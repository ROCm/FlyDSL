#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU contracts for the direct Softmax autotune adopter.

Selection behavior is validated with deterministic synthetic ranking, never with
shared-runner timing: CI must not publish a deployment winner.
"""

import json
import os
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import kernels.norm.softmax_autotune as softmax_autotune_module  # noqa: E402
from kernels.norm.softmax_autotune import (  # noqa: E402
    _RTOL,
    _softmax_hot_cache,
    _softmax_tuner,
    candidate_configs,
    softmax_autotuned,
    uses_fast_path,
)
from kernels.norm.softmax_kernel import TUNING_SCHEMA, softmax_direct  # noqa: E402

DTYPES = [(torch.float32, "f32"), (torch.float16, "f16"), (torch.bfloat16, "bf16")]

# Small, medium, wide, aligned, arbitrary-column-tail and partial-row-block cases.
SHAPES = [
    (8, 128),
    (17, 128),
    (16, 2000),
    (32, 2048),
    (8, 4096),
    (8, 4097),
    (4, 8192),
]


@pytest.fixture(autouse=True)
def _isolated_tuner(tmp_path, monkeypatch):
    """The tuner is a module-level singleton built at import, so every test needs its
    winner cache, artifact cache, disk cache and artifact directory isolated."""
    _softmax_tuner.cache.clear()
    _softmax_tuner._artifact_cache.clear()
    _softmax_hot_cache.clear()
    monkeypatch.setattr(_softmax_tuner, "_cache_file", tmp_path / "softmax.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    yield
    _softmax_tuner.cache.clear()
    _softmax_tuner._artifact_cache.clear()
    _softmax_hot_cache.clear()


def _inputs(M, N, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.rand((M, N), generator=generator, device="cuda", dtype=torch.float32) * 8.0 - 4.0).to(dtype)
    return x, torch.empty_like(x)


def _assert_close(out, reference, dtype_str):
    """Same scale-aware criterion the candidate gate uses, so the default and every
    candidate are held to one standard."""
    tol = _RTOL[dtype_str]
    bound = tol * (reference.abs() + reference.amax(dim=-1, keepdim=True))
    error = (out.float() - reference).abs()
    worst = (error / (reference.abs() + reference.amax(dim=-1, keepdim=True))).max().item()
    assert bool((error <= bound).all()), f"{dtype_str} relative error {worst:.3e} exceeds {tol:g}"


def _reference(x):
    return torch.softmax(x.float(), dim=-1)


# ── kernel specialization ────────────────────────────────────────────────
@pytest.mark.parametrize("block_threads", [256, 512])
def test_workgroup_size_tracks_block_threads(block_threads):
    """A 512-thread candidate is only a real candidate if the workgroup limit follows it
    past the 256 default. The launcher passes a static block dim, so the compiler infers
    known_block_size and no explicit attribute is needed -- this pins that inference,
    since a regression would silently shrink the search space to the sizes that fit 256.
    """
    x, out = _inputs(4, 4096, torch.bfloat16)
    stream = torch.cuda.current_stream()
    compiled = flyc.compile(
        softmax_direct,
        x,
        out,
        x.shape[0],
        4096,
        "bf16",
        block_threads,
        TUNING_SCHEMA,
        stream,
    )
    stream.synchronize()
    artifact = compiled._keepalive

    assert f"known_block_size = array<i32: {block_threads}, 1, 1>" in artifact.source_ir
    match = re.search(r"max_flat_workgroup_size\\CD\\([0-9A-Fa-f]{2})\\([0-9A-Fa-f]{2})", artifact.ir)
    assert match is not None and int("".join(match.groups()), 16) == block_threads
    _assert_close(out, _reference(x), "bf16")


# ── every legal candidate is numerically correct ─────────────────────────
@pytest.mark.parametrize("dtype,dtype_str", DTYPES)
@pytest.mark.parametrize("M,N", SHAPES)
def test_every_legal_candidate_is_correct(dtype, dtype_str, M, N):
    """Enumerate and validate the whole search space before any ranking, across both
    data-movement paths."""
    x, out = _inputs(M, N, dtype, seed=M * 1000 + N)
    reference = _reference(x)
    stream = torch.cuda.current_stream()

    paths = set()
    for config in candidate_configs(m_in=M, N=N, dtype_str=dtype_str):
        block_threads = config.kwargs["BLOCK_THREADS"]
        threads_per_row = config.kwargs["THREADS_PER_ROW"]
        paths.add(uses_fast_path(N, dtype_str, threads_per_row))
        kernel_kwargs = dict(config.kwargs)
        kernel_kwargs.pop("BLOCK_THREADS")
        out.zero_()
        softmax_direct(
            x,
            out,
            M,
            N,
            dtype_str,
            block_threads,
            TUNING_SCHEMA,
            stream,
            **kernel_kwargs,
        )
        torch.cuda.synchronize()
        _assert_close(out, reference, dtype_str)
        row_sums = out.float().sum(dim=-1)
        assert torch.isfinite(out.float()).all()
        assert bool(((row_sums - 1.0).abs() < 0.1).all())

    if N == 2048 and dtype_str in ("f16", "bf16"):
        assert paths == {True, False}, "N=2048 should exercise both paths in one search"


# ── wrapper contract ─────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype,dtype_str", DTYPES)
def test_wrapper_matches_the_torch_reference(dtype, dtype_str):
    x, out = _inputs(64, 4096, dtype)
    softmax_autotuned(x, out)
    torch.cuda.synchronize()
    _assert_close(out, _reference(x), dtype_str)


def test_zero_rows_is_a_no_launch(monkeypatch):
    """M == 0 must return before reaching the tuner: a zero-sized grid is not a legal
    launch, and there is nothing to compute."""
    monkeypatch.setattr(
        _softmax_tuner,
        "resolve_config",
        lambda *a, **k: pytest.fail("unexpected resolution"),
    )
    monkeypatch.setattr(
        softmax_autotune_module,
        "_compile_resolved",
        lambda *a, **k: pytest.fail("unexpected compilation"),
    )
    x = torch.empty((0, 4096), device="cuda", dtype=torch.bfloat16)
    assert softmax_autotuned(x, torch.empty_like(x)) is None


def _rank_mismatch():
    return torch.empty((4, 8, 16), device="cuda"), torch.empty((4, 8, 16), device="cuda")


def _shape_mismatch():
    return torch.empty((4, 16), device="cuda"), torch.empty((4, 32), device="cuda")


def _dtype_mismatch():
    return (
        torch.empty((4, 16), device="cuda", dtype=torch.bfloat16),
        torch.empty((4, 16), device="cuda", dtype=torch.float32),
    )


def _non_contiguous():
    return torch.empty((16, 4), device="cuda").t(), torch.empty((4, 16), device="cuda")


def _aliased():
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    return x, x


def _partial_overlap():
    storage = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    return storage[0:4], storage[2:6]


@pytest.mark.parametrize(
    "make_args,message",
    [
        (_rank_mismatch, "2-D"),
        (_shape_mismatch, "shape mismatch"),
        (_dtype_mismatch, "dtype mismatch"),
        (_non_contiguous, "contiguous"),
        (_aliased, "out-of-place"),
        (_partial_overlap, "out-of-place"),
    ],
    ids=["rank", "shape", "dtype", "non_contiguous", "aliased", "partial_overlap"],
)
def test_input_contract_failures_are_rejected(make_args, message):
    a, b = make_args()
    with pytest.raises(ValueError, match=message):
        softmax_autotuned(a, b)


# ── serving order: default -> forced search -> cache hit -> artifact ─────
def test_default_serves_without_searching_on_the_current_stream(monkeypatch):
    monkeypatch.setattr(_softmax_tuner, "_bench_one", lambda *a, **k: pytest.fail("unexpected search"))
    observed = []
    original_resolve = _softmax_tuner.resolve_config

    def record(*args, **kwargs):
        config = original_resolve(*args, **kwargs)
        observed.append((config.kwargs["BLOCK_THREADS"], kwargs["stream"].cuda_stream))
        return config

    monkeypatch.setattr(_softmax_tuner, "resolve_config", record)

    x, out = _inputs(64, 4096, torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        softmax_autotuned(x, out)
    stream.synchronize()

    assert observed == [(256, stream.cuda_stream)]
    _assert_close(out, _reference(x), "bf16")


def test_compiled_cache_hit_bypasses_resolution_and_accepts_new_buffers_and_stream(
    monkeypatch,
):
    x1, out1 = _inputs(64, 4096, torch.bfloat16, seed=1)
    softmax_autotuned(x1, out1)
    torch.cuda.synchronize()
    _assert_close(out1, _reference(x1), "bf16")

    monkeypatch.setattr(
        _softmax_tuner,
        "resolve_config",
        lambda *a, **k: pytest.fail("compiled cache hit resolved again"),
    )
    x2, out2 = _inputs(64, 4096, torch.bfloat16, seed=2)
    stream = torch.cuda.Stream()
    softmax_autotuned(x2, out2, stream=stream)
    stream.synchronize()

    _assert_close(out2, _reference(x2), "bf16")


def test_forced_search_then_cache_hit_then_artifact_load(monkeypatch, tmp_path):
    """Deterministic synthetic ranking: the winner is chosen by construction, not by
    timing, so this asserts selection mechanics without publishing a real winner."""
    configs = candidate_configs()
    target = next(
        index
        for index, config in enumerate(configs, start=1)
        if config.kwargs["BLOCK_THREADS"] == 128 and config.waves_per_eu == 2
    )
    completed = 0

    def bench_once(call, warmup, rep):
        nonlocal completed
        call()
        torch.cuda.synchronize()
        completed += 1
        return 0.0 if completed == target else float(completed)

    monkeypatch.setattr(_softmax_tuner, "_do_bench", bench_once)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")

    x, out = _inputs(64, 4096, torch.bfloat16)
    softmax_autotuned(x, out)
    torch.cuda.synchronize()
    _assert_close(out, _reference(x), "bf16")
    assert completed == len(configs), "every legal candidate should be benched under a forced search"

    artifacts = sorted(Path(os.environ["FLYDSL_AUTOTUNE_CONFIG_DIR"]).glob("softmax_fwd-*.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text())
    assert payload["identity"]["key"] == {
        "m_in": 64,
        "N": 4096,
        "dtype_str": "bf16",
        "tuning_schema": TUNING_SCHEMA,
    }
    assert payload["identity"]["device"]["name"] == torch.cuda.get_device_name(x.device)
    assert payload["config"]["BLOCK_THREADS"] == 128
    assert payload["config"]["waves_per_eu"] == 2

    # Cache hit: the searched winner outranks the default and does not re-search.
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    monkeypatch.setattr(_softmax_tuner, "default", lambda *a, **k: pytest.fail("cached winner must win"))
    softmax_autotuned(x, out)
    torch.cuda.synchronize()
    assert completed == len(configs)
    _assert_close(out, _reference(x), "bf16")

    # A fresh serving decision with no winner cache must load the emitted artifact
    # without evaluating the default or the search space.
    _softmax_tuner.cache.clear()
    _softmax_tuner._artifact_cache.clear()
    _softmax_hot_cache.clear()
    monkeypatch.setattr(
        _softmax_tuner,
        "configs",
        lambda *a, **k: pytest.fail("artifact must skip the search"),
    )
    softmax_autotuned(x, out)
    torch.cuda.synchronize()
    assert completed == len(configs)
    _assert_close(out, _reference(x), "bf16")


@pytest.mark.parametrize("damage", ["corrupt", "identity", "version"])
def test_invalid_artifact_falls_back_to_the_default(monkeypatch, damage):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setattr(_softmax_tuner, "_do_bench", lambda call, warmup, rep: (call(), 1.0)[1])
    x, out = _inputs(64, 4096, torch.bfloat16)
    softmax_autotuned(x, out)
    torch.cuda.synchronize()

    artifact = next(Path(os.environ["FLYDSL_AUTOTUNE_CONFIG_DIR"]).glob("softmax_fwd-*.json"))
    if damage == "corrupt":
        artifact.write_text("{not json")
    else:
        payload = json.loads(artifact.read_text())
        if damage == "identity":
            payload["identity"]["key"]["N"] = 1
        else:
            payload["version"] = 999
        artifact.write_text(json.dumps(payload))

    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    _softmax_tuner.cache.clear()
    _softmax_tuner._artifact_cache.clear()
    _softmax_hot_cache.clear()
    served = []
    original_resolve = _softmax_tuner.resolve_config

    def record(*args, **kwargs):
        config = original_resolve(*args, **kwargs)
        served.append(config.kwargs["BLOCK_THREADS"])
        return config

    monkeypatch.setattr(_softmax_tuner, "resolve_config", record)
    softmax_autotuned(x, out)
    torch.cuda.synchronize()

    assert served == [256], "an unusable artifact must fall back to the compatibility default"
    _assert_close(out, _reference(x), "bf16")


# ── candidate correctness gate ───────────────────────────────────────────
def test_a_numerically_wrong_candidate_cannot_win(monkeypatch):
    """The gate is the whole reason a fast-but-wrong candidate is not selectable."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    poisoned = 512

    original_run = _softmax_tuner._run_with_hints

    def sabotage(compiler_opts, args, kwargs):
        result = original_run(compiler_opts, args, kwargs)
        if kwargs.get("BLOCK_THREADS") == poisoned:
            args[1].zero_()  # launches fine, computes garbage
        return result

    monkeypatch.setattr(_softmax_tuner, "_run_with_hints", sabotage)

    # Make the poisoned candidate the fastest, so only the gate can keep it from winning.
    def bench(call, warmup, rep):
        call()
        return 0.0 if _last_block[0] == poisoned else 1.0

    _last_block = [None]
    original_bench_one = _softmax_tuner._bench_one

    def track(config, args, kwargs):
        _last_block[0] = config.kwargs["BLOCK_THREADS"]
        return original_bench_one(config, args, kwargs)

    monkeypatch.setattr(_softmax_tuner, "_do_bench", bench)
    monkeypatch.setattr(_softmax_tuner, "_bench_one", track)

    x, out = _inputs(64, 4096, torch.bfloat16)
    softmax_autotuned(x, out)
    torch.cuda.synchronize()

    winners = {config.kwargs["BLOCK_THREADS"] for config in _softmax_tuner.cache.values()}
    assert poisoned not in winners, "the preflight gate must reject the sabotaged candidate"


def test_all_candidates_rejected_raises_with_the_cause(monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setattr(
        _softmax_tuner,
        "validate_hook",
        lambda sig_args: (_ for _ in ()).throw(ValueError("synthetic rejection")),
    )
    x, out = _inputs(8, 4096, torch.bfloat16)
    with pytest.raises(RuntimeError, match="All autotune configs failed") as excinfo:
        softmax_autotuned(x, out)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "synthetic rejection" in str(excinfo.value.__cause__)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
