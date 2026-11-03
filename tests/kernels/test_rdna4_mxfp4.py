# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""MXFP4 bit packing, scale modes, tails, replay and device ownership on RDNA4."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kernels.gemm.rdna4_mxfp4 import gemm, prepare_weights, tune_gemm

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.fixture(autouse=True)
def require_rdna4():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm RDNA4")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] not in ("gfx1200", "gfx1201"):
        pytest.skip("requires gfx1200/gfx1201")


def make_case(shape, device="cuda", seed=19):
    m, n, k = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(m, k, generator=generator, device=device) * 3).to(torch.float8_e4m3fn)
    packed = torch.randint(0, 256, (n, k // 2), device=device, generator=generator, dtype=torch.uint8)
    scales = torch.randint(124, 132, (n, k // 32), device=device, generator=generator, dtype=torch.uint8)
    x_scale = torch.linspace(0.5, 1.5, m, device=device, dtype=torch.float32)
    return x, packed, x_scale, scales


def reference(x, packed, x_scale, scales):
    # Independent OCP MXFP4 decoding; no FlyDSL packing or conversion helpers.
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    values = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    weights = values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return (x.float() @ weights.T) * x_scale[:, None]


def check(actual, expected):
    assert torch.isfinite(actual).all()
    relative_rms = ((actual.float() - expected).square().mean() / expected.square().mean().clamp_min(1e-35)).sqrt()
    assert float(relative_rms) < 0.006


@pytest.mark.parametrize("shape", [(1, 70, 32), (7, 128, 96), (33, 70, 160), (73, 513, 1120), (257, 128, 512)])
@pytest.mark.parametrize("mode", ["e4m3", "e5m2", "grouped"])
@pytest.mark.parametrize("strategy", ["register", "hybrid", "auto"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_formats_tails_and_strategies(shape, mode, strategy, dtype):
    if strategy == "hybrid" and mode == "grouped":
        pytest.skip("exact grouped scaling uses register")
    x, packed, xs, scales = make_case(shape)
    prepared = prepare_weights(packed, scales, mode=mode)
    cfg = None if strategy == "auto" else {"engine": strategy}
    if strategy == "hybrid":
        cfg.update(tile_k=32, operand_prefetch=1)
    y = gemm(x, prepared, xs, dtype=dtype, config=cfg)
    assert y.dtype == dtype and y.shape == shape[:2]
    check(y, reference(x, packed, xs, scales))


@pytest.mark.parametrize("cache_policy", ["auto", "hot", "rotating"])
@pytest.mark.parametrize("mode", ["e4m3", "e5m2", "grouped"])
def test_default_policy_reads_changed_inputs_during_replay(cache_policy, mode):
    x, packed, xs, scales = make_case((73, 2117, 1120))
    prepared = prepare_weights(packed, scales, mode=mode)
    y = gemm(x, prepared, xs, cache_policy=cache_policy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(x, prepared, xs, out=y, cache_policy=cache_policy)
    x.view(torch.uint8).bitwise_xor_(128)
    xs.mul_(0.5)
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(y, reference(x, packed, xs, scales))


@pytest.mark.parametrize("mode,gap", [("e4m3", 14), ("e5m2", 28)])
def test_folded_subnormal_bits_are_lossless(mode, gap):
    x = torch.zeros(16, 64, device="cuda")
    x[:, :32] = 1
    x = x.to(torch.float8_e4m3fn)
    codes = torch.arange(16, device="cuda", dtype=torch.uint8)[:, None]
    packed = (codes | codes << 4).expand(16, 32).contiguous()
    scales = torch.full((16, 2), 100, device="cuda", dtype=torch.uint8)
    scales[:, 1] += gap
    xs = torch.ones(16, device="cuda")
    y = gemm(x, prepare_weights(packed, scales, mode=mode), xs)
    torch.testing.assert_close(y, reference(x, packed, xs, scales).to(y.dtype), atol=0, rtol=0)


def test_unsafe_scale_spans_use_exact_grouped_fallback():
    x, packed, xs, scales = make_case((33, 70, 160))
    scales[:, 0] = 100
    scales[:, 1] = 129
    prepared = prepare_weights(packed, scales)
    assert prepared.mode == "grouped"
    check(gemm(x, prepared, xs), reference(x, packed, xs, scales))
    with pytest.raises(ValueError, match="losslessly"):
        prepare_weights(packed, scales, mode="e4m3")
    with pytest.raises(ValueError, match="losslessly folded"):
        gemm(x, prepared, xs, config={"engine": "hybrid"})


@pytest.mark.parametrize("strategy", ["register", "hybrid"])
def test_tail_stores_preserve_output_storage_guards(strategy):
    x, packed, xs, scales = make_case((73, 70, 160))
    storage = torch.full((73 * 70 + 128,), 127.0, device="cuda", dtype=torch.bfloat16)
    out = storage[64:-64].view(73, 70)
    config = {"engine": strategy}
    if strategy == "hybrid":
        config.update(tile_k=32, operand_prefetch=1)
    result = gemm(x, prepare_weights(packed, scales), xs, out=out, config=config)
    assert result.data_ptr() == out.data_ptr()
    check(result, reference(x, packed, xs, scales))
    assert bool((storage[:64] == 127).all())
    assert bool((storage[-64:] == 127).all())


@pytest.mark.parametrize("mode", ["e4m3", "e5m2", "grouped"])
def test_compile_only_warmup_serves_unseen_rows_with_guarded_stores(mode, tmp_path, monkeypatch):
    """Startup warms compile classes, then arbitrary rows need no new compiler."""
    from kernels.gemm.rdna4_mxfp4 import compile_gemm, get_warmup_keys

    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "kernels"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "tuning"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    _, packed, _, scales = make_case((73, 93, 672), seed=301)
    prepared = prepare_weights(packed, scales, mode=mode)
    keys = get_warmup_keys(
        prepared.n,
        prepared.k,
        256,
        mode=prepared.mode,
        normal_only=prepared.normal_only,
        device_id=packed.device.index,
        cache_policy="rotating",
    )
    allocated = torch.cuda.memory_allocated()
    for key in keys:
        compile_gemm(key)
    assert torch.cuda.memory_allocated() == allocated

    # A missing specialization now raises, rather than compiling on demand.
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    for rows in (73, 91, 128, 192):
        x, _, xs, _ = make_case((rows, 93, 672), seed=301 + rows)
        storage = torch.full((rows * 93 + 128,), 127.0, device="cuda", dtype=torch.bfloat16)
        out = storage[64:-64].view(rows, 93)
        gemm(x, prepared, xs, out=out, cache_policy="rotating")
        check(out, reference(x, packed, xs, scales))
        assert bool((storage[:64] == 127).all())
        assert bool((storage[-64:] == 127).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(x, prepared, xs, out=out, cache_policy="rotating")
    x.view(torch.uint8).bitwise_xor_(128)
    graph.replay()
    torch.cuda.synchronize()
    check(out, reference(x, packed, xs, scales))


def test_bf16_epilogue_preserves_subnormal_output():
    k = 17408
    x = torch.full((16, k), 448.0, device="cuda").to(torch.float8_e4m3fn)
    packed = torch.full((16, k // 2), 0x77, device="cuda", dtype=torch.uint8)
    scales = torch.zeros(16, k // 32, device="cuda", dtype=torch.uint8)
    xs = torch.full((16,), 2.0**-30, device="cuda")
    y = gemm(x, prepare_weights(packed, scales), xs)
    bits = int(torch.tensor(448 * 6 * k * 2.0**-157, dtype=torch.bfloat16).view(torch.int16))
    assert bits != 0
    assert bool((y.view(torch.int16) == bits).all())


@pytest.mark.multi_gpu
def test_input_device_owns_compilation_and_execution():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two devices")
    if any(torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] not in ("gfx1200", "gfx1201") for i in (0, 1)):
        pytest.skip("requires two RDNA4 devices")
    original = torch.cuda.current_device()
    values = []
    for device in (0, 1, 0):
        with torch.cuda.device(device):
            x, packed, xs, scales = make_case((33, 70, 160), device=f"cuda:{device}")
            prepared = prepare_weights(packed, scales)
            values.append((x, prepared, xs, reference(x, packed, xs, scales)))
    torch.cuda.set_device(0)
    try:
        for x, prepared, xs, expected in values:
            check(gemm(x, prepared, xs), expected)
            assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(original)


def test_native_calibration_persists_and_replays_changed_inputs(tmp_path, monkeypatch):
    from kernels.gemm.rdna4_mxfp4 import get_warmup_keys
    from kernels.gemm.rdna4_mxfp4.entrypoint import specialization_key
    from kernels.gemm.rdna4_mxfp4.tuning import cached_configurations

    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    x, packed, xs, scales = make_case((7, 80, 224))
    prepared = prepare_weights(packed, scales)
    check(gemm(x, prepared, xs, cache_policy="hot"), reference(x, packed, xs, scales))
    assert not list(tmp_path.iterdir())
    choice = tune_gemm(x, prepared, xs[:, None], cache_policy="hot")
    cached = cached_configurations(
        prepared.n,
        prepared.k,
        7,
        mode=prepared.mode,
        normal_only=prepared.normal_only,
        nibble_layout=prepared.nibble_layout,
        dtype=torch.bfloat16,
        device_id=x.device.index,
        cache_policy="hot",
    )
    assert (7, choice) in cached
    keys = get_warmup_keys(
        prepared.n,
        prepared.k,
        7,
        mode=prepared.mode,
        normal_only=prepared.normal_only,
        dtype=torch.bfloat16,
        device_id=x.device.index,
        cache_policy="hot",
    )
    assert (
        specialization_key(
            7,
            prepared.n,
            prepared.k,
            mode=prepared.mode,
            normal_only=prepared.normal_only,
            dtype=torch.bfloat16,
            device_id=x.device.index,
            config=choice,
        )
        in keys
    )
    cache = tmp_path / "rdna4_mxfp4_choice_d0.json"
    assert cache.exists() and json.loads(cache.read_text())
    timestamp = cache.stat().st_mtime_ns
    y = gemm(x, prepared, xs, cache_policy="hot")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(x, prepared, xs, out=y, cache_policy="hot")
    x.view(torch.uint8).bitwise_xor_(128)
    xs.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    check(y, reference(x, packed, xs, scales))

    # A new process has no in-memory choice: explicit calibration must load
    # the native disk cache without measuring configurations again.
    code = """
import json
import torch
from kernels.gemm.rdna4_mxfp4 import prepare_weights, tune_gemm
g = torch.Generator(device='cuda').manual_seed(19)
x = (torch.randn(7, 224, generator=g, device='cuda') * 3).to(torch.float8_e4m3fn)
p = torch.randint(0, 256, (80, 112), generator=g, device='cuda', dtype=torch.uint8)
s = torch.randint(124, 132, (80, 7), generator=g, device='cuda', dtype=torch.uint8)
xs = torch.linspace(0.5, 1.5, 7, device='cuda')
print(json.dumps(tune_gemm(x, prepare_weights(p, s), xs, cache_policy='hot'), sort_keys=True))
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[autotune]" not in result.stdout
    assert json.loads(result.stdout.strip().splitlines()[-1]) == choice
    assert cache.stat().st_mtime_ns == timestamp
