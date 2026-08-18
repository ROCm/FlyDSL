#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU-free contracts for the Softmax autotune adopter.

Candidate generation, path classification and cache-key partitioning are pure host-side
logic, so they are tested here rather than behind an l2_device gate. The wave size is
injected explicitly: kernels/norm/softmax_kernel.py resolves WARP_SIZE at import time, so
an ambient value would make these assertions a property of whatever host ran them.
"""

import importlib

import pytest

from flydsl.autotune import Config, do_bench
from kernels.norm.softmax_autotune import (
    MAX_BLOCK_THREADS,
    _default_config,
    _quack_threads_per_row,
    _softmax_select_config,
    _softmax_tuner,
    candidate_configs,
    is_legal,
    tile_cols,
    uses_fast_path,
)
from kernels.norm.softmax_kernel import BLOCK_THREADS, TUNING_SCHEMA

pytestmark = [pytest.mark.l0_backend_agnostic]

WAVE_SIZES = (32, 64)


class FakeTensor:
    """Minimal tensor stand-in with the attributes _make_key reads."""

    def __init__(self, shape, dtype="bfloat16"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def stride(self):
        strides, acc = [], 1
        for s in reversed(self.shape):
            strides.append(acc)
            acc *= s
        return tuple(reversed(strides))


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)


def _identities(configs):
    return [
        (
            c.kwargs["BLOCK_THREADS"],
            c.kwargs.get("THREADS_PER_ROW", c.kwargs["BLOCK_THREADS"]),
            c.kwargs.get("ROWS_PER_BLOCK", 1),
            c.waves_per_eu,
        )
        for c in configs
    ]


# ── candidate generation ─────────────────────────────────────────────────
@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_full_row_candidates_cross_geometry_and_occupancy(warp_size):
    configs = candidate_configs(warp_size, m_in=1, N=4096, dtype_str="bf16")
    full_row = [identity for identity in _identities(configs) if identity[1:3] == (identity[0], 1)]
    expected = [
        (threads, threads, 1, waves_per_eu) for threads in (64, 128, 256, 512) for waves_per_eu in (None, 1, 2, 4)
    ]
    assert full_row == expected


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_short_rows_add_quack_style_subgroup_packing(warp_size):
    configs = candidate_configs(warp_size, m_in=64, N=128, dtype_str="bf16")
    packed = [identity for identity in _identities(configs) if identity[2] > 1]
    assert packed == [
        (128, 8, 16, None),
        (128, 16, 8, None),
        (256, 16, 16, None),
        (128, 16, 8, 2),
    ]


def test_quack_row_width_heuristic_is_preserved_without_nvidia_clusters():
    assert [_quack_threads_per_row(n) for n in (64, 128, 3072, 6144, 16384, 16385)] == [
        8,
        16,
        32,
        64,
        128,
        256,
    ]


def test_search_space_excludes_state_dependent_and_rejected_algorithm_policies():
    configs = candidate_configs(m_in=64, N=16384, dtype_str="bf16")
    assert all("BUFFER_POLICY" not in config.kwargs for config in configs)
    assert all("INPUT_CACHE_MODIFIER" not in config.kwargs for config in configs)


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_candidate_order_is_deterministic(warp_size):
    assert _identities(candidate_configs(warp_size)) == _identities(candidate_configs(warp_size))


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_candidates_are_deduplicated(warp_size):
    identities = _identities(candidate_configs(warp_size))
    assert len(identities) == len(set(identities))


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_compatibility_default_is_always_retained(warp_size):
    """256 with no compiler override must survive every legal set, because it is what a
    non-searching call serves."""
    assert (
        BLOCK_THREADS,
        BLOCK_THREADS,
        1,
        None,
    ) in _identities(candidate_configs(warp_size))
    default = _default_config()
    assert default.kwargs["BLOCK_THREADS"] == BLOCK_THREADS
    assert default.waves_per_eu is None


# ── objective legality ───────────────────────────────────────────────────
@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_legality_rejects_non_positive_and_oversized(warp_size):
    assert not is_legal(0, warp_size)
    assert not is_legal(-64, warp_size)
    assert is_legal(MAX_BLOCK_THREADS, warp_size)
    assert not is_legal(MAX_BLOCK_THREADS + warp_size, warp_size)


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_legality_rejects_wave_misalignment(warp_size):
    assert not is_legal(warp_size + 1, warp_size)
    assert is_legal(warp_size * 2, warp_size)


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_legality_requires_exact_row_partitioning(warp_size):
    assert is_legal(128, warp_size, threads_per_row=16, rows_per_block=8)
    assert not is_legal(128, warp_size, threads_per_row=16, rows_per_block=4)
    assert not is_legal(256, warp_size, threads_per_row=128, rows_per_block=2)
    assert not is_legal(128, warp_size, threads_per_row=24, rows_per_block=1)


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_single_wave_reduction_holds_for_every_legal_size(warp_size):
    """The block reduction finalizes on one wave indexing lane < RED_SLOTS, so RED_SLOTS
    must fit a wave. At MAX_BLOCK_THREADS=1024 the workgroup limit already implies this
    for both wave sizes; the rule is kept so that raising either bound cannot silently
    produce a broken reduction. This pins the implication rather than pretending the
    rule is currently the binding constraint."""
    legal = [bt for bt in range(warp_size, MAX_BLOCK_THREADS + 1, warp_size) if is_legal(bt, warp_size)]
    assert legal, "expected at least one legal block size"
    assert all(-(-bt // warp_size) <= warp_size for bt in legal)


@pytest.mark.parametrize("warp_size", WAVE_SIZES)
def test_every_generated_candidate_is_legal(warp_size):
    for block_threads, threads_per_row, rows_per_block, _ in _identities(candidate_configs(warp_size)):
        assert is_legal(
            block_threads,
            warp_size,
            threads_per_row=threads_per_row,
            rows_per_block=rows_per_block,
        )


# ── data-movement path classification ────────────────────────────────────
def test_tile_cols_follows_the_128_bit_transaction_contract():
    assert tile_cols("bf16", 256) == 256 * 8
    assert tile_cols("f16", 256) == 256 * 8
    assert tile_cols("f32", 256) == 256 * 4


@pytest.mark.parametrize("dtype_str", ["f32", "f16", "bf16"])
def test_all_candidates_take_the_fast_path(dtype_str):
    assert all(uses_fast_path(4096, dtype_str, bt) for bt in (64, 128, 256, 512))


def test_mixed_path_set():
    """bf16 N=2048: 512 threads needs 4096 columns per tile, so it falls to the scalar
    path while every smaller candidate stays vectorized. One search compares both."""
    assert [uses_fast_path(2048, "bf16", bt) for bt in (64, 128, 256, 512)] == [
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("dtype_str", ["f32", "f16", "bf16"])
def test_all_candidates_take_the_generic_path(dtype_str):
    assert not any(uses_fast_path(4097, dtype_str, bt) for bt in (64, 128, 256, 512))


# ── cache-key partitioning ───────────────────────────────────────────────
def _key(monkeypatch, *, m_in=64, N=4096, dtype_str="bf16", tuning_schema=TUNING_SCHEMA, arch="gfx950"):
    at = importlib.import_module("flydsl.autotune")
    monkeypatch.setattr(at, "_device_fingerprint", lambda: arch)
    args = (FakeTensor((m_in, N)), FakeTensor((m_in, N)), m_in)
    kwargs = {
        "N": N,
        "dtype_str": dtype_str,
        "tuning_schema": tuning_schema,
        "stream": None,
    }
    return _softmax_tuner._make_key(args, kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"m_in": 65},
        {"N": 8192},
        {"dtype_str": "f16"},
        {"tuning_schema": TUNING_SCHEMA + 1},
        {"arch": "gfx942"},
    ],
    ids=["m_in", "N", "dtype", "tuning_schema", "arch"],
)
def test_each_declared_axis_partitions_the_winner_cache(monkeypatch, override):
    assert _key(monkeypatch) != _key(monkeypatch, **override)


def test_identical_calls_share_one_key(monkeypatch):
    assert _key(monkeypatch) == _key(monkeypatch)


def test_declared_key_names_real_kernel_parameters():
    """artifact_name validation depends on this, and a typo would silently widen the
    cache instead of failing."""
    assert _softmax_tuner.key == ["m_in", "N", "dtype_str", "tuning_schema"]
    assert all(name in _softmax_tuner._signature.parameters for name in _softmax_tuner.key)
    assert _softmax_tuner.artifact_name == "softmax_fwd"


# ── serving path ─────────────────────────────────────────────────────────
def test_default_serving_never_searches(monkeypatch):
    """With FLYDSL_AUTOTUNE unset or 0 an ordinary call must not benchmark anything."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setattr(_softmax_tuner, "_bench_one", lambda *a, **k: pytest.fail("unexpected search"))
    monkeypatch.setattr(
        _softmax_tuner,
        "configs",
        lambda *a, **k: pytest.fail("unexpected candidate generation"),
    )
    served = []
    monkeypatch.setattr(
        _softmax_tuner,
        "_run_config",
        lambda config, args, kwargs: served.append(config),
    )

    args = (FakeTensor((64, 4096)), FakeTensor((64, 4096)), 64)
    _softmax_tuner(*args, N=4096, dtype_str="bf16", tuning_schema=TUNING_SCHEMA, stream=None)

    assert len(served) == 1
    assert served[0].kwargs["BLOCK_THREADS"] == BLOCK_THREADS


def test_validate_hook_is_wired_up():
    """The gate is the reason a wrong candidate cannot win; losing the wiring would be
    invisible until a real miscompile."""
    assert _softmax_tuner.validate_hook is not None


def test_search_space_is_the_shared_candidate_owner():
    configs = _softmax_tuner.configs(None, None, 64, N=4096, dtype_str="bf16")
    assert _identities(configs) == _identities(candidate_configs(m_in=64, N=4096, dtype_str="bf16"))
    assert all(isinstance(c, Config) for c in configs)


# ── timing contract ──────────────────────────────────────────────────────
def test_softmax_tuner_uses_the_shared_backlogged_timer():
    assert _softmax_tuner._do_bench is do_bench
    assert (_softmax_tuner.warmup, _softmax_tuner.rep) == (10, 100)


def test_tie_policy_prefers_the_compatibility_default_within_two_percent():
    fastest = Config(
        BLOCK_THREADS=512,
        THREADS_PER_ROW=512,
        ROWS_PER_BLOCK=1,
        waves_per_eu=4,
    )
    compatibility = Config(
        BLOCK_THREADS=BLOCK_THREADS,
        THREADS_PER_ROW=BLOCK_THREADS,
        ROWS_PER_BLOCK=1,
    )
    assert _softmax_select_config([(fastest, 1.0), (compatibility, 1.019)]) == (
        compatibility,
        1.019,
    )
    assert _softmax_select_config([(fastest, 1.0), (compatibility, 1.021)]) == (
        fastest,
        1.0,
    )


def test_tie_policy_prefers_no_wpe_then_more_packed_rows():
    wpe = Config(
        BLOCK_THREADS=128,
        THREADS_PER_ROW=16,
        ROWS_PER_BLOCK=8,
        waves_per_eu=2,
    )
    packed = Config(
        BLOCK_THREADS=256,
        THREADS_PER_ROW=16,
        ROWS_PER_BLOCK=16,
    )
    assert _softmax_select_config([(wpe, 1.0), (packed, 1.019)]) == (packed, 1.019)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
