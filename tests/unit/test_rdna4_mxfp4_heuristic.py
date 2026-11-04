# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU selector invariants: unseen shapes, tails, K factors and cache hints."""

import builtins
import subprocess
import sys

import pytest

from kernels.gemm.rdna4_mxfp4.heuristic import (
    _legal_config,
    _selected,
    choose_config,
)

pytestmark = pytest.mark.l0_backend_agnostic


@pytest.mark.parametrize("first_module", ["entrypoint", "heuristic"])
def test_public_gemm_is_callable_after_importing_support_modules(first_module):
    """Internal imports must not shadow the package's public GEMM function."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import torch
importlib.import_module("kernels.gemm.rdna4_mxfp4.{first_module}")
from kernels.gemm.rdna4_mxfp4 import gemm, prepare_weights, get_warmup_keys, compile_gemm
assert all(callable(f) for f in (gemm, prepare_weights, get_warmup_keys, compile_gemm))
assert not torch.cuda.is_initialized()
""",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 17, 32),
        (13, 2053, 1120),
        (33, 70, 160),
        (73, 4099, 1568),
        (97, 10003, 2080),
        (257, 1536, 544),
        (1025, 2053, 1120),
        (2049, 10003, 2080),
        (8191, 6144, 8192),
        (10240, 3072, 2048),
    ],
)
@pytest.mark.parametrize("cache_policy", ["auto", "hot", "rotating"])
def test_unseen_shapes_select_legal_tiles(shape, cache_policy):
    cfg = choose_config(*shape, cache_policy=cache_policy)
    assert cfg["engine"] in ("register", "hybrid")
    assert _legal_config(cfg, *shape)
    if cfg["engine"] == "register":
        assert shape[2] % (32 * cfg["k_unroll"]) == 0
        if not cfg["buffer_io"]:
            assert shape[0] % (16 * cfg["waves_m"] * cfg["repeat_m"]) == 0
            assert shape[1] % (16 * cfg["waves_n"] * cfg["repeat_n"]) == 0
    else:
        assert shape[2] % cfg["tile_k"] == 0
        assert 0 <= cfg["operand_prefetch"] < cfg["tile_k"] // 16


@pytest.mark.parametrize("shape", [(17, 513, 544), (192, 3840, 3840), (8193, 4097, 1568)])
def test_grouped_fallback_preserves_k32_scaling(shape):
    cfg = choose_config(*shape, "grouped")
    assert cfg["engine"] == "register"
    assert cfg["k_unroll"] == 1
    assert _legal_config(cfg, *shape)


def test_mutating_returned_dictionary_cannot_corrupt_cache():
    original = choose_config(73, 4099, 1568)
    changed = choose_config(73, 4099, 1568)
    changed["engine"] = "invalid"
    assert choose_config(73, 4099, 1568) == original
    assert _selected.cache_info().hits > 0


def test_selector_has_no_gpu_or_machine_learning_dependency(monkeypatch):
    ordinary_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "flydsl", "numpy", "sklearn"):
            raise AssertionError("runtime selector imported " + name)
        return ordinary_import(name, *args, **kwargs)

    _selected.cache_clear()
    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert choose_config(57, 6913, 2112)["engine"] in ("register", "hybrid")


@pytest.mark.parametrize("shape", [(0, 64, 32), (1, 0, 32), (1, 64, 0), (1, 64, 33)])
def test_invalid_shapes_rejected(shape):
    with pytest.raises(ValueError):
        choose_config(*shape)


def test_unknown_policy_or_mode_rejected():
    with pytest.raises(ValueError, match="cache_policy"):
        choose_config(1, 64, 256, cache_policy="unknown")
    with pytest.raises(ValueError, match="mode"):
        choose_config(1, 64, 256, "unknown")


@pytest.mark.parametrize(
    "hardware",
    [
        dict(num_processors=8, cache_size_bytes=0, lds_size_bytes=16384),
        dict(num_processors=64, cache_size_bytes=8 * 1024**2, lds_size_bytes=32768),
        dict(num_processors=32, cache_size_bytes=64 * 1024**2, lds_size_bytes=65536),
    ],
)
def test_unseen_shapes_respect_explicit_hardware_budgets(hardware):
    from kernels.gemm.rdna4_mxfp4.heuristic import _geometry

    for shape in ((17, 291, 1376), (97, 12345, 8320), (8193, 7011, 2080)):
        config = choose_config(*shape, **hardware)
        assert _legal_config(config, *shape)
        assert _geometry(config)[-1] <= hardware["lds_size_bytes"]


def test_compile_only_warmup_matches_shape_selection_without_gpu_initialization():
    """Nondefault capacities must survive transfer to compile-only workers."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from kernels.gemm.rdna4_mxfp4 import choose_config, get_warmup_keys
from kernels.gemm.rdna4_mxfp4.entrypoint import specialization_key
assert not torch.cuda.is_initialized()
hardware = dict(num_processors=12, cache_size_bytes=8 * 1024**2, lds_size_bytes=32768)
keys = get_warmup_keys(2053, 1120, 257, cache_policy="rotating", **hardware)
for m in range(1, 258):
    config = choose_config(m, 2053, 1120, cache_policy="rotating", **hardware)
    assert specialization_key(m, 2053, 1120, config=config) in keys
assert not torch.cuda.is_initialized()
""",
        ],
        check=True,
    )


def test_explicit_hybrid_rejects_unsupported_copy_layout_before_compilation():
    """A K160 copy layout compiled but returned incorrect values at K640."""
    from kernels.gemm.rdna4_mxfp4.entrypoint import specialization_key

    with pytest.raises(ValueError, match="power-of-two K tile"):
        specialization_key(33, 3840, 640, config=dict(engine="hybrid", tile_k=160))
