# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU selector invariants: unseen shapes, tails, K factors and cache hints."""

import builtins

import pytest

from kernels.gemm.rdna4_mxfp4.dispatch import _candidates, _model, _selected, choose_config
from kernels.gemm.rdna4_mxfp4.dispatch_features import legal_config

pytestmark = pytest.mark.l0_backend_agnostic


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
    assert legal_config(cfg, *shape)
    if cfg["engine"] == "register":
        assert shape[2] % (32 * cfg["k_unroll"]) == 0
        if not cfg["buffer_io"]:
            assert shape[0] % (16 * cfg["waves_m"] * cfg["repeat_m"]) == 0
            assert shape[1] % (16 * cfg["waves_n"] * cfg["repeat_n"]) == 0
    else:
        assert shape[2] % cfg["tile_k"] == 0
        assert 0 <= cfg["operand_prefetch"] < cfg["tile_k"] // 16


@pytest.mark.parametrize("shape", [(1, 17, 32), (73, 4099, 1568), (257, 1536, 544)])
def test_all_candidates_obey_factory_requirements(shape):
    candidates = list(_candidates(*shape))
    assert candidates
    assert {c["engine"] for c in candidates} <= {"register", "hybrid"}
    assert all(legal_config(c, *shape) for c in candidates)


@pytest.mark.parametrize("shape", [(17, 513, 544), (192, 3840, 3840), (8193, 4097, 1568)])
def test_grouped_fallback_preserves_k32_scaling(shape):
    cfg = choose_config(*shape, "grouped")
    assert cfg["engine"] == "register"
    assert cfg["k_unroll"] == 1
    assert legal_config(cfg, *shape)


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

    _model.cache_clear()
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
