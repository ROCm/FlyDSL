# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU selector invariants: unseen shapes, tails, K factors and cache hints."""

import builtins

import pytest

from kernels.gemm.rdna4_mxfp4.dispatch import _candidates, _model, _selected, choose_config, tuning_candidates
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


@pytest.mark.parametrize("shape", [(7, 6147, 2464), (40, 5120, 8704), (80, 17408, 5120), (1, 17, 32)])
@pytest.mark.parametrize("cache_policy", ["auto", "hot", "rotating"])
def test_calibration_shortlist_is_bounded_legal_and_contains_proposal(shape, cache_policy):
    candidates = tuning_candidates(*shape, cache_policy=cache_policy)
    assert 1 <= len(candidates) <= 64
    assert len({tuple(sorted(c.items())) for c in candidates}) == len(candidates)
    assert all(legal_config(c, *shape) for c in candidates)
    assert choose_config(*shape, cache_policy=cache_policy) in candidates
    assert {c["engine"] for c in candidates} == {"register", "hybrid"}


def test_calibration_can_explore_odd_k_factors_beyond_static_tail_guard():
    candidates = tuning_candidates(7, 6147, 2464)
    assert any(c["engine"] == "register" and c["k_unroll"] == 7 for c in candidates)
    assert any(c["engine"] == "register" and c["k_unroll"] == 11 for c in candidates)


def _one_profile(m, n, k, config):
    return {
        "candidates": [config],
        "families": {f"{n}:{k}": [[m, 0, 0, 0]]},
        "neighbors": 2,
        "grid_weight": 0.2,
    }


def test_profile_cannot_extrapolate_a_small_launch_to_large_prefill():
    """The old M129 -> M17021 transfer caused a measured skinny-GEMM cliff."""
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt
    from kernels.gemm.rdna4_mxfp4.dispatch_features import REGISTER_DEFAULTS
    from kernels.gemm.rdna4_mxfp4.profile_policy import profile_choice

    config = dict(REGISTER_DEFAULTS, waves_m=1, waves_n=1, repeat_m=1,
                  repeat_n=1, k_unroll=1)
    model = _one_profile(129, 23, 992, config)
    assert profile_choice(model, 257, 23, 992, "hot", _adapt) is not None
    assert profile_choice(model, 17021, 23, 992, "hot", _adapt) is None


@pytest.mark.parametrize("query", [(40, 192), (64, 256)])
def test_profile_transfer_rejects_changed_register_tail_or_k_loop(query):
    """Nearby dimensions alone do not preserve a measured register strategy."""
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt
    from kernels.gemm.rdna4_mxfp4.dispatch_features import REGISTER_DEFAULTS
    from kernels.gemm.rdna4_mxfp4.profile_transfer import transferred_choice

    config = dict(REGISTER_DEFAULTS, waves_m=1, waves_n=4, repeat_m=1,
                  repeat_n=1, k_unroll=3)
    model = _one_profile(17, 64, 192, config)
    assert transferred_choice(model, 17, 128, 192, "hot", _adapt, legal_config)
    assert transferred_choice(model, 17, *query, "hot", _adapt, legal_config) is None


def test_profile_transfer_rejects_crossing_the_footprint_guard():
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt
    from kernels.gemm.rdna4_mxfp4.dispatch_features import REGISTER_DEFAULTS
    from kernels.gemm.rdna4_mxfp4.profile_transfer import transferred_choice

    config = dict(REGISTER_DEFAULTS, waves_m=1, waves_n=4, repeat_m=1,
                  repeat_n=1, k_unroll=1)
    model = _one_profile(129, 8192, 8192, config)
    assert transferred_choice(model, 129, 16384, 8192, "hot", _adapt, legal_config) is None


def test_profile_transfer_checks_new_unroll_opportunities():
    """Unroll1 remaining legal does not justify ignoring a newly available13."""
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt
    from kernels.gemm.rdna4_mxfp4.dispatch_features import REGISTER_DEFAULTS
    from kernels.gemm.rdna4_mxfp4.profile_transfer import transferred_choice

    config = dict(REGISTER_DEFAULTS, waves_m=1, waves_n=4, repeat_m=1,
                  repeat_n=1, k_unroll=1)
    model = _one_profile(17, 64, 1184, config)
    assert transferred_choice(model, 17, 64, 1312, "hot", _adapt, legal_config)
    assert transferred_choice(model, 17, 64, 1248, "hot", _adapt, legal_config) is None


@pytest.mark.parametrize("n", [24, 96, 128])
def test_skinny_aligned_query_uses_aligned_profile_source(n):
    """A nearby odd M can hide the register winner at an aligned prefill."""
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt
    from kernels.gemm.rdna4_mxfp4.dispatch_features import HYBRID_DEFAULTS
    from kernels.gemm.rdna4_mxfp4.profile_policy import profile_choice

    config = dict(HYBRID_DEFAULTS, tile_m=128, tile_n=32, tile_k=256,
                  waves_m=4, waves_n=1, schedule="groups")
    model = _one_profile(1345, n, 5120, config)
    # Both inputs use the same ragged-M executable, so specialization alone
    # does not detect this bad source for the profile's measured strategy.
    assert profile_choice(model, 1344, n, 5120, "hot", _adapt) is None
    model["families"][f"{n}:5120"].insert(0, [1328, 0, 0, 0])
    assert profile_choice(model, 1344, n, 5120, "hot", _adapt)[0] == 1328


@pytest.mark.parametrize("m,n", [(1343, 24), (1344, 129), (1345, 24)])
def test_alignment_source_guard_preserves_other_profile_queries(m, n):
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt
    from kernels.gemm.rdna4_mxfp4.dispatch_features import HYBRID_DEFAULTS
    from kernels.gemm.rdna4_mxfp4.profile_policy import profile_choice

    config = dict(HYBRID_DEFAULTS, tile_m=128, tile_n=32, tile_k=256,
                  waves_m=4, waves_n=1, schedule="groups")
    model = _one_profile(1345, n, 5120, config)
    assert profile_choice(model, m, n, 5120, "hot", _adapt)[0] == 1345


@pytest.mark.parametrize("m", [17, 31, 47, 63])
def test_cost_pool_retains_measured_repeat_one_odd_divisor_winner(m):
    """The KU5/RM1 family was missing despite beating the selected tile."""
    from kernels.gemm.rdna4_mxfp4.cost_candidates import expanded_candidates
    from kernels.gemm.rdna4_mxfp4.dispatch import _adapt, _proposal_regional
    from kernels.gemm.rdna4_mxfp4.dispatch_features import HYBRID_DEFAULTS, REGISTER_DEFAULTS

    shape = (m, 4091, 1760)
    pool = expanded_candidates(*shape, lambda hint: _proposal_regional(*shape, hint),
                               _adapt, legal_config, REGISTER_DEFAULTS, HYBRID_DEFAULTS)
    for table in ("alu", "lut"):
        assert dict(REGISTER_DEFAULTS, waves_m=1, waves_n=4, repeat_m=1,
                    repeat_n=1, k_unroll=5, table_mode=table) in pool
