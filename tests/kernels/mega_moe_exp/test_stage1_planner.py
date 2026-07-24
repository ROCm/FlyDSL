# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Backend-agnostic contracts for MegaMoE v2 stage1."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l0_backend_agnostic

_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLANNER = _load_module(
    "_mega_moe_v2_planner_test",
    _ROOT / "kernels/mega_moe/mega_moe_exp/planner.py",
)
_TUNE = _load_module(
    "_mega_moe_v2_tune_test",
    _ROOT / "kernels/mega_moe/mega_moe_exp/group_gemm/tune_config.py",
)
_CALIBRATION = _ROOT / "kernels/comm/mega_moe_tuning_config" / "flydsl_gfx950_mi355x_MegaStage1V2_ep4.json"


def _v4_pro_plan(batch_size, *, use_per_tile_payload_resource=False, tile_m=32):
    return _PLANNER.make_stage1_dispatch_plan(
        batch_size=batch_size,
        npes=8,
        experts_per_rank=48,
        topk=6,
        tile_m=tile_m,
        row_bytes=7168,
        use_per_tile_payload_resource=use_per_tile_payload_resource,
    )


@pytest.mark.parametrize("batch_size", [2048, 4096, 8192])
def test_compact_capacity(batch_size):
    plan = _v4_pro_plan(batch_size)
    assert plan.max_rows == 8 * batch_size * 6 + 48 * 32
    assert plan.epoch_increment == 8
    assert plan.payload_bytes < _PLANNER.BUFFER_OFFSET_ABI_BYTES


@pytest.mark.parametrize("batch_size", [16384, 32768])
def test_large_payload_requires_per_tile_resource(batch_size):
    with pytest.raises(ValueError, match="per-tile payload resource"):
        _v4_pro_plan(batch_size)
    plan = _v4_pro_plan(batch_size, use_per_tile_payload_resource=True)
    assert plan.payload_bytes >= _PLANNER.BUFFER_OFFSET_ABI_BYTES


def test_m128_capacity_includes_per_expert_padding():
    plan = _PLANNER.make_stage1_dispatch_plan(
        batch_size=8192,
        npes=4,
        experts_per_rank=48,
        topk=6,
        tile_m=128,
        row_bytes=7168,
        use_per_tile_payload_resource=True,
    )
    assert plan.max_rows == 4 * 8192 * 6 + 48 * 128


def test_expert_major_tasks_balance_each_producer_round():
    npes, experts_per_rank, dispatch_cu = 4, 48, 32
    first_round = []
    for task in range(dispatch_cu):
        local_expert = task // npes
        destination = task % npes
        global_expert = destination * experts_per_rank + local_expert
        first_round.append((destination, local_expert, global_expert))
    assert [sum(destination == rank for destination, _, _ in first_round) for rank in range(npes)] == [
        8,
        8,
        8,
        8,
    ]
    assert {local_expert for _, local_expert, _ in first_round} == set(range(8))
    assert all(
        global_expert == destination * experts_per_rank + local_expert
        for destination, local_expert, global_expert in first_round
    )


def test_dispatch_table_contracts_have_only_live_slots():
    assert not hasattr(_PLANNER.DispatchSlot, "EXPERT_COUNT")
    assert not hasattr(_PLANNER.DispatchSlot, "P2P_INDEX")
    assert not hasattr(_PLANNER.SmallFixedSlot, "ROUTE_QUEUE")
    assert not hasattr(_PLANNER.SmallFixedSlot, "P2P_INDEX")
    assert max(_PLANNER.DispatchSlot) == _PLANNER.DispatchSlot.PAIR_READY
    assert max(_PLANNER.SmallFixedSlot) == _PLANNER.SmallFixedSlot.P2P_ENTRY_DONE


def test_autotune_space_contains_calibrated_ep4_winner():
    configs = _TUNE.get_stage1_autotune_configs(tile_m_values=(32, 64, 128))
    expected_keys = {
        "sort_block_m",
        "tile_n",
        "tile_k",
        "num_waves",
        "wgm",
        "grid_mult",
        "sched_nmajor",
        "pipe_weights",
        "mfma_amajor",
        "swizzle_a",
        "num_dispatch_cu",
        "tune_use_xcd",
        "use_tile_resource",
        "waves_per_eu_hint",
    }
    assert all(set(config.kwargs) == expected_keys for config in configs)

    calibration = json.loads(_CALIBRATION.read_text())
    row = calibration["megastage1_v2"][0]
    winner = {key: row[key] for key in expected_keys}
    signatures = {tuple(sorted(config.kwargs.items())) for config in configs}
    assert tuple(sorted(winner.items())) in signatures


@pytest.mark.parametrize("batch_size", [8, 64, 512, 1024, 8192])
def test_autotune_pruning_keeps_valid_batch_specific_configs(batch_size):
    configs = _TUNE.get_stage1_autotune_configs(tile_m_values=(32, 64, 128))
    selected = _TUNE.prune_stage1_autotune_configs(
        configs,
        {
            "tune_tokens": batch_size,
            "model_dim": 7168,
            "inter_dim": 3072,
            "num_cu": 256,
            "fuse_npes": 4,
            "fuse_topk": 6,
            "fuse_mtpr": 8192,
            "experts_per_rank": 48,
        },
    )
    assert selected
    assert len(selected) < len(configs)
