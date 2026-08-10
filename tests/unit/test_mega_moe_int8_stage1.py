# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect

import pytest

import kernels.mega_moe.mega_moe_stage1 as stage1_module
from kernels.mega_moe.dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from kernels.mega_moe.mega_moe import Int8Stage1Output
from kernels.mega_moe.mega_moe_stage1 import (
    _stage1_quant_traits,
    compile_mega_moe_stage1,
)


@pytest.mark.parametrize(
    "mode,packed,n_tiles,mx_scale_lds",
    [
        ("a8w4", False, 2, True),
        ("a8w4smooth", True, 1, False),
        ("w8a8smooth", False, 1, False),
    ],
)
def test_stage1_quant_geometry(mode, packed, n_tiles, mx_scale_lds):
    traits = _stage1_quant_traits(mode, inter_dim=256, tile_n=256)
    assert traits["packed_int4"] is packed
    assert traits["n_tiles"] == n_tiles
    assert traits["uses_mx_scale_lds"] is mx_scale_lds


def test_int8_dispatch_table_has_exact_recv_slots():
    assert DISPATCH_TABLE_SIZE == max(DispatchSlot) + 1
    assert DispatchSlot.TOTAL_RECV < DISPATCH_TABLE_SIZE
    assert DispatchSlot.P2P_RECV_NUM < DISPATCH_TABLE_SIZE


def test_stage1_cache_key_contains_quant_mode_and_qparams():
    parameters = inspect.signature(compile_mega_moe_stage1).parameters
    assert parameters["quant_mode"].default == "a8w4"
    assert "quant_mode" in parameters
    assert {
        "compact_src",
        "compact_experts",
        "compact_weights",
        "qscale_w",
        "qzero_w",
    }.issubset(
        inspect.signature(
            __import__(
                "kernels.mega_moe.mega_moe_stage1",
                fromlist=["run_mega_moe_stage1"],
            ).run_mega_moe_stage1
        ).parameters
    )


def test_int8_stage1_output_exposes_stage2_metadata():
    assert set(Int8Stage1Output.__dataclass_fields__) == {
        "a2",
        "sorted_token_ids",
        "sorted_expert_ids",
        "sorted_weights",
        "num_valid_ids",
        "sort_block_m",
    }


def test_int8_launcher_construction_is_quant_specialized(monkeypatch):
    monkeypatch.setattr(stage1_module, "get_rocm_arch", lambda: "gfx950")
    kwargs = dict(
        model_dim=256,
        inter_dim=256,
        rank=0,
        experts_per_rank=8,
        fuse_npes=1,
        fuse_topk=2,
        fuse_cap=32,
        fuse_mtpr=16,
        fuse_scale_dim=4,
        fixed_slot_dispatch=False,
        sort_block_m=32,
        tile_n=256,
        tile_k=256,
        num_waves=4,
        grid_mult=1,
        num_cu=304,
        num_dispatch_cu=64,
    )
    compile_mega_moe_stage1.cache_clear()
    packed = compile_mega_moe_stage1(**kwargs, quant_mode="a8w4smooth")
    full = compile_mega_moe_stage1(**kwargs, quant_mode="w8a8smooth")
    assert callable(packed)
    assert callable(full)
    assert packed is not full
