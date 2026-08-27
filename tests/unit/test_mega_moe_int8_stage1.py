# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect

import pytest

import kernels.mega_moe.mega_moe_stage1 as stage1_module
import kernels.mega_moe.mega_moe_stage1_main_a8w4 as native_stage1_module
import kernels.mega_moe.mega_moe_stage1_smooth as smooth_stage1_module
from kernels.mega_moe.dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from kernels.mega_moe.mega_moe import Int8Stage1Output
from kernels.mega_moe.mega_moe_stage1 import (
    ENTRY_EPOCH_SLOT_COUNT,
    _entry_epoch_slot,
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
    assert DISPATCH_TABLE_SIZE == 46
    assert DispatchSlot.TOTAL_RECV < DISPATCH_TABLE_SIZE
    assert DispatchSlot.P2P_RECV_NUM < DISPATCH_TABLE_SIZE


def test_entry_epoch_slot_includes_dispatch_width_for_retiring_roles():
    smooth_slots = {
        _entry_epoch_slot(grid_mult, 64, False)
        for grid_mult in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    }
    retiring_slots = {
        _entry_epoch_slot(grid_mult, dispatch_cu, True)
        for grid_mult in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
        for dispatch_cu in range(1, 256)
    }

    assert len(smooth_slots) == 10
    assert len(retiring_slots) == 10 * 255
    assert smooth_slots.isdisjoint(retiring_slots)
    assert max(retiring_slots) < ENTRY_EPOCH_SLOT_COUNT
    assert _entry_epoch_slot(1, 224, True) != _entry_epoch_slot(1, 128, True)


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
    monkeypatch.setattr(native_stage1_module, "get_rocm_arch", lambda: "gfx950")
    monkeypatch.setattr(smooth_stage1_module, "get_rocm_arch", lambda: "gfx950")
    native_kwargs = dict(
        model_dim=3584,
        inter_dim=1280,
        rank=0,
        experts_per_rank=48,
        fuse_npes=8,
        fuse_topk=8,
        fuse_cap=128,
        fuse_mtpr=16,
        fuse_scale_dim=112,
        fixed_slot_dispatch=True,
        sort_block_m=32,
        tile_n=128,
        tile_k=256,
        num_waves=4,
        grid_mult=1,
        num_cu=256,
        num_dispatch_cu=32,
    )
    w8_kwargs = dict(
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
    packed = compile_mega_moe_stage1(**native_kwargs, quant_mode="a8w4")
    full = compile_mega_moe_stage1(**w8_kwargs, quant_mode="w8a8smooth")
    assert callable(packed)
    assert callable(full)
    assert packed is not full


def test_mxfp4_transport_specializes_int8_compute_without_changing_quant_traits(
    monkeypatch,
):
    monkeypatch.setattr(stage1_module, "get_rocm_arch", lambda: "gfx950")
    monkeypatch.setattr(smooth_stage1_module, "get_rocm_arch", lambda: "gfx950")
    kwargs = dict(
        model_dim=3584,
        inter_dim=1280,
        rank=0,
        experts_per_rank=48,
        fuse_npes=8,
        fuse_topk=8,
        fuse_cap=256,
        fuse_mtpr=32,
        fuse_scale_dim=112,
        fixed_slot_dispatch=False,
        sort_block_m=32,
        tile_n=256,
        tile_k=256,
        num_waves=4,
        grid_mult=1,
        num_cu=304,
        num_dispatch_cu=64,
        quant_mode="w8a8smooth",
    )
    compile_mega_moe_stage1.cache_clear()
    baseline = compile_mega_moe_stage1(**kwargs, mxfp4_transport=False)
    transported = compile_mega_moe_stage1(**kwargs, mxfp4_transport=True)
    assert callable(baseline) and callable(transported)
    assert baseline is not transported


def test_a8w4smooth_rejects_mxfp4_transport_before_compilation(monkeypatch):
    monkeypatch.setattr(stage1_module, "get_rocm_arch", lambda: "gfx950")
    with pytest.raises(ValueError, match="does not support MXFP4"):
        compile_mega_moe_stage1(
            model_dim=3584,
            inter_dim=1280,
            rank=0,
            experts_per_rank=48,
            fuse_npes=8,
            fuse_topk=8,
            fuse_cap=512,
            fuse_mtpr=128,
            fuse_scale_dim=1,
            fixed_slot_dispatch=True,
            num_cu=256,
            quant_mode="a8w4smooth",
            smoothquant_mode="bf16_route",
            mxfp4_transport=True,
        )
