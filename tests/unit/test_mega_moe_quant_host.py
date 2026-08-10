# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect

import pytest
import torch

from kernels.comm.flydsl_dispatch_combine_intranode_op import FlyDSLDispatchCombineConfig
from kernels.mega_moe import (
    convert_aiter_lqq_to_megamoe as exported_convert_aiter_lqq_to_megamoe,
)
from kernels.mega_moe.mega_moe import (
    MegaMoEV2,
    _combine_launch_geometry,
    _dispatch_quant_config,
)
from kernels.mega_moe.quant import (
    convert_aiter_lqq_to_megamoe,
    repack_megamoe_lqq_for_int8_loader,
)


@pytest.mark.parametrize(
    "quant,expected",
    [
        ("a8w4", (torch.float8_e4m3fn, 8, 1)),
        ("a8w4smooth", (torch.int8, 1, 4)),
        ("w8a8smooth", (torch.int8, 1, 4)),
    ],
)
def test_quant_mode_dispatch_contract(quant, expected):
    assert _dispatch_quant_config(quant, 256) == expected


def test_unknown_quant_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported quant"):
        _dispatch_quant_config("int8", 256)


@pytest.mark.parametrize(
    "quant,mtpr,expected",
    [
        ("a8w4smooth", 1, (64, 4)),
        ("a8w4smooth", 2, (32, 4)),
        ("a8w4smooth", 4, (32, 4)),
        ("a8w4smooth", 8, (64, 4)),
        ("a8w4smooth", 16, (None, None)),
        ("w8a8smooth", 4, (None, None)),
        ("a8w4", 4, (None, None)),
    ],
)
def test_small_a8w4_combine_geometry(quant, mtpr, expected):
    assert _combine_launch_geometry(quant, mtpr) == expected


def test_int8_combine_config_contract():
    cfg = FlyDSLDispatchCombineConfig(
        rank=0,
        world_size=1,
        hidden_dim=256,
        max_num_inp_token_per_rank=16,
        num_experts_per_rank=8,
        num_experts_per_token=2,
        dispatch_dtype=torch.int8,
        combine_dtype=torch.bfloat16,
        scale_dim=1,
        scale_type_size=4,
    )
    assert cfg.dispatch_dtype == torch.int8
    assert cfg.combine_dtype == torch.bfloat16
    assert cfg.scale_bytes == 4


def test_a8w4_constructor_contract_is_unchanged():
    params = inspect.signature(MegaMoEV2.__init__).parameters
    required = (
        "rank",
        "world_size",
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "quant",
        "w1",
        "w1_scale",
        "w2",
        "w2_scale",
        "max_tok_per_rank",
    )
    assert all(params[name].default is inspect.Parameter.empty for name in required)
    assert params["mega_scheme"].default == "fixedslot"
    assert params["swiglu_limit"].default == 0.0
    for name in (
        "w1_lqq_scale",
        "w1_lqq_zero",
        "w2_lqq_scale",
        "w2_lqq_zero",
        "fc1_smooth_scale",
        "fc2_smooth_scale",
    ):
        assert params[name].default is None
    assert params["weight_format"].default == "megamoe"


def test_lqq_conversion_shape_and_layout_formula():
    assert exported_convert_aiter_lqq_to_megamoe is convert_aiter_lqq_to_megamoe
    experts, rows, k_dim = 1, 16, 256
    u4 = (torch.arange(experts * rows * k_dim, dtype=torch.int64) % 16).to(torch.uint8)
    u4 = u4.view(experts, rows, k_dim)
    scale = torch.arange(experts * rows * (k_dim // 64), dtype=torch.uint8).view(
        experts, rows, k_dim // 64
    )
    zero = (255 - scale).to(torch.uint8)

    weight, packed_scale, packed_zero = convert_aiter_lqq_to_megamoe(u4, scale, zero)

    assert weight.dtype == torch.int8
    assert weight.shape == (experts * rows * k_dim // 2,)
    assert packed_scale.shape == packed_zero.shape == (experts, rows // 16, k_dim // 256, 16)
    assert packed_scale.dtype == packed_zero.dtype == torch.int32

    # Independent scalar reference for the (16,16) preshuffle, K64 interleave,
    # and low/high-nibble byte packing.
    shuffled = []
    for k_block in range(k_dim // 32):
        for row in range(rows):
            shuffled.extend(u4[0, row, k_block * 32 : (k_block + 1) * 32].tolist())
    shuffled = torch.tensor(shuffled, dtype=torch.uint8).view(rows, k_dim)
    interleaved = torch.empty_like(shuffled)
    for row in range(rows):
        for chunk in range(k_dim // 128):
            source = shuffled[row, chunk * 128 : (chunk + 1) * 128]
            interleaved[row, chunk * 128 : (chunk + 1) * 128 : 2] = source[:64]
            interleaved[row, chunk * 128 + 1 : (chunk + 1) * 128 : 2] = source[64:]
    expected_weight = (
        interleaved.reshape(-1, 2)[:, 0]
        | (interleaved.reshape(-1, 2)[:, 1] << 4)
    )
    assert torch.equal(weight.view(torch.uint8), expected_weight)

    for row in range(rows):
        expected_scale = sum(int(scale[0, row, group]) << (8 * group) for group in range(4))
        expected_zero = sum(int(zero[0, row, group]) << (8 * group) for group in range(4))
        expected_scale = torch.tensor(expected_scale & 0xFFFFFFFF, dtype=torch.uint32).view(torch.int32)
        expected_zero = torch.tensor(expected_zero & 0xFFFFFFFF, dtype=torch.uint32).view(torch.int32)
        assert packed_scale[0, 0, 0, row] == expected_scale
        assert packed_zero[0, 0, 0, row] == expected_zero


def test_lqq_conversion_rejects_invalid_shape():
    u4 = torch.zeros((1, 16, 128), dtype=torch.uint8)
    qparam = torch.zeros((1, 16, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="K % 256"):
        convert_aiter_lqq_to_megamoe(u4, qparam, qparam)


def test_legacy_lqq_repack_matches_direct_k64_pairs():
    rows = k_dim = 256
    u4 = torch.randint(0, 16, (1, rows, k_dim), dtype=torch.uint8)
    qparam = torch.ones((1, rows, k_dim // 64), dtype=torch.uint8)
    legacy, _, _ = convert_aiter_lqq_to_megamoe(u4, qparam, qparam)
    repacked = repack_megamoe_lqq_for_int8_loader(legacy, rows, k_dim)

    full_layout = (
        u4.view(1, rows // 16, 16, k_dim // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(1, rows // 16, k_dim // 64, 4, 16, 16)
    )
    expected = (
        full_layout[:, :, 0::2] | (full_layout[:, :, 1::2] << 4)
    ).contiguous()
    assert torch.equal(repacked.view(torch.uint8), expected.view(-1))
