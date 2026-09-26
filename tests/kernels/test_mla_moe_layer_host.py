# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CPU checks for indexed MLA + MoE configuration and weight packing."""

import pytest
import torch

from kernels.common.mx_formats import dequantize_mxfp4, quant_dequant_mxfp8, quantize_mxfp4
from kernels.mla_moe_layer.config import (
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    as_moe_mode,
    moe_format,
    validate_shard,
)
from kernels.mla_moe_layer.packing import (
    pack_bf16,
    pack_fp8,
    pack_mxfp4,
)


def test_pack_fp8_uses_mfma_lane_order():
    raw = torch.arange(16 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16, 64)
    packed = pack_fp8(raw.view(torch.float8_e4m3fn)).view(torch.uint8)
    expected = torch.tensor(
        [*range(0, 8), *range(32, 40), *range(64, 72), *range(96, 104)],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_pack_bf16_uses_mfma_lane_order():
    raw = torch.arange(16 * 64, dtype=torch.int16).reshape(16, 64)
    packed = pack_bf16(raw.view(torch.bfloat16)).view(torch.int16)
    expected = torch.tensor([*range(8), *range(64, 72)], dtype=torch.int16)
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_pack_mxfp4_matches_bf16_mfma_k32_steps():
    raw = torch.arange(16 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16, 64)
    packed = pack_mxfp4(raw)
    expected = torch.tensor(
        [*range(0, 4), *range(16, 20), *range(32, 36), *range(48, 52)],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_mxfp4_quantization_round_trips_known_values():
    values = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.0] * 4], dtype=torch.float32)
    packed, scale = quantize_mxfp4(values)
    torch.testing.assert_close(dequantize_mxfp4(packed, scale), values, atol=0, rtol=0)


def test_mxfp8_quantization_uses_independent_32_value_groups():
    values = torch.cat([torch.full((32,), 0.5), torch.full((32,), 64.0)]).reshape(1, 64)
    got = quant_dequant_mxfp8(values)
    expected = torch.cat([torch.full((32,), 0.4375), torch.full((32,), 56.0)]).reshape(1, 64)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize("value", tuple(mode.value for mode in MoeMode))
def test_moe_mode_accepts_public_values(value):
    assert as_moe_mode(value).value == value


def test_moe_modes_map_to_independent_activation_and_weight_formats():
    assert moe_format(MoeMode.W8A8).activation is ExpertActivation.FP8_BLOCK128
    assert moe_format(MoeMode.W8A16).activation is ExpertActivation.BF16
    assert moe_format(MoeMode.A16W4).weight is ExpertWeight.MXFP4_BLOCK32
    assert moe_format(MoeMode.A8W4).activation is ExpertActivation.MXFP8_BLOCK32


@pytest.mark.parametrize(
    "args",
    [
        (3, 8, 0, 1, 2048),
        (16, 8, 0, 1, 2048),
        (1, 10, 0, 1, 2048),
        (1, 8, 1, 1, 2048),
        (1, 8, 0, 3, 2048),
        (1, 8, 0, 1, 2000),
    ],
)
def test_validate_shard_rejects_unsupported_contract(args):
    with pytest.raises(ValueError):
        validate_shard(*args)


def test_validate_shard_accepts_eight_samples():
    validate_shard(8, 8, 0, 1, 2048)
