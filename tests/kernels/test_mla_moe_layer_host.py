# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CPU checks for shared/reuse layer configuration and weight packing."""

import pytest
import torch

from kernels.mla_moe_layer.config import MoeMode, as_moe_mode, validate_shard
from kernels.mla_moe_layer.packing import pack_bf16, pack_fp8


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


@pytest.mark.parametrize("value", tuple(mode.value for mode in MoeMode))
def test_moe_mode_accepts_public_values(value):
    assert as_moe_mode(value).value == value


@pytest.mark.parametrize(
    "args",
    [
        (3, 8, 0, 1, 2048),
        (1, 10, 0, 1, 2048),
        (1, 8, 1, 1, 2048),
        (1, 8, 0, 3, 2048),
        (1, 8, 0, 1, 2000),
    ],
)
def test_validate_shard_rejects_unsupported_contract(args):
    with pytest.raises(ValueError):
        validate_shard(*args)
