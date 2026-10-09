# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CPU checks of the MXFP8 quantizers that the kernel tests use for their inputs and references."""

import pytest
import torch

from tests.kernels.utils import gemm_common_utils as gcu

pytestmark = [pytest.mark.l0_backend_agnostic]

E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _blocks(rows, cols, block_m, block_k):
    """Values in [-1, 1]; every (block_m, block_k) block has max |x| = (1 + j / 64) * 448 * 2^k for j < 64, |k| <= 10."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(rows, cols, generator=g, device="cpu") * 2 - 1
    x[::block_m, ::block_k] = -1.0
    i = torch.arange(rows // block_m, dtype=torch.float32, device="cpu")
    row_scale = (1 + i.remainder(64) / 64) * E4M3_MAX * torch.pow(2.0, i.remainder(21) - 10)
    return x * row_scale.repeat_interleave(block_m).unsqueeze(1)


def _assert_e4m3_rounding(x, dq, scale):
    """Each element is off by at most half an E4M3 step: 2^-4 * |x| for normals, 2^-10 * scale for subnormals."""
    bound = torch.maximum(x.abs() * 2.0**-4, scale * 2.0**-10) * (1 + 1e-6)
    ratio = (dq - x).abs() / bound
    assert bool((ratio <= 1).all()), f"error up to {ratio.max().item():.2f}x half an E4M3 step"


def test_per_1x32_f8_quant_does_not_saturate():
    x = _blocks(1344, 64, 1, 32)
    x_q, scale = gcu.per_1x32_f8_quant(x)
    scale_f32 = gcu.e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
    _assert_e4m3_rounding(x, x_q.float() * scale_f32, scale_f32)


def test_per_block_f8_quant_does_not_saturate():
    x = _blocks(1344 * 4, 256, 4, 128)
    x_q, scale = gcu.per_block_f8_quant(x, 4, 128)
    scale_f32 = gcu.e8m0_to_f32(scale).repeat_interleave(4, dim=0).repeat_interleave(128, dim=1)
    _assert_e4m3_rounding(x, x_q.float() * scale_f32, scale_f32)
