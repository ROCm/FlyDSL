#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 8-wave implicit-GEMM conv3d kernel.

Compares ``conv3d_implicit_8wave`` against ``torch.nn.functional.conv3d`` on
NCDHW/OIDHW bf16 inputs across stride/padding and M%TILE_M / K%TILE_N tail paths.
Channels must satisfy the kernel's ``c % 8 == 0`` and ``crs = c*kt*kh*kw`` a
multiple of TILE_K (32) constraints.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels.conv3d_implicit_8wave import conv3d_implicit_8wave

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


# (N, C, T, H, W, K), kernel 3x3x3. Covers stride/padding and tile-tail paths.
@pytest.mark.parametrize(
    "n,c,t,h,w,k,stride,padding",
    [
        (1, 32, 8, 16, 16, 64, 1, 0),
        (1, 32, 9, 17, 17, 96, 1, 1),
        (2, 64, 6, 18, 18, 192, 1, 1),
        (1, 32, 10, 20, 20, 64, 2, 1),
    ],
)
def test_conv3d_vs_torch(n, c, t, h, w, k, stride, padding):
    torch.manual_seed(2000 + h + w + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit_8wave(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)
