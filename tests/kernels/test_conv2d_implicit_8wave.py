#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 8-wave implicit-GEMM conv2d kernel.

Compares ``conv2d_implicit_8wave`` against ``torch.nn.functional.conv2d`` on
NCHW/KCRS bf16 inputs across stride/padding and M%TILE_M / K%TILE_N tail paths.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv2d_implicit_mfma import conv2d_implicit_8wave

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
# mfma_f32_16x16x32_bf16 is only available on CDNA4 (gfx95x)
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.startswith("gfx95")),
    reason=f"conv2d 8-wave BF16 needs mfma_f32_16x16x32_bf16 (CDNA4 gfx95x), got {_ARCH}",
)


# (N, C, H, W, K, R, S, stride, padding)
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,h,w,k,r,s,stride,padding",
    [
        # clean tile, valid conv
        (1, 32, 34, 34, 128, 3, 3, 1, 0),
        # same-size conv (pad=1)
        (1, 32, 32, 32, 128, 3, 3, 1, 1),
        # K tail (K not multiple of TILE_N=128)
        (1, 32, 34, 34, 96, 3, 3, 1, 0),
        # M tail (NPQ not multiple of TILE_M=128)
        (1, 32, 17, 17, 128, 3, 3, 1, 0),
        # stride=2
        (1, 32, 66, 66, 128, 3, 3, 2, 1),
        # larger C and K
        (2, 64, 34, 34, 256, 3, 3, 1, 0),
        # 1×1 conv (CRS = C, needs C%32==0)
        (1, 128, 32, 32, 256, 1, 1, 1, 0),
        # bias
        (1, 32, 34, 34, 128, 3, 3, 1, 0),
    ],
)
def test_conv2d_vs_torch(n, c, h, w, k, r, s, stride, padding):
    torch.manual_seed(1000 + h + w + k)
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, r, s), device="cuda", dtype=torch.bfloat16)
    use_bias = k == 128 and h == 34 and r == 3 and stride == 1 and padding == 0 and n == 1 and c == 32
    bias = torch.randn((k,), device="cuda", dtype=torch.float32) if use_bias else None

    y = conv2d_implicit_8wave(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv2d(
        x,
        weight,
        bias=bias.to(torch.bfloat16) if bias is not None else None,
        stride=stride,
        padding=padding,
    )
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape, f"shape mismatch: got {y.shape}, expected {y_ref.shape}"
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2), f"max abs err: {(y - y_ref).abs().max().item():.4f}"
