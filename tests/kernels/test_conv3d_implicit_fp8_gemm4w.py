#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the FP8 4-wave (fp8_gemm-pipeline) implicit-GEMM conv3d.

The kernel consumes FP8 (E4M3FN) inputs natively and requires the rigid 4-wave
GEMM constraints: crs % 128 == 0, k % 256 == 0, c % 16 == 0. Tests use only
aligned shapes and compare against the FP8-cast conv reference (the same FP8
tensors cast back to bf16 through torch's conv). Requires CDNA4 (gfx95x).
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_implicit_fp8_gemm4w import conv3d_implicit_fp8_gemm4w

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
_IS_CDNA4 = isinstance(_ARCH, str) and _ARCH.startswith("gfx95")
_skip_no_fp8 = pytest.mark.skipif(not _IS_CDNA4, reason=f"FP8 16x16x128 MFMA needs CDNA4 (gfx95x), got {_ARCH}")


def _fp8(t):
    return t.to(torch.float8_e4m3fn)


# Aligned shapes only: crs = c*kt*kh*kw % 128 == 0, k % 256 == 0.
#   1x3x3, c=256 -> crs=2304 (%128=0); 3x3x3, c=128 -> crs=3456 (%128=0);
#   1x1x1, c=256 -> crs=256.
@_skip_no_fp8
@pytest.mark.parametrize(
    "n,c,t,h,w,k,kt,kh,kw,stride,padding",
    [
        (1, 256, 3, 16, 16, 256, 1, 3, 3, 1, (0, 1, 1)),
        (1, 128, 4, 12, 12, 256, 3, 3, 3, 1, 1),
        (1, 256, 4, 16, 16, 256, 1, 1, 1, 1, 0),
        (1, 256, 3, 8, 9, 512, 1, 3, 3, 1, (0, 1, 1)),  # npq not %256 -> row mask
    ],
)
def test_conv3d_fp8_gemm4w_vs_reference(n, c, t, h, w, k, kt, kh, kw, stride, padding):
    torch.manual_seed(2600 + c + k + h)
    x = _fp8(torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16))
    weight = _fp8(torch.randn((k, c, kt, kh, kw), device="cuda", dtype=torch.bfloat16))

    y = conv3d_implicit_fp8_gemm4w(x, weight, stride=stride, padding=padding)
    ref = F.conv3d(x.to(torch.bfloat16), weight.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"FP8 gemm4w conv rel_err {rel.item():.3e} too high vs FP8-cast reference"


@_skip_no_fp8
def test_conv3d_fp8_gemm4w_bias():
    torch.manual_seed(4242)
    n, c, t, h, w, k = 1, 256, 3, 16, 16, 256
    x = _fp8(torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16))
    weight = _fp8(torch.randn((k, c, 1, 3, 3), device="cuda", dtype=torch.bfloat16))
    bias = torch.randn((k,), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit_fp8_gemm4w(x, weight, bias=bias, stride=1, padding=(0, 1, 1))
    ref = F.conv3d(
        x.to(torch.bfloat16), weight.to(torch.bfloat16), bias=bias.to(torch.bfloat16), stride=1, padding=(0, 1, 1)
    )
    torch.cuda.synchronize()
    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"FP8 gemm4w conv+bias rel_err {rel.item():.3e}"


# 2D via the depth-1 wrapper (aligned: c=256, k=256, 3x3 -> crs=2304).
@_skip_no_fp8
def test_conv2d_fp8_gemm4w():
    torch.manual_seed(5252)
    n, c, h, w, k = 1, 256, 24, 24, 256
    x = _fp8(torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16))
    weight = _fp8(torch.randn((k, c, 3, 3), device="cuda", dtype=torch.bfloat16))

    y = conv3d_implicit_fp8_gemm4w(x, weight, padding=1)
    ref = F.conv2d(x.to(torch.bfloat16), weight.to(torch.bfloat16), padding=1)
    torch.cuda.synchronize()
    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"FP8 gemm4w conv2d rel_err {rel.item():.3e}"
