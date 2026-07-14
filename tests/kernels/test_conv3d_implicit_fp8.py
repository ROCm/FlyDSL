#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the FP8 (E4M3FN) 8-wave implicit-GEMM conv3d kernel.

The kernel quantizes the bf16 inputs to FP8, so it is checked against an
FP8-cast reference (``x.to(float8_e4m3fn)`` / weight likewise) rather than the
full-precision bf16 conv. Requires the CDNA4 (gfx95x) 16x16x128 FP8 MFMA. Only
``c % 16 == 0`` is required; partial M/N/K tiles (NPQ, K, CRS not multiples of
128) are masked, so misaligned channel counts and frame counts are covered too.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
_IS_CDNA4 = isinstance(_ARCH, str) and _ARCH.startswith("gfx95")
_skip_no_fp8 = pytest.mark.skipif(not _IS_CDNA4, reason=f"FP8 16x16x128 MFMA needs CDNA4 (gfx95x), got {_ARCH}")


@_skip_no_fp8
@pytest.mark.parametrize(
    "n,c,t,h,w,k,stride,padding",
    [
        (1, 128, 3, 18, 18, 128, 1, 0),
        (1, 256, 3, 18, 18, 256, 1, 0),
        (1, 128, 3, 16, 16, 256, 1, 1),
        # Partial-tile cases (masked): C=192 -> CRS%128=64, K%128=64;
        # C=96 -> CRS%128=32; NPQ not 128-aligned.
        (1, 192, 6, 16, 20, 192, 1, 1),
        (1, 96, 4, 8, 9, 96, 1, 1),
        (1, 384, 5, 8, 9, 384, 1, 1),
        # K=32 tiny N-tile: split-K forced by JIT cap must predicate the atomic
        # store (WAN VAE conv_out: C384 -> K32).
        (1, 384, 6, 16, 20, 32, 1, 1),
    ],
)
def test_conv3d_fp8_vs_fp8cast_reference(n, c, t, h, w, k, stride, padding):
    torch.manual_seed(2500 + h + w + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit_8wave_fp8(x, weight, stride=stride, padding=padding)
    ref = F.conv3d(
        x.to(torch.float8_e4m3fn).to(torch.bfloat16),
        weight.to(torch.float8_e4m3fn).to(torch.bfloat16),
        stride=stride,
        padding=padding,
    )
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    # Aligned shapes (CRS%128==0): kernel matches FP8-cast reference exactly (<1%).
    # Partial K-tile shapes (CRS%128!=0): the partial K region is zeroed in the
    # kernel but not in the reference, so the bound is the FP8 quantization floor (~5%).
    crs = c * 3 * 3 * 3
    threshold = 5e-2 if crs % 128 != 0 else 1e-2
    assert rel.item() < threshold, f"FP8 conv rel_err {rel.item():.3e} too high vs FP8-cast reference"


# Tile-size sweep on an aligned shape (C, K, CRS all 128-multiples) so the only
# error source is the FP8 quantization floor; each forced tile must match.
@_skip_no_fp8
@pytest.mark.parametrize(
    "tile",
    [
        (128, 128, 2, 4),  # default
        (128, 256, 2, 4),
        (256, 128, 2, 4),
        (256, 256, 2, 4),
        (128, 128, 4, 2),
        (64, 128, 1, 4),
    ],
)
def test_conv3d_fp8_tile_configs(tile):
    torch.manual_seed(3300 + sum(tile))
    n, c, t, h, w, k, stride, padding = 1, 128, 3, 18, 18, 256, 1, 1
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit_8wave_fp8(x, weight, stride=stride, padding=padding, tile=tile)
    ref = F.conv3d(
        x.to(torch.float8_e4m3fn).to(torch.bfloat16),
        weight.to(torch.float8_e4m3fn).to(torch.bfloat16),
        stride=stride,
        padding=padding,
    )
    torch.cuda.synchronize()
    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 6e-2, f"FP8 conv rel_err {rel.item():.3e} too high for tile {tile}"


def _fp8cast(t):
    return t.to(torch.float8_e4m3fn).to(torch.bfloat16)


# 2D FP8 conv via the depth-1 wrapper. NPQ-aligned so only the FP8 quant floor
# contributes (partial-tile masking accuracy is covered by the 3D tests).
@_skip_no_fp8
@pytest.mark.parametrize("kernel_shape,padding", [((3, 3), 1), ((1, 1), 0)])
def test_conv2d_fp8_vs_reference(kernel_shape, padding):
    torch.manual_seed(5100 + sum(kernel_shape))
    n, c, h, w, k = 1, 128, 32, 32, 128
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit_8wave_fp8(x, weight, padding=padding)
    ref = F.conv2d(_fp8cast(x), _fp8cast(weight), padding=padding)
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"FP8 conv2d rel_err {rel.item():.3e}"


# 1D FP8 conv via the depth/height-1 wrapper.
@_skip_no_fp8
@pytest.mark.parametrize("s,padding", [(3, 1), (1, 0)])
def test_conv1d_fp8_vs_reference(s, padding):
    torch.manual_seed(6100 + s)
    n, c, w, k = 1, 128, 256, 128
    x = torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, s), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit_8wave_fp8(x, weight, padding=padding)
    ref = F.conv1d(_fp8cast(x), _fp8cast(weight), padding=padding)
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"FP8 conv1d rel_err {rel.item():.3e}"
