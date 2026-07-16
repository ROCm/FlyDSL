#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the FP8 (E4M3FN) implicit-GEMM conv3d kernel.

The kernel consumes FP8 (E4M3FN) inputs natively, so the tests quantize the bf16
inputs to FP8 and pass the FP8 tensors in, checking against an FP8-cast reference
(the same FP8 tensors cast back to bf16) rather than the full-precision bf16 conv.
Requires the CDNA4 (gfx95x) 16x16x128 FP8 MFMA. Only ``c % 16 == 0`` is required;
partial M/N/K tiles (NPQ, K, CRS not multiples of 128) are masked, so misaligned
channel counts and frame counts are covered too.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_implicit_fp8 import conv3d_implicit_fp8

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
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    y = conv3d_implicit_fp8(x, weight, stride=stride, padding=padding)
    ref = F.conv3d(
        x.to(torch.bfloat16),
        weight.to(torch.bfloat16),
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
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    y = conv3d_implicit_fp8(x, weight, stride=stride, padding=padding, tile=tile)
    ref = F.conv3d(
        x.to(torch.bfloat16),
        weight.to(torch.bfloat16),
        stride=stride,
        padding=padding,
    )
    torch.cuda.synchronize()
    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 6e-2, f"FP8 conv rel_err {rel.item():.3e} too high for tile {tile}"


# Aligned shapes (k%256==0, crs%128==0, c%16==0, crs//128>=2) are routed by the
# public entry to the fp8_gemm 8-wave fast path (conv3d_implicit_fp8_gemm8w). These
# cases exercise that dispatch and confirm the fast kernel matches the FP8-cast ref.
@_skip_no_fp8
@pytest.mark.parametrize(
    "n,c,t,h,w,k,kt,kh,kw,stride,padding",
    [
        (1, 256, 3, 16, 16, 256, 1, 3, 3, 1, (0, 1, 1)),  # crs=2304
        (1, 128, 4, 12, 12, 256, 3, 3, 3, 1, 1),  # crs=3456
        (1, 256, 4, 16, 16, 256, 1, 1, 1, 1, 0),  # crs=256 (min k_iters=2)
    ],
)
def test_conv3d_fp8_gemm8w_dispatch(n, c, t, h, w, k, kt, kh, kw, stride, padding):
    from kernels.conv.conv3d_implicit_fp8 import _gemm8w_eligible

    crs = c * kt * kh * kw
    assert _gemm8w_eligible(k, crs, c), "shape must route to gemm8w"

    torch.manual_seed(2600 + c + k + h)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weight = torch.randn((k, c, kt, kh, kw), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    y = conv3d_implicit_fp8(x, weight, stride=stride, padding=padding)
    ref = F.conv3d(x.to(torch.bfloat16), weight.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"gemm8w-routed FP8 conv rel_err {rel.item():.3e}"


# The gemm8w fast path has no tile sweep (fixed 256x256x128); its only autotune knob
# is WGM (workgroup L2-swizzle). autotune=True must sweep WGM, cache, and stay correct.
@_skip_no_fp8
def test_conv3d_fp8_gemm8w_autotune_wgm(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    torch.manual_seed(2700)
    n, c, t, h, w, k = 1, 256, 3, 16, 16, 256
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weight = torch.randn((k, c, 1, 3, 3), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    y = conv3d_implicit_fp8(x, weight, stride=1, padding=(0, 1, 1), autotune=True)
    y2 = conv3d_implicit_fp8(x, weight, stride=1, padding=(0, 1, 1), autotune=True)  # cache hit
    ref = F.conv3d(x.to(torch.bfloat16), weight.to(torch.bfloat16), stride=1, padding=(0, 1, 1))
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"gemm8w autotune rel_err {rel.item():.3e}"
    assert torch.equal(y, y2), "cached WGM re-run must be deterministic"


def _fp8(t):
    return t.to(torch.float8_e4m3fn)


# 2D FP8 conv via the depth-1 wrapper. NPQ-aligned so only the FP8 quant floor
# contributes (partial-tile masking accuracy is covered by the 3D tests).
@_skip_no_fp8
@pytest.mark.parametrize("kernel_shape,padding", [((3, 3), 1), ((1, 1), 0)])
def test_conv2d_fp8_vs_reference(kernel_shape, padding):
    torch.manual_seed(5100 + sum(kernel_shape))
    n, c, h, w, k = 1, 128, 32, 32, 128
    x = _fp8(torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16))
    weight = _fp8(torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16))

    y = conv3d_implicit_fp8(x, weight, padding=padding)
    ref = F.conv2d(x.to(torch.bfloat16), weight.to(torch.bfloat16), padding=padding)
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
    x = _fp8(torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16))
    weight = _fp8(torch.randn((k, c, s), device="cuda", dtype=torch.bfloat16))

    y = conv3d_implicit_fp8(x, weight, padding=padding)
    ref = F.conv1d(x.to(torch.bfloat16), weight.to(torch.bfloat16), padding=padding)
    torch.cuda.synchronize()

    assert y.shape == ref.shape
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 2e-2, f"FP8 conv1d rel_err {rel.item():.3e}"
