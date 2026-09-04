#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 implicit-GEMM conv3d kernel.

Compares ``conv3d_implicit`` against ``torch.nn.functional.conv3d`` on
NCDHW/OIDHW bf16 inputs across stride/padding and M%TILE_M / K%TILE_N tail paths.
Any channel count and spatial extent is supported.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_implicit import conv3d_implicit

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
# mfma_f32_16x16x32_bf16 is only available on CDNA4 (gfx95x)
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.startswith("gfx95")),
    reason=f"conv3d BF16 needs mfma_f32_16x16x32_bf16 (CDNA4 gfx95x), got {_ARCH}",
)


# (N, C, T, H, W, K), kernel 3x3x3. Covers stride/padding and tile-tail paths.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,t,h,w,k,stride,padding",
    [
        (1, 32, 8, 16, 16, 64, 1, 0),
        (1, 32, 9, 17, 17, 96, 1, 1),
        (2, 64, 6, 18, 18, 192, 1, 1),
        (1, 32, 10, 20, 20, 64, 2, 1),
        # Partial K-tile: C=16 -> CRS=432, 432 % TILE_K(32) = 16 (masked).
        (1, 16, 6, 16, 20, 16, 1, 1),
        (1, 16, 4, 12, 16, 384, 1, 1),
        (1, 3, 4, 12, 12, 32, 1, 1),
        (1, 12, 4, 12, 12, 32, 1, 1),
        (2, 5, 4, 10, 14, 48, 1, 1),
        (1, 6, 3, 11, 11, 32, 1, 1),
    ],
)
def test_conv3d_vs_torch(n, c, t, h, w, k, stride, padding):
    torch.manual_seed(2000 + h + w + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,padding",
    [
        ((1, 3, 3), (0, 1, 1)),
        ((3, 1, 1), (1, 0, 0)),
    ],
)
def test_conv3d_factorized_filters_vs_torch(kernel_shape, padding):
    """Cover the spatial-only and temporal-only filter dispatch paths."""
    torch.manual_seed(3100 + sum(kernel_shape))
    n, c, t, h, w, k = 1, 64, 6, 18, 20, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=1, padding=padding)
    y_ref = F.conv3d(x, weight, stride=1, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize("c", [16, 64])
def test_conv3d_runtime_k_loop_short_problems(c):
    """Exercise one- and two-K-tile runtime-pipeline epilogues."""
    torch.manual_seed(3200 + c)
    n, t, h, w, k = 1, 3, 8, 8, 64
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 1, 1, 1), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight)
    y_ref = F.conv3d(x, weight)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Tile-size sweep: each forced (TILE_M, TILE_N, WAVE_M, WAVE_N) must stay correct.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "tile",
    [
        (128, 128, 2, 4),  # default
        (128, 256, 2, 4),
        (256, 128, 2, 4),
        (256, 256, 2, 4),
        (256, 256, 4, 4),
        (128, 128, 4, 2),
        (64, 128, 1, 4),
        (64, 64, 2, 2),
    ],
)
def test_conv3d_tile_configs(tile):
    torch.manual_seed(4000 + sum(tile))
    n, c, t, h, w, k, stride, padding = 2, 64, 6, 18, 18, 192, 1, 1
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding, tile=tile)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
def test_conv3d_autotune(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "at"))
    from kernels.conv import conv3d_autotune

    conv3d_autotune._MEM_CACHE.clear()

    torch.manual_seed(4242)
    n, c, t, h, w, k = 1, 128, 6, 40, 40, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=1, padding=1, autotune=True)
    y_ref = F.conv3d(x, weight, stride=1, padding=1)
    torch.cuda.synchronize()
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)

    # A tile was chosen and persisted; the second call must hit the cache.
    assert len(conv3d_autotune._MEM_CACHE) == 1
    calls = {"n": 0}
    orig = conv3d_autotune.do_bench

    def _counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(conv3d_autotune, "do_bench", _counting)
    y2 = conv3d_implicit(x, weight, stride=1, padding=1, autotune=True)
    torch.cuda.synchronize()
    assert torch.allclose(y2, y_ref, rtol=2e-2, atol=2e-2)
    assert calls["n"] == 0  # cached, no re-benchmark


# 2D conv via the depth-1 degenerate path through the 3D kernel.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,stride,padding",
    [
        ((3, 3), 1, 1),
        ((1, 1), 1, 0),  # 1x1 -> temporal_only_fast-style vectorized epilogue
        ((5, 5), 1, 2),
        ((3, 3), 2, 1),
    ],
)
def test_conv2d_vs_torch(kernel_shape, stride, padding):
    torch.manual_seed(5000 + sum(kernel_shape) + stride + padding)
    n, c, h, w, k = 2, 64, 24, 28, 128
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv2d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# Unaligned channel counts and spatial extents.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "c,h,w,k,kernel_shape,stride,padding",
    [
        (3, 32, 32, 64, (3, 3), 1, 1),
        (3, 24, 28, 32, (7, 7), 2, 3),
        (1, 24, 24, 32, (3, 3), 1, 1),
        (12, 16, 16, 32, (3, 3), 1, 1),
        (64, 33, 33, 64, (3, 3), 2, 0),
        (128, 17, 17, 64, (3, 3), 2, 0),
        (6, 21, 21, 32, (3, 3), 1, 1),
    ],
)
def test_conv2d_unaligned_channels_and_spatial(c, h, w, k, kernel_shape, stride, padding):
    torch.manual_seed(7000 + c + h + k)
    x = torch.randn((1, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=stride, padding=padding)
    y_ref = F.conv2d(x, weight, stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# The transpose needs C aligned to the vector width; S may be anything.
@_skip_non_cdna4
@pytest.mark.parametrize("c", [8, 16, 64, 512])
@pytest.mark.parametrize("h,w", [(3, 3), (5, 5), (17, 15), (33, 33), (8, 8)])
def test_transpose_unaligned_spatial(c, h, w):
    from kernels.conv.conv3d_implicit import _ncdhw_to_ndhwc

    torch.manual_seed(8000 + c + h * w)
    x = torch.randn((1, c, 1, h, w), device="cuda", dtype=torch.bfloat16)

    got = _ncdhw_to_ndhwc(x, torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(got, x.permute(0, 2, 3, 4, 1).contiguous())


# 1D conv via the depth/height-1 degenerate path through the 3D kernel.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "s,stride,padding",
    [
        (3, 1, 1),
        (1, 1, 0),
        (5, 2, 2),
    ],
)
def test_conv1d_vs_torch(s, stride, padding):
    torch.manual_seed(6000 + s + stride + padding)
    n, c, w, k = 2, 64, 96, 128
    x = torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, s), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv1d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


# ---- Qwen-Image VAE classic conv (T2I T=1) ---------------------------------
# CausalConv3d 3x3x3 degenerates to conv2d with weight[:, :, 2, :, :].
# Shapes below are the 1024x1024 spatial ladder plus the two hottest layers of
# the 1328x1328 default resolution, taken from forward-hook traces of
# AutoencoderKLQwenImage rather than from the config alone: the decoder halves
# its channel count inside the UpBlock loop (in_dim // 2) before each stage, so
# its ResBlock channel pairs coincide with the encoder ones instead of
# continuing 384 -> 192 -> 96 -> 48.


_QWENIMAGE_T1_RES3 = [
    pytest.param(3, 96, 1024, 1024, id="enc_conv_in"),
    pytest.param(96, 96, 1024, 1024, id="enc_e0_res__dec_d3_res"),
    pytest.param(96, 192, 512, 512, id="enc_e1_res1"),
    pytest.param(192, 192, 512, 512, id="enc_e1_res2__dec_d2_res"),
    pytest.param(192, 384, 256, 256, id="enc_e2_res1__dec_d1_res1"),
    pytest.param(384, 384, 256, 256, id="enc_e2_res2__dec_d1_res"),
    pytest.param(384, 384, 128, 128, id="enc_e3_mid__dec_mid_d0"),
    pytest.param(384, 32, 128, 128, id="enc_conv_out"),
    pytest.param(16, 384, 128, 128, id="dec_conv_in"),
    pytest.param(96, 3, 1024, 1024, id="dec_conv_out"),
    pytest.param(384, 384, 166, 166, id="dec_bottleneck_1328"),
    pytest.param(96, 96, 1328, 1328, id="dec_d3_res_hot_1328"),
]

# Resample downsample2d: ZeroPad2d((0, 1, 0, 1)) then Conv2d(k=3, s=2, p=0).
_QWENIMAGE_DOWN2D = [
    pytest.param(96, 1024, 1024, id="enc_e0_downsample"),
    pytest.param(192, 512, 512, id="enc_e1_downsample_spatial"),
    pytest.param(384, 256, 256, id="enc_e2_downsample_spatial"),
]

# Resample upsample2d/3d: nearest-exact x2 happens outside the kernel, so the
# conv runs at the already-doubled resolution with Conv2d(dim, dim // 2, k=3,
# s=1, p=1).
_QWENIMAGE_UP2D = [
    pytest.param(384, 192, 256, 256, id="dec_d0_upsample"),
    pytest.param(384, 192, 512, 512, id="dec_d1_upsample"),
    pytest.param(192, 96, 1024, 1024, id="dec_d2_upsample"),
]


@_skip_non_cdna4
@pytest.mark.parametrize("c_in,c_out,h,w", _QWENIMAGE_T1_RES3)
def test_qwenimage_vae_t1_res3_bf16(c_in, c_out, h, w):
    torch.manual_seed(8800 + c_in + c_out + h + w)
    x2 = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.bfloat16)
    weight5 = torch.randn((c_out, c_in, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    weight2 = weight5[:, :, 2, :, :]
    bias = torch.randn((c_out,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x2, weight2, bias=bias, stride=1, padding=1)
    y_ref = F.conv2d(x2, weight2, bias=bias.to(torch.bfloat16), stride=1, padding=1)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape == (1, c_out, h, w)
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize("c,h,w", _QWENIMAGE_DOWN2D)
def test_qwenimage_vae_downsample2d_bf16(c, h, w):
    torch.manual_seed(9100 + c + h + w)
    x2 = torch.randn((1, c, h, w), device="cuda", dtype=torch.bfloat16)
    x_pad = F.pad(x2, (0, 1, 0, 1))
    weight = torch.randn((c, c, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((c,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x_pad, weight, bias=bias, stride=2, padding=0)
    y_ref = F.conv2d(x_pad, weight, bias=bias.to(torch.bfloat16), stride=2, padding=0)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
@pytest.mark.parametrize("c_in,c_out,h,w", _QWENIMAGE_UP2D)
def test_qwenimage_vae_upsample2d_bf16(c_in, c_out, h, w):
    torch.manual_seed(9400 + c_in + c_out + h + w)
    x2 = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((c_out,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x2, weight, bias=bias, stride=1, padding=1)
    y_ref = F.conv2d(x2, weight, bias=bias.to(torch.bfloat16), stride=1, padding=1)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape == (1, c_out, h, w)
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)
