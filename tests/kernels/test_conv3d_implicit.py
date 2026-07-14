#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 implicit-GEMM conv3d kernel.

Compares ``conv3d_implicit`` against ``torch.nn.functional.conv3d`` on
NCDHW/OIDHW bf16 inputs across stride/padding and M%TILE_M / K%TILE_N tail paths.
Channels must satisfy the kernel's ``c % 8 == 0`` and ``crs = c*kt*kh*kw`` a
multiple of TILE_K (32) constraints.
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
