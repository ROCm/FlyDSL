#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x W8A16 linear (bf16 WMMA; int8 / FP8 e4m3 / e5m2 weights)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_w8a16_linear import (  # noqa: E402
    fp8_max_for,
    pick_tile_config,
    w8a16_gemm,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"W8A16 linear requires gfx120x, got {ARCH}", allow_module_level=True)


def _ref(a: object, b_nk: object, w_scale: object) -> torch.Tensor:
    # Offline dequant ref (W8A16 linear contract — not HIP dyn-quant bits).
    w = b_nk.float() * w_scale.reshape(-1, 1)
    return (a.float() @ w.T).to(a.dtype)


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(32, 64, 64, id="tiny-32x64x64"),
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(37, 70, 80, id="ragged-37x70x80"),
    ],
)
def test_w8a16_linear_vs_offline_ref(m: int, n: int, k: int) -> None:
    torch.manual_seed(20260930 + m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    ws = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    out = w8a16_gemm(a, b, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = _ref(a, b, ws)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


def test_w8a16_linear_fp8_e4m3fn_smoke() -> None:
    """One-shape smoke: float8_e4m3fn weights → float WMMA (cast in-reg)."""
    m, n, k = 32, 64, 64
    torch.manual_seed(20260930 + 41)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    # Quantize small bf16 weights into e4m3 with per-N scale (max 448).
    w_f = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    fp8_max = fp8_max_for(torch.float8_e4m3fn)
    ws = (w_f.abs().amax(dim=1) / fp8_max).clamp_min(1e-6).contiguous()
    b = (w_f / ws.reshape(-1, 1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    out = w8a16_gemm(a, b, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = _ref(a, b, ws)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


def test_w8a16_linear_fp8_e5m2_smoke() -> None:
    """One-shape smoke: float8_e5m2 weights → float WMMA (cast in-reg)."""
    m, n, k = 32, 64, 64
    torch.manual_seed(20260930 + 52)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_f = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    fp8_max = fp8_max_for(torch.float8_e5m2)
    ws = (w_f.abs().amax(dim=1) / fp8_max).clamp_min(1e-6).contiguous()
    b = (w_f / ws.reshape(-1, 1)).clamp(-fp8_max, fp8_max).to(torch.float8_e5m2)
    out = w8a16_gemm(a, b, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = _ref(a, b, ws)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


def test_w8a16_linear_tile_pick() -> None:
    """Exact tile fields for each pick_tile_config branch so a wrong arm fails."""
    # Skinny (M<=64 or N<=64) → 64x64x64 w2x2 t2x2
    cfg = pick_tile_config(32, 64, 64, wgps=48)
    assert (cfg.bm, cfg.bn, cfg.bk, cfg.warps_m, cfg.warps_n, cfg.tm, cfg.tn) == (64, 64, 64, 2, 2, 2, 2)

    # Non-skinny, blocks_128 < wgps → 128x128x64 w4x2 t2x4
    cfg = pick_tile_config(128, 128, 128, wgps=48)
    assert (cfg.bm, cfg.bn, cfg.bk, cfg.warps_m, cfg.warps_n, cfg.tm, cfg.tn) == (128, 128, 64, 4, 2, 2, 4)

    # Non-skinny, blocks_128 >= wgps → 256x128x64 w4x2 t4x4
    cfg = pick_tile_config(256, 256, 256, wgps=4)
    assert (cfg.bm, cfg.bn, cfg.bk, cfg.warps_m, cfg.warps_n, cfg.tm, cfg.tn) == (256, 128, 64, 4, 2, 4, 4)


def test_w8a16_gemm_e5m2_odd_k_stays_in_kernel() -> None:
    """Odd K stays in the kernel for e5m2 weights (no host pad, never rejects)."""
    m, n, k = 32, 64, 24  # 24 % 16 != 0
    torch.manual_seed(20261006)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_f = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    fp8_max = fp8_max_for(torch.float8_e5m2)
    ws = (w_f.abs().amax(dim=1) / fp8_max).clamp_min(1e-6).contiguous()
    b = (w_f / ws.reshape(-1, 1)).clamp(-fp8_max, fp8_max).to(torch.float8_e5m2)
    out = w8a16_gemm(a, b, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = _ref(a, b, ws)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


def test_w8a16_k0_is_zeros_plus_bias() -> None:
    a = torch.empty((3, 0), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((5, 0), device="cuda", dtype=torch.int8)
    ws = torch.ones(5, device="cuda")
    bias = torch.randn(5, device="cuda", dtype=torch.bfloat16)
    out = w8a16_gemm(a, b, ws, bias=bias)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), bias.float().unsqueeze(0).expand(3, 5).contiguous())
