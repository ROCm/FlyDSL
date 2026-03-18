#!/usr/bin/env python3
"""WMMA FP8 GEMM correctness tests for RDNA4 (gfx12xx, wave32).

Kernel implementation: kernels/wmma_fp8_gemm.py
"""

import os
import sys
import logging

import torch
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.wmma_fp8_gemm import (
    compile_fp8_gemm,
    preshuffle_b_fp8,
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
)
from tests.test_common import verify_output
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna4():
    if "gfx12" not in ARCH:
        pytest.skip(f"WMMA FP8 GEMM requires RDNA4 (gfx12xx), got {ARCH}")


def _run_fp8_gemm(M, N, K, tile_m=32, tile_n=None, tile_k=32):
    """Helper: quantize (per-token/per-channel), preshuffle B, compile, launch, return (C, C_ref)."""
    launch_fn = compile_fp8_gemm(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    A_f32 = torch.randn(M, K, device="cuda") * 0.1
    B_f32 = torch.randn(K, N, device="cuda") * 0.1

    A_fp8, scale_a = fp8_quantize_per_token(A_f32)       # [M]
    B_fp8, scale_b = fp8_quantize_per_channel(B_f32)     # [N]

    B_shuf = preshuffle_b_fp8(B_fp8)

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    sa = scale_a.to(device="cuda", dtype=torch.float32).contiguous()  # [M]
    sb = scale_b.to(device="cuda", dtype=torch.float32).contiguous()  # [N]

    A_f32_view = A_fp8.view(torch.float32).contiguous()
    B_shuf_f32 = B_shuf.view(torch.float32).contiguous()

    launch_fn(C, A_f32_view, B_shuf_f32, sa, sb, torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Reference: per-token scale_a[m] * A[m,:] @ per-channel scale_b[n] * B[:,n]
    C_ref = (A_fp8.float() * scale_a.unsqueeze(1)) @ (B_fp8.float() * scale_b.unsqueeze(0))
    return C, C_ref


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(32, 128, 128, id="32x128x128"),
        pytest.param(32, 128, 256, id="32x128x256"),
        pytest.param(32, 256, 256, id="32x256x256"),
    ],
)
def test_wmma_fp8_gemm_correctness(M, N, K):
    """Test FP8 WMMA GEMM correctness with preshuffle."""
    _requires_rdna4()
    torch.manual_seed(42)

    C, C_ref = _run_fp8_gemm(M, N, K)
    assert verify_output(C.float(), C_ref.float(), atol=0.5, rtol=0.1)


def test_wmma_fp8_preshuffle_b():
    """Test preshuffle_b_fp8 produces correct layout."""
    _requires_rdna4()

    K, N = 64, 32
    B = torch.arange(K * N, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn).reshape(K, N)
    B_shuf = preshuffle_b_fp8(B)
    # Shape should be [N//16, K//16, 2, 16, 8]
    assert B_shuf.shape == (N // 16, K // 16, 2, 16, 8), f"Wrong shape: {B_shuf.shape}"


def test_wmma_fp8_quantize():
    """Test fp8_quantize_per_token roundtrip."""
    _requires_rdna4()

    x = torch.randn(64, 64, device="cuda")
    x_fp8, scale = fp8_quantize_per_token(x)

    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (64,)
    assert (scale > 0).all()

    # Roundtrip should preserve order of magnitude
    x_roundtrip = x_fp8.float() * scale.unsqueeze(1)
    rel_err = ((x - x_roundtrip).abs() / (x.abs() + 1e-6)).mean().item()
    assert rel_err < 0.2, f"Mean relative roundtrip error too large: {rel_err}"
