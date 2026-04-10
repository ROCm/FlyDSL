#!/usr/bin/env python3
"""RDNA4 GEMM correctness tests (gfx120x, wave32).

Kernel implementations:
  kernels/rdna_f16_gemm.py          — BF16/F16 GEMM with LDS
  kernels/rdna_fp8_preshuffle_gemm.py — FP8 GEMM with B preshuffle
"""

import os
import sys
import logging

import torch
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.rdna_f16_gemm import create_wmma_gemm_module
from kernels.rdna_fp8_preshuffle_gemm import (
    compile_fp8_gemm,
    preshuffle_b_fp8,
    fp8_quantize_per_token,
    fp8_quantize_per_channel,
)
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna4():
    if not ARCH.startswith("gfx120"):
        pytest.skip(f"RDNA4 GEMM requires gfx120x, got {ARCH}")


# ── BF16/F16 GEMM ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(256, 256, 256, id="256x256x256"),
        pytest.param(256, 256, 512, id="256x256x512"),
        pytest.param(512, 512, 512, id="512x512x512", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_f16_gemm_correctness(M, N, K, dtype):
    """Test BF16/F16 GEMM correctness for various shapes and dtypes."""
    _requires_rdna4()

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(42)

    launch_fn, BLOCK_M, BLOCK_N, BLOCK_K = create_wmma_gemm_module(M, N, K, in_dtype=dtype, out_dtype="bf16")

    A = torch.randn(M, K, dtype=torch_dtype, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch_dtype, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    launch_fn(C, A, B_T, torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(256, 256, 256, id="256x256x256"),
    ],
)
def test_f16_gemm_f32_output(M, N, K):
    """Test BF16 GEMM with f32 output accumulation."""
    _requires_rdna4()

    torch.manual_seed(42)
    launch_fn, _, _, _ = create_wmma_gemm_module(M, N, K, in_dtype="bf16", out_dtype="f32")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn(C, A, B_T, torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(1024, 1024, 1024, id="1k"),
        pytest.param(2048, 2048, 2048, id="2k", marks=pytest.mark.large_shape),
    ],
)
def test_f16_gemm_benchmark(M, N, K):
    """Benchmark BF16 GEMM throughput."""
    _requires_rdna4()

    torch.manual_seed(42)
    launch_fn, _, _, _ = create_wmma_gemm_module(M, N, K, in_dtype="bf16", out_dtype="bf16")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.01
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.01
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    def run_kernel():
        launch_fn(C, A, B_T, torch.cuda.current_stream())

    _, avg_us = run_perftest(run_kernel, num_iters=20, num_warmup=3)

    flops = 2 * M * N * K
    tflops = flops / (avg_us / 1e6) / 1e12
    logging.getLogger("flydsl").info(f"[f16_gemm] {M}x{N}x{K} bf16: {avg_us:.1f} us, {tflops:.2f} TFLOPS")

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.1, rtol=0.1, msg=f"{M}x{N}x{K}")


# ── FP8 Preshuffle GEMM ──────────────────────────────────────────────────────


def _run_fp8_gemm(M, N, K, tile_m=32, tile_n=None, tile_k=32):
    """Helper: quantize (per-token/per-channel), preshuffle B, compile, launch."""
    launch_fn = compile_fp8_gemm(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    A_f32 = torch.randn(M, K, device="cuda") * 0.1
    B_f32 = torch.randn(K, N, device="cuda") * 0.1

    A_fp8, scale_a = fp8_quantize_per_token(A_f32)
    B_fp8, scale_b = fp8_quantize_per_channel(B_f32)

    B_shuf = preshuffle_b_fp8(B_fp8)

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    sa = scale_a.to(device="cuda", dtype=torch.float32).contiguous()
    sb = scale_b.to(device="cuda", dtype=torch.float32).contiguous()

    A_f32_view = A_fp8.view(torch.float32).contiguous()
    B_shuf_f32 = B_shuf.view(torch.float32).contiguous()

    launch_fn(C, A_f32_view, B_shuf_f32, sa, sb, torch.cuda.current_stream())
    torch.cuda.synchronize()

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
def test_fp8_gemm_correctness(M, N, K):
    """Test FP8 preshuffle GEMM correctness."""
    _requires_rdna4()
    torch.manual_seed(42)

    C, C_ref = _run_fp8_gemm(M, N, K)
    assert verify_output(C.float(), C_ref.float(), atol=0.5, rtol=0.1)


def test_fp8_preshuffle_b():
    """Test preshuffle_b_fp8 produces correct layout."""
    _requires_rdna4()

    K, N = 64, 32
    B = torch.arange(K * N, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn).reshape(K, N)
    B_shuf = preshuffle_b_fp8(B)
    assert B_shuf.shape == (N // 16, K // 16, 2, 16, 8), f"Wrong shape: {B_shuf.shape}"


def test_fp8_quantize():
    """Test fp8_quantize_per_token roundtrip."""
    _requires_rdna4()

    x = torch.randn(64, 64, device="cuda")
    x_fp8, scale = fp8_quantize_per_token(x)

    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (64,)
    assert (scale > 0).all()

    x_roundtrip = x_fp8.float() * scale.unsqueeze(1)
    rel_err = ((x - x_roundtrip).abs() / (x.abs() + 1e-6)).mean().item()
    assert rel_err < 0.2, f"Mean relative roundtrip error too large: {rel_err}"
