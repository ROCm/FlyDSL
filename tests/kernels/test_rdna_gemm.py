#!/usr/bin/env python3
"""WMMA GEMM correctness tests for RDNA4 (gfx120x, wave32).

Kernel implementation: kernels/rdna_gemm.py
"""

import os
import sys
import logging

import torch
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.rdna_gemm import create_wmma_gemm_module
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna4():
    if not ARCH.startswith("gfx120"):
        pytest.skip(f"WMMA GEMM requires RDNA4 (gfx120x), got {ARCH}")


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
def test_wmma_gemm_correctness(M, N, K, dtype):
    """Test WMMA GEMM correctness for various shapes and dtypes."""
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
def test_wmma_gemm_f32_output(M, N, K):
    """Test WMMA GEMM with f32 output accumulation."""
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
def test_wmma_gemm_benchmark(M, N, K):
    """Benchmark WMMA GEMM throughput."""
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
    logging.getLogger("flydsl").info(f"[wmma_gemm] {M}x{N}x{K} bf16: {avg_us:.1f} us, {tflops:.2f} TFLOPS")

    # Verify correctness after benchmark
    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.1, rtol=0.1, msg=f"{M}x{N}x{K}")
