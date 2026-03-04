#!/usr/bin/env python3
"""Test for the WMMA GEMM kernel on RDNA4 (gfx12xx)."""

import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import flydsl
from flydsl.runtime.device import get_rocm_arch

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

gpu_arch = get_rocm_arch()
if not gpu_arch.startswith("gfx12"):
    pytest.skip(
        f"WMMA GEMM test requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )


from kernels.wmma_gemm import create_wmma_gemm_module, BLOCK_M, BLOCK_N, BLOCK_K


TEST_SHAPES = [
    (32, 32, 16),  # 1 workgroup, 1 K tile
    (32, 32, 32),  # 1 workgroup, 2 K tiles
    (64, 64, 64),  # 4 workgroups, 4 K tiles
    (128, 128, 128),  # 16 workgroups
    (256, 256, 256),  # 64 workgroups
    (512, 512, 512),  # medium
]


@pytest.mark.parametrize("M,N,K", TEST_SHAPES)
def test_wmma_gemm_f32_output(M, N, K):
    """Test WMMA GEMM with f32 output."""
    print(f"\n{'=' * 60}")
    print(f"WMMA GEMM Test: M={M}, N={N}, K={K}, output=f32")
    print(f"GPU: {gpu_arch}")
    print(f"{'=' * 60}")

    m = create_wmma_gemm_module(M, N, K, out_dtype="f32")
    exe = flydsl.compile(m)

    np.random.seed(42)
    a_np = np.random.randn(M, K).astype(np.float16) * 0.1
    b_np = np.random.randn(K, N).astype(np.float16) * 0.1
    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    A = torch.tensor(a_np, device="cuda", dtype=torch.float16)
    B = torch.tensor(b_np, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    exe(A, B, C)
    torch.cuda.synchronize()

    c_host = C.cpu().numpy()
    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)

    print(f"Max absolute error: {error:.2e}")
    print(f"Max relative error: {rel_error:.2e}")

    # f16 inputs with f32 accumulation: expect ~1e-3 relative error for larger K
    tol = 1e-2
    assert rel_error < tol, f"WMMA GEMM error too high: rel_error={rel_error:.2e}"
    print("PASS")


def test_wmma_gemm_benchmark():
    """Benchmark WMMA GEMM at a meaningful size."""
    M, N, K = 1024, 1024, 1024

    print(f"\n{'=' * 60}")
    print(f"WMMA GEMM Benchmark: M={M}, N={N}, K={K}")
    print(f"GPU: {gpu_arch}")
    print(f"{'=' * 60}")

    m = create_wmma_gemm_module(M, N, K, out_dtype="f32")
    exe = flydsl.compile(m)

    A = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.01
    B = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.01
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(3):
        exe(A, B, C)
    torch.cuda.synchronize()

    # Benchmark
    import time

    num_iters = 20
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        exe(A, B, C)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iters) * 1000
    flops = 2 * M * N * K  # multiply-add = 2 FLOPs per element
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"Average time: {avg_ms:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

    # Verify correctness
    expected = A.float() @ B.float()
    c_host = C.cpu()
    e_host = expected.cpu()
    error = torch.max(torch.abs(c_host - e_host)).item()
    rel_error = error / (torch.max(torch.abs(e_host)).item() + 1e-8)
    print(f"Max relative error: {rel_error:.2e}")

    assert rel_error < 0.05, f"Benchmark result incorrect: rel_error={rel_error:.2e}"
    print("PASS")


if __name__ == "__main__":
    # Run smallest test first
    test_wmma_gemm_f32_output(32, 32, 16)
    test_wmma_gemm_f32_output(32, 32, 32)
    test_wmma_gemm_f32_output(128, 128, 128)
    test_wmma_gemm_benchmark()
