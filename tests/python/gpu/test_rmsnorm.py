#!/usr/bin/env python3
"""
RMSNorm Operator Test
Implementation of a Block-wise RMSNorm:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (sum of squares)

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
"""

import sys
import os

# Add paths to find flir and mlir packages (prefer embedded MLIR to avoid mixing runtimes)
repo_root = os.path.join(os.path.dirname(__file__), "../../..")
embedded_pkgs = os.path.join(repo_root, "build", "python_packages", "flir")
if os.path.isdir(os.path.join(embedded_pkgs, "_mlir")):
    sys.path.insert(0, embedded_pkgs)
else:
    sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, repo_root)

import pyflir
import pytest
import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import numpy as np
import time

from gpu_common import EPS, bf16_to_fp32_cpu, fp32_to_bf16_rne_cpu
from examples.rmsnorm_kernel import (
    build_rmsnorm_module,
    KERNEL_NAME as RMSNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100
fp32_to_bf16_cpu = fp32_to_bf16_rne_cpu

def run_test(M: int, N: int, dtype: str = "f32") -> bool:
    print(f"\nTesting RMSNorm (M={M}, N={N}, dtype={dtype})")

    ctx = build_rmsnorm_module(M, N, dtype)
    try:
        exe = flir.compile(ctx)
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(ctx.module)
        raise e
    print(" Compiled")

    np.random.seed(42)
    input_f32 = np.random.randn(M, N).astype(np.float32)
    gamma_f32 = np.random.rand(N).astype(np.float32)

    if dtype == "f32":
        input_host = input_f32
        gamma_host = gamma_f32
        output_host = np.zeros((M, N), dtype=np.float32)
        elem_bytes = 4
        input_ref = input_f32
        gamma_ref = gamma_f32
        atol = 1e-4
    elif dtype == "f16":
        input_host = input_f32.astype(np.float16)
        gamma_host = gamma_f32.astype(np.float16)
        output_host = np.zeros((M, N), dtype=np.float16)
        elem_bytes = 2
        input_ref = input_host.astype(np.float32)
        gamma_ref = gamma_host.astype(np.float32)
        atol = 1e-2
    elif dtype == "bf16":
        input_host = fp32_to_bf16_cpu(input_f32)
        gamma_host = fp32_to_bf16_cpu(gamma_f32)
        output_host = np.zeros((M, N), dtype=np.uint16)
        elem_bytes = 2
        input_ref = bf16_to_fp32_cpu(input_host)
        gamma_ref = bf16_to_fp32_cpu(gamma_host)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # Numpy Reference
    # RMS(x) = sqrt(mean(x^2) + eps) RMSNorm(x) = x / RMS(x) * gamma
    sq_mean = np.mean(input_ref**2, axis=1, keepdims=True)
    rms = np.sqrt(sq_mean + EPS)
    expected = (input_ref / rms) * gamma_ref

    print("Launching kernel...")
    if dtype == "f32":
        x = torch.tensor(input_host, device="cuda", dtype=torch.float32)
        gamma = torch.tensor(gamma_host, device="cuda", dtype=torch.float32)
        y = torch.empty((M, N), device="cuda", dtype=torch.float32)
    elif dtype == "f16":
        x = torch.tensor(input_host, device="cuda", dtype=torch.float16)
        gamma = torch.tensor(gamma_host, device="cuda", dtype=torch.float16)
        y = torch.empty((M, N), device="cuda", dtype=torch.float16)
    else:  # bf16
        x = torch.tensor(input_ref, device="cuda", dtype=torch.bfloat16)
        gamma = torch.tensor(gamma_ref, device="cuda", dtype=torch.bfloat16)
        y = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP_ITERS):
        exe(x, gamma, y)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(BENCH_ITERS):
        exe(x, gamma, y)
    stop_event.record()
    stop_event.synchronize()
    avg_ms = start_event.elapsed_time(stop_event) / BENCH_ITERS
    print(f"Kernel avg time: {avg_ms:.4f} ms (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")

    output_ref = y.float().cpu().numpy()

    # Verification
    error = np.max(np.abs(output_ref - expected))
    print(f"Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("✅ PASSED")
        ok = True
    else:
        print("❌ FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_host[0, :5])
        ok = False

    return ok

def test_all():
    print("="*80)
    print("Running RMSNorm Tests")
    print("="*80)

    configs = [
        # (64, 256, "f32"),    # Aligned
        # (128, 1024, "f32"),  # Aligned
        # (32, 128, "f16"),    # Aligned
        # (64, 2000, "f32"),   # Unaligned (tail handling)
        # (16, 512, "bf16"),   # BF16
        # (256, 65536, "bf16"),# BF16
        (32768, 8192, "bf16"),  # BF16

    ]

    failures = 0
    for M, N, dtype in configs:
        if not run_test(M, N, dtype):
            failures += 1

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)

if __name__ == "__main__":
    test_all()

