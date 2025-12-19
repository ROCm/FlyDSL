#!/usr/bin/env python3
"""
LayerNorm Operator Test
Implementation of a Block-wise LayerNorm:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (mean and variance)

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
"""

import sys
import os

# Add paths to find rocdsl and mlir packages (prefer embedded MLIR to avoid mixing runtimes)
repo_root = os.path.join(os.path.dirname(__file__), "../../..")
embedded_pkgs = os.path.join(repo_root, "build", "python_packages", "rocdsl")
if os.path.isdir(os.path.join(embedded_pkgs, "_mlir")):
    sys.path.insert(0, embedded_pkgs)
else:
    sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, repo_root)

from rocdsl.runtime.hip_util import hip_check
from tests.utils import compile_to_hsaco
import pytest
try:
    from hip import hip
except ImportError:
    pytest.skip("HIP module not found. Skipping GPU tests.", allow_module_level=True)

import numpy as np
import ctypes
import time

from gpu_common import EPS, bf16_to_fp32_cpu, fp32_to_bf16_rne_cpu
from examples.layernorm_kernel import (
    build_layernorm_module,
    KERNEL_NAME as LAYERNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

fp32_to_bf16_cpu = fp32_to_bf16_rne_cpu

def run_test(M: int, N: int, dtype: str = "f32") -> bool:
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")

    if hip is None:
        print("HIP not available, skipping...")
        return True

    ctx = build_layernorm_module(M, N, dtype)
    try:
        hsaco = compile_to_hsaco(ctx.module, kernel_name=LAYERNORM_KERNEL_NAME)
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(ctx.module)
        raise e

    print(f" HSACO size: {len(hsaco)} bytes")

    np.random.seed(42)
    input_f32 = np.random.randn(M, N).astype(np.float32)
    gamma_f32 = np.random.rand(N).astype(np.float32)
    beta_f32 = np.random.rand(N).astype(np.float32)

    if dtype == "f32":
        input_host = input_f32
        gamma_host = gamma_f32
        beta_host = beta_f32
        output_host = np.zeros((M, N), dtype=np.float32)
        elem_bytes = 4
        input_ref = input_f32
        gamma_ref = gamma_f32
        beta_ref = beta_f32
        atol = 1e-4
    elif dtype == "f16":
        input_host = input_f32.astype(np.float16)
        gamma_host = gamma_f32.astype(np.float16)
        beta_host = beta_f32.astype(np.float16)
        output_host = np.zeros((M, N), dtype=np.float16)
        elem_bytes = 2
        input_ref = input_host.astype(np.float32)
        gamma_ref = gamma_host.astype(np.float32)
        beta_ref = beta_host.astype(np.float32)
        atol = 1e-2
    elif dtype == "bf16":
        input_host = fp32_to_bf16_cpu(input_f32)
        gamma_host = fp32_to_bf16_cpu(gamma_f32)
        beta_host = fp32_to_bf16_cpu(beta_f32)
        output_host = np.zeros((M, N), dtype=np.uint16)
        elem_bytes = 2
        input_ref = bf16_to_fp32_cpu(input_host)
        gamma_ref = bf16_to_fp32_cpu(gamma_host)
        beta_ref = bf16_to_fp32_cpu(beta_host)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # Numpy Reference
    mean = np.mean(input_ref, axis=1, keepdims=True)
    var = np.var(input_ref, axis=1, keepdims=True)
    expected = (input_ref - mean) / np.sqrt(var + EPS) * gamma_ref + beta_ref

    # Allocate GPU Memory
    d_input = hip_check(hip.hipMalloc(M * N * elem_bytes))
    d_gamma = hip_check(hip.hipMalloc(N * elem_bytes))
    d_beta = hip_check(hip.hipMalloc(N * elem_bytes))
    d_output = hip_check(hip.hipMalloc(M * N * elem_bytes))

    hip_check(hip.hipMemcpy(d_input, input_host.ctypes.data, M * N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_gamma, gamma_host.ctypes.data, N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_beta, beta_host.ctypes.data, N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # Load Kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"layernorm_kernel"))

    # Launch Config
    grid_x, grid_y, grid_z = M, 1, 1
    block_x, block_y, block_z = BLOCK_THREADS, 1, 1
    smem_size = 0

    arg_ptrs = [
        ctypes.c_void_p(int(d_input)),
        ctypes.c_void_p(int(d_gamma)),
        ctypes.c_void_p(int(d_beta)),
        ctypes.c_void_p(int(d_output))
    ]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])

    print("Launching kernel...")
    start_time = time.time()
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, grid_z, block_x, block_y, block_z, smem_size, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    end_time = time.time()
    
    print(f"Kernel execution time: {(end_time - start_time)*1000:.4f} ms")

    # Copy back
    hip_check(hip.hipMemcpy(output_host.ctypes.data, d_output, M * N * elem_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    if dtype == "f32":
        output_ref = output_host
    elif dtype == "f16":
        output_ref = output_host.astype(np.float32)
    else:  # bf16
        output_ref = bf16_to_fp32_cpu(output_host)

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

    # Cleanup
    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_gamma))
    hip_check(hip.hipFree(d_beta))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(hip_module))
    return ok

def test_all():
    print("="*80)
    print("Running LayerNorm Tests")
    print("="*80)

    configs = [
        # (64, 256, "f32"),    # Aligned
        # (128, 1024, "f32"),  # Aligned
        # (32, 128, "f16"),    # Aligned
        # (64, 2000, "f32"),   # Unaligned (tail handling)
        # (16, 512, "bf16"),   # BF16
        # (1024, 8192, "bf16"),# BF16
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

