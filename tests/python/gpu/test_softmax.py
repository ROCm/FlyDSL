#!/usr/bin/env python3
"""
Softmax Operator Test with Manual Vectorization and Register Buffering
Implementation based on high-performance C++ kernel logic:
- Vectorized Loads/Stores (WIDTH=8/4)
- Register Buffering (Row kept in registers)
- Warp Reductions (Shuffle)
- Shared Memory Block Reductions
"""

import sys
import os

# Add paths (prefer embedded MLIR to avoid mixing multiple runtimes)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, repo_root)

from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from tests.utils import compile_to_hsaco
from tests.test_common import run_perftest
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import numpy as np
import ctypes

from gpu_common import bf16_to_fp32_cpu, fp32_to_bf16_trunc_cpu, next_power_of_2
from examples.softmax_kernel import build_softmax_module, KERNEL_NAME as SOFTMAX_KERNEL_NAME

fp32_to_bf16_cpu = fp32_to_bf16_trunc_cpu
WARMUP_ITERS = 10
BENCH_ITERS = 100

def run_test(M, N, dtype_str):
    print(f"\nTesting Softmax (Vectorized): M={M}, N={N}, dtype={dtype_str}")

    # CPU reference + run_perftest both rely on torch.
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "test_softmax requires torch (ROCm build recommended). "
            "It is used for the PyTorch CPU reference and for run_perftest."
        ) from e
    
    try:
        ctx = build_softmax_module(M, N, dtype_str)
        hsaco = compile_to_hsaco(ctx.module, kernel_name=SOFTMAX_KERNEL_NAME)
    except Exception as e:
        print(f"❌ Compilation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    np.random.seed(42)
    a_f32 = (np.random.rand(M, N).astype(np.float32) * 4.0) - 2.0

    # PyTorch CPU reference (stable softmax)
    expected_f32 = torch.softmax(torch.from_numpy(a_f32), dim=1).cpu().numpy()
    
    if dtype_str == "f32":
        a_host = a_f32
        c_host = np.zeros_like(a_f32)
        nbytes = M * N * 4
    elif dtype_str == "f16":
        a_host = a_f32.astype(np.float16)
        c_host = np.zeros_like(a_host)
        nbytes = M * N * 2
    elif dtype_str == "bf16":
        a_host = fp32_to_bf16_cpu(a_f32)
        c_host = np.zeros_like(a_host)
        nbytes = M * N * 2
    
    d_a = hip_check(hip.hipMalloc(nbytes))
    d_c = hip_check(hip.hipMalloc(nbytes))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, SOFTMAX_KERNEL_NAME.encode("utf-8")))
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Dynamic block size based on N
    blk = min(256, next_power_of_2(N))
    if blk < 32: blk = 32

    def hip_kernel_launch():
        hip_check(
            hip.hipModuleLaunchKernel(
                kernel_func,
                M, 1, 1,
                blk, 1, 1,
                0, 0, args, None
            )
        )

    # One run for correctness visibility, then benchmark via shared harness.
    hip_kernel_launch()
    hip_check(hip.hipDeviceSynchronize())

    _, avg_us = run_perftest(hip_kernel_launch, num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    hip_check(hip.hipDeviceSynchronize())
    avg_ms = avg_us / 1000.0
    total_bytes = 2 * M * N * (4 if dtype_str == "f32" else 2)  # read input + write output
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    if dtype_str == "f32":
        res_f32 = c_host
        atol = 1e-5
    elif dtype_str == "f16":
        res_f32 = c_host.astype(np.float32)
        atol = 1e-2 
    elif dtype_str == "bf16":
        res_f32 = bf16_to_fp32_cpu(c_host)
        atol = 2e-2 
        
    diff = np.abs(res_f32 - expected_f32)
    max_err = np.max(diff)
    print(f"  Max Absolute Error: {max_err:.2e} (atol={atol})")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if max_err < atol:
        print("  ✅ Passed")
        return True
    else:
        print("  ❌ Failed")
        return False

def test_all():
    print("="*80)
    print("Running Softmax Vectorized Tests")
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
