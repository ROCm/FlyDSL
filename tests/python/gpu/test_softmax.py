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
import math

# Add paths (prefer embedded MLIR to avoid mixing multiple runtimes)
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

from gpu_common import bf16_to_fp32_cpu
from examples.softmax_kernel import build_softmax_module, KERNEL_NAME as SOFTMAX_KERNEL_NAME

def run_test(M, N, dtype_str):
    print(f"\nTesting Softmax (Vectorized): M={M}, N={N}, dtype={dtype_str}")
    
    try:
        m = build_softmax_module(M, N, dtype_str)
        exe = flir.compile(m)
    except Exception as e:
        print(f"❌ Compilation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    np.random.seed(42)
    a_f32 = (np.random.rand(M, N).astype(np.float32) * 4.0) - 2.0
    
    max_vals = np.max(a_f32, axis=1, keepdims=True)
    exp_vals = np.exp(a_f32 - max_vals)
    sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
    expected_f32 = exp_vals / sum_vals
    
    if dtype_str == "f32":
        a = torch.tensor(a_f32, device="cuda", dtype=torch.float32)
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        atol = 1e-5
    elif dtype_str == "f16":
        a = torch.tensor(a_f32, device="cuda", dtype=torch.float16)
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)
        atol = 1e-2
    elif dtype_str == "bf16":
        a = torch.tensor(a_f32, device="cuda", dtype=torch.bfloat16)
        c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        atol = 2e-2
    else:
        raise ValueError(dtype_str)

    exe(a, c)
    torch.cuda.synchronize()
    res_f32 = c.float().cpu().numpy()
        
    diff = np.abs(res_f32 - expected_f32)
    max_err = np.max(diff)
    print(f"  Max Absolute Error: {max_err:.2e} (atol={atol})")
    
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
