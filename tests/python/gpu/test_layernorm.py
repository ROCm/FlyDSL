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
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, repo_root)

from rocdsl.runtime.hip_util import hip_check
from tests.utils import compile_to_hsaco
from tests.test_common import run_perftest
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import ctypes

EPS: float = 1e-5
from examples.layernorm_kernel import (
    build_layernorm_module,
    KERNEL_NAME as LAYERNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100

def run_test(M: int, N: int, dtype: str = "f32") -> bool:
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")

    if hip is None:
        print("HIP not available, skipping...")
        return True

    # CPU reference + run_perftest both rely on torch.
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "test_layernorm requires torch (ROCm build recommended). "
            "It is used for the PyTorch CPU reference and for run_perftest."
        ) from e

    ctx = build_layernorm_module(M, N, dtype)
    try:
        hsaco = compile_to_hsaco(ctx.module, kernel_name=LAYERNORM_KERNEL_NAME)
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(ctx.module)
        raise e

    print(f" HSACO size: {len(hsaco)} bytes")

    torch.manual_seed(42)
    input_t = torch.randn((M, N), dtype=torch.float32)
    gamma_t = torch.rand((N,), dtype=torch.float32)
    beta_t = torch.rand((N,), dtype=torch.float32)

    if dtype == "f32":
        input_host = input_t.contiguous()
        gamma_host = gamma_t.contiguous()
        beta_host = beta_t.contiguous()
        output_host = torch.empty((M, N), dtype=torch.float32)
        elem_bytes = 4
        input_ref = input_host.to(torch.float32)
        gamma_ref = gamma_host.to(torch.float32)
        beta_ref = beta_host.to(torch.float32)
        atol = 1e-4
    elif dtype == "f16":
        input_host = input_t.to(torch.float16).contiguous()
        gamma_host = gamma_t.to(torch.float16).contiguous()
        beta_host = beta_t.to(torch.float16).contiguous()
        output_host = torch.empty((M, N), dtype=torch.float16)
        elem_bytes = 2
        input_ref = input_host.to(torch.float32)
        gamma_ref = gamma_host.to(torch.float32)
        beta_ref = beta_host.to(torch.float32)
        atol = 1e-2
    elif dtype == "bf16":
        input_host = input_t.to(torch.bfloat16).view(torch.uint16).contiguous()
        gamma_host = gamma_t.to(torch.bfloat16).view(torch.uint16).contiguous()
        beta_host = beta_t.to(torch.bfloat16).view(torch.uint16).contiguous()
        output_host = torch.empty((M, N), dtype=torch.uint16)
        elem_bytes = 2
        input_ref = input_host.view(torch.bfloat16).to(torch.float32)
        gamma_ref = gamma_host.view(torch.bfloat16).to(torch.float32)
        beta_ref = beta_host.view(torch.bfloat16).to(torch.float32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # PyTorch CPU Reference (variance uses unbiased=False)
    x = input_ref
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    expected = expected.to(torch.float32)

    # Allocate GPU Memory
    d_input = hip_check(hip.hipMalloc(M * N * elem_bytes))
    d_gamma = hip_check(hip.hipMalloc(N * elem_bytes))
    d_beta = hip_check(hip.hipMalloc(N * elem_bytes))
    d_output = hip_check(hip.hipMalloc(M * N * elem_bytes))

    hip_check(hip.hipMemcpy(d_input, int(input_host.data_ptr()), M * N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_gamma, int(gamma_host.data_ptr()), N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_beta, int(beta_host.data_ptr()), N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

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

    def hip_kernel_launch():
        hip_check(
            hip.hipModuleLaunchKernel(
                kernel_func,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                smem_size, 0, args, None
            )
        )

    # One run for correctness visibility, then benchmark via shared harness.
    hip_kernel_launch()
    hip_check(hip.hipDeviceSynchronize())

    _, avg_us = run_perftest(hip_kernel_launch, num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    hip_check(hip.hipDeviceSynchronize())
    avg_ms = avg_us / 1000.0
    total_bytes = (2 * M * N + 2 * N) * elem_bytes  # read input + write output + (gamma+beta)
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")

    # Copy back
    hip_check(hip.hipMemcpy(int(output_host.data_ptr()), d_output, M * N * elem_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    # Verification (pure torch style; compute max error in torch)
    if dtype == "f32":
        output_ref = output_host.to(torch.float32)
    elif dtype == "f16":
        output_ref = output_host.to(torch.float32)
    else:  # bf16 payload
        output_ref = output_host.view(torch.bfloat16).to(torch.float32)

    error = (output_ref - expected).abs().max().item()
    print(f"Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("✅ PASSED")
        ok = True
    else:
        print("❌ FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        # Print raw host values (uint16 for bf16) to help debugging.
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

