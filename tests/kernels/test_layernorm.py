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
from pathlib import Path

# Prefer embedded MLIR/rocdsl to avoid mixing multiple runtimes.
_repo = Path(__file__).resolve().parents[3]
_embedded = _repo / "build" / "python_packages" / "rocdsl"
if _embedded.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

from tests.utils import compile_to_hsaco
from tests.test_common import run_perftest
from tests.kernels.perf_compare_common import (
    PerfRow,
    bench_gpu_us_hip,
    bench_gpu_us_torch,
    hip_check,
    maybe_enable_aiter,
    print_perf_table,
)
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import ctypes

EPS: float = 1e-5
from kernels.layernorm_kernel import (
    build_layernorm_module,
    KERNEL_NAME as LAYERNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100

def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")

    if hip is None:
        print("HIP not available, skipping...")
        return True, None

    # CPU reference + run_perftest both rely on torch.
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "test_layernorm requires torch (ROCm build recommended). "
            "It is used for the PyTorch CPU reference and for run_perftest."
        ) from e

    try:
        ctx = build_layernorm_module(M, N, dtype)
    except ValueError as e:
        print(f"[Skip] {e}")
        return True, None
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
    flir_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flir_gpu_us = bench_gpu_us_hip(hip_kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0
    total_bytes = (2 * M * N + 2 * N) * elem_bytes  # read input + write output + (gamma+beta)
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flir_gpu_us is not None:
        print(f"[Perf] FLIR layernorm gpu: {flir_gpu_us:.1f} us")

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
    return ok, flir_gpu_us

def test_all():
    print("="*80)
    print("Running LayerNorm Tests")
    print("="*80)

    shapes_env = os.environ.get("ROCDSL_LAYERNORM_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
    else:
        configs = [
            (64, 256, "f32"),     # Aligned
            (128, 1024, "f32"),   # Aligned
            (32, 128, "f16"),     # Aligned
            (64, 2000, "f32"),    # Unaligned (tail handling)
            (16, 512, "bf16"),    # BF16
            (1024, 8192, "bf16"), # BF16
            (32768, 8192, "bf16"),
        ]

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flir_gpu_us = run_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch
            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.norm import layer_norm as aiter_layer_norm
                    x = torch.randn((M, N), device="cuda", dtype=torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "f16" else torch.float32))
                    w = torch.rand((N,), device="cuda", dtype=x.dtype)
                    b = torch.rand((N,), device="cuda", dtype=x.dtype)

                    def run_aiter():
                        aiter_layer_norm(x, w, b, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter layernorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter layernorm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(PerfRow(op="layernorm", shape=f"{M}x{N}", dtype=dtype, flir_gpu_us=flir_gpu_us, aiter_gpu_us=aiter_us))

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)

if __name__ == "__main__":
    test_all()

