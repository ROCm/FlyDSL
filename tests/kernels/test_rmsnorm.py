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
from pathlib import Path

# Prefer embedded MLIR/rocdsl to avoid mixing multiple runtimes.
# Repo root is two levels above `tests/kernels/`.
_repo = Path(__file__).resolve().parents[2]
_embedded = _repo / "build" / "python_packages" / "rocdsl"
if _embedded.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)
import pytest
try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

import flydsl

EPS: float = 1e-5
from kernels.rmsnorm_kernel import (
    build_rmsnorm_module,
    KERNEL_NAME as RMSNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100

def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting RMSNorm (M={M}, N={N}, dtype={dtype})")

    try:
        m = build_rmsnorm_module(M, N, dtype)
        exe = flydsl.compile(m)
    except Exception as e:
        # Treat compile-time failures as test failures so wrapper scripts (run_tests.sh)
        # can detect them via a non-zero exit code.
        print(f"[FAIL] Compile failed for (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None
    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # PyTorch CPU Reference:
    # RMS(x) = sqrt(mean(x^2) + eps) ; RMSNorm(x) = x / RMS(x) * gamma
    x = input_ref
    gamma = gamma_ref
    sq_mean = (x * x).mean(dim=1, keepdim=True)
    rms = torch.sqrt(sq_mean + EPS)
    expected = (x / rms) * gamma
    expected = expected.to(DTYPE_FP32)

    print("Launching kernel...")

    def kernel_launch():
        exe(input_dev, gamma_dev, output_dev, M)

    # run_perftest returns (data, avg_us)
    _, avg_us = run_perftest(lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    torch.cuda.synchronize()
    flir_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flir_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    # Bandwidth estimate: read input + read gamma + write output
    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = 2 * M * N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flir_gpu_us is not None:
        print(f"[Perf] FLIR rmsnorm gpu: {flir_gpu_us:.1f} us")

    # Verification (pure torch style; compute max error in torch)
    output_ref = output_dev.to(DTYPE_FP32)

    error = (output_ref - expected).abs().max().item()
    print(f"Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        ok = False
    return ok, flir_gpu_us

def test_all():
    print("="*80)
    print("Running RMSNorm Tests")
    print("="*80)

    shapes_env = os.environ.get("ROCDSL_RMSNORM_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
    else:
        # Prefer N multiples of BLOCK_THREADS*VEC_WIDTH (=2048) to exercise the fast path.
        configs = [
            # (64, 256, "f32"),     # Aligned
            # (128, 1024, "f32"),   # Aligned
            # (32, 128, "f16"),     # Aligned
            # (64, 2000, "f32"),    # Unaligned (tail handling)
            # (16, 512, "bf16"),    # BF16
            # (1024, 8192, "bf16"), # BF16
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
                    from aiter.ops.triton.rmsnorm import rms_norm as aiter_rms_norm
                    x = torch.randn((M, N), device="cuda", dtype=DTYPE_BF16 if dtype == "bf16" else (DTYPE_FP16 if dtype == "f16" else DTYPE_FP32))
                    w = torch.rand((N,), device="cuda", dtype=x.dtype)

                    def run_aiter():
                        aiter_rms_norm(x, w, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter rmsnorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter rmsnorm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(PerfRow(op="rmsnorm", shape=f"{M}x{N}", dtype=dtype, flir_gpu_us=flir_gpu_us, aiter_gpu_us=aiter_us))

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    test_all()

