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
from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)

def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

from kernels.softmax_kernel import build_softmax_module, KERNEL_NAME as SOFTMAX_KERNEL_NAME

WARMUP_ITERS = 10
BENCH_ITERS = 100

def run_test(M, N, dtype_str):
    print(f"\nTesting Softmax (Vectorized): M={M}, N={N}, dtype={dtype_str}")

    try:
        m = build_softmax_module(M, N, dtype_str)
        exe = flydsl.compile(m)
    except Exception as e:
        print(f"❌ Compilation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
        
    torch.manual_seed(42)
    a_t = (torch.rand((M, N), device="cuda", dtype=DTYPE_FP32) * 4.0) - 2.0

    # PyTorch CPU reference (stable softmax)
    expected = torch.softmax(a_t, dim=1).to(DTYPE_FP32)
    
    if dtype_str == "f32":
        a_dev = a_t.contiguous()
        c_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
    elif dtype_str == "f16":
        a_dev = a_t.to(DTYPE_FP16).contiguous()
        c_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
    elif dtype_str == "bf16":
        a_dev = a_t.to(DTYPE_BF16).contiguous()
        c_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
    
    def kernel_launch():
        exe(a_dev, c_dev)

    # One run for correctness visibility, then benchmark via shared harness.
    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    torch.cuda.synchronize()
    # Optional: more stable device timing via HIP events (for FLIR kernel).
    flir_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flir_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0
    total_bytes = 2 * M * N * (4 if dtype_str == "f32" else 2)  # read input + write output
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flir_gpu_us is not None:
        print(f"[Perf] FLIR softmax gpu: {flir_gpu_us:.1f} us")

    # Verify in pure torch style (keep tensors, compute max error in torch), similar to test_mfma_gemm_fp8_rocir.py
    if dtype_str == "f32":
        res = c_dev.to(DTYPE_FP32)
        atol = 1e-5
    elif dtype_str == "f16":
        res = c_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype_str == "bf16":
        res = c_dev.to(DTYPE_FP32)
        atol = 2e-2

    max_err = (res - expected).abs().max().item()
    print(f"  Max Absolute Error: {max_err:.2e} (atol={atol})")
    
    if max_err < atol:
        print("  ✅ Passed")
        return True, flir_gpu_us
    else:
        print("  ❌ Failed")
        return False, flir_gpu_us

def test_all():
    print("="*80)
    print("Running Softmax Vectorized Tests")
    print("="*80)
    
    # Default shape sweep (override with ROCDSL_SOFTMAX_SHAPES="M,N,dtype;M,N,dtype;...")
    shapes_env = os.environ.get("ROCDSL_SOFTMAX_SHAPES", "").strip()
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
            # Re-run just perf on device tensors (avoid H2D/D2H in the comparison).
            import torch
            if not torch.cuda.is_available():
                continue

            # FLIR side: reuse this test's hip kernel launch via compilation path in run_test is expensive to re-plumb.
            # We provide AIter numbers here; FLIR gpu-us is printed from run_test via HIP events.
            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.softmax import softmax as aiter_softmax
                    x = (torch.rand((M, N), device="cuda", dtype=DTYPE_FP16) * 4.0) - 2.0
                    if dtype == "f32":
                        x = x.to(DTYPE_FP32)
                    elif dtype == "bf16":
                        x = x.to(DTYPE_BF16)

                    def run_aiter():
                        aiter_softmax(x)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter softmax gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter softmax skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(PerfRow(op="softmax", shape=f"{M}x{N}", dtype=dtype, flir_gpu_us=flir_gpu_us, aiter_gpu_us=aiter_us))
            
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
