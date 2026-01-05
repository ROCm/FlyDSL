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

from flydsl.runtime.hip_util import hip_check, get_hip_arch
from tests.utils import compile_to_hsaco
from tests.test_common import run_perftest
from tests.kernels.perf_compare_common import (
    PerfRow,
    bench_gpu_us_hip,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import ctypes

def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

from kernels.softmax_kernel import build_softmax_module, KERNEL_NAME as SOFTMAX_KERNEL_NAME

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
        return False, None
        
    torch.manual_seed(42)
    a_t = (torch.rand((M, N), dtype=torch.float32) * 4.0) - 2.0

    # PyTorch CPU reference (stable softmax)
    expected = torch.softmax(a_t, dim=1).to(torch.float32)
    
    if dtype_str == "f32":
        a_host = a_t.contiguous()
        c_host = torch.empty((M, N), dtype=torch.float32)
        nbytes = M * N * 4
    elif dtype_str == "f16":
        a_host = a_t.to(torch.float16).contiguous()
        c_host = torch.empty((M, N), dtype=torch.float16)
        nbytes = M * N * 2
    elif dtype_str == "bf16":
        # BF16 host buffer uses uint16 payload holding raw bf16 bits.
        a_host = a_t.to(torch.bfloat16).view(torch.uint16).contiguous()
        c_host = torch.empty((M, N), dtype=torch.uint16)
        nbytes = M * N * 2
    
    d_a = hip_check(hip.hipMalloc(nbytes))
    d_c = hip_check(hip.hipMalloc(nbytes))
    
    hip_check(hip.hipMemcpy(d_a, int(a_host.data_ptr()), nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
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
    # Optional: more stable device timing via HIP events (for FLIR kernel).
    flir_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flir_gpu_us = bench_gpu_us_hip(hip_kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0
    total_bytes = 2 * M * N * (4 if dtype_str == "f32" else 2)  # read input + write output
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flir_gpu_us is not None:
        print(f"[Perf] FLIR softmax gpu: {flir_gpu_us:.1f} us")
    
    hip_check(hip.hipMemcpy(int(c_host.data_ptr()), d_c, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    # Verify in pure torch style (keep tensors, compute max error in torch), similar to test_mfma_gemm_fp8_rocir.py
    if dtype_str == "f32":
        res = c_host.to(torch.float32)
        atol = 1e-5
    elif dtype_str == "f16":
        res = c_host.to(torch.float32)
        atol = 1e-2
    elif dtype_str == "bf16":
        # BF16 payload stored as uint16; reinterpret then convert to fp32
        res = c_host.view(torch.bfloat16).to(torch.float32)
        atol = 2e-2

    max_err = (res - expected).abs().max().item()
    print(f"  Max Absolute Error: {max_err:.2e} (atol={atol})")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
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
                    x = (torch.rand((M, N), device="cuda", dtype=torch.float16) * 4.0) - 2.0
                    if dtype == "f32":
                        x = x.to(torch.float32)
                    elif dtype == "bf16":
                        x = x.to(torch.bfloat16)

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
