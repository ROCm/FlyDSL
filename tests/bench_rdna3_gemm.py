#!/usr/bin/env python3
"""Benchmark the RDNA 3 / 3.5 atom-based GEMM (kernels.rdna3_gemm)
against a rocBLAS / PyTorch hgemm reference on the same gfx11x device.

Prints TFLOPS, percentage of rocBLAS, and percentage of theoretical
WMMA peak. On gfx1151 (40 CUs @ 2.9 GHz), the WMMA F16 peak is
~59 TFLOPS, and rocBLAS hgemm typically reaches ~30 TFLOPS (~50% MFU).
"""

import argparse
import os
import sys
import time

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch
from kernels.rdna3_gemm import create_rdna3_gemm_module


# Per-CU per-GHz F16 WMMA throughput on RDNA 3 / 3.5: ~511 ops/cycle/CU
# (back-calculated from RX 7900 XTX's 122.6 TFLOPS marketing number /
# 96 CUs / 2.5 GHz). Same family ⇒ same per-CU rate on gfx115x/gfx110x.
WMMA_F16_OPS_PER_CU_PER_CYCLE = 511

# Rough chip lookup for theoretical peak hint. Set GHz to your boost clock.
CHIP_INFO = {
    "gfx1100": (96, 2.5),  # RX 7900 XTX
    "gfx1101": (60, 2.5),  # RX 7800 XT
    "gfx1102": (32, 2.5),  # RX 7600
    "gfx1150": (32, 2.9),  # Strix Halo desktop fallback
    "gfx1151": (40, 2.9),  # Strix Halo iGPU (Ryzen AI Max+)
    "gfx1152": (40, 2.9),  # Strix Halo variant
}


def theoretical_wmma_tflops(arch: str) -> float:
    cus, ghz = CHIP_INFO.get(arch, (None, None))
    if cus is None:
        return 0.0
    return cus * WMMA_F16_OPS_PER_CU_PER_CYCLE * ghz / 1000.0


def bench_one(M, N, K, in_dtype="f16", out_dtype="f32", iters=100, warmup=20, **kw):
    torch_in = {"f16": torch.float16, "bf16": torch.bfloat16}[in_dtype]
    torch_out = {"f32": torch.float32, "f16": torch.float16,
                 "bf16": torch.bfloat16}[out_dtype]
    A = (torch.randn(M, K, dtype=torch_in, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch_in, device="cuda") * 0.1)

    # FlyDSL kernel
    C = torch.zeros(M, N, dtype=torch_out, device="cuda")
    launch_fn, BM, BN, BK = create_rdna3_gemm_module(
        M, N, K, in_dtype=in_dtype, out_dtype=out_dtype, **kw
    )
    for _ in range(warmup):
        launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    fly_us = (time.perf_counter() - t) / iters * 1e6
    fly_tflops = 2 * M * N * K / fly_us / 1e6

    # PyTorch / rocBLAS reference (always uses native dtype output, e.g.
    # f16@f16 -> f16. Same FLOPS budget regardless of out_dtype.)
    B = B_T.t().contiguous()
    for _ in range(warmup):
        C2 = torch.matmul(A, B)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        C2 = torch.matmul(A, B)
    torch.cuda.synchronize()
    ref_us = (time.perf_counter() - t) / iters * 1e6
    ref_tflops = 2 * M * N * K / ref_us / 1e6

    # Correctness check
    Cref = A.float() @ B_T.float().T
    cos = ((C.float() * Cref).sum() /
           ((C.float() * C.float()).sum().sqrt() *
            (Cref * Cref).sum().sqrt())).item()

    return {
        "M": M, "N": N, "K": K,
        "BLOCK": (BM, BN, BK),
        "fly_us": fly_us, "fly_tflops": fly_tflops,
        "ref_us": ref_us, "ref_tflops": ref_tflops,
        "cos": cos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dtype", default="f16", choices=["f16", "bf16"])
    parser.add_argument("--out-dtype", default="f32",
                        choices=["f32", "f16", "bf16"])
    parser.add_argument("--shapes", default="1024,2048,4096",
                        help="comma-separated square sizes")
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    peak = theoretical_wmma_tflops(arch)
    print(f"GPU arch: {arch}")
    if peak > 0:
        print(f"Theoretical WMMA F16 peak: {peak:.1f} TFLOPS")
    print()

    print(f"{'Shape':<22} {'BLOCK':<14} {'FlyDSL':>10}  {'rocBLAS':>10}  "
          f"{'%rocBLAS':>9}  {'%peak':>7}  {'cos':>10}")
    print("-" * 95)

    for s in args.shapes.split(","):
        M = N = K = int(s)
        r = bench_one(M, N, K, in_dtype=args.in_dtype, out_dtype=args.out_dtype)
        pct_ref = 100.0 * r["fly_tflops"] / r["ref_tflops"]
        pct_peak = 100.0 * r["fly_tflops"] / peak if peak > 0 else 0.0
        BM, BN, BK = r["BLOCK"]
        print(f"{M}x{N}x{K:<14}"
              f"{BM}x{BN}x{BK:<5}"
              f"  {r['fly_tflops']:6.2f} TF  "
              f"{r['ref_tflops']:6.2f} TF  "
              f"{pct_ref:7.1f}%  "
              f"{pct_peak:5.1f}%  "
              f"{r['cos']:10.6f}")


if __name__ == "__main__":
    main()
