#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Benchmark: preshuffle_gemm_v2 (layout API) vs old preshuffle_gemm.

Usage:
    # Run all default configs
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_v2.py

    # Specific dtype
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_v2.py --dtype fp16
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_v2.py --dtype bf16
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_v2.py --dtype fp8

    # Custom shape
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_v2.py --dtype fp16 -M 5120 -N 5120 -K 8192 --tile_m 64 --tile_n 128 --tile_k 64

    # All tiles sweep for a given shape
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_v2.py --dtype fp16 -M 128 -N 5120 -K 8192 --sweep
"""

import os
import sys
import argparse

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import flydsl.compiler as flyc
from kernels.preshuffle_gemm_v2 import compile_preshuffle_gemm_v2
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import pertoken_quant, shuffle_weight
from flydsl.runtime.device import get_rocm_arch

ARCH = str(get_rocm_arch())
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz
DEVICE = torch.device("cuda")


def _bench_kernel(compiled_fn, args, warmup=5, iters=20):
    for _ in range(warmup):
        compiled_fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        compiled_fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters  # us


def _make_data(M, N, K, in_dtype):
    is_fp = in_dtype in ("fp16", "bf16")
    if is_fp:
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a = torch.rand(M, K, device=DEVICE, dtype=torch_dtype)
        b_raw = torch.rand(N, K, device=DEVICE, dtype=torch_dtype)
        sa = sb = torch.empty(0, device=DEVICE, dtype=torch.float32)
        ref = a.float() @ b_raw.float().T
    else:
        a_f = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
        b_f = torch.rand(N, K, device=DEVICE, dtype=torch.float32)
        a, sa = pertoken_quant(a_f, quant_dtype=DTYPE_FP8)
        b_raw, sb = pertoken_quant(b_f, quant_dtype=DTYPE_FP8)
        ref = (a.float() * sa.view(-1, 1)) @ (b_raw.float() * sb.view(-1, 1)).T
    b_shuf = shuffle_weight(b_raw, layout=(16, 16))
    return a, b_raw, b_shuf, sa, sb, ref


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _make_args(c, a, b_shuf, sa, sb, M, N, *, include_bias=False):
    args = [
        c.view(-1),
        _as_i8(a.view(-1)),
        _as_i8(b_shuf.view(-1)),
        sa.view(-1) if sa.numel() > 0 else sa,
        sb.view(-1) if sb.numel() > 0 else sb,
    ]
    if include_bias:
        args.append(torch.empty(0, device=c.device, dtype=c.dtype))
    args.extend([M, N, torch.cuda.current_stream()])
    return tuple(args)


def compile_one(M, N, K, tile_m, tile_n, tile_k, in_dtype, out_dtype="bf16",
                waves_per_eu=None, enable_scheduler=True, maxnreg=None,
                opt_level=None):
    """Compile v2 and old kernels, return compilation status."""
    elem_bytes = 1 if in_dtype in ("fp8",) else 2
    smem = tile_m * tile_k * elem_bytes * 2
    if smem > 65536:
        return None

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    a, b_raw, b_shuf, sa, sb, ref = _make_data(M, N, K, in_dtype)
    c = torch.zeros(M, N, device=DEVICE, dtype=torch_out_dtype)
    args = _make_args(c, a, b_shuf, sa, sb, M, N)

    hints = {}
    if maxnreg:
        hints["maxnreg"] = maxnreg
    if opt_level is not None:
        hints["opt_level"] = opt_level

    import time
    t0 = time.time()
    fn_v2 = compile_preshuffle_gemm_v2(
        N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype, out_dtype=out_dtype,
        waves_per_eu=waves_per_eu, enable_scheduler=enable_scheduler,
    )
    compiled_v2 = flyc.compile[hints](fn_v2, *args) if hints else flyc.compile(fn_v2, *args)
    t_v2 = time.time() - t0

    t0 = time.time()
    fn_old = compile_preshuffle_gemm_a8(
        N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    args_old = _make_args(c, a, b_shuf, sa, sb, M, N, include_bias=True)
    compiled_old = flyc.compile(fn_old, *args_old)
    t_old = time.time() - t0

    tile_str = f"{tile_m}x{tile_n}x{tile_k}"
    print(f"  {tile_str:>14s}  v2: {t_v2:.1f}s  old: {t_old:.1f}s  [OK]")
    return {"tile": tile_str}


def bench_one(M, N, K, tile_m, tile_n, tile_k, in_dtype, out_dtype="bf16",
              warmup=5, iters=20, check_correctness=True, waves_per_eu=None,
              enable_scheduler=True, maxnreg=None, opt_level=None,
              llvm_opts=None):
    elem_bytes = 1 if in_dtype in ("fp8",) else 2
    smem = tile_m * tile_k * elem_bytes * 2
    if smem > 65536:
        return None  # LDS overflow

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    a, b_raw, b_shuf, sa, sb, ref = _make_data(M, N, K, in_dtype)

    # ── v2 (layout API) ──────────────────────────────────────────
    hints = {}
    if maxnreg:
        hints["maxnreg"] = maxnreg
    if opt_level is not None:
        hints["opt_level"] = opt_level
    if llvm_opts:
        hints["llvm_options"] = llvm_opts
    fn_v2 = compile_preshuffle_gemm_v2(
        N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype, out_dtype=out_dtype,
        waves_per_eu=waves_per_eu, enable_scheduler=enable_scheduler,
    )
    c_v2 = torch.zeros(M, N, device=DEVICE, dtype=torch_out_dtype)
    args_v2 = _make_args(c_v2, a, b_shuf, sa, sb, M, N)
    compiled_v2 = flyc.compile[hints](fn_v2, *args_v2) if hints else flyc.compile(fn_v2, *args_v2)
    us_v2 = _bench_kernel(compiled_v2, args_v2, warmup=warmup, iters=iters)

    # ── old path ──────────────────────────────────────────────────
    us_old = 0.0
    compiled_old = None
    try:
        fn_old = compile_preshuffle_gemm_a8(
            N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )
        c_old = torch.zeros(M, N, device=DEVICE, dtype=torch_out_dtype)
        args_old = _make_args(c_old, a, b_shuf, sa, sb, M, N, include_bias=True)
        compiled_old = flyc.compile(fn_old, *args_old)
        us_old = _bench_kernel(compiled_old, args_old, warmup=warmup, iters=iters)
    except (ValueError, RuntimeError) as e:
        print(f"    (old kernel unsupported: {e})")
        c_old = torch.zeros(M, N, device=DEVICE, dtype=torch_out_dtype)
        args_old = _make_args(c_old, a, b_shuf, sa, sb, M, N, include_bias=True)

    flops = 2 * M * N * K
    tflops_v2 = flops / (us_v2 / 1e6) / 1e12
    tflops_old = flops / (us_old / 1e6) / 1e12 if us_old > 0 else 0.0
    ratio = tflops_v2 / tflops_old * 100 if tflops_old > 0 else float("inf")

    # correctness
    err_v2 = err_old = None
    if check_correctness:
        compiled_v2(*args_v2)
        if compiled_old:
            compiled_old(*args_old)
        torch.cuda.synchronize()
        err_v2 = ((c_v2.float() - ref).abs() / (ref.abs() + 1e-6)).mean().item()
        if compiled_old:
            err_old = ((c_old.float() - ref).abs() / (ref.abs() + 1e-6)).mean().item()

    return {
        "M": M, "N": N, "K": K,
        "tile": f"{tile_m}x{tile_n}x{tile_k}",
        "k_iters": tile_k // 32,
        "us_v2": us_v2, "us_old": us_old,
        "tflops_v2": tflops_v2, "tflops_old": tflops_old,
        "ratio": ratio,
        "err_v2": err_v2, "err_old": err_old,
    }


# ── Default benchmark configs ─────────────────────────────────────

DEFAULT_CONFIGS = {
    "fp16": [
        # (M, N, K, tile_m, tile_n, tile_k)
        (128,  5120, 8192,  64, 128,  64),   # k=2, best tile
        (128,  5120, 8192, 128, 128,  64),   # k=2
        (128,  5120, 8192,  64, 128, 128),   # k=4
        (32,   5120, 8192,  32,  64, 512),   # k=16, stress test
        (5120, 5120, 8192,  64, 128,  64),   # large, compute-bound
        (5120, 5120, 8192,  64, 128, 128),   # large, k=4
    ],
    "bf16": [
        (128,  5120, 8192,  64, 128,  64),
        (128,  5120, 8192,  64, 128, 128),
        (5120, 5120, 8192,  64, 128,  64),
    ],
    "fp8": [
        (16,   5120, 8192,  16,  64, 256),
        (128,  5120, 8192,  64, 128, 128),
        (128,  5120, 8192,  64, 256, 256),
        (5120, 5120, 8320,  64, 256, 128),
        (5120, 5120, 8320,  64, 256, 256),
    ],
}

SWEEP_TILES = [
    (32,  64,  64),
    (32,  64, 128),
    (32,  64, 256),
    (32,  64, 512),
    (64, 128,  64),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256,  64),
    (64, 256, 128),
    (128, 128, 64),
    (128, 256, 64),
]


def print_results(results, in_dtype):
    print()
    hdr = f"{'tile':>14s} {'k':>2s} | {'v2 us':>8s} {'v2 TF':>7s} | {'old us':>8s} {'old TF':>7s} | {'ratio':>6s}"
    if results and results[0].get("err_v2") is not None:
        hdr += f" | {'err_v2':>7s} {'err_old':>7s}"
    print(f"  {in_dtype.upper()} (M={results[0]['M']}, N={results[0]['N']}, K={results[0]['K']})")
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for r in results:
        old_us_str = f"{r['us_old']:>8.1f}" if r['us_old'] > 0 else f"{'n/a':>8s}"
        old_tf_str = f"{r['tflops_old']:>7.1f}" if r['tflops_old'] > 0 else f"{'n/a':>7s}"
        ratio_str = f"{r['ratio']:>5.1f}%" if r['ratio'] != float("inf") else f"{'v2only':>6s}"
        line = (f"  {r['tile']:>14s} {r['k_iters']:>2d} | "
                f"{r['us_v2']:>8.1f} {r['tflops_v2']:>7.1f} | "
                f"{old_us_str} {old_tf_str} | "
                f"{ratio_str}")
        if r.get("err_v2") is not None:
            ev2 = f"{r['err_v2']:>7.4f}" if r['err_v2'] is not None else f"{'n/a':>7s}"
            eo = f"{r['err_old']:>7.4f}" if r['err_old'] is not None else f"{'n/a':>7s}"
            line += f" | {ev2} {eo}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Benchmark preshuffle_gemm v2 vs old")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["fp16", "bf16", "fp8"],
                        help="Data type to benchmark (default: all)")
    parser.add_argument("-M", type=int, default=None)
    parser.add_argument("-N", type=int, default=None)
    parser.add_argument("-K", type=int, default=None)
    parser.add_argument("--tile_m", type=int, default=None)
    parser.add_argument("--tile_n", type=int, default=None)
    parser.add_argument("--tile_k", type=int, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all tile configs for given M/N/K")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness check")
    parser.add_argument("--waves_per_eu", type=int, default=None,
                        help="Set waves_per_eu hint for v2 kernel (e.g. 3)")
    parser.add_argument("--compile_only", action="store_true",
                        help="Compile only (no benchmark). Use with FLYDSL_DUMP_IR=1 for VGPR analysis")
    parser.add_argument("--no_scheduler", action="store_true",
                        help="Disable hot_loop_scheduler in v2 kernel")
    parser.add_argument("--maxnreg", type=int, default=None,
                        help="Set max VGPR count for v2 kernel (e.g. 168)")
    parser.add_argument("--opt_level", type=int, default=None,
                        help="LLVM optimization level for v2 kernel (default: 2)")
    parser.add_argument("--no_post_misched", action="store_true",
                        help="Disable LLVM post-RA machine scheduling")
    parser.add_argument("--lsr_drop", action="store_true",
                        help="Set lsr-drop-solution=True")
    args = parser.parse_args()

    llvm_opts = {}
    if args.no_post_misched:
        llvm_opts["enable-post-misched"] = False
    if args.lsr_drop:
        llvm_opts["lsr-drop-solution"] = True
    if not llvm_opts:
        llvm_opts = None

    dtypes = [args.dtype] if args.dtype else ["fp16", "bf16", "fp8"]

    print(f"GPU: {ARCH}")
    wpe_str = f", waves_per_eu={args.waves_per_eu}" if args.waves_per_eu else ""
    print(f"Pipeline comparison: v2 (layout API{wpe_str}) vs old (manual)")
    print("=" * 78)

    for dt in dtypes:
        # Compile-only mode
        if args.compile_only:
            if args.M and args.tile_m:
                M, N, K = args.M, args.N or 5120, args.K or 8192
                compile_one(M, N, K, args.tile_m, args.tile_n or 128, args.tile_k or 64,
                            dt, waves_per_eu=args.waves_per_eu,
                            enable_scheduler=not args.no_scheduler,
                            maxnreg=args.maxnreg, opt_level=args.opt_level)
            else:
                configs = DEFAULT_CONFIGS.get(dt, [])
                for M, N, K, tm, tn, tk in configs:
                    compile_one(M, N, K, tm, tn, tk, dt, waves_per_eu=args.waves_per_eu,
                                enable_scheduler=not args.no_scheduler,
                            maxnreg=args.maxnreg, opt_level=args.opt_level)
            continue

        # Custom single config
        if args.M and args.tile_m and not args.sweep:
            M, N, K = args.M, args.N or 5120, args.K or 8192
            r = bench_one(M, N, K, args.tile_m, args.tile_n or 128, args.tile_k or 64,
                          dt, warmup=args.warmup, iters=args.iters,
                          check_correctness=not args.no_check,
                          waves_per_eu=args.waves_per_eu,
                          enable_scheduler=not args.no_scheduler,
                          maxnreg=args.maxnreg, opt_level=args.opt_level,
                          llvm_opts=llvm_opts)
            if r:
                print_results([r], dt)
            continue

        # Sweep mode
        if args.sweep:
            M = args.M or 128
            N = args.N or 5120
            K = args.K or 8192
            results = []
            for tm, tn, tk in SWEEP_TILES:
                if tm > M:
                    continue
                r = bench_one(M, N, K, tm, tn, tk, dt,
                              warmup=args.warmup, iters=args.iters,
                              check_correctness=not args.no_check,
                              waves_per_eu=args.waves_per_eu,
                              enable_scheduler=not args.no_scheduler,
                              maxnreg=args.maxnreg, opt_level=args.opt_level,
                              llvm_opts=llvm_opts)
                if r:
                    results.append(r)
            if results:
                print_results(results, dt)
            continue

        # Default configs
        configs = DEFAULT_CONFIGS.get(dt, [])
        if not configs:
            continue
        # Group by (M, N, K)
        groups = {}
        for M, N, K, tm, tn, tk in configs:
            key = (M, N, K)
            groups.setdefault(key, []).append((tm, tn, tk))

        for (M, N, K), tiles in groups.items():
            results = []
            for tm, tn, tk in tiles:
                r = bench_one(M, N, K, tm, tn, tk, dt,
                              warmup=args.warmup, iters=args.iters,
                              check_correctness=not args.no_check,
                              waves_per_eu=args.waves_per_eu,
                              enable_scheduler=not args.no_scheduler,
                              maxnreg=args.maxnreg, opt_level=args.opt_level,
                              llvm_opts=llvm_opts)
                if r:
                    results.append(r)
            if results:
                print_results(results, dt)

    print()
    print("=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
