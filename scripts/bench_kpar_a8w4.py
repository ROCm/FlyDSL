#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the k_warp (K-parallel warp) A8W4 GEMM vs the current main kernel.

Shape matches the main decode script (M=1, N=12288, K=3072 by default). Sweeps
the k_warp kernel over tile_n in {16,32,64} x tile_k in {512,1024} with
n_warp=1, k_warp=4 (k_warp>1 requires tile_m==16, m_warp==1, and forbids
wave_specialized_tdm), and compares against the current main kernel at the same
tile (k_warp=1, wave-specialized) as the reference point.

Each measurement runs in an isolated subprocess, repeated --repeat times; median
us is reported.

Run via scripts/run_bench_kpar_a8w4.sh (sets env + triton path).
"""
from __future__ import annotations

import argparse
import functools
import os
import statistics
import subprocess
import sys
from types import SimpleNamespace

_RESULT_PREFIX = "BENCH_RESULT "

# Sweep: (tile_n, tile_k)
TILE_SWEEP = [
    (16, 512), (32, 512), (64, 512),
    (16, 1024), (32, 1024), (64, 1024),
]


def _make_args(M, N, K, tm, tn, tk, mw, nw, nb, warmup, iters, wave_spec):
    return SimpleNamespace(
        data_format="a8w4", scale_mode="mxscale",
        M=M, N=N, K=K,
        tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=mw, n_warp=nw,
        num_buffers=nb, split_k=1,
        cluster_m=1, cluster_n=1,
        l2_prefetch_distance=0,
        out_dtype="bf16", fill_mode="random",
        benchmark=True, warmup=warmup, iters=iters, use_graph=True,
        no_flush_l2=False, no_tdm_store=False,
        wave_spec_tdm=wave_spec, waves_per_eu=None,
        use_scale_opsel=False, inst_prefetch=False,
        expert_sched_mode=True, atomic_barrier_enable=False,
        b_streaming=False, scale_load_path="tdm", verify_graph=False,
    )


def _worker(kind, cfg_vals, warmup, iters):
    """kind = 'main' (current kernel, k_warp=1, wave-spec) or 'kpar' (k_warp=4)."""
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    M, N, K, tm, tn, tk = cfg_vals

    if kind == "kpar":
        from kernels import gemm_fp8fp4_gfx1250_kpar as mod
        # k_warp>1 requires m_warp=1, n_warp=1, tile_m=16, no wave-spec.
        bench_mod.compile_mxscale_gemm = functools.partial(mod.compile_mxscale_gemm, k_warp=4)
        args = _make_args(M, N, K, tm, tn, tk, mw=1, nw=1, nb=2,
                          warmup=warmup, iters=iters, wave_spec=False)
    else:
        from kernels import gemm_fp8fp4_gfx1250 as mod
        bench_mod.compile_mxscale_gemm = mod.compile_mxscale_gemm
        # Current main kernel at the same tile: n_warp=4 (its decode config), wave-spec on.
        args = _make_args(M, N, K, tm, tn, tk, mw=1, nw=4, nb=4,
                          warmup=warmup, iters=iters, wave_spec=True)

    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs}", flush=True)


def _run_subproc(kind, cfg, warmup, iters):
    cfg_csv = ",".join(str(v) for v in cfg)
    cmd = [sys.executable, "-u", os.path.abspath(__file__),
           "--worker", "--kind", kind, "--config", cfg_csv,
           "--warmup", str(warmup), "--iters", str(iters)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", file=sys.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            us, tf, gbs = (float(x) for x in line[len(_RESULT_PREFIX):].split(","))
            return (us, tf, gbs)
    tail = "\n".join(out.stderr.strip().splitlines()[-3:])
    print(f"    FAILED (rc={out.returncode}): {tail}", file=sys.stderr)
    return None


def _median(kind, cfg, warmup, iters, repeat):
    samples = []
    for r in range(repeat):
        res = _run_subproc(kind, cfg, warmup, iters)
        if res is not None:
            samples.append(res)
            print(f"  [{kind}] {cfg[3]}x{cfg[4]} ... wait", end="", file=sys.stderr)
    if not samples:
        return None
    return (statistics.median(s[0] for s in samples),
            statistics.median(s[1] for s in samples))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=1)
    p.add_argument("-N", type=int, default=12288)
    p.add_argument("-K", type=int, default=3072)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--kind", choices=["main", "kpar"], help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        cfg_vals = tuple(int(x) for x in args.config.split(","))
        _worker(args.kind, cfg_vals, args.warmup, args.iters)
        return

    rows = []
    for tn, tk in TILE_SWEEP:
        cfg = (args.M, args.N, args.K, tn, tk)  # passed as (M,N,K,tile_n,tile_k)
        # worker expects (M,N,K,tm,tn,tk); tile_m fixed 16.
        wcfg = (args.M, args.N, args.K, 16, tn, tk)
        tag = f"t(16,{tn},{tk})"
        print(f"\n{'='*60}\n{tag}\n{'='*60}", flush=True)

        main_res = []
        kpar_res = []
        for r in range(args.repeat):
            m = _run_subproc("main", wcfg, args.warmup, args.iters)
            if m: main_res.append(m)
            k = _run_subproc("kpar", wcfg, args.warmup, args.iters)
            if k: kpar_res.append(k)
            print(f"  run {r+1}/{args.repeat}: "
                  f"main={'-' if not m else f'{m[0]:.2f}us'} "
                  f"kpar={'-' if not k else f'{k[0]:.2f}us'}", flush=True)

        mm = (statistics.median(s[0] for s in main_res),
              statistics.median(s[1] for s in main_res)) if main_res else None
        kk = (statistics.median(s[0] for s in kpar_res),
              statistics.median(s[1] for s in kpar_res)) if kpar_res else None
        rows.append((tag, mm, kk))

    print(f"\n\n{'='*78}\n  A8W4 decode: k_warp=4 (kpar) vs current main kernel  "
          f"(median of {args.repeat})\n  M={args.M} N={args.N} K={args.K}\n{'='*78}")
    hdr = f"{'tile':<16} {'main us':>9} {'kpar us':>9} {'speedup':>9} {'main TF':>9} {'kpar TF':>9}"
    print(hdr); print("-" * len(hdr))
    for tag, mm, kk in rows:
        m_us = f"{mm[0]:.1f}" if mm else "n/a"
        k_us = f"{kk[0]:.1f}" if kk else "n/a"
        sp = f"{mm[0]/kk[0]:.2f}x" if (mm and kk and kk[0] > 0) else "--"
        m_tf = f"{mm[1]:.2f}" if mm else "n/a"
        k_tf = f"{kk[1]:.2f}" if kk else "n/a"
        print(f"{tag:<16} {m_us:>9} {k_us:>9} {sp:>9} {m_tf:>9} {k_tf:>9}")
    print("-" * len(hdr))
    print("speedup > 1.0 means kpar (k_warp=4) is faster than the current main kernel")


if __name__ == "__main__":
    main()
