#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare A8W4 GEMM perf: current kernel vs the reference baseline kernel.

For each shape/tile config, compile+benchmark both
  kernels.gemm_fp8fp4_gfx1250            (current, with tail prefetch fix)
  kernels.gemm_fp8fp4_gfx1250_baseline   (reference, no LDS prefetch)
reusing the test harness's data prep + hipGraph timing, and print a ratio table.

Each measurement runs in an ISOLATED SUBPROCESS (fresh CUDA context / L2 / clock
state) and is repeated --repeat times; the median us is reported. This removes
the cross-kernel context pollution that made same-process timing noisy.

Run via scripts/run_compare_a8w4_baseline.sh (sets env + triton path).

Usage:
  bash scripts/run_compare_a8w4_baseline.sh
  bash scripts/run_compare_a8w4_baseline.sh --repeat 5 --warmup 10 --iters 50
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from types import SimpleNamespace


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
# Mix of loop_iters==0 (large tile_k, the fixed path) and loop_iters>0 configs.
DEFAULT_CONFIGS = [
    # tile_k=512 -> few K-tiles -> loop_iters==0 (the path the fix targets)
    (1, 12288, 3072, 16, 64, 512, 1, 4, 4),
    (16, 12288, 3072, 16, 64, 512, 1, 4, 4),
    # (1, 12288, 3072, 16, 128, 512, 1, 4, 4),
    # (64, 12288, 3072, 16, 64, 512, 1, 4, 4),
    # tile_k=256 -> loop_iters>0 (regression / already-prefetched baseline path)
    # (1, 12288, 3072, 16, 128, 256, 1, 4, 4),
    (64, 12288, 3072, 16, 128, 256, 1, 4, 4),
    (4096, 12288, 3072, 32, 256, 256, 1, 4, 4),
]

_RESULT_PREFIX = "BENCH_RESULT "


def _make_args(M, N, K, tm, tn, tk, mw, nw, nb, warmup, iters):
    return SimpleNamespace(
        data_format="a8w4",
        scale_mode="mxscale",
        M=M, N=N, K=K,
        tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=mw, n_warp=nw,
        num_buffers=nb,
        split_k=1,
        cluster_m=1, cluster_n=1,
        l2_prefetch_distance=0,
        out_dtype="bf16",
        fill_mode="random",
        benchmark=True,
        warmup=warmup, iters=iters,
        use_graph=True,
        no_flush_l2=False,
        no_tdm_store=False,
        wave_spec_tdm=True,
        waves_per_eu=None,
        use_scale_opsel=False,
        inst_prefetch=False,
        expert_sched_mode=True,
        atomic_barrier_enable=False,
        b_streaming=False,
        scale_load_path="tdm",
        verify_graph=False,
    )


def _worker(kernel, cfg_vals, warmup, iters):
    """Run a single benchmark in this (isolated) process; print BENCH_RESULT us,tf,gbs."""
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    if kernel == "baseline":
        from kernels import gemm_fp8fp4_gfx1250_baseline as mod
        # bench_mod hardcodes a8w4 -> preshuffle_b_16x16_tiled (the current
        # kernel's tile-contiguous layout). The baseline kernel reads the flat
        # 16x16 layout, so route its B preshuffle back to the flat variant.
        from tests.kernels.utils import fp4_utils
        fp4_utils.preshuffle_b_16x16_tiled = (
            lambda b, rows, cols, tile_n, tile_kb: fp4_utils.preshuffle_b_16x16(b, rows, cols)
        )
    else:
        from kernels import gemm_fp8fp4_gfx1250 as mod
    bench_mod.compile_mxscale_gemm = mod.compile_mxscale_gemm

    args = _make_args(*cfg_vals, warmup, iters)
    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs}", flush=True)


def _run_subproc(kernel, cfg, warmup, iters):
    """Spawn an isolated worker; return (us, tf, gbs) or None on failure."""
    cfg_csv = ",".join(str(v) for v in cfg)
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--worker", "--kernel", kernel, "--config", cfg_csv,
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", file=sys.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            us, tf, gbs = (float(x) for x in line[len(_RESULT_PREFIX):].split(","))
            return (us, tf, gbs)
    # No result line: surface a short tail of the worker's stderr for diagnosis.
    tail = "\n".join(out.stderr.strip().splitlines()[-3:])
    print(f"    FAILED (rc={out.returncode}): {tail}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=5,
                   help="isolated subprocess runs per (kernel,config); median reported")
    # worker-mode args (internal)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--kernel", choices=["baseline", "current"], help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        cfg_vals = tuple(int(x) for x in args.config.split(","))
        _worker(args.kernel, cfg_vals, args.warmup, args.iters)
        return

    rows = []
    for cfg in DEFAULT_CONFIGS:
        M, N, K, tm, tn, tk, mw, nw, nb = cfg
        tag = f"M{M} N{N} K{K} t({tm},{tn},{tk}) w{mw}x{nw} b{nb}"
        print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)

        med = {}
        for kernel in ("baseline", "current"):
            samples = []
            for r in range(args.repeat):
                res = _run_subproc(kernel, cfg, args.warmup, args.iters)
                if res is not None:
                    samples.append(res)
                    print(f"  [{kernel}] run {r+1}/{args.repeat}: "
                          f"{res[0]:.2f} us, {res[1]:.2f} TF", flush=True)
            if samples:
                us_med = statistics.median(s[0] for s in samples)
                tf_med = statistics.median(s[1] for s in samples)
                gbs_med = statistics.median(s[2] for s in samples)
                med[kernel] = (us_med, tf_med, gbs_med, len(samples))
            else:
                med[kernel] = None
        rows.append((tag, med.get("baseline"), med.get("current")))

    # ── summary table (medians) ──
    print(f"\n\n{'='*92}\n  A8W4 GEMM: current vs baseline  "
          f"(median of {args.repeat} isolated runs)\n{'='*92}")
    hdr = (f"{'config':<40} {'base us':>9} {'cur us':>9} {'speedup':>9} "
           f"{'base TF':>9} {'cur TF':>9}")
    print(hdr)
    print("-" * len(hdr))
    for tag, base, cur in rows:
        if base is None or cur is None:
            b = "n/a" if base is None else f"{base[0]:.1f}"
            c = "n/a" if cur is None else f"{cur[0]:.1f}"
            print(f"{tag:<40} {b:>9} {c:>9} {'--':>9}")
            continue
        b_us, b_tf = base[0], base[1]
        c_us, c_tf = cur[0], cur[1]
        speedup = b_us / c_us if c_us > 0 else 0.0
        print(f"{tag:<40} {b_us:>9.1f} {c_us:>9.1f} {speedup:>8.2f}x "
              f"{b_tf:>9.2f} {c_tf:>9.2f}")
    print("-" * len(hdr))
    print("speedup > 1.0 means current is faster than baseline")


if __name__ == "__main__":
    main()
