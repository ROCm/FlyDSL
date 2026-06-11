#!/usr/bin/env python3
"""A8W4 GEMM: current kernel vs reference baseline, isolated and cache-safe.

For each config, benchmarks:
  - baseline : kernels.gemm_fp8fp4_gfx1250_baseline (reference, no LDS prefetch)
  - current  : kernels.gemm_fp8fp4_gfx1250 (FLYDSL_TDM_LOAD_ONLY=all -> full,
               correctness-valid kernel; the ablation knob is inert at "all")

Each (kernel, config, repeat) runs in its own subprocess with a UNIQUE JIT
cache dir (FLYDSL_RUNTIME_CACHE_DIR), so the env-insensitive compile cache key
can never serve a wrong artifact across runs. Reports avg / min / max us and
speedup (baseline / current) as a markdown table.

Run (same env as run_compare_a8w4_baseline.sh):
    bash scripts/run_compare_a8w4_baseline.sh   # then swap the exec target, OR
    PYTHONPATH=... python3 scripts/compare_tdm_vs_baseline.py
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace

# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
DEFAULT_CONFIGS = [
    (1, 12288, 3072, 16, 64, 512, 1, 4, 4),
    (16, 12288, 3072, 16, 64, 512, 1, 4, 4),
    (64, 12288, 3072, 16, 128, 256, 1, 4, 4),
    (4096, 12288, 3072, 32, 256, 256, 1, 4, 4),
]

_RESULT_PREFIX = "BENCH_RESULT "


def _make_args(M, N, K, tm, tn, tk, mw, nw, nb, warmup, iters):
    return SimpleNamespace(
        data_format="a8w4", scale_mode="mxscale",
        M=M, N=N, K=K,
        tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=mw, n_warp=nw, num_buffers=nb,
        split_k=1, cluster_m=1, cluster_n=1,
        l2_prefetch_distance=0, out_dtype="bf16",
        fill_mode="random", benchmark=True,
        warmup=warmup, iters=iters, use_graph=False,
        no_flush_l2=False, no_tdm_store=False,
        wave_spec_tdm=True, waves_per_eu=None,
        use_scale_opsel=False, inst_prefetch=False,
        expert_sched_mode=True, atomic_barrier_enable=False,
        b_streaming=False, scale_load_path="tdm",
        verify_graph=False,
    )


def _worker(kernel, cfg_vals, warmup, iters):
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    if kernel == "baseline":
        from kernels import gemm_fp8fp4_gfx1250_baseline as mod
    else:
        from kernels import gemm_fp8fp4_gfx1250 as mod
    bench_mod.compile_mxscale_gemm = mod.compile_mxscale_gemm
    args = _make_args(*cfg_vals, warmup, iters)
    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs}", flush=True)


def _run_subproc(kernel, cfg, warmup, iters):
    cfg_csv = ",".join(str(v) for v in cfg)
    env = dict(os.environ)
    # Unique cache dir per subprocess: defeats the env-insensitive compile cache
    # key so a prior run can never serve the wrong binary.
    env["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"flycache_{kernel}_")
    if kernel == "current":
        env["FLYDSL_TDM_LOAD_ONLY"] = "all"
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--worker", "--kernel", kernel, "--config", cfg_csv,
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
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


def _agg(samples):
    if not samples:
        return None
    us = [s[0] for s in samples]
    tf = [s[1] for s in samples]
    return (statistics.mean(us), min(us), max(us), statistics.median(tf), len(us))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--repeat", type=int, default=5)
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
        tag = f"M{M} t({tm},{tn},{tk})"
        print(f"\n{'='*60}\n{tag}  (N={N} K={K} w{mw}x{nw} b{nb})\n{'='*60}", flush=True)
        agg = {}
        for kernel in ("baseline", "current"):
            samples = []
            for r in range(args.repeat):
                res = _run_subproc(kernel, cfg, args.warmup, args.iters)
                if res is not None:
                    samples.append(res)
                    print(f"  [{kernel:>8}] {r+1}/{args.repeat}: {res[0]:.2f} us", flush=True)
            agg[kernel] = _agg(samples)
        rows.append((tag, agg["baseline"], agg["current"]))

    print(f"\n\n## current (TDM_LOAD_ONLY=all) vs baseline  "
          f"(avg/min/max us over {args.repeat} isolated runs, unique cache/run)\n")
    print("| config | base avg | base min | base max | cur avg | cur min | cur max | speedup(avg) |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, base, cur in rows:
        if base is None or cur is None:
            print(f"| {tag} | n/a | | | n/a | | | -- |")
            continue
        sp = base[0] / cur[0] if cur[0] > 0 else 0.0
        print(f"| {tag} | {base[0]:.1f} | {base[1]:.1f} | {base[2]:.1f} "
              f"| {cur[0]:.1f} | {cur[1]:.1f} | {cur[2]:.1f} | {sp:.3f}x |")
    print("\nspeedup > 1.0 => current faster than baseline. "
          "current runs the full correctness-valid kernel (knob inert at 'all').")


if __name__ == "__main__":
    main()
