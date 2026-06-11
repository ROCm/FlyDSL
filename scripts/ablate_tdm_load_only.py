#!/usr/bin/env python3
"""TDM load-only ablation: per-tensor DMA latency on the real A8W4 kernel.

For each config, runs the current kernel with FLYDSL_TDM_LOAD_ONLY set to each
of {a, b, as, bs, all}. That env var (read at trace time in
kernels/gemm_fp8fp4_gfx1250.py) suppresses the TDM predicate of the loader
waves not selected, so only the chosen tensor's DMA is actually issued while
all scheduling/fences/WMMA stay byte-identical. Compute still runs but reads
stale LDS for the suppressed tensors — fine, we only measure latency.

Each variant runs in its own subprocess (the kernel is lru_cached by compile
args, not by env, so a fresh process is required to re-read the env). Reports
avg / min / max us per variant, per config.

Run via the same env as the baseline compare:
    bash scripts/run_compare_a8w4_baseline.sh \
        python3 scripts/ablate_tdm_load_only.py   # (or just exec it directly)
or simply:
    PYTHONPATH=... python3 scripts/ablate_tdm_load_only.py
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

VARIANTS = ["a", "b", "as", "bs", "all"]

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
        use_graph=False,
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


def _worker(cfg_vals, warmup, iters):
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    from kernels import gemm_fp8fp4_gfx1250 as mod
    bench_mod.compile_mxscale_gemm = mod.compile_mxscale_gemm
    args = _make_args(*cfg_vals, warmup, iters)
    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs}", flush=True)


def _run_subproc(variant, cfg, warmup, iters):
    cfg_csv = ",".join(str(v) for v in cfg)
    env = dict(os.environ)
    env["FLYDSL_TDM_LOAD_ONLY"] = variant
    # Unique JIT cache dir per subprocess: the compile cache key does NOT include
    # FLYDSL_TDM_LOAD_ONLY, so a shared cache could serve a different variant's
    # binary. Isolate to keep each variant's timing faithful.
    env["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"flycache_{variant}_")
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--worker", "--config", cfg_csv,
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=5,
                   help="isolated subprocess runs per (variant,config)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        cfg_vals = tuple(int(x) for x in args.config.split(","))
        _worker(cfg_vals, args.warmup, args.iters)
        return

    rows = []
    for cfg in DEFAULT_CONFIGS:
        M, N, K, tm, tn, tk, mw, nw, nb = cfg
        tag = f"M{M} N{N} K{K} t({tm},{tn},{tk}) w{mw}x{nw} b{nb}"
        print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)

        stats = {}
        for variant in VARIANTS:
            samples = []
            for r in range(args.repeat):
                res = _run_subproc(variant, cfg, args.warmup, args.iters)
                if res is not None:
                    samples.append(res[0])
                    print(f"  [{variant:>3}] run {r+1}/{args.repeat}: {res[0]:.2f} us", flush=True)
            if samples:
                stats[variant] = (statistics.mean(samples), min(samples), max(samples), len(samples))
            else:
                stats[variant] = None
        rows.append((tag, stats))

    # ── summary markdown table ──
    print(f"\n\n## TDM load-only ablation (avg/min/max us over {args.repeat} runs)\n")
    print("| config | metric | " + " | ".join(VARIANTS) + " |")
    print("|---|---|" + "---|" * len(VARIANTS))
    for tag, stats in rows:
        for mi, metric in enumerate(("avg", "min", "max")):
            cells = []
            for v in VARIANTS:
                s = stats.get(v)
                cells.append("n/a" if s is None else f"{s[mi]:.1f}")
            label = tag if mi == 0 else ""
            print(f"| {label} | {metric} | " + " | ".join(cells) + " |")
    print("\nwave map: a=wave0, b=wave1, as=wave2(A_scale), bs=wave3(B_scale); "
          "all=production. Compute runs in every variant; only the listed "
          "tensor's TDM DMA is issued.")


if __name__ == "__main__":
    main()
