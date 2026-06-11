#!/usr/bin/env python3
"""Simulation: cooperative TDM (tile_n=32, n_warp=2) loading ONLY A,B vs all.

This is the user's requested experiment: a non-wave-specialized config where the
4 tensors are distributed cooperatively across 2 warps, and we ablate scale by
issuing only the A and B TDM loads (tdm_load_only='a,b'). Compares against the
full 'all' load on the SAME config, plus the production wave-spec config for
reference. Same isolated-subprocess methodology as compare_a8w4_baseline.py.

Variants (config baked per row): see ROWS.
"""
import argparse
import functools
import os
import statistics
import subprocess
import sys
from types import SimpleNamespace

# Each row: label, (M,N,K,tm,tn,tk,mw,nw,nb), wave_spec, load_only(or None)
ROWS = [
    # cooperative tile_n=32 / n_warp=2
    ("coop32 a,b", (1, 12288, 3072, 16, 32, 512, 1, 2, 4), False, "a,b"),
    ("coop32 all", (1, 12288, 3072, 16, 32, 512, 1, 2, 4), False, None),
    # cooperative tile_n=64 / n_warp=2 (same warps, wider N tile)
    ("coop64 a,b", (1, 12288, 3072, 16, 64, 512, 1, 2, 4), False, "a,b"),
    ("coop64 all", (1, 12288, 3072, 16, 64, 512, 1, 2, 4), False, None),
    # production reference: wave-spec tile_n=64 / n_warp=4
    ("ws64   all", (1, 12288, 3072, 16, 64, 512, 1, 4, 4), True, None),
    ("ws64   a,b", (1, 12288, 3072, 16, 64, 512, 1, 4, 4), True, "a,b"),
]

_RESULT_PREFIX = "BENCH_RESULT "


def _make_args(cfg, warmup, iters, cold, wave_spec):
    M, N, K, tm, tn, tk, mw, nw, nb = cfg
    return SimpleNamespace(
        data_format="a8w4", scale_mode="mxscale",
        M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=mw, n_warp=nw, num_buffers=nb,
        split_k=1, cluster_m=1, cluster_n=1,
        l2_prefetch_distance=0, out_dtype="bf16",
        fill_mode="random", benchmark=True,
        warmup=warmup, iters=iters, use_graph=not cold,
        no_flush_l2=False, no_tdm_store=False,
        wave_spec_tdm=wave_spec, waves_per_eu=None,
        use_scale_opsel=False, inst_prefetch=False,
        expert_sched_mode=True, atomic_barrier_enable=False,
        b_streaming=False, scale_load_path="tdm",
        verify_graph=False,
    )


def _worker(row_idx, warmup, iters, cold):
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    from kernels import gemm_fp8fp4_gfx1250 as mod

    _, cfg, wave_spec, load_only = ROWS[row_idx]
    base = mod.compile_mxscale_gemm
    inject = {} if load_only is None else {"tdm_load_only": load_only}

    @functools.wraps(base)
    def patched(**kw):
        kw.update(inject)
        return base(**kw)

    bench_mod.compile_mxscale_gemm = patched
    args = _make_args(cfg, warmup, iters, cold, wave_spec)
    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs}", flush=True)


def _run_subproc(row_idx, warmup, iters, cold):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
           "--row", str(row_idx), "--warmup", str(warmup), "--iters", str(iters)]
    if cold:
        cmd.append("--cold")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", file=sys.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return float(line[len(_RESULT_PREFIX):].split(",")[0])
    tail = "\n".join(out.stderr.strip().splitlines()[-3:])
    print(f"    FAILED (rc={out.returncode}): {tail}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--row", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        _worker(args.row, args.warmup, args.iters, args.cold)
        return

    mode = "COLD" if args.cold else "WARM(graph)"
    results = []
    for i, (label, cfg, ws, lo) in enumerate(ROWS):
        print(f"\n[{label}] cfg={cfg} wave_spec={ws} load_only={lo}  [{mode}]", flush=True)
        samples = []
        for r in range(args.repeat):
            v = _run_subproc(i, args.warmup, args.iters, args.cold)
            if v is not None:
                samples.append(v)
                print(f"   {r+1}/{args.repeat}: {v:.2f} us", flush=True)
        if samples:
            results.append((label, statistics.mean(samples), min(samples), max(samples)))
        else:
            results.append((label, None, None, None))

    print(f"\n\n## A,B-only simulation ({mode}, avg/min/max us over {args.repeat} runs)\n")
    print("| variant | avg | min | max |")
    print("|---|---|---|---|")
    for label, a, mn, mx in results:
        if a is None:
            print(f"| {label} | n/a | | |")
        else:
            print(f"| {label} | {a:.2f} | {mn:.2f} | {mx:.2f} |")
    print("\ncoop32/64 = cooperative TDM (n_warp=2); ws64 = wave-spec (n_warp=4). "
          "'a,b' loads only A,B TDM (scale skipped, numerically invalid).")


if __name__ == "__main__":
    main()
