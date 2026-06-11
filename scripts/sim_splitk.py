#!/usr/bin/env python3
"""M=1 split-K sweep: does splitting K across CUs hide the serial TDM/fence chain?

All rows share tile=(16,64,256), wave-spec, a8w4. split_k>1 forces buffer-store
atomic epilogue (use_tdm_store=False). Isolated subprocess per (row,repeat);
--cold = per-iter L2 flush. Same methodology as sim_ab_only.py.
"""
import argparse
import os
import statistics
import subprocess
import sys
from types import SimpleNamespace

# label, split_k, num_buffers
ROWS = [
    ("split_k=1 nb4", 1, 4),
    ("split_k=2 nb4", 2, 4),
    ("split_k=2 nb2", 2, 2),
    ("split_k=4 nb3", 4, 3),
    ("split_k=4 nb2", 4, 2),
]
CFG = (1, 12288, 3072, 16, 64, 256, 1, 4)  # M,N,K,tm,tn,tk,mw,nw
_RESULT_PREFIX = "BENCH_RESULT "


def _make_args(split_k, nb, warmup, iters, cold):
    M, N, K, tm, tn, tk, mw, nw = CFG
    return SimpleNamespace(
        data_format="a8w4", scale_mode="mxscale",
        M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=mw, n_warp=nw, num_buffers=nb,
        split_k=split_k, cluster_m=1, cluster_n=1,
        l2_prefetch_distance=0, out_dtype="bf16",
        fill_mode="random", benchmark=True,
        warmup=warmup, iters=iters, use_graph=not cold,
        no_flush_l2=False, no_tdm_store=(split_k > 1),
        wave_spec_tdm=True, waves_per_eu=None,
        use_scale_opsel=False, inst_prefetch=False,
        expert_sched_mode=True, atomic_barrier_enable=False,
        b_streaming=False, scale_load_path="tdm",
        verify_graph=False,
    )


def _worker(row_idx, warmup, iters, cold):
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    from kernels import gemm_fp8fp4_gfx1250 as mod
    bench_mod.compile_mxscale_gemm = mod.compile_mxscale_gemm
    _, sk, nb = ROWS[row_idx]
    args = _make_args(sk, nb, warmup, iters, cold)
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
    for i, (label, sk, nb) in enumerate(ROWS):
        print(f"\n[{label}] cfg={CFG} split_k={sk} nb={nb}  [{mode}]", flush=True)
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

    print(f"\n\n## M=1 split-K sweep ({mode}, avg/min/max us over {args.repeat} runs, "
          f"tile=(16,64,256))\n")
    print("| variant | avg | min | max |")
    print("|---|---|---|---|")
    for label, a, mn, mx in results:
        print(f"| {label} | " + ("n/a | |" if a is None else f"{a:.2f} | {mn:.2f} | {mx:.2f}") + " |")
    print("\nsplit_k>1 uses buffer-store atomic epilogue. All wave-spec a8w4.")


if __name__ == "__main__":
    main()
