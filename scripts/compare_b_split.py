#!/usr/bin/env python3
"""A8W4 GEMM: B-split (B over 2 loader waves) vs production 'all', via the same
isolated-subprocess methodology as compare_a8w4_baseline.py (low variance).

Each (variant, config, repeat) runs in its own subprocess calling the test
harness's _run_benchmark. We monkeypatch compile_mxscale_gemm to inject the
ablation knob:
  - all     : production kernel (tdm_load_only='all')
  - b_split : B streamed by two loader waves (tile_n/2 N each), A_scale dropped

NOTE: b_split is numerically INVALID (A_scale not loaded); this measures latency
only. Reports avg/min/max us + speedup(all/b_split).
"""
import argparse
import functools
import os
import statistics
import subprocess
import sys
from types import SimpleNamespace

DEFAULT_CONFIGS = [
    (1, 12288, 3072, 16, 64, 512, 1, 4, 4),
    (4096, 12288, 3072, 32, 256, 256, 1, 4, 4),
]

# variant -> compile-knob injection. tdm_load_only accepts a comma list of
# {a,b,as,bs}; b_split is the two-wave B path.
VARIANT_KNOBS = {
    "all": {},                                  # production (loads A,B,As,Bs)
    # single-tensor (ONLY this one loaded)
    "b": {"tdm_load_only": "b"},                # only B
    "a": {"tdm_load_only": "a"},                # only A
    "as": {"tdm_load_only": "as"},              # only A_scale
    "bs": {"tdm_load_only": "bs"},              # only B_scale
    # leave-one-out (everything EXCEPT this)
    "no_b": {"tdm_load_only": "a,as,bs"},       # all but B
    "no_a": {"tdm_load_only": "b,as,bs"},       # all but A
    "no_as": {"tdm_load_only": "a,b,bs"},       # all but A_scale
    "no_bs": {"tdm_load_only": "a,b,as"},       # all but B_scale
    "no_scale": {"tdm_load_only": "a,b"},       # neither scale
    # B distributed over two loader waves
    "b_split": {"tdm_b_split": True},
}
DEFAULT_VARIANTS = ["all", "b", "no_b", "a", "no_a", "as", "bs",
                    "no_as", "no_bs", "no_scale", "b_split"]
_RESULT_PREFIX = "BENCH_RESULT "


def _make_args(M, N, K, tm, tn, tk, mw, nw, nb, warmup, iters, cold):
    return SimpleNamespace(
        data_format="a8w4", scale_mode="mxscale",
        M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=mw, n_warp=nw, num_buffers=nb,
        split_k=1, cluster_m=1, cluster_n=1,
        l2_prefetch_distance=0, out_dtype="bf16",
        fill_mode="random", benchmark=True,
        warmup=warmup, iters=iters,
        # cold: no graph + per-iter L2 flush. warm: graph replay, L2 resident.
        use_graph=not cold,
        no_flush_l2=False, no_tdm_store=False,
        wave_spec_tdm=True, waves_per_eu=None,
        use_scale_opsel=False, inst_prefetch=False,
        expert_sched_mode=True, atomic_barrier_enable=False,
        b_streaming=False, scale_load_path="tdm",
        verify_graph=False,
    )


def _worker(variant, cfg_vals, warmup, iters, cold):
    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    from kernels import gemm_fp8fp4_gfx1250 as mod

    base = mod.compile_mxscale_gemm
    inject = VARIANT_KNOBS[variant]

    @functools.wraps(base)
    def patched(**kw):
        kw.update(inject)
        return base(**kw)

    bench_mod.compile_mxscale_gemm = patched
    args = _make_args(*cfg_vals, warmup, iters, cold)
    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs}", flush=True)


def _run_subproc(variant, cfg, warmup, iters, cold):
    cfg_csv = ",".join(str(v) for v in cfg)
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--worker", "--variant", variant, "--config", cfg_csv,
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    if cold:
        cmd.append("--cold")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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
    return (statistics.mean(us), min(us), max(us), len(us))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--cold", action="store_true",
                   help="cold: no graph + per-iter L2 flush (B read from HBM each iter)")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                   help="comma list from: " + ",".join(VARIANT_KNOBS))
    p.add_argument("--configs", default="1",
                   help="comma list of M values to run (from DEFAULT_CONFIGS)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=list(VARIANT_KNOBS), help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        cfg_vals = tuple(int(x) for x in args.config.split(","))
        _worker(args.variant, cfg_vals, args.warmup, args.iters, args.cold)
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    want_M = {int(x) for x in args.configs.split(",")}
    configs = [c for c in DEFAULT_CONFIGS if c[0] in want_M]
    mode = "COLD (L2 flush/iter)" if args.cold else "WARM (graph replay)"

    rows = []
    for cfg in configs:
        M, N, K, tm, tn, tk, mw, nw, nb = cfg
        tag = f"M{M} t({tm},{tn},{tk})"
        print(f"\n{'='*60}\n{tag}  (N={N} K={K} w{mw}x{nw} b{nb})  [{mode}]\n{'='*60}", flush=True)
        agg = {}
        for v in variants:
            samples = []
            for r in range(args.repeat):
                res = _run_subproc(v, cfg, args.warmup, args.iters, args.cold)
                if res is not None:
                    samples.append(res)
                    print(f"  [{v:>9}] {r+1}/{args.repeat}: {res[0]:.2f} us", flush=True)
            agg[v] = _agg(samples)
        rows.append((tag, agg))

    base_ref = "all"
    print(f"\n\n## TDM load ablation ({mode}, avg/min/max us over {args.repeat} isolated runs)\n")
    print("| config | variant | avg | min | max | vs all |")
    print("|---|---|---|---|---|---|")
    for tag, agg in rows:
        ref = agg.get(base_ref)
        ref_us = ref[0] if ref else None
        for v in variants:
            s = agg.get(v)
            if s is None:
                print(f"| {tag} | {v} | n/a | | | |")
                continue
            rel = f"{ref_us / s[0]:.3f}x" if (ref_us and s[0] > 0) else "--"
            print(f"| {tag} | {v} | {s[0]:.1f} | {s[1]:.1f} | {s[2]:.1f} | {rel} |")
    print("\nvariants: only/leave-one-out of {A,B,A_scale(as),B_scale(bs)}; "
          "no_scale=A,B only; b_split=B over 2 waves. 'vs all'>1 => faster than "
          "production. Only 'all' is numerically valid; others read stale LDS.")


if __name__ == "__main__":
    main()
