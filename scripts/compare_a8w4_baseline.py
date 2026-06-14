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
    (1, 12288, 3072, 16, 128, 512, 1, 4, 4),
    (64, 12288, 3072, 16, 64, 512, 1, 4, 4),
    # tile_k=256 -> loop_iters>0 (regression / already-prefetched baseline path)
    (1, 12288, 3072, 16, 128, 256, 1, 4, 4),
    (64, 12288, 3072, 16, 128, 256, 1, 4, 4),
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
        pf_depth_wmma=int(os.environ.get("PF_DEPTH_WMMA", "4")),
    )


# Compile-time env per mode. PF_* flags are read inside the kernel at trace time,
# so setting them in this (isolated) process before compile selects the variant.
_MODES = ("baseline", "legacy", "fullpf")
_MODE_ENV = {
    "baseline": {"PF_QUADRANT": None, "PF_PIPELINE": None, "PF_FULL_PREFETCH": None},
    "legacy": {"PF_QUADRANT": "1", "PF_PIPELINE": "1", "PF_FULL_PREFETCH": None},
    "fullpf": {"PF_QUADRANT": "1", "PF_PIPELINE": "1", "PF_FULL_PREFETCH": "1"},
}


def _worker(mode, cfg_vals, warmup, iters):
    """Run accuracy + benchmark for one mode in this (isolated) process.

    Prints: BENCH_RESULT us,tf,gbs,cosine,passed
    """
    import contextlib
    import io

    for k, v in _MODE_ENV[mode].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    # Isolate each mode's disk cache: the PF_* flags are not part of the JIT cache
    # key, so a shared dir would let one mode load another mode's compiled kernel.
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = f"/tmp/cmp_cache_{mode}"

    import tests.kernels.test_gemm_fp8fp4_gfx1250 as bench_mod
    if mode == "baseline":
        from kernels import gemm_fp8fp4_gfx1250_baseline as mod

        # The reference kernel predates pf_depth_wmma (and any other newer pipeline
        # knobs); drop kwargs it does not accept so the shared harness drives both.
        _orig_compile = mod.compile_mxscale_gemm

        def _baseline_compile(**kw):
            kw.pop("pf_depth_wmma", None)
            return _orig_compile(**kw)

        bench_mod.compile_mxscale_gemm = _baseline_compile
    else:
        from kernels import gemm_fp8fp4_gfx1250 as mod
        bench_mod.compile_mxscale_gemm = mod.compile_mxscale_gemm

    M, N, K, tm, tn, tk, mw, nw, nb = cfg_vals
    pf_depth = None if mode == "baseline" else int(os.environ.get("PF_DEPTH_WMMA", "4"))

    # ── accuracy: run the test body once, capture cosine, pass = no AssertionError ──
    cos = float("nan")
    passed = 0
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            bench_mod._run_mxscale_gemm_test(
                "a8w4", M, N, K, tm, tn, tk, mw, nw, nb,
                use_tdm_store=True, out_dtype="bf16",
                wave_specialized_tdm=True, cluster_m=1, cluster_n=1,
                split_k=1, scale_load_path="tdm", pf_depth_wmma=pf_depth,
            )
        passed = 1
    except AssertionError:
        passed = 0
    for line in _buf.getvalue().splitlines():
        if "Cosine similarity:" in line:
            cos = float(line.split(":")[1].strip())

    # ── perf ──
    args = _make_args(*cfg_vals, warmup, iters)
    us, tf, gbs = bench_mod._run_benchmark(args)
    print(f"{_RESULT_PREFIX}{us},{tf},{gbs},{cos},{passed}", flush=True)


def _run_subproc(mode, cfg, warmup, iters):
    """Spawn an isolated worker; return (us, tf, gbs, cos, passed) or None on failure."""
    cfg_csv = ",".join(str(v) for v in cfg)
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--worker", "--kernel", mode, "--config", cfg_csv,
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", file=sys.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            parts = line[len(_RESULT_PREFIX):].split(",")
            us, tf, gbs, cos = (float(x) for x in parts[:4])
            passed = int(parts[4]) if len(parts) > 4 else -1
            return (us, tf, gbs, cos, passed)
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
    p.add_argument("--configs", default=None,
                   help="override DEFAULT_CONFIGS: 'M,N,K,tm,tn,tk,mw,nw,nb' tuples "
                        "separated by ';' (e.g. '1,12288,4096,32,256,256,1,4,2;...')")
    # worker-mode args (internal)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--kernel", choices=list(_MODES), help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        cfg_vals = tuple(int(x) for x in args.config.split(","))
        _worker(args.kernel, cfg_vals, args.warmup, args.iters)
        return

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split(",")) for c in args.configs.split(";") if c.strip()]

    rows = []
    for cfg in configs:
        M, N, K, tm, tn, tk, mw, nw, nb = cfg
        tag = f"M{M} N{N} K{K} t({tm},{tn},{tk}) w{mw}x{nw} b{nb}"
        print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)

        med = {}
        for mode in _MODES:
            samples = []
            for r in range(args.repeat):
                res = _run_subproc(mode, cfg, args.warmup, args.iters)
                if res is not None:
                    samples.append(res)
                    _ok = "ok" if res[4] == 1 else ("FAIL" if res[4] == 0 else "?")
                    print(f"  [{mode}] run {r+1}/{args.repeat}: "
                          f"{res[0]:.2f} us, {res[1]:.2f} TF, cos={res[3]:.5f} {_ok}", flush=True)
            if samples:
                med[mode] = (
                    statistics.median(s[0] for s in samples),  # us
                    statistics.median(s[1] for s in samples),  # tf
                    statistics.median(s[2] for s in samples),  # gbs
                    statistics.median(s[3] for s in samples),  # cosine
                    min(s[4] for s in samples),                # passed (worst)
                )
            else:
                med[mode] = None
        rows.append((tag, med))

    # ── perf summary table (medians) ── speedups vs baseline ──
    print(f"\n\n{'='*104}\n  A8W4 GEMM: baseline vs legacy vs full-prefetch  "
          f"(median of {args.repeat} isolated runs)\n{'='*104}")
    hdr = (f"{'config':<38} {'base us':>8} {'leg us':>8} {'pf us':>8} "
           f"{'leg/base':>9} {'pf/base':>9} {'pf/leg':>8}")
    print(hdr)
    print("-" * len(hdr))

    def _us(m):
        return m[0] if m else None

    for tag, med in rows:
        b, lg, pf = _us(med.get("baseline")), _us(med.get("legacy")), _us(med.get("fullpf"))
        def _f(x):
            return f"{x:.1f}" if x is not None else "n/a"
        def _sp(num, den):
            return f"{den / num:.2f}x" if (num and den) else "--"
        print(f"{tag:<38} {_f(b):>8} {_f(lg):>8} {_f(pf):>8} "
              f"{_sp(lg, b):>9} {_sp(pf, b):>9} {_sp(pf, lg):>8}")
    print("-" * len(hdr))
    print("ratio > 1.0 = faster than the denominator (base=baseline, leg=legacy)")

    # ── accuracy summary ── cosine + pass/fail per mode ──
    print(f"\n{'='*104}\n  Accuracy (cosine vs torch ref; PASS = within a8w4 tolerance)\n{'='*104}")
    ahdr = f"{'config':<38} {'baseline':>18} {'legacy':>18} {'full-prefetch':>18}"
    print(ahdr)
    print("-" * len(ahdr))
    for tag, med in rows:
        def _acc(m):
            if not m:
                return "n/a"
            return f"{m[3]:.5f} {'PASS' if m[4] == 1 else 'FAIL'}"
        print(f"{tag:<38} {_acc(med.get('baseline')):>18} "
              f"{_acc(med.get('legacy')):>18} {_acc(med.get('fullpf')):>18}")
    print("-" * len(ahdr))


if __name__ == "__main__":
    main()
