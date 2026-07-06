#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Head-to-head fp8 preshuffle GEMM bench: FlyDSL repo kernel vs aiter flydsl-fork.

Compares ``kernels.preshuffle_gemm.compile_preshuffle_gemm_a8`` (this repo)
against ``aiter.ops.flydsl.kernels.preshuffle_gemm.compile_preshuffle_gemm_a8``
(the aiter fork), compiled in-env by the SAME FlyDSL compiler, on the tuned
DeepSeek-V3 fp8 shapes. Because both sides go through the same compiler/codegen,
the delta isolates the kernel-source / scheduling differences (the ~6-13%
structural fork gap tracked in shapes_regression.md).

Methodology (matters — see the perf-diagnosis notes):
  * COLD timing: run_perftest is fed tensor args so its arg-rotation exceeds L2.
  * Same card, same harness for both sides.
  * INTERLEAVED per round (ours, aiter, ...) with alternating order to cancel
    clock/thermal drift; report the MEDIAN over rounds.
  * ISOLATED subprocess per shape by default (batch-in-one-process contaminates
    via clock/contention).

Usage:
    # default: the regressed shapes {4,5,6,8,10}, one isolated process each
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_vs_aiter.py

    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_vs_aiter.py --all
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_vs_aiter.py --shapes 4,6,10
    PYTHONPATH=./ python tests/kernels/bench_preshuffle_gemm_vs_aiter.py --gpu 4 --rounds 7

Requires the aiter checkout at $AITER_ROOT (default /root/aiter).
"""

import argparse
import os
import subprocess
import sys


# --- Pre-parse --gpu / --aiter-root before importing torch (HIP_VISIBLE_DEVICES
#     must be set first) -------------------------------------------------------
def _preparse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=int, default=int(os.environ.get("HIP_VISIBLE_DEVICES", "4")))
    p.add_argument("--aiter-root", default=os.environ.get("AITER_ROOT", "/root/aiter"))
    known, _ = p.parse_known_args()
    return known


_PRE = _preparse()
os.environ["HIP_VISIBLE_DEVICES"] = str(_PRE.gpu)
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "build-fly", "python_packages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Per-shape tuned configs extracted from aiter's flydsl rows of
# dsv3_a8w8_bpreshuffle_tuned_gemm.csv. aiter_tflops = the tuned CSV number.
# fields: M, N, K, tile_m, tile_n, tile_k, lds_stage, cshuffle, async, wpe, xcd
SHAPES = {
    1: dict(M=1, N=256, K=7168, tm=16, tn=64, tk=512, lds=2, csh=0, asy=1, wpe=1, xcd=0, aiter_tflops=0.54),
    2: dict(M=8, N=2112, K=7168, tm=16, tn=64, tk=512, lds=2, csh=1, asy=0, wpe=2, xcd=0, aiter_tflops=20.61),
    3: dict(M=16, N=3072, K=1536, tm=16, tn=64, tk=512, lds=1, csh=0, asy=0, wpe=0, xcd=0, aiter_tflops=31.59),
    4: dict(M=128, N=3072, K=1536, tm=32, tn=64, tk=512, lds=1, csh=0, asy=0, wpe=1, xcd=4, aiter_tflops=226.69),
    5: dict(M=256, N=6144, K=1536, tm=64, tn=128, tk=256, lds=2, csh=0, asy=1, wpe=0, xcd=4, aiter_tflops=547.33),
    6: dict(M=512, N=7168, K=4096, tm=128, tn=128, tk=256, lds=2, csh=0, asy=1, wpe=4, xcd=4, aiter_tflops=1272.9),
    7: dict(M=1024, N=256, K=7168, tm=16, tn=64, tk=512, lds=2, csh=0, asy=1, wpe=3, xcd=0, aiter_tflops=321.17),
    8: dict(M=2048, N=4096, K=512, tm=128, tn=128, tk=256, lds=1, csh=0, asy=1, wpe=0, xcd=4, aiter_tflops=793.01),
    9: dict(M=8192, N=7168, K=2048, tm=128, tn=256, tk=128, lds=2, csh=1, asy=1, wpe=2, xcd=4, aiter_tflops=2011.98),
    10: dict(M=32768, N=3072, K=1536, tm=128, tn=256, tk=128, lds=1, csh=0, asy=0, wpe=2, xcd=4, aiter_tflops=1901.37),
}
REGRESSED = [4, 5, 6, 8, 10]  # >8% gap in shapes_regression.md


def _run_one_shape(idx, args):
    """In-process: compile both kernels, verify, interleaved-median timing."""
    import statistics

    import torch

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.runtime.device import get_rocm_arch

    sys.path.insert(0, args.aiter_root)
    import aiter.ops.flydsl.kernels.preshuffle_gemm as aiter_pg

    from tests.test_common import run_perftest
    from tests.utils import pertoken_quant, shuffle_weight

    # Shim: new bindings hand a memref where the aiter fork expects a pointer.
    _orig_ptrtoint = fx.ptrtoint

    def _shim_ptrtoint(ptr):
        try:
            is_memref = "memref" in str(ptr.type)
        except Exception:
            is_memref = False
        return _orig_ptrtoint(fx.get_iter(ptr) if is_memref else ptr)

    fx.ptrtoint = _shim_ptrtoint
    aiter_pg.fx.ptrtoint = _shim_ptrtoint

    # Repo kernel: origin/main exposes compile_preshuffle_gemm; older branches
    # (pre-#754 rename) exposed compile_preshuffle_gemm_a8. Neither honors the
    # aiter-fork-only lds_stage / cshuffle knobs.
    import kernels.preshuffle_gemm as repo_pg

    repo_compile = getattr(repo_pg, "compile_preshuffle_gemm", None) or repo_pg.compile_preshuffle_gemm_a8

    cfg = SHAPES[idx]
    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    arch = str(get_rocm_arch())
    fp8 = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz
    dev = torch.device("cuda")

    # --- data (fp8 per-token quant, preshuffled B) ---
    a_f = torch.randn(M, K, device=dev, dtype=torch.float32)
    b_f = torch.randn(N, K, device=dev, dtype=torch.float32)
    a_q, sa = pertoken_quant(a_f, quant_dtype=fp8)
    b_q, sb = pertoken_quant(b_f, quant_dtype=fp8)
    a_q, b_q = a_q.contiguous(), b_q.contiguous()
    b_shuf = shuffle_weight(b_q, layout=(16, 16))
    ref = (a_q.float() * sa.view(-1, 1)) @ (b_q.float() * sb.view(-1, 1)).T
    bias = torch.empty(0, device=dev, dtype=torch.bfloat16)

    def as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def make_args(c, a, b, s_a, s_b):
        return (
            c.view(-1),
            as_i8(a.contiguous().view(-1)),
            as_i8(b.contiguous().view(-1)),
            s_a.contiguous().view(-1),
            s_b.contiguous().view(-1),
            bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    # --- compile both sides ---
    repo_fn = repo_compile(
        N=N,
        K=K,
        tile_m=cfg["tm"],
        tile_n=cfg["tn"],
        tile_k=cfg["tk"],
        in_dtype="fp8",
        out_dtype="bf16",
        waves_per_eu=cfg["wpe"],
        use_async_copy=bool(cfg["asy"]),
        xcd_swizzle=cfg["xcd"],
    )
    aiter_fn = aiter_pg.compile_preshuffle_gemm_a8(
        N=N,
        K=K,
        tile_m=cfg["tm"],
        tile_n=cfg["tn"],
        tile_k=cfg["tk"],
        in_dtype="fp8",
        out_dtype="bf16",
        lds_stage=cfg["lds"],
        use_cshuffle_epilog=bool(cfg["csh"]),
        waves_per_eu=cfg["wpe"],
        use_async_copy=bool(cfg["asy"]),
        xcd_swizzle=cfg["xcd"],
    )

    c0 = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)

    # Compile-only ASM dump mode: FLYDSL_DUMP_DIR/IR are set by the caller so a
    # single side lands in its own dir (NN_final_isa.s under a per-kernel subdir).
    if args.dump_side:
        fn = repo_fn if args.dump_side == "repo" else aiter_fn
        flyc.compile(fn, *make_args(c0, a_q, b_shuf, sa, sb))
        print(f"  [dump] {args.dump_side} shape{idx} -> {os.environ.get('FLYDSL_DUMP_DIR')}", flush=True)
        return True

    repo_c = flyc.compile(repo_fn, *make_args(c0, a_q, b_shuf, sa, sb))
    aiter_c = flyc.compile(aiter_fn, *make_args(c0, a_q, b_shuf, sa, sb))

    # --- correctness (cosine sim vs torch ref) ---
    def cos(fn):
        c = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        fn(*make_args(c, a_q, b_shuf, sa, sb))
        torch.cuda.synchronize()
        return torch.nn.functional.cosine_similarity(c.float().flatten(), ref.flatten(), dim=0).item()

    repo_cos, aiter_cos = cos(repo_c), cos(aiter_c)

    # --- interleaved median timing (cold: tensor args rotate past L2) ---
    def time_us(compiled):
        def launch(c, a, b, s_a, s_b):
            compiled(*make_args(c, a, b, s_a, s_b))

        _, us = run_perftest(
            launch,
            c0,
            a_q,
            b_shuf,
            sa,
            sb,
            num_iters=args.iters,
            num_warmup=args.warmup,
        )
        return us

    repo_us, aiter_us = [], []
    for r in range(args.rounds):
        if r % 2 == 0:
            repo_us.append(time_us(repo_c))
            aiter_us.append(time_us(aiter_c))
        else:
            aiter_us.append(time_us(aiter_c))
            repo_us.append(time_us(repo_c))

    r_us, a_us = statistics.median(repo_us), statistics.median(aiter_us)
    flop = 2.0 * M * N * K
    r_tf, a_tf = flop / (r_us * 1e-6) / 1e12, flop / (a_us * 1e-6) / 1e12
    gap = (a_tf - r_tf) / r_tf * 100.0  # aiter is `gap`% faster than ours

    tile = f"{cfg['tm']}x{cfg['tn']}x{cfg['tk']}"
    print(
        f"shape{idx:<2d} {M}x{N}x{K:<6d} tile={tile:<12s} xcd={cfg['xcd']} async={cfg['asy']} | "
        f"ours {r_tf:8.1f} TF ({r_us:8.2f}us cos={repo_cos:.4f}) | "
        f"aiter {a_tf:8.1f} TF ({a_us:8.2f}us cos={aiter_cos:.4f}) | "
        f"aiter {gap:+.1f}% vs ours | csv {cfg['aiter_tflops']:.1f}",
        flush=True,
    )
    ok = repo_cos > args.tol and aiter_cos > args.tol
    if not ok:
        print(f"  !! correctness below tol={args.tol} (ours {repo_cos:.4f} aiter {aiter_cos:.4f})", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=_PRE.gpu)
    p.add_argument("--aiter-root", default=_PRE.aiter_root)
    p.add_argument("--shapes", type=str, default=None, help="comma-separated shape indices (1-10)")
    p.add_argument("--all", action="store_true", help="all 10 DSV3 shapes")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--rounds", type=int, default=7, help="interleaved timing rounds; median reported")
    p.add_argument("--tol", type=float, default=0.99, help="min cosine similarity")
    p.add_argument("--no-isolate", action="store_true", help="run all shapes in one process (contaminates)")
    p.add_argument("--dump-asm", type=str, default=None, help="compile-only; dump final ISA of both sides under DIR")
    p.add_argument("--_shape", type=int, default=None, help=argparse.SUPPRESS)  # subprocess worker
    p.add_argument("--dump-side", choices=["repo", "aiter"], default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.all:
        indices = sorted(SHAPES)
    elif args.shapes:
        indices = [int(x) for x in args.shapes.split(",")]
    else:
        indices = REGRESSED

    # Worker mode: one shape, in-process.
    if args._shape is not None:
        ok = _run_one_shape(args._shape, args)
        sys.exit(0 if ok else 1)

    # ASM dump mode: one subprocess per side per shape, each with its own dump dir.
    if args.dump_asm:
        for i in indices:
            for side in ("repo", "aiter"):
                ddir = os.path.join(os.path.abspath(args.dump_asm), f"shape{i}", side)
                os.makedirs(ddir, exist_ok=True)
                env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=ddir)
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--_shape",
                    str(i),
                    "--dump-side",
                    side,
                    "--gpu",
                    str(args.gpu),
                    "--aiter-root",
                    args.aiter_root,
                ]
                subprocess.run(cmd, env=env)
                isa = subprocess.run(
                    ["bash", "-c", f"find {ddir} -name '*_final_isa.s' | head -1"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                print(f"shape{i} {side}: {isa or '(no isa found)'}", flush=True)
        return

    if args.no_isolate:
        all_ok = all(_run_one_shape(i, args) for i in indices)
        sys.exit(0 if all_ok else 1)

    # Default: isolated subprocess per shape (clean clock state per shape).
    print(
        f"# fp8 preshuffle GEMM: FlyDSL repo vs aiter fork | gpu={args.gpu} "
        f"iters={args.iters} warmup={args.warmup} rounds={args.rounds}",
        flush=True,
    )
    all_ok = True
    for i in indices:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--_shape",
            str(i),
            "--gpu",
            str(args.gpu),
            "--aiter-root",
            args.aiter_root,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--rounds",
            str(args.rounds),
            "--tol",
            str(args.tol),
        ]
        rc = subprocess.run(cmd).returncode
        all_ok = all_ok and rc == 0
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
