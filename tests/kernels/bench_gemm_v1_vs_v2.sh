#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# One-command benchmark: old pipeline vs layout API (v2)
#
# Usage:
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh          # default shapes
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh fp16     # fp16 only
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh bf16     # bf16 only
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh fp8      # fp8 only
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh sweep     # tile sweep

set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH=./
export FLYDSL_RUNTIME_ENABLE_CACHE=0

FILTER="${1:-all}"

python -u - "$FILTER" << 'PYEOF'
import os, sys, time
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
sys.path.insert(0, os.getcwd())

import torch
import flydsl.compiler as flyc
from kernels.preshuffle_gemm_v2 import compile_preshuffle_gemm_v2
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import pertoken_quant, shuffle_weight
from flydsl.runtime.device import get_rocm_arch

ARCH = str(get_rocm_arch())
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz
DEVICE = torch.device("cuda")
ITERS, WARMUP = 20, 5
SEP = "=" * 80


def bench(cf, args):
    for _ in range(WARMUP):
        cf(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        cf(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / ITERS


def run(dt, M, N, K, tm, tn, tk):
    is_fp = dt in ("fp16", "bf16")
    elem_bytes = 2 if is_fp else 1
    smem = tm * tk * elem_bytes * 2
    if smem > 65536:
        return None
    torch_out = torch.bfloat16
    if is_fp:
        td = torch.float16 if dt == "fp16" else torch.bfloat16
        a = torch.rand(M, K, device=DEVICE, dtype=td)
        b_raw = torch.rand(N, K, device=DEVICE, dtype=td)
        sa = sb = torch.empty(0, device=DEVICE, dtype=torch.float32)
    else:
        af = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
        bf = torch.rand(N, K, device=DEVICE, dtype=torch.float32)
        a, sa = pertoken_quant(af, quant_dtype=DTYPE_FP8)
        b_raw, sb = pertoken_quant(bf, quant_dtype=DTYPE_FP8)
    b_shuf = shuffle_weight(b_raw, layout=(16, 16))

    def _i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def make_args(c):
        return (c.view(-1), _i8(a.view(-1)), _i8(b_shuf.view(-1)),
                sa.view(-1) if sa.numel() > 0 else sa,
                sb.view(-1) if sb.numel() > 0 else sb,
                M, N, torch.cuda.current_stream())

    # old
    fn_old = compile_preshuffle_gemm_a8(
        N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        in_dtype=dt, out_dtype="bf16")
    c_old = torch.zeros(M, N, device=DEVICE, dtype=torch_out)
    co = flyc.compile(fn_old, *make_args(c_old))
    us_old = bench(co, make_args(c_old))

    # v2
    fn_v2 = compile_preshuffle_gemm_v2(
        N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        in_dtype=dt, out_dtype="bf16")
    c_v2 = torch.zeros(M, N, device=DEVICE, dtype=torch_out)
    cv = flyc.compile(fn_v2, *make_args(c_v2))
    us_v2 = bench(cv, make_args(c_v2))

    flops = 2 * M * N * K
    tf_old = flops / (us_old / 1e6) / 1e12
    tf_v2 = flops / (us_v2 / 1e6) / 1e12
    ratio = tf_v2 / tf_old * 100 if tf_old > 0 else 0
    return us_old, tf_old, us_v2, tf_v2, ratio


def print_header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(f"  {'tile':>14s} {'k':>2s} | {'old us':>8s} {'old TF':>7s} | {'v2 us':>8s} {'v2 TF':>7s} | {'ratio':>6s}")
    print(f"  {'-' * 66}")


def print_row(tm, tn, tk, r):
    if r is None:
        print(f"  {tm}x{tn}x{tk:>3d} {tk//32:>2d} | {'SKIP (LDS)':>40s}")
        return
    us_old, tf_old, us_v2, tf_v2, ratio = r
    tag = " <--" if ratio >= 95 else ""
    print(f"  {tm}x{tn}x{tk:>3d} {tk//32:>2d} |"
          f" {us_old:>8.1f} {tf_old:>7.1f} |"
          f" {us_v2:>8.1f} {tf_v2:>7.1f} |"
          f" {ratio:>5.1f}%{tag}")


# ── Configs ──────────────────────────────────────────────────────
FP16_SHAPES = [
    # (M, N, K, [(tm,tn,tk), ...])
    (5120, 5120, 8192, [
        (64, 128, 64), (64, 256, 64), (128, 128, 64), (128, 256, 64),
        (64, 128, 128), (64, 256, 128),
        (32, 64, 128), (32, 64, 512),
    ]),
    (128, 5120, 8192, [
        (64, 128, 64), (64, 256, 64), (128, 128, 64),
        (64, 128, 128), (32, 64, 512),
    ]),
]

BF16_SHAPES = [
    (5120, 5120, 8192, [
        (64, 128, 64), (64, 256, 64), (128, 128, 64),
        (64, 128, 128),
    ]),
]

FP8_SHAPES = [
    (5120, 5120, 8320, [(64, 256, 128)]),
    (128, 5120, 8192, [(64, 128, 128), (16, 64, 256)]),
]

SWEEP_TILES = [
    (32,64,64), (32,64,128), (32,64,256), (32,64,512),
    (64,128,64), (64,128,128), (64,128,256),
    (64,256,64), (64,256,128),
    (128,128,64), (128,256,64),
]


def run_group(dt, shapes):
    for M, N, K, tiles in shapes:
        print_header(f"{dt.upper()} M={M} N={N} K={K}")
        for tm, tn, tk in tiles:
            r = run(dt, M, N, K, tm, tn, tk)
            print_row(tm, tn, tk, r)


def run_sweep():
    for M in [128, 5120]:
        N, K = 5120, 8192
        print_header(f"FP16 sweep M={M} N={N} K={K}")
        for tm, tn, tk in SWEEP_TILES:
            if tm > M:
                continue
            r = run("fp16", M, N, K, tm, tn, tk)
            print_row(tm, tn, tk, r)


# ── Main ─────────────────────────────────────────────────────────
filt = sys.argv[1] if len(sys.argv) > 1 else "all"
print(SEP)
print(f"  Preshuffle GEMM: old pipeline vs v2 (layout API)")
print(f"  GPU: {ARCH}  |  iters={ITERS}  warmup={WARMUP}")
print(SEP)

if filt in ("all", "fp16"):
    run_group("fp16", FP16_SHAPES)
if filt in ("all", "bf16"):
    run_group("bf16", BF16_SHAPES)
if filt in ("all", "fp8"):
    run_group("fp8", FP8_SHAPES)
if filt == "sweep":
    run_sweep()

print(f"\n{SEP}\nDone.\n")
PYEOF
