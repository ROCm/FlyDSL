#!/usr/bin/env python3
"""Latency sweep for bmm_a16w8_gfx1250 across all tunable knobs.

Covers:
  A. M sweep       — memory→compute roofline crossing
  B. tile_m sweep  — decode-regime tile size (M=64)
  C. warp shape    — m_warp × n_warp (M=64, tile_m=64)
  D. cluster_n     — A-matrix multicast benefit (M=64)
  E. scale mode    — no_scale vs e8m0 vs fp32 (M=64)
  F. num_buffers   — pipeline depth (M=64, tile_m=64)

Usage:
    cd ~/flydsl
    python tests/kernels/bench_bmm_a16w8_sweep.py              # all groups
    python tests/kernels/bench_bmm_a16w8_sweep.py --group m_sweep
    python tests/kernels/bench_bmm_a16w8_sweep.py --group tile_warp
    python tests/kernels/bench_bmm_a16w8_sweep.py --warmup 20 --iters 200
    python tests/kernels/bench_bmm_a16w8_sweep.py --csv results.csv

Output: pretty table + optional CSV.
Each row = one config: label, M, tile_m, m_warp×n_warp, cluster_n, nb, scale_mode,
           p50_us, TFLOPS, BW_TBps.
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from flydsl.runtime.device import get_rocm_arch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

# ---------------------------------------------------------------------------
# V4 fixed shape (B=16, K=4096, N=1024, group_k=group_n=128)
# ---------------------------------------------------------------------------
B, K, N = 16, 4096, 1024
GROUP_K, GROUP_N = 128, 128
# tile_k == group_k and tile_n == group_n are hard constraints in the kernel
TILE_K, TILE_N = 128, 128

# ---------------------------------------------------------------------------
# Sweep groups
# ---------------------------------------------------------------------------

# Each entry: (label, M, tile_m, m_warp, n_warp, num_buffers, cluster_n,
#              use_e8m0_scale, no_scale, waves_per_eu)
#
# LDS footprint reference (3 stages, tile_n=tile_k=128):
#   tile_m=32: stage≈26KB, arena≈81KB  — fits, but 1 WG/WGP (>64KB)
#   tile_m=64: stage≈36KB, arena≈108KB — fits, 1 WG/WGP
#   tile_m=128: stage≈53KB, nb=3→159KB (over limit); nb=2→106KB (fits)

# A. M sweep — roofline crossing. Best decode/prefill config each regime.
#
# NOTE: tile_m=128 with gx >= 4 (M > 256) causes MEMORY_APERTURE_VIOLATION on silicon
# (confirmed 2026-05-27). Precision tests only cover M <= 256 for tile_m=128.
# Using tile_m=64 for M >= 512 until the tile_m=128 / large-grid bug is root-caused.
M_SWEEP = [
    # label               M    tm   mw nw nb  cl  e8m0  nosc  wpe
    ("M=1   dec",          1,  64,  2, 4,  3,  1, True, False, None),
    ("M=8   dec",          8,  64,  2, 4,  3,  1, True, False, None),
    ("M=32  dec",         32,  64,  2, 4,  3,  1, True, False, None),
    ("M=64  dec",         64,  64,  2, 4,  3,  1, True, False, None),
    ("M=128 dec",        128,  64,  2, 4,  3,  1, True, False, None),
    ("M=256 pre",        256, 128,  2, 4,  2,  1, True, False, None),
    # tile_m=64 below: tile_m=128 crashes for M>256 (gx>=4, untested kernel path)
    ("M=512 pre tm64",   512,  64,  2, 4,  3,  1, True, False, None),
    ("M=1024 pre tm64", 1024,  64,  2, 4,  3,  1, True, False, None),
]

# B. tile_m sweep at M=64 — how tile size affects decode throughput
TILE_WARP_SWEEP = [
    # tile_m=32 variants
    ("tm32 mw1n4 nb3",    64,  32,  1, 4,  3,  1, True, False, None),
    ("tm32 mw2n4 nb3",    64,  32,  2, 4,  3,  1, True, False, None),
    # tile_m=64 variants (current default: mw2n4 nb3)
    ("tm64 mw1n4 nb2",    64,  64,  1, 4,  2,  1, True, False, None),
    ("tm64 mw1n4 nb3",    64,  64,  1, 4,  3,  1, True, False, None),
    ("tm64 mw2n4 nb2",    64,  64,  2, 4,  2,  1, True, False, None),
    ("tm64 mw2n4 nb3",    64,  64,  2, 4,  3,  1, True, False, None),  # ← current
    ("tm64 mw4n4 nb3",    64,  64,  4, 4,  3,  1, True, False, None),
    ("tm64 mw2n8 nb3",    64,  64,  2, 8,  3,  1, True, False, None),
]

# C. cluster_n sweep at M=64 — A-matrix multicast (gy=N/tile_n=8, cluster_n ∈ {1,2,4,8})
CLUSTER_SWEEP = [
    ("cl1 (baseline)",    64,  64,  2, 4,  3,  1, True, False, None),
    ("cl2",               64,  64,  2, 4,  3,  2, True, False, None),
    ("cl4",               64,  64,  2, 4,  3,  4, True, False, None),
    ("cl8",               64,  64,  2, 4,  3,  8, True, False, None),
]

# D. scale mode comparison — same config, different dequant path
SCALE_MODE_SWEEP = [
    ("fp32 scale",        64,  64,  2, 4,  3,  1, False, False, None),
    ("e8m0 scale",        64,  64,  2, 4,  3,  1, True,  False, None),  # ← current
    ("no_scale",          64,  64,  2, 4,  3,  1, False, True,  None),
]

# E. Prefill tile/warp sweep at M=256 (nb=2 because tile_m=128 needs it)
# M=256 is safe for tile_m=128 (gx=2, tested). Do NOT increase M here until
# the tile_m=128 large-grid bug (M>256 → crash) is fixed.
PREFILL_SWEEP = [
    ("pre tm64 mw2n4",   256,  64,  2, 4,  3,  1, True, False, None),
    ("pre tm128 mw2n4",  256, 128,  2, 4,  2,  1, True, False, None),  # ← current
    ("pre tm128 mw4n4",  256, 128,  4, 4,  2,  1, True, False, None),
    ("pre tm128 mw2n8",  256, 128,  2, 8,  2,  1, True, False, None),
]

SWEEP_GROUPS = {
    "m_sweep":    M_SWEEP,
    "tile_warp":  TILE_WARP_SWEEP,
    "cluster":    CLUSTER_SWEEP,
    "scale_mode": SCALE_MODE_SWEEP,
    "prefill":    PREFILL_SWEEP,
}

# ---------------------------------------------------------------------------
# Data preparation (shared across configs)
# ---------------------------------------------------------------------------

def _align_up(v, a):
    return ((v + a - 1) // a) * a


def _make_inputs(M_pad, use_e8m0_scale, no_scale):
    torch.manual_seed(0)
    a = torch.randn((B, M_pad, K), dtype=torch.bfloat16).cuda().contiguous()
    b_raw = torch.randn((B, K, N), dtype=torch.float32).clamp(-1, 1)
    b_fp8 = b_raw.to(torch.float8_e4m3fn).cuda().contiguous()

    if no_scale:
        scale = torch.zeros(1, dtype=torch.uint8).cuda()
    elif use_e8m0_scale:
        scale_fp32 = torch.rand((B, K // GROUP_K, N // GROUP_N)) * 0.1 + 0.01
        log2_s = torch.log2(scale_fp32.clamp(min=1e-38))
        e8m0 = (log2_s.round().to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
        scale = e8m0.cuda().contiguous()
    else:
        scale = (torch.rand((B, K // GROUP_K, N // GROUP_N)) * 0.1 + 0.01).cuda().contiguous()

    c = torch.zeros((B, M_pad, N), dtype=torch.bfloat16).cuda()
    return a.view(-1), b_fp8.view(-1), scale.view(-1), c.view(-1)


# ---------------------------------------------------------------------------
# Benchmark one config
# ---------------------------------------------------------------------------

def bench_config(label, M, tile_m, m_warp, n_warp, num_buffers, cluster_n,
                 use_e8m0_scale, no_scale, waves_per_eu,
                 warmup, iters):
    M_pad = _align_up(M, tile_m)

    try:
        fn = compile_bmm_a16w8_gfx1250(
            B=B, M=M_pad, N=N, K=K,
            group_k=GROUP_K, group_n=GROUP_N,
            tile_m=tile_m, tile_n=TILE_N, tile_k=TILE_K,
            m_warp=m_warp, n_warp=n_warp,
            num_buffers=num_buffers,
            cluster_n=cluster_n, cluster_m=1,
            use_e8m0_scale=use_e8m0_scale,
            no_scale=no_scale,
            waves_per_eu=waves_per_eu,
        )
    except Exception as e:
        return None, f"compile error: {e}"

    a, b, scale, c = _make_inputs(M_pad, use_e8m0_scale, no_scale)
    stream = torch.cuda.current_stream()

    # Warmup — also warms L2 (B matrix = 64MB, fits in 192MB GL2)
    try:
        for _ in range(warmup):
            fn(c, a, b, scale, M_pad, stream)
        torch.cuda.synchronize()
    except Exception as e:
        return None, f"launch error: {e}"

    # Measure with CUDA events (avoid host-sync overhead per iter)
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_start.record(stream)
    for _ in range(iters):
        fn(c, a, b, scale, M_pad, stream)
    t_end.record(stream)
    torch.cuda.synchronize()

    us = t_start.elapsed_time(t_end) * 1e3 / iters  # ms → µs

    flops = 2 * B * M * N * K
    tflops = flops / (us * 1e-6) / 1e12

    # Memory bytes: A bf16 + B fp8 + scale + C bf16
    scale_elem_bytes = 1 if use_e8m0_scale else (0 if no_scale else 4)
    mem = (B * M * K * 2
           + B * K * N * 1
           + B * (K // GROUP_K) * (N // GROUP_N) * scale_elem_bytes
           + B * M * N * 2)
    bw = mem / (us * 1e-6) / 1e12

    return (us, tflops, bw), None


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------

HDR = (
    f"{'label':<24} {'M':>5} {'tm':>4} {'mw×nw':>6} {'cl':>3} "
    f"{'nb':>3} {'scale':>6}  {'µs':>8} {'TFLOPS':>7} {'BW(TB/s)':>9}  note"
)
SEP = "-" * len(HDR)


def _scale_str(use_e8m0, no_sc):
    if no_sc:
        return "noscl"
    if use_e8m0:
        return "e8m0"
    return "fp32"


def print_row(label, M, tile_m, m_warp, n_warp, nb, cl, use_e8m0, no_sc,
              result, err):
    sc = _scale_str(use_e8m0, no_sc)
    prefix = (f"{label:<24} {M:>5} {tile_m:>4} {m_warp}×{n_warp:>1}   {cl:>3} "
              f"{nb:>3} {sc:>6}")
    if err:
        print(f"{prefix}  {'SKIP':>8}  {'':>7}  {'':>9}  {err}")
    else:
        us, tflops, bw = result
        print(f"{prefix}  {us:>8.2f}  {tflops:>7.2f}  {bw:>9.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="bmm_a16w8 latency sweep")
    parser.add_argument("--group", default="all",
                        choices=["all"] + list(SWEEP_GROUPS.keys()),
                        help="Which sweep group to run")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations (warm L2, default=20)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Measurement iterations (default=100)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV file")
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    if not arch.startswith("gfx1250"):
        print(f"ERROR: requires gfx1250, got {arch}")
        sys.exit(1)

    if args.group == "all":
        groups = list(SWEEP_GROUPS.items())
    else:
        groups = [(args.group, SWEEP_GROUPS[args.group])]

    csv_rows = []
    csv_file = open(args.csv, "w", newline="") if args.csv else None
    writer = None
    if csv_file:
        fieldnames = ["group", "label", "M", "tile_m", "m_warp", "n_warp",
                      "num_buffers", "cluster_n", "scale_mode",
                      "us", "TFLOPS", "BW_TBps", "error"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for group_name, configs in groups:
        print(f"\n{'='*60}")
        print(f"  Group: {group_name}  (warmup={args.warmup}, iters={args.iters})")
        print(f"  Shape: B={B} K={K} N={N} tile_n={TILE_N} tile_k={TILE_K} group_k/n=128")
        print(f"{'='*60}")
        print(HDR)
        print(SEP)

        for cfg in configs:
            label, M, tile_m, m_warp, n_warp, nb, cl, e8m0, nosc, wpe = cfg
            result, err = bench_config(
                label, M, tile_m, m_warp, n_warp, nb, cl, e8m0, nosc, wpe,
                warmup=args.warmup, iters=args.iters)
            print_row(label, M, tile_m, m_warp, n_warp, nb, cl, e8m0, nosc,
                      result, err)
            if writer:
                row = {
                    "group": group_name,
                    "label": label,
                    "M": M, "tile_m": tile_m,
                    "m_warp": m_warp, "n_warp": n_warp,
                    "num_buffers": nb, "cluster_n": cl,
                    "scale_mode": _scale_str(e8m0, nosc),
                }
                if err:
                    row.update({"us": "", "TFLOPS": "", "BW_TBps": "", "error": err})
                else:
                    us, tflops, bw = result
                    row.update({"us": f"{us:.3f}", "TFLOPS": f"{tflops:.3f}",
                                "BW_TBps": f"{bw:.4f}", "error": ""})
                writer.writerow(row)

    if csv_file:
        csv_file.close()
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
