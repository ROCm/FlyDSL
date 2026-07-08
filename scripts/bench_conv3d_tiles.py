#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Microbenchmark: compare conv3d tile configs and show the autotuner pick.

For a set of representative 3x3x3 conv shapes, times every legal tile candidate
with ``do_bench`` and prints a per-shape table (tile -> ms -> TFLOP/s), the best
tile, and what ``autotune_conv3d`` selects. Read-only; no correctness asserts.

Usage (inside a FlyDSL GPU env):
    python3 scripts/bench_conv3d_tiles.py            # BF16
    python3 scripts/bench_conv3d_tiles.py --fp8      # FP8 (CDNA4)
"""

import argparse

import torch

from flydsl.autotune import do_bench
from kernels.conv.conv3d_autotune import BF16_CANDIDATES, FP8_CANDIDATES, autotune_conv3d
from kernels.conv.conv3d_implicit_8wave import conv3d_implicit_8wave
from kernels.conv.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8

# (N, C, T, H, W, K), 3x3x3 stride=1 pad=1. From the PR #794 perf table (C=K=128).
SHAPES = [
    (1, 128, 6, 40, 40, 128),
    (1, 128, 6, 56, 56, 128),
    (1, 128, 6, 72, 72, 128),
    (1, 128, 6, 104, 104, 128),
    (1, 128, 6, 144, 144, 128),
]


def _tflops(n, c, t, h, w, k, ms):
    do, ho, wo = t, h, w  # stride=1, pad=1, 3x3x3 -> same spatial dims
    macs = n * do * ho * wo * k * c * 27
    return (2 * macs) / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", action="store_true", help="benchmark the FP8 kernel")
    args = ap.parse_args()

    if args.fp8:
        conv, cands, kind = conv3d_implicit_8wave_fp8, FP8_CANDIDATES, "fp8"
    else:
        conv, cands, kind = conv3d_implicit_8wave, BF16_CANDIDATES, "bf16"

    print(f"conv3d tile benchmark ({kind}), 3x3x3 stride=1 pad=1\n")
    for shp in SHAPES:
        n, c, t, h, w, k = shp
        x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)

        print(f"shape N={n} C={c} T={t} H={h} W={w} K={k}  (NPQ={n*t*h*w})")
        results = []
        for tile in cands:
            try:
                ms = do_bench(lambda: conv(x, weight, stride=1, padding=1, tile=tile), warmup=5, rep=20)
                results.append((tile, ms))
                print(f"    tile={tile}  {ms:.4f} ms  {_tflops(*shp, ms):.1f} TF")
            except Exception as e:
                print(f"    tile={tile}  FAILED: {type(e).__name__}")
        if results:
            best = min(results, key=lambda r: r[1])
            print(f"  best tile: {best[0]}  ({best[1]:.4f} ms, {_tflops(*shp, best[1]):.1f} TF)")

        shape_key = (n, c, t, h, w, k, 3, 3, 3, 1, 1, 1, 1, 1, 1, False)
        picked = autotune_conv3d(
            kind, shape_key, kind, cands, x.device, lambda tl: conv(x, weight, stride=1, padding=1, tile=tl)
        )
        print(f"  autotuner picked: {picked}\n")


if __name__ == "__main__":
    main()
