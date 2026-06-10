#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A8W4 GEMM precision check for the current pipeline.

Runs the test harness's correctness path (kernel output vs torch f32 reference)
across a set of shapes/tiles and tallies PASS/FAIL. Covers both:
  - tile_k=512 -> loop_iters==0  (the tail-prefetch fix path)
  - tile_k=256 -> loop_iters>0   (regular pipelined path)

Each config prints Cosine similarity + abs diff + PASS/FAIL (from the harness's
scale-aware assert_close). Use realistic M/N — tiny shapes (M=1,N=256) with bf16
output fail on tolerance for BOTH baseline and current (not a kernel bug).

Run via scripts/run_check_a8w4_precision.sh (sets env + triton path).
"""
from __future__ import annotations

import sys

from tests.kernels import test_gemm_fp8fp4_gfx1250 as T


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
CONFIGS = [
    # loop_iters==0 (tail-prefetch fix): num_k_tiles in [4,6], num_buffers=4
    (128, 2816, 2048, 16, 64, 512, 1, 4, 4),   # 4 k-tiles -> loop_iters 0
    (256, 1024, 2048, 16, 64, 512, 1, 4, 4),
    (128, 4096, 3072, 16, 128, 512, 1, 4, 4),  # 6 k-tiles -> loop_iters 0
    # loop_iters>0 (regular pipeline): tile_k=256
    (128, 2816, 2816, 16, 64, 256, 1, 4, 2),
    (256, 4096, 3072, 16, 128, 256, 1, 4, 4),
]


def main():
    results = []
    for cfg in CONFIGS:
        M, N, K, tm, tn, tk, mw, nw, nb = cfg
        tag = f"M{M} N{N} K{K} t({tm},{tn},{tk}) w{mw}x{nw} b{nb}"
        print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)
        try:
            T._run_mxscale_gemm_test(
                "a8w4", M, N, K, tm, tn, tk, mw, nw, nb,
                use_tdm_store=True, out_dtype="bf16",
                wave_specialized_tdm=True, l2_prefetch_distance=2,
            )
            results.append((tag, "PASS"))
        except AssertionError as e:
            # Extract the mismatch summary line for the table.
            msg = str(e).splitlines()
            short = next((l.strip() for l in msg if "Mismatched" in l), "assert_close failed")
            results.append((tag, f"FAIL ({short})"))
        except Exception as e:  # noqa: BLE001
            results.append((tag, f"ERROR ({type(e).__name__}: {e})"))

    print(f"\n\n{'='*72}\n  A8W4 precision summary (current pipeline)\n{'='*72}")
    npass = sum(1 for _, s in results if s == "PASS")
    for tag, status in results:
        print(f"  {status:<6}  {tag}" if status == "PASS" else f"  {status}\n          {tag}")
    print(f"\n  {npass}/{len(results)} PASS")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
