#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep the tile ladder over a set of shapes to check the pick_tile heuristic.

Reports the measured latency of every feasible tile, so the shape-selected
default can be compared against the best tile actually available.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.gemm.rdna3_int8_gemm_autotune import (  # noqa: E402
    _build,
    _tile_geometry,
    feasible_tiles,
    pick_split_k,
    pick_tile,
)
from scripts.bench_rdna3_int8_gemm import _bench_us, _parse_shape  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="512,1024,2048,4096,64x4096x4096,128x4096x4096,256x4096x4096")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--split-k", action="store_true", help="give every tile its heuristic split_k")
    args = parser.parse_args()

    for m, n, k in (_parse_shape(item.strip()) for item in args.shapes.split(",") if item.strip()):
        torch.manual_seed(2026)
        A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        B_T = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        C = torch.zeros((m, n), dtype=torch.int32, device="cuda")
        reference = torch._int_mm(A, B_T.T.contiguous())
        default = pick_tile(m, n, k)

        print(f"\n=== {m}x{n}x{k} (heuristic picks {'x'.join(map(str, _tile_geometry(default)[:3]))}) ===")
        rows = []
        for tile, workgroups in feasible_tiles(m, n, k):
            split_k = pick_split_k(m, n, k, tile) if args.split_k else 1
            launch = _build(m, n, k, "int8", "i32", "none", *tile, False, split_k, k, k, n)

            def run():
                if split_k > 1:
                    C.zero_()
                launch(C, A, B_T, torch.cuda.current_stream(), None, None)

            C.zero_()
            run()
            torch.cuda.synchronize()
            if not torch.equal(C, reference):
                print(f"  {tile} produced a wrong result; skipping")
                continue
            block = "x".join(map(str, _tile_geometry(tile)[:3]))
            _bm, _bn, _bk, threads = _tile_geometry(tile)
            us = _bench_us(run, warmup=args.warmup, iters=args.iters, use_cudagraph=True)
            rows.append((us, block, threads, workgroups, split_k, tile))

        best = min(us for us, *_ in rows)
        for us, block, threads, workgroups, split_k, tile in sorted(rows):
            marks = "  <- heuristic" if tile == default else ""
            marks += "  <- best" if us == best else ""
            print(
                f"  {us:8.2f} us  {us / best:5.2f}x  {block:<12s} threads={threads:<4d} "
                f"wgs={workgroups:<5d} split_k={split_k}{marks}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
