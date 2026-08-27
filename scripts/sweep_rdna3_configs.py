#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep the full kernel config space on one shape.

Goes wider than the tile ladder: register blocking, wave grid, LDS layout,
tile-swizzle group and stagger are all varied, so the ladder's fixed choices
can be checked against what the kernel can actually reach.
"""

import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.gemm.rdna3_int8_gemm import WMMA_K, WMMA_M, WMMA_N, create_wmma_int8_gemm_module  # noqa: E402
from scripts.bench_rdna3_int8_gemm import _bench_us, _parse_shape  # noqa: E402


def _candidates(args):
    tiles = itertools.product(args.reg_m, args.reg_n, args.reg_k, args.waves_m, args.waves_n)
    for reg_m, reg_n, reg_k, waves_m, waves_n in tiles:
        for group_m, layout, stagger in itertools.product(args.group_m, args.lds_layout, args.stagger):
            yield {
                "reg_m": reg_m,
                "reg_n": reg_n,
                "reg_k": reg_k,
                "waves_m": waves_m,
                "waves_n": waves_n,
                "group_m": group_m,
                "lds_layout": layout,
                "stagger": stagger,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="4096")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--reg-m", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--reg-n", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--reg-k", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--waves-m", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--waves-n", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--group-m", type=int, nargs="+", default=[8])
    parser.add_argument("--lds-layout", nargs="+", default=["pad"])
    parser.add_argument("--stagger", type=int, nargs="+", default=[1])
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    m, n, k = _parse_shape(args.shape)
    torch.manual_seed(2026)
    A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    B_T = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    C = torch.zeros((m, n), dtype=torch.int32, device="cuda")
    reference = torch._int_mm(A, B_T.T.contiguous())

    rows = []
    skipped = 0
    for config in _candidates(args):
        block_m = WMMA_M * config["reg_m"] * config["waves_m"]
        block_n = WMMA_N * config["reg_n"] * config["waves_n"]
        block_k = WMMA_K * config["reg_k"]
        if block_m > m or block_n > n or k % block_k or k // block_k < 2:
            skipped += 1
            continue
        try:
            launch, _, _, _ = create_wmma_int8_gemm_module(m, n, k, in_dtype="int8", out_dtype="i32", **config)
        except (AssertionError, ValueError, RuntimeError):
            skipped += 1
            continue

        def run():
            launch(C, A, B_T, torch.cuda.current_stream(), None, None)

        C.zero_()
        try:
            run()
            torch.cuda.synchronize()
        except RuntimeError:
            skipped += 1
            continue
        if not torch.equal(C, reference):
            skipped += 1
            continue
        us = _bench_us(run, warmup=args.warmup, iters=args.iters, use_cudagraph=True)
        rows.append((us, f"{block_m}x{block_n}x{block_k}", config))

    rows.sort()
    ops = 2 * m * n * k
    print(f"{m}x{n}x{k}: {len(rows)} configs measured, {skipped} skipped")
    for us, block, config in rows[: args.top]:
        threads = config["waves_m"] * config["waves_n"] * 32
        print(
            f"  {us:9.2f} us  {ops / (us * 1e-6) / 1e12:6.2f} TOP/s  {block:<12s} threads={threads:<4d} "
            f"group_m={config['group_m']:<3d} lds={config['lds_layout']:<7s} stagger={config['stagger']} "
            f"regs=({config['reg_m']},{config['reg_n']},{config['reg_k']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
