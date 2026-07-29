# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 10 - Multi-block GEMM with a K-loop (C = A @ B^T).

GOAL: extend puzzle 09 to full size (256x256x256). Use a 2D grid of output tiles
and loop over K. Each block owns one (BLOCK_M, BLOCK_N) output tile; frag_C lives
in registers and accumulates across all K tiles. Every K iteration loads a fresh
(BLOCK_M, BLOCK_K) slice of A and (BLOCK_N, BLOCK_K) slice of B from global into
fragments (no LDS yet) and issues one fx.gemm.

CONCEPTS: 2D grid tiling, nested-coord K-slice, register accumulation.
READ: FlyDSL for Dummies - Chapter 6 (MMA).
HINT: the K index is the SECOND mode of the zipped_divide, so slice with a nested
coord: fx.slice(zA, (None, (bm, ki))). Build frag_C ONCE, fill(0), accumulate in
the K-loop, then store after the loop. Grid = (N//BLOCK_N, M//BLOCK_M, 1);
bm = fx.block_idx.y, bn = fx.block_idx.x. K is compile-time -> range_constexpr.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8


def build(M: int, N: int, K: int):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
    num_k = K // BLOCK_K

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.y
        bn = fx.block_idx.x

        # ==== YOUR CODE HERE ====
        # 1. Buffer tensors; zA = zipped_divide(A, (BLOCK_M, BLOCK_K)); zB likewise for B;
        #    bC = slice(zipped_divide(C, (BLOCK_M, BLOCK_N)), (None, (bm, bn))).
        # 2. mma_atom / tiled_mma / thr_mma and tiled_copy_A/B/C + thr_copy_* as in puzzle 09.
        # 3. frag_C = thr_mma.make_fragment_C(bC); frag_C.fill(0).
        # 4. for ki in range_constexpr(num_k): bA = slice(zA, (None, (bm, ki)));
        #    bB = slice(zB, (None, (bn, ki))); make_fragment_A/B; retile; fx.copy A and B;
        #    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C).
        # 5. After the loop: retile frag_C and fx.copy to thr_copy_C.partition_S(bC).
        raise NotImplementedError("Fill in gemm_kernel")

    @flyc.jit
    def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        grid = (N // BLOCK_N, M // BLOCK_M, 1)
        gemm_kernel(A, B, C).launch(grid=grid, block=(256, 1, 1), stream=stream)

    return gemm


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n, _ = b.shape
    c = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    build(m, n, k)(a, b, c, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(256, 256, dtype=torch.float32).cuda()
    b = torch.randn(256, 256, dtype=torch.float32).cuda()
    common.check(run(a, b), common.ref_gemm(a, b), name="puzzle10 gemm_kloop", atol=5e-2, rtol=5e-2)
