# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 09 - Single-tile MFMA GEMM (C = A @ B^T, one block).

GOAL: compute one (BLOCK_M, BLOCK_N) output tile with a single K tile - no K-loop
yet. Load A and B into MFMA fragments, run one fx.gemm, store C. This is the
"hello world" of the tiled_mma abstraction (identical to examples/03-tiledMma.py).

A is (M, K), B is (N, K) (row-major), C = A @ B^T is (M, N).

CONCEPTS: MMA atom, tiled_mma wave layout, make_fragment_A/B/C, fx.gemm.
READ: FlyDSL for Dummies - Chapter 6 (MMA); reference examples/03-tiledMma.py.
HINT: mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32));
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)));
use make_tiled_copy_A/B/C(copy_atom, tiled_mma) to build per-operand copies.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # ==== YOUR CODE HERE ====
    # 1. Buffer tensors; bA/bB/bC = slice(zipped_divide(X, (BLOCK_*, BLOCK_*)), (None, bid)).
    # 2. mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32));
    #    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)));
    #    thr_mma = tiled_mma.thr_slice(tid).
    # 3. copy_atom = BufferCopy32b; tiled_copy_A/B/C = fx.make_tiled_copy_A/B/C(copy_atom, tiled_mma);
    #    thr_copy_* = .get_slice(tid); partition_S(bA/bB/bC).
    # 4. frag_A/B/C = thr_mma.make_fragment_A/B/C(bA/bB/bC); copy_frag_* = thr_copy_*.retile(frag_*).
    # 5. fx.copy A and B (pred=None); frag_C.fill(0); fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C);
    #    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None).
    raise NotImplementedError("Fill in gemm_kernel")


@flyc.jit
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n, _ = b.shape
    c = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    gemm(a, b, c, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(64, 8, dtype=torch.float32).cuda()
    b = torch.randn(64, 8, dtype=torch.float32).cuda()
    common.check(run(a, b), common.ref_gemm(a, b), name="puzzle09 gemm_single_tile", atol=1e-2, rtol=1e-2)
