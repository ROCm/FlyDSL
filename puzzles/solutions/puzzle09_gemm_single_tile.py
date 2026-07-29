# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 09 (solution) - Single-tile MFMA GEMM (C = A @ B^T, one block).

The whole problem is one (BLOCK_M, BLOCK_N) output tile with a single K tile, so
there is no K-loop yet: just load A and B into MFMA fragments, run one fx.gemm,
store C. This is the "hello world" of the tiled_mma abstraction and is identical
in spirit to examples/03-tiledMma.py.

A is (M, K), B is (N, K) (row-major), C = A @ B^T is (M, N).
The MMA atom is MFMA(16, 16, 4, f32); a 2x2 wave tiling (256 threads) covers the
64x64 tile, and fx.gemm iterates the atom over the fragments for us.

Concepts: MMA atom, tiled_mma wave layout, make_fragment_A/B/C, fx.gemm.
Book: Chapter 6 (MMA), reference examples/03-tiledMma.py.
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

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.slice(fx.zipped_divide(A, (BLOCK_M, BLOCK_K)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, (BLOCK_N, BLOCK_K)), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, (BLOCK_M, BLOCK_N)), (None, bid))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)

    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)


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
