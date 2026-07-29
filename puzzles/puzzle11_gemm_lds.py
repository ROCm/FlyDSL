# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 11 - GEMM with LDS-staged K-tiles (C = A @ B^T).

GOAL: same math as puzzle 10, but route each K-tile of A and B through LDS before
the MFMA: global -> registers -> LDS -> registers -> MFMA. This extra hop does not
speed anything up on its own; it introduces the machinery (an LDS-backed tensor
view + gpu.barrier) that puzzles 12 (double-buffer) and 13 (swizzle) build on.

Key insight: because the LDS view sA has the same (BLOCK_M, BLOCK_K) logical shape
as the global tile, the SAME thr_copy_A.partition_S works on both - only the
memory space differs.

CONCEPTS: SharedAllocator LDS tensor, partition on LDS, gpu.barrier.
READ: FlyDSL for Dummies - Chapter 5 (LDS) + Chapter 6 (MMA).
HINT: sA = lds.sA.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1))). Use a
UniversalCopy32b atom for the LDS copies. Barrier after writing LDS and again
after reading it back (before the next iteration overwrites it).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8


@fx.struct
class SharedStorage:
    sA: fx.Array[fx.Float32, BLOCK_M * BLOCK_K]
    sB: fx.Array[fx.Float32, BLOCK_N * BLOCK_K]


def build(M: int, N: int, K: int):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
    num_k = K // BLOCK_K

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.y
        bn = fx.block_idx.x

        # ==== YOUR CODE HERE ====
        # 1. lds = SharedAllocator().allocate(SharedStorage).peek();
        #    sA = lds.sA.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1))); sB likewise.
        # 2. Buffer tensors, zA/zB/bC, mma_atom/tiled_mma/thr_mma, thr_copy_A/B/C as in puzzle 10.
        #    Add lds_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32).
        # 3. frag_C = make_fragment_C(bC); fill(0).
        # 4. for ki in range_constexpr(num_k):
        #      bA/bB = slice(zX, (None,(b*,ki))); frag_A/B + copy_frag_A/B (retile).
        #      global->registers: fx.copy(copy_atom, thr_copy_*.partition_S(bX), copy_frag_*).
        #      registers->LDS:    fx.copy(lds_atom, copy_frag_*, thr_copy_*.partition_S(sX)); gpu.barrier().
        #      LDS->registers:    fx.copy(lds_atom, thr_copy_*.partition_S(sX), copy_frag_*); gpu.barrier().
        #      fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C).
        # 5. Store frag_C to bC as in puzzle 10.
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
    common.check(run(a, b), common.ref_gemm(a, b), name="puzzle11 gemm_lds", atol=5e-2, rtol=5e-2)
