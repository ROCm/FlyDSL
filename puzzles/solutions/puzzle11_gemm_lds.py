# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 11 (solution) - GEMM with LDS-staged K-tiles (C = A @ B^T).

Same math as puzzle 10, but each K-tile of A and B is routed through LDS before
being fed to the MFMA. The round-trip is: global -> registers -> LDS -> registers
-> MFMA. On its own this extra hop does not speed anything up; it introduces the
machinery (an LDS-backed tensor view + gpu.barrier synchronization) that the
double-buffer (puzzle 12) and swizzle (puzzle 13) puzzles build on to actually
overlap loads with compute and remove bank conflicts.

Key insight: because the LDS view sA has the same (BLOCK_M, BLOCK_K) logical shape
as the global tile, the SAME thr_copy_A.partition_S works on both - only the
memory space differs.

Concepts: SharedAllocator LDS tensor, partition on LDS, gpu.barrier.
Book: Chapter 5 (LDS) + Chapter 6 (MMA).
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

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        sA = lds.sA.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
        sB = lds.sB.view(fx.make_layout((BLOCK_N, BLOCK_K), (BLOCK_K, 1)))

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        zA = fx.zipped_divide(A, (BLOCK_M, BLOCK_K))
        zB = fx.zipped_divide(B, (BLOCK_N, BLOCK_K))
        bC = fx.slice(fx.zipped_divide(C, (BLOCK_M, BLOCK_N)), (None, (bm, bn)))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        lds_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma).get_slice(tid)

        frag_C = thr_mma.make_fragment_C(bC)
        frag_C.fill(0)

        for ki in range_constexpr(num_k):
            bA = fx.slice(zA, (None, (bm, ki)))
            bB = fx.slice(zB, (None, (bn, ki)))
            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            copy_frag_A = thr_copy_A.retile(frag_A)
            copy_frag_B = thr_copy_B.retile(frag_B)

            # global -> registers -> LDS
            fx.copy(copy_atom, thr_copy_A.partition_S(bA), copy_frag_A, pred=None)
            fx.copy(copy_atom, thr_copy_B.partition_S(bB), copy_frag_B, pred=None)
            fx.copy(lds_atom, copy_frag_A, thr_copy_A.partition_S(sA), pred=None)
            fx.copy(lds_atom, copy_frag_B, thr_copy_B.partition_S(sB), pred=None)
            gpu.barrier()

            # LDS -> registers (MFMA fragment)
            fx.copy(lds_atom, thr_copy_A.partition_S(sA), copy_frag_A, pred=None)
            fx.copy(lds_atom, thr_copy_B.partition_S(sB), copy_frag_B, pred=None)
            gpu.barrier()

            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        copy_frag_C = thr_copy_C.retile(frag_C)
        fx.copy(copy_atom, copy_frag_C, thr_copy_C.partition_S(bC), pred=None)

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
