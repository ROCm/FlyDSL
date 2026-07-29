# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 13 (solution) - GEMM with a swizzled (padded) LDS layout (C = A @ B^T).

The final refinement over puzzle 12. When many lanes read the same LDS column at
once, their addresses can collide on the same memory bank (a "bank conflict"),
serializing the reads. The classic fix is to *pad* each LDS row so consecutive
rows start in different banks: the logical tile stays (BLOCK_M, BLOCK_K), but the
physical row stride becomes BLOCK_K_PAD = BLOCK_K + PAD. Because writes and reads
use the same padded view, the result is unchanged - only the bank mapping shifts.

This is the simplest correct form of LDS swizzling (padding); production kernels
often use an XOR swizzle instead, but the principle - decorrelate lane->bank - is
the same.

Concepts: LDS bank conflicts, padded/swizzled layouts, strided LDS views.
Book: Chapter 5 (LDS); reference kernels/gemm/preshuffle_gemm.py (XOR16 swizzle).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8
PAD = 4
BLOCK_K_PAD = BLOCK_K + PAD  # 12: pad each LDS row to shift bank mapping


@fx.struct
class SharedStorage:
    sA0: fx.Array[fx.Float32, BLOCK_M * BLOCK_K_PAD]
    sA1: fx.Array[fx.Float32, BLOCK_M * BLOCK_K_PAD]
    sB0: fx.Array[fx.Float32, BLOCK_N * BLOCK_K_PAD]
    sB1: fx.Array[fx.Float32, BLOCK_N * BLOCK_K_PAD]


def build(M: int, N: int, K: int):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
    num_k = K // BLOCK_K

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.y
        bn = fx.block_idx.x

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # Padded (swizzled) layout: logical (BLOCK, BLOCK_K), physical row stride BLOCK_K_PAD.
        LA = fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K_PAD, 1))
        LB = fx.make_layout((BLOCK_N, BLOCK_K), (BLOCK_K_PAD, 1))
        sA = [lds.sA0.view(LA), lds.sA1.view(LA)]
        sB = [lds.sB0.view(LB), lds.sB1.view(LB)]

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

        def prefetch_to_lds(ki, buf):
            bA = fx.slice(zA, (None, (bm, ki)))
            bB = fx.slice(zB, (None, (bn, ki)))
            rA = fx.make_fragment_like(thr_copy_A.partition_S(bA))
            rB = fx.make_fragment_like(thr_copy_B.partition_S(bB))
            fx.copy(copy_atom, thr_copy_A.partition_S(bA), rA, pred=None)
            fx.copy(copy_atom, thr_copy_B.partition_S(bB), rB, pred=None)
            fx.copy(lds_atom, rA, thr_copy_A.partition_S(sA[buf]), pred=None)
            fx.copy(lds_atom, rB, thr_copy_B.partition_S(sB[buf]), pred=None)

        frag_C = thr_mma.make_fragment_C(bC)
        frag_C.fill(0)

        prefetch_to_lds(0, 0)
        gpu.barrier()

        for ki in range_constexpr(num_k):
            cur = ki % 2
            if const_expr(ki + 1 < num_k):
                prefetch_to_lds(ki + 1, 1 - cur)
            frag_A = thr_mma.make_fragment_A(sA[cur])
            frag_B = thr_mma.make_fragment_B(sB[cur])
            fx.copy(lds_atom, thr_copy_A.partition_S(sA[cur]), thr_copy_A.retile(frag_A), pred=None)
            fx.copy(lds_atom, thr_copy_B.partition_S(sB[cur]), thr_copy_B.retile(frag_B), pred=None)
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
            gpu.barrier()

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
    common.check(run(a, b), common.ref_gemm(a, b), name="puzzle13 gemm_swizzled", atol=5e-2, rtol=5e-2)
