# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 13 - GEMM with a swizzled (padded) LDS layout (C = A @ B^T).

GOAL: the final refinement over puzzle 12. When many lanes read the same LDS
column at once, their addresses can collide on the same memory bank (a "bank
conflict"), serializing the reads. The classic fix is to PAD each LDS row so
consecutive rows start in different banks: the logical tile stays
(BLOCK_M, BLOCK_K), but the physical row stride becomes BLOCK_K_PAD = BLOCK_K + PAD.
Because writes and reads use the same padded view, the result is unchanged - only
the bank mapping shifts.

CONCEPTS: LDS bank conflicts, padded/swizzled layouts, strided LDS views.
READ: FlyDSL for Dummies - Chapter 5; kernels/gemm/preshuffle_gemm.py (XOR16 swizzle).
HINT: this is puzzle 12 with ONE change - the LDS view stride. Use
fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K_PAD, 1)) and size the LDS arrays
BLOCK_M * BLOCK_K_PAD. Everything else (double-buffer pipeline) is identical.
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

        # ==== YOUR CODE HERE ====
        # This is puzzle 12 with ONE change: the LDS views use a PADDED row stride.
        #   LA = fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K_PAD, 1))
        #   LB = fx.make_layout((BLOCK_N, BLOCK_K), (BLOCK_K_PAD, 1))
        #   sA = [lds.sA0.view(LA), lds.sA1.view(LA)]; sB = [lds.sB0.view(LB), lds.sB1.view(LB)]
        # The rest (buffer tensors, tiled_mma, prefetch_to_lds helper, prologue, the
        # ping-pong K-loop, and the frag_C store) is identical to puzzle 12.
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
    common.check(run(a, b), common.ref_gemm(a, b), name="puzzle13 gemm_swizzled", atol=5e-2, rtol=5e-2)
