# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 12 - GEMM with double-buffered LDS (C = A @ B^T).

GOAL: overlap loading and compute. Puzzle 11 loaded a K-tile, waited on a barrier,
then computed. Here two LDS buffers ping-pong: while the MFMA consumes buffer
`cur`, the next K-tile is prefetched from global into buffer `1-cur`. The K-loop is
compile-time unrolled, so the buffer index is a Python constant each iteration -
no scf.for loop-carried state needed.

Structure: prologue loads tile 0 into buf 0; each iteration prefetches tile ki+1
into the other buffer, then computes tile ki from the current buffer.

CONCEPTS: software pipelining, ping-pong LDS, prefetch/compute overlap.
READ: FlyDSL for Dummies - Chapter 5 + Chapter 6; kernels/gemm/preshuffle_gemm.py.
HINT: keep two views per operand, sA = [lds.sA0.view(LA), lds.sA1.view(LA)]. A
nested prefetch_to_lds(ki, buf) helper (read-only capture) keeps the loop tidy.
Guard the prefetch with `if const_expr(ki + 1 < num_k):`.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8


@fx.struct
class SharedStorage:
    sA0: fx.Array[fx.Float32, BLOCK_M * BLOCK_K]
    sA1: fx.Array[fx.Float32, BLOCK_M * BLOCK_K]
    sB0: fx.Array[fx.Float32, BLOCK_N * BLOCK_K]
    sB1: fx.Array[fx.Float32, BLOCK_N * BLOCK_K]


def build(M: int, N: int, K: int):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
    num_k = K // BLOCK_K

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bm = fx.block_idx.y
        bn = fx.block_idx.x

        # ==== YOUR CODE HERE ====
        # 1. lds + two views per operand: sA = [lds.sA0.view(LA), lds.sA1.view(LA)]; sB likewise.
        # 2. Buffer tensors / zA / zB / bC / mma_atom / tiled_mma / thr_mma / thr_copy_A/B/C
        #    and lds_atom, exactly as puzzle 11.
        # 3. Nested helper prefetch_to_lds(ki, buf): load global K-tile ki into registers, then
        #    store registers into sA[buf] / sB[buf] (global->reg->LDS). Read-only capture only.
        # 4. frag_C = make_fragment_C(bC); fill(0). Prologue: prefetch_to_lds(0, 0); gpu.barrier().
        # 5. for ki in range_constexpr(num_k):
        #        cur = ki % 2
        #        if const_expr(ki + 1 < num_k): prefetch_to_lds(ki + 1, 1 - cur)
        #        frag_A/B = make_fragment_A/B(sA[cur]/sB[cur]); copy LDS->frag; fx.gemm; gpu.barrier().
        # 6. Store frag_C to bC.
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
    common.check(run(a, b), common.ref_gemm(a, b), name="puzzle12 gemm_double_buffer", atol=5e-2, rtol=5e-2)
