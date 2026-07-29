# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 06 - Row-sum reduction (out[m] = sum_n A[m, n]).

GOAL: one block per row. Each thread strides across the row loading float4s and
accumulates a lane-local partial sum, then the block cooperatively reduces to a
single scalar per row: first a warp shuffle_xor tree, then an LDS exchange across
the 2 warps of the 128-thread block.

This is your first reduction. The machinery mirrors kernels/norm/softmax_kernel.py.

CONCEPTS: warp shuffle_xor reduction, LDS block reduce, SharedAllocator/@fx.struct.
READ: FlyDSL for Dummies - Chapter 5 (LDS); reference softmax_kernel.py.
HINT: build(N) is a factory returning the launch wrapper (N is compile-time so the
tile loop can be unrolled with range_constexpr).
"""

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp

BLOCK_THREADS = 128
WARP_SIZE = 64
VEC = 4  # f32 float4
TILE_COLS = BLOCK_THREADS * VEC  # 512
RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)  # 2


@fx.struct
class SharedStorage:
    s_red: fx.Array[fx.Float32, RED_SLOTS]


def build(N: int):
    assert N % TILE_COLS == 0, "puzzle06 fast path needs N % 512 == 0"
    num_tiles = N // TILE_COLS

    @flyc.kernel
    def row_sum_kernel(A: fx.Tensor, C: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        fm = arith.FastMathFlags.fast

        # ==== YOUR CODE HERE ====
        # 1. Allocate LDS: lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        #    s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1)); c_zero = fx.Float32(0.0)
        # 2. Nested wave_reduce_sum(x): shuffle_xor tree over log2(WARP_SIZE) steps,
        #    off = WARP_SIZE // (2 << e); w = w.addf(w.shuffle_xor(off, WARP_SIZE), fastmath=fm).
        # 3. Nested block_reduce_sum(val): wave_reduce_sum per warp; lane 0 writes to s_red[wave];
        #    gpu.barrier(); warp 0 reduces the RED_SLOTS partials; store to s_red[0]; barrier; load s_red[0].
        # 4. Load: for t in range_constexpr(num_tiles): make_rmem_tensor(VEC), copy_atom_call(BufferCopy128b),
        #    memref_load_vec, thread_sum += x.reduce(ReductionOp.ADD, fastmath=fm).
        # 5. total = block_reduce_sum(thread_sum); if tid == 0: store scalar to C[bid] via BufferCopy32b.
        raise NotImplementedError("Fill in row_sum_kernel")

    @flyc.jit
    def launch(A: fx.Tensor, C: fx.Tensor, m: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        row_sum_kernel(A, C).launch(grid=(m, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def run(a: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    c = torch.zeros(m, 1, dtype=torch.float32, device=a.device)
    build(n)(a, c, m, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c.squeeze(1)


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(128, 512, dtype=torch.float32).cuda()
    common.check(run(a), common.ref_row_sum(a), name="puzzle06 row_sum")
