# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 06 (solution) - Row-sum reduction (out[m] = sum_n A[m, n]).

One block per row. Each thread loads a float4, sums its lane-local values, then
the block cooperatively reduces: a warp shuffle_xor tree, then an LDS exchange
across the 2 warps. Mirrors the reduction machinery in kernels/norm/softmax_kernel.py.

Fast path only: N must be a multiple of tile_cols = BLOCK_THREADS * 4.
Concepts: warp shuffle_xor reduction, LDS block reduce, SharedAllocator/@fx.struct.
Book: Chapter 5 (LDS), reference softmax_kernel.py.
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

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        c_zero = fx.Float32(0.0)

        def wave_reduce_sum(x):
            w = x
            for e in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << e)
                w = w.addf(w.shuffle_xor(off, WARP_SIZE), fastmath=fm)
            return w

        def block_reduce_sum(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_sum(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_sum(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = wave_reduce_sum(in_range.select(v, c_zero))
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        row_a = fx.slice(A_buf, (bid, None))
        a_div = fx.logical_divide(row_a, fx.make_layout(VEC, 1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)

        thread_sum = c_zero
        for t in range_constexpr(num_tiles):
            idx = tid + t * BLOCK_THREADS
            r = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx)), r)
            x = fx.memref_load_vec(r)
            thread_sum = thread_sum + x.reduce(ReductionOp.ADD, fastmath=fm)

        total = block_reduce_sum(thread_sum)

        # thread 0 writes the row's scalar sum to C[bid]
        row_c = fx.slice(C_buf, (bid, None))
        c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))
        if tid == 0:
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store_vec(fx.full(1, total, fx.Float32), r)
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
            fx.copy_atom_call(store_atom, r, fx.slice(c_div, (None, 0)))

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
