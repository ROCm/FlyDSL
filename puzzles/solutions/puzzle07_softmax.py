# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 07 (solution) - Row-wise softmax (out[m, :] = softmax(A[m, :])).

One block per row, three register-buffered passes:
  1. load the whole row into registers, track the running max;
  2. exp2((x - max) * log2e), track the running sum;
  3. multiply by 1/sum and store.

Each pass ends in a block reduce (max, then sum) built from a warp shuffle_xor
tree plus an LDS exchange across the 2 warps. This is puzzle 06's reduction reused
twice, with an exp in the middle. Mirrors kernels/norm/softmax_kernel.py's fast path.

Fast path only: N must be a multiple of tile_cols = BLOCK_THREADS * 4.
Concepts: multi-pass reduction, fast exp2, register buffering.
Book: Chapter 5 (LDS), reference softmax_kernel.py.
"""

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp

BLOCK_THREADS = 128
WARP_SIZE = 64
VEC = 4  # f32 float4
TILE_COLS = BLOCK_THREADS * VEC  # 512
RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)  # 2
LOG2E = 1.4426950408889634


@fx.struct
class SharedStorage:
    s_red: fx.Array[fx.Float32, RED_SLOTS]


def build(N: int):
    assert N % TILE_COLS == 0, "puzzle07 fast path needs N % 512 == 0"
    num_tiles = N // TILE_COLS

    @flyc.kernel
    def softmax_kernel(A: fx.Tensor, C: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        fm = arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        c_zero = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))

        def wave_reduce(x, mode):
            w = x
            for e in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << e)
                peer = w.shuffle_xor(off, WARP_SIZE)
                if const_expr(mode == "max"):
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm)
            return w

        def block_reduce(val, mode):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce(val, mode)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            neutral = c_neg_inf if mode == "max" else c_zero
            w = wave_reduce(val, mode)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = wave_reduce(in_range.select(v, neutral), mode)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        row_a = fx.slice(A_buf, (bid, None))
        row_c = fx.slice(C_buf, (bid, None))
        a_div = fx.logical_divide(row_a, fx.make_layout(VEC, 1))
        c_div = fx.logical_divide(row_c, fx.make_layout(VEC, 1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)

        # Pass 1: load the row into registers, track the running max.
        row_buffer = []
        thread_max = c_neg_inf
        for t in range_constexpr(num_tiles):
            idx = tid + t * BLOCK_THREADS
            r = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx)), r)
            x = fx.memref_load_vec(r)
            row_buffer.append(x)
            thread_max = thread_max.maximumf(x.reduce(ReductionOp.MAX))
        global_max = block_reduce(thread_max, "max")

        # Pass 2: exp2((x - max) * log2e), track the running sum.
        thread_sum = c_zero
        for i in range_constexpr(num_tiles):
            scaled = (row_buffer[i] - global_max) * LOG2E
            exp_val = fmath.exp2(scaled, fastmath=fm)
            row_buffer[i] = exp_val
            thread_sum = thread_sum + exp_val.reduce(ReductionOp.ADD, fastmath=fm)
        global_sum = block_reduce(thread_sum, "sum")

        # Pass 3: normalize and store.
        inv_sum = 1.0 / global_sum
        for t in range_constexpr(num_tiles):
            out_vec = row_buffer[t] * inv_sum
            idx = tid + t * BLOCK_THREADS
            r = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.memref_store_vec(out_vec, r)
            fx.copy_atom_call(copy_atom, r, fx.slice(c_div, (None, idx)))

    @flyc.jit
    def launch(A: fx.Tensor, C: fx.Tensor, m: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        softmax_kernel(A, C).launch(grid=(m, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def run(a: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    c = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    build(n)(a, c, m, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(128, 512, dtype=torch.float32).cuda()
    common.check(run(a), common.ref_softmax(a), name="puzzle07 softmax")
