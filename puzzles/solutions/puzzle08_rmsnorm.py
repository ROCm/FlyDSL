# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 08 (solution) - RMSNorm (out[m, :] = A[m, :] * rsqrt(mean(A^2) + eps) * gamma).

One block per row, two register-buffered passes:
  1. load the row, accumulate sum of squares -> block reduce -> mean -> rsqrt;
  2. multiply each element by the shared inv_rms and by the per-column gamma, store.

Only ONE reduction (sum of squares). gamma is a length-N vector broadcast across
rows, loaded with the same float4 buffer copy. Builds directly on puzzle 06/07.

Fast path only: N must be a multiple of tile_cols = BLOCK_THREADS * 4.
Concepts: sum-of-squares reduction, rsqrt, broadcast weight.
Book: Chapter 5 (LDS).
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
EPS = 1e-6


@fx.struct
class SharedStorage:
    s_red: fx.Array[fx.Float32, RED_SLOTS]


def build(N: int):
    assert N % TILE_COLS == 0, "puzzle08 fast path needs N % 512 == 0"
    num_tiles = N // TILE_COLS
    inv_n = 1.0 / float(N)

    @flyc.kernel
    def rmsnorm_kernel(A: fx.Tensor, G: fx.Tensor, C: fx.Tensor):
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
        G_buf = fx.rocdl.make_buffer_tensor(G)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        row_a = fx.slice(A_buf, (bid, None))
        row_c = fx.slice(C_buf, (bid, None))
        a_div = fx.logical_divide(row_a, fx.make_layout(VEC, 1))
        c_div = fx.logical_divide(row_c, fx.make_layout(VEC, 1))
        g_div = fx.logical_divide(G_buf, fx.make_layout(VEC, 1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)

        # Pass 1: load row, accumulate sum of squares.
        row_buffer = []
        thread_sq = c_zero
        for t in range_constexpr(num_tiles):
            idx = tid + t * BLOCK_THREADS
            r = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx)), r)
            x = fx.memref_load_vec(r)
            row_buffer.append(x)
            sq = x * x
            thread_sq = thread_sq + sq.reduce(ReductionOp.ADD, fastmath=fm)
        mean_sq = block_reduce_sum(thread_sq) * inv_n
        inv_rms = fmath.rsqrt(mean_sq + EPS, fastmath=fm)

        # Pass 2: scale by inv_rms and gamma, store.
        for t in range_constexpr(num_tiles):
            idx = tid + t * BLOCK_THREADS
            g = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.copy_atom_call(copy_atom, fx.slice(g_div, (None, idx)), g)
            gamma = fx.memref_load_vec(g)
            out_vec = row_buffer[t] * inv_rms * gamma
            r = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.memref_store_vec(out_vec, r)
            fx.copy_atom_call(copy_atom, r, fx.slice(c_div, (None, idx)))

    @flyc.jit
    def launch(A: fx.Tensor, G: fx.Tensor, C: fx.Tensor, m: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        rmsnorm_kernel(A, G, C).launch(grid=(m, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def run(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    c = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    build(n)(a, gamma, c, m, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(128, 512, dtype=torch.float32).cuda()
    gamma = torch.randn(512, dtype=torch.float32).cuda()
    common.check(run(a, gamma), common.ref_rmsnorm(a, gamma), name="puzzle08 rmsnorm")
