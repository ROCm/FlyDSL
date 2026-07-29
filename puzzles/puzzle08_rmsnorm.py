# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 08 - RMSNorm (out[m, :] = A[m, :] * rsqrt(mean(A^2) + eps) * gamma).

GOAL: one block per row, two register-buffered passes:
  1. load the row, accumulate sum of squares -> block reduce -> mean -> rsqrt;
  2. multiply each element by the shared inv_rms and the per-column gamma, store.

Only ONE reduction (sum of squares). gamma is a length-N vector broadcast across
rows, loaded with the same float4 buffer copy. Builds directly on puzzle 06/07.

CONCEPTS: sum-of-squares reduction, rsqrt, broadcast weight.
READ: FlyDSL for Dummies - Chapter 5 (LDS).
HINT: reuse wave_reduce_sum / block_reduce_sum from puzzle 06. rsqrt is
fmath.rsqrt (from flydsl.expr import math as fmath). gamma has its own g_div view.
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

        # ==== YOUR CODE HERE ====
        # 1. LDS + wave_reduce_sum / block_reduce_sum (sum only) as in puzzle 06.
        # 2. Buffer tensors for A, G (gamma), C. a_div/c_div = logical_divide(row, make_layout(VEC,1));
        #    g_div = logical_divide(G_buf, make_layout(VEC,1)) (gamma is 1-D, no row slice); BufferCopy128b.
        # 3. Pass 1: for t in range_constexpr(num_tiles): load float4 into row_buffer[]; sq = x*x;
        #    thread_sq += sq.reduce(ReductionOp.ADD, fastmath=fm).
        #    mean_sq = block_reduce_sum(thread_sq) * inv_n; inv_rms = fmath.rsqrt(mean_sq + EPS, fastmath=fm).
        # 4. Pass 2: load gamma float4 for this idx; out = row_buffer[t] * inv_rms * gamma; store to c_div.
        raise NotImplementedError("Fill in rmsnorm_kernel")

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
