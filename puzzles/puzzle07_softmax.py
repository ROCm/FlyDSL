# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 07 - Row-wise softmax (out[m, :] = softmax(A[m, :])).

GOAL: one block per row, three register-buffered passes:
  1. load the whole row into registers, track the running max;
  2. exp2((x - max) * log2e), track the running sum;
  3. multiply by 1/sum and store.

Each pass ends in a block reduce. This is puzzle 06's reduction reused twice
(once for max, once for sum) with a fast exp2 in the middle.

CONCEPTS: multi-pass reduction, fast exp2, register buffering.
READ: FlyDSL for Dummies - Chapter 5 (LDS); reference softmax_kernel.py.
HINT: reuse the wave_reduce / block_reduce machinery from puzzle 06, generalized
to a mode ("max" or "sum"). exp2 lives in flydsl.expr.math (fmath.exp2).
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

        # ==== YOUR CODE HERE ====
        # 1. LDS + wave_reduce(x, mode) / block_reduce(val, mode) as in puzzle 06,
        #    but mode "max" uses .maximumf and neutral -inf; mode "sum" uses .addf and neutral 0.
        # 2. Buffer tensors for A and C; a_div / c_div = logical_divide(row, make_layout(VEC, 1));
        #    copy_atom = BufferCopy128b.
        # 3. Pass 1: for t in range_constexpr(num_tiles): load float4 into row_buffer[],
        #    thread_max = thread_max.maximumf(x.reduce(ReductionOp.MAX)); global_max = block_reduce(.., "max").
        # 4. Pass 2: row_buffer[i] = fmath.exp2((x - global_max) * LOG2E, fastmath=fm);
        #    thread_sum += exp_val.reduce(ReductionOp.ADD, fastmath=fm); global_sum = block_reduce(.., "sum").
        # 5. Pass 3: out = row_buffer[t] * (1.0 / global_sum); store via copy_atom_call to c_div.
        raise NotImplementedError("Fill in softmax_kernel")

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
