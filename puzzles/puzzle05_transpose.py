# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 05 - Transpose as a layout operation.

GOAL: compute B = A.T. The trick: do NOT shuffle data. Copy A[m, n] into a
column-major (transposed-stride) view of the output, so logical (m, n) lands at
physical B[n, m]. The kernel is an ordinary tiled copy; the transpose lives
entirely in the output view's strides (see run()).

Use a scalar BufferCopy32b atom because the output view is strided (no contiguous
128-bit stores).

CONCEPTS: layout vs data movement, strided views, coordinate mapping.
READ: FlyDSL for Dummies - Chapter 3 (layouts) and Chapter 5.
HINT: the kernel is identical to puzzle 04 but with a BufferCopy32b atom.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_M = 16
BLOCK_N = 16


@flyc.kernel
def transpose_kernel(A: fx.Tensor, Bt: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # ==== YOUR CODE HERE ====
    # Ordinary tiled copy A -> Bt using a BufferCopy32b atom (scalar stores, since
    # Bt is a strided/transposed view). thr=(16,2), val=(1,8) -> tile (16,16).
    raise NotImplementedError("Fill in transpose_kernel")


@flyc.jit
def transpose(A: fx.Tensor, Bt: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    M, N = A.shape.unpack()
    grid = ((M // BLOCK_M) * (N // BLOCK_N), 1, 1)
    transpose_kernel(A, Bt).launch(grid=grid, block=(32, 1, 1), stream=stream)


def run(a: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    b = torch.zeros(n, m, dtype=a.dtype, device=a.device)  # holds A.T
    bt = b.t()  # (M, N) view with strides (1, M): writing (m,n) -> b[n,m]
    transpose(a, bt, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return b


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(64, 48, dtype=torch.float32).cuda()
    common.check(run(a), common.ref_transpose(a), name="puzzle05 transpose")
