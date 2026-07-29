# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 05 (solution) - Transpose as a layout operation.

The lesson: transpose need not shuffle data at all - it is a *layout* change. We
copy A[m, n] -> Bt[m, n], where Bt is a column-major (transposed-stride) view of
the output buffer B. Writing logical (m, n) into a column-major view lands the
value at physical B[n, m], i.e. B = A.T. The kernel is an ordinary tiled copy.

Because the output view is strided (its fast axis has stride M, not 1), we use a
scalar BufferCopy32b atom so no store needs contiguous 128-bit lanes.

Concepts: layout vs data movement, strided views, coordinate mapping.
Book: Chapter 3 (layouts), Chapter 5.
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

    A = fx.rocdl.make_buffer_tensor(A)
    Bt = fx.rocdl.make_buffer_tensor(Bt)  # column-major view of the output

    bA = fx.slice(fx.zipped_divide(A, (BLOCK_M, BLOCK_N)), (None, bid))
    bB = fx.slice(fx.zipped_divide(Bt, (BLOCK_M, BLOCK_N)), (None, bid))

    thr_layout = fx.make_layout((16, 2), (2, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    src = thr_copy.partition_S(bA)
    dst = thr_copy.partition_D(bB)
    frag = fx.make_fragment_like(src)

    fx.copy(copy_atom, src, frag)
    fx.copy(copy_atom, frag, dst)


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
