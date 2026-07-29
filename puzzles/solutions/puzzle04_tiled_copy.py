# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 04 (solution) - 2D tiled copy with a 2D thread layout.

Same result as puzzle 01, but here the thread layout is genuinely 2D: 8 threads
down M and 3 across N (24 threads), each owning a 1x8 strip. The TV layout's tile
(shape = per-mode product of thread x value) exactly matches the (8, 24) block
tile, so each thread issues one copy-tile with no iteration.

Concepts: make_layout_tv internals (tile = thr_shape (x) val_shape per mode),
2D thread layouts, coalesced access.
Book: Chapter 4 (TV layouts), Chapter 5.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_M = 8
BLOCK_N = 24


@flyc.kernel
def tiled_copy_kernel(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    bA = fx.slice(fx.zipped_divide(A, (BLOCK_M, BLOCK_N)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, (BLOCK_M, BLOCK_N)), (None, bid))

    # 8 threads down M, 3 across N = 24 threads; each owns 1x8 along N.
    thr_layout = fx.make_layout((8, 3), (3, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    src = thr_copy.partition_S(bA)
    dst = thr_copy.partition_D(bB)
    frag = fx.make_fragment_like(src)

    fx.copy(copy_atom, src, frag)
    fx.copy(copy_atom, frag, dst)


@flyc.jit
def tiled_copy(A: fx.Tensor, B: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    M, N = A.shape.unpack()
    grid = ((M // BLOCK_M) * (N // BLOCK_N), 1, 1)
    tiled_copy_kernel(A, B).launch(grid=grid, block=(24, 1, 1), stream=stream)


def run(a: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(a)
    tiled_copy(a, b, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return b


if __name__ == "__main__":
    from puzzles import common

    a = torch.arange(24 * 120, dtype=torch.float32).reshape(24, 120).cuda()
    common.check(run(a), common.ref_copy(a), name="puzzle04 tiled_copy")
