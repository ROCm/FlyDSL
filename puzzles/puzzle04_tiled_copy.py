# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 04 - 2D tiled copy with a 2D thread layout.

GOAL: copy an (M, N) matrix, but design a genuinely 2D thread layout so the TV
tile matches the (BLOCK_M, BLOCK_N) block tile exactly (one copy-tile per thread,
no iteration).

Key fact: make_layout_tv(thr, val) produces a tile whose shape is the per-mode
product of the thread and value shapes. Choose thr=(8, 3), val=(1, 8) so the tile
is (8*1, 3*8) = (8, 24) == (BLOCK_M, BLOCK_N).

CONCEPTS: make_layout_tv internals, 2D thread layouts, coalesced access.
READ: FlyDSL for Dummies - Chapter 4 (TV layouts) and Chapter 5.
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

    # ==== YOUR CODE HERE ====
    # 1. Make buffer tensors; slice this block's (BLOCK_M, BLOCK_N) tile of A and B.
    # 2. thr_layout = fx.make_layout((8, 3), (3, 1)); val_layout = fx.make_layout((1, 8), (1, 1))
    # 3. copy_atom + tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
    #    tiled_copy = fx.make_tiled_copy(copy_atom, tv, tile_mn)
    # 4. thr_copy.partition_S/D, make_fragment_like, two fx.copy calls.
    raise NotImplementedError("Fill in tiled_copy_kernel")


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
