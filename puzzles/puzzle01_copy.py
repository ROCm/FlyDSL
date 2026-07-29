# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 01 - Copy a tensor.

GOAL: copy an (M, N) fp32 matrix from A into B, one (BLOCK_M, BLOCK_N) tile per
block, moving the data global -> registers -> global.

This is the "hello world" of tile programming. Every later puzzle reuses this
skeleton: make buffer tensors, tile + slice this block, build a tiled copy,
partition per thread, and issue fx.copy.

CONCEPTS: buffer tensors, zipped_divide + slice, thread/value layout,
make_tiled_copy, partition_S/D, make_fragment_like, fx.copy.
READ: FlyDSL for Dummies - Chapter 5 (data movement) and Chapter 7.3.
HINT: closely follow examples/02-tiledCopy.py.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_M = 8
BLOCK_N = 24


@flyc.kernel
def copy_kernel(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # ==== YOUR CODE HERE ====
    # 1. Wrap A and B with fx.rocdl.make_buffer_tensor(...).
    # 2. bA = this block's tile: fx.slice(fx.zipped_divide(A, (BLOCK_M, BLOCK_N)), (None, bid))
    #    (same for bB).
    # 3. thr_layout = fx.make_layout((4, 1), (1, 1)); val_layout = fx.make_layout((1, 8), (1, 1))
    # 4. copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    #    tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
    #    tiled_copy = fx.make_tiled_copy(copy_atom, tv, tile_mn)
    # 5. thr_copy = tiled_copy.get_slice(tid)
    #    src = thr_copy.partition_S(bA); dst = thr_copy.partition_D(bB)
    #    frag = fx.make_fragment_like(src)
    # 6. fx.copy(copy_atom, src, frag); fx.copy(copy_atom, frag, dst)
    raise NotImplementedError("Fill in copy_kernel")


@flyc.jit
def copy(A: fx.Tensor, B: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    M, N = A.shape.unpack()
    grid = ((M // BLOCK_M) * (N // BLOCK_N), 1, 1)
    copy_kernel(A, B).launch(grid=grid, block=(4, 1, 1), stream=stream)


def run(a: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(a)
    copy(a, b, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return b


if __name__ == "__main__":
    from puzzles import common

    a = torch.arange(24 * 120, dtype=torch.float32).reshape(24, 120).cuda()
    common.check(run(a), common.ref_copy(a), name="puzzle01 copy")
