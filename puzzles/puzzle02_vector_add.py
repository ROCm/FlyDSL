# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 02 - Vectorized, predicated vector add (C = A + B).

GOAL: add two (M, N) fp32 matrices where M, N need NOT be multiples of the tile.
Each thread loads a float4 from A and B, adds in registers, stores a float4 to C,
and a per-value predicate masks out-of-bounds lanes on the border blocks.

CONCEPTS: make_tiled_copy_tv, float4 value layout, identity/coordinate tensor,
elem_less predicate, register-fragment arithmetic (.load()/.store()).
READ: FlyDSL for Dummies - Chapter 5 (predication) and Chapter 7.1.
HINT: closely follow examples/01-vectorAdd.py.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, tiled_copy: fx.TiledCopy):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    # ==== YOUR CODE HERE ====
    # 1. M, N = A.shape.unpack(); build an identity/coordinate tensor idC over (M, N)
    #    with fx.make_view((0, 0), fx.make_identity_layout((M, N))).
    # 2. TileMN = tiled_copy.tile_mn; for A, B, C and idC, take this block's tile via
    #    fx.flat_divide(X, TileMN)[None, None, bid_x, bid_y].
    # 3. thr_copy = tiled_copy.get_slice(tid); partition_S the sources / partition_D C /
    #    partition_S the coordinate tensor (slice its inner value mode).
    # 4. Allocate register fragments with fx.make_fragment_like(...) (a Boolean one for
    #    the predicate). Fill the predicate: thr_pC[a] = fx.elem_less(thr_cC[a], (M, N)).
    # 5. copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    #    fx.copy(..., pred=thr_pC) A and B into registers; add via .load()/.store();
    #    fx.copy(..., pred=thr_pC) the result back to C.
    raise NotImplementedError("Fill in vector_add_kernel")


@flyc.jit
def vector_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((8, 16), order=(1, 0)),
        fx.make_ordered_layout((1, 4), order=(0, 1)),
    )
    tile_m, tile_n = tiled_copy.tile_mn.unpack()
    M, N = A.shape.unpack()
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    vector_add_kernel(A, B, C, tiled_copy).launch(
        grid=(grid_m, grid_n, 1), block=(8 * 16, 1, 1), stream=stream
    )


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.zeros_like(a)
    vector_add(a, b, c, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(100, 1000, dtype=torch.float32).cuda()
    b = torch.randn(100, 1000, dtype=torch.float32).cuda()
    common.check(run(a, b), common.ref_vector_add(a, b), name="puzzle02 vector_add")
