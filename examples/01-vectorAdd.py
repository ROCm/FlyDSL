# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Vectorized, predicated, target-neutral 2D elementwise add (C = A + B).

This example is **target-neutral**: it uses only the backend-agnostic ``flydsl.expr`` API, so it
supports on any backend.

Highlights:
  1. **float4 vectorization** via ``UniversalCopy128b`` -- each copy atom moves 128 bits
     (4 x f32) along the contiguous (N) axis, so every thread loads/stores one ``float4``.
  2. **Predicated OOB masking**: the (M, N) shape need not be a multiple of the block tile,
     so border blocks have threads whose float4 lies past the tensor. A per-atom boolean
     predicate (``coord < (M, N)``) gates each copy, so a load/store never touches OOB memory.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    tiled_copy: fx.TiledCopy,
):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    # Identity (coordinate) tensor: value == logical coord (m, n).
    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))

    TileMN = tiled_copy.tile_mn

    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]

    thr_copy = tiled_copy.get_slice(tid)

    thr_gA = thr_copy.partition_S(gA)
    thr_gB = thr_copy.partition_S(gB)
    thr_gC = thr_copy.partition_D(gC)
    thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]

    thr_rA = fx.make_fragment_like(thr_gA)
    thr_rB = fx.make_fragment_like(thr_gB)
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)

    for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
        thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))

    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)
    fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)

    thr_rC.store(thr_rA.load() + thr_rB.load())

    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)


@flyc.jit
def vector_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
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
    vector_add_kernel(A, B, C, tiled_copy).launch(grid=(grid_m, grid_n, 1), block=(8 * 16, 1, 1), stream=stream)


M, N = 100, 1000

A = torch.randn(M, N, dtype=torch.float32, device=torch.device("cuda"))
B = torch.randn(M, N, dtype=torch.float32, device=torch.device("cuda"))
C = torch.zeros(M, N, dtype=torch.float32, device=torch.device("cuda"))

vector_add(A, B, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()

if torch.allclose(A + B, C):
    print("PASS")
else:
    print("FAIL:")
    print(A + B)
    print(C)
