# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 03 (solution) - Scale and bias: C = A * alpha + beta.

Introduces Constexpr specialization: alpha and beta are baked into the IR (like
C++ non-type template params), so each (alpha, beta) compiles a specialized
kernel. Compute happens on register fragments.

Concepts: fx.Constexpr, elementwise register math with a scalar.
Book: Chapter 1.4 (Constexpr), Chapter 5.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def scale_bias_kernel(
    A: fx.Tensor,
    C: fx.Tensor,
    tiled_copy: fx.TiledCopy,
    ALPHA: fx.Constexpr,
    BETA: fx.Constexpr,
):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
    TileMN = tiled_copy.tile_mn

    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]

    thr_copy = tiled_copy.get_slice(tid)
    thr_gA = thr_copy.partition_S(gA)
    thr_gC = thr_copy.partition_D(gC)
    thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]

    thr_rA = fx.make_fragment_like(thr_gA)
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)

    for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
        thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))

    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)

    thr_rC.store(thr_rA.load() * ALPHA + BETA)

    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)


@flyc.jit
def scale_bias(
    A: fx.Tensor,
    C: fx.Tensor,
    alpha: fx.Constexpr,
    beta: fx.Constexpr,
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
    scale_bias_kernel(A, C, tiled_copy, alpha, beta).launch(
        grid=(grid_m, grid_n, 1), block=(8 * 16, 1, 1), stream=stream
    )


def run(a: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    c = torch.zeros_like(a)
    scale_bias(a, c, float(alpha), float(beta), stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c


if __name__ == "__main__":
    from puzzles import common

    a = torch.randn(64, 256, dtype=torch.float32).cuda()
    common.check(run(a, 2.5, -1.25), common.ref_scale_bias(a, 2.5, -1.25), name="puzzle03 scale_bias")
