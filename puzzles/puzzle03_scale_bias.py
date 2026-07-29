# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 03 - Scale and bias: C = A * alpha + beta.

GOAL: compute C = A * alpha + beta elementwise, where alpha and beta are scalars
baked into the kernel as fx.Constexpr (like C++ non-type template parameters).

CONCEPTS: fx.Constexpr specialization, elementwise register math with a scalar.
READ: FlyDSL for Dummies - Chapter 1.4 (Constexpr) and Chapter 5.
HINT: start from puzzle02, but with a single input tensor and the compute line
      thr_rC.store(thr_rA.load() * ALPHA + BETA).
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

    # ==== YOUR CODE HERE ====
    # Same structure as puzzle02, but one input A and the compute:
    #     thr_rC.store(thr_rA.load() * ALPHA + BETA)
    raise NotImplementedError("Fill in scale_bias_kernel")


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
