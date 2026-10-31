#!/usr/bin/env python3
"""Vector Abs - C = |A| element-wise using FlyDSL"""

import sys
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith


@flyc.kernel
def vecAbsKernel(
    A: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    print_debug: fx.Constexpr[bool] = True,
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    # if print_debug and bid == 0:
    if print_debug and bid == 0 and tid <= 2:
        fx.printf("[kernel] bid={}, tid={}", bid, tid)

@flyc.jit
def vecAbs(
    A: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    vecAbsKernel(A, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


if __name__ == "__main__":
    THREADS = 256
    VEC = 4
    TILE = THREADS * VEC
    SIZE = TILE * 1000

    A = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    stream = torch.cuda.Stream()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=VEC)

    vecAbs(tA, C, SIZE, SIZE, THREADS, VEC, stream=stream)
    torch.cuda.synchronize()