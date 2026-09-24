#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""A second, unrelated kernel for the FLYDSL_HACK_UT_ASM scoping test.

It lives in a real file because the AST rewriter reads kernels back with
``inspect.getsource``, which cannot see a kernel defined in a ``python3 -c`` string.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def scaleKernel(A: fx.Tensor, C: fx.Tensor, n: fx.Int32, block_dim: fx.Constexpr[int]):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    C = fx.rocdl.make_buffer_tensor(C)
    if idx < n:
        C[idx] = A[idx] * fx.Float32(2.0)


@flyc.jit
def scale(
    A: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (const_n + block_dim - 1) // block_dim
    scaleKernel(A, C, n, block_dim).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)
