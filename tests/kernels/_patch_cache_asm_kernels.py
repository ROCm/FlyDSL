# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Kernels whose cache entries tests/unit/test_patch_cache_asm.py patches.

They live in a real file because the AST rewriter reads kernels back with
``inspect.getsource``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def tiny_kernel():
    pass


@flyc.kernel
def other_kernel():
    pass


@flyc.jit
def single():
    tiny_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))


@flyc.jit
def both():
    tiny_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))
    other_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))


@flyc.jit
def sized(block: fx.Constexpr[int]):
    # Each value is its own cache entry, all launching the same kernel symbol.
    tiny_kernel().launch(grid=(1, 1, 1), block=(block, 1, 1))
