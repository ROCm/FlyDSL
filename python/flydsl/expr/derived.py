# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Derived tiled operations for GPU kernel partitioning.

This module provides high-level abstractions for tiled copy and MMA (Matrix
Multiply-Accumulate) operations. These classes partition tensors across
threads and values according to the Fly dialect's layout algebra, enabling
efficient MFMA-based GEMM and data movement patterns.

Typical usage::

    import flydsl.expr as fx

    copy_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float16)
    mma_atom = fx.make_mma_atom(fx.MFMA(16, 16, 16, fx.Float32), fx.Float16)

    tiled_mma = fx.make_tiled_mma(mma_atom, ...)
    thr_mma = tiled_mma.get_slice(fx.thread_idx.x)
    tAr = thr_mma.partition_A(sA)
"""

from .._mlir.dialects._fly_enum_gen import MmaOperand
from .meta import traced_op
from .primitive import *
from .typing import Tensor, TiledCopy, TiledMma

__all__ = [
    # Tiled Operation
    "ThrCopy",
    "ThrMma",
    "make_layout_tv",
    "make_tiled_copy_tv",
    "make_tiled_copy",
    "make_tiled_copy_A",
    "make_tiled_copy_B",
    "make_tiled_copy_C",
]


class ThrCopy(TiledCopy):
    """Per-thread view of a TiledCopy for partitioning source/destination tensors.

    Obtained via ``TiledCopy.get_slice(thr_idx)``. Provides ``partition_S``,
    ``partition_D``, and ``retile`` methods for tensor partitioning.
    """

    def __init__(self, tiled_copy: TiledCopy, thr_idx):
        super().__init__(tiled_copy)
        self.tiled_copy = tiled_copy
        self._thr_idx = thr_idx
        self._thr_idx_int = make_int_tuple(self.thr_idx)

    @property
    def thr_idx(self):
        return self._thr_idx

    @traced_op
    def partition_S(self, src: Tensor, loc=None, ip=None):
        return tiled_copy_partition_src(self, src, self._thr_idx_int, loc=loc, ip=ip)

    @traced_op
    def partition_D(self, dst: Tensor, loc=None, ip=None):
        return tiled_copy_partition_dst(self, dst, self._thr_idx_int, loc=loc, ip=ip)

    @traced_op
    def retile(self, t: Tensor, loc=None, ip=None):
        return tiled_copy_retile(self, t, loc=loc, ip=ip)


class ThrMma(TiledMma):
    """Per-thread view of a TiledMma for partitioning A, B, C operands.

    Obtained via ``TiledMma.get_slice(thr_idx)``. Provides ``partition_A``,
    ``partition_B``, and ``partition_C`` methods.
    """

    def __init__(self, tiled_mma: TiledMma, thr_idx):
        super().__init__(tiled_mma)
        self.tiled_mma = tiled_mma
        self._thr_idx = thr_idx
        self._thr_idx_int = make_int_tuple(self.thr_idx)

    @property
    def thr_idx(self):
        return self._thr_idx

    @traced_op
    def partition_A(self, a: Tensor, loc=None, ip=None):
        return tiled_mma_partition(MmaOperand.A, self.tiled_mma, a, self._thr_idx_int, loc=loc, ip=ip)

    @traced_op
    def partition_B(self, b: Tensor, loc=None, ip=None):
        return tiled_mma_partition(MmaOperand.B, self.tiled_mma, b, self._thr_idx_int, loc=loc, ip=ip)

    @traced_op
    def partition_C(self, c: Tensor, loc=None, ip=None):
        return tiled_mma_partition(MmaOperand.C, self.tiled_mma, c, self._thr_idx_int, loc=loc, ip=ip)


def make_layout_tv(thr_layout, val_layout, loc=None, ip=None):
    """Build a thread-value (TV) layout from separate thread and value layouts.

    Computes the raked product of *thr_layout* and *val_layout*, then
    derives a TV mapping via ``composition(right_inverse(layout_mn), ...)``.

    Returns:
        Tuple of (tiler_mn, layout_tv).
    """
    if not thr_layout.is_static:
        raise ValueError("thr_layout is not static")
    if not val_layout.is_static:
        raise ValueError("val_layout is not static")

    layout_mn = raked_product(thr_layout, val_layout)
    thr_size = size(thr_layout).to_py_value()
    val_size = size(val_layout).to_py_value()
    tmp = make_layout((thr_size, val_size), (1, thr_size))

    layout_tv = composition(right_inverse(layout_mn), tmp)

    tiler_mn = int_tuple_product_each(get_shape(layout_mn)).to_py_value()
    return (tiler_mn, layout_tv)


def make_tiled_copy_tv(atom, thr_layout, val_layout):
    (tiler_mn, layout_tv) = make_layout_tv(thr_layout, val_layout)
    return make_tiled_copy(atom, layout_tv, tiler_mn)


def make_tiled_copy_A(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand A of *tiled_mma*."""
    layout_tv = tiled_mma.tv_layout_A_tiled
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        make_layout(select(tile_size, [0]), 1),
        make_layout(select(tile_size, [2]), 1),
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)


def make_tiled_copy_B(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand B of *tiled_mma*."""
    layout_tv = tiled_mma.tv_layout_B_tiled
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        make_layout(select(tile_size, [1]), 1),
        make_layout(select(tile_size, [2]), 1),
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)


def make_tiled_copy_C(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand C of *tiled_mma*."""
    layout_tv = tiled_mma.tv_layout_C_tiled
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        make_layout(select(tile_size, [0]), 1),
        make_layout(select(tile_size, [1]), 1),
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)
