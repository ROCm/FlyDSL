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

from .._mlir import ir
from .._mlir.dialects._fly_enum_gen import MmaOperand
from .primitive import *
from .typing import Layout, Tensor

__all__ = [
    # Tiled Operation
    "CopyAtom",
    "MmaAtom",
    "TiledCopy",
    "TiledMma",
    "ThrCopy",
    "ThrMma",
    "make_layout_tv",
    "make_tiled_copy",
    "make_tiled_copy_A",
    "make_tiled_copy_B",
    "make_tiled_copy_C",
]


class Atom:
    """Base class for hardware instruction atoms (copy/MMA).

    An atom wraps a single MLIR ``ir.Value`` that represents a hardware
    instruction descriptor in the Fly dialect type system.
    """

    def __init__(self, value: ir.Value):
        self.value = value
        self.atom_ty = self.value.type

    @classmethod
    def __fly_construct__(cls, values):
        return cls(values[0])

    def __fly_values__(self):
        return [self.value]


class CopyAtom:
    """Atom describing a single hardware copy instruction.

    Wraps the thread-value layouts that define how threads cooperate
    to perform a single copy operation (e.g., ``UniversalCopy(128)``
    for 128-bit copies).

    Properties:
        thr_layout: Thread layout of the copy atom.
        tv_layout_src: Thread-value layout for the source.
        tv_layout_dst: Thread-value layout for the destination.
        tv_layout_ref: Thread-value reference layout.
    """

    def __init__(self, value: ir.Value):
        self.value = value
        self.atom_ty = self.value.type

    @classmethod
    def __fly_construct__(cls, values):
        return cls(values[0])

    def __fly_values__(self):
        return [self.value]

    def __str__(self):
        return f"CopyAtom({self.atom_ty})"

    @property
    def thr_layout(self):
        return static(self.atom_ty.thr_layout)

    @property
    def tv_layout_src(self):
        return static(self.atom_ty.tv_layout_src)

    @property
    def tv_layout_dst(self):
        return static(self.atom_ty.tv_layout_dst)

    @property
    def tv_layout_ref(self):
        return static(self.atom_ty.tv_layout_ref)


class MmaAtom(Atom):
    """Atom describing a single MMA (matrix multiply-accumulate) instruction.

    Wraps MFMA instruction descriptors with thread-value layouts for
    operands A, B, and C, plus the MNK shape of the instruction.

    Properties:
        thr_layout: Thread layout of the MMA atom.
        shape_mnk: The (M, N, K) shape of the MMA instruction.
        tv_layout_A/B/C: Thread-value layouts for operands A, B, C.
    """

    def __str__(self):
        return f"MmaAtom({self.atom_ty})"

    @property
    def thr_layout(self):
        return static(self.atom_ty.thr_layout)

    @property
    def shape_mnk(self):
        return static(self.atom_ty.shape_mnk)

    @property
    def tv_layout_A(self):
        return static(self.atom_ty.tv_layout_a)

    @property
    def tv_layout_B(self):
        return static(self.atom_ty.tv_layout_b)

    @property
    def tv_layout_C(self):
        return static(self.atom_ty.tv_layout_c)


class TiledCopy:
    """Tiled copy operation composed from a CopyAtom repeated across threads.

    Use ``fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)`` to construct.
    Call ``get_slice(thr_idx)`` to obtain a per-thread ``ThrCopy`` view for
    partitioning source/destination tensors.

    Properties:
        tiled_tv_layout_S: Full tiled thread-value layout for source.
        tiled_tv_layout_D: Full tiled thread-value layout for destination.
    """

    def __init__(self, value):
        self.value = value
        self.tiled_copy_ty = self.value.type

    @classmethod
    def __fly_construct__(cls, values):
        return cls(values[0])

    def __fly_values__(self):
        return [self.value]

    def __str__(self):
        return f"TiledCopy({self.tiled_copy_ty})"

    def get_slice(self, thr_idx):
        return ThrCopy(self, thr_idx)

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)

    @property
    def tiled_tv_layout_S(self):
        return static(self.tiled_copy_ty.tiled_tv_layout_src)

    @property
    def tiled_tv_layout_D(self):
        return static(self.tiled_copy_ty.tiled_tv_layout_dst)


class TiledMma:
    """Tiled MMA operation composed from an MmaAtom repeated across threads.

    Use ``fx.make_tiled_mma(mma_atom, ...)`` to construct. Call
    ``get_slice(thr_idx)`` to obtain a per-thread ``ThrMma`` view for
    partitioning A, B, C operands.

    Properties:
        tile_size_mnk: The tiled (M, N, K) size.
        thr_layout_vmnk: Thread layout in (V, M, N, K) order.
        tiled_tv_layout_A/B/C: Full tiled thread-value layouts for operands.
    """

    def __init__(self, value):
        self.value = value
        self.tiled_mma_ty = self.value.type

    @classmethod
    def __fly_construct__(cls, values):
        return cls(values[0])

    def __fly_values__(self):
        return [self.value]

    def __str__(self):
        return f"TiledMma({self.tiled_mma_ty})"

    def get_slice(self, thr_idx):
        return ThrMma(self, thr_idx)

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)

    def make_fragment_A(self, a: Tensor):
        return make_fragment_like(a)

    def make_fragment_B(self, b: Tensor):
        return make_fragment_like(b)

    def make_fragment_C(self, c: Tensor):
        return make_fragment_like(c)

    @property
    def tile_size_mnk(self):
        return static(self.tiled_mma_ty.tile_size_mnk)

    @property
    def thr_layout_vmnk(self):
        return static(self.tiled_mma_ty.thr_layout_vmnk)

    @property
    def tiled_tv_layout_A(self):
        return static(self.tiled_mma_ty.tiled_tv_layout_a)

    @property
    def tiled_tv_layout_B(self):
        return static(self.tiled_mma_ty.tiled_tv_layout_b)

    @property
    def tiled_tv_layout_C(self):
        return static(self.tiled_mma_ty.tiled_tv_layout_c)


class ThrCopy(TiledCopy):
    """Per-thread view of a TiledCopy for partitioning source/destination tensors.

    Obtained via ``TiledCopy.get_slice(thr_idx)``. Provides ``partition_S``,
    ``partition_D``, and ``retile`` methods for tensor partitioning.
    """

    def __init__(self, tiled_copy: TiledCopy, thr_idx):
        super().__init__(tiled_copy.value)
        self.tiled_copy = tiled_copy
        self._thr_idx = thr_idx
        self._thr_idx_int = make_int_tuple(self.thr_idx)

    @property
    def thr_idx(self):
        return self._thr_idx

    def partition_S(self, src: Tensor):
        return tiled_copy_partition_src(self.value, src, self._thr_idx_int)

    def partition_D(self, dst: Tensor):
        return tiled_copy_partition_dst(self.value, dst, self._thr_idx_int)

    def retile(self, t: Tensor):
        return tiled_copy_retile(self.value, t)


class ThrMma(TiledMma):
    """Per-thread view of a TiledMma for partitioning A, B, C operands.

    Obtained via ``TiledMma.get_slice(thr_idx)``. Provides ``partition_A``,
    ``partition_B``, and ``partition_C`` methods.
    """

    def __init__(self, tiled_mma: TiledMma, thr_idx):
        super().__init__(tiled_mma.value)
        self.tiled_mma = tiled_mma
        self._thr_idx = thr_idx
        self._thr_idx_int = make_int_tuple(self.thr_idx)

    @property
    def thr_idx(self):
        return self._thr_idx

    def partition_A(self, a: Tensor):
        return tiled_mma_partition(MmaOperand.A, self.tiled_mma.value, a, self._thr_idx_int)

    def partition_B(self, b: Tensor):
        return tiled_mma_partition(MmaOperand.B, self.tiled_mma.value, b, self._thr_idx_int)

    def partition_C(self, c: Tensor):
        return tiled_mma_partition(MmaOperand.C, self.tiled_mma.value, c, self._thr_idx_int)


def make_layout_tv(thr_layout, val_layout, loc=None, ip=None):
    """Build a thread-value (TV) layout from separate thread and value layouts.

    Computes the raked product of *thr_layout* and *val_layout*, then
    derives a TV mapping via ``composition(right_inverse(layout_mn), ...)``.

    Returns:
        Tuple of (tiler_mn, layout_tv).
    """
    layout_mn = raked_product(thr_layout, val_layout)
    thr_size = size(thr_layout)
    val_size = size(val_layout)
    tmp = make_layout((thr_size, val_size), (1, thr_size))

    layout_tv = composition(right_inverse(layout_mn), tmp)

    tiler_mn = int_tuple_product_each(get_shape(layout_mn))
    return (tiler_mn, layout_tv)


def make_tiled_copy_A(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand A of *tiled_mma*."""
    layout_tv = tiled_mma.tiled_tv_layout_A
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        [
            make_layout(select(tile_size, [0]), 1),
            make_layout(select(tile_size, [2]), 1),
        ]
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)


def make_tiled_copy_B(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand B of *tiled_mma*."""
    layout_tv = tiled_mma.tiled_tv_layout_B
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        [
            make_layout(select(tile_size, [1]), 1),
            make_layout(select(tile_size, [2]), 1),
        ]
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)


def make_tiled_copy_C(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand C of *tiled_mma*."""
    layout_tv = tiled_mma.tiled_tv_layout_C
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        [
            make_layout(select(tile_size, [0]), 1),
            make_layout(select(tile_size, [1]), 1),
        ]
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)
