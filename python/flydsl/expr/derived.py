# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


from .._mlir.dialects import fly
from .._mlir.dialects._fly_enum_gen import MmaOperand
from .meta import dsl_loc_tracing
from .numeric import Boolean, Numeric
from .primitive import *
from .typing import Int8, Layout, Tensor, TiledCopy, TiledMma

__all__ = [
    # Tiled Operation
    "ThrCopy",
    "ThrMma",
    "make_rmem_tensor",
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

    @dsl_loc_tracing
    def partition_S(self, src: Tensor):
        return tiled_copy_partition_src(self, src, self._thr_idx_int)

    @dsl_loc_tracing
    def partition_D(self, dst: Tensor):
        return tiled_copy_partition_dst(self, dst, self._thr_idx_int)

    @dsl_loc_tracing
    def retile(self, t: Tensor):
        return tiled_copy_retile(self, t)


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

    @dsl_loc_tracing
    def partition_A(self, a: Tensor):
        return tiled_mma_partition(MmaOperand.A, self.tiled_mma, a, self._thr_idx_int)

    @dsl_loc_tracing
    def partition_B(self, b: Tensor):
        return tiled_mma_partition(MmaOperand.B, self.tiled_mma, b, self._thr_idx_int)

    @dsl_loc_tracing
    def partition_C(self, c: Tensor):
        return tiled_mma_partition(MmaOperand.C, self.tiled_mma, c, self._thr_idx_int)


@dsl_loc_tracing
def make_rmem_tensor(shape_or_layout, dtype):
    """Creates a tensor in register memory with the specified layout/shape and data type.

    If shape_or_layout is a shape, it is converted to a layout with column-major ordering.
    Booleans are canonically stored as Int8.

    Examples:
        tensor = make_rmem_tensor(8, fx.Float32)
        tensor = make_rmem_tensor(make_layout(4, 1), fx.Float16)
    """
    if not (isinstance(dtype, type) and issubclass(dtype, Numeric)):
        raise TypeError(f"dtype must be a Numeric subclass, but got {dtype!r}")
    elem_ty = dtype.ir_type if dtype is not Boolean else Int8.ir_type

    if not isinstance(shape_or_layout, Layout):
        layout = make_ordered_layout(shape_or_layout, 0)
    else:
        layout = shape_or_layout

    tensorTy = fly.MemRefType.get(elem_ty, layout.type, fly.AddressSpace.Register)
    return memref_alloca(tensorTy, layout=layout)


@dsl_loc_tracing
def make_layout_tv(thr_layout, val_layout):
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


@dsl_loc_tracing
def make_tiled_copy_tv(atom, thr_layout, val_layout):
    tiler_mn, layout_tv = make_layout_tv(thr_layout, val_layout)
    return make_tiled_copy(atom, layout_tv, tiler_mn)


@dsl_loc_tracing
def make_tiled_copy_A(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand A of *tiled_mma*."""
    layout_tv = tiled_mma.tv_layout_A_tiled
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        make_layout(select(tile_size, [0]), 1),
        make_layout(select(tile_size, [2]), 1),
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)


@dsl_loc_tracing
def make_tiled_copy_B(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand B of *tiled_mma*."""
    layout_tv = tiled_mma.tv_layout_B_tiled
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        make_layout(select(tile_size, [1]), 1),
        make_layout(select(tile_size, [2]), 1),
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)


@dsl_loc_tracing
def make_tiled_copy_C(copy_atom, tiled_mma):
    """Create a TiledCopy matched to operand C of *tiled_mma*."""
    layout_tv = tiled_mma.tv_layout_C_tiled
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = make_tile(
        make_layout(select(tile_size, [0]), 1),
        make_layout(select(tile_size, [1]), 1),
    )
    return make_tiled_copy(copy_atom, layout_tv, tile_mn)
