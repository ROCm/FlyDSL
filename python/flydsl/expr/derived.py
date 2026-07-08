# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


from .._mlir.dialects import fly
from .._mlir.dialects._fly_enum_gen import MmaOperand
from .meta import dsl_loc_tracing
from .numeric import Numeric
from .primitive import *
from .typing import Layout, Tensor, TiledCopy, TiledMma

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
    "gather",
    "scatter",
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

    Examples:
        tensor = make_rmem_tensor(8, fx.Float32)
        tensor = make_rmem_tensor(make_layout(4, 1), fx.Float16)
    """
    if not (isinstance(dtype, type) and issubclass(dtype, Numeric)):
        raise TypeError(f"dtype must be a Numeric subclass, but got {dtype!r}")

    if not isinstance(shape_or_layout, Layout):
        layout = make_ordered_layout(shape_or_layout, 0)
    else:
        layout = shape_or_layout

    tensorTy = fly.MemRefType.get(dtype.ir_type, layout.type, fly.AddressSpace.Register)
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


def _gather_scatter_expand(offset_tensor, operand, pred):
    offset_rank = rank(offset_tensor)
    if offset_rank < 1:
        raise ValueError("offset_tensor must have at least the TV mode")
    if pred is not None:
        pred = make_view(get_iter(pred), prepend(get_layout(pred), make_layout(1, 0)))

    tv = offset_tensor.shape[0].unpack()
    if offset_rank == 1:
        for v in range(tv):
            operand_v = operand[None, v]
            pred_v = None if pred is None else pred[None, v]
            yield offset_tensor[v], operand_v, pred_v
        return

    offset_tensor = group(offset_tensor, 1, offset_rank)
    operand = group(operand, 1, rank(operand))
    if pred is not None:
        pred = group(pred, 2, rank(pred))

    rest = size(offset_tensor.shape[1]).unpack()

    for v in range(tv):
        for i in range(rest):
            operand_v = operand[(None, v), i]
            pred_v = None if pred is None else pred[None, v, i]
            yield offset_tensor[v, i], operand_v, pred_v


@dsl_loc_tracing
def gather(copy_atom, base_iter, offset_tensor, dst_tensor, *, pred=None):
    """indexed load ``dst_tensor = base[offset]``

    Layout contract:

    .. code-block:: text

        copy_atom src value layout : copy_atom.layout_src_tv[1]
        dst_tensor                 : ((AtomV, TV), Rest...)
        offset_tensor              : (TV, Rest...)
        pred                       : (TV, Rest...)  optional

    For each ``(v, rest)`` instance, ``offset_tensor[v, rest]`` advances
    ``base_iter``. The reconstructed source view uses the copy atom's source
    value layout, while ``dst_tensor[(None, v), rest]`` supplies the matching
    destination ``(AtomV,)`` slice.
    """
    src_layout = copy_atom.layout_src_tv[1]
    for off, dst_v, pred_v in _gather_scatter_expand(offset_tensor, dst_tensor, pred):
        src_v = make_view(base_iter + off, src_layout)
        copy(copy_atom, src_v, dst_v, pred=pred_v)


@dsl_loc_tracing
def scatter(copy_atom, src_tensor, base_iter, offset_tensor, *, pred=None):
    """indexed store ``base[offset] = src_tensor``

    Layout contract:

    .. code-block:: text

        src_tensor                 : ((AtomV, TV), Rest...)
        copy_atom dst value layout : copy_atom.layout_dst_tv[1]
        offset_tensor              : (TV, Rest...)
        pred                       : (TV, Rest...)  optional

    For each ``(v, rest)`` instance, ``src_tensor[(None, v), rest]`` supplies
    the source ``(AtomV,)`` slice. The reconstructed destination view uses the copy
    atom's destination value layout at ``base_iter + offset_tensor[v, rest]``.
    """
    dst_layout = copy_atom.layout_dst_tv[1]
    for off, src_v, pred_v in _gather_scatter_expand(offset_tensor, src_tensor, pred):
        dst_v = make_view(base_iter + off, dst_layout)
        copy(copy_atom, src_v, dst_v, pred=pred_v)
