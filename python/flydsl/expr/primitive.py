# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import overload

from .._mlir import ir
from .._mlir.dialects import arith as _arith
from .._mlir.dialects import fly
from .._mlir.dialects.fly import (
    AddressSpace,
    AtomicOp,
    CachePolicy,
    ComposedLayoutType,
    CoordTensorType,
    CopyAtomType,
    CopyOpUniversalAtomicType,
    CopyOpUniversalCopyType,
    GemmTraversalOrder,
    IntTupleType,
    LayoutType,
    MemRefType,
    MmaAtomType,
    MmaOperand,
    MmaOpUniversalFMAType,
    PointerType,
    SwizzleType,
    TiledCopyType,
    TiledMmaType,
    TileType,
    #
    has_none,
)
from .._mlir.extras import types as T
from .meta import traced_op

__all__ = [
    # Maybe remove it in the future
    "T",
    # "arith",
    # Enum Attributes
    "AtomicOp",
    "AddressSpace",
    "CachePolicy",
    "MmaOperand",
    "GemmTraversalOrder",
    # Types
    "IntTupleType",
    "TileType",
    "LayoutType",
    "SwizzleType",
    "ComposedLayoutType",
    "PointerType",
    "MemRefType",
    "CoordTensorType",
    "CopyAtomType",
    "MmaAtomType",
    "TiledCopyType",
    "TiledMmaType",
    "CopyOpUniversalCopyType",
    "CopyOpUniversalAtomicType",
    "MmaOpUniversalFMAType",
    # UniversalOps
    "UniversalCopy",
    "UniversalCopy8b",
    "UniversalCopy16b",
    "UniversalCopy32b",
    "UniversalCopy64b",
    "UniversalCopy128b",
    "UniversalAtomic",
    "UniversalAtomicAdd",
    "UniversalAtomicMax",
    "UniversalAtomicMin",
    "UniversalAtomicAnd",
    "UniversalAtomicOr",
    "UniversalAtomicInc",
    "UniversalAtomicDec",
    "UniversalFMA",
    # Constexpr functions
    "const_expr",
    "range_constexpr",
    "rank",
    "depth",
    "has_none",
    # DSL functions
    "static",
    "make_int_tuple",
    "make_shape",
    "make_stride",
    "make_coord",
    "make_layout",
    "make_layout_like",
    "make_ordered_layout",
    "make_composed_layout",
    "make_identity_layout",
    "make_view",
    "make_fragment_like",
    "get_scalar",
    "get_leaves",
    "get_shape",
    "get_stride",
    "get_layout",
    "get_iter",
    "composed_get_inner",
    "composed_get_offset",
    "composed_get_outer",
    "int_tuple_add",
    "int_tuple_sub",
    "int_tuple_mul",
    "int_tuple_div",
    "int_tuple_mod",
    "int_tuple_product",
    "int_tuple_product_each",
    "int_tuple_product_like",
    "shape_div",
    "ceil_div",
    "elem_less",
    "equal",
    "get",
    "get_",
    "take",
    "select",
    "group",
    "append",
    "prepend",
    "slice",
    "dice",
    "size",
    "coprofile",
    "coshape",
    "cosize",
    "crd2idx",
    "idx2crd",
    "get_flat_coord",
    "get_1d_coord",
    "coalesce",
    "composition",
    "complement",
    "right_inverse",
    "left_inverse",
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    "block_product",
    "raked_product",
    "recast_layout",
    "tile_to_shape",
    "make_mma_atom",
    "make_copy_atom",
    "copy_atom_call",
    "mma_atom_call",
    "make_tiled_copy",
    "make_tiled_mma",
    "tiled_copy_partition_src",
    "tiled_copy_partition_dst",
    "tiled_copy_retile",
    "tiled_mma_partition",
    "tiled_mma_partition_shape",
    "mma_make_fragment",
    "copy",
    "gemm",
    "make_ptr",
    "get_dyn_shared",
    "inttoptr",
    "ptrtoint",
    "add_offset",
    "apply_swizzle",
    "ptr_load",
    "ptr_store",
    "recast_iter",
    "memref_alloca",
    "memref_load_vec",
    "memref_store_vec",
    "memref_load",
    "memref_store",
    "printf",
    "assume",
    "make_tile",
]


UniversalCopy = lambda bit_size: CopyOpUniversalCopyType.get(bit_size)
UniversalCopy8b = lambda: CopyOpUniversalCopyType.get(8)
UniversalCopy16b = lambda: CopyOpUniversalCopyType.get(16)
UniversalCopy32b = lambda: CopyOpUniversalCopyType.get(32)
UniversalCopy64b = lambda: CopyOpUniversalCopyType.get(64)
UniversalCopy128b = lambda: CopyOpUniversalCopyType.get(128)

UniversalAtomic = lambda atomic_op, val_type: CopyOpUniversalAtomicType.get(int(atomic_op), val_type)
UniversalAtomicAdd = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Add), val_type)
UniversalAtomicMax = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Max), val_type)
UniversalAtomicMin = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Min), val_type)
UniversalAtomicAnd = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.And), val_type)
UniversalAtomicOr = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Or), val_type)
UniversalAtomicInc = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Inc), val_type)
UniversalAtomicDec = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Dec), val_type)

UniversalFMA = lambda ty: MmaOpUniversalFMAType.get(ty.ir_type)


def const_expr(x):
    return x


def range_constexpr(*args):
    return range(*args)


def rank(int_or_tuple):
    return fly.rank(int_or_tuple)


def depth(int_or_tuple):
    return fly.depth(int_or_tuple)


# ===----------------------------------------------------------------------=== #
# Constructors
# ===----------------------------------------------------------------------=== #


@traced_op
def static(result_type, loc=None, ip=None):
    return fly.static(result_type, loc=loc, ip=ip)


@traced_op
def make_int_tuple(elems, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(elems)
    return fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_shape(*shape, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(shape)
    return fly.make_shape(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_stride(*stride, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(stride)
    return fly.make_stride(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_coord(*coord, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(coord)
    return fly.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_layout(shape, stride, loc=None, ip=None):
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(shape)
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(stride, ir.Value):
        strideTy, dyncElems = fly.infer_int_tuple_type(stride)
        stride = fly.make_stride(strideTy, dyncElems, loc=loc, ip=ip)
    return fly.make_layout(shape, stride=stride, loc=loc, ip=ip)


@traced_op
def make_layout_like(ref, loc=None, ip=None):
    return fly.make_layout_like(ref, loc=loc, ip=ip)


@traced_op
def make_ordered_layout(shape, order, loc=None, ip=None):
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(shape)
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(order, ir.Value):
        orderTy, dyncElems = fly.infer_int_tuple_type(order)
        order = fly.make_int_tuple(orderTy, dyncElems, loc=loc, ip=ip)
    return fly.make_ordered_layout(shape, order, loc=loc, ip=ip)


@overload
def make_composed_layout(inner, offset, outer, loc=None, ip=None): ...
@overload
def make_composed_layout(inner, outer, loc=None, ip=None): ...


@traced_op
def make_composed_layout(inner, offset_or_outer, outer=None, loc=None, ip=None):
    if outer is None:
        outer = offset_or_outer
        offset = make_int_tuple(0, loc=loc, ip=ip)
    else:
        offset = offset_or_outer
    return fly.make_composed_layout(inner, offset, outer, loc=loc, ip=ip)


@traced_op
def make_identity_layout(shape, loc=None, ip=None):
    return fly.make_identity_layout(shape, loc=loc, ip=ip)


@traced_op
def make_view(iter, layout, loc=None, ip=None):
    return fly.make_view(iter, layout, loc=loc, ip=ip)


@traced_op
def make_fragment_like(tensor, dtype=None, loc=None, ip=None):
    return fly.make_fragment_like(tensor, dtype=dtype, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Extractors
# ===----------------------------------------------------------------------=== #


@traced_op
def get_scalar(int_tuple, loc=None, ip=None):
    return fly.get_scalar(int_tuple, loc=loc, ip=ip)


@traced_op
def get_leaves(input, loc=None, ip=None):
    return fly.get_leaves(input, loc=loc, ip=ip)


@traced_op
def get_shape(layout, loc=None, ip=None):
    return fly.get_shape(layout, loc=loc, ip=ip)


@traced_op
def get_stride(layout, loc=None, ip=None):
    return fly.get_stride(layout, loc=loc, ip=ip)


@traced_op
def get_layout(memref, loc=None, ip=None):
    return fly.get_layout(memref, loc=loc, ip=ip)


@traced_op
def get_iter(memref, loc=None, ip=None):
    return fly.get_iter(memref, loc=loc, ip=ip)


@traced_op
def composed_get_inner(input, loc=None, ip=None):
    return fly.composed_get_inner(input, loc=loc, ip=ip)


@traced_op
def composed_get_offset(input, loc=None, ip=None):
    return fly.composed_get_offset(input, loc=loc, ip=ip)


@traced_op
def composed_get_outer(input, loc=None, ip=None):
    return fly.composed_get_outer(input, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# IntTuple operations
# ===----------------------------------------------------------------------=== #


@traced_op
def int_tuple_add(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_add(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_sub(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_sub(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_mul(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_mul(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_div(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_div(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_mod(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_mod(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_product(int_tuple, loc=None, ip=None):
    return fly.int_tuple_product(int_tuple, loc=loc, ip=ip)


@traced_op
def int_tuple_product_each(int_tuple, loc=None, ip=None):
    return fly.int_tuple_product_each(int_tuple, loc=loc, ip=ip)


@traced_op
def int_tuple_product_like(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_product_like(lhs, rhs, loc=loc, ip=ip)


@traced_op
def shape_div(lhs, rhs, loc=None, ip=None):
    return fly.shape_div(lhs, rhs, loc=loc, ip=ip)


@traced_op
def ceil_div(lhs, rhs, loc=None, ip=None):
    return fly.ceil_div(lhs, rhs, loc=loc, ip=ip)


@traced_op
def elem_less(lhs, rhs, loc=None, ip=None):
    return fly.elem_less(lhs, rhs, loc=loc, ip=ip)


@traced_op
def equal(lhs, rhs, loc=None, ip=None):
    return fly.equal(lhs, rhs, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# IntTupleLike operations
# ===----------------------------------------------------------------------=== #


@traced_op
def get(int_tuple, mode, loc=None, ip=None):
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    selected = fly.select(int_tuple, indices=[mode], loc=loc, ip=ip)
    result = fly.get_scalar(selected, loc=loc, ip=ip)
    if isinstance(result, ir.Value) and not isinstance(result.type, ir.IndexType):
        result = _arith.IndexCastOp(T.index(), result).result
    return result


@traced_op
def get_(int_tuple, mode, loc=None, ip=None):
    if isinstance(mode, int):
        mode = [mode]
    return fly.get(int_tuple, mode, loc=loc, ip=ip)


@traced_op
def take(int_tuple, begin: int, end: int, loc=None, ip=None):
    return fly.take(int_tuple, begin=begin, end=end, loc=loc, ip=ip)


@traced_op
def select(int_tuple, indices, loc=None, ip=None):
    return fly.select(int_tuple, indices=indices, loc=loc, ip=ip)


@traced_op
def group(int_tuple, begin: int, end: int, loc=None, ip=None):
    return fly.group(int_tuple, begin=begin, end=end, loc=loc, ip=ip)


@traced_op
def append(base, elem, n: int | None = None, loc=None, ip=None):
    return fly.append(base, elem, n=n, loc=loc, ip=ip)


@traced_op
def prepend(base, elem, n: int | None = None, loc=None, ip=None):
    return fly.prepend(base, elem, n=n, loc=loc, ip=ip)


@traced_op
def slice(src, coord, loc=None, ip=None):
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = fly.infer_int_tuple_type(coord)
        coord = fly.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return fly.slice(src, coord, loc=loc, ip=ip)


@traced_op
def dice(src, coord, loc=None, ip=None):
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = fly.infer_int_tuple_type(coord)
        coord = fly.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return fly.dice(src, coord, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# LayoutLike operations
# ===----------------------------------------------------------------------=== #


@traced_op
def size(int_tuple, loc=None, ip=None):
    return fly.size(int_tuple, loc=loc, ip=ip)


@traced_op
def coprofile(layout, loc=None, ip=None):
    return fly.coprofile(layout, loc=loc, ip=ip)


@traced_op
def coshape(layout, loc=None, ip=None):
    return fly.coshape(layout, loc=loc, ip=ip)


@traced_op
def cosize(layout, loc=None, ip=None):
    return fly.cosize(layout, loc=loc, ip=ip)


def _to_i32(v):
    """Cast index-type ir.Value to i32 (required by fly.make_int_tuple)."""
    if isinstance(v, ir.Value) and isinstance(v.type, ir.IndexType):
        return _arith.IndexCastOp(T.i32(), v).result
    return v


@traced_op
def crd2idx(crd, layout, loc=None, ip=None):
    if not isinstance(crd, ir.Value):
        if isinstance(crd, (list, tuple)):
            crd = tuple(_to_i32(c) for c in crd)
        crdTy, dyncElems = fly.infer_int_tuple_type(crd)
        crd = fly.make_coord(crdTy, dyncElems, loc=loc, ip=ip)
    return fly.crd2idx(crd, layout, loc=loc, ip=ip)


@traced_op
def idx2crd(index, layout, loc=None, ip=None):
    if isinstance(index, ir.Value) and not str(index.type).startswith("!fly.int_tuple"):
        index = _to_i32(index)
        IntTupleTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(index, ir.Value):
        indexTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(indexTy, dyncElems, loc=loc, ip=ip)
    return fly.idx2crd(index, layout, loc=loc, ip=ip)


@traced_op
def get_flat_coord(index, layout, loc=None, ip=None):
    if not isinstance(index, ir.Value):
        indexTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(indexTy, dyncElems, loc=loc, ip=ip)
    return fly.get_flat_coord(index, layout, loc=loc, ip=ip)


@traced_op
def get_1d_coord(index, layout, loc=None, ip=None):
    if not isinstance(index, ir.Value):
        indexTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(indexTy, dyncElems, loc=loc, ip=ip)
    return fly.get_1d_coord(index, layout, loc=loc, ip=ip)


@traced_op
def coalesce(layout, pattern=None, loc=None, ip=None):
    return fly.coalesce(layout, pattern=pattern, loc=loc, ip=ip)


@traced_op
def composition(layout, tiler, loc=None, ip=None):
    return fly.composition(layout, tiler, loc=loc, ip=ip)


@traced_op
def complement(layout, codomain_size=None, loc=None, ip=None):
    if codomain_size is not None and not isinstance(codomain_size, ir.Value):
        codomain_sizeTy, dyncElems = fly.infer_int_tuple_type(codomain_size)
        codomain_size = fly.make_shape(codomain_sizeTy, dyncElems, loc=loc, ip=ip)
    return fly.complement(layout, codomain_size=codomain_size, loc=loc, ip=ip)


@traced_op
def right_inverse(layout, loc=None, ip=None):
    return fly.right_inverse(layout, loc=loc, ip=ip)


@traced_op
def left_inverse(layout, loc=None, ip=None):
    return fly.left_inverse(layout, loc=loc, ip=ip)


@traced_op
def logical_divide(layout, divisor, loc=None, ip=None):
    return fly.logical_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def zipped_divide(layout, divisor, loc=None, ip=None):
    return fly.zipped_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def tiled_divide(layout, divisor, loc=None, ip=None):
    return fly.tiled_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def flat_divide(layout, divisor, loc=None, ip=None):
    return fly.flat_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def logical_product(layout, tiler, loc=None, ip=None):
    return fly.logical_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def zipped_product(layout, tiler, loc=None, ip=None):
    return fly.zipped_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def tiled_product(layout, tiler, loc=None, ip=None):
    return fly.tiled_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def flat_product(layout, tiler, loc=None, ip=None):
    return fly.flat_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def block_product(layout, tiler, loc=None, ip=None):
    return fly.block_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def raked_product(layout, tiler, loc=None, ip=None):
    return fly.raked_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def recast_layout(layout, old_type_bits, new_type_bits, loc=None, ip=None):
    def _to_static_bits(v):
        if isinstance(v, int):
            return v
        if isinstance(v, ir.Type):
            if hasattr(v, "width"):
                return int(v.width)
            raise TypeError(f"recast_layout only supports int/type-with-width, got type {v}")
        raise TypeError(f"recast_layout only supports int/Type, got {type(v)}")

    old_type_bits = _to_static_bits(old_type_bits)
    new_type_bits = _to_static_bits(new_type_bits)
    return fly.recast_layout(new_type_bits=new_type_bits, old_type_bits=old_type_bits, src=layout, loc=loc, ip=ip)


@traced_op
def tile_to_shape(block, trg_shape, ord_shape, loc=None, ip=None):
    return fly.tile_to_shape(block, trg_shape, ord_shape, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Atom and Tiled Mma/Copy ops
# ===----------------------------------------------------------------------=== #


@traced_op
def make_mma_atom(mma_op_type, loc=None, ip=None):
    mma_atom_ty = MmaAtomType.get(mma_op=mma_op_type)
    return fly.make_mma_atom(mma_atom_ty, loc=loc, ip=ip)


@traced_op
def make_copy_atom(copy_op_type, elem_type, loc=None, ip=None):
    from .numeric import NumericMeta

    if isinstance(elem_type, NumericMeta):
        val_bits = elem_type.width
    elif isinstance(elem_type, ir.Type):
        if hasattr(elem_type, "width"):
            val_bits = int(elem_type.width)
        else:
            raise TypeError(f"make_copy_atom: elem_type must have a width, got {elem_type}")
    elif isinstance(elem_type, int):
        val_bits = elem_type
    else:
        raise TypeError(f"make_copy_atom: elem_type must be NumericType, ir.Type, or int, got {type(elem_type)}")
    copy_atom_ty = CopyAtomType.get(copy_op=copy_op_type, val_bits=val_bits)
    return fly.make_copy_atom(copy_atom_ty, val_bits=val_bits, loc=loc, ip=ip)


@traced_op
def copy_atom_call(copy_atom, src, dst, *, pred=None, loc=None, ip=None):
    return fly.copy_atom_call(copy_atom, src, dst, pred=pred, loc=loc, ip=ip)


@traced_op
def mma_atom_call(mma_atom, d, a, b, c, loc=None, ip=None):
    return fly.mma_atom_call(mma_atom, d, a, b, c, loc=loc, ip=ip)


@traced_op
def make_tiled_copy(copy_atom, layout_thr_val, tile_mn, loc=None, ip=None):
    return fly.make_tiled_copy(copy_atom, layout_thr_val, tile_mn, loc=loc, ip=ip)


@traced_op
def make_tiled_mma(mma_atom, atom_layout, permutation=None, loc=None, ip=None):
    return fly.make_tiled_mma(mma_atom, atom_layout, permutation=permutation, loc=loc, ip=ip)


@traced_op
def tiled_copy_partition_src(tiled_copy, src, thr_int_tuple, loc=None, ip=None):
    return fly.tiled_copy_partition_src(tiled_copy, src, thr_int_tuple, loc=loc, ip=ip)


@traced_op
def tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple, loc=None, ip=None):
    return fly.tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple, loc=loc, ip=ip)


@traced_op
def tiled_copy_retile(tiled_copy, t, loc=None, ip=None):
    return fly.tiled_copy_retile(tiled_copy, t, loc=loc, ip=ip)


@traced_op
def tiled_mma_partition(operand_id, tiled_mma, t, coord, loc=None, ip=None):
    return fly.tiled_mma_partition(operand_id, tiled_mma, t, coord, loc=loc, ip=ip)


@traced_op
def tiled_mma_partition_shape(operand_id, tiled_mma, shape, loc=None, ip=None):
    return fly.tiled_mma_partition_shape(operand_id, tiled_mma, shape, loc=loc, ip=ip)


@traced_op
def mma_make_fragment(operand_id, tiled_mma, input, loc=None, ip=None):
    return fly.mma_make_fragment(operand_id, tiled_mma, input, loc=loc, ip=ip)


@traced_op
def copy(copy_atom, src, dst, *, pred=None, loc=None, ip=None):
    return fly.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip)


@traced_op
def gemm(mma_atom, d, a, b, c, *, traversal_order=None, traversal_layout=None, loc=None, ip=None):
    if traversal_order is not None and traversal_layout is not None:
        raise ValueError("Only one of 'traversal_order' or 'traversal_layout' can be specified, not both")
    return fly.gemm(
        mma_atom, d, a, b, c, traversal_order=traversal_order, traversal_layout=traversal_layout, loc=loc, ip=ip
    )


# ===----------------------------------------------------------------------=== #
# MemRef and Ptr operations
# ===----------------------------------------------------------------------=== #


@traced_op
def make_ptr(result_type, args, loc=None, ip=None):
    return fly.make_ptr(result_type, args, loc=loc, ip=ip)


@traced_op
def get_dyn_shared(loc=None, ip=None):
    return fly.get_dyn_shared(loc=loc, ip=ip)


@traced_op
def inttoptr(result_type, src, loc=None, ip=None):
    return fly.inttoptr(result_type, src, loc=loc, ip=ip)


@traced_op
def ptrtoint(ptr, loc=None, ip=None):
    return fly.ptrtoint(ptr, loc=loc, ip=ip)


@traced_op
def add_offset(ptr, offset, loc=None, ip=None):
    if not isinstance(offset, ir.Value):
        offset = make_int_tuple(offset, loc=loc, ip=ip)
    return fly.add_offset(ptr, offset, loc=loc, ip=ip)


@traced_op
def apply_swizzle(ptr, swizzle, loc=None, ip=None):
    return fly.apply_swizzle(ptr, swizzle, loc=loc, ip=ip)


@traced_op
def ptr_load(ptr, result_type=None, loc=None, ip=None):
    if result_type is None:
        result_type = ptr.element_type
    return fly.ptr_load(result_type.ir_type, ptr, loc=loc, ip=ip)


@traced_op
def ptr_store(value, ptr, loc=None, ip=None):
    return fly.ptr_store(value, ptr, loc=loc, ip=ip)


@traced_op
def recast_iter(result_type, src, loc=None, ip=None):
    return fly.recast_iter(result_type, src, loc=loc, ip=ip)


@traced_op
def memref_alloca(memref_type, layout, loc=None, ip=None):
    return fly.memref_alloca(memref_type, layout, loc=loc, ip=ip)


@traced_op
def memref_load_vec(memref, loc=None, ip=None):
    return fly.memref_load_vec(memref, loc=loc, ip=ip)


@traced_op
def memref_store_vec(vector, memref, loc=None, ip=None):
    return fly.memref_store_vec(vector, memref, loc=loc, ip=ip)


@traced_op
def memref_load(memref, indices, loc=None, ip=None):
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return fly.memref_load(memref, indices, loc=loc, ip=ip)
        if str(indices.type) == "index":
            indices = _arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return fly.memref_load(memref, indices, loc=loc, ip=ip)

    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return fly.memref_load(memref, indices, loc=loc, ip=ip)


@traced_op
def memref_store(value, memref, indices, loc=None, ip=None):
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return fly.memref_store(value, memref, indices, loc=loc, ip=ip)
        if str(indices.type) == "index":
            indices = _arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return fly.memref_store(value, memref, indices, loc=loc, ip=ip)

    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return fly.memref_store(value, memref, indices, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Utility ops
# ===----------------------------------------------------------------------=== #


@traced_op
def printf(*args, format_str="", loc=None, ip=None):
    def _convert_printf_value(val):
        if isinstance(val, ir.Value):
            return (False, val)
        elif isinstance(val, type):
            return (True, val.__name__)
        elif isinstance(val, str):
            return (True, val)
        elif isinstance(val, bool):
            return (False, _arith.constant(T.bool(), int(val)))
        elif isinstance(val, int):
            return (False, _arith.constant(T.i32(), val))
        elif isinstance(val, float):
            return (False, _arith.constant(T.f64(), val))
        elif hasattr(val, "__fly_values__"):
            ir_values = val.__fly_values__()
            if len(ir_values) == 1:
                return (False, ir_values[0])
            raise ValueError(f"Cannot use multi-value type in printf: {type(val)}")
        elif hasattr(val, "value") and isinstance(val.value, ir.Value):
            return (False, val.value)
        else:
            raise ValueError(f"Cannot convert {type(val)} to MLIR Value for printf")

    if len(args) > 0 and isinstance(args[0], str):
        format_str = args[0]
        raw_values = list(args[1:])
    else:
        raw_values = list(args)

    converted = [_convert_printf_value(v) for v in raw_values]

    final_format = format_str
    ir_values = []
    placeholder_idx = 0
    result_parts = []
    i = 0
    while i < len(final_format):
        if i + 1 < len(final_format) and final_format[i : i + 2] == "{}":
            if placeholder_idx < len(converted):
                is_static, val = converted[placeholder_idx]
                if is_static:
                    result_parts.append(str(val))
                else:
                    result_parts.append("{}")
                    ir_values.append(val)
                placeholder_idx += 1
            else:
                result_parts.append("{}")
            i += 2
        else:
            result_parts.append(final_format[i])
            i += 1

    final_format = "".join(result_parts)
    return fly.print_(final_format, ir_values, loc=loc, ip=ip)


@traced_op
def assume(result_type, dst, src, loc=None, ip=None):
    return fly.assume(result_type, dst, src, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Deprecated
# ===----------------------------------------------------------------------=== #


@traced_op
def make_tile(*args, loc=None, ip=None):
    from .typing import Layout

    def _resolve(m):
        if isinstance(m, int) or m is None:
            return m
        if isinstance(m, tuple):
            return tuple(_resolve(e) for e in m)
        if isinstance(m, Layout):
            return m.type
        raise ValueError(f"make_tile: expected int, None, tuple, or Layout, got {type(m)}")

    resolved = [_resolve(m) for m in args]
    if len(resolved) == 1:
        tile_type = TileType.get(resolved[0])
    else:
        tile_type = TileType.get(resolved)
    return static(tile_type, loc=loc, ip=ip)
