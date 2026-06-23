# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect
from enum import IntEnum
from functools import wraps
from typing import overload

from .._mlir import ir
from .._mlir.dialects import arith as _arith
from .._mlir.dialects import fly
from .._mlir.dialects.fly import (
    AddressSpace,
    AtomicOp,
    CachePolicy,
    ComposedLayoutType,
    CoordSwizzleType,
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
from .meta import dsl_loc_tracing, dsl_wrap_result

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
    "CoordSwizzleType",
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
    "make_fragment_layout_like",
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
    "blocked_product",
    "raked_product",
    "recast_layout",
    "tile_to_shape",
    "make_mma_atom",
    "make_copy_atom",
    "atom_set_value",
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

UniversalAtomic = lambda atomic_op, val_type: CopyOpUniversalAtomicType.get(int(atomic_op), val_type.ir_type)
UniversalAtomicAdd = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Add), val_type.ir_type)
UniversalAtomicMax = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Max), val_type.ir_type)
UniversalAtomicMin = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Min), val_type.ir_type)
UniversalAtomicAnd = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.And), val_type.ir_type)
UniversalAtomicOr = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Or), val_type.ir_type)
UniversalAtomicInc = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Inc), val_type.ir_type)
UniversalAtomicDec = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Dec), val_type.ir_type)

UniversalFMA = lambda ty: MmaOpUniversalFMAType.get(ty.ir_type)


# ===----------------------------------------------------------------------=== #
# Internal
# ===----------------------------------------------------------------------=== #


def _is_int_tuple_value(value):
    return isinstance(value, ir.Value) and isinstance(value.type, IntTupleType)


def _expand_int_tuple_leaves(value):
    from .numeric import Int32, Int64, Numeric

    if _is_int_tuple_value(value):
        return _expand_int_tuple_leaves(value.to_py_value())
    if isinstance(value, (list, tuple)):
        return tuple(_expand_int_tuple_leaves(v) for v in value)
    # widen narrow dynamic ints to i32
    if isinstance(value, Numeric):
        if isinstance(value.value, ir.Value) and type(value).width < 32:
            return Int32(value).value
        return value.value
    if isinstance(value, ir.Value) and isinstance(value.type, ir.IntegerType) and value.type.width < 32:
        return Int32(value).value
    if isinstance(value, ir.Value) and isinstance(value.type, ir.IndexType):
        return Int64(value).value
    return value


def _infer_int_tuple_type(value):
    return fly.infer_int_tuple_type(_expand_int_tuple_leaves(value))


def _infer_variadic_int_tuple_type(values):
    if len(values) == 1 and _is_int_tuple_value(values[0]):
        values = values[0]
    return _infer_int_tuple_type(values)


is_profile_congruent = fly.is_profile_congruent
is_profile_weakly_congruent = fly.is_profile_weakly_congruent


def _check_profile(match_func, lhs, rhs):
    if not match_func(lhs, rhs):
        raise ValueError(f"profile mismatch: {match_func.__name__}({lhs.type}, {rhs.type}) is False")


# ---- IntTuple covariance ----
# Covariance rules (Python value → fly.IntTuple):
#   int             <: fly.IntTuple<int>           (leaf)
#   Numeric         <: fly.IntTuple<Numeric>       (leaf, e.g. Int32(5))
#   tuple(X1, ...)  <: fly.IntTuple<(X1, ...)>     (non-leaf; tuple is constructor)
#   fly.IntTuple    <: fly.IntTuple                (trivial)


def _coerce_int_tuple(v):
    if _is_int_tuple_value(v):
        return v
    return make_int_tuple(v)


def _coerce_int_tuple_permissive(v):
    if isinstance(v, ir.Value):
        return v
    return make_int_tuple(v)


def coerce_int_tuple_args(*arg_names, permissive=False):
    coerce = _coerce_int_tuple_permissive if permissive else _coerce_int_tuple

    def decorator(fn):
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            for name in arg_names:
                v = bound.arguments.get(name)
                if v is None:
                    continue
                bound.arguments[name] = coerce(v)
            return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


# ===----------------------------------------------------------------------=== #
# Compile-time utility
# ===----------------------------------------------------------------------=== #


def const_expr(x):
    return x


def range_constexpr(*args):
    return range(*args)


def rank(int_or_tuple):
    """Number of top-level elements of a tuple / layout.

    A leaf integer has rank 1; each child of a nested tuple counts as one mode.

    Examples:
        rank(8)              -> 1
        rank((8, 16))        -> 2
        rank((8, (4, 2)))    -> 2   (the nested (4, 2) still counts as one mode)
    """
    if isinstance(int_or_tuple, int):
        return 1
    if isinstance(int_or_tuple, tuple):
        return len(int_or_tuple)
    return fly.rank(int_or_tuple)


def depth(int_or_tuple):
    """How deeply the tuple is nested.

    A leaf integer has depth 0; a flat tuple has depth 1; each extra level of
    nesting adds one.

    Examples:
        depth(8)             -> 0
        depth((8, 16))       -> 1
        depth((8, (4, 2)))   -> 2
    """
    if isinstance(int_or_tuple, int):
        return 0
    if isinstance(int_or_tuple, tuple):
        return 1 + max((depth(c) for c in int_or_tuple), default=0)
    return fly.depth(int_or_tuple)


# ===----------------------------------------------------------------------=== #
# Constructors
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
def static(result_type):
    """Materialize a value whose entire content is encoded in *result_type*.

    Used for fully known compile-time objects: static tuples, tiles, swizzles, layout, etc.
    All information lives in the type, so no runtime operands are needed.

    Examples:
        static(IntTupleType.get((4, 8)))          -> a static (4, 8) tuple
        static(SwizzleType.get(3, 3, 3))          -> a static swizzle descriptor
    """
    return fly.static(result_type)


@dsl_loc_tracing
def make_int_tuple(elems):
    """Build a (possibly nested) integer tuple from Python ints or runtime values.

    Integers become static entries; `ir.Value` operands become dynamic entries.

    Examples:
        make_int_tuple((4, 8))           -> static tuple (4, 8)
        make_int_tuple((m, 8))           -> (m, 8) where m is a runtime int
    """
    IntTupleTy, dyncElems = _infer_int_tuple_type(elems)
    return fly.make_int_tuple(IntTupleTy, dyncElems)


@dsl_loc_tracing
def make_shape(*shape):
    """Build a shape tuple describing the extent of each mode.

    Supports nested shapes for hierarchical tiling.

    Examples:
        make_shape(8, 16)          -> (8, 16)
        make_shape(9, (4, 8))      -> (9, (4, 8))  (second mode is sub-structured)
    """
    IntTupleTy, dyncElems = _infer_variadic_int_tuple_type(shape)
    return fly.make_shape(IntTupleTy, dyncElems)


@dsl_loc_tracing
def make_stride(*stride):
    """Build a stride tuple: the step (in elements) when moving along each mode.

    Nested structure must mirror the shape it will be paired with.

    Examples:
        make_stride(1, 8)                  -> column-major stride for (8, 16)
        make_stride(16, 1)                 -> row-major stride for (8, 16)
    """
    IntTupleTy, dyncElems = _infer_variadic_int_tuple_type(stride)
    return fly.make_stride(IntTupleTy, dyncElems)


@dsl_loc_tracing
def make_coord(*coord):
    """Build a coordinate used for indexing / slicing a layout.

    Use `None` in a mode to mean "all positions of that mode" (a free axis).

    Examples:
        make_coord(3, 5)           -> point coordinate (row 3, col 5)
        make_coord(None, bid)      -> (:, bid)  keep first axis free, pick second
    """
    IntTupleTy, dyncElems = _infer_variadic_int_tuple_type(coord)
    return fly.make_coord(IntTupleTy, dyncElems)


@dsl_loc_tracing
def make_layout(shape, stride):
    """Pair a *shape* with a *stride* to describe how logical coords map to memory.

    Accepts Python tuples directly (auto-converted). The mapping is:
    `index = sum(coord_i * stride_i)`.

    Examples:
        make_layout((4, 8), (1, 4))      -> ((4, 8), (1, 4))
        make_layout((4, 8), (8, 1))      -> ((4, 8), (8, 1))
    """
    if not _is_int_tuple_value(shape):
        shape = make_int_tuple(shape)
    if not _is_int_tuple_value(stride):
        stride = make_int_tuple(stride)
    _check_profile(is_profile_congruent, shape, stride)
    return fly.make_layout(shape, stride=stride)


@dsl_loc_tracing
def make_layout_like(ref):
    return fly.make_layout_like(ref)


@dsl_loc_tracing
def make_ordered_layout(shape, order):
    """Build a compact layout whose stride order matches *order*.

    `order[i]` says where mode *i* sits when ranking strides from fastest
    (smallest value) to slowest. Lower means more contiguous.

    Examples:
        make_ordered_layout((M, N), (0, 1))  # column-major: M iterates fastest
        make_ordered_layout((M, N), (1, 0))  # row-major:    N iterates fastest
    """
    if not _is_int_tuple_value(shape):
        shape = make_int_tuple(shape)
    if not _is_int_tuple_value(order):
        order = make_int_tuple(order)
    _check_profile(is_profile_weakly_congruent, order, shape)
    return fly.make_ordered_layout(shape, order)


@overload
def make_composed_layout(inner, offset, outer): ...
@overload
def make_composed_layout(inner, outer): ...
@dsl_loc_tracing
def make_composed_layout(inner, offset_or_outer, outer=None):
    """Stack two layouts: a coord is first mapped by *outer*, then by *inner*.

    An optional constant *offset* is added after the outer mapping. The outer
    mapping may itself be a composed layout, allowing composition chains.

    Examples:
        make_composed_layout(swizzle, layout)           # no offset
        make_composed_layout(swizzle, 16, layout)       # with offset = 16
    """
    if outer is None:
        outer = offset_or_outer
        offset = coprofile(outer)
    else:
        offset = offset_or_outer
        if not _is_int_tuple_value(offset):
            offset = make_int_tuple(offset)
    return fly.make_composed_layout(inner, offset, outer)


@dsl_loc_tracing
def make_identity_layout(shape):
    """Build the identity layout in FlyDSL's layout-algebra sense.

    The result keeps *shape* and uses basis-tuple strides derived from that
    shape's profile (e.g. `(4, 8) -> (1E0, 1E1)`), so coordinates stay symbolic
    instead of being collapsed to one flat linear address.

    Examples:
        make_identity_layout((4, 8))   -> ((4, 8), (1E0, 1E1))
    """
    if not _is_int_tuple_value(shape):
        shape = make_int_tuple(shape)
    return fly.make_identity_layout(shape)


@dsl_loc_tracing
def make_view(iter, layout):
    return fly.make_view(iter, layout)


@dsl_loc_tracing
def make_fragment_layout_like(tensor):
    return fly.make_fragment_layout_like(tensor)


@dsl_loc_tracing
def make_fragment_like(tensor, dtype=None):
    if hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    return fly.make_fragment_like(tensor, dtype=dtype)


# ===----------------------------------------------------------------------=== #
# Extractors
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
@dsl_wrap_result
def get_scalar(int_tuple):
    """Unwrap a rank-1, single-element tuple back to a plain scalar value.

    Fails if the input has more than one leaf - use this only when you know
    the tuple is a trivial wrapper.

    Examples:
        get_scalar(make_coord(tid)) -> Int32(tid)
        get_scalar(make_int_tuple(5)) -> 5
    """
    if not _is_int_tuple_value(int_tuple):
        return int_tuple
    if int_tuple.is_leaf and int_tuple.is_static:
        return int_tuple.get_static_leaf_int
    return fly.get_scalar(int_tuple)


@dsl_loc_tracing
@dsl_wrap_result
def get_leaves(input, dynamic_only=False):
    """Flatten an IntTuple into a flat sequence of leaf values.

    Set *dynamic_only=True* to keep only runtime values and drop static
    constants - handy when you need the inputs that were passed at call time.

    Examples:
        get_leaves(make_coord(tid, 0)) -> (Int32(tid), 0)
        get_leaves(make_coord(tid, 0), dynamic_only=True) -> (Int32(tid),) # 0 is static, dropped
    """
    if dynamic_only:
        res_lists = fly.GetLeavesOp(input, dynamicOnly=True)
        return tuple(res_lists.results)

    def _walk_int_tuple_leaves(ty):
        if ty.is_leaf:
            yield ty
            return
        for i in range(ty.rank):
            yield from _walk_int_tuple_leaves(ty.at(i))

    ty = IntTupleType(input.type)
    res_lists = fly.GetLeavesOp(input, dynamicOnly=True)
    dyn_iter = iter(res_lists.results)
    out = []
    for leaf_ty in _walk_int_tuple_leaves(ty):
        if leaf_ty.is_static:
            out.append(leaf_ty.get_static_leaf_int)
        else:
            out.append(next(dyn_iter))
    return tuple(out)


@dsl_loc_tracing
def get_shape(layout):
    return fly.get_shape(layout)


@dsl_loc_tracing
def get_stride(layout):
    return fly.get_stride(layout)


@dsl_loc_tracing
def get_layout(memref):
    return fly.get_layout(memref)


@dsl_loc_tracing
def get_iter(memref):
    return fly.get_iter(memref)


@dsl_loc_tracing
def composed_get_inner(input):
    return fly.composed_get_inner(input)


@dsl_loc_tracing
def composed_get_offset(input):
    return fly.composed_get_offset(input)


@dsl_loc_tracing
def composed_get_outer(input):
    return fly.composed_get_outer(input)


# ===----------------------------------------------------------------------=== #
# IntTuple operations
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def int_tuple_add(lhs, rhs):
    return fly.int_tuple_add(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def int_tuple_sub(lhs, rhs):
    return fly.int_tuple_sub(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def int_tuple_mul(lhs, rhs):
    return fly.int_tuple_mul(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def int_tuple_div(lhs, rhs):
    return fly.int_tuple_div(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def int_tuple_mod(lhs, rhs):
    return fly.int_tuple_mod(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple")
def int_tuple_product(int_tuple):
    return fly.int_tuple_product(int_tuple)


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple")
def int_tuple_product_each(int_tuple):
    return fly.int_tuple_product_each(int_tuple)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def int_tuple_product_like(lhs, rhs):
    return fly.int_tuple_product_like(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def shape_div(lhs, rhs):
    return fly.shape_div(lhs, rhs)


@dsl_loc_tracing
@coerce_int_tuple_args("lhs", "rhs")
def ceil_div(lhs, rhs):
    return fly.ceil_div(lhs, rhs)


@dsl_loc_tracing
@dsl_wrap_result
@coerce_int_tuple_args("lhs", "rhs")
def elem_less(lhs, rhs):
    return fly.elem_less(lhs, rhs)


@dsl_loc_tracing
@dsl_wrap_result
@coerce_int_tuple_args("lhs", "rhs")
def equal(lhs, rhs):
    return fly.equal(lhs, rhs)


# ===----------------------------------------------------------------------=== #
# IntTupleLike operations
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
def get(int_tuple, mode):
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    selected = fly.select(int_tuple, indices=[mode])
    result = fly.get_scalar(selected)
    if isinstance(result, ir.Value) and not isinstance(result.type, ir.IndexType):
        result = _arith.IndexCastOp(T.index(), result).result
    return result


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple", permissive=True)
def get_(int_tuple, mode):
    if isinstance(mode, int):
        mode = [mode]
    return fly.get(int_tuple, mode)


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple", permissive=True)
def take(int_tuple, begin: int, end: int):
    return fly.take(int_tuple, begin=begin, end=end)


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple", permissive=True)
def select(int_tuple, indices):
    return fly.select(int_tuple, indices=indices)


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple", permissive=True)
def group(int_tuple, begin: int, end: int):
    return fly.group(int_tuple, begin=begin, end=end)


@dsl_loc_tracing
@coerce_int_tuple_args("base", "elem", permissive=True)
def append(base, elem, *, n: int | None = None):
    return fly.append(base, elem, n=n)


@dsl_loc_tracing
@coerce_int_tuple_args("base", "elem", permissive=True)
def prepend(base, elem, *, n: int | None = None):
    return fly.prepend(base, elem, n=n)


@dsl_loc_tracing
def slice(src, coord):
    """Keep the modes where *coord* has `None` (wildcard), drop the rest.

    A None in coord means "all of this axis"; a fixed integer picks that index
    and the mode disappears from the result.

    Examples:
        slice((4, 8, 16), (None, 3, None))   -> (4, 16)   # mode 1 fixed, dropped
        slice(layout, make_coord(None, bid)) -> sub-layout for column `bid`
    """
    if not _is_int_tuple_value(coord):
        coord = make_int_tuple(coord)
    _check_profile(is_profile_weakly_congruent, coord, src)
    return fly.slice(src, coord)


@dsl_loc_tracing
def dice(src, coord):
    """Complement of `slice`: keep the *fixed* modes, drop the `None` (wildcard) ones.

    Useful for extracting the per-tile / per-thread coordinate from a partitioned layout.

    Examples:
        dice((4, 8, 16), (None, 3, None))    -> (8,)
        dice(coord_tensor, make_coord(tid, None)) -> the thread-only part
    """
    if not _is_int_tuple_value(coord):
        coord = make_int_tuple(coord)
    _check_profile(is_profile_weakly_congruent, coord, src)
    return fly.dice(src, coord)


# ===----------------------------------------------------------------------=== #
# LayoutLike operations
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
@coerce_int_tuple_args("int_tuple", permissive=True)
def size(int_tuple):
    return fly.size(int_tuple)


@dsl_loc_tracing
def coprofile(layout):
    return fly.coprofile(layout)


@dsl_loc_tracing
def coshape(layout):
    return fly.coshape(layout)


@dsl_loc_tracing
def cosize(layout):
    return fly.cosize(layout)


@dsl_loc_tracing
def crd2idx(crd, layout):
    """Map a coordinate tuple to an index through *layout*.

    For flat layouts this reduces to the familiar `sum(coord_i * stride_i)`.
    Nested / composed layouts recurse through sub-layouts, apply offsets, and may
    apply swizzles, so the general case is richer than a single multiply-add.

    Examples:
        crd2idx((1, 2), make_layout((4, 8), (1, 4)))   -> 9
        crd2idx(7, make_layout((4, 8), (1, 4)))        -> 7
    """
    if not _is_int_tuple_value(crd):
        crd = make_int_tuple(crd)
    _check_profile(is_profile_weakly_congruent, crd, layout)
    return fly.crd2idx(crd, layout)


@dsl_loc_tracing
def idx2crd(index, layout):
    """Map an index back to a coordinate tuple for a plain `Layout`.

    This is the inverse of `crd2idx` for non-composed layouts; the result keeps
    the same nested structure as the layout's shape. Composed layouts / swizzles
    are not accepted here.

    Examples:
        idx2crd(9, make_layout((4, 8), (1, 4)))        -> (1, 2)
        idx2crd(5, make_layout((4, 8), (8, 1)))        -> (0, 5)
    """
    if not _is_int_tuple_value(index):
        index = make_int_tuple(index)
    return fly.idx2crd(index, layout)


@dsl_loc_tracing
def get_flat_coord(index, layout):
    """Map an index to a *fully flattened* coordinate, ignoring nested grouping.

    Unlike `idx2crd`, the result is always a flat tuple of length `rank` of
    shape's flattened form - convenient when you want per-axis coordinates.

    Examples:
        get_flat_coord(9, make_layout((4, 8), (1, 4)))            -> (1, 2)
        get_flat_coord(3, make_layout(((2, 2), 4), ((1, 2), 4)))  -> (1, 1, 0)
    """
    if not _is_int_tuple_value(index):
        index = make_int_tuple(index)
    return fly.get_flat_coord(index, layout)


@dsl_loc_tracing
def get_1d_coord(index, layout):
    """Map an index to a single 1-D coordinate in the layout's shape space.

    Examples:
        get_1d_coord(9, make_layout((4, 8), (1, 4)))   -> 9
        get_1d_coord(5, make_layout((4, 8), (8, 1)))   -> 20
    """
    if not _is_int_tuple_value(index):
        index = make_int_tuple(index)
    return fly.get_1d_coord(index, layout)


@dsl_loc_tracing
@coerce_int_tuple_args("pattern")
def coalesce(layout, pattern=None):
    return fly.coalesce(layout, pattern=pattern)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def composition(layout, tiler):
    return fly.composition(layout, tiler)


@dsl_loc_tracing
@coerce_int_tuple_args("codomain_size")
def complement(layout, codomain_size=None):
    return fly.complement(layout, codomain_size=codomain_size)


@dsl_loc_tracing
def right_inverse(layout):
    return fly.right_inverse(layout)


@dsl_loc_tracing
def left_inverse(layout):
    return fly.left_inverse(layout)


@dsl_loc_tracing
def logical_divide(layout, divisor):
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor)
    return fly.logical_divide(layout, divisor)


@dsl_loc_tracing
def zipped_divide(layout, divisor):
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor)
    return fly.zipped_divide(layout, divisor)


@dsl_loc_tracing
def tiled_divide(layout, divisor):
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor)
    return fly.tiled_divide(layout, divisor)


@dsl_loc_tracing
def flat_divide(layout, divisor):
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor)
    return fly.flat_divide(layout, divisor)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def logical_product(layout, tiler):
    return fly.logical_product(layout, tiler)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def zipped_product(layout, tiler):
    return fly.zipped_product(layout, tiler)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def tiled_product(layout, tiler):
    return fly.tiled_product(layout, tiler)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def flat_product(layout, tiler):
    return fly.flat_product(layout, tiler)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def blocked_product(layout, tiler):
    return fly.blocked_product(layout, tiler)


@dsl_loc_tracing
@coerce_int_tuple_args("tiler", permissive=True)
def raked_product(layout, tiler):
    return fly.raked_product(layout, tiler)


@dsl_loc_tracing
def recast_layout(layout, old_type_bits, new_type_bits):
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
    return fly.recast_layout(new_type_bits=new_type_bits, old_type_bits=old_type_bits, src=layout)


@dsl_loc_tracing
@coerce_int_tuple_args("trg_shape", "ord_shape")
def tile_to_shape(block, trg_shape, ord_shape):
    return fly.tile_to_shape(block, trg_shape, ord_shape)


# ===----------------------------------------------------------------------=== #
# Atom and Tiled Mma/Copy ops
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
def make_mma_atom(mma_op_type):
    mma_atom_ty = MmaAtomType.get(mma_op=mma_op_type)
    return fly.make_mma_atom(mma_atom_ty)


@dsl_loc_tracing
def make_copy_atom(copy_op_type, elem_type):
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
    return fly.make_copy_atom(copy_atom_ty, val_bits=val_bits)


@dsl_loc_tracing
def atom_set_value(atom, field, value):
    from .typing import as_ir_value

    if isinstance(field, IntEnum):
        field = str(field)
    return fly.atom_set_value(atom, field, as_ir_value(value))


@dsl_loc_tracing
def copy_atom_call(copy_atom, src, dst, *, pred=None):
    return fly.copy_atom_call(copy_atom, src, dst, pred=pred)


@dsl_loc_tracing
def mma_atom_call(mma_atom, d, a, b, c):
    return fly.mma_atom_call(mma_atom, d, a, b, c)


@dsl_loc_tracing
def make_tiled_copy(copy_atom, layout_thr_val, tile_mn):
    if not isinstance(tile_mn, ir.Value):
        tile_mn = make_tile(*tile_mn)
    return fly.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)


@dsl_loc_tracing
def make_tiled_mma(mma_atom, atom_layout, permutation=None):
    if permutation is not None and not isinstance(permutation, ir.Value):
        permutation = make_tile(*permutation)
    return fly.make_tiled_mma(mma_atom, atom_layout, permutation=permutation)


@dsl_loc_tracing
@coerce_int_tuple_args("thr_int_tuple")
def tiled_copy_partition_src(tiled_copy, src, thr_int_tuple):
    return fly.tiled_copy_partition_src(tiled_copy, src, thr_int_tuple)


@dsl_loc_tracing
@coerce_int_tuple_args("thr_int_tuple")
def tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple):
    return fly.tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple)


@dsl_loc_tracing
def tiled_copy_retile(tiled_copy, t):
    return fly.tiled_copy_retile(tiled_copy, t)


@dsl_loc_tracing
@coerce_int_tuple_args("coord")
def tiled_mma_partition(operand_id, tiled_mma, t, coord):
    return fly.tiled_mma_partition(operand_id, tiled_mma, t, coord)


@dsl_loc_tracing
@coerce_int_tuple_args("shape")
def tiled_mma_partition_shape(operand_id, tiled_mma, shape):
    return fly.tiled_mma_partition_shape(operand_id, tiled_mma, shape)


@dsl_loc_tracing
def mma_make_fragment(operand_id, tiled_mma, input, *, stages=None):
    return fly.mma_make_fragment(operand_id, tiled_mma, input, stages=stages)


@dsl_loc_tracing
def copy(copy_atom, src, dst, *, pred=None, **kwargs):
    return fly.copy(copy_atom.set_value(kwargs), src, dst, pred=pred)


@dsl_loc_tracing
def gemm(mma_atom, d, a, b, c, *, traversal_order=None, traversal_layout=None, **kwargs):
    if traversal_order is not None and traversal_layout is not None:
        raise ValueError("Only one of 'traversal_order' or 'traversal_layout' can be specified, not both")
    return fly.gemm(
        mma_atom if (not kwargs) else mma_atom.set_value(kwargs),
        d,
        a,
        b,
        c,
        traversal_order=traversal_order,
        traversal_layout=traversal_layout,
    )


# ===----------------------------------------------------------------------=== #
# MemRef and Ptr operations
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
def make_ptr(result_type, args, *, dict_attrs=None):
    result = fly.make_ptr(result_type, args)
    if dict_attrs is not None:
        result.owner.attributes["dictAttrs"] = dict_attrs
    return result


@dsl_loc_tracing
def get_dyn_shared(dtype=None):
    """Return a pointer to the start of the kernel's dynamic shared-memory buffer.

    Examples:
        smem_base = get_dyn_shared()
        sA = make_view(recast_iter(fx.Float32, smem_base), sA_layout)
    """
    raw_ptr = fly.get_dyn_shared()
    if dtype is None:
        return raw_ptr
    return recast_iter(dtype, raw_ptr)


@dsl_loc_tracing
def inttoptr(result_type, src):
    """Interpret an integer address *src* as a pointer of *result_type*.

    Requirement: ptr.address_space != Register
    """
    from .typing import as_ir_value, is_generic_address_space

    if is_generic_address_space(result_type.address_space, AddressSpace.Register):
        raise ValueError("inttoptr is not supported for register address space")
    return fly.inttoptr(result_type, as_ir_value(src))


@dsl_loc_tracing
@dsl_wrap_result
def ptrtoint(ptr):
    """Get the raw integer address underlying *ptr*.

    Requirement: ptr.address_space != Register

    Examples:
        addr = ptrtoint(global_ptr)
    """
    from .typing import is_generic_address_space

    if is_generic_address_space(ptr.address_space, AddressSpace.Register):
        raise ValueError("ptrtoint is not supported for register address space")
    return fly.ptrtoint(ptr)


@dsl_loc_tracing
def add_offset(ptr, offset):
    """Shift *ptr* by *offset* elements

    Examples:
        ptr2 = add_offset(ptr, 16)            # move forward 16 elements
        ptr2 = add_offset(ptr, tile_id * BM)  # runtime offset
    """
    if not _is_int_tuple_value(offset):
        offset = make_int_tuple(offset)
    return fly.add_offset(ptr, offset)


@dsl_loc_tracing
def apply_swizzle(ptr, swizzle):
    return fly.apply_swizzle(ptr, swizzle)


@dsl_loc_tracing
@dsl_wrap_result
def ptr_load(ptr, result_type=None):
    """Load one value (scalar or vector) from *ptr*; dtype defaults to ptr's element type.

    Examples:
        v = ptr_load(ptr)
    """
    if result_type is None:
        result_type = ptr.element_type
    if not isinstance(result_type, ir.Type):
        result_type = result_type.ir_type
    return fly.ptr_load(result_type, ptr)


@dsl_loc_tracing
def ptr_store(value, ptr):
    """Store *value* into *ptr*. Types must match the pointer's element type.

    Examples:
        ptr_store(val, ptr)
    """
    from .numeric import Numeric

    if isinstance(value, Numeric):
        value = value.ir_value()
    elif not isinstance(value, ir.Value):
        value = ptr.element_type(value).ir_value()
    return fly.ptr_store(value, ptr)


@dsl_loc_tracing
def recast_iter(result_type, src):
    """Reinterpret a pointer / iterator as another element type (like `reinterpret_cast`).

    Examples:
        smem_f32 = recast_iter(fx.Float32, get_dyn_shared())
    """
    from .numeric import Numeric

    if isinstance(result_type, type):
        if issubclass(result_type, Numeric):
            result_type = result_type.ir_type
        else:
            raise TypeError(
                f"result_type must be a Numeric subclass or a fly Pointer, got unsupported class {result_type!r}"
            )
        result_type = PointerType.get(result_type, src.memspace, src.alignment)
    return fly.recast_iter(result_type, src)


@dsl_loc_tracing
def memref_alloca(memref_type, layout):
    return fly.memref_alloca(memref_type, layout)


@dsl_loc_tracing
def memref_load_vec(memref):
    from .typing import Vector

    return Vector(fly.memref_load_vec(memref), memref.shape.to_py_value(), memref.dtype)


@dsl_loc_tracing
def memref_store_vec(vector, memref):
    return fly.memref_store_vec(vector, memref)


@dsl_loc_tracing
@dsl_wrap_result
def memref_load(memref, indices):
    if isinstance(indices, ir.Value):
        if not _is_int_tuple_value(indices):
            indices = make_int_tuple(indices)
        return fly.memref_load(memref, indices)

    indices = make_int_tuple(indices)
    _check_profile(is_profile_weakly_congruent, indices, memref)
    return fly.memref_load(memref, indices)


@dsl_loc_tracing
def memref_store(value, memref, indices):
    from .typing import as_ir_value

    value = as_ir_value(value)
    if isinstance(indices, ir.Value):
        if not _is_int_tuple_value(indices):
            indices = make_int_tuple(indices)
        return fly.memref_store(value, memref, indices)
    indices = make_int_tuple(indices)
    _check_profile(is_profile_weakly_congruent, indices, memref)
    return fly.memref_store(value, memref, indices)


# ===----------------------------------------------------------------------=== #
# Utility ops
# ===----------------------------------------------------------------------=== #


@dsl_loc_tracing
def printf(*args, format_str=""):
    def _convert_printf_value(val):
        if isinstance(val, ir.Value):
            return (False, val)
        elif isinstance(val, type):
            return (True, val.__name__)
        elif isinstance(val, str):
            return (True, val)
        elif isinstance(val, bool):
            return (True, val)
        elif isinstance(val, int):
            return (True, val)
        elif isinstance(val, float):
            return (True, val)
        elif hasattr(val, "__extract_to_ir_values__"):
            ir_values = val.__extract_to_ir_values__()
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
    return fly.print_(final_format, ir_values)


@dsl_loc_tracing
def assume(result_type, dst, src):
    """
    WIP, unsupported for now
    """
    return fly.assume(result_type, dst, src)


@dsl_loc_tracing
def make_tile(*args):
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
    return static(tile_type)
