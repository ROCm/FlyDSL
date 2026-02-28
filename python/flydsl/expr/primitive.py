from .._mlir import ir
from .._mlir.dialects import arith as _arith, fly
from .._mlir.dialects.fly import (
    # Enum Attributes
    AddressSpace,
    CachePolicy,
    CopyAtomType,
    # Type
    CopyOpUniversalCopyType,
    IntTupleType,
    LayoutType,
    MemRefType,
    MmaAtomUniversalFMAType,
    PointerType,
    SwizzleType,
)
from .._mlir.extras import types as T
from .meta import dsl_api_wrapper
from .typing import Int32

UniversalCopy = lambda bit_size: CopyOpUniversalCopyType.get(bit_size)  # noqa: E731
UniversalCopy32b = lambda: CopyOpUniversalCopyType.get(32)  # noqa: E731
UniversalCopy64b = lambda: CopyOpUniversalCopyType.get(64)  # noqa: E731
UniversalCopy128b = lambda: CopyOpUniversalCopyType.get(128)  # noqa: E731

UniversalFMA = lambda ty: MmaAtomUniversalFMAType.get(ty.ir_type)  # noqa: E731

# __all__ = [
#     # Maybe remove it in the future
#     "T",
#     "arith",
#     # Enum Attributes
#     "AddressSpace",
#     "CachePolicy",
#     # Types
#     "CopyOpUniversalCopyType",
#     "IntTupleType",
#     "LayoutType",
#     "MemRefType",
#     "MmaAtomUniversalFMAType",
#     "PointerType",
#     "SwizzleType",
#     # DSL functions
#     "const_expr",
#     "range_constexpr",
#     "rank",
#     "depth",
#     "static",
#     "int_tuple_add",
#     "int_tuple_sub",
#     "int_tuple_mul",
#     "int_tuple_div",
#     "int_tuple_product",
#     "int_tuple_product_each",
#     "make_identity_tensor",
#     "make_identity_layout",
#     "make_shape",
#     "make_stride",
#     "make_coord",
#     "make_int_tuple",
#     "make_layout",
#     "size",
#     "get_scalar",
#     "get_shape",
#     "get_stride",
#     "slice",
#     "crd2idx",
#     "composition",
#     "complement",
#     "right_inverse",
#     "coalesce",
#     "zip",
#     "select",
#     "group",
#     "append",
#     "prepend",
#     "logical_divide",
#     "zipped_divide",
#     "tiled_divide",
#     "flat_divide",
#     "logical_product",
#     "zipped_product",
#     "tiled_product",
#     "flat_product",
#     "block_product",
#     "raked_product",
#     "make_copy_atom",
#     "make_mma_atom",
#     "make_tile",
#     "mma_atom_call",
#     "copy_atom_call",
#     "make_tiled_copy",
#     "memref_alloca",
#     "memref_load",
#     "memref_store",
#     "memref_load_vec",
#     "memref_store_vec",
#     "get_layout",
#     "get_iter",
#     "make_view",
#     "add_offset",
#     "gemm",
#     "copy",
#     "printf",
# ]


def const_expr(x):
    return x


def range_constexpr(*args):
    return range(*args)


def make_int32(value):
    return fly.make_int32(value)


def make_int32_tuple(value):
    return fly.make_int32_tuple(value)


def rank(int_or_tuple):
    return fly.rank(int_or_tuple)


def depth(int_or_tuple):
    return fly.depth(int_or_tuple)


@dsl_api_wrapper
def static(result_type, loc=None, ip=None):
    return fly.static(result_type, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_add(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_add(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_sub(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_sub(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_mul(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_mul(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_div(lhs, rhs, loc=None, ip=None):
    return fly.int_tuple_div(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_product(int_tuple, loc=None, ip=None):
    return fly.int_tuple_product(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_product_each(int_tuple, loc=None, ip=None):
    return fly.int_tuple_product_each(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def make_identity_tensor(shape, loc=None, ip=None):
    return fly.make_identity_tensor(shape, loc=loc, ip=ip)


@dsl_api_wrapper
def make_identity_layout(shape, loc=None, ip=None):
    return fly.make_identity_layout(shape, loc=loc, ip=ip)


@dsl_api_wrapper
def make_shape(*shape, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(shape)
    return fly.make_shape(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_stride(*stride, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(stride)
    return fly.make_stride(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_coord(*coord, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(coord)
    return fly.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_int_tuple(elems, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(elems)
    return fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_layout(shape, stride, loc=None, ip=None):
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(shape)
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(stride, ir.Value):
        strideTy, dyncElems = fly.infer_int_tuple_type(stride)
        stride = fly.make_stride(strideTy, dyncElems, loc=loc, ip=ip)
    return fly.make_layout(shape, stride=stride, loc=loc, ip=ip)


@dsl_api_wrapper
def make_ordered_layout(shape, order, loc=None, ip=None):
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(shape)
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(order, ir.Value):
        orderTy, dyncElems = fly.infer_int_tuple_type(order)
        order = fly.make_int_tuple(orderTy, dyncElems, loc=loc, ip=ip)
    return fly.make_ordered_layout(shape, order, loc=loc, ip=ip)


@dsl_api_wrapper
def make_fragment_like(tensor, dtype=None, loc=None, ip=None):
    return fly.make_fragment_like(tensor, dtype=dtype, loc=loc, ip=ip)


@dsl_api_wrapper
def size(int_tuple, loc=None, ip=None):
    result = fly.size(int_tuple, loc=loc, ip=ip)
    # If the int_tuple is static, return the static value
    result_ty = IntTupleType(result.type)
    if result_ty.is_leaf and result_ty.is_static:
        return Int32(result_ty.static_value)
    return result


@dsl_api_wrapper
def get_scalar(int_tuple, loc=None, ip=None):
    return fly.get_scalar(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def get_shape(layout, loc=None, ip=None):
    return fly.get_shape(layout, loc=loc, ip=ip)


@dsl_api_wrapper
def get_stride(layout, loc=None, ip=None):
    return fly.get_stride(layout, loc=loc, ip=ip)


@dsl_api_wrapper
def slice(src, coord, loc=None, ip=None):
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = fly.infer_int_tuple_type(coord)
        coord = fly.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return fly.slice(src, coord, loc=loc, ip=ip)


@dsl_api_wrapper
def get_leaf(int_tuple, leaf_idx, loc=None, ip=None):
    return fly.get_leaf(int_tuple, leaf_idx, loc=loc, ip=ip)


@dsl_api_wrapper
def get_flat_coord(index, layout, loc=None, ip=None):
    return fly.get_flat_coord(index, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def crd2idx(crd, layout, loc=None, ip=None):
    return fly.crd2idx(crd, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def idx2crd(idx, layout, loc=None, ip=None):
    if isinstance(idx, ir.Value) and not str(idx.type).startswith("!fly.int_tuple"):
        IntTupleTy, dyncElems = fly.infer_int_tuple_type((idx,))
        idx = fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)
    return fly.idx2crd(idx, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def get(int_tuple, mode, loc=None, ip=None):
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    selected = fly.select(int_tuple, indices=[mode], loc=loc, ip=ip)
    result = fly.get_scalar(selected, loc=loc, ip=ip)
    if isinstance(result, ir.Value) and not isinstance(result.type, ir.IndexType):
        result = _arith.IndexCastOp(ir.IndexType.get(), result).result
    return result


@dsl_api_wrapper
def composition(layout, tiler, loc=None, ip=None):
    return fly.composition(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def complement(layout, codomain_size, loc=None, ip=None):
    if not isinstance(codomain_size, ir.Value):
        codomain_sizeTy, dyncElems = fly.infer_int_tuple_type(codomain_size)
        codomain_size = fly.make_shape(codomain_sizeTy, dyncElems, loc=loc, ip=ip)
    return fly.complement(layout, codomain_size=codomain_size, loc=loc, ip=ip)


@dsl_api_wrapper
def right_inverse(layout, loc=None, ip=None):
    return fly.right_inverse(layout, loc=loc, ip=ip)


@dsl_api_wrapper
def coalesce(layout, pattern=None, loc=None, ip=None):
    return fly.coalesce(layout, pattern=pattern, loc=loc, ip=ip)


@dsl_api_wrapper
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


@dsl_api_wrapper
def zip(lhs, rhs, loc=None, ip=None):
    return fly.zip(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def select(int_tuple, indices, loc=None, ip=None):
    return fly.select(int_tuple, indices=indices, loc=loc, ip=ip)


@dsl_api_wrapper
def group(int_tuple, begin: int, end: int, loc=None, ip=None):
    return fly.group(int_tuple, begin=begin, end=end, loc=loc, ip=ip)


@dsl_api_wrapper
def append(base, elem, n: int | None = None, loc=None, ip=None):
    return fly.append(base, elem, n=n, loc=loc, ip=ip)


@dsl_api_wrapper
def prepend(base, elem, n: int | None = None, loc=None, ip=None):
    return fly.prepend(base, elem, n=n, loc=loc, ip=ip)


@dsl_api_wrapper
def logical_divide(layout, divisor, loc=None, ip=None):
    return fly.logical_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def zipped_divide(layout, divisor, loc=None, ip=None):
    return fly.zipped_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_divide(layout, divisor, loc=None, ip=None):
    return fly.tiled_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def flat_divide(layout, divisor, loc=None, ip=None):
    return fly.flat_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def logical_product(layout, tiler, loc=None, ip=None):
    return fly.logical_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def zipped_product(layout, tiler, loc=None, ip=None):
    return fly.zipped_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_product(layout, tiler, loc=None, ip=None):
    return fly.tiled_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def flat_product(layout, tiler, loc=None, ip=None):
    return fly.flat_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def block_product(layout, tiler, loc=None, ip=None):
    return fly.block_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def raked_product(layout, tiler, loc=None, ip=None):
    return fly.raked_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_alloca(memref_type, layout, loc=None, ip=None):
    return fly.memref_alloca(memref_type, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_load(memref, indices, loc=None, ip=None):
    # `fly.memref.load` expects `indices` as `!fly.int_tuple` (typically a scalar offset).
    # Accept convenience forms:
    # - int_tuple Value (pass through)
    # - python int / tuple/list (make_int_tuple)
    # - index/i32/i64 Value (cast index->i32 then make_int_tuple)
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return fly.memref_load(memref, indices, loc=loc, ip=ip)
        # Common case: user passes `index` as a 1-D coordinate/offset.
        if str(indices.type) == "index":
            indices = _arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return fly.memref_load(memref, indices, loc=loc, ip=ip)

    # List/tuple (e.g. [row]) or python int.
    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return fly.memref_load(memref, indices, loc=loc, ip=ip)


@dsl_api_wrapper
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


@dsl_api_wrapper
def memref_load_vec(memref, loc=None, ip=None):
    return fly.memref_load_vec(memref, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_store_vec(vector, memref, loc=None, ip=None):
    return fly.memref_store_vec(vector, memref, loc=loc, ip=ip)


@dsl_api_wrapper
def get_layout(memref, loc=None, ip=None):
    return fly.get_layout(memref, loc=loc, ip=ip)


@dsl_api_wrapper
def get_iter(memref, loc=None, ip=None):
    return fly.get_iter(memref, loc=loc, ip=ip)


@dsl_api_wrapper
def make_view(iter, layout, loc=None, ip=None):
    return fly.make_view(iter, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def make_ptr(result_type, args, loc=None, ip=None):
    return fly.make_ptr(result_type, args, loc=loc, ip=ip)


@dsl_api_wrapper
def add_offset(ptr, offset, loc=None, ip=None):
    if not isinstance(offset, ir.Value):
        offset = make_int_tuple(offset, loc=loc, ip=ip)
    return fly.add_offset(ptr, offset, loc=loc, ip=ip)


@dsl_api_wrapper
def make_copy_atom(copy_op_type, elem_type, loc=None, ip=None):
    from .derived import CopyAtom
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
    copy_atom_ty = CopyAtomType.get(copy_op_type, val_bits)
    return CopyAtom(fly.make_copy_atom(copy_atom_ty, val_bits=val_bits, loc=loc, ip=ip))


@dsl_api_wrapper
def make_mma_atom(atom_type, loc=None, ip=None):
    from .derived import MmaAtom

    return MmaAtom(fly.make_mma_atom(atom_type, loc=loc, ip=ip))


@dsl_api_wrapper
def make_tile(*args, loc=None, ip=None):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        modes = args[0]
    else:
        modes = args
    resolved = []
    for m in modes:
        if isinstance(m, int):
            resolved.append(make_layout(m, 1, loc=loc, ip=ip))
        else:
            resolved.append(m)
    return fly.make_tile(resolved, loc=loc, ip=ip)


@dsl_api_wrapper
def mma_atom_call(mma_atom, d, a, b, c, loc=None, ip=None):
    return fly.mma_atom_call(mma_atom, d, a, b, c, loc=loc, ip=ip)


@dsl_api_wrapper
def copy_atom_call(copy_atom, src, dst, loc=None, ip=None):
    return fly.copy_atom_call(copy_atom, src, dst, loc=loc, ip=ip)


@dsl_api_wrapper
def make_tiled_copy(copy_atom, layout_thr_val, tile_mn, loc=None, ip=None):
    from .derived import TiledCopy

    return TiledCopy(fly.make_tiled_copy(copy_atom, layout_thr_val, tile_mn, loc=loc, ip=ip))


@dsl_api_wrapper
def make_tiled_mma(mma_atom, atom_layout, permutation=None, loc=None, ip=None):
    from .derived import TiledMma

    return TiledMma(fly.make_tiled_mma(mma_atom, atom_layout, permutation=permutation, loc=loc, ip=ip))


@dsl_api_wrapper
def tiled_copy_partition_src(tiled_copy, src, thr_int_tuple, loc=None, ip=None):
    return fly.tiled_copy_partition_src(tiled_copy, src, thr_int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple, loc=None, ip=None):
    return fly.tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_copy_retile(tiled_copy, t, loc=None, ip=None):
    return fly.tiled_copy_retile(tiled_copy, t, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_mma_partition(operand_id, tiled_mma, t, coord, loc=None, ip=None):
    return fly.tiled_mma_partition(operand_id, tiled_mma, t, coord, loc=loc, ip=ip)


@dsl_api_wrapper
def copy(copy_atom, src, dst, *, pred=None, loc=None, ip=None):
    return fly.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip)


@dsl_api_wrapper
def gemm(mma_atom, d, a, b, c, loc=None, ip=None):
    return fly.gemm(mma_atom, d, a, b, c, loc=loc, ip=ip)


@dsl_api_wrapper
def printf(*args, format_str="", loc=None, ip=None):
    def _convert_printf_value(val):
        """Convert Python values to MLIR Values for printf.
        Returns tuple of (is_static, value) where is_static=True means value is a string to embed."""
        if isinstance(val, ir.Value):
            return (False, val)
        elif isinstance(val, type):
            return (True, val.__name__)
        elif isinstance(val, str):
            return (True, val)
        elif isinstance(val, bool):
            return (False, _arith.constant(T.i1(), int(val)))
        elif isinstance(val, int):
            return (False, _arith.constant(T.i32(), val))
        elif isinstance(val, float):
            return (False, _arith.constant(T.f64(), val))
        elif hasattr(val, "__extract_ir_values__"):
            ir_values = val.__extract_ir_values__()
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
