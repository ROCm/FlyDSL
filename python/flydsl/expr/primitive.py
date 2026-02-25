from .._mlir import ir
from .._mlir.dialects import arith, fly
from .._mlir.dialects.fly import (
    # Enum Attributes
    AddressSpace,
    CachePolicy,
    # Type
    CopyAtomUniversalCopyType,
    IntTupleType,
    LayoutType,
    MemRefType,
    MmaAtomUniversalFMAType,
    PointerType,
    SwizzleType,
)
from .._mlir.extras import types as _T
from .meta import dsl_api_wrapper
from .typing import Int32


def _unwrap(v):
    """Unwrap ArithValue or other wrappers to raw ir.Value or keep Python int/None."""
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "_value"):
        return _unwrap(v._value)
    if hasattr(v, "value") and isinstance(getattr(v, "value", None), ir.Value):
        return v.value
    return v


def _idx2crd_arith(idx, layout, loc=None, ip=None):
    """Pure-arith idx2crd for simple product layouts: (s0, s1, ...):(d0, d1, ...).
    Returns a Python list of index-typed Values [coord0, coord1, ...].
    """
    import builtins as _builtins
    import re
    v = _unwrap(idx)
    ly = _unwrap(layout)
    ly_str = str(ly.type) if hasattr(ly, 'type') else str(ly)
    shape_match = re.search(r'\(([^)]+)\):\(([^)]+)\)', ly_str)
    if shape_match:
        raw_shapes = [s.strip() for s in shape_match.group(1).split(',')]
        raw_strides = [s.strip() for s in shape_match.group(2).split(',')]
        shapes = [None if s == '?' else int(s) for s in raw_shapes]
        strides = [None if d == '?' else int(d) for d in raw_strides]
        if all(s is None for s in strides):
            return [v]
    else:
        return [v]
    if isinstance(v, ir.Value) and str(v.type) != 'index':
        v = arith.IndexCastOp(ir.IndexType.get(), v).result
    import builtins as _builtins
    idx_type = ir.IndexType.get()
    def _const(val):
        return arith.ConstantOp(idx_type, ir.IntegerAttr.get(idx_type, val)).result
    ndims = len(strides)
    indexed = list(_builtins.zip(range(ndims), strides, shapes))
    has_stride = [(i, s, sz) for i, s, sz in indexed if s is not None]
    has_stride.sort(key=lambda x: x[1], reverse=True)
    coords = [None] * ndims
    remaining = v
    for i, stride_val, size_val in has_stride:
        if stride_val == 0:
            coords[i] = _const(0)
            continue
        c_stride = _const(stride_val)
        coord = arith.DivUIOp(remaining, c_stride).result
        if size_val is not None:
            c_size = _const(size_val)
            coord = arith.RemUIOp(coord, c_size).result
        coords[i] = coord
    for i in range(ndims):
        if coords[i] is None:
            coords[i] = remaining
    return coords


def _unwrap_tuple(elems):
    """Unwrap each element in a tuple/list that may contain ArithValue wrappers.
    Also converts index-typed Values to i32 for fly dialect compatibility."""
    def _conv(e):
        v = _unwrap(e)
        if isinstance(v, ir.Value) and v.type == ir.IndexType.get():
            v = arith.IndexCastOp(ir.IntegerType.get_signless(32), v).result
        return v
    if isinstance(elems, (tuple, list)):
        return type(elems)(_conv(e) for e in elems)
    return _conv(elems)

__all__ = [
    # Enum Attributes
    "AddressSpace",
    "CachePolicy",
    # Types
    "CopyAtomUniversalCopyType",
    "IntTupleType",
    "LayoutType",
    "MemRefType",
    "MmaAtomUniversalFMAType",
    "PointerType",
    "SwizzleType",
    # DSL functions
    "const_expr",
    "range_constexpr",
    "rank",
    "depth",
    "static",
    "int_tuple_add",
    "int_tuple_sub",
    "int_tuple_mul",
    "int_tuple_div",
    "int_tuple_product",
    "idx2crd",
    "get",
    "make_identity_tensor",
    "make_identity_layout",
    "make_shape",
    "make_stride",
    "make_coord",
    "make_int_tuple",
    "make_layout",
    "size",
    "get_scalar",
    "slice",
    "crd2idx",
    "composition",
    "complement",
    "right_inverse",
    "coalesce",
    "zip",
    "select",
    "group",
    "append",
    "prepend",
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
    "make_atom",
    "make_tile",
    "mma_atom_call",
    "copy_atom_call",
    "make_tiled_copy",
    "memref_alloca",
    "memref_load",
    "memref_store",
    "memref_load_vec",
    "memref_store_vec",
    "get_layout",
    "get_iter",
    "make_view",
    "add_offset",
    "cooperative_copy",
    "gemm",
    "copy",
    "printf",
]


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
def make_identity_tensor(shape, loc=None, ip=None):
    return fly.make_identity_tensor(shape, loc=loc, ip=ip)


@dsl_api_wrapper
def make_identity_layout(shape, loc=None, ip=None):
    return fly.make_identity_layout(shape, loc=loc, ip=ip)


@dsl_api_wrapper
def make_shape(*shape, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(_unwrap_tuple(shape))
    return fly.make_shape(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_stride(*stride, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(_unwrap_tuple(stride))
    return fly.make_stride(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_coord(*coord, loc=None, ip=None):
    return [_unwrap(c) for c in coord]


@dsl_api_wrapper
def make_int_tuple(elems, loc=None, ip=None):
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(_unwrap_tuple(elems))
    return fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_layout(shape, stride, loc=None, ip=None):
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(_unwrap_tuple(shape))
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    else:
        shape = _unwrap(shape)
    if not isinstance(stride, ir.Value):
        strideTy, dyncElems = fly.infer_int_tuple_type(_unwrap_tuple(stride))
        stride = fly.make_stride(strideTy, dyncElems, loc=loc, ip=ip)
    else:
        stride = _unwrap(stride)
    return fly.make_layout(shape, stride=stride, loc=loc, ip=ip)


@dsl_api_wrapper
def size(int_tuple, loc=None, ip=None):
    return fly.size(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def get(int_tuple, mode, loc=None, ip=None):
    """Extract element `mode` from a coordinate."""
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    v = _unwrap(int_tuple)
    tp = str(v.type) if hasattr(v, 'type') else ''
    if 'int_tuple' in tp:
        selected = fly.select(v, indices=[mode], loc=loc, ip=ip)
        result = fly.get_scalar(selected, loc=loc, ip=ip)
        if hasattr(result, 'type') and str(result.type) != 'index':
            result = arith.IndexCastOp(ir.IndexType.get(), result).result
        return result
    return v


@dsl_api_wrapper
def get_scalar(int_tuple, loc=None, ip=None):
    return fly.get_scalar(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def slice(src, coord, loc=None, ip=None):
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = fly.infer_int_tuple_type(coord)
        coord = fly.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return fly.slice(src, coord, loc=loc, ip=ip)


@dsl_api_wrapper
def idx2crd(idx, layout, loc=None, ip=None):
    """idx2crd: returns a Python list of index Values (pure arith decomposition)."""
    return _idx2crd_arith(idx, layout, loc=loc, ip=ip)


def _crd2idx_fly_fallback(crd, ly, loc=None, ip=None):
    """Fall back to fly.crd2idx for layouts with dynamic strides."""
    if isinstance(crd, (list, tuple)):
        unwrapped = [_unwrap(c) for c in crd]
        idx_type = ir.IndexType.get()
        i32_type = ir.IntegerType.get_signless(32)
        converted = []
        for v in unwrapped:
            if isinstance(v, ir.Value):
                if str(v.type) == 'index':
                    v = arith.IndexCastOp(i32_type, v).result
                converted.append(v)
            elif isinstance(v, int):
                converted.append(arith.ConstantOp(i32_type, ir.IntegerAttr.get(i32_type, v)).result)
            else:
                converted.append(v)
        IntTupleTy, dyncElems = fly.infer_int_tuple_type(tuple(converted))
        crd_val = fly.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)
    else:
        crd_val = _unwrap(crd)
    result = fly.crd2idx(crd_val, _unwrap(ly), loc=loc, ip=ip)
    if hasattr(result, 'type') and 'int_tuple' in str(result.type):
        result = fly.get_scalar(result, loc=loc, ip=ip)
    if hasattr(result, 'type') and str(result.type) != 'index':
        result = arith.IndexCastOp(ir.IndexType.get(), result).result
    return result


@dsl_api_wrapper
def crd2idx(crd, layout, loc=None, ip=None):
    """Pure-arith crd2idx: idx = sum(coord[i] * stride[i])."""
    import builtins as _b
    import re as _re
    ly = _unwrap(layout)
    ly_str = str(ly.type) if hasattr(ly, 'type') else ''
    match = _re.search(r'\(([^)]+)\):\(([^)]+)\)', ly_str)
    if not match:
        return _crd2idx_fly_fallback(crd, ly, loc=loc, ip=ip)
    raw_strides = [s.strip() for s in match.group(2).split(',')]
    strides = [None if s == '?' else int(s) for s in raw_strides]
    if any(s is None for s in strides):
        return _crd2idx_fly_fallback(crd, ly, loc=loc, ip=ip)
    if isinstance(crd, (list, tuple)):
        coords = [_unwrap(c) for c in crd]
    else:
        coords = [_unwrap(crd)]
    idx_type = ir.IndexType.get()
    def _to_val(v):
        if isinstance(v, ir.Value):
            if str(v.type) != 'index':
                return arith.IndexCastOp(idx_type, v).result
            return v
        if isinstance(v, int):
            return arith.ConstantOp(idx_type, ir.IntegerAttr.get(idx_type, v)).result
        return _unwrap(v)
    coords = [_to_val(c) for c in coords]
    idx_type = ir.IndexType.get()
    def _c(v):
        return arith.ConstantOp(idx_type, ir.IntegerAttr.get(idx_type, v)).result
    result = None
    for i, (coord_v, stride_v) in enumerate(_b.zip(coords, strides)):
        cv = _unwrap(coord_v)
        if isinstance(cv, ir.Value) and str(cv.type) != 'index':
            cv = arith.IndexCastOp(idx_type, cv).result
        if stride_v is None or stride_v == 1:
            term = cv
        elif stride_v == 0:
            continue
        else:
            term = arith.MulIOp(cv, _c(stride_v)).result
        if result is None:
            result = term
        else:
            result = arith.AddIOp(result, term).result
    return result if result is not None else _c(0)


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
def make_atom(atom_type, loc=None, ip=None):
    return fly.make_atom(atom_type, loc=loc, ip=ip)


@dsl_api_wrapper
def make_tile(layouts, loc=None, ip=None):
    return fly.make_tile(layouts, loc=loc, ip=ip)


@dsl_api_wrapper
def mma_atom_call(mma_atom, d, a, b, c, loc=None, ip=None):
    return fly.mma_atom_call(mma_atom, d, a, b, c, loc=loc, ip=ip)


@dsl_api_wrapper
def copy_atom_call(copy_atom, src, dst, loc=None, ip=None):
    return fly.copy_atom_call(copy_atom, src, dst, loc=loc, ip=ip)


@dsl_api_wrapper
def make_tiled_copy(copy_atom, layout_tv, tile_mn, loc=None, ip=None):
    return fly.make_tiled_copy(copy_atom, layout_tv, tile_mn, loc=loc, ip=ip)


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
            indices = arith.IndexCastOp(T.i32(), indices)
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
            indices = arith.IndexCastOp(T.i32(), indices)
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
def add_offset(ptr, offset, loc=None, ip=None):
    if not isinstance(offset, ir.Value):
        offset = make_int_tuple(offset, loc=loc, ip=ip)
    return fly.add_offset(ptr, offset, loc=loc, ip=ip)


@dsl_api_wrapper
def cooperative_copy(tiled_copy, partition_idx, src, dst, loc=None, ip=None):
    return fly.cooperative_copy(
        tiled_copy,
        partition_idx,
        src,
        dst,
        loc=loc,
        ip=ip,
    )


@dsl_api_wrapper
def gemm(mma_atom, d, a, b, c, loc=None, ip=None):
    return fly.gemm(mma_atom, d, a, b, c, loc=loc, ip=ip)


@dsl_api_wrapper
def copy(copy_atom, src, dst, loc=None, ip=None):
    return fly.copy(copy_atom, src, dst, loc=loc, ip=ip)


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
            return (False, arith.constant(T.i1(), int(val)))
        elif isinstance(val, int):
            return (False, arith.constant(T.i32(), val))
        elif isinstance(val, float):
            return (False, arith.constant(T.f64(), val))
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
