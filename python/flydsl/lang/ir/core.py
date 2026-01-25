from functools import partialmethod
from functools import lru_cache

from flydsl.lang.meta import dsl_api_wrapper


from .module import _global_ctx

from ..._mlir import ir
from ..._mlir.dialects import fly as _fly_ir
from ..._mlir.dialects._fly_enum_gen import AddressSpace, CachePolicy

from ..._mlir.dialects import arith
from ..._mlir.extras import types as T


def _binary_op(lhs, rhs, op: str) -> "ArithValue":
    op = op.capitalize()
    if arith._is_float_type(lhs.type) and arith._is_float_type(rhs.type):
        op += "F"
    elif arith._is_integer_like_type(lhs.type) and arith._is_integer_like_type(
        lhs.type
    ):
        op += "I"
    else:
        raise NotImplementedError(f"Unsupported '{op}' operands: {lhs}, {rhs}")

    op = getattr(arith, f"{op}Op")
    return op(lhs, rhs).result


@ir.register_value_caster(T.F16Type.static_typeid)
@ir.register_value_caster(T.F32Type.static_typeid)
@ir.register_value_caster(T.F64Type.static_typeid)
@ir.register_value_caster(T.IntegerType.static_typeid)
class ArithValue(ir.Value):
    def __init__(self, v):
        super().__init__(v)

    __add__ = partialmethod(_binary_op, op="add")
    __sub__ = partialmethod(_binary_op, op="sub")
    __mul__ = partialmethod(_binary_op, op="mul")

    def __str__(self):
        return super().__str__().replace(ir.Value.__name__, ArithValue.__name__)


def make_int32(value):
    return _fly_ir.make_int32(value)


def make_int32_tuple(value):
    return _fly_ir.make_int32_tuple(value)


def rank(int_or_tuple):
    return _fly_ir.rank(int_or_tuple)


def depth(int_or_tuple):
    return _fly_ir.depth(int_or_tuple)


@dsl_api_wrapper
def int_tuple_add(lhs, rhs, loc=None, ip=None):
    return _fly_ir.int_tuple_add(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_sub(lhs, rhs, loc=None, ip=None):
    return _fly_ir.int_tuple_sub(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_mul(lhs, rhs, loc=None, ip=None):
    return _fly_ir.int_tuple_mul(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_div(lhs, rhs, loc=None, ip=None):
    return _fly_ir.int_tuple_div(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def int_tuple_product(int_tuple, loc=None, ip=None):
    return _fly_ir.int_tuple_product(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def make_identity_tensor(shape, loc=None, ip=None):
    return _fly_ir.make_identity_tensor(shape, loc=loc, ip=ip)


@dsl_api_wrapper
def make_identity_layout(shape, loc=None, ip=None):
    return _fly_ir.make_identity_layout(shape, loc=loc, ip=ip)


@dsl_api_wrapper
def make_shape(*shape, loc=None, ip=None):
    IntTupleTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, shape)
    return _fly_ir.make_shape(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_stride(*stride, loc=None, ip=None):
    IntTupleTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, stride)
    return _fly_ir.make_stride(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_coord(*coord, loc=None, ip=None):
    IntTupleTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, coord)
    return _fly_ir.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_int_tuple(elems, loc=None, ip=None):
    IntTupleTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, elems)
    return _fly_ir.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)


@dsl_api_wrapper
def make_layout(shape, stride, loc=None, ip=None):
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, shape)
        shape = _fly_ir.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(stride, ir.Value):
        strideTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, stride)
        stride = _fly_ir.make_stride(strideTy, dyncElems, loc=loc, ip=ip)
    return _fly_ir.make_layout(shape, stride=stride, loc=loc, ip=ip)


@dsl_api_wrapper
def size(int_tuple, loc=None, ip=None):
    return _fly_ir.size(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def get_scalar(int_tuple, loc=None, ip=None):
    return _fly_ir.get_scalar(int_tuple, loc=loc, ip=ip)


@dsl_api_wrapper
def slice(src, coord, loc=None, ip=None):
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = _fly_ir.infer_int_tuple_type(ir.Context.current, coord)
        coord = _fly_ir.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return _fly_ir.slice(src, coord, loc=loc, ip=ip)


@dsl_api_wrapper
def crd2idx(crd, layout, loc=None, ip=None):
    return _fly_ir.crd2idx(crd, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def composition(layout, tiler, loc=None, ip=None):
    return _fly_ir.composition(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def complement(layout, codomain_size, loc=None, ip=None):
    if not isinstance(codomain_size, ir.Value):
        codomain_sizeTy, dyncElems = _fly_ir.infer_int_tuple_type(
            ir.Context.current, codomain_size
        )
        codomain_size = _fly_ir.make_shape(codomain_sizeTy, dyncElems, loc=loc, ip=ip)
    return _fly_ir.complement(layout, codomain_size=codomain_size, loc=loc, ip=ip)


@dsl_api_wrapper
def coalesce(layout, pattern=None, loc=None, ip=None):
    return _fly_ir.coalesce(layout, pattern=pattern, loc=loc, ip=ip)


@dsl_api_wrapper
def zip(lhs, rhs, loc=None, ip=None):
    return _fly_ir.zip(lhs, rhs, loc=loc, ip=ip)


@dsl_api_wrapper
def select(int_tuple, indices, loc=None, ip=None):
    return _fly_ir.select(int_tuple, indices=indices, loc=loc, ip=ip)


@dsl_api_wrapper
def group(int_tuple, begin: int, end: int, loc=None, ip=None):
    return _fly_ir.group(int_tuple, begin=begin, end=end, loc=loc, ip=ip)


@dsl_api_wrapper
def append(base, elem, n: int | None = None, loc=None, ip=None):
    return _fly_ir.append(base, elem, n=n, loc=loc, ip=ip)


@dsl_api_wrapper
def prepend(base, elem, n: int | None = None, loc=None, ip=None):
    return _fly_ir.prepend(base, elem, n=n, loc=loc, ip=ip)


@dsl_api_wrapper
def logical_divide(layout, divisor, loc=None, ip=None):
    return _fly_ir.logical_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def zipped_divide(layout, divisor, loc=None, ip=None):
    return _fly_ir.zipped_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_divide(layout, divisor, loc=None, ip=None):
    return _fly_ir.tiled_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def flat_divide(layout, divisor, loc=None, ip=None):
    return _fly_ir.flat_divide(layout, divisor, loc=loc, ip=ip)


@dsl_api_wrapper
def logical_product(layout, tiler, loc=None, ip=None):
    return _fly_ir.logical_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def zipped_product(layout, tiler, loc=None, ip=None):
    return _fly_ir.zipped_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def tiled_product(layout, tiler, loc=None, ip=None):
    return _fly_ir.tiled_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def flat_product(layout, tiler, loc=None, ip=None):
    return _fly_ir.flat_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def block_product(layout, tiler, loc=None, ip=None):
    return _fly_ir.block_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def raked_product(layout, tiler, loc=None, ip=None):
    return _fly_ir.raked_product(layout, tiler, loc=loc, ip=ip)


@dsl_api_wrapper
def make_atom(atom_type, loc=None, ip=None):
    return _fly_ir.make_atom(atom_type, loc=loc, ip=ip)


@dsl_api_wrapper
def make_tile(layouts, loc=None, ip=None):
    return _fly_ir.make_tile(layouts, loc=loc, ip=ip)


@dsl_api_wrapper
def mma_atom_call(mma_atom, d, a, b, c, loc=None, ip=None):
    return _fly_ir.mma_atom_call(mma_atom, d, a, b, c, loc=loc, ip=ip)


@dsl_api_wrapper
def copy_atom_call(copy_atom, src, dst, loc=None, ip=None):
    return _fly_ir.copy_atom_call(copy_atom, src, dst, loc=loc, ip=ip)


@dsl_api_wrapper
def make_tiled_copy(copy_atom, layout_tv, tile_mn, loc=None, ip=None):
    return _fly_ir.make_tiled_copy(copy_atom, layout_tv, tile_mn, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_alloca(memref_type, layout, loc=None, ip=None):
    return _fly_ir.memref_alloca(memref_type, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_load(memref, indices, loc=None, ip=None):
    # `fly.memref.load` expects `indices` as `!fly.int_tuple` (typically a scalar offset).
    # Accept convenience forms:
    # - int_tuple Value (pass through)
    # - python int / tuple/list (make_int_tuple)
    # - index/i32/i64 Value (cast index->i32 then make_int_tuple)
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return _fly_ir.memref_load(memref, indices, loc=loc, ip=ip)
        # Common case: user passes `index` as a 1-D coordinate/offset.
        if str(indices.type) == "index":
            indices = arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return _fly_ir.memref_load(memref, indices, loc=loc, ip=ip)

    # List/tuple (e.g. [row]) or python int.
    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return _fly_ir.memref_load(memref, indices, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_store(value, memref, indices, loc=None, ip=None):
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return _fly_ir.memref_store(value, memref, indices, loc=loc, ip=ip)
        if str(indices.type) == "index":
            indices = arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return _fly_ir.memref_store(value, memref, indices, loc=loc, ip=ip)

    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return _fly_ir.memref_store(value, memref, indices, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_load_vec(memref, loc=None, ip=None):
    return _fly_ir.memref_load_vec(memref, loc=loc, ip=ip)


@dsl_api_wrapper
def memref_store_vec(vector, memref, loc=None, ip=None):
    return _fly_ir.memref_store_vec(vector, memref, loc=loc, ip=ip)


@dsl_api_wrapper
def get_layout(memref, loc=None, ip=None):
    return _fly_ir.get_layout(memref, loc=loc, ip=ip)


@dsl_api_wrapper
def get_iter(memref, loc=None, ip=None):
    return _fly_ir.get_iter(memref, loc=loc, ip=ip)


@dsl_api_wrapper
def make_view(iter, layout, loc=None, ip=None):
    return _fly_ir.make_view(iter, layout, loc=loc, ip=ip)


@dsl_api_wrapper
def add_offset(ptr, offset, loc=None, ip=None):
    if not isinstance(offset, ir.Value):
        offset = make_int_tuple(offset, loc=loc, ip=ip)
    return _fly_ir.add_offset(ptr, offset, loc=loc, ip=ip)


@dsl_api_wrapper
def cooperative_copy(tiled_copy, partition_idx, src, dst, loc=None, ip=None):
    return _fly_ir.cooperative_copy(
        tiled_copy,
        partition_idx,
        src,
        dst,
        loc=loc,
        ip=ip,
    )


@dsl_api_wrapper
def print_op(*values, format_str="", loc=None, ip=None):
    """
    Print operation for debugging. Supports IntTuple and other value types.
    Lowers to printf for host code or gpu.printf for device code.

    Example:
        fx.print_op(int_tuple)
        fx.print_op(layout)
        fx.print_op(value1, value2, value3)
        fx.print_op(value1, format_str="v1=%d\n")
    """
    return _fly_ir.print_(format_str, list(values), loc=loc, ip=ip)


# ==============================================================================
# Fly Type Classes (MLIR-style API)
# ==============================================================================


class PointerType:
    """
    Fly Pointer Type with MLIR-style static get() method.

    Example:
        ptr_ty = PointerType.get(T.f32(), AddressSpace.Global)
        ptr_ty = PointerType.get(T.f32(), AddressSpace.Register, alignment=16)
    """

    @staticmethod
    def get(elem_ty, address_space, alignment=None):
        """
        Create a PointerType.

        Args:
            elem_ty: Element type (e.g., T.f32())
            address_space: Address space (AddressSpace.Global, AddressSpace.Shared, AddressSpace.Register)
            alignment: Optional alignment value

        Returns:
            PointerType as ir.Type
        """
        return _fly_ir.PointerType.get(elem_ty, int(address_space), alignment)


class MemRefType:
    """
    Fly MemRef Type with MLIR-style static get() method.

    Example:
        layout_ty = LayoutType.get(ir.Context.current, 16, 1)
        memref_ty = MemRefType.get(T.f32(), AddressSpace.Global, layout_ty)
    """

    @staticmethod
    def get(elem_ty, address_space, layout, alignment=None):
        """
        Create a MemRefType.

        Args:
            elem_ty: Element type (e.g., T.f32())
            address_space: Address space (AddressSpace.Global, AddressSpace.Shared, AddressSpace.Register)
            layout: Layout type (LayoutType or ir.Type)
            alignment: Optional alignment value

        Returns:
            MemRefType as ir.Type
        """
        # If layout is an ir.Value (from make_layout), get its type
        if isinstance(layout, ir.Value):
            layout = layout.type
        return _fly_ir.MemRefType.get(elem_ty, int(address_space), layout, alignment)


class LayoutType:
    """
    Fly Layout Type with MLIR-style static get() method.

    Example:
        layout_ty = LayoutType.get(ir.Context.current, 16, 1)
        layout_ty = LayoutType.get(ir.Context.current, (4, 4), (4, 1))
    """

    @staticmethod
    def get(context, shape, stride):
        """
        Create a LayoutType.

        Args:
            context: MLIR context
            shape: Shape as int or tuple
            stride: Stride as int or tuple

        Returns:
            LayoutType as ir.Type
        """
        return _fly_ir.LayoutType.get(context, shape, stride)


class IntTupleType:
    """
    Fly IntTuple Type with MLIR-style static get() method.

    Example:
        int_tuple_ty = IntTupleType.get(ir.Context.current, (4, 4))
    """

    @staticmethod
    def get(context, int_or_tuple):
        """
        Create an IntTupleType.

        Args:
            context: MLIR context
            int_or_tuple: Python int or tuple

        Returns:
            Tuple of (IntTupleType as ir.Type, list of dynamic elements)
        """
        return _fly_ir.IntTupleType.get(context, int_or_tuple)
