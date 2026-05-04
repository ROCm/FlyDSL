# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import enum
import operator
from inspect import isclass
from typing import Generic, Type, TypeVar, overload

from flydsl.runtime.device import get_rocm_arch

from .._mlir import ir
from .._mlir.dialects import gpu
from .._mlir.dialects import vector as _vector
from .meta import traced_op
from .numeric import (
    BFloat16,
    Boolean,
    Float,
    Float4E2M1FN,
    Float6E2M3FN,
    Float6E3M2FN,
    Float8E4M3,
    Float8E4M3B11FNUZ,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E8M0FNU,
    Float16,
    Float32,
    Float64,
    Index,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Integer,
    Numeric,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from .primitive import *
from .utils.arith import (
    ArithValue,
    _to_raw,
    element_type,
    fp_to_fp,
    fp_to_int,
    int_to_fp,
    int_to_int,
)


def _vec(n: int, elem: ir.Type) -> ir.Type:
    return ir.VectorType.get([int(n)], elem)


def default_f8_type() -> ir.Type:
    """Select E4M3 f8 type compatible with the current GPU arch.

    - gfx95* (MI350): FP8 E4M3FN (OCP)
    - gfx94* (MI300): FP8 E4M3FNUZ
    """
    arch = ""
    try:
        arch = str(get_rocm_arch())
    except Exception:
        arch = ""
    if "gfx95" in arch or "gfx12" in arch:
        return Float8E4M3FN.ir_type
    return Float8E4M3FNUZ.ir_type


class Types:
    """Property-based MLIR type constructors backed by DSL numeric classes.

    Scalar properties delegate to ``<DslClass>.ir_type`` (single source of
    truth in ``numeric.py``).  Vector shortcuts and ``vec()`` use
    ``ir.VectorType`` directly.

    Usage::

        from flydsl.expr.typing import T
        T.f16            # ir.F16Type
        T.i32x4          # vector<4xi32>
        T.vec(8, T.f16)  # vector<8xf16>
    """

    # ---- Index ----
    @property
    def index(self) -> ir.Type:
        return ir.IndexType.get()

    # ---- Integer scalars & vectors ----
    @property
    def i8(self) -> ir.Type:
        return Int8.ir_type

    @property
    def i8x2(self) -> ir.Type:
        return _vec(2, Int8.ir_type)

    @property
    def i8x4(self) -> ir.Type:
        return _vec(4, Int8.ir_type)

    @property
    def i8x8(self) -> ir.Type:
        return _vec(8, Int8.ir_type)

    @property
    def i8x16(self) -> ir.Type:
        return _vec(16, Int8.ir_type)

    @property
    def i16(self) -> ir.Type:
        return Int16.ir_type

    @property
    def i16x2(self) -> ir.Type:
        return _vec(2, Int16.ir_type)

    @property
    def i16x4(self) -> ir.Type:
        return _vec(4, Int16.ir_type)

    @property
    def i16x8(self) -> ir.Type:
        return _vec(8, Int16.ir_type)

    @property
    def i32(self) -> ir.Type:
        return Int32.ir_type

    @property
    def i32x2(self) -> ir.Type:
        return _vec(2, Int32.ir_type)

    @property
    def i32x4(self) -> ir.Type:
        return _vec(4, Int32.ir_type)

    @property
    def i64(self) -> ir.Type:
        return Int64.ir_type

    @property
    def i64x2(self) -> ir.Type:
        return _vec(2, Int64.ir_type)

    # ---- Float scalars & vectors ----
    @property
    def f16(self) -> ir.Type:
        return Float16.ir_type

    @property
    def f16x2(self) -> ir.Type:
        return _vec(2, Float16.ir_type)

    @property
    def f16x4(self) -> ir.Type:
        return _vec(4, Float16.ir_type)

    @property
    def f16x8(self) -> ir.Type:
        return _vec(8, Float16.ir_type)

    @property
    def bf16(self) -> ir.Type:
        return BFloat16.ir_type

    @property
    def bf16x2(self) -> ir.Type:
        return _vec(2, BFloat16.ir_type)

    @property
    def bf16x4(self) -> ir.Type:
        return _vec(4, BFloat16.ir_type)

    @property
    def bf16x8(self) -> ir.Type:
        return _vec(8, BFloat16.ir_type)

    @property
    def f32(self) -> ir.Type:
        return Float32.ir_type

    @property
    def f32x2(self) -> ir.Type:
        return _vec(2, Float32.ir_type)

    @property
    def f32x4(self) -> ir.Type:
        return _vec(4, Float32.ir_type)

    @property
    def f64(self) -> ir.Type:
        return Float64.ir_type

    # ---- FP8 (arch-dependent shortcut) ----
    @property
    def f8(self) -> ir.Type:
        return default_f8_type()

    @property
    def f8x2(self) -> ir.Type:
        return _vec(2, default_f8_type())

    @property
    def f8x4(self) -> ir.Type:
        return _vec(4, default_f8_type())

    @property
    def f8x8(self) -> ir.Type:
        return _vec(8, default_f8_type())

    @property
    def f8x16(self) -> ir.Type:
        return _vec(16, default_f8_type())

    # ---- Dynamic vector constructor ----
    def vec(self, n: int, elem: ir.Type) -> ir.Type:
        return _vec(n, elem)


T = Types()


__all__ = [
    # MLIR type helpers
    "Types",
    "T",
    "default_f8_type",
    "is_generic_address_space",
    "is_target_address_space",
    # DSL value types
    "Numeric",
    "Boolean",
    "Float",
    "BFloat16",
    "Float4E2M1FN",
    "Float6E2M3FN",
    "Float6E3M2FN",
    "Float8E4M3",
    "Float8E4M3B11FNUZ",
    "Float8E4M3FN",
    "Float8E4M3FNUZ",
    "Float8E5M2",
    "Float8E8M0FNU",
    "Float16",
    "Float32",
    "Float64",
    "Int4",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Index",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Constexpr",
    "IntTuple",
    "Layout",
    "Swizzle",
    "ComposedLayout",
    "Tensor",
    "CopyAtom",
    "Tile",
    "TiledCopy",
    "TiledMma",
    "Stream",
    "Tuple3D",
    # Vector types
    "Vector",
    "ReductionOp",
    "empty_like",
    "full",
    "full_like",
    "ones_like",
    "zeros_like",
]


def address_space_from_attr(address_space):
    """Normalize core Fly address spaces while preserving target-specific attrs."""
    if isinstance(address_space, AddressSpace):
        return address_space
    if isinstance(address_space, int):
        try:
            return AddressSpace(address_space)
        except ValueError as exc:
            valid = ", ".join(f"{int(candidate)} ({candidate})" for candidate in AddressSpace)
            raise ValueError(f"unknown Fly address space integer {address_space}; expected one of: {valid}") from exc

    text = str(address_space)
    for candidate in AddressSpace:
        if text == str(candidate) or text == f"#fly<address_space {candidate}>":
            return candidate
    return address_space


def is_generic_address_space(address_space, expected: AddressSpace) -> bool:
    if not isinstance(expected, AddressSpace):
        raise TypeError(f"expected must be an AddressSpace enum, got {type(expected).__name__}")
    return address_space_from_attr(address_space) == expected


def is_target_address_space(address_space, expected) -> bool:
    if isinstance(expected, AddressSpace):
        raise TypeError("expected must be a target-specific address-space attribute; use is_generic_address_space")

    exp = address_space_from_attr(expected)
    actual = address_space_from_attr(address_space)
    if isinstance(actual, AddressSpace):
        return False
    return str(actual) == str(exp)


ValueT = TypeVar("ValueT")


class Constexpr(Generic[ValueT]):
    pass


class BuiltinDslType(ir.Value):
    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        type_str = self.type.__str__()
        return f"{type(self).__name__}{type_str[type_str.find('<') : type_str.rfind('>') + 1]}"

    def __repr__(self):
        return f"{type(self).__name__}<{super().__str__()}>"

    @classmethod
    def __fly_construct__(cls, values):
        return cls(values[0])

    def __fly_values__(self):
        return [self]


@ir.register_value_caster(IntTupleType.static_typeid, replace=True)
class IntTuple(BuiltinDslType):
    @property
    def rank(self) -> int:
        return self.type.rank

    @property
    def depth(self) -> int:
        return self.type.depth

    @property
    def is_leaf(self) -> bool:
        return self.type.is_leaf

    @property
    def is_static(self) -> bool:
        return self.type.is_static

    @property
    def get_static_leaf_int(self) -> int:
        if not self.type.is_leaf or not self.type.is_static:
            raise ValueError("IntTuple is not a static leaf")
        return self.type.get_static_leaf_int

    @staticmethod
    def _static_to_py_value(ty):
        if ty.is_leaf:
            return ty.get_static_leaf_int
        return tuple(IntTuple._static_to_py_value(ty.at(i)) for i in range(ty.rank))

    def _rebuild_py_value(self, leaf_iter):
        if self.is_leaf:
            if self.is_static:
                return self.get_static_leaf_int
            val = next(leaf_iter)
            width = ir.IntegerType(val.type).width
            wrapper = Int64 if width == 64 else Int32
            return wrapper(val)
        return tuple(IntTuple(get_(self, i))._rebuild_py_value(leaf_iter) for i in range(self.rank))

    @traced_op
    def to_py_value(self, loc=None, ip=None):
        if self.is_static:
            return IntTuple._static_to_py_value(self.type)
        leaves = get_leaves(self, dynamic_only=True, loc=loc, ip=ip)
        leaf_iter = iter(leaves)
        return self._rebuild_py_value(leaf_iter)

    @traced_op
    def __getitem__(self, mode, loc=None, ip=None):
        if isinstance(mode, int):
            mode = [mode]
        if self.rank <= mode[0]:
            raise IndexError(f"Index {mode[0]} out of range for int tuple with rank {self.rank}")
        return get_(self, mode, loc=loc, ip=ip)


@ir.register_value_caster(TileType.static_typeid, replace=True)
class Tile(BuiltinDslType):
    @property
    def rank(self) -> int:
        return self.type.rank


@ir.register_value_caster(LayoutType.static_typeid, replace=True)
class Layout(BuiltinDslType):
    @property
    def rank(self) -> int:
        return self.type.rank

    @property
    def depth(self) -> int:
        return self.type.depth

    @property
    def is_leaf(self) -> bool:
        return self.type.is_leaf

    @property
    def is_static(self) -> bool:
        return self.type.is_static

    @property
    def is_static_shape(self) -> bool:
        return self.type.is_static_shape

    @property
    def is_static_stride(self) -> bool:
        return self.type.is_static_stride

    @property
    @traced_op
    def shape(self, loc=None, ip=None) -> IntTuple:
        return get_shape(self, loc=loc, ip=ip)

    @property
    @traced_op
    def stride(self, loc=None, ip=None) -> IntTuple:
        return get_stride(self, loc=loc, ip=ip)

    @traced_op
    def __getitem__(self, mode, loc=None, ip=None):
        if isinstance(mode, int):
            mode = [mode]
        if self.rank <= mode[0]:
            raise IndexError(f"Index {mode[0]} out of range for layout with rank {self.rank}")
        return get_(self, mode, loc=loc, ip=ip)

    @traced_op
    def __call__(self, *coord, loc=None, ip=None):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord, loc=loc, ip=ip)

        if has_none(coord):
            return slice(self, coord, loc=loc, ip=ip)
        else:
            return crd2idx(coord, self, loc=loc, ip=ip)

    @traced_op
    def get_hier_coord(self, index, loc=None, ip=None):
        return idx2crd(index, self, loc=loc, ip=ip)

    @traced_op
    def get_flat_coord(self, index, loc=None, ip=None):
        return get_flat_coord(index, self, loc=loc, ip=ip)

    @traced_op
    def get_1d_coord(self, index, loc=None, ip=None):
        return get_1d_coord(index, self, loc=loc, ip=ip)


@ir.register_value_caster(SwizzleType.static_typeid, replace=True)
class Swizzle(BuiltinDslType):
    @property
    def mask(self) -> int:
        return self.type.mask

    @property
    def base(self) -> int:
        return self.type.base

    @property
    def shift(self) -> int:
        return self.type.shift


@ir.register_value_caster(ComposedLayoutType.static_typeid, replace=True)
class ComposedLayout(BuiltinDslType):
    @property
    def rank(self) -> int:
        return self.type.rank

    @property
    def depth(self) -> int:
        return self.type.depth

    @property
    def is_leaf(self) -> bool:
        return self.type.is_leaf

    @property
    def is_static(self) -> bool:
        return self.type.is_static

    @property
    def is_static_outer(self) -> bool:
        return self.type.is_static_outer

    @property
    def is_static_inner(self) -> bool:
        return self.type.is_static_inner

    @property
    def is_static_offset(self) -> bool:
        return self.type.is_static_offset

    @property
    def shape(self) -> IntTuple:
        return get_shape(self)

    @property
    def stride(self) -> IntTuple:
        raise TypeError("ComposedLayout doesn't have a meaningful stride")

    @property
    @traced_op
    def inner(self, loc=None, ip=None):
        return composed_get_inner(self, loc=loc, ip=ip)

    @property
    @traced_op
    def offset(self, loc=None, ip=None) -> IntTuple:
        return composed_get_offset(self, loc=loc, ip=ip)

    @property
    @traced_op
    def outer(self, loc=None, ip=None) -> Layout:
        return composed_get_outer(self, loc=loc, ip=ip)

    @traced_op
    def __getitem__(self, mode, loc=None, ip=None):
        if isinstance(mode, int):
            mode = [mode]
        if self.rank <= mode[0]:
            raise IndexError(f"Index {mode[0]} out of range for composed layout with rank {self.rank}")
        return get_(self, mode, loc=loc, ip=ip)

    @traced_op
    def __call__(self, *coord, loc=None, ip=None):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord, loc=loc, ip=ip)

        if has_none(coord):
            return slice(self, coord, loc=loc, ip=ip)
        else:
            return crd2idx(coord, self, loc=loc, ip=ip)


@ir.register_value_caster(PointerType.static_typeid, replace=True)
class Pointer(BuiltinDslType):
    @property
    def element_type(self):
        return Numeric.from_ir_type(self.type.element_type)

    @property
    def dtype(self):
        return self.element_type

    @property
    def value_type(self):
        return self.element_type

    @property
    def address_space(self):
        return address_space_from_attr(self.type.address_space)

    @property
    def memspace(self):
        return self.address_space

    @property
    def alignment(self):
        return self.type.alignment


@ir.register_value_caster(MemRefType.static_typeid, replace=True)
@ir.register_value_caster(CoordTensorType.static_typeid, replace=True)
class Tensor(BuiltinDslType):
    @property
    def element_type(self):
        if isinstance(self.type, CoordTensorType):
            raise TypeError("CoordTensor doesn't have an element type")
        return Numeric.from_ir_type(self.type.element_type)

    @property
    def dtype(self):
        return self.element_type

    @property
    def value_type(self):
        return self.element_type

    @property
    def address_space(self):
        return address_space_from_attr(self.type.address_space)

    @property
    def memspace(self):
        return self.address_space

    @property
    def alignment(self):
        return self.type.alignment

    @property
    def leading_dim(self):
        return self.type.leading_dim

    @property
    def layout(self) -> Layout:
        return get_layout(self)

    @property
    def shape(self) -> IntTuple:
        return self.layout.shape

    @property
    def stride(self) -> IntTuple:
        return self.layout.stride

    @traced_op
    def __getitem__(self, coord, loc=None, ip=None):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord, loc=loc, ip=ip)

        if has_none(coord):
            return slice(self, coord, loc=loc, ip=ip)
        else:
            return memref_load(self, coord, loc=loc, ip=ip)

    @traced_op
    def __setitem__(self, coord, value, loc=None, ip=None):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord, loc=loc, ip=ip)

        if has_none(coord):
            self.__getitem__(coord, loc=loc, ip=ip).store(value, loc=loc, ip=ip)
        else:
            memref_store(value, self, coord, loc=loc, ip=ip)

    @traced_op
    def load(self, loc=None, ip=None):
        return Vector(memref_load_vec(self, loc=loc, ip=ip), self.shape.to_py_value(), self.dtype)

    @traced_op
    def store(self, vector, loc=None, ip=None):
        return memref_store_vec(vector, self, loc=loc, ip=ip)

    @traced_op
    def fill(self, value, loc=None, ip=None):
        filled_vec = full(self.shape.to_py_value(), value, self.dtype, loc=loc, ip=ip)
        return self.store(filled_vec, loc=loc, ip=ip)


@ir.register_value_caster(CopyAtomType.static_typeid, replace=True)
class CopyAtom(BuiltinDslType):
    @property
    def val_bits(self):
        return self.type.val_bits

    @property
    def thr_layout(self):
        return static(self.type.thr_layout)

    @property
    def thr_id(self):
        return self.thr_layout

    @property
    def layout_src_tv(self):
        return static(self.type.tv_layout_src)

    @property
    def layout_dst_tv(self):
        return static(self.type.tv_layout_dst)

    @property
    def layout_ref_tv(self):
        return static(self.type.tv_layout_ref)

    @overload
    def set_value(self, field: str, value, loc=None, ip=None): ...
    @overload
    def set_value(self, field: dict, loc=None, ip=None): ...

    @traced_op
    def set_value(self, field, value=None, loc=None, ip=None):
        if isinstance(field, dict):
            result = self
            for k, v in field.items():
                result = atom_set_value(result, k, v, loc=loc, ip=ip)
            return result
        return atom_set_value(self, field, value, loc=loc, ip=ip)


@ir.register_value_caster(MmaAtomType.static_typeid, replace=True)
class MmaAtom(BuiltinDslType):
    @property
    def thr_layout(self):
        return static(self.type.thr_layout)

    @property
    def thr_id(self):
        return self.thr_layout

    @property
    def shape_mnk(self):
        return static(self.type.shape_mnk)

    @property
    def layout_A_tv(self):
        return static(self.type.tv_layout_a)

    @property
    def layout_B_tv(self):
        return static(self.type.tv_layout_b)

    @property
    def layout_C_tv(self):
        return static(self.type.tv_layout_c)

    @overload
    def set_value(self, field: str, value, loc=None, ip=None): ...
    @overload
    def set_value(self, field: dict, loc=None, ip=None): ...

    @traced_op
    def set_value(self, field, value=None, loc=None, ip=None):
        if isinstance(field, dict):
            result = self
            for k, v in field.items():
                result = atom_set_value(result, k, v, loc=loc, ip=ip)
            return result
        return atom_set_value(self, field, value, loc=loc, ip=ip)


@ir.register_value_caster(TiledCopyType.static_typeid, replace=True)
class TiledCopy(BuiltinDslType):
    @property
    def tile_mn(self):
        return static(self.type.tile_mn)

    @property
    def layout_tv_tiled(self):
        return static(self.type.layout_thr_val)

    @property
    def layout_src_tv_tiled(self):
        return static(self.type.tiled_tv_layout_src)

    @property
    def layout_dst_tv_tiled(self):
        return static(self.type.tiled_tv_layout_dst)

    def get_slice(self, thr_idx):
        from .derived import ThrCopy

        return ThrCopy(self, thr_idx)

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)


@ir.register_value_caster(TiledMmaType.static_typeid, replace=True)
class TiledMma(BuiltinDslType):
    @property
    def mma_atom(self):
        return self.type.mma_atom

    @property
    def atom_layout(self):
        return static(self.type.atom_layout)

    @property
    def permutation_mnk(self):
        return static(self.type.permutation)

    @property
    def tile_size_mnk(self):
        return static(self.type.tile_size_mnk)

    @property
    def thr_layout_vmnk(self):
        return static(self.type.thr_layout_vmnk)

    @property
    def tv_layout_A_tiled(self):
        return static(self.type.tiled_tv_layout_a)

    @property
    def tv_layout_B_tiled(self):
        return static(self.type.tiled_tv_layout_b)

    @property
    def tv_layout_C_tiled(self):
        return static(self.type.tiled_tv_layout_c)

    def get_slice(self, thr_idx):
        from .derived import ThrMma

        return ThrMma(self, thr_idx)

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)

    @traced_op
    def make_fragment_A(self, a: Tensor, *, stages=None, loc=None, ip=None):
        return mma_make_fragment(MmaOperand.A, self, a, stages=stages, loc=loc, ip=ip)

    @traced_op
    def make_fragment_B(self, b: Tensor, *, stages=None, loc=None, ip=None):
        return mma_make_fragment(MmaOperand.B, self, b, stages=stages, loc=loc, ip=ip)

    @traced_op
    def make_fragment_C(self, c: Tensor, *, stages=None, loc=None, ip=None):
        return mma_make_fragment(MmaOperand.C, self, c, stages=stages, loc=loc, ip=ip)


class Stream:
    """Opaque async queue handle for kernel launch.

    ``None`` is the default queue; an :class:`int` is a raw pointer. Any other
    value is interpreted by the active device runtime
    (:mod:`flydsl.runtime.device_runtime`).
    """

    _is_stream_param = True

    def __init__(self, value=None):
        self.value = value
        self._stream_storage = None

    def __fly_types__(self):
        return [gpu.AsyncTokenType.get()]

    def __fly_ptrs__(self):
        if isinstance(self.value, int):
            self._stream_storage = ctypes.c_void_p(self.value)
        elif self.value is None:
            self._stream_storage = ctypes.c_void_p(0)
        else:
            self._stream_storage = ctypes.c_void_p(self.value.cuda_stream)
        return [ctypes.cast(ctypes.pointer(self._stream_storage), ctypes.c_void_p)]

    @staticmethod
    def _extract_stream_value(arg):
        raw = arg.value if isinstance(arg, Stream) else arg
        if raw is None:
            return 0
        elif isinstance(raw, int):
            return raw
        return raw.cuda_stream

    @classmethod
    def _reusable_slot_spec(cls, arg):
        return ctypes.c_void_p, cls._extract_stream_value

    @classmethod
    def __fly_construct__(cls, values):
        return Stream(values[0])

    def __fly_values__(self):
        return [self.value]


class Tuple3D:
    def __init__(self, factory, dtype=Int32):
        self.factory = factory
        self.dtype = dtype

    def __getattr__(self, name):
        if name in ("x", "y", "z"):
            return self.dtype(self.factory(name))
        raise AttributeError(name)

    def __iter__(self):
        return iter((self.x, self.y, self.z))


# ═══════════════════════════════════════════════════════════════════════
# Vector — register vector with value semantics
# ═══════════════════════════════════════════════════════════════════════


class ReductionOp(enum.Enum):
    ADD = "add"
    MUL = "mul"
    MAX = "max"
    MIN = "min"


_REDUCE_KINDS = {
    "add": (_vector.CombiningKind.ADD, _vector.CombiningKind.ADD, _vector.CombiningKind.ADD),
    "mul": (_vector.CombiningKind.MUL, _vector.CombiningKind.MUL, _vector.CombiningKind.MUL),
    "max": (_vector.CombiningKind.MAXNUMF, _vector.CombiningKind.MAXSI, _vector.CombiningKind.MAXUI),
    "min": (_vector.CombiningKind.MINIMUMF, _vector.CombiningKind.MINSI, _vector.CombiningKind.MINUI),
}

_VECTOR_OP_METHODS = {
    operator.add: "__add__",
    operator.sub: "__sub__",
    operator.mul: "__mul__",
    operator.truediv: "__truediv__",
    operator.floordiv: "__floordiv__",
    operator.mod: "__mod__",
    operator.pow: "__pow__",
    operator.lshift: "__lshift__",
    operator.rshift: "__rshift__",
    operator.and_: "__and__",
    operator.or_: "__or__",
    operator.xor: "__xor__",
    operator.lt: "__lt__",
    operator.le: "__le__",
    operator.gt: "__gt__",
    operator.ge: "__ge__",
    operator.eq: "__eq__",
    operator.ne: "__ne__",
}

_VECTOR_REVERSE_OP_METHODS = {
    "__add__": "__radd__",
    "__sub__": "__rsub__",
    "__mul__": "__rmul__",
    "__truediv__": "__rtruediv__",
    "__floordiv__": "__rfloordiv__",
    "__mod__": "__rmod__",
    "__pow__": "__rpow__",
    "__lshift__": "__rlshift__",
    "__rshift__": "__rrshift__",
    "__and__": "__rand__",
    "__or__": "__ror__",
    "__xor__": "__rxor__",
}


def _resolve_combining_kind(op, is_float, signed):
    if isinstance(op, _vector.CombiningKind):
        return op
    if isinstance(op, ReductionOp):
        key = op.value
    elif isinstance(op, str):
        key = op.lower()
    else:
        raise TypeError(f"reduce op must be str, ReductionOp, or CombiningKind, got {type(op)}")
    triple = _REDUCE_KINDS.get(key)
    if triple is None:
        raise ValueError(f"unknown reduction kind {op!r}; expected one of {list(_REDUCE_KINDS)}")
    return triple[0] if is_float else (triple[1] if signed else triple[2])


@ir.register_value_caster(ir.VectorType.static_typeid, replace=True)
class Vector(ArithValue):
    """Thread-local register vector with value semantics.

    Wraps a flat ``vector<NxTy>`` ir.Value with shape and dtype metadata.
    Arithmetic operators are inherited from ArithValue; scalar operands
    are auto-broadcast via ``_coerce_other``.
    """

    def __init__(self, value, shape=None, dtype=None):
        if not isinstance(value, ir.Value) and hasattr(value, "ir_value"):
            value = value.ir_value()
        vty = ir.VectorType(value.type)
        if shape is None:
            shape = tuple(vty.shape)
            dtype = Numeric.from_ir_type(vty.element_type)
        elif dtype is None:
            dtype = Numeric.from_ir_type(vty.element_type)
        shape = self._canonical_shape(shape)
        if not all(isinstance(dim, int) for dim in self._flatten_static(shape)):
            raise ValueError("dynamic vector shape is not supported")
        if self._numel_from_shape(shape) != self._numel_from_shape(tuple(vty.shape)):
            raise ValueError(
                f"shape {shape} has {self._numel_from_shape(shape)} elements, but value has type {value.type}"
            )
        if dtype.ir_type != vty.element_type:
            raise ValueError(f"dtype {dtype} does not match vector element type {vty.element_type}")
        signed = dtype.signed if isclass(dtype) and issubclass(dtype, Integer) else False
        super().__init__(value, signed)
        self._shape = shape
        self._dtype = dtype

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def element_type(self) -> Type[Numeric]:
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def numel(self) -> int:
        return self._numel_from_shape(self._shape)

    @staticmethod
    def _canonical_shape(shape):
        return (shape,) if isinstance(shape, int) else tuple(shape)

    @staticmethod
    def _flatten_static(value):
        if isinstance(value, (tuple, list)):
            out = []
            for elem in value:
                out.extend(Vector._flatten_static(elem))
            return out
        return [value]

    @staticmethod
    def _slice_shape(shape, coord):
        out = []
        for dim, index in zip(shape, coord, strict=True):
            if isinstance(dim, (tuple, list)):
                if index is None:
                    out.append(tuple(dim))
                    continue
                sub_shape = Vector._slice_shape(tuple(dim), Vector._canonical_shape(index))
                if sub_shape != ():
                    out.append(sub_shape)
            elif index is None:
                out.append(int(dim))
        return tuple(out)

    @staticmethod
    def _flat_static_index(shape, coord) -> int:
        shape = Vector._flatten_static(shape)
        coord = Vector._flatten_static(coord)
        if len(shape) != len(coord):
            raise ValueError(f"coordinate rank {len(coord)} does not match shape rank {len(shape)}")
        idx = 0
        for dim, value in zip(shape, coord, strict=True):
            idx = idx * int(dim) + int(value)
        return idx

    @staticmethod
    def _numel_from_shape(shape) -> int:
        n = 1
        for dim in shape:
            if isinstance(dim, (tuple, list)):
                n *= Vector._numel_from_shape(dim)
            else:
                n *= int(dim)
        return n

    @staticmethod
    def make_type(shape, dtype: Type[Numeric]) -> ir.Type:
        """Return the flat MLIR vector type for a FlyDSL vector shape/dtype."""
        if not isclass(dtype) or not issubclass(dtype, Numeric):
            raise TypeError(f"dtype must be a Numeric type, got {type(dtype)}")
        shape = Vector._canonical_shape(shape)
        return ir.VectorType.get([Vector._numel_from_shape(shape)], dtype.ir_type)

    @staticmethod
    def _infer_broadcast_shape(lhs_shape, rhs_shape):
        if lhs_shape == rhs_shape:
            return lhs_shape
        lhs_flat = Vector._flatten_static(lhs_shape)
        rhs_flat = Vector._flatten_static(rhs_shape)
        if Vector._numel_from_shape(lhs_shape) == 1:
            return rhs_shape
        if Vector._numel_from_shape(rhs_shape) == 1:
            return lhs_shape
        rank = max(len(lhs_flat), len(rhs_flat))
        lhs_dims = [1] * (rank - len(lhs_flat)) + lhs_flat
        rhs_dims = [1] * (rank - len(rhs_flat)) + rhs_flat
        result = []
        for lhs_dim, rhs_dim in zip(lhs_dims, rhs_dims, strict=True):
            if lhs_dim == rhs_dim:
                result.append(lhs_dim)
            elif lhs_dim == 1:
                result.append(rhs_dim)
            elif rhs_dim == 1:
                result.append(lhs_dim)
            else:
                raise ValueError(f"cannot broadcast shapes {lhs_shape} and {rhs_shape}")
        return tuple(result)

    def __str__(self):
        return f"Vector({self.type} o {self._shape}, {self._dtype.__name__})"

    def __repr__(self):
        return self.__str__()

    __hash__ = ArithValue.__hash__

    def __fly_values__(self):
        return [self]

    @classmethod
    def __fly_construct__(cls, values):
        return cls(values[0])

    def __fly_construct_from_values__(self, values):
        return Vector(values[0], self.shape, self.dtype)

    def to(self, dtype: Type[Numeric], *, loc=None, ip=None) -> "Vector":
        if dtype is ir.Value:
            return self
        if not isclass(dtype) or not issubclass(dtype, Numeric):
            raise TypeError(f"dtype must be a Numeric type, got {type(dtype)}")
        src_dtype = self._dtype
        if src_dtype is dtype:
            return self
        src_float = getattr(src_dtype, "is_float", False)
        dst_float = getattr(dtype, "is_float", False)
        if src_float and dst_float:
            res = fp_to_fp(self, dtype.ir_type, loc=loc, ip=ip)
        elif src_float:
            res = fp_to_int(self, dtype.signed, dtype.ir_type, loc=loc, ip=ip)
        elif dst_float:
            res = int_to_fp(self, src_dtype.signed, dtype.ir_type, loc=loc, ip=ip)
        else:
            res = int_to_int(self, dtype, loc=loc, ip=ip)
        return Vector(res, self._shape, dtype)

    def ir_value(self, *, loc=None, ip=None):
        return self

    def with_signedness(self, signed):
        return ArithValue(self, signed)

    def _wrap_op_result(self, result, shape):
        if isinstance(result, ir.Value) and isinstance(result.type, ir.VectorType):
            return Vector(result, shape, Numeric.from_ir_type(result.type.element_type))
        if isinstance(result, Numeric):
            return result
        if isinstance(result, ir.Value):
            return Numeric.from_ir_type(result.type)(result)
        return result

    def _apply_op(self, method_name, op, other, flip=False, *, loc=None, ip=None):
        lhs = self
        rhs = other
        shape = self.shape
        if isinstance(other, Vector):
            shape = self._infer_broadcast_shape(self.shape, other.shape)
            lhs = self.broadcast_to(shape, loc=loc, ip=ip)
            rhs = other.broadcast_to(shape, loc=loc, ip=ip)
        method = getattr(ArithValue, method_name)
        if flip:
            if isinstance(rhs, Vector):
                result = method(rhs, lhs, loc=loc, ip=ip)
            else:
                reverse_name = _VECTOR_REVERSE_OP_METHODS.get(method_name, method_name)
                result = getattr(ArithValue, reverse_name)(lhs, rhs, loc=loc, ip=ip)
        else:
            result = method(lhs, rhs, loc=loc, ip=ip)
        return self._wrap_op_result(result, shape)

    def apply_op(self, op, other, flip=False, *, loc=None, ip=None):
        method_name = _VECTOR_OP_METHODS.get(op)
        if method_name is None:
            raise NotImplementedError(f"Vector.apply_op does not support {op}")
        return self._apply_op(method_name, op, other, flip=flip, loc=loc, ip=ip)

    def __add__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.add, other, loc=loc, ip=ip)

    def __radd__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.add, other, flip=True, loc=loc, ip=ip)

    def __sub__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.sub, other, loc=loc, ip=ip)

    def __rsub__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.sub, other, flip=True, loc=loc, ip=ip)

    def __mul__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.mul, other, loc=loc, ip=ip)

    def __rmul__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.mul, other, flip=True, loc=loc, ip=ip)

    def __truediv__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.truediv, other, loc=loc, ip=ip)

    def __rtruediv__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.truediv, other, flip=True, loc=loc, ip=ip)

    def __floordiv__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.floordiv, other, loc=loc, ip=ip)

    def __rfloordiv__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.floordiv, other, flip=True, loc=loc, ip=ip)

    def __mod__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.mod, other, loc=loc, ip=ip)

    def __rmod__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.mod, other, flip=True, loc=loc, ip=ip)

    def __pow__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.pow, other, loc=loc, ip=ip)

    def __rpow__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.pow, other, flip=True, loc=loc, ip=ip)

    def __lshift__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.lshift, other, loc=loc, ip=ip)

    def __rlshift__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.lshift, other, flip=True, loc=loc, ip=ip)

    def __rshift__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.rshift, other, loc=loc, ip=ip)

    def __rrshift__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.rshift, other, flip=True, loc=loc, ip=ip)

    def __and__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.and_, other, loc=loc, ip=ip)

    def __rand__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.and_, other, flip=True, loc=loc, ip=ip)

    def __or__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.or_, other, loc=loc, ip=ip)

    def __ror__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.or_, other, flip=True, loc=loc, ip=ip)

    def __xor__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.xor, other, loc=loc, ip=ip)

    def __rxor__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.xor, other, flip=True, loc=loc, ip=ip)

    def __lt__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.lt, other, loc=loc, ip=ip)

    def __le__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.le, other, loc=loc, ip=ip)

    def __gt__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.gt, other, loc=loc, ip=ip)

    def __ge__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.ge, other, loc=loc, ip=ip)

    def __eq__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.eq, other, loc=loc, ip=ip)

    def __ne__(self, other, *, loc=None, ip=None):
        return self.apply_op(operator.ne, other, loc=loc, ip=ip)

    def reduce(self, op, init_val=None, reduction_profile=None, *, fastmath=None, loc=None, ip=None):
        is_fp = self._dtype.is_float
        signed = getattr(self._dtype, "signed", True)
        kind = _resolve_combining_kind(op, is_fp, signed)
        et = element_type(self.type)
        kwargs = {}
        if fastmath is not None:
            kwargs["fastmath"] = fastmath
        if init_val is not None:
            if isinstance(init_val, Numeric):
                init_val = init_val.ir_value(loc=loc, ip=ip)
            kwargs["acc"] = _to_raw(init_val)
        res = _vector.reduction(et, kind, self, loc=loc, ip=ip, **kwargs)
        return self._dtype(res)

    @staticmethod
    def _coerce_element(element, dtype: Type[Numeric], *, loc=None, ip=None):
        if isinstance(element, (int, float, bool)):
            return dtype(element)
        if isinstance(element, Numeric):
            return element.to(dtype, loc=loc, ip=ip)
        if isinstance(element, ir.Value):
            return Numeric.from_ir_type(element.type)(element).to(dtype, loc=loc, ip=ip)
        if hasattr(element, "ir_value"):
            value = element.ir_value(loc=loc, ip=ip)
            return Numeric.from_ir_type(value.type)(value).to(dtype, loc=loc, ip=ip)
        raise ValueError(f"expected numeric vector element, got {type(element)}")

    def __getitem__(self, idx):
        if idx is None:
            return self
        if isinstance(idx, int):
            res = _vector.ExtractOp(self, static_position=[idx], dynamic_position=[]).result
            return self._dtype(res)
        if isinstance(idx, (Numeric, ArithValue, ir.Value)):
            dyn_idx = _to_raw(Index(idx))
            res = _vector.ExtractOp(
                self,
                static_position=[ir.ShapedType.get_dynamic_size()],
                dynamic_position=[dyn_idx],
            ).result
            return self._dtype(res)
        if isinstance(idx, tuple):
            coord = self._canonical_shape(idx)
            if not any(part is None for part in self._flatten_static(coord)):
                flat_idx = self._flat_static_index(self._shape, coord)
                return self[flat_idx]

            flat_shape = self._flatten_static(self._shape)
            flat_coord = self._flatten_static(coord)
            if len(flat_shape) != len(flat_coord):
                raise ValueError(f"coordinate rank {len(flat_coord)} does not match shape rank {len(flat_shape)}")

            offsets = [0 if c is None else int(c) for c in flat_coord]
            sizes = [int(s) if c is None else 1 for s, c in zip(flat_shape, flat_coord, strict=True)]
            tmp_ty = ir.VectorType.get(list(flat_shape), self._dtype.ir_type)
            tmp = _vector.shape_cast(tmp_ty, self)
            res_ty = ir.VectorType.get(sizes, self._dtype.ir_type)
            res = _vector.extract_strided_slice(
                res_ty,
                tmp,
                offsets=offsets,
                sizes=sizes,
                strides=[1] * len(flat_shape),
            )
            res_shape = self._slice_shape(self._shape, coord)
            return self._build_result(res, res_shape, row_major=True)
        raise TypeError(f"unsupported index type: {type(idx)}")

    def _build_result(self, value, shape, *, row_major=False, loc=None, ip=None) -> "Vector":
        shape = self._canonical_shape(shape)
        flat_ty = self.make_type(shape, self._dtype)
        flat_value = _vector.shape_cast(flat_ty, value, loc=loc, ip=ip)
        return Vector(flat_value, shape, self._dtype)

    def reshape(self, shape, *, loc=None, ip=None) -> "Vector":
        shape = self._canonical_shape(shape)
        if self.numel != self._numel_from_shape(shape):
            raise ValueError(f"expected reshaped size to match: {self._shape} -> {shape}")
        return Vector(self, shape, self._dtype)

    def broadcast_to(self, target_shape, *, loc=None, ip=None) -> "Vector":
        target_shape = self._canonical_shape(target_shape)
        if self._shape == target_shape:
            return self
        src_flat_shape = self._flatten_static(self._shape)
        target_flat_shape = self._flatten_static(target_shape)
        if self.numel == 1:
            scalar = self[0].ir_value(loc=loc, ip=ip)
            target_ty = self.make_type(target_shape, self._dtype)
            res = _vector.broadcast(target_ty, scalar, loc=loc, ip=ip)
            return Vector(res, target_shape, self._dtype)
        if len(src_flat_shape) > len(target_flat_shape):
            raise ValueError(f"cannot broadcast shape {self._shape} to {target_shape}")
        padded_src = [1] * (len(target_flat_shape) - len(src_flat_shape)) + src_flat_shape
        for src_dim, dst_dim in zip(padded_src, target_flat_shape, strict=True):
            if src_dim != dst_dim and src_dim != 1:
                raise ValueError(f"cannot broadcast shape {self._shape} to {target_shape}")
        src_ty = ir.VectorType.get(padded_src, self._dtype.ir_type)
        src = _vector.shape_cast(src_ty, self, loc=loc, ip=ip)
        target_ty_nd = ir.VectorType.get(target_flat_shape, self._dtype.ir_type)
        res = _vector.broadcast(target_ty_nd, src, loc=loc, ip=ip)
        return self._build_result(res, target_shape, row_major=True, loc=loc, ip=ip)

    def bitcast(self, dtype: Type[Numeric], *, loc=None, ip=None) -> "Vector":
        src_bits = self.numel * self._dtype.width
        dst_count = src_bits // dtype.width
        dst_vec_ty = ir.VectorType.get([dst_count], dtype.ir_type)
        res = _vector.BitCastOp(dst_vec_ty, self, loc=loc, ip=ip).result
        return Vector(res, (dst_count,), dtype)

    def shuffle(self, other, mask, *, loc=None, ip=None) -> "Vector":
        other_val = other if not isinstance(other, Vector) else ir.Value(other)
        res = _vector.shuffle(self, other_val, mask, loc=loc, ip=ip)
        return Vector(res, (len(mask),), self._dtype)

    @classmethod
    def from_elements(cls, elements, dtype: Type[Numeric] | None = None, *, loc=None, ip=None) -> "Vector":
        elements = list(elements)
        if not elements:
            raise ValueError("Vector.from_elements requires at least one element")
        if dtype is None:
            first = elements[0]
            if isinstance(first, Numeric):
                dtype = type(first)
            elif isinstance(first, ir.Value):
                dtype = Numeric.from_ir_type(first.type)
            elif hasattr(first, "ir_value"):
                dtype = Numeric.from_ir_type(first.ir_value(loc=loc, ip=ip).type)
            else:
                dtype = type(Numeric.from_python_value(first))
        vec_ty = cls.make_type(len(elements), dtype)
        raw_elements = [_to_raw(cls._coerce_element(element, dtype, loc=loc, ip=ip)) for element in elements]
        res = _vector.from_elements(vec_ty, raw_elements, loc=loc, ip=ip)
        return cls(res, (len(elements),), dtype)

    @classmethod
    def load(cls, result_type, memref, indices, *, loc=None, ip=None) -> "Vector":
        vty = ir.VectorType(result_type)
        dtype = Numeric.from_ir_type(vty.element_type)
        raw_indices = []
        for index in indices:
            if isinstance(index, int):
                index = Index(index, loc=loc, ip=ip)
            elif not isinstance(index, ir.Value) and not hasattr(index, "ir_value"):
                index = Index(index, loc=loc, ip=ip)
            raw_indices.append(_to_raw(index))
        res = _vector.LoadOp(result_type, _to_raw(memref), raw_indices, loc=loc, ip=ip).result
        return cls(res, tuple(vty.shape), dtype)

    def store(self, memref, indices, *, alignment=None, loc=None, ip=None):
        raw_indices = []
        for index in indices:
            if isinstance(index, int):
                index = Index(index, loc=loc, ip=ip)
            elif not isinstance(index, ir.Value) and not hasattr(index, "ir_value"):
                index = Index(index, loc=loc, ip=ip)
            raw_indices.append(_to_raw(index))
        kwargs = {}
        if alignment is not None:
            kwargs["alignment"] = alignment
        return _vector.store(_to_raw(self), _to_raw(memref), raw_indices, loc=loc, ip=ip, **kwargs)

    @classmethod
    def filled(cls, shape, fill_value, dtype: Type[Numeric], *, loc=None, ip=None) -> "Vector":
        shape = cls._canonical_shape(shape)
        n = cls._numel_from_shape(shape)
        if isinstance(fill_value, (int, float, bool)):
            fill_value = dtype(fill_value)
        elif isinstance(fill_value, Numeric):
            fill_value = fill_value.to(dtype, loc=loc, ip=ip)
        else:
            raise ValueError(f"expected numeric fill_value, got {type(fill_value)}")
        vec_ty = cls.make_type(n, dtype)
        val = _vector.broadcast(vec_ty, fill_value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        return cls(val, shape, dtype)

    @classmethod
    def filled_like(cls, template: "Vector", fill_value, dtype=None, *, loc=None, ip=None) -> "Vector":
        if dtype is None:
            dtype = template.dtype
        return cls.filled(template.shape, fill_value, dtype, loc=loc, ip=ip)

    @classmethod
    def zeros_like(cls, template: "Vector", dtype=None, *, loc=None, ip=None) -> "Vector":
        if dtype is None:
            dtype = template.dtype
        return cls.filled(template.shape, 0.0 if dtype.is_float else 0, dtype, loc=loc, ip=ip)


def full(shape, fill_value, dtype: Type[Numeric], *, loc=None, ip=None) -> Vector:
    return Vector.filled(shape, fill_value, dtype, loc=loc, ip=ip)


def full_like(a: Vector, fill_value, dtype=None, *, loc=None, ip=None) -> Vector:
    return Vector.filled_like(a, fill_value, dtype, loc=loc, ip=ip)


def empty_like(a: Vector, dtype=None, *, loc=None, ip=None) -> Vector:
    return Vector.filled_like(a, 0, dtype, loc=loc, ip=ip)


def ones_like(a: Vector, dtype=None, *, loc=None, ip=None) -> Vector:
    return Vector.filled_like(a, 1, dtype, loc=loc, ip=ip)


def zeros_like(a: Vector, dtype=None, *, loc=None, ip=None) -> Vector:
    return Vector.zeros_like(a, dtype, loc=loc, ip=ip)
