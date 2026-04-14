# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import enum
from inspect import isclass
from typing import Generic, Type, TypeVar

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
    # DSL value types
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
    "full",
    "full_like",
    "zeros_like",
]


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

    @traced_op
    def to_py_value(self, loc=None, ip=None):
        if self.is_static:
            if self.is_leaf:
                return self.get_static_leaf_int
            return tuple(get_(self, i).to_py_value() for i in range(self.rank))
        leaves = get_leaves(self, dynamic_only=True, loc=loc, ip=ip)
        leaf_iter = iter(leaves)
        return self._rebuild_py_value(leaf_iter)

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
        return AddressSpace(self.type.address_space)

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
        return AddressSpace(self.type.address_space)

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
        return memref_load_vec(self, loc=loc, ip=ip)

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

    @traced_op
    def set_value(self, field, value, loc=None, ip=None):
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

    @traced_op
    def set_value(self, field, value, loc=None, ip=None):
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
    def make_fragment_A(self, a: Tensor, loc=None, ip=None):
        return mma_make_fragment(MmaOperand.A, self, a, loc=loc, ip=ip)

    @traced_op
    def make_fragment_B(self, b: Tensor, loc=None, ip=None):
        return mma_make_fragment(MmaOperand.B, self, b, loc=loc, ip=ip)

    @traced_op
    def make_fragment_C(self, c: Tensor, loc=None, ip=None):
        return mma_make_fragment(MmaOperand.C, self, c, loc=loc, ip=ip)


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
        if shape is None:
            vty = ir.VectorType(value.type)
            shape = tuple(vty.shape)
            dtype = Numeric.from_ir_type(vty.element_type)
        signed = dtype.signed if isclass(dtype) and issubclass(dtype, Integer) else False
        super().__init__(value, signed)
        self._shape = (shape,) if isinstance(shape, int) else tuple(shape)
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
        r = 1
        for s in self._shape:
            r *= s
        return r

    def __str__(self):
        return f"Vector({self.type} o {self._shape}, {self._dtype.__name__})"

    def __repr__(self):
        return self.__str__()

    def __fly_values__(self):
        return [self]

    @classmethod
    def __fly_construct__(cls, values):
        return values[0]

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

    def __getitem__(self, idx):
        if isinstance(idx, int):
            res = _vector.ExtractOp(self, static_position=[idx], dynamic_position=[]).result
            return self._dtype(res)
        raise TypeError(f"unsupported index type: {type(idx)}")

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
    def filled(cls, shape, fill_value, dtype: Type[Numeric], *, loc=None, ip=None) -> "Vector":
        def _shape_numel(dims):
            n = 1
            for dim in dims:
                if isinstance(dim, (tuple, list)):
                    n *= _shape_numel(dim)
                else:
                    n *= dim
            return n

        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        n = _shape_numel(shape)
        if isinstance(fill_value, (int, float, bool)):
            fill_value = dtype(fill_value)
        elif isinstance(fill_value, Numeric):
            fill_value = fill_value.to(dtype, loc=loc, ip=ip)
        else:
            raise ValueError(f"expected numeric fill_value, got {type(fill_value)}")
        vec_ty = ir.VectorType.get([n], dtype.ir_type)
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


def zeros_like(a: Vector, dtype=None, *, loc=None, ip=None) -> Vector:
    return Vector.zeros_like(a, dtype, loc=loc, ip=ip)
