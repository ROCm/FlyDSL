# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import enum
import operator
import types
from collections.abc import Callable as AbcCallable
from inspect import isclass
from typing import Any, Callable, List, Type, get_origin, overload

from flydsl.runtime.device import get_rocm_arch

from .._mlir import ir
from .._mlir.dialects import gpu
from .._mlir.dialects import vector as _vector
from .meta import dsl_loc_tracing
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
    Int128,
    Integer,
    Numeric,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,
    as_numeric,
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


def as_ir_value(value, *, keep_static=False):
    """Convert any DslType value into a raw ``ir.Value``

    This is the *canonical* "DSL -> ir.Value" converter. Body code that
    needs to feed an MLIR builder should call this explicitly per argument.

    Behavior summary:
      - ``None``                                    -> ``None``
      - ``ir.Value``                                -> returned unchanged
      - ``Numeric`` holding a Python literal, when
        ``keep_static=True``                        -> returned unchanged
        ``keep_static=False``                       -> promoted via ``as_numeric(value).ir_value()``
      - ``tuple`` / ``list``                        -> recursed, shape preserved
      - object with ``__extract_to_ir_values__``    -> single value extracted; multi-value returns a list
      - ``bool`` / ``int`` / ``float``              -> promoted via ``as_numeric(value).ir_value()``
      - object with ``ir_value()``                  -> called as a fallback
      - anything else                               -> returned unchanged
    """
    if value is None:
        return None
    if isinstance(value, ir.Value):
        return value
    if keep_static and isinstance(value, Numeric) and not isinstance(value.value, ir.Value):
        return value
    if isinstance(value, tuple):
        return tuple(as_ir_value(v, keep_static=keep_static) for v in value)
    if isinstance(value, list):
        return [as_ir_value(v, keep_static=keep_static) for v in value]
    if hasattr(value, "__extract_to_ir_values__"):
        values = value.__extract_to_ir_values__()
        if len(values) == 1:
            return values[0]
        return values
    if isinstance(value, (bool, int, float)):
        return as_numeric(value).ir_value()
    if hasattr(value, "ir_value"):
        return value.ir_value()
    return value


def as_dsl_value(value, exemplar=None):
    """Wrap a raw ``ir.Value`` back into a DSL value. This is the inverse
    of :func:`as_ir_value` (``ir.Value -> DslType``).

    ``exemplar`` is an optional *type template* describing how to wrap ``value``:
      - a DslType class                         -> constructed directly via ``exemplar(value)``
      - a DslType instance                      -> ``type(exemplar)(value)``

    Behavior summary (mirrors the branches of :func:`as_ir_value`):
      - ``None``                                    -> ``None``
      - ``tuple`` / ``list``                        -> recursed, shape preserved,
        paired element-wise with ``exemplar`` (a non-sequence ``exemplar`` is
        broadcast to every element)
      - with no usable ``exemplar``: a ``value`` already satisfying the
        ``DslType`` protocol is returned unchanged; a bare scalar ``ir.Value``
        is dispatched by ``value.type`` via ``Numeric.from_ir_type``; any other
        non-``ir.Value`` is returned unchanged.

    Raises ``TypeError`` when a bare ``ir.Value`` cannot be wrapped into any DSL
    value.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        exemplars = exemplar if isinstance(exemplar, (tuple, list)) else [exemplar] * len(value)
        return type(value)(as_dsl_value(v, ex) for v, ex in zip(value, exemplars))

    if exemplar is not None and isinstance(value, ir.Value):
        if isclass(exemplar):
            return exemplar(value)
        if isinstance(exemplar, Numeric):
            return type(exemplar)(value)
        ctor = getattr(type(exemplar), "__construct_from_ir_values__", None)
        if ctor is not None:
            try:
                return ctor([value])
            except Exception:
                raise ValueError(f"failed to construct {type(exemplar)} from {value}")

    from ..compiler.protocol import DslType

    if isinstance(value, DslType):
        return value
    if not isinstance(value, ir.Value):
        return value
    try:
        return Numeric.from_ir_type(value.type)(value)
    except Exception as e:
        raise TypeError(f"as_dsl_value cannot wrap ir.Value of type {value.type!s} into a DSL value") from e


def _vec(n: int, elem: ir.Type) -> ir.Type:
    return ir.VectorType.get([int(n)], elem)


def default_f8_type() -> ir.Type:
    """Select E4M3 f8 type compatible with the current GPU arch.

    - gfx95* (MI350): FP8 E4M3FN (OCP)
    - gfx12*: FP8 E4M3FN (OCP)
    - gfx94* (MI300): FP8 E4M3FNUZ

    Raises ``RuntimeError`` on gfx11* (RDNA3/RDNA3.5): these chips have no
    native FP8 instructions, so FP8 compute would surface as a late LLVM
    "cannot select" error. Fail early with a clear message instead.
    """
    arch = ""
    try:
        arch = str(get_rocm_arch())
    except Exception:
        arch = ""
    if "gfx95" in arch or "gfx12" in arch:
        return Float8E4M3FN.ir_type
    if arch.startswith("gfx11"):
        raise RuntimeError(
            f"default_f8_type(): no native FP8 support on {arch}; "
            "FP8 instructions are available on gfx94*, gfx95*, and gfx12*. "
            "Use bf16/f16 GEMM via "
            "`rdna3_f16_gemm.create_wmma_gemm_module` on gfx11* targets."
        )
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

    @property
    def i128(self) -> ir.Type:
        return Int128.ir_type

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
    # DSL utilities
    "as_ir_value",
    "as_dsl_value",
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
    "Int128",
    "Index",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Uint128",
    "Constexpr",
    "IntTuple",
    "Layout",
    "Swizzle",
    "ComposedLayout",
    "Pointer",
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


class Constexpr:
    _annotation_cache: dict = {}
    _value_cache: dict = {}

    value_type: type | None = None
    value: Any = None
    is_specialized: bool = False

    @staticmethod
    def _type_name(param) -> str:
        if Constexpr._is_callable_annotation(param):
            return "Callable"
        return getattr(param, "__name__", repr(param))

    @staticmethod
    def _is_callable_annotation(param) -> bool:
        if param is Callable or param is AbcCallable:
            return True
        return get_origin(param) is AbcCallable

    @staticmethod
    def _is_tuple_annotation(param) -> bool:
        return param is tuple or get_origin(param) is tuple

    @staticmethod
    def _is_supported_annotation(param) -> bool:
        return param in (int, bool, float) or Constexpr._is_tuple_annotation(param)

    @staticmethod
    def _scalar_cache_signature(value):
        if type(value) is bool:
            return (bool, value)
        if type(value) is int:
            return (int, value)
        if type(value) is float:
            return (float, value)
        return None

    @staticmethod
    def _tuple_cache_signature(value):
        return ("tuple", tuple(Constexpr.value_signature(item) for item in value))

    @staticmethod
    def _lambda_cache_signature(value):
        if not isinstance(value, types.FunctionType) or value.__name__ != "<lambda>":
            return None
        if value.__code__.co_freevars or value.__closure__:
            raise TypeError("Constexpr lambda values must not capture free variables")
        global_refs = [name for name in value.__code__.co_names if name in value.__globals__]
        if global_refs:
            raise TypeError(f"Constexpr lambda values must not reference globals: {global_refs}")
        defaults = value.__defaults__ or ()
        kwdefaults = value.__kwdefaults__ or {}
        if kwdefaults:
            raise TypeError("Constexpr lambda values must not use keyword-only defaults")
        return (
            "lambda",
            value.__code__.co_argcount,
            value.__code__.co_posonlyargcount,
            value.__code__.co_kwonlyargcount,
            value.__code__.co_nlocals,
            value.__code__.co_stacksize,
            value.__code__.co_flags,
            value.__code__.co_code,
            tuple(Constexpr._lambda_const_cache_signature(item) for item in value.__code__.co_consts),
            value.__code__.co_names,
            value.__code__.co_varnames,
            tuple(Constexpr.value_signature(item) for item in defaults),
        )

    @staticmethod
    def _lambda_const_cache_signature(value):
        if value is None:
            return (type(None), None)
        return Constexpr.value_signature(value)

    @staticmethod
    def value_signature(value):
        scalar_sig = Constexpr._scalar_cache_signature(value)
        if scalar_sig is not None:
            return scalar_sig
        if isinstance(value, tuple):
            return Constexpr._tuple_cache_signature(value)
        lambda_sig = Constexpr._lambda_cache_signature(value)
        if lambda_sig is not None:
            return lambda_sig
        raise TypeError(
            "Constexpr values support only int, bool, float, tuples of those scalar values, "
            "and lambdas without free variables"
        )

    def __class_getitem__(cls, param):
        if cls is not Constexpr:
            raise TypeError(f"{cls.__name__} cannot be re-parametrized")
        if not Constexpr._is_supported_annotation(param) and not Constexpr._is_callable_annotation(param):
            raise TypeError(
                "Constexpr[...] supports only int, bool, float, tuple, or Callable annotations; " f"got {param!r}"
            )
        cached = Constexpr._annotation_cache.get(param)
        if cached is not None:
            return cached
        result = type(
            f"Constexpr[{Constexpr._type_name(param)}]",
            (Constexpr,),
            {
                "__origin__": Constexpr,
                "__args__": (param,),
                "value_type": param,
                "value": None,
                "is_specialized": False,
            },
        )
        Constexpr._annotation_cache[param] = result
        return result

    @classmethod
    def _specialize(cls, value):
        cache_key = Constexpr.value_signature(value)
        cached = Constexpr._value_cache.get(cache_key)
        if cached is not None:
            return cached
        result = type(
            f"Constexpr[{value!r}]",
            (Constexpr,),
            {
                "__origin__": Constexpr,
                "__args__": (type(value),),
                "value_type": type(value),
                "value": value,
                "is_specialized": True,
            },
        )
        Constexpr._value_cache[cache_key] = result
        return result

    @classmethod
    def __construct_from_ir_values__(cls, values):
        if values:
            raise ValueError(f"{cls.__name__} expects 0 ir.Values, got {len(values)}")
        if not cls.is_specialized:
            raise TypeError(
                f"{cls.__name__} must be value-specialized (e.g. Constexpr[42]) "
                f"before reconstruction; the surrounding schema did not bind a value."
            )
        return cls.value

    @classmethod
    def __extract_to_ir_values__(cls):
        return []

    @classmethod
    def __get_ir_types__(cls):
        return []

    def __c_abi_spec__(self):
        return []

    @classmethod
    def __coerce__(cls, value):
        inner = cls.value_type
        if inner is not None:
            if Constexpr._is_callable_annotation(inner):
                lambda_sig = Constexpr._lambda_cache_signature(value)
                if lambda_sig is None:
                    raise TypeError(f"expects lambda without free variables, got {type(value).__name__}")
            elif Constexpr._is_tuple_annotation(inner):
                if not isinstance(value, tuple):
                    raise TypeError(f"expects tuple, got {type(value).__name__}")
                Constexpr.value_signature(value)
            elif inner in (int, bool, float):
                if type(value) is not inner:
                    raise TypeError(f"expects {inner.__name__}, got {type(value).__name__}")
                Constexpr.value_signature(value)
            elif not isinstance(value, inner):
                raise TypeError(f"expects {getattr(inner, '__name__', repr(inner))}, got {type(value).__name__}")
            else:
                Constexpr.value_signature(value)
        return value

    @classmethod
    def __specialize_for_value__(cls, value):
        if cls.is_specialized:
            return cls
        return Constexpr._specialize(value)

    @staticmethod
    def is_constexpr_annotation(annotation) -> bool:
        if annotation is Constexpr:
            return True
        return isinstance(annotation, type) and issubclass(annotation, Constexpr)


class BuiltinDslType(ir.Value):
    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        type_str = self.type.__str__()
        return f"{type(self).__name__}{type_str[type_str.find('<') : type_str.rfind('>') + 1]}"

    def __repr__(self):
        return f"{type(self).__name__}<{super().__str__()}>"

    @classmethod
    def __construct_from_ir_values__(cls, values):
        return cls(values[0])

    def __extract_to_ir_values__(self):
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
            return next(leaf_iter)
        return tuple(get_(self, i)._rebuild_py_value(leaf_iter) for i in range(self.rank))

    @dsl_loc_tracing
    def to_py_value(self):
        if self.is_static:
            return IntTuple._static_to_py_value(self.type)
        leaves = get_leaves(self, dynamic_only=True)
        leaf_iter = iter(leaves)
        return self._rebuild_py_value(leaf_iter)

    @dsl_loc_tracing
    def __getitem__(self, mode):
        if isinstance(mode, int):
            mode = [mode]
        if self.rank <= mode[0]:
            raise IndexError(f"Index {mode[0]} out of range for int tuple with rank {self.rank}")
        return get_(self, mode)


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
    @dsl_loc_tracing
    def shape(self) -> IntTuple:
        return get_shape(self)

    @property
    @dsl_loc_tracing
    def stride(self) -> IntTuple:
        return get_stride(self)

    @dsl_loc_tracing
    def __getitem__(self, mode):
        if isinstance(mode, int):
            mode = [mode]
        if self.rank <= mode[0]:
            raise IndexError(f"Index {mode[0]} out of range for layout with rank {self.rank}")
        return get_(self, mode)

    @dsl_loc_tracing
    def __call__(self, *coord):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord)

        if has_none(coord):
            return slice(self, coord)
        else:
            return crd2idx(coord, self)

    @dsl_loc_tracing
    def get_hier_coord(self, index):
        return idx2crd(index, self)

    @dsl_loc_tracing
    def get_flat_coord(self, index):
        return get_flat_coord(index, self)

    @dsl_loc_tracing
    def get_1d_coord(self, index):
        return get_1d_coord(index, self)


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


@ir.register_value_caster(CoordSwizzleType.static_typeid, replace=True)
class CoordSwizzle(BuiltinDslType):
    @property
    def mask(self) -> int:
        return self.type.mask

    @property
    def base_row(self) -> int:
        return self.type.base_row

    @property
    def mode_row(self) -> list[int]:
        return self.type.mode_row

    @property
    def base_col(self) -> int:
        return self.type.base_col

    @property
    def mode_col(self) -> list[int]:
        return self.type.mode_col


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
    @dsl_loc_tracing
    def inner(self):
        return composed_get_inner(self)

    @property
    @dsl_loc_tracing
    def offset(self) -> IntTuple:
        return composed_get_offset(self)

    @property
    @dsl_loc_tracing
    def outer(self) -> "Layout | ComposedLayout":
        return composed_get_outer(self)

    @dsl_loc_tracing
    def __getitem__(self, mode):
        if isinstance(mode, int):
            mode = [mode]
        if self.rank <= mode[0]:
            raise IndexError(f"Index {mode[0]} out of range for composed layout with rank {self.rank}")
        return get_(self, mode)

    @dsl_loc_tracing
    def __call__(self, *coord):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord)

        if has_none(coord):
            return slice(self, coord)
        else:
            return crd2idx(coord, self)


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

    @dsl_loc_tracing
    def load(self):
        return ptr_load(self)

    @dsl_loc_tracing
    def store(self, value):
        if isinstance(value, (bool, int, float)):
            value = self.element_type(value)
        return ptr_store(value, self)

    @dsl_loc_tracing
    def __getitem__(self, offset):
        return (self + offset).load()

    @dsl_loc_tracing
    def __setitem__(self, offset, value):
        (self + offset).store(value)

    @dsl_loc_tracing
    def __add__(self, offset):
        return add_offset(self, offset)

    __radd__ = __add__

    @dsl_loc_tracing
    def __sub__(self, offset):
        if isinstance(offset, ir.Value) and not isinstance(offset, ArithValue):
            offset = ArithValue(offset)
        return add_offset(self, -offset)

    @dsl_loc_tracing
    def view(self, layout):
        return make_view(self, layout)


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

    @dsl_loc_tracing
    def __getitem__(self, coord):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord)

        if has_none(coord):
            return slice(self, coord)
        else:
            return memref_load(self, coord)

    @dsl_loc_tracing
    def __setitem__(self, coord, value):
        if not isinstance(coord, IntTuple):
            coord = make_int_tuple(coord)

        if has_none(coord):
            self.__getitem__(coord).store(value)
        else:
            memref_store(value, self, coord)

    @dsl_loc_tracing
    def load(self):
        return Vector(memref_load_vec(self), self.shape.to_py_value(), self.dtype)

    @dsl_loc_tracing
    def store(self, vector):
        return memref_store_vec(vector, self)

    @dsl_loc_tracing
    def fill(self, value):
        filled_vec = full(self.shape.to_py_value(), value, self.dtype)
        return self.store(filled_vec)


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
    def set_value(self, field: str, value): ...
    @overload
    def set_value(self, field: dict): ...

    @dsl_loc_tracing
    def set_value(self, field, value=None):
        if isinstance(field, dict):
            result = self
            for k, v in field.items():
                result = atom_set_value(result, k, v)
            return result
        return atom_set_value(self, field, value)


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
    def set_value(self, field: str, value): ...
    @overload
    def set_value(self, field: dict): ...

    @dsl_loc_tracing
    def set_value(self, field, value=None):
        if isinstance(field, dict):
            result = self
            for k, v in field.items():
                result = atom_set_value(result, k, v)
            return result
        return atom_set_value(self, field, value)


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

    @dsl_loc_tracing
    def make_fragment_A(self, a: Tensor, *, stages=None):
        return mma_make_fragment(MmaOperand.A, self, a, stages=stages)

    @dsl_loc_tracing
    def make_fragment_B(self, b: Tensor, *, stages=None):
        return mma_make_fragment(MmaOperand.B, self, b, stages=stages)

    @dsl_loc_tracing
    def make_fragment_C(self, c: Tensor, *, stages=None):
        return mma_make_fragment(MmaOperand.C, self, c, stages=stages)


class Stream:
    """Opaque async queue handle for kernel launch.

    ``None`` is the default queue; an :class:`int` is a raw pointer. Any other
    value is interpreted by the active device runtime
    (:mod:`flydsl.runtime.device_runtime`).
    """

    _is_stream_param = True

    def __init__(self, value=None):
        self.value = value

    def __get_ir_types__(self):
        return [gpu.AsyncTokenType.get()]

    def __cache_signature__(self):
        return (type(self),)

    def __c_abi_spec__(self):
        def fill(a, s):
            raw = a.value if hasattr(a, "_is_stream_param") else a
            if raw is None:
                s.value = 0
            elif isinstance(raw, int):
                s.value = raw
            elif hasattr(raw, "cuda_stream"):
                s.value = raw.cuda_stream
            else:
                raise ValueError(f"invalid stream value: {raw}")

        return [(ctypes.c_void_p, fill)]

    @classmethod
    def __construct_from_ir_values__(cls, values):
        return Stream(values[0])

    def __extract_to_ir_values__(self):
        return [self.value]


class Tuple3D:
    def __init__(self, factory, dtype=Int32):
        self.factory = factory
        self.dtype = dtype

    def __getattr__(self, name):
        if name in ("x", "y", "z"):
            from .meta import capture_user_location

            return self.dtype(self.factory(name, loc=capture_user_location()))
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

    def __extract_to_ir_values__(self):
        return [self]

    @classmethod
    def __construct_from_ir_values__(cls, values):
        return cls(values[0])

    def to(self, dtype: Type[Numeric]) -> "Vector":
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
            res = fp_to_fp(self, dtype.ir_type)
        elif src_float:
            res = fp_to_int(self, dtype.signed, dtype.ir_type)
        elif dst_float:
            res = int_to_fp(self, src_dtype.signed, dtype.ir_type)
        else:
            res = int_to_int(self, dtype)
        return Vector(res, self._shape, dtype)

    def ir_value(self):
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

    def _apply_op(self, method_name, op, other, flip=False):
        lhs = self
        rhs = other
        shape = self.shape
        if isinstance(other, Vector):
            shape = self._infer_broadcast_shape(self.shape, other.shape)
            lhs = self.broadcast_to(shape)
            rhs = other.broadcast_to(shape)
        method = getattr(ArithValue, method_name)
        if flip:
            if isinstance(rhs, Vector):
                result = method(rhs, lhs)
            else:
                reverse_name = _VECTOR_REVERSE_OP_METHODS.get(method_name, method_name)
                result = getattr(ArithValue, reverse_name)(lhs, rhs)
        else:
            result = method(lhs, rhs)
        return self._wrap_op_result(result, shape)

    def apply_op(self, op, other, flip=False):
        method_name = _VECTOR_OP_METHODS.get(op)
        if method_name is None:
            raise NotImplementedError(f"Vector.apply_op does not support {op}")
        return self._apply_op(method_name, op, other, flip=flip)

    def __add__(self, other):
        return self.apply_op(operator.add, other)

    def __radd__(self, other):
        return self.apply_op(operator.add, other, flip=True)

    def __sub__(self, other):
        return self.apply_op(operator.sub, other)

    def __rsub__(self, other):
        return self.apply_op(operator.sub, other, flip=True)

    def __mul__(self, other):
        return self.apply_op(operator.mul, other)

    def __rmul__(self, other):
        return self.apply_op(operator.mul, other, flip=True)

    def __truediv__(self, other):
        return self.apply_op(operator.truediv, other)

    def __rtruediv__(self, other):
        return self.apply_op(operator.truediv, other, flip=True)

    def __floordiv__(self, other):
        return self.apply_op(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return self.apply_op(operator.floordiv, other, flip=True)

    def __mod__(self, other):
        return self.apply_op(operator.mod, other)

    def __rmod__(self, other):
        return self.apply_op(operator.mod, other, flip=True)

    def __pow__(self, other):
        return self.apply_op(operator.pow, other)

    def __rpow__(self, other):
        return self.apply_op(operator.pow, other, flip=True)

    def __lshift__(self, other):
        return self.apply_op(operator.lshift, other)

    def __rlshift__(self, other):
        return self.apply_op(operator.lshift, other, flip=True)

    def __rshift__(self, other):
        return self.apply_op(operator.rshift, other)

    def __rrshift__(self, other):
        return self.apply_op(operator.rshift, other, flip=True)

    def __and__(self, other):
        return self.apply_op(operator.and_, other)

    def __rand__(self, other):
        return self.apply_op(operator.and_, other, flip=True)

    def __or__(self, other):
        return self.apply_op(operator.or_, other)

    def __ror__(self, other):
        return self.apply_op(operator.or_, other, flip=True)

    def __xor__(self, other):
        return self.apply_op(operator.xor, other)

    def __rxor__(self, other):
        return self.apply_op(operator.xor, other, flip=True)

    def __lt__(self, other):
        return self.apply_op(operator.lt, other)

    def __le__(self, other):
        return self.apply_op(operator.le, other)

    def __gt__(self, other):
        return self.apply_op(operator.gt, other)

    def __ge__(self, other):
        return self.apply_op(operator.ge, other)

    def __eq__(self, other):
        return self.apply_op(operator.eq, other)

    def __ne__(self, other):
        return self.apply_op(operator.ne, other)

    @dsl_loc_tracing
    def reduce(self, op, init_val=None, reduction_profile=None, *, fastmath=None):
        is_fp = self._dtype.is_float
        signed = getattr(self._dtype, "signed", True)
        kind = _resolve_combining_kind(op, is_fp, signed)
        et = element_type(self.type)
        kwargs = {}
        if fastmath is not None:
            kwargs["fastmath"] = fastmath
        if init_val is not None:
            if isinstance(init_val, Numeric):
                init_val = init_val.ir_value()
            kwargs["acc"] = _to_raw(init_val)
        res = _vector.reduction(et, kind, self, **kwargs)
        return self._dtype(res)

    @staticmethod
    def _coerce_element(element, dtype: Type[Numeric]):
        if isinstance(element, (int, float, bool)):
            return dtype(element)
        if isinstance(element, Numeric):
            return element.to(dtype)
        if isinstance(element, ir.Value):
            return Numeric.from_ir_type(element.type)(element).to(dtype)
        if hasattr(element, "ir_value"):
            value = element.ir_value()
            return Numeric.from_ir_type(value.type)(value).to(dtype)
        raise ValueError(f"expected numeric vector element, got {type(element)}")

    @dsl_loc_tracing
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

    def _build_result(self, value, shape, *, row_major=False) -> "Vector":
        shape = self._canonical_shape(shape)
        flat_ty = self.make_type(shape, self._dtype)
        flat_value = _vector.shape_cast(flat_ty, value)
        return Vector(flat_value, shape, self._dtype)

    def reshape(self, shape) -> "Vector":
        shape = self._canonical_shape(shape)
        if self.numel != self._numel_from_shape(shape):
            raise ValueError(f"expected reshaped size to match: {self._shape} -> {shape}")
        return Vector(self, shape, self._dtype)

    @dsl_loc_tracing
    def broadcast_to(self, target_shape) -> "Vector":
        target_shape = self._canonical_shape(target_shape)
        if self._shape == target_shape:
            return self
        src_flat_shape = self._flatten_static(self._shape)
        target_flat_shape = self._flatten_static(target_shape)
        if self.numel == 1:
            scalar = self[0].ir_value()
            target_ty = self.make_type(target_shape, self._dtype)
            res = _vector.broadcast(target_ty, scalar)
            return Vector(res, target_shape, self._dtype)
        if len(src_flat_shape) > len(target_flat_shape):
            raise ValueError(f"cannot broadcast shape {self._shape} to {target_shape}")
        padded_src = [1] * (len(target_flat_shape) - len(src_flat_shape)) + src_flat_shape
        for src_dim, dst_dim in zip(padded_src, target_flat_shape, strict=True):
            if src_dim != dst_dim and src_dim != 1:
                raise ValueError(f"cannot broadcast shape {self._shape} to {target_shape}")
        src_ty = ir.VectorType.get(padded_src, self._dtype.ir_type)
        src = _vector.shape_cast(src_ty, self)
        target_ty_nd = ir.VectorType.get(target_flat_shape, self._dtype.ir_type)
        res = _vector.broadcast(target_ty_nd, src)
        return self._build_result(res, target_shape, row_major=True)

    @dsl_loc_tracing
    def bitcast(self, dtype: Type[Numeric]) -> "Vector":
        src_bits = self.numel * self._dtype.width
        dst_count = src_bits // dtype.width
        dst_vec_ty = ir.VectorType.get([dst_count], dtype.ir_type)
        res = _vector.BitCastOp(dst_vec_ty, self).result
        return Vector(res, (dst_count,), dtype)

    @dsl_loc_tracing
    def shuffle(self, other, mask) -> "Vector":
        other_val = other if not isinstance(other, Vector) else ir.Value(other)
        res = _vector.shuffle(self, other_val, mask)
        return Vector(res, (len(mask),), self._dtype)

    @classmethod
    @dsl_loc_tracing
    def from_elements(cls, elements, dtype: Type[Numeric] | None = None) -> "Vector":
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
                dtype = Numeric.from_ir_type(first.ir_value().type)
            else:
                dtype = type(Numeric.from_python_value(first))
        vec_ty = cls.make_type(len(elements), dtype)
        raw_elements = [_to_raw(cls._coerce_element(element, dtype)) for element in elements]
        res = _vector.from_elements(vec_ty, raw_elements)
        return cls(res, (len(elements),), dtype)

    @classmethod
    @dsl_loc_tracing
    def load(cls, result_type, memref, indices) -> "Vector":
        vty = ir.VectorType(result_type)
        dtype = Numeric.from_ir_type(vty.element_type)
        raw_indices = []
        for index in indices:
            if isinstance(index, int):
                index = Index(index)
            elif not isinstance(index, ir.Value) and not hasattr(index, "ir_value"):
                index = Index(index)
            raw_indices.append(_to_raw(index))
        res = _vector.LoadOp(result_type, _to_raw(memref), raw_indices).result
        return cls(res, tuple(vty.shape), dtype)

    @dsl_loc_tracing
    def store(self, memref, indices, *, alignment=None):
        raw_indices = []
        for index in indices:
            if isinstance(index, int):
                index = Index(index)
            elif not isinstance(index, ir.Value) and not hasattr(index, "ir_value"):
                index = Index(index)
            raw_indices.append(_to_raw(index))
        kwargs = {}
        if alignment is not None:
            kwargs["alignment"] = alignment
        return _vector.store(_to_raw(self), _to_raw(memref), raw_indices, **kwargs)

    @classmethod
    @dsl_loc_tracing
    def filled(cls, shape, fill_value, dtype: Type[Numeric]) -> "Vector":
        shape = cls._canonical_shape(shape)
        n = cls._numel_from_shape(shape)
        if isinstance(fill_value, (int, float, bool)):
            fill_value = dtype(fill_value)
        elif isinstance(fill_value, Numeric):
            fill_value = fill_value.to(dtype)
        else:
            raise ValueError(f"expected numeric fill_value, got {type(fill_value)}")
        vec_ty = cls.make_type(n, dtype)
        val = _vector.broadcast(vec_ty, fill_value.ir_value())
        return cls(val, shape, dtype)

    @classmethod
    def filled_like(cls, template: "Vector", fill_value, dtype=None) -> "Vector":
        if dtype is None:
            dtype = template.dtype
        return cls.filled(template.shape, fill_value, dtype)

    @classmethod
    def zeros_like(cls, template: "Vector", dtype=None) -> "Vector":
        if dtype is None:
            dtype = template.dtype
        return cls.filled(template.shape, 0.0 if dtype.is_float else 0, dtype)


def full(shape, fill_value, dtype: Type[Numeric]) -> Vector:
    return Vector.filled(shape, fill_value, dtype)


def full_like(a: Vector, fill_value, dtype=None) -> Vector:
    return Vector.filled_like(a, fill_value, dtype)


def empty_like(a: Vector, dtype=None) -> Vector:
    return Vector.filled_like(a, 0, dtype)


def ones_like(a: Vector, dtype=None) -> Vector:
    return Vector.filled_like(a, 1, dtype)


def zeros_like(a: Vector, dtype=None) -> Vector:
    return Vector.zeros_like(a, dtype)


class Array:
    _cache: dict[tuple, type] = {}

    class _Base:
        dtype = None
        size = None
        align = None

        def __init__(self, ptr_value):
            self._ptr_value = ptr_value

        def __repr__(self):
            cls = type(self)
            name = getattr(cls.dtype, "__name__", repr(cls.dtype))
            suffix = f", {cls.align}" if cls.align != max(1, cls.dtype.width // 8) else ""
            return f"Array[{name}, {cls.size}{suffix}]({self._ptr_value})"

        @property
        def ptr(self):
            return self._ptr_value

        @classmethod
        def __construct_from_ir_values__(cls, values):
            if len(values) != 1:
                raise ValueError(f"{cls.__name__} expects 1 ir.Value, got {len(values)}")
            return cls(values[0])

        def __extract_to_ir_values__(self) -> List[ir.Value]:
            return [self._ptr_value]

        @classmethod
        def __dsl_size_of__(cls) -> int:
            total_bytes = max(1, cls.dtype.width * cls.size // 8)
            return total_bytes

        @classmethod
        def __dsl_align_of__(cls) -> int:
            return cls.align

        @classmethod
        def __peek_from_ptr__(cls, ptr):
            typed_ptr = recast_iter(cls.dtype, ptr)
            return cls(typed_ptr)

        @classmethod
        def __poke_into_ptr__(cls, ptr, value):
            raise NotImplementedError(f"{cls.__name__} does not support __poke_into_ptr__ yet")

        @dsl_loc_tracing
        def __getitem__(self, offset):
            return self.ptr.__getitem__(offset)

        @dsl_loc_tracing
        def __setitem__(self, offset, value):
            self.ptr.__setitem__(offset, value)

        def view(self, layout):
            return make_view(self._ptr_value, layout)

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) == 2:
            dtype, size = params
            align = None
        elif len(params) == 3:
            dtype, size, align = params
        else:
            raise TypeError("Array expects Array[dtype, size] or Array[dtype, size, align]")

        if not (isinstance(dtype, type) and issubclass(dtype, Numeric)):
            raise TypeError(f"Array dtype must be a Numeric subclass, got {dtype!r}")
        if not isinstance(size, int) or size <= 0:
            raise TypeError(f"Array size must be a positive integer, got {size!r}")

        elem_byte_size = max(1, dtype.width // 8)
        if align is None:
            align = elem_byte_size
        else:
            if not isinstance(align, int) or align <= 0:
                raise TypeError(f"Array align must be a positive integer, got {align!r}")

        cache_key = (dtype, size, align)
        cached = cls._cache.get(cache_key)
        if cached is not None:
            return cached

        name = getattr(dtype, "__name__", repr(dtype))
        suffix = f", {align}" if align != elem_byte_size else ""
        array_type = type(
            f"Array[{name}, {size}{suffix}]",
            (cls._Base,),
            {"dtype": dtype, "size": size, "align": align},
        )
        cls._cache[cache_key] = array_type
        return array_type
