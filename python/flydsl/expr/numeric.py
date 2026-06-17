# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import operator
from typing import Type

import numpy as np

from .._mlir import ir
from .._mlir.dialects import arith
from .._mlir.extras import types as T
from .meta import dsl_loc_tracing
from .utils.arith import (
    ArithValue,
    _to_raw,
    arith_const,
    fp_to_fp,
    fp_to_int,
    index_cast,
    int_to_fp,
    int_to_int,
    is_float_type,
)


def _infer_np_dtype(width, signed, name):
    if signed is not None:
        if width == 1:
            return np.bool_
        elif width < 8:
            return None
        elif signed:
            return getattr(np, f"int{width}", None)
        return getattr(np, f"uint{width}", None)
    return getattr(np, name.lower(), None)


class NumericMeta(type):
    width: int
    log_width: int
    _ir_type = None
    _np_dtype = None

    def __new__(
        cls,
        name,
        bases,
        attrs,
        width=8,
        np_dtype=None,
        ir_type=None,
        signed=None,
        zero=None,
        **kwargs,
    ):
        def _extract_to_ir_values(self):
            return [self.ir_value()]

        def _construct_from_ir_values(cls, values):
            return cls(values[0])

        inferred_np = np_dtype if np_dtype is not None else _infer_np_dtype(width, signed, name)
        is_storable = width >= 8

        def _dsl_size_of(cls):
            return 1 if cls.width < 8 else (cls.width + 7) // 8

        def _dsl_align_of(cls):
            return 1 if cls.width < 8 else (cls.width + 7) // 8

        def _peek_from_ptr(cls, ptr):
            from .primitive import ptr_load, recast_iter

            typed_ptr = recast_iter(cls, ptr)
            return cls(ptr_load(typed_ptr, cls))

        def _poke_into_ptr(cls, ptr, value):
            from .primitive import ptr_store, recast_iter

            typed_ptr = recast_iter(cls, ptr)
            coerced = value.to(cls) if isinstance(value, Numeric) else cls(value)
            ptr_store(coerced.ir_value(), typed_ptr)

        def _not_storable(cls):
            raise TypeError(f"sub-byte type {cls.__name__} (width={cls.width}) is not Storable")

        new_attrs = {
            "__extract_to_ir_values__": _extract_to_ir_values,
            "__construct_from_ir_values__": classmethod(_construct_from_ir_values),
        }
        if is_storable:
            new_attrs["__dsl_size_of__"] = classmethod(_dsl_size_of)
            new_attrs["__dsl_align_of__"] = classmethod(_dsl_align_of)
            new_attrs["__peek_from_ptr__"] = classmethod(_peek_from_ptr)
            new_attrs["__poke_into_ptr__"] = classmethod(_poke_into_ptr)
        elif any(hasattr(base, "__dsl_size_of__") for base in bases):
            new_attrs["__dsl_size_of__"] = classmethod(_not_storable)
            new_attrs["__dsl_align_of__"] = classmethod(_not_storable)
            new_attrs["__peek_from_ptr__"] = classmethod(_not_storable)
            new_attrs["__poke_into_ptr__"] = classmethod(lambda cls, ptr, value: _not_storable(cls))
        if signed is not None:

            def __c_abi_spec__(self):
                w = self.width
                ct = ctypes.c_bool if w == 1 else getattr(ctypes, f"c_{'int' if self.signed else 'uint'}{w}", None)
                if ct is None:
                    raise TypeError(
                        f"{type(self).__name__} (width={w}) has no host C-ABI and cannot be a launch argument"
                    )

                def fill(a, s):
                    # ``a`` may be a bare Python int (fast path) or a Numeric
                    # instance (slow path); read via the ``hasattr(a, "value")`` shim.
                    s.value = a.value if hasattr(a, "value") else a

                return [(ct, fill)]

            new_attrs["__c_abi_spec__"] = __c_abi_spec__

        new_cls = super().__new__(cls, name, bases, new_attrs | attrs)
        if ir_type is not None:
            new_cls._ir_type = staticmethod(ir_type)
        new_cls.width = width
        new_cls.log_width = (width - 1).bit_length()
        new_cls._np_dtype = inferred_np
        new_cls.signed = signed
        new_cls._zero = zero
        if signed is not None:
            prefix = "c_int" if signed else "c_uint"
            ctype = getattr(ctypes, f"{prefix}{width}", None)
            if ctype is not None:
                new_cls._reusable_ctype = ctype
        return new_cls

    def __str__(cls):
        return f"{cls.__name__}"

    @property
    def numpy_dtype(cls):
        return cls._np_dtype

    @property
    def ir_type(cls):
        if cls._ir_type is not None:
            return cls._ir_type()
        return None

    @property
    def is_integer(cls) -> bool:
        return cls.signed is not None

    @property
    def is_float(cls) -> bool:
        return cls.signed is None and cls._ir_type is not None

    @property
    def zero(cls):
        if cls._zero is not None:
            return cls._zero
        elif cls.is_integer:
            return 0
        elif cls.is_float:
            return 0.0
        else:
            raise ValueError(f"no zero value for {cls}")


_CMP_OPS = frozenset({operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne})


def _widen_bool_to_int32(x, widen_bool=False):
    """Promote Boolean to Int32 for arithmetic when widen_bool=True.

    Per C++-style usual arithmetic conversions, we deliberately do NOT apply
    integer promotion: i8/i16/u8/u16 stay at their narrow width.
    Same-width same-signedness operands keep their type; cross-width or
    cross-sign mixing is resolved by ``_coerce_operands``.
    """
    if widen_bool and type(x) is Boolean:
        return x.to(Int32), Int32
    return x, type(x)


def _resolve_float_type(ta, tb):
    """Pick the wider float type, or the one with higher rank at equal width."""
    # Use module-level _FLOAT_RANK (defined after all classes)
    if ta.is_float and not tb.is_float:
        return ta
    if tb.is_float and not ta.is_float:
        return tb
    wa, wb = ta.width, tb.width
    if wa != wb:
        wider = ta if wa > wb else tb
        if wider.width >= 16:
            return wider
    ra = _FLOAT_RANK.get(ta, 0)
    rb = _FLOAT_RANK.get(tb, 0)
    if ra >= rb and ra > 0:
        return ta
    if rb > ra:
        return tb
    raise ValueError(f"no common float type for {ta} and {tb}; cast explicitly")


def _coerce_operands(a, b, widen_bool=False):
    """Promote *a* and *b* to a common scalar type."""
    ta, tb = type(a), type(b)
    a, ta = _widen_bool_to_int32(a, widen_bool=widen_bool)
    b, tb = _widen_bool_to_int32(b, widen_bool=widen_bool)

    if ta is tb:
        return a, b, ta

    if ta.is_float or tb.is_float:
        dest = _resolve_float_type(ta, tb)
        return (a if type(a) is dest else a.to(dest), b if type(b) is dest else b.to(dest), dest)

    # Both integers — pick wider; on tie, prefer unsigned when mixed sign
    if ta.signed == tb.signed:
        wider = ta if ta.width >= tb.width else tb
        return (a if type(a) is wider else a.to(wider), b if type(b) is wider else b.to(wider), wider)

    u, s = (ta, tb) if not ta.signed else (tb, ta)
    dest = u if u.width >= s.width else s
    return (a if type(a) is dest else a.to(dest), b if type(b) is dest else b.to(dest), dest)


def _try_coerce_rhs(rhs):
    """Try converting *rhs* to a Numeric; return None on failure."""
    if isinstance(rhs, Numeric):
        return rhs
    if isinstance(rhs, ArithValue):
        if isinstance(rhs.type, ir.VectorType):
            return None
        if isinstance(rhs.type, ir.IndexType):
            return Index(rhs)
        try:
            return Numeric.from_ir_type(rhs.type)(rhs)
        except (ValueError, KeyError):
            return None
    if isinstance(rhs, (int, float, bool)):
        return as_numeric(rhs)
    return None


def _extract_arith(val, signed):
    """Unwrap Numeric.value, attaching signedness if it's an ArithValue."""
    v = val.value
    return v.with_signedness(signed) if isinstance(v, ArithValue) else v


def _make_binop(op, promote=True, widen_bool=False, swap=False):
    """Create a binary-operator closure for Numeric subclasses."""

    def _apply(lhs, rhs):
        rhs = _try_coerce_rhs(rhs)
        if rhs is None:
            return NotImplemented

        out_type = type(lhs)
        if promote:
            lhs, rhs, out_type = _coerce_operands(lhs, rhs, widen_bool)
        else:
            rhs = type(lhs)(rhs)

        if op in _CMP_OPS:
            out_type = Boolean
        elif op is operator.truediv and isinstance(lhs, Integer):
            out_type = Float64 if out_type.width > 32 else Float32

        lv, rv = _extract_arith(lhs, lhs.signed), _extract_arith(rhs, rhs.signed)
        if swap:
            lv, rv = rv, lv
        return out_type(op(lv, rv))

    return _apply


class Numeric(metaclass=NumericMeta):
    def __init__(self, value):
        self.value = value

    def __str__(self) -> str:
        return "?"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.value)})"

    def __hash__(self):
        return hash(type(self)) ^ hash(self.value)

    def select(self, true_value, false_value):
        """Ternary select (for Boolean conditions from Int32 comparisons)."""
        from .typing import as_dsl_value

        result = ArithValue(self).select(true_value, false_value)
        return as_dsl_value(result, true_value)

    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except Exception:
            raise TypeError(f"expects {cls.__name__}, got {type(value).__name__}")

    @property
    def dtype(self) -> Type["Numeric"]:
        return type(self)

    def to(self, dtype):
        if dtype is type(self):
            return self
        elif isinstance(dtype, type) and issubclass(dtype, Numeric):
            return dtype(self)
        elif dtype is ir.Value:
            if isinstance(self.value, (int, float, bool)):
                return arith_const(self.value, type(self).ir_type)
            elif isinstance(self.value, ir.Value):
                res = self.value
                if not isinstance(res, ArithValue):
                    raise ValueError(f"expected ArithValue, got {type(res)}")
                return res.with_signedness(getattr(type(self), "signed", None))
            else:
                raise ValueError(f"cannot convert {type(self)} to {dtype}")
        elif dtype in (int, float, bool):
            if isinstance(self.value, ir.Value):
                raise ValueError(f"dynamic IR value cannot be materialized as {dtype}")
            return dtype(self.value)
        else:
            raise ValueError(f"unable to convert {type(self)} to {dtype}")

    def ir_value(self) -> ir.Value:
        return self.to(ir.Value)

    def __get_ir_types__(self):
        return [type(self).ir_type]

    def __cache_signature__(self):
        return (type(self),)

    def __neg__(self):
        if isinstance(self.value, (bool, int, float)):
            return type(self)(-self.value)
        return type(self)(-self.value)

    def __fly_bool__(self):
        if isinstance(self.value, (int, float, bool)):
            return Boolean(bool(self.value))
        zero = arith_const(type(self).zero, type(self).ir_type)
        return self.__ne__(type(self)(zero))

    def __fly_not__(self):
        b = self.__fly_bool__()
        if isinstance(b.value, bool):
            return Boolean(not b.value)
        zero = arith_const(0, T.bool())
        return Boolean(b.ir_value().__eq__(zero))

    def __fly_and__(self, other):
        lhs = self.__fly_bool__()
        rhs = as_numeric(other).__fly_bool__()
        if isinstance(lhs.value, bool) and isinstance(rhs.value, bool):
            return Boolean(lhs.value and rhs.value)
        return Boolean(lhs.ir_value().__and__(rhs.ir_value()))

    def __fly_or__(self, other):
        lhs = self.__fly_bool__()
        rhs = as_numeric(other).__fly_bool__()
        if isinstance(lhs.value, bool) and isinstance(rhs.value, bool):
            return Boolean(lhs.value or rhs.value)
        return Boolean(lhs.ir_value().__or__(rhs.ir_value()))

    def __bool__(self):
        if isinstance(self.value, (int, float, bool)):
            return bool(self.value)
        raise RuntimeError(f"cannot evaluate dynamic '{type(self).__name__}' as Python bool during tracing")

    def __index__(self):
        if isinstance(self.value, (int, float, bool)):
            return int(self.value)
        raise RuntimeError(f"dynamic '{type(self.value).__name__}' has no Python integer representation")

    @staticmethod
    def from_python_value(value):
        if isinstance(value, Numeric):
            return value
        elif isinstance(value, ArithValue):
            return Numeric.from_ir_type(value.type)(value)
        elif isinstance(value, bool):
            return Boolean(value)
        elif isinstance(value, int):
            return Int32(value) if -2147483648 <= value <= 2147483647 else Int64(value)
        elif isinstance(value, float):
            return Float32(value)
        raise ValueError(f"cannot convert {value} ({type(value)}) to Numeric")

    @staticmethod
    def from_ir_type(ir_type):
        ir2dsl_map = {
            T.bool(): Boolean,
            T.f64(): Float64,
            T.f32(): Float32,
            T.f16(): Float16,
            T.bf16(): BFloat16,
            T.i64(): Int64,
            T.i32(): Int32,
            T.i16(): Int16,
            T.i8(): Int8,
            T.si64(): Int64,
            T.si32(): Int32,
            T.si16(): Int16,
            T.si8(): Int8,
            T.ui64(): Uint64,
            T.ui32(): Uint32,
            T.ui16(): Uint16,
            T.ui8(): Uint8,
            T.i(128): Int128,
            T.si(128): Int128,
            T.ui(128): Uint128,
            T.f8E5M2(): Float8E5M2,
            T.f8E4M3(): Float8E4M3,
            T.f8E4M3FN(): Float8E4M3FN,
            Float8E4M3FNUZ.ir_type: Float8E4M3FNUZ,  # not in upstream MLIR extras T
            T.f8E4M3B11FNUZ(): Float8E4M3B11FNUZ,
            T.f8E8M0FNU(): Float8E8M0FNU,
            T.f6E2M3FN(): Float6E2M3FN,
            T.f6E3M2FN(): Float6E3M2FN,
            T.f4E2M1FN(): Float4E2M1FN,
        }
        # Handle IndexType specially since it maps to Index
        if isinstance(ir_type, ir.IndexType):
            return Index
        if ir_type not in ir2dsl_map:
            raise ValueError(f"unsupported mlir type: {ir_type}")
        return ir2dsl_map[ir_type]

    def __add__(self, other):
        return _make_binop(operator.add, widen_bool=True)(self, other)

    def __sub__(self, other):
        return _make_binop(operator.sub, widen_bool=True)(self, other)

    def __mul__(self, other):
        return _make_binop(operator.mul, widen_bool=True)(self, other)

    def __floordiv__(self, other):
        return _make_binop(operator.floordiv, widen_bool=True)(self, other)

    def __truediv__(self, other):
        return _make_binop(operator.truediv, widen_bool=True)(self, other)

    def __mod__(self, other):
        return _make_binop(operator.mod, widen_bool=True)(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return _make_binop(operator.sub, widen_bool=True, swap=True)(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rfloordiv__(self, other):
        return _make_binop(operator.floordiv, widen_bool=True, swap=True)(self, other)

    def __rtruediv__(self, other):
        return _make_binop(operator.truediv, widen_bool=True, swap=True)(self, other)

    def __rmod__(self, other):
        return _make_binop(operator.mod, widen_bool=True, swap=True)(self, other)

    def __pow__(self, other):
        return _make_binop(operator.pow)(self, other)

    def __eq__(self, other):
        return _make_binop(operator.eq)(self, other)

    def __ne__(self, other):
        return _make_binop(operator.ne)(self, other)

    # ── Proxy methods: delegate ArithValue-specific ops via ir_value() ──
    def maximumf(self, other):
        """Float maximum — delegates to ArithValue.maximumf."""
        return type(self)(self.ir_value().maximumf(_to_raw(other)))

    def minimumf(self, other):
        """Float minimum — delegates to ArithValue.minimumf."""
        return type(self)(self.ir_value().minimumf(_to_raw(other)))

    def exp2(self, *, fastmath=None):
        """Base-2 exponential — delegates to ArithValue.exp2."""
        return type(self)(self.ir_value().exp2(fastmath=fastmath))

    def shuffle_xor(self, offset, width):
        """GPU warp shuffle XOR — delegates to ArithValue.shuffle_xor."""
        return type(self)(self.ir_value().shuffle_xor(offset, width))

    def shrui(self, amount):
        """Unsigned right shift — delegates to ArithValue.shrui."""
        return type(self)(self.ir_value().shrui(amount))

    def addf(self, other, *, fastmath=None):
        """Float add with fastmath — delegates to ArithValue.addf."""
        return type(self)(self.ir_value().addf(_to_raw(other), fastmath=fastmath))

    def __lt__(self, other):
        return _make_binop(operator.lt)(self, other)

    def __le__(self, other):
        return _make_binop(operator.le)(self, other)

    def __gt__(self, other):
        return _make_binop(operator.gt)(self, other)

    def __ge__(self, other):
        return _make_binop(operator.ge)(self, other)

    @dsl_loc_tracing
    def bitcast(self, dtype):
        """Reinterpret this value's bits as *dtype* (a same-width Numeric type)."""
        if not (isinstance(dtype, type) and issubclass(dtype, Numeric)):
            raise TypeError(f"dtype must be a Numeric subclass, but got {dtype!r}")
        res = arith.bitcast(dtype.ir_type, self.ir_value())
        return dtype(res)


def as_numeric(obj):
    if isinstance(obj, Numeric):
        return obj
    return Numeric.from_python_value(obj)


class Integer(Numeric, metaclass=NumericMeta, width=32, signed=True, ir_type=T.i32):
    def __init__(self, x):
        ty = type(self)

        if isinstance(x, (bool, int, float)):
            if isinstance(x, float):
                if np.isnan(x):
                    raise ValueError("float NaN is not representable as integer")
                elif np.isinf(x):
                    raise OverflowError("float infinity is not representable as integer")
            np_dtype = ty.numpy_dtype
            if np_dtype is not None:
                x_val = int(np.array(x).astype(np_dtype))
            else:
                x_val = int(x)
        elif type(x) is ty:
            x_val = x.value
        elif isinstance(x, ir.Value):
            x_val = x
            if isinstance(x.type, ir.IndexType):
                x_val = index_cast(ty.ir_type, x)
            elif isinstance(x.type, ir.IntegerType):
                if x.type.width != ty.width:
                    x_val = int_to_int(x, ty, signed=ty.signed)
            elif is_float_type(x.type):
                x_val = fp_to_int(x, ty.signed, ty.ir_type)
        elif isinstance(x, Integer):
            if isinstance(x.value, ir.Value):
                raw = x.ir_value()
                if isinstance(raw.type, ir.IndexType):
                    x_val = index_cast(ty.ir_type, raw)
                else:
                    x_val = int_to_int(raw, ty)
            else:
                src_dtype = type(x).numpy_dtype
                dst_dtype = ty.numpy_dtype
                if src_dtype is not None and dst_dtype is not None:
                    x_val = int(np.array(x.value, dtype=src_dtype).astype(dst_dtype))
                else:
                    x_val = int(x.value)
        elif isinstance(x, Float):
            Integer.__init__(self, x.value)
            return
        else:
            raise ValueError(f"{x} to integer conversion is not supported")

        super().__init__(x_val)

    def __invert__(self):
        res_type = type(self)
        return res_type(self.ir_value().__invert__())

    def __lshift__(self, other):
        return _make_binop(operator.lshift)(self, other)

    def __rlshift__(self, other):
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"left-shift requires integer operands, got {other_}")
        return other_.__lshift__(self)

    def __rshift__(self, other):
        return _make_binop(operator.rshift)(self, other)

    def __rrshift__(self, other):
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"right-shift requires integer operands, got {other_}")
        return other_.__rshift__(self)

    def __and__(self, other):
        return _make_binop(operator.and_)(self, other)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return _make_binop(operator.or_)(self, other)

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        return _make_binop(operator.xor)(self, other)

    def __rxor__(self, other):
        return self.__xor__(other)

    def is_static(self):
        return not isinstance(self.value, ir.Value)


class Float(Numeric, metaclass=NumericMeta, width=32, ir_type=T.f32):
    def __init__(self, x):
        ty = type(self)

        if isinstance(x, (bool, int, float)):
            super().__init__(float(x))
        elif isinstance(x, ir.Value):
            if isinstance(x.type, ir.IntegerType):
                raise ValueError("bare signless integer cannot be promoted to float; use a typed wrapper")
            elif is_float_type(x.type):
                if x.type != ty.ir_type:
                    x = fp_to_fp(x, ty.ir_type)
            super().__init__(x)
        elif isinstance(x, Integer):
            if isinstance(x.value, ir.Value):
                x = int_to_fp(x.value, type(x).signed, ty.ir_type)
            else:
                x = float(x.value)
            super().__init__(x)
        elif isinstance(x, Float):
            Float.__init__(self, x.value)
        else:
            raise ValueError(f"{x} to float conversion is not supported")


class Boolean(Integer, metaclass=NumericMeta, width=1, signed=True, ir_type=T.bool):
    def __init__(self, a):
        value = None
        if isinstance(a, (bool, int, float)):
            value = bool(a)
        elif isinstance(a, Numeric):
            Boolean.__init__(self, a.value)
            return
        elif isinstance(a, ArithValue):
            if a.type == T.bool():
                value = a
            else:
                value = a != arith_const(0, a.type)
        if value is None:
            raise ValueError(f"no Boolean coercion defined for {a}")
        super().__init__(value)

    def __neg__(self):
        raise TypeError("unary minus is undefined for booleans")


class Int4(Integer, metaclass=NumericMeta, width=4, signed=True, ir_type=lambda: T.IntegerType.get_signless(4)):
    pass


class Int8(Integer, metaclass=NumericMeta, width=8, signed=True, ir_type=T.i8):
    pass


class Int16(Integer, metaclass=NumericMeta, width=16, signed=True, ir_type=T.i16):
    pass


class Int32(Integer, metaclass=NumericMeta, width=32, signed=True, ir_type=T.i32):
    pass


class Int64(Integer, metaclass=NumericMeta, width=64, signed=True, ir_type=T.i64):
    pass


class Int128(Integer, metaclass=NumericMeta, width=128, signed=True, ir_type=lambda: T.i(128)):
    def __get_c_pointers__(self):
        raise TypeError("Int128 is not a JitArgument for now. ctypes has no support for 128b integers.")


class Uint8(Integer, metaclass=NumericMeta, width=8, signed=False, ir_type=T.i8):
    pass


class Uint16(Integer, metaclass=NumericMeta, width=16, signed=False, ir_type=T.i16):
    pass


class Uint32(Integer, metaclass=NumericMeta, width=32, signed=False, ir_type=T.i32):
    pass


class Uint64(Integer, metaclass=NumericMeta, width=64, signed=False, ir_type=T.i64):
    pass


class Uint128(Integer, metaclass=NumericMeta, width=128, signed=False, ir_type=lambda: T.i(128)):
    def __get_c_pointers__(self):
        raise TypeError("Uint128 is not a JitArgument for now. ctypes has no support for 128b integers.")


class Float16(Float, metaclass=NumericMeta, width=16, ir_type=T.f16):
    def __c_abi_spec__(self):
        def fill(a, s):
            v = a.value if hasattr(a, "value") else a
            s.value = int(np.float16(v).view(np.uint16))

        return [(ctypes.c_short, fill)]


class BFloat16(Float, metaclass=NumericMeta, width=16, ir_type=T.bf16):
    def __c_abi_spec__(self):
        def fill(a, s):
            v = a.value if hasattr(a, "value") else a
            bits = np.float32(v).view(np.uint32)
            s.value = int(np.uint16(bits >> 16))

        return [(ctypes.c_short, fill)]


class Float32(Float, metaclass=NumericMeta, width=32, ir_type=T.f32):
    def __c_abi_spec__(self):
        def fill(a, s):
            s.value = a.value if hasattr(a, "value") else a

        return [(ctypes.c_float, fill)]


class Float64(Float, metaclass=NumericMeta, width=64, ir_type=T.f64):
    def __c_abi_spec__(self):
        def fill(a, s):
            s.value = a.value if hasattr(a, "value") else a

        return [(ctypes.c_double, fill)]


class Float8E5M2(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E5M2): ...


class Float8E4M3FN(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E4M3FN): ...


class Float8E4M3FNUZ(
    Float, metaclass=NumericMeta, width=8, ir_type=lambda: ir.Float8E4M3FNUZType.get()
): ...  # not in upstream MLIR extras T


class Float8E4M3B11FNUZ(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E4M3B11FNUZ): ...


class Float8E4M3(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E4M3): ...


class Float6E2M3FN(Float, metaclass=NumericMeta, width=6, ir_type=T.f6E2M3FN): ...


class Float6E3M2FN(Float, metaclass=NumericMeta, width=6, ir_type=T.f6E3M2FN): ...


class Float8E8M0FNU(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E8M0FNU): ...


class Float4E2M1FN(Float, metaclass=NumericMeta, width=4, ir_type=T.f4E2M1FN): ...


# Float type rank for promotion (must be after class definitions)
_FLOAT_RANK = {Float64: 3, Float32: 2, Float16: 1, BFloat16: 1}

# ── Type promotion (added to Numeric after all subclasses exist) ──────

_FLOAT_BY_MIN_WIDTH = {16: Float16, 32: Float32, 64: Float64}


def _widen_float(float_type, min_width):
    """Return the narrowest standard float type with width >= *min_width*."""
    if float_type.width >= min_width:
        return float_type
    for w in (32, 64):
        if w >= min_width:
            return _FLOAT_BY_MIN_WIDTH[w]
    return Float64


@classmethod
def _promote(cls, a_type, b_type):
    """Resolve the promoted result type for two Numeric types.

    :param a_type: Left Numeric class (e.g. Float16)
    :param b_type: Right Numeric class (e.g. Float32)
    :return: The common Numeric class both can be safely promoted to
    """
    if a_type is b_type:
        return a_type

    a_float = a_type.is_float
    b_float = b_type.is_float

    if a_float and not b_float:
        return _widen_float(a_type, b_type.width)
    if b_float and not a_float:
        return _widen_float(b_type, a_type.width)

    if a_float and b_float:
        aw, bw = a_type.width, b_type.width
        if aw > bw and aw >= 16:
            return a_type
        if bw > aw and bw >= 16:
            return b_type
        if aw == bw:
            ra = _FLOAT_RANK.get(a_type, 0)
            rb = _FLOAT_RANK.get(b_type, 0)
            return a_type if ra >= rb else b_type
        raise ValueError(f"cannot promote {a_type} and {b_type}; cast explicitly")

    # Both integers
    if a_type.signed == b_type.signed:
        return a_type if a_type.width >= b_type.width else b_type
    u, s = (a_type, b_type) if not a_type.signed else (b_type, a_type)
    return u if u.width >= s.width else s


Numeric.promote = _promote


class Index(Integer, metaclass=NumericMeta, width=64, signed=False, ir_type=lambda: ir.IndexType.get()):
    """DSL Numeric for MLIR index type. Replaces arith.index(N).

    Usage:
        fx.Index(64)       # compile-time constant → arith.index(64)
        fx.Index(i32_val)  # cast i32/i64 ir.Value or Numeric to index
    """

    def __init__(self, x):
        from .utils.arith import index_cast

        # Unwrap DSL Numeric to ir.Value first
        if isinstance(x, Index):
            x = x.value
        elif isinstance(x, Numeric):
            x = x.ir_value()
        # Cast integer ir.Value to index (skip if already index type)
        if isinstance(x, ir.Value) and not isinstance(x.type, ir.IndexType):
            x = index_cast(ir.IndexType.get(), x)
        # x is now either: Python int, or index-typed ir.Value
        # Pass directly to Numeric.__init__ (bypass Integer conversion logic)
        Numeric.__init__(self, x)
