import ctypes
import operator
from typing import Type

import numpy as np

from .._mlir import ir
from .._mlir.dialects import arith
from .._mlir.extras import types as T
from .utils.arith import (
    ArithValue,
    arith_const,
    fp_to_fp,
    fp_to_int,
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
        def _extract_ir_values(self):
            return [self.ir_value()]

        def _new_from_ir_values(cls, values):
            return cls(values[0])

        def _c_pointers(self):
            if width == 1:
                c_value = ctypes.c_bool(self.value)
            elif signed:
                c_value = getattr(ctypes, f"c_int{width}")(self.value)
            else:
                c_value = getattr(ctypes, f"c_uint{width}")(self.value)
            return [ctypes.cast(ctypes.pointer(c_value), ctypes.c_void_p)]

        inferred_np = np_dtype if np_dtype is not None else _infer_np_dtype(width, signed, name)

        new_attrs = {
            "__extract_ir_values__": _extract_ir_values,
            "__new_from_ir_values__": classmethod(_new_from_ir_values),
        }
        if signed is not None:
            new_attrs["__c_pointers__"] = _c_pointers

        new_cls = super().__new__(cls, name, bases, new_attrs | attrs)
        if ir_type is not None:
            new_cls._ir_type = staticmethod(ir_type)
        new_cls.width = width
        new_cls._np_dtype = inferred_np
        new_cls.signed = signed
        new_cls._zero = zero
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


def _integer_promote(x, promote_bool=False):
    ty = type(x)
    if ty is Boolean:
        if promote_bool:
            return x.to(Int32), Int32
        return x, ty
    if ty.is_integer and ty.width < 32:
        return x.to(Int32), Int32
    return x, ty


def _numeric_binary_op_type_promote(a, b, promote_bool=False):
    a_type = type(a)
    b_type = type(b)

    a, a_type = _integer_promote(a, promote_bool=promote_bool)
    b, b_type = _integer_promote(b, promote_bool=promote_bool)

    if a_type == b_type:
        return a, b, a_type

    if a_type.is_float or b_type.is_float:
        a_width = getattr(a_type, "width", 0)
        b_width = getattr(b_type, "width", 0)

        if a_type.is_float and not b_type.is_float:
            res_type = a_type
        elif b_type.is_float and not a_type.is_float:
            res_type = b_type
        elif a_width > b_width and a_width >= 16:
            res_type = a_type
        elif b_width > a_width and b_width >= 16:
            res_type = b_type
        elif a_width == b_width:
            if a_type is Float64 or b_type is Float64:
                res_type = Float64
            elif a_type is Float32 or b_type is Float32:
                res_type = Float32
            elif a_type is Float16 or b_type is Float16:
                res_type = Float16
            else:
                raise ValueError(f"implicit float promotion of {a_type} and {b_type} not supported")
        else:
            raise ValueError(f"implicit float promotion of {a_type} and {b_type} not supported")

        new_a = a.to(res_type) if type(a) is not res_type else a
        new_b = b.to(res_type) if type(b) is not res_type else b
        return new_a, new_b, res_type

    a_width = a_type.width
    b_width = b_type.width
    a_signed = a_type.signed
    b_signed = b_type.signed

    if a_signed == b_signed:
        if a_width >= b_width:
            return a, b.to(a_type), a_type
        return a.to(b_type), b, b_type

    unsigned_type = a_type if not a_signed else b_type
    signed_type = a_type if a_signed else b_type
    if unsigned_type.width >= signed_type.width:
        res_type = unsigned_type
    else:
        res_type = signed_type
    new_a = a.to(res_type) if type(a) is not res_type else a
    new_b = b.to(res_type) if type(b) is not res_type else b
    return new_a, new_b, res_type


def _numeric_binary_op(op, promote_operand=True, promote_bool=False, flip=False):
    def wrapper(lhs, rhs, *, loc=None, ip=None):
        if not isinstance(rhs, Numeric):
            if not isinstance(rhs, (ArithValue, int, float, bool)):
                return NotImplemented
            if isinstance(rhs, ArithValue) and isinstance(rhs.type, ir.VectorType):
                return NotImplemented
            rhs = as_numeric(rhs)

        res_type = type(lhs)

        if promote_operand:
            lhs, rhs, res_type = _numeric_binary_op_type_promote(lhs, rhs, promote_bool)
        else:
            rhs = type(lhs)(rhs)

        if op in (
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.eq,
            operator.ne,
        ):
            res_type = Boolean
        elif op == operator.truediv and isinstance(lhs, Integer):
            res_type = Float64 if res_type.width > 32 else Float32

        if isinstance(lhs.value, ArithValue):
            lhs_val = lhs.value.with_signedness(lhs.signed)
        else:
            lhs_val = lhs.value

        if isinstance(rhs.value, ArithValue):
            rhs_val = rhs.value.with_signedness(rhs.signed)
        else:
            rhs_val = rhs.value

        if flip:
            lhs_val, rhs_val = rhs_val, lhs_val

        res_val = op(lhs_val, rhs_val)
        return res_type(res_val, loc=loc, ip=ip)

    return wrapper


class Numeric(metaclass=NumericMeta):
    def __init__(self, value, *, loc=None, ip=None):
        self.value = value

    def __str__(self) -> str:
        return "?"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.value)})"

    def __hash__(self):
        return hash(type(self)) ^ hash(self.value)

    @property
    def dtype(self) -> Type["Numeric"]:
        return type(self)

    def to(self, dtype, *, loc=None, ip=None):
        if dtype is type(self):
            return self
        elif isinstance(dtype, type) and issubclass(dtype, Numeric):
            return dtype(self)
        elif dtype is ir.Value:
            if isinstance(self.value, (int, float, bool)):
                return arith_const(self.value, type(self).ir_type, loc=loc, ip=ip)
            elif isinstance(self.value, ir.Value):
                res = self.value
                if not isinstance(res, ArithValue):
                    raise ValueError(f"expected ArithValue, got {type(res)}")
                return res.with_signedness(getattr(type(self), "signed", None))
            else:
                raise ValueError(f"cannot convert {type(self)} to {dtype}")
        elif dtype in (int, float, bool):
            if isinstance(self.value, ir.Value):
                raise ValueError(f"unable to convert dynamic value to static type: {dtype}")
            return dtype(self.value)
        else:
            raise ValueError(f"unable to convert {type(self)} to {dtype}")

    def ir_value(self, *, loc=None, ip=None) -> ir.Value:
        return self.to(ir.Value, loc=loc, ip=ip)

    def __ir_types__(self):
        return [type(self).ir_type]

    def __neg__(self, *, loc=None, ip=None):
        if isinstance(self.value, (bool, int, float)):
            return type(self)(-self.value)
        return type(self)(-self.value, loc=loc, ip=ip)

    def __dsl_bool__(self, *, loc=None, ip=None):
        if isinstance(self.value, (int, float, bool)):
            return Boolean(bool(self.value))
        zero = arith_const(type(self).zero, type(self).ir_type, loc=loc, ip=ip)
        return self.__ne__(type(self)(zero, loc=loc, ip=ip), loc=loc, ip=ip)

    def __dsl_not__(self, *, loc=None, ip=None):
        b = self.__dsl_bool__(loc=loc, ip=ip)
        if isinstance(b.value, bool):
            return Boolean(not b.value)
        zero = arith_const(0, T.bool(), loc=loc, ip=ip)
        return Boolean(b.ir_value().__eq__(zero), loc=loc, ip=ip)

    def __dsl_and__(self, other, *, loc=None, ip=None):
        lhs = self.__dsl_bool__(loc=loc, ip=ip)
        rhs = as_numeric(other).__dsl_bool__(loc=loc, ip=ip)
        if isinstance(lhs.value, bool) and isinstance(rhs.value, bool):
            return Boolean(lhs.value and rhs.value)
        return Boolean(lhs.ir_value().__and__(rhs.ir_value()), loc=loc, ip=ip)

    def __dsl_or__(self, other, *, loc=None, ip=None):
        lhs = self.__dsl_bool__(loc=loc, ip=ip)
        rhs = as_numeric(other).__dsl_bool__(loc=loc, ip=ip)
        if isinstance(lhs.value, bool) and isinstance(rhs.value, bool):
            return Boolean(lhs.value or rhs.value)
        return Boolean(lhs.ir_value().__or__(rhs.ir_value()), loc=loc, ip=ip)

    def __bool__(self):
        if isinstance(self.value, (int, float, bool)):
            return bool(self.value)
        raise RuntimeError(f"unable to convert dynamic '{type(self).__name__}' value to bool at compile time")

    def __index__(self):
        if isinstance(self.value, (int, float, bool)):
            return int(self.value)
        raise RuntimeError(f"'{type(self.value)}' object cannot be interpreted as an integer")

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
            T.f8E5M2(): Float8E5M2,
            T.f8E4M3(): Float8E4M3,
            T.f8E4M3FN(): Float8E4M3FN,
            T.f8E4M3B11FNUZ(): Float8E4M3B11FNUZ,
            T.f8E8M0FNU(): Float8E8M0FNU,
            T.f6E2M3FN(): Float6E2M3FN,
            T.f6E3M2FN(): Float6E3M2FN,
            T.f4E2M1FN(): Float4E2M1FN,
        }
        if ir_type not in ir2dsl_map:
            raise ValueError(f"unsupported mlir type: {ir_type}")
        return ir2dsl_map[ir_type]

    def __add__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.add, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __sub__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.sub, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __mul__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.mul, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __floordiv__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.floordiv, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __truediv__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.truediv, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __mod__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.mod, promote_bool=True)(self, other, loc=loc, ip=ip)

    def __radd__(self, other, *, loc=None, ip=None):
        return self.__add__(other, loc=loc, ip=ip)

    def __rsub__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.sub, promote_bool=True, flip=True)(self, other, loc=loc, ip=ip)

    def __rmul__(self, other, *, loc=None, ip=None):
        return self.__mul__(other, loc=loc, ip=ip)

    def __rfloordiv__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.floordiv, promote_bool=True, flip=True)(self, other, loc=loc, ip=ip)

    def __rtruediv__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.truediv, promote_bool=True, flip=True)(self, other, loc=loc, ip=ip)

    def __rmod__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.mod, promote_bool=True, flip=True)(self, other, loc=loc, ip=ip)

    def __pow__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.pow)(self, other, loc=loc, ip=ip)

    def __eq__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.eq)(self, other, loc=loc, ip=ip)

    def __ne__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.ne)(self, other, loc=loc, ip=ip)

    def __lt__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.lt)(self, other, loc=loc, ip=ip)

    def __le__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.le)(self, other, loc=loc, ip=ip)

    def __gt__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.gt)(self, other, loc=loc, ip=ip)

    def __ge__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.ge)(self, other, loc=loc, ip=ip)


def as_numeric(obj):
    if isinstance(obj, Numeric):
        return obj
    return Numeric.from_python_value(obj)


class Integer(Numeric, metaclass=NumericMeta, width=32, signed=True, ir_type=T.i32):
    def __init__(self, x, *, loc=None, ip=None):
        ty = type(self)

        if isinstance(x, (bool, int, float)):
            if isinstance(x, float):
                if np.isnan(x):
                    raise ValueError("cannot convert float NaN to integer")
                elif np.isinf(x):
                    raise OverflowError("cannot convert float infinity to integer")
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
                x_val = arith.index_cast(ty.ir_type, x, loc=loc, ip=ip)
            elif isinstance(x.type, ir.IntegerType):
                if x.type.width != ty.width:
                    x_val = int_to_int(x, ty, signed=ty.signed)
            elif is_float_type(x.type):
                x_val = fp_to_int(x, ty.signed, ty.ir_type, loc=loc, ip=ip)
        elif isinstance(x, Integer):
            if isinstance(x.value, ir.Value):
                x_val = int_to_int(x.ir_value(), ty)
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

    def __invert__(self, *, loc=None, ip=None):
        res_type = type(self)
        return res_type(self.ir_value(loc=loc, ip=ip).__invert__(loc=loc, ip=ip))

    def __lshift__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.lshift)(self, other, loc=loc, ip=ip)

    def __rlshift__(self, other, *, loc=None, ip=None):
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"cannot left shift {other_} with {self}")
        return other_.__lshift__(self, loc=loc, ip=ip)

    def __rshift__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.rshift)(self, other, loc=loc, ip=ip)

    def __rrshift__(self, other, *, loc=None, ip=None):
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"cannot right shift {other_} with {self}")
        return other_.__rshift__(self, loc=loc, ip=ip)

    def __and__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.and_)(self, other, loc=loc, ip=ip)

    def __rand__(self, other, *, loc=None, ip=None):
        return self.__and__(other, loc=loc, ip=ip)

    def __or__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.or_)(self, other, loc=loc, ip=ip)

    def __ror__(self, other, *, loc=None, ip=None):
        return self.__or__(other, loc=loc, ip=ip)

    def __xor__(self, other, *, loc=None, ip=None):
        return _numeric_binary_op(operator.xor)(self, other, loc=loc, ip=ip)

    def __rxor__(self, other, *, loc=None, ip=None):
        return self.__xor__(other, loc=loc, ip=ip)

    def is_static(self):
        return not isinstance(self.value, ir.Value)


class Float(Numeric, metaclass=NumericMeta, width=32, ir_type=T.f32):
    def __init__(self, x, *, loc=None, ip=None):
        ty = type(self)

        if isinstance(x, (bool, int, float)):
            super().__init__(float(x))
        elif isinstance(x, ir.Value):
            if isinstance(x.type, ir.IntegerType):
                raise ValueError("signless integer to float conversion is not supported directly")
            elif is_float_type(x.type):
                if x.type != ty.ir_type:
                    x = fp_to_fp(x, ty.ir_type, loc=loc, ip=ip)
            super().__init__(x)
        elif isinstance(x, Integer):
            if isinstance(x.value, ir.Value):
                x = int_to_fp(x.value, type(x).signed, ty.ir_type, loc=loc, ip=ip)
            else:
                x = float(x.value)
            super().__init__(x)
        elif isinstance(x, Float):
            Float.__init__(self, x.value)
        else:
            raise ValueError(f"{x} to float conversion is not supported")


class Boolean(Integer, metaclass=NumericMeta, width=1, signed=True, ir_type=T.bool):
    def __init__(self, a, *, loc=None, ip=None):
        value = None
        if isinstance(a, (bool, int, float)):
            value = bool(a)
        elif isinstance(a, Numeric):
            Boolean.__init__(self, a.value, loc=loc, ip=ip)
            return
        elif isinstance(a, ArithValue):
            if a.type == T.bool():
                value = a
            else:
                value = a != arith_const(0, a.type, loc=loc, ip=ip)
        if value is None:
            raise ValueError(f"cannot convert {a} to Boolean")
        super().__init__(value, loc=loc, ip=ip)

    def __neg__(self, *, loc=None, ip=None):
        raise TypeError("negation is not supported for boolean type")


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


class Uint8(Integer, metaclass=NumericMeta, width=8, signed=False, ir_type=T.i8):
    pass


class Uint16(Integer, metaclass=NumericMeta, width=16, signed=False, ir_type=T.i16):
    pass


class Uint32(Integer, metaclass=NumericMeta, width=32, signed=False, ir_type=T.i32):
    pass


class Uint64(Integer, metaclass=NumericMeta, width=64, signed=False, ir_type=T.i64):
    pass


class Float16(Float, metaclass=NumericMeta, width=16, ir_type=T.f16):
    def __c_pointers__(self):
        if not isinstance(self.value, float):
            raise ValueError("only float is supported")
        f16_val = np.float16(self.value)
        bits = f16_val.view(np.uint16)
        c_val = ctypes.c_short(bits)
        return [ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p)]


class BFloat16(Float, metaclass=NumericMeta, width=16, ir_type=T.bf16):
    def __c_pointers__(self):
        if not isinstance(self.value, float):
            raise ValueError("only float is supported")
        f32_val = np.float32(self.value)
        bits = f32_val.view(np.uint32)
        bf16_bits = np.uint16(bits >> 16)
        c_val = ctypes.c_short(bf16_bits)
        return [ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p)]


class Float32(Float, metaclass=NumericMeta, width=32, ir_type=T.f32):
    def __c_pointers__(self):
        if not isinstance(self.value, float):
            raise ValueError("only float is supported")
        return [ctypes.cast(ctypes.pointer(ctypes.c_float(self.value)), ctypes.c_void_p)]


class Float64(Float, metaclass=NumericMeta, width=64, ir_type=T.f64):
    def __c_pointers__(self):
        if not isinstance(self.value, float):
            raise ValueError("only float is supported")
        return [ctypes.cast(ctypes.pointer(ctypes.c_double(self.value)), ctypes.c_void_p)]


class Float8E5M2(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E5M2): ...


class Float8E4M3FN(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E4M3FN): ...


class Float8E4M3B11FNUZ(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E4M3B11FNUZ): ...


class Float8E4M3(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E4M3): ...


class Float6E2M3FN(Float, metaclass=NumericMeta, width=6, ir_type=T.f6E2M3FN): ...


class Float6E3M2FN(Float, metaclass=NumericMeta, width=6, ir_type=T.f6E3M2FN): ...


class Float8E8M0FNU(Float, metaclass=NumericMeta, width=8, ir_type=T.f8E8M0FNU): ...


class Float4E2M1FN(Float, metaclass=NumericMeta, width=4, ir_type=T.f4E2M1FN): ...
