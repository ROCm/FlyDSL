# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import builtins
from functools import partialmethod

from ..._mlir import ir
from ..._mlir.dialects import arith, math
from ..._mlir.extras import types as T
from ..meta import dsl_loc_tracing


def element_type(ty) -> ir.Type:
    if isinstance(ty, ir.VectorType):
        return ty.element_type
    return ty


def is_integer_like_type(ty) -> bool:
    elem_ty = element_type(ty)
    return isinstance(elem_ty, ir.IntegerType) or isinstance(elem_ty, ir.IndexType)


def is_narrow_float_type(ty) -> bool:
    elem_ty = element_type(ty)
    return isinstance(elem_ty, ir.FloatType) and elem_ty.width <= 8


def is_float_type(ty) -> bool:
    elem_ty = element_type(ty)
    return isinstance(elem_ty, ir.FloatType)


def recast_type(src_type, res_elem_type) -> ir.Type:
    if isinstance(src_type, ir.VectorType):
        return ir.VectorType.get(list(src_type.shape), res_elem_type)
    return res_elem_type


@dsl_loc_tracing
def arith_const(value, ty=None):
    if isinstance(value, ir.Value):
        return value

    if ty is None:
        if isinstance(value, float):
            ty = T.f32()
        elif isinstance(value, bool):
            ty = T.bool()
        elif isinstance(value, int):
            ty = T.i32()
        else:
            raise ValueError(f"unsupported constant type: {type(value)}")

    if isinstance(ty, ir.VectorType):
        elem_ty = element_type(ty)
        if isinstance(elem_ty, ir.IntegerType):
            attr = ir.IntegerAttr.get(elem_ty, int(value))
        else:
            attr = ir.FloatAttr.get(elem_ty, float(value))
        value = ir.DenseElementsAttr.get_splat(ty, attr)
    elif is_integer_like_type(ty):
        value = int(value)
    elif is_float_type(ty):
        value = float(value)
    else:
        raise ValueError(f"unsupported constant type: {type(value)}")
    return arith.constant(ty, value)


@dsl_loc_tracing
def fp_to_fp(src, res_elem_type):
    if not isinstance(src, ir.Value) and hasattr(src, "ir_value"):
        src = src.ir_value()
    src_elem_type = element_type(src.type)
    if res_elem_type == src_elem_type:
        return src
    res_type = recast_type(src.type, res_elem_type)
    if res_elem_type.width > src_elem_type.width:
        return arith.extf(res_type, src)
    return arith.truncf(res_type, src)


@dsl_loc_tracing
def fp_to_int(src, signed, res_elem_type):
    if not isinstance(src, ir.Value) and hasattr(src, "ir_value"):
        src = src.ir_value()
    res_type = recast_type(src.type, res_elem_type)
    if signed:
        return arith.fptosi(res_type, src)
    return arith.fptoui(res_type, src)


@dsl_loc_tracing
def int_to_fp(src, signed, res_elem_type):
    if not isinstance(src, ir.Value) and hasattr(src, "ir_value"):
        src = src.ir_value()
    res_type = recast_type(src.type, res_elem_type)
    if signed and element_type(src.type).width > 1:
        return arith.sitofp(res_type, src)
    return arith.uitofp(res_type, src)


@dsl_loc_tracing
def int_to_int(src, dst_type, *, signed=None):
    if not isinstance(src, ir.Value) and hasattr(src, "ir_value"):
        src = src.ir_value()
    src_width = element_type(src.type).width
    dst_width = dst_type.width
    dst_ir_type = recast_type(src.type, dst_type.ir_type)
    if dst_width == src_width:
        return src
    elif dst_width > src_width:
        if signed is None:
            signed = getattr(src, "signed", None)
        if signed and src_width > 1:
            return arith.extsi(dst_ir_type, src)
        return arith.extui(dst_ir_type, src)
    return arith.trunci(dst_ir_type, src)


@dsl_loc_tracing
def _coerce_other(self, other):
    if isinstance(other, (int, float, bool)):
        return arith_const(other, self.type).with_signedness(self.signed)
    if not isinstance(other, ArithValue):
        # Accept DSL Numeric types (Int32, Float32, etc.) by unwrapping via ir_value()
        if hasattr(other, "ir_value"):
            other = ArithValue(other.ir_value())
        else:
            return NotImplemented
    # Broadcast scalar to vector when self is a vector and other is scalar
    if isinstance(self.type, ir.VectorType) and not isinstance(other.type, ir.VectorType):
        from ..._mlir.dialects import vector as _vector

        return _vector.broadcast(self.type, _to_raw(other))
    return other


_ARITH_OPS = {
    "add": (arith.addf, arith.addi),
    "sub": (arith.subf, arith.subi),
    "mul": (arith.mulf, arith.muli),
}


@dsl_loc_tracing
def _binary_op(self, other, op):
    other = _coerce_other(self, other)
    if other is NotImplemented:
        return NotImplemented

    if op in _ARITH_OPS:
        float_fn, int_fn = _ARITH_OPS[op]
        if self.is_float:
            return float_fn(self, other)
        return int_fn(self, other)

    if op == "div":
        if self.is_float:
            return arith.divf(self, other)
        et = element_type(self.type)
        if isinstance(et, ir.IndexType):
            return arith.divui(self, other)
        fp_ty = T.f64() if et.width > 32 else T.f32()
        lhs = int_to_fp(self, self.signed, fp_ty)
        rhs = int_to_fp(other, other.signed, fp_ty)
        return arith.divf(lhs, rhs)

    if op == "floordiv":
        if self.is_float:
            q = arith.divf(self, other)
            return math.floor(q)
        et = element_type(self.type)
        if isinstance(et, ir.IndexType):
            return arith.divui(self, other)
        if self.signed is not False:
            return arith.floordivsi(self, other)
        return arith.divui(self, other)

    if op == "mod":
        if self.is_float:
            return arith.remf(self, other)
        et = element_type(self.type)
        if isinstance(et, ir.IndexType):
            return arith.remui(self, other)
        if self.signed is not False:
            return arith.remsi(self, other)
        return arith.remui(self, other)

    raise ValueError(f"unknown binary op: {op}")


@dsl_loc_tracing
def _rbinary_op(self, other, op):
    other = _coerce_other(self, other)
    if other is NotImplemented:
        return NotImplemented
    return _binary_op(other, self, op)


_CMP_FLOAT_PRED = {
    "lt": arith.CmpFPredicate.OLT,
    "le": arith.CmpFPredicate.OLE,
    "eq": arith.CmpFPredicate.OEQ,
    "ne": arith.CmpFPredicate.UNE,
    "gt": arith.CmpFPredicate.OGT,
    "ge": arith.CmpFPredicate.OGE,
}
_CMP_INT_SIGNED = {
    "lt": arith.CmpIPredicate.slt,
    "le": arith.CmpIPredicate.sle,
    "eq": arith.CmpIPredicate.eq,
    "ne": arith.CmpIPredicate.ne,
    "gt": arith.CmpIPredicate.sgt,
    "ge": arith.CmpIPredicate.sge,
}
_CMP_INT_UNSIGNED = {
    "lt": arith.CmpIPredicate.ult,
    "le": arith.CmpIPredicate.ule,
    "eq": arith.CmpIPredicate.eq,
    "ne": arith.CmpIPredicate.ne,
    "gt": arith.CmpIPredicate.ugt,
    "ge": arith.CmpIPredicate.uge,
}


@dsl_loc_tracing
def _comparison_op(self, other, predicate):
    other = _coerce_other(self, other)
    if other is NotImplemented:
        return NotImplemented

    if self.is_float:
        return arith.cmpf(_CMP_FLOAT_PRED[predicate], self, other)
    if self.signed is not False:
        return arith.cmpi(_CMP_INT_SIGNED[predicate], self, other)
    return arith.cmpi(_CMP_INT_UNSIGNED[predicate], self, other)


_BITWISE_OPS = {
    "and": arith.andi,
    "or": arith.ori,
    "xor": arith.xori,
}


@dsl_loc_tracing
def _bitwise_op(self, other, op, reverse=False):
    other = _coerce_other(self, other)
    if other is NotImplemented:
        return NotImplemented
    fn = _BITWISE_OPS[op]
    if reverse:
        return fn(other, self)
    return fn(self, other)


@dsl_loc_tracing
def _shift_op(self, other, op, reverse=False):
    other = _coerce_other(self, other)
    if other is NotImplemented:
        return NotImplemented
    lhs, rhs = (other, self) if reverse else (self, other)
    if op == "shl":
        return arith.shli(lhs, rhs)
    signed = getattr(lhs, "signed", None)
    if signed is True:
        return arith.shrsi(lhs, rhs)
    return arith.shrui(lhs, rhs)


@dsl_loc_tracing
def _pow_op(self, other, reverse=False):
    other = _coerce_other(self, other)
    if other is NotImplemented:
        return NotImplemented
    if reverse:
        self, other = other, self
    if self.is_float and other.is_float:
        return math.powf(self, other)
    if self.is_float and not other.is_float:
        return math.fpowi(self, other)
    if not self.is_float and other.is_float:
        fp_ty = element_type(other.type)
        lhs = int_to_fp(self, self.signed, fp_ty)
        return math.powf(lhs, other)
    return math.ipowi(self, other)


@dsl_loc_tracing
def _neg_op(self):
    if self.type == T.bool():
        raise TypeError("negation is not supported for boolean type")
    if self.is_float:
        return arith.negf(self)
    c0 = arith_const(0, self.type)
    return arith.subi(c0, self)


@dsl_loc_tracing
def _invert_op(self):
    return arith.xori(self, arith_const(-1, self.type))


@dsl_loc_tracing
def _select_raw_operand(value, other):
    if isinstance(value, (int, float, bool)):
        return _to_raw(arith_const(value, _to_raw(other).type))
    return _to_raw(value)


@ir.register_value_caster(ir.Float4E2M1FNType.static_typeid)
@ir.register_value_caster(ir.Float6E2M3FNType.static_typeid)
@ir.register_value_caster(ir.Float6E3M2FNType.static_typeid)
@ir.register_value_caster(ir.Float8E4M3FNType.static_typeid)
@ir.register_value_caster(ir.Float8E4M3B11FNUZType.static_typeid)
@ir.register_value_caster(ir.Float8E5M2Type.static_typeid)
@ir.register_value_caster(ir.Float8E4M3Type.static_typeid)
@ir.register_value_caster(ir.Float8E8M0FNUType.static_typeid)
@ir.register_value_caster(ir.BF16Type.static_typeid)
@ir.register_value_caster(ir.F16Type.static_typeid)
@ir.register_value_caster(ir.F32Type.static_typeid)
@ir.register_value_caster(ir.F64Type.static_typeid)
@ir.register_value_caster(ir.IntegerType.static_typeid)
@ir.register_value_caster(ir.IndexType.static_typeid)
@ir.register_value_caster(ir.VectorType.static_typeid)
class ArithValue(ir.Value):
    def __init__(self, v, signed=None):
        if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
            v = v.ir_value()
        super().__init__(v)
        elem_ty = element_type(self.type)
        self.is_float = not is_integer_like_type(elem_ty)
        self.signed = signed and elem_ty.width > 1

    def with_signedness(self, signed):
        return type(self)(self, signed)

    __neg__ = _neg_op
    __invert__ = _invert_op

    __add__ = partialmethod(_binary_op, op="add")
    __sub__ = partialmethod(_binary_op, op="sub")
    __mul__ = partialmethod(_binary_op, op="mul")
    __truediv__ = partialmethod(_binary_op, op="div")
    __floordiv__ = partialmethod(_binary_op, op="floordiv")
    __mod__ = partialmethod(_binary_op, op="mod")

    __radd__ = partialmethod(_rbinary_op, op="add")
    __rsub__ = partialmethod(_rbinary_op, op="sub")
    __rmul__ = partialmethod(_rbinary_op, op="mul")
    __rtruediv__ = partialmethod(_rbinary_op, op="div")
    __rfloordiv__ = partialmethod(_rbinary_op, op="floordiv")
    __rmod__ = partialmethod(_rbinary_op, op="mod")

    __pow__ = partialmethod(_pow_op)
    __rpow__ = partialmethod(_pow_op, reverse=True)

    __lt__ = partialmethod(_comparison_op, predicate="lt")
    __le__ = partialmethod(_comparison_op, predicate="le")
    __eq__ = partialmethod(_comparison_op, predicate="eq")
    __ne__ = partialmethod(_comparison_op, predicate="ne")
    __gt__ = partialmethod(_comparison_op, predicate="gt")
    __ge__ = partialmethod(_comparison_op, predicate="ge")

    __and__ = partialmethod(_bitwise_op, op="and")
    __or__ = partialmethod(_bitwise_op, op="or")
    __xor__ = partialmethod(_bitwise_op, op="xor")
    __rand__ = partialmethod(_bitwise_op, op="and", reverse=True)
    __ror__ = partialmethod(_bitwise_op, op="or", reverse=True)
    __rxor__ = partialmethod(_bitwise_op, op="xor", reverse=True)

    __lshift__ = partialmethod(_shift_op, op="shl")
    __rshift__ = partialmethod(_shift_op, op="shr")
    __rlshift__ = partialmethod(_shift_op, op="shl", reverse=True)
    __rrshift__ = partialmethod(_shift_op, op="shr", reverse=True)

    @dsl_loc_tracing
    def select(self, true_value, false_value):
        """Ternary select: self (i1 condition) ? true_value : false_value."""
        true_value = _select_raw_operand(true_value, false_value)
        false_value = _select_raw_operand(false_value, true_value)
        return arith.SelectOp(_to_raw(self), true_value, false_value).result

    @dsl_loc_tracing
    def extf(self, target_type):
        """Extend float precision (e.g. bf16 → f32)."""
        return arith.ExtFOp(target_type, self).result

    @dsl_loc_tracing
    def truncf(self, target_type):
        """Truncate float precision (e.g. f32 → bf16)."""
        return arith.TruncFOp(target_type, self).result

    @dsl_loc_tracing
    def extui(self, target_type):
        """Zero-extend integer to wider type (e.g. i32 → i64)."""
        return arith.ExtUIOp(target_type, self).result

    @dsl_loc_tracing
    def extsi(self, target_type):
        """Sign-extend integer to wider type (e.g. i32 → i64)."""
        return arith.ExtSIOp(target_type, self).result

    @dsl_loc_tracing
    def trunci(self, target_type):
        """Truncate integer to narrower type (e.g. i64 → i32)."""
        return arith.TruncIOp(target_type, self).result

    @dsl_loc_tracing
    def bitcast(self, target_type):
        """Reinterpret bits as different type (same bit width)."""
        return arith.BitcastOp(target_type, self).result

    @dsl_loc_tracing
    def shrui(self, amount):
        """Unsigned right shift (zero-fills high bits)."""
        return arith.ShRUIOp(self, _to_raw(amount)).result

    @dsl_loc_tracing
    def addf(self, other, *, fastmath=None):
        """Float add with optional fastmath flags."""
        return arith.addf(self, _to_raw(other), fastmath=fastmath)

    @dsl_loc_tracing
    def maximumf(self, other):
        """Float maximum (NaN-propagating)."""
        return arith.maximumf(self, _to_raw(other))

    @dsl_loc_tracing
    def rsqrt(self, *, fastmath=None):
        """Reciprocal square root: 1/sqrt(self)."""
        from ..._mlir.dialects import math as _math

        return _math.rsqrt(self, fastmath=fastmath)

    @dsl_loc_tracing
    def exp2(self, *, fastmath=None):
        """Base-2 exponential: 2^self."""
        from ..._mlir.dialects import math as _math

        return _math.exp2(self, fastmath=fastmath)

    @dsl_loc_tracing
    def shuffle_xor(self, offset, width):
        """GPU warp shuffle with XOR mode."""
        from ..._mlir.dialects.gpu import ShuffleOp

        if isinstance(offset, int):
            offset = constant(offset, type=T.i32())
        if isinstance(width, int):
            width = constant(width, type=T.i32())
        return ShuffleOp(_to_raw(self), _to_raw(offset), _to_raw(width), mode="xor").shuffleResult

    @dsl_loc_tracing
    def index_cast(self, target_type):
        """Cast between index and integer types."""
        if self.type == target_type:
            return self
        return arith.IndexCastOp(target_type, self).result

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        try:
            ty = str(self.type)
            owner = self.owner
            if isinstance(owner, ir.Block):
                return f"ArithValue(type={ty}, block_arg)"
            op_name = owner.name
            return f"ArithValue(type={ty}, op={op_name})"
        except Exception:
            return "ArithValue(type=?)"

    def __repr__(self):
        return self.__str__()


# =========================================================================
# Function-level arith API
# =========================================================================


def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


@dsl_loc_tracing
def constant(value, *, type=None, index=False):
    """Create a constant value.

    Args:
        value: Python int/float/bool
        type: Explicit MLIR type (optional)
        index: If True, create index type constant
    """
    if index:
        mlir_type = ir.IndexType.get()
    elif type is not None:
        mlir_type = type
    elif isinstance(value, float):
        mlir_type = T.f32()
    elif isinstance(value, bool):
        mlir_type = T.bool()
    elif isinstance(value, int):
        mlir_type = T.i32()
    else:
        raise ValueError(f"unsupported constant type: {builtins.type(value)}")
    if isinstance(mlir_type, (ir.F16Type, ir.F32Type, ir.F64Type, ir.BF16Type)):
        value = float(value)
    return arith.constant(mlir_type, value)


@dsl_loc_tracing
def index(value):
    """Create an index constant."""
    return constant(value, index=True)


@dsl_loc_tracing
def constant_vector(element_value, vector_type):
    """Create a splat constant vector."""
    elem_ty = element_type(vector_type)
    if is_float_type(elem_ty):
        attr = ir.FloatAttr.get(elem_ty, float(element_value))
    else:
        attr = ir.IntegerAttr.get(elem_ty, int(element_value))
    dense = ir.DenseElementsAttr.get_splat(vector_type, attr)
    return arith.constant(vector_type, dense)


@dsl_loc_tracing
def index_cast(target_type, value):
    """Cast between index and integer types."""
    v = _to_raw(value)
    if v.type == target_type:
        return v
    return arith.IndexCastOp(target_type, v).result


@dsl_loc_tracing
def select(condition, true_value, false_value):
    """Select between two values based on a boolean condition."""
    true_value = _select_raw_operand(true_value, false_value)
    false_value = _select_raw_operand(false_value, true_value)
    return arith.SelectOp(_to_raw(condition), true_value, false_value).result


@dsl_loc_tracing
def sitofp(target_type, value):
    """Convert signed integer to floating point."""
    return arith.SIToFPOp(target_type, _to_raw(value)).result


@dsl_loc_tracing
def trunc_f(target_type, value):
    """Truncate floating point to narrower type (e.g. f32 -> f16)."""
    return arith.TruncFOp(target_type, _to_raw(value)).result


@dsl_loc_tracing
def andi(lhs, rhs):
    """Bitwise AND."""
    return arith.AndIOp(_to_raw(lhs), _to_raw(rhs)).result


@dsl_loc_tracing
def xori(lhs, rhs):
    """Bitwise XOR."""
    return arith.XOrIOp(_to_raw(lhs), _to_raw(rhs)).result


@dsl_loc_tracing
def shli(lhs, rhs):
    """Left shift."""
    return arith.ShLIOp(_to_raw(lhs), _to_raw(rhs)).result


def unwrap(val, *, type=None, index=False):
    """Unwrap ArithValue to raw ir.Value. Materializes Python scalars."""
    if isinstance(val, (int, float, bool)):
        return _to_raw(constant(val, type=type, index=index))
    return _to_raw(val)
