import builtins
from functools import partialmethod

from ..._mlir import ir
from ..._mlir.dialects import arith, math
from ..._mlir.extras import types as T


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


def arith_const(value, ty=None, *, loc=None, ip=None):
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
    return arith.constant(ty, value, loc=loc, ip=ip)


def fp_to_fp(src, res_elem_type, *, loc=None, ip=None):
    src_elem_type = element_type(src.type)
    if res_elem_type == src_elem_type:
        return src
    res_type = recast_type(src.type, res_elem_type)
    if res_elem_type.width > src_elem_type.width:
        return arith.extf(res_type, src, loc=loc, ip=ip)
    return arith.truncf(res_type, src, loc=loc, ip=ip)


def fp_to_int(src, signed, res_elem_type, *, loc=None, ip=None):
    res_type = recast_type(src.type, res_elem_type)
    if signed:
        return arith.fptosi(res_type, src, loc=loc, ip=ip)
    return arith.fptoui(res_type, src, loc=loc, ip=ip)


def int_to_fp(src, signed, res_elem_type, *, loc=None, ip=None):
    res_type = recast_type(src.type, res_elem_type)
    if signed and element_type(src.type).width > 1:
        return arith.sitofp(res_type, src, loc=loc, ip=ip)
    return arith.uitofp(res_type, src, loc=loc, ip=ip)


def int_to_int(src, dst_type, *, signed=None, loc=None, ip=None):
    src_width = element_type(src.type).width
    dst_width = dst_type.width
    dst_ir_type = recast_type(src.type, dst_type.ir_type)
    if dst_width == src_width:
        return src
    elif dst_width > src_width:
        if signed is None:
            signed = getattr(src, "signed", None)
        if signed and src_width > 1:
            return arith.extsi(dst_ir_type, src, loc=loc, ip=ip)
        return arith.extui(dst_ir_type, src, loc=loc, ip=ip)
    return arith.trunci(dst_ir_type, src, loc=loc, ip=ip)


def _coerce_other(self, other, *, loc=None, ip=None):
    if isinstance(other, (int, float, bool)):
        return arith_const(other, self.type, loc=loc, ip=ip).with_signedness(self.signed)
    if not isinstance(other, ArithValue):
        return NotImplemented
    return other


_ARITH_OPS = {
    "add": (arith.addf, arith.addi),
    "sub": (arith.subf, arith.subi),
    "mul": (arith.mulf, arith.muli),
}


def _binary_op(self, other, op, *, loc=None, ip=None):
    other = _coerce_other(self, other, loc=loc, ip=ip)
    if other is NotImplemented:
        return NotImplemented

    if op in _ARITH_OPS:
        float_fn, int_fn = _ARITH_OPS[op]
        if self.is_float:
            return float_fn(self, other, loc=loc, ip=ip)
        return int_fn(self, other, loc=loc, ip=ip)

    if op == "div":
        if self.is_float:
            return arith.divf(self, other, loc=loc, ip=ip)
        et = element_type(self.type)
        if isinstance(et, ir.IndexType):
            return arith.divui(self, other, loc=loc, ip=ip)
        fp_ty = T.f64() if et.width > 32 else T.f32()
        lhs = int_to_fp(self, self.signed, fp_ty, loc=loc, ip=ip)
        rhs = int_to_fp(other, other.signed, fp_ty, loc=loc, ip=ip)
        return arith.divf(lhs, rhs, loc=loc, ip=ip)

    if op == "floordiv":
        if self.is_float:
            q = arith.divf(self, other, loc=loc, ip=ip)
            return math.floor(q, loc=loc, ip=ip)
        if self.signed is not False:
            return arith.floordivsi(self, other, loc=loc, ip=ip)
        return arith.divui(self, other, loc=loc, ip=ip)

    if op == "mod":
        if self.is_float:
            return arith.remf(self, other, loc=loc, ip=ip)
        if self.signed is not False:
            return arith.remsi(self, other, loc=loc, ip=ip)
        return arith.remui(self, other, loc=loc, ip=ip)

    raise ValueError(f"unknown binary op: {op}")


def _rbinary_op(self, other, op, *, loc=None, ip=None):
    other = _coerce_other(self, other, loc=loc, ip=ip)
    if other is NotImplemented:
        return NotImplemented
    return _binary_op(other, self, op, loc=loc, ip=ip)


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


def _comparison_op(self, other, predicate, *, loc=None, ip=None):
    other = _coerce_other(self, other, loc=loc, ip=ip)
    if other is NotImplemented:
        return NotImplemented

    if self.is_float:
        return arith.cmpf(_CMP_FLOAT_PRED[predicate], self, other, loc=loc, ip=ip)
    if self.signed is not False:
        return arith.cmpi(_CMP_INT_SIGNED[predicate], self, other, loc=loc, ip=ip)
    return arith.cmpi(_CMP_INT_UNSIGNED[predicate], self, other, loc=loc, ip=ip)


_BITWISE_OPS = {
    "and": arith.andi,
    "or": arith.ori,
    "xor": arith.xori,
}


def _bitwise_op(self, other, op, reverse=False, *, loc=None, ip=None):
    other = _coerce_other(self, other, loc=loc, ip=ip)
    if other is NotImplemented:
        return NotImplemented
    fn = _BITWISE_OPS[op]
    if reverse:
        return fn(other, self, loc=loc, ip=ip)
    return fn(self, other, loc=loc, ip=ip)


def _shift_op(self, other, op, reverse=False, *, loc=None, ip=None):
    other = _coerce_other(self, other, loc=loc, ip=ip)
    if other is NotImplemented:
        return NotImplemented
    lhs, rhs = (other, self) if reverse else (self, other)
    if op == "shl":
        return arith.shli(lhs, rhs, loc=loc, ip=ip)
    signed = getattr(lhs, "signed", None)
    if signed is not False:
        return arith.shrsi(lhs, rhs, loc=loc, ip=ip)
    return arith.shrui(lhs, rhs, loc=loc, ip=ip)


def _pow_op(self, other, reverse=False, *, loc=None, ip=None):
    other = _coerce_other(self, other, loc=loc, ip=ip)
    if other is NotImplemented:
        return NotImplemented
    if reverse:
        self, other = other, self
    if self.is_float and other.is_float:
        return math.powf(self, other, loc=loc, ip=ip)
    if self.is_float and not other.is_float:
        return math.fpowi(self, other, loc=loc, ip=ip)
    if not self.is_float and other.is_float:
        fp_ty = element_type(other.type)
        lhs = int_to_fp(self, self.signed, fp_ty, loc=loc, ip=ip)
        return math.powf(lhs, other, loc=loc, ip=ip)
    return math.ipowi(self, other, loc=loc, ip=ip)


def _neg_op(self, *, loc=None, ip=None):
    if self.type == T.bool():
        raise TypeError("negation is not supported for boolean type")
    if self.is_float:
        return arith.negf(self, loc=loc, ip=ip)
    c0 = arith.constant(self.type, 0, loc=loc, ip=ip)
    return arith.subi(c0, self, loc=loc, ip=ip)


def _invert_op(self, *, loc=None, ip=None):
    return arith.xori(self, arith.constant(self.type, -1))


def _safe_register(typeid):
    """Register value caster, silently skipping if already registered."""
    def decorator(cls):
        try:
            ir.register_value_caster(typeid)(cls)
        except RuntimeError:
            pass
        return cls
    return decorator

@_safe_register(ir.Float4E2M1FNType.static_typeid)
@_safe_register(ir.Float6E2M3FNType.static_typeid)
@_safe_register(ir.Float6E3M2FNType.static_typeid)
@_safe_register(ir.Float8E4M3FNType.static_typeid)
@_safe_register(ir.Float8E4M3B11FNUZType.static_typeid)
@_safe_register(ir.Float8E5M2Type.static_typeid)
@_safe_register(ir.Float8E4M3Type.static_typeid)
@_safe_register(ir.Float8E8M0FNUType.static_typeid)
@_safe_register(ir.BF16Type.static_typeid)
@_safe_register(ir.F16Type.static_typeid)
@_safe_register(ir.F32Type.static_typeid)
@_safe_register(ir.F64Type.static_typeid)
@_safe_register(ir.IntegerType.static_typeid)
@_safe_register(ir.IndexType.static_typeid)
@_safe_register(ir.VectorType.static_typeid)
class ArithValue(ir.Value):
    def __init__(self, v, signed=None, *, loc=None, ip=None):
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

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return "?"

    def __repr__(self):
        return self.__str__()


# =========================================================================
# Function-level arith API
# =========================================================================

def _to_raw(v):
    """Convert ArithValue to raw ir.Value for nanobind MLIR ops."""
    if isinstance(v, ir.Value):
        return v
    return ir.Value._CAPICreate(v._CAPIPtr)


def constant(value, *, type=None, index=False, loc=None, ip=None):
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
        mlir_type = ir.IntegerType.get_signless(1)
    elif isinstance(value, int):
        mlir_type = ir.IntegerType.get_signless(32)
    else:
        raise ValueError(f"unsupported constant type: {builtins.type(value)}")
    if isinstance(mlir_type, (ir.F16Type, ir.F32Type, ir.F64Type, ir.BF16Type)):
        value = float(value)
    return arith.constant(mlir_type, value, loc=loc, ip=ip)


def index(value, *, loc=None, ip=None):
    """Create an index constant."""
    return constant(value, index=True, loc=loc, ip=ip)


def constant_vector(element_value, vector_type, *, loc=None):
    """Create a splat constant vector."""
    elem_ty = element_type(vector_type)
    if is_float_type(elem_ty):
        attr = ir.FloatAttr.get(elem_ty, float(element_value))
    else:
        attr = ir.IntegerAttr.get(elem_ty, int(element_value))
    dense = ir.DenseElementsAttr.get_splat(vector_type, attr)
    return arith.constant(vector_type, dense, loc=loc)


def index_cast(target_type, value, *, loc=None):
    """Cast between index and integer types."""
    v = _to_raw(value)
    return arith.IndexCastOp(target_type, v, loc=loc).result


def select(condition, true_value, false_value, *, loc=None):
    """Select between two values based on a boolean condition."""
    return arith.SelectOp(_to_raw(condition), _to_raw(true_value),
                          _to_raw(false_value), loc=loc).result


def sitofp(target_type, value, *, loc=None):
    """Convert signed integer to floating point."""
    return arith.SIToFPOp(target_type, _to_raw(value), loc=loc).result


def trunc_f(target_type, value, *, loc=None):
    """Truncate floating point to narrower type (e.g. f32 -> f16)."""
    return arith.TruncFOp(target_type, _to_raw(value), loc=loc).result


def andi(lhs, rhs, *, loc=None):
    """Bitwise AND."""
    return arith.AndIOp(_to_raw(lhs), _to_raw(rhs), loc=loc).result


def xori(lhs, rhs, *, loc=None):
    """Bitwise XOR."""
    return arith.XOrIOp(_to_raw(lhs), _to_raw(rhs), loc=loc).result


def shli(lhs, rhs, *, loc=None):
    """Left shift."""
    return arith.ShLIOp(_to_raw(lhs), _to_raw(rhs), loc=loc).result


def unwrap(val, *, type=None, index=False, loc=None):
    """Unwrap ArithValue to raw ir.Value. Materializes Python scalars."""
    if isinstance(val, (int, float, bool)):
        return _to_raw(constant(val, type=type, index=index, loc=loc))
    return _to_raw(val)
