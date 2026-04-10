# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TensorSSA — register vector tensor with value semantics.

Wraps a flat ``vector<NxTy>`` MLIR SSA value with shape and dtype metadata.
All arithmetic returns TensorSSA with auto type-promotion and scalar broadcast.

Usage in kernels::

    vec = _load_vec(in_div, idx)       # → TensorSSA
    x = vec.to(Float32)               # type conversion
    x2 = x * x                        # elementwise (auto-broadcast scalars)
    s = x2.reduce(ReductionOp.ADD)    # full reduction → scalar
    y = x * rstd * g                  # scalar rstd auto-broadcasts
    out = y.to(BFloat16)              # convert back
"""

import enum
import operator
from inspect import isclass
from typing import Type, Union

from .._mlir import ir
from .._mlir.dialects import arith as _arith, vector as _vector
from .._mlir.extras import types as T

from .utils.arith import (
    ArithValue,
    element_type,
    fp_to_fp,
    fp_to_int,
    int_to_fp,
    int_to_int,
    _to_raw,
)
from .numeric import (
    Numeric,
    Integer,
    Float,
    Boolean,
    as_numeric,
    Int32,
    Float32,
    Float16,
    BFloat16,
    Float64,
)


# ═══════════════════════════════════════════════════════════════════════
# ReductionOp enum
# ═══════════════════════════════════════════════════════════════════════

class ReductionOp(enum.Enum):
    """Predefined reduction operators for TensorSSA.reduce()."""
    ADD = "add"
    MUL = "mul"
    MAX = "max"
    MIN = "min"


# ═══════════════════════════════════════════════════════════════════════
# Type promotion
# ═══════════════════════════════════════════════════════════════════════

_FLOAT_RANK = {Float64: 3, Float32: 2, Float16: 1, BFloat16: 1}
_FLOAT_BY_MIN_WIDTH = {16: Float16, 32: Float32, 64: Float64}


def _widen_float(float_type, min_width):
    """Return the narrowest standard float type with width >= *min_width*."""
    if float_type.width >= min_width:
        return float_type
    for w in (32, 64):
        if w >= min_width:
            return _FLOAT_BY_MIN_WIDTH[w]
    return Float64


def _resolve_promote_type(a_type, b_type):
    """Resolve the promoted result type for two Numeric types."""
    if a_type is b_type:
        return a_type

    a_float = a_type.is_float if hasattr(a_type, 'is_float') else False
    b_float = b_type.is_float if hasattr(b_type, 'is_float') else False

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


def _binary_op_type_promote(a, b):
    """Promote operands for binary operations.

    Returns (lhs, rhs, result_dtype) where lhs/rhs are converted as needed.
    Works with TensorSSA and Numeric operands.
    """
    a_dtype = a.dtype if isinstance(a, TensorSSA) else type(a)
    b_dtype = b.dtype if isinstance(b, TensorSSA) else type(b)

    if a_dtype is b_dtype:
        return a, b, a_dtype

    res_dtype = _resolve_promote_type(a_dtype, b_dtype)

    if isinstance(a, TensorSSA) and a.dtype is not res_dtype:
        a = a.to(res_dtype)
    elif isinstance(a, Numeric) and type(a) is not res_dtype:
        a = a.to(res_dtype)

    if isinstance(b, TensorSSA) and b.dtype is not res_dtype:
        b = b.to(res_dtype)
    elif isinstance(b, Numeric) and type(b) is not res_dtype:
        b = b.to(res_dtype)

    return a, b, res_dtype


# ═══════════════════════════════════════════════════════════════════════
# TensorSSA
# ═══════════════════════════════════════════════════════════════════════

class TensorSSA(ArithValue):
    """Thread-local register vector tensor with value semantics (immutable).

    Wraps a flat ``vector<NxTy>`` ir.Value with shape and dtype metadata.
    All arithmetic returns TensorSSA with auto type promotion and scalar broadcast.

    :param value: Flat vector as ir.Value
    :param shape: Shape as int or tuple of ints (e.g. ``8`` or ``(8,)``)
    :param dtype: Element data type (Numeric class, e.g. ``Float32``)
    """

    def __init__(self, value, shape, dtype: Type[Numeric]):
        if not isinstance(value, ir.Value) and hasattr(value, "ir_value"):
            value = value.ir_value()

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
        return f"TensorSSA({self.type} o {self._shape}, {self._dtype.__name__})"

    def __repr__(self):
        return self.__str__()

    # ── scf.for integration ──────────────────────────────────────────
    def __fly_values__(self):
        return [self]

    def __fly_construct__(cls_unused, values, *, shape=None, dtype=None):
        return values  # placeholder, handled by __new_from_mlir_values__

    def __extract_mlir_values__(self):
        return [self]

    def __new_from_mlir_values__(self, values):
        return TensorSSA(values[0], self._shape, self._dtype)

    # ── Core: _apply_op ───────────────────────────────────────────────

    def _apply_op(self, op, other, flip=False, *, loc=None, ip=None):
        # 1. Canonicalize non-TensorSSA, non-Numeric to Numeric
        if isinstance(other, (int, float, bool)):
            other = as_numeric(other)
        elif not isinstance(other, TensorSSA) and isinstance(other, ArithValue):
            # Raw ArithValue → treat as scalar Numeric
            try:
                other = Numeric.from_ir_type(element_type(other.type))(other)
            except (ValueError, KeyError):
                other = as_numeric(other)

        # 2. Type promotion
        lhs, rhs, res_dtype = _binary_op_type_promote(self, other)

        # 3. Scalar → vector broadcast
        if not isinstance(rhs, TensorSSA):
            if isinstance(rhs, Numeric):
                rhs_val = rhs.ir_value(loc=loc, ip=ip)
            else:
                rhs_val = _to_raw(rhs)
            vec_val = _vector.broadcast(lhs.type, rhs_val, loc=loc, ip=ip)
            rhs = TensorSSA(vec_val, lhs._shape, lhs.dtype)

        if flip:
            lhs, rhs = rhs, lhs

        # Comparison ops return Boolean dtype
        if op in (operator.lt, operator.le, operator.gt, operator.ge,
                  operator.eq, operator.ne):
            res_dtype = Boolean

        # 4. Compute via ArithValue operators
        lhs_av = ArithValue(lhs, lhs.signed)
        rhs_av = ArithValue(rhs, rhs.signed)
        res = op(lhs_av, rhs_av)

        # 5. Wrap result
        return TensorSSA(res, lhs._shape, res_dtype)

    # ── Arithmetic operators ─────────────────────────────────────────

    def __add__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.add, other, loc=loc, ip=ip)

    def __radd__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.add, other, flip=True, loc=loc, ip=ip)

    def __sub__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.sub, other, loc=loc, ip=ip)

    def __rsub__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.sub, other, flip=True, loc=loc, ip=ip)

    def __mul__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.mul, other, loc=loc, ip=ip)

    def __rmul__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.mul, other, flip=True, loc=loc, ip=ip)

    def __truediv__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.truediv, other, loc=loc, ip=ip)

    def __rtruediv__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.truediv, other, flip=True, loc=loc, ip=ip)

    def __floordiv__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.floordiv, other, loc=loc, ip=ip)

    def __rfloordiv__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.floordiv, other, flip=True, loc=loc, ip=ip)

    def __mod__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.mod, other, loc=loc, ip=ip)

    def __rmod__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.mod, other, flip=True, loc=loc, ip=ip)

    def __pow__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.pow, other, loc=loc, ip=ip)

    def __rpow__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.pow, other, flip=True, loc=loc, ip=ip)

    def __neg__(self, *, loc=None, ip=None):
        return self._apply_op(operator.sub, 0, flip=True, loc=loc, ip=ip)

    # ── Comparison operators ─────────────────────────────────────────

    def __lt__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.lt, other, loc=loc, ip=ip)

    def __le__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.le, other, loc=loc, ip=ip)

    def __eq__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.eq, other, loc=loc, ip=ip)

    def __ne__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.ne, other, loc=loc, ip=ip)

    def __gt__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.gt, other, loc=loc, ip=ip)

    def __ge__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.ge, other, loc=loc, ip=ip)

    # ── Bitwise operators ────────────────────────────────────────────

    def __and__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.and_, other, loc=loc, ip=ip)

    def __rand__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.and_, other, flip=True, loc=loc, ip=ip)

    def __or__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.or_, other, loc=loc, ip=ip)

    def __ror__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.or_, other, flip=True, loc=loc, ip=ip)

    def __xor__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.xor, other, loc=loc, ip=ip)

    def __rxor__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.xor, other, flip=True, loc=loc, ip=ip)

    def __lshift__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.lshift, other, loc=loc, ip=ip)

    def __rlshift__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.lshift, other, flip=True, loc=loc, ip=ip)

    def __rshift__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.rshift, other, loc=loc, ip=ip)

    def __rrshift__(self, other, *, loc=None, ip=None):
        return self._apply_op(operator.rshift, other, flip=True, loc=loc, ip=ip)

    # ── Type conversion ──────────────────────────────────────────────

    def to(self, dtype: Type[Numeric], *, loc=None, ip=None) -> "TensorSSA":
        """Convert the tensor to a different numeric type.

        :param dtype: Target Numeric type class (e.g. Float32, BFloat16)
        :return: New TensorSSA with converted elements
        """
        if dtype is ir.Value:
            return self

        if not isclass(dtype) or not issubclass(dtype, Numeric):
            raise TypeError(f"dtype must be a Numeric type, got {type(dtype)}")

        src_dtype = self._dtype
        if src_dtype is dtype:
            return self

        src_float = src_dtype.is_float if hasattr(src_dtype, 'is_float') else False
        dst_float = dtype.is_float if hasattr(dtype, 'is_float') else False

        if src_float and dst_float:
            res = fp_to_fp(self, dtype.ir_type, loc=loc, ip=ip)
        elif src_float and not dst_float:
            res = fp_to_int(self, dtype.signed, dtype.ir_type, loc=loc, ip=ip)
        elif not src_float and dst_float:
            res = int_to_fp(self, src_dtype.signed, dtype.ir_type, loc=loc, ip=ip)
        else:
            res = int_to_int(self, dtype, loc=loc, ip=ip)

        return TensorSSA(res, self._shape, dtype)

    def ir_value(self, *, loc=None, ip=None):
        """Return self as ir.Value (TensorSSA IS-A ir.Value)."""
        return self

    # ── Reduction ────────────────────────────────────────────────────

    def reduce(self, op: ReductionOp, init_val=None, reduction_profile=None,
               *, fastmath=None, loc=None, ip=None):
        """Full reduction over vector elements.

        :param op: Reduction operation (ReductionOp.ADD, .MAX, etc.)
        :param init_val: Initial accumulator value (optional)
        :param reduction_profile: Not used for 1D (reserved for multi-dim)
        :param fastmath: FastMath flags (optional)
        :return: Scalar result as Numeric or ArithValue
        """
        _kind_map = {
            ReductionOp.ADD: _vector.CombiningKind.ADD,
            ReductionOp.MUL: _vector.CombiningKind.MUL,
            ReductionOp.MAX: _vector.CombiningKind.MAXNUMF,
            ReductionOp.MIN: _vector.CombiningKind.MINIMUMF,
        }
        kind = _kind_map[op]
        et = element_type(self.type)

        kwargs = {}
        if fastmath is not None:
            kwargs["fastmath"] = fastmath

        if init_val is not None:
            if isinstance(init_val, Numeric):
                init_val = init_val.ir_value(loc=loc, ip=ip)
            kwargs["acc"] = _to_raw(init_val)

        return _vector.reduction(et, kind, self, loc=loc, ip=ip, **kwargs)

    # ── Element access ───────────────────────────────────────────────

    def __getitem__(self, idx):
        """Extract element at integer index, returning scalar."""
        if isinstance(idx, int):
            res = _vector.ExtractOp(
                self, static_position=[idx], dynamic_position=[]
            ).result
            return self._dtype(res)
        raise TypeError(f"unsupported index type: {type(idx)}")

    # ── Vector operations ────────────────────────────────────────────

    def bitcast(self, dtype: Type[Numeric], *, loc=None, ip=None) -> "TensorSSA":
        """Reinterpret bits as different element type.

        Element count may change if bit widths differ.
        """
        src_bits = self.numel * self._dtype.width
        dst_elem_bits = dtype.width
        dst_count = src_bits // dst_elem_bits
        dst_vec_ty = ir.VectorType.get([dst_count], dtype.ir_type)
        res = _vector.BitCastOp(dst_vec_ty, self, loc=loc, ip=ip).result
        return TensorSSA(res, (dst_count,), dtype)

    def shuffle(self, other, mask, *, loc=None, ip=None) -> "TensorSSA":
        """Vector shuffle with integer mask."""
        other_val = other if not isinstance(other, TensorSSA) else ir.Value(other)
        res = _vector.shuffle(self, other_val, mask, loc=loc, ip=ip)
        return TensorSSA(res, (len(mask),), self._dtype)

    # ── Hash (required since __eq__ is overridden) ───────────────────

    def __hash__(self):
        return super().__hash__()


# ═══════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════

def full(shape, fill_value, dtype: Type[Numeric], *, loc=None, ip=None) -> TensorSSA:
    """Create a TensorSSA filled with *fill_value*.

    :param shape: int or tuple of ints
    :param fill_value: scalar value
    :param dtype: Element type
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    n = 1
    for s in shape:
        n *= s

    if isinstance(fill_value, (int, float, bool)):
        fill_value = dtype(fill_value)
    elif isinstance(fill_value, Numeric):
        fill_value = fill_value.to(dtype, loc=loc, ip=ip)
    else:
        raise ValueError(f"expected numeric fill_value, got {type(fill_value)}")

    vec_ty = ir.VectorType.get([n], dtype.ir_type)
    val = _vector.broadcast(vec_ty, fill_value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    return TensorSSA(val, shape, dtype)


def full_like(a: TensorSSA, fill_value, dtype=None, *, loc=None, ip=None) -> TensorSSA:
    """Create a TensorSSA with same shape/dtype as *a*, filled with *fill_value*."""
    if dtype is None:
        dtype = a.dtype
    return full(a.shape, fill_value, dtype, loc=loc, ip=ip)


def zeros_like(a: TensorSSA, dtype=None, *, loc=None, ip=None) -> TensorSSA:
    """Create a zero-filled TensorSSA with same shape/dtype as *a*."""
    if dtype is None:
        dtype = a.dtype
    zero = 0.0 if dtype.is_float else 0
    return full(a.shape, zero, dtype, loc=loc, ip=ip)
