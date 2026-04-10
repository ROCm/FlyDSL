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

import collections
import enum
import operator
from inspect import isclass
from typing import Type

from .._mlir import ir
from .._mlir.dialects import vector as _vector

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
    Boolean,
    as_numeric,
)


# ═══════════════════════════════════════════════════════════════════════
# Reduction kinds
# ═══════════════════════════════════════════════════════════════════════

class ReductionOp(enum.Enum):
    """Reduction operators for TensorSSA.reduce().

    Backward-compatible enum.  ``reduce()`` also accepts plain strings
    (``"add"``, ``"max"``, …) or raw ``vector.CombiningKind`` values.
    """
    ADD = "add"
    MUL = "mul"
    MAX = "max"
    MIN = "min"


# Maps a canonical string key → (float CombiningKind, signed-int CK, unsigned-int CK).
# Using a triple avoids rebuilding dicts per call.
_REDUCE_KINDS = {
    "add": (_vector.CombiningKind.ADD,  _vector.CombiningKind.ADD,  _vector.CombiningKind.ADD),
    "mul": (_vector.CombiningKind.MUL,  _vector.CombiningKind.MUL,  _vector.CombiningKind.MUL),
    "max": (_vector.CombiningKind.MAXNUMF, _vector.CombiningKind.MAXSI, _vector.CombiningKind.MAXUI),
    "min": (_vector.CombiningKind.MINIMUMF, _vector.CombiningKind.MINSI, _vector.CombiningKind.MINUI),
}


def _resolve_combining_kind(op, is_float, signed):
    """Normalize *op* (str | ReductionOp | CombiningKind) → ``CombiningKind``.

    :param op: Reduction specifier
    :param is_float: Whether the operand dtype is floating-point
    :param signed: Whether the operand dtype is signed (ignored for floats)
    """
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
    if is_float:
        return triple[0]
    return triple[1] if signed else triple[2]


# ═══════════════════════════════════════════════════════════════════════
# Type promotion
# ═══════════════════════════════════════════════════════════════════════

PromotedPair = collections.namedtuple("PromotedPair", ("lhs", "rhs", "dtype"))


def _promote_pair(a, b):
    """Promote operands for binary operations.

    Returns ``PromotedPair(lhs, rhs, dtype)`` where lhs/rhs are converted
    as needed.  Works with TensorSSA and Numeric operands.
    """
    a_dtype = a.dtype if isinstance(a, TensorSSA) else type(a)
    b_dtype = b.dtype if isinstance(b, TensorSSA) else type(b)

    if a_dtype is b_dtype:
        return PromotedPair(a, b, a_dtype)

    res_dtype = Numeric.promote(a_dtype, b_dtype)

    if isinstance(a, TensorSSA) and a.dtype is not res_dtype:
        a = a.to(res_dtype)
    elif isinstance(a, Numeric) and type(a) is not res_dtype:
        a = a.to(res_dtype)

    if isinstance(b, TensorSSA) and b.dtype is not res_dtype:
        b = b.to(res_dtype)
    elif isinstance(b, Numeric) and type(b) is not res_dtype:
        b = b.to(res_dtype)

    return PromotedPair(a, b, res_dtype)


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

    # ── FlyDSL protocol ────────────────────────────────────────────
    def __fly_values__(self):
        return [self]

    @classmethod
    def __fly_construct__(cls, values):
        return values[0]

    # ── Binary operation pipeline ────────────────────────────────────

    @staticmethod
    def _coerce_operand(other):
        """Coerce a non-TensorSSA operand into Numeric for type promotion."""
        if isinstance(other, (int, float, bool)):
            return as_numeric(other)
        if not isinstance(other, TensorSSA) and isinstance(other, ArithValue):
            try:
                return Numeric.from_ir_type(element_type(other.type))(other)
            except (ValueError, KeyError):
                return as_numeric(other)
        return other

    def _broadcast_to_vec(self, scalar, *, loc=None, ip=None):
        """Broadcast a scalar Numeric or ArithValue to match this tensor's shape."""
        if isinstance(scalar, Numeric):
            raw = scalar.ir_value(loc=loc, ip=ip)
        else:
            raw = _to_raw(scalar)
        vec_val = _vector.broadcast(self.type, raw, loc=loc, ip=ip)
        return TensorSSA(vec_val, self._shape, self.dtype)

    @staticmethod
    def _dispatch_arith(op, lhs, rhs, shape, res_dtype):
        """Unwrap to ArithValue, apply op, wrap result as TensorSSA."""
        lhs_av = ArithValue(lhs, lhs.signed)
        rhs_av = ArithValue(rhs, rhs.signed)
        return TensorSSA(op(lhs_av, rhs_av), shape, res_dtype)

    _CMP_OPS = frozenset({
        operator.lt, operator.le, operator.gt, operator.ge,
        operator.eq, operator.ne,
    })

    def _apply_op(self, op, other, flip=False, *, loc=None, ip=None):
        other = self._coerce_operand(other)
        pp = _promote_pair(self, other)
        lhs, rhs, res_dtype = pp.lhs, pp.rhs, pp.dtype

        if not isinstance(rhs, TensorSSA):
            rhs = lhs._broadcast_to_vec(rhs, loc=loc, ip=ip)

        if flip:
            lhs, rhs = rhs, lhs

        if op in self._CMP_OPS:
            res_dtype = Boolean

        return self._dispatch_arith(op, lhs, rhs, lhs._shape, res_dtype)

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

    def reduce(self, op, init_val=None, reduction_profile=None,
               *, fastmath=None, loc=None, ip=None):
        """Full reduction over vector elements.

        :param op: Reduction — ``ReductionOp`` enum, string (``"add"``,
            ``"max"``, ``"min"``, ``"mul"``), or raw ``CombiningKind``.
        :param init_val: Initial accumulator value (optional)
        :param reduction_profile: Not used for 1D (reserved for multi-dim)
        :param fastmath: FastMath flags (optional)
        :return: Scalar result as Numeric
        """
        is_fp = self._dtype.is_float
        signed = getattr(self._dtype, 'signed', True)
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

    # ── Factory classmethods ────────────────────────────────────────

    @classmethod
    def filled(cls, shape, fill_value, dtype: Type[Numeric],
               *, loc=None, ip=None) -> "TensorSSA":
        """Create a TensorSSA filled with *fill_value*.

        :param shape: int or tuple of ints
        :param fill_value: scalar value (int/float/bool/Numeric)
        :param dtype: Element type (e.g. Float32)
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
        val = _vector.broadcast(vec_ty, fill_value.ir_value(loc=loc, ip=ip),
                                loc=loc, ip=ip)
        return cls(val, shape, dtype)

    @classmethod
    def filled_like(cls, template: "TensorSSA", fill_value, dtype=None,
                    *, loc=None, ip=None) -> "TensorSSA":
        """Create a TensorSSA with same shape/dtype as *template*, filled with *fill_value*."""
        if dtype is None:
            dtype = template.dtype
        return cls.filled(template.shape, fill_value, dtype, loc=loc, ip=ip)

    @classmethod
    def zeros_like(cls, template: "TensorSSA", dtype=None,
                   *, loc=None, ip=None) -> "TensorSSA":
        """Create a zero-filled TensorSSA with same shape/dtype as *template*."""
        if dtype is None:
            dtype = template.dtype
        zero = 0.0 if dtype.is_float else 0
        return cls.filled(template.shape, zero, dtype, loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════════
# Backward-compatible module-level aliases
# ═══════════════════════════════════════════════════════════════════════

def full(shape, fill_value, dtype: Type[Numeric], *, loc=None, ip=None) -> TensorSSA:
    """Create a TensorSSA filled with *fill_value*.  Alias for ``TensorSSA.filled()``."""
    return TensorSSA.filled(shape, fill_value, dtype, loc=loc, ip=ip)


def full_like(a: TensorSSA, fill_value, dtype=None, *, loc=None, ip=None) -> TensorSSA:
    """Alias for ``TensorSSA.filled_like()``."""
    return TensorSSA.filled_like(a, fill_value, dtype, loc=loc, ip=ip)


def zeros_like(a: TensorSSA, dtype=None, *, loc=None, ip=None) -> TensorSSA:
    """Alias for ``TensorSSA.zeros_like()``."""
    return TensorSSA.zeros_like(a, dtype, loc=loc, ip=ip)
