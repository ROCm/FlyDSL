# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: I001

"""Arith dialect API — operator overloading + function-level builders.

Usage:
    from flydsl.expr import arith

    c = arith.constant(42, index=True)
    v = arith.index_cast(T.index, val)
    r = arith.select(cond, a, b)
    # ArithValue operator overloading: c + 1, c * 2, c / 4, c % 16
"""

import builtins
import math as _math

from .._mlir import ir
from .._mlir.dialects.arith import *
from .._mlir.dialects import arith
from .math import dsl_math_wrap_result
from .meta import dsl_loc_tracing
from .utils.arith import (  # noqa: F401
    ArithValue,
    _to_raw,
    andi,
    constant,
    constant_vector,
    fastmath,
    index,
    index_cast,
    int_to_fp,
    select,
    shli,
    sitofp,
    trunc_f,
    unwrap,
    xori,
    resolve_fastmath,
)
from .typing import as_ir_value

__all__ = [
    "constant_vector",  # Deprecated: will be removed in a future release
    "index_cast",  # Deprecated: will be removed in a future release
    # Enums
    "FastMathFlags",
    "RoundingMode",
    # Fastmath context
    "fastmath",
    # Binary ops
    "cmpi",
    "cmpf",
    "max",
    "min",
    "maxnumf",
    "maximumf",
    "minimumf",
    "ceildiv",
    "shrui",
]


@dsl_loc_tracing
def cmpi(predicate, lhs, rhs, **kwargs):
    """Integer comparison accepting DSL numeric types (Int32, ArithValue, etc.).

    Args:
        predicate: ``arith.CmpIPredicate`` (e.g., ``eq``, ``slt``, ``uge``).
        lhs: Left-hand operand.
        rhs: Right-hand operand.

    Returns:
        An ``i1`` comparison result.
    """
    return arith.cmpi(predicate, as_ir_value(lhs), as_ir_value(rhs), **kwargs)


@dsl_loc_tracing
def cmpf(predicate, lhs, rhs, **kwargs):
    """Floating-point comparison accepting DSL numeric types.

    Args:
        predicate: ``arith.CmpFPredicate`` (e.g., ``olt``, ``oeq``, ``une``).
        lhs: Left-hand operand.
        rhs: Right-hand operand.

    Returns:
        An ``i1`` comparison result.
    """
    return arith.cmpf(predicate, as_ir_value(lhs), as_ir_value(rhs), **kwargs)


def _flatten_extreme_operands(operands):
    flat = []
    for operand in operands:
        if isinstance(operand, (list, tuple)):
            flat.extend(_flatten_extreme_operands(operand))
        else:
            flat.append(operand)
    return flat


def _normalize_typed_operand(value, operation):
    from .numeric import Numeric, as_numeric
    from .typing import Vector

    if isinstance(value, (Numeric, Vector)):
        return value
    if isinstance(value, ir.Value):
        try:
            if isinstance(value.type, ir.VectorType):
                return Vector(value)
            return Numeric.from_ir_type(value.type)(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError(f"{operation} does not support raw value type {value.type}") from exc
    if isinstance(value, (bool, int, float)):
        return as_numeric(value)
    raise TypeError(f"{operation} expects Numeric, Vector, ir.Value, or a numeric literal, got {type(value).__name__}")


def _validated_operand_dtype(dtype, operation, *, bool_widen, integer_only):
    from .numeric import BFloat16, Boolean, Float16, Float32, Float64, Index, Int32

    if dtype is Index:
        raise TypeError(f"{operation} does not support Index; cast to an explicit-width integer type")
    if dtype is Boolean:
        if not bool_widen:
            raise TypeError(f"{operation} does not support Boolean operands")
        return Int32
    if integer_only and not dtype.is_integer:
        raise TypeError(f"{operation} requires integer operands, got {dtype.__name__}")
    if dtype.is_float and dtype not in (Float16, BFloat16, Float32, Float64):
        raise TypeError(f"{operation} does not support narrow storage float {dtype.__name__}")
    return dtype


def _resolve_typed_operands(values, operation, *, bool_widen=False, integer_only=False):
    from .numeric import Numeric, _resolve_numeric_type
    from .typing import Vector

    normalized = [_normalize_typed_operand(value, operation) for value in values]
    dtypes = []
    for value in normalized:
        dtype = value.dtype if isinstance(value, Vector) else type(value)
        dtypes.append(
            _validated_operand_dtype(
                dtype,
                operation,
                bool_widen=bool_widen,
                integer_only=integer_only,
            )
        )

    common_dtype = dtypes[0]
    for dtype in dtypes[1:]:
        common_dtype = _resolve_numeric_type(common_dtype, dtype)
    if integer_only and not common_dtype.is_integer:
        raise TypeError(f"{operation} requires integer operands, got {common_dtype.__name__}")

    result_shape = None
    for value in normalized:
        if isinstance(value, Vector):
            result_shape = (
                value.shape if result_shape is None else Vector._infer_broadcast_shape(result_shape, value.shape)
            )

    prepared = []
    for value, dtype in zip(normalized, dtypes, strict=True):
        if isinstance(value, Vector):
            item = value if value.dtype is dtype else value.to(dtype)
            item = item if item.dtype is common_dtype else item.to(common_dtype)
            if item.shape != result_shape:
                item = item.broadcast_to(result_shape)
        else:
            item = value if type(value) is dtype else value.to(dtype)
            item = item if type(item) is common_dtype else item.to(common_dtype)
            if result_shape is not None:
                item = Vector.filled(result_shape, item, common_dtype)
        prepared.append(item)

    all_static = result_shape is None and all(isinstance(value, Numeric) and value.is_static() for value in prepared)
    return prepared, common_dtype, result_shape, all_static


def _wrap_typed_result(result, dtype, shape):
    if shape is None:
        return dtype(result)
    from .typing import Vector

    return Vector(result, shape, dtype)


def _fold_float_extreme(lhs, rhs, is_max):
    if _math.isnan(lhs) or _math.isnan(rhs):
        return float("nan")
    if lhs == 0.0 and rhs == 0.0:
        lhs_negative = _math.copysign(1.0, lhs) < 0.0
        rhs_negative = _math.copysign(1.0, rhs) < 0.0
        if is_max:
            return -0.0 if lhs_negative and rhs_negative else 0.0
        return -0.0 if lhs_negative or rhs_negative else 0.0
    return builtins.max(lhs, rhs) if is_max else builtins.min(lhs, rhs)


def _extreme(is_max, operands, *, fastmath=None, **kwargs):
    operands = _flatten_extreme_operands(operands)
    if not operands:
        name = "max" if is_max else "min"
        raise ValueError(f"fx.{name} requires at least one operand")

    prepared, dtype, shape, all_static = _resolve_typed_operands(
        operands,
        "max" if is_max else "min",
        bool_widen=True,
    )
    if len(prepared) == 1:
        return prepared[0]

    if all_static:
        result = prepared[0].value
        for operand in prepared[1:]:
            if dtype.is_float:
                result = _fold_float_extreme(result, operand.value, is_max)
            else:
                result = builtins.max(result, operand.value) if is_max else builtins.min(result, operand.value)
        return dtype(result)

    lhs = as_ir_value(prepared[0])
    if dtype.is_float:
        op = arith.maximumf if is_max else arith.minimumf
        fastmath = resolve_fastmath(fastmath)
        for operand in prepared[1:]:
            lhs = op(lhs, as_ir_value(operand), fastmath=fastmath, **kwargs)
    else:
        if dtype.signed:
            op = arith.maxsi if is_max else arith.minsi
        else:
            op = arith.maxui if is_max else arith.minui
        for operand in prepared[1:]:
            lhs = op(lhs, as_ir_value(operand), **kwargs)
    return _wrap_typed_result(lhs, dtype, shape)


@dsl_loc_tracing
def max(*operands, fastmath=None, **kwargs):
    """Return the type-promoted maximum of one or more numeric operands.

    Floating-point operands use NaN-propagating ``arith.maximumf``. Signed and
    unsigned integers use ``arith.maxsi`` and ``arith.maxui`` respectively.
    """
    return _extreme(True, operands, fastmath=fastmath, **kwargs)


@dsl_loc_tracing
def min(*operands, fastmath=None, **kwargs):
    """Return the type-promoted minimum of one or more numeric operands.

    Floating-point operands use NaN-propagating ``arith.minimumf``. Signed and
    unsigned integers use ``arith.minsi`` and ``arith.minui`` respectively.
    """
    return _extreme(False, operands, fastmath=fastmath, **kwargs)


@dsl_loc_tracing
def ceildiv(lhs, rhs, **kwargs):
    """Return integer ``lhs / rhs`` rounded toward positive infinity.

    The operation dispatches to ``arith.ceildivsi`` or ``arith.ceildivui`` from
    the promoted operand signedness. It is distinct from the layout/int-tuple
    :func:`flydsl.expr.ceil_div` API.
    """
    prepared, dtype, shape, all_static = _resolve_typed_operands(
        (lhs, rhs),
        "ceildiv",
        integer_only=True,
    )
    lhs, rhs = prepared
    if all_static:
        if rhs.value == 0:
            raise ZeroDivisionError("fx.ceildiv division by zero")
        if dtype.signed and lhs.value == -(1 << (dtype.width - 1)) and rhs.value == -1:
            raise OverflowError("fx.ceildiv signed division overflow")
        return dtype(-(-lhs.value // rhs.value))

    op = arith.ceildivsi if dtype.signed else arith.ceildivui
    result = op(as_ir_value(lhs), as_ir_value(rhs), **kwargs)
    return _wrap_typed_result(result, dtype, shape)


@dsl_loc_tracing
@dsl_math_wrap_result
def maxnumf(a, b, *, fastmath=None, **kwargs):
    """Floating-point maximum, returning the non-NaN operand when one input is NaN (libm ``fmax``)."""
    return arith.maxnumf(as_ir_value(a), as_ir_value(b), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def maximumf(a, b, *, fastmath=None, **kwargs):
    """NaN-propagating floating-point maximum."""
    return arith.maximumf(as_ir_value(a), as_ir_value(b), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def minimumf(a, b, *, fastmath=None, **kwargs):
    """NaN-propagating floating-point minimum."""
    return arith.minimumf(as_ir_value(a), as_ir_value(b), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result(preserve_numeric_type=True)
def shrui(value, amount, *, is_exact=None, **kwargs):
    """Unsigned right shift that preserves the DSL type of ``value``."""
    return arith.shrui(as_ir_value(value), as_ir_value(amount), is_exact=is_exact, **kwargs)
