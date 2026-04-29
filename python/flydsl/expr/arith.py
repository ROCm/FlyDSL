# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from .._mlir.dialects.arith import *  # noqa: F401,F403
"""Arith dialect API — operator overloading + function-level builders.

Usage:
    from flydsl.expr import arith

    c = arith.constant(42, index=True)
    v = arith.index_cast(T.index, val)
    r = arith.select(cond, a, b)
    # ArithValue operator overloading: c + 1, c * 2, c / 4, c % 16
"""
from .meta import traced_op
from .utils.arith import (  # noqa: F401
    ArithValue,
    constant,
    constant_vector,
    index,
    index_cast,
    int_to_fp,
    select,
    sitofp,
    trunc_f,
    andi,
    xori,
    shli,
    unwrap,
    _to_raw,
)

# Override star-import cmpi/cmpf to accept Numeric types (Int32, etc.)
from .._mlir.dialects import arith as _mlir_arith  # noqa: E402


@traced_op
def cmpi(predicate, lhs, rhs, **kwargs):
    """Integer comparison accepting DSL numeric types (Int32, ArithValue, etc.).

    Args:
        predicate: ``arith.CmpIPredicate`` (e.g., ``eq``, ``slt``, ``uge``).
        lhs: Left-hand operand.
        rhs: Right-hand operand.

    Returns:
        An ``i1`` comparison result.
    """
    return _mlir_arith.cmpi(predicate, _to_raw(lhs), _to_raw(rhs), **kwargs)


@traced_op
def cmpf(predicate, lhs, rhs, **kwargs):
    """Floating-point comparison accepting DSL numeric types.

    Args:
        predicate: ``arith.CmpFPredicate`` (e.g., ``olt``, ``oeq``, ``une``).
        lhs: Left-hand operand.
        rhs: Right-hand operand.

    Returns:
        An ``i1`` comparison result.
    """
    return _mlir_arith.cmpf(predicate, _to_raw(lhs), _to_raw(rhs), **kwargs)


@traced_op
def divui(lhs, rhs, **kwargs):
    """Unsigned integer divide accepting DSL types and Python int constants.

    Generates ``arith.divui`` (efficient ``udiv`` on AMD GPU).

    Args:
        lhs: Dividend (ArithValue, ir.Value, or DSL Numeric).
        rhs: Divisor (ArithValue, ir.Value, DSL Numeric, or Python int).
    """
    lhs_v = _to_raw(lhs)
    if isinstance(rhs, int):
        rhs_v = _to_raw(constant(rhs, type=lhs_v.type))
    else:
        rhs_v = _to_raw(rhs)
    return _mlir_arith.DivUIOp(lhs_v, rhs_v, **kwargs).result


@traced_op
def remui(lhs, rhs, **kwargs):
    """Unsigned integer remainder accepting DSL types and Python int constants.

    Generates ``arith.remui`` (efficient ``urem`` on AMD GPU).

    Args:
        lhs: Dividend (ArithValue, ir.Value, or DSL Numeric).
        rhs: Divisor (ArithValue, ir.Value, DSL Numeric, or Python int).
    """
    lhs_v = _to_raw(lhs)
    if isinstance(rhs, int):
        rhs_v = _to_raw(constant(rhs, type=lhs_v.type))
    else:
        rhs_v = _to_raw(rhs)
    return _mlir_arith.RemUIOp(lhs_v, rhs_v, **kwargs).result


def zext_i64(val):
    """Zero-extend integer value to i64, idempotent if already i64.

    Returns ArithValue for use in arithmetic expressions.
    """
    from .._mlir.extras import types as T
    v = _to_raw(val)
    i64 = T.i64()
    if v.type == i64:
        return v
    return _mlir_arith.ExtUIOp(i64, v).result


@traced_op
def select_by_index(index_val, values):
    """Select one of *values* by integer *index_val* via chained ``arith.select``.

    Equivalent to a compile-time switch: returns ``values[index_val]``.

    Args:
        index_val: Integer index (i32 ``ir.Value``).
        values: List of ``ir.Value`` to select from.

    Returns:
        The selected ``ir.Value``.
    """
    out = values[0]
    for i in range(1, len(values)):
        pred = _mlir_arith.CmpIOp(
            _mlir_arith.CmpIPredicate.eq, index_val, constant(i, type=index_val.type)
        ).result
        out = _mlir_arith.SelectOp(pred, values[i], out).result
    return out

