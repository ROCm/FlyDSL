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
    "maxnumf",
    "maximumf",
    "minimumf",
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


@dsl_loc_tracing
def maxnumf(a, b, **kwargs):
    """Floating-point maximum, returning the non-NaN operand when one input is NaN (libm ``fmax``)."""
    from .numeric import Numeric
    from .typing import Vector

    result = arith.maxnumf(as_ir_value(a), as_ir_value(b), **kwargs)
    if isinstance(a, Vector):
        return Vector(result, a.shape, a.dtype)
    if isinstance(a, Numeric):
        return Numeric.from_ir_type(result.type)(result)
    return result


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
