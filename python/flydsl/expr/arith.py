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

from .._mlir.dialects.arith import *  # noqa: F401,F403

__all__ = [
    "ArithValue",  # Deprecated: will be removed in a future release
    "_to_raw",  # Deprecated: will be removed in a future release
    "FastMathFlags",
    "andi",
    "constant",
    "constant_vector",
    "fastmath",
    "index",  # Deprecated: will be removed in a future release
    "index_cast",  # Deprecated: will be removed in a future release
    "int_to_fp",
    "maxnumf",
    "minnumf",
    "maximumf",
    "minimumf",
    "shli",
    "sitofp",
    "trunc_f",
    "unwrap",  # Deprecated: will be removed in a future release
    "xori",
    "cmpi",
    "cmpf",
]

# Override star-import cmpi/cmpf to accept Numeric types (Int32, etc.)
from .._mlir.dialects import arith
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
from .math import dsl_math_wrap_result
from .typing import as_ir_value


@dsl_loc_tracing
@dsl_math_wrap_result
def cmpi(predicate, lhs, rhs, **kwargs):
    """Integer comparison accepting DSL numeric types (Int32, ArithValue, etc.).

    Args:
        predicate: ``arith.CmpIPredicate`` (e.g., ``eq``, ``slt``, ``uge``).
        lhs: Left-hand operand.
        rhs: Right-hand operand.

    Returns:
        A ``Boolean`` (scalar) or ``Vector(Boolean)`` comparison result.
    """
    return arith.cmpi(predicate, as_ir_value(lhs), as_ir_value(rhs), **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def cmpf(predicate, lhs, rhs, **kwargs):
    """Floating-point comparison accepting DSL numeric types.

    Args:
        predicate: ``arith.CmpFPredicate`` (e.g., ``olt``, ``oeq``, ``une``).
        lhs: Left-hand operand.
        rhs: Right-hand operand.

    Returns:
        A ``Boolean`` (scalar) or ``Vector(Boolean)`` comparison result.
    """
    return arith.cmpf(predicate, as_ir_value(lhs), as_ir_value(rhs), **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def maximumf(lhs, rhs, *, fastmath=None):
    return arith.maximumf(as_ir_value(lhs), as_ir_value(rhs), fastmath=fastmath)


@dsl_loc_tracing
@dsl_math_wrap_result
def minimumf(lhs, rhs, *, fastmath=None):
    return arith.minimumf(as_ir_value(lhs), as_ir_value(rhs), fastmath=fastmath)


@dsl_loc_tracing
@dsl_math_wrap_result
def maxnumf(lhs, rhs, *, fastmath=None):
    return arith.maxnumf(as_ir_value(lhs), as_ir_value(rhs), fastmath=fastmath)


@dsl_loc_tracing
@dsl_math_wrap_result
def minnumf(lhs, rhs, *, fastmath=None):
    return arith.minnumf(as_ir_value(lhs), as_ir_value(rhs), fastmath=fastmath)
