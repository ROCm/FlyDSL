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
    "andi",
    "constant",
    "constant_vector",
    "index",  # Deprecated: will be removed in a future release
    "index_cast",  # Deprecated: will be removed in a future release
    "int_to_fp",
    "shli",
    "sitofp",
    "trunc_f",
    "unwrap",  # Deprecated: will be removed in a future release
    "xori",
    "cmpi",
    "cmpf",
]

# Override star-import cmpi/cmpf to accept Numeric types (Int32, etc.)
from .._mlir.dialects import arith as _mlir_arith
from .meta import dsl_loc_tracing
from .utils.arith import (  # noqa: F401
    ArithValue,
    _to_raw,
    andi,
    constant,
    constant_vector,
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
    return _mlir_arith.cmpi(predicate, _to_raw(lhs), _to_raw(rhs), **kwargs)


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
    return _mlir_arith.cmpf(predicate, _to_raw(lhs), _to_raw(rhs), **kwargs)
