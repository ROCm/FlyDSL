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
    shli,
    sitofp,
    trunc_f,
    unwrap,
    xori,
)
from .typing import as_ir_value


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
    """Floating-point maximum, returning the non-NaN operand when one input is NaN (libm ``fmax``).

    Accepts DSL numeric types (Float32, Vector, ...) and preserves the DSL type of ``a`` so the
    result can be chained with further DSL operations (e.g. ``.shuffle_xor(...)``).
    """
    from .numeric import Numeric
    from .typing import Vector

    result = arith.maxnumf(as_ir_value(a), as_ir_value(b), **kwargs)
    if isinstance(a, Vector):
        return Vector(result, a.shape, a.dtype)
    if isinstance(a, Numeric):
        return Numeric.from_ir_type(result.type)(result)
    return result
