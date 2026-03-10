from .._mlir.dialects.arith import *  # noqa: F401,F403
"""Arith dialect API — operator overloading + function-level builders.

Usage:
    from flydsl.expr import arith

    c = arith.constant(42, index=True)
    v = arith.index_cast(T.index, val)
    r = arith.select(cond, a, b)
    # ArithValue operator overloading: c + 1, c * 2, c / 4, c % 16
"""
from .utils.arith import (  # noqa: F401
    ArithValue,
    constant,
    constant_vector,
    index,
    index_cast,
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


def cmpi(predicate, lhs, rhs, **kwargs):
    return _mlir_arith.cmpi(predicate, _to_raw(lhs), _to_raw(rhs), **kwargs)


def cmpf(predicate, lhs, rhs, **kwargs):
    return _mlir_arith.cmpf(predicate, _to_raw(lhs), _to_raw(rhs), **kwargs)

