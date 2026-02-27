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
