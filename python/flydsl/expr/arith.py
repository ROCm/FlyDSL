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
    "max",
    "min",
]

# Override star-import cmpi/cmpf to accept Numeric types (Int32, etc.)
from .._mlir.dialects import arith as _mlir_arith
from .meta import dsl_loc_tracing
from .utils.arith import (  # noqa: F401
    ArithValue,
    _default_fastmath,
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


# ── Type-generic max / min ──────────────────────────────────────────────────
# One entry point for any DSL numeric type (Float32/Int32/Int64/unsigned/...) and
# Python scalars, any number of args. Reuses the shared numeric coercion
# (as_numeric + _coerce_operands) and dispatches by the resulting type:
#   float          -> maximumf / minimumf   (NaN-propagating, matches cutlass.max)
#   int, signed    -> maxsi / minsi
#   int, unsigned  -> maxui / minui
# The maximum-vs-maxnum choice is NOT exposed (matches cutlass). Return type
# follows the operands' static type.


def _minmax_pair(is_max, a, b):
    from .numeric import _coerce_operands, as_numeric

    a, b, out_ty = _coerce_operands(as_numeric(a), as_numeric(b))
    lv, rv = a.ir_value(), b.ir_value()
    if out_ty.is_float:
        fn = _mlir_arith.maximumf if is_max else _mlir_arith.minimumf
        res = fn(lv, rv, fastmath=_default_fastmath())
    elif out_ty.signed:
        fn = _mlir_arith.maxsi if is_max else _mlir_arith.minsi
        res = fn(lv, rv)
    else:
        fn = _mlir_arith.maxui if is_max else _mlir_arith.minui
        res = fn(lv, rv)
    return out_ty(res)


def _minmax(is_max, args):
    flat = []
    for a in args:
        if isinstance(a, (list, tuple)):
            flat.extend(a)
        else:
            flat.append(a)
    if not flat:
        raise ValueError("max()/min() requires at least one argument")
    acc = flat[0]
    for x in flat[1:]:
        acc = _minmax_pair(is_max, acc, x)
    return acc


@dsl_loc_tracing
def max(*args):
    """Type-generic maximum over any number of DSL numeric args (and Python scalars).

    Return type follows the operands' static types (not values). Accepts
    ``max(a, b)``, ``max(a, b, c, ...)``, ``max([a, b, ...])``, ``max(a, [x, y])``.
    Float uses NaN-propagating ``maximumf``; signed/unsigned int use ``maxsi``/``maxui``.
    """
    return _minmax(True, args)


@dsl_loc_tracing
def min(*args):
    """Type-generic minimum. See :func:`max`."""
    return _minmax(False, args)
