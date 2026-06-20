# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Math dialect API — thin DSL wrappers over the MLIR ``math`` dialect.

Usage:
    import flydsl.expr as fx

    y = fx.exp(x)
    y = fx.sqrt(x, fastmath="fast")
    y = fx.fma(a, b, c)
    pred = fx.isnan(x)
"""

from functools import wraps

from .._mlir import ir
from .._mlir.dialects import math
from .meta import dsl_loc_tracing
from .numeric import Numeric
from .typing import as_ir_value

__all__ = [
    "absf",
    "ceil",
    "floor",
    "trunc",
    "round",
    "roundeven",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "rsqrt",
    "cbrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "sincos",
    "absi",
    "ctlz",
    "cttz",
    "ctpop",
    "powf",
    "fpowi",
    "ipowi",
    "atan2",
    "copysign",
    "fma",
    "clampf",
    "isnan",
    "isinf",
    "isfinite",
    "isnormal",
]


def dsl_math_wrap_result(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from .typing import Vector

        first = args[0] if args else None
        is_vector = isinstance(first, Vector)
        is_numeric = isinstance(first, Numeric)

        result = fn(*args, **kwargs)

        if not (is_vector or is_numeric):
            return tuple(result) if not isinstance(result, ir.Value) and hasattr(result, "__iter__") else result

        def dsl_wrap(value):
            if not isinstance(value, ir.Value):
                return value
            if is_vector:
                elem_dtype = Numeric.from_ir_type(ir.VectorType(value.type).element_type)
                return Vector(value, first.shape, elem_dtype)
            return Numeric.from_ir_type(value.type)(value)

        if isinstance(result, ir.Value):
            return dsl_wrap(result)
        return tuple(dsl_wrap(r) for r in result)

    return wrapper


# ---------------------------------------------------------------------------
# Unary float ops
# ---------------------------------------------------------------------------


@dsl_loc_tracing
@dsl_math_wrap_result
def absf(x, *, fastmath=None, **kwargs):
    return math.absf(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def ceil(x, *, fastmath=None, **kwargs):
    return math.ceil(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def floor(x, *, fastmath=None, **kwargs):
    return math.floor(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def trunc(x, *, fastmath=None, **kwargs):
    return math.trunc(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def round(x, *, fastmath=None, **kwargs):
    return math.round(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def roundeven(x, *, fastmath=None, **kwargs):
    return math.roundeven(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def exp(x, *, fastmath=None, **kwargs):
    return math.exp(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def exp2(x, *, fastmath=None, **kwargs):
    return math.exp2(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def expm1(x, *, fastmath=None, **kwargs):
    return math.expm1(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def log(x, *, fastmath=None, **kwargs):
    return math.log(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def log2(x, *, fastmath=None, **kwargs):
    return math.log2(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def log10(x, *, fastmath=None, **kwargs):
    return math.log10(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def log1p(x, *, fastmath=None, **kwargs):
    return math.log1p(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def sqrt(x, *, fastmath=None, **kwargs):
    return math.sqrt(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def rsqrt(x, *, fastmath=None, **kwargs):
    return math.rsqrt(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def cbrt(x, *, fastmath=None, **kwargs):
    return math.cbrt(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def sin(x, *, fastmath=None, **kwargs):
    return math.sin(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def cos(x, *, fastmath=None, **kwargs):
    return math.cos(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def tan(x, *, fastmath=None, **kwargs):
    return math.tan(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def asin(x, *, fastmath=None, **kwargs):
    return math.asin(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def acos(x, *, fastmath=None, **kwargs):
    return math.acos(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def atan(x, *, fastmath=None, **kwargs):
    return math.atan(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def sinh(x, *, fastmath=None, **kwargs):
    return math.sinh(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def cosh(x, *, fastmath=None, **kwargs):
    return math.cosh(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def tanh(x, *, fastmath=None, **kwargs):
    return math.tanh(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def asinh(x, *, fastmath=None, **kwargs):
    return math.asinh(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def acosh(x, *, fastmath=None, **kwargs):
    return math.acosh(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def atanh(x, *, fastmath=None, **kwargs):
    return math.atanh(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def erf(x, *, fastmath=None, **kwargs):
    return math.erf(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def erfc(x, *, fastmath=None, **kwargs):
    return math.erfc(as_ir_value(x), fastmath=fastmath, **kwargs)


# ---------------------------------------------------------------------------
# Multi-result unary float ops
# ---------------------------------------------------------------------------


@dsl_loc_tracing
@dsl_math_wrap_result
def sincos(x, *, fastmath=None, **kwargs):
    """Simultaneous sin and cos.  Returns ``(sin(x), cos(x))``."""
    return math.sincos(as_ir_value(x), fastmath=fastmath, **kwargs)


# ---------------------------------------------------------------------------
# Unary integer ops
# ---------------------------------------------------------------------------


@dsl_loc_tracing
@dsl_math_wrap_result
def absi(x, **kwargs):
    return math.absi(as_ir_value(x), **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def ctlz(x, **kwargs):
    return math.ctlz(as_ir_value(x), **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def cttz(x, **kwargs):
    return math.cttz(as_ir_value(x), **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def ctpop(x, **kwargs):
    return math.ctpop(as_ir_value(x), **kwargs)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------


@dsl_loc_tracing
@dsl_math_wrap_result
def powf(base, exp, *, fastmath=None, **kwargs):
    return math.powf(as_ir_value(base), as_ir_value(exp), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def fpowi(base, exp, *, fastmath=None, **kwargs):
    return math.fpowi(as_ir_value(base), as_ir_value(exp), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def ipowi(base, exp, **kwargs):
    return math.ipowi(as_ir_value(base), as_ir_value(exp), **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def atan2(y, x, *, fastmath=None, **kwargs):
    return math.atan2(as_ir_value(y), as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def copysign(mag, sign, *, fastmath=None, **kwargs):
    return math.copysign(as_ir_value(mag), as_ir_value(sign), fastmath=fastmath, **kwargs)


# ---------------------------------------------------------------------------
# Ternary ops
# ---------------------------------------------------------------------------


@dsl_loc_tracing
@dsl_math_wrap_result
def fma(a, b, c, *, fastmath=None, **kwargs):
    return math.fma(as_ir_value(a), as_ir_value(b), as_ir_value(c), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def clampf(x, lo, hi, *, fastmath=None, **kwargs):
    return math.clampf(as_ir_value(x), as_ir_value(lo), as_ir_value(hi), fastmath=fastmath, **kwargs)


# ---------------------------------------------------------------------------
# Predicates :: Float -> Boolean
# ---------------------------------------------------------------------------


@dsl_loc_tracing
@dsl_math_wrap_result
def isnan(x, *, fastmath=None, **kwargs):
    return math.isnan(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def isinf(x, *, fastmath=None, **kwargs):
    return math.isinf(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def isfinite(x, *, fastmath=None, **kwargs):
    return math.isfinite(as_ir_value(x), fastmath=fastmath, **kwargs)


@dsl_loc_tracing
@dsl_math_wrap_result
def isnormal(x, *, fastmath=None, **kwargs):
    return math.isnormal(as_ir_value(x), fastmath=fastmath, **kwargs)
