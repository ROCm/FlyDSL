# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Math dialect API — DSL-friendly wrappers with traced locations and auto-unwrap.

Usage:
    from flydsl.expr import math

    y = math.exp(x)
    y = math.sqrt(x, fastmath="fast")
    y = math.fma(a, b, c)
    pred = math.isnan(x)
"""

from functools import wraps

from .._mlir import ir
from .._mlir.dialects.math import *  # noqa: F401,F403
from .._mlir.dialects import math as _mlir_math
from .meta import _caller_location, _flatten_args
from .numeric import Numeric
from .utils.arith import _to_raw


def _traced_math_op(fn):
    """Like @traced_op, but re-wraps results to preserve Numeric class hierarchy.

    If the first positional arg is a Numeric (Float32, Int32, …), the MLIR
    result is wrapped back into the appropriate Numeric subclass via
    ``Numeric.from_ir_type``.  Raw ir.Value inputs pass through unchanged.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        first = args[0] if args else None
        do_rewrap = isinstance(first, Numeric)

        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = _caller_location(depth=1)
        args, kwargs = _flatten_args(args, kwargs)
        with loc:
            result = fn(*args, **kwargs)

        if not do_rewrap:
            return result
        if isinstance(result, ir.Value):
            return Numeric.from_ir_type(result.type)(result)
        # Multi-result (e.g. sincos)
        return tuple(Numeric.from_ir_type(r.type)(r) for r in result)

    return wrapper


# ---------------------------------------------------------------------------
# Unary float ops
# ---------------------------------------------------------------------------

@_traced_math_op
def absf(x, *, fastmath=None, **kw):
    return _mlir_math.absf(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def ceil(x, *, fastmath=None, **kw):
    return _mlir_math.ceil(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def floor(x, *, fastmath=None, **kw):
    return _mlir_math.floor(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def trunc(x, *, fastmath=None, **kw):
    return _mlir_math.trunc(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def round(x, *, fastmath=None, **kw):
    return _mlir_math.round(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def roundeven(x, *, fastmath=None, **kw):
    return _mlir_math.roundeven(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def exp(x, *, fastmath=None, **kw):
    return _mlir_math.exp(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def exp2(x, *, fastmath=None, **kw):
    return _mlir_math.exp2(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def expm1(x, *, fastmath=None, **kw):
    return _mlir_math.expm1(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def log(x, *, fastmath=None, **kw):
    return _mlir_math.log(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def log2(x, *, fastmath=None, **kw):
    return _mlir_math.log2(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def log10(x, *, fastmath=None, **kw):
    return _mlir_math.log10(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def log1p(x, *, fastmath=None, **kw):
    return _mlir_math.log1p(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def sqrt(x, *, fastmath=None, **kw):
    return _mlir_math.sqrt(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def rsqrt(x, *, fastmath=None, **kw):
    return _mlir_math.rsqrt(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def cbrt(x, *, fastmath=None, **kw):
    return _mlir_math.cbrt(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def sin(x, *, fastmath=None, **kw):
    return _mlir_math.sin(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def cos(x, *, fastmath=None, **kw):
    return _mlir_math.cos(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def tan(x, *, fastmath=None, **kw):
    return _mlir_math.tan(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def asin(x, *, fastmath=None, **kw):
    return _mlir_math.asin(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def acos(x, *, fastmath=None, **kw):
    return _mlir_math.acos(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def atan(x, *, fastmath=None, **kw):
    return _mlir_math.atan(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def sinh(x, *, fastmath=None, **kw):
    return _mlir_math.sinh(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def cosh(x, *, fastmath=None, **kw):
    return _mlir_math.cosh(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def tanh(x, *, fastmath=None, **kw):
    return _mlir_math.tanh(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def asinh(x, *, fastmath=None, **kw):
    return _mlir_math.asinh(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def acosh(x, *, fastmath=None, **kw):
    return _mlir_math.acosh(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def atanh(x, *, fastmath=None, **kw):
    return _mlir_math.atanh(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def erf(x, *, fastmath=None, **kw):
    return _mlir_math.erf(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def erfc(x, *, fastmath=None, **kw):
    return _mlir_math.erfc(_to_raw(x), fastmath=fastmath, **kw)


# ---------------------------------------------------------------------------
# Multi-result unary float ops
# ---------------------------------------------------------------------------

@_traced_math_op
def sincos(x, *, fastmath=None, **kw):
    """Simultaneous sin and cos.  Returns ``(sin(x), cos(x))``."""
    return _mlir_math.sincos(_to_raw(x), fastmath=fastmath, **kw)


# ---------------------------------------------------------------------------
# Unary integer ops
# ---------------------------------------------------------------------------

@_traced_math_op
def absi(x, **kw):
    return _mlir_math.absi(_to_raw(x), **kw)


@_traced_math_op
def ctlz(x, **kw):
    return _mlir_math.ctlz(_to_raw(x), **kw)


@_traced_math_op
def cttz(x, **kw):
    return _mlir_math.cttz(_to_raw(x), **kw)


@_traced_math_op
def ctpop(x, **kw):
    return _mlir_math.ctpop(_to_raw(x), **kw)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

@_traced_math_op
def powf(base, exp, *, fastmath=None, **kw):
    return _mlir_math.powf(_to_raw(base), _to_raw(exp), fastmath=fastmath, **kw)


@_traced_math_op
def fpowi(base, exp, *, fastmath=None, **kw):
    return _mlir_math.fpowi(_to_raw(base), _to_raw(exp), fastmath=fastmath, **kw)


@_traced_math_op
def ipowi(base, exp, **kw):
    return _mlir_math.ipowi(_to_raw(base), _to_raw(exp), **kw)


@_traced_math_op
def atan2(y, x, *, fastmath=None, **kw):
    return _mlir_math.atan2(_to_raw(y), _to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def copysign(mag, sign, *, fastmath=None, **kw):
    return _mlir_math.copysign(_to_raw(mag), _to_raw(sign), fastmath=fastmath, **kw)


# ---------------------------------------------------------------------------
# Ternary ops
# ---------------------------------------------------------------------------

@_traced_math_op
def fma(a, b, c, *, fastmath=None, **kw):
    return _mlir_math.fma(_to_raw(a), _to_raw(b), _to_raw(c), fastmath=fastmath, **kw)


@_traced_math_op
def clampf(x, lo, hi, *, fastmath=None, **kw):
    return _mlir_math.clampf(_to_raw(x), _to_raw(lo), _to_raw(hi), fastmath=fastmath, **kw)


# ---------------------------------------------------------------------------
# Predicates (return i1)
# ---------------------------------------------------------------------------

@_traced_math_op
def isnan(x, *, fastmath=None, **kw):
    return _mlir_math.isnan(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def isinf(x, *, fastmath=None, **kw):
    return _mlir_math.isinf(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def isfinite(x, *, fastmath=None, **kw):
    return _mlir_math.isfinite(_to_raw(x), fastmath=fastmath, **kw)


@_traced_math_op
def isnormal(x, *, fastmath=None, **kw):
    return _mlir_math.isnormal(_to_raw(x), fastmath=fastmath, **kw)
