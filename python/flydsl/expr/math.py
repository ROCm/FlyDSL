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

import inspect
from functools import wraps

from .._mlir import ir
from .._mlir.dialects import arith, llvm, math, vector
from .meta import dsl_loc_tracing
from .numeric import Numeric
from .typing import as_ir_value
from .utils.arith import current_fastmath

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


def dsl_math_wrap_result(fn=None, *, exemplar=None, preserve_numeric_type=False):
    """Wrap raw builder results back into DSL ``Numeric`` / ``Vector`` values.

    The DSL type of the result is shaped after an *exemplar* operand:

    - ``exemplar=None`` (default): the first positional argument is the
      exemplar. This fits the ``x``-first math builders (``exp(x)``, ...).
    - ``exemplar="<name>"``: the argument bound to that parameter name is the
      exemplar. Use this for builders whose first argument is not the operand,
      e.g. ``cmpi(predicate, lhs, rhs)`` -> ``exemplar="lhs"``.
    - ``preserve_numeric_type=True``: preserve the exemplar's exact DSL scalar
      type or vector dtype. Use this for same-type operations whose MLIR result
      does not encode DSL metadata such as integer signedness.
    """
    if fn is None:
        return lambda f: dsl_math_wrap_result(
            f,
            exemplar=exemplar,
            preserve_numeric_type=preserve_numeric_type,
        )

    sig = inspect.signature(fn)
    accepts_fastmath = "fastmath" in sig.parameters

    @wraps(fn)
    def wrapper(*args, **kwargs):
        from .typing import Vector

        if accepts_fastmath and kwargs.get("fastmath") is None:
            ambient = current_fastmath()
            if ambient is not None:
                kwargs["fastmath"] = ambient

        if exemplar is None:
            first = args[0] if args else None
        else:
            try:
                bound = sig.bind_partial(*args, **kwargs)
                first = bound.arguments.get(exemplar)
            except TypeError:
                first = kwargs.get(exemplar)
        is_vector = isinstance(first, Vector)
        is_numeric = isinstance(first, Numeric)

        result = fn(*args, **kwargs)

        if not (is_vector or is_numeric):
            return tuple(result) if not isinstance(result, ir.Value) and hasattr(result, "__iter__") else result

        def dsl_wrap(value):
            if not isinstance(value, ir.Value):
                return value
            if is_vector:
                if preserve_numeric_type:
                    return Vector(value, first.shape, first.dtype)
                elem_dtype = Numeric.from_ir_type(ir.VectorType(value.type).element_type)
                return Vector(value, first.shape, elem_dtype)
            if preserve_numeric_type:
                return type(first)(value)
            return Numeric.from_ir_type(value.type)(value)

        if isinstance(result, ir.Value):
            return dsl_wrap(result)
        return tuple(dsl_wrap(r) for r in result)

    return wrapper


def _float_splat(ty, value):
    """A constant ``value`` of float type ``ty``, splatted when ``ty`` is a vector type."""
    if isinstance(ty, ir.VectorType):
        return arith.constant(ty, ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get(ty.element_type, value)))
    return arith.constant(ty, ir.FloatAttr.get(ty, value))


def _lanewise(scalar_fn, x):
    """``scalar_fn(x)`` for a scalar ``x``; lane by lane for a vector ``x``, reassembled into its shape."""
    ty = x.type
    if not isinstance(ty, ir.VectorType):
        return scalar_fn(x)
    n = 1
    for extent in ty.shape:
        n *= extent
    flat_ty = ir.VectorType.get([n], ty.element_type)
    flat = vector.shape_cast(flat_ty, x) if ty.rank != 1 else x
    lanes = [scalar_fn(vector.extract(flat, [], [i])) for i in range(n)]
    out = vector.from_elements(flat_ty, lanes)
    return vector.shape_cast(ty, out) if ty.rank != 1 else out


def _via_f32(scalar_fn, x):
    """``scalar_fn`` computed in f32 for a float scalar of another type."""
    return arith.truncf(x.type, scalar_fn(arith.extf(ir.F32Type.get(), x)))


_OCML_SUFFIXES = ((ir.F16Type, "f16"), (ir.F32Type, "f32"), (ir.F64Type, "f64"))


def _ocml_declared(symbol, fn_type):
    """Declare ``symbol`` in the nearest enclosing ``gpu.module`` / ``builtin.module`` unless it is there."""
    op = ir.InsertionPoint.current.block.owner
    while op.operation.name not in ("gpu.module", "builtin.module"):
        op = op.operation.parent
    if symbol not in ir.SymbolTable(op.operation):
        with ir.InsertionPoint.at_block_begin(op.operation.regions[0].blocks[0]):
            llvm.LLVMFuncOp(symbol, ir.TypeAttr.get(fn_type), sym_visibility="private")
    return ir.FlatSymbolRefAttr.get(symbol)


def _ocml_scalar(name, x):
    """``__ocml_<name>_<f16|f32|f64>(x)`` for a float scalar; other float types are computed in f32."""
    ty = x.type
    suffix = next((s for cls, s in _OCML_SUFFIXES if isinstance(ty, cls)), None)
    if suffix is None:
        return _via_f32(lambda v: _ocml_scalar(name, v), x)
    callee = _ocml_declared(f"__ocml_{name}_{suffix}", ir.Type.parse(f"!llvm.func<{ty} ({ty})>"))
    return llvm.CallOp(ty, [x], [], ir.DenseI32ArrayAttr.get([]), callee=callee).result


_LN2 = 0.6931471805599453


def _native_log_scalar(x):
    """``log(x)`` as the hardware ``log2`` (``rocdl.log``, ``v_log_f32`` / ``v_log_f16``) times ln 2, for a float
    scalar that is 0 or not denormal (the instruction flushes denormal inputs); f64 uses ``math.log``, other
    float types f32. Without the ROCDL bindings, ``math.log``."""
    try:
        from .._mlir.dialects import rocdl
    except ImportError:
        return math.log(x)
    ty = x.type
    if isinstance(ty, ir.F64Type):
        return math.log(x)
    if not isinstance(ty, (ir.F32Type, ir.F16Type)):
        return _via_f32(_native_log_scalar, x)
    return arith.mulf(rocdl.log(ty, x), _float_splat(ty, _LN2))


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
    """``exp(x) - 1``, accurate for small ``|x|``: a call to OCML's ``__ocml_expm1_*``.

    ``math.expm1`` is not emitted: ``convert-gpu-to-rocdl`` lowers it to ``exp(x) - 1``, which is 0 for
    ``|x|`` below half an ulp of 1 (6e-8 in f32). ``fastmath`` is accepted for API compatibility only.
    """
    return _lanewise(lambda v: _ocml_scalar("expm1", v), as_ir_value(x))


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
    """``log(1 + x)``, accurate for small ``|x|``: ``log(w) + c`` with ``w = 1 + x`` and ``c = x - (w - 1)``.

    ``math.log1p`` is not emitted: ``convert-gpu-to-rocdl`` lowers it to ``log(1 + x)``, which is 0 for
    ``|x|`` below half an ulp of 1 (6e-8 in f32).

    For ``|x| <= 1``, ``c`` is the exact rounding error of ``w``, so ``log(1 + x) = log(w) + c / w`` to first
    order. For ``w < 2``, ``c`` stands in for ``c / w`` at a cost of about an ulp: ``|c|`` is at most half an
    ulp of 1, and ``|c - c / w|`` is at most ``|c| * |log w|`` for ``w >= 1`` and ``1.5 |c| * |log w|`` for
    ``0.5 <= w < 1`` (below 0.5, ``w`` is exact and ``c == 0``). For ``w >= 2``, ``|c / w|`` is below an ulp
    of ``log w`` and ``c`` is dropped. ``w == 1`` returns ``x`` exactly, except that ``-0`` gives ``+0``;
    ``+inf``, ``-1``, ``x < -1`` and NaN give ``+inf``, ``-inf``, NaN and NaN.

    ``log(w)`` is the hardware ``log2`` times ln 2, one transcendental instruction (``w`` is 0 or at least
    2^-24, never denormal); the result is within about 4 ulps in f32. ``fastmath`` is not applied:
    reassociating ``(w - 1)`` would undo the correction.
    """
    x = as_ir_value(x)
    ty = x.type
    one = _float_splat(ty, 1.0)
    w = arith.addf(one, x)
    c = arith.subf(x, arith.subf(w, one))
    c = arith.select(arith.cmpf(arith.CmpFPredicate.OLT, w, _float_splat(ty, 2.0)), c, _float_splat(ty, 0.0))
    return arith.addf(_lanewise(_native_log_scalar, w), c)


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
