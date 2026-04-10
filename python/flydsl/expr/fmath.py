# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Math function dispatcher for TensorSSA and Numeric scalars.

Each function works on both TensorSSA (returns TensorSSA) and Numeric
scalars (returns scalar).

Usage::

    from flydsl.expr import fmath
    y = fmath.exp2(x, fastmath=True)
    z = fmath.rsqrt(x)
"""

from typing import Union

from .._mlir.dialects import arith as _arith, math as _math

from .numeric import Numeric
from .tensor_ssa import TensorSSA


def _math_op(func, fastmath, *args, loc=None, ip=None):
    """Dispatch *func* to TensorSSA or Numeric operands.

    :param func: MLIR math dialect function (e.g. ``math.exp2``)
    :param fastmath: Whether to enable fast-math flags
    :param args: Operands (TensorSSA or Numeric, must be same kind)
    """
    fm_flag = _arith.FastMathFlags.fast if fastmath else _arith.FastMathFlags.none

    if isinstance(args[0], TensorSSA):
        for a in args:
            if not isinstance(a, TensorSSA):
                raise TypeError(f"expected TensorSSA, got {type(a)}")
        result = func(*args, fastmath=fm_flag, loc=loc, ip=ip)
        return TensorSSA(result, args[0].shape, args[0].dtype)
    else:
        ir_args = []
        result_type = None
        for a in args:
            if isinstance(a, TensorSSA):
                raise TypeError("cannot mix TensorSSA and Numeric in math ops")
            elif isinstance(a, Numeric):
                if not type(a).is_float:
                    raise TypeError(
                        f"expected a float Numeric, got {type(a).__name__}"
                    )
                if result_type is None:
                    result_type = type(a)
                ir_args.append(a.ir_value(loc=loc, ip=ip))
            else:
                raise TypeError(f"expected TensorSSA or Numeric, got {type(a)}")
        res = func(*ir_args, fastmath=fm_flag, loc=loc, ip=ip)
        return result_type(res) if result_type is not None else res


def exp2(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Base-2 exponential: 2^a."""
    return _math_op(_math.exp2, fastmath, a, **kw)


def rsqrt(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Reciprocal square root: 1/sqrt(a)."""
    return _math_op(_math.rsqrt, fastmath, a, **kw)


def sqrt(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Square root."""
    return _math_op(_math.sqrt, fastmath, a, **kw)


def log2(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Base-2 logarithm."""
    return _math_op(_math.log2, fastmath, a, **kw)


def log(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Natural logarithm."""
    return _math_op(_math.log, fastmath, a, **kw)


def log10(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Base-10 logarithm."""
    return _math_op(_math.log10, fastmath, a, **kw)


def exp(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Natural exponential: e^a."""
    return _math_op(_math.exp, fastmath, a, **kw)


def floor(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Floor."""
    return _math_op(_math.floor, fastmath, a, **kw)


def ceil(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Ceiling."""
    return _math_op(_math.ceil, fastmath, a, **kw)


def sin(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Sine."""
    return _math_op(_math.sin, fastmath, a, **kw)


def cos(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Cosine."""
    return _math_op(_math.cos, fastmath, a, **kw)


def tan(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Tangent."""
    return _math_op(_math.tan, fastmath, a, **kw)


def asin(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Arc sine."""
    return _math_op(_math.asin, fastmath, a, **kw)


def acos(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Arc cosine."""
    return _math_op(_math.acos, fastmath, a, **kw)


def atan(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Arc tangent."""
    return _math_op(_math.atan, fastmath, a, **kw)


def atan2(
    a: Union[TensorSSA, Numeric],
    b: Union[TensorSSA, Numeric],
    fastmath: bool = False,
    **kw,
):
    """Arc tangent of two values."""
    return _math_op(_math.atan2, fastmath, a, b, **kw)


def tanh(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Hyperbolic tangent."""
    return _math_op(_math.tanh, fastmath, a, **kw)


def erf(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Error function."""
    return _math_op(_math.erf, fastmath, a, **kw)


def abs(a: Union[TensorSSA, Numeric], fastmath: bool = False, **kw):
    """Absolute value (float)."""
    return _math_op(_math.absf, fastmath, a, **kw)


__all__ = [
    "exp2", "rsqrt", "sqrt", "log2", "log", "log10",
    "exp", "floor", "ceil",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "tanh", "erf", "abs",
]
