# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx


def run(body, *args):
    """Trace ``body(*args)`` through ``@flyc.jit`` (frontend-only); return the JitFunction.

    ``args`` map to ``body``'s parameters, so an annotated parameter (e.g.
    ``def body(n: fx.Int32)``) becomes a *run-time* operand inside the trace —
    the way to exercise dynamic control flow / promotion paths.
    """
    jit_fn = flyc.jit(body)
    jit_fn(*args)
    return jit_fn


def source_ir(body, *args):
    """Trace ``body(*args)`` and return its emitted (pre-lowering) MLIR text."""
    return run(body, *args)._last_compiled[1].source_ir


def dtype_of(value):
    """Result dtype for a scalar ``Numeric`` or a ``Vector``."""
    return value.dtype if isinstance(value, fx.Vector) else type(value)


def _sample(dtype):
    """A Python value acceptable as a dynamic jit argument of ``dtype``."""
    if dtype is fx.Boolean:
        return True
    if dtype.is_float:
        return 1.0
    return 1


def dynamic_binop(a_ty, b_ty, op):
    """Apply ``op`` to two *dynamic* operands (jit parameters) of the given types.

    Returns ``(result_dtype, source_ir_text)`` so callers can assert both the
    result type and the intermediate casts the promotion emits.
    """
    captured = {}

    @flyc.jit
    def build(a: a_ty, b: b_ty):
        captured["dtype"] = dtype_of(op(a, b))

    build(_sample(a_ty), _sample(b_ty))
    return captured["dtype"], build._last_compiled[1].source_ir


def dynamic_literal_binop(arg_ty, literal, op):
    """Apply ``op(arg, literal)`` with a *dynamic* ``arg`` of ``arg_ty``; return result dtype."""
    captured = {}

    @flyc.jit
    def build(a: arg_ty):
        captured["dtype"] = dtype_of(op(a, literal))

    build(_sample(arg_ty))
    return captured["dtype"]


def vec(dtype, n=8, fill=1):
    """Build a constant ``Vector`` of ``dtype`` (call inside a traced ``body``)."""
    return fx.Vector.filled(n, fill, dtype)
