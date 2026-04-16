# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vector dialect helpers and re-exports.

The ``Vector`` class itself lives in ``typing.py`` alongside other builtin
DSL types.  This module re-exports it for convenience and provides thin
wrappers around upstream ``_mlir.dialects.vector`` ops.
"""

from __future__ import annotations

from .._mlir import ir
from .._mlir.dialects import vector as _vector
from .meta import traced_op

# Re-export upstream dialect for ``from flydsl.expr import vector; vector.broadcast(...)``
from .._mlir.dialects.vector import *  # noqa: F401,F403,E402

# Re-export Vector and friends so ``from flydsl.expr.vector import Vector`` works
from .typing import Vector, ReductionOp, full, full_like, zeros_like  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Dialect helper wrappers (legacy, will be deprecated)
# Prefer using Vector methods or _mlir.dialects.vector directly.
# ═══════════════════════════════════════════════════════════════════════

@traced_op
def from_elements(*args, loc=None, ip=None, **kwargs):
    """Construct a vector from scalar elements, auto-unwrapping ArithValue wrappers."""
    from . import arith as _arith_ext

    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [_arith_ext.unwrap(v) for v in elems]
        return _vector.from_elements(*args, loc=loc, ip=ip, **kwargs)

    return _vector.from_elements(*args, loc=loc, ip=ip, **kwargs)


@traced_op
def store(value, memref, indices, *, loc=None, ip=None, **kwargs):
    """Vector store wrapper that accepts ArithValue/wrappers for value/indices."""
    from . import arith as _arith_ext

    return _vector.store(
        _arith_ext.unwrap(value),
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=loc,
        ip=ip,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Thin wrappers for common op classes that otherwise require `.result` access.
# -----------------------------------------------------------------------------


@traced_op
def extract(vector, static_position=None, dynamic_position=None, *, loc=None, ip=None):
    """Wrapper around `vector.ExtractOp(...).result`.

    When only ``dynamic_position`` is supplied (without explicit
    ``static_position``), each dynamic index needs a corresponding
    ``kDynamic`` sentinel in the static attribute so the ODS builder
    pairs them correctly.  This wrapper fills in the sentinels
    automatically.
    """
    from . import arith as _arith_ext

    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_arith_ext.unwrap(i, index=True, loc=loc) for i in dynamic_position]

    n_static = len(static_position)
    n_dynamic = len(dynamic_position)
    if n_dynamic > 0 and n_static < n_dynamic:
        kDynamic = ir.ShapedType.get_dynamic_size()
        static_position = list(static_position) + [kDynamic] * (n_dynamic - n_static)

    return _vector.ExtractOp(
        _arith_ext.unwrap(vector, loc=loc),
        static_position=static_position,
        dynamic_position=dynamic_position,
        loc=loc,
        ip=ip,
    ).result


@traced_op
def load_op(result_type, memref, indices, *, loc=None, ip=None):
    """Wrapper around `vector.LoadOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.LoadOp(
        result_type,
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=loc,
        ip=ip,
    ).result


@traced_op
def bitcast(result_type, source, *, loc=None, ip=None):
    """Wrapper around `vector.BitCastOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.BitCastOp(
        result_type,
        _arith_ext.unwrap(source, loc=loc),
        loc=loc,
        ip=ip,
    ).result


# ═══════════════════════════════════════════════════════════════════════
# Scalar ↔ vector bitcast (requires llvm.BitcastOp)
# arith.bitcast and vector.BitCastOp do not support shape changes
# (e.g. i32 ↔ vector<2xbf16>); llvm.BitcastOp is required.
# ═══════════════════════════════════════════════════════════════════════

def bitcast_i32_to_v2bf16(val, *, loc=None):
    """Bitcast i32 scalar to vector<2xbf16> (bit-identical reinterpretation).

    Used to reinterpret a packed i32 load result as two bf16 elements.
    """
    from . import arith as _arith_ext
    from .._mlir.dialects import llvm as _llvm
    from .._mlir.extras import types as _T

    v2bf16 = _T.VectorType.get([2], _T.bf16())
    return _llvm.BitcastOp(v2bf16, _arith_ext.unwrap(val, loc=loc), loc=loc).res


def bitcast_v2bf16_to_i32(val, *, loc=None):
    """Bitcast vector<2xbf16> to i32 (bit-identical reinterpretation).

    Used to pack two bf16 accumulator results into an i32 for store.
    """
    from . import arith as _arith_ext
    from .._mlir.dialects import llvm as _llvm
    from .._mlir.ir import IntegerType

    i32 = IntegerType.get_signless(32)
    return _llvm.BitcastOp(i32, _arith_ext.unwrap(val, loc=loc), loc=loc).res
