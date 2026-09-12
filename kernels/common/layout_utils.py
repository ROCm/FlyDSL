# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Layout helpers for GEMM kernels.

Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
idx2crd / crd2idx with plain arith ops for static layouts.
Falls back to fly dialect ops for dynamic layouts.

Optimisation: power-of-2 strides/shapes emit ``shrui`` / ``andi`` instead of
``divui`` / ``remui``, avoiding 10-15-cycle V_DIV sequences on CDNA GPUs.
"""

import builtins as _builtins
import math as _math
import re

import flydsl.expr as fx
from flydsl._mlir import ir


def _is_pow2(n):
    """Return True when *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _div_pow2(val, divisor):
    """Unsigned divide index *val* by a **compile-time** power-of-2 *divisor*.

    Emits ``arith.shrui`` (1 VALU cycle) instead of ``arith.divui``
    (10-15 VALU cycles on CDNA).
    """
    shift = _math.log2(divisor)
    assert shift == int(shift), f"{divisor} is not a power of 2"
    return fx.Index(val) >> fx.Index(int(shift))


def _mod_pow2(val, modulus):
    """Unsigned remainder of index *val* by a **compile-time** power-of-2 *modulus*.

    Emits ``arith.andi`` (1 VALU cycle) instead of ``arith.remui``.
    """
    return fx.Index(val) & fx.Index(modulus - 1)


def _parse_dim(tok):
    """Parse a single dimension token: '?' -> None, otherwise int."""
    tok = tok.strip()
    return None if tok == "?" else int(tok)


def _parse_layout(ly):
    """Parse '(s0,s1,...):(d0,d1,...)' -> (shapes, strides) as lists (None for '?')."""
    ly_str = str(ly.type) if hasattr(ly, "type") else str(ly)
    m = re.search(r"\(([^)]+)\):\(([^)]+)\)", ly_str)
    if not m:
        return None
    shapes = [_parse_dim(s) for s in m.group(1).split(",")]
    strides = [_parse_dim(s) for s in m.group(2).split(",")]
    return shapes, strides


def _has_dynamic_strides(strides):
    """Check if any stride is dynamic (None)."""
    return any(s is None for s in strides)


def idx2crd(idx, layout):
    """Decompose flat index into a list of coordinate values.

    For static layouts, computes coordinates with plain arith ops.
    Power-of-2 strides/shapes use shift/mask instead of div/rem.
    For dynamic layouts, falls back to fx.idx2crd + fx.get.
    """
    parsed = _parse_layout(layout)

    if hasattr(idx, "ir_value"):
        idx = idx.ir_value()

    if parsed is None or _has_dynamic_strides(parsed[1]):
        result = fx.idx2crd(fx.Int32(idx), layout)
        ndims = len(parsed[1]) if parsed else 1
        return [fx.get(result, i) for i in range(ndims)]

    if isinstance(idx, ir.Value) and not isinstance(idx.type, ir.IndexType):
        idx = fx.Index(idx).ir_value()
    shapes, strides = parsed
    ndims = len(strides)

    ordered = sorted(
        [(i, s, sz) for i, s, sz in _builtins.zip(range(ndims), strides, shapes) if s != 0],
        key=lambda x: x[1],
        reverse=True,
    )
    coords = [None] * ndims
    remaining = idx
    for i, stride_val, size_val in ordered:
        if stride_val == 1:
            c = remaining
        elif _is_pow2(stride_val):
            c = _div_pow2(remaining, stride_val)
        else:
            c = fx.Index(remaining) // fx.Index(stride_val)
        if size_val is not None:
            if _is_pow2(size_val):
                c = _mod_pow2(c, size_val)
            else:
                c = fx.Index(c) % fx.Index(size_val)
        coords[i] = c
    for i in range(ndims):
        if coords[i] is None:
            coords[i] = remaining
    return coords


def crd2idx(crd, layout):
    """Compute flat index from a coordinate tuple/list.

    For static layouts, computes with plain arith ops.
    For dynamic layouts, falls back to fx.crd2idx with fx.make_coord.
    """
    if not isinstance(crd, (list, tuple)):
        crd = [crd]
    parsed = _parse_layout(layout)

    if parsed is None or _has_dynamic_strides(parsed[1]):
        crd_i32 = [
            (
                fx.Int32(c).ir_value()
                if isinstance(c, int) or isinstance(fx.as_ir_value(c).type, ir.IndexType)
                else fx.as_ir_value(c)
            )
            for c in crd
        ]
        coord_val = fx.make_coord(*crd_i32)
        scalar = fx.get_scalar(fx.crd2idx(coord_val, layout)).ir_value()
        if not isinstance(scalar.type, ir.IndexType):
            scalar = fx.Index(scalar).ir_value()
        return scalar

    _, strides = parsed
    result = None
    for coord_v, stride_v in _builtins.zip(crd, strides):
        if stride_v == 0:
            continue
        term = coord_v if stride_v == 1 else coord_v * fx.Index(stride_v)
        result = term if result is None else result + term
    return result if result is not None else fx.Index(0).ir_value()


def get(int_tuple, mode):
    """Extract element at `mode` from a Python list/tuple."""
    return int_tuple[mode]
