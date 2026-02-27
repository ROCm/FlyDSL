"""Pure-arith layout utilities for idx2crd, crd2idx, get.

These optimize the common case of static-stride layouts by decomposing
the layout operations into plain arith ops, avoiding fly dialect round-trips.
Falls back to fly.crd2idx/fly.idx2crd for dynamic layouts.
"""

import re
import builtins as _builtins

from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl.expr import arith
from flydsl.expr.meta import dsl_api_wrapper
from flydsl.expr.typing import T


def _unwrap(v):
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "_value"):
        return _unwrap(v._value)
    if hasattr(v, "value") and isinstance(getattr(v, "value", None), ir.Value):
        return v.value
    return v


def _to_i32(v):
    """Convert a value to i32 for fly dialect ops."""
    v = _unwrap(v)
    if isinstance(v, int):
        return arith.constant(v, type=T.i32)
    if isinstance(v, ir.Value) and str(v.type) == 'index':
        return arith.index_cast(T.i32, v)
    return v


def _to_coord(crd, loc=None, ip=None):
    """Convert a Python list/tuple of values to a fly coord, or unwrap a single value."""
    if isinstance(crd, (list, tuple)):
        converted = tuple(_unwrap(_to_i32(c)) for c in crd)
        IntTupleTy, dyncElems = fly.infer_int_tuple_type(converted)
        return fly.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)
    return _unwrap(crd)


def _scalar_to_index(result, loc=None, ip=None):
    """Extract scalar from int_tuple if needed, cast to index type."""
    if hasattr(result, 'type') and 'int_tuple' in str(result.type):
        result = fly.get_scalar(result, loc=loc, ip=ip)
    if hasattr(result, 'type') and str(result.type) != 'index':
        result = arith.index_cast(T.index, result)
    return result


def _parse_layout(ly):
    """Parse a fly.layout type string like '(4,64):(64,1)' into (shapes, strides).
    Returns (shapes, strides) as lists of int|None, or None if unparseable."""
    ly_str = str(ly.type) if hasattr(ly, 'type') else str(ly)
    match = re.search(r'\(([^)]+)\):\(([^)]+)\)', ly_str)
    if not match:
        return None
    raw_shapes = [s.strip() for s in match.group(1).split(',')]
    raw_strides = [s.strip() for s in match.group(2).split(',')]
    shapes = [None if s == '?' else int(s) for s in raw_shapes]
    strides = [None if s == '?' else int(s) for s in raw_strides]
    return shapes, strides


@dsl_api_wrapper
def idx2crd(idx, layout, loc=None, ip=None):
    """idx2crd: decompose flat index into coordinates. Returns list of index Values."""
    v = _unwrap(idx)
    if isinstance(v, ir.Value) and str(v.type) != 'index':
        v = arith.index_cast(T.index, v)
    ly = _unwrap(layout)
    parsed = _parse_layout(ly)
    if parsed is None:
        return [v]
    shapes, strides = parsed
    if all(s is None for s in strides):
        return [v]

    ndims = len(strides)
    indexed = list(_builtins.zip(range(ndims), strides, shapes))
    has_stride = [(i, s, sz) for i, s, sz in indexed if s is not None]
    has_stride.sort(key=lambda x: x[1], reverse=True)

    coords = [None] * ndims
    remaining = v
    for i, stride_val, size_val in has_stride:
        if stride_val == 0:
            coords[i] = arith.index(0)
            continue
        coord = remaining / arith.index(stride_val)
        if size_val is not None:
            coord = coord % arith.index(size_val)
        coords[i] = coord
    for i in range(ndims):
        if coords[i] is None:
            coords[i] = remaining
    return coords


@dsl_api_wrapper
def crd2idx(crd, layout, loc=None, ip=None):
    """crd2idx: compute flat index from coordinates. Returns index Value."""
    ly = _unwrap(layout)

    if not isinstance(crd, (list, tuple)):
        return _crd2idx_via_fly(crd, ly, loc=loc, ip=ip)

    parsed = _parse_layout(ly)
    if parsed is None or any(s is None for s in parsed[1]):
        return _crd2idx_via_fly(crd, ly, loc=loc, ip=ip)

    _, strides = parsed
    coords = [_unwrap(c) for c in crd]

    result = None
    for coord_v, stride_v in _builtins.zip(coords, strides):
        if stride_v == 0:
            continue
        term = coord_v if stride_v == 1 else coord_v * stride_v
        result = term if result is None else result + term
    return result if result is not None else arith.index(0)


def _crd2idx_via_fly(crd, ly, loc=None, ip=None):
    """Fallback: use fly.crd2idx for dynamic layouts."""
    crd_val = _to_coord(crd, loc=loc, ip=ip)
    result = fly.crd2idx(crd_val, ly, loc=loc, ip=ip)
    return _scalar_to_index(result, loc=loc, ip=ip)


@dsl_api_wrapper
def get(int_tuple, mode, loc=None, ip=None):
    """Extract element `mode` from a coordinate (int_tuple or Python list)."""
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    v = _unwrap(int_tuple)
    tp = str(v.type) if hasattr(v, 'type') else ''
    if 'int_tuple' in tp:
        selected = fly.select(v, indices=[mode], loc=loc, ip=ip)
        return _scalar_to_index(selected, loc=loc, ip=ip)
    return v
