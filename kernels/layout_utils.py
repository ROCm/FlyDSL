"""Pure-arith layout helpers for static-stride layouts.

Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
idx2crd / crd2idx with plain arith ops, avoiding fly dialect round-trips.
"""

import re
import builtins as _builtins

from flydsl.expr import arith
from flydsl.expr.typing import T


def _parse_layout(ly):
    """Parse '(s0,s1,...):(d0,d1,...)' → (shapes, strides) as int lists."""
    ly_str = str(ly.type) if hasattr(ly, 'type') else str(ly)
    m = re.search(r'\(([^)]+)\):\(([^)]+)\)', ly_str)
    if not m:
        return None
    shapes = [int(s) for s in m.group(1).split(',')]
    strides = [int(s) for s in m.group(2).split(',')]
    return shapes, strides


def idx2crd(idx, layout):
    """Decompose flat index into a list of coordinate values (index-typed)."""
    if hasattr(idx, 'type') and str(idx.type) != 'index':
        idx = arith.index_cast(T.index, idx)
    parsed = _parse_layout(layout)
    assert parsed is not None, f"idx2crd: cannot parse layout {layout}"
    shapes, strides = parsed
    ndims = len(strides)

    ordered = sorted(
        [(i, s, sz) for i, s, sz in _builtins.zip(range(ndims), strides, shapes) if s != 0],
        key=lambda x: x[1], reverse=True,
    )
    coords = [None] * ndims
    remaining = idx
    for i, stride_val, size_val in ordered:
        coords[i] = (remaining / arith.index(stride_val)) % arith.index(size_val)
    for i in range(ndims):
        if coords[i] is None:
            coords[i] = remaining
    return coords


def crd2idx(crd, layout):
    """Compute flat index from a coordinate tuple/list or make_coord result."""
    if not isinstance(crd, (list, tuple)):
        crd = [crd]
    parsed = _parse_layout(layout)
    assert parsed is not None, f"crd2idx: cannot parse layout {layout}"
    _, strides = parsed

    result = None
    for coord_v, stride_v in _builtins.zip(crd, strides):
        if stride_v == 0:
            continue
        term = coord_v if stride_v == 1 else coord_v * stride_v
        result = term if result is None else result + term
    return result if result is not None else arith.index(0)


def get(int_tuple, mode):
    """Extract element at `mode` from a Python list/tuple."""
    return int_tuple[mode]
