"""
Layout coordinate utilities (idx2crd / crd2idx / get).
temp compat mode before we migrate to fly dialect.
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, fly


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
    v = _unwrap(v)
    if isinstance(v, ir.Value) and v.type == ir.IndexType.get():
        return arith.IndexCastOp(ir.IntegerType.get_signless(32), v).result
    return v


def _to_index(v):
    if hasattr(v, "type") and v.type != ir.IndexType.get():
        return arith.IndexCastOp(ir.IndexType.get(), v).result
    return v


def idx2crd(idx, layout, *, loc=None, ip=None):
    """idx2crd via fly dialect.  Returns a Python list of index-typed Values."""
    idx_val = _to_i32(_unwrap(idx))
    ly = _unwrap(layout)
    idx_tuple_ty, dync = fly.infer_int_tuple_type((idx_val,))
    idx_tuple = fly.make_int_tuple(idx_tuple_ty, (idx_val,), loc=loc, ip=ip)
    result = fly.idx2crd(idx_tuple, ly, loc=loc, ip=ip)
    ndims = fly.rank(ly)
    coords = []
    for i in range(ndims):
        selected = fly.select(result, indices=[i], loc=loc, ip=ip)
        scalar = fly.get_scalar(selected, loc=loc, ip=ip)
        coords.append(_to_index(scalar))
    return coords


def crd2idx(crd, layout, *, loc=None, ip=None):
    """crd2idx via fly dialect.  Returns an index-typed Value."""
    ly = _unwrap(layout)
    if isinstance(crd, (list, tuple)):
        converted = [_to_i32(_unwrap(c)) for c in crd]
        IntTupleTy, dyncElems = fly.infer_int_tuple_type(tuple(converted))
        crd_val = fly.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)
    else:
        crd_val = _unwrap(crd)
        if isinstance(crd_val, ir.Value) and crd_val.type == ir.IndexType.get():
            crd_val = _to_i32(crd_val)
            i32_ty, dync = fly.infer_int_tuple_type((crd_val,))
            crd_val = fly.make_int_tuple(i32_ty, (crd_val,), loc=loc, ip=ip)
    result = fly.crd2idx(crd_val, ly, loc=loc, ip=ip)
    if hasattr(result, "type") and "int_tuple" in str(result.type):
        result = fly.get_scalar(result, loc=loc, ip=ip)
    return _to_index(result)


def get(int_tuple, mode, *, loc=None, ip=None):
    """Extract element *mode* from a coordinate / int_tuple."""
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    v = _unwrap(int_tuple)
    selected = fly.select(v, indices=[mode], loc=loc, ip=ip)
    result = fly.get_scalar(selected, loc=loc, ip=ip)
    return _to_index(result)


__all__ = ["idx2crd", "crd2idx", "get"]
