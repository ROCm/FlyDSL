# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Guarded scalar and packed I/O shared by warp and block tile loaders."""

from ...compiler import jit
from ...expr.primitive import const_expr, make_layout, make_view
from ...expr.typing import Numeric, Vector
from ._values import _as_items, _from_items, _item_dtype, _record_cast


@jit
def _load_item(source, index, offset, valid_items, default):
    value = default
    if const_expr(valid_items is None):
        value = _record_cast(source[offset + index], _item_dtype(default))
    else:
        if index < valid_items:
            value = _record_cast(source[offset + index], _item_dtype(default))
    return value


@jit
def _store_item(destination, value, index, offset, valid_items):
    if const_expr(valid_items is None):
        destination[offset + index] = _record_cast(value, destination.dtype)
    else:
        if index < valid_items:
            destination[offset + index] = _record_cast(value, destination.dtype)


def _pack_width(tensor, count):
    # A packed operation is legal only for a numeric contiguous 1-D view.
    if not issubclass(tensor.dtype, Numeric):
        return 1
    if not hasattr(tensor, "stride") or not hasattr(tensor, "iter"):
        return 1
    try:
        stride = tensor.stride.to_py_value()
    except (TypeError, ValueError):
        return 1
    if stride != 1:
        return 1
    width = 1
    while width * 2 <= count and count % (width * 2) == 0 and width * 2 * tensor.dtype.width <= 128:
        width *= 2
    return width


def _scalar_pack(source, index, offset, valid_items, default, width):
    return _from_items([_load_item(source, index + i, offset, valid_items, default) for i in range(width)])


@jit
def _load_pack(source, index, offset, valid_items, default, width):
    result = Vector.filled(width, default, default.dtype)
    if const_expr(valid_items is None):
        result = make_view(source.iter + offset + index, make_layout(width, 1)).load().to(default.dtype)
    else:
        if index + width <= valid_items:
            result = make_view(source.iter + offset + index, make_layout(width, 1)).load().to(default.dtype)
        else:
            result = _scalar_pack(source, index, offset, valid_items, default, width)
    return result


def _load_vectorized(source, index, count, offset, valid_items, default):
    width = _pack_width(source, count)
    if width == 1:
        return _scalar_pack(source, index, offset, valid_items, default, count)
    items = []
    for i in range(0, count, width):
        items.extend(_load_pack(source, index + i, offset, valid_items, default, width))
    return Vector.from_elements(items)


def _scalar_store_pack(destination, value, index, offset, valid_items):
    for i, item in enumerate(value):
        _store_item(destination, item, index + i, offset, valid_items)


@jit
def _store_pack(destination, value, index, offset, valid_items, width):
    if const_expr(valid_items is None):
        make_view(destination.iter + offset + index, make_layout(width, 1)).store(value.to(destination.dtype))
    else:
        if index + width <= valid_items:
            make_view(destination.iter + offset + index, make_layout(width, 1)).store(value.to(destination.dtype))
        else:
            _scalar_store_pack(destination, value, index, offset, valid_items)


def _store_vectorized(destination, value, index, offset, valid_items):
    items = _as_items(value)
    width = _pack_width(destination, len(items))
    for i in range(0, len(items), width):
        if width == 1:
            _store_item(destination, items[i], index + i, offset, valid_items)
        else:
            _store_pack(destination, Vector.from_elements(items[i : i + width]), index + i, offset, valid_items, width)
