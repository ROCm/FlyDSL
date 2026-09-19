# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shape and strict-order helpers shared by cooperative sorting algorithms."""

from ...expr.numeric import Int32
from ._values import _as_items, _from_items, _is_items, _item_dtype, _items_dtype, _record_default, _record_select


def _unpack(keys, values=None, value_dtype=None):
    vector = _is_items(keys)
    items = list(_as_items(keys))
    payload = list(_as_items(values)) if values is not None else None
    if payload is not None:
        if len(payload) != len(items) or _is_items(values) != vector:
            raise ValueError("values must have the same shape as keys")
        if value_dtype is not None and _items_dtype(values) is not value_dtype:
            raise TypeError(f"expected value dtype {value_dtype.__name__}, got {_items_dtype(values).__name__}")
    return items, payload, vector


def _pack(items, payload, vector, keys=None, values=None):
    keys = _from_items(items, like=keys) if vector else items[0]
    if payload is None:
        return keys
    return keys, _from_items(payload, like=values) if vector else payload[0]


def _before(a, b, descending=False, compare_op=None):
    if compare_op is None:
        return a > b if descending else a < b
    return compare_op(b, a) if descending else compare_op(a, b)


def _valid_before(a, b, va, vb, descending=False, compare_op=None):
    # Invalid entries follow every valid key even when no numeric sentinel exists.
    return ((va != 0) & (vb == 0)) | ((va == vb) & _before(a, b, descending, compare_op))


def _prepare(items, payload, indices, valid_items=None, oob_default=None):
    valid = [Int32(1) if valid_items is None else Int32(index < valid_items) for index in indices]
    if oob_default is not None:
        items = [_record_select(v, key, _record_default(_item_dtype(key), oob_default)) for key, v in zip(items, valid)]
        if payload is not None:
            payload = [
                _record_select(v, value, _record_default(_item_dtype(value))) for value, v in zip(payload, valid)
            ]
    return items, payload, valid


def _local_sort(items, payload, valid, descending=False, compare_op=None):
    items, valid = list(items), list(valid)
    payload = list(payload) if payload is not None else None
    for i in range(1, len(items)):
        for j in range(i, 0, -1):
            a, b = items[j - 1], items[j]
            va, vb = valid[j - 1], valid[j]
            swap = _valid_before(b, a, vb, va, descending, compare_op)
            items[j - 1], items[j] = _record_select(swap, b, a), _record_select(swap, a, b)
            valid[j - 1], valid[j] = swap.select(vb, va), swap.select(va, vb)
            if payload is not None:
                a, b = payload[j - 1], payload[j]
                payload[j - 1], payload[j] = _record_select(swap, b, a), _record_select(swap, a, b)
    return items, payload, valid
