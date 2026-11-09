# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``warp_merge_sort``: one item per lane ranks and scatters instead of merging.

Several items per lane keep the portable merge path.
"""

from .....expr.gpu import lane_id
from .....expr.numeric import Int32, Integer
from ..._common import _resolve_warp_width, _validate_valid_items
from ..._sorting import _pack, _unpack, _valid_before
from ..._values import _is_items, _record_select
from .. import merge_sort as _portable
from ..merge_sort import warp_merge_sort as _portable_warp_merge_sort
from ._common import _XOR_DPP, _dpp, _is_gfx9, _replacing
from .exchange import _ds_bpermute, _ds_permute, _map_words, _movable

__all__ = ["WarpMergeSort"]


def _valid_at(position, valid_items):
    """Whether tile position *position* is valid; see the portable ``merge_sort._valid_at``."""
    return Int32(1) if valid_items is None else Int32(position < valid_items)


# quad_perm controls for "the other half of the quad", element 0 / 1.
_QUAD_OTHER = (0x0A, 0x5F)  # [2,2,0,0], [3,3,1,1]


def _rank_merge(key, value, lane, width, compare_op, valid_items):
    """One item per lane: every pass ranks each key in the other run and pushes it there.

    A stable merge of runs A and B puts ``A[j]`` at ``j + #{b : b before A[j]}``
    and ``B[j]`` at ``j + #{a : not B[j] before a}``. Each lane binary-searches
    the other run for its own rank -- one gather per step, against the merge
    path's two -- and then key and payload travel to their slot with one
    ``ds_permute`` each, against the merge path's final two gathers plus the
    payload's. The ranks form a permutation for any strict weak ordering, which
    is what ``compare_op`` is required to be.

    The two smallest runs need no LDS at all: a pair of lanes is a
    compare-exchange through ``quad_perm``, and the other run of a quad is two
    ``quad_perm`` broadcasts.
    """
    physical = lane_id()
    start = physical - lane
    mine = _valid_at(lane, valid_items)

    def precedes(x, vx, in_b):
        """For A: x comes before me. For B: x does not come after me."""
        first, second = _record_select(in_b, key, x), _record_select(in_b, x, key)
        if valid_items is None:  # keep the comparator's validity branch a trace-time constant
            v1 = v2 = Int32(1)
        else:
            v1, v2 = in_b.select(mine, vx), in_b.select(vx, mine)
        return _valid_before(first, second, v1, v2, False, compare_op) != in_b

    def quad(item, ctrl):
        return _map_words(lambda w: _dpp(w, ctrl, bound_ctrl=True), item)

    run = 1
    while run < width:
        in_b = (lane & run) != 0
        base = lane & ~(2 * run - 1)
        if run == 1:
            # Rank 1 for A means the pair swaps; rank 1 for B means it does not.
            partner = quad(key, _XOR_DPP[1])
            ranked = precedes(partner, _valid_at(lane ^ 1, valid_items), in_b)
            swap = in_b.select(~ranked, ranked)
            key = _record_select(swap, partner, key)
            if value is not None:
                value = _record_select(swap, quad(value, _XOR_DPP[1]), value)
        else:
            other = base + in_b.select(Int32(0), Int32(run))
            fetched = [quad(key, ctrl) for ctrl in _QUAD_OTHER] if run == 2 else None
            low, high = Int32(0), Int32(run)
            for _ in range(run.bit_length()):
                mid = (low + high) // 2
                index = other + (mid < run).select(mid, Int32(run - 1))
                if fetched is not None:
                    x = _record_select(mid == 0, fetched[0], fetched[1])
                else:
                    address = (start + index) << 2
                    x = _map_words(lambda w: _ds_bpermute(w, address), key)
                counted = precedes(x, _valid_at(index, valid_items), in_b) & (mid < run)
                low = counted.select(mid + 1, low)
                high = counted.select(high, mid)
            address = (start + base + (lane & (run - 1)) + low) << 2
            key = _map_words(lambda w: _ds_permute(w, address), key)
            if value is not None:
                value = _map_words(lambda w: _ds_permute(w, address), value)
        run *= 2
    return key, value


def warp_merge_sort(
    keys,
    values=None,
    *,
    width: int | None = None,
    compare_op=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
    _value_dtype=None,
):
    if compare_op is not None and not callable(compare_op):
        raise TypeError("compare_op must be a strict ordering callable")
    if not _is_items(keys, _dtype):
        raise TypeError(
            "keys must be a nonempty outer item sequence; wrap one complete key in [key] or (key,). "
            "A Vector represents an item sequence only for a Numeric key dtype"
        )
    width = _resolve_warp_width(width, "warp_merge_sort width")
    items, payload, vector = _unpack(keys, values, _value_dtype, _dtype)
    count = len(items)
    _validate_valid_items(valid_items, width * count)
    lane = lane_id() % width
    if count == 1 and _is_gfx9() and _movable(items[0]) and (payload is None or _movable(payload[0])):
        key, value = _rank_merge(
            items[0], payload[0] if payload is not None else None, lane, width, compare_op, valid_items
        )
        return _pack([key], [value] if payload is not None else None, vector, keys, values)
    return _portable_warp_merge_sort(
        keys,
        values,
        width=width,
        compare_op=compare_op,
        valid_items=valid_items,
        _dtype=_dtype,
        _value_dtype=_value_dtype,
    )


warp_merge_sort.__doc__ = _portable_warp_merge_sort.__doc__


class WarpMergeSort(_portable.WarpMergeSort):
    __doc__ = _portable.WarpMergeSort.__doc__

    _implementations = _replacing(_portable, warp_merge_sort)
