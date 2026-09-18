# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Stable merge-path sorting of arbitrary per-lane blocked register tiles."""

from ....expr.gpu import lane_id
from ....expr.numeric import Int32, Integer
from ....expr.typing import Vector
from .._common import _resolve_warp_width, _validate_valid_items
from .._sorting import _local_sort, _pack, _prepare, _unpack, _valid_before
from .._values import _from_items, _is_items, _record_select
from ._spec import WarpPrimitive
from .exchange import _warp_gather

__all__ = [
    "WarpMergeSort",
    "warp_merge_sort",
]


def _gather(items, index, width):
    return _warp_gather(items, index, width=width)


def warp_merge_sort(
    keys,
    values=None,
    *,
    width: int | None = None,
    compare_op,
    valid_items: int | Integer | None = None,
):
    """Stably sort a blocked logical-warp tile, optionally carrying payloads.

    All lanes of the logical warp participate. Any positive per-lane item count
    is supported. Invalid entries sort after valid keys in either direction.
    Comparator-equivalent keys retain their original blocked order.

    Args:
        keys: This lane's fixed-size Vector or tuple/list of keys in blocked order, with a nonempty item count. Supply compare_op when keys
            require a custom ordering.
        values: Optional payloads with the same item count and shape as keys.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        compare_op: Required strict weak ordering predicate (a, b). Use a < b for
            ascending order or a > b for descending order.
            When NaNs participate, supply a comparator defining a consistent NaN order.
        valid_items: Uniform static or runtime length of the valid blocked prefix in [0,
            width * items_per_thread]. None marks all keys valid.

    Returns:
        Sorted keys, or (keys, values) when payloads are provided, in blocked ownership.
        Results retain the input item count and shape. Output slots beyond
        valid_items are unspecified.

    Examples:
        # Stably sort each group in blocked order. Payloads identify the original input positions.
        sorted_keys, sorted_values = fx.coop.warp_merge_sort(keys, values, width=4, compare_op=lambda a, b: a < b)
        # Group             |            group 0            ||                group 1
        # Data              | L0    | L1    | L2    | L3    || L4      | L5      | L6      | L7
        # ------------------+-------+-------+-------+-------++---------+---------+---------+--------
        # keys in (blocked) | [3,1] | [2,1] | [4,2] | [3,0] || [2,1]   | [2,0]   | [1,2]   | [0,1]
        # values in         | [0,1] | [2,3] | [4,5] | [6,7] || [8,9]   | [10,11] | [12,13] | [14,15]
        # keys out          | [0,1] | [1,2] | [2,3] | [3,4] || [0,0]   | [1,1]   | [1,2]   | [2,2]
        # values out        | [7,1] | [3,2] | [5,0] | [6,4] || [11,14] | [9,12]  | [15,8]  | [10,13]

        # Equal keys retain their input order: group 0's key 1 carries payloads [1,3];
        # group 1's key 1 carries [9,12,15]. Identical keys in different groups never mix.

        # A separate partial tile: two keys per lane, with five valid ITEMS per group.
        # valid_items follows blocked order; it does not count lanes.
        # Group       |             group 0              ||              group 1
        # Data        | L0    | L1    | L2     | L3      || L4     | L5    | L6     | L7
        # ------------+-------+-------+--------+---------++--------+-------+--------+--------
        # keys in     | [3,1] | [4,0] | [2,99] | [98,97] || [-2,5] | [1,0] | [3,99] | [98,97]
        # valid slots | [1,1] | [1,1] | [1,0]  | [0,0]   || [1,1]  | [1,1] | [1,0]  | [0,0]

        # Each input payload is 10 * its key, including the invalid placeholders.
        sorted_keys, sorted_values = fx.coop.warp_merge_sort(
            keys, values, width=4, compare_op=lambda a, b: a > b,
            valid_items=5,
        )
        # Group      |               group 0                ||               group 1
        # Data       | L0      | L1      | L2     | L3      || L4      | L5     | L6      | L7
        # -----------+---------+---------+--------+---------++---------+--------+---------+--------
        # keys out   | [4,3]   | [2,1]   | [0,?]  | [?,?]   || [5,3]   | [1,0]  | [-2,?]  | [?,?]
        # values out | [40,30] | [20,10] | [0,?]  | [?,?]   || [50,30] | [10,0] | [-20,?] | [?,?]

        # Only the first five logical output positions are specified.
        # ? marks unspecified keys and payloads beyond that valid prefix.
    """
    if not callable(compare_op):
        raise TypeError("compare_op must be a strict ordering callable")
    if not _is_items(keys):
        raise TypeError("keys must be a nonempty Vector/tuple/list")
    descending = False
    width = _resolve_warp_width(width, "warp_merge_sort width")
    items, payload, vector = _unpack(keys, values)
    count = len(items)
    _validate_valid_items(valid_items, width * count)
    lane = lane_id() % width
    items, payload, valid = _prepare(items, payload, [lane * count + i for i in range(count)], valid_items, None)
    items, payload, valid = _local_sort(items, payload, valid, descending, compare_op)
    run = count
    while run < width * count:
        current = _from_items(items)
        validity = Vector.from_elements(valid)
        carried = _from_items(payload) if payload is not None else None
        out, out_values, out_valid = [], [], []
        for i in range(count):
            index = lane * count + i
            base = (index // (2 * run)) * (2 * run)
            diagonal = index % (2 * run)
            low = (diagonal > run).select(diagonal - run, Int32(0))
            high = (diagonal < run).select(diagonal, Int32(run))
            for _ in range(run.bit_length()):
                mid = (low + high) // 2
                other = diagonal - mid
                ai = base + (mid < run).select(mid, Int32(run - 1))
                bi = base + run + (other > 0).select(other - 1, Int32(0))
                a, b = _gather(current, ai, width), _gather(current, bi, width)
                va, vb = _warp_gather(validity, ai, width=width), _warp_gather(validity, bi, width=width)
                advance = (mid < run) & (other > 0) & ~_valid_before(b, a, vb, va, descending, compare_op)
                low = advance.select(mid + 1, low)
                high = advance.select(high, mid)
            other = diagonal - low
            ai = base + (low < run).select(low, Int32(run - 1))
            bi = base + run + (other < run).select(other, Int32(run - 1))
            a, b = _gather(current, ai, width), _gather(current, bi, width)
            va, vb = _warp_gather(validity, ai, width=width), _warp_gather(validity, bi, width=width)
            take_a = (low < run) & ((other >= run) | ~_valid_before(b, a, vb, va, descending, compare_op))
            out.append(_record_select(take_a, a, b))
            out_valid.append(take_a.select(va, vb))
            if carried is not None:
                out_values.append(_gather(carried, take_a.select(ai, bi), width))
        items, valid = out, out_valid
        if payload is not None:
            payload = out_values
        run *= 2
    return _pack(items, payload, vector, keys, values)


class WarpMergeSort(WarpPrimitive):
    """Stably sort blocked logical-warp keys and optional payloads.

    Specialize with ``[dtype, width, items_per_thread]``. ``None`` selects the target's
    physical warp width. Every lane in each logical warp must participate.
    SharedStorage is Empty: explicit storage has a zero-byte layout.
    There is no algorithm parameter.
    The corresponding ``warp_*`` functions infer dtype and tile extent.
    Use ``(key_dtype, value_dtype)`` as dtype when carrying payloads.
    Key-only specializations reject payloads; pair specializations require them.

    Examples:
        P = fx.coop.WarpMergeSort[fx.Int32, 8, 2]
        result = P.sort(x, compare_op=lambda a, b: a < b)
    """

    _tile = True
    _pairs = True

    @classmethod
    def sort(
        cls,
        keys,
        values=None,
        *,
        compare_op,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Stably sort a blocked logical-warp tile, optionally carrying payloads.

        All lanes of the logical warp participate. Any positive per-lane item count
        is supported. Invalid entries sort after valid keys in either direction.
        Comparator-equivalent keys retain their original blocked order.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            keys: This lane's fixed-size Vector or tuple/list of keys in blocked order, with a nonempty item count. Supply compare_op when keys
                require a custom ordering.
            values: Optional payloads with the same item count and shape as keys.
            compare_op: Required strict weak ordering predicate (a, b). Use a < b for
                ascending order or a > b for descending order.
                When NaNs participate, supply a comparator defining a consistent NaN order.
            valid_items: Uniform static or runtime length of the valid blocked prefix in [0,
                width * items_per_thread]. None marks all keys valid.

        Returns:
            Sorted keys, or (keys, values) when payloads are provided, in blocked ownership.
            Results retain the input item count and shape. Output slots beyond
            valid_items are unspecified.
        """
        keys = cls._prepare(keys)
        if (values is None) != (cls.value_dtype is None):
            raise TypeError("payload presence must match the key/value specialization")
        if values is not None:
            values = cls._prepare(values, dtype=cls.value_dtype)
        return cls._invoke(
            warp_merge_sort, keys, values, compare_op=compare_op, valid_items=valid_items, storage=storage
        )
