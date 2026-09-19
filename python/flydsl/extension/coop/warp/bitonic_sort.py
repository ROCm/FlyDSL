# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Bitonic sorting networks over striped logical-warp tiles."""

from ....expr.gpu import lane_id, shuffle_xor
from ....expr.numeric import Int32, Integer
from .._common import _resolve_warp_width, _shuffle_value, _validate_valid_items
from .._sorting import _pack, _prepare, _unpack, _valid_before
from .._values import (
    _is_items,
    _record_select,
)
from ._spec import WarpPrimitive

__all__ = [
    "WarpBitonicSort",
    "warp_bitonic_sort",
]


def _shuffle(value, distance, width):
    return _shuffle_value(value, distance, width, mode="xor")


def _network(keys, values, width, compare_op, valid_items):
    descending = False
    width = _resolve_warp_width(width, "warp_bitonic_sort width")
    items, payload, vector = _unpack(keys, values)
    count = len(items)
    _validate_valid_items(valid_items, width * count)
    lane = lane_id() % width
    items, payload, valid = _prepare(
        items,
        payload,
        [lane + i * width for i in range(count)],
        valid_items,
    )
    padded = 1 << (count - 1).bit_length()
    # Invalid padding reuses a well-typed value; no default constructor is needed.
    items += [items[-1]] * (padded - count)
    valid += [Int32(0)] * (padded - count)
    if payload is not None:
        payload += [payload[-1]] * (padded - count)
    total = width * padded
    span = 2
    while span <= total:
        distance = span // 2
        while distance:
            out, carried, out_valid = [], [], []
            for i, key in enumerate(items):
                j = i ^ (distance // width)
                partner = _shuffle(key, distance, width) if distance < width else items[j]
                other_valid = shuffle_xor(valid[i], distance, width) if distance < width else valid[j]
                index = lane + i * width
                smaller = ((index & span) == 0) == ((index & distance) == 0)
                # Both ends must agree which original element wins a tie.
                # Using strict comparisons in both directions preserves pairs.
                take = smaller.select(
                    _valid_before(partner, key, other_valid, valid[i], descending, compare_op),
                    _valid_before(key, partner, valid[i], other_valid, descending, compare_op),
                )
                out.append(_record_select(take, partner, key))
                out_valid.append(take.select(other_valid, valid[i]))
                if payload is not None:
                    other = _shuffle(payload[i], distance, width) if distance < width else payload[j]
                    carried.append(_record_select(take, other, payload[i]))
            items, valid = out, out_valid
            if payload is not None:
                payload = carried
            distance //= 2
        span *= 2
    return _pack(
        items[:count],
        payload[:count] if payload is not None else None,
        vector,
        keys,
        values,
    )


def warp_bitonic_sort(
    keys,
    values=None,
    *,
    width: int | None = None,
    compare_op,
    valid_items: int | Integer | None = None,
):
    """Sort a striped logical-warp tile with a bitonic network.

    All lanes of the logical warp participate. Any positive per-lane item count
    is supported. Invalid entries sort after valid keys in either direction.
    Equal keys may be reordered; this operation does not guarantee stability.

    Args:
        keys: This lane's fixed-size Vector or tuple/list of keys in striped order, with a nonempty item count. Supply compare_op when keys
            require a custom ordering.
        values: Optional payloads with the same item count and shape as keys.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        compare_op: Required strict weak ordering predicate (a, b). Use a < b for
            ascending order or a > b for descending order.
            When NaNs participate, supply a comparator defining a consistent NaN order.
        valid_items: Uniform static or runtime length of the valid striped prefix in [0,
            width * items_per_thread]. None marks all keys valid.

    Returns:
        Sorted keys, or (keys, values) when payloads are provided, in striped ownership.
        Results retain the input item count and shape. Output slots beyond
        valid_items are unspecified.

    Examples:
        # Sort each group in striped order. Group 1 includes negative and duplicate keys.
        y = fx.coop.warp_bitonic_sort(keys, width=4, compare_op=lambda a, b: a < b)
        # Group             |            group 0            ||             group 1
        # Data              | L0    | L1    | L2    | L3    || L4     | L5     | L6     | L7
        # ------------------+-------+-------+-------+-------++--------+--------+--------+------
        # keys in (striped) | [8,4] | [7,3] | [6,2] | [5,1] || [2,0]  | [-3,5] | [2,-1] | [4,2]
        # y out (striped)   | [1,5] | [2,6] | [3,7] | [4,8] || [-3,2] | [-1,2] | [0,4]  | [2,5]

        # Read slot 0 across a group, then slot 1: group 0 becomes [1,2,3,4,5,6,7,8],
        # group 1 becomes [-3,-1,0,2,2,2,4,5]. Equal-key order is not guaranteed to be stable.

        # A separate partial tile: two keys per lane, with five valid ITEMS per group.
        # valid_items follows striped order; it does not count lanes.
        # Group       |             group 0              ||              group 1
        # Data        | L0    | L1     | L2     | L3     || L4     | L5     | L6     | L7
        # ------------+-------+--------+--------+--------++--------+--------+--------+-------
        # keys in     | [3,2] | [1,99] | [4,98] | [0,97] || [-2,3] | [5,99] | [1,98] | [0,97]
        # valid slots | [1,1] | [1,0]  | [1,0]  | [1,0]  || [1,1]  | [1,0]  | [1,0]  | [1,0]

        # Each input payload is 10 * its key, including the invalid placeholders.
        sorted_keys, sorted_values = fx.coop.warp_bitonic_sort(
            keys, values, width=4, compare_op=lambda a, b: a > b,
            valid_items=5,
        )
        # Group      |              group 0              ||               group 1
        # Data       | L0     | L1     | L2     | L3     || L4       | L5     | L6     | L7
        # -----------+--------+--------+--------+--------++----------+--------+--------+-------
        # keys out   | [4,0]  | [3,?]  | [2,?]  | [1,?]  || [5,-2]   | [3,?]  | [1,?]  | [0,?]
        # values out | [40,0] | [30,?] | [20,?] | [10,?] || [50,-20] | [30,?] | [10,?] | [0,?]

        # Only the first five logical output positions are specified.
        # ? marks unspecified keys and payloads beyond that valid prefix.
    """
    if not callable(compare_op):
        raise TypeError("compare_op must be a strict ordering callable")
    if not _is_items(keys):
        raise TypeError("keys must be a nonempty Vector/tuple/list")
    return _network(keys, values, width, compare_op, valid_items)


class WarpBitonicSort(WarpPrimitive):
    """Sort striped logical-warp keys and optional payloads.

    Specialize with ``[dtype, width, items_per_thread]``. ``None`` selects the target's
    physical warp width. Every lane in each logical warp must participate.
    SharedStorage is Empty: explicit storage has a zero-byte layout.
    There is no algorithm parameter.
    The corresponding ``warp_*`` functions infer dtype and tile extent.
    Use ``(key_dtype, value_dtype)`` as dtype when carrying payloads.
    Key-only specializations reject payloads; pair specializations require them.

    Examples:
        P = fx.coop.WarpBitonicSort[fx.Int32, 8, 2]
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
        """Sort a striped logical-warp tile with a bitonic network.

        All lanes of the logical warp participate. Any positive per-lane item count
        is supported. Invalid entries sort after valid keys in either direction.
        Equal keys may be reordered; this operation does not guarantee stability.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            keys: This lane's fixed-size Vector or tuple/list of keys in striped order, with a nonempty item count. Supply compare_op when keys
                require a custom ordering.
            values: Optional payloads with the same item count and shape as keys.
            compare_op: Required strict weak ordering predicate (a, b). Use a < b for
                ascending order or a > b for descending order.
                When NaNs participate, supply a comparator defining a consistent NaN order.
            valid_items: Uniform static or runtime length of the valid striped prefix in [0,
                width * items_per_thread]. None marks all keys valid.

        Returns:
            Sorted keys, or (keys, values) when payloads are provided, in striped ownership.
            Results retain the input item count and shape. Output slots beyond
            valid_items are unspecified.
        """
        keys = cls._prepare(keys)
        if (values is None) != (cls.value_dtype is None):
            raise TypeError("payload presence must match the key/value specialization")
        if values is not None:
            values = cls._prepare(values, dtype=cls.value_dtype)
        return cls._invoke(
            warp_bitonic_sort, keys, values, compare_op=compare_op, valid_items=valid_items, storage=storage
        )
