# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Sorting a blocked logical-warp tile by what the caller needs, not by algorithm."""

from ....expr.numeric import Integer, Numeric
from .._common import _resolve_warp_width
from .._values import _as_items, _is_items
from .bitonic_sort import warp_bitonic_sort
from .merge_sort import warp_merge_sort

__all__ = [
    "warp_sort",
]


def _order_is_unique(key, values, compare_op):
    """Whether every correct sort returns the same keys.

    Keys alone in the default order come back as bare values, and equal
    integers are the same bits, so no order of equal keys shows. A float
    ``-0.0`` equals ``0.0`` with other bits, a comparator can call different
    bits equal, and a payload shows which equal key came first.
    """
    return values is None and compare_op is None and isinstance(key, Numeric) and issubclass(key.dtype, Integer)


def _takes_network(keys, values, compare_op, stable):
    """Whether the sorting network may serve this call.

    One item per lane is blocked and striped at once, so the network returns
    the blocked tile. Malformed keys go to the merge, which reports them.
    """
    if not _is_items(keys):
        return False
    items = _as_items(keys)
    return len(items) == 1 and (not stable or _order_is_unique(items[0], values, compare_op))


def _sort(network, merge, keys, values, compare_op, stable, width, valid_items):
    """``warp_sort`` over one target's two sorts; *network* serves the calls ``_takes_network`` allows."""
    if not isinstance(stable, bool):
        raise TypeError("stable must be a compile-time bool")
    width = _resolve_warp_width(width, "warp_sort width")
    chosen = network if _takes_network(keys, values, compare_op, stable) else merge
    return chosen(keys, values, width=width, compare_op=compare_op, valid_items=valid_items)


def warp_sort(
    keys,
    values=None,
    *,
    width: int | None = None,
    compare_op=None,
    valid_items: int | Integer | None = None,
    stable: bool = False,
):
    """Sort a blocked logical-warp tile, stably on request.

    All lanes of the logical warp participate. Any positive per-lane item count
    is supported. Invalid entries sort after valid keys in either direction and never reach compare_op.
    With ``stable=True`` the result is that of ``warp_merge_sort``. Otherwise
    comparator-equivalent keys may come back in any order.

    Which sort runs is not part of the contract and can differ between targets.
    One item per lane may run the network of ``warp_bitonic_sort`` when the order
    of equal keys is free or cannot show; several items per lane are merged.

    Args:
        keys: Nonempty outer tuple/list of complete keys in blocked order.
            For one key, pass ``[key]`` or ``(key,)``, including Vector or Struct
            keys. A Vector can represent a sequence of Numeric scalar keys.
            Supply compare_op when keys require a custom ordering.
        values: Optional outer sequence of payloads with the same item count
            as keys. Payload element types and component shapes are independent
            of key element types and component shapes.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        compare_op: Optional strict weak ordering predicate (a, b). None sorts ascending
            with a < b; required for record and Vector keys. Use a > b for descending order.
            When NaNs participate, supply a comparator defining a consistent NaN order.
        valid_items: Uniform static or runtime length of the valid blocked prefix in [0,
            width * items_per_thread]. None marks all keys valid.
        stable: Compile-time bool. True keeps comparator-equivalent keys in their original
            blocked order. False, the default, may reorder them. Integer keys alone in the
            default order sort to the same tile either way.

    Returns:
        Sorted keys, or (keys, values) when payloads are provided, in blocked ownership.
        Results retain the input item count and shape. Output slots beyond
        valid_items are unspecified.

    Examples:
        # One key per lane with the lane index as payload; equal keys may swap.
        sorted_keys, sorted_values = fx.coop.warp_sort([key], [lane], width=4)
        # Group      |      group 0      ||      group 1
        # Data       | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # -----------+----+----+----+----++----+----+----+----
        # keys in    | 3  | 1  | 2  | 1  || 0  | 2  | 0  | 1
        # values in  | 0  | 1  | 2  | 3  || 4  | 5  | 6  | 7
        # keys out   | 1  | 1  | 2  | 3  || 0  | 0  | 1  | 2
        # values out | 1  | 3  | 2  | 0  || 4  | 6  | 7  | 5

        # Group 0's two keys 1 carry payloads 1 and 3 in either order, as do group 1's
        # two keys 0 with 4 and 6. stable=True fixes the order shown.
        sorted_keys, sorted_values = fx.coop.warp_sort([key], [lane], width=4, stable=True)

        # Integer keys alone: the same tile with or without stable.
        sorted_keys = fx.coop.warp_sort([key], width=4)
    """
    return _sort(warp_bitonic_sort, warp_merge_sort, keys, values, compare_op, stable, width, valid_items)
