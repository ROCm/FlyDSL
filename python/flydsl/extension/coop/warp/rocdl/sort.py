# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCm ``warp_sort``: the portable choice, except where the gfx9 rank path beats the network.

Where ``warp_sort`` may take either sort, the faster one depends on the items.
One 32-bit word per key and payload is the DPP network of ``bitonic_sort``.
Any other item gets the portable network there, and on gfx9 the rank path of
``merge_sort`` sorts those faster. Without that rank path -- on the other
targets, and for an item it cannot move, such as a pointer -- the choice is
the portable one.
"""

from .....expr.numeric import Integer
from ..._sorting import _unpack
from ..sort import _sort
from ..sort import warp_sort as _portable_warp_sort
from ._common import _is_gfx9
from .bitonic_sort import _applies, warp_bitonic_sort
from .exchange import _movable
from .merge_sort import warp_merge_sort

__all__ = ["warp_sort"]


def _network(keys, values=None, **options):
    items, payload, _ = _unpack(keys, values)
    # merge_sort ranks one item per lane only on gfx9, and only items it can move as 32-bit words.
    ranked = _is_gfx9() and _movable(items[0]) and (payload is None or _movable(payload[0]))
    chosen = warp_merge_sort if ranked and not _applies(items, payload) else warp_bitonic_sort
    return chosen(keys, values, **options)


def warp_sort(
    keys,
    values=None,
    *,
    width: int | None = None,
    compare_op=None,
    valid_items: int | Integer | None = None,
    stable: bool = False,
):
    return _sort(_network, warp_merge_sort, keys, values, compare_op, stable, width, valid_items)


warp_sort.__doc__ = _portable_warp_sort.__doc__
