# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-scope cooperative algorithms.

Every public class here specializes into the compile backend's override of
it, when the backend has one (``rocdl`` for ROCm). A ``warp_*`` function
belongs to the class defined next to it and runs the implementation that
class's override gives it, so a function and the member it backs never
diverge. ``fx.coop.universal`` keeps the portable classes and functions.
"""

from functools import wraps

from ..._dispatch import Dispatcher, dispatch_target
from . import bitonic_sort as _bitonic_sort
from . import merge_sort as _merge_sort
from . import reduce as _reduce
from . import reduce_batched as _reduce_batched
from . import scan as _scan
from .bitonic_sort import *
from .exchange import *
from .load import *
from .merge_sort import *
from .reduce import *
from .reduce_batched import *
from .scan import *
from .store import *

__all__ = [
    "WarpReduce",
    "WarpScan",
    "WarpReduceBatched",
    "WarpExchange",
    "WarpLoad",
    "WarpStore",
    "WarpBitonicSort",
    "WarpMergeSort",
    "WarpExchangeAlgorithm",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
    "warp_head_segmented_reduce",
    "warp_tail_segmented_reduce",
    "warp_head_segmented_sum",
    "warp_tail_segmented_sum",
    "warp_broadcast",
    "warp_reduce_batched_to_blocked",
    "warp_reduce_batched_to_striped",
    "warp_sum_batched",
    "warp_sum_batched_to_blocked",
    "warp_sum_batched_to_striped",
    "warp_bitonic_sort",
    "warp_merge_sort",
    "warp_reduce",
    "warp_sum",
    "warp_max",
    "warp_min",
    "warp_reduce_batched",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_inclusive_sum",
    "warp_exclusive_sum",
    "warp_scan",
    "warp_scan_with_aggregate",
]

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

WarpReduce = _dispatch.public_class(WarpReduce)
WarpScan = _dispatch.public_class(WarpScan)
WarpReduceBatched = _dispatch.public_class(WarpReduceBatched)
WarpExchange = _dispatch.public_class(WarpExchange)
WarpLoad = _dispatch.public_class(WarpLoad)
WarpStore = _dispatch.public_class(WarpStore)
WarpBitonicSort = _dispatch.public_class(WarpBitonicSort)
WarpMergeSort = _dispatch.public_class(WarpMergeSort)


def _through(owner, portable):
    """The public form of *portable*: it runs the implementation *owner*'s override gives it."""

    @wraps(portable)
    def function(*args, **kwargs):
        return dispatch_target(owner)._implementation(portable)(*args, **kwargs)

    return function


for _owner, _module in (
    (WarpReduce, _reduce),
    (WarpScan, _scan),
    (WarpReduceBatched, _reduce_batched),
    (WarpBitonicSort, _bitonic_sort),
    (WarpMergeSort, _merge_sort),
):
    for _name in _module.__all__:
        if _name.startswith("warp_"):
            globals()[_name] = _through(_owner, getattr(_module, _name))
del _owner, _module, _name
