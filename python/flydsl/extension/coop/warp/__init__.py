# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-scope cooperative algorithms."""

from ..._dispatch import Dispatcher
from .bitonic_sort import *
from .exchange import *
from .load import *
from .merge_sort import *
from .reduce import *
from .reduce import WarpReduce as _WarpReduce
from .reduce_batched import *
from .scan import *
from .scan import WarpScan as _WarpScan
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
    "warp_broadcast",
    "warp_reduce_batched_to_blocked",
    "warp_reduce_batched_to_striped",
    "warp_bitonic_sort",
    "warp_merge_sort",
    "warp_reduce",
    "warp_reduce_batched",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
]

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

_dispatch.dispatch_all(
    globals(),
    [name for name in __all__ if name.startswith("warp_")],
)


class WarpReduce(_WarpReduce):
    __doc__ = _WarpReduce.__doc__

    _dispatcher = _dispatch


class WarpScan(_WarpScan):
    __doc__ = _WarpScan.__doc__

    _dispatcher = _dispatch
