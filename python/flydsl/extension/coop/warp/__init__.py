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
from .vote import *
from .vote import WarpVote as _WarpVote

__all__ = [
    "WarpReduce",
    "WarpScan",
    "WarpVote",
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
    "warp_argmax",
    "warp_argmin",
    # Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see reduce.py / rocdl.py.
    # "warp_reduce_e4m3x4",
    # "warp_reduce_e5m2x4",
    # "warp_argmax_e4m3x4",
    # "warp_argmin_e4m3x4",
    "warp_reduce_batched",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_inclusive_sum",
    "warp_exclusive_sum",
    "warp_scan",
    "warp_scan_with_aggregate",
    "warp_any",
    "warp_all",
    "warp_ballot",
    "warp_count",
    "warp_ballot_rank",
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


class WarpVote(_WarpVote):
    __doc__ = _WarpVote.__doc__

    _dispatcher = _dispatch
