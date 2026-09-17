# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-scope cooperative algorithms."""

from ..._dispatch import Dispatcher
from .bitonic_sort import *
from .exchange import *
from .load import *
from .merge_sort import *
from .reduce import *
from .reduce_batched import *
from .scan import *
from .store import *

__all__ = [
    "WarpExchangeAlgorithm",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
    "warp_exchange_storage",
    "warp_scatter_to_striped",
    "warp_head_segmented_reduce",
    "warp_tail_segmented_reduce",
    "warp_broadcast",
    "warp_reduce_batched_to_blocked",
    "warp_reduce_batched_to_striped",
    "warp_bitonic_sort",
    "warp_blocked_to_striped",
    "warp_striped_to_blocked",
    "warp_load",
    "warp_merge_sort",
    "warp_reduce",
    "warp_reduce_batched",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
    "warp_store",
]

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

_dispatch.dispatch_all(
    globals(),
    [name for name in __all__ if name.startswith("warp_") and name != "warp_exchange_storage"],
)
