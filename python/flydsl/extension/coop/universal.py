# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expose portable collectives under the same names as the dispatched API.

``fx.coop.universal.warp_reduce`` always uses the portable warp implementation.
``fx.coop.warp_reduce`` may select a backend override when its requirements are
met. Both expose the same argument and result contracts; floating-point
rounding can depend on the reduction tree.

The block subclasses in this module also route their internal warp operations
and scans through portable implementations. This namespace supports explicit
implementation selection and comparison in correctness tests and benchmarks.
"""

from types import SimpleNamespace

from .block import reduce as _block_reduce
from .block import scan as _block_scan

# Portable implementations of the additional warp primitive families.
from .warp import __all__ as _warp_names
from .warp import reduce as _warp_reduce
from .warp import scan as _warp_scan
from .warp.bitonic_sort import *
from .warp.exchange import *
from .warp.load import *
from .warp.merge_sort import *
from .warp.reduce_batched import *
from .warp.store import *

warp_reduce = _warp_reduce.warp_reduce
warp_inclusive_scan = _warp_scan.warp_inclusive_scan
warp_exclusive_scan = _warp_scan.warp_exclusive_scan
warp_scan = _warp_scan.warp_scan
warp_scan_with_aggregate = _warp_scan.warp_scan_with_aggregate


# What the block classes below fold through, in place of the dispatched warp
# namespace. Only the two names block scope actually reaches for are here, so a
# warp primitive that gains a block-scope caller has to be added deliberately.
_UNIVERSAL_WARP = SimpleNamespace(
    warp_reduce=warp_reduce,
    warp_inclusive_scan=warp_inclusive_scan,
    warp_scan_with_aggregate=warp_scan_with_aggregate,
)


# The policy enums describe what an algorithm does, not how it is compiled, so
# they are the dispatched ones rather than copies: a caller must be able to pass
# ``fx.coop.BlockReduceAlgorithm.RAKING`` to either spelling of ``BlockReduce``.
BlockReduceAlgorithm = _block_reduce.BlockReduceAlgorithm
BlockScanAlgorithm = _block_scan.BlockScanAlgorithm


class BlockReduce(_block_reduce.BlockReduce):
    """:class:`~flydsl.extension.coop.BlockReduce`, folding through portable warps."""

    warp_ops = _UNIVERSAL_WARP


class BlockScan(_block_scan.BlockScan):
    """:class:`~flydsl.extension.coop.BlockScan`, folding through portable warps."""

    warp_ops = _UNIVERSAL_WARP


warp_head_segmented_reduce = _warp_reduce.warp_head_segmented_reduce
warp_tail_segmented_reduce = _warp_reduce.warp_tail_segmented_reduce
warp_broadcast = _warp_scan.warp_broadcast
__all__ = [
    *_warp_names,
    "BlockReduceAlgorithm",
    "BlockReduce",
    "BlockScanAlgorithm",
    "BlockScan",
]
