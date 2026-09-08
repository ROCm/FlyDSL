# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The portable implementations, under the public names, never displaced.

``fx.coop.warp_reduce`` resolves through ``extension/_dispatch.py``: on the ROCm
backend it is the DPP sequence in ``warp/rocdl.py``, not the shuffle butterfly in
``warp/reduce.py``. ``fx.coop.universal.warp_reduce`` is that butterfly, and stays
it on every target.

The names, the signatures and the results are the same either way — a target
override is only ever a faster route to the same answer — so ``universal`` is not
a second API to learn. It is the same one with dispatch turned off, which two
callers want:

- **A kernel that measured it.** An override is faster on the shapes it was
  tuned for, not on every shape; a caller that found the portable form better on
  its own can say so, rather than being stuck with whatever the target picked.
- **A test.** Running both and comparing them against each other is what catches
  an override that is wrong in a way a host reference would not show, and it is
  the only way to reach the portable code at all once an override exists for the
  target being tested.
"""

from types import SimpleNamespace

from .block import reduce as _block_reduce
from .block import scan as _block_scan
from .warp import reduce as _warp_reduce
from .warp import scan as _warp_scan

__all__ = [
    # warp scope
    "warp_reduce",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
    # block scope
    "BlockReduceAlgorithm",
    "BlockReduce",
    "BlockScanAlgorithm",
    "BlockScan",
]


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
