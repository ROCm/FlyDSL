# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL overrides for the block-scope collectives.

Each module subclasses the portable class of the same name and overrides only
the parts its gfx9 sequence replaces -- a hook such as ``_scan_warp_totals`` or
``_sort_runs``, or a layout method, each marked ``override`` -- handing every
other call to the portable implementation. Specializing a public
``fx.coop.BlockX`` specializes the class here instead (see ``block/_spec.py``);
``fx.coop.universal`` keeps the portable classes. The parts an override
contains (``scan_type``, ``sort_type``) are inherited from the portable class,
which names them by their public, dispatched spelling.

Each module's docstring describes what its override replaces and when it
applies.
"""

from .adjacent_difference import BlockAdjacentDifference
from .discontinuity import BlockDiscontinuity
from .exchange import BlockExchange
from .merge_sort import BlockMergeSort
from .radix_rank import BlockRadixRank
from .radix_sort import BlockRadixSort
from .reduce import BlockReduce
from .scan import BlockScan
from .shuffle import BlockShuffle
from .topk import BlockTopK

__all__ = [
    "BlockAdjacentDifference",
    "BlockDiscontinuity",
    "BlockExchange",
    "BlockMergeSort",
    "BlockRadixRank",
    "BlockRadixSort",
    "BlockReduce",
    "BlockScan",
    "BlockShuffle",
    "BlockTopK",
]
