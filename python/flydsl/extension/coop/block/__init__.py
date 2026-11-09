# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-scope cooperative algorithms."""

from ..._dispatch import Dispatcher
from .adjacent_difference import *
from .discontinuity import *
from .exchange import *
from .histogram import *
from .load import *
from .merge_sort import *
from .radix_rank import *
from .radix_sort import *
from .reduce import *
from .run_length_decode import *
from .scan import *
from .shuffle import *
from .store import *
from .topk import *

__all__ = [
    "BlockAdjacentDifference",
    "BlockDiscontinuity",
    "BlockExchangeAlgorithm",
    "BlockExchange",
    "BlockHistogramAlgorithm",
    "BlockHistogram",
    "BlockLoadAlgorithm",
    "BlockLoad",
    "BlockMergeSortAlgorithm",
    "BlockMergeSort",
    "BlockRadixRankAlgorithm",
    "BlockRadixRank",
    "BlockRadixSortAlgorithm",
    "BlockRadixSort",
    "BlockReduceAlgorithm",
    "BlockReduce",
    "BlockRunLengthDecodeAlgorithm",
    "BlockRunLengthDecode",
    "BlockScanAlgorithm",
    "BlockScan",
    "BlockShuffleAlgorithm",
    "BlockShuffle",
    "BlockStoreAlgorithm",
    "BlockStore",
    "BlockTopKAlgorithm",
    "BlockTopK",
]

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

# Specializing a public class specializes the backend's same-named override.
# A collective that contains another (``scan_type``, ``sort_type``,
# ``exchange_type``) names that part by its public spelling (``_spec._Public``),
# so a collective without an override of its own still reaches its parts'
# overrides. ``fx.coop.universal`` keeps the portable classes and pins their parts.
BlockReduce = _dispatch.public_class(BlockReduce)
BlockScan = _dispatch.public_class(BlockScan)
BlockExchange = _dispatch.public_class(BlockExchange)
BlockShuffle = _dispatch.public_class(BlockShuffle)
BlockAdjacentDifference = _dispatch.public_class(BlockAdjacentDifference)
BlockDiscontinuity = _dispatch.public_class(BlockDiscontinuity)
BlockMergeSort = _dispatch.public_class(BlockMergeSort)
BlockRadixRank = _dispatch.public_class(BlockRadixRank)
BlockRadixSort = _dispatch.public_class(BlockRadixSort)
BlockTopK = _dispatch.public_class(BlockTopK)
BlockRunLengthDecode = _dispatch.public_class(BlockRunLengthDecode)
BlockHistogram = _dispatch.public_class(BlockHistogram)
BlockLoad = _dispatch.public_class(BlockLoad)
BlockStore = _dispatch.public_class(BlockStore)
