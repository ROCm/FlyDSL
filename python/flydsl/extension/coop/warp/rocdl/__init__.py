# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL overrides for the warp-scope collectives.

Everything here is gfx9-only. ``row_bcast`` and ``wave_shr`` are CDNA DPP
controls that RDNA dropped in favour of ``row_share`` / ``permlanex16``, which
is a different sequence rather than a different constant -- and one with much
less to gain, since LLVM's own rewrite already reaches pure DPP there. A few
sequences also need ``row_newbcast``, which gfx90a introduced; older gfx9 keeps
the portable form for those.

Each module exports the override of the portable class of the same name, which
the public ``fx.coop.Warp*`` class specializes into on ROCm. An override either
replaces methods outright (marked ``override``) or, where the portable methods
call a ``warp_*`` function, maps that function to its own in
``_implementations`` -- the table the public ``fx.coop.warp_*`` function runs
too. Each covers the calls its sequence can take -- typically a
``ReductionOp`` or the default comparator over 32-bit scalars -- and hands
every other call to the portable implementation it displaces, which
``fx.coop.universal`` keeps reachable:

- ``reduce``: ``WarpReduce`` -- ``warp_reduce`` and the head/tail segmented reductions.
- ``scan``: ``WarpScan`` -- the scans and ``warp_broadcast``.
- ``reduce_batched``, ``bitonic_sort``, ``merge_sort``: ``WarpReduceBatched``,
  ``WarpBitonicSort``, ``WarpMergeSort`` and their functions.
- ``exchange``: ``WarpExchange`` with the SHUFFLE algorithm.
"""

from . import bitonic_sort as _bitonic_sort
from . import exchange as _exchange
from . import merge_sort as _merge_sort
from . import reduce as _reduce
from . import reduce_batched as _reduce_batched
from . import scan as _scan
from .bitonic_sort import *
from .exchange import *
from .merge_sort import *
from .reduce import *
from .reduce_batched import *
from .scan import *

__all__ = [
    *_reduce.__all__,
    *_scan.__all__,
    *_reduce_batched.__all__,
    *_bitonic_sort.__all__,
    *_merge_sort.__all__,
    *_exchange.__all__,
]
