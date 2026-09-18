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

from . import block as _block
from . import warp as _warp
from .block import reduce as _block_reduce
from .block import scan as _block_scan
from .warp import reduce as _warp_reduce
from .warp import scan as _warp_scan
from .warp.bitonic_sort import *
from .warp.exchange import *
from .warp.load import *
from .warp.merge_sort import *
from .warp.reduce_batched import *
from .warp.store import *

WarpReduce = _warp_reduce.WarpReduce
WarpScan = _warp_scan.WarpScan

warp_head_segmented_reduce = _warp_reduce.warp_head_segmented_reduce
warp_tail_segmented_reduce = _warp_reduce.warp_tail_segmented_reduce
warp_broadcast = _warp_scan.warp_broadcast
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
    """Reduce a block using portable warp operations for every policy.

    Specialization, arguments, result ownership and shared-storage requirements
    are those of :class:`~flydsl.extension.coop.block.reduce.BlockReduce`.

    Examples:
        # x in thread t is [2*t+1, 2*t+2], so the block owns integers 1 through 128.
        P = fx.coop.universal.BlockReduce[fx.Int32, 64, fx.coop.universal.BlockReduceAlgorithm.WARP_REDUCTIONS]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y = P(x, fx.ReductionOp.ADD, storage=storage)
        # Every thread receives y=8256.
        fx.barrier()
        partial = P(x, fx.ReductionOp.ADD, storage=storage, valid_items=5, identity=fx.Int32(0))
        # Every thread receives partial=15. valid_items counts elements, not threads.
    """

    warp_ops = _UNIVERSAL_WARP


class BlockScan(_block_scan.BlockScan):
    """Scan a block using portable warp operations for every policy.

    Specialization, arguments, callbacks and shared-storage requirements are
    those of :class:`~flydsl.extension.coop.block.scan.BlockScan`.

    Examples:
        # Compute prefix sums of 256 ones, with four consecutive items per thread. The scan continues
        # across thread boundaries.

        P = fx.coop.universal.BlockScan[fx.Int32, 64, fx.coop.universal.BlockScanAlgorithm.WARP_SCANS]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        exclusive = P.exclusive(x, fx.ReductionOp.ADD, init=0, storage=storage)
        fx.barrier()
        inclusive = P.inclusive(x, fx.ReductionOp.ADD, storage=storage)

        # Tables below show T0..T3; each of T4..T63 also supplies four ones.
        # Data           | T0        | T1        | T2           | T3
        # ---------------+-----------+-----------+--------------+--------------
        # x in (blocked) | [1,1,1,1] | [1,1,1,1] | [1,1,1,1]    | [1,1,1,1]
        # exclusive      | [0,1,2,3] | [4,5,6,7] | [8,9,10,11]  | [12,13,14,15]
        # inclusive      | [1,2,3,4] | [5,6,7,8] | [9,10,11,12] | [13,14,15,16]

        # The corresponding exclusive_with_aggregate and inclusive_with_aggregate methods also return 256
        # to every thread. With 128 threads and four ones per thread, the last thread receives
        # [508,509,510,511] from the exclusive scan.

        # Keep the first six ones and seed with 10. Each method keeps its normal return shape.
        fx.barrier()
        inclusive = P.inclusive(
            x, fx.ReductionOp.ADD, storage=storage, init=10, valid_items=6, identity=fx.Int32(0)
        )
        fx.barrier()
        exclusive = P.exclusive(
            x, fx.ReductionOp.ADD, storage=storage, init=10, valid_items=6, identity=fx.Int32(0)
        )
        fx.barrier()
        inclusive, aggregate = P.inclusive_with_aggregate(
            x, fx.ReductionOp.ADD, storage=storage, init=10, valid_items=6, identity=fx.Int32(0)
        )
        fx.barrier()
        exclusive, aggregate = P.exclusive_with_aggregate(
            x, fx.ReductionOp.ADD, storage=storage, init=10, valid_items=6, identity=fx.Int32(0)
        )
        # Data      | T0            | T1            | T2            | T3
        # ----------+---------------+---------------+---------------+--------------
        # inclusive | [11,12,13,14] | [15,16,16,16] | [16,16,16,16] | [16,16,16,16]
        # exclusive | [10,11,12,13] | [14,15,16,16] | [16,16,16,16] | [16,16,16,16]
        # aggregate | 6             | 6             | 6             | 6

        # T4..T63 receive [16,16,16,16] in both guarded scans.

        # A callback can supply the prefix from an earlier tile; omit init in these calls.
        def previous_prefix(aggregate):
            return fx.Int32(10)

        fx.barrier()
        inclusive = P.inclusive(
            x, fx.ReductionOp.ADD, storage=storage, valid_items=6,
            identity=fx.Int32(0), prefix_callback=previous_prefix,
        )
        fx.barrier()
        exclusive = P.exclusive(
            x, fx.ReductionOp.ADD, storage=storage, valid_items=6,
            identity=fx.Int32(0), prefix_callback=previous_prefix,
        )
        fx.barrier()
        inclusive, aggregate = P.inclusive_with_aggregate(
            x, fx.ReductionOp.ADD, storage=storage, valid_items=6,
            identity=fx.Int32(0), prefix_callback=previous_prefix,
        )
        fx.barrier()
        exclusive, aggregate = P.exclusive_with_aggregate(
            x, fx.ReductionOp.ADD, storage=storage, valid_items=6,
            identity=fx.Int32(0), prefix_callback=previous_prefix,
        )
        # The callback receives 6; all four methods produce the same respective results as above.
    """

    warp_ops = _UNIVERSAL_WARP


__all__ = [
    *_warp.__all__,
    *_block.__all__,
]
