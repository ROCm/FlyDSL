# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide prefix scan."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.primitive import const_expr, range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array, Vector
from .. import warp as _dispatched_warp
from .._common import combine, linear_thread_id, seed
from ._spec import BlockAlgorithmMeta

__all__ = [
    "BlockScanAlgorithm",
    "BlockScan",
]


class BlockScanAlgorithm(enum.Enum):
    """How a block turns per-thread values into a block-wide running fold.

    ``WARP_SCANS``
        Scan inside each warp with shuffles, then fold the per-warp aggregates that precede each
        warp back in. Only ``num_warps`` slots of shared memory and one barrier.
    ``RAKING``
        *Planned, not implemented.* Stage every thread's value in shared memory, then let a single
        warp walk ("rake") equal-length segments of it: an upsweep to get the segment totals, a scan
        of those, and a downsweep to push the prefixes back out. ``block_threads`` slots of shared
        memory and three barriers, but the cross-lane work collapses to one warp.
    ``RAKING_MEMOIZE``
        *Planned, not implemented.* Raking that keeps the upsweep's intermediate values in registers
        instead of re-reading them during the downsweep: fewer shared memory reads at the cost of
        ``segment_length`` live registers in the raking warp.
    """

    RAKING = "raking"  # planned, not implemented
    RAKING_MEMOIZE = "raking_memoize"  # planned, not implemented
    WARP_SCANS = "warp_scans"


# ── policy implementations ─────────────────────────────────────────────────
# Each one answers the same question — what precedes this thread — and is paired
# with the shared storage it needs in _SHARED_STORAGE below. Everything else the
# scan produces follows from that exclusive prefix by one `combine`, so the
# inclusive and exclusive forms do not need separate policies.


@jit
def _prefix_warp_scans(partial, tid, slots, op, warp_scan_with_aggregate, warp_threads, num_warps):
    """Scan within each warp, then fold in the aggregates of the warps in front.

    Returns the thread's exclusive prefix together with the block's own
    aggregate. The aggregate is free: with more than one warp every thread can
    already read every warp's total out of *slots*, and with a single warp the
    scan's top lane already holds it.
    """
    inclusive, prefix, aggregate = warp_scan_with_aggregate(partial, op, width=warp_threads)
    if const_expr(num_warps > 1):
        lane = tid % warp_threads
        warp_id = tid // warp_threads
        if lane == warp_threads - 1:
            slots[warp_id] = inclusive
        barrier()
        # Highest warp first, so the operands stay in block order: a thread in
        # warp w ends up with slots[0] ⊕ ... ⊕ slots[w-1] ⊕ its own warp prefix.

        # TODO: this and the aggregate below are linear in num_warps, and every
        # thread walks both — 2 * (num_warps - 1) folds each, so 30 at a
        # 1024-thread wave64 block. Scanning *slots* in one warp instead would
        # make it logarithmic: warp 0 scans the num_warps totals, and each
        # thread then reads the single entry in front of its own warp.
        for i in range_constexpr(num_warps - 2, -1, -1):
            prefix = (warp_id > i).select(combine(op, slots[i], prefix), prefix)
        # Same slots, folded unconditionally: that is the whole block.
        aggregate = slots[0]
        for i in range_constexpr(1, num_warps):
            aggregate = combine(op, aggregate, slots[i])
    return prefix, aggregate


def _storage_warp_scans(dtype, block_threads, warp_threads):
    return Struct["slots" : Array[dtype, block_threads // warp_threads]]


# Registry of the implemented policies. An unlisted member of the enum names a
# strategy this library has not implemented yet.
_SHARED_STORAGE = {
    BlockScanAlgorithm.WARP_SCANS: _storage_warp_scans,
}


class _BlockScanMeta(BlockAlgorithmMeta):
    """Gives ``BlockScan`` its ``[...]`` specialization syntax."""

    _algorithms = BlockScanAlgorithm
    _shared_storage = _SHARED_STORAGE

    def _default_algorithm_for(cls, target):
        """``WARP_SCANS`` on every target, being the only policy implemented.

        The other two members of the enum are declared but have no entry in
        ``_SHARED_STORAGE``, so there is nothing yet for a target to choose
        between. Whichever lands first gets its branch here.
        """
        return BlockScanAlgorithm.WARP_SCANS

    def __call__(cls, *args, **kwargs):
        raise TypeError("a scan has two forms; call .inclusive(...) or .exclusive(...)")


class BlockScan(metaclass=_BlockScanMeta):
    """Block-wide prefix scan.

    Specialize it, allocate its shared storage, then ask for the form you want::

        block_scan = fx.coop.BlockScan[fx.Float32, 256]
        storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()
        running = block_scan.inclusive(value, fx.ReductionOp.ADD, storage=storage)

    The parameters are ``[dtype, block_size, algorithm]``, exactly as for
    :class:`~flydsl.extension.coop.BlockReduce` — *block_size* is either the x extent or the full
    ``(x, y, z)`` — except that ``algorithm`` defaults to ``BlockScanAlgorithm.WARP_SCANS``.

    ``value`` is either one scalar per thread or a ``Vector`` of several per-thread items. A
    ``Vector`` scans as if its items sat consecutively in the block's sequence — thread ``t`` owns
    items ``t * n .. t * n + n - 1`` — and the result is a ``Vector`` of the same length.

    Passing ``init=`` folds a value in ahead of the whole block, so the first thread's exclusive
    result is ``init`` rather than :func:`~flydsl.extension.coop._common.identity`.

    :meth:`inclusive_with_aggregate` and :meth:`exclusive_with_aggregate` return ``(result,
    block_aggregate)`` instead. The aggregate is the fold of every thread's input, valid in all of
    them. It reuses what the scan already staged in shared memory, so it adds ``num_warps - 1``
    folds and no traffic at all — and nothing whatsoever when a caller ignores it. It is what lets a
    kernel carry a running total across tiles.

    Every thread of the block has to reach the call, and reach it together, exactly as for
    :class:`~flydsl.extension.coop.BlockReduce`: it synchronizes the block and reads across lanes,
    and the specialization's *block_size* has to be the size the kernel is actually launched with.

    Threads are ordered by their linear id, and *op* has to be associative but not commutative:
    every fold on the way, the warp-scope scan included, keeps its operands in that order. That is
    the opposite of :class:`~flydsl.extension.coop.BlockReduce`, whose implemented policies all
    need a commutative *op* today — see :class:`BlockReduceAlgorithm` for why.

    Reusing *storage* for a second collective call needs a :func:`~flydsl.expr.gpu.barrier` in
    between, since this one leaves the block unsynchronized after its last read.
    """

    dtype = None
    block_size = None
    block_threads = None
    algorithm = None
    warp_threads = None
    num_warps = None
    SharedStorage = None

    # Where the warp-scope scan underneath comes from; see
    # :class:`~flydsl.extension.coop.BlockReduce` for what overriding it buys.
    warp_ops = _dispatched_warp

    @classmethod
    def inclusive(cls, value, op, *, storage, init=None):
        """Fold everything up to and including this thread's own value."""
        return cls._scan(value, op, storage, inclusive=True, init=init)[0]

    @classmethod
    def exclusive(cls, value, op, *, storage, init=None):
        """Fold everything strictly in front of this thread's own value.

        The first thread has nothing in front of it and gets *init*, or
        :func:`~flydsl.extension.coop._common.identity` when none is given.
        """
        return cls._scan(value, op, storage, inclusive=False, init=init)[0]

    @classmethod
    def inclusive_with_aggregate(cls, value, op, *, storage, init=None):
        """:meth:`inclusive`, plus the block aggregate every thread can read.

        The aggregate covers the input alone: *init* seeds the scan but is not
        folded into it.
        """
        return cls._scan(value, op, storage, inclusive=True, init=init)

    @classmethod
    def exclusive_with_aggregate(cls, value, op, *, storage, init=None):
        """:meth:`exclusive`, plus the block aggregate every thread can read."""
        return cls._scan(value, op, storage, inclusive=False, init=init)

    @classmethod
    def _scan(cls, value, op, storage, *, inclusive, init=None):
        """The one implementation; returns ``(result, block_aggregate)``."""
        if cls.block_threads is None:
            raise TypeError("specialize first, e.g. BlockScan[fx.Float32, 256]")

        if not isinstance(value, Vector):
            prefix, aggregate = cls._block_prefix(value, op, storage)
            prefix = seed(prefix, op, init)
            return (combine(op, prefix, value) if inclusive else prefix), aggregate

        # Scan the thread's own items first: its aggregate is then a single
        # value the block-wide scan can treat exactly like a scalar input.
        items = list(value)
        running = items[0]
        scanned = [running]
        for item in items[1:]:
            running = combine(op, running, item)
            scanned.append(running)

        prefix, aggregate = cls._block_prefix(running, op, storage)
        prefix = seed(prefix, op, init)
        # The exclusive form is the inclusive one shifted a slot to the right,
        # with the block prefix moving into the hole at the front.
        heads = scanned if inclusive else scanned[:-1]
        out = [combine(op, prefix, head) for head in heads]
        return Vector.from_elements(out if inclusive else [prefix, *out]), aggregate

    @classmethod
    def _block_prefix(cls, partial, op, storage):
        """``(prefix, aggregate)``: what precedes this thread, and the block's total."""
        tid = linear_thread_id(cls.block_size)
        return _prefix_warp_scans(
            partial,
            tid,
            storage.slots,
            op,
            cls.warp_ops.warp_scan_with_aggregate,
            cls.warp_threads,
            cls.num_warps,
        )
