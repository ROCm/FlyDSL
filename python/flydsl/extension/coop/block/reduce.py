# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide reduction."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.primitive import const_expr, range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array
from .. import warp as _dispatched_warp
from .._common import combine, linear_thread_id, thread_partial
from ._spec import BlockAlgorithmMeta

__all__ = [
    "BlockReduceAlgorithm",
    "BlockReduce",
]


class BlockReduceAlgorithm(enum.Enum):
    """How a block folds its per-thread partials into one value.

    Every policy implemented so far needs a commutative *op*, whatever its name suggests. The block
    layer itself folds in order — a raking lane walks its segment by ascending index, and the
    per-warp totals are combined lowest warp first — but the warp-scope reduction underneath does
    not, so the requirement belongs to the collective rather than to any one policy.

    So no member below is currently distinguished by accepting a non-commutative *op*, and none can
    be while every :class:`~flydsl.expr.typing.ReductionOp` on offer is commutative.

    ``RAKING``
        Stage every thread's partial in shared memory, then let a single warp walk ("rake")
        equal-length segments of it. ``block_threads`` slots of shared memory, but the cross-lane
        work collapses to one warp. Named for the variant that honours operand order, which this
        one does not yet.
    ``RAKING_COMMUTATIVE_ONLY``
        *Planned, not implemented.* Raking that spends the freedom to reorder: the first warp keeps
        its partial in registers rather than staging it, and the rake strides by the warp width
        instead of walking a contiguous segment, which is bank-conflict-free where a contiguous
        walk is not. That access pattern is what it would add — the relaxed ordering it is named
        for is already in force above.
    ``WARP_REDUCTIONS``
        Reduce inside each warp with shuffles, then fold the per-warp aggregates through shared
        memory. Only ``num_warps`` slots of shared memory and one barrier.
    ``WARP_REDUCTIONS_NONDETERMINISTIC``
        *Planned, not implemented.* Warp reductions whose combining order may vary between runs,
        which makes floating-point results irreproducible.
    """

    RAKING = "raking"
    RAKING_COMMUTATIVE_ONLY = "raking_commutative_only"  # planned, not implemented
    WARP_REDUCTIONS = "warp_reductions"
    WARP_REDUCTIONS_NONDETERMINISTIC = "warp_reductions_nondeterministic"  # planned, not implemented


# ── policy implementations ─────────────────────────────────────────────────
# One function per BlockReduceAlgorithm, each paired with the shared storage it
# needs in _SHARED_STORAGE below.


@jit
def _reduce_warp_reductions(partial, tid, slots, op, warp_reduce, warp_threads, num_warps):
    """Reduce within each warp, then fold the per-warp aggregates through shared memory."""
    aggregate = warp_reduce(partial, op, width=warp_threads)
    if const_expr(num_warps == 1):
        total = aggregate
    else:
        lane = tid % warp_threads
        warp_id = tid // warp_threads
        if lane == 0:
            slots[warp_id] = aggregate
        barrier()
        # Every thread folds the same num_warps values, so the result is valid
        # block-wide and no second barrier is needed to broadcast it.

        # TODO: that fold is linear in num_warps and every thread walks it —
        # num_warps - 1 combines each, so 15 at a 1024-thread wave64 block.
        # Reducing *slots* in one warp instead would make it logarithmic, at the
        # cost of the second barrier this shape is currently free of.
        total = slots[0]
        for i in range_constexpr(1, num_warps):
            total = combine(op, total, slots[i])
    return total


@jit
def _reduce_raking(partial, tid, slots, result, op, warp_reduce, warp_threads, segment_length):
    """Stage every thread's partial in shared memory, then rake it with a single warp."""
    if const_expr(segment_length == 1):
        # One segment per raking lane means the block is already one warp, so
        # staging it would only be a round trip out to shared memory and back into the
        # same cross-lane fold. This is the warp reduction, unchanged.
        total = warp_reduce(partial, op, width=warp_threads)
    else:
        slots[tid] = partial
        barrier()
        if tid < warp_threads:
            base = tid * segment_length
            raked = slots[base]
            for i in range_constexpr(1, segment_length):
                raked = combine(op, raked, slots[base + i])
            raked = warp_reduce(raked, op, width=warp_threads)
            if tid == 0:
                result[0] = raked
        barrier()
        total = result[0]
    return total


def _storage_warp_reductions(dtype, block_threads, warp_threads):
    return Struct["slots" : Array[dtype, block_threads // warp_threads]]


def _storage_raking(dtype, block_threads, warp_threads):
    # A single-warp block never reaches the raking grid — see _reduce_raking —
    # so reserving a slot per thread for it would just be shared memory nothing writes.
    slots = block_threads if block_threads > warp_threads else 1
    return Struct["slots" : Array[dtype, slots], "result" : Array[dtype, 1]]


# Registry of the implemented policies. An unlisted member of the enum names a
# strategy this library has not implemented yet.
_SHARED_STORAGE = {
    BlockReduceAlgorithm.WARP_REDUCTIONS: _storage_warp_reductions,
    BlockReduceAlgorithm.RAKING: _storage_raking,
}


class _BlockReduceMeta(BlockAlgorithmMeta):
    """Gives ``BlockReduce`` its ``[...]`` specialization and call syntax."""

    _algorithms = BlockReduceAlgorithm
    _shared_storage = _SHARED_STORAGE

    def _default_algorithm_for(cls, target):
        """``WARP_REDUCTIONS`` everywhere so far, and this is where that is decided.

        Measured on both a wave64 and a wave32 target, it comes out ahead of
        ``RAKING`` at every block width — fewer shared memory accesses, half the
        barriers, and a gap that widens with the warp count. A target that
        inverts that gets its branch here.
        """
        return BlockReduceAlgorithm.WARP_REDUCTIONS

    def __call__(cls, value, op, *, storage):
        if cls.block_threads is None:
            raise TypeError("specialize first, e.g. BlockReduce[fx.Float32, 256]")

        partial = thread_partial(value, op)
        tid = linear_thread_id(cls.block_size)
        if cls.algorithm is BlockReduceAlgorithm.WARP_REDUCTIONS:
            return _reduce_warp_reductions(
                partial,
                tid,
                storage.slots,
                op,
                cls.warp_ops.warp_reduce,
                cls.warp_threads,
                cls.num_warps,
            )
        return _reduce_raking(
            partial,
            tid,
            storage.slots,
            storage.result,
            op,
            cls.warp_ops.warp_reduce,
            cls.warp_threads,
            cls.num_warps,
        )


class BlockReduce(metaclass=_BlockReduceMeta):
    """Block-wide reduction.

    Specialize it, allocate its shared storage, then call it::

        block_reduce = fx.coop.BlockReduce[fx.Float32, 256]
        storage = fx.SharedAllocator().allocate(block_reduce.SharedStorage).peek()
        total = block_reduce(value, fx.ReductionOp.ADD, storage=storage)

    The parameters are ``[dtype, block_size, algorithm]``. *block_size* is either the x extent on
    its own or the full ``(x, y, z)``, and must be a power of two in total.

    Where the kernel declares its own block size — the usual case, either through
    ``known_block_size`` or inferred from a static launch — passing
    :func:`~flydsl.expr.gpu.known_block_size` straight through is the way to keep the two in step
    without repeating the dimensions::

        block_reduce = fx.coop.BlockReduce[fx.Float32, fx.known_block_size()]

    ``value`` is either one scalar per thread or a ``Vector`` of several per-thread items. The
    result is valid in every thread of the block, not only in one of them.

    Every thread of the block has to reach this call, and reach it together. It synchronizes the
    block and reads across lanes, so a call made under a condition that is not uniform block-wide
    hangs on the barrier or folds in lanes with no defined value.

    *op* must be commutative under every algorithm implemented so far, and associative under all of
    them; :class:`BlockReduceAlgorithm` says where that comes from and why no policy is currently
    free of it. Nothing depends on the distinction while every
    :class:`~flydsl.expr.typing.ReductionOp` is commutative.

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

    # Where the warp-scope fold underneath comes from. The default is the
    # dispatched namespace, so a block reduction picks up whatever override the
    # target supplies. :mod:`flydsl.extension.coop.universal` subclasses this and
    # points it at the portable implementations instead.
    warp_ops = _dispatched_warp
