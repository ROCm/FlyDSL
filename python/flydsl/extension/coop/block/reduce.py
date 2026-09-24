# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block reductions with ordered, commutative, and arrival-order policies."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.llvm import atomic_add
from ....expr.numeric import Int32, Integer
from ....expr.primitive import const_expr, range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array
from .. import warp as _dispatched_warp
from .._common import (
    _combine,
    _convert_value,
    _linear_thread_id,
    _optional_identity,
    _require_commutative,
    _select_value,
    _thread_partial,
    _validate_scalar_valid_items,
)
from .._values import _item_dtype, _shared_array, _shared_load, _shared_store
from ._spec import BlockAlgorithmMeta

__all__ = ["BlockReduceAlgorithm", "BlockReduce"]


class BlockReduceAlgorithm(enum.Enum):
    """Shared-memory policies for block reduction.

    All policies require associative operators. Ordered policies preserve
    flattened thread and register-item order but may regroup operations.

    Attributes:
        RAKING: Reduce contiguous segments in order using shared memory.
        RAKING_COMMUTATIVE_ONLY: Rake strided columns while keeping the first
            warp's partials in registers. Requires a built-in ReductionOp or
            a callable declaring ``commutative = True``.
        WARP_REDUCTIONS: Reduce each warp, then combine warp totals in thread
            order. This is the default policy.
        WARP_REDUCTIONS_NONDETERMINISTIC: Assign shared slots to warp totals
            with atomic arrival tickets and fold them in ticket order. Requires
            a commutative operator; floating-point results may vary across runs.
    """

    RAKING = "raking"
    RAKING_COMMUTATIVE_ONLY = "raking_commutative_only"
    WARP_REDUCTIONS = "warp_reductions"
    WARP_REDUCTIONS_NONDETERMINISTIC = "warp_reductions_nondeterministic"

    Raking = RAKING
    RakingCommutativeOnly = RAKING_COMMUTATIVE_ONLY
    WarpReductions = WARP_REDUCTIONS
    WarpReductionsNondeterministic = WARP_REDUCTIONS_NONDETERMINISTIC


@jit
def _reduce_warps(partial, tid, storage, op, warp_reduce, warp_threads, block_threads, nondeterministic, valid_threads):
    lane = tid % warp_threads
    warp_id = tid // warp_threads
    num_warps = (block_threads + warp_threads - 1) // warp_threads
    active = Int32(warp_threads)
    if const_expr(valid_threads is not None):
        remaining = valid_threads - warp_id * warp_threads
        active = (remaining < warp_threads).select(remaining, active)
        # Empty warps contribute no slot to the final fold; use an existing
        # lane as their temporary source without inventing a semigroup identity.
        active = (active > 0).select(active, Int32(1))
        aggregate = warp_reduce(partial, op, width=warp_threads, valid_items=active, _dtype=_item_dtype(partial))
    elif const_expr(block_threads % warp_threads != 0):
        active = (warp_id == num_warps - 1).select(Int32(block_threads % warp_threads), active)
        aggregate = warp_reduce(partial, op, width=warp_threads, valid_items=active, _dtype=_item_dtype(partial))
    else:
        aggregate = warp_reduce(partial, op, width=warp_threads, _dtype=_item_dtype(partial))
    if const_expr(num_warps == 1):
        total = aggregate
    else:
        if const_expr(nondeterministic):
            if tid == 0:
                storage.counter[0] = Int32(0)
            barrier()
            eligible = lane == 0
            if const_expr(valid_threads is not None):
                eligible = eligible & (warp_id * warp_threads < valid_threads)
            if eligible:
                ticket = atomic_add(storage.counter.ptr, Int32(1))
                _shared_store(storage.slots, ticket, aggregate)
        else:
            if lane == 0:
                _shared_store(storage.slots, warp_id, aggregate)
        barrier()
        total = _shared_load(storage.slots, 0)
        for i in range_constexpr(1, num_warps):
            if const_expr(valid_threads is not None):
                # Do not load uninitialized arrival slots for empty warps.
                if i * warp_threads < valid_threads:
                    total = _combine(op, total, _shared_load(storage.slots, i))
            else:
                total = _combine(op, total, _shared_load(storage.slots, i))
    return total


@jit
def _reduce_raking(partial, tid, storage, op, warp_reduce, warp_threads, block_threads, commutative, valid_threads):
    if const_expr(valid_threads is None):
        # Complete logical warps make every raking segment complete. Do not
        # manufacture a valid_items argument: that would disable the DPP path.
        if const_expr(block_threads == warp_threads):
            return warp_reduce(partial, op, width=warp_threads, _dtype=_item_dtype(partial))
        if const_expr(commutative):
            if tid >= warp_threads:
                _shared_store(storage.slots, tid, partial)
        else:
            _shared_store(storage.slots, tid, partial)
        barrier()
        if tid < warp_threads:
            if const_expr(commutative):
                raked = partial
                for offset in range_constexpr(warp_threads, block_threads, warp_threads):
                    raked = _combine(op, raked, _shared_load(storage.slots, tid + offset))
            else:
                segment = block_threads // warp_threads
                base = tid * segment
                raked = _shared_load(storage.slots, base)
                for i in range_constexpr(1, segment):
                    raked = _combine(op, raked, _shared_load(storage.slots, base + i))
            raked = warp_reduce(raked, op, width=warp_threads, _dtype=_item_dtype(raked))
            if tid == 0:
                _shared_store(storage.result, 0, raked)
        barrier()
        return _shared_load(storage.result, 0)
    limit = valid_threads
    if const_expr(block_threads == warp_threads):
        total = warp_reduce(partial, op, width=warp_threads, valid_items=limit, _dtype=_item_dtype(partial))
    else:
        if const_expr(commutative):
            if tid >= warp_threads:
                _shared_store(storage.slots, tid, partial)
        else:
            _shared_store(storage.slots, tid, partial)
        barrier()
        if tid < warp_threads:
            if const_expr(commutative):
                raked = partial
                for offset in range_constexpr(warp_threads, block_threads, warp_threads):
                    index = tid + offset
                    safe = (index < block_threads).select(index, Int32(block_threads - 1))
                    raked = _select_value(index < limit, _combine(op, raked, _shared_load(storage.slots, safe)), raked)
                active = (limit < warp_threads).select(limit, Int32(warp_threads))
                raked = warp_reduce(raked, op, width=warp_threads, valid_items=active, _dtype=_item_dtype(raked))
            else:
                segment = (block_threads + warp_threads - 1) // warp_threads
                base = tid * segment
                safe = (base < block_threads).select(base, Int32(block_threads - 1))
                raked = _shared_load(storage.slots, safe)
                for i in range_constexpr(1, segment):
                    index = base + i
                    safe = (index < block_threads).select(index, Int32(block_threads - 1))
                    raked = _select_value(index < limit, _combine(op, raked, _shared_load(storage.slots, safe)), raked)
                raked = warp_reduce(
                    raked,
                    op,
                    width=warp_threads,
                    valid_items=(limit + segment - 1) // segment,
                    _dtype=_item_dtype(raked),
                )
            if tid == 0:
                _shared_store(storage.result, 0, raked)
        barrier()
        total = _shared_load(storage.result, 0)
    return total


def _storage_warps(dtype, block_threads, warp_threads):
    return Struct["slots" : _shared_array(dtype, (block_threads + warp_threads - 1) // warp_threads)]


def _storage_nondeterministic(dtype, block_threads, warp_threads):
    return Struct[
        "slots" : _shared_array(dtype, (block_threads + warp_threads - 1) // warp_threads), "counter" : Array[Int32, 1]
    ]


def _storage_raking(dtype, block_threads, warp_threads):
    slots = block_threads if block_threads > warp_threads else 1
    return Struct["slots" : _shared_array(dtype, slots), "result" : _shared_array(dtype, 1)]


class _BlockReduceMeta(BlockAlgorithmMeta):
    _supports_subwarp = True
    _algorithms = BlockReduceAlgorithm
    _shared_storage = {
        BlockReduceAlgorithm.WARP_REDUCTIONS: _storage_warps,
        BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC: _storage_nondeterministic,
        BlockReduceAlgorithm.RAKING: _storage_raking,
        BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY: _storage_raking,
    }

    def _default_algorithm_for(cls, target):
        return BlockReduceAlgorithm.WARP_REDUCTIONS

    def __call__(
        cls,
        value,
        op,
        *,
        storage,
        valid_items: int | Integer | None = None,
        identity=None,
    ):
        """Reduce a flattened blocked input sequence and broadcast the aggregate.

        All block threads must participate. Ordered policies preserve operand
        order, while commutative policies may reorder it. Synchronize before
        reusing the shared allocation.

        Args:
            value: One element or a nonempty list/tuple/Vector of consecutive
                elements, converted to the specialized dtype. All threads must
                provide the same item count; arrays form one block-wide task.
            op: Associative ReductionOp or binary callable returning the element
                dtype. Values that need a custom operation require a callable. Commutative policies
                require ReductionOp or a callable with ``commutative = True``.
            storage: Shared instance of the specialization's ``SharedStorage``.
            valid_items: Uniform number of leading contributing threads in
                ``[0, block_threads]``, or ``None`` for all inputs. Only the
                single-item overload accepts this argument; omit it for an
                item range. Runtime bounds are the caller's responsibility.
            identity: Optional neutral element or callable ``identity(dtype)`` used
                for masking a valid prefix. Overrides a built-in identity or
                ``op.identity``. Without one, nonempty prefixes are still supported
                and the empty reduction has an unspecified result.

        Returns:
            One aggregate value of the specialized dtype in every
            participating thread, regardless of the per-thread input tile length.
            An empty valid prefix returns the identity when available.
        """
        if cls.block_threads is None:
            raise TypeError("specialize first, e.g. BlockReduce[fx.Float32, 256]")
        commutative = cls.algorithm is BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY
        nondeterministic = cls.algorithm is BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC
        if commutative or nondeterministic:
            _require_commutative(op, cls.algorithm.name)
        # Tensor signatures may carry signless integer storage. Honor the
        # specialization's signedness before any thread-local min/max fold.
        _validate_scalar_valid_items(value, valid_items, cls.block_threads, cls.dtype)
        value = _convert_value(value, cls.dtype)
        tid = _linear_thread_id(cls.block_size)
        valid_threads = None
        if valid_items is not None:
            neutral = _optional_identity(op, cls.dtype, identity)
            if neutral is None:
                valid_threads = Int32(valid_items)
                # Without an identity, an empty aggregate is unspecified. Read
                # a real input lane for that case, never uninitialized scratch.
                valid_threads = (valid_threads > 0).select(valid_threads, Int32(1))
                partial = value
            else:
                partial = _select_value(tid < valid_items, value, neutral)
        else:
            partial = _thread_partial(value, op, cls.dtype)
        if cls.algorithm in (
            BlockReduceAlgorithm.WARP_REDUCTIONS,
            BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
        ):
            return _reduce_warps(
                partial,
                tid,
                storage,
                op,
                cls.warp_ops.warp_reduce,
                cls.warp_threads,
                cls.block_threads,
                nondeterministic,
                valid_threads,
            )
        return _reduce_raking(
            partial,
            tid,
            storage,
            op,
            cls.warp_ops.warp_reduce,
            cls.warp_threads,
            cls.block_threads,
            commutative,
            valid_threads,
        )


class BlockReduce(metaclass=_BlockReduceMeta):
    """Reduce a block's input values in blocked per-thread order.

    Specialize as ``BlockReduce[dtype, block_size, algorithm]``. The default
    policy is ``WARP_REDUCTIONS``. ``block_size`` is a positive thread count or
    positive ``(x, y, z)`` extents. ``dtype`` specifies one element's type,
    including structured or vector elements. A bare Vector is one element
    for a Vector dtype, and a scalar item range for a Numeric dtype.
    A list/tuple of vector elements
    keeps each vector intact. Structured elements require a callable operator.
    Tile lengths are inferred from inputs and must be uniform.

    Every thread receives the aggregate. Item ranges follow blocked order:
    each thread's items precede those of the next linear thread.
    ``valid_items`` is supported only for a single item per thread and counts
    leading contributing threads. RAKING and WARP_REDUCTIONS preserve this
    order for associative noncommutative operators. The two commutative policies
    require a built-in ReductionOp or ``op.commutative = True`` and may reorder
    operands. Floating-point regrouping may affect results even in ordered policies.

    An empty valid prefix returns the explicit or operator-provided identity
    when available; without an identity its result is unspecified. Every block
    thread must enter the collective with uniform options. Use one shared
    allocation of ``SharedStorage`` and synchronize before reusing it.

    A block smaller than the target's physical warp (64 on CDNA, 32 on RDNA)
    must have a power-of-two thread count; its logical warp narrows to that
    count. Larger blocks must contain complete physical warps. valid_items
    controls a partial input tile; all launched threads still participate.

    Attributes:
        dtype: Specialized reduction element type.
        block_size: Specialized ``(x, y, z)`` shape.
        block_threads: Number of participating threads.
        algorithm: Selected reduction policy.
        warp_threads: Logical warp width selected for the target and block.
        num_warps: Number of logical warps in the block.
        SharedStorage: Shared-memory Struct type required by the specialization.

    Examples:
        # x in thread t is [2*t+1, 2*t+2], so the block owns integers 1 through 128.
        P = fx.coop.BlockReduce[fx.Int32, 64, fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y = P(x, fx.ReductionOp.ADD, storage=storage)
        # Every thread receives y=8256.
        fx.barrier()
        partial = P(x[0], fx.ReductionOp.ADD, storage=storage, valid_items=5, identity=fx.Int32(0))
        # The first five threads contribute 1, 3, 5, 7, 9; every thread receives 25.
    """

    dtype = None
    block_size = None
    block_threads = None
    algorithm = None
    warp_threads = None
    num_warps = None
    SharedStorage = None
    warp_ops = _dispatched_warp
