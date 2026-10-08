# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Ordered block scans with warp and two independently implemented raking policies."""

import enum
from typing import NoReturn

from ....compiler import jit
from ....expr.gpu import barrier, num_warp_threads
from ....expr.numeric import Int32, Integer
from ....expr.primitive import const_expr, range_constexpr
from ....expr.struct import Struct
from ....expr.typing import ReductionOp
from .. import warp as _dispatched_warp
from .._common import (
    _cast_value,
    _combine,
    _convert_item,
    _convert_value,
    _identity,
    _linear_thread_id,
    _optional_identity,
    _seed,
    _select_value,
    _shuffle_value,
    _validate_scalar_valid_items,
)
from .._values import _as_items, _from_items, _is_items, _item_dtype, _shared_array, _shared_load, _shared_store
from ._spec import BlockAlgorithmMeta

__all__ = ["BlockScanAlgorithm", "BlockScan"]


class BlockScanAlgorithm(enum.Enum):
    """Ordered shared-memory policies for block scans.

    All policies preserve flattened thread and register-item order and support
    associative noncommutative operators; parenthesization may differ.

    Attributes:
        RAKING: Stage thread totals and reload each raking segment during
            the downsweep.
        RAKING_MEMOIZE: Keep segment-local prefixes in registers and write
            them after scanning segment totals.
        WARP_SCANS: Stage warp totals and combine ordered warp prefixes.
            This is the default policy.
    """

    RAKING = "raking"
    RAKING_MEMOIZE = "raking_memoize"
    WARP_SCANS = "warp_scans"

    Raking = RAKING
    RakingMemoize = RAKING_MEMOIZE
    WarpScans = WARP_SCANS


@jit
def _prefix_warp_scans_full(partial, tid, slots, op, warp_scan_with_aggregate, warp_threads, num_warps):
    """Scan within each warp, then fold in the aggregates of the warps in front.

    Returns the thread's exclusive prefix together with the block's own
    aggregate. The aggregate is free: with more than one warp every thread can
    already read every warp's total out of *slots*, and with a single warp the
    scan's top lane already holds it.
    """
    inclusive, prefix, aggregate = warp_scan_with_aggregate(
        partial, op, width=warp_threads, _dtype=_item_dtype(partial)
    )
    if const_expr(num_warps > 1):
        lane = tid % warp_threads
        warp_id = tid // warp_threads
        if lane == warp_threads - 1:
            _shared_store(slots, warp_id, inclusive)
        barrier()
        # Highest warp first, so the operands stay in block order: a thread in
        # warp w ends up with slots[0] ⊕ ... ⊕ slots[w-1] ⊕ its own warp prefix.

        # TODO: this and the aggregate below are linear in num_warps, and every
        # thread walks both — 2 * (num_warps - 1) folds each, so 30 at a
        # 1024-thread wave64 block. Scanning *slots* in one warp instead would
        # make it logarithmic: warp 0 scans the num_warps totals, and each
        # thread then reads the single entry in front of its own warp.
        for i in range_constexpr(num_warps - 2, -1, -1):
            prefix = _select_value(warp_id > i, _combine(op, _shared_load(slots, i), prefix), prefix)
        # Same slots, folded unconditionally: that is the whole block.
        aggregate = _shared_load(slots, 0)
        for i in range_constexpr(1, num_warps):
            aggregate = _combine(op, aggregate, _shared_load(slots, i))
    return prefix, aggregate


@jit
def _prefix_warp_scans(partial, tid, storage, op, warp_inclusive_scan, warp_threads, block_threads):
    lane = tid % warp_threads
    warp_id = tid // warp_threads
    num_warps = (block_threads + warp_threads - 1) // warp_threads
    raw = warp_inclusive_scan(partial, op, width=warp_threads, _dtype=_item_dtype(partial))
    prefix = _shuffle_value(raw, 1, warp_threads, mode="up")
    last = Int32(warp_threads - 1)
    if const_expr(block_threads % warp_threads != 0):
        last = (warp_id == num_warps - 1).select(Int32(block_threads % warp_threads - 1), last)
    # shuffle_idx uses a physical-wave lane index, even for logical subwarps.
    from ....expr.gpu import lane_id

    base = (lane_id() // warp_threads) * warp_threads
    aggregate = _shuffle_value(raw, base + last, warp_threads)
    if const_expr(num_warps > 1):
        if lane == 0:
            _shared_store(storage.slots, warp_id, aggregate)
        barrier()
        for i in range_constexpr(num_warps - 2, -1, -1):
            preceding = _shared_load(storage.slots, i)
            joined = _combine(op, preceding, prefix)
            joined = _select_value((lane == 0) & (warp_id == i + 1), preceding, joined)
            prefix = _select_value(warp_id > i, joined, prefix)
        aggregate = _shared_load(storage.slots, 0)
        for i in range_constexpr(1, num_warps):
            aggregate = _combine(op, aggregate, _shared_load(storage.slots, i))
    return prefix, aggregate


def _segment_prefixes(slots, base, segment, block_threads, op):
    """Register memoization used only by RAKING_MEMOIZE."""
    values = []
    running = None
    for i in range(segment):
        index = base + i
        safe = (index < block_threads).select(index, Int32(block_threads - 1))
        item = _shared_load(slots, safe)
        if running is None:
            running = item
        else:
            running = _select_value(index < block_threads, _combine(op, running, item), running)
        values.append(running)
    return _from_items(values)


@jit
def _prefix_raking(partial, tid, storage, op, warp_inclusive_scan, warp_threads, block_threads, memoize):
    segment = (block_threads + warp_threads - 1) // warp_threads
    segments = (block_threads + segment - 1) // segment
    _shared_store(storage.slots, tid, partial)
    barrier()
    if tid < warp_threads:
        base = tid * segment
        if const_expr(memoize):
            saved = _segment_prefixes(storage.slots, base, segment, block_threads, op)
            total = saved[segment - 1]
        else:
            safe = (base < block_threads).select(base, Int32(block_threads - 1))
            total = _shared_load(storage.slots, safe)
            for i in range_constexpr(1, segment):
                index = base + i
                safe = (index < block_threads).select(index, Int32(block_threads - 1))
                total = _select_value(
                    index < block_threads, _combine(op, total, _shared_load(storage.slots, safe)), total
                )
        scanned = warp_inclusive_scan(total, op, width=warp_threads, _dtype=_item_dtype(total))
        preceding = _shuffle_value(scanned, 1, warp_threads, mode="up")
        aggregate = _shuffle_value(scanned, segments - 1, warp_threads)
        if tid == 0:
            _shared_store(storage.aggregate, 0, aggregate)
        if const_expr(memoize):
            for i in range_constexpr(segment):
                index = base + i
                if index < block_threads:
                    local_prefix = saved[i]
                    _shared_store(
                        storage.output,
                        index,
                        _select_value(tid > 0, _combine(op, preceding, local_prefix), local_prefix),
                    )
        else:
            safe = (base < block_threads).select(base, Int32(block_threads - 1))
            running = _shared_load(storage.slots, safe)
            for i in range_constexpr(segment):
                index = base + i
                if index < block_threads:
                    if const_expr(i > 0):
                        running = _combine(op, running, _shared_load(storage.slots, index))
                    _shared_store(
                        storage.output, index, _select_value(tid > 0, _combine(op, preceding, running), running)
                    )
    barrier()
    previous = (tid > 0).select(tid - 1, Int32(0))
    return _shared_load(storage.output, previous), _shared_load(storage.aggregate, 0)


@jit
def _callback_prefix(aggregate, tid, storage, callback, warp_threads):
    # The policy selects callback threads in the first warp; only lane zero's
    # return value is used. Pointer-backed state can carry prefixes across tiles.
    barrier()
    if tid < warp_threads:
        prefix = _convert_item(callback(aggregate), _item_dtype(aggregate))
        if tid == 0:
            _shared_store(storage.callback, 0, prefix)
    barrier()
    return _shared_load(storage.callback, 0)


def _storage_warp_scans(dtype, block_threads, warp_threads):
    @Struct
    class Storage:
        slots: _shared_array(dtype, block_threads // warp_threads)

        @property
        def callback(self):
            # Callback use starts after the warp-prefix phase; reuse slot zero.
            return self.slots

    return Storage


def _storage_raking(dtype, block_threads, warp_threads):
    return Struct[
        "slots" : _shared_array(dtype, block_threads),
        "output" : _shared_array(dtype, block_threads),
        "aggregate" : _shared_array(dtype, 1),
        "callback" : _shared_array(dtype, 1),
    ]


class _BlockScanMeta(BlockAlgorithmMeta):
    _supports_subwarp = True
    _algorithms = BlockScanAlgorithm
    _shared_storage = {
        BlockScanAlgorithm.WARP_SCANS: _storage_warp_scans,
        BlockScanAlgorithm.RAKING: _storage_raking,
        BlockScanAlgorithm.RAKING_MEMOIZE: _storage_raking,
    }

    def _default_algorithm_for(cls, target):
        return BlockScanAlgorithm.WARP_SCANS

    def __call__(cls, *args, **kwargs) -> NoReturn:
        """Reject an ambiguous direct scan call.

        Use ``inclusive(...)``, ``exclusive(...)``, or their ``_with_aggregate``
        variants on a specialization to select the desired scan form.

        Args:
            *args: Positional arguments rejected by this interface.
            **kwargs: Keyword arguments rejected by this interface.
        """
        raise TypeError("a scan has two forms; call .inclusive(...) or .exclusive(...)")


class BlockScan(metaclass=_BlockScanMeta):
    """Scan input values in blocked order.

    Specialize as ``BlockScan[dtype, block_size, algorithm]``; ``algorithm``
    defaults to ``WARP_SCANS``. ``block_size`` is a positive thread count or
    positive ``(x, y, z)`` extents.
    The declared dtype is one complete element. A bare Vector is one element
    for a Vector dtype, and a scalar item sequence for a Numeric dtype.
    Prefixes preserve the input item count; seeds and aggregate are one dtype.
    Tile lengths are inferred from inputs and must be uniform across threads.
    All policies support associative operators and preserve flattened blocked
    order: thread ``t``'s items precede thread ``t + 1``'s items. Regrouping may
    affect floating-point results. Values that need a custom operation require a callable operator.

    Custom callables may supply ``identity(dtype)`` or a constant ``identity``
    member, overridden by explicit ``identity=``. Inclusive scans need no
    identity. Exclusive scans without an identity or ``init`` leave the first
    flattened output unspecified. ``valid_items`` is supported only for one
    item per thread, counts leading contributing threads, and requires an
    identity to mask the remainder. Item ranges require ``valid_items=None``.
    The aggregate excludes both ``init`` and any callback prefix.

    ``prefix_callback(aggregate)`` executes in the first physical warp: all its
    active lanes for WARP_SCANS, and the raking-thread subset for RAKING and
    RAKING_MEMOIZE. Its lane-zero return value seeds the entire block; other
    return values are ignored. Use pointer-backed state when a callback carries
    a prefix between tiles. An explicit ``init`` and a callback are mutually
    exclusive.

    All threads must participate with uniform options and a common allocation
    of ``SharedStorage``. Synchronize the block before reusing storage for
    another collective. Result values retain the input item count and shape.

    A block smaller than the target's physical warp (64 on CDNA, 32 on RDNA)
    must have a power-of-two thread count; its logical warp narrows to that
    count. Larger blocks must contain complete physical warps. valid_items
    controls a partial input tile; all launched threads still participate.

    Attributes:
        dtype: Specialized scan element type.
        block_size: Specialized ``(x, y, z)`` shape.
        block_threads: Number of participating threads.
        algorithm: Selected scan policy.
        warp_threads: Logical warp width selected for the target and block.
        num_warps: Number of logical warps in the block.
        SharedStorage: Shared-memory Struct type required by the specialization.

    Examples:
        # Compute prefix sums of 256 ones, with four consecutive items per thread. The scan continues
        # across thread boundaries.

        P = fx.coop.BlockScan[fx.Int32, 64, fx.coop.BlockScanAlgorithm.WARP_SCANS]
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

        # Guarded scans accept one item per thread. The first six threads
        # contribute one each; aggregate=6 and inclusive[T0..T5]=11..16.
        fx.barrier()
        inclusive, aggregate = P.inclusive_with_aggregate(
            x[0], fx.ReductionOp.ADD, storage=storage, init=10, valid_items=6
        )

        # A callback also supplies one prefix for a complete array tile.
        def previous_prefix(aggregate):
            return fx.Int32(10)

        fx.barrier()
        exclusive, aggregate = P.exclusive_with_aggregate(
            x, fx.ReductionOp.ADD, storage=storage, prefix_callback=previous_prefix
        )
        # aggregate=256; T0 receives [10,11,12,13], T1 receives [14,15,16,17].
    """

    dtype = None
    block_size = None
    block_threads = None
    algorithm = None
    warp_threads = None
    num_warps = None
    SharedStorage = None
    warp_ops = _dispatched_warp

    @classmethod
    def inclusive(
        cls,
        value,
        op,
        *,
        storage,
        init=None,
        identity=None,
        valid_items: int | Integer | None = None,
        prefix_callback=None,
    ):
        """Compute inclusive prefixes, including each output item's own input.

        Without a seed or identity, the first output is the first input.
        All threads must participate and synchronize before reusing storage.

        Args:
            value: One element or a nonempty list/tuple/Vector of consecutive
                elements, converted to the specialized dtype. Tile lengths must
                match across threads; items are ordered by thread, then item index.
                A list/tuple of vector elements keeps each vector intact.
            op: Associative ReductionOp or binary callable returning the element
                dtype. Values that need a custom operation require a callable. Operand order is preserved
                and commutativity is not required.
            storage: Shared instance of the specialization's ``SharedStorage``.
            init: Optional single value of the element dtype, applied as the
                left operand before the entire input sequence. Mutually exclusive
                with ``prefix_callback``.
            identity: Optional neutral element or callable ``identity(dtype)``.
                Overrides a built-in identity or ``op.identity``. Required from
                one of these sources when ``valid_items`` is supplied.
            valid_items: Uniform number of leading contributing threads in
                ``[0, block_threads]``, or ``None`` for all inputs. Only the
                single-item overload accepts this argument; omit it for an
                item range. Runtime bounds are the caller's responsibility.
            prefix_callback: Optional callable receiving the input aggregate.
                Runs in the first physical warp's active lanes for WARP_SCANS,
                or its raking-thread subset for raking policies. Only lane zero's
                returned prefix seeds the scan. Pointer-backed state may persist
                between tiles; the aggregate excludes the returned prefix.

        Returns:
            Prefix values in blocked order and the input's per-thread shape. Seeds precede the input sequence as
            left operands.
        """
        return cls._scan(
            value,
            op,
            storage,
            inclusive=True,
            init=init,
            identity=identity,
            valid_items=valid_items,
            prefix_callback=prefix_callback,
        )[0]

    @classmethod
    def exclusive(
        cls,
        value,
        op,
        *,
        storage,
        init=None,
        identity=None,
        valid_items: int | Integer | None = None,
        prefix_callback=None,
    ):
        """Compute exclusive prefixes, excluding each output item's own input.

        The first flattened output is the seed or identity when available;
        otherwise only that first output is unspecified.
        All threads must participate and synchronize before reusing storage.

        Args:
            value: One element or a nonempty list/tuple/Vector of consecutive
                elements, converted to the specialized dtype. Tile lengths must
                match across threads; items are ordered by thread, then item index.
                A list/tuple of vector elements keeps each vector intact.
            op: Associative ReductionOp or binary callable returning the element
                dtype. Values that need a custom operation require a callable. Operand order is preserved
                and commutativity is not required.
            storage: Shared instance of the specialization's ``SharedStorage``.
            init: Optional single value of the element dtype, applied as the
                left operand before the entire input sequence. Mutually exclusive
                with ``prefix_callback``.
            identity: Optional neutral element or callable ``identity(dtype)``.
                Overrides a built-in identity or ``op.identity``. Required from
                one of these sources when ``valid_items`` is supplied.
            valid_items: Uniform number of leading contributing threads in
                ``[0, block_threads]``, or ``None`` for all inputs. Only the
                single-item overload accepts this argument; omit it for an
                item range. Runtime bounds are the caller's responsibility.
            prefix_callback: Optional callable receiving the input aggregate.
                Runs in the first physical warp's active lanes for WARP_SCANS,
                or its raking-thread subset for raking policies. Only lane zero's
                returned prefix seeds the scan. Pointer-backed state may persist
                between tiles; the aggregate excludes the returned prefix.

        Returns:
            Prefix values in blocked order and the input's per-thread shape. Seeds precede the input sequence as
            left operands.
        """
        return cls._scan(
            value,
            op,
            storage,
            inclusive=False,
            init=init,
            identity=identity,
            valid_items=valid_items,
            prefix_callback=prefix_callback,
        )[0]

    @classmethod
    def inclusive_with_aggregate(
        cls,
        value,
        op,
        *,
        storage,
        init=None,
        identity=None,
        valid_items: int | Integer | None = None,
        prefix_callback=None,
    ):
        """Compute inclusive prefixes, including each output item's own input.

        Without a seed or identity, the first output is the first input.
        All threads must participate and synchronize before reusing storage.

        Args:
            value: One element or a nonempty list/tuple/Vector of consecutive
                elements, converted to the specialized dtype. Tile lengths must
                match across threads; items are ordered by thread, then item index.
                A list/tuple of vector elements keeps each vector intact.
            op: Associative ReductionOp or binary callable returning the element
                dtype. Values that need a custom operation require a callable. Operand order is preserved
                and commutativity is not required.
            storage: Shared instance of the specialization's ``SharedStorage``.
            init: Optional single value of the element dtype, applied as the
                left operand before the entire input sequence. Mutually exclusive
                with ``prefix_callback``.
            identity: Optional neutral element or callable ``identity(dtype)``.
                Overrides a built-in identity or ``op.identity``. Required from
                one of these sources when ``valid_items`` is supplied.
            valid_items: Uniform number of leading contributing threads in
                ``[0, block_threads]``, or ``None`` for all inputs. Only the
                single-item overload accepts this argument; omit it for an
                item range. Runtime bounds are the caller's responsibility.
            prefix_callback: Optional callable receiving the input aggregate.
                Runs in the first physical warp's active lanes for WARP_SCANS,
                or its raking-thread subset for raking policies. Only lane zero's
                returned prefix seeds the scan. Pointer-backed state may persist
                between tiles; the aggregate excludes the returned prefix.

        Returns:
            A pair ``(prefixes, aggregate)``. Prefixes have the same per-thread
            shape as the input. Every thread receives the aggregate value of the masked input sequence, excluding
            ``init`` and any callback prefix.
        """
        return cls._scan(
            value,
            op,
            storage,
            inclusive=True,
            init=init,
            identity=identity,
            valid_items=valid_items,
            prefix_callback=prefix_callback,
        )

    @classmethod
    def exclusive_with_aggregate(
        cls,
        value,
        op,
        *,
        storage,
        init=None,
        identity=None,
        valid_items: int | Integer | None = None,
        prefix_callback=None,
    ):
        """Compute exclusive prefixes, excluding each output item's own input.

        The first flattened output is the seed or identity when available;
        otherwise only that first output is unspecified.
        All threads must participate and synchronize before reusing storage.

        Args:
            value: One element or a nonempty list/tuple/Vector of consecutive
                elements, converted to the specialized dtype. Tile lengths must
                match across threads; items are ordered by thread, then item index.
                A list/tuple of vector elements keeps each vector intact.
            op: Associative ReductionOp or binary callable returning the element
                dtype. Values that need a custom operation require a callable. Operand order is preserved
                and commutativity is not required.
            storage: Shared instance of the specialization's ``SharedStorage``.
            init: Optional single value of the element dtype, applied as the
                left operand before the entire input sequence. Mutually exclusive
                with ``prefix_callback``.
            identity: Optional neutral element or callable ``identity(dtype)``.
                Overrides a built-in identity or ``op.identity``. Required from
                one of these sources when ``valid_items`` is supplied.
            valid_items: Uniform number of leading contributing threads in
                ``[0, block_threads]``, or ``None`` for all inputs. Only the
                single-item overload accepts this argument; omit it for an
                item range. Runtime bounds are the caller's responsibility.
            prefix_callback: Optional callable receiving the input aggregate.
                Runs in the first physical warp's active lanes for WARP_SCANS,
                or its raking-thread subset for raking policies. Only lane zero's
                returned prefix seeds the scan. Pointer-backed state may persist
                between tiles; the aggregate excludes the returned prefix.

        Returns:
            A pair ``(prefixes, aggregate)``. Prefixes have the same per-thread
            shape as the input. Every thread receives the aggregate value of the masked input sequence, excluding
            ``init`` and any callback prefix.
        """
        return cls._scan(
            value,
            op,
            storage,
            inclusive=False,
            init=init,
            identity=identity,
            valid_items=valid_items,
            prefix_callback=prefix_callback,
        )

    @classmethod
    def _scan(cls, value, op, storage, *, inclusive, init=None, identity=None, valid_items=None, prefix_callback=None):
        if cls.block_threads is None:
            raise TypeError("specialize first, e.g. BlockScan[fx.Float32, 256]")
        if prefix_callback is not None and init is not None:
            raise ValueError("init and prefix_callback are mutually exclusive")
        _validate_scalar_valid_items(value, valid_items, cls.block_threads, cls.dtype)
        value = _convert_value(value, cls.dtype)
        if init is not None:
            init = _cast_value(cls.dtype, init)
        tid = _linear_thread_id(cls.block_size)
        items = _as_items(value, cls.dtype)
        if (
            cls.algorithm is BlockScanAlgorithm.WARP_SCANS
            and isinstance(op, ReductionOp)
            and valid_items is None
            and identity is None
            and prefix_callback is None
        ):
            # The old numeric/default contract needs neither callback work nor
            # the boundary selects required by semigroups without an identity.
            scanned = [items[0]]
            for item in items[1:]:
                scanned.append(_combine(op, scanned[-1], item))
            prefix, aggregate = _prefix_warp_scans_full(
                scanned[-1],
                tid,
                storage.slots,
                op,
                cls.warp_ops.warp_scan_with_aggregate,
                cls.warp_threads,
                cls.num_warps,
            )
            prefix = _seed(prefix, op, init)
            heads = scanned if inclusive else scanned[:-1]
            output = [_combine(op, prefix, head) for head in heads]
            if not inclusive:
                output.insert(0, prefix)
            result = _from_items(output, like=value) if _is_items(value, cls.dtype) else output[0]
            return result, aggregate
        neutral = _optional_identity(op, cls.dtype, identity)
        if valid_items is not None:
            neutral = _identity(op, cls.dtype, identity)
            items = (_select_value(tid < valid_items, value, neutral),)
        scanned = [items[0]]
        for item in items[1:]:
            scanned.append(_combine(op, scanned[-1], item))
        prefix, aggregate = cls._block_prefix(scanned[-1], op, storage)
        if prefix_callback is not None:
            callback_threads = min(cls.block_threads, num_warp_threads())
            if cls.algorithm is not BlockScanAlgorithm.WARP_SCANS:
                segment = (cls.block_threads + callback_threads - 1) // callback_threads
                callback_threads = (cls.block_threads + segment - 1) // segment
            init = _callback_prefix(aggregate, tid, storage, prefix_callback, callback_threads)
        if init is not None:
            prefix = _select_value(tid > 0, _combine(op, init, prefix), _cast_value(cls.dtype, init))
        elif neutral is not None:
            prefix = _select_value(tid > 0, prefix, neutral)
        heads = scanned if inclusive else scanned[:-1]
        output = []
        for head in heads:
            joined = _combine(op, prefix, head)
            if init is None and neutral is None:
                joined = _select_value(tid > 0, joined, head)
            output.append(joined)
        if not inclusive:
            output.insert(0, prefix)
        result = _from_items(output, like=value) if _is_items(value, cls.dtype) else output[0]
        return result, aggregate

    @classmethod
    def _block_prefix(cls, partial, op, storage):
        tid = _linear_thread_id(cls.block_size)
        if cls.algorithm is BlockScanAlgorithm.WARP_SCANS:
            return _prefix_warp_scans(
                partial, tid, storage, op, cls.warp_ops.warp_inclusive_scan, cls.warp_threads, cls.block_threads
            )
        return _prefix_raking(
            partial,
            tid,
            storage,
            op,
            cls.warp_ops.warp_inclusive_scan,
            cls.warp_threads,
            cls.block_threads,
            cls.algorithm is BlockScanAlgorithm.RAKING_MEMOIZE,
        )
