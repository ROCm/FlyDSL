# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Stable, shared-memory merge sort of blocked keys and value payloads."""

import enum

from ...._mlir.dialects import gpu
from ....compiler import jit
from ....compiler.protocol import dsl_size_of
from ....expr.gpu import barrier
from ....expr.numeric import Int32, Integer, Uint16
from ....expr.primitive import const_expr, range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array, Vector
from .._common import _linear_thread_id, _validate_valid_items
from .._sorting import _local_sort, _pack, _prepare, _unpack, _valid_before
from .._values import (
    _item_dtype,
    _record_cast,
    _record_default,
    _record_select,
    _shared_array,
    _shared_load,
    _shared_store,
)
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta

__all__ = [
    "BlockMergeSortAlgorithm",
    "BlockMergeSort",
]


class BlockMergeSortAlgorithm(enum.Enum):
    """Execution policy for :class:`BlockMergeSort`.

    Attributes:
        MERGE: Stably sort local runs and merge them through shared memory.
            This is the default and only policy.
    """

    MERGE = "merge"

    Merge = MERGE


def _storage(dtype, threads, warp, count):
    key_dtype, value_dtype = dtype if isinstance(dtype, tuple) else (dtype, None)
    size = threads * count
    # The threshold still counts the 4-byte validity word every item once kept
    # in shared memory, so exactly the tiles that were compact stay compact.
    footprint = (
        dsl_size_of(_shared_array(key_dtype, size))
        + size * 4
        + (dsl_size_of(_shared_array(value_dtype, size)) if value_dtype else 0)
    )
    compact = footprint > 60 * 1024
    slots = threads if compact else size
    return Struct[
        "slots" : _shared_array(key_dtype, slots),
        "values" : _shared_array(value_dtype or key_dtype, slots if value_dtype else 1),
        "permutation" : Array[Uint16 if size <= 65536 else Int32, size if compact else 1],
    ]


def _valid_at(position, valid_items):
    """Whether tile position *position* holds a valid entry.

    Invalid entries order after every valid one without consulting the
    comparator, and the valid entries start as the blocked prefix
    ``[0, valid_items)``. Every run therefore keeps its valid entries in
    front, so after the local sort and after every merge pass the valid set is
    that same prefix: validity is a function of position and never has to be
    stored. Without a count every entry is valid and the checks fold away.
    """
    return Int32(1) if valid_items is None else Int32(position < valid_items)


def _scope(run, warp, count):
    """Merging runs of *run* items stays inside one logical warp's tile when ``2 * run <= warp * count``."""
    return gpu.BarrierScope.Subgroup if 2 * run <= warp * count else gpu.BarrierScope.Workgroup


@jit
def _stage_column(key, value, storage, tid):
    _shared_store(storage.slots, tid, key)
    if const_expr(value is not None):
        _shared_store(storage.values, tid, value)
    barrier()


def _gather_columns(items, payload, queries, storage, tid):
    """Gather register tiles with one shared slot per thread, not per item."""
    count = len(items)
    keys = [_record_default(_item_dtype(items[0]))] * count
    values = [_record_default(_item_dtype(payload[0]))] * count if payload is not None else None
    for item in range(count):
        _stage_column(items[item], payload[item] if payload is not None else None, storage, tid)
        for q, query in enumerate(queries):
            source, selected = query // count, (query % count) == item
            keys[q] = _record_select(
                selected, _record_cast(_shared_load(storage.slots, source), _item_dtype(items[0])), keys[q]
            )
            if payload is not None:
                values[q] = _record_select(
                    selected, _record_cast(_shared_load(storage.values, source), _item_dtype(payload[0])), values[q]
                )
        barrier()
    return keys, values


@jit
def _publish_permutation(ranks, storage, tid, count):
    for i in range_constexpr(count):
        storage.permutation[ranks[i]] = storage.permutation.dtype(tid * count + i)
    barrier()


def _compact_merge(items, payload, valid, storage, tid, threads, descending, compare_op, valid_items):
    count, total = len(items), threads * len(items)
    run = count
    indices = [tid * count + i for i in range(count)]
    while run < total:
        right = [((index // run) % 2) != 0 for index in indices]
        base = [(index // (2 * run)) * (2 * run) for index in indices]
        other = [b + r.select(Int32(0), Int32(run)) for b, r in zip(base, right)]
        sizes = [((total - o) < run).select(total - o, Int32(run)) for o in other]
        sizes = [(size > 0).select(size, Int32(0)) for size in sizes]
        low, high = [Int32(0)] * count, sizes
        for step in range(run.bit_length()):
            mids = [(lo + hi) // 2 for lo, hi in zip(low, high)]
            addresses = [o + (mid < size).select(mid, size - 1) for o, mid, size in zip(other, mids, sizes)]
            addresses = [((a >= 0) & (a < total)).select(a, Int32(0)) for a in addresses]
            candidates, _ = _gather_columns(items, None, addresses, storage, tid)
            for i in range(count):
                candidate_valid = _valid_at(addresses[i], valid_items)
                before = _valid_before(candidates[i], items[i], candidate_valid, valid[i], descending, compare_op)
                equal = ~before & ~_valid_before(
                    items[i], candidates[i], valid[i], candidate_valid, descending, compare_op
                )
                advance = (mids[i] < sizes[i]) & (before | (equal & right[i]))
                low[i] = advance.select(mids[i] + 1, low[i])
                high[i] = advance.select(high[i], mids[i])
        ranks = Vector.from_elements([b + (index % run) + lo for b, index, lo in zip(base, indices, low)])
        _publish_permutation(ranks, storage, tid, count)
        queries = [Int32(_shared_load(storage.permutation, index)) for index in indices]
        barrier()
        items, payload = _gather_columns(items, payload, queries, storage, tid)
        run *= 2
    return items, payload


@jit
def _stage(keys, values, storage, tid, count, scope):
    for i in range_constexpr(count):
        _shared_store(storage.slots, tid * count + i, keys[i])
        if const_expr(values is not None):
            _shared_store(storage.values, tid * count + i, values[i])
    barrier(scope=scope)


@jit
def _scatter(keys, values, ranks, storage, count, scope):
    for i in range_constexpr(count):
        _shared_store(storage.slots, ranks[i], keys[i])
        if const_expr(values is not None):
            _shared_store(storage.values, ranks[i], values[i])
    barrier(scope=scope)


@jit
def _merge_rank(key, valid, storage, index, run, total, descending, compare_op, valid_items):
    right = ((index // run) % 2) != 0
    base = (index // (2 * run)) * (2 * run)
    other = base + right.select(Int32(0), Int32(run))
    size = ((total - other) < run).select(total - other, Int32(run))
    size = (size > 0).select(size, Int32(0))
    low, high = Int32(0), size
    for step in range_constexpr(run.bit_length()):
        mid = (low + high) // 2
        address = other + (mid < size).select(mid, size - 1)
        address = ((address >= 0) & (address < total)).select(address, Int32(0))
        candidate = _record_cast(_shared_load(storage.slots, address), _item_dtype(key))
        candidate_valid = _valid_at(address, valid_items)
        before = _valid_before(candidate, key, candidate_valid, valid, descending, compare_op)
        equivalent = ~before & ~_valid_before(key, candidate, valid, candidate_valid, descending, compare_op)
        advance = (mid < size) & (before | (equivalent & right))
        low = advance.select(mid + 1, low)
        high = advance.select(high, mid)
    return base + (index % run) + low


def _merge_rounds(items, payload, storage, tid, run, threads, warp, valid_items, descending, compare_op):
    """Merge sorted runs of *run* items, pass by pass, until the tile is one run.

    The tile lives in shared memory in blocked order between passes. Every
    pass ranks each item in the run it merges with, scatters it to that rank
    and reads its blocked slots back, so a pass needs only two barriers: one
    after its reads, before the scatter overwrites them, and one after the
    scatter. A pass whose merged runs fit one logical warp's tile only
    synchronizes that warp.
    """
    count = len(items)
    total = threads * count
    if run >= total:
        return items, payload
    key_dtype = _item_dtype(items[0])
    value_dtype = _item_dtype(payload[0]) if payload is not None else None
    indices = [tid * count + i for i in range(count)]
    valid = [_valid_at(index, valid_items) for index in indices]
    _stage(items, payload, storage, tid, count, _scope(run, warp, count))
    while run < total:
        ranks = [
            _merge_rank(item, valid[i], storage, indices[i], run, total, descending, compare_op, valid_items)
            for i, item in enumerate(items)
        ]
        barrier(scope=_scope(run, warp, count))
        _scatter(items, payload, ranks, storage, count, _scope(2 * run if 2 * run < total else run, warp, count))
        items = [_record_cast(_shared_load(storage.slots, index), key_dtype) for index in indices]
        if payload is not None:
            payload = [_record_cast(_shared_load(storage.values, index), value_dtype) for index in indices]
        run *= 2
    return items, payload


class _BlockMergeSortMeta(BlockTileAlgorithmMeta):
    _supports_pairs = True
    _algorithms = BlockMergeSortAlgorithm
    _shared_storage = {BlockMergeSortAlgorithm.MERGE: _storage}

    def _default_algorithm_for(cls, target):
        return BlockMergeSortAlgorithm.MERGE

    def __call__(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        compare_op=None,
        valid_items: int | Integer | None = None,
        oob_default=None,
    ):
        """Implement :meth:`BlockMergeSort.__call__`."""
        cls._check(keys)
        items, payload, vector = _unpack(keys, values, cls.value_dtype, cls.dtype)
        if payload is not None and cls.value_dtype is None:
            raise TypeError("specialize [(key_dtype, value_dtype), block_size, items_per_thread] for pairs")
        count = cls.items_per_thread
        _validate_valid_items(valid_items, cls.block_threads * count)
        if isinstance(valid_items, int) and valid_items == cls.block_threads * count:
            valid_items = None  # every slot is valid: let the validity checks fold away
        tid = _linear_thread_id(cls.block_size)
        items, payload, valid = _prepare(
            items, payload, [tid * count + i for i in range(count)], valid_items, oob_default
        )
        if storage.permutation.size > 1:
            items, payload, valid = _local_sort(items, payload, valid, descending, compare_op)
            items, payload = _compact_merge(
                items, payload, valid, storage, tid, cls.block_threads, descending, compare_op, valid_items
            )
            return _pack(items, payload, vector, keys, values)
        items, payload, run = cls._sort_runs(items, payload, valid, tid, storage, descending, compare_op, valid_items)
        items, payload = _merge_rounds(
            items,
            payload,
            storage,
            tid,
            run,
            cls.block_threads,
            cls.warp_threads,
            valid_items,
            descending,
            compare_op,
        )
        if cls.block_threads > 1:
            barrier()  # the storage is free again on return, as after every merge pass
        return _pack(items, payload, vector, keys, values)


class BlockMergeSort(BlockPrimitive, metaclass=_BlockMergeSortMeta):
    """Stably sort a blocked tile of keys and optional value payloads.

    Specialize with ``BlockMergeSort[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockMergeSort[...]``, in
    bracket order.

    Args:
        dtype: Required key element type for keys only, or (key_dtype, value_dtype) for
            key/value pairs. Each type describes one complete element, including Vector or
            Struct elements; payload types are independent of key types. Supply compare_op
            at call time when keys require a custom ordering.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        items_per_thread: Required positive Python int giving the number of complete items
            owned by each thread. The block tile contains block_threads * items_per_thread
            items. This count excludes element components/fields and need not be a power of
            two. Pair payloads have the same per-thread count as keys.
        algorithm: Optional member of BlockMergeSortAlgorithm selecting the implementation
            policy. Defaults to ``BlockMergeSortAlgorithm.MERGE`` when omitted. It fixes the
            specialization and its SharedStorage layout.

    Input and output use blocked order: item ``i`` of linear thread ``t`` has
    flattened position ``t * items_per_thread + i``. Equivalent valid keys
    keep that original order, including when sorting descending. Numeric
    ordering uses ``<`` or ``>``: signed floating zeros are equivalent, and
    NaNs have no special total-order treatment. Supply an appropriate strict
    comparator when NaNs must be ordered. Invalid entries sort after every
    valid key in both directions.

    All block threads must participate with uniform options and one shared
    allocation of the specialization's ``SharedStorage``. Synchronize the
    block before reusing that allocation for another collective. Large tiles
    automatically use a compact shared-memory merge representation.

    Attributes:
        dtype: Specialized key element type.
        value_dtype: Specialized payload type, or ``None`` for keys only.
        block_size: Specialized ``(x, y, z)`` block shape.
        block_threads: Total number of participating threads.
        items_per_thread: Number of keys supplied by each thread.
        algorithm: Selected merge policy.
        SharedStorage: Shared-memory Struct type required by the collective.

    Examples:
        # T0..T3 supply keys={[3,1], [2,1], [4,2], [3,0]}; T4..T63 supply [99,99].
        # Thread t supplies payload values=[2*t, 2*t+1], the original blocked positions.
        P = fx.coop.BlockMergeSort[(fx.Int32, fx.Int32), 64, 2, fx.coop.BlockMergeSortAlgorithm.MERGE]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        sorted_keys, sorted_values = P(keys, values, storage=storage)
        # T0..T3 keys: {[0,1], [1,2], [2,3], [3,4]}; values: {[7,1], [3,2], [5,0], [6,4]}.
        # T4..T63 keep their [99,99] keys and original payloads, since equal keys are stable.
        fx.barrier()
        partial_keys, partial_values = P(
            keys, values, storage=storage, descending=True, compare_op=lambda a, b: a < b,
            valid_items=6, oob_default=-1,
        )
        # The first six flattened keys are [4,3,2,2,1,1], with payloads [4,0,2,5,1,3].
        # All remaining keys are -1 and payloads zero, after the valid prefix.
        fx.barrier()
        partial_keys, partial_values = P.sort(
            keys, values, storage=storage, descending=True, compare_op=lambda a, b: a < b,
            valid_items=6, oob_default=-1,
        )
        fx.barrier()
        partial_keys, partial_values = P.stable_sort(
            keys, values, storage=storage, descending=True, compare_op=lambda a, b: a < b,
            valid_items=6, oob_default=-1,
        )
        # Named entry points return the same stable pairs.
    """

    @classmethod
    def _sort_runs(cls, items, payload, valid, tid, storage, descending, compare_op, valid_items):
        """Sort the tile into runs the shared-memory passes merge; returns ``(items, payload, run)``.

        Portably each thread sorts its own items, one run per thread. A target
        may sort longer runs in registers, such as a whole warp's tile.
        """
        items, payload, valid = _local_sort(items, payload, valid, descending, compare_op)
        return items, payload, cls.items_per_thread

    @classmethod
    def __call__(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        compare_op=None,
        valid_items: int | Integer | None = None,
        oob_default=None,
    ):
        """Stably sort keys, carrying optional values to the same blocked positions.

        All threads must call this collective with uniform options. Equivalent
        keys preserve flattened blocked input order. Synchronize before reusing
        ``storage``; see :class:`BlockMergeSort` for the ordering contract.

        Args:
            keys: Key value matching the specialized dtype and item count.
            values: Optional payload with the same item count and single-item/sequence
                form as ``keys``; its element shape may differ. Must match the
                specialized ``value_dtype``. Requires pair specialization.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean reversing the key comparison.
            compare_op: Optional strict weak ordering ``compare_op(a, b)``.
                Returns whether ``a`` precedes ``b`` in ascending order; required
                for record keys. Descending order reverses its arguments.
            valid_items: Uniform Python integer or runtime scalar counting valid
                leading items across the entire flattened block tile. Must be in
                ``[0, block_threads * items_per_thread]``; ``None`` means all items.
                Runtime bounds are the caller's responsibility.
            oob_default: Optional fill for invalid keys before sorting. The fill value must be compatible with the
                key dtype.
                When supplied, invalid payloads are zero-filled. Otherwise invalid
                input keys and payloads are retained and moved to the tail.

        Returns:
            Sorted keys, or ``(keys, values)`` if payloads were supplied. Results retain the item count and complete element types. Only the first ``valid_items`` flattened outputs are valid.

        Raises:
            TypeError: The operator must be specialized with ``BlockMergeSort[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(
                f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, items_per_thread]"
            )
        return type(cls).__call__(
            cls,
            keys,
            values,
            storage=storage,
            descending=descending,
            compare_op=compare_op,
            valid_items=valid_items,
            oob_default=oob_default,
        )

    @classmethod
    def sort(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        compare_op=None,
        valid_items: int | Integer | None = None,
        oob_default=None,
    ):
        """Stably sort a blocked tile through the class-call interface.

        Equivalent keys retain their original blocked order in both directions.
        All block threads must participate and synchronize before reusing storage.

        Args:
            keys: Key value matching the specialization.
            values: Optional payload matching the key item count and single-item/sequence
                form, with the specialized payload dtype; requires pair specialization.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean reversing the key comparison.
            compare_op: Optional strict weak ordering for keys; see ``__call__``.
            valid_items: Number of leading valid items in the whole blocked tile.
            oob_default: Optional fill value for invalid keys; see ``__call__``.

        Returns:
            Sorted keys, or ``(keys, values)`` for pairs. Results retain the item count and complete element types.
        """
        return cls(
            keys,
            values,
            storage=storage,
            descending=descending,
            compare_op=compare_op,
            valid_items=valid_items,
            oob_default=oob_default,
        )

    @classmethod
    def stable_sort(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        compare_op=None,
        valid_items: int | Integer | None = None,
        oob_default=None,
    ):
        """Stably sort a blocked tile through the class-call interface.

        Equivalent keys retain their original blocked order in both directions.
        All block threads must participate and synchronize before reusing storage.

        Args:
            keys: Key value matching the specialization.
            values: Optional payload matching the key item count and single-item/sequence
                form, with the specialized payload dtype; requires pair specialization.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean reversing the key comparison.
            compare_op: Optional strict weak ordering for keys; see ``__call__``.
            valid_items: Number of leading valid items in the whole blocked tile.
            oob_default: Optional fill value for invalid keys; see ``__call__``.

        Returns:
            Sorted keys, or ``(keys, values)`` for pairs. Results retain the item count and complete element types.
        """
        return cls(
            keys,
            values,
            storage=storage,
            descending=descending,
            compare_op=compare_op,
            valid_items=valid_items,
            oob_default=oob_default,
        )
