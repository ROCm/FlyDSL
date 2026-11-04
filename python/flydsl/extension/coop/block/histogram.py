# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Atomic and sort-based shared-memory histograms."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.llvm import atomic_add
from ....expr.numeric import Float32, Int32, Numeric
from ....expr.primitive import range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array, Vector
from .._common import _linear_thread_id
from .._values import _as_items, _items_dtype
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta
from .merge_sort import BlockMergeSort

__all__ = [
    "BlockHistogramAlgorithm",
    "BlockHistogram",
]


class BlockHistogramAlgorithm(enum.Enum):
    """Accumulation policies for shared-memory block histograms.

    Attributes:
        ATOMIC: Increment shared counters with an atomic addition per valid key.
        SORT: Sort one per-thread key column at a time and count equal-key runs
            with one writer per bin, without atomic counter updates.
    """

    ATOMIC = "atomic"
    SORT = "sort"

    Atomic = ATOMIC
    Sort = SORT


@jit
def _initialize(storage, tid, threads, bins, counter):
    for i in range_constexpr((bins + threads - 1) // threads):
        index = tid + i * threads
        if index < bins:
            storage.counts[index] = counter(0)
    barrier()


def _bin_index(item, bins):
    """Convert one key to its Int32 bin, or -1 when the converted index is out of range."""
    dtype = item.dtype
    if dtype.is_float:
        # trunc(x) is in [0, bins) exactly when -1 < x < bins; NaN fails both.
        wide = item if dtype.width >= 32 else item.to(Float32)
        valid = (wide > -1.0) & (wide < float(bins))
    elif dtype.width > 32:
        valid = (item >= 0) & (item < bins)
    else:
        # Narrower integers convert exactly or wrap to a negative index.
        return item.to(Int32)
    return valid.select(item.to(Int32), Int32(-1))


@jit
def _accumulate(items, storage, count, bins, counter):
    for i in range_constexpr(count):
        key = _bin_index(items[i], bins)
        if (key >= 0) & (key < bins):
            atomic_add(storage.counts.ptr + key, counter(1))
    barrier()


@jit
def _publish_sorted(key, storage, tid):
    storage.sorted[tid] = key
    barrier()


@jit
def _bound(keys, key, size, upper):
    low, high = Int32(0), Int32(size)
    for step in range_constexpr(size.bit_length()):
        mid = (low + high) // 2
        candidate = keys[(mid < size).select(mid, Int32(size - 1))]
        before = (candidate <= key) if upper else (candidate < key)
        advance = (mid < size) & before
        low = advance.select(mid + 1, low)
        high = advance.select(high, mid)
    return low


@jit
def _count_sorted(storage, tid, threads, bins, counter):
    # Each bin has one writer. Equal sorted keys form a contiguous run, so
    # its length can be accumulated without atomic updates.
    for i in range_constexpr((bins + threads - 1) // threads):
        index = tid + i * threads
        if index < bins:
            count = _bound(storage.sorted, index, threads, True) - _bound(storage.sorted, index, threads, False)
            storage.counts[index] = storage.counts[index].to(counter) + count.to(counter)
    barrier()


def _counter(dtype):
    item, counter = dtype if isinstance(dtype, tuple) else (dtype, Int32)
    if not isinstance(item, type) or not issubclass(item, Numeric):
        raise TypeError("histogram input elements must be numeric scalar bin values")
    if not isinstance(counter, type) or not issubclass(counter, Numeric):
        raise TypeError("histogram counter type must be a numeric scalar")
    return counter


def _storage(dtype, threads, warp, bins):
    return Struct["counts" : Array[_counter(dtype), bins]]


def _sort_storage(dtype, threads, warp, bins):
    return Struct[
        "counts" : Array[_counter(dtype), bins],
        "sorted" : Array[Int32, threads],
        "sort" : BlockMergeSort[Int32, threads, 1].SharedStorage,
    ]


class _BlockHistogramMeta(BlockTileAlgorithmMeta):
    _supports_pairs = True
    _parameter_name = "bins"
    _algorithms = BlockHistogramAlgorithm
    _shared_storage = {BlockHistogramAlgorithm.ATOMIC: _storage, BlockHistogramAlgorithm.SORT: _sort_storage}

    def _default_algorithm_for(cls, target):
        return BlockHistogramAlgorithm.ATOMIC

    def __call__(cls, value, *, storage):
        """Implement :meth:`BlockHistogram.__call__`."""
        cls.init(storage=storage)
        return cls.composite(value, storage=storage)


class BlockHistogram(BlockPrimitive, metaclass=_BlockHistogramMeta):
    """Build or accumulate a histogram in block-shared memory.

    Specialize with ``BlockHistogram[dtype, block_size, bins, algorithm]``. The parameters
    below are compile-time positional arguments to ``BlockHistogram[...]``, in bracket
    order.

    Args:
        dtype: Required numeric scalar input type, or (input_dtype, counter_dtype) to select
            a numeric scalar counter type. A single input type uses Int32 counters. Inputs
            convert to Int32 bin indices. With ATOMIC, counter_dtype must support shared
            atomic addition on the target.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        bins: Required positive Python int giving the number of histogram bins and shared
            counters. Valid bin indices are [0, bins). This is independent of the per-thread
            input count, which is inferred from each call and must be uniform across the
            block.
        algorithm: Optional member of BlockHistogramAlgorithm selecting the implementation
            policy. Defaults to ``BlockHistogramAlgorithm.ATOMIC`` when omitted. It fixes
            the specialization and its SharedStorage layout.

    Keys are converted to integer bin indices. Out-of-range indices
    are ignored. ``init`` clears the bins; ``composite`` adds to existing bins.
    SORT sorts one column at a time and counts equal-key runs without atomics,
    keeping storage bounded for arbitrary per-thread item counts.
    All block threads must participate with equal per-thread item counts.
    Initialization and accumulation finish with block barriers. Synchronize
    consumers before clearing or repurposing the returned shared counts.

    Attributes:
        dtype: Specialized numeric input key type.
        value_dtype: Explicit counter type, or ``None`` for the ``Int32`` default.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        bins: Number of shared counters, independent of the per-thread item count.
        algorithm: Selected ``BlockHistogramAlgorithm`` policy.
        SharedStorage: Shared-memory struct type containing ``counts``; the
            sort policy also includes one key per thread and sorting scratch.

    Examples:
        # T0..T3 supply x={[0,1], [1,3], [0,1], [2,3]}; T4..T63 supply [0,0].
        P = fx.coop.BlockHistogram[fx.Int32, 64, 4, fx.coop.BlockHistogramAlgorithm.ATOMIC]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        counts = P(x, storage=storage)
        # All threads see the same shared counts: [122,3,1,2].
        fx.barrier()
        P.init(storage=storage)
        P.composite(x, storage=storage)
        fx.barrier()
        P.composite(x, storage=storage)
        fx.barrier()
        # Explicit initialization followed by two accumulations gives [244,6,2,4].
    """

    @classmethod
    def __call__(cls, value, *, storage):
        """Clear the histogram and accumulate one block's input items.

        Args:
            value: Per-thread input value of the specialized dtype. Keys are converted to ``Int32`` bin indices, and
                indices outside ``[0, bins)`` are ignored.
            storage: Shared instance of ``SharedStorage`` used by the block.

        Returns:
            The shared ``storage.counts`` array, containing the completed
            histogram and visible to every participating block thread.

        Raises:
            TypeError: The operator must be specialized with ``BlockHistogram[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, bins]")
        return type(cls).__call__(cls, value, storage=storage)

    @classmethod
    def init(cls, *, storage) -> "None":
        """Collectively zero every histogram bin and synchronize the block.

        Args:
            storage: Shared instance of ``SharedStorage`` whose ``counts``
                array is initialized using the specialized counter dtype.

        Returns:
            None. Every shared bin contains zero on return.
        """
        cls._check()
        _initialize(storage, _linear_thread_id(cls.block_size), cls.block_threads, cls.bins, cls.value_dtype or Int32)

    @classmethod
    def composite(cls, value, *, storage):
        """Add input keys to an initialized histogram and synchronize.

        Args:
            value: Scalar or list/tuple/Vector of keys of the specialized dtype,
                contributing to one histogram. Each item is converted to an ``Int32`` index;
                indices outside ``[0, bins)`` contribute nothing.
            storage: Shared instance of ``SharedStorage`` with existing counters,
                typically initialized by ``init`` or a prior histogram call.

        Returns:
            The shared ``storage.counts`` array after accumulation, using the
            explicit counter dtype or ``Int32`` by default.
        """
        cls._check()
        if _items_dtype(value, cls.dtype) is not cls.dtype:
            raise TypeError("histogram keys must have the specialized dtype")
        items = Vector.from_elements(_as_items(value, cls.dtype))
        counter = cls.value_dtype or Int32
        if cls.algorithm is BlockHistogramAlgorithm.ATOMIC:
            _accumulate(items, storage, items.numel, cls.bins, counter)
        else:
            tid = _linear_thread_id(cls.block_size)
            sort = BlockMergeSort[Int32, cls.block_size, 1]
            for item in items:
                key = sort(_bin_index(item, cls.bins), storage=storage.sort)
                _publish_sorted(key, storage, tid)
                _count_sorted(storage, tid, cls.block_threads, cls.bins, counter)
        return storage.counts
