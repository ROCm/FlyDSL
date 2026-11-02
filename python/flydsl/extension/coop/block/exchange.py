# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide blocked, striped, warp-striped and ranked exchanges."""

import enum

from ....compiler import jit
from ....expr.arith import min as arith_min
from ....expr.gpu import barrier
from ....expr.primitive import range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Vector
from .._common import _linear_thread_id
from .._values import _as_items, _from_items, _is_items, _record_default, _shared_array, _shared_load, _shared_store
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta

__all__ = [
    "BlockExchangeAlgorithm",
    "BlockExchange",
]


class BlockExchangeAlgorithm(enum.Enum):
    """Shared-memory policies for exchanging a block's item ownership.

    Attributes:
        SHARED_MEMORY: Stage the complete tile in shared memory.
        WARP_TIME_SLICING: Stage one logical-warp-sized tile window at a time;
            all block threads participate in every window.
    """

    SHARED_MEMORY = "shared_memory"
    WARP_TIME_SLICING = "warp_time_slicing"

    SharedMemory = SHARED_MEMORY
    WarpTimeSlicing = WARP_TIME_SLICING


def _storage(dtype, threads, warp, count):
    return Struct["slots" : _shared_array(dtype, threads * count)]


def _timesliced_storage(dtype, threads, warp, count):
    return Struct["slots" : _shared_array(dtype, warp * count)]


def _warp_layout(tid, threads, warp, count):
    """Return indexing components for a complete logical warp.

    Args:
        tid: Linear thread index in the block.
        threads: Total number of block threads.
        warp: Logical warp width.
        count: Number of items owned by each thread.

    Returns:
        ``(base, lane, extent)``, where ``base`` is the first item index of
        the warp's tile, ``lane`` is the thread's local lane, and ``extent``
        is the number of participating threads in this warp.
    """
    first = (tid // warp) * warp
    return first * count, tid % warp, arith_min(threads - first, warp)


@jit
def _write_window(value, ranks, flags, slots, count, first, extent):
    for i in range_constexpr(count):
        if flags[i] != 0:
            if (ranks[i] >= first) & (ranks[i] < first + extent):
                _shared_store(slots, ranks[i] - first, value[i])


@jit
def _read_window(slots, index, first, extent, previous):
    result = previous
    if (index >= first) & (index < first + extent):
        result = _shared_load(slots, index - first)
    return result


@jit
def _clear_window(slots, tid, threads, count, extent, default):
    for i in range_constexpr(count):
        index = tid + i * threads
        if index < extent:
            _shared_store(slots, index, default)
    barrier()


class _BlockExchangeMeta(BlockTileAlgorithmMeta):
    _algorithms = BlockExchangeAlgorithm
    _shared_storage = {
        BlockExchangeAlgorithm.SHARED_MEMORY: _storage,
        BlockExchangeAlgorithm.WARP_TIME_SLICING: _timesliced_storage,
    }

    def _default_algorithm_for(cls, target):
        return BlockExchangeAlgorithm.SHARED_MEMORY


class BlockExchange(BlockPrimitive, metaclass=_BlockExchangeMeta):
    """Convert blocked, striped, warp-striped, and ranked item ownership.

    Specialize with ``BlockExchange[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockExchange[...]``, in
    bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        items_per_thread: Required positive Python int giving the number of complete items
            owned by each thread. The block tile contains block_threads * items_per_thread
            items. This count excludes element components/fields and need not be a power of
            two.
        algorithm: Optional member of BlockExchangeAlgorithm selecting the implementation
            policy. Defaults to ``BlockExchangeAlgorithm.SHARED_MEMORY`` when omitted. It
            fixes the specialization and its SharedStorage layout.

    For linear thread ``t`` and local
    item ``i``, blocked order uses ``t * items_per_thread + i`` and striped
    order uses ``t + i * block_threads``. Warp-striped order applies stripes
    within each logical warp, using the specialization's logical warp width.
    All methods return the values owned by the calling thread.
    WARP_TIME_SLICING exchanges one warp-sized
    tile window at a time, reducing LDS from block_threads * items to
    warp_threads * items. All block threads participate in every window.
    Non-guarded scatter ranks must be a permutation. Guarded/flagged scatters
    ignore invalid inputs and fill holes with ``default`` (zero by default);
    remaining ranks must be distinct. Synchronize before reusing storage.

    Attributes:
        dtype: Specialized value type.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        items_per_thread: Positive compile-time number of items per thread.
        algorithm: Selected ``BlockExchangeAlgorithm`` policy.
        warp_threads: Logical warp width used for warp-striped layouts.
        SharedStorage: Shared-memory struct with a ``slots`` array containing
            the full tile or one warp-sized window, according to the policy.

    Examples:
        # Thread t supplies x=[2*t, 2*t+1], and ranks=[127-2*t, 126-2*t].
        P = fx.coop.BlockExchange[fx.Int32, 64, 2, fx.coop.BlockExchangeAlgorithm.SHARED_MEMORY]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y = P.blocked_to_striped(x, storage=storage)
        # Thread t receives [t, t+64].
        fx.barrier()
        restored = P.striped_to_blocked(y, storage=storage)
        fx.barrier()
        warp_striped = P.blocked_to_warp_striped(x, storage=storage)
        fx.barrier()
        restored = P.warp_striped_to_blocked(warp_striped, storage=storage)
        # Both round trips recover x. Warp stripes use the target's physical warp width.
        fx.barrier()
        reversed_blocked = P.scatter_to_blocked(x, ranks, storage=storage)
        fx.barrier()
        reversed_striped = P.scatter_to_striped(x, ranks, storage=storage)
        # Thread t receives [127-2*t,126-2*t] and [127-t,63-t], respectively.

        # guarded_ranks equals ranks except T0's first rank is -1.
        # flags is [1,1] everywhere except T0, which supplies [0,1].
        fx.barrier()
        guarded = P.scatter_to_striped_guarded(x, guarded_ranks, storage=storage, default=-1)
        fx.barrier()
        flagged = P.scatter_to_striped_flagged(x, ranks, flags, storage=storage, default=-1)
        # Both equal reversed_striped except T63's second slot is -1 instead of 0.
    """

    @classmethod
    def _exchange(cls, value, destinations, sources, storage, flags=None, default=0):
        cls._check(value)
        count = cls.items_per_thread
        template = value
        value = _from_items(_as_items(value, cls.dtype))
        tid = _linear_thread_id(cls.block_size)
        guarded = flags is not None
        flags = flags if guarded else Vector.filled(count, 1, destinations.dtype)
        window = (
            cls.warp_threads * count
            if cls.algorithm is BlockExchangeAlgorithm.WARP_TIME_SLICING
            else cls.block_threads * count
        )
        out = [_record_default(cls.dtype, default) for _ in range(count)]
        for first in range(0, cls.block_threads * count, window):
            extent = min(window, cls.block_threads * count - first)
            if guarded:
                _clear_window(storage.slots, tid, cls.block_threads, count, extent, _record_default(cls.dtype, default))
            _write_window(value, destinations, flags, storage.slots, count, first, extent)
            barrier()
            out = [_read_window(storage.slots, index, first, extent, prior) for index, prior in zip(sources, out)]
            if cls.algorithm is BlockExchangeAlgorithm.WARP_TIME_SLICING:
                barrier()
        return _from_items(out, dtype=cls.dtype, like=template) if _is_items(template, cls.dtype) else out[0]

    @classmethod
    def _indices(cls, layout):
        cls._check()
        tid = _linear_thread_id(cls.block_size)
        count = cls.items_per_thread
        if layout == "blocked":
            return Vector.from_elements([tid * count + i for i in range(count)])
        if layout == "striped":
            return Vector.from_elements([tid + i * cls.block_threads for i in range(count)])
        base, lane, extent = _warp_layout(tid, cls.block_threads, cls.warp_threads, count)
        return Vector.from_elements([base + lane + i * extent for i in range(count)])

    @classmethod
    def blocked_to_striped(cls, value, *, storage):
        """Redistribute a blocked tile into block-striped ownership.

        Args:
            value: ``items_per_thread`` inputs of the specialized dtype in
                blocked order.
            storage: Shared instance of ``SharedStorage`` used by all threads.

        Returns:
            The calling thread's result in striped ownership order. A single T
            returns one T; an item sequence returns the same number of T items.
        """
        return cls._exchange(value, cls._indices("blocked"), cls._indices("striped"), storage)

    @classmethod
    def striped_to_blocked(cls, value, *, storage):
        """Redistribute a block-striped tile into blocked ownership.

        Args:
            value: ``items_per_thread`` inputs of the specialized dtype in
                striped order.
            storage: Shared instance of ``SharedStorage`` used by all threads.

        Returns:
            The calling thread's result value in blocked ownership order.
        """
        return cls._exchange(value, cls._indices("striped"), cls._indices("blocked"), storage)

    @classmethod
    def blocked_to_warp_striped(cls, value, *, storage):
        """Redistribute a blocked tile into stripes within each logical warp.

        Args:
            value: ``items_per_thread`` inputs of the specialized dtype in
                blocked order.
            storage: Shared instance of ``SharedStorage`` used by all threads.

        Returns:
            A result value in warp-striped ownership order,
            using the target warp width.
        """
        return cls._exchange(value, cls._indices("blocked"), cls._indices("warp_striped"), storage)

    @classmethod
    def warp_striped_to_blocked(cls, value, *, storage):
        """Redistribute warp-striped items into blocked ownership.

        Args:
            value: ``items_per_thread`` inputs of the specialized dtype in
                warp-striped order.
            storage: Shared instance of ``SharedStorage`` used by all threads.

        Returns:
            A result value in blocked ownership order,
            using the target warp width.
        """
        return cls._exchange(value, cls._indices("warp_striped"), cls._indices("blocked"), storage)

    @classmethod
    def _scatter(cls, value, ranks, storage, striped, flags=None, default=0, guarded=False):
        cls._check(value)
        count = cls.items_per_thread
        ranks = Vector.from_elements(_as_items(ranks))
        if ranks.numel != count or ranks.dtype.is_float:
            raise TypeError("ranks must contain items_per_thread integer entries")
        if flags is not None:
            flags = Vector.from_elements(_as_items(flags))
            if flags.numel != count:
                raise TypeError("is_valid must contain items_per_thread entries")
        if guarded:
            flags = ranks >= 0
        sources = cls._indices("striped" if striped else "blocked")
        return cls._exchange(value, ranks, sources, storage, flags, default)

    @classmethod
    def scatter_to_blocked(cls, value, ranks, *, storage):
        """Scatter ranked inputs and read the result in blocked order.

        Args:
            value: ``items_per_thread`` input items of the specialized dtype.
            ranks: Integer scalar or list/tuple/Vector of destination indices, one per input.
                Across the block, ranks must form a permutation of the tile.
            storage: Shared instance of ``SharedStorage`` used by all threads.

        Returns:
            A result value with blocked output ownership.
        """
        return cls._scatter(value, ranks, storage, False)

    @classmethod
    def scatter_to_striped(cls, value, ranks, *, storage):
        """Scatter ranked inputs and read the result in striped order.

        Args:
            value: ``items_per_thread`` input items of the specialized dtype.
            ranks: Integer scalar or list/tuple/Vector of destination indices, one per input.
                Across the block, ranks must form a permutation of the tile.
            storage: Shared instance of ``SharedStorage`` used by all threads.

        Returns:
            A result value with striped output ownership.
        """
        return cls._scatter(value, ranks, storage, True)

    @classmethod
    def scatter_to_striped_guarded(
        cls,
        value,
        ranks,
        *,
        storage,
        default=0,
    ):
        """Ignore negative/out-of-tile ranks and fill unassigned outputs.

        Args:
            value: ``items_per_thread`` input items of the specialized dtype.
            ranks: Integer scalar or list/tuple/Vector of destination indices. In-range ranks
                must be distinct across the block; out-of-range ranks are ignored.
            storage: Shared instance of ``SharedStorage`` used by all threads.
            default: Fill value for unassigned outputs, compatible with the specialized dtype.

        Returns:
            A result value with striped output ownership.
        """
        return cls._scatter(value, ranks, storage, True, default=default, guarded=True)

    @classmethod
    def scatter_to_striped_flagged(
        cls,
        value,
        ranks,
        is_valid,
        *,
        storage,
        default=0,
    ):
        """Scatter flagged items and fill holes in striped output ownership.

        Args:
            value: ``items_per_thread`` input items of the specialized dtype.
            ranks: Integer scalar or list/tuple/Vector of destination indices. Participating
                in-range ranks must be distinct; out-of-tile ranks are ignored.
            is_valid: Scalar or list/tuple/Vector with one flag per input; nonzero entries
                participate in the scatter.
            storage: Shared instance of ``SharedStorage`` used by all threads.
            default: Fill for unassigned outputs; scalar fills broadcast to
                record fields.

        Returns:
            A result value with striped output ownership.
        """
        return cls._scatter(value, ranks, storage, True, is_valid, default)
