# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block tile stores with direct, vector and shared-memory transpose policies."""

import enum

from ....expr.numeric import Integer
from .._common import _linear_thread_id
from .._io import _store_item, _store_vectorized
from .._values import _as_items
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta
from .exchange import BlockExchange, BlockExchangeAlgorithm, _storage, _timesliced_storage, _warp_layout
from .load import _direct_storage

__all__ = [
    "BlockStoreAlgorithm",
    "BlockStore",
]


class BlockStoreAlgorithm(enum.Enum):
    """Input ownership and global-memory access policies for block stores.

    Attributes:
        DIRECT: Store blocked inputs at contiguous per-thread positions.
        STRIPED: Store striped inputs at block-striped positions.
        VECTORIZE: Store blocked inputs with packed writes where possible and
            scalar writes for guarded tails or unsupported destination containers.
        TRANSPOSE: Exchange blocked inputs into block stripes before storing.
        WARP_TRANSPOSE: Exchange blocked inputs into stripes within each logical
            warp before storing, using the specialization's logical warp width.
        WARP_TRANSPOSE_TIMESLICED: Store blocked inputs through warp stripes
            using a shared buffer for one warp-sized window at a time.
    """

    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"
    WARP_TRANSPOSE = "warp_transpose"
    WARP_TRANSPOSE_TIMESLICED = "warp_transpose_timesliced"

    Direct = DIRECT
    Striped = STRIPED
    Vectorize = VECTORIZE
    Transpose = TRANSPOSE
    WarpTranspose = WARP_TRANSPOSE
    WarpTransposeTimesliced = WARP_TRANSPOSE_TIMESLICED


class _BlockStoreMeta(BlockTileAlgorithmMeta):
    _algorithms = BlockStoreAlgorithm
    _shared_storage = {
        BlockStoreAlgorithm.DIRECT: _direct_storage,
        BlockStoreAlgorithm.STRIPED: _direct_storage,
        BlockStoreAlgorithm.VECTORIZE: _direct_storage,
        BlockStoreAlgorithm.TRANSPOSE: _storage,
        BlockStoreAlgorithm.WARP_TRANSPOSE: _storage,
        BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED: _timesliced_storage,
    }

    def _default_algorithm_for(cls, target):
        return BlockStoreAlgorithm.DIRECT

    def __call__(
        cls,
        destination,
        value,
        *,
        storage=None,
        offset: int | Integer = 0,
        valid_items: int | Integer | None = None,
    ) -> "None":
        """Implement :meth:`BlockStore.__call__`."""
        cls._check(value)
        tid = _linear_thread_id(cls.block_size)
        count = cls.items_per_thread
        if cls.algorithm is BlockStoreAlgorithm.VECTORIZE:
            _store_vectorized(destination, value, tid * count, offset, valid_items, cls.dtype)
            return
        warp_transpose = cls.algorithm in (
            BlockStoreAlgorithm.WARP_TRANSPOSE,
            BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
        )
        transpose = warp_transpose or cls.algorithm is BlockStoreAlgorithm.TRANSPOSE
        if transpose:
            if storage is None:
                raise TypeError(f"{cls.algorithm.name} requires storage")
            policy = (
                BlockExchangeAlgorithm.WARP_TIME_SLICING
                if cls.algorithm is BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED
                else BlockExchangeAlgorithm.SHARED_MEMORY
            )
            exchange = cls.exchange_type[cls.dtype, cls.block_size, count, policy]
            if warp_transpose:
                value = exchange.blocked_to_warp_striped(value, storage=storage)
            else:
                value = exchange.blocked_to_striped(value, storage=storage)
        items = _as_items(value, cls.dtype)
        if warp_transpose:
            base, lane, extent = _warp_layout(tid, cls.block_threads, cls.warp_threads, count)
            indices = [base + lane + i * extent for i in range(count)]
        elif cls.algorithm in (BlockStoreAlgorithm.STRIPED, BlockStoreAlgorithm.TRANSPOSE):
            indices = [tid + i * cls.block_threads for i in range(count)]
        else:
            indices = [tid * count + i for i in range(count)]
        for index, item in zip(indices, items):
            _store_item(destination, item, index, offset, valid_items)


class BlockStore(BlockPrimitive, metaclass=_BlockStoreMeta):
    """Store a block tile with inverse ownership of ``BlockLoad``.

    Specialize with ``BlockStore[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockStore[...]``, in
    bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. It is also the stored
            element type.
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
        algorithm: Optional member of BlockStoreAlgorithm selecting the implementation
            policy. Defaults to ``BlockStoreAlgorithm.DIRECT`` when omitted. It fixes the
            specialization and its SharedStorage layout.

    All policies consume
    blocked items except ``STRIPED``, which consumes block-striped items.
    ``valid_items`` guards a prefix relative to the destination ``offset``.
    Transpose policies require uniform participation by every block thread
    and include internal barriers. Synchronize before reusing their storage.

    Attributes:
        dtype: Specialized input value type.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        items_per_thread: Positive compile-time number of inputs per thread.
        algorithm: Selected ``BlockStoreAlgorithm`` policy.
        warp_threads: Logical warp width used by warp-transpose policies.
        SharedStorage: Shared-memory struct type with ``slots``. Transpose
            policies allocate a full tile, except the time-sliced policy which
            allocates one warp-sized window; other policies use no scratch data.
        exchange_type: Exchange primitive used by transpose policies.

    Examples:
        # x in thread t is [10+2*t, 11+2*t]. destination initially holds 130 copies of -9.
        P = fx.coop.BlockStore[fx.Int32, 64, 2, fx.coop.BlockStoreAlgorithm.DIRECT]
        P(destination, x, valid_items=5)
        # destination starts [10,11,12,13,14]; the remaining 125 entries retain -9.

        # For this separate scenario, use a fresh destination of 130 copies of -9.
        Q = fx.coop.BlockStore[fx.Int32, 64, 2, fx.coop.BlockStoreAlgorithm.TRANSPOSE]
        scratch = fx.SharedAllocator().allocate(Q.SharedStorage).peek()
        Q(destination, x, storage=scratch, offset=2, valid_items=5)
        # destination starts [-9,-9,10,11,12,13,14]; all remaining entries retain -9.
        fx.barrier()
    """

    exchange_type = BlockExchange

    @classmethod
    def __call__(
        cls,
        destination,
        value,
        *,
        storage=None,
        offset: int | Integer = 0,
        valid_items: int | Integer | None = None,
    ) -> "None":
        """Store one block tile with optional guarded writes.

        Args:
            destination: Element-indexable writable container. Items are
                converted to the destination's dtype.
            value: ``items_per_thread`` inputs of the specialized dtype. Inputs
                use blocked ownership except for ``STRIPED``.
            storage: Shared instance of ``SharedStorage``. Required by transpose
                policies and unused by direct, striped, and vectorized policies.
            offset: Destination starting offset in elements.
            valid_items: Number of writable elements relative to ``offset``, or
                ``None`` for the entire tile. Items outside this prefix are
                skipped without accessing destination memory.

        Returns:
            None. Writes the valid items to ``destination``.

        Raises:
            TypeError: The operator must be specialized with ``BlockStore[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(
                f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, items_per_thread]"
            )
        return type(cls).__call__(cls, destination, value, storage=storage, offset=offset, valid_items=valid_items)
