# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block tile loads with direct, vector and shared-memory transpose policies."""

import enum

from ....expr.numeric import Integer
from ....expr.struct import Struct
from .._common import _linear_thread_id
from .._io import _load_item, _load_vectorized
from .._values import _from_items, _record_default, _shared_array
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta
from .exchange import BlockExchange, BlockExchangeAlgorithm, _storage, _timesliced_storage, _warp_layout

__all__ = [
    "BlockLoadAlgorithm",
    "BlockLoad",
]


class BlockLoadAlgorithm(enum.Enum):
    """Global-memory access and output ownership policies for block loads.

    Attributes:
        DIRECT: Load contiguous per-thread items into blocked ownership.
        STRIPED: Load block stripes and retain striped output ownership.
        VECTORIZE: Load blocked items with packed reads where possible and
            scalar reads for guarded tails or unsupported source containers.
        TRANSPOSE: Load block stripes, then exchange into blocked ownership.
        WARP_TRANSPOSE: Load stripes within each logical warp, then exchange
            into blocked ownership, using the specialization's logical warp width.
        WARP_TRANSPOSE_TIMESLICED: Produce blocked output using warp stripes
            and one warp-sized shared buffer reused over sequential windows.
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


def _direct_storage(dtype, threads, warp, count):
    return Struct["slots" : _shared_array(dtype, 1)]


class _BlockLoadMeta(BlockTileAlgorithmMeta):
    _algorithms = BlockLoadAlgorithm
    _shared_storage = {
        BlockLoadAlgorithm.DIRECT: _direct_storage,
        BlockLoadAlgorithm.STRIPED: _direct_storage,
        BlockLoadAlgorithm.VECTORIZE: _direct_storage,
        BlockLoadAlgorithm.TRANSPOSE: _storage,
        BlockLoadAlgorithm.WARP_TRANSPOSE: _storage,
        BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED: _timesliced_storage,
    }

    def _default_algorithm_for(cls, target):
        return BlockLoadAlgorithm.DIRECT

    def __call__(
        cls,
        source,
        *,
        storage=None,
        offset: int | Integer = 0,
        valid_items: int | Integer | None = None,
        default=0,
    ):
        """Implement :meth:`BlockLoad.__call__`."""
        cls._check()
        tid = _linear_thread_id(cls.block_size)
        count = cls.items_per_thread
        if cls.algorithm is BlockLoadAlgorithm.VECTORIZE:
            return _load_vectorized(
                source, tid * count, count, offset, valid_items, _record_default(cls.dtype, default)
            )
        warp_transpose = cls.algorithm in (
            BlockLoadAlgorithm.WARP_TRANSPOSE,
            BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED,
        )
        transpose = warp_transpose or cls.algorithm is BlockLoadAlgorithm.TRANSPOSE
        if transpose and storage is None:
            raise TypeError(f"{cls.algorithm.name} requires storage")
        if warp_transpose:
            base, lane, extent = _warp_layout(tid, cls.block_threads, cls.warp_threads, count)
            indices = [base + lane + i * extent for i in range(count)]
        elif cls.algorithm in (BlockLoadAlgorithm.STRIPED, BlockLoadAlgorithm.TRANSPOSE):
            indices = [tid + i * cls.block_threads for i in range(count)]
        else:
            indices = [tid * count + i for i in range(count)]
        items = _from_items(
            [_load_item(source, index, offset, valid_items, _record_default(cls.dtype, default)) for index in indices]
        )
        if transpose:
            policy = (
                BlockExchangeAlgorithm.WARP_TIME_SLICING
                if cls.algorithm is BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED
                else BlockExchangeAlgorithm.SHARED_MEMORY
            )
            exchange = cls.exchange_type[cls.dtype, cls.block_size, count, policy]
            if warp_transpose:
                items = exchange.warp_striped_to_blocked(items, storage=storage)
            else:
                items = exchange.striped_to_blocked(items, storage=storage)
        return items


class BlockLoad(BlockPrimitive, metaclass=_BlockLoadMeta):
    """Collectively load a block tile with configurable memory access.

    Specialize with ``BlockLoad[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockLoad[...]``, in bracket
    order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. It is also the loaded
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
        algorithm: Optional member of BlockLoadAlgorithm selecting the implementation
            policy. Defaults to ``BlockLoadAlgorithm.DIRECT`` when omitted. It fixes the
            specialization and its SharedStorage layout.

    Every policy returns blocked items except STRIPED. VECTORIZE packs
    contiguous items into vector reads when possible, with scalar guarded
    tails. TRANSPOSE reads block stripes; WARP_TRANSPOSE reads independent
    warp stripes, using the specialization's logical warp width. TIMESLICED uses a warp-sized
    shared buffer and sequential warp windows. ``valid_items`` is relative to
    ``offset``; invalid entries are filled without accessing source memory.
    Transpose policies require uniform participation by every block thread
    and include internal barriers. Synchronize before reusing their storage.

    Attributes:
        dtype: Specialized output value type.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        items_per_thread: Positive compile-time number of outputs per thread.
        algorithm: Selected ``BlockLoadAlgorithm`` policy.
        warp_threads: Logical warp width used by warp-transpose policies.
        SharedStorage: Shared-memory struct type with ``slots``. Transpose
            policies allocate a full tile, except the time-sliced policy which
            allocates one warp-sized window; other policies use no scratch data.
        exchange_type: Exchange primitive used by transpose policies.

    Examples:
        # source contains [10,11,...,137], two items per thread.
        P = fx.coop.BlockLoad[fx.Int32, 64, 2, fx.coop.BlockLoadAlgorithm.DIRECT]
        x = P(source)
        # Thread t receives [10+2*t, 11+2*t].

        # A transpose load uses shared scratch and guards a prefix relative to offset=2.
        Q = fx.coop.BlockLoad[fx.Int32, 64, 2, fx.coop.BlockLoadAlgorithm.TRANSPOSE]
        scratch = fx.SharedAllocator().allocate(Q.SharedStorage).peek()
        partial = Q(source, storage=scratch, offset=2, valid_items=5, default=-1)
        # T0..T3: {[12,13], [14,15], [16,-1], [-1,-1]}; T4..T63: [-1,-1].
        fx.barrier()
    """

    exchange_type = BlockExchange

    @classmethod
    def __call__(
        cls,
        source,
        *,
        storage=None,
        offset: int | Integer = 0,
        valid_items: int | Integer | None = None,
        default=0,
    ):
        """Load one block tile from an element-indexable source.

        Args:
            source: Source value supporting element indexing. Items are converted to the specialized dtype.
            storage: Shared instance of ``SharedStorage``. Required by transpose
                policies and unused by direct, striped, and vectorized policies.
            offset: Source starting offset in elements.
            valid_items: Number of readable elements relative to ``offset``, or
                ``None`` to load the entire tile. Invalid items use ``default``
                without reading the source.
            default: Fill value for invalid items, compatible with the specialized dtype.

        Returns:
            A result value containing ``items_per_thread`` items.
            Ownership is blocked except for the ``STRIPED`` policy.

        Raises:
            TypeError: The operator must be specialized with ``BlockLoad[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(
                f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, items_per_thread]"
            )
        return type(cls).__call__(cls, source, storage=storage, offset=offset, valid_items=valid_items, default=default)
