# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register and shared-memory exchanges within independent logical warps."""

import enum
import math

from ...._mlir.dialects import gpu
from ....expr.gpu import barrier, known_block_size, lane_id
from ....expr.struct import Struct
from ....expr.typing import Int32, Vector
from .._common import _linear_thread_id, _resolve_warp_width
from .._values import (
    _as_items,
    _from_items,
    _is_items,
    _record_select,
    _record_shuffle,
    _shared_array,
    _shared_load,
    _shared_store,
)

__all__ = [
    "WarpExchangeAlgorithm",
    "warp_exchange_storage",
    "warp_blocked_to_striped",
    "warp_striped_to_blocked",
    "warp_scatter_to_striped",
]


class WarpExchangeAlgorithm(enum.Enum):
    """Select the transport for logical-warp item exchanges.

    Attributes:
        SHUFFLE: Move items through warp register shuffles without shared storage.
        SMEM: Move items through disjoint per-group regions of caller-owned shared
            storage, with subgroup synchronization.
    """

    SHUFFLE = "shuffle"
    SMEM = "smem"


def _check_algorithm(algorithm, storage):
    if not isinstance(algorithm, WarpExchangeAlgorithm):
        raise TypeError("algorithm must be a WarpExchangeAlgorithm")
    if algorithm is WarpExchangeAlgorithm.SMEM and storage is None:
        raise TypeError("SMEM requires storage allocated from warp_exchange_storage")


def warp_exchange_storage(
    dtype,
    items_per_thread: int,
    *,
    block_size: int | tuple[int, int, int] | list[int] | None = None,
):
    """Create the shared-storage type for all logical warps in a block.

    Args:
        dtype: Input value type.
        items_per_thread: Positive compile-time number of items per lane.
        block_size: Block thread count or (x, y, z) dimensions. None uses the known kernel
            block shape.

    Returns:
        A Struct type with disjoint regions for each logical warp. Allocate it with
        SharedAllocator and pass the resulting view to SMEM exchanges.

    Raises:
        ValueError: If items_per_thread is not a positive Python integer or the block size
            is unknown.

    Examples:
        # L0..L7 are lanes of the SAME physical warp. width=4 forms two independent groups:
        # group 0 = L0..L3, group 1 = L4..L7. All lanes in each group participate.
        # Numeric inputs use fx.Int32; [] denotes per-lane items.

        # This block has eight threads, each holding two Int32 items: 16 slots, totaling 64 bytes.
        # Allocate scratch for BOTH groups; each group exchanges only its own eight items.
        Storage = fx.coop.warp_exchange_storage(fx.Int32, 2, block_size=8)
        storage = fx.SharedAllocator().allocate(Storage).peek()
        y = fx.coop.warp_blocked_to_striped(
            x, width=4, algorithm=fx.coop.WarpExchangeAlgorithm.SMEM, storage=storage
        )
        # Group         |            group 0            ||                    group 1
        # Data          | L0    | L1    | L2    | L3    || L4        | L5        | L6        | L7
        # --------------+-------+-------+-------+-------++-----------+-----------+-----------+----------
        # in (blocked)  | [0,1] | [2,3] | [4,5] | [6,7] || [100,101] | [102,103] | [104,105] | [106,107]
        # out (striped) | [0,4] | [1,5] | [2,6] | [3,7] || [100,104] | [101,105] | [102,106] | [103,107]

        # Blocked order reads both items of L0, then both of L1, and so on within a group.
        # Striped order reads slot 0 of all four lanes, then slot 1 of all four lanes.
        # The logical sequences stay [0,1,...,7] and [100,101,...,107]; groups never exchange items.
    """
    if not isinstance(items_per_thread, int) or isinstance(items_per_thread, bool) or items_per_thread < 1:
        raise ValueError("items_per_thread must be a positive Python int")
    shape = known_block_size() if block_size is None else block_size
    if shape is None:
        raise ValueError("shared warp exchange requires a known block size")
    threads = shape if isinstance(shape, int) else math.prod(shape)
    return Struct["slots" : _shared_array(dtype, threads * items_per_thread)]


def _warp_gather(
    value,
    indices,
    *,
    width: int | None = None,
    algorithm: WarpExchangeAlgorithm = WarpExchangeAlgorithm.SHUFFLE,
    storage=None,
):
    """Internal gather from a blocked warp tile; indices are group-relative."""
    width = _resolve_warp_width(width, "_warp_gather width")
    _check_algorithm(algorithm, storage)
    items = _as_items(value)
    requested = list(indices) if isinstance(indices, Vector) else [indices]
    out = []
    if algorithm is WarpExchangeAlgorithm.SMEM:
        tid = _linear_thread_id(known_block_size())
        for i, item in enumerate(items):
            _shared_store(storage.slots, tid * len(items) + i, item)
        barrier(scope=gpu.BarrierScope.Subgroup)
        base = (tid // width) * width * len(items)
        out = [_shared_load(storage.slots, base + index) for index in requested]
    else:
        for index in requested:
            index = Int32(index)
            source_lane = (lane_id() // width) * width + index // len(items)
            source_item = index % len(items)
            result = _record_shuffle(items[0], source_lane, width)
            for i, item in enumerate(items[1:], 1):
                moved = _record_shuffle(item, source_lane, width)
                result = _record_select(source_item == i, moved, result)
            out.append(result)
    return _from_items(out, like=value) if isinstance(indices, Vector) else out[0]


def warp_blocked_to_striped(
    value,
    *,
    width: int | None = None,
    algorithm: WarpExchangeAlgorithm = WarpExchangeAlgorithm.SMEM,
    storage=None,
):
    """Convert blocked item ownership to striped within a logical warp.

    Blocked index is lane * items_per_thread + item; striped index is
    item * width + lane. All lanes of the logical warp must participate.

    Args:
        value: This lane's input value in blocked order.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        algorithm: WarpExchangeAlgorithm policy; defaults to SMEM. SHUFFLE requires
            items_per_thread == width and does not use storage.
        storage: Caller-allocated view of warp_exchange_storage for the block and item
            count. Required for SMEM; unused for SHUFFLE. Synchronize the logical group
            before reusing it.

    Returns:
        This lane's result value in striped order. A single input item is returned unchanged.

    Raises:
        TypeError: If algorithm is invalid or required shared storage is missing.
        ValueError: If width or the item count is unsupported by the selected algorithm.

    Examples:
        # L0..L7 are lanes of the SAME physical warp. width=4 forms two independent groups:
        # group 0 = L0..L3, group 1 = L4..L7. All lanes in each group participate.
        # Numeric inputs use fx.Int32; [] denotes per-lane items.

        Storage = fx.coop.warp_exchange_storage(fx.Int32, 2, block_size=8)
        storage = fx.SharedAllocator().allocate(Storage).peek()
        y = fx.coop.warp_blocked_to_striped(x, width=4, storage=storage)
        # Group         |            group 0            ||                    group 1
        # Data          | L0    | L1    | L2    | L3    || L4        | L5        | L6        | L7
        # --------------+-------+-------+-------+-------++-----------+-----------+-----------+----------
        # in (blocked)  | [0,1] | [2,3] | [4,5] | [6,7] || [100,101] | [102,103] | [104,105] | [106,107]
        # out (striped) | [0,4] | [1,5] | [2,6] | [3,7] || [100,104] | [101,105] | [102,106] | [103,107]

        # Blocked order reads both items of L0, then both of L1, and so on within a group.
        # Striped order reads slot 0 of all four lanes, then slot 1 of all four lanes.
        # The logical sequences stay [0,1,...,7] and [100,101,...,107]; groups never exchange items.

        # Shared-memory transport gives the same results; all eight block threads participate.
        fx.barrier()  # Finish reading scratch before reusing it.
        shared_y = fx.coop.warp_blocked_to_striped(
            x, width=4, algorithm=fx.coop.WarpExchangeAlgorithm.SMEM, storage=storage
        )
        fx.barrier()  # Finish reading scratch before it is reused.
    """
    width = _resolve_warp_width(width, "warp_blocked_to_striped width")
    _check_algorithm(algorithm, storage)
    if algorithm is WarpExchangeAlgorithm.SHUFFLE and len(_as_items(value)) != width:
        raise ValueError("SHUFFLE requires items_per_thread == width")
    if not _is_items(value):
        return value
    lane = lane_id() % width
    indices = Vector.from_elements([lane + i * width for i in range(len(_as_items(value)))])
    return _warp_gather(value, indices, width=width, algorithm=algorithm, storage=storage)


def warp_striped_to_blocked(
    value,
    *,
    width: int | None = None,
    algorithm: WarpExchangeAlgorithm = WarpExchangeAlgorithm.SMEM,
    storage=None,
):
    """Convert striped item ownership to blocked within a logical warp.

    Blocked index is lane * items_per_thread + item; striped index is
    item * width + lane. All lanes of the logical warp must participate.

    Args:
        value: This lane's input value in striped order.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        algorithm: WarpExchangeAlgorithm policy; defaults to SMEM. SHUFFLE requires
            items_per_thread == width and does not use storage.
        storage: Caller-allocated view of warp_exchange_storage for the block and item
            count. Required for SMEM; unused for SHUFFLE. Synchronize the logical group
            before reusing it.

    Returns:
        This lane's result value in blocked order. A single input item is returned unchanged.

    Raises:
        TypeError: If algorithm is invalid or required shared storage is missing.
        ValueError: If width or the item count is unsupported by the selected algorithm.

    Examples:
        # L0..L7 are lanes of the SAME physical warp. width=4 forms two independent groups:
        # group 0 = L0..L3, group 1 = L4..L7. All lanes in each group participate.
        # Numeric inputs use fx.Int32; [] denotes per-lane items.

        Storage = fx.coop.warp_exchange_storage(fx.Int32, 2, block_size=8)
        storage = fx.SharedAllocator().allocate(Storage).peek()
        y = fx.coop.warp_striped_to_blocked(x, width=4, storage=storage)
        # Group         |            group 0            ||                    group 1
        # Data          | L0    | L1    | L2    | L3    || L4        | L5        | L6        | L7
        # --------------+-------+-------+-------+-------++-----------+-----------+-----------+----------
        # in (striped)  | [0,4] | [1,5] | [2,6] | [3,7] || [100,104] | [101,105] | [102,106] | [103,107]
        # out (blocked) | [0,1] | [2,3] | [4,5] | [6,7] || [100,101] | [102,103] | [104,105] | [106,107]

        # Blocked order reads both items of L0, then both of L1, and so on within a group.
        # Striped order reads slot 0 of all four lanes, then slot 1 of all four lanes.
        # The logical sequences stay [0,1,...,7] and [100,101,...,107]; groups never exchange items.

        # Shared-memory transport gives the same results; all eight block threads participate.
        fx.barrier()  # Finish reading scratch before reusing it.
        shared_y = fx.coop.warp_striped_to_blocked(
            x, width=4, algorithm=fx.coop.WarpExchangeAlgorithm.SMEM, storage=storage
        )
        fx.barrier()  # Finish reading scratch before it is reused.
    """
    width = _resolve_warp_width(width, "warp_striped_to_blocked width")
    _check_algorithm(algorithm, storage)
    if algorithm is WarpExchangeAlgorithm.SHUFFLE and len(_as_items(value)) != width:
        raise ValueError("SHUFFLE requires items_per_thread == width")
    if not _is_items(value):
        return value
    lane = lane_id() % width
    count = len(_as_items(value))
    indices = [lane * count + i for i in range(count)]
    sources = Vector.from_elements([(i % width) * count + i // width for i in indices])
    return _warp_gather(value, sources, width=width, algorithm=algorithm, storage=storage)


def warp_scatter_to_striped(
    value,
    ranks,
    *,
    width: int | None = None,
    algorithm: WarpExchangeAlgorithm = WarpExchangeAlgorithm.SMEM,
    storage=None,
):
    """Scatter a permutation of logical-warp tile ranks into striped ownership.

    Ranks over the entire group must be a permutation of the tile indices, without
    duplicates or out-of-range destinations. All logical-warp lanes participate.

    Args:
        value: Input value containing nonempty items of the same type.
        ranks: Integer Vector with one destination rank per item, in [0, width *
            items_per_thread).
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        algorithm: WarpExchangeAlgorithm.SMEM; scatter requires shared storage.
        storage: Caller-allocated view of warp_exchange_storage for the block and item
            count. Required for SMEM; unused for SHUFFLE. Synchronize the logical group
            before reusing it.

    Returns:
        A result value whose slot i in lane l contains the input with rank i * width + l.

    Raises:
        TypeError: If ranks have the wrong type/extent, algorithm is invalid, or SMEM
            storage is missing.
        ValueError: If width or the item count is unsupported by the selected algorithm.

    Examples:
        # L0..L7 are lanes of the SAME physical warp. width=4 forms two independent groups:
        # group 0 = L0..L3, group 1 = L4..L7. All lanes in each group participate.
        # Numeric inputs use fx.Int32; [] denotes per-lane items.

        # Each input item supplies its destination rank in the group's logical sequence.
        # The ranks form an arbitrary permutation of 0..7 in EACH group; no duplicates are allowed.
        Storage = fx.coop.warp_exchange_storage(fx.Int32, 2, block_size=8)
        storage = fx.SharedAllocator().allocate(Storage).peek()
        y = fx.coop.warp_scatter_to_striped(x, ranks, width=4, storage=storage)
        # Group           |                group 0                ||                    group 1
        # Data            | L0      | L1      | L2      | L3      || L4        | L5        | L6        | L7
        # ----------------+---------+---------+---------+---------++-----------+-----------+-----------+----------
        # x in (blocked)  | [10,20] | [30,40] | [50,60] | [70,80] || [110,120] | [130,140] | [150,160] | [170,180]
        # ranks           | [2,7]   | [0,5]   | [6,1]   | [4,3]   || [2,7]     | [0,5]     | [6,1]     | [4,3]
        # y out (striped) | [30,70] | [60,40] | [10,50] | [80,20] || [130,170] | [160,140] | [110,150] | [180,120]

        # Group 0's logical output is [30,60,10,80,70,40,50,20]. Striped ownership puts
        # ranks [0,4] in L0, [1,5] in L1, [2,6] in L2 and [3,7] in L3; group 1 follows the same rule.

        # Shared-memory transport preserves these results. All eight block threads participate.
        fx.barrier()  # Finish reading scratch before reusing it.
        shared_y = fx.coop.warp_scatter_to_striped(
            x, ranks, width=4, algorithm=fx.coop.WarpExchangeAlgorithm.SMEM, storage=storage
        )
        fx.barrier()  # Finish reading scratch before it is reused.
    """
    width = _resolve_warp_width(width, "warp_scatter_to_striped width")
    _check_algorithm(algorithm, storage)
    if algorithm is WarpExchangeAlgorithm.SHUFFLE:
        raise ValueError("Scatter to striped supports only SMEM")
    items = _as_items(value)
    if not isinstance(ranks, Vector) or ranks.numel != len(items) or ranks.dtype.is_float:
        raise TypeError("ranks must be an integer Vector with one rank per input item")
    lane = lane_id() % width
    tid = _linear_thread_id(known_block_size())
    base = (tid // width) * width * len(items)
    for i, item in enumerate(items):
        _shared_store(storage.slots, base + ranks[i], item)
    barrier(scope=gpu.BarrierScope.Subgroup)
    return _from_items(
        [_shared_load(storage.slots, base + lane + i * width) for i in range(len(items))],
        like=value,
    )
