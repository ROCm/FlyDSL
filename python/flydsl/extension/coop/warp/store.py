# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Store tiles distributed across independent logical warps."""

import enum

from ....expr.gpu import lane_id
from ....expr.numeric import Integer
from ....expr.typing import Vector
from .._common import _validate_valid_items
from .._io import _store_item, _store_vectorized
from .._values import _as_items
from ._spec import WarpPrimitive
from .exchange import _warp_gather

__all__ = [
    "WarpStoreAlgorithm",
    "WarpStore",
]


class WarpStoreAlgorithm(enum.Enum):
    """Select the memory access and ownership policy for a warp store.

    Attributes:
        DIRECT: Access each lane's contiguous blocked items independently.
        STRIPED: Access strided items and retain striped register ownership.
        VECTORIZE: Use packed contiguous accesses where legal, with scalar fallbacks.
        TRANSPOSE: Access striped memory and exchange with blocked register ownership
            using logical-warp shuffles. All lanes of the group participate.
    """

    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"

    Direct = DIRECT
    Striped = STRIPED
    Vectorize = VECTORIZE
    Transpose = TRANSPOSE


class WarpStore(WarpPrimitive):
    """Store a logical-warp tile with a fixed memory-access policy.

    Specialize with ``[dtype, width, items_per_thread, algorithm]``. The final
    policy is optional and defaults to ``WarpStoreAlgorithm.DIRECT``.
    ``None`` selects the target's physical warp width. All participating lanes
    use the same specialization and per-call options. Algorithm selection is
    part of the operator type; member functions do not accept an algorithm.
    STRIPED retains striped ownership; other policies use blocked ownership.
    SharedStorage is Empty with a zero-byte layout. The caller selects each
    logical warp's tile address explicitly with a view or offset.

    Examples:
        P = fx.coop.WarpStore[fx.Int32, 8, 2]
        P.store(destination, x, offset=(fx.thread_idx.x // 8) * 16)
    """

    _tile = True
    _algorithms = WarpStoreAlgorithm
    _default_algorithm = WarpStoreAlgorithm.DIRECT

    @classmethod
    def store(
        cls,
        destination,
        value,
        *,
        offset: int | Integer = 0,
        valid_items: int | Integer | None = None,
        storage=None,
    ) -> "None":
        """Store a tile for each logical warp with optional prefix guards.

        Consume blocked items except for STRIPED policy. Invalid elements perform
        no memory write. Destination plus offset is the calling
        logical warp's tile base; no warp offset is implicit. TRANSPOSE requires
        all logical-warp lanes to participate. Packed accesses fall back to scalar
        accesses for partial packs or unsupported layouts.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            destination: Writable indexable DSL destination with a dtype. Values convert to its
                element type. With offset=0, its first element is the calling logical
                warp's tile base.
            value: This lane's input value with a fixed item count, blocked except for STRIPED policy.
            offset: Element offset of the calling logical warp's tile in destination. Include
                any grid, block, and warp displacement not already applied to destination.
            valid_items: Valid prefix length measured in items, not lanes; None accesses the
                whole tile. The count is relative to this logical warp tile and must be
                uniform within the logical warp.

        Examples:
            # destination initially contains 16 copies of -9. Each group writes to its own eight-item tile.
            # Group          |                group 0                ||                group 1
            # Data           | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
            # ---------------+---------+---------+---------+---------++---------+---------+---------+--------
            # x in (blocked) | [10,11] | [12,13] | [14,15] | [16,17] || [20,21] | [22,23] | [24,25] | [26,27]

            group_id = fx.thread_idx.x // 4
            fx.coop.WarpStore[fx.Int32, 4, 2, fx.coop.WarpStoreAlgorithm.DIRECT].store(destination, x, offset=group_id * 8, valid_items=6)
            # destination[0:8]  = [10,11,12,13,14,15,-9,-9]  (group 0)
            # destination[8:16] = [20,21,22,23,24,25,-9,-9]  (group 1)
            # Masked stores leave the previous -9 values in memory.

            # Store striped input in a separate scenario; destination initially contains 18 copies of -9.
            # Group        |                group 0                ||                group 1
            # Data         | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
            # -------------+---------+---------+---------+---------++---------+---------+---------+--------
            # striped_x in | [10,14] | [11,15] | [12,16] | [13,17] || [20,24] | [21,25] | [22,26] | [23,27]

            fx.coop.WarpStore[fx.Int32, 4, 2, fx.coop.WarpStoreAlgorithm.STRIPED].store(destination, striped_x, offset=2 + group_id * 8, valid_items=fx.min(fx.max(13 - group_id * 8, 0), 8))
            # destination[0:2]   = [-9,-9]                       (before the explicit offset)
            # destination[2:10]  = [10,11,12,13,14,15,16,17]     (all eight items of group 0)
            # destination[10:18] = [20,21,22,23,24,-9,-9,-9]     (five valid items of group 1)
            # L4 writes both slots; L5..L7 write only slot 0. The unwritten locations keep -9.
        """
        cls._check_storage(storage)
        value = cls._prepare(value)
        algorithm = cls.algorithm
        width = cls.warp_threads
        _validate_valid_items(valid_items, width * cls.items_per_thread)
        count = len(_as_items(value))
        lane = lane_id() % width
        if algorithm is WarpStoreAlgorithm.VECTORIZE:
            _store_vectorized(destination, value, lane * count, offset, valid_items)
            return
        if algorithm is WarpStoreAlgorithm.TRANSPOSE:
            indices = Vector.from_elements([lane + i * width for i in range(count)])
            value = _warp_gather(value, indices, width=width)
        striped = algorithm in (WarpStoreAlgorithm.STRIPED, WarpStoreAlgorithm.TRANSPOSE)
        items = _as_items(value)
        for i, item in enumerate(items):
            index = i * width + lane if striped else lane * count + i
            _store_item(destination, item, index, offset, valid_items)
