# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Store tiles distributed across independent logical warps."""

import enum

from ....expr.gpu import known_block_size
from ....expr.numeric import Integer
from ....expr.typing import Vector
from .._common import _linear_thread_id, _resolve_warp_width
from .._io import _store_item, _store_vectorized
from .._values import _as_items
from .exchange import _warp_gather

__all__ = ["WarpStoreAlgorithm", "warp_store"]


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


def warp_store(
    destination,
    value,
    *,
    width: int | None = None,
    offset: int | Integer = 0,
    valid_items: int | Integer | None = None,
    algorithm: WarpStoreAlgorithm = WarpStoreAlgorithm.DIRECT,
) -> "None":
    """Store a tile for each logical warp with optional prefix guards.

    Consume blocked items except for STRIPED policy. Invalid elements perform
    no memory write. Destination plus offset is the calling
    logical warp's tile base; no warp offset is implicit. TRANSPOSE requires
    all logical-warp lanes to participate. Packed accesses fall back to scalar
    accesses for partial packs or unsupported layouts.

    Args:
        destination: Writable indexable DSL destination with a dtype. Values convert to its
            element type. With offset=0, its first element is the calling logical
            warp's tile base.
        value: This lane's input value with a fixed item count, blocked except for STRIPED policy.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        offset: Element offset of the calling logical warp's tile in destination. Include
            any grid, block, and warp displacement not already applied to destination.
        valid_items: Valid prefix length measured in items, not lanes; None accesses the
            whole tile. The count is relative to this logical warp tile and must be
            uniform within the logical warp.
        algorithm: WarpStoreAlgorithm policy; defaults to DIRECT.
            STRIPED uses striped ownership; the other policies use blocked ownership.

    Raises:
        TypeError: If algorithm is not a WarpStoreAlgorithm.
        ValueError: If width is unsupported.

    Examples:
        # L0..L7 are lanes of the SAME physical warp. width=4 forms two independent groups:
        # group 0 = L0..L3, group 1 = L4..L7. All lanes in each group participate.
        # Numeric inputs use fx.Int32; [] denotes per-lane items.

        # destination initially contains 16 copies of -9. Each group writes to its own eight-item tile.
        # Group          |                group 0                ||                group 1
        # Data           | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
        # ---------------+---------+---------+---------+---------++---------+---------+---------+--------
        # x in (blocked) | [10,11] | [12,13] | [14,15] | [16,17] || [20,21] | [22,23] | [24,25] | [26,27]

        group_id = fx.thread_idx.x // 4
        fx.coop.warp_store(destination, x, width=4, offset=group_id * 8, valid_items=6)
        # destination[0:8]  = [10,11,12,13,14,15,-9,-9]  (group 0)
        # destination[8:16] = [20,21,22,23,24,25,-9,-9]  (group 1)
        # Masked stores leave the previous -9 values in memory.

        # Store striped input in a separate scenario; destination initially contains 18 copies of -9.
        # Group        |                group 0                ||                group 1
        # Data         | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
        # -------------+---------+---------+---------+---------++---------+---------+---------+--------
        # striped_x in | [10,14] | [11,15] | [12,16] | [13,17] || [20,24] | [21,25] | [22,26] | [23,27]

        fx.coop.warp_store(
            destination, striped_x, width=4, offset=2 + group_id * 8, valid_items=fx.min(fx.max(13 - group_id * 8, 0), 8),
            algorithm=fx.coop.WarpStoreAlgorithm.STRIPED,
        )
        # destination[0:2]   = [-9,-9]                       (before the explicit offset)
        # destination[2:10]  = [10,11,12,13,14,15,16,17]     (all eight items of group 0)
        # destination[10:18] = [20,21,22,23,24,-9,-9,-9]     (five valid items of group 1)
        # L4 writes both slots; L5..L7 write only slot 0. The unwritten locations keep -9.
    """
    width = _resolve_warp_width(width, "warp_store width")
    if not isinstance(algorithm, WarpStoreAlgorithm):
        raise TypeError("algorithm must be a WarpStoreAlgorithm")
    tid = _linear_thread_id(known_block_size())
    count = len(_as_items(value))
    lane = tid % width
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
