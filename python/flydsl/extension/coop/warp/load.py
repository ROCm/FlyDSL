# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Load tiles distributed across independent logical warps."""

import enum

from ....expr.gpu import known_block_size
from ....expr.numeric import Integer
from ....expr.typing import Vector
from .._common import _linear_thread_id, _resolve_warp_width
from .._io import _load_item, _load_vectorized
from .._values import _from_items, _record_default
from .exchange import _warp_gather

__all__ = ["WarpLoadAlgorithm", "warp_load"]


class WarpLoadAlgorithm(enum.Enum):
    """Select the memory access and ownership policy for a warp load.

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


def warp_load(
    source,
    items_per_thread: int = 1,
    *,
    width: int | None = None,
    offset: int | Integer = 0,
    valid_items: int | Integer | None = None,
    default=None,
    algorithm: WarpLoadAlgorithm = WarpLoadAlgorithm.DIRECT,
):
    """Load a tile for each logical warp with optional prefix guards.

    Return items in blocked order except for STRIPED policy. Invalid elements
    do not read source memory and use default when it is provided. Source plus offset is the calling
    logical warp's tile base; no warp offset is implicit. TRANSPOSE requires all
    logical-warp lanes to participate. Packed accesses fall back to scalar accesses
    for partial packs or unsupported layouts.

    Args:
        source: Indexable source value with a dtype.
            With offset=0, its first element is the calling logical warp's tile base.
            May be a view already offset to that tile.
        items_per_thread: Positive compile-time number of items returned per lane.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        offset: Element offset of the calling logical warp's tile in source. Include
            any grid, block, and warp displacement not already applied to source.
        valid_items: Valid prefix length measured in items, not lanes; None accesses the
            whole tile. The count is relative to this logical warp tile and must be
            uniform within the logical warp.
        default: Optional value used for invalid items, converted to source.dtype; a
            scalar fills all record leaves. Requires valid_items. Without an explicit
            default, invalid returned items are unspecified.
        algorithm: WarpLoadAlgorithm policy; defaults to DIRECT.
            STRIPED uses striped ownership; the other policies use blocked ownership.

    Returns:
        A Vector for numeric elements or a tuple for plain Struct elements, with
        items_per_thread entries in the selected ownership layout and source.dtype.

    Raises:
        TypeError: If algorithm is not a WarpLoadAlgorithm.
        ValueError: If width is unsupported. items_per_thread must also be a positive Python integer.

    Examples:
        # L0..L7 are lanes of the SAME physical warp. width=4 forms two independent groups:
        # group 0 = L0..L3, group 1 = L4..L7. All lanes in each group participate.
        # Numeric inputs use fx.Int32; [] denotes per-lane items.

        # source contains [10,11,...,27]. The same source and offset select the SAME tile in both groups.
        # valid_items=6 applies separately to each group's eight-item tile.
        x = fx.coop.warp_load(source, 2, width=4, valid_items=6, default=-1)
        # Group           |                group 0                ||                group 1
        # Data            | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
        # ----------------+---------+---------+---------+---------++---------+---------+---------+--------
        # x out (blocked) | [10,11] | [12,13] | [14,15] | [-1,-1] || [10,11] | [12,13] | [14,15] | [-1,-1]

        # Now choose different tiles: group 0 starts at source[2], group 1 at source[10].
        # The eight-thread block has 13 valid items: eight in group 0 and five in group 1.
        group_id = fx.thread_idx.x // 4
        striped_x = fx.coop.warp_load(
            source, 2, width=4, offset=2 + group_id * 8, valid_items=fx.min(fx.max(13 - group_id * 8, 0), 8), default=-1,
            algorithm=fx.coop.WarpLoadAlgorithm.STRIPED,
        )
        # Group          |                group 0                ||                group 1
        # Data           | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
        # ---------------+---------+---------+---------+---------++---------+---------+---------+--------
        # source indices | [2,6]   | [3,7]   | [4,8]   | [5,9]   || [10,14] | [11,15] | [12,16] | [13,17]
        # valid slots    | [1,1]   | [1,1]   | [1,1]   | [1,1]   || [1,1]   | [1,0]   | [1,0]   | [1,0]
        # striped_x      | [12,16] | [13,17] | [14,18] | [15,19] || [20,24] | [21,-1] | [22,-1] | [23,-1]

        # In group 1, L4 owns two valid items; L5..L7 each own one valid item and one default.
        # The caller computes each group's valid count and explicitly selects its tile address.
    """
    dtype = source.dtype
    if default is not None and valid_items is None:
        raise ValueError("default requires valid_items")
    fill = 0 if default is None else default
    width = _resolve_warp_width(width, "warp_load width")
    if not isinstance(items_per_thread, int) or isinstance(items_per_thread, bool) or items_per_thread < 1:
        raise ValueError("items_per_thread must be a positive Python int")
    if not isinstance(algorithm, WarpLoadAlgorithm):
        raise TypeError("algorithm must be a WarpLoadAlgorithm")
    tid = _linear_thread_id(known_block_size())
    lane = tid % width
    if algorithm is WarpLoadAlgorithm.VECTORIZE:
        return _load_vectorized(
            source,
            lane * items_per_thread,
            items_per_thread,
            offset,
            valid_items,
            _record_default(dtype, fill),
        )
    striped = algorithm in (WarpLoadAlgorithm.STRIPED, WarpLoadAlgorithm.TRANSPOSE)
    items = _from_items(
        [
            _load_item(
                source,
                i * width + lane if striped else lane * items_per_thread + i,
                offset,
                valid_items,
                _record_default(dtype, fill),
            )
            for i in range(items_per_thread)
        ]
    )
    if algorithm is WarpLoadAlgorithm.TRANSPOSE:
        indices = [lane * items_per_thread + i for i in range(items_per_thread)]
        sources = Vector.from_elements([i % width * items_per_thread + i // width for i in indices])
        items = _warp_gather(items, sources, width=width)
    return items
