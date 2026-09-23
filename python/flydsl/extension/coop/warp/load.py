# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Load tiles distributed across independent logical warps."""

import enum

from ....expr.gpu import lane_id
from ....expr.numeric import Integer
from ....expr.typing import Vector
from .._common import _validate_valid_items
from .._io import _load_item, _load_vectorized
from .._values import _from_items, _record_default
from ._spec import WarpPrimitive
from .exchange import _warp_gather

__all__ = [
    "WarpLoadAlgorithm",
    "WarpLoad",
]


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

    Direct = DIRECT
    Striped = STRIPED
    Vectorize = VECTORIZE
    Transpose = TRANSPOSE


class WarpLoad(WarpPrimitive):
    """Load a logical-warp tile with a fixed memory-access policy.

    Specialize with ``[dtype, width, items_per_thread, algorithm]``. The final
    policy is optional and defaults to ``WarpLoadAlgorithm.DIRECT``.
    ``None`` selects the target's physical warp width. All participating lanes
    use the same specialization and per-call options. Algorithm selection is
    part of the operator type; member functions do not accept an algorithm.
    STRIPED retains striped ownership; other policies use blocked ownership.
    SharedStorage is Empty with a zero-byte layout. The caller selects each
    logical warp's tile address explicitly with a view or offset.

    Examples:
        P = fx.coop.WarpLoad[fx.Int32, 8, 2]
        x = P.load(source, offset=(fx.thread_idx.x // 8) * 16)
    """

    _tile = True
    _algorithms = WarpLoadAlgorithm
    _default_algorithm = WarpLoadAlgorithm.DIRECT

    @classmethod
    def load(
        cls,
        source,
        *,
        offset: int | Integer = 0,
        valid_items: int | Integer | None = None,
        default=None,
        storage=None,
    ):
        """Load a tile for each logical warp with optional prefix guards.

        Return items in blocked order except for STRIPED policy. Invalid elements
        do not read source memory and use default when it is provided. Source plus offset is the calling
        logical warp's tile base; no warp offset is implicit. TRANSPOSE requires all
        logical-warp lanes to participate. Packed accesses fall back to scalar accesses
        for partial packs or unsupported layouts.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            source: Indexable source value with a dtype.
                With offset=0, its first element is the calling logical warp's tile base.
                May be a view already offset to that tile.
            offset: Element offset of the calling logical warp's tile in source. Include
                any grid, block, and warp displacement not already applied to source.
            valid_items: Valid prefix length measured in items, not lanes; None accesses the
                whole tile. The count is relative to this logical warp tile and must be
                uniform within the logical warp.
            default: Optional value used for invalid items, converted to the specialized dtype; a
                scalar fills all record leaves. Requires valid_items. Without an explicit
                default, invalid returned items are unspecified.

        Returns:
            A Vector for numeric elements or a tuple for plain Struct elements, with
            items_per_thread entries in the selected ownership layout and specialized dtype.

        Examples:
            # source contains [10,11,...,27]. The same source and offset select the SAME tile in both groups.
            # valid_items=6 applies separately to each group's eight-item tile.
            x = fx.coop.WarpLoad[fx.Int32, 4, 2, fx.coop.WarpLoadAlgorithm.DIRECT].load(source, valid_items=6, default=-1)
            # Group           |                group 0                ||                group 1
            # Data            | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
            # ----------------+---------+---------+---------+---------++---------+---------+---------+--------
            # x out (blocked) | [10,11] | [12,13] | [14,15] | [-1,-1] || [10,11] | [12,13] | [14,15] | [-1,-1]

            # Now choose different tiles: group 0 starts at source[2], group 1 at source[10].
            # The eight-thread block has 13 valid items: eight in group 0 and five in group 1.
            group_id = fx.thread_idx.x // 4
            striped_x = fx.coop.WarpLoad[fx.Int32, 4, 2, fx.coop.WarpLoadAlgorithm.STRIPED].load(source, offset=2 + group_id * 8, valid_items=fx.min(fx.max(13 - group_id * 8, 0), 8), default=-1)
            # Group          |                group 0                ||                group 1
            # Data           | L0      | L1      | L2      | L3      || L4      | L5      | L6      | L7
            # ---------------+---------+---------+---------+---------++---------+---------+---------+--------
            # source indices | [2,6]   | [3,7]   | [4,8]   | [5,9]   || [10,14] | [11,15] | [12,16] | [13,17]
            # valid slots    | [1,1]   | [1,1]   | [1,1]   | [1,1]   || [1,1]   | [1,0]   | [1,0]   | [1,0]
            # striped_x      | [12,16] | [13,17] | [14,18] | [15,19] || [20,24] | [21,-1] | [22,-1] | [23,-1]

            # In group 1, L4 owns two valid items; L5..L7 each own one valid item and one default.
            # The caller computes each group's valid count and explicitly selects its tile address.
        """
        cls._check_storage(storage)
        dtype = cls.dtype
        items_per_thread = cls.items_per_thread
        algorithm = cls.algorithm
        if default is not None and valid_items is None:
            raise ValueError("default requires valid_items")
        fill = 0 if default is None else default
        width = cls.warp_threads
        _validate_valid_items(valid_items, width * items_per_thread)
        lane = lane_id() % width
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
