# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide reduction — the portable form."""

from ....compiler import jit
from ....expr.gpu import lane_id, shuffle_down
from ....expr.numeric import Int32, Integer
from ....expr.primitive import range_constexpr
from ....expr.typing import ReductionOp
from .._common import (
    _combine,
    _resolve_warp_width,
    _shuffle_value,
    _thread_partial,
    _validate_valid_items,
)
from .._values import _as_items, _is_items
from ._spec import WarpPrimitive
from .scan import _hillis_steele

__all__ = [
    "WarpReduce",
    "warp_reduce",
    "warp_head_segmented_reduce",
    "warp_tail_segmented_reduce",
]


@jit
def _reduce_valid(value, op, width, valid_items, dtype=None):
    lane = lane_id() % width
    partial = _as_items(value, dtype)[0]
    if lane < valid_items:
        partial = _thread_partial(value, op, dtype)
    for stage in range_constexpr(width.bit_length() - 1):
        offset = 1 << stage
        # All lanes shuffle together; only valid prefixes evaluate the operator.
        other = _shuffle_value(partial, offset, width, mode="up")
        if (lane < valid_items) & (lane >= offset):
            partial = _combine(op, other, partial)
    count = Int32(valid_items)
    last = (count > 0).select(count - 1, Int32(0))
    base = (lane_id() // width) * width
    return _shuffle_value(partial, base + last, width)


@jit
def _segmented_reduce(value, tail_flag, op, width, dtype=None):
    """Suffix doubling: the aggregate is valid at each segment's head lane."""
    value = _thread_partial(value, op, dtype)
    lane = lane_id() % width
    ended = Int32(tail_flag != 0) | Int32(lane == width - 1)
    for stage in range_constexpr(width.bit_length() - 1):
        offset = 1 << stage
        other = _shuffle_value(value, offset, width, mode="down")
        other_end = shuffle_down(ended, offset, width)
        take = (ended == 0) & (lane + offset < width)
        # Shuffles stay converged; operators only consume this segment's data.
        if take:
            value = _combine(op, value, other)
        ended = take.select(other_end, ended)
    return value


def warp_reduce(
    value,
    op,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Reduce lane-local values and return the aggregate to every lane.

    All lanes of each logical warp must participate. A final partially populated
    warp is supported when valid_items gives its active lane count. Invalid lanes
    skip the binary operator. Empty groups have unspecified results.

    Args:
        value: This lane's input value. Local items must be nonempty. Reduction folds local items first, then
            combines lanes in blocked order. Use warp_reduce_batched for independent
            reductions of register columns.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Runtime counts must stay in range. Only the single-item overload accepts this argument;
            omit it when reducing a per-lane item range.

    Returns:
        The aggregate in every lane of the logical warp, including lanes outside
        valid_items. Empty groups have unspecified results.

    Examples:
        # Each group of four lanes reduces independently; every lane gets its aggregate.

        y = fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=4)

        # Group |      group 0      ||      group 1
        # Data  | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ------+----+----+----+----++----+----+----+---
        # x in  | 1  | 2  | 3  | 4  || 5  | 6  | 7  | 8
        # y out | 10 | 10 | 10 | 10 || 26 | 26 | 26 | 26

        # Array inputs fold all items across participating lanes into one value.

        # Include the first two lanes of each group: L0/L1 and L4/L5. All eight lanes call.
        partial = fx.coop.warp_reduce(
            x, fx.ReductionOp.ADD, width=4, valid_items=2
        )
        # L0..L3 receive 3; L4..L7 receive 11. valid_items counts contributing lanes in each group.

        # items in L0..L7 are {[1,10], [2,20], [3,30], [4,40],
        #                        [5,50], [6,60], [7,70], [8,80]}.
        total = fx.coop.warp_reduce(items, fx.ReductionOp.ADD, width=4)
        # L0..L3 receive 110; L4..L7 receive 286.
        # A guarded item range first folds its local items explicitly.
        local_total = items[0] + items[1]
        partial_tile = fx.coop.warp_reduce(
            local_total, fx.ReductionOp.ADD, width=4, valid_items=2
        )
        # L0..L3 receive 33; L4..L7 receive 121.
    """
    width = _resolve_warp_width(width, "warp_reduce width")
    _validate_valid_items(valid_items, width)
    if valid_items is not None:
        if _is_items(value, _dtype):
            raise TypeError("valid_items is supported only for a single item per lane")
        return _reduce_valid(value, op, width, valid_items, _dtype)

    value = _thread_partial(value, op, _dtype)
    if isinstance(op, ReductionOp):
        # Full tiles with built-in commutative operators keep the original
        # butterfly. Ordered scans/broadcasts are only needed by the extensions.
        offset = 1
        while offset < width:
            value = _combine(op, value, _shuffle_value(value, offset, width, mode="xor"))
            offset <<= 1
        return value

    raw = _hillis_steele(value, op, width)
    base = (lane_id() // width) * width
    return _shuffle_value(raw, base + width - 1, width)


def warp_head_segmented_reduce(
    value,
    head_flag: int | Integer,
    op,
    *,
    width: int | None = None,
    _dtype=None,
):
    """Reduce head-delimited segments within a logical warp.

    Lane zero implicitly starts a segment. Every lane in the logical warp must participate.
    Operand order within each segment follows ascending lanes.

    Args:
        value: This lane's input value. Local items are folded in order before
            combining lanes into one segment aggregate.
        head_flag: Nonzero when this lane's first item starts a segment.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.

    Returns:
        The segment aggregate at each segment's first lane. Results at other lanes are
        unspecified. A one-lane segment returns its locally folded items at that lane.

    Examples:
        # Split eight lanes into three segments. A nonzero head_flag starts a segment; sum the values in
        # each segment.

        y = fx.coop.warp_head_segmented_reduce(x, head_flag, fx.ReductionOp.ADD, width=8)

        # Data      | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7
        # ----------+----+----+----+----+----+----+----+---
        # x in      | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7
        # head_flag | 1  | 0  | 1  | 0  | 1  | 0  | 0  | 0
        # y out     | 1  | ?  | 5  | ?  | 22 | ?  | ?  | ?

        # Only segment heads L0, L2 and L4 own valid results. L0 implicitly starts the first segment.
    """
    width = _resolve_warp_width(width, "warp_head_segmented_reduce width")
    tail = shuffle_down(Int32(head_flag != 0), 1, width)
    return _segmented_reduce(value, tail, op, width, _dtype)


def warp_tail_segmented_reduce(
    value,
    tail_flag: int | Integer,
    op,
    *,
    width: int | None = None,
    _dtype=None,
):
    """Reduce tail-delimited segments within a logical warp.

    The last lane implicitly ends a segment. Every lane in the logical warp must participate.
    Operand order within each segment follows ascending lanes.

    Args:
        value: This lane's input value. Local items are folded in order before
            combining lanes into one segment aggregate.
        tail_flag: Nonzero when this lane's last item ends a segment.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.

    Returns:
        The segment aggregate at each segment's first lane. Results at other lanes are
        unspecified. A one-lane segment returns its locally folded items at that lane.

    Examples:
        # Four segments have lengths 2, 3, 2 and 1. L7 is an implicit tail even with tail_flag=0.
        y = fx.coop.warp_tail_segmented_reduce(x, tail_flag, fx.ReductionOp.MAX, width=8)

        # Data      | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7
        # ----------+----+----+----+----+----+----+----+---
        # x in      | 4  | 1  | 7  | 2  | 6  | 3  | 8  | 5
        # tail_flag | 0  | 1  | 0  | 0  | 1  | 0  | 1  | 0
        # segment   | 0  | 0  | 1  | 1  | 1  | 2  | 2  | 3
        # y out     | 4  | ?  | 7  | ?  | ?  | 8  | ?  | 5

        # Segments are L0..L1, L2..L4, L5..L6 and L7. Results belong to their HEADS:
        # L0 gets max(4,1)=4; L2 gets max(7,2,6)=7; L5 gets max(3,8)=8; L7 gets 5.
        # tail_flag=1 ENDS a segment; it does not make that lane a one-lane segment.
        # L1, L4 and L6 are tails of multi-lane segments, so their outputs are unspecified.
        # L7 is both head and tail of its one-lane segment, so its result is its input, 5.

        # Consecutive tails also form one-lane segments away from the group's last lane.
        singletons = fx.coop.warp_tail_segmented_reduce(
            x, single_tail_flag, fx.ReductionOp.MAX, width=8
        )

        # Data             | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7
        # -----------------+----+----+----+----+----+----+----+---
        # x in             | 4  | 1  | 7  | 2  | 6  | 3  | 8  | 5
        # single_tail_flag | 1  | 1  | 0  | 1  | 1  | 0  | 1  | 0
        # segment          | 0  | 1  | 2  | 2  | 3  | 4  | 4  | 5
        # singletons       | 4  | 1  | 7  | ?  | 6  | 8  | ?  | 5

        # L0, L1, L4 and L7 are one-lane segments and return 4, 1, 6 and 5 respectively.
    """
    width = _resolve_warp_width(width, "warp_tail_segmented_reduce width")
    return _segmented_reduce(value, tail_flag, op, width, _dtype)


class WarpReduce(WarpPrimitive):
    """Reduce logical-warp inputs into their designated head lanes.

    Specialize with ``[dtype, width=None]``. ``None`` selects the target's
    physical warp width. Every lane in each logical warp must participate.
    SharedStorage is Empty: explicit storage has a zero-byte layout.
    There is no algorithm parameter.
    The corresponding ``warp_*`` functions infer dtype and tile extent.
    The declared dtype is one complete element: a bare Vector is one element
    for a Vector dtype, and a scalar item range for a Numeric dtype. Outer
    lists/tuples contain complete elements. The result is always one dtype.

    Examples:
        P = fx.coop.WarpReduce[fx.Int32, 8]
        result = P.reduce(x, fx.ReductionOp.ADD)
    """

    @classmethod
    def reduce(
        cls,
        value,
        op,
        *,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Reduce lane-local values and return the aggregate to every lane.

        All lanes of each logical warp must participate. A final partially populated
        warp is supported when valid_items gives its active lane count. Invalid lanes
        skip the binary operator. Empty groups have unspecified results.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. Local items must be nonempty. Reduction folds local items first, then
                combines lanes in blocked order. Use warp_reduce_batched for independent
                reductions of register columns.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            valid_items: Uniform number of leading contributing lanes in [0, width], or None for
                all lanes. Runtime counts must stay in range. Only the single-item overload accepts this argument;
                omit it when reducing a per-lane item range.

        Returns:
            The aggregate in every lane of the logical warp, including lanes outside
            valid_items. Empty groups have unspecified results.
        """
        value = cls._prepare(value)
        return cls._invoke(warp_reduce, value, op, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

    @classmethod
    def head_segmented_reduce(
        cls,
        value,
        head_flag: int | Integer,
        op,
        *,
        storage=None,
    ):
        """Reduce head-delimited segments within a logical warp.

        Lane zero implicitly starts a segment. Every lane in the logical warp must participate.
        Operand order within each segment follows ascending lanes.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. Local items are folded in order before
                combining lanes into one segment aggregate.
            head_flag: Nonzero when this lane's first item starts a segment.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.

        Returns:
            The segment aggregate at each segment's first lane. Results at other lanes are
            unspecified. A one-lane segment returns its locally folded items at that lane.
        """
        value = cls._prepare(value)
        return cls._invoke(warp_head_segmented_reduce, value, head_flag, op, storage=storage, _dtype=cls.dtype)

    @classmethod
    def tail_segmented_reduce(
        cls,
        value,
        tail_flag: int | Integer,
        op,
        *,
        storage=None,
    ):
        """Reduce tail-delimited segments within a logical warp.

        The last lane implicitly ends a segment. Every lane in the logical warp must participate.
        Operand order within each segment follows ascending lanes.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. Local items are folded in order before
                combining lanes into one segment aggregate.
            tail_flag: Nonzero when this lane's last item ends a segment.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.

        Returns:
            The segment aggregate at each segment's first lane. Results at other lanes are
            unspecified. A one-lane segment returns its locally folded items at that lane.
        """
        value = cls._prepare(value)
        return cls._invoke(warp_tail_segmented_reduce, value, tail_flag, op, storage=storage, _dtype=cls.dtype)
