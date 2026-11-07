# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide reduction — the portable form."""

from ....compiler import jit
from ....expr.arith import max as _max
from ....expr.arith import min as _min
from ....expr.gpu import lane_id, shuffle_down
from ....expr.numeric import Boolean, Float32, Int32, Integer, Numeric, Uint32
from ....expr.primitive import range_constexpr
from ....expr.struct import Empty
from ....expr.typing import ReductionOp
from .._common import (
    _combine,
    _resolve_warp_width,
    _shuffle_value,
    _thread_partial,
    _validate_valid_items,
)
from .._values import _as_items, _is_items, _normalize_value
from ._spec import WarpPrimitive
from .scan import _hillis_steele

__all__ = [
    "WarpReduce",
    "warp_reduce",
    "warp_sum",
    "warp_max",
    "warp_min",
    "warp_argmax",
    "warp_argmin",
    "warp_head_segmented_reduce",
    "warp_tail_segmented_reduce",
    "warp_head_segmented_sum",
    "warp_tail_segmented_sum",
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


def _argext_promote(item: Numeric) -> Numeric:
    """Promote narrow scalar numerics losslessly to 32-bit for True16-safe shuffles/DPP."""
    dtype = item.dtype
    if dtype is Boolean:
        return item.select(Int32(1), Int32(0))
    if dtype.width < 32:
        if dtype.is_float:
            return Float32(item)
        return Int32(item) if dtype.signed else Uint32(item)
    return item


def _argext_unpromote(value: Numeric, dtype: type[Numeric]) -> Numeric:
    if dtype is Boolean:
        return value != Int32(0)
    if value.dtype is dtype:
        return value
    return dtype(value)


def _combine_argext(
    is_max: bool,
    a_val: Numeric,
    a_idx: Int32,
    b_val: Numeric,
    b_idx: Int32,
) -> tuple[Numeric, Int32]:
    """Combine two ``(value, index)`` candidates with NaN propagation and lowest-index tie-break."""
    if a_val.dtype.is_float:
        a_nan = a_val != a_val
        b_nan = b_val != b_val
        b_better = (b_val > a_val) if is_max else (b_val < a_val)
        take_b = (b_nan & (~a_nan | (b_idx < a_idx))) | (~a_nan & (b_better | ((b_val == a_val) & (b_idx < a_idx))))
    else:
        b_better = (b_val > a_val) if is_max else (b_val < a_val)
        take_b = b_better | ((b_val == a_val) & (b_idx < a_idx))
    best_val = _max(a_val, b_val) if is_max else _min(a_val, b_val)
    best_idx = take_b.select(b_idx, a_idx)
    return best_val, best_idx


def _argext_local_fold(
    value: object,
    is_max: bool,
    width: int,
    dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32, type[Numeric], int]:
    """Validate and fold one lane's items into ``(promoted_val, local_blocked_idx, orig_dtype, count)``."""
    if dtype is not None and not issubclass(dtype, Numeric):
        raise TypeError("warp_argmax and warp_argmin require scalar Numeric elements")
    value = _normalize_value(value)
    items = _as_items(value, dtype)
    if not items or not isinstance(items[0], Numeric):
        raise TypeError("warp_argmax and warp_argmin require scalar Numeric elements")
    orig_dtype = items[0].dtype if dtype is None else dtype
    count = len(items)
    lane = lane_id() % width
    base_idx = lane if count == 1 else lane * count
    best_val = _argext_promote(orig_dtype(items[0]) if items[0].dtype is not orig_dtype else items[0])
    best_idx = Int32(base_idx)
    for j in range_constexpr(1, count):
        cand = orig_dtype(items[j]) if items[j].dtype is not orig_dtype else items[j]
        best_val, best_idx = _combine_argext(is_max, best_val, best_idx, _argext_promote(cand), Int32(base_idx + j))
    return best_val, best_idx, orig_dtype, count


@jit
def _argext_valid(
    value: object,
    is_max: bool,
    width: int,
    valid_items: int | Integer,
    dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32]:
    best_val, best_idx, orig_dtype, _ = _argext_local_fold(value, is_max, width, dtype)
    lane = lane_id() % width
    for stage in range_constexpr(width.bit_length() - 1):
        offset = 1 << stage
        other_val = _shuffle_value(best_val, offset, width, mode="up")
        other_idx = _shuffle_value(best_idx, offset, width, mode="up")
        if (lane < valid_items) & (lane >= offset):
            best_val, best_idx = _combine_argext(is_max, other_val, other_idx, best_val, best_idx)
    count = Int32(valid_items)
    last = (count > 0).select(count - 1, Int32(0))
    base = (lane_id() // width) * width
    out_val = _shuffle_value(best_val, base + last, width)
    out_idx = _shuffle_value(best_idx, base + last, width)
    return _argext_unpromote(out_val, orig_dtype), out_idx


def _portable_argext(
    value: object,
    is_max: bool,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32]:
    what = "warp_argmax width" if is_max else "warp_argmin width"
    width = _resolve_warp_width(width, what)
    _validate_valid_items(valid_items, width)
    if valid_items is not None:
        if _is_items(value, _dtype):
            raise TypeError("valid_items is supported only for a single item per lane")
        return _argext_valid(value, is_max, width, valid_items, _dtype)
    best_val, best_idx, orig_dtype, _ = _argext_local_fold(value, is_max, width, _dtype)
    offset = 1
    while offset < width:
        other_val = _shuffle_value(best_val, offset, width, mode="xor")
        other_idx = _shuffle_value(best_idx, offset, width, mode="xor")
        best_val, best_idx = _combine_argext(is_max, best_val, best_idx, other_val, other_idx)
        offset <<= 1
    return _argext_unpromote(best_val, orig_dtype), best_idx


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


def warp_sum(
    value,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Shorthand for ``warp_reduce(value, fx.ReductionOp.ADD, ...)``."""
    return warp_reduce(value, ReductionOp.ADD, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_max(
    value,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Shorthand for ``warp_reduce(value, fx.ReductionOp.MAX, ...)``."""
    return warp_reduce(value, ReductionOp.MAX, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_min(
    value,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Shorthand for ``warp_reduce(value, fx.ReductionOp.MIN, ...)``."""
    return warp_reduce(value, ReductionOp.MIN, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_argmax(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32]:
    """Return ``(max_value, winning_index)`` across a logical warp with lowest-index tie-breaking.

    All lanes of each logical warp must participate. Multi-item per-lane inputs use
    blocked linear group indices ``(lane_id() % width) * items_per_lane + item``.
    ``NaN`` inputs propagate matching ``warp_max`` and tie-break to the lowest ``NaN``
    index; equal values (including ``-0.0`` and ``+0.0``) select the lowest index.

    Args:
        value: Scalar ``Numeric`` or nonempty per-lane item range of scalar ``Numeric`` values.
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.
        valid_items: Uniform number of leading contributing lanes in ``[0, width]`` for
            single-item inputs, or ``None`` for all lanes. Empty groups have unspecified results.

    Returns:
        Tuple ``(max_value, winning_index)`` in every lane of the logical warp, where
        ``max_value`` has the input element type and ``winning_index`` is ``Int32``.
    """
    return _portable_argext(value, True, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_argmin(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32]:
    """Return ``(min_value, winning_index)`` across a logical warp with lowest-index tie-breaking.

    All lanes of each logical warp must participate. Multi-item per-lane inputs use
    blocked linear group indices ``(lane_id() % width) * items_per_lane + item``.
    ``NaN`` inputs propagate matching ``warp_min`` and tie-break to the lowest ``NaN``
    index; equal values (including ``-0.0`` and ``+0.0``) select the lowest index.

    Args:
        value: Scalar ``Numeric`` or nonempty per-lane item range of scalar ``Numeric`` values.
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.
        valid_items: Uniform number of leading contributing lanes in ``[0, width]`` for
            single-item inputs, or ``None`` for all lanes. Empty groups have unspecified results.

    Returns:
        Tuple ``(min_value, winning_index)`` in every lane of the logical warp, where
        ``min_value`` has the input element type and ``winning_index`` is ``Int32``.
    """
    return _portable_argext(value, False, width=width, valid_items=valid_items, _dtype=_dtype)


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


def warp_head_segmented_sum(
    value,
    head_flag: int | Integer,
    *,
    width: int | None = None,
    _dtype=None,
):
    """Shorthand for ``warp_head_segmented_reduce(value, head_flag, fx.ReductionOp.ADD, ...)``."""
    return warp_head_segmented_reduce(value, head_flag, ReductionOp.ADD, width=width, _dtype=_dtype)


def warp_tail_segmented_sum(
    value,
    tail_flag: int | Integer,
    *,
    width: int | None = None,
    _dtype=None,
):
    """Shorthand for ``warp_tail_segmented_reduce(value, tail_flag, fx.ReductionOp.ADD, ...)``."""
    return warp_tail_segmented_reduce(value, tail_flag, ReductionOp.ADD, width=width, _dtype=_dtype)


class WarpReduce(WarpPrimitive):
    """Reduce logical-warp inputs into their designated head lanes.

    Specialize with ``WarpReduce[dtype, width]``. The parameters below are compile-time
    positional arguments to ``WarpReduce[...]``, in bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype.
        width: Optional; defaults to None. A positive power-of-two Python int no larger than
            the target physical warp width, or None to use that physical width. This is the
            number of participating lanes in each logical warp, not the block thread count.

    Every lane in each logical warp must participate.
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
    def sum(
        cls,
        value,
        *,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Shorthand for ``reduce(value, fx.ReductionOp.ADD, ...)``."""
        value = cls._prepare(value)
        return cls._invoke(warp_sum, value, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

    @classmethod
    def max(
        cls,
        value,
        *,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Shorthand for ``reduce(value, fx.ReductionOp.MAX, ...)``."""
        value = cls._prepare(value)
        return cls._invoke(warp_max, value, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

    @classmethod
    def min(
        cls,
        value,
        *,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Shorthand for ``reduce(value, fx.ReductionOp.MIN, ...)``."""
        value = cls._prepare(value)
        return cls._invoke(warp_min, value, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

    @classmethod
    def argmax(
        cls,
        value: object,
        *,
        valid_items: int | Integer | None = None,
        storage: Empty | None = None,
    ) -> tuple[Numeric, Int32]:
        """Return ``(max_value, winning_index)`` with lowest-index tie-breaking."""
        if cls.dtype is not None and not issubclass(cls.dtype, Numeric):
            raise TypeError(f"WarpReduce.argmax requires a scalar Numeric dtype, got {cls.dtype.__name__}")
        value = cls._prepare(value)
        return cls._invoke(warp_argmax, value, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

    @classmethod
    def argmin(
        cls,
        value: object,
        *,
        valid_items: int | Integer | None = None,
        storage: Empty | None = None,
    ) -> tuple[Numeric, Int32]:
        """Return ``(min_value, winning_index)`` with lowest-index tie-breaking."""
        if cls.dtype is not None and not issubclass(cls.dtype, Numeric):
            raise TypeError(f"WarpReduce.argmin requires a scalar Numeric dtype, got {cls.dtype.__name__}")
        value = cls._prepare(value)
        return cls._invoke(warp_argmin, value, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

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

    @classmethod
    def head_segmented_sum(
        cls,
        value,
        head_flag: int | Integer,
        *,
        storage=None,
    ):
        """Shorthand for ``head_segmented_reduce(value, head_flag, fx.ReductionOp.ADD, ...)``."""
        value = cls._prepare(value)
        return cls._invoke(warp_head_segmented_sum, value, head_flag, storage=storage, _dtype=cls.dtype)

    @classmethod
    def tail_segmented_sum(
        cls,
        value,
        tail_flag: int | Integer,
        *,
        storage=None,
    ):
        """Shorthand for ``tail_segmented_reduce(value, tail_flag, fx.ReductionOp.ADD, ...)``."""
        value = cls._prepare(value)
        return cls._invoke(warp_tail_segmented_sum, value, tail_flag, storage=storage, _dtype=cls.dtype)
