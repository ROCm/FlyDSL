# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Ordered warp prefix scans and logical-warp broadcast."""

from ....compiler import jit
from ....expr.gpu import lane_id
from ....expr.numeric import Int32, Integer
from ....expr.primitive import range_constexpr
from ....expr.typing import ReductionOp
from .._common import (
    _cast_seed,
    _combine,
    _identity,
    _normalize_columns,
    _resolve_warp_width,
    _seed,
    _select_value,
    _shuffle_value,
    _validate_scalar_valid_items,
)
from .._values import _as_items, _from_items, _is_items, _item_dtype, _items_dtype
from ._spec import WarpPrimitive

__all__ = [
    "WarpScan",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
    "warp_broadcast",
]


def _shuffle_up(value, offset, width):
    return _shuffle_value(value, offset, width, mode="up"), lane_id() % width >= offset


def _shift_up(inclusive, op, width):
    shifted, valid = _shuffle_up(inclusive, 1, width)
    return _select_value(valid, shifted, _identity(op, _item_dtype(inclusive)))


def _broadcast_last(value, width):
    return warp_broadcast(value, width - 1, width=width)


def _hillis_steele(value, op, width):
    offset = 1
    while offset < width:
        shifted, valid = _shuffle_up(value, offset, width)
        value = _select_value(valid, _combine(op, shifted, value), value)
        offset <<= 1
    return value


@jit
def _scan_valid(value, op, width, valid_items):
    lane = lane_id() % width
    for stage in range_constexpr(width.bit_length() - 1):
        offset = 1 << stage
        # Keep all lanes in the shuffle, and evaluate the operator only for valid prefixes.
        shifted = _shuffle_value(value, offset, width, mode="up")
        if (lane < valid_items) & (lane >= offset):
            value = _combine(op, shifted, value)
    return value


def _raw(value, op, width, valid_items):
    value = _normalize_columns(value)
    dtype = _item_dtype(value)
    _validate_scalar_valid_items(value, valid_items, width, dtype)
    if valid_items is not None:
        return _scan_valid(value, op, width, valid_items)
    return _hillis_steele(value, op, width)


def _aggregate(raw, width, valid_items):
    if valid_items is None:
        return _broadcast_last(raw, width)
    count = Int32(valid_items)
    last = (count > 0).select(count - 1, Int32(0))
    # For an empty group, lane zero supplies an unspecified result.
    return warp_broadcast(raw, last, width=width)


@jit
def _seed_valid(raw, op, init, width, valid_items):
    result = raw
    if lane_id() % width < valid_items:
        result = _seed(raw, op, init)
    return result


def _inclusive(raw, op, width, init, valid_items):
    if init is None:
        return raw
    if valid_items is None:
        return _seed(raw, op, init)
    return _seed_valid(raw, op, init, width, valid_items)


@jit
def _seed_exclusive(shifted, op, init, width, valid_items):
    lane = lane_id() % width
    result = shifted
    if (lane > 0) & (lane < valid_items):
        result = _seed(shifted, op, init)
    # The first prefix is init itself: there is no input to combine with it.
    return _select_value(lane != 0, result, _cast_seed(shifted, init))


def _exclusive(raw, op, width, init, valid_items):
    if valid_items is None and isinstance(op, ReductionOp):
        return _seed(_shift_up(raw, op, width), op, init)
    shifted, _ = _shuffle_up(raw, 1, width)
    if init is None and op is ReductionOp.ADD:
        init = _item_dtype(raw)(0)
    if init is not None:
        return _seed_exclusive(shifted, op, init, width, width if valid_items is None else valid_items)
    # Without a seed, the first exclusive prefix is unspecified.
    return shifted


def _validate_scan(value, init, valid_items, width, dtype=None):
    dtype = _items_dtype(value) if dtype is None else dtype
    _validate_scalar_valid_items(value, valid_items, width, dtype)
    if _is_items(init, dtype):
        raise TypeError("scan init must be one item, not an item range")


@jit
def _join_array_prefix(local, prefix, op, init, width):
    result = local
    if lane_id() % width != 0:
        result = _combine(op, prefix, local)
    return _seed(result, op, init)


def _array_scan(value, op, width, init, dtype=None):
    """Scan a complete blocked tile, returning one unseeded aggregate."""
    items = _as_items(value, dtype)
    local = [items[0]]
    for item in items[1:]:
        local.append(_combine(op, local[-1], item))
    raw = _hillis_steele(local[-1], op, width)
    prefix, _ = _shuffle_up(raw, 1, width)
    inclusive = [_join_array_prefix(item, prefix, op, init, width) for item in local]
    exclusive = [_exclusive(raw, op, width, init, None), *inclusive[:-1]]
    return (
        _from_items(inclusive, like=value),
        _from_items(exclusive, like=value),
        _broadcast_last(raw, width),
    )


def warp_broadcast(
    value,
    source_lane: int | Integer,
    *,
    width: int | None = None,
    _dtype=None,
):
    """Read a value from a lane relative to the calling logical warp.

    All lanes of the logical warp must participate, and the source lane must be
    active. Each lane may select its own source index.

    Args:
        value: This lane's input value.
        source_lane: Source index in [0, width), relative to the logical group. Runtime
            indices must stay in range.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.

    Returns:
        The source lane's value, retaining its type and shape.

    Examples:
        # Source lane 2 means physical L2 in group 0 and physical L6 in group 1.
        y = fx.coop.warp_broadcast(x, 2, width=4)
        # Group |      group 0      ||      group 1
        # Data  | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ------+----+----+----+----++----+----+----+---
        # x in  | 10 | 20 | 30 | 40 || 50 | 60 | 70 | 80
        # y out | 30 | 30 | 30 | 30 || 70 | 70 | 70 | 70

        # Each lane may also choose its own source; repeated requests are allowed.
        selected = fx.coop.warp_broadcast(x, source_lane, width=4)
        # Group       |      group 0      ||      group 1
        # Data        | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ------------+----+----+----+----++----+----+----+---
        # source_lane | 3  | 0  | 3  | 1  || 2  | 3  | 0  | 2
        # selected    | 40 | 10 | 40 | 20 || 70 | 80 | 50 | 70
    """
    width = _resolve_warp_width(width, "warp_broadcast width")
    if isinstance(source_lane, int) and not 0 <= source_lane < width:
        raise ValueError("source_lane must be inside the logical warp")
    base = (lane_id() // width) * width
    return _shuffle_value(_normalize_columns(value), base + source_lane, width)


def warp_inclusive_scan(
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Compute the inclusive prefix in ascending lane order.

    All lanes of the logical warp participate; a final partial warp can specify its
    contributing count with valid_items. Invalid lanes do not evaluate the binary
    operator; their outputs and empty-group aggregates are unspecified. The
    aggregate excludes init. Without init, ADD has a zero first exclusive output;
    for other operators the first exclusive output is unspecified.

    Args:
        value: This lane's input value. An item range forms part of one blocked
            sequence: all of this lane's items precede the next lane's items.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        init: One initial item combined on the left of each prefix, converted
            to the input element type. Per-item seed ranges are rejected.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        This lane's inclusive prefix, with the input value type and shape.

    Examples:
        # Prefixes restart in each group. An exclusive prefix excludes the current lane.
        y = fx.coop.warp_inclusive_scan(x, fx.ReductionOp.ADD, width=4)
        # Group |      group 0      ||      group 1
        # Data  | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ------+----+----+----+----++----+----+----+---
        # x in  | 1  | 2  | 3  | 4  || 5  | 6  | 7  | 8
        # y out | 1  | 3  | 6  | 10 || 5  | 11 | 18 | 26

        # valid_items=3 masks L3 and L7. Each group starts with init=10.
        # Masked lanes do not evaluate the operator; ? denotes an unspecified output.
        seeded = fx.coop.warp_inclusive_scan(
            x, fx.ReductionOp.ADD, width=4, init=10, valid_items=3
        )
        # Group  |      group 0      ||      group 1
        # Data   | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # -------+----+----+----+----++----+----+----+---
        # seeded | 11 | 13 | 16 | ?  || 15 | 21 | 28 | ?

        # An explicit prefix derived from aggregate + 10: group 0 gets seed 16; group 1 gets seed 28.
        # The aggregate is the unseeded valid total (6 or 18).
        _, _, total = fx.coop.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=4, valid_items=3
        )

        carried = fx.coop.warp_inclusive_scan(
            x, fx.ReductionOp.ADD, width=4, valid_items=3,
            init=total + 10,
        )
        # Group   |      group 0      ||      group 1
        # Data    | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # --------+----+----+----+----++----+----+----+---
        # carried | 17 | 19 | 22 | ?  || 33 | 39 | 46 | ?
    """
    width = _resolve_warp_width(width, "warp_inclusive_scan width")
    _validate_scan(value, init, valid_items, width, _dtype)
    if _is_items(value, _dtype):
        return _array_scan(value, op, width, init, _dtype)[0]
    raw = _raw(value, op, width, valid_items)
    return _inclusive(raw, op, width, init, valid_items)


def warp_exclusive_scan(
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Compute the exclusive prefix in ascending lane order.

    All lanes of the logical warp participate; a final partial warp can specify its
    contributing count with valid_items. Invalid lanes do not evaluate the binary
    operator; their outputs and empty-group aggregates are unspecified. The
    aggregate excludes init. Without init, ADD has a zero first exclusive output;
    for other operators the first exclusive output is unspecified.

    Args:
        value: This lane's input value. An item range forms part of one blocked
            sequence: all of this lane's items precede the next lane's items.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        init: One initial item combined on the left of each prefix, converted
            to the input element type. Per-item seed ranges are rejected.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        This lane's exclusive prefix, with the input value type and shape.

    Examples:
        # Prefixes restart in each group. An exclusive prefix excludes the current lane.
        y = fx.coop.warp_exclusive_scan(x, fx.ReductionOp.ADD, width=4, init=0)
        # Group |      group 0      ||      group 1
        # Data  | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ------+----+----+----+----++----+----+----+---
        # x in  | 1  | 2  | 3  | 4  || 5  | 6  | 7  | 8
        # y out | 0  | 1  | 3  | 6  || 0  | 5  | 11 | 18

        # valid_items=3 masks L3 and L7. Each group starts with init=10.
        # Masked lanes do not evaluate the operator; ? denotes an unspecified output.
        seeded = fx.coop.warp_exclusive_scan(
            x, fx.ReductionOp.ADD, width=4, init=10, valid_items=3
        )
        # Group  |      group 0      ||      group 1
        # Data   | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # -------+----+----+----+----++----+----+----+---
        # seeded | 10 | 11 | 13 | ?  || 10 | 15 | 21 | ?

        # An explicit prefix derived from aggregate + 10: group 0 gets seed 16; group 1 gets seed 28.
        # The aggregate is the unseeded valid total (6 or 18).
        _, _, total = fx.coop.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=4, valid_items=3
        )

        carried = fx.coop.warp_exclusive_scan(
            x, fx.ReductionOp.ADD, width=4, valid_items=3,
            init=total + 10,
        )
        # Group   |      group 0      ||      group 1
        # Data    | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # --------+----+----+----+----++----+----+----+---
        # carried | 16 | 17 | 19 | ?  || 28 | 33 | 39 | ?
    """
    width = _resolve_warp_width(width, "warp_exclusive_scan width")
    _validate_scan(value, init, valid_items, width, _dtype)
    if _is_items(value, _dtype):
        return _array_scan(value, op, width, init, _dtype)[1]
    raw = _raw(value, op, width, valid_items)
    exclusive = _exclusive(raw, op, width, init, valid_items)
    return exclusive if valid_items is None else _select_value(lane_id() % width < valid_items, exclusive, value)


def warp_scan(
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Compute inclusive and exclusive prefixes with one scan.

    All lanes of the logical warp participate; a final partial warp can specify its
    contributing count with valid_items. Invalid lanes do not evaluate the binary
    operator; their outputs and empty-group aggregates are unspecified. The
    aggregate excludes init. Without init, ADD has a zero first exclusive output;
    for other operators the first exclusive output is unspecified.

    Args:
        value: This lane's input value. An item range forms part of one blocked
            sequence: all of this lane's items precede the next lane's items.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        init: One initial item combined on the left of each prefix, converted
            to the input element type. Per-item seed ranges are rejected.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        A tuple (inclusive, exclusive) of this lane's prefixes.

    Examples:
        # Prefixes restart in each group. An exclusive prefix excludes the current lane.
        inclusive, exclusive = fx.coop.warp_scan(x, fx.ReductionOp.ADD, width=4, init=0)
        # Group     |      group 0      ||      group 1
        # Data      | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ----------+----+----+----+----++----+----+----+---
        # x in      | 1  | 2  | 3  | 4  || 5  | 6  | 7  | 8
        # inclusive | 1  | 3  | 6  | 10 || 5  | 11 | 18 | 26
        # exclusive | 0  | 1  | 3  | 6  || 0  | 5  | 11 | 18

        # valid_items=3 masks L3 and L7. Each group starts with init=10.
        # Masked lanes do not evaluate the operator; ? denotes an unspecified output.
        seeded = fx.coop.warp_scan(
            x, fx.ReductionOp.ADD, width=4, init=10, valid_items=3
        )
        # Group               |      group 0      ||      group 1
        # Data                | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # --------------------+----+----+----+----++----+----+----+---
        # seeded[0] inclusive | 11 | 13 | 16 | ?  || 15 | 21 | 28 | ?
        # seeded[1] exclusive | 10 | 11 | 13 | ?  || 10 | 15 | 21 | ?

        # An explicit prefix derived from aggregate + 10: group 0 gets seed 16; group 1 gets seed 28.
        # The aggregate is the unseeded valid total (6 or 18).
        _, _, total = fx.coop.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=4, valid_items=3
        )

        carried = fx.coop.warp_scan(
            x, fx.ReductionOp.ADD, width=4, valid_items=3,
            init=total + 10,
        )
        # Group                |      group 0      ||      group 1
        # Data                 | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ---------------------+----+----+----+----++----+----+----+---
        # carried[0] inclusive | 17 | 19 | 22 | ?  || 33 | 39 | 46 | ?
        # carried[1] exclusive | 16 | 17 | 19 | ?  || 28 | 33 | 39 | ?
    """
    result = warp_scan_with_aggregate(value, op, width=width, init=init, valid_items=valid_items, _dtype=_dtype)
    return result[0], result[1]


def warp_scan_with_aggregate(
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
    """Compute both prefixes and the unseeded group aggregate.

    All lanes of the logical warp participate; a final partial warp can specify its
    contributing count with valid_items. Invalid lanes do not evaluate the binary
    operator; their outputs and empty-group aggregates are unspecified. The
    aggregate excludes init. Without init, ADD has a zero first exclusive output;
    for other operators the first exclusive output is unspecified.

    Args:
        value: This lane's input value. An item range forms part of one blocked
            sequence: all of this lane's items precede the next lane's items.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        init: One initial item combined on the left of each prefix, converted
            to the input element type. Per-item seed ranges are rejected.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        A tuple (inclusive, exclusive, aggregate). The aggregate is available in every
        participating lane and has the input element type, even for an item range.

    Examples:
        # Prefixes restart in each group. An exclusive prefix excludes the current lane.
        inclusive, exclusive, aggregate = fx.coop.warp_scan_with_aggregate(x, fx.ReductionOp.ADD, width=4, init=10)
        # Group     |      group 0      ||      group 1
        # Data      | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ----------+----+----+----+----++----+----+----+---
        # x in      | 1  | 2  | 3  | 4  || 5  | 6  | 7  | 8
        # inclusive | 11 | 13 | 16 | 20 || 15 | 21 | 28 | 36
        # exclusive | 10 | 11 | 13 | 16 || 10 | 15 | 21 | 28
        # aggregate | 10 | 10 | 10 | 10 || 26 | 26 | 26 | 26

        # The aggregate excludes init: group totals remain 10 and 26, despite the seed of 10.

        # valid_items=3 masks L3 and L7. Each group starts with init=10.
        # Masked lanes do not evaluate the operator; ? denotes an unspecified output.
        seeded = fx.coop.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=4, init=10, valid_items=3
        )
        # Group               |      group 0      ||      group 1
        # Data                | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # --------------------+----+----+----+----++----+----+----+---
        # seeded[0] inclusive | 11 | 13 | 16 | ?  || 15 | 21 | 28 | ?
        # seeded[1] exclusive | 10 | 11 | 13 | ?  || 10 | 15 | 21 | ?
        # seeded[2] aggregate | 6  | 6  | 6  | 6  || 18 | 18 | 18 | 18

        # An explicit prefix derived from aggregate + 10: group 0 gets seed 16; group 1 gets seed 28.
        # The aggregate is the unseeded valid total (6 or 18).
        _, _, total = fx.coop.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=4, valid_items=3
        )

        carried = fx.coop.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=4, valid_items=3,
            init=total + 10,
        )
        # Group                |      group 0      ||      group 1
        # Data                 | L0 | L1 | L2 | L3 || L4 | L5 | L6 | L7
        # ---------------------+----+----+----+----++----+----+----+---
        # carried[0] inclusive | 17 | 19 | 22 | ?  || 33 | 39 | 46 | ?
        # carried[1] exclusive | 16 | 17 | 19 | ?  || 28 | 33 | 39 | ?
        # carried[2] aggregate | 6  | 6  | 6  | 6  || 18 | 18 | 18 | 18

        # Per-lane arrays form a single blocked sequence within each group.
        tile = fx.coop.warp_scan_with_aggregate(items, fx.ReductionOp.ADD, width=4, init=0)
        # Data              | L0     | L1      | L2      | L3
        # ------------------+--------+---------+---------+---------
        # items in          | [1,10] | [2,20]  | [3,30]  | [4,40]
        # tile[0] inclusive | [1,11] | [13,33] | [36,66] | [70,110]
        # tile[1] exclusive | [0,1]  | [11,13] | [33,36] | [66,70]
        # tile[2] aggregate | 110    | 110     | 110     | 110
    """
    width = _resolve_warp_width(width, "warp_scan_with_aggregate width")
    _validate_scan(value, init, valid_items, width, _dtype)
    if _is_items(value, _dtype):
        return _array_scan(value, op, width, init, _dtype)
    if valid_items is None and isinstance(op, ReductionOp):
        # Keep the old construction order, including for callers that combine
        # all three results; otherwise LLVM can schedule the aggregate first.
        raw = _hillis_steele(_normalize_columns(value), op, width)
        inclusive = _seed(raw, op, init)
        exclusive = _seed(_shift_up(raw, op, width), op, init)
        return inclusive, exclusive, _broadcast_last(raw, width)
    raw = _raw(value, op, width, valid_items)
    aggregate = _aggregate(raw, width, valid_items)
    exclusive = _exclusive(raw, op, width, init, valid_items)
    if valid_items is not None:
        exclusive = _select_value(lane_id() % width < valid_items, exclusive, value)
    return _inclusive(raw, op, width, init, valid_items), exclusive, aggregate


class WarpScan(WarpPrimitive):
    """Compute logical-warp prefixes and broadcast lane values.

    Specialize with ``[dtype, width=None]``. ``None`` selects the target's
    physical warp width. Every lane in each logical warp must participate.
    SharedStorage is Empty: explicit storage has a zero-byte layout.
    There is no algorithm parameter.
    The corresponding ``warp_*`` functions infer dtype and tile extent.

    Outer item sequences (including Vector only for a Numeric dtype) form one
    complete blocked tile: each lane's items precede the next lane's items. Arrays
    require ``valid_items=None``, even with one item per lane. One element-valued
    ``init`` seeds the whole sequence; the aggregate is one unseeded element.
    A single complete dtype value, including Vector or Struct, contributes
    one item per lane and accepts
    ``valid_items`` as the number of contributing leading lanes.

    Examples:
        P = fx.coop.WarpScan[fx.Int32, 8]
        result = P.inclusive_scan(x, fx.ReductionOp.ADD)
    """

    @classmethod
    def inclusive_scan(
        cls,
        value,
        op,
        *,
        init=None,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Compute the inclusive prefix in ascending lane order.

        All lanes of the logical warp participate; a final partial warp can specify its
        contributing count with valid_items. Invalid lanes do not evaluate the binary
        operator; their outputs and empty-group aggregates are unspecified. The
        aggregate excludes init. Without init, ADD has a zero first exclusive output;
        for other operators the first exclusive output is unspecified.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. An item range forms part of one blocked
                sequence: all of this lane's items precede the next lane's items.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            init: One initial item combined on the left of each prefix, converted
                to the input element type. Per-item seed ranges are rejected.
            valid_items: Uniform number of leading contributing lanes in [0, width], or None for
                all lanes. Only single-item inputs accept this argument; omit it for
                an item range. Runtime counts must stay in range.

        Returns:
            This lane's inclusive prefix, with the input value type and shape.
        """
        value = cls._prepare(value)
        return cls._invoke(
            warp_inclusive_scan, value, op, init=init, valid_items=valid_items, storage=storage, _dtype=cls.dtype
        )

    @classmethod
    def exclusive_scan(
        cls,
        value,
        op,
        *,
        init=None,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Compute the exclusive prefix in ascending lane order.

        All lanes of the logical warp participate; a final partial warp can specify its
        contributing count with valid_items. Invalid lanes do not evaluate the binary
        operator; their outputs and empty-group aggregates are unspecified. The
        aggregate excludes init. Without init, ADD has a zero first exclusive output;
        for other operators the first exclusive output is unspecified.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. An item range forms part of one blocked
                sequence: all of this lane's items precede the next lane's items.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            init: One initial item combined on the left of each prefix, converted
                to the input element type. Per-item seed ranges are rejected.
            valid_items: Uniform number of leading contributing lanes in [0, width], or None for
                all lanes. Only single-item inputs accept this argument; omit it for
                an item range. Runtime counts must stay in range.

        Returns:
            This lane's exclusive prefix, with the input value type and shape.
        """
        value = cls._prepare(value)
        return cls._invoke(
            warp_exclusive_scan, value, op, init=init, valid_items=valid_items, storage=storage, _dtype=cls.dtype
        )

    @classmethod
    def scan(
        cls,
        value,
        op,
        *,
        init=None,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Compute inclusive and exclusive prefixes with one scan.

        All lanes of the logical warp participate; a final partial warp can specify its
        contributing count with valid_items. Invalid lanes do not evaluate the binary
        operator; their outputs and empty-group aggregates are unspecified. The
        aggregate excludes init. Without init, ADD has a zero first exclusive output;
        for other operators the first exclusive output is unspecified.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. An item range forms part of one blocked
                sequence: all of this lane's items precede the next lane's items.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            init: One initial item combined on the left of each prefix, converted
                to the input element type. Per-item seed ranges are rejected.
            valid_items: Uniform number of leading contributing lanes in [0, width], or None for
                all lanes. Only single-item inputs accept this argument; omit it for
                an item range. Runtime counts must stay in range.

        Returns:
            A tuple (inclusive, exclusive) of this lane's prefixes.
        """
        value = cls._prepare(value)
        return cls._invoke(warp_scan, value, op, init=init, valid_items=valid_items, storage=storage, _dtype=cls.dtype)

    @classmethod
    def scan_with_aggregate(
        cls,
        value,
        op,
        *,
        init=None,
        valid_items: int | Integer | None = None,
        storage=None,
    ):
        """Compute both prefixes and the unseeded group aggregate.

        All lanes of the logical warp participate; a final partial warp can specify its
        contributing count with valid_items. Invalid lanes do not evaluate the binary
        operator; their outputs and empty-group aggregates are unspecified. The
        aggregate excludes init. Without init, ADD has a zero first exclusive output;
        for other operators the first exclusive output is unspecified.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value. An item range forms part of one blocked
                sequence: all of this lane's items precede the next lane's items.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            init: One initial item combined on the left of each prefix, converted
                to the input element type. Per-item seed ranges are rejected.
            valid_items: Uniform number of leading contributing lanes in [0, width], or None for
                all lanes. Only single-item inputs accept this argument; omit it for
                an item range. Runtime counts must stay in range.

        Returns:
            A tuple (inclusive, exclusive, aggregate). The aggregate is available in every
            participating lane and has the input element type, even for an item range.
        """
        value = cls._prepare(value)
        return cls._invoke(
            warp_scan_with_aggregate, value, op, init=init, valid_items=valid_items, storage=storage, _dtype=cls.dtype
        )

    @classmethod
    def broadcast(
        cls,
        value,
        source_lane: int | Integer,
        *,
        storage=None,
    ):
        """Read a value from a lane relative to the calling logical warp.

        All lanes of the logical warp must participate, and the source lane must be
        active. Each lane may select its own source index.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: This lane's input value.
            source_lane: Source index in [0, width), relative to the logical group. Runtime
                indices must stay in range.

        Returns:
            The source lane's value, retaining its type and shape.
        """
        value = cls._prepare(value)
        return cls._invoke(warp_broadcast, value, source_lane, storage=storage, _dtype=cls.dtype)
