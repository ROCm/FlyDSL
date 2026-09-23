# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Batched reductions with output ownership distributed across lanes."""

from ....expr.gpu import lane_id, num_warp_threads
from .._common import _combine, _resolve_warp_width, _select_value, _shuffle_value
from .._values import _as_items, _from_items, _is_items
from ._spec import WarpPrimitive

__all__ = [
    "WarpReduceBatched",
    "warp_reduce_batched",
    "warp_reduce_batched_to_blocked",
    "warp_reduce_batched_to_striped",
]


def _reduce_batched(value, op, width, output_layout, sync_physical_warp=False, dtype=None):
    width = _resolve_warp_width(width, "warp_reduce_batched width")
    if not isinstance(sync_physical_warp, bool):
        raise TypeError("sync_physical_warp must be a Python bool")
    if output_layout not in ("scalar", "blocked", "striped"):
        raise ValueError("output_layout must be scalar, blocked, or striped")
    if not _is_items(value, dtype):
        raise TypeError("warp_reduce_batched expects a Vector or a fixed-size tuple/list")
    items = list(_as_items(value, dtype)) if len(value) else []
    batches = len(items)
    if output_layout == "scalar" and not 1 <= batches <= width:
        raise ValueError("scalar batched reduction requires 1 <= batches <= width; use a distributed layout")
    if not batches:
        return ()
    count = (batches + width - 1) // width
    lane = lane_id() % width
    shuffle_width = num_warp_threads() if sync_physical_warp else width

    def _tree(base, previous_stride):
        # Recursively exchange batches against lanes: each round
        # halves the number of live register columns and doubles the number of
        # input lanes represented by each remaining column.
        batch = base // width + (base % width) * count if output_layout == "blocked" else base
        if batch >= batches:
            return items[-1]
        if previous_stride == 1:
            return items[batch]
        stride = previous_stride // 2
        left = _tree(base, stride)
        right = _tree(base + stride, stride)
        is_left = lane % previous_stride < stride
        kept = _select_value(is_left, left, right)
        exchanged = _shuffle_value(_select_value(is_left, right, left), stride, shuffle_width, mode="xor")
        # The extra conditional swap preserves operand order for associative,
        # noncommutative operators without changing batch ownership.
        return _combine(
            op,
            _select_value(is_left, kept, exchanged),
            _select_value(is_left, exchanged, kept),
        )

    outputs = [_tree(slot * width, width) for slot in range(count)]
    return outputs[0] if output_layout == "scalar" else _from_items(outputs, like=value)


def warp_reduce_batched(
    value,
    op,
    *,
    width: int | None = None,
    sync_physical_warp: bool = False,
    _dtype=None,
):
    """Reduce register columns and distribute the batch aggregates across lanes.

    Every lane of the logical warp must participate. Batch i is reduced in ascending
    lane order. Only lane i owns the result of batch i; lanes beyond the batch count
    have unspecified results. Requires 1 <= batches <= width.

    Args:
        value: Input value whose item i is this lane's contribution
            to batch i. All lanes provide the same number of batches.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        sync_physical_warp: Whether the reduction uses physical-warp shuffle width. True
            requires full physical-warp participation; False allows independent logical
            warps.

    Returns:
        One result value per lane; logical lane i owns batch i. Use
        warp_reduce_batched_to_blocked or warp_reduce_batched_to_striped when the
        batch count exceeds width or is zero.

    Examples:
        # Item 0 is batch 0; item 1 is batch 1. Reduce columns independently within each group.
        # Scalar output gives batch i to group-relative lane i; lanes 2 and 3 have no batch.
        y = fx.coop.warp_reduce_batched(x, fx.ReductionOp.ADD, width=4)
        # Group |              group 0              ||              group 1
        # Data  | L0     | L1     | L2     | L3     || L4     | L5     | L6     | L7
        # ------+--------+--------+--------+--------++--------+--------+--------+-------
        # x in  | [1,10] | [2,20] | [3,30] | [4,40] || [5,50] | [6,60] | [7,70] | [8,80]
        # y out | 10     | 100    | ?      | ?      || 26     | 260    | ?      | ?

        # A physical-warp shuffle width changes participation requirements, not reduction groups.
        physical_sync = fx.coop.warp_reduce_batched_to_striped(
            x, fx.ReductionOp.ADD, width=4, sync_physical_warp=True
        )
        # Group         |         group 0          ||         group 1
        # Data          | L0   | L1    | L2  | L3  || L4   | L5    | L6  | L7
        # --------------+------+-------+-----+-----++------+-------+-----+----
        # physical_sync | [10] | [100] | [?] | [?] || [26] | [260] | [?] | [?]
    """
    return _reduce_batched(value, op, width, "scalar", sync_physical_warp, _dtype)


def warp_reduce_batched_to_blocked(
    value,
    op,
    *,
    width: int | None = None,
    sync_physical_warp: bool = False,
    _dtype=None,
):
    """Reduce register columns into blocked batch ownership.

    All lanes of each logical warp participate; batch contributions retain ascending lane
    order. Slots beyond the batch count are unspecified.

    Args:
        value: Input value whose item i is this lane's contribution
            to batch i. All lanes provide the same number of batches.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        sync_physical_warp: Whether the reduction uses physical-warp shuffle width. True
            requires full physical-warp participation; False allows independent logical
            warps.

    Returns:
        Result values with ceil(batches / width) items per lane, or an empty result for zero batches.
        Lane l, slot j owns batch l * ceil(batches / width) + j.

    Examples:
        # Five batches: in lane L, item b is (L+1)*(b+1), for the eight lanes shown.
        # group 0 batch totals = [10,20,30,40,50]; group 1 = [26,52,78,104,130].
        # Group |                           group 0                           ||                                group 1
        # Data  | L0          | L1           | L2            | L3             || L4              | L5              | L6              | L7
        # ------+-------------+--------------+---------------+----------------++-----------------+-----------------+-----------------+----------------
        # x in  | [1,2,3,4,5] | [2,4,6,8,10] | [3,6,9,12,15] | [4,8,12,16,20] || [5,10,15,20,25] | [6,12,18,24,30] | [7,14,21,28,35] | [8,16,24,32,40]

        # ceil(5/4)=2 slots per lane. Positions beyond the five batches are unspecified.
        y = fx.coop.warp_reduce_batched_to_blocked(x, fx.ReductionOp.ADD, width=4)
        # Group         |              group 0               ||               group 1
        # Data          | L0      | L1      | L2     | L3    || L4      | L5       | L6      | L7
        # --------------+---------+---------+--------+-------++---------+----------+---------+------
        # batch indices | [0,1]   | [2,3]   | [4,?]  | [?,?] || [0,1]   | [2,3]    | [4,?]   | [?,?]
        # y out         | [10,20] | [30,40] | [50,?] | [?,?] || [26,52] | [78,104] | [130,?] | [?,?]

        # Local lane l, slot j owns batch 2*l+j: each lane gets consecutive batch indices.

        # Synchronizing shuffles over the full physical warp keeps exactly the same per-group results.
        physical_sync = fx.coop.warp_reduce_batched_to_blocked(
            x, fx.ReductionOp.ADD, width=4, sync_physical_warp=True
        )
        # physical_sync has the same lane/slot results as y, including its unspecified slots.
    """
    return _reduce_batched(value, op, width, "blocked", sync_physical_warp, _dtype)


def warp_reduce_batched_to_striped(
    value,
    op,
    *,
    width: int | None = None,
    sync_physical_warp: bool = False,
    _dtype=None,
):
    """Reduce register columns into striped batch ownership.

    All lanes of each logical warp participate; batch contributions retain ascending lane
    order. Slots beyond the batch count are unspecified.

    Args:
        value: Input value whose item i is this lane's contribution
            to batch i. All lanes provide the same number of batches.
        op: ReductionOp or associative binary callable. Operand order follows ascending
            lanes; reassociation is allowed.
        width: Compile-time power-of-two logical width, at most the native warp width. None
            uses the native width.
        sync_physical_warp: Whether the reduction uses physical-warp shuffle width. True
            requires full physical-warp participation; False allows independent logical
            warps.

    Returns:
        Result values with ceil(batches / width) items per lane, or an empty result for zero batches.
        Lane l, slot j owns batch j * width + l.

    Examples:
        # Five batches: in lane L, item b is (L+1)*(b+1), for the eight lanes shown.
        # group 0 batch totals = [10,20,30,40,50]; group 1 = [26,52,78,104,130].
        # Group |                           group 0                           ||                                group 1
        # Data  | L0          | L1           | L2            | L3             || L4              | L5              | L6              | L7
        # ------+-------------+--------------+---------------+----------------++-----------------+-----------------+-----------------+----------------
        # x in  | [1,2,3,4,5] | [2,4,6,8,10] | [3,6,9,12,15] | [4,8,12,16,20] || [5,10,15,20,25] | [6,12,18,24,30] | [7,14,21,28,35] | [8,16,24,32,40]

        # ceil(5/4)=2 slots per lane. Positions beyond the five batches are unspecified.
        y = fx.coop.warp_reduce_batched_to_striped(x, fx.ReductionOp.ADD, width=4)
        # Group         |              group 0               ||               group 1
        # Data          | L0      | L1     | L2     | L3     || L4       | L5     | L6     | L7
        # --------------+---------+--------+--------+--------++----------+--------+--------+--------
        # batch indices | [0,4]   | [1,?]  | [2,?]  | [3,?]  || [0,4]    | [1,?]  | [2,?]  | [3,?]
        # y out         | [10,50] | [20,?] | [30,?] | [40,?] || [26,130] | [52,?] | [78,?] | [104,?]

        # Local lane l, slot j owns batch 4*j+l: slot 0 spans batches 0..3; slot 1 starts at batch 4.

        # Synchronizing shuffles over the full physical warp keeps exactly the same per-group results.
        physical_sync = fx.coop.warp_reduce_batched_to_striped(
            x, fx.ReductionOp.ADD, width=4, sync_physical_warp=True
        )
        # physical_sync has the same lane/slot results as y, including its unspecified slots.
    """
    return _reduce_batched(value, op, width, "striped", sync_physical_warp, _dtype)


class WarpReduceBatched(WarpPrimitive):
    """Reduce independent register columns into distributed lane outputs.

    Specialize with ``[dtype, width, items_per_thread]``. ``None`` selects the target's
    physical warp width. Every lane in each logical warp must participate.
    SharedStorage is Empty: explicit storage has a zero-byte layout.
    There is no algorithm parameter.
    The corresponding ``warp_*`` functions infer dtype and tile extent.
    The tile extent is the number of independent batches. Zero batches are
    accepted by the distributed methods; scalar reduce requires 1..width batches.

    Examples:
        P = fx.coop.WarpReduceBatched[fx.Int32, 8, 2]
        result = P.reduce(x, fx.ReductionOp.ADD)
    """

    _tile = True
    _minimum_items = 0

    @classmethod
    def reduce(
        cls,
        value,
        op,
        *,
        sync_physical_warp: bool = False,
        storage=None,
    ):
        """Reduce register columns and distribute the batch aggregates across lanes.

        Every lane of the logical warp must participate. Batch i is reduced in ascending
        lane order. Only lane i owns the result of batch i; lanes beyond the batch count
        have unspecified results. Requires 1 <= batches <= width.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: Input value whose item i is this lane's contribution
                to batch i. All lanes provide the same number of batches.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            sync_physical_warp: Whether the reduction uses physical-warp shuffle width. True
                requires full physical-warp participation; False allows independent logical
                warps.

        Returns:
            One result value per lane; logical lane i owns batch i. Use
            warp_reduce_batched_to_blocked or warp_reduce_batched_to_striped when the
            batch count exceeds width or is zero.
        """
        value = cls._prepare(value)
        return cls._invoke(
            warp_reduce_batched, value, op, sync_physical_warp=sync_physical_warp, storage=storage, _dtype=cls.dtype
        )

    @classmethod
    def reduce_to_blocked(
        cls,
        value,
        op,
        *,
        sync_physical_warp: bool = False,
        storage=None,
    ):
        """Reduce register columns into blocked batch ownership.

        All lanes of each logical warp participate; batch contributions retain ascending lane
        order. Slots beyond the batch count are unspecified.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: Input value whose item i is this lane's contribution
                to batch i. All lanes provide the same number of batches.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            sync_physical_warp: Whether the reduction uses physical-warp shuffle width. True
                requires full physical-warp participation; False allows independent logical
                warps.

        Returns:
            Result values with ceil(batches / width) items per lane, or an empty result for zero batches.
            Lane l, slot j owns batch l * ceil(batches / width) + j.
        """
        value = cls._prepare(value)
        return cls._invoke(
            warp_reduce_batched_to_blocked,
            value,
            op,
            sync_physical_warp=sync_physical_warp,
            storage=storage,
            _dtype=cls.dtype,
        )

    @classmethod
    def reduce_to_striped(
        cls,
        value,
        op,
        *,
        sync_physical_warp: bool = False,
        storage=None,
    ):
        """Reduce register columns into striped batch ownership.

        All lanes of each logical warp participate; batch contributions retain ascending lane
        order. Slots beyond the batch count are unspecified.

        The logical width is fixed by the operator specialization.

        Args:
            storage: Optional instance of this specialization's empty SharedStorage.
                Allocate Array[SharedStorage, num_warps] with SharedAllocator,
                peek the array and pass this warp's element. None is also allowed.
            value: Input value whose item i is this lane's contribution
                to batch i. All lanes provide the same number of batches.
            op: ReductionOp or associative binary callable. Operand order follows ascending
                lanes; reassociation is allowed.
            sync_physical_warp: Whether the reduction uses physical-warp shuffle width. True
                requires full physical-warp participation; False allows independent logical
                warps.

        Returns:
            Result values with ceil(batches / width) items per lane, or an empty result for zero batches.
            Lane l, slot j owns batch j * width + l.
        """
        value = cls._prepare(value)
        return cls._invoke(
            warp_reduce_batched_to_striped,
            value,
            op,
            sync_physical_warp=sync_physical_warp,
            storage=storage,
            _dtype=cls.dtype,
        )
