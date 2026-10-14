# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register and shared-memory exchanges within independent logical warps."""

import enum

from ...._mlir.dialects import gpu
from ....expr.gpu import barrier, lane_id
from ....expr.struct import Struct
from ....expr.typing import Int32, Vector
from .._common import _resolve_warp_width
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
from ._spec import WarpPrimitive

__all__ = [
    "WarpExchangeAlgorithm",
    "WarpExchange",
]


class WarpExchangeAlgorithm(enum.Enum):
    """Select the transport for logical-warp item exchanges.

    Attributes:
        SHUFFLE: Move items through warp register shuffles without shared storage.
        Shared: Move items through disjoint per-group regions of caller-owned shared
            storage, with subgroup synchronization.
    """

    SHUFFLE = "shuffle"
    SHARED = "shared"

    Shuffle = SHUFFLE
    Shared = SHARED


def _check_algorithm(algorithm, storage):
    if not isinstance(algorithm, WarpExchangeAlgorithm):
        raise TypeError("algorithm must be a WarpExchangeAlgorithm")
    if algorithm is WarpExchangeAlgorithm.Shared and storage is None:
        raise TypeError("Shared requires storage allocated from WarpExchange.SharedStorage")


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
    if algorithm is WarpExchangeAlgorithm.Shared:
        lane = lane_id() % width
        for i, item in enumerate(items):
            _shared_store(storage.slots, lane * len(items) + i, item)
        barrier(scope=gpu.BarrierScope.Subgroup)
        out = [_shared_load(storage.slots, index) for index in requested]
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


class WarpExchange(WarpPrimitive):
    """Exchange blocked, striped and ranked logical-warp ownership.

    Specialize with ``[dtype, width, items_per_thread, algorithm]``. The final
    policy is optional and defaults to ``WarpExchangeAlgorithm.Shared``.
    ``None`` selects the target's physical warp width. All participating lanes
    use the same specialization and per-call options. Algorithm selection is
    part of the operator type; member functions do not accept an algorithm.
    Shared requires one SharedStorage instance per logical warp. SHUFFLE needs
    an empty SharedStorage and requires items_per_thread == width. Synchronize each group
    before reusing its scratch; scatter is supported only by Shared.

    Examples:
        P = fx.coop.WarpExchange[fx.Int32, 8, 2]
        # A 64-thread block contains eight independent logical warps.
        scratch = fx.SharedAllocator().allocate(fx.Array[P.SharedStorage, 8]).peek()
        storage = scratch[fx.thread_idx.x // 8]
        striped = P.blocked_to_striped(x, storage=storage)
        fx.barrier()
        blocked = P.striped_to_blocked(striped, storage=storage)
    """

    _tile = True
    _algorithms = WarpExchangeAlgorithm
    _default_algorithm = WarpExchangeAlgorithm.Shared

    @classmethod
    def _validate_specialization(cls):
        if cls.algorithm is WarpExchangeAlgorithm.SHUFFLE and cls.items_per_thread != cls.warp_threads:
            raise ValueError("SHUFFLE requires items_per_thread == width")

    @classmethod
    def _make_storage(cls):
        if cls.algorithm is WarpExchangeAlgorithm.Shared:
            return Struct["slots" : _shared_array(cls.dtype, cls.warp_threads * cls.items_per_thread)]
        return super()._make_storage()

    @classmethod
    def blocked_to_striped(
        cls,
        value,
        *,
        storage=None,
    ):
        """Convert blocked item ownership to striped within a logical warp.

        Blocked index is lane * items_per_thread + item; striped index is
        item * width + lane. All lanes of the logical warp must participate.

        Args:
            value: This lane's input value in blocked order.
            storage: View of this specialization's SharedStorage for the calling logical warp. Required for Shared; unused for SHUFFLE. Synchronize the logical group
                before reusing it.

        Returns:
            This lane's result value in striped order. A single input item is returned unchanged.

        Examples:
            Storage = fx.Array[fx.coop.WarpExchange[fx.Int32, 4, 2].SharedStorage, 2]
            storage = fx.SharedAllocator().allocate(Storage).peek()[fx.thread_idx.x // 4]
            y = fx.coop.WarpExchange[fx.Int32, 4, 2, fx.coop.WarpExchangeAlgorithm.Shared].blocked_to_striped(x, storage=storage)
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
            shared_y = fx.coop.WarpExchange[fx.Int32, 4, 2, fx.coop.WarpExchangeAlgorithm.Shared].blocked_to_striped(x, storage=storage)
            fx.barrier()  # Finish reading scratch before it is reused.
        """
        cls._check()
        algorithm = cls.algorithm
        width = cls.warp_threads
        _check_algorithm(algorithm, storage)
        value = cls._prepare(value)
        if not _is_items(value):
            return value
        lane = lane_id() % width
        indices = Vector.from_elements([lane + i * width for i in range(len(_as_items(value)))])
        return _warp_gather(value, indices, width=width, algorithm=algorithm, storage=storage)

    @classmethod
    def striped_to_blocked(
        cls,
        value,
        *,
        storage=None,
    ):
        """Convert striped item ownership to blocked within a logical warp.

        Blocked index is lane * items_per_thread + item; striped index is
        item * width + lane. All lanes of the logical warp must participate.

        Args:
            value: This lane's input value in striped order.
            storage: View of this specialization's SharedStorage for the calling logical warp. Required for Shared; unused for SHUFFLE. Synchronize the logical group
                before reusing it.

        Returns:
            This lane's result value in blocked order. A single input item is returned unchanged.

        Examples:
            Storage = fx.Array[fx.coop.WarpExchange[fx.Int32, 4, 2].SharedStorage, 2]
            storage = fx.SharedAllocator().allocate(Storage).peek()[fx.thread_idx.x // 4]
            y = fx.coop.WarpExchange[fx.Int32, 4, 2, fx.coop.WarpExchangeAlgorithm.Shared].striped_to_blocked(x, storage=storage)
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
            shared_y = fx.coop.WarpExchange[fx.Int32, 4, 2, fx.coop.WarpExchangeAlgorithm.Shared].striped_to_blocked(x, storage=storage)
            fx.barrier()  # Finish reading scratch before it is reused.
        """
        cls._check()
        algorithm = cls.algorithm
        width = cls.warp_threads
        _check_algorithm(algorithm, storage)
        value = cls._prepare(value)
        if not _is_items(value):
            return value
        lane = lane_id() % width
        count = len(_as_items(value))
        indices = [lane * count + i for i in range(count)]
        sources = Vector.from_elements([(i % width) * count + i // width for i in indices])
        return _warp_gather(value, sources, width=width, algorithm=algorithm, storage=storage)

    @classmethod
    def scatter_to_striped(
        cls,
        value,
        ranks,
        *,
        storage=None,
    ):
        """Scatter a permutation of logical-warp tile ranks into striped ownership.

        Ranks over the entire group must be a permutation of the tile indices, without
        duplicates or out-of-range destinations. All logical-warp lanes participate.

        Args:
            value: Input value containing nonempty items of the same type.
            ranks: Integer Vector with one destination rank per item, in [0, width *
                items_per_thread).
            storage: View of this specialization's SharedStorage for the calling logical warp. Required for Shared; unused for SHUFFLE. Synchronize the logical group
                before reusing it.

        Returns:
            A result value whose slot i in lane l contains the input with rank i * width + l.

        Examples:
            # Each input item supplies its destination rank in the group's logical sequence.
            # The ranks form an arbitrary permutation of 0..7 in EACH group; no duplicates are allowed.
            Storage = fx.Array[fx.coop.WarpExchange[fx.Int32, 4, 2].SharedStorage, 2]
            storage = fx.SharedAllocator().allocate(Storage).peek()[fx.thread_idx.x // 4]
            y = fx.coop.WarpExchange[fx.Int32, 4, 2, fx.coop.WarpExchangeAlgorithm.Shared].scatter_to_striped(x, ranks, storage=storage)
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
            shared_y = fx.coop.WarpExchange[fx.Int32, 4, 2, fx.coop.WarpExchangeAlgorithm.Shared].scatter_to_striped(x, ranks, storage=storage)
            fx.barrier()  # Finish reading scratch before it is reused.
        """
        cls._check()
        algorithm = cls.algorithm
        width = cls.warp_threads
        _check_algorithm(algorithm, storage)
        if algorithm is WarpExchangeAlgorithm.SHUFFLE:
            raise ValueError("Scatter to striped supports only Shared")
        value = cls._prepare(value)
        items = _as_items(value)
        if not isinstance(ranks, Vector) or ranks.numel != len(items) or ranks.dtype.is_float:
            raise TypeError("ranks must be an integer Vector with one rank per input item")
        lane = lane_id() % width
        for i, item in enumerate(items):
            _shared_store(storage.slots, ranks[i], item)
        barrier(scope=gpu.BarrierScope.Subgroup)
        return _from_items(
            [_shared_load(storage.slots, lane + i * width) for i in range(len(items))],
            like=value,
        )
