# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockReduce``: the warp totals combined across lanes with DPP.

The portable warp-reduction policies stage one total per warp in shared memory
and then let *every* thread read *every* total and fold them itself. At 1024
threads (16 wave64 warps) that is four ``ds_read2_b64`` per wave -- 4 KiB of
LDS return data per wave for 64 bytes of totals -- and it is the LDS return
path, not the VALU or the barrier, that the phase waits on.

Here lane ``i`` of every 16-lane row reads total ``i`` only (one
``ds_read_b32``) and a ``log2(num_warps)``-step XOR butterfly inside the row
(``quad_perm`` / ``row_half_mirror`` / ``row_ror``) leaves the block total in
every lane. The barrier count is unchanged.

The arrival-order policy (WARP_REDUCTIONS_NONDETERMINISTIC) folds its totals in
slot order: that is one legal arrival order, and it drops the policy's counter
reset barrier and LDS atomic.

The override covers a ``ReductionOp`` over 32-bit scalars (what the DPP
sequences move) on a wave64 target with ``row_newbcast`` (gfx90a / gfx94x /
gfx95x), with 4 to 16 warps; everything else is the portable implementation.
Integers are bit-identical to it; floating-point totals are regrouped, as the
policies allow, and every thread of the block still receives the same value.
"""

from .....compiler import jit
from .....expr.gpu import barrier
from .....expr.numeric import Int32
from .....expr.typing import ReductionOp
from ...._dispatch import override
from ..._common import _identity
from ..._values import _item_dtype, _shared_load, _shared_store
from ...warp.rocdl._common import _dpp_applies, _has_row_newbcast
from ...warp.rocdl.reduce import _butterfly_reduce
from .. import reduce as _reduce

__all__ = ["BlockReduce"]

# DPP row width: every cross-lane step below stays inside one 16-lane row.
_ROW = 16


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _lanes_apply(cls, item, op, min_warps):
    """Whether the DPP cross-warp phase covers this specialization and call."""
    return (
        isinstance(op, ReductionOp)
        and min_warps <= cls.num_warps <= _ROW
        and cls.warp_threads == 64
        and _has_row_newbcast()
        and _dpp_applies(item)
    )


def _slot_of_lane(slots, tid, op, dtype, num_warps):
    """Lane ``i`` of each power-of-two group reads total ``i``; padding lanes hold the identity.

    Returns ``(index, total)``. The read is clamped, so it never touches
    storage past the last slot.
    """
    group = _pow2(num_warps)
    index = tid & (group - 1)
    if num_warps == group:
        return index, _shared_load(slots, index)
    total = _shared_load(slots, (index < num_warps).select(index, Int32(num_warps - 1)))
    return index, (index < num_warps).select(total, _identity(op, dtype))


@jit
def _reduce_warps_lanes(partial, tid, storage, op, warp_reduce, warp_threads, num_warps):
    aggregate = warp_reduce(partial, op, width=warp_threads, _dtype=_item_dtype(partial))
    if tid % warp_threads == 0:
        _shared_store(storage.slots, tid // warp_threads, aggregate)
    barrier()
    _, total = _slot_of_lane(storage.slots, tid, op, partial.dtype, num_warps)
    # The butterfly ends in a combine, never a bare DPP move: nothing the
    # caller does with the total can fold a move (GCNDPPCombine) into itself.
    return _butterfly_reduce(total, op, _pow2(num_warps))


class BlockReduce(_reduce.BlockReduce):
    __doc__ = _reduce.BlockReduce.__doc__

    # At 2-3 warps the portable form reads every total with one or two LDS
    # instructions and is as fast.
    _lane_min_warps = 4

    @override
    @classmethod
    def _reduce_warp_totals(cls, partial, tid, storage, op, nondeterministic, valid_threads):
        if valid_threads is None and isinstance(op, ReductionOp):
            if _lanes_apply(cls, partial, op, cls._lane_min_warps):
                return _reduce_warps_lanes(
                    partial, tid, storage, op, cls.warp_ops.warp_reduce, cls.warp_threads, cls.num_warps
                )
            # Slot order is one legal arrival order: no tickets, no counter reset.
            nondeterministic = False
        return super()._reduce_warp_totals(partial, tid, storage, op, nondeterministic, valid_threads)
