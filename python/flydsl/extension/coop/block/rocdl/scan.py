# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockScan``: the warp totals scanned across lanes with DPP.

The portable WARP_SCANS cross-warp phase has every thread read every warp
total and fold the ones in front of its warp (and all of them for the
aggregate): at 1024 threads, eight LDS reads and ~30 dependent folds per
thread, and an LDS return path that 16 waves saturate.

Here lane ``i`` of every 16-lane row reads total ``i`` only (one
``ds_read_b32``):

- up to 8 warps, two butterflies run side by side inside the row: one over all
  totals (the aggregate) and one over the totals in front of this warp (lanes
  ``>= warp_id`` contribute the identity). Pure VALU, lowest latency.
- 9 to 16 warps, a ``row_shr`` scan of the totals; the aggregate is lane
  ``num_warps - 1`` (``row_newbcast``) and the warp's prefix lane
  ``warp_id - 1``, fetched with one ``ds_bpermute`` so that step runs on the
  LDS pipe rather than the VALU 16 waves contend for.

The barrier count is unchanged. The override covers ReductionOp scans of
32-bit scalars with 8 to 16 wave64 warps on a target with ``row_newbcast`` --
both the default form and the guarded / callback forms, which share the
``_scan_warp_totals`` hook; everything else is the portable implementation.
Integers are bit-identical to it; floating-point prefixes are regrouped, as
the policy allows.
"""

from .....compiler import jit
from .....expr.gpu import barrier
from .....expr.numeric import Int32
from .....expr.rocdl import ds_bpermute
from ...._dispatch import override
from ..._common import _combine, _identity
from ..._values import _item_dtype, _shared_store
from ...warp.rocdl._common import _ALL_ROWS, _ROW_SHR, _dpp
from ...warp.rocdl.reduce import _butterfly_reduce
from ...warp.rocdl.scan import _from_lane
from .. import scan as _scan
from .reduce import _ROW, _lanes_apply, _pow2, _slot_of_lane

__all__ = ["BlockScan"]


def _row_scan(value, op, dtype, num_warps):
    """Inclusive scan over the first ``num_warps`` lanes of every row.

    ``row_shr`` never leaves the row and hands the lanes it shifts in from
    nowhere the identity, so a group that starts at the row needs no masks.
    """
    neutral = _identity(op, dtype)
    shift = 1
    while shift < num_warps:
        value = _combine(op, _dpp(value, _ROW_SHR[shift], _ALL_ROWS, neutral), value)
        shift <<= 1
    return value


def _bpermute(value, lane):
    raw = value if value.dtype is Int32 else value.bitcast(Int32)
    moved = Int32(ds_bpermute(Int32.ir_type, (lane << 2).ir_value(), raw.ir_value()))
    return moved if value.dtype is Int32 else moved.bitcast(value.dtype)


def _front_and_aggregate(slots, tid, warp_id, op, dtype, num_warps):
    """(fold of the totals of warps ``< warp_id``, fold of all totals), in every lane.

    Neither result is a bare DPP move, which GCNDPPCombine could fold into a
    caller's subtraction (``v_subrev_u32_dpp``, wrong on gfx942): the
    butterflies end in a combine, the broadcast keeps itself as *old*, and the
    front passes through a select.
    """
    index, total = _slot_of_lane(slots, tid, op, dtype, num_warps)
    neutral = _identity(op, dtype)
    if num_warps <= 8:
        group = _pow2(num_warps)
        front = (index < warp_id).select(total, neutral)
        return _butterfly_reduce(front, op, group), _butterfly_reduce(total, op, group)
    scanned = _row_scan(total, op, dtype, num_warps)
    aggregate = dtype(_from_lane(scanned.ir_value(), num_warps - 1, _ROW))
    # Lane w - 1 of the first row; warp 0 reads lane 15 and discards it.
    front = _bpermute(scanned, (warp_id + (_ROW - 1)) & (_ROW - 1))
    return (warp_id > 0).select(front, neutral), aggregate


@jit
def _prefix_warp_scans_lanes(partial, tid, slots, op, warp_scan_with_aggregate, warp_threads, num_warps):
    """(exclusive block prefix, block aggregate); thread 0's prefix is the identity."""
    inclusive, prefix, aggregate = warp_scan_with_aggregate(
        partial, op, width=warp_threads, _dtype=_item_dtype(partial)
    )
    warp_id = tid // warp_threads
    if tid % warp_threads == warp_threads - 1:
        _shared_store(slots, warp_id, inclusive)
    barrier()
    front, aggregate = _front_and_aggregate(slots, tid, warp_id, op, partial.dtype, num_warps)
    return _combine(op, front, prefix), aggregate


class BlockScan(_scan.BlockScan):
    __doc__ = _scan.BlockScan.__doc__

    # At 4 warps the portable form reads 3-4 totals cheaply (one form loses 2%
    # throughput there); from 8 warps on every form gains.
    _lane_min_warps = 8

    @override
    @classmethod
    def _scan_warp_totals(cls, partial, tid, slots, op):
        if cls.algorithm is _scan.BlockScanAlgorithm.WARP_SCANS and _lanes_apply(cls, partial, op, cls._lane_min_warps):
            return _prefix_warp_scans_lanes(
                partial, tid, slots, op, cls.warp_ops.warp_scan_with_aggregate, cls.warp_threads, cls.num_warps
            )
        return super()._scan_warp_totals(partial, tid, slots, op)
