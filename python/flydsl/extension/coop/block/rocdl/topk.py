# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockTopK``: MERGE sorts with the dispatched merge sort; AIR counts with wave ballots.

AIR refines the threshold one bit at a time from the top, and the portable
form pays a block scan plus a barrier per bit for a single block-wide count.
With the bits above fixed at ``T``, the next bit of the k-th smallest ordered
key is set exactly when fewer than ``k`` valid items lie below ``T | bit`` --
the portable ``remaining > zeros`` test, since the items below ``T`` always
number ``k - remaining``. Each item's test is one ``v_cmp`` whose lane mask is
already the ballot, ``s_bcnt1`` counts it, and lane 0 of every wave stores
the count; the waves' counts meet in the AIR scratch words, two buffers by
bit parity, so a bit costs one barrier and no cross-lane VALU.

Selection needs the "less" and "equal" prefix counts in blocked order; both
come from one exchange of wave totals, with ``v_mbcnt`` counting the lanes
below. At most ``remaining`` equal items are taken, in input order, so the
selected prefix is ``less_before + min(equal_before, remaining)``. Ranks, and
therefore the output, are the portable ones bit for bit for
``0 <= k <= num_valid``.
"""

from ....._mlir import ir
from ....._mlir.dialects import arith as _arith
from ....._mlir.dialects import llvm as _llvm
from .....compiler import jit
from .....expr.gpu import barrier, lane_id
from .....expr.math import ctpop
from .....expr.numeric import Int32, Uint32
from .....expr.primitive import range_constexpr
from .....expr.rocdl import ballot
from .....expr.typing import Vector
from ...._dispatch import override
from ...warp.rocdl._common import _is_gfx9
from .. import topk as _portable

__all__ = ["BlockTopK"]


def _i64():
    return ir.IntegerType.get_signless(64)


def _wave_mask(pred):
    """The lanes of this wave where *pred* holds: a ``v_cmp`` result, already a uniform 64-bit mask."""
    return ballot(_i64(), pred.ir_value())


def _popcount(mask):
    """``s_bcnt1_i32_b64`` of a uniform mask."""
    return Int32(_arith.trunci(Int32.ir_type, ctpop(mask)))


def _count_below(mask, acc):
    """``acc`` plus the set bits of *mask* in lanes below this one: ``v_mbcnt_lo`` + ``v_mbcnt_hi``."""
    i32 = Int32.ir_type
    lo = _arith.trunci(i32, mask)
    hi = _arith.trunci(i32, _arith.shrui(mask, _arith.constant(_i64(), 32)))
    acc = _llvm.call_intrinsic(i32, "llvm.amdgcn.mbcnt.lo", [lo, acc.ir_value()], [], [])
    return Int32(_llvm.call_intrinsic(i32, "llvm.amdgcn.mbcnt.hi", [hi, acc], [], []))


def _wave_count(preds, valid_masks):
    """Number of this wave's (lane, item) pairs where ``preds[i]`` holds for a valid item: uniform."""
    counts = [_popcount(_arith.andi(_wave_mask(pred), mask)) for pred, mask in zip(preds, valid_masks)]
    return sum(counts[1:], counts[0])


@jit
def _publish_counts(counts, words, base, stride, lane):
    """Lane 0 of each wave stores the wave's counts at ``words[base + j * stride]``."""
    if lane == 0:
        for j in range_constexpr(counts.numel):
            words[base + j * stride] = Uint32(counts[j])


def _wave_totals(counts, words, base, warp, warps):
    """Publish every wave's *counts*; return (block totals, totals over lower waves). One barrier.

    ``words[base + j * warps + w]`` holds count ``j`` of wave ``w``.
    """
    _publish_counts(Vector.from_elements(counts), words, base + warp, warps, lane_id())
    barrier()
    totals, lower = [], []
    for j in range(len(counts)):
        parts = [Int32(words[base + j * warps + w]) for w in range(warps)]
        totals.append(sum(parts[1:], parts[0]))
        lower.append(sum(((warp > w).select(parts[w], Int32(0)) for w in range(warps - 1)), Int32(0)))
    return totals, lower


def _air_ranks(cls, ordered, k, validity, tid, storage):
    count, warps = cls.items_per_thread, cls.num_warps
    words = storage.words
    warp = tid // cls.warp_threads
    dtype = ordered.dtype
    valid = [validity[i] != 0 for i in range(count)]
    valid_masks = [_wave_mask(v) for v in valid]
    threshold = dtype(0)
    for bit in range(dtype.width - 1, -1, -1):
        below = _wave_count([ordered[i] < (threshold | dtype(1 << bit)) for i in range(count)], valid_masks)
        (total,), _ = _wave_totals([below], words, (bit % 2) * warps, warp, warps)
        threshold = (total < k).select(threshold | dtype(1 << bit), threshold)
    less = [valid[i] & (ordered[i] < threshold) for i in range(count)]
    equal = [valid[i] & (ordered[i] == threshold) for i in range(count)]
    less_masks = [_wave_mask(x) for x in less]
    equal_masks = [_wave_mask(x) for x in equal]
    less_below, equal_below = Int32(0), Int32(0)
    for i in range(count):
        less_below = _count_below(less_masks[i], less_below)
        equal_below = _count_below(equal_masks[i], equal_below)
    wave_less = sum((_popcount(m) for m in less_masks[1:]), _popcount(less_masks[0]))
    wave_equal = sum((_popcount(m) for m in equal_masks[1:]), _popcount(equal_masks[0]))
    (total_less, total_equal), (lower_less, lower_equal) = _wave_totals(
        [wave_less, wave_equal], words, 2 * warps, warp, warps
    )
    remaining = k - total_less
    less_before = lower_less + less_below
    equal_before = lower_equal + equal_below
    selected_count = total_less + (total_equal < remaining).select(total_equal, remaining)
    ranks = []
    for i in range(count):
        prefix = less_before + (equal_before < remaining).select(equal_before, remaining)
        selected = less[i] | (equal[i] & (equal_before < remaining))
        ranks.append(selected.select(prefix, selected_count + tid * count + i - prefix))
        less_before = less_before + Int32(less[i])
        equal_before = equal_before + Int32(equal[i])
    barrier()  # the exchange that follows reuses the words
    return Vector.from_elements(ranks)


class BlockTopK(_portable.BlockTopK):
    __doc__ = _portable.BlockTopK.__doc__

    @override
    @classmethod
    def _air_ranks(cls, ordered, k, validity, tid, storage):
        # Two count buffers and the two prefix totals per wave must fit the AIR words.
        if _is_gfx9() and 4 * cls.num_warps <= cls.block_threads * cls.items_per_thread:
            return _air_ranks(cls, ordered, k, validity, tid, storage)
        return super()._air_ranks(ordered, k, validity, tid, storage)
