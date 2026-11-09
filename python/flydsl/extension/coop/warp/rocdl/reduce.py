# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 warp reductions: the full fold and the segmented ones."""

from ....._mlir import ir
from ....._mlir.dialects import arith as _arith
from ....._mlir.dialects import llvm as _llvm
from ....._mlir.dialects import math as _math
from .....compiler import jit
from .....expr.gpu import lane_id
from .....expr.numeric import Boolean, Int32, Integer
from .....expr.rocdl import ballot, readlane
from .....expr.typing import ReductionOp
from ..._common import (
    _combine,
    _identity,
    _resolve_warp_width,
    _thread_partial,
)
from ..._values import _as_items
from .. import reduce as _portable
from ..reduce import warp_head_segmented_reduce as _portable_head_segmented_reduce
from ..reduce import warp_reduce as _portable_warp_reduce
from ..reduce import warp_tail_segmented_reduce as _portable_tail_segmented_reduce
from ._common import (
    _ALL_ROWS,
    _ROW_BCAST15,
    _ROW_BCAST31,
    _ROW_HALF_MIRROR,
    _ROW_NEWBCAST,
    _ROW_SHL,
    _ROW_SHR,
    _SWIZZLE_XOR16,
    _WAVE_SHL1,
    _XOR_DPP,
    _dpp,
    _dpp_applies,
    _has_row_newbcast,
    _replacing,
    _swizzle,
)

__all__ = ["WarpReduce"]


# What a butterfly step of each distance costs. Only the first two and the
# last are the XOR they stand for: ``quad_perm[1,0,3,2]`` and
# ``quad_perm[2,3,0,1]`` are XOR by 1 and 2 outright, and rotating a 16-lane
# row by 8 is XOR by 8 outright. ``row_half_mirror`` (lane ``i`` reads ``7 - i``
# within each group of 8) is not XOR by 4 -- but by the time a butterfly takes
# that step every quad already holds one value, so reading *any* lane of the
# neighbouring quad reads the right one, and the mirror does that.
_BUTTERFLY_DPP = {**_XOR_DPP, 4: _ROW_HALF_MIRROR}


# -- reduce -----------------------------------------------------------------


def _wave64_reduce(value, op):
    """Fold a full 64-lane wave with DPP, then broadcast out of an SGPR.

    Requires a commutative *op*, and in a different way from the portable
    butterfly: ``row_shr`` brings in the value from *below*, and it is folded
    in as ``_combine(op, acc, moved)`` -- the accumulator on the left. So the
    total lane 63 accumulates, and that ``readlane`` then broadcasts, is the
    wave folded in reverse lane order. Every lane agrees on it, unlike the
    butterfly below; it is simply the mirror image of the lane order a caller
    would expect. Writing ``_combine(op, moved, acc)`` instead would put it back
    in order at no cost -- the scan further down uses the same DPP sequence
    that way -- but the two are indistinguishable for the commutative ops this
    library accepts, so the flip is left for whoever needs it.
    """
    neutral = _identity(op, value.dtype)

    acc = value
    for shift in (1, 2, 4, 8):
        acc = _combine(op, acc, _dpp(acc, _ROW_SHR[shift], _ALL_ROWS, neutral))
    # Each 16-lane row now holds its own running fold, so its last lane -- 15,
    # 31, 47, 63 -- holds that row's total. ``row_bcast15`` hands each of those
    # to the row above (rows 1 and 3, hence row_mask 0xa) and ``row_bcast31``
    # hands lane 31 to the upper half (rows 2 and 3, row_mask 0xc), which
    # leaves lane 63 holding all four rows.
    acc = _combine(op, acc, _dpp(acc, _ROW_BCAST15, 0xA, neutral))
    acc = _combine(op, acc, _dpp(acc, _ROW_BCAST31, 0xC, neutral))

    return value.dtype(readlane(value.dtype.ir_type, acc, 63))


def _butterfly_reduce(value, op, width):
    """Fold a group narrower than the wave by XOR distance.

    The butterfly is what a narrow group needs anyway -- there is no lane the
    whole group can read a total out of the way lane 63 serves a full wave --
    and every one of its steps stays inside the group, so no step leaves a lane
    without a source and the identity ``_dpp`` takes is never read. It is
    passed for the other reason ``_dpp`` gives: it is what lets each move fold
    into the ``_combine`` that consumes it.

    Requires a commutative *op*, and needs it harder than the portable
    butterfly does. There, lane 0 at least folds the group in lane order; here
    the substitutes noted at ``_BUTTERFLY_DPP`` -- ``row_half_mirror`` for
    XOR 4, ``row_ror:8`` for XOR 8 -- read *a* lane of the neighbouring group
    rather than the matching one. Those lanes hold the same values as the
    matching lane but in a different order, which is exactly the thing
    commutativity makes invisible, so from width 8 up not even lane 0 folds in
    lane order. Restoring that would mean a different sequence, not a different
    operand order.
    """
    neutral = _identity(op, value.dtype)
    acc = value
    offset = 1
    while offset < width:
        if offset in _BUTTERFLY_DPP:
            partner = _dpp(acc, _BUTTERFLY_DPP[offset], _ALL_ROWS, neutral)
        else:
            partner = _swizzle(acc, _SWIZZLE_XOR16)
        acc = _combine(op, acc, partner)
        offset <<= 1
    return acc


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
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

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
    if valid_items is not None:
        return _portable_warp_reduce(value, op, width=width, valid_items=valid_items, _dtype=_dtype)
    # Dispatch before folding: a partial may itself be an item range, which
    # the portable entry point would otherwise interpret and reduce again.
    if not isinstance(op, ReductionOp) or not _dpp_applies(_as_items(value, _dtype)[0]):
        return _portable_warp_reduce(value, op, width=width, _dtype=_dtype)
    value = _thread_partial(value, op, _dtype)
    if width == 64:
        return _wave64_reduce(value, op)
    return _butterfly_reduce(value, op, width)


def warp_sum(value, *, width: int | None = None, valid_items: int | Integer | None = None, _dtype=None):
    """``warp_reduce`` with ``ReductionOp.ADD``."""
    return warp_reduce(value, ReductionOp.ADD, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_max(value, *, width: int | None = None, valid_items: int | Integer | None = None, _dtype=None):
    """``warp_reduce`` with ``ReductionOp.MAX``."""
    return warp_reduce(value, ReductionOp.MAX, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_min(value, *, width: int | None = None, valid_items: int | Integer | None = None, _dtype=None):
    """``warp_reduce`` with ``ReductionOp.MIN``."""
    return warp_reduce(value, ReductionOp.MIN, width=width, valid_items=valid_items, _dtype=_dtype)


# -- segmented reduce ------------------------------------------------------
#
# The portable form doubles a suffix through ``2 * log2(width)`` shuffles -- the
# value and the segment-end flag each take a ``ds_bpermute_b32`` per step -- so
# the whole fold sits on the LDS pipe. This one moves neither through LDS:
#
# - **The flags never move.** One ``ballot`` turns them into a 64-bit mask of
#   segment ends. It is uniform, so which lanes may fold at each step is SALU
#   arithmetic on that mask, handed to ``v_cndmask`` as the SGPR pair it is.
# - **Inside each 16-lane row the values move by ``row_shl``,** which reads the
#   lane ``k`` above.
# - **Between rows** a lane needs the *first* lane of the row above it, a
#   direction no gfx9 DPP control reaches in one move: ``wave_shl:1`` followed
#   by gfx90a's ``row_newbcast:15`` gets rows 0 and 2 theirs, and a full wave's
#   lower half reads lane 32 out of an SGPR with ``v_readlane``.
#
# Every fold is ``lower lanes ⊕ higher lanes``, so the operand order is exact
# and a callable need not commute.

# Rows 0 and 2 -- the ones whose next row is in the same 32-lane half.
_ROWS_0_2 = 0x5


def _i64(value):
    return _arith.constant(ir.IntegerType.get_signless(64), value - (1 << 64) if value >= 1 << 63 else value)


def _segment_ends(flag, width, *, head):
    """Bit ``j`` set when lane ``j`` ends a segment: one ``v_cmp`` and a little SALU.

    Every group's last lane ends a segment whatever its flag says. A head at
    lane ``j`` ends the segment at ``j - 1``, so the head form shifts its
    ballot down a lane: lane 0 of each group -- an implicit head -- lands on
    the previous group's last lane, which is set anyway, and bit 63 shifts in
    zero.
    """
    i64 = ir.IntegerType.get_signless(64)
    nonzero = Int32(flag != 0) != 0
    mask = ballot(i64, nonzero.ir_value())
    if head:
        mask = _arith.shrui(mask, _i64(1))
    group_ends = sum(1 << (base + width - 1) for base in range(0, 64, width))
    return _arith.ori(mask, _i64(group_ends))


def _inverse_ballot(mask):
    """This lane's bit of a uniform mask -- the SGPR pair itself, used as a lane mask."""
    i1 = ir.IntegerType.get_signless(1)
    return Boolean(_llvm.call_intrinsic(i1, "llvm.amdgcn.inverse.ballot.i64", [mask], [], []))


def _carries(lhs, rhs):
    """Whether ``lhs + rhs`` overflows 32 bits: one ``v_add_co_u32``, read from its carry-out."""
    i1, i32 = ir.IntegerType.get_signless(1), ir.IntegerType.get_signless(32)
    _, overflow = _arith.addui_extended(i32, i1, lhs.ir_value(), rhs.ir_value())
    return Boolean(overflow)


def _dist_to_end(ends, lane):
    """Lanes from this lane to the end of its segment, saturating at 32.

    Every comparison made against it is against at most 32, so the low word
    of ``ends >> lane`` is all it needs: one 64-bit shift and a ``v_ffbl``.
    """
    i32, i64 = ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64)
    shifted = _arith.shrui(ends, _arith.extui(i64, lane.ir_value()))
    return Int32(_math.cttz(_arith.trunci(i32, shifted)))


@jit
def _fold_if(take, op, acc, other):
    """``acc ⊕ other`` on the lanes that *take*, the op not evaluated elsewhere.

    A callable gets the portable form's guarantee: it only ever sees two
    pieces of one segment, and a lane that folds nothing never calls it. LLVM
    if-converts the branch back into a select when the op is cheap and pure.
    """
    if take:
        acc = _combine(op, acc, other)
    return acc


def _rows_with_identity(value, ends, op, width, neutral):
    """Suffix doubling inside each row, masked on the *sending* side (ReductionOp).

    Lane ``i`` may fold lane ``i + k`` only when no segment ends in
    ``[i, i + k - 1]`` -- equivalently, when no segment ends in the ``k``
    lanes below the sender. So lane ``j`` sends its value under that
    condition and the identity otherwise, and the fold itself is
    unconditional: it fuses with the move (``v_add_u32_dpp``), where masking
    the receiver would leave ``select(c, a + b, a)``, which instcombine turns
    into ``a + select(c, b, 0)`` and the move into a separate ``v_mov_dpp``.
    The masks are ``ends << 1`` smeared upwards by doubling -- two SALU
    instructions a step. A move off the end of the row reads the identity, so
    each lane ends up holding its segment up to the end of its row.
    """
    blocked = _arith.shli(ends, _i64(1))  # bit j: a segment ends in [j - k, j - 1]
    acc = value
    shift = 1
    while shift < min(width, 16):
        if shift > 1:
            blocked = _arith.ori(blocked, _arith.shli(blocked, _i64(shift // 2)))
        sent = _inverse_ballot(blocked).select(neutral, acc)
        acc = _combine(op, acc, _dpp(sent, _ROW_SHL[shift], _ALL_ROWS, neutral))
        shift <<= 1
    return acc


def _rows_guarded(value, ends, op, width):
    """The same doubling for a callable: masked on the *receiving* side, the op under ``if``.

    There is no identity to send, so lane ``i`` takes lane ``i + k`` when no
    segment ends in ``[i, i + k - 1]`` (``ends`` smeared downwards) and
    ``i + k`` is still in its row; a move off the row reads the lane's own
    value, which is never taken.
    """
    ends_within = ends  # bit i: a segment ends in [i, i + k - 1]
    acc = value
    shift = 1
    while shift < min(width, 16):
        if shift > 1:
            ends_within = _arith.ori(ends_within, _arith.shrui(ends_within, _i64(shift // 2)))
        in_row = (0xFFFF >> shift) * 0x0001_0001_0001_0001
        take = _arith.andi(_arith.xori(ends_within, _i64(-1 + (1 << 64))), _i64(in_row))
        acc = _fold_if(_inverse_ballot(take), op, acc, _dpp(acc, _ROW_SHL[shift], _ALL_ROWS, acc))
        shift <<= 1
    return acc


def _segmented_suffix(value, ends, op, width):
    """Each lane's fold up to its segment's end, in lane order; the head's is the aggregate.

    **Across rows** (32- and 64-lane groups, ``ReductionOp`` only -- see
    ``_segmented_applies``) a lane whose segment runs past its row needs the
    fold of the next row's first lane. For rows 0 and 2 ``wave_shl:1`` moves
    lane 16 (48) into lane 15 (47) and ``row_newbcast:15`` hands that to the
    whole row; then, for a full wave, the lower half reads lane 32 with
    ``v_readlane``. Whether a lane takes it is decided by its distance ``d`` to
    its segment end, as the carry-out of ``bias + d``: ``bias`` is
    ``(lane & 15) - 16`` on rows 0 and 2 (``lane - 32`` on the lower half for
    the second step) and 0 elsewhere, so the add carries exactly when the
    segment reaches the next row and never on a lane that must not take it.
    """
    if not isinstance(op, ReductionOp):
        return _rows_guarded(value, ends, op, width)
    neutral = _identity(op, value.dtype)
    acc = _rows_with_identity(value, ends, op, width, neutral)
    if width >= 32:
        lane = lane_id()
        dist = _dist_to_end(ends, lane)
        below = _dpp(_dpp(acc, _WAVE_SHL1, _ALL_ROWS, neutral), _ROW_NEWBCAST + 15, _ROWS_0_2, neutral)
        bias = ((lane & 16) == 0).select((lane & 15) - 16, Int32(0))
        acc = _carries(bias, dist).select(_combine(op, acc, below), acc)
        if width == 64:
            dtype = acc.dtype
            upper = dtype(readlane(dtype.ir_type, acc, 32))
            bias = (lane < 32).select(lane - 32, Int32(0))
            acc = _carries(bias, dist).select(_combine(op, acc, upper), acc)
    return acc


def _segmented_applies(value, op, dtype, width):
    """One 32-bit scalar item type on gfx9; records, vectors and other widths are portable.

    Decided before the items are folded, as in ``warp_reduce``: the portable
    entry point folds an item range itself.

    - Groups that span rows need ``row_newbcast``, which gfx90a introduced.
    - A callable runs under a branch at every step, where the portable form
      skips the op on a wave in which no lane folds. Up to 16 lanes the
      sequence still wins for any op measured; from 32 lanes up an expensive
      callable loses throughput to the portable form, so those keep it.
    """
    if not _dpp_applies(_as_items(value, dtype)[0]):
        return False
    if width >= 32 and not (isinstance(op, ReductionOp) and _has_row_newbcast()):
        return False
    return True


def warp_head_segmented_reduce(
    value,
    head_flag: int | Integer,
    op,
    *,
    width: int | None = None,
    _dtype=None,
):
    width = _resolve_warp_width(width, "warp_head_segmented_reduce width")
    if not _segmented_applies(value, op, _dtype, width):
        return _portable_head_segmented_reduce(value, head_flag, op, width=width, _dtype=_dtype)
    value = _thread_partial(value, op, _dtype)
    return _segmented_suffix(value, _segment_ends(head_flag, width, head=True), op, width)


def warp_tail_segmented_reduce(
    value,
    tail_flag: int | Integer,
    op,
    *,
    width: int | None = None,
    _dtype=None,
):
    width = _resolve_warp_width(width, "warp_tail_segmented_reduce width")
    if not _segmented_applies(value, op, _dtype, width):
        return _portable_tail_segmented_reduce(value, tail_flag, op, width=width, _dtype=_dtype)
    value = _thread_partial(value, op, _dtype)
    return _segmented_suffix(value, _segment_ends(tail_flag, width, head=False), op, width)


def warp_head_segmented_sum(value, head_flag: int | Integer, *, width: int | None = None, _dtype=None):
    """``warp_head_segmented_reduce`` with ``ReductionOp.ADD``."""
    return warp_head_segmented_reduce(value, head_flag, ReductionOp.ADD, width=width, _dtype=_dtype)


def warp_tail_segmented_sum(value, tail_flag: int | Integer, *, width: int | None = None, _dtype=None):
    """``warp_tail_segmented_reduce`` with ``ReductionOp.ADD``."""
    return warp_tail_segmented_reduce(value, tail_flag, ReductionOp.ADD, width=width, _dtype=_dtype)


# The contracts are the portable ones, and so is their documentation.
for _name, _doc in (
    ("warp_head_segmented_reduce", _portable_head_segmented_reduce),
    ("warp_tail_segmented_reduce", _portable_tail_segmented_reduce),
):
    globals()[_name].__doc__ = _doc.__doc__
del _name, _doc


class WarpReduce(_portable.WarpReduce):
    __doc__ = _portable.WarpReduce.__doc__

    _implementations = _replacing(
        _portable,
        warp_reduce,
        warp_sum,
        warp_max,
        warp_min,
        warp_head_segmented_reduce,
        warp_tail_segmented_reduce,
        warp_head_segmented_sum,
        warp_tail_segmented_sum,
    )
