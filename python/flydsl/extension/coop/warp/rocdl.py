# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL overrides for the warp-scope collectives.

Covers both wave64 CDNA (``gfx9*``, using ``row_bcast`` / ``wave_shr`` /
``ds_swizzle``) and wave32 GFX10+/RDNA/CDNA5 (``gfx10*``, ``gfx11*``, ``gfx12*``,
using ``row_xmask`` / ``row_share`` / ``permlanex16``), plus native wave32/wave64
``ballot`` + ``mbcnt`` + ``ctpop``/``cttz`` overrides for predicate voting,
compaction ranking, and ``warp_argmax``/``warp_argmin``.
"""

from collections.abc import Callable

from ....compiler.backends import current_target
from ....expr.gpu import lane_id
from ....expr.math import ctpop, cttz
from ....expr.numeric import Boolean, Int32, Int64, Integer, Numeric, Uint32, Uint64
from ....expr.rocdl import ballot, ds_swizzle, mbcnt_hi, mbcnt_lo, permlanex16, readlane, update_dpp
from ....expr.typing import ReductionOp
from .._common import (
    _combine,
    _identity,
    _resolve_warp_width,
    _seed,
    _shuffle_value,
    _thread_partial,
    _validate_valid_items,
)
from .._values import _as_items, _normalize_value
from . import scan as _universal_scan
from .reduce import (
    _argext_local_fold,
    _argext_unpromote,
    _portable_argext,
)
from .reduce import warp_reduce as _portable_warp_reduce
from .vote import _as_predicate
from .vote import (
    warp_all as _portable_warp_all,
)
from .vote import (
    warp_any as _portable_warp_any,
)
from .vote import (
    warp_ballot as _portable_warp_ballot,
)
from .vote import (
    warp_ballot_rank as _portable_warp_ballot_rank,
)
from .vote import (
    warp_count as _portable_warp_count,
)

__all__ = [
    "warp_reduce",
    "warp_sum",
    "warp_max",
    "warp_min",
    "warp_argmax",
    "warp_argmin",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_inclusive_sum",
    "warp_exclusive_sum",
    "warp_scan",
    "warp_scan_with_aggregate",
    "warp_any",
    "warp_all",
    "warp_ballot",
    "warp_count",
    "warp_ballot_rank",
]


# DPP controls, spelled as the ISA encodes them.
_ROW_SHR = {1: 0x111, 2: 0x112, 4: 0x114, 8: 0x118}
_WAVE_SHR1 = 0x138
_ROW_BCAST15 = 0x142
_ROW_BCAST31 = 0x143
_ROW_SHARE7 = 0x157
_ROW_SHARE15 = 0x15F
_RDNA_XOR_DPP = {1: 0x161, 2: 0x162, 4: 0x164, 8: 0x168}
_QUAD_BCAST_PAIR = 0xF5  # quad_perm:[1, 1, 3, 3]
_QUAD_BCAST_LAST = 0xFF  # quad_perm:[3, 3, 3, 3]
_ALL_ROWS = 0xF
_ROW_MASK_ODD = 0xA
_ROW_MASK_HI2 = 0xC
_ALL_BANKS = 0xF
_BANK_MASK_LO2 = 0x3
_BANK_MASK_HI2 = 0xC
_PERMLANEX16_BCAST15 = 0xFFFFFFFF
_FULL_MASK32 = 0xFFFFFFFF
_FULL_MASK64 = 0xFFFFFFFFFFFFFFFF

# What a butterfly step of each distance costs. Only the first two and the
# last are the XOR they stand for: ``quad_perm[1,0,3,2]`` and
# ``quad_perm[2,3,0,1]`` are XOR by 1 and 2 outright, and rotating a 16-lane
# row by 8 is XOR by 8 outright. ``row_half_mirror`` (lane ``i`` reads ``7 - i``
# within each group of 8) is not XOR by 4 -- but by the time a butterfly takes
# that step every quad already holds one value, so reading *any* lane of the
# neighbouring quad reads the right one, and the mirror does that.
_BUTTERFLY_DPP = {1: 0xB1, 2: 0x4E, 4: 0x141, 8: 0x128}

# XOR by 16 crosses the 16-lane rows DPP is built around, and gfx9 has no
# control that does. ``ds_swizzle_b32`` in its 32-lane mode does: a lane reads
# ``((lane & and_mask) | or_mask) ^ xor_mask``, and the offset packs the three
# as ``xor_mask << 10 | or_mask << 5 | and_mask``.
_SWIZZLE_XOR16 = (16 << 10) | 0x1F


def _swizzle_broadcast(width: int) -> int:
    """Offset that hands every lane of a *width*-lane group its last lane.

    ``and_mask`` keeps the bits above the group, ``or_mask`` sets every bit
    inside it, so the source lane is the group's base plus ``width - 1``.
    """
    return ((width - 1) << 5) | (0x1F & ~(width - 1))


def _dpp_banks(value: Numeric, ctrl: int, row_mask: int, bank_mask: int, old: Numeric) -> Numeric:
    """One DPP move of *value* with explicit row and bank masks."""
    dtype = value.dtype
    moved = update_dpp(
        dtype.ir_type,
        old.ir_value(),
        value.ir_value(),
        ctrl,
        row_mask,
        bank_mask,
        False,
    )
    return dtype(moved)


def _dpp(value: Numeric, ctrl: int, row_mask: int, old: Numeric) -> Numeric:
    """One DPP move of *value*, with *old* wherever the move reaches nothing.

    ``bound_ctrl`` is off, which is what puts *old* there: the hardware's own
    out-of-range fill writes zero, and zero is the identity of ``ADD`` alone.
    Every caller passes the identity of the *op* it is folding with, so the
    lanes a move does not reach drop out of that fold whatever the op is. Turning
    ``bound_ctrl`` on to save the operand would silently break ``MUL``, ``MAX``
    and ``MIN``.
    """
    return _dpp_banks(value, ctrl, row_mask, _ALL_BANKS, old)


def _permlanex16(value: Numeric, sel_lo: int, sel_hi: int, old: Numeric) -> Numeric:
    """Cross-row 16-lane exchange on wave32 GFX10+/RDNA/CDNA5 targets."""
    dtype = value.dtype
    raw = value if dtype is Int32 else value.bitcast(Int32)
    raw_old = old if dtype is Int32 else old.bitcast(Int32)
    s1 = Uint32(sel_lo).bitcast(Int32).ir_value()
    s2 = Uint32(sel_hi).bitcast(Int32).ir_value()
    moved = Int32(permlanex16(Int32.ir_type, raw_old.ir_value(), raw.ir_value(), s1, s2, False, False))
    return moved if dtype is Int32 else moved.bitcast(dtype)


def _swizzle(value: Numeric, offset: int) -> Numeric:
    """One ``ds_swizzle_b32``, which speaks 32-bit integers and nothing else."""
    dtype = value.dtype
    raw = value if dtype is Int32 else value.bitcast(Int32)
    moved = Int32(ds_swizzle(Int32.ir_type, raw.ir_value(), Int32(offset).ir_value()))
    return moved if dtype is Int32 else moved.bitcast(dtype)


def _is_cdna() -> bool:
    return current_target().arch.startswith("gfx9")


def _is_wave32_dpp() -> bool:
    target = current_target()
    return target.warp_size == 32 and target.arch.startswith(("gfx10", "gfx11", "gfx12"))


def _dpp_applies(value: object) -> bool:
    """Whether the DPP sequences below cover this call, or the portable ones have to.

    - **Supported arch.** Wave64 CDNA (``gfx9*``) or wave32 GFX10+/RDNA/CDNA5
      (``gfx10*``, ``gfx11*``, ``gfx12*``).
    - **A 32-bit scalar.** ``update_dpp``, ``permlanex16`` and ``ds_swizzle``
      all move one 32-bit lane value; a narrower type would have to be packed
      and a wider one split.
    """
    if not (_is_cdna() or _is_wave32_dpp()):
        return False
    return isinstance(value, Numeric) and value.dtype.width == 32


def _argext_applies(value: object, dtype: type[Numeric] | None = None) -> bool:
    """Whether ``warp_argmax``/``warp_argmin`` can promote losslessly to 32-bit DPP."""
    if not (_is_cdna() or _is_wave32_dpp()):
        return False
    if dtype is not None and not issubclass(dtype, Numeric):
        return False
    items = _as_items(_normalize_value(value), dtype)
    if not items or not isinstance(items[0], Numeric):
        return False
    elem_dtype = items[0].dtype if dtype is None else dtype
    return elem_dtype.width <= 32


# -- reduce -----------------------------------------------------------------


def _wave64_reduce(value: Numeric, op: ReductionOp) -> Numeric:
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
    acc = _combine(op, acc, _dpp(acc, _ROW_BCAST15, _ROW_MASK_ODD, neutral))
    acc = _combine(op, acc, _dpp(acc, _ROW_BCAST31, _ROW_MASK_HI2, neutral))

    return value.dtype(readlane(value.dtype.ir_type, acc, 63))


def _butterfly_reduce(value: Numeric, op: ReductionOp, width: int) -> Numeric:
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


def _wave32_reduce(value: Numeric, op: ReductionOp) -> Numeric:
    """Fold a full 32-lane wave on GFX10+/RDNA/CDNA5, then broadcast via SGPR."""
    neutral = _identity(op, value.dtype)
    acc = value
    for shift in (1, 2, 4, 8):
        acc = _combine(op, acc, _dpp(acc, _ROW_SHR[shift], _ALL_ROWS, neutral))
    # Row 0's total is in lane 15; row 1's total is in lane 31. ``permlanex16``
    # with all-0xF nibbles hands row 0's lane 15 to every lane of row 1, so
    # lane 31 accumulates both rows and ``readlane`` lifts the uniform result.
    acc = _combine(op, acc, _permlanex16(acc, _PERMLANEX16_BCAST15, _PERMLANEX16_BCAST15, neutral))
    return value.dtype(readlane(value.dtype.ir_type, acc, 31))


def _wave32_butterfly_reduce(value: Numeric, op: ReductionOp, width: int) -> Numeric:
    """Fold a sub-wave group (width <= 16) on wave32 using exact ``row_xmask`` DPP."""
    neutral = _identity(op, value.dtype)
    acc = value
    offset = 1
    while offset < width:
        acc = _combine(op, acc, _dpp(acc, _RDNA_XOR_DPP[offset], _ALL_ROWS, neutral))
        offset <<= 1
    return acc


def warp_reduce(
    value: object,
    op: ReductionOp | Callable,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
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
    if width == 1:
        return value
    if _is_wave32_dpp():
        return _wave32_reduce(value, op) if width == 32 else _wave32_butterfly_reduce(value, op, width)
    if width == 64:
        return _wave64_reduce(value, op)
    return _butterfly_reduce(value, op, width)


def warp_sum(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
    """``warp_reduce`` with ``ReductionOp.ADD``."""
    return warp_reduce(value, ReductionOp.ADD, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_max(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
    """``warp_reduce`` with ``ReductionOp.MAX``."""
    return warp_reduce(value, ReductionOp.MAX, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_min(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
    """``warp_reduce`` with ``ReductionOp.MIN``."""
    return warp_reduce(value, ReductionOp.MIN, width=width, valid_items=valid_items, _dtype=_dtype)


def _rocdl_argext(
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
    if valid_items is not None or not _argext_applies(value, _dtype):
        return _portable_argext(value, is_max, width=width, valid_items=valid_items, _dtype=_dtype)
    local_val, local_idx, orig_dtype, count = _argext_local_fold(value, is_max, width, _dtype)
    if width == 1:
        return _argext_unpromote(local_val, orig_dtype), local_idx
    op = ReductionOp.MAX if is_max else ReductionOp.MIN
    extremum = warp_reduce(local_val, op, width=width)
    if local_val.dtype.is_float:
        is_winner = (local_val == extremum) | ((local_val != local_val) & (extremum != extremum))
    else:
        is_winner = local_val == extremum
    winners = warp_ballot(is_winner, width=width)
    if width <= 32:
        win_lane = cttz(Uint32(winners)).to(Int32)
    else:
        win_lane = cttz(winners).to(Int32)
    if count == 1:
        win_idx = win_lane
    elif width == current_target().warp_size:
        win_idx = Int32(readlane(Int32.ir_type, local_idx, win_lane))
    else:
        base = (lane_id() // width) * width
        win_idx = _shuffle_value(local_idx, base + win_lane, width)
    return _argext_unpromote(extremum, orig_dtype), win_idx


def warp_argmax(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32]:
    """Return ``(max_value, winning_index)`` via DPP extremum + ``ballot`` + ``cttz``."""
    return _rocdl_argext(value, True, width=width, valid_items=valid_items, _dtype=_dtype)


def warp_argmin(
    value: object,
    *,
    width: int | None = None,
    valid_items: int | Integer | None = None,
    _dtype: type[Numeric] | None = None,
) -> tuple[Numeric, Int32]:
    """Return ``(min_value, winning_index)`` via DPP extremum + ``ballot`` + ``cttz``."""
    return _rocdl_argext(value, False, width=width, valid_items=valid_items, _dtype=_dtype)


# -- scan -------------------------------------------------------------------


def _inclusive_scan(value: Numeric, op: ReductionOp, width: int) -> Numeric:
    """The raw inclusive scan: ``log2(width)`` rounds, all of them DPP moves."""
    neutral = _identity(op, value.dtype)
    # Within a 16-lane row ``row_shr:k`` is exactly the round's shift, and the
    # lanes at the row's start read nothing, so ``_dpp`` hands them the identity
    # and they fold away for free. A group narrower than a row shares that row
    # with its neighbours, so it is the one case that has to mask the reads
    # crossing into them itself.
    in_group = lane_id() % width if width < 16 else None

    acc = value
    shift = 1
    while shift < min(width, 16):
        moved = _dpp(acc, _ROW_SHR[shift], _ALL_ROWS, neutral)
        if in_group is not None:
            moved = (in_group >= shift).select(moved, neutral)
        acc = _combine(op, moved, acc)
        shift <<= 1

    if _is_wave32_dpp():
        if width == 32:
            bcast15 = _permlanex16(acc, _PERMLANEX16_BCAST15, _PERMLANEX16_BCAST15, neutral)
            carry = (lane_id() >= 16).select(bcast15, neutral)
            acc = _combine(op, carry, acc)
        return acc

    # Row ``r`` now holds its own prefix, so its last lane holds the row's
    # total. ``row_bcast15`` hands that to the row above -- masked to rows 1
    # and 3, whose lanes are the ones missing a row -- and ``row_bcast31``
    # hands lane 31's running total to the wave's upper half.
    if width > 16:
        acc = _combine(op, _dpp(acc, _ROW_BCAST15, _ROW_MASK_ODD, neutral), acc)
    if width > 32:
        acc = _combine(op, _dpp(acc, _ROW_BCAST31, _ROW_MASK_HI2, neutral), acc)
    return acc


def _shift_up(inclusive: Numeric, op: ReductionOp, width: int) -> Numeric:
    """Turn an inclusive scan into the exclusive one by moving it up a lane.

    ``row_shr:1`` covers a group inside a row and ``wave_shr:1`` one spanning
    rows on CDNA; on wave32 RDNA/GFX12 ``permlanex16`` carries lane 15 into
    lane 16. Either way the lane that shifted in from nowhere reads nothing, so
    ``_dpp`` gives it the identity. That lane is the group's first only when the
    group starts where the row (or the wave) does, so the two widths that sit
    inside a larger unit -- under 16 lanes, or 32 on wave64 -- name their own
    first lane.
    """
    neutral = _identity(op, inclusive.dtype)
    if width == 1:
        return neutral
    if _is_wave32_dpp():
        shifted = _dpp(inclusive, _ROW_SHR[1], _ALL_ROWS, neutral)
        if width < 16:
            return ((lane_id() % width) == 0).select(neutral, shifted)
        if width == 16:
            return shifted
        bcast15 = _permlanex16(inclusive, _PERMLANEX16_BCAST15, _PERMLANEX16_BCAST15, neutral)
        return (lane_id() == 16).select(bcast15, shifted)
    ctrl = _ROW_SHR[1] if width <= 16 else _WAVE_SHR1
    shifted = _dpp(inclusive, ctrl, _ALL_ROWS, neutral)
    if width < 16 or width == 32:
        at_group_start = (lane_id() % width) == 0
        shifted = at_group_start.select(neutral, shifted)
    return shifted


def _aggregate(inclusive: Numeric, width: int) -> Numeric:
    """Hand every lane of the group what its last lane holds.

    A full wave reads its last lane (63 on wave64, 31 on wave32), which is
    uniform, so it lifts into an SGPR. On wave32 narrower groups stay in VALU
    with ``quad_perm`` / ``row_share``; on wave64 narrower groups use
    ``ds_swizzle`` in its 32-lane mode without any address arithmetic.
    """
    if width == 1:
        return inclusive
    if _is_wave32_dpp():
        if width == 2:
            return _dpp(inclusive, _QUAD_BCAST_PAIR, _ALL_ROWS, inclusive)
        if width == 4:
            return _dpp(inclusive, _QUAD_BCAST_LAST, _ALL_ROWS, inclusive)
        if width == 8:
            lo = _dpp_banks(inclusive, _ROW_SHARE7, _ALL_ROWS, _BANK_MASK_LO2, inclusive)
            return _dpp_banks(inclusive, _ROW_SHARE15, _ALL_ROWS, _BANK_MASK_HI2, lo)
        if width == 16:
            return _dpp(inclusive, _ROW_SHARE15, _ALL_ROWS, inclusive)
        return inclusive.dtype(readlane(inclusive.dtype.ir_type, inclusive, 31))
    if width == 64:
        return inclusive.dtype(readlane(inclusive.dtype.ir_type, inclusive, 63))
    return _swizzle(inclusive, _swizzle_broadcast(width))


def warp_inclusive_scan(
    value: object,
    op: ReductionOp | Callable,
    *,
    width: int | None = None,
    init: object = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
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
            to the input element type. Array inputs reject per-item seed ranges.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        This lane's inclusive prefix, with the input value type and shape.
    """
    width = _resolve_warp_width(width, "warp_inclusive_scan width")
    _universal_scan._validate_scan(value, init, valid_items, width, _dtype)
    if not _dpp_applies(value) or not isinstance(op, ReductionOp) or valid_items is not None:
        return _universal_scan.warp_inclusive_scan(
            value,
            op,
            width=width,
            init=init,
            valid_items=valid_items,
            _dtype=_dtype,
        )
    return _seed(_inclusive_scan(value, op, width), op, init)


def warp_exclusive_scan(
    value: object,
    op: ReductionOp | Callable,
    *,
    width: int | None = None,
    init: object = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
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
            to the input element type. Array inputs reject per-item seed ranges.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        This lane's exclusive prefix, with the input value type and shape.
    """
    width = _resolve_warp_width(width, "warp_exclusive_scan width")
    _universal_scan._validate_scan(value, init, valid_items, width, _dtype)
    if not _dpp_applies(value) or not isinstance(op, ReductionOp) or valid_items is not None:
        return _universal_scan.warp_exclusive_scan(
            value,
            op,
            width=width,
            init=init,
            valid_items=valid_items,
            _dtype=_dtype,
        )
    return _seed(_shift_up(_inclusive_scan(value, op, width), op, width), op, init)


def warp_inclusive_sum(
    value: object,
    *,
    width: int | None = None,
    init: object = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
    """``warp_inclusive_scan`` with ``ReductionOp.ADD``."""
    return warp_inclusive_scan(value, ReductionOp.ADD, width=width, init=init, valid_items=valid_items, _dtype=_dtype)


def warp_exclusive_sum(
    value: object,
    *,
    width: int | None = None,
    init: object = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> object:
    """``warp_exclusive_scan`` with ``ReductionOp.ADD``."""
    return warp_exclusive_scan(value, ReductionOp.ADD, width=width, init=init, valid_items=valid_items, _dtype=_dtype)


def warp_scan(
    value: object,
    op: ReductionOp | Callable,
    *,
    width: int | None = None,
    init: object = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> tuple[object, object]:
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
            to the input element type. Array inputs reject per-item seed ranges.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        A tuple (inclusive, exclusive) of this lane's prefixes.
    """
    width = _resolve_warp_width(width, "warp_scan width")
    _universal_scan._validate_scan(value, init, valid_items, width, _dtype)
    if not _dpp_applies(value) or not isinstance(op, ReductionOp) or valid_items is not None:
        return _universal_scan.warp_scan(
            value,
            op,
            width=width,
            init=init,
            valid_items=valid_items,
            _dtype=_dtype,
        )
    raw = _inclusive_scan(value, op, width)
    return _seed(raw, op, init), _seed(_shift_up(raw, op, width), op, init)


def warp_scan_with_aggregate(
    value: object,
    op: ReductionOp | Callable,
    *,
    width: int | None = None,
    init: object = None,
    valid_items: int | Integer | None = None,
    _dtype: type | None = None,
) -> tuple[object, object, object]:
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
            to the input element type. Array inputs reject per-item seed ranges.
        valid_items: Uniform number of leading contributing lanes in [0, width], or None for
            all lanes. Only single-item inputs accept this argument; omit it for
            an item range. Runtime counts must stay in range.

    Returns:
        A tuple (inclusive, exclusive, aggregate). The aggregate is available in every
        participating lane and has the input element type, even for an item range.
    """
    width = _resolve_warp_width(width, "warp_scan_with_aggregate width")
    _universal_scan._validate_scan(value, init, valid_items, width, _dtype)
    if not _dpp_applies(value) or not isinstance(op, ReductionOp) or valid_items is not None:
        return _universal_scan.warp_scan_with_aggregate(
            value,
            op,
            width=width,
            init=init,
            valid_items=valid_items,
            _dtype=_dtype,
        )
    raw = _inclusive_scan(value, op, width)
    inclusive = _seed(raw, op, init)
    exclusive = _seed(_shift_up(raw, op, width), op, init)
    return inclusive, exclusive, _aggregate(raw, width)


# -- vote -------------------------------------------------------------------


def _raw_ballot32(pred: Boolean) -> Uint32:
    return Uint32(Int32(ballot(Int32.ir_type, pred)))


def _raw_ballot64(pred: Boolean) -> Uint64:
    return Uint64(Int64(ballot(Int64.ir_type, pred)))


def warp_any(predicate: bool | int | Integer, *, width: int | None = None) -> Boolean:
    """Return ``True`` if any lane in the logical warp has a nonzero predicate."""
    width = _resolve_warp_width(width, "warp_any width")
    warp_threads = current_target().warp_size
    if warp_threads not in (32, 64):
        return _portable_warp_any(predicate, width=width)
    pred = _as_predicate(predicate)
    if width == 1:
        return pred
    if warp_threads == 32:
        raw32 = _raw_ballot32(pred)
        if width == 32:
            return raw32 != Uint32(0)
        base = Uint32(lane_id() & Int32(~(width - 1)))
        window = raw32 & (Uint32((1 << width) - 1) << base)
        return window != Uint32(0)
    raw64 = _raw_ballot64(pred)
    if width == 64:
        return raw64 != Uint64(0)
    base = Uint64(Uint32(lane_id() & Int32(~(width - 1))))
    window = raw64 & (Uint64((1 << width) - 1) << base)
    return window != Uint64(0)


def warp_all(predicate: bool | int | Integer, *, width: int | None = None) -> Boolean:
    """Return ``True`` if every lane in the logical warp has a nonzero predicate."""
    width = _resolve_warp_width(width, "warp_all width")
    warp_threads = current_target().warp_size
    if warp_threads not in (32, 64):
        return _portable_warp_all(predicate, width=width)
    pred = _as_predicate(predicate)
    if width == 1:
        return pred
    if warp_threads == 32:
        raw32 = _raw_ballot32(pred)
        if width == 32:
            return raw32 == Uint32(_FULL_MASK32)
        base = Uint32(lane_id() & Int32(~(width - 1)))
        low_mask = Uint32((1 << width) - 1)
        return ((raw32 >> base) & low_mask) == low_mask
    raw64 = _raw_ballot64(pred)
    if width == 64:
        return raw64 == Uint64(_FULL_MASK64)
    base = Uint64(Uint32(lane_id() & Int32(~(width - 1))))
    low_mask = Uint64((1 << width) - 1)
    return ((raw64 >> base) & low_mask) == low_mask


def warp_ballot(predicate: bool | int | Integer, *, width: int | None = None) -> Uint64:
    """Collect lane predicates into a ``Uint64`` logical-warp bitmask in ``[0, 2**width)``."""
    width = _resolve_warp_width(width, "warp_ballot width")
    warp_threads = current_target().warp_size
    if warp_threads not in (32, 64):
        return _portable_warp_ballot(predicate, width=width)
    pred = _as_predicate(predicate)
    if width == 1:
        return pred.select(Uint64(1), Uint64(0))
    if warp_threads == 32:
        raw32 = _raw_ballot32(pred)
        if width == 32:
            return Uint64(raw32)
        base = Uint32(lane_id() & Int32(~(width - 1)))
        return Uint64((raw32 >> base) & Uint32((1 << width) - 1))
    raw64 = _raw_ballot64(pred)
    if width == 64:
        return raw64
    base = Uint64(Uint32(lane_id() & Int32(~(width - 1))))
    return (raw64 >> base) & Uint64((1 << width) - 1)


def warp_count(predicate: bool | int | Integer, *, width: int | None = None) -> Int32:
    """Count lanes with a nonzero predicate via wave-native ``ballot`` + ``ctpop``."""
    width = _resolve_warp_width(width, "warp_count width")
    warp_threads = current_target().warp_size
    if warp_threads not in (32, 64):
        return _portable_warp_count(predicate, width=width)
    pred = _as_predicate(predicate)
    if width == 1:
        return pred.select(Int32(1), Int32(0))
    if warp_threads == 32:
        raw32 = _raw_ballot32(pred)
        if width < 32:
            base = Uint32(lane_id() & Int32(~(width - 1)))
            raw32 = raw32 & (Uint32((1 << width) - 1) << base)
        return ctpop(raw32).to(Int32)
    raw64 = _raw_ballot64(pred)
    if width == 64:
        return ctpop(raw64).to(Int32)
    base = Uint64(Uint32(lane_id() & Int32(~(width - 1))))
    logical32 = Uint32(raw64 >> base) & Uint32((1 << width) - 1)
    return ctpop(logical32).to(Int32)


def warp_ballot_rank(predicate: bool | int | Integer, *, width: int | None = None) -> tuple[Int32, Int32]:
    """Compute ``(exclusive_rank, total_count)`` via wave-native ``ballot`` + ``mbcnt``."""
    width = _resolve_warp_width(width, "warp_ballot_rank width")
    warp_threads = current_target().warp_size
    if warp_threads not in (32, 64):
        return _portable_warp_ballot_rank(predicate, width=width)
    pred = _as_predicate(predicate)
    if width == 1:
        return Int32(0), pred.select(Int32(1), Int32(0))
    zero_i32 = Int32(0).ir_value()
    if warp_threads == 32:
        mask32 = _raw_ballot32(pred)
        if width < 32:
            base = Uint32(lane_id() & Int32(~(width - 1)))
            mask32 = mask32 & (Uint32((1 << width) - 1) << base)
        rank = Int32(mbcnt_lo(Int32.ir_type, mask32.ir_value(), zero_i32))
        total = ctpop(mask32).to(Int32)
        return rank, total
    raw64 = _raw_ballot64(pred)
    if width == 64:
        lo32 = Uint32(raw64)
        hi32 = Uint32(raw64 >> Uint64(32))
        lo_cnt = mbcnt_lo(Int32.ir_type, lo32.ir_value(), zero_i32)
        rank = Int32(mbcnt_hi(Int32.ir_type, hi32.ir_value(), lo_cnt))
        total = ctpop(raw64).to(Int32)
        return rank, total
    base = Uint64(Uint32(lane_id() & Int32(~(width - 1))))
    low_mask64 = Uint64((1 << width) - 1)
    window64 = raw64 & (low_mask64 << base)
    lo32 = Uint32(window64)
    hi32 = Uint32(window64 >> Uint64(32))
    lo_cnt = mbcnt_lo(Int32.ir_type, lo32.ir_value(), zero_i32)
    rank = Int32(mbcnt_hi(Int32.ir_type, hi32.ir_value(), lo_cnt))
    logical32 = Uint32(raw64 >> base) & Uint32((1 << width) - 1)
    total = ctpop(logical32).to(Int32)
    return rank, total
