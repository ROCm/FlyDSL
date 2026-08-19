# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL overrides for the warp-scope collectives.

Everything here is gfx9-only. ``row_bcast`` and ``wave_shr`` are CDNA DPP
controls that RDNA dropped in favour of ``row_share`` / ``permlanex16``, which
is a different sequence rather than a different constant -- and one with much
less to gain, since LLVM's own rewrite already reaches pure DPP there.
"""

from ....compiler.backends import current_target
from ....expr.gpu import lane_id
from ....expr.numeric import Int32, Numeric
from ....expr.rocdl import ds_swizzle, readlane, update_dpp
from .._common import combine, identity, resolve_warp_width, seed
from . import scan as _universal_scan
from .reduce import warp_reduce as _portable_warp_reduce

__all__ = [
    "warp_reduce",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
]


# DPP controls, spelled as the ISA encodes them.
_ROW_SHR = {1: 0x111, 2: 0x112, 4: 0x114, 8: 0x118}
_WAVE_SHR1 = 0x138
_ROW_BCAST15 = 0x142
_ROW_BCAST31 = 0x143
_ALL_ROWS = 0xF
_ALL_BANKS = 0xF

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


def _swizzle_broadcast(width):
    """Offset that hands every lane of a *width*-lane group its last lane.

    ``and_mask`` keeps the bits above the group, ``or_mask`` sets every bit
    inside it, so the source lane is the group's base plus ``width - 1``.
    """
    return ((width - 1) << 5) | (0x1F & ~(width - 1))


def _dpp(value, ctrl, row_mask, old):
    """One DPP move of *value*, with *old* wherever the move reaches nothing."""
    dtype = value.dtype
    moved = update_dpp(
        dtype.ir_type,
        old.ir_value(),
        value.ir_value(),
        ctrl,
        row_mask,
        _ALL_BANKS,
        False,
    )
    return dtype(moved)


def _swizzle(value, offset):
    """One ``ds_swizzle_b32``, which speaks 32-bit integers and nothing else."""
    dtype = value.dtype
    raw = value if dtype is Int32 else value.bitcast(Int32)
    moved = Int32(ds_swizzle(Int32.ir_type, raw.ir_value(), Int32(offset).ir_value()))
    return moved if dtype is Int32 else moved.bitcast(dtype)


def _dpp_applies(value, width):
    """Whether the sequences below cover this call, or the portable ones have to.

    - **gfx9.** Every control used here -- ``row_bcast``, ``wave_shr``,
      ``row_ror`` -- is CDNA-only.
    - **A 32-bit scalar.** ``update_dpp`` and ``ds_swizzle`` both move one
      32-bit lane value; a narrower type would have to be packed and a wider
      one split.
    - **A group no wider than the wave.** gfx9 is wave64 throughout, so this
      only rejects a caller that asked for something meaningless.
    """
    if width > 64 or not current_target().arch.startswith("gfx9"):
        return False
    return isinstance(value, Numeric) and value.dtype.width == 32


# -- reduce -----------------------------------------------------------------


def _wave64_reduce(value, op):
    """Fold a full 64-lane wave with DPP, then broadcast out of an SGPR.

    Requires a commutative *op*, and in a different way from the portable
    butterfly: ``row_shr`` brings in the value from *below*, and it is folded
    in as ``combine(op, acc, moved)`` -- the accumulator on the left. So the
    total lane 63 accumulates, and that ``readlane`` then broadcasts, is the
    wave folded in reverse lane order. Every lane agrees on it, unlike the
    butterfly below; it is simply the mirror image of the lane order a caller
    would expect. Writing ``combine(op, moved, acc)`` instead would put it back
    in order at no cost -- the scan further down uses the same DPP sequence
    that way -- but the two are indistinguishable for the commutative ops this
    library accepts, so the flip is left for whoever needs it.
    """
    neutral = identity(op, value.dtype)

    acc = value
    for shift in (1, 2, 4, 8):
        acc = combine(op, acc, _dpp(acc, _ROW_SHR[shift], _ALL_ROWS, neutral))
    # Each 16-lane row now holds its own running fold, so its last lane -- 15,
    # 31, 47, 63 -- holds that row's total. ``row_bcast15`` hands each of those
    # to the row above (rows 1 and 3, hence row_mask 0xa) and ``row_bcast31``
    # hands lane 31 to the upper half (rows 2 and 3, row_mask 0xc), which
    # leaves lane 63 holding all four rows.
    acc = combine(op, acc, _dpp(acc, _ROW_BCAST15, 0xA, neutral))
    acc = combine(op, acc, _dpp(acc, _ROW_BCAST31, 0xC, neutral))

    return value.dtype(readlane(value.dtype.ir_type, acc, 63))


def _butterfly_reduce(value, op, width):
    """Fold a group narrower than the wave by XOR distance.

    The butterfly is what a narrow group needs anyway -- there is no lane the
    whole group can read a total out of the way lane 63 serves a full wave --
    and every one of its steps stays inside the group, so no step leaves a lane
    without a source and the identity ``_dpp`` takes is never read. It is
    passed for the other reason ``_dpp`` gives: it is what lets each move fold
    into the ``combine`` that consumes it.

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
    neutral = identity(op, value.dtype)
    acc = value
    offset = 1
    while offset < width:
        if offset in _BUTTERFLY_DPP:
            partner = _dpp(acc, _BUTTERFLY_DPP[offset], _ALL_ROWS, neutral)
        else:
            partner = _swizzle(acc, _SWIZZLE_XOR16)
        acc = combine(op, acc, partner)
        offset <<= 1
    return acc


def warp_reduce(value, op, *, width=None):
    """Reduce *value* across *width* lanes; every participating lane gets the result.

    Same contract as the portable implementation this displaces -- see
    :func:`~flydsl.extension.coop.warp.reduce.warp_reduce`.
    """
    width = resolve_warp_width(width, "warp_reduce width")
    if not _dpp_applies(value, width):
        return _portable_warp_reduce(value, op, width=width)
    if width == 64:
        return _wave64_reduce(value, op)
    return _butterfly_reduce(value, op, width)


# -- scan -------------------------------------------------------------------


def _inclusive_scan(value, op, width):
    """The raw inclusive scan: ``log2(width)`` rounds, all of them DPP moves."""
    neutral = identity(op, value.dtype)
    # Within a 16-lane row ``row_shr:k`` is exactly the round's shift, and
    # ``bound_ctrl`` supplies identity at the row's start for free. A group
    # narrower than a row shares that row with its neighbours, so it is the one
    # case that has to mask the reads crossing into them itself.
    in_group = lane_id() % width if width < 16 else None

    acc = value
    shift = 1
    while shift < min(width, 16):
        moved = _dpp(acc, _ROW_SHR[shift], _ALL_ROWS, neutral)
        if in_group is not None:
            moved = (in_group >= shift).select(moved, neutral)
        acc = combine(op, moved, acc)
        shift <<= 1

    # Row ``r`` now holds its own prefix, so its last lane holds the row's
    # total. ``row_bcast15`` hands that to the row above -- masked to rows 1
    # and 3, whose lanes are the ones missing a row -- and ``row_bcast31``
    # hands lane 31's running total to the wave's upper half.
    if width > 16:
        acc = combine(op, _dpp(acc, _ROW_BCAST15, 0xA, neutral), acc)
    if width > 32:
        acc = combine(op, _dpp(acc, _ROW_BCAST31, 0xC, neutral), acc)
    return acc


def _shift_up(inclusive, op, width):
    """Turn an inclusive scan into the exclusive one by moving it up a lane.

    ``row_shr:1`` covers a group inside a row and ``wave_shr:1`` one spanning
    rows; either way ``bound_ctrl`` puts identity in front of the lane it
    shifted in from nowhere. That lane is the group's first only when the group
    starts where the row (or the wave) does, so the two widths that sit inside
    a larger unit -- under 16 lanes, or 32 -- name their own first lane.
    """
    neutral = identity(op, inclusive.dtype)
    ctrl = _ROW_SHR[1] if width <= 16 else _WAVE_SHR1
    shifted = _dpp(inclusive, ctrl, _ALL_ROWS, neutral)
    if width < 16 or width == 32:
        at_group_start = (lane_id() % width) == 0
        shifted = at_group_start.select(neutral, shifted)
    return shifted


def _aggregate(inclusive, width):
    """Hand every lane of the group what its last lane holds.

    A full wave reads lane 63, which is uniform, so it lifts into an SGPR. A
    narrower group's last lane differs per group, which is what ``ds_swizzle``
    in its 32-lane mode names without any address arithmetic.
    """
    if width == 64:
        return inclusive.dtype(readlane(inclusive.dtype.ir_type, inclusive, 63))
    return _swizzle(inclusive, _swizzle_broadcast(width))


def warp_inclusive_scan(value, op, *, width=None, init=None):
    """Scan *value* across *width* lanes; lane ``k`` gets lanes ``0..k`` folded.

    Same contract as the portable implementation this displaces -- see
    :func:`~flydsl.extension.coop.warp.scan.warp_inclusive_scan`.
    """
    width = resolve_warp_width(width, "warp_inclusive_scan width")
    if not _dpp_applies(value, width):
        return _universal_scan.warp_inclusive_scan(value, op, width=width, init=init)
    return seed(_inclusive_scan(value, op, width), op, init)


def warp_exclusive_scan(value, op, *, width=None, init=None):
    """Scan *value* across *width* lanes; lane ``k`` gets lanes ``0..k-1`` folded.

    Same contract as the portable implementation this displaces -- see
    :func:`~flydsl.extension.coop.warp.scan.warp_exclusive_scan`.
    """
    width = resolve_warp_width(width, "warp_exclusive_scan width")
    if not _dpp_applies(value, width):
        return _universal_scan.warp_exclusive_scan(value, op, width=width, init=init)
    return seed(_shift_up(_inclusive_scan(value, op, width), op, width), op, init)


def warp_scan(value, op, *, width=None, init=None):
    """Return ``(inclusive, exclusive)`` for this lane, sharing one scan.

    Same contract as the portable implementation this displaces -- see
    :func:`~flydsl.extension.coop.warp.scan.warp_scan`.
    """
    width = resolve_warp_width(width, "warp_scan width")
    if not _dpp_applies(value, width):
        return _universal_scan.warp_scan(value, op, width=width, init=init)
    raw = _inclusive_scan(value, op, width)
    return seed(raw, op, init), seed(_shift_up(raw, op, width), op, init)


def warp_scan_with_aggregate(value, op, *, width=None, init=None):
    """Return ``(inclusive, exclusive, aggregate)``, sharing one scan.

    Same contract as the portable implementation this displaces -- see
    :func:`~flydsl.extension.coop.warp.scan.warp_scan_with_aggregate`.
    """
    width = resolve_warp_width(width, "warp_scan_with_aggregate width")
    if not _dpp_applies(value, width):
        return _universal_scan.warp_scan_with_aggregate(value, op, width=width, init=init)
    raw = _inclusive_scan(value, op, width)
    inclusive = seed(raw, op, init)
    exclusive = seed(_shift_up(raw, op, width), op, init)
    return inclusive, exclusive, _aggregate(raw, width)
