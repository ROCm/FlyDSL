# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide prefix scan."""

from ....expr.gpu import lane_id, shuffle_idx, shuffle_up
from .._common import combine, identity, resolve_warp_width, seed

__all__ = [
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
]


def _shuffle_up(value, offset, width):
    """Read lane ``k - offset``, together with the flag saying that lane exists."""
    within_group = lane_id() % width
    return shuffle_up(value, offset, width), within_group >= offset


def _shift_up(inclusive, op, width):
    """Turn an inclusive scan into the exclusive one by moving it up a lane."""
    shifted, valid = _shuffle_up(inclusive, 1, width)
    return valid.select(shifted, identity(op, inclusive.dtype))


def _broadcast_last(value, width):
    """Hand every lane of the group what its highest lane holds.

    After an inclusive scan that lane holds the whole group's fold, so this is
    what turns a scan into an aggregate — one shuffle rather than a second pass
    over the data.

    ``shuffle_idx`` takes an *absolute* lane, not one relative to the group, so
    a group narrower than the warp has to name its own last lane.
    """
    group_base = (lane_id() // width) * width
    return shuffle_idx(value, group_base + (width - 1), width)


def _hillis_steele(value, op, width):
    """The raw inclusive scan: ``log2(width)`` rounds of shuffle-and-fold.

    Each round folds in the lane *offset* below, whose result is kept only
    where that lane exists.
    """
    offset = 1
    while offset < width:
        shifted, valid = _shuffle_up(value, offset, width)
        value = valid.select(combine(op, shifted, value), value)
        offset <<= 1
    return value


def warp_inclusive_scan(value, op, *, width=None, init=None):
    """Scan *value* across *width* lanes; lane ``k`` gets lanes ``0..k`` folded.

    *width* defaults to the target's warp size; a narrower power-of-two width
    scans independent groups of lanes side by side. Passing *init* folds it in
    ahead of every lane, so lane ``k`` gets ``init`` followed by lanes ``0..k``.
    """
    width = resolve_warp_width(width, "warp_inclusive_scan width")
    return seed(_hillis_steele(value, op, width), op, init)


def warp_exclusive_scan(value, op, *, width=None, init=None):
    """Scan *value* across *width* lanes; lane ``k`` gets lanes ``0..k-1`` folded.

    *width* defaults to the target's warp size, as in
    :func:`warp_inclusive_scan`. Lane 0 has nothing in front of it and gets
    *init*, or ``identity(op, ...)`` when no *init* is given.
    """
    width = resolve_warp_width(width, "warp_exclusive_scan width")
    return seed(_shift_up(_hillis_steele(value, op, width), op, width), op, init)


def warp_scan(value, op, *, width=None, init=None):
    """Return ``(inclusive, exclusive)`` for this lane, sharing one scan.

    *width* defaults to the target's warp size, as in :func:`warp_inclusive_scan`.
    """
    width = resolve_warp_width(width, "warp_scan width")
    raw = _hillis_steele(value, op, width)
    return seed(raw, op, init), seed(_shift_up(raw, op, width), op, init)


def warp_scan_with_aggregate(value, op, *, width=None, init=None):
    """Return ``(inclusive, exclusive, aggregate)``, sharing one scan.

    *width* defaults to the target's warp size, as in :func:`warp_inclusive_scan`.
    """
    width = resolve_warp_width(width, "warp_scan_with_aggregate width")
    raw = _hillis_steele(value, op, width)
    inclusive = seed(raw, op, init)
    exclusive = seed(_shift_up(raw, op, width), op, init)
    return inclusive, exclusive, _broadcast_last(raw, width)
