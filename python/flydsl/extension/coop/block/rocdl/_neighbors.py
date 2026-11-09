# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 neighbors of blocked items in a single-wave block: DPP moves, no shared memory.

``BlockShuffle``, ``BlockAdjacentDifference`` and ``BlockDiscontinuity`` all read
each thread's left and/or right neighbor through ``_neighbors``. When the whole
block is one wave (``block_threads <= 64``) the neighbor is one ``wave_shr:1`` /
``wave_shl:1`` DPP move per 32-bit word and the wrap-around item -- the portable
layout's ``(tid -/+ 1) % block_threads`` -- one ``v_readlane``: no shared memory
and no barrier.

Blocks of several waves keep the portable publication (one boundary store, one
barrier, one load). The wave-boundary items still need that round trip, and the
DPP moves and selects added on top measured slower on gfx942.
"""

from .....expr.numeric import Int32, Uint32
from .....expr.rocdl import readlane
from ...._dispatch import override
from ..._common import _linear_thread_id
from ..._values import _as_items, _record_select
from ...warp.rocdl._common import _WAVE_SHL1, _WAVE_SHR1, _dpp, _is_gfx9
from ...warp.rocdl.exchange import _map_words, _movable

_WAVE = 64


def _shift(item, ctrl, edge):
    """``wave_shr:1`` / ``wave_shl:1`` of every 32-bit word; the lane without a source keeps ``edge``.

    ``edge`` is the DPP ``old`` operand with ``bound_ctrl`` off. Besides filling
    the edge lane, a register ``old`` keeps the move a plain ``v_mov_b32_dpp``:
    with ``bound_ctrl`` on, GCNDPPCombine folds the move into its consumer, and
    for ``x - neighbor`` that is ``v_subrev_u32_dpp``, which gfx942 executes with
    the DPP applied to the other operand.
    """
    return _map_words(lambda w, e: _dpp(w, ctrl, old=e), item, edge)


def _lane_value(item, lane):
    """The item of a constant lane, broadcast (``v_readlane`` per 32-bit word)."""
    return _map_words(lambda w: Uint32(readlane(Int32.ir_type, w, lane)), item)


class _WaveNeighbors:
    """Mixin: ``_neighbors`` with in-wave DPP moves for single-wave blocks."""

    @classmethod
    def _wave_neighbors_apply(cls, items):
        return cls.num_warps == 1 and _is_gfx9() and _movable(items[0])

    @override
    @classmethod
    def _neighbors(cls, value, storage, left=True, right=True):
        cls._check(value)
        items = list(_as_items(value, cls.dtype))
        if not cls._wave_neighbors_apply(items):
            return super()._neighbors(value, storage, left=left, right=right)
        tid = _linear_thread_id(cls.block_size)
        warp = cls.warp_threads
        first, last = items[0], items[-1]
        prior = following = None
        if left:
            prior = _shift(last, _WAVE_SHR1, _lane_value(last, warp - 1))
        if right:
            edge = _lane_value(first, 0)
            following = _shift(first, _WAVE_SHL1, edge)
            if warp < _WAVE:
                # Lane W-1 of a partial wave would read a lane that was never launched.
                following = _record_select(tid % warp == warp - 1, edge, following)
        return items, tid, prior, following

    @classmethod
    def _single_wave_items(cls, value):
        """The items of a call the DPP path covers, else None."""
        items = list(_as_items(value, cls.dtype))
        return items if cls._wave_neighbors_apply(items) else None
