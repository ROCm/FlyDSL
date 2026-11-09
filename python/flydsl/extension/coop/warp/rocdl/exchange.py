# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``WarpExchange``: the SHUFFLE algorithm as a register-tile transpose.

With ``items_per_thread == width`` both directions, blocked->striped and
striped->blocked, are the same permutation: the ``width x width`` (lane,
register) tile transposed. The portable form gathers every output register from
every input register of its source lane -- ``width**2`` ``ds_bpermute`` --
because the register a lane needs depends on its lane id. A butterfly avoids
dynamic register indexing altogether: stage ``k`` swaps lane bit ``k`` with
register bit ``k``, i.e. the registers whose bit ``k`` differs from the lane's
move to lane ``l ^ (1 << k)`` and register ``r ^ (1 << k)``. Which register is
which is decided at trace time; only the lane bit is a runtime value.
``log2(width)`` stages of ``width / 2`` pairs replace the ``width**2`` gathers,
and the stages inside a 16-lane row are DPP.
"""

from ....._mlir.dialects import llvm as _llvm
from .....expr.gpu import lane_id
from .....expr.numeric import Index, Int32, Integer, Numeric, Uint8, Uint16, Uint32, Uint64, Uint128
from .....expr.rocdl import ds_bpermute
from .....expr.typing import Vector
from ...._dispatch import override
from ..._values import _as_items, _flatten_record, _from_items, _is_items, _item_dtype, _rebuild_record, _record_select
from .. import exchange as _exchange
from ..exchange import WarpExchangeAlgorithm
from ._common import (
    _BANKS_BIT_CLEAR,
    _BANKS_BIT_SET,
    _ROW_SHL,
    _ROW_SHR,
    _SWIZZLE_XOR16,
    _dpp,
    _is_gfx9,
    _swizzle,
)

__all__ = ["WarpExchange"]


# -- moving any complete item as 32-bit words -------------------------------------
#
# DPP, ds_swizzle and ds_bpermute all move one 32-bit lane value. A transpose
# only moves and selects, so any item -- scalar, Vector, Struct -- can travel
# as its 32-bit words: narrower leaves widen to one word, wider ones split.

_UINT = {8: Uint8, 16: Uint16, 32: Uint32, 64: Uint64, 128: Uint128}


def _leaf_movable(leaf):
    if not isinstance(leaf, (Numeric, Vector)):
        return False  # pointers keep the portable path
    dtype = leaf.dtype
    if issubclass(dtype, Index):
        return False  # no bitcast from index
    return dtype.width in _UINT or (issubclass(dtype, Integer) and dtype.width < 32)


def _movable(item):
    if isinstance(item, (Numeric, Vector)):
        return _leaf_movable(item)
    return all(_leaf_movable(leaf) for leaf in _flatten_record(item))


def _words(x):
    width = x.dtype.width
    bits = x if width not in _UINT or x.dtype is _UINT[width] else x.bitcast(_UINT[width])
    if width <= 32:
        return [bits.to(Uint32)]
    return [(bits >> (32 * i)).to(Uint32) for i in range(width // 32)]


def _from_words(words, dtype):
    width = dtype.width
    if width == 32:
        return words[0] if dtype is Uint32 else words[0].bitcast(dtype)
    if width < 32:
        if width not in _UINT:
            return words[0].to(dtype)
        bits = words[0].to(_UINT[width])
        return bits if dtype is bits.dtype else bits.bitcast(dtype)
    wide = _UINT[width]
    bits = words[0].to(wide)
    for i in range(1, width // 32):
        bits = bits | (words[i].to(wide) << (32 * i))
    return bits if dtype is wide else bits.bitcast(dtype)


def _map_words(fn, *items):
    """Apply ``fn(word_of_item0, word_of_item1, ...) -> word`` to every word of same-typed items."""
    first = items[0]
    if isinstance(first, Numeric):
        return _from_words([fn(*words) for words in zip(*(_words(x) for x in items))], first.dtype)
    if isinstance(first, Vector):
        elements = [_map_words(fn, *column) for column in zip(*(list(x) for x in items))]
        return Vector(Vector.from_elements(elements, dtype=first.dtype), shape=first.shape, dtype=first.dtype)
    leaves = zip(*(_flatten_record(x) for x in items))
    return _rebuild_record(_item_dtype(first), [_map_words(fn, *column) for column in leaves], first)


def _ds_permute(word, address):
    """Push *word* to the lane at byte *address* (``ds_permute_b32``; ROCDL has no op for it)."""
    raw = _llvm.call_intrinsic(Int32.ir_type, "llvm.amdgcn.ds.permute", [address.ir_value(), word.ir_value()], [], [])
    return Uint32(raw)


def _ds_bpermute(word, address):
    return Uint32(ds_bpermute(Int32.ir_type, address.ir_value(), word.ir_value()))


# -- one butterfly stage -------------------------------------------------------------


def _bank_stage(r0, r1, d):
    """d = 4, 8: one ``v_mov_b32_dpp`` per register, the select being the bank mask.

    Lanes with bit ``d`` clear take the partner's ``r0`` into ``r1``
    (``row_shl``, written only in the low banks, the rest keep ``r1`` through
    ``old``); lanes with it set take the partner's ``r1`` into ``r0``.
    """
    new_r1 = _map_words(lambda a, b: _dpp(a, _ROW_SHL[d], old=b, bank_mask=_BANKS_BIT_CLEAR[d]), r0, r1)
    new_r0 = _map_words(lambda b, a: _dpp(b, _ROW_SHR[d], old=a, bank_mask=_BANKS_BIT_SET[d]), r1, r0)
    return new_r0, new_r1


def _quad_stage(r0, r1, d, lower, upper):
    """d = 1, 2: lane bits below a bank need a ``v_cndmask`` per register.

    Each register moves in its own direction (``row_shl`` / ``row_shr``), so a
    stage is a DPP and a select per register. The ``bound_ctrl`` lanes a shift
    leaves without a source are exactly the ones the select discards.
    """
    from_above = _map_words(lambda w: _dpp(w, _ROW_SHL[d], bound_ctrl=True), r0)
    from_below = _map_words(lambda w: _dpp(w, _ROW_SHR[d], bound_ctrl=True), r1)
    return _record_select(lower, r0, from_below), _record_select(upper, r1, from_above)


def _lds_stage(r0, r1, move, lower, upper):
    """d = 16, 32: no gfx9 DPP crosses a row, so each pair moves one register.

    The lane sends the register its partner needs, so a pair costs one
    LDS-pipe move and three selects instead of two moves.
    """
    sent = _map_words(move, _record_select(upper, r0, r1))
    return _record_select(lower, r0, sent), _record_select(upper, r1, sent)


def _transpose(items, width):
    """Transpose the ``width x width`` (lane, register) tile of each logical warp."""
    regs = list(items)
    lane = lane_id()
    d = 1
    while d < width:
        bit = lane & d
        lower, upper = bit == 0, bit != 0
        if d == 16:
            move = lambda w: _swizzle(w, _SWIZZLE_XOR16)  # noqa: E731
        elif d == 32:
            partner = (lane ^ 32) << 2
            move = lambda w: _ds_bpermute(w, partner)  # noqa: E731
        for r0 in range(width):
            if r0 & d:
                continue
            r1 = r0 | d
            if d in _BANKS_BIT_CLEAR:
                regs[r0], regs[r1] = _bank_stage(regs[r0], regs[r1], d)
            elif d < 16:
                regs[r0], regs[r1] = _quad_stage(regs[r0], regs[r1], d, lower, upper)
            else:
                regs[r0], regs[r1] = _lds_stage(regs[r0], regs[r1], move, lower, upper)
        d <<= 1
    return regs


class WarpExchange(_exchange.WarpExchange):
    __doc__ = _exchange.WarpExchange.__doc__

    @classmethod
    def _shuffle_transpose(cls, value, storage):
        """The butterfly, or None when the portable gather has to run instead."""
        cls._check()
        if cls.algorithm is not WarpExchangeAlgorithm.SHUFFLE or not _is_gfx9():
            return None
        _exchange._check_algorithm(cls.algorithm, storage)
        value = cls._prepare(value)
        if not _is_items(value, cls.dtype):
            return value
        items = _as_items(value, cls.dtype)
        if not _movable(items[0]):
            return None
        return _from_items(_transpose(items, cls.warp_threads), like=value)

    @override
    @classmethod
    def blocked_to_striped(cls, value, *, storage=None):
        result = cls._shuffle_transpose(value, storage)
        return super().blocked_to_striped(value, storage=storage) if result is None else result

    @override
    @classmethod
    def striped_to_blocked(cls, value, *, storage=None):
        result = cls._shuffle_transpose(value, storage)
        return super().striped_to_blocked(value, storage=storage) if result is None else result


WarpExchange.blocked_to_striped.__func__.__doc__ = _exchange.WarpExchange.blocked_to_striped.__doc__
WarpExchange.striped_to_blocked.__func__.__doc__ = _exchange.WarpExchange.striped_to_blocked.__doc__
