# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 warp scans and logical-warp broadcast."""

from ....._mlir import ir
from ....._mlir.dialects import arith, llvm
from .....expr.gpu import lane_id, num_warp_threads
from .....expr.numeric import Int32, Integer, Numeric, Uint64
from .....expr.primitive import inttoptr, ptrtoint
from .....expr.rocdl import ds_swizzle, readlane
from .....expr.typing import Pointer, ReductionOp, Vector
from ..._common import (
    _combine,
    _identity,
    _normalize_columns,
    _resolve_warp_width,
    _seed,
    _shuffle_value,
)
from ..._values import _flatten_record, _item_dtype, _rebuild_record
from .. import scan as _universal_scan
from ._common import (
    _ALL_ROWS,
    _ROW_BCAST15,
    _ROW_BCAST31,
    _ROW_NEWBCAST,
    _ROW_SHR,
    _WAVE_SHR1,
    _XOR_DPP,
    _dpp,
    _dpp_applies,
    _has_permlane_swap,
    _has_row_newbcast,
    _permlane_swap,
    _swizzle,
)

__all__ = [
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_inclusive_sum",
    "warp_exclusive_sum",
    "warp_scan",
    "warp_scan_with_aggregate",
    "warp_broadcast",
]


def _swizzle_broadcast(width):
    """Offset that hands every lane of a *width*-lane group its last lane.

    ``and_mask`` keeps the bits above the group, ``or_mask`` sets every bit
    inside it, so the source lane is the group's base plus ``width - 1``.
    """
    return ((width - 1) << 5) | (0x1F & ~(width - 1))


# -- moving a lane's value to a whole group ----------------------------------
#
# A broadcast moves bits and never combines them, so every value -- whatever its
# dtype, a Vector, a pointer or a record -- goes through the same sequences one
# word at a time (32 bits, or 64 where ``v_mov_b64_dpp`` can take the move).


def _from_lane(word, k, width):
    """Every lane of each *width*-lane group (``width >= 2``) reads the group's lane *k*.

    The result goes back to the caller, so the move that produces it must not
    be one ``GCNDPPCombine`` can fold into whatever consumes it: folded into an
    integer subtraction it becomes ``v_subrev_u32_dpp``, which gfx942 computes
    wrongly (``tests/extension/coop/test_warp_rocdl.py`` pins this). Every lane
    reads inside its own row, so *old* is never read; passing the source as
    *old* with ``bound_ctrl`` off is what keeps the move a plain
    ``v_mov_b32_dpp``. The moves that only feed the next move may fold freely.
    """
    if width <= 4:
        # quad_perm reads within each quad; a pair is half a quad.
        first = (k, k, k, k) if width == 4 else (k, k, 2 + k, 2 + k)
        return _dpp(word, sum(lane << (2 * i) for i, lane in enumerate(first)), old=word)
    if width == 16:
        return _dpp(word, _ROW_NEWBCAST + k, old=word)
    if width == 8:
        # A row holds two groups: broadcast the first group's lane to the whole
        # row, then the second group's lane over banks 2 and 3 (lanes 8..15).
        low = _dpp(word, _ROW_NEWBCAST + k, bound_ctrl=True)
        return _dpp(word, _ROW_NEWBCAST + 8 + k, old=low, bank_mask=0b1100)
    if k < 16:
        # Width 32, source in the group's first row: broadcast it inside that
        # row, then ``row_bcast:15`` carries it up into rows 1 and 3.
        row = _dpp(word, _ROW_NEWBCAST + k, bound_ctrl=True)
        return _dpp(row, _ROW_BCAST15, 0b1010, row)
    if _has_permlane_swap():
        # Width 32, source in the group's second row, on gfx950: broadcast it
        # inside that row, then ``v_permlane16_swap`` hands the row down to the
        # first one -- two VALU moves, no LDS round trip.
        row = _dpp(word, _ROW_NEWBCAST + k - 16, bound_ctrl=True)
        row = Int32(row if row.type == Int32.ir_type else llvm.bitcast(Int32.ir_type, row))
        moved = _permlane_swap(row, row, 16)[1].ir_value()
        return moved if word.type == moved.type else llvm.bitcast(word.type, moved)
    # Width 32, source in the group's second row: DPP only carries values up
    # the wave, so reaching down a row takes three moves. ``ds_swizzle``'s
    # 32-lane bitmask mode names the lane in one (and_mask 0, or_mask k).
    return ds_swizzle(word.type, word, Int32(k << 5).ir_value())


def _int(bits):
    return ir.IntegerType.get_signless(bits)


def _words(raw, bits, size):
    """Split *raw* (*bits* wide) into *size*-bit words, padding the last one.

    A value narrower than a word is zero-extended into one. Padding an 8- or
    16-bit value with poison instead would save the ``v_and``, but bitcasting a
    vector with a poison lane makes the whole word poison, and ``update.dpp``
    takes neither a narrow integer nor a packed vector.
    """
    exact = _int(bits)
    as_int = raw if raw.type == exact else llvm.bitcast(exact, raw)
    count = -(-bits // size)
    if bits != size * count:
        as_int = arith.extui(_int(size * count), as_int)
    if count == 1:
        return [as_int]
    words = llvm.bitcast(ir.VectorType.get([count], _int(size)), as_int)
    return [llvm.extractelement(words, Int32(i).ir_value()) for i in range(count)]


def _join(words, ty, bits):
    """Inverse of ``_words``: reassemble a value of IR type *ty*."""
    if len(words) == 1:
        as_int = words[0]
    else:
        size = ir.IntegerType(words[0].type).width
        packed = llvm.mlir_poison(ir.VectorType.get([len(words)], _int(size)))
        for i, word in enumerate(words):
            packed = llvm.insertelement(packed, word, Int32(i).ir_value())
        as_int = llvm.bitcast(_int(size * len(words)), packed)
    if as_int.type != _int(bits):
        as_int = arith.trunci(_int(bits), as_int)
    return as_int if ty == _int(bits) else llvm.bitcast(ty, as_int)


def _move(value, move, size):
    """Apply *move* to every *size*-bit word of *value*, leaf by leaf.

    Values up to 32 bits always travel as one 32-bit word.
    """
    if isinstance(value, Pointer):
        return type(value)(inttoptr(value.type, _move(ptrtoint(value).to(Uint64), move, size)))
    if isinstance(value, (tuple, list)):
        return type(value)(_move(item, move, size) for item in value)
    if not isinstance(value, (Numeric, Vector)):
        leaves = [_move(leaf, move, size) for leaf in _flatten_record(value)]
        return _rebuild_record(_item_dtype(value), leaves, value)
    raw = value.ir_value()
    bits = value.dtype.width * (value.numel if isinstance(value, Vector) else 1)
    words = _words(raw, bits, size if bits > 32 else 32)
    moved = _join([move(word) for word in words], raw.type, bits)
    if isinstance(value, Vector):
        return Vector(moved, shape=value.shape, dtype=value.dtype)
    return value.dtype(moved)


# -- scan -------------------------------------------------------------------


def _inclusive_scan(value, op, width):
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

    # Row ``r`` now holds its own prefix, so its last lane holds the row's
    # total. ``row_bcast15`` hands that to the row above -- masked to rows 1
    # and 3, whose lanes are the ones missing a row -- and ``row_bcast31``
    # hands lane 31's running total to the wave's upper half.
    if width > 16:
        acc = _combine(op, _dpp(acc, _ROW_BCAST15, 0xA, neutral), acc)
    if width > 32:
        acc = _combine(op, _dpp(acc, _ROW_BCAST31, 0xC, neutral), acc)
    return acc


def _shift_up(inclusive, op, width):
    """Turn an inclusive scan into the exclusive one by moving it up a lane.

    ``row_shr:1`` covers a group inside a row and ``wave_shr:1`` one spanning
    rows. Each group's first lane takes the identity through a select, never
    through the move's *old*: the result goes back to the caller, and a move
    whose *old* is the identity is one ``GCNDPPCombine`` folds into the
    caller's instruction -- into an integer subtraction as
    ``v_subrev_u32_dpp``, which gfx942 computes wrongly. The select consumes
    the move, so nothing downstream can fold it.
    """
    neutral = _identity(op, inclusive.dtype)
    ctrl = _ROW_SHR[1] if width <= 16 else _WAVE_SHR1
    shifted = _dpp(inclusive, ctrl, _ALL_ROWS, inclusive)
    at_group_start = (lane_id() % width) == 0
    return at_group_start.select(neutral, shifted)


def _aggregate(inclusive, width):
    """Hand every lane of the group what its last lane holds.

    A full wave reads lane 63, which is uniform, so it lifts into an SGPR. A
    narrower group's last lane differs per group. A pair, a quad and a row
    (gfx90a+) name it with one DPP control that folds into its consumer, and
    gfx950 reaches a 32-lane group's last lane with ``row_newbcast`` and
    ``v_permlane16_swap``; otherwise ``ds_swizzle`` in its 32-lane mode names
    it without any address arithmetic. Eight lanes would take two DPP moves,
    which halves the latency of the swizzle but costs more VALU throughput
    than it saves.
    """
    if width == 64:
        return inclusive.dtype(readlane(inclusive.dtype.ir_type, inclusive, 63))
    if 2 <= width <= 4 or (width == 16 and _has_row_newbcast()) or (width == 32 and _has_permlane_swap()):
        return inclusive.dtype(_from_lane(inclusive.ir_value(), width - 1, width))
    return _swizzle(inclusive, _swizzle_broadcast(width))


def warp_inclusive_scan(
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
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
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
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
    value, *, width: int | None = None, init=None, valid_items: int | Integer | None = None, _dtype=None
):
    """``warp_inclusive_scan`` with ``ReductionOp.ADD``."""
    return warp_inclusive_scan(value, ReductionOp.ADD, width=width, init=init, valid_items=valid_items, _dtype=_dtype)


def warp_exclusive_sum(
    value, *, width: int | None = None, init=None, valid_items: int | Integer | None = None, _dtype=None
):
    """``warp_exclusive_scan`` with ``ReductionOp.ADD``."""
    return warp_exclusive_scan(value, ReductionOp.ADD, width=width, init=init, valid_items=valid_items, _dtype=_dtype)


def warp_scan(
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
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
    value,
    op,
    *,
    width: int | None = None,
    init=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
):
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


def warp_broadcast(
    value,
    source_lane: int | Integer,
    *,
    width: int | None = None,
    _dtype=None,
):
    width = _resolve_warp_width(width, "warp_broadcast width")
    # A full wave is left to the portable form, which LLVM already turns into
    # ``v_readlane_b32`` for a constant lane; a single lane has nothing to move.
    if width in (1, num_warp_threads()) or not _has_row_newbcast():
        return _universal_scan.warp_broadcast(value, source_lane, width=width, _dtype=_dtype)
    value = _normalize_columns(value)
    if isinstance(source_lane, int):
        if not 0 <= source_lane < width:
            raise ValueError("source_lane must be inside the logical warp")
        # A constant source has a fixed DPP control that names it (or one
        # ``ds_swizzle`` offset where DPP would need three moves). 64-bit DPP
        # (``v_mov_b64_dpp``) takes row_newbcast, so wide values move in 64-bit
        # words; ds_swizzle only speaks 32 bits.
        size = 32 if width == 32 and source_lane >= 16 else 64
        return _move(value, lambda word: _from_lane(word, source_lane, width), size)
    lane = lane_id()
    source_lane = Int32(source_lane)
    if width == 2:
        # Two candidates only: this lane, or its neighbour by quad_perm.
        own = ((source_lane ^ lane) & 1) == 0
        return _move(
            value,
            lambda word: arith.select(own.ir_value(), word, _dpp(word, _XOR_DPP[1], bound_ctrl=True)),
            64,
        )
    # Each lane may name its own source, so a runtime one keeps
    # ``ds_bpermute_b32``. A source inside the group needs no bounds check, so
    # the group base is applied here and the shuffle spans the whole wave.
    source = (lane & ~(width - 1)) | (source_lane & (width - 1))
    return _shuffle_value(value, source, num_warp_threads())


warp_broadcast.__doc__ = _universal_scan.warp_broadcast.__doc__
