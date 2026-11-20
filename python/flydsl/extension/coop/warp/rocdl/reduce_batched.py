# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 batched warp reductions.

The portable tree (``reduce_batched._reduce_batched``) merges, at stride ``s``,
a ``left`` subtree that lanes with bit ``s`` clear keep and a ``right`` subtree
that the other lanes keep. Each lane takes its partner's copy of the subtree it
keeps through ``shuffle_xor(s)`` -- on gfx942 one ``ds_swizzle_b32`` (or, for
xor 32, ``ds_bpermute_b32``) plus an ``lgkmcnt`` wait per node -- wrapped in four
``v_cndmask`` that pick what to send and restore operand order. A 64-batch,
64-lane call is 63 such nodes, so 63 LDS round trips in a dependent chain.

**Directional nodes** (``_merge_directional``). The xor moves more than either
side needs: a left lane only reads the partner *above* it (``l + s``), and only
that partner's ``left``; a right lane only the partner *below* (``l - s``), and
only its ``right``. Inside a 16-lane row those are ``row_shl:s`` and
``row_shr:s``, so for ``s`` = 1, 2, 4, 8 a node is two DPP moves, each folded by
LLVM into the combine that consumes it (``v_add_u32_dpp`` ...), and one select:

    left lanes:  op(left, left[l + s])     right lanes:  op(right[l - s], right)

That is the same tree with the same operands in the same order as the portable
one. It pays two combines per node, which is free only while a combine is one
DPP-foldable VOP2 instruction: 32-bit integer ``ADD``/``MAX``/``MIN``.

**Commutative nodes** (``_merge_commutative``) serve every other
``ReductionOp`` in groups of up to 32 lanes. They keep the portable send/keep
selects but drop the two ordering selects (every ``ReductionOp`` commutes) and
move by an exact xor DPP (``quad_perm`` for 1/2, two bank-masked shifts for 4,
``row_ror:8`` for 8): one combine per node. Integer ``MUL`` is a VOP3
``v_mul_lo_u32`` and float ``MAX``/``MIN`` the three-instruction
NaN-propagating form, so two of them per node would cost more than the moves
save. Float ``ADD``/``MUL`` would fold, but LLVM's SLP vectorizer pairs the
independent sibling folds into packed ``v_pk_add_f32``/``v_pk_mul_f32`` that no
DPP move folds into. Across a full wave the moves' VALU slots cost more
throughput than the shorter chain saves, so a 64-lane group keeps the portable
tree for these ops.

``s`` = 16 and 32 cross the rows DPP is confined to; both kinds of node keep one
LDS move there -- ``ds_swizzle_b32`` xor 16 and the ``ds_bpermute_b32`` LLVM
emits for xor 32 -- but around fewer selects. gfx950 makes either node one
``v_permlane16_swap`` / ``v_permlane32_swap`` and one combine: the swap hands
every lane both operands of its node in lane order, so no select is left.

A node whose right subtree holds no batch at all (fewer batches than the tree
has leaves) only needs its left lanes: one ``row_shl`` move and one combine.

A callable ``op`` keeps the portable tree: it needs operand order, so one combine
per node means four selects plus a move, and a DPP move costs a VALU slot where
the portable LDS move costs none; measured, that trades latency for throughput.
"""

from .....expr.gpu import lane_id, num_warp_threads
from .....expr.numeric import Integer
from .....expr.typing import ReductionOp
from ..._common import _combine, _resolve_warp_width, _select_value, _shuffle_value
from ..._values import _as_items, _from_items, _is_items
from .. import reduce_batched as _portable
from ._common import (
    _ALL_ROWS,
    _BANKS_BIT_CLEAR,
    _BANKS_BIT_SET,
    _ROW_SHL,
    _ROW_SHR,
    _SWIZZLE_XOR16,
    _XOR_DPP,
    _dpp,
    _dpp_applies,
    _has_permlane_swap,
    _permlane_swap,
    _swizzle,
)

__all__ = [
    "warp_reduce_batched",
    "warp_reduce_batched_to_blocked",
    "warp_reduce_batched_to_striped",
    "warp_sum_batched",
    "warp_sum_batched_to_blocked",
    "warp_sum_batched_to_striped",
]


def _zero(value):
    return value.dtype(0)


def _from_above(value, stride):
    """``value[l + stride]`` in every lane with bit *stride* clear.

    The lanes whose read leaves the row are right lanes, whose copy of this fold
    is discarded, so ``old`` need not be the op's identity. It is zero because,
    with every row and bank enabled, ``old == 0`` is indistinguishable from
    ``bound_ctrl:0``: that is what lets ``GCNDPPCombine`` fold the move into
    *any* consuming VOP2 op, MAX and MIN included, without an identity check.
    """
    return _dpp(value, _ROW_SHL[stride], _ALL_ROWS, _zero(value))


def _from_below(value, stride):
    """``value[l - stride]`` in every lane with bit *stride* set (see ``_from_above``)."""
    return _dpp(value, _ROW_SHR[stride], _ALL_ROWS, _zero(value))


def _xor(value, stride, shuffle_width):
    """``value[l ^ stride]``, by DPP inside a row and the cheapest LDS move across rows."""
    if stride in _XOR_DPP:
        return _dpp(value, _XOR_DPP[stride], _ALL_ROWS, _zero(value))
    if stride == 4:
        # No single control is xor 4: lanes with bit 2 clear shift down, the
        # others up, and the second move keeps the first one's banks as ``old``.
        down = _dpp(value, _ROW_SHL[4], _ALL_ROWS, _zero(value), bank_mask=_BANKS_BIT_CLEAR[4])
        return _dpp(value, _ROW_SHR[4], _ALL_ROWS, down, bank_mask=_BANKS_BIT_SET[4])
    if stride == 16:
        return _swizzle(value, _SWIZZLE_XOR16)
    return _shuffle_value(value, stride, shuffle_width, mode="xor")


def _crosses_by_swap(stride):
    """gfx950 crosses rows (16) and halves (32) with ``v_permlane*_swap``, not the LDS pipe."""
    return stride >= 16 and _has_permlane_swap()


def _merge_swapped(op, left, right, stride):
    """One swap hands every lane both operands of its node, lower lane first.

    A left lane keeps its ``left`` and gains its partner's, a right lane keeps
    its ``right`` and gains its partner's -- exactly the pairs the node folds,
    already in lane order, so there is nothing to select before or after.
    """
    return _combine(op, *_permlane_swap(left, right, stride))


def _merge_directional(op, left, right, stride, is_left, shuffle_width):
    if _crosses_by_swap(stride):
        return _merge_swapped(op, left, right, stride)
    if stride <= 8:
        lower = _combine(op, left, _from_above(left, stride))
        upper = _combine(op, _from_below(right, stride), right)
    else:
        exchanged = _xor(_select_value(is_left, right, left), stride, shuffle_width)
        lower = _combine(op, left, exchanged)
        upper = _combine(op, exchanged, right)
    return _select_value(is_left, lower, upper)


def _merge_commutative(op, left, right, stride, is_left, shuffle_width):
    if _crosses_by_swap(stride):
        return _merge_swapped(op, left, right, stride)
    kept = _select_value(is_left, left, right)
    return _combine(op, kept, _xor(_select_value(is_left, right, left), stride, shuffle_width))


def _merge_left(op, left, stride, shuffle_width):
    """A node with no right batches: only its left lanes own anything."""
    if _crosses_by_swap(stride):
        return _merge_swapped(op, left, left, stride)
    if stride <= 8:
        return _combine(op, left, _from_above(left, stride))
    return _combine(op, left, _xor(left, stride, shuffle_width))


def _merge_for(op, item, width):
    """The node shape for this call, or None for the portable tree.

    A commutative node spends VALU slots on moves the portable tree makes on
    the LDS pipe. Up to 32 lanes the shorter chain still wins on both latency
    and throughput; across a full wave its extra VALU work costs more
    throughput than its latency saves, so those calls keep the portable tree.
    """
    if not isinstance(op, ReductionOp) or not _dpp_applies(item):
        return None
    if isinstance(item, Integer) and op is not ReductionOp.MUL:
        return _merge_directional
    return _merge_commutative if width <= 32 else None


def _reduce_batched(value, op, width, output_layout, sync_physical_warp=False, dtype=None):
    items = list(_as_items(value, dtype)) if _is_items(value, dtype) and len(value) else []
    width = _resolve_warp_width(width, "warp_reduce_batched width")
    merge = _merge_for(op, items[0], width) if items else None
    if merge is None:
        return _portable._reduce_batched(value, op, width, output_layout, sync_physical_warp, dtype)
    if not isinstance(sync_physical_warp, bool):
        raise TypeError("sync_physical_warp must be a Python bool")
    if output_layout not in ("scalar", "blocked", "striped"):
        raise ValueError("output_layout must be scalar, blocked, or striped")
    batches = len(items)
    if output_layout == "scalar" and not 1 <= batches <= width:
        raise ValueError("scalar batched reduction requires 1 <= batches <= width; use a distributed layout")
    count = (batches + width - 1) // width
    # DPP and ds_swizzle act on the whole wave whatever the shuffle width; only
    # the xor-32 shuffle takes one, and the full wave is the only group it serves.
    shuffle_width = num_warp_threads() if sync_physical_warp else width
    masks = {}

    def is_left(stride):
        if stride not in masks:
            masks[stride] = (lane_id() & stride) == 0
        return masks[stride]

    def batch_of(base):
        return base // width + (base % width) * count if output_layout == "blocked" else base

    def tree(base, span):
        # Within one output slot ``batch_of`` grows with ``base``, so a subtree
        # holds a batch iff its first leaf does; the root's first leaf always does.
        if span == 1:
            return items[batch_of(base)]
        stride = span // 2
        left = tree(base, stride)
        if batch_of(base + stride) >= batches:
            return _merge_left(op, left, stride, shuffle_width)
        return merge(op, left, tree(base + stride, stride), stride, is_left(stride), shuffle_width)

    outputs = [tree(slot * width, width) for slot in range(count)]
    return outputs[0] if output_layout == "scalar" else _from_items(outputs, like=value)


def warp_reduce_batched(value, op, *, width: int | None = None, sync_physical_warp: bool = False, _dtype=None):
    return _reduce_batched(value, op, width, "scalar", sync_physical_warp, _dtype)


def warp_reduce_batched_to_blocked(
    value, op, *, width: int | None = None, sync_physical_warp: bool = False, _dtype=None
):
    return _reduce_batched(value, op, width, "blocked", sync_physical_warp, _dtype)


def warp_reduce_batched_to_striped(
    value, op, *, width: int | None = None, sync_physical_warp: bool = False, _dtype=None
):
    return _reduce_batched(value, op, width, "striped", sync_physical_warp, _dtype)


def warp_sum_batched(value, *, width: int | None = None, sync_physical_warp: bool = False, _dtype=None):
    """``warp_reduce_batched`` with ``ReductionOp.ADD``."""
    return warp_reduce_batched(
        value, ReductionOp.ADD, width=width, sync_physical_warp=sync_physical_warp, _dtype=_dtype
    )


def warp_sum_batched_to_blocked(value, *, width: int | None = None, sync_physical_warp: bool = False, _dtype=None):
    """``warp_reduce_batched_to_blocked`` with ``ReductionOp.ADD``."""
    return warp_reduce_batched_to_blocked(
        value, ReductionOp.ADD, width=width, sync_physical_warp=sync_physical_warp, _dtype=_dtype
    )


def warp_sum_batched_to_striped(value, *, width: int | None = None, sync_physical_warp: bool = False, _dtype=None):
    """``warp_reduce_batched_to_striped`` with ``ReductionOp.ADD``."""
    return warp_reduce_batched_to_striped(
        value, ReductionOp.ADD, width=width, sync_physical_warp=sync_physical_warp, _dtype=_dtype
    )


# The contracts are the portable ones, and so is their documentation.
for _name in ("warp_reduce_batched", "warp_reduce_batched_to_blocked", "warp_reduce_batched_to_striped"):
    globals()[_name].__doc__ = getattr(_portable, _name).__doc__
del _name
