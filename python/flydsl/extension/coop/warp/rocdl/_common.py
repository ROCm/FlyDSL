# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""DPP controls and the cross-lane moves every gfx9 override is built from."""

from ....._mlir.dialects import llvm
from .....compiler.backends import current_target
from .....expr.numeric import Int32, Numeric
from .....expr.rocdl import ds_swizzle, update_dpp

# DPP controls, spelled as the ISA encodes them.
#
# ``row_shl:k`` hands lane ``i`` the value of lane ``i + k`` of its 16-lane row,
# ``row_shr:k`` that of lane ``i - k``; ``wave_shl:1`` / ``wave_shr:1`` do the
# same by one lane across rows.
_ROW_SHL = {1: 0x101, 2: 0x102, 4: 0x104, 8: 0x108}
_ROW_SHR = {1: 0x111, 2: 0x112, 4: 0x114, 8: 0x118}
_WAVE_SHL1 = 0x130
_WAVE_SHR1 = 0x138
_ROW_BCAST15 = 0x142
_ROW_BCAST31 = 0x143
# ``row_newbcast:n`` (gfx90a and gfx94x/gfx95x; RDNA spends the encoding on
# ``row_share``) hands lane ``n`` of every row to the whole row.
_ROW_NEWBCAST = 0x150
_ROW_HALF_MIRROR = 0x141  # lane i reads 7 - i within each group of 8: xor 7
_QUAD_REVERSE = 0x1B  # quad_perm[3,2,1,0]: xor 3
# The XORs one control spells exactly: ``quad_perm[1,0,3,2]``,
# ``quad_perm[2,3,0,1]``, and rotating a 16-lane row by 8.
_XOR_DPP = {1: 0xB1, 2: 0x4E, 8: 0x128}
_ALL_ROWS = 0xF
_ALL_BANKS = 0xF
# A bank is a 4-lane quarter of a row, so lane bits 2 and 3 are bank bits: the
# banks whose lanes have bit ``d`` clear, and the banks whose lanes have it set.
_BANKS_BIT_CLEAR = {4: 0b0101, 8: 0b0011}
_BANKS_BIT_SET = {4: 0b1010, 8: 0b1100}

# XOR by 16 crosses the 16-lane rows DPP is built around, and gfx9 has no
# control that does. ``ds_swizzle_b32`` in its 32-lane mode does: a lane reads
# ``((lane & and_mask) | or_mask) ^ xor_mask``, and the offset packs the three
# as ``xor_mask << 10 | or_mask << 5 | and_mask``.
_SWIZZLE_XOR16 = (16 << 10) | 0x1F


def _is_gfx9():
    return current_target().arch.startswith("gfx9")


def _has_row_newbcast():
    return current_target().arch.startswith(("gfx90a", "gfx94", "gfx95"))


def _dpp(value, ctrl, row_mask=_ALL_ROWS, old=None, *, bank_mask=_ALL_BANKS, bound_ctrl=False):
    """One DPP move of *value*: a 32-bit ``Numeric``, or a raw i32/i64 IR value.

    The lanes *row_mask* and *bank_mask* leave out keep *old*. So do the lanes
    whose source the control puts outside the row -- unless *bound_ctrl* is
    on, which writes zero there instead. ``old=None`` is poison.

    The two settings serve different moves:

    - **A fold that reads past the edge of a row** (the reduce and scan
      ladders) passes the identity of the op it folds with as *old*, so the
      lanes the move does not reach drop out of that fold whatever the op is.
      ``bound_ctrl``'s zero is the identity of ``ADD`` alone; turning it on
      there would silently break ``MUL``, ``MAX`` and ``MIN``.
    - **A move none of whose stray lanes survives** -- an exact XOR, a
      broadcast, a shift whose out-of-row lanes a select then discards -- sets
      ``bound_ctrl`` (or passes a zero *old* under full masks, which is the
      same thing). That is what lets ``GCNDPPCombine`` fold the move into
      whichever VALU instruction consumes it, without an identity to check.
    """
    raw = value.ir_value() if isinstance(value, Numeric) else value
    if old is None:
        old = llvm.mlir_poison(raw.type)
    elif isinstance(old, Numeric):
        old = old.ir_value()
    moved = update_dpp(raw.type, old, raw, ctrl, row_mask, bank_mask, bound_ctrl)
    return value.dtype(moved) if isinstance(value, Numeric) else moved


def _swizzle(value, offset):
    """One ``ds_swizzle_b32``, which speaks 32-bit integers and nothing else."""
    dtype = value.dtype
    raw = value if dtype is Int32 else value.bitcast(Int32)
    moved = Int32(ds_swizzle(Int32.ir_type, raw.ir_value(), Int32(offset).ir_value()))
    return moved if dtype is Int32 else moved.bitcast(dtype)


def _dpp_applies(value):
    """Whether the DPP sequences cover this call, or the portable ones have to.

    - **gfx9.** ``row_bcast``, ``wave_shr`` and ``row_ror`` are CDNA-only.
      RDNA dropped them in favour of ``row_share`` / ``permlanex16``, which is
      a different sequence rather than a different constant.
    - **A 32-bit scalar.** ``update_dpp`` and ``ds_swizzle`` both move one
      32-bit lane value; a narrower type would have to be packed and a wider
      one split.
    """
    if not _is_gfx9():
        return False
    return isinstance(value, Numeric) and value.dtype.width == 32


def _replacing(portable, *implementations):
    """An override's ``_implementations``: each function replaces the portable one of its name.

    A name the portable module no longer defines fails here, when the override
    is imported, rather than leaving its implementation silently unused.
    """
    return {getattr(portable, implementation.__name__): implementation for implementation in implementations}
