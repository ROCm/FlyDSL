# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL overrides for Fast Walsh-Hadamard Transform collectives.

On wave32 GFX10+/RDNA/CDNA5 targets (``gfx10*``, ``gfx11*``, ``gfx12*``), cross-lane
XOR butterflies use ``row_xmask:1/2/4/8`` DPP and ``v_permlanex16_b32``, keeping
all intra-warp stages 100% in VALU with zero LDS instructions.

On wave64 CDNA targets (``gfx9*``), exact XOR distances 1, 2, and 8 use VALU DPP
(``quad_perm`` and ``row_ror:8``), distance 4 uses ``ds_swizzle_b32`` (plus distance
16 on pre-CDNA4 ``gfx90*``/``gfx94*``), while CDNA4 (``gfx95*``) lowers distances 16
and 32 to VALU ``v_permlane16_swap_b32`` and ``v_permlane32_swap_b32``.
"""

import math

from ...compiler.backends import current_target
from ...expr.gpu import shuffle_xor
from ...expr.numeric import Float32, Int32, Uint32
from ...expr.rocdl import ds_swizzle, permlanex16, update_dpp
from .universal import (
    _apply_butterfly_stages_f32,
    _FWHTOutput,
    _FWHTValue,
    _local_fwht_f32,
    _portable_xor_exchange,
    _repack_items,
    _resolve_warp_width,
    _unpack_items,
)

__all__ = [
    "warp_fwht",
]

_ALL_ROWS = 0xF
_ALL_BANKS = 0xF

# v_permlanex16_b32 lane-select words swapping lane l in 0..15 with lane l ^ 16 in 16..31.
_PERMLANEX16_SWAP_LO = 0x76543210
_PERMLANEX16_SWAP_HI = 0xFEDCBA98

# ds_swizzle_b32 XOR mode base mask (AND mask = 0x1F; 5-bit XOR mask sits at bits [14:10]).
_SWIZZLE_XOR_MASK = 0x1F

# Exact XOR DPP controls on wave32 GFX10+/RDNA/CDNA5 (row_xmask:1/2/4/8).
_RDNA_XOR_DPP = {1: 0x161, 2: 0x162, 4: 0x164, 8: 0x168}

# Exact XOR DPP controls on wave64 CDNA: quad_perm[1,0,3,2], quad_perm[2,3,0,1],
# and row_ror:8 (since (r + 8) mod 16 == r ^ 8 for all r in 0..15).
# Note: row_half_mirror (7 - r) is NOT r ^ 4, so distance 4 uses ds_swizzle_b32.
_CDNA_EXACT_XOR_DPP = {1: 0xB1, 2: 0x4E, 8: 0x128}


def _dpp_f32(val: Float32, ctrl: int) -> Float32:
    moved = update_dpp(
        Float32.ir_type,
        val.ir_value(),
        val.ir_value(),
        ctrl,
        _ALL_ROWS,
        _ALL_BANKS,
        False,
    )
    return Float32(moved)


def _permlanex16_xor16_f32(val: Float32) -> Float32:
    raw = val.bitcast(Int32)
    s1 = Uint32(_PERMLANEX16_SWAP_LO).bitcast(Int32).ir_value()
    s2 = Uint32(_PERMLANEX16_SWAP_HI).bitcast(Int32).ir_value()
    moved = Int32(permlanex16(Int32.ir_type, raw.ir_value(), raw.ir_value(), s1, s2, False, False))
    return moved.bitcast(Float32)


def _swizzle_xor_f32(val: Float32, offset: int) -> Float32:
    raw = val.bitcast(Int32)
    swz_imm = (offset << 10) | _SWIZZLE_XOR_MASK
    moved = Int32(ds_swizzle(Int32.ir_type, raw.ir_value(), Int32(swz_imm).ir_value()))
    return moved.bitcast(Float32)


def _rocdl_xor_exchange(val: Float32, offset: int, width: int) -> Float32:
    target = current_target()
    arch = target.arch
    if target.warp_size == 32 and arch.startswith(("gfx10", "gfx11", "gfx12")):
        if offset in _RDNA_XOR_DPP:
            return _dpp_f32(val, _RDNA_XOR_DPP[offset])
        if offset == 16:
            return _permlanex16_xor16_f32(val)
        return _portable_xor_exchange(val, offset, width)

    if arch.startswith("gfx9"):
        if offset in _CDNA_EXACT_XOR_DPP:
            return _dpp_f32(val, _CDNA_EXACT_XOR_DPP[offset])
        if offset == 4 or (offset == 16 and not arch.startswith("gfx95")):
            return _swizzle_xor_f32(val, offset)
        return shuffle_xor(val, offset, width)

    return _portable_xor_exchange(val, offset, width)


def warp_fwht(
    value: _FWHTValue,
    *,
    width: int | None = None,
    normalize: bool = False,
) -> _FWHTOutput:
    """ROCDL-accelerated Sylvester Fast Walsh-Hadamard Transform across a logical warp."""
    resolved_width = _resolve_warp_width(width)
    regs, elem_dtype, kind = _unpack_items(value)
    regs = _local_fwht_f32(regs)
    regs = _apply_butterfly_stages_f32(regs, resolved_width, _rocdl_xor_exchange)
    if normalize:
        scale = Float32(1.0 / math.sqrt(resolved_width * len(regs)))
        regs = [r * scale for r in regs]
    return _repack_items(regs, elem_dtype, kind)
