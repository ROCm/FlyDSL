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
from ...expr.numeric import Float32, Int32
from ...expr.rocdl import ds_swizzle, permlanex16, update_dpp
from ...expr.typing import T

# Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see _warp_fwht_fp8x4_rocdl below.
# from ...expr.numeric import Uint32
# from ...expr.rocdl import cvt_pk_bf8_f32, cvt_pk_f32_bf8, cvt_pk_f32_fp8, cvt_pk_fp8_f32
# from ...expr.typing import Vector
# from ...expr.utils.arith import ArithValue
# from .universal import warp_fwht_e4m3x4 as _portable_warp_fwht_e4m3x4
# from .universal import warp_fwht_e5m2x4 as _portable_warp_fwht_e5m2x4
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
    # Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see below.
    # "warp_fwht_e4m3x4",
    # "warp_fwht_e5m2x4",
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

# Finite saturation bounds and IEEE masks for packed OCP FP8 FWHT (used when ROCm/FlyDSL#1376 lands).
# _E4M3_MAX_FINITE = 448.0
# _E5M2_MAX_FINITE = 57344.0
# _F32_SIGN_BIT = 0x80000000
# _F32_ABS_MASK = 0x7FFFFFFF
# _F32_EXP_INF_BITS = 0x7F800000


def _dpp_f32(val: Float32, ctrl: int) -> Float32:
    moved = update_dpp(
        T.f32,
        val,
        val,
        ctrl,
        _ALL_ROWS,
        _ALL_BANKS,
        False,
    )
    return Float32(moved)


def _permlanex16_xor16_f32(val: Float32) -> Float32:
    raw = val.bitcast(Int32)
    moved = Int32(permlanex16(T.i32, raw, raw, _PERMLANEX16_SWAP_LO, _PERMLANEX16_SWAP_HI, False, False))
    return moved.bitcast(Float32)


def _swizzle_xor_f32(val: Float32, offset: int) -> Float32:
    raw = val.bitcast(Int32)
    swz_imm = (offset << 10) | _SWIZZLE_XOR_MASK
    moved = Int32(ds_swizzle(T.i32, raw, swz_imm))
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


# NOTE: Packed OCP FP8 warp Hadamard overrides (``warp_fwht_e4m3x4`` and
# ``warp_fwht_e5m2x4``) are intentionally disabled until upstream LLVM PR
# https://github.com/llvm/llvm-project/pull/231721 lands in FlyDSL's pinned LLVM
# commit (tracked in https://github.com/ROCm/FlyDSL/issues/1376).
# In unmodified LLVM (``AMDGPU/VOP1Instructions.td``), ``HasCvtFP8ByteSel`` defines
# ISel patterns only for ``V_CVT_PK_F32_{FP8,BF8}_fake16`` and omits
# ``UseRealTrue16Insts`` (``_t16``) patterns, causing 2-instruction
# ``cvt_pk_f32_{fp8,bf8}`` unpacking to fail MC assembly on ``+real-true16``
# targets (``gfx120x``/``gfx125x``). Once ``llvm/llvm-project#231721`` merges,
# uncomment to enable fused packed-FP8 unpack + VALU DPP FWHT + repack.
#
# def _has_ocp_fp8_pk_hw(arch: str) -> bool:
#     return arch.startswith(("gfx95", "gfx120", "gfx125"))
#
#
# def _clamp_ocp_f32(val: Float32, max_finite: float, *, saturate: bool) -> Float32:
#     if not saturate:
#         return val
#     bits = val.bitcast(Uint32)
#     abs_bits = bits & Uint32(_F32_ABS_MASK)
#     is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)
#     sign_bit = bits & Uint32(_F32_SIGN_BIT)
#     max_bits = Float32(max_finite).bitcast(Uint32)
#     sat_val = (sign_bit | max_bits).bitcast(Float32)
#     return is_nan.select(val, (abs_bits > max_bits).select(sat_val, val))
#
#
# def _warp_fwht_fp8x4_rocdl(
#     packed: Int32 | Uint32 | int | ArithValue,
#     *,
#     is_bf8: bool,
#     width: int | None,
#     normalize: bool,
#     saturate: bool,
# ) -> Uint32:
#     arch = current_target().arch
#     if not _has_ocp_fp8_pk_hw(arch):
#         fallback = _portable_warp_fwht_e5m2x4 if is_bf8 else _portable_warp_fwht_e4m3x4
#         return fallback(packed, width=width, normalize=normalize, saturate=saturate)
#     resolved_width = _resolve_warp_width(width)
#     src_u32 = packed if isinstance(packed, Uint32) else (
#         packed.bitcast(Uint32) if isinstance(packed, Int32) else Uint32(packed)
#     )
#     unpack_op = cvt_pk_f32_bf8 if is_bf8 else cvt_pk_f32_fp8
#     pack_op = cvt_pk_bf8_f32 if is_bf8 else cvt_pk_fp8_f32
#     max_finite = _E5M2_MAX_FINITE if is_bf8 else _E4M3_MAX_FINITE
#     lo = Vector(unpack_op(T.f32x2, src_u32, False), shape=(2,), dtype=Float32)
#     hi = Vector(unpack_op(T.f32x2, src_u32, True), shape=(2,), dtype=Float32)
#     regs = _local_fwht_f32([lo[0], lo[1], hi[0], hi[1]])
#     regs = _apply_butterfly_stages_f32(regs, resolved_width, _rocdl_xor_exchange)
#     if normalize:
#         scale = Float32(1.0 / math.sqrt(4 * resolved_width))
#         regs = [r * scale for r in regs]
#     clamped = [_clamp_ocp_f32(r, max_finite, saturate=saturate) for r in regs]
#     pk_lo = pack_op(T.i32, clamped[0], clamped[1], 0, False)
#     pk_all = pack_op(T.i32, clamped[2], clamped[3], pk_lo, True)
#     return Uint32(pk_all)
#
#
# def warp_fwht_e4m3x4(
#     packed: Int32 | Uint32 | int | ArithValue,
#     *,
#     width: int | None = None,
#     normalize: bool = True,
#     saturate: bool = True,
# ) -> Uint32:
#     """ROCDL override for :func:`flydsl.extension.hadamard.universal.warp_fwht_e4m3x4`."""
#     return _warp_fwht_fp8x4_rocdl(packed, is_bf8=False, width=width, normalize=normalize, saturate=saturate)
#
#
# def warp_fwht_e5m2x4(
#     packed: Int32 | Uint32 | int | ArithValue,
#     *,
#     width: int | None = None,
#     normalize: bool = True,
#     saturate: bool = True,
# ) -> Uint32:
#     """ROCDL override for :func:`flydsl.extension.hadamard.universal.warp_fwht_e5m2x4`."""
#     return _warp_fwht_fp8x4_rocdl(packed, is_bf8=True, width=width, normalize=normalize, saturate=saturate)
