# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL-specific hardware overrides for ``flydsl.extension.quant``.

Architecture dispatch rules across AMDGPU generations:

- **OCP FP8 hardware conversion** (``gfx950`` CDNA4, ``gfx120x`` RDNA4,
  ``gfx1250`` CDNA5): ``v_cvt_*_fp8`` / ``v_cvt_*_bf8`` / ``v_cvt_sr_*`` implement
  OCP ``E4M3FN`` and ``E5M2``. Note that ``gfx942`` (CDNA3) uses **FNUZ**
  encodings (bias 8/16, no Inf) for ``v_cvt_*_fp8``/``bf8`` and ``gfx11xx``
  (RDNA3) has no hardware FP8 conversion instructions; both fall back to the
  portable bit-exact VALU paths for OCP FP8.
- **True16 packed FP8 unpack boundary**
  (https://github.com/ROCm/FlyDSL/issues/1376 /
  https://github.com/llvm/llvm-project/pull/231721): On ``HasCvtFP8ByteSel``
  True16 targets (``gfx120x``, ``gfx125x``), FlyDSL's pinned LLVM lacks
  ``UseRealTrue16Insts`` (``_t16``) ISel patterns for
  ``v_cvt_pk_f32_{fp8,bf8}``. Those targets use 32-bit scalar byte-indexed
  ``cvt_f32_{fp8,bf8}`` (``byte_sel=0..3``), while ``gfx95x``
  (``HasCvtFP8SDWASrcSel``, non-True16) uses packed ``cvt_pk_f32_{fp8,bf8}``.
  Once ``llvm/llvm-project#231721`` lands in FlyDSL's LLVM pin, uncommenting the
  True16 gate in ``_has_safe_pk_f32_fp8`` and ``ue4m3x2_to_f32`` below enables
  the 2-op ``cvt_pk_f32_*`` path on ``gfx120x``/``gfx125x`` as well.
- **Native hardware FP4** (``gfx950`` CDNA4 ``FeatureFP4ConversionScaleInsts``):
  uses ``v_cvt_scalef32_pk_fp4_f32``, ``v_cvt_scalef32_pk_f32_fp4``, and
  ``v_cvt_scalef32_pk_bf16_fp4``.
- **1-cycle ``v_perm_b32`` nibble LUTs** (all GCN/CDNA/RDNA targets: ``gfx942``,
  ``gfx950``, ``gfx11xx``, ``gfx120x``, ``gfx1250``): ``llvm.amdgcn.perm`` drives
  8-entry register table lookups for ``E2M1 -> bf16x2``, ``E2M1 -> e4m3x4``, and
  lossless ``E2M1 -> int4`` ``(p, q)`` pair decomposition, plus ``f32x2 -> bf16x2``
  high-half byte gather when unpacking FP8 to ``bf16x2``.
"""

from collections.abc import Sequence

from ...compiler.backends import current_target
from ...expr.numeric import BFloat16, Float32, Uint8, Uint32
from ...expr.primitive import range_constexpr
from ...expr.rocdl import (
    cvt_f32_bf8,
    cvt_f32_fp8,
    cvt_pk_bf8_f32,
    cvt_pk_f32_fp8,
    cvt_pk_fp8_f32,
    cvt_scalef32_pk_bf16_fp4,
    cvt_scalef32_pk_f32_fp4,
    cvt_scalef32_pk_fp4_f32,
    cvt_sr_bf8_f32,
    cvt_sr_fp8_f32,
    perm_b32,
)

# Disabled in expr.rocdl until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see below.
# from ...expr.rocdl import bf8x4_to_bf16x2_pair, cvt_pk_f32_bf8, fp8x4_to_bf16x2_pair
from ...expr.typing import T, Vector
from . import universal as _u

__all__ = [
    "f32_to_ue4m3",
    "ue4m3_to_f32",
    # Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see below.
    # "ue4m3x2_to_f32",
    "pack_e4m3x2",
    "pack_e4m3x4",
    "unpack_e4m3x4",
    "unpack_e4m3x4_bf16x2",
    "stoch_round_e4m3x4",
    "pack_e5m2x2",
    "pack_e5m2x4",
    "unpack_e5m2x4",
    "unpack_e5m2x4_bf16x2",
    "stoch_round_e5m2x4",
    "pack_e2m1x8",
    "unpack_e2m1x4",
    "unpack_e2m1x8",
    "unpack_e2m1x4_bf16x2",
    "unpack_e2m1x4_e4m3x4",
    "unpack_e2m1x8_int4_pair",
]

# Finite saturation bounds for OCP E4M3FN and E5M2.
_E4M3_MAX_FINITE = 448.0
_E5M2_MAX_FINITE = 57344.0

# Mask clearing bit 7 of two packed UE4M3 bytes in bits [15:0] (used when ROCm/FlyDSL#1376 lands).
# _UE4M3_PAIR_MASK = 0x7F7F

# v_perm_b32 selector gathering upper 16 bits of two Float32 lanes into bf16x2.
_PERM_F32_PAIR_TO_BF16X2 = 0x07060302

# 8-byte v_perm_b32 lookup tables for E2M1 magnitudes m in 0..7:
# - BFloat16 [0000, 3F00, 3F80, 3FC0, 4000, 4040, 4080, 40C0] split into low/high bytes.
_E2M1_BF16_LO_HI = 0xC0804000
_E2M1_BF16_LO_LO = 0xC0800000
_E2M1_BF16_HI_HI = 0x40404040
_E2M1_BF16_HI_LO = 0x3F3F3F00

# - OCP E4M3FN [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C].
_E2M1_E4M3_LUT_HI = 0x4C484440
_E2M1_E4M3_LUT_LO = 0x3C383000

# - Lossless INT4 pair p = [0, 1, 1, 2, 2, 3, 4, 6], q = [0, 0, 1, 1, 2, 3, 4, 6].
_E2M1_INT4_PQ_HI = 0x06040302
_E2M1_INT4_P_LO = 0x02010100
_E2M1_INT4_Q_LO = 0x01010000


def _arch() -> str:
    return current_target().arch


def _has_ocp_fp8_hw(arch: str) -> bool:
    """Return True iff *arch* has hardware OCP E4M3FN / E5M2 conversion ISA."""
    return arch.startswith(("gfx95", "gfx120", "gfx125"))


def _has_safe_pk_f32_fp8(arch: str) -> bool:
    """Return True iff ``v_cvt_pk_f32_{fp8,bf8}`` lowers cleanly on pinned LLVM.

    ``gfx95x`` uses SDWA source selection (non-True16) and works today.
    ``gfx120x`` and ``gfx125x`` hit missing ``_t16`` ISel patterns in pinned LLVM
    until https://github.com/llvm/llvm-project/pull/231721
    (https://github.com/ROCm/FlyDSL/issues/1376) is integrated.
    """
    # NOTE: Uncomment the True16 target tuple below once ROCm/FlyDSL#1376
    # (https://github.com/llvm/llvm-project/pull/231721) lands in FlyDSL's pinned
    # LLVM commit. Doing so switches ``_unpack_fp8x4_hw_f32`` (used by
    # ``unpack_e4m3x4``, ``unpack_e4m3x4_bf16x2``, ``unpack_e5m2x4``, and
    # ``unpack_e5m2x4_bf16x2``) on ``gfx120x``/``gfx125x`` from four scalar
    # ``cvt_f32_{fp8,bf8}`` instructions to two packed ``cvt_pk_f32_{fp8,bf8}``
    # instructions.
    # return arch.startswith(("gfx95", "gfx120", "gfx125"))
    return arch.startswith("gfx95")


def _has_fp4_scale_hw(arch: str) -> bool:
    """Return True iff *arch* implements ``FeatureFP4ConversionScaleInsts``."""
    return arch.startswith("gfx95")


def _perm_u32(src_hi: Uint32 | int, src_lo: Uint32 | int, sel: Uint32 | int) -> Uint32:
    return Uint32(perm_b32(src_hi, src_lo, sel))


def _clamp_ocp_f32(val: Float32, max_finite: float, *, saturate: bool) -> Float32:
    if not saturate:
        return val
    bits = val.bitcast(Uint32)
    abs_bits = bits & Uint32(_u._F32_ABS_MASK)
    is_nan = abs_bits > Uint32(_u._F32_EXP_INF_BITS)
    sign_bit = bits & Uint32(_u._F32_SIGN_BIT)
    max_bits = Float32(max_finite).bitcast(Uint32)
    sat_val = (sign_bit | max_bits).bitcast(Float32)
    over = abs_bits > max_bits
    return is_nan.select(val, over.select(sat_val, val))


# ---------------------------------------------------------------------------
# UE4M3 scale overrides
# ---------------------------------------------------------------------------


def f32_to_ue4m3(x: _u._ScalarInput) -> Uint8:
    """ROCDL override for :func:`flydsl.extension.quant.universal.f32_to_ue4m3`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.f32_to_ue4m3(x)
    val = _u._as_f32(x)
    bits = val.bitcast(Uint32)
    is_nan = (bits & Uint32(_u._F32_ABS_MASK)) > Uint32(_u._F32_EXP_INF_BITS)
    is_neg = (bits & Uint32(_u._F32_SIGN_BIT)) != Uint32(0)
    nonneg = is_neg.select(Float32(0.0), val)
    clamped = _clamp_ocp_f32(nonneg, _E4M3_MAX_FINITE, saturate=True)
    packed = Uint32(cvt_pk_fp8_f32(T.i32, clamped, 0.0, 0, False))
    code = is_nan.select(Uint32(_u._E4M3_NAN_MAG), packed & Uint32(_u._E4M3_NAN_MAG))
    return Uint8(code)


def ue4m3_to_f32(code: _u._WordInput) -> Float32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.ue4m3_to_f32`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.ue4m3_to_f32(code)
    masked = _u._as_u32(code) & Uint32(_u._E4M3_NAN_MAG)
    return Float32(cvt_f32_fp8(T.f32, masked, 0))


# NOTE: ``ue4m3x2_to_f32`` is intentionally disabled until upstream LLVM PR
# https://github.com/llvm/llvm-project/pull/231721 lands in FlyDSL's pinned LLVM
# commit (tracked in https://github.com/ROCm/FlyDSL/issues/1376). Once merged
# (alongside enabling ``gfx120``/``gfx125`` in ``_has_safe_pk_f32_fp8`` above and
# ``_UE4M3_PAIR_MASK``), uncomment to decode two adjacent ``UE4M3`` scale bytes
# in one ``v_cvt_pk_f32_fp8`` instruction on ``gfx95x``, ``gfx120x``, and
# ``gfx125x``.
#
# def ue4m3x2_to_f32(packed_pair: _u._WordInput) -> tuple[Float32, Float32]:
#     """ROCDL override for :func:`flydsl.extension.quant.universal.ue4m3x2_to_f32`."""
#     arch = _arch()
#     if not _has_ocp_fp8_hw(arch):
#         return _u.ue4m3x2_to_f32(packed_pair)
#     if _has_safe_pk_f32_fp8(arch):
#         masked = _u._as_u32(packed_pair) & Uint32(_UE4M3_PAIR_MASK)
#         pair = Vector(cvt_pk_f32_fp8(T.f32x2, masked, False), shape=(2,), dtype=Float32)
#         return pair[0], pair[1]
#     word = _u._as_u32(packed_pair)
#     return ue4m3_to_f32(word), ue4m3_to_f32(word >> Uint32(8))


# ---------------------------------------------------------------------------
# OCP FP8 (E4M3FN and E5M2) overrides
# ---------------------------------------------------------------------------


def pack_e4m3x2(
    v0: _u._ScalarInput,
    v1: _u._ScalarInput,
    *,
    scale_inv: _u._ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.pack_e4m3x2`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.pack_e4m3x2(v0, v1, scale_inv=scale_inv, saturate=saturate)
    f0 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v0), scale_inv), _E4M3_MAX_FINITE, saturate=saturate)
    f1 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v1), scale_inv), _E4M3_MAX_FINITE, saturate=saturate)
    return Uint32(cvt_pk_fp8_f32(T.i32, f0, f1, 0, False))


def pack_e4m3x4(
    v0: _u._ScalarInput,
    v1: _u._ScalarInput,
    v2: _u._ScalarInput,
    v3: _u._ScalarInput,
    *,
    scale_inv: _u._ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.pack_e4m3x4`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.pack_e4m3x4(v0, v1, v2, v3, scale_inv=scale_inv, saturate=saturate)
    f0 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v0), scale_inv), _E4M3_MAX_FINITE, saturate=saturate)
    f1 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v1), scale_inv), _E4M3_MAX_FINITE, saturate=saturate)
    f2 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v2), scale_inv), _E4M3_MAX_FINITE, saturate=saturate)
    f3 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v3), scale_inv), _E4M3_MAX_FINITE, saturate=saturate)
    lo = cvt_pk_fp8_f32(T.i32, f0, f1, 0, False)
    return Uint32(cvt_pk_fp8_f32(T.i32, f2, f3, lo, True))


def _unpack_fp8x4_hw_f32(packed: _u._WordInput, *, is_bf8: bool) -> tuple[Float32, Float32, Float32, Float32]:
    arch = _arch()
    src_u32 = _u._as_u32(packed)
    # NOTE: Once ROCm/FlyDSL#1376 (https://github.com/llvm/llvm-project/pull/231721)
    # lands and ``cvt_pk_f32_bf8`` is uncommented in ``flydsl.expr.rocdl``, switch the
    # guard below to ``if _has_safe_pk_f32_fp8(arch):`` and select
    # ``pk_op = cvt_pk_f32_bf8 if is_bf8 else cvt_pk_f32_fp8``.
    if not is_bf8 and _has_safe_pk_f32_fp8(arch):
        lo_vec = Vector(cvt_pk_f32_fp8(T.f32x2, src_u32, False), shape=(2,), dtype=Float32)
        hi_vec = Vector(cvt_pk_f32_fp8(T.f32x2, src_u32, True), shape=(2,), dtype=Float32)
        return lo_vec[0], lo_vec[1], hi_vec[0], hi_vec[1]
    # True16-safe scalar byte-indexed unpack via PR #1375 wrappers
    # (see ROCm/FlyDSL#1376 and llvm/llvm-project#231721).
    scalar_op = cvt_f32_bf8 if is_bf8 else cvt_f32_fp8
    return (
        Float32(scalar_op(T.f32, src_u32, 0)),
        Float32(scalar_op(T.f32, src_u32, 1)),
        Float32(scalar_op(T.f32, src_u32, 2)),
        Float32(scalar_op(T.f32, src_u32, 3)),
    )


def unpack_e4m3x4(
    packed: _u._WordInput,
    *,
    scale: _u._ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e4m3x4`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.unpack_e4m3x4(packed, scale=scale)
    v0, v1, v2, v3 = _unpack_fp8x4_hw_f32(packed, is_bf8=False)
    return _u._postscale4(v0, v1, v2, v3, scale)


def unpack_e4m3x4_bf16x2(packed: _u._WordInput) -> tuple[Uint32, Uint32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e4m3x4_bf16x2`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.unpack_e4m3x4_bf16x2(packed)
    # NOTE: Once ROCm/FlyDSL#1376 (https://github.com/llvm/llvm-project/pull/231721)
    # lands and ``fp8x4_to_bf16x2_pair`` is uncommented in ``flydsl.expr.rocdl``:
    # bf01, bf23 = fp8x4_to_bf16x2_pair(_u._as_u32(packed))
    # return Uint32(bf01), Uint32(bf23)
    v0, v1, v2, v3 = _unpack_fp8x4_hw_f32(packed, is_bf8=False)
    bf01 = _perm_u32(v1.bitcast(Uint32), v0.bitcast(Uint32), _PERM_F32_PAIR_TO_BF16X2)
    bf23 = _perm_u32(v3.bitcast(Uint32), v2.bitcast(Uint32), _PERM_F32_PAIR_TO_BF16X2)
    return bf01, bf23


def pack_e5m2x2(
    v0: _u._ScalarInput,
    v1: _u._ScalarInput,
    *,
    scale_inv: _u._ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.pack_e5m2x2`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.pack_e5m2x2(v0, v1, scale_inv=scale_inv, saturate=saturate)
    f0 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v0), scale_inv), _E5M2_MAX_FINITE, saturate=saturate)
    f1 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v1), scale_inv), _E5M2_MAX_FINITE, saturate=saturate)
    return Uint32(cvt_pk_bf8_f32(T.i32, f0, f1, 0, False))


def pack_e5m2x4(
    v0: _u._ScalarInput,
    v1: _u._ScalarInput,
    v2: _u._ScalarInput,
    v3: _u._ScalarInput,
    *,
    scale_inv: _u._ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.pack_e5m2x4`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.pack_e5m2x4(v0, v1, v2, v3, scale_inv=scale_inv, saturate=saturate)
    f0 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v0), scale_inv), _E5M2_MAX_FINITE, saturate=saturate)
    f1 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v1), scale_inv), _E5M2_MAX_FINITE, saturate=saturate)
    f2 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v2), scale_inv), _E5M2_MAX_FINITE, saturate=saturate)
    f3 = _clamp_ocp_f32(_u._prescale(_u._as_f32(v3), scale_inv), _E5M2_MAX_FINITE, saturate=saturate)
    lo = cvt_pk_bf8_f32(T.i32, f0, f1, 0, False)
    return Uint32(cvt_pk_bf8_f32(T.i32, f2, f3, lo, True))


def unpack_e5m2x4(
    packed: _u._WordInput,
    *,
    scale: _u._ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e5m2x4`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.unpack_e5m2x4(packed, scale=scale)
    v0, v1, v2, v3 = _unpack_fp8x4_hw_f32(packed, is_bf8=True)
    return _u._postscale4(v0, v1, v2, v3, scale)


def unpack_e5m2x4_bf16x2(packed: _u._WordInput) -> tuple[Uint32, Uint32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e5m2x4_bf16x2`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.unpack_e5m2x4_bf16x2(packed)
    # NOTE: Once ROCm/FlyDSL#1376 (https://github.com/llvm/llvm-project/pull/231721)
    # lands and ``bf8x4_to_bf16x2_pair`` is uncommented in ``flydsl.expr.rocdl``:
    # bf01, bf23 = bf8x4_to_bf16x2_pair(_u._as_u32(packed))
    # return Uint32(bf01), Uint32(bf23)
    v0, v1, v2, v3 = _unpack_fp8x4_hw_f32(packed, is_bf8=True)
    bf01 = _perm_u32(v1.bitcast(Uint32), v0.bitcast(Uint32), _PERM_F32_PAIR_TO_BF16X2)
    bf23 = _perm_u32(v3.bitcast(Uint32), v2.bitcast(Uint32), _PERM_F32_PAIR_TO_BF16X2)
    return bf01, bf23


def stoch_round_e4m3x4(
    v0: _u._ScalarInput,
    v1: _u._ScalarInput,
    v2: _u._ScalarInput,
    v3: _u._ScalarInput,
    entropy: _u._EntropyInput,
    *,
    scale_inv: _u._ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.stoch_round_e4m3x4`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.stoch_round_e4m3x4(v0, v1, v2, v3, entropy, scale_inv=scale_inv, saturate=saturate)
    e0, e1, e2, e3 = _u._expand_entropy4(entropy)
    inputs = (
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v0), scale_inv), _E4M3_MAX_FINITE, saturate=saturate), e0),
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v1), scale_inv), _E4M3_MAX_FINITE, saturate=saturate), e1),
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v2), scale_inv), _E4M3_MAX_FINITE, saturate=saturate), e2),
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v3), scale_inv), _E4M3_MAX_FINITE, saturate=saturate), e3),
    )
    acc: object = 0
    for byte_sel in range_constexpr(4):
        fval, seed = inputs[byte_sel]
        # Clamping |fval| <= 448.0 zeros all 20 dropped mantissa bits at +/-448.0,
        # so v_cvt_sr_fp8_f32 never carries past 0x7E/0xFE for any 32-bit seed.
        acc = cvt_sr_fp8_f32(T.i32, fval, seed, acc, byte_sel)
    return Uint32(acc)


def stoch_round_e5m2x4(
    v0: _u._ScalarInput,
    v1: _u._ScalarInput,
    v2: _u._ScalarInput,
    v3: _u._ScalarInput,
    entropy: _u._EntropyInput,
    *,
    scale_inv: _u._ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.stoch_round_e5m2x4`."""
    if not _has_ocp_fp8_hw(_arch()):
        return _u.stoch_round_e5m2x4(v0, v1, v2, v3, entropy, scale_inv=scale_inv, saturate=saturate)
    e0, e1, e2, e3 = _u._expand_entropy4(entropy)
    inputs = (
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v0), scale_inv), _E5M2_MAX_FINITE, saturate=saturate), e0),
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v1), scale_inv), _E5M2_MAX_FINITE, saturate=saturate), e1),
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v2), scale_inv), _E5M2_MAX_FINITE, saturate=saturate), e2),
        (_clamp_ocp_f32(_u._prescale(_u._as_f32(v3), scale_inv), _E5M2_MAX_FINITE, saturate=saturate), e3),
    )
    acc: object = 0
    for byte_sel in range_constexpr(4):
        fval, seed = inputs[byte_sel]
        acc = cvt_sr_bf8_f32(T.i32, fval, seed, acc, byte_sel)
    return Uint32(acc)


# ---------------------------------------------------------------------------
# Sub-byte E2M1 (MXFP4 / NVFP4) overrides (gfx950 FP4 ISA + v_perm_b32 LUTs)
# ---------------------------------------------------------------------------


def pack_e2m1x8(
    values: Vector | Sequence[_u._ScalarInput],
    *,
    scale_inv: _u._ScalarInput | None = None,
) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.pack_e2m1x8`."""
    if not _has_fp4_scale_hw(_arch()):
        return _u.pack_e2m1x8(values, scale_inv=scale_inv)
    elems = [_u._prescale(_u._as_f32(v), scale_inv) for v in _u._extract_8(values)]
    acc: object = 0
    for sel in range_constexpr(4):
        acc = cvt_scalef32_pk_fp4_f32(T.i32, acc, elems[2 * sel], elems[2 * sel + 1], 1.0, sel)
    return Uint32(acc)


def _perm_e2m1_pair_bf16x2(nibble_lo: Uint32, nibble_hi: Uint32) -> Uint32:
    """Decode two ``E2M1`` nibbles into one packed ``bf16x2`` ``Uint32`` via ``v_perm_b32``.

    Splits the 8-entry ``BFloat16`` magnitude table
    ``[0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0]`` into
    low-byte and high-byte 8-entry tables so two ``v_perm_b32`` lookups cover
    all eight ``E2M1`` magnitudes simultaneously.
    """
    m0 = nibble_lo & Uint32(0x7)
    m1 = nibble_hi & Uint32(0x7)
    s0 = (nibble_lo & Uint32(0x8)) << Uint32(12)
    s1 = (nibble_hi & Uint32(0x8)) << Uint32(28)

    lo_bytes = _perm_u32(_E2M1_BF16_LO_HI, _E2M1_BF16_LO_LO, m0 | (m1 << Uint32(16)))
    hi_bytes = _perm_u32(_E2M1_BF16_HI_HI, _E2M1_BF16_HI_LO, (m0 << Uint32(8)) | (m1 << Uint32(24)))
    mag_bf16x2 = (lo_bytes & Uint32(0x00FF00FF)) | (hi_bytes & Uint32(0xFF00FF00))
    return mag_bf16x2 | s0 | s1


def unpack_e2m1x4_bf16x2(packed: _u._WordInput, *, half: int = 0) -> tuple[Uint32, Uint32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e2m1x4_bf16x2`."""
    half = _u._validate_half(half)
    if _has_fp4_scale_hw(_arch()):
        src_u32 = _u._as_u32(packed)
        b01 = Vector(
            cvt_scalef32_pk_bf16_fp4(T.bf16x2, src_u32, 1.0, 2 * half),
            shape=(2,),
            dtype=BFloat16,
        ).bitcast(
            Uint32
        )[0]
        b23 = Vector(
            cvt_scalef32_pk_bf16_fp4(T.bf16x2, src_u32, 1.0, 2 * half + 1),
            shape=(2,),
            dtype=BFloat16,
        ).bitcast(Uint32)[0]
        return b01, b23

    word = _u._as_u32(packed) >> Uint32(half * 16)
    bf01 = _perm_e2m1_pair_bf16x2(word, word >> Uint32(4))
    bf23 = _perm_e2m1_pair_bf16x2(word >> Uint32(8), word >> Uint32(12))
    return bf01, bf23


def unpack_e2m1x4(
    packed: _u._WordInput,
    *,
    half: int = 0,
    scale: _u._ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e2m1x4`."""
    half = _u._validate_half(half)
    if _has_fp4_scale_hw(_arch()):
        src_u32 = _u._as_u32(packed)
        scale_val = 1.0 if scale is None else _u._as_f32(scale)
        p0 = Vector(
            cvt_scalef32_pk_f32_fp4(T.f32x2, src_u32, scale_val, 2 * half),
            shape=(2,),
            dtype=Float32,
        )
        p1 = Vector(
            cvt_scalef32_pk_f32_fp4(T.f32x2, src_u32, scale_val, 2 * half + 1),
            shape=(2,),
            dtype=Float32,
        )
        return p0[0], p0[1], p1[0], p1[1]

    bf01, bf23 = unpack_e2m1x4_bf16x2(packed, half=half)
    v0, v1 = _u.unpack_bf16x2(bf01)
    v2, v3 = _u.unpack_bf16x2(bf23)
    return _u._postscale4(v0, v1, v2, v3, scale)


def unpack_e2m1x8(
    packed: _u._WordInput,
    *,
    scale: _u._ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e2m1x8`."""
    lo = unpack_e2m1x4(packed, half=0, scale=scale)
    hi = unpack_e2m1x4(packed, half=1, scale=scale)
    return (*lo, *hi)


def unpack_e2m1x4_e4m3x4(packed: _u._WordInput, *, half: int = 0) -> Uint32:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e2m1x4_e4m3x4`.

    Decodes all four ``E2M1`` magnitudes simultaneously in one ``v_perm_b32``
    instruction against the 8-byte OCP ``E4M3FN`` table
    ``[0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C]``.
    """
    half = _u._validate_half(half)
    word = _u._as_u32(packed) >> Uint32(half * 16)
    m0 = word & Uint32(0x07)
    m1 = (word >> Uint32(4)) & Uint32(0x07)
    m2 = (word >> Uint32(8)) & Uint32(0x07)
    m3 = (word >> Uint32(12)) & Uint32(0x07)
    sel = m0 | (m1 << Uint32(8)) | (m2 << Uint32(16)) | (m3 << Uint32(24))
    mag_bytes = _perm_u32(_E2M1_E4M3_LUT_HI, _E2M1_E4M3_LUT_LO, sel)

    signs = (
        ((word & Uint32(0x0008)) << Uint32(4))
        | ((word & Uint32(0x0080)) << Uint32(8))
        | ((word & Uint32(0x0800)) << Uint32(12))
        | ((word & Uint32(0x8000)) << Uint32(16))
    )
    return mag_bytes | signs


def _pack_bytes_to_nibbles(lo4: Uint32, hi4: Uint32) -> Uint32:
    """Pack low nibbles of four bytes in ``lo4`` and ``hi4`` into one 32-bit nibble word."""
    lo_even = (lo4 & Uint32(0x000F000F)) | ((lo4 & Uint32(0x0F000F00)) >> Uint32(4))
    lo_u16 = (lo_even & Uint32(0x00FF)) | ((lo_even >> Uint32(8)) & Uint32(0xFF00))
    hi_even = (hi4 & Uint32(0x000F000F)) | ((hi4 & Uint32(0x0F000F00)) >> Uint32(4))
    hi_u16 = (hi_even & Uint32(0x00FF)) | ((hi_even >> Uint32(8)) & Uint32(0xFF00))
    return lo_u16 | (hi_u16 << Uint32(16))


def unpack_e2m1x8_int4_pair(packed: _u._WordInput) -> tuple[Uint32, Uint32]:
    """ROCDL override for :func:`flydsl.extension.quant.universal.unpack_e2m1x8_int4_pair`.

    Uses ``v_perm_b32`` table lookups + SWAR sign negation with zero cross-nibble carry.
    """
    word = _u._as_u32(packed)
    sel_lo = (
        (word & Uint32(0x7))
        | (((word >> Uint32(4)) & Uint32(0x7)) << Uint32(8))
        | (((word >> Uint32(8)) & Uint32(0x7)) << Uint32(16))
        | (((word >> Uint32(12)) & Uint32(0x7)) << Uint32(24))
    )
    sel_hi = (
        ((word >> Uint32(16)) & Uint32(0x7))
        | (((word >> Uint32(20)) & Uint32(0x7)) << Uint32(8))
        | (((word >> Uint32(24)) & Uint32(0x7)) << Uint32(16))
        | (((word >> Uint32(28)) & Uint32(0x7)) << Uint32(24))
    )
    p_lo4 = _perm_u32(_E2M1_INT4_PQ_HI, _E2M1_INT4_P_LO, sel_lo)
    p_hi4 = _perm_u32(_E2M1_INT4_PQ_HI, _E2M1_INT4_P_LO, sel_hi)
    q_lo4 = _perm_u32(_E2M1_INT4_PQ_HI, _E2M1_INT4_Q_LO, sel_lo)
    q_hi4 = _perm_u32(_E2M1_INT4_PQ_HI, _E2M1_INT4_Q_LO, sel_hi)

    p_u = _pack_bytes_to_nibbles(p_lo4, p_hi4)
    q_u = _pack_bytes_to_nibbles(q_lo4, q_hi4)
    sign_bits = (word >> Uint32(3)) & Uint32(_u._NIBBLE_LSB_MASK)
    return _u._swar_negate_nibbles(p_u, sign_bits), _u._swar_negate_nibbles(q_u, sign_bits)
