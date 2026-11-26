# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral micro-scaled and sub-byte quantization codecs."""

from collections.abc import Sequence

from ...expr.numeric import Float32, Int32, Numeric, Uint8, Uint32
from ...expr.primitive import range_constexpr
from ...expr.typing import Vector
from ...expr.utils.arith import ArithValue

__all__ = [
    "f32_to_e8m0",
    "e8m0_to_f32",
    "f32_to_ue4m3",
    "ue4m3_to_f32",
    # Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see below.
    # "ue4m3x2_to_f32",
    "pack_bf16x2",
    "unpack_bf16x2",
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
    "pack_int4x8",
    "unpack_int4x8",
    "unpack_uint4x8",
]

_ScalarInput = Numeric | float | int | ArithValue
_WordInput = Numeric | int | ArithValue
_EntropyInput = _WordInput | Vector | Sequence[_WordInput]

# IEEE-754 Float32 bit masks and special encodings.
_F32_SIGN_BIT = 0x80000000
_F32_ABS_MASK = 0x7FFFFFFF
_F32_EXP_INF_BITS = 0x7F800000
_F32_MANT_MASK = 0x007FFFFF
_F32_QNAN_BITS = 0x7FC00000

# OCP MX E8M0 shared-exponent encoding constants.
_E8M0_SUBNORM_MIN_BITS = 0x00400000  # Exact 2^-127 as an FP32 subnormal (code 0x00).
_E8M0_SUBNORM_MID_BITS = 0x00600000  # Exact 1.5 * 2^-127 RNE midpoint between codes 0 and 1.
_E8M0_RNE_BIAS_BITS = 0x003FFFFF
_E8M0_MAX_FINITE_CODE = 0xFE
_E8M0_NAN_CODE = 0xFF

# Magic FP32 constants whose ULP matches the subnormal quantum of OCP E4M3FN
# (2^-9 -> 1.5 * 2^14 = 24576.0) and OCP E5M2 (2^-16 -> 1.5 * 2^7 = 192.0).
_E4M3_SUBNORM_MAGIC_BITS = 0x46400000
_E4M3_SUBNORM_MAGIC_F32 = 24576.0
_E5M2_SUBNORM_MAGIC_BITS = 0x43400000
_E5M2_SUBNORM_MAGIC_F32 = 192.0

# OCP E4M3FN normal/saturation/overflow thresholds (FP32 bit encodings and 7-bit magnitudes).
_E4M3_MIN_NORM_BITS = 0x3C800000  # 2^-6
_E4M3_RNE_BIAS_BITS = 0x0007FFFF
_E4M3_SAT_THRESHOLD_BITS = 0x43E00000  # 448.0
_E4M3_OVF_RNE_BITS = 0x43E80000  # 464.0
_E4M3_OVF_SR_BITS = 0x43F00000  # 480.0
_E4M3_MAX_FINITE_MAG = 0x7E
_E4M3_NAN_MAG = 0x7F

# OCP E5M2 normal/saturation/overflow thresholds (FP32 bit encodings and 7-bit magnitudes).
_E5M2_MIN_NORM_BITS = 0x38800000  # 2^-14
_E5M2_RNE_BIAS_BITS = 0x000FFFFF
_E5M2_SAT_THRESHOLD_BITS = 0x47600000  # 57344.0
_E5M2_OVF_RNE_BITS = 0x47700000  # 61440.0
_E5M2_OVF_SR_BITS = 0x47800000  # 65536.0
_E5M2_MAX_FINITE_MAG = 0x7B
_E5M2_INF_MAG = 0x7C
_E5M2_QNAN_MAG = 0x7E

# BFloat16 and SWAR nibble masks.
_BF16_RNE_BIAS = 0x7FFF
_BF16_QNAN_BIT = 0x0040
_BF16_HI_MASK = 0xFFFF0000
_NIBBLE_LSB_MASK = 0x11111111

# Golden-ratio mixing constants used when a single 32-bit entropy word drives
# four independent stochastic-rounding lanes within one packed dword.
_SR_MIX_1 = 0x9E3779B9
_SR_MIX_2 = 0x85EBCA6B
_SR_MIX_3 = 0xC2B2AE35


def _as_u32(word: _WordInput) -> Uint32:
    if isinstance(word, Uint32):
        return word
    if isinstance(word, Int32):
        return word.bitcast(Uint32)
    return Uint32(word)


def _as_f32(value: _ScalarInput) -> Float32:
    if isinstance(value, Float32):
        return value
    return Float32(value)


def _extract_8(values: Vector | Sequence[_ScalarInput]) -> tuple[_ScalarInput, ...]:
    if isinstance(values, Vector):
        if values.numel != 8:
            raise ValueError(f"expected 8 elements, got Vector with {values.numel}")
        return tuple(values[i] for i in range_constexpr(8))
    items = tuple(values)
    if len(items) != 8:
        raise ValueError(f"expected 8 elements, got {len(items)}")
    return items


def _expand_entropy4(entropy: _EntropyInput) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    if isinstance(entropy, Vector):
        if entropy.numel != 4:
            raise ValueError(f"expected 4 entropy words, got Vector with {entropy.numel}")
        return (_as_u32(entropy[0]), _as_u32(entropy[1]), _as_u32(entropy[2]), _as_u32(entropy[3]))
    if isinstance(entropy, (tuple, list)):
        if len(entropy) != 4:
            raise ValueError(f"expected 4 entropy words, got {len(entropy)}")
        return (_as_u32(entropy[0]), _as_u32(entropy[1]), _as_u32(entropy[2]), _as_u32(entropy[3]))
    e0 = _as_u32(entropy)
    e1 = (e0 ^ (e0 >> Uint32(16))) * Uint32(_SR_MIX_1)
    e2 = (e1 ^ (e1 >> Uint32(13))) * Uint32(_SR_MIX_2)
    e3 = (e2 ^ (e2 >> Uint32(16))) * Uint32(_SR_MIX_3)
    return e0, e1, e2, e3


def _validate_half(half: int) -> int:
    if not isinstance(half, int) or isinstance(half, bool) or half not in (0, 1):
        raise ValueError(f"half must be Python int 0 or 1, got {half!r}")
    return half


# ---------------------------------------------------------------------------
# Block-scale codecs (OCP MX E8M0 and NVFP4 UE4M3)
# ---------------------------------------------------------------------------


def f32_to_e8m0(x: _ScalarInput, *, round_up: bool = True) -> Uint8:
    """Encode a non-negative ``Float32`` block scale as an OCP MX ``E8M0`` byte.

    OCP MX ``E8M0`` encodes unbiased power-of-two scales ``2^(code - 127)`` for
    ``code`` in ``[0, 254]`` and ``NaN`` at ``0xFF``.

    Args:
        x: Non-negative ``Float32`` (or convertible scalar) scale value.
        round_up: When ``True`` (default), values strictly above a power of two
            round up to the next exponent so quantized elements never exceed the
            target format range. When ``False``, rounds to the nearest-even
            power-of-two exponent.

    Returns:
        ``Uint8`` ``E8M0`` code. ``NaN``, ``Inf``, and negative inputs map to
        ``0xFF``; finite overflow above ``2^127`` saturates to ``0xFE``; zeros
        and magnitudes at or below ``2^-127`` map to ``0x00``.
    """
    val = _as_f32(x)
    bits = val.bitcast(Uint32)
    is_neg = (bits & Uint32(_F32_SIGN_BIT)) != Uint32(0)
    abs_bits = bits & Uint32(_F32_ABS_MASK)
    is_zero = abs_bits == Uint32(0)
    is_nan_or_inf = abs_bits >= Uint32(_F32_EXP_INF_BITS)

    exp = abs_bits >> Uint32(23)
    mant = abs_bits & Uint32(_F32_MANT_MASK)

    if round_up:
        # Normal: increment exponent whenever mantissa is nonzero.
        norm_code = exp + Uint32(mant != Uint32(0))
        # Subnormal (exp == 0): 2^-127 maps to code 0; (2^-127, 2^-126) rounds up to code 1.
        sub_code = (abs_bits > Uint32(_E8M0_SUBNORM_MIN_BITS)).select(Uint32(1), Uint32(0))
    else:
        tie_even = (abs_bits >> Uint32(23)) & Uint32(1)
        norm_code = (abs_bits + Uint32(_E8M0_RNE_BIAS_BITS) + tie_even) >> Uint32(23)
        sub_code = (abs_bits > Uint32(_E8M0_SUBNORM_MID_BITS)).select(Uint32(1), Uint32(0))

    code = (exp == Uint32(0)).select(sub_code, norm_code)
    code = (code > Uint32(_E8M0_MAX_FINITE_CODE)).select(Uint32(_E8M0_MAX_FINITE_CODE), code)
    code = is_zero.select(Uint32(0x00), code)
    invalid = is_nan_or_inf | (is_neg & ~is_zero)
    code = invalid.select(Uint32(_E8M0_NAN_CODE), code)
    return Uint8(code)


def e8m0_to_f32(code: _WordInput) -> Float32:
    """Decode an OCP MX ``E8M0`` scale byte to ``Float32``.

    Args:
        code: ``Uint8`` or integer ``E8M0`` byte in ``[0, 255]``.

    Returns:
        ``Float32`` equal to ``2^(code - 127)`` for ``0 <= code <= 254``
        (including ``2^-127`` as an exact ``Float32`` subnormal when ``code`` is
        ``0``), or quiet ``NaN`` when ``code == 0xFF``.
    """
    raw = _as_u32(code) & Uint32(_E8M0_NAN_CODE)
    bits = raw << Uint32(23)
    bits = (raw == Uint32(0)).select(Uint32(_E8M0_SUBNORM_MIN_BITS), bits)
    bits = (raw == Uint32(_E8M0_NAN_CODE)).select(Uint32(_F32_QNAN_BITS), bits)
    return bits.bitcast(Float32)


def _f32_to_e4m3_u32(val: Float32, *, saturate: bool) -> Uint32:
    bits = val.bitcast(Uint32)
    sign = (bits >> Uint32(24)) & Uint32(0x80)
    abs_bits = bits & Uint32(_F32_ABS_MASK)
    is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)

    # Subnormal / underflow path (|x| < 2^-6): IEEE RNE at ULP 2^-9.
    sub_f32 = abs_bits.bitcast(Float32) + Float32(_E4M3_SUBNORM_MAGIC_F32)
    sub_code = sub_f32.bitcast(Uint32) - Uint32(_E4M3_SUBNORM_MAGIC_BITS)

    # Normal path (2^-6 <= |x| <= 464.0): drop 20 mantissa bits with RNE.
    lsb = (abs_bits >> Uint32(20)) & Uint32(1)
    rounded = abs_bits + Uint32(_E4M3_RNE_BIAS_BITS) + lsb
    norm_code = (rounded >> Uint32(20)) - Uint32(120 << 3)

    mag = (abs_bits < Uint32(_E4M3_MIN_NORM_BITS)).select(sub_code, norm_code)
    if saturate:
        mag = (abs_bits >= Uint32(_E4M3_SAT_THRESHOLD_BITS)).select(Uint32(_E4M3_MAX_FINITE_MAG), mag)
    else:
        mag = (abs_bits > Uint32(_E4M3_OVF_RNE_BITS)).select(Uint32(_E4M3_NAN_MAG), mag)
    mag = is_nan.select(Uint32(_E4M3_NAN_MAG), mag)
    return sign | (mag & Uint32(_E4M3_NAN_MAG))


def _e4m3_byte_to_f32(byte_u32: Uint32) -> Float32:
    b = byte_u32 & Uint32(0xFF)
    sign = (b & Uint32(0x80)) << Uint32(24)
    abs_b = b & Uint32(_E4M3_NAN_MAG)
    exp = abs_b >> Uint32(3)
    mant = abs_b & Uint32(0x7)

    norm_bits = sign | ((exp + Uint32(120)) << Uint32(23)) | (mant << Uint32(20))
    sub_bits = (Float32(Int32(mant)) * Float32(1.0 / 512.0)).bitcast(Uint32) | sign

    out_bits = (exp == Uint32(0)).select(sub_bits, norm_bits)
    out_bits = (abs_b == Uint32(0)).select(sign, out_bits)
    out_bits = (abs_b == Uint32(_E4M3_NAN_MAG)).select(sign | Uint32(_F32_QNAN_BITS), out_bits)
    return out_bits.bitcast(Float32)


def _f32_to_e5m2_u32(val: Float32, *, saturate: bool) -> Uint32:
    bits = val.bitcast(Uint32)
    sign = (bits >> Uint32(24)) & Uint32(0x80)
    abs_bits = bits & Uint32(_F32_ABS_MASK)
    is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)

    # Subnormal / underflow path (|x| < 2^-14): IEEE RNE at ULP 2^-16.
    sub_f32 = abs_bits.bitcast(Float32) + Float32(_E5M2_SUBNORM_MAGIC_F32)
    sub_code = sub_f32.bitcast(Uint32) - Uint32(_E5M2_SUBNORM_MAGIC_BITS)

    # Normal path (2^-14 <= |x| < 61440.0): drop 21 mantissa bits with RNE.
    lsb = (abs_bits >> Uint32(21)) & Uint32(1)
    rounded = abs_bits + Uint32(_E5M2_RNE_BIAS_BITS) + lsb
    norm_code = (rounded >> Uint32(21)) - Uint32(112 << 2)

    mag = (abs_bits < Uint32(_E5M2_MIN_NORM_BITS)).select(sub_code, norm_code)
    if saturate:
        mag = (abs_bits >= Uint32(_E5M2_SAT_THRESHOLD_BITS)).select(Uint32(_E5M2_MAX_FINITE_MAG), mag)
    else:
        mag = (abs_bits >= Uint32(_E5M2_OVF_RNE_BITS)).select(Uint32(_E5M2_INF_MAG), mag)
    mag = is_nan.select(Uint32(_E5M2_QNAN_MAG), mag)
    return sign | (mag & Uint32(0x7F))


def _e5m2_byte_to_f32(byte_u32: Uint32) -> Float32:
    b = byte_u32 & Uint32(0xFF)
    sign = (b & Uint32(0x80)) << Uint32(24)
    abs_b = b & Uint32(0x7F)
    exp = abs_b >> Uint32(2)
    mant = abs_b & Uint32(0x3)

    norm_bits = sign | ((exp + Uint32(112)) << Uint32(23)) | (mant << Uint32(21))
    sub_bits = (Float32(Int32(mant)) * Float32(1.0 / 65536.0)).bitcast(Uint32) | sign

    out_bits = (exp == Uint32(0)).select(sub_bits, norm_bits)
    out_bits = (abs_b == Uint32(0)).select(sign, out_bits)
    out_bits = (abs_b == Uint32(_E5M2_INF_MAG)).select(sign | Uint32(_F32_EXP_INF_BITS), out_bits)
    out_bits = (abs_b > Uint32(_E5M2_INF_MAG)).select(sign | Uint32(_F32_QNAN_BITS), out_bits)
    return out_bits.bitcast(Float32)


def f32_to_ue4m3(x: _ScalarInput) -> Uint8:
    """Encode a non-negative ``Float32`` block scale as a ``UE4M3`` byte.

    ``UE4M3`` is the unsigned/non-negative subset of OCP ``E4M3FN`` (``sign=0``,
    bias 7, max finite ``448.0`` at ``0x7E``, ``NaN`` at ``0x7F``) used for
    per-16-element NVFP4 micro-block scales.

    Args:
        x: ``Float32`` (or convertible scalar) scale value.

    Returns:
        ``Uint8`` ``UE4M3`` code in ``[0x00, 0x7F]``. Negative finite inputs
        clamp to ``0x00``; finite/infinite values above ``448.0`` saturate to
        ``0x7E``; ``NaN`` maps to ``0x7F``.
    """
    val = _as_f32(x)
    bits = val.bitcast(Uint32)
    is_nan = (bits & Uint32(_F32_ABS_MASK)) > Uint32(_F32_EXP_INF_BITS)
    is_neg = (bits & Uint32(_F32_SIGN_BIT)) != Uint32(0)
    clamped = is_neg.select(Float32(0.0), val)
    code = _f32_to_e4m3_u32(clamped, saturate=True) & Uint32(_E4M3_NAN_MAG)
    code = is_nan.select(Uint32(_E4M3_NAN_MAG), code)
    return Uint8(code)


def ue4m3_to_f32(code: _WordInput) -> Float32:
    """Decode a ``UE4M3`` micro-block scale byte to ``Float32``.

    Args:
        code: ``Uint8`` or integer ``UE4M3`` code (low 7 bits; bit 7 is masked
            off so the decoded scale is always non-negative or ``NaN``).

    Returns:
        ``Float32`` scale in ``[0.0, 448.0]``, or ``NaN`` when
        ``(code & 0x7F) == 0x7F``.
    """
    return _e4m3_byte_to_f32(_as_u32(code) & Uint32(_E4M3_NAN_MAG))


# NOTE: ``ue4m3x2_to_f32`` is intentionally disabled until upstream LLVM PR
# https://github.com/llvm/llvm-project/pull/231721 lands in FlyDSL's pinned LLVM
# commit (tracked in https://github.com/ROCm/FlyDSL/issues/1376).
# Its ROCDL hardware override decodes two adjacent ``UE4M3`` micro-block scale
# bytes in a single ``cvt_pk_f32_fp8`` instruction; in unmodified LLVM
# (``AMDGPU/VOP1Instructions.td``), ``HasCvtFP8ByteSel`` defines ISel patterns only
# for ``V_CVT_PK_F32_{FP8,BF8}_fake16`` and omits ``UseRealTrue16Insts`` (``_t16``)
# patterns, failing MC assembly on ``+real-true16`` targets (``gfx120x``/``gfx125x``).
# Decode individual scale bytes with ``ue4m3_to_f32`` until the LLVM fix merges.
#
# def ue4m3x2_to_f32(packed_pair: _WordInput) -> tuple[Float32, Float32]:
#     """Decode two packed ``UE4M3`` micro-block scale bytes to ``Float32``.
#
#     Args:
#         packed_pair: ``Uint32`` or ``Int32`` holding two ``UE4M3`` scale bytes
#             in bits ``[7:0]`` and ``[15:8]`` (bit 7 of each byte is masked off
#             so decoded scales are always non-negative or ``NaN``).
#
#     Returns:
#         Tuple ``(s0, s1)`` of ``Float32`` scales in ``[0.0, 448.0]``, or ``NaN``
#         when the masked 7-bit code equals ``0x7F``.
#     """
#     word = _as_u32(packed_pair)
#     return ue4m3_to_f32(word), ue4m3_to_f32(word >> Uint32(8))


# ---------------------------------------------------------------------------
# BFloat16 packed register helpers
# ---------------------------------------------------------------------------


def _f32_to_bf16_u16(val: Float32) -> Uint32:
    bits = val.bitcast(Uint32)
    is_nan = (bits & Uint32(_F32_ABS_MASK)) > Uint32(_F32_EXP_INF_BITS)
    lsb = (bits >> Uint32(16)) & Uint32(1)
    rounded = (bits + Uint32(_BF16_RNE_BIAS) + lsb) >> Uint32(16)
    nan_u16 = ((bits >> Uint32(16)) | Uint32(_BF16_QNAN_BIT)) & Uint32(0xFFFF)
    return is_nan.select(nan_u16, rounded & Uint32(0xFFFF))


def pack_bf16x2(v0: _ScalarInput, v1: _ScalarInput) -> Uint32:
    """Round two ``Float32`` values to ``BFloat16`` (RNE) and pack into ``Uint32``.

    Args:
        v0: Low-half ``Float32`` value (placed in bits ``[15:0]``).
        v1: High-half ``Float32`` value (placed in bits ``[31:16]``).

    Returns:
        Packed ``Uint32`` holding ``(bf16(v1) << 16) | bf16(v0)``.
    """
    lo = _f32_to_bf16_u16(_as_f32(v0))
    hi = _f32_to_bf16_u16(_as_f32(v1))
    return lo | (hi << Uint32(16))


def unpack_bf16x2(packed: _WordInput) -> tuple[Float32, Float32]:
    """Unpack a packed ``bf16x2`` 32-bit word into two exact ``Float32`` values.

    Args:
        packed: ``Uint32`` or ``Int32`` holding two ``BFloat16`` values.

    Returns:
        Tuple ``(v0, v1)`` of ``Float32`` values from the low and high halves.
    """
    word = _as_u32(packed)
    v0 = (word << Uint32(16)).bitcast(Float32)
    v1 = (word & Uint32(_BF16_HI_MASK)).bitcast(Float32)
    return v0, v1


# ---------------------------------------------------------------------------
# OCP FP8 (E4M3FN and E5M2) pack, unpack, BF16x2 bridge, stochastic rounding
# ---------------------------------------------------------------------------


def _prescale(val: Float32, scale_inv: _ScalarInput | None) -> Float32:
    if scale_inv is None:
        return val
    return val * _as_f32(scale_inv)


def _postscale4(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    scale: _ScalarInput | None,
) -> tuple[Float32, Float32, Float32, Float32]:
    if scale is None:
        return v0, v1, v2, v3
    s = _as_f32(scale)
    return v0 * s, v1 * s, v2 * s, v3 * s


def pack_e4m3x2(
    v0: _ScalarInput,
    v1: _ScalarInput,
    *,
    scale_inv: _ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """Quantize two ``Float32`` values into the low 16 bits of a ``Uint32`` as OCP ``E4M3FN``.

    Args:
        v0: First ``Float32`` value (bits ``[7:0]``).
        v1: Second ``Float32`` value (bits ``[15:8]``).
        scale_inv: Optional inverse block scale multiplied into inputs before rounding.
        saturate: Clamp finite/infinite overflow to ``±448.0`` when ``True`` (default).

    Returns:
        ``Uint32`` with packed ``E4M3FN`` bytes in bits ``[15:0]`` and zeros above.
    """
    b0 = _f32_to_e4m3_u32(_prescale(_as_f32(v0), scale_inv), saturate=saturate)
    b1 = _f32_to_e4m3_u32(_prescale(_as_f32(v1), scale_inv), saturate=saturate)
    return b0 | (b1 << Uint32(8))


def pack_e4m3x4(
    v0: _ScalarInput,
    v1: _ScalarInput,
    v2: _ScalarInput,
    v3: _ScalarInput,
    *,
    scale_inv: _ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """Quantize four ``Float32`` values into one ``Uint32`` of four OCP ``E4M3FN`` bytes.

    Args:
        v0: Element 0 (bits ``[7:0]``).
        v1: Element 1 (bits ``[15:8]``).
        v2: Element 2 (bits ``[23:16]``).
        v3: Element 3 (bits ``[31:24]``).
        scale_inv: Optional inverse block scale multiplied into inputs before rounding.
        saturate: Clamp finite/infinite overflow to ``±448.0`` when ``True`` (default).

    Returns:
        Packed ``Uint32`` containing four OCP ``E4M3FN`` bytes.
    """
    lo = pack_e4m3x2(v0, v1, scale_inv=scale_inv, saturate=saturate)
    hi = pack_e4m3x2(v2, v3, scale_inv=scale_inv, saturate=saturate)
    return lo | (hi << Uint32(16))


def unpack_e4m3x4(
    packed: _WordInput,
    *,
    scale: _ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Unpack four OCP ``E4M3FN`` bytes from a 32-bit word into ``Float32`` scalars.

    Args:
        packed: ``Uint32`` or ``Int32`` holding four ``E4M3FN`` bytes.
        scale: Optional ``Float32`` block scale multiplied into each decoded output.

    Returns:
        4-tuple ``(v0, v1, v2, v3)`` of ``Float32`` values.
    """
    word = _as_u32(packed)
    v0 = _e4m3_byte_to_f32(word)
    v1 = _e4m3_byte_to_f32(word >> Uint32(8))
    v2 = _e4m3_byte_to_f32(word >> Uint32(16))
    v3 = _e4m3_byte_to_f32(word >> Uint32(24))
    return _postscale4(v0, v1, v2, v3, scale)


def _e4m3_byte_to_bf16_u16(byte_u32: Uint32) -> Uint32:
    # Every OCP E4M3FN value (bias 7, 3 mantissa bits, subnormals >= 2^-9) is
    # 0-ULP exact in BFloat16 (bias 127, 7 mantissa bits).
    return _e4m3_byte_to_f32(byte_u32).bitcast(Uint32) >> Uint32(16)


def unpack_e4m3x4_bf16x2(packed: _WordInput) -> tuple[Uint32, Uint32]:
    """Losslessly unpack four OCP ``E4M3FN`` bytes into two packed ``bf16x2`` ``Uint32`` words.

    Every finite, zero, and ``NaN`` OCP ``E4M3FN`` code has an exact ``BFloat16``
    representation (0 ULP error), making this directly usable with BF16 dot/WMMA
    or register dequantization pipelines without touching 16-bit subregisters.

    Args:
        packed: ``Uint32`` or ``Int32`` holding four ``E4M3FN`` bytes.

    Returns:
        Tuple ``(bf16_01, bf16_23)`` of packed ``bf16x2`` ``Uint32`` words.
    """
    word = _as_u32(packed)
    b0 = _e4m3_byte_to_bf16_u16(word)
    b1 = _e4m3_byte_to_bf16_u16(word >> Uint32(8))
    b2 = _e4m3_byte_to_bf16_u16(word >> Uint32(16))
    b3 = _e4m3_byte_to_bf16_u16(word >> Uint32(24))
    return b0 | (b1 << Uint32(16)), b2 | (b3 << Uint32(16))


def pack_e5m2x2(
    v0: _ScalarInput,
    v1: _ScalarInput,
    *,
    scale_inv: _ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """Quantize two ``Float32`` values into the low 16 bits of a ``Uint32`` as OCP ``E5M2``.

    Args:
        v0: First ``Float32`` value (bits ``[7:0]``).
        v1: Second ``Float32`` value (bits ``[15:8]``).
        scale_inv: Optional inverse block scale multiplied into inputs before rounding.
        saturate: Clamp finite/infinite overflow to ``±57344.0`` when ``True`` (default).

    Returns:
        ``Uint32`` with packed ``E5M2`` bytes in bits ``[15:0]`` and zeros above.
    """
    b0 = _f32_to_e5m2_u32(_prescale(_as_f32(v0), scale_inv), saturate=saturate)
    b1 = _f32_to_e5m2_u32(_prescale(_as_f32(v1), scale_inv), saturate=saturate)
    return b0 | (b1 << Uint32(8))


def pack_e5m2x4(
    v0: _ScalarInput,
    v1: _ScalarInput,
    v2: _ScalarInput,
    v3: _ScalarInput,
    *,
    scale_inv: _ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """Quantize four ``Float32`` values into one ``Uint32`` of four OCP ``E5M2`` bytes.

    Args:
        v0: Element 0 (bits ``[7:0]``).
        v1: Element 1 (bits ``[15:8]``).
        v2: Element 2 (bits ``[23:16]``).
        v3: Element 3 (bits ``[31:24]``).
        scale_inv: Optional inverse block scale multiplied into inputs before rounding.
        saturate: Clamp finite/infinite overflow to ``±57344.0`` when ``True`` (default).

    Returns:
        Packed ``Uint32`` containing four OCP ``E5M2`` bytes.
    """
    lo = pack_e5m2x2(v0, v1, scale_inv=scale_inv, saturate=saturate)
    hi = pack_e5m2x2(v2, v3, scale_inv=scale_inv, saturate=saturate)
    return lo | (hi << Uint32(16))


def unpack_e5m2x4(
    packed: _WordInput,
    *,
    scale: _ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Unpack four OCP ``E5M2`` bytes from a 32-bit word into ``Float32`` scalars.

    Args:
        packed: ``Uint32`` or ``Int32`` holding four ``E5M2`` bytes.
        scale: Optional ``Float32`` block scale multiplied into each decoded output.

    Returns:
        4-tuple ``(v0, v1, v2, v3)`` of ``Float32`` values.
    """
    word = _as_u32(packed)
    v0 = _e5m2_byte_to_f32(word)
    v1 = _e5m2_byte_to_f32(word >> Uint32(8))
    v2 = _e5m2_byte_to_f32(word >> Uint32(16))
    v3 = _e5m2_byte_to_f32(word >> Uint32(24))
    return _postscale4(v0, v1, v2, v3, scale)


def _e5m2_byte_to_bf16_u16(byte_u32: Uint32) -> Uint32:
    return _e5m2_byte_to_f32(byte_u32).bitcast(Uint32) >> Uint32(16)


def unpack_e5m2x4_bf16x2(packed: _WordInput) -> tuple[Uint32, Uint32]:
    """Losslessly unpack four OCP ``E5M2`` bytes into two packed ``bf16x2`` ``Uint32`` words.

    Args:
        packed: ``Uint32`` or ``Int32`` holding four ``E5M2`` bytes.

    Returns:
        Tuple ``(bf16_01, bf16_23)`` of packed ``bf16x2`` ``Uint32`` words.
    """
    word = _as_u32(packed)
    b0 = _e5m2_byte_to_bf16_u16(word)
    b1 = _e5m2_byte_to_bf16_u16(word >> Uint32(8))
    b2 = _e5m2_byte_to_bf16_u16(word >> Uint32(16))
    b3 = _e5m2_byte_to_bf16_u16(word >> Uint32(24))
    return b0 | (b1 << Uint32(16)), b2 | (b3 << Uint32(16))


def _stoch_round_e4m3_byte(val: Float32, seed: Uint32, *, saturate: bool) -> Uint32:
    bits = val.bitcast(Uint32)
    sign = (bits >> Uint32(24)) & Uint32(0x80)
    abs_bits = bits & Uint32(_F32_ABS_MASK)
    is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)
    is_zero = abs_bits == Uint32(0)

    # Normal range: align top 20 bits of seed with dropped mantissa bits [19:0],
    # matching AMDGPU v_cvt_sr_fp8_f32 bit alignment.
    noise_norm = seed >> Uint32(12)
    norm_rounded = abs_bits + noise_norm
    norm_code = (norm_rounded >> Uint32(20)) - Uint32(120 << 3)

    # Subnormal range (|x| < 2^-6, quantum = 2^-9): add uniform [0, 2^-9) noise
    # from the same top 20 seed bits so Float32 addition never double-rounds.
    noise_sub = Float32(Int32(noise_norm)) * Float32(2.0**-29)
    sub_scaled = (abs_bits.bitcast(Float32) + noise_sub) * Float32(512.0)
    sub_code = Uint32(Int32(sub_scaled))

    mag = (abs_bits < Uint32(_E4M3_MIN_NORM_BITS)).select(sub_code, norm_code)
    mag = is_zero.select(Uint32(0), mag)
    if saturate:
        mag = ((abs_bits >= Uint32(_E4M3_SAT_THRESHOLD_BITS)) | (mag > Uint32(_E4M3_MAX_FINITE_MAG))).select(
            Uint32(_E4M3_MAX_FINITE_MAG), mag
        )
    else:
        mag = ((abs_bits >= Uint32(_E4M3_OVF_SR_BITS)) | (mag >= Uint32(_E4M3_NAN_MAG))).select(
            Uint32(_E4M3_NAN_MAG), mag
        )
    mag = is_nan.select(Uint32(_E4M3_NAN_MAG), mag)
    return sign | (mag & Uint32(_E4M3_NAN_MAG))


def _stoch_round_e5m2_byte(val: Float32, seed: Uint32, *, saturate: bool) -> Uint32:
    bits = val.bitcast(Uint32)
    sign = (bits >> Uint32(24)) & Uint32(0x80)
    abs_bits = bits & Uint32(_F32_ABS_MASK)
    is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)
    is_zero = abs_bits == Uint32(0)

    # Normal range: align top 21 bits of seed with dropped mantissa bits [20:0],
    # matching AMDGPU v_cvt_sr_bf8_f32 bit alignment.
    noise_norm = seed >> Uint32(11)
    norm_rounded = abs_bits + noise_norm
    norm_code = (norm_rounded >> Uint32(21)) - Uint32(112 << 2)

    # Subnormal range (|x| < 2^-14, quantum = 2^-16): add uniform [0, 2^-16) noise
    # from the same top 21 seed bits so Float32 addition never double-rounds.
    noise_sub = Float32(Int32(noise_norm)) * Float32(2.0**-37)
    sub_scaled = (abs_bits.bitcast(Float32) + noise_sub) * Float32(65536.0)
    sub_code = Uint32(Int32(sub_scaled))

    mag = (abs_bits < Uint32(_E5M2_MIN_NORM_BITS)).select(sub_code, norm_code)
    mag = is_zero.select(Uint32(0), mag)
    if saturate:
        mag = ((abs_bits >= Uint32(_E5M2_SAT_THRESHOLD_BITS)) | (mag > Uint32(_E5M2_MAX_FINITE_MAG))).select(
            Uint32(_E5M2_MAX_FINITE_MAG), mag
        )
    else:
        mag = ((abs_bits >= Uint32(_E5M2_OVF_SR_BITS)) | (mag >= Uint32(_E5M2_INF_MAG))).select(
            Uint32(_E5M2_INF_MAG), mag
        )
    mag = is_nan.select(Uint32(_E5M2_QNAN_MAG), mag)
    return sign | (mag & Uint32(0x7F))


def stoch_round_e4m3x4(
    v0: _ScalarInput,
    v1: _ScalarInput,
    v2: _ScalarInput,
    v3: _ScalarInput,
    entropy: _EntropyInput,
    *,
    scale_inv: _ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """Stochastically round four ``Float32`` values into four packed OCP ``E4M3FN`` bytes.

    Pairs directly with :func:`flydsl.extension.random.randint4x` or
    :func:`flydsl.extension.random.randint`.

    Args:
        v0: Element 0 (bits ``[7:0]``).
        v1: Element 1 (bits ``[15:8]``).
        v2: Element 2 (bits ``[23:16]``).
        v3: Element 3 (bits ``[31:24]``).
        entropy: Either one 32-bit random word (expanded into four decorrelated
            lane seeds) or a 4-element tuple/list/``Vector`` of 32-bit random words.
        scale_inv: Optional inverse block scale multiplied into inputs before rounding.
        saturate: Clamp finite/infinite overflow to ``±448.0`` when ``True`` (default).

    Returns:
        Packed ``Uint32`` containing four stochastically rounded OCP ``E4M3FN`` bytes.
    """
    e0, e1, e2, e3 = _expand_entropy4(entropy)
    b0 = _stoch_round_e4m3_byte(_prescale(_as_f32(v0), scale_inv), e0, saturate=saturate)
    b1 = _stoch_round_e4m3_byte(_prescale(_as_f32(v1), scale_inv), e1, saturate=saturate)
    b2 = _stoch_round_e4m3_byte(_prescale(_as_f32(v2), scale_inv), e2, saturate=saturate)
    b3 = _stoch_round_e4m3_byte(_prescale(_as_f32(v3), scale_inv), e3, saturate=saturate)
    return b0 | (b1 << Uint32(8)) | (b2 << Uint32(16)) | (b3 << Uint32(24))


def stoch_round_e5m2x4(
    v0: _ScalarInput,
    v1: _ScalarInput,
    v2: _ScalarInput,
    v3: _ScalarInput,
    entropy: _EntropyInput,
    *,
    scale_inv: _ScalarInput | None = None,
    saturate: bool = True,
) -> Uint32:
    """Stochastically round four ``Float32`` values into four packed OCP ``E5M2`` bytes.

    Args:
        v0: Element 0 (bits ``[7:0]``).
        v1: Element 1 (bits ``[15:8]``).
        v2: Element 2 (bits ``[23:16]``).
        v3: Element 3 (bits ``[31:24]``).
        entropy: Either one 32-bit random word or a 4-element tuple/list/``Vector``
            of 32-bit random words (e.g. from ``fx.random.randint4x``).
        scale_inv: Optional inverse block scale multiplied into inputs before rounding.
        saturate: Clamp finite/infinite overflow to ``±57344.0`` when ``True`` (default).

    Returns:
        Packed ``Uint32`` containing four stochastically rounded OCP ``E5M2`` bytes.
    """
    e0, e1, e2, e3 = _expand_entropy4(entropy)
    b0 = _stoch_round_e5m2_byte(_prescale(_as_f32(v0), scale_inv), e0, saturate=saturate)
    b1 = _stoch_round_e5m2_byte(_prescale(_as_f32(v1), scale_inv), e1, saturate=saturate)
    b2 = _stoch_round_e5m2_byte(_prescale(_as_f32(v2), scale_inv), e2, saturate=saturate)
    b3 = _stoch_round_e5m2_byte(_prescale(_as_f32(v3), scale_inv), e3, saturate=saturate)
    return b0 | (b1 << Uint32(8)) | (b2 << Uint32(16)) | (b3 << Uint32(24))


# ---------------------------------------------------------------------------
# Sub-byte FP4 (OCP E2M1: MXFP4 / NVFP4) and INT4 / UINT4 codecs
# ---------------------------------------------------------------------------


def _f32_to_e2m1_nibble(val: Float32) -> Uint32:
    bits = val.bitcast(Uint32)
    sign_nib = (bits >> Uint32(28)) & Uint32(0x8)
    ax = (bits & Uint32(_F32_ABS_MASK)).bitcast(Float32)
    # Exact IEEE round-to-nearest-even across all 7 E2M1 midpoints:
    # even codes {0: 0.0, 2: 1.0, 4: 2.0, 6: 4.0} win ties (>= below, > above).
    mag = (
        Uint32(ax > Float32(0.25))
        + Uint32(ax >= Float32(0.75))
        + Uint32(ax > Float32(1.25))
        + Uint32(ax >= Float32(1.75))
        + Uint32(ax > Float32(2.5))
        + Uint32(ax >= Float32(3.5))
        + Uint32(ax > Float32(5.0))
    )
    return sign_nib | mag


def _e2m1_nibble_to_bf16_u16(nib_u32: Uint32) -> Uint32:
    nib = nib_u32 & Uint32(0xF)
    sign = (nib & Uint32(0x8)) << Uint32(12)
    mag = nib & Uint32(0x7)
    # Exact BFloat16 upper-16-bit encodings for {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}:
    # mag == 0 -> 0x0000; mag == 1 -> 0x3F00; mag in 2..7 -> ((0x7E + (mag >> 1)) << 7) | ((mag & 1) << 6).
    exp_biased = Uint32(0x7E) + (mag >> Uint32(1))
    mant_bit = (mag & Uint32(1)) << Uint32(6)
    norm_u16 = (exp_biased << Uint32(7)) | mant_bit
    abs_u16 = (mag == Uint32(1)).select(Uint32(0x3F00), norm_u16)
    abs_u16 = (mag == Uint32(0)).select(Uint32(0x0000), abs_u16)
    return sign | abs_u16


def _e2m1_nibble_to_e4m3_byte(nib_u32: Uint32) -> Uint32:
    nib = nib_u32 & Uint32(0xF)
    sign = (nib & Uint32(0x8)) << Uint32(4)
    mag = nib & Uint32(0x7)
    # Exact OCP E4M3FN encodings for {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}:
    # mag == 0 -> 0x00; mag == 1 -> 0x30; mag in 2..7 -> ((6 + (mag >> 1)) << 3) | ((mag & 1) << 2).
    norm_byte = ((Uint32(6) + (mag >> Uint32(1))) << Uint32(3)) | ((mag & Uint32(1)) << Uint32(2))
    abs_byte = (mag == Uint32(1)).select(Uint32(0x30), norm_byte)
    abs_byte = (mag == Uint32(0)).select(Uint32(0x00), abs_byte)
    return sign | abs_byte


def pack_e2m1x8(
    values: Vector | Sequence[_ScalarInput],
    *,
    scale_inv: _ScalarInput | None = None,
) -> Uint32:
    """Quantize eight ``Float32`` values into one ``Uint32`` of eight 4-bit ``E2M1`` nibbles.

    Uses IEEE round-to-nearest-even across the ``E2M1`` grid
    ``{0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}`` (saturating finite and
    infinite magnitudes above ``5.0`` to ``±6.0``).

    Args:
        values: 8-element sequence or ``Vector`` of ``Float32`` values. Element
            ``i`` occupies bits ``[4*i + 3 : 4*i]``.
        scale_inv: Optional inverse micro-block scale multiplied into each input
            before ``E2M1`` rounding.

    Returns:
        Packed ``Uint32`` containing eight ``E2M1`` nibbles.
    """
    elems = _extract_8(values)
    acc = Uint32(0)
    for idx in range_constexpr(8):
        nib = _f32_to_e2m1_nibble(_prescale(_as_f32(elems[idx]), scale_inv))
        acc = acc | (nib << Uint32(idx * 4)) if idx > 0 else nib
    return acc


def unpack_e2m1x4_bf16x2(packed: _WordInput, *, half: int = 0) -> tuple[Uint32, Uint32]:
    """Losslessly unpack four ``E2M1`` nibbles into two packed ``bf16x2`` ``Uint32`` words.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight ``E2M1`` nibbles.
        half: Compile-time Python ``int``: ``0`` selects nibbles ``0..3`` (bits
            ``[15:0]``), ``1`` selects nibbles ``4..7`` (bits ``[31:16]``).

    Returns:
        Tuple ``(bf16_01, bf16_23)`` of packed ``bf16x2`` ``Uint32`` words.
    """
    half = _validate_half(half)
    word = _as_u32(packed) >> Uint32(half * 16)
    b0 = _e2m1_nibble_to_bf16_u16(word)
    b1 = _e2m1_nibble_to_bf16_u16(word >> Uint32(4))
    b2 = _e2m1_nibble_to_bf16_u16(word >> Uint32(8))
    b3 = _e2m1_nibble_to_bf16_u16(word >> Uint32(12))
    return b0 | (b1 << Uint32(16)), b2 | (b3 << Uint32(16))


def unpack_e2m1x4(
    packed: _WordInput,
    *,
    half: int = 0,
    scale: _ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Unpack four ``E2M1`` nibbles into four ``Float32`` scalars.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight ``E2M1`` nibbles.
        half: Compile-time Python ``int``: ``0`` for nibbles ``0..3``, ``1`` for
            nibbles ``4..7``.
        scale: Optional ``Float32`` micro-block scale multiplied into each output.

    Returns:
        4-tuple ``(v0, v1, v2, v3)`` of ``Float32`` values.
    """
    bf01, bf23 = unpack_e2m1x4_bf16x2(packed, half=half)
    v0, v1 = unpack_bf16x2(bf01)
    v2, v3 = unpack_bf16x2(bf23)
    return _postscale4(v0, v1, v2, v3, scale)


def unpack_e2m1x8(
    packed: _WordInput,
    *,
    scale: _ScalarInput | None = None,
) -> tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    """Unpack all eight ``E2M1`` nibbles of a 32-bit word into ``Float32`` scalars.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight ``E2M1`` nibbles.
        scale: Optional ``Float32`` micro-block scale multiplied into each output.

    Returns:
        8-tuple of ``Float32`` values corresponding to nibbles ``0..7``.
    """
    lo = unpack_e2m1x4(packed, half=0, scale=scale)
    hi = unpack_e2m1x4(packed, half=1, scale=scale)
    return (*lo, *hi)


def unpack_e2m1x4_e4m3x4(packed: _WordInput, *, half: int = 0) -> Uint32:
    """Losslessly unpack four ``E2M1`` nibbles into four packed OCP ``E4M3FN`` bytes.

    Every ``E2M1`` value is 0-ULP exact in OCP ``E4M3FN``, enabling zero-loss
    upconversion into FP8 matrix-core accumulators.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight ``E2M1`` nibbles.
        half: Compile-time Python ``int``: ``0`` for nibbles ``0..3``, ``1`` for
            nibbles ``4..7``.

    Returns:
        ``Uint32`` containing four packed OCP ``E4M3FN`` bytes.
    """
    half = _validate_half(half)
    word = _as_u32(packed) >> Uint32(half * 16)
    b0 = _e2m1_nibble_to_e4m3_byte(word)
    b1 = _e2m1_nibble_to_e4m3_byte(word >> Uint32(4))
    b2 = _e2m1_nibble_to_e4m3_byte(word >> Uint32(8))
    b3 = _e2m1_nibble_to_e4m3_byte(word >> Uint32(12))
    return b0 | (b1 << Uint32(8)) | (b2 << Uint32(16)) | (b3 << Uint32(24))


def _swar_negate_nibbles(unsigned_nibbles: Uint32, sign_bits: Uint32) -> Uint32:
    # Negate each 4-bit unsigned nibble u in {0, 1, 2, 3, 4, 6} in place without
    # cross-nibble carry when its sign bit is 1 and u != 0: -u mod 16 = (u ^ 0xF) + 1.
    nonzero = (unsigned_nibbles | (unsigned_nibbles >> Uint32(1)) | (unsigned_nibbles >> Uint32(2))) & Uint32(
        _NIBBLE_LSB_MASK
    )
    neg_nz = sign_bits & nonzero
    mask = neg_nz * Uint32(0xF)
    return (unsigned_nibbles ^ mask) + neg_nz


def _e2m1_mag_to_pq_nibbles(mag: Uint32) -> tuple[Uint32, Uint32]:
    # Magnitude m in 0..7 represents 2*|w| in {0, 1, 2, 3, 4, 6, 8, 12}, split into
    # p in {0, 1, 1, 2, 2, 3, 4, 6} and q in {0, 0, 1, 1, 2, 3, 4, 6} so p + q == 2*|w|.
    p = (mag == Uint32(7)).select(
        Uint32(6),
        (mag == Uint32(6)).select(
            Uint32(4),
            (mag == Uint32(5)).select(Uint32(3), (mag + Uint32(1)) >> Uint32(1)),
        ),
    )
    q = (mag == Uint32(7)).select(
        Uint32(6),
        (mag == Uint32(6)).select(
            Uint32(4),
            (mag == Uint32(5)).select(Uint32(3), mag >> Uint32(1)),
        ),
    )
    return p, q


def unpack_e2m1x8_int4_pair(packed: _WordInput) -> tuple[Uint32, Uint32]:
    """Decompose eight ``E2M1`` nibbles into two lossless signed ``INT4`` words ``(p, q)``.

    Every ``E2M1`` value ``w`` satisfies ``2 * w in {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}``.
    Since a signed 4-bit integer spans ``[-8, 7]``, ``2 * w`` splits with zero
    rounding error into two signed ``INT4`` nibbles ``p + q == 2 * w`` (absorbing
    the factor ``0.5`` into the outer FP32 block scale), allowing ``INT4`` dot and
    matrix instructions to evaluate exact ``E2M1`` products.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight ``E2M1`` nibbles.

    Returns:
        Tuple ``(p_packed, q_packed)`` of ``Uint32`` words, each holding eight
        two's-complement signed ``INT4`` nibbles in ``[-6, 6]``.
    """
    word = _as_u32(packed)
    p_u = Uint32(0)
    q_u = Uint32(0)
    for idx in range_constexpr(8):
        shift = Uint32(idx * 4)
        mag = (word >> shift) & Uint32(0x7)
        pi, qi = _e2m1_mag_to_pq_nibbles(mag)
        p_u = p_u | (pi << shift) if idx > 0 else pi
        q_u = q_u | (qi << shift) if idx > 0 else qi
    sign_bits = (word >> Uint32(3)) & Uint32(_NIBBLE_LSB_MASK)
    return _swar_negate_nibbles(p_u, sign_bits), _swar_negate_nibbles(q_u, sign_bits)


def pack_int4x8(values: Vector | Sequence[_WordInput]) -> Uint32:
    """Pack eight 4-bit integers (low 4 bits of each element) into one ``Uint32``.

    Args:
        values: 8-element sequence or ``Vector`` of integer values. Element ``i``
            is masked to 4 bits and placed in ``[4*i + 3 : 4*i]``.

    Returns:
        Packed ``Uint32`` holding eight 4-bit nibbles.
    """
    elems = _extract_8(values)
    acc = Uint32(0)
    for idx in range_constexpr(8):
        nib = _as_u32(elems[idx]) & Uint32(0xF)
        acc = acc | (nib << Uint32(idx * 4)) if idx > 0 else nib
    return acc


def unpack_int4x8(packed: _WordInput) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Unpack eight two's-complement signed 4-bit integers ``[-8, 7]`` from a 32-bit word.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight signed 4-bit nibbles.

    Returns:
        8-tuple of sign-extended ``Int32`` values in ``[-8, 7]``.
    """
    s_word = _as_u32(packed).bitcast(Int32)
    return (
        (s_word << Int32(28)) >> Int32(28),
        (s_word << Int32(24)) >> Int32(28),
        (s_word << Int32(20)) >> Int32(28),
        (s_word << Int32(16)) >> Int32(28),
        (s_word << Int32(12)) >> Int32(28),
        (s_word << Int32(8)) >> Int32(28),
        (s_word << Int32(4)) >> Int32(28),
        s_word >> Int32(28),
    )


def unpack_uint4x8(packed: _WordInput) -> tuple[Uint32, Uint32, Uint32, Uint32, Uint32, Uint32, Uint32, Uint32]:
    """Unpack eight unsigned 4-bit integers ``[0, 15]`` from a 32-bit word.

    Args:
        packed: ``Uint32`` or ``Int32`` holding eight unsigned 4-bit nibbles.

    Returns:
        8-tuple of zero-extended ``Uint32`` values in ``[0, 15]``.
    """
    u_word = _as_u32(packed)
    return (
        u_word & Uint32(0xF),
        (u_word >> Uint32(4)) & Uint32(0xF),
        (u_word >> Uint32(8)) & Uint32(0xF),
        (u_word >> Uint32(12)) & Uint32(0xF),
        (u_word >> Uint32(16)) & Uint32(0xF),
        (u_word >> Uint32(20)) & Uint32(0xF),
        (u_word >> Uint32(24)) & Uint32(0xF),
        (u_word >> Uint32(28)) & Uint32(0xF),
    )
