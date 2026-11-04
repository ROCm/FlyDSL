# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FP8 / MXFP8 arithmetic of the V4.1 stages, identical to the MI350 mono
kernels' (the numerics are part of the contract, not of the architecture).

The original path quantizes activations by three rules and each stage uses the
one its original kernel does:

- ``code_ceil`` (ATOM ``quantize_fp8``, the window row, aiter ``rmsnorm_quant``
  MXFP8): the smallest power of two >= amax / 448;
- ``code_round`` (aiter Triton ``_mxfp8_quant_op``, the q_lora norm): amax's
  exponent after adding 0x200000 to its bits, minus 8.

A value is quantized as ``clamp_fp8(x * rcp_pow2(code))`` and converted with
``cvt_pk_fp8_f32`` (round to nearest even). The conversion's CLAMP bit is never
set: on MI450-B0 it selects E5M3.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

FP8_MAX = 448.0
UNIT_SCALE = 127  # the E8M0 code of 1.0
ROUND_BIAS = 0x200000
EXP_MASK = -8388608  # 0xFF800000
FP8_EXP = 8  # log2(448) rounded down


def clamp_fp8(v):
    return fx.max(fx.min(v, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


def code_ceil(v):
    """Biased exponent of the smallest power of two >= ``v`` (v >= 0, finite),
    clamped to [1, 254]."""
    bits = v.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    return fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))


def code_round(amax):
    """``_mxfp8_quant_op``'s code: amax rounded to a power of two, minus 2^8,
    clamped to +-127 unbiased."""
    p2 = (amax.bitcast(fx.Int32) + ROUND_BIAS) & fx.Int32(EXP_MASK)
    unbiased = ((p2 >> 23) & 255) - 127 - FP8_EXP
    return fx.max(fx.min(unbiased, 127), -127) + 127


def pow2(code):
    return (code << 23).bitcast(fx.Float32)


def rcp_pow2(code):
    """1 / ``pow2(code)``, exact for code in [1, 253]."""
    return ((254 - code) << 23).bitcast(fx.Float32)


def fp8_value(v):
    """``v`` (in range) rounded to E4M3 and back, as the hardware converts."""
    w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, v, v, fx.Int32(0), False))
    return fx.Float32(rocdl.cvt_f32_fp8(T.f32, w.ir_value(), 0))
