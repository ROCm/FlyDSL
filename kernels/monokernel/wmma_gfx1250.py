# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The gfx1250 (wave32) WMMA instructions the mono kernels issue, and their
operand / accumulator lane layouts.

16 x 16 x K, 8-bit operands (the layout of aiter's hardware-validated gfx1250
GEMMs and of ``aiter.ops.shuffle.shuffle_weight_gfx1250``): lane l holds row
``l % 16`` (an A row, or a B column) and half ``h = l // 16`` of each 32-byte K
block -- bytes [32 j + 16 h, +16) of a 128-K step for j = 0..3, the operand's
dwords 4 j .. 4 j + 3. A 4-bit operand halves the bytes: [16 j + 8 h, +8) for
j = 0..3.

Accumulator (16 x 16 f32, 8 VGPRs): lane l holds column ``l % 16`` and rows
``8 (l // 16) + i``, i = 0..7.

MX scales (E8M0, block 32): one dword a lane holds the four codes of one 128-K
step, byte g for K group g; for a 16-row operand lanes 0-15 supply rows 0-15
(``opsel`` 0) or lanes 16-31 do (``opsel`` 1), so two 16-row tiles can share a
dword (aiter's ``n32k4`` scale layout).

The mono kernels put the weight in A (16 output rows) and a tile of 16 tokens
in B, so a lane's accumulator is one token's 8 consecutive output rows.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import rocdl
from flydsl.expr.typing import T

# the f8f6f4 operand format codes
FP8, BF8, FP6, BF6, FP4 = 0, 1, 2, 3, 4
STEP_K = 128  # K a scaled WMMA consumes
ACC = 8  # f32 accumulator registers a lane


def acc_zero():
    return fx.Vector.filled(ACC, 0.0, fx.Float32)


def fp8_chunk_bytes(h, j):
    """Byte offset within a 128-K step of 8-bit operand dwords 4 j .. 4 j + 3 of a
    lane in half ``h``."""
    return 32 * j + 16 * h


def fp4_chunk_bytes(h, j):
    return 16 * j + 8 * h


def acc_row(lane, i):
    """The output row (within the 16) of accumulator register i of ``lane``."""
    return 8 * (lane // 16) + i


def acc_col(lane):
    return lane % 16


def wmma_scaled(a, b, c, sa, sb, fmt_a=FP8, fmt_b=FP8, opsel_a=0, opsel_b=0):
    """V_WMMA_SCALE_F32_16X16X128_F8F6F4: c + (A sa) (B sb)^T over one 128-K step.
    ``a`` / ``b``: v16i32 (8-bit) or v8i32 (4-bit); ``sa`` / ``sb``: E8M0 dwords."""
    return fx.Vector(
        rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            T.vec(ACC, T.f32),
            a,
            b,
            c,
            sa,
            sb,
            fmtA=fmt_a,
            fmtB=fmt_b,
            scaleAType=opsel_a,
            scaleBType=opsel_b,
        )
    )


def wmma_f4_32x16(a, b, c, sa, sb, opsel_a=0, opsel_b=0):
    """V_WMMA_SCALE_F32_32X16X128_F4 (MI450-B0 only): a v16i32 (32 rows of FP4),
    b v8i32, c v16f32."""
    return fx.Vector(
        rocdl.wmma_scale_f32_32x16x128_f4(
            T.vec(16, T.f32), a, b, c, sa, sb, scaleAType=opsel_a, scaleBType=opsel_b
        )
    )


def wmma_bf16(a, b, c):
    """V_WMMA_F32_16X16X32_BF16: a, b v16bf16; c v8f32."""
    return fx.Vector(_rocdl_ods.wmma_f32_16x16x32_bf16_(T.vec(ACC, T.f32), a, b, c))


def wmma_f32(a, b, c):
    """V_WMMA_F32_16X16X4_F32: a, b v2f32; c v8f32."""
    return fx.Vector(_rocdl_ods.wmma_f32_16x16x4_f32_(T.vec(ACC, T.f32), a, b, c))


def red_index(wave, lane, i):
    """Word of accumulator register i of (wave, lane) in a CTA's ``red`` buffer
    (WAVES x 32 lanes x 8)."""
    return (wave * 32 + lane) * ACC + i


def row_sum(red, r, t, waves):
    """Output row r, token t of the waves' accumulators in ``red``, summed in the
    given wave order."""
    lane = 16 * (r // 8) + t
    tot = fx.Float32(0.0)
    for w in waves:
        tot = tot + fx.ptr_load(red + red_index(w, lane, r % 8))
    return tot
