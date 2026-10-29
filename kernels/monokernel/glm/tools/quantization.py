# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared quantization primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.monokernel.config import FP8_MAX, HIDDEN
from kernels.monokernel.glm.layout import XQ_BLOCKS, XQ_GROUPS
from kernels.monokernel.layout import THREADS
from kernels.monokernel.ops import f8_word
from kernels.monokernel.ops import fp8_roundtrip as _fp8_roundtrip
from kernels.monokernel.ops import rcp as _rcp
from kernels.monokernel.ops import xred as _xred
from kernels.monokernel.ops import xshfl as _xshfl


class QuantizationTools(NamedTuple):
    quant_moe: Callable
    quant_index_pair: Callable
    quant_scaled: Callable
    quant_mxfp8: Callable
    pack_moe4: Callable
    publish_moe_quant: Callable
    moe_roundtrip: Callable
    stage_xq: Callable
    st_f8: Callable
    st_mxfp8: Callable
    st_ptpc: Callable


@flyc.jit
def build_quantization(plan, state, memory, reductions):
    UG_SCALE_ROW = plan.UG_SCALE_ROW
    UG_X_ROW = plan.UG_X_ROW
    native_fp4_mfma = plan.native_fp4_mfma
    activation_fp4 = plan.activation_fp4
    native_fp4_mfma = plan.native_fp4_mfma
    lane = state.lane
    misc = state.misc
    tid = state.tid
    xs = state.xs
    lds_st = memory.lds_st
    bf16_round = memory.bf16_round
    mb = memory.mb
    put = memory.put
    poll = memory.poll
    wave_max = reductions.wave_max

    def quant_index_pair(v0, v1):
        """One lane's BF16 pair; 16 neighboring lanes share one E8M0 scale."""
        v0, v1 = bf16_round(v0), bf16_round(v1)
        amax = fx.max(fmath.absf(v0), fmath.absf(v1))
        for delta in (1, 2, 4, 8):
            amax = _xred(amax, delta, fx.max)
        bits = (fx.max(amax, fx.Float32(1e-10)) * (1.0 / 6.0)).bitcast(fx.Int32)
        exponent = (bits >> 23) & 255
        exponent = exponent + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
        scale4 = (exponent << 23).bitcast(fx.Float32)
        inverse = _rcp(scale4)
        q0, q1 = v0 * inverse, v1 * inverse
        pair4 = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, fx.Int32(0), q0, q1, fx.Float32(1.0), 0)) & 255
        word4 = pair4 | (_xshfl(pair4, 1) << 8) | (_xshfl(pair4, 2) << 16) | (_xshfl(_xshfl(pair4, 1), 2) << 24)
        return pair4, word4, exponent

    def quant_scaled(a0, a1):
        """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
        amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
        nz = amax > 0.0
        qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
        inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))  # hardware rcp, no IEEE divide
        q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
        q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
        return q0, q1, qs

    def quant_mxfp8(a0, a1):
        """Per-32 E4M3 values and E8M0 scale used by ATOM/AITER W4A8."""
        if const_expr(activation_fp4):
            # Match the BF16 input boundary of the native A4 quantizer.
            a0, a1 = bf16_round(a0), bf16_round(a1)
        amax = fx.max(fmath.absf(a0), fmath.absf(a1))
        if const_expr(activation_fp4):
            amax = fx.max(amax, fx.Float32(1e-10))
        for offset in (1, 2, 4, 8):
            amax = _xred(amax, offset, fx.max)
        bits = (amax * fx.Float32(1.0 / (6.0 if activation_fp4 else FP8_MAX))).bitcast(fx.Int32)
        exponent = (bits >> 23) & fx.Int32(0xFF)
        exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(fx.Int32(1), fx.Int32(0))
        exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
        inverse = _rcp((exponent << 23).bitcast(fx.Float32))
        mx = 6.0 if activation_fp4 else FP8_MAX
        q0 = fx.min(fx.max(a0 * inverse, -mx), mx)
        q1 = fx.min(fx.max(a1 * inverse, -mx), mx)
        return q0, q1, exponent

    def quant_moe(a0, a1):
        """Common MoE quantization entry; retain each format's scale and rounding."""
        if const_expr(native_fp4_mfma):
            return quant_mxfp8(a0, a1)
        return quant_scaled(a0, a1)

    def pack_moe4(q0, q1):
        pair4 = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, fx.Int32(0), q0, q1, fx.Float32(1.0), 0)) & fx.Int32(255)
        word4 = pair4 | (_xshfl(pair4, 1) << 8) | (_xshfl(pair4, 2) << 16) | (_xshfl(_xshfl(pair4, 1), 2) << 24)
        return word4

    def publish_moe_quant(ptr, k, q0, q1):
        if const_expr(activation_fp4):
            word4 = pack_moe4(q0, q1)
            if lane % 4 == 0:
                address4 = (k // 128) * 32 + (k % 128) // 8
                put(ptr, address4, word4)
                put(ptr, address4 + 16, fx.Int32(0))
        else:
            word8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
            neighbor8 = _xshfl(word8, 1)
            if lane % 2 == 0:
                put(ptr, k // 4, word8 | (neighbor8 << 16))

    def moe_roundtrip(q0, q1):
        if const_expr(activation_fp4):
            p4 = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, fx.Int32(0), q0, q1, fx.Float32(1.0), 0))
            out4 = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), p4, fx.Float32(1.0), 0)).to(fx.Float32)
            return out4[0], out4[1]
        return _fp8_roundtrip(q0, q1)

    def stage_xq(samples):
        """Poll the router's packed FP8 activation + block scales of ``samples``
        (sample list, or one runtime sample) into LDS, slot 0 for one runtime sample.
        """
        nxw = HIDDEN // 4 // THREADS
        nscale = XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS
        got = poll(
            [(mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
            + [(mb("xqs"), sx * nscale + fx.min(tid, nscale - 1), 1) for sx in samples]
        )
        for j in range_constexpr(len(samples)):
            for i in range_constexpr(nxw):
                wd = tid + i * THREADS if const_expr(native_fp4_mfma) else f8_word((tid + i * THREADS) * 4)
                lds_st(
                    xs,
                    j * UG_X_ROW + wd,
                    got[j * nxw + i][0].bitcast(fx.Float32),
                )
            if tid < nscale:
                lds_st(
                    misc,
                    8 + j * (UG_SCALE_ROW if native_fp4_mfma else nscale) + tid,
                    got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                )

    def st_f8(k, q0, q1):
        """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
        k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
        w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
        nb = _xshfl(w, 1)
        if lane % 2 == 0:
            lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

    def st_mxfp8(k, q0, q1):
        if const_expr(activation_fp4):
            packed4 = pack_moe4(q0, q1)
            if lane % 4 == 0:
                address4 = (k // 128) * 32 + (k % 128) // 8
                lds_st(xs, address4, packed4.bitcast(fx.Float32))
                lds_st(xs, address4 + 16, fx.Float32(0.0))
        else:
            """Store one row-major MXFP8 pair for the scaled MFMA."""
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

    def st_ptpc(k, q0, q1):
        """Store one wave's PTPC FP8 pairs in the AITER GEMM's linear K order."""
        word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
        neighbor = _xshfl(word, 1)
        if lane % 2 == 0:
            lds_st(xs, k // 4, (word | (neighbor << 16)).bitcast(fx.Float32))

    return QuantizationTools(
        quant_moe=quant_moe,
        quant_index_pair=quant_index_pair,
        quant_scaled=quant_scaled,
        quant_mxfp8=quant_mxfp8,
        pack_moe4=pack_moe4,
        publish_moe_quant=publish_moe_quant,
        moe_roundtrip=moe_roundtrip,
        stage_xq=stage_xq,
        st_f8=st_f8,
        st_mxfp8=st_mxfp8,
        st_ptpc=st_ptpc,
    )
