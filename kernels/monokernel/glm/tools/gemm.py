# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared gemm primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import SCALE_BM
from kernels.monokernel.glm.layout import WAVES
from kernels.monokernel.layout import THREADS, atom_mxfp4_scale_index
from kernels.monokernel.ops import fp8_to_bf16x8 as _fp8_to_bf16x8
from kernels.monokernel.ops import mxfp4_to_bf16x8 as _mxfp4_to_bf16x8


class GemmTools(NamedTuple):
    unit_fp8: Callable
    unit_fp8x2: Callable
    unit_f8f8: Callable
    unit_ptpc: Callable
    unit_ptpc64: Callable
    unit_mxfp4: Callable
    unit_mxfp4_bf16: Callable
    unit_native_mxfp4: Callable
    lds_mxfp8: Callable
    unit_bf16: Callable
    unit_attention: Callable
    mma_units: Callable
    run_units: Callable
    reduce_rows: Callable
    emit_out: Callable
    attention_b_word: Callable
    emit_ptpc: Callable
    emit_ptpc_scalar: Callable


@flyc.jit
def build_gemm(plan, state, memory):
    S = plan.S
    activation_fp4 = plan.activation_fp4
    atom_experts = plan.atom_experts
    attention_bf16 = plan.attention_bf16
    attention_ptpc = plan.attention_ptpc
    lane = state.lane
    misc = state.misc
    outs = state.outs
    red = state.red
    tid = state.tid
    v4f = state.v4f
    wave = state.wave
    xs = state.xs
    ld_f32 = memory.ld_f32
    lds_ld = memory.lds_ld
    lds_st = memory.lds_st
    bf16_round = memory.bf16_round
    pick = memory.pick

    def packed_fp8_weights(w_rsrc, rg, kc, NKC, ln):
        """Issue the shared two-half 128-K packed FP8 tile."""
        return [
            fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            for h in range(2)
        ]

    def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None):
        """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
        bf16 activation chunk starts at LDS word ``b_word``."""
        wv = fx.Vector(
            bo.buffer_load(
                w_rsrc,
                ((rg * NKC + kc) * 64 + lane) * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
        if const_expr(callable(coef)):  # factor known only after a later wait
            return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
        if const_expr(coef is not None):
            s = s * coef
        return ("fp8", [wv], s, b_word + (lane // 16) * 4)

    def unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, scale_rows=SCALE_BM):
        """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
        wv = packed_fp8_weights(w_rsrc, rg, kc, NKC, lane)
        s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
        if const_expr(callable(coef)):
            return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
        if const_expr(coef is not None):
            s = s * coef
        return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

    def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
        """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
        ``rg`` against the FP8 activation of LDS words ``b_word`` + [0, 32) (``f8_word``
        order); ``coef()`` = activation block scale (times route weight).  ``ln``
        = the lane whose weights are loaded (default: own lane)."""
        ln = lane if ln is None else ln
        wv = packed_fp8_weights(w_rsrc, rg, kc, NKC, ln)
        s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
        return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

    def unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
        """Issue one packed 128-K PTPC FP8 tile; output-channel scale is applied after reduction."""
        ln = lane if ln is None else ln
        wv = packed_fp8_weights(w_rsrc, rg, kc, NKC, ln)
        return ("ptpc", wv, coef, b_word + (lane // 16) * 4)

    def unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
        """Issue one packed 64-K PTPC FP8 tile."""
        ln = lane if ln is None else ln
        weight = fx.Vector(
            bo.buffer_load(
                w_rsrc,
                ((rg * NKC + kc) * 64 + ln) * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        return ("ptpc64", [weight], coef, b_word + (lane // 16) * 4)

    def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
        """Issue one native packed 128-K MXFP4 tile and four E8M0 row scales."""
        ln = lane if ln is None else ln
        row = rg * 16 + ln % 16
        if const_expr(atom_experts):
            g4 = ln // 16
            raw = fx.Vector.from_elements(
                [
                    fx.Int32(
                        bo.buffer_load(
                            w_rsrc,
                            (rg * (K // 64) + kc * 2 + sp // 2) * 128 + (sp % 2) * 64 + (row % 16) * 4 + g4,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    for sp in range_constexpr(4)
                ],
                fx.Int32,
            )
            scales = [
                (
                    fx.Int32(
                        bo.buffer_load(
                            s_rsrc,
                            atom_mxfp4_scale_index(row, kc * 4 + sp, K // 32),
                            vec_width=1,
                            dtype=T.i8,
                        )
                    )
                    << fx.Int32(23)
                ).bitcast(fx.Float32)
                for sp in range_constexpr(4)
            ]
        else:
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((rg * (K // 128) + kc) * 64 + ln) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            packed_scale = fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    (rg * (K // 128) + kc) * 16 + ln % 16,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            scales = [
                ((packed_scale.shrui(fx.Int32(sp * 8)) & 0xFF) << 23).bitcast(fx.Float32) for sp in range_constexpr(4)
            ]
        return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

    def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
        fmt, weights, factor, _ = unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln)
        return ("mxfp4_bf16", weights, factor, b_word + (lane // 16) * 4)

    def unit_native_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, b_scale_word, coef=None, ln=None):
        """Issue ATOM's 16-byte/lane W4 tile directly on the scaled MFMA."""
        ln = lane if ln is None else ln
        raw = fx.Vector(
            bo.buffer_load(
                w_rsrc,
                (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        scale = fx.Int32(
            bo.buffer_load(
                s_rsrc,
                atom_mxfp4_scale_index(rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32),
                vec_width=1,
                dtype=T.i8,
            )
        )
        return ("native_mxfp4", (raw, scale, b_scale_word), coef, b_word)

    def lds_mxfp8(chunk_word):
        """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
        if const_expr(activation_fp4):
            # FP4 uses contiguous K[32*g:32*g+32] for lane group g.
            return fx.ptr_load(xs + chunk_word + 4 * (lane // 16), result_type=v4f).bitcast(fx.Int32)
        words = [
            fx.ptr_load(
                xs + chunk_word + 16 * half + 4 * (lane // 16),
                result_type=v4f,
            ).bitcast(fx.Int32)
            for half in range(2)
        ]
        return fx.Vector.from_elements([words[half][e] for half in range(2) for e in range(4)], fx.Int32)

    def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
        ln = lane if ln is None else ln
        wv = [
            fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for sp in range(2)
        ]
        return ("bf16", wv, None, b_word + (lane // 16) * 4)

    def unit_attention(
        w_rsrc,
        s_rsrc,
        rg,
        kc,
        NKC,
        K,
        BK,
        b_word,
        ln=None,
        scale_rows=SCALE_BM,
        coef=None,
    ):
        if const_expr(attention_bf16):
            return unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln)
        if const_expr(attention_ptpc):
            if const_expr(BK == 64):
                return unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
            return unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
        if const_expr(BK == 64):
            return unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word)
        return unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, scale_rows=scale_rows)

    def mma_units(acc, units):
        """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
        for fmt, wv, coef, bw in units:
            if const_expr(callable(coef)):
                coef = coef()
            c = fx.full(4, 0.0, fx.Float32)
            if const_expr(fmt in ("mxfp4", "mxfp4_bf16")):
                raw, scales = wv
                for sp in range_constexpr(4):
                    a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                    if const_expr(fmt == "mxfp4"):
                        wh, ws = sp // 2, sp % 2
                        bv = fx.ptr_load(xs + (bw + wh * 16), result_type=v4f).bitcast(fx.Int32)
                        b = _fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                    else:
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(fmt == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                bv = [fx.ptr_load(xs + (bw + h * 16), result_type=v4f).bitcast(fx.Int32) for h in range(2)]
                b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                one = fx.Int32(127)
                c = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))
            if const_expr(fmt == "native_mxfp4"):
                raw, weight_scale, activation_scale_word = wv
                activation_scale = lds_ld(misc, activation_scale_word).bitcast(fx.Int32)
                c = fx.Vector(
                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32),
                        [
                            raw,
                            lds_mxfp8(bw),
                            c,
                            4,
                            4 if activation_fp4 else 0,
                            0,
                            weight_scale,
                            0,
                            activation_scale,
                        ],
                    )
                )
            if const_expr(fmt in ("ptpc", "ptpc64")):
                halves64 = 2 if fmt == "ptpc" else 1
                for half64 in range_constexpr(halves64):
                    a = wv[half64].bitcast(fx.Int64)
                    b = fx.ptr_load(xs + (bw + half64 * 16), result_type=v4f).bitcast(fx.Int64)
                    for half32 in range_constexpr(2):
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.vec(4, T.f32),
                                [a[half32], b[half32], c, 0, 0, 0],
                            )
                        )
            nsp = (
                4
                if fmt == "fp8x2"
                else (
                    2
                    if fmt
                    not in (
                        "f8f8",
                        "ptpc",
                        "ptpc64",
                        "mxfp4",
                        "mxfp4_bf16",
                        "native_mxfp4",
                    )
                    else 0
                )
            )
            for sp in range_constexpr(nsp):
                if const_expr(fmt in ("fp8", "fp8x2")):
                    wh = sp // 2 if fmt == "fp8x2" else 0
                    ws = sp % 2 if fmt == "fp8x2" else sp
                    a = _fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                else:
                    a = wv[sp].bitcast(fx.BFloat16)
                b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(coef is None):
                acc = [acc[e] + c[e] for e in range(4)]
            else:
                acc = [acc[e] + c[e] * coef for e in range(4)]
        return acc

    def run_units(make_unit, cpw, batch, pre=None):
        """Software pipelined: issue batch b+1's loads before computing batch b.
        ``pre`` = the already-issued first batch (prefetched before a wait)."""
        acc = [fx.Float32(0.0) for _ in range(4)]
        starts = list(range(0, cpw, batch))
        cur = pre if pre is not None else [make_unit(c) for c in range(0, min(batch, cpw))]
        for bi in range_constexpr(len(starts)):
            nxt = None
            if const_expr(bi + 1 < len(starts)):
                n0 = starts[bi + 1]
                nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
            acc = mma_units(acc, cur)
            cur = nxt
        return acc

    def reduce_rows(R, acc, emit, count=S):
        """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
        wpr = WAVES // R
        fx.ptr_store(
            fx.Vector.from_elements(acc, fx.Float32),
            red + (wave * 64 + lane) * 4,
        )
        gpu.barrier()
        n_out = R * 16 * count
        for i in range_constexpr((n_out + THREADS - 1) // THREADS):
            t = tid + i * THREADS
            if t < n_out:
                rl = t % (R * 16)
                n = t // (R * 16)
                r = rl % 16
                tot = fx.Float32(0.0)
                for j in range_constexpr(wpr):
                    ww = (rl // 16) * wpr + j
                    tot = tot + lds_ld(red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4)
                emit(rl, n, tot)

    def emit_out(stride):
        def f(rl, n, v):
            lds_st(outs, n * stride + rl, v)

        return f

    def attention_b_word(element):
        return element // 4 if const_expr(attention_ptpc) else element // 2

    def emit_ptpc(stride, row_base, scale_rsrc, x_scales):
        def f(rl, n, v):
            lds_st(
                outs,
                n * stride + rl,
                bf16_round(v * pick(x_scales, n) * ld_f32(scale_rsrc, row_base + rl)),
            )

        return f

    def emit_ptpc_scalar(stride, scale):
        def f(rl, n, v):
            lds_st(outs, n * stride + rl, bf16_round(v * scale))

        return f

    return GemmTools(
        unit_fp8=unit_fp8,
        unit_fp8x2=unit_fp8x2,
        unit_f8f8=unit_f8f8,
        unit_ptpc=unit_ptpc,
        unit_ptpc64=unit_ptpc64,
        unit_mxfp4=unit_mxfp4,
        unit_mxfp4_bf16=unit_mxfp4_bf16,
        unit_native_mxfp4=unit_native_mxfp4,
        lds_mxfp8=lds_mxfp8,
        unit_bf16=unit_bf16,
        unit_attention=unit_attention,
        mma_units=mma_units,
        run_units=run_units,
        reduce_rows=reduce_rows,
        emit_out=emit_out,
        attention_b_word=attention_b_word,
        emit_ptpc=emit_ptpc,
        emit_ptpc_scalar=emit_ptpc_scalar,
    )
