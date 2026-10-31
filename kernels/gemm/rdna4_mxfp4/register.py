# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register-fed MXFP4 GEMM expressed with FlyDSL tiled MMA and copies.

The MMA atom defines thread ownership and register fragments. A native copy
partitions A, compressed B is a small format adapter, and the epilogue copies
C through the same tiled layout. No lane-specific WMMA ABI is encoded here.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .folding import magnitude_words
from .layouts import (
    expand_nibbles,
    global_tensor,
    identity_tensor,
    tensor_load,
    tiled_wmma,
)


@lru_cache(maxsize=256)
def build_register(
    m,
    n,
    k,
    *,
    waves_m=1,
    waves_n=2,
    repeat_m=1,
    repeat_n=2,
    fold_mode="grouped",
    table_mode="alu",
    normal_only=False,
    separated_nibbles=False,
    k_unroll=1,
    output_dtype="bf16",
    buffer_io=True,
    weight_bits=4,
    fast_epilogue=True,
    device_id=0,
):
    assert fold_mode in ("grouped", "e4m3", "e5m2")
    assert isinstance(k_unroll, int) and 1 <= k_unroll <= 16
    assert weight_bits in (4, 8) and (weight_bits == 4 or fold_mode != "grouped")
    assert k % 32 == 0 and m > 0 and n > 0
    folded = fold_mode != "grouped"
    out_type = fx.BFloat16 if output_dtype == "bf16" else fx.Float16
    b_dtype = fx.Float8E5M2 if fold_mode == "e5m2" else fx.Float8E4M3FN
    bm, bn = waves_m * repeat_m * 16, waves_n * repeat_n * 16
    threads = waves_m * waves_n * 32
    npad = (n + 15) // 16 * 16
    nt, groups = npad // 16, k // 32
    grid_m, grid_n = (m + bm - 1) // bm, (n + bn - 1) // bn
    k_unroll = max(d for d in range(1, k_unroll + 1) if groups % d == 0)
    word_width = weight_bits // 4
    table_bias = 15 if fold_mode == "e5m2" else 8

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def mxfp4_register(
        A: fx.Tensor,
        W: fx.Tensor,
        S: fx.Tensor,
        XScale: fx.Tensor,
        Y: fx.Tensor,
        Ref: fx.Tensor,
        LUT: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        pid = fx.Int32(fx.block_idx.x)
        pm, pn = pid // grid_n, pid % grid_n
        tiled_mma = tiled_wmma(waves_m, waves_n, repeat_m, repeat_n, b_dtype)
        thr_mma = tiled_mma.thr_slice(tid)
        gA = global_tensor(A, fx.Float8E4M3FN, fx.make_layout((m, k), (k, 1)), m * k, buffer_io)
        gC = global_tensor(Y, out_type, fx.make_layout((m, n), (n, 1)), m * n * 2, buffer_io)
        gW = global_tensor(
            W,
            fx.Int32,
            fx.make_layout(
                (nt, k // 16, 2, 16, word_width),
                (
                    (k // 16) * 32 * word_width,
                    32 * word_width,
                    16 * word_width,
                    word_width,
                    1,
                ),
            ),
            npad * k * weight_bits // 8,
            buffer_io,
        )
        gS = global_tensor(
            S,
            fx.Uint8,
            fx.make_layout((nt, groups, 16), (groups * 16, 16, 1)),
            npad * groups,
            buffer_io,
        )
        gRef = global_tensor(Ref, fx.Int32, fx.make_layout(npad, 1), npad * 4, buffer_io)
        gLUT = global_tensor(
            LUT,
            fx.Uint32,
            fx.make_layout((29 if fold_mode == "e5m2" else 15) * 2, 1),
            (29 if fold_mode == "e5m2" else 15) * 8,
            buffer_io,
        )
        gX = global_tensor(
            XScale,
            fx.Float32,
            fx.make_layout((m, n), (0 if m == 1 else 1, 0)),
            m * 4,
            buffer_io,
        )
        tA = fx.flat_divide(gA, fx.make_tile(bm, 16))[None, None, pm, None]
        tC = fx.flat_divide(gC, fx.make_tile(bm, bn))[None, None, pm, pn]
        tX = fx.flat_divide(gX, fx.make_tile(bm, bn))[None, None, pm, pn]
        cB = thr_mma.partition_B(identity_tensor((bn, 16)))
        copy_op = fx.rocdl.BufferCopy if const_expr(buffer_io) else fx.UniversalCopy
        copy_a = fx.make_copy_atom(copy_op(64), fx.Float8E4M3FN)
        thr_copy_a = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        pA = thr_copy_a.partition_S(tA)
        frag_A = thr_mma.make_fragment_A(tA[None, None, 0])
        # Shape-only logical B view: its compressed storage is supplied by the
        # format adapter below, while the MMA atom owns the fragment layout.
        logical_B = fx.make_view(fx.recast_iter(b_dtype, fx.get_iter(W)), fx.make_layout((bn, 16), (16, 1)))
        frag_B = thr_mma.make_fragment_B(logical_B)
        frag_C = thr_mma.make_fragment_C(tC)
        copy_frag_A = thr_copy_a.retile(frag_A)
        refs = []
        if const_expr(folded):
            for j in range_constexpr(repeat_n):
                nc, _ = cB[0, j, 0].unpack()
                refs.append(tensor_load(gRef, (pn * bn + fx.Int32(nc),), fx.Int32, buffer_io=buffer_io))
        frag_C.fill(0)
        for gbase, carried in range(0, groups, k_unroll, init=[frag_C.load()]):
            frag_C.store(carried[0])
            for ku in range_constexpr(k_unroll):
                group = fx.Int32(gbase) + ku
                if const_expr(not folded):
                    total = Vec(frag_C.load())
                    frag_C.fill(0)
                tables = []
                if const_expr(folded and weight_bits == 4):
                    for j in range_constexpr(repeat_n):
                        nc, _ = cB[0, j, 0].unpack()
                        nc = pn * bn + fx.Int32(nc)
                        exponent = fx.Int32(
                            tensor_load(
                                gS,
                                (nc // 16, group, nc % 16),
                                fx.Uint8,
                                buffer_io=buffer_io,
                            )
                        )
                        if const_expr(table_mode == "alu"):
                            t0, t1 = magnitude_words(exponent - refs[j], fold_mode, normal_only)
                            tables.append(Vec.from_elements([t0, t1], fx.Uint32))
                        else:
                            ti = (exponent - refs[j] + table_bias) * 2
                            tables.append(tensor_load(gLUT, (ti,), fx.Uint32, 2, buffer_io))
                for step in range_constexpr(2):
                    fx.copy(copy_a, pA[None, None, None, group * 2 + step], copy_frag_A)
                    for j in range_constexpr(repeat_n):
                        nc, kc = cB[0, j, 0].unpack()
                        nc, kc = pn * bn + fx.Int32(nc), fx.Int32(kc)
                        raw = tensor_load(
                            gW,
                            (nc // 16, group * 2 + step, kc // 8, nc % 16, 0),
                            fx.Int32,
                            word_width,
                            buffer_io,
                        )
                        if const_expr(weight_bits == 8):
                            decoded = Vec(raw)
                        elif const_expr(folded):
                            decoded = expand_nibbles(
                                raw,
                                fx.Uint32(tables[j][0]),
                                fx.Uint32(tables[j][1]),
                                separated_nibbles,
                            )
                        else:
                            decoded = expand_nibbles(
                                raw,
                                fx.Uint32(0x3C383000),
                                fx.Uint32(0x4C484440),
                                separated_nibbles,
                            )
                        frag_B[None, j, 0].store(decoded.bitcast(b_dtype))
                    fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
                if const_expr(not folded):
                    scales = fx.make_fragment_like(frag_C)
                    for j in range_constexpr(repeat_n):
                        nc, _ = cB[0, j, 0].unpack()
                        nc = pn * bn + fx.Int32(nc)
                        exponent = fx.Uint32(
                            tensor_load(
                                gS,
                                (nc // 16, group, nc % 16),
                                fx.Uint8,
                                buffer_io=buffer_io,
                            )
                        )
                        sf = (exponent << 23).bitcast(fx.Float32)
                        sf = fx.Float32(fx.arith.select(exponent == 0, fx.Float32(2.0**-127), sf))
                        sf = fx.Float32(
                            fx.arith.select(
                                exponent == 255,
                                fx.Uint32(0x7FC00000).bitcast(fx.Float32),
                                sf,
                            )
                        )
                        for i in range_constexpr(repeat_m):
                            scales[None, i, j].fill(sf)
                    frag_C.store(total + Vec(frag_C.load()) * Vec(scales.load()))
            result = yield [frag_C.load()]
        frag_C.store(result)
        if const_expr(folded):
            scales = fx.make_fragment_like(frag_C)
            for j in range_constexpr(repeat_n):
                for i in range_constexpr(repeat_m):
                    scales[None, i, j].fill((refs[j] << 23).bitcast(fx.Float32))
            frag_C.store(Vec(frag_C.load()) * Vec(scales.load()))
        copy_x = fx.make_copy_atom(copy_op(32), fx.Float32)
        thr_copy_x = fx.make_tiled_copy_C(copy_x, tiled_mma).get_slice(tid)
        frag_X = fx.make_fragment_like(frag_C)
        fx.copy(copy_x, thr_copy_x.partition_S(tX), thr_copy_x.retile(frag_X))
        values = Vec(frag_C.load()) * Vec(frag_X.load())
        frag_out = fx.make_fragment_like(frag_C, out_type)
        frag_out.store(values.to(out_type))
        copy_out = fx.make_copy_atom(copy_op(out_type.width), out_type)
        thr_copy_c = fx.make_tiled_copy_C(copy_out, tiled_mma).get_slice(tid)
        pC = thr_copy_c.partition_S(tC)
        if const_expr(fast_epilogue and m % bm == 0 and n % bn == 0):
            fx.copy(copy_out, thr_copy_c.retile(frag_out), pC)
        else:
            coords = fx.flat_divide(
                fx.make_view((0, 0), fx.make_identity_layout((m, n))),
                fx.make_tile(bm, bn),
            )[None, None, pm, pn]
            pcoords = thr_copy_c.partition_S(coords)
            pred = fx.make_fragment_like(pC, fx.Boolean)
            for q in range_constexpr(fx.size(pred.shape).unpack()):
                row, column = pcoords[q].unpack()
                if const_expr(fast_epilogue and n % bn == 0):
                    pred[q] = fx.Int32(row) < m
                elif const_expr(fast_epilogue and m % bm == 0):
                    pred[q] = fx.Int32(column) < n
                else:
                    pred[q] = fx.elem_less(pcoords[q], (m, n))
            fx.copy(copy_out, thr_copy_c.retile(frag_out), pC, pred=pred)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        S: fx.Tensor,
        XScale: fx.Tensor,
        Y: fx.Tensor,
        Ref: fx.Tensor,
        LUT: fx.Tensor,
        stream: fx.Stream,
    ):
        mxfp4_register(A, W, S, XScale, Y, Ref, LUT).launch(
            grid=(grid_m * grid_n + 0 * device_id, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch
