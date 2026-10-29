# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared reductions primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from kernels.monokernel.glm.layout import WAVES
from kernels.monokernel.ops import xred as _xred
from kernels.monokernel.ops import xshfl as _xshfl


class ReductionsTools(NamedTuple):
    wave_sum: Callable
    wave_max: Callable
    block_sums: Callable
    block_sums_f64: Callable
    block_maxs: Callable
    block_sum: Callable


@flyc.jit
def build_reductions(plan, state, memory):
    indexer_quant = plan.indexer_quant
    lane = state.lane
    red = state.red
    wave = state.wave
    lds_ld = memory.lds_ld
    lds_st = memory.lds_st

    def wave_sum(v):
        if const_expr(indexer_quant):
            # FP4 query/key rounding retains v163's descending XOR tree.
            # The current cooperative ADD uses ascending XOR offsets;
            # reassociation can change the BF16 quantization boundary.
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v
        return fx.coop.warp_reduce(v, fx.ReductionOp.ADD, width=64)

    def wave_max(v):
        return fx.coop.warp_reduce(v, fx.ReductionOp.MAX, width=64)

    def block_sums(vs):
        """Block-wide sums of several per-thread values with one LDS exchange."""
        ws = [wave_sum(v) for v in vs]
        if lane == 0:
            for i in range_constexpr(len(vs)):
                lds_st(red, i * WAVES + wave, ws[i])
        gpu.barrier()
        tots = []
        for i in range_constexpr(len(vs)):
            t = lds_ld(red, i * WAVES)
            for w in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i * WAVES + w)
            tots.append(t)
        gpu.barrier()
        return tots

    def block_sums_f64(vs):
        """Retain precision before the index query's BF16 rounding boundary."""
        ws = []
        for v in vs:
            for sh in range_constexpr(6):
                bits = v.bitcast(fx.Int64)
                lo = _xshfl(fx.Int32(bits), 32 >> sh)
                hi = _xshfl(fx.Int32(bits >> 32), 32 >> sh)
                other = (fx.Int64(fx.Uint32(lo)) | (fx.Int64(hi) << 32)).bitcast(fx.Float64)
                v = v + other
            ws.append(v)
        # Some row layouts only promise four-byte alignment for red.
        # Exchange each double as two words rather than recasting it.
        if lane == 0:
            for i in range_constexpr(len(vs)):
                bits = ws[i].bitcast(fx.Int64)
                index = 2 * (i * WAVES + wave)
                lds_st(red, index, fx.Int32(bits).bitcast(fx.Float32))
                lds_st(red, index + 1, fx.Int32(bits >> 32).bitcast(fx.Float32))
        gpu.barrier()

        def load_sum(index):
            lo = lds_ld(red, index * 2).bitcast(fx.Int32)
            hi = lds_ld(red, index * 2 + 1).bitcast(fx.Int32)
            return (fx.Int64(fx.Uint32(lo)) | (fx.Int64(hi) << 32)).bitcast(fx.Float64)

        tots = []
        for i in range_constexpr(len(vs)):
            t = load_sum(i * WAVES)
            for w in range_constexpr(1, WAVES):
                t = t + load_sum(i * WAVES + w)
            tots.append(t)
        gpu.barrier()
        return tots

    def block_maxs(vs):
        """Block-wide maxima of several per-thread values with one LDS exchange."""
        ws = [wave_max(v) for v in vs]
        if lane == 0:
            for i in range_constexpr(len(vs)):
                lds_st(red, i * WAVES + wave, ws[i])
        gpu.barrier()
        tots = []
        for i in range_constexpr(len(vs)):
            t = lds_ld(red, i * WAVES)
            for w in range_constexpr(1, WAVES):
                t = fx.max(t, lds_ld(red, i * WAVES + w))
            tots.append(t)
        gpu.barrier()
        return tots

    def block_sum(v):
        w = wave_sum(v)
        if lane == 0:
            lds_st(red, wave, w)
        gpu.barrier()
        t = lds_ld(red, 0)
        for i in range_constexpr(1, WAVES):
            t = t + lds_ld(red, i)
        gpu.barrier()
        return t

    return ReductionsTools(
        wave_sum=wave_sum,
        wave_max=wave_max,
        block_sums=block_sums,
        block_sums_f64=block_sums_f64,
        block_maxs=block_maxs,
        block_sum=block_sum,
    )
