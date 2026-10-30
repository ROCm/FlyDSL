# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared normalization primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import EPS, FP8_MAX
from kernels.monokernel.layout import THREADS
from kernels.monokernel.ops import div_rn, fp8_pack4
from kernels.monokernel.ops import rsq as _rsq
from kernels.monokernel.ops import rsrc as _rsrc


class NormalizationTools(NamedTuple):
    stage_x_rmsnorm_ptpc: Callable
    stage_x_pairs_ptpc: Callable
    stage_x_rmsnorm: Callable
    load_x_rmsnorm: Callable
    stage_x_pairs: Callable


@flyc.jit
def build_normalization(plan, execution, memory, reductions):
    INDEX_BF16_OFF = plan.INDEX_BF16_OFF
    INDEX_Q_BF16_OFF = plan.INDEX_Q_BF16_OFF
    S = plan.S
    with_indexer = plan.with_indexer
    tid = execution.thread.thread_id
    xs = execution.workspace.activations
    lds_st = memory.lds_st
    bf16_pair = memory.bf16_pair
    mb = memory.mb
    bf2_f32 = memory.bf2_f32
    poll = memory.poll
    block_sums = reductions.block_sums
    block_sums_f64 = reductions.block_sums_f64
    block_maxs = reductions.block_maxs

    def stage_x_rmsnorm_ptpc(ld4s, n, gamma, loaded=None, count=S):
        """RMSNorm followed by production per-token FP8 quantization."""
        per = n // (4 * THREADS)
        ks = [(tid + i * THREADS) * 4 for i in range(per)]
        gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
        sums = []
        for s in range_constexpr(count):
            ss = fx.Float32(0.0)
            for i in range_constexpr(per):
                for value in vals[s * per + i]:
                    ss = ss + value * value
            sums.append(ss)
        totals = block_sums(sums)
        normed = []
        for s in range_constexpr(count):
            rstd = _rsq(totals[s] * (1.0 / n) + EPS)
            for i in range_constexpr(per):
                normed.append([vals[s * per + i][j] * rstd * gs[i][j] for j in range_constexpr(4)])
        local_max = []
        for s in range_constexpr(count):
            amax = fx.Float32(0.0)
            for i in range_constexpr(per):
                for value in normed[s * per + i]:
                    amax = fx.max(amax, fmath.absf(value))
            local_max.append(amax)
        maxima = block_maxs(local_max)
        scales = []
        for s in range_constexpr(count):
            scale = (maxima[s] == 0.0).select(fx.Float32(1.0), maxima[s] * (1.0 / FP8_MAX))
            scales.append(scale)
            reciprocal = 1.0 / scale
            for i in range_constexpr(per):
                if const_expr(with_indexer):
                    for pair in range_constexpr(2):
                        lds_st(
                            xs,
                            INDEX_BF16_OFF + (s * n + ks[i]) // 2 + pair,
                            bf16_pair(
                                normed[s * per + i][pair * 2],
                                normed[s * per + i][pair * 2 + 1],
                            ),
                        )
                values = [div_rn(value, scale, reciprocal) for value in normed[s * per + i]]
                lds_st(
                    xs,
                    (s * n + ks[i]) // 4,
                    fp8_pack4(values[0], values[1], values[2], values[3]).bitcast(fx.Float32),
                )
        return scales

    def stage_x_pairs_ptpc(name, count, n, src_of, group=None, retain_bf16=False):
        """Quantize packed-BF16 mailbox rows with one scale per activation group."""
        group = n if group is None else group
        groups = (n + group - 1) // group
        scales = []
        for s in range_constexpr(count):
            for g in range_constexpr(groups):
                width = min(group, n - g * group)
                words = width // 4
                per = (words + THREADS - 1) // THREADS
                local_max = fx.Float32(0.0)
                row_values = []
                valids = []
                for i in range_constexpr(per):
                    word = tid + i * THREADS
                    source_word = fx.min(word, words - 1)
                    element = s * n + g * group + source_word * 4
                    got = poll([(mb(name), src_of(element) // 2, 2)])[0]
                    first = bf2_f32(got[0])
                    second = bf2_f32(got[1])
                    values = [first[0], first[1], second[0], second[1]]
                    valid = word < words
                    row_values.append(values)
                    valids.append(valid)
                    for value in values:
                        local_max = fx.max(
                            local_max,
                            valid.select(fmath.absf(value), fx.Float32(0.0)),
                        )
                maximum = block_maxs([local_max])[0]
                scale = (maximum == 0.0).select(fx.Float32(1.0), maximum * (1.0 / FP8_MAX))
                scales.append(scale)
                reciprocal = 1.0 / scale
                for i in range_constexpr(per):
                    word = tid + i * THREADS
                    if valids[i]:
                        values = row_values[i]
                        if const_expr(retain_bf16):
                            pair_word = (s * n + g * group + word * 4) // 2
                            lds_st(
                                xs,
                                INDEX_Q_BF16_OFF + pair_word,
                                bf16_pair(values[0], values[1]),
                            )
                            lds_st(
                                xs,
                                INDEX_Q_BF16_OFF + pair_word + 1,
                                bf16_pair(values[2], values[3]),
                            )
                        quantized = [div_rn(value, scale, reciprocal) for value in values]
                        lds_st(
                            xs,
                            (s * n + g * group + word * 4) // 4,
                            fp8_pack4(
                                quantized[0],
                                quantized[1],
                                quantized[2],
                                quantized[3],
                            ).bitcast(fx.Float32),
                        )
        return scales

    def stage_x_rmsnorm(ld4s, n, gamma, loaded=None, count=S, precise=False):
        """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
        ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
        ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
        per = n // (4 * THREADS)
        ks = [(tid + i * THREADS) * 4 for i in range(per)]
        gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
        sss = []
        for s in range_constexpr(count):
            ss = fx.Float64(0.0) if const_expr(precise) else fx.Float32(0.0)
            for i in range_constexpr(per):
                for a in vals[s * per + i]:
                    if const_expr(precise):
                        a = fx.Float64(a)
                    ss = ss + a * a
            sss.append(ss)
        if const_expr(precise):
            rstds = [fx.Float64(1.0) / fx.sqrt(tot * (1.0 / n) + EPS, fastmath="none") for tot in block_sums_f64(sss)]
        else:
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
        for s in range_constexpr(count):
            for i in range_constexpr(per):
                a = vals[s * per + i]
                for j in range_constexpr(2):
                    if const_expr(precise):
                        v0 = fx.Float32(fx.Float64(a[2 * j]) * rstds[s] * fx.Float64(gs[i][2 * j]))
                        v1 = fx.Float32(fx.Float64(a[2 * j + 1]) * rstds[s] * fx.Float64(gs[i][2 * j + 1]))
                    else:
                        v0 = a[2 * j] * rstds[s] * gs[i][2 * j]
                        v1 = a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]
                    lds_st(
                        xs,
                        (s * n + ks[i]) // 2 + j,
                        bf16_pair(
                            v0,
                            v1,
                        ),
                    )
        return rstds

    def load_x_rmsnorm(ld4s, n, gamma, count=S):
        """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
        rg_ = _rsrc(gamma)
        ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
        gs = []
        for k in ks:
            g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
            gs.append([g[j] for j in range(4)])
        return gs, ld4s([(s, k) for s in range(count) for k in ks])

    def stage_x_pairs(name, n_total, src_of, padded_row=0):
        """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
        (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements.
        """
        nq = n_total // 4
        full = nq // THREADS
        vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
        for i in range_constexpr(full):
            for j in range_constexpr(2):
                xp_word = (tid + i * THREADS) * 2 + j
                if const_expr(padded_row > 0):
                    xp_word = (xp_word // (padded_row // 2)) * (padded_row // 2 + 4) + xp_word % (padded_row // 2)
                lds_st(xs, xp_word, vals[i][j].bitcast(fx.Float32))
        if const_expr(nq % THREADS):
            w = tid + full * THREADS
            if w < nq:
                v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                for j in range_constexpr(2):
                    xp_word = w * 2 + j
                    if const_expr(padded_row > 0):
                        xp_word = (xp_word // (padded_row // 2)) * (padded_row // 2 + 4) + xp_word % (padded_row // 2)
                    lds_st(xs, xp_word, v[j].bitcast(fx.Float32))

    return NormalizationTools(
        stage_x_rmsnorm_ptpc=stage_x_rmsnorm_ptpc,
        stage_x_pairs_ptpc=stage_x_pairs_ptpc,
        stage_x_rmsnorm=stage_x_rmsnorm,
        load_x_rmsnorm=load_x_rmsnorm,
        stage_x_pairs=stage_x_pairs,
    )
