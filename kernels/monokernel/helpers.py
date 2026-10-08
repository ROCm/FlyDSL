# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Helpers every MonoKernel binds to its own launch: tagged-pair mailboxes, block reductions,
MFMA GEMV units and their reduction, activation staging and task placement.

``bind_helpers`` returns plain functions closed over the values a kernel passes in; a kernel calls
it in its body once its own ``poll``, ``stamp`` and ``mma_units`` exist.
"""

import hashlib
from pathlib import Path

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel import ops
from kernels.monokernel.config import FP8_MAX, SCALE_BM
from kernels.monokernel.layout import CM_DEV, THREADS, WAVES
from kernels.monokernel.ops import bf16_pair, f8_word, ld_f32, lds_ld, lds_st, rcp, rsrc, wave_max, wave_sum, xshfl

# A kernel's JIT disk-cache key covers only the sources in its own directory. A kernel that binds these
# helpers captures this digest of the shared device sources, so editing this file or ops.py recompiles it.
SHARED_SOURCE_KEY = hashlib.sha256(Path(__file__).read_bytes() + Path(ops.__file__).read_bytes()).hexdigest()[:16]


@ASTRewriter.transform
def bind_helpers(
    *,
    source_key,
    S,
    tid,
    lane,
    wave,
    bid,
    G,
    base,
    scratch,
    SC,
    tag,
    red,
    outs,
    xs,
    bias=None,
    N_EXPERTS=None,
    poll,
    stamp,
    mma_units
):
    """The shared helpers, bound to one kernel's launch state (a dict of name -> function).

    ``source_key`` is the caller's captured ``SHARED_SOURCE_KEY`` (see there)."""
    assert source_key == SHARED_SOURCE_KEY, "the kernel was built against older shared helper sources"

    def mb(name):
        return scratch + fx.Int64(SC[name])

    def put(base_addr, i, v, cm=CM_DEV):
        """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), rsrc(base_addr), i * 2, cache_modifier=cm)

    def put2(base_addr, i, v0, v1, cm=CM_DEV):
        """Pairs i, i+1 (i even) in one 16-byte store."""
        vec = fx.Vector.from_elements(
            [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
        )
        bo.buffer_store(vec, rsrc(base_addr), i * 2, cache_modifier=cm)

    def put_bf(base_addr, i, vs, cm=CM_DEV, store_tag=None):
        """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
        i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
        write_tag = tag if store_tag is None else store_tag
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), write_tag]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(base_addr), i, cache_modifier=cm)

    def get(base_addr, i):
        return poll([(base_addr, i, 1)])[0][0]

    def getf(base_addr, i):
        return get(base_addr, i).bitcast(fx.Float32)

    def getf_many(specs):
        """[(base, i)] single pairs -> list of f32."""
        return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]

    def get2_many(specs):
        """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
        return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]

    def pre_poll(n, addr_of):
        """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
        before a large payload poll, so waiting CTAs do not flood memory."""
        if wave == 0:
            b, i = addr_of(fx.min(lane, n - 1))
            poll([(b, i, 1)])
        gpu.barrier()

    def hint_wait(n, addr_of, mark=None):
        """Block barrier before a stage polls its inputs; consumers poll their payload
        directly (tight per-wave spins). ``mark`` stamps timeline column 1."""
        if const_expr(mark is not None):
            stamp(mark[0], mark[1], 1)
        gpu.barrier()

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

    def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None, ln=None):
        """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
        bf16 activation chunk starts at LDS word ``b_word``."""
        ln = lane if ln is None else ln
        wv = fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
        if const_expr(callable(coef)):  # factor known only after a later wait
            return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
        if const_expr(coef is not None):
            s = s * coef
        return ("fp8", [wv], s, b_word + (lane // 16) * 4)

    def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
        """Issue one 128-k chunk (64-k chunks kc, kc + 1) of row group ``rg`` against the
        FP8 activation at LDS word ``b_word``; ``coef()`` = activation scale (x route weight)."""
        ln = lane if ln is None else ln
        wv = [
            fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            for h in range(2)
        ]
        s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
        return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

    def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
        ln = lane if ln is None else ln
        wv = [
            fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            for sp in range(2)
        ]
        return ("bf16", wv, None, b_word + (lane // 16) * 4)

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
        fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
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

    def stage_x_pairs(name, n_total, src_of):
        """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
        (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements."""
        nq = n_total // 4
        full = nq // THREADS
        vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
        for i in range_constexpr(full):
            for j in range_constexpr(2):
                lds_st(xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32))
        if const_expr(nq % THREADS):
            w = tid + full * THREADS
            if w < nq:
                v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                for j in range_constexpr(2):
                    lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))

    def quant_scaled(a0, a1):
        """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
        amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
        nz = amax > 0.0
        qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
        inv = nz.select(rcp(amax) * FP8_MAX, fx.Float32(1.0))
        q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
        q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
        return q0, q1, qs

    def st_f8(k, q0, q1):
        """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
        k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
        w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
        nb = xshfl(w, 1)
        if lane % 2 == 0:
            lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

    def load_bias():
        """This lane's 4 expert biases (issue before the scores wait)."""
        return [ld_f32(rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

    def start(name):
        return (bid + (G - base[name])) & (G - 1)

    def n_sel(count=S):
        """This lane's local MFMA B column; inactive columns duplicate the last one."""
        return fx.min(lane % 16, count - 1)

    return dict(
        mb=mb,
        put=put,
        put2=put2,
        put_bf=put_bf,
        get=get,
        getf=getf,
        getf_many=getf_many,
        get2_many=get2_many,
        pre_poll=pre_poll,
        hint_wait=hint_wait,
        block_sums=block_sums,
        block_sum=block_sum,
        unit_fp8=unit_fp8,
        unit_f8f8=unit_f8f8,
        unit_bf16=unit_bf16,
        run_units=run_units,
        reduce_rows=reduce_rows,
        emit_out=emit_out,
        stage_x_pairs=stage_x_pairs,
        quant_scaled=quant_scaled,
        st_f8=st_f8,
        load_bias=load_bias,
        start=start,
        n_sel=n_sel,
    )
