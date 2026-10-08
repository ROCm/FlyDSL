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
from kernels.monokernel.layout import CM_DEV, CM_SYS, POLL_MAX, THREADS, TL_COLS, WAVES
from kernels.monokernel.ops import (
    bf16_pair,
    f8_word,
    fp8_to_bf16x8,
    ld_f32,
    lds_ld,
    lds_st,
    mem_realtime,
    mxfp4_to_bf16x8,
    rcp,
    rsrc,
    spin_pause,
    uniform,
    wave_max,
    wave_sum,
    xshfl,
)

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
    hang=None,
    poll_timeout_ticks=None,
    retry_spin_pause=False,
    timeline=False,
    timeline_buf=None,
    first=None,
    stamp_fence=False,
    timeline_addr=None,
    tl_cols=TL_COLS,
    mxfp4_layout="atom",
):
    """The shared helpers, bound to one kernel's launch state (a dict of name -> function).

    ``source_key`` is the caller's captured ``SHARED_SOURCE_KEY`` (see there)."""
    assert source_key == SHARED_SOURCE_KEY, "the kernel was built against older shared helper sources"

    v4f = fx.Vector.make_type(4, fx.Float32)

    def poll(specs, scope="agent", batch=POLL_MAX, expected_tag=None):
        """Batched poll of mailbox pairs ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

        All pairs are loaded together with plain 8 / 16-byte coherent buffer loads (sc1 locally,
        sc0 sc1 for peer memory); while any tag is not ``expected_tag`` (default: this launch's)
        the whole batch is re-loaded, so a batch costs one round trip after its last producer
        lands. A side-effecting op in the retry loop keeps the loads from being hoisted. With
        ``poll_timeout_ticks`` a timed-out poll stores the tag into ``hang``, and later polls
        seeing it give up. Returns one list of Int32 value bits per spec."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):  # bound live registers
            return poll(specs[:batch], scope, batch, expected_tag) + poll(specs[batch:], scope, batch, expected_tag)
        cm = CM_DEV if const_expr(scope == "agent") else CM_SYS
        wanted_tag = tag if expected_tag is None else expected_tag

        def load_all():
            words = []
            for b, i, n in specs:
                w = fx.Vector(bo.buffer_load(rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm))
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def unpack(v):
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def pending(v):
            bad = v[1] != wanted_tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != wanted_tag)
            return bad

        v = load_all()
        if const_expr(poll_timeout_ticks is not None):
            t0 = mem_realtime()
            stop = fx.Int32(0)
            while pending(v) & (stop == 0):
                rocdl.s_nop(0)
                v = load_all()
                flagged = uniform(bo.buffer_load(rsrc(hang), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) == tag
                late = (mem_realtime() - t0) > fx.Int64(poll_timeout_ticks)
                stop = late.select(fx.Int32(2), flagged.select(fx.Int32(1), fx.Int32(0)))
            if stop == 2:
                bo.buffer_store(tag, rsrc(hang), 0, cache_modifier=CM_DEV)
        else:
            while pending(v):
                if const_expr(retry_spin_pause):
                    spin_pause()
                else:
                    rocdl.s_nop(0)
                v = load_all()
        return unpack(v)

    def stamp(name, t, which, lead=0, pred=None):
        """Timeline column ``which`` of stage ``name``'s task ``t`` := s_memrealtime, written by
        thread ``lead`` (and only where ``pred``); timeline builds only."""
        if const_expr(timeline):
            if const_expr(stamp_fence):
                # nothing may cross the clock read (constrains timeline builds only)
                rocdl.sched_barrier(0)
            # a masked-off rep must not overwrite the live task's row
            ok = (tid == lead) if const_expr(pred is None) else ((tid == lead) & pred)
            if ok:
                now = mem_realtime()
                tl_addr = timeline_buf if const_expr(timeline_addr is None) else timeline_addr()
                fx.generic_store(
                    fx.inttoptr(
                        fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                        tl_addr + fx.Int64((first[name] + t) * tl_cols + which) * 8,
                    ),
                    now,
                )

    def unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, ln=None, scale_rows=SCALE_BM):
        """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
        ln = lane if ln is None else ln
        wv = [
            fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            for h in range(2)
        ]
        s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
        if const_expr(callable(coef)):
            return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
        if const_expr(coef is not None):
            s = s * coef
        return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

    def unit_fp8mx(w_rsrc, s_rsrc, rg, s_rg, kc, K, b_word, coef=None, ln=None):
        """One 128-K chunk of row group ``rg`` of an FP8 128x128-scaled matrix (``s_rg`` =
        the row group of this lane's output rows) against bf16 at LDS word ``b_word``."""
        ln = lane if ln is None else ln
        wv = [
            fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 64) + kc * 2 + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            for h in range(2)
        ]
        s = ld_f32(s_rsrc, (s_rg * 16 // SCALE_BM) * (K // 128) + kc)
        return ("fp8mx", wv, (s, coef), b_word + (lane // 16) * 4)

    def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
        """Issue one 128-K MXFP4 tile and its E8M0 scales, in this kernel's ``mxfp4_layout``:
        "atom" (ATOM's gfx950 bank: lane ``16 * kl + r`` holds one 32-K scale block of row
        ``r``; the scale dword's four bytes cover tiles kc (even, odd) x row groups (even, odd)),
        or "rows_fp8" / "rows_split" (four per-row E8M0 scales, against FP8 activations, or bf16
        ones with one MFMA chain per 32-K part)."""
        ln = lane if ln is None else ln
        if const_expr(mxfp4_layout == "atom"):
            k1n = -(-(K // 32) // 8)  # scale K blocks, padded to 8, in groups of 8
            raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            word = fx.Int32(bo.buffer_load(s_rsrc, ((rg // 2) * k1n + kc // 2) * 64 + ln, vec_width=1, dtype=T.i32))
            sc_byte = word.shrui(((kc % 2) * 2 + rg % 2) * 8) & fx.Int32(0xFF)
            return ("mxfp4_atom", (raw, sc_byte), coef, b_word + (lane // 16) * 16)
        raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        row = rg * 16 + ln % 16
        packed_scale = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32))
        scales = [
            ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
            for sp in range_constexpr(4)
        ]
        unit_format = "mxfp4_rows_fp8" if const_expr(mxfp4_layout == "rows_fp8") else "mxfp4_rows_split"
        return (unit_format, (raw, scales), coef, b_word + (lane // 16) * 4)

    def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
        """A row-scaled MXFP4 tile (``mxfp4_layout`` "rows_*") against bf16 activations."""
        _, weights, factor, _ = unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln)
        return ("mxfp4_rows_bf16", weights, factor, b_word + (lane // 16) * 4)

    def mma_units(acc, units):
        """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
        for unit_format, wv, coef, bw in units:
            if const_expr(unit_format == "fp8mx"):
                # one factor per unit, so the four K32 MFMAs chain into one partial
                ws, f = coef
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                for sp in range_constexpr(4):
                    a = fp8_to_bf16x8(wv[sp // 2][(sp % 2) * 2], wv[sp // 2][(sp % 2) * 2 + 1])
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                f = ws if const_expr(f is None) else ws * (f() if const_expr(callable(f)) else f)
                acc = [acc[e] + c[e] * f for e in range(4)]
                continue
            if const_expr(callable(coef) and unit_format not in ("mxfp4_atom", "mxfp4_rows_split")):
                coef = coef()
            if const_expr(unit_format == "mxfp4_atom"):
                # step sp takes K 32 * kl + 8 * sp .. of the lane's block (B words 16 * kl + 4 * sp)
                raw, sc_byte = wv
                assert not isinstance(coef, list), "per-K32 factors are folded into the operands"
                c = fx.Vector.from_elements(acc, fx.Float32) if coef is None else fx.Vector.filled(4, 0.0, fx.Float32)
                sc = (sc_byte << fx.Int32(23)).bitcast(fx.Float32)
                for sp in range_constexpr(4):
                    a = mxfp4_to_bf16x8(raw[sp], sc)
                    b = fx.ptr_load(xs + (bw + sp * 4), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                if const_expr(coef is None):
                    acc = [c[e] for e in range(4)]
                else:
                    f = coef() if const_expr(callable(coef)) else coef
                    acc = [acc[e] + c[e] * f for e in range(4)]
                continue
            if const_expr(unit_format == "mxfp4_rows_split"):
                # one MFMA per 32-K part, each with its own factor (``coef`` may be a list)
                raw, scales = wv
                for sp in range_constexpr(4):
                    a = mxfp4_to_bf16x8(raw[sp], scales[sp])
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    part_coef = coef[sp] if const_expr(isinstance(coef, list)) else coef
                    if const_expr(callable(part_coef)):
                        part_coef = part_coef()
                    if const_expr(part_coef is None):
                        acc = [acc[e] + c[e] for e in range(4)]
                    else:
                        acc = [acc[e] + c[e] * part_coef for e in range(4)]
                continue
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            if const_expr(unit_format in ("mxfp4_rows_fp8", "mxfp4_rows_bf16")):
                raw, scales = wv
                for sp in range_constexpr(4):
                    a = mxfp4_to_bf16x8(raw[sp], scales[sp])
                    if const_expr(unit_format == "mxfp4_rows_fp8"):
                        wh, ws = sp // 2, sp % 2
                        bv = fx.Vector(fx.ptr_load(xs + (bw + wh * 16), result_type=v4f)).bitcast(fx.Int32)
                        b = fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                    else:
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(unit_format == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                bv = [fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32) for h in range(2)]
                b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                one = fx.Int32(127)
                c = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))
            nsp = (
                4
                if unit_format == "fp8x2"
                else 2 if unit_format not in ("f8f8", "mxfp4_rows_fp8", "mxfp4_rows_bf16") else 0
            )
            for sp in range_constexpr(nsp):
                if const_expr(unit_format in ("fp8", "fp8x2")):
                    wh = sp // 2 if unit_format == "fp8x2" else 0
                    ws = sp % 2 if unit_format == "fp8x2" else sp
                    a = fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                else:
                    a = wv[sp].bitcast(fx.BFloat16)
                b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(coef is None):
                acc = [acc[e] + c[e] for e in range(4)]
            else:
                acc = [acc[e] + c[e] * coef for e in range(4)]
        return acc

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
        poll=poll,
        stamp=stamp,
        mma_units=mma_units,
        unit_fp8x2=unit_fp8x2,
        unit_fp8mx=unit_fp8mx,
        unit_mxfp4=unit_mxfp4,
        unit_mxfp4_bf16=unit_mxfp4_bf16,
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
