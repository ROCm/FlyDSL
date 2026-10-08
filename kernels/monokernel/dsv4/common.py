# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device helpers shared by every stage of the DeepSeek-V4 MonoKernel."""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.config import EPS, FP8_MAX, ROUTE_SCALE, SCALE_BM
from kernels.monokernel.dsv4.plan import MIN_I32, POLL_MAX, ROW_TILE, THREADS, TL_COLS, WAVES
from kernels.monokernel.layout import CM_DEV, CM_SYS
from kernels.monokernel.ops import (
    bf2_f32,
    bf16_pair,
    exp,
    f8_word,
    fp8_roundtrip,
    fp8_to_bf16x8,
    ld_f32,
    lds_ld,
    lds_st,
    mem_realtime,
    mxfp4_to_bf16x8,
    mxfp8_to_bf16x8,
    rcp,
    rsq,
    rsrc,
    uniform,
    wave_max,
    wave_sum,
    wave_umax,
    write_lane_i32,
    xred,
    xshfl,
)


def wave_any(pred):
    """Whether ``pred`` holds on any lane of the wave (wave-uniform)."""
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, pred.ir_value())) != fx.Int64(0)


def sqrt_softplus(x):
    """V4's router score sqrt(softplus(x)), softplus as max(x, 0) + log1p(exp(-|x|))."""
    sp = fx.max(x, fx.Float32(0.0)) + fmath.log1p(exp(-fmath.absf(x)))
    return fmath.sqrt(sp)


def swiglu(g, u, limit):
    """SwiGLU with V4's clamp: ``up`` on both sides, ``gate`` only from above."""
    if const_expr(limit > 0):
        g = fx.min(g, fx.Float32(limit))
        u = fx.min(fx.max(u, fx.Float32(-limit)), fx.Float32(limit))
    return g * rcp(1.0 + exp(-g)) * u


def sort_network(n):
    """Odd-even transposition compare-exchange pairs for ``n`` elements."""
    pairs = []
    for r in range(n):
        for i in range(r % 2, n - 1, 2):
            pairs.append((i, i + 1))
    return pairs


FP4_MAX = 6.0


def fp4_roundtrip(a, b):
    """f32 pair -> E2M1 -> f32 pair (pre-scaled inputs), plus the codes word (``a``
    in the low nibble). Scaling is done in f32 around this; the scale operand is 1.0."""
    one = as_ir_value(fx.Float32(1.0))
    word = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, as_ir_value(fx.Int32(0)), a, b, one, 0))
    v2 = fx.Vector.make_type(2, fx.Float32)
    out = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp4(res=v2, src=as_ir_value(word), scale=one, src_sel_index=0))
    return out[0], out[1], word


def pow2_ceil(x):
    """Smallest power of two >= x: V4's FP4 block scale (reference.quant_dequant_fp4)."""
    bits = fx.Float32(x).bitcast(fx.Int32)
    man = bits & ((1 << 23) - 1)
    e = ((bits >> 23) & 0xFF) - 127 + (man != 0).select(fx.Int32(1), fx.Int32(0))
    return ((e + 127) << 23).bitcast(fx.Float32)


def bf16x2_has_nan(w):
    """Whether either bf16 half of a word is NaN."""
    w = fx.Int32(w)
    return ((w & 0x7FFF) > fx.Int32(0x7F80)) | (((w >> 16) & 0x7FFF) > fx.Int32(0x7F80))


@ASTRewriter.transform
def common_defs(ctx):
    """The helpers every stage shares: mailboxes, reductions, the MFMA GEMV machinery,
    activation staging, routing and the TP peer reduce."""
    BOUNDED_POLL = ctx["BOUNDED_POLL"]
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    ID_BITS = ctx["ID_BITS"]
    ID_MASK = ctx["ID_MASK"]
    N_COMP = ctx["N_COMP"]
    N_EXPERTS = ctx["N_EXPERTS"]
    POLL_TIMEOUT_TICKS = ctx["POLL_TIMEOUT_TICKS"]
    S = ctx["S"]
    SC = ctx["SC"]
    SY = ctx["SY"]
    TK_PARTS = ctx["TK_PARTS"]
    TK_PER = ctx["TK_PER"]
    TK_TRIPS = ctx["TK_TRIPS"]
    TOPK_SUM_OFFS = ctx["TOPK_SUM_OFFS"]
    TOP_K = ctx["TOP_K"]
    W = ctx["W"]
    XQ_BLOCKS = ctx["XQ_BLOCKS"]
    base = ctx["base"]
    bias = ctx["bias"]
    bid = ctx["bid"]
    first = ctx["first"]
    hang = ctx["hang"]
    lane = ctx["lane"]
    misc = ctx["misc"]
    outs = ctx["outs"]
    peer_dst = ctx["peer_dst"]
    rank = ctx["rank"]
    red = ctx["red"]
    scratch = ctx["scratch"]
    sym = ctx["sym"]
    tag = ctx["tag"]
    tid = ctx["tid"]
    tid2eid = ctx["tid2eid"]
    timeline = ctx["timeline"]
    timeline_buf = ctx["timeline_buf"]
    tok_ids = ctx["tok_ids"]
    use_fp8_block128 = ctx["use_fp8_block128"]
    use_hash = ctx["use_hash"]
    use_mxfp8_block32 = ctx["use_mxfp8_block32"]
    v4f = ctx["v4f"]
    wave = ctx["wave"]
    xs = ctx["xs"]

    # ------------------------------------------------------------ helpers
    # ---- tagged-pair mailboxes
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

    def put_bf(base_addr, i, vs, cm=CM_DEV):
        """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
        i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(base_addr), i, cache_modifier=cm)

    def poll(specs, scope="agent", batch=POLL_MAX):
        """Batched poll of mailbox pairs ``specs`` = [(base_addr, pair index, npairs in
        {1, 2})]: re-load the batch until every tag matches; one Int32 list per spec.
        The s_nop in the retry loop keeps the loads from being hoisted."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):  # bound live registers
            return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
        cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

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
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        # bounded: a timed-out poll stores the tag into ``hang``; later polls seeing it give up
        v = load_all()
        if const_expr(BOUNDED_POLL):
            t0 = mem_realtime()
            stop = fx.Int32(0)
            while pending(v) & (stop == 0):
                rocdl.s_nop(0)
                v = load_all()
                flagged = uniform(bo.buffer_load(rsrc(hang), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) == tag
                late = (mem_realtime() - t0) > fx.Int64(POLL_TIMEOUT_TICKS)
                stop = late.select(fx.Int32(2), flagged.select(fx.Int32(1), fx.Int32(0)))
            if stop == 2:
                bo.buffer_store(tag, rsrc(hang), 0, cache_modifier=CM_DEV)
        else:
            while pending(v):
                rocdl.s_nop(0)
                v = load_all()
        return unpack(v)

    def hint_wait(n, addr_of, mark=None):
        """Block barrier before a stage polls its inputs; consumers poll their payload
        directly (tight per-wave spins). ``mark`` stamps timeline column 1."""
        if const_expr(mark is not None):
            stamp(mark[0], mark[1], 1)
        gpu.barrier()

    def pre_poll(n, addr_of):
        """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
        before a large payload poll, so waiting CTAs do not flood memory."""
        if wave == 0:
            b, i = addr_of(fx.min(lane, n - 1))
            poll([(b, i, 1)])
        gpu.barrier()

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

    # ---- wave reductions
    def subgroup16_max(v):
        for off in (8, 4, 2, 1):
            v = xred(v, off, fx.max)
        return v

    def _other_parts(part):
        """Every top-k part but this one, starting just after it (a part never reads
        its own global store back)."""
        return [(part + 1 + k) % TK_PARTS for k in range(TK_PARTS - 1)]

    def part_keys(sbase, part, n_live, n_parts):
        """(trip bases, unsigned order-preserving keys) of this thread's candidates in part
        ``part``, held in registers. Bases are clamped and trips past ``n_live`` poll
        candidate 0 (never unwritten); the caller masks on the unclamped index."""
        cbs = [((fx.Int32(j) * n_parts + part) * THREADS + tid) * TK_PER for j in range(TK_TRIPS)]
        specs = []
        for cb in cbs:
            a = (cb < n_live).select(fx.min(cb, fx.Int32(N_COMP - TK_PER)), fx.Int32(0))
            specs += [(mb("i_score"), sbase + a + 2 * q, 2) for q in range(TK_PER // 2)]
        ws = [w[e] for w in poll(specs) for e in range(2)]
        return cbs, [(w ^ ((w >> 31) & 0x7FFFFFFF)) ^ MIN_I32 for w in ws]

    def block_excl_scan(v):
        """Exclusive prefix sum of a per-thread int32 over the block, and the block
        total: an xor-butterfly scan per wave, then the wave totals through LDS."""
        x = fx.Float32(v)
        pre = fx.Float32(0.0)
        for sh in range_constexpr(6):
            off = 1 << sh
            p = xshfl(x, off)
            pre = ((lane & off) != 0).select(pre + p, pre)
            x = x + p  # every lane now holds the sum of its 2 * off block
        if lane == 0:
            lds_st(red, wave, x)
        gpu.barrier()
        tot = fx.Float32(0.0)
        for i in range_constexpr(WAVES):
            t = lds_ld(red, i)
            pre = (fx.Int32(i) < wave).select(pre + t, pre)
            tot = tot + t
        gpu.barrier()
        return fx.Int32(pre), fx.Int32(tot)

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

    def block_max(v):
        w = wave_max(v)
        if lane == 0:
            lds_st(red, wave, w)
        gpu.barrier()
        t = lds_ld(red, 0)
        for i in range_constexpr(1, WAVES):
            t = fx.max(t, lds_ld(red, i))
        gpu.barrier()
        return t

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

    # ------------------------------------------------ MFMA GEMV machinery
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
        """Issue one 128-K tile of an MXFP4 bank in ATOM's gfx950 layout and its E8M0
        scale (lane ``16 * kl + r`` holds one 32-K scale block of row ``r``; the scale
        dword's four bytes cover tiles kc (even, odd) x row groups (even, odd))."""
        ln = lane if ln is None else ln
        k1n = -(-(K // 32) // 8)  # scale K blocks, padded to 8, in groups of 8
        raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        word = fx.Int32(bo.buffer_load(s_rsrc, ((rg // 2) * k1n + kc // 2) * 64 + ln, vec_width=1, dtype=T.i32))
        sc_byte = word.shrui(((kc % 2) * 2 + rg % 2) * 8) & fx.Int32(0xFF)
        return ("mxfp4", (raw, sc_byte), coef, b_word + (lane // 16) * 16)

    def mx_rg(c):
        """Up/gate tile ``c`` (8 intermediates: gate rows on lanes r < 8, up rows on
        r >= 8) as this lane's row group of the gate/up bank in ATOM's order."""
        return (c // 2) * 2 + (lane % 16) // 8

    def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
        ln = lane if ln is None else ln
        wv = [
            fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            for sp in range(2)
        ]
        return ("bf16", wv, None, b_word + (lane // 16) * 4)

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
            if const_expr(callable(coef) and unit_format != "mxfp4"):
                coef = coef()
            if const_expr(unit_format == "mxfp4"):
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
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            if const_expr(unit_format == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                bv = [fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32) for h in range(2)]
                b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                one = fx.Int32(127)
                c = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))
            for sp in range_constexpr(2 if unit_format != "f8f8" else 0):
                if const_expr(unit_format == "fp8"):
                    a = fp8_to_bf16x8(wv[0][sp * 2], wv[0][sp * 2 + 1])
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

    def reduce_rows(R, acc, emit):
        """Sum the per-wave MFMA tiles of each of R row groups; emit(row_local, n, v) for n < S."""
        wpr = WAVES // R
        fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
        gpu.barrier()
        n_out = R * 16 * S
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

    def _rmsnorm_tail_ks(n):
        """This thread's group-of-4 starting indices of n elements. A ragged tail is
        clamped to the last group (idempotent rewrites); ``active`` masks its
        contribution to sums (None when there is no tail)."""
        nq = n // 4
        full = nq // THREADS
        ks = [(tid + i * THREADS) * 4 for i in range(full)]
        active = None
        if const_expr(nq % THREADS):
            w = tid + full * THREADS
            active = w < nq
            ks.append(fx.min(w, nq - 1) * 4)
        return ks, active

    def stage_x_rmsnorm(ld4s, n, gamma, loaded=None, count=S):
        """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
        ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
        ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
        ks, active = _rmsnorm_tail_ks(n)
        per = len(ks)
        gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
        sss = []
        for s in range_constexpr(count):
            ss = fx.Float32(0.0)
            for i in range_constexpr(per):
                for a in vals[s * per + i]:
                    term = a * a
                    if const_expr(active is not None and i == per - 1):
                        term = active.select(term, fx.Float32(0.0))
                    ss = ss + term
            sss.append(ss)
        rstds = [rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
        for s in range_constexpr(count):
            for i in range_constexpr(per):
                a = vals[s * per + i]
                for j in range_constexpr(2):
                    lds_st(
                        xs,
                        (s * n + ks[i]) // 2 + j,
                        bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                    )
        return rstds

    def load_x_rmsnorm(ld4s, n, gamma, count=S):
        """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
        rg_ = rsrc(gamma)
        ks, _ = _rmsnorm_tail_ks(n)
        gs = []
        for k in ks:
            g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
            gs.append([g[j] for j in range(4)])
        return gs, ld4s([(s, k) for s in range(count) for k in ks])

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

    def quant_mxfp8(a0, a1):
        """Per-16-lane/32-value MXFP8 quantization; the E8M0 scale rounds up so the
        block max never clips."""

        amax = subgroup16_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
        nz = amax > 0.0
        scale = nz.select(pow2_ceil(amax * (1.0 / FP8_MAX)), fx.Float32(1.0))
        inv = nz.select(rcp(scale), fx.Float32(1.0))
        q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
        q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
        d0, d1 = fp8_roundtrip(q0, q1)
        return d0, d1, scale

    def stage_moe_input(samples):
        """Stage normalized expert inputs published by the router into LDS."""
        if const_expr(use_fp8_block128):
            # HIDDEN // 4 slots of 4 FP8 bytes; ragged tail clamped (see _rmsnorm_tail_ks)
            nq = HIDDEN // 4
            full = nq // THREADS
            xk = [tid + i * THREADS for i in range(full)]
            if const_expr(nq % THREADS):
                xk.append(fx.min(tid + full * THREADS, nq - 1))
            nxw = len(xk)
            got = poll(
                [(mb("xq"), sx * nq + k, 1) for sx in samples for k in xk]
                + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    wd = f8_word(xk[i] * 4)
                    lds_st(xs, j * nq + wd, got[j * nxw + i][0].bitcast(fx.Float32))
                if tid < XQ_BLOCKS:
                    lds_st(
                        misc,
                        8 + j * XQ_BLOCKS + tid,
                        got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                    )
        elif const_expr(use_mxfp8_block32):
            chunks = HIDDEN // 8
            per_thread = (chunks + THREADS - 1) // THREADS
            # the power-of-two block scale folds exactly into the conversion
            data_specs, scale_specs = [], []
            for sx in samples:
                for i in range_constexpr(per_thread):
                    chunk = fx.min(tid + i * THREADS, chunks - 1)
                    data_specs.append((mb("xq"), sx * (HIDDEN // 4) + chunk * 2, 2))
                    scale_specs.append((mb("xqs"), sx * XQ_BLOCKS + chunk // 4, 1))
            got = poll(data_specs + scale_specs)
            nd = len(data_specs)
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(per_thread):
                    chunk = tid + i * THREADS
                    if chunk < chunks:
                        words = got[j * per_thread + i]
                        qs = got[nd + j * per_thread + i][0].bitcast(fx.Float32)
                        values = mxfp8_to_bf16x8(words[0], words[1], qs)
                        for pair in range_constexpr(4):
                            lds_st(
                                xs,
                                j * (HIDDEN // 2) + chunk * 4 + pair,
                                fx.Vector.from_elements([values[2 * pair], values[2 * pair + 1]], fx.BFloat16).bitcast(
                                    fx.Float32
                                )[0],
                            )
        else:
            nxw = HIDDEN // 2 // THREADS
            got = poll([(mb("xq"), sx * (HIDDEN // 2) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)])
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    lds_st(
                        xs,
                        j * (HIDDEN // 2) + tid + i * THREADS,
                        got[j * nxw + i][0].bitcast(fx.Float32),
                    )

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

    def route_topk(s, raws=None, bs=None):
        """Flat top-``TOP_K`` of sample s (one whole wave): each round is one u32 wave
        max over packed keys of (score + bias). Returns (expert id, raw score / sum *
        ROUTE_SCALE) of pick ``lane``, valid in lanes < TOP_K. A hash-routed layer
        (``use_hash``, runtime) takes the ids from ``tid2eid[token]`` instead."""
        KPL = N_EXPERTS // 64  # selection keys held per lane
        if const_expr(bs is None):
            bs = load_bias()
        if const_expr(raws is None):
            raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * 64) for i in range(KPL)])
        ks = []
        for i in range_constexpr(KPL):
            kb = (raws[i] + bs[i]).bitcast(fx.Int32)
            ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
            ks.append(fx.Uint32((ok & fx.Int32(-(1 << ID_BITS))) | (ID_MASK - (lane + i * 64))))
        # sort each lane's keys descending; a round takes the wave max of the heads
        for a, b in sort_network(KPL):
            ks[a], ks[b] = fx.max(ks[a], ks[b]), fx.min(ks[a], ks[b])
        ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
        mv = fx.Int32(0)  # lane k: the key of pick k
        for k in range_constexpr(TOP_K):
            m = wave_umax(ks[0])
            hit = ks[0] == m
            ks = [hit.select(ks[i + 1], ks[i]) for i in range(KPL)] + [ks[KPL]]
            mv = write_lane_i32(m, k, mv)
        e = ID_MASK - (mv & ID_MASK)
        # a scored layer passes null tables: zero records makes the loads return 0
        nrec = (use_hash != 0).select(fx.Int32(0x7FFFFFF0), fx.Int32(0))
        r_tok = bo.create_buffer_resource_from_addr(tok_ids, num_records_bytes=nrec)
        r_t2e = bo.create_buffer_resource_from_addr(tid2eid, num_records_bytes=nrec)
        tok = fx.Int32(bo.buffer_load(r_tok, s, vec_width=1, dtype=T.i32))
        hk = tok * TOP_K + fx.min(lane, fx.Int32(TOP_K - 1))
        e_hash = fx.Int32(bo.buffer_load(r_t2e, hk, vec_width=1, dtype=T.i32))
        e = (use_hash != 0).select(e_hash, e)
        src = (e % 64) * 4
        got = [fx.Int32(rocdl.ds_bpermute(T.i32, src.ir_value(), r.bitcast(fx.Int32).ir_value())) for r in raws]
        raw = got[0]
        for i in range_constexpr(1, N_EXPERTS // 64):
            raw = (e // 64 == i).select(got[i], raw)
        raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
        tot = raw
        for off in TOPK_SUM_OFFS:
            tot = xred(tot, off, lambda a, b: a + b)
        return e, raw * (rcp(tot) * ROUTE_SCALE)

    def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
        """Push BF16 partials to every peer, then sum all ranks' in rank order.
        ``residual``: fn(s, row) -> (r0, r1), a mailbox base polled with the peers,
        or None when the caller adds its own (hc_post)."""
        if const_expr(W > 1):
            # one wave per destination peer
            if wave < W:
                pair_count = S * tile // 2
                for batch in range_constexpr((pair_count + 63) // 64):
                    pair = lane + batch * 64
                    if pair < pair_count:
                        si = pair // (tile // 2)
                        ri = (pair % (tile // 2)) * 2
                        put_bf(
                            peer_dst + fx.Int64(SY[region]),
                            (rank * S + si) * HIDDEN + t * tile + ri,
                            [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
                            CM_SYS,
                        )
            gpu.barrier()
        if tid < S * tile // 2:
            s = tid // (tile // 2)
            r = (tid % (tile // 2)) * 2
            row = t * tile + r
            r0 = fx.Float32(0.0)
            r1 = fx.Float32(0.0)
            if const_expr(callable(residual)):
                r0, r1 = residual(s, row)
            v0 = lds_ld(outs, s * tile + r)
            v1 = lds_ld(outs, s * tile + r + 1)
            if const_expr(W == 1):
                parts = [(v0, v1)]
                got = []
                if const_expr(residual is not None and not callable(residual)):
                    got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
            else:
                own = sym + fx.Int64(SY[region])
                specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                if const_expr(residual is not None and not callable(residual)):
                    specs.append((residual, (s * HIDDEN + row) // 2, 1))
                got = poll(specs, "one-as")
                parts = [bf2_f32(v[0]) for v in got[:W]]
                got = got[W:]
            if const_expr(residual is not None and not callable(residual)):
                r0, r1 = bf2_f32(got[0][0])
            t0 = fx.Float32(0.0)
            t1 = fx.Float32(0.0)
            for src in range_constexpr(W):
                t0 = t0 + parts[src][0]
                t1 = t1 + parts[src][1]
            out_fn(s, row, r0 + t0, r1 + t1)

    def start(name):
        return (bid + (G - base[name])) & (G - 1)

    def stamp(name, t, which, pred=None):
        if const_expr(timeline):
            # nothing may cross the clock read (constrains timeline builds only)
            rocdl.sched_barrier(0)
            # a masked-off rep must not overwrite the live task's row
            ok = (tid == 0) if const_expr(pred is None) else ((tid == 0) & pred)
            if ok:
                now = mem_realtime()
                fx.generic_store(
                    fx.inttoptr(
                        fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                        timeline_buf + fx.Int64((first[name] + t) * TL_COLS + which) * 8,
                    ),
                    now,
                )

    def n_sel():
        """This lane's MFMA B column (sample); columns >= S duplicate the last one."""
        return fx.min(lane % 16, S - 1)

    return dict(
        _other_parts=_other_parts,
        _rmsnorm_tail_ks=_rmsnorm_tail_ks,
        block_excl_scan=block_excl_scan,
        block_max=block_max,
        block_sum=block_sum,
        block_sums=block_sums,
        emit_out=emit_out,
        get=get,
        get2_many=get2_many,
        getf=getf,
        getf_many=getf_many,
        hint_wait=hint_wait,
        load_bias=load_bias,
        load_x_rmsnorm=load_x_rmsnorm,
        mb=mb,
        mma_units=mma_units,
        mx_rg=mx_rg,
        n_sel=n_sel,
        part_keys=part_keys,
        peer_reduce=peer_reduce,
        poll=poll,
        pre_poll=pre_poll,
        put=put,
        put2=put2,
        put_bf=put_bf,
        quant_mxfp8=quant_mxfp8,
        quant_scaled=quant_scaled,
        reduce_rows=reduce_rows,
        route_topk=route_topk,
        run_units=run_units,
        st_f8=st_f8,
        stage_moe_input=stage_moe_input,
        stage_x_pairs=stage_x_pairs,
        stage_x_rmsnorm=stage_x_rmsnorm,
        stamp=stamp,
        start=start,
        unit_bf16=unit_bf16,
        unit_f8f8=unit_f8f8,
        unit_fp8=unit_fp8,
        unit_fp8mx=unit_fp8mx,
        unit_mxfp4=unit_mxfp4,
    )
