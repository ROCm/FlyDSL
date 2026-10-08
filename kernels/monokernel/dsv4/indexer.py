# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The CSA lightning indexer of the DeepSeek-V4 MonoKernel: compressor, query, scores and top-k."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.common import FP4_MAX, fp4_roundtrip, pow2_ceil
from kernels.monokernel.dsv4.config import EPS, ROPE_DIM
from kernels.monokernel.dsv4.plan import IH_TASK, MIN_I32, Q_B_TILE, QKV_A_TILE, SCORE_TILE, THREADS, WAVES, q_b_groups
from kernels.monokernel.helpers import traced
from kernels.monokernel.layout import NEG
from kernels.monokernel.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    ld_bf16,
    ld_f32,
    lds_ld,
    lds_st,
    mxfp4_to_bf16x8,
    rcp,
    rsq,
    rsrc,
    wave_sum,
    xred,
    xshfl,
)


@traced
def index_query_stages(ctx):
    """2c-3c. The indexer's compressor, query and head weights (CSA only)."""
    CR = ctx["CR"]
    C_COFF = ctx["C_COFF"]
    C_RING = ctx["C_RING"]
    G = ctx["G"]
    HC = ctx["HC"]
    HIDDEN = ctx["HIDDEN"]
    IC_BLK_WORDS = ctx["IC_BLK_WORDS"]
    IC_GRP_WORDS = ctx["IC_GRP_WORDS"]
    IC_S_BLK = ctx["IC_S_BLK"]
    IH = ctx["IH"]
    IHD = ctx["IHD"]
    INDEXER_HADAMARD = ctx["INDEXER_HADAMARD"]
    IW = ctx["IW"]
    K_PB = ctx["K_PB"]
    N_IQB = ctx["N_IQB"]
    OVERLAP = ctx["OVERLAP"]
    Q_LORA = ctx["Q_LORA"]
    S = ctx["S"]
    TOK = ctx["TOK"]
    _rmsnorm_tail_ks = ctx["_rmsnorm_tail_ks"]
    block_sums = ctx["block_sums"]
    bt_block = ctx["bt_block"]
    emit_out = ctx["emit_out"]
    g_ickv = ctx["g_ickv"]
    g_in = ctx["g_in"]
    g_q = ctx["g_q"]
    get2_many = ctx["get2_many"]
    getf = ctx["getf"]
    getf_many = ctx["getf_many"]
    hint_wait = ctx["hint_wait"]
    i_ape = ctx["i_ape"]
    i_cache = ctx["i_cache"]
    i_cache_s = ctx["i_cache_s"]
    i_kv_state = ctx["i_kv_state"]
    i_score_state = ctx["i_score_state"]
    i_w = ctx["i_w"]
    lane = ctx["lane"]
    ld_pos = ctx["ld_pos"]
    ld_slot = ctx["ld_slot"]
    load_x_rmsnorm = ctx["load_x_rmsnorm"]
    mb = ctx["mb"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    r_h = ctx["r_h"]
    reduce_rows = ctx["reduce_rows"]
    ring_row = ctx["ring_row"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    run_units = ctx["run_units"]
    s_i_q_b = ctx["s_i_q_b"]
    slot_rsrc = ctx["slot_rsrc"]
    st_i = ctx["st_i"]
    st_ic = ctx["st_ic"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_fp8 = ctx["unit_fp8"]
    w_i_q_b = ctx["w_i_q_b"]
    wave = ctx["wave"]
    window_pool = ctx["window_pool"]

    def had_pair(v0, v1, ln):
        """FWHT over IHD channels (lane ln holds ln and ln + 64), scaled by IHD**-0.5;
        the identity without INDEXER_HADAMARD (ATOM's indexer)."""
        if const_expr(not INDEXER_HADAMARD):
            return v0, v1
        # range_constexpr, not a while: the shuffle offset must stay a constant
        for h in range_constexpr(6):
            st = 1 << h
            p0, p1 = xshfl(v0, st), xshfl(v1, st)
            hi = (ln & st) != 0
            v0 = hi.select(p0 - v0, v0 + p0)
            v1 = hi.select(p1 - v1, v1 + p1)
        v0, v1 = v0 + v1, v0 - v1
        sc = float(IHD) ** -0.5
        return v0 * sc, v1 * sc

    def fp4_block(v):
        """FP4 round trip over 32-lane blocks, power-of-two scale -> (value, code, scale)."""
        amax = fmath.absf(v)
        for off in (16, 8, 4, 2, 1):
            amax = xred(amax, off, fx.max)
        sc = pow2_ceil(fx.max(amax, fx.Float32(FP4_MAX * 2.0**-126)) * (1.0 / FP4_MAX))
        q = fx.min(fx.max(v * rcp(sc), -FP4_MAX), FP4_MAX)
        d, _, word = fp4_roundtrip(q, fx.Float32(0.0))
        return d * sc, word & 0xF, sc

    # ========== 2c. the indexer's compressor: same pooling, Hadamard + FP4 tail
    if const_expr(IHD):
        for tt in range(start("i_cmp"), S, G):
            tt = fx.Int32(tt)
            stamp("i_cmp", tt, 0)
            # one WAVE covers the row: lane ln holds channels ln and ln + 64
            ln = lane
            ilive = wave == 0
            p = ld_pos(tt)
            rs_ikv, rs_isc, isb = slot_rsrc(i_kv_state, tt, st_i), slot_rsrc(i_score_state, tt, st_i), 0
            icb = ld_slot(tt, st_ic)  # the indexer's key cache slot (bytes)
            slot = p % CR
            chs = [ln, ln + 64]
            ap0 = [[ld_f32(rsrc(i_ape), slot * IW + j * IHD + c) for j in range(C_COFF)] for c in chs]
            gg = [ld_bf16(rsrc(g_ickv), c) for c in chs]
            anchor = fx.max(p + 1 - CR, fx.Int32(0))
            kvv = [[getf(mb("i_kv"), (tt * C_COFF + j) * IHD + c) for j in range(C_COFF)] for c in chs]
            gtv = [[getf(mb("i_gate"), (tt * C_COFF + j) * IHD + c) for j in range(C_COFF)] for c in chs]
            stamp("i_cmp", tt, 2)
            if ilive:
                for e in range_constexpr(2):
                    for j in range_constexpr(C_COFF):
                        w = isb + (p % C_RING) * IW + j * IHD + chs[e]
                        bo.buffer_store(kvv[e][j], rs_ikv, w)
                        bo.buffer_store(gtv[e][j] + ap0[e][j], rs_isc, w)
            if (p + 1) % CR == 0:
                j_tok = tt % TOK
                pooled = []
                for e in range_constexpr(2):

                    def i_ring(i, e=e):
                        coff = (i >= CR).select(fx.Int32(IHD), fx.Int32(0)) if OVERLAP else 0
                        wi = isb + ring_row(p, i) * IW + coff + chs[e]
                        return ld_f32(rs_isc, wi), ld_f32(rs_ikv, wi)

                    def i_own(i, e=e):
                        h = 1 if (OVERLAP and i >= CR) else 0
                        return gtv[e][h] + ap0[e][h], kvv[e][h]

                    def i_launch(d, i, e=e):
                        h = 1 if (OVERLAP and i >= CR) else 0
                        sm = tt - fx.min(fx.Int32(d), j_tok)
                        q = fx.max(p - d, fx.Int32(0))
                        apq = ld_f32(rsrc(i_ape), (q % CR) * IW + h * IHD + chs[e])
                        return (
                            getf(mb("i_gate"), (sm * C_COFF + h) * IHD + chs[e]) + apq,
                            getf(mb("i_kv"), (sm * C_COFF + h) * IHD + chs[e]),
                        )

                    den_, num_ = window_pool(p, j_tok, i_ring, i_launch, i_own)
                    pooled.append(bf16_round(num_ * rcp(den_)))
                sq = pooled[0] * pooled[0] + pooled[1] * pooled[1]
                for off in range_constexpr(6):
                    sq = xred(sq, 32 >> off, lambda a, b: a + b)
                rs = rsq(sq * (1.0 / IHD) + EPS)
                nv = [bf16_round(pooled[e] * rs * gg[e]) for e in range(2)]
                # rope is entirely in the second half, so only channel ln + 64 rotates
                rc = ld_f32(rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ln // 2)
                rs2 = ld_f32(rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ln // 2)
                partner = xshfl(nv[1], 1)
                even = ln % 2 == 0
                nv[1] = bf16_round(even.select(nv[1] * rc - partner * rs2, partner * rs2 + nv[1] * rc))
                h0, h1 = had_pair(nv[0], nv[1], ln)
                q0, q1 = bf16_round(h0), bf16_round(h1)
                (o0, k0, s0), (o1, k1, s1) = fp4_block(q0), fp4_block(q1)
                # eight lanes' codes OR into one word (nibble j); group g's exponent in byte g
                cw = [k0 << ((ln % 8) * 4), k1 << ((ln % 8) * 4)]
                for off in (1, 2, 4):
                    cw = [xred(w, off, lambda a, b: a | b) for w in cw]
                e8 = [(sc.bitcast(fx.Int32) >> 23) & 0xFF for sc in (s0, s1)]
                sw = (e8[0] << ((ln // 32) * 8)) | (e8[1] << ((ln // 32 + 2) * 8))
                sw = xred(sw, 32, lambda a, b: a | b)
                if ilive:
                    # FP4 pool (see K_PB): word w is group w // 4, dword w % 4 of its 16 bytes
                    e_i = p // CR
                    blk_i = bt_block(tt, e_i)
                    sl = e_i % K_PB
                    dbase = icb // 4 + blk_i * IC_BLK_WORDS + sl * 4
                    if ln % 8 == 0:
                        for h in range_constexpr(2):
                            w = ln // 8 + 8 * h
                            bo.buffer_store(cw[h], rsrc(i_cache), dbase + (w // 4) * IC_GRP_WORDS + w % 4)
                    if ln == 0:
                        sbase = icb // 16 + blk_i * IC_S_BLK + (sl % 16) * 4 + (sl % K_PB) // 16
                        for g in range_constexpr(IHD // 32):
                            bo.buffer_store(fx.Int8((sw >> (8 * g)) & 0xFF), rsrc(i_cache_s), sbase + g * K_PB)
                    put(mb("i_cnew"), tt * IHD + ln, bf16_round(o0))
                    put(mb("i_cnew"), tt * IHD + ln + 64, bf16_round(o1))
            stamp("i_cmp", tt, 4)

    # ========= 3b. the indexer's query: its own projection off q_a (no per-head RMS)
    if const_expr(IHD):
        r_wiqb, r_siqb = rsrc(w_i_q_b), rsrc(s_i_q_b)
        IQB_NKC = Q_LORA // 64
        IQB_R = q_b_groups(N_IQB)
        IQB_WPR = WAVES // IQB_R
        IQB_UPW = IQB_NKC // IQB_WPR
        IQB_ROWS = Q_B_TILE * IQB_R
        for t in range(start("i_q_b"), N_IQB // IQB_R, G):
            t = fx.Int32(t)
            stamp("i_q_b", t, 0)
            iqb_rg = t * IQB_R + wave // IQB_WPR

            def u_iqb(c):
                kc = (wave % IQB_WPR) * IQB_UPW + c
                return unit_fp8(r_wiqb, r_siqb, iqb_rg, kc, IQB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

            pre = [u_iqb(c) for c in range(IQB_NKC // WAVES)]
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("i_q_b", t),
            )

            def ld_iqa(sks):
                v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
                return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

            stage_x_rmsnorm(ld_iqa, Q_LORA, g_q)
            stamp("i_q_b", t, 2)
            gpu.barrier()
            acc = run_units(u_iqb, IQB_UPW, IQB_NKC // WAVES, pre)
            reduce_rows(IQB_R, acc, emit_out(IQB_ROWS))
            stamp("i_q_b", t, 3)
            gpu.barrier()
            if tid < S * IQB_ROWS:
                s = tid // IQB_ROWS
                row = t * IQB_ROWS + tid % IQB_ROWS  # row of the [IH * IHD] query
                put(mb("i_q_raw"), s * IH * IHD + row, lds_ld(outs, tid))
            stamp("i_q_b", t, 4)

        # ---- rope -> Hadamard -> FP4, one whole index head per wave
        for tt in range(start("i_q"), S * IH // IH_TASK, G):
            tt = fx.Int32(tt)
            stamp("i_q", tt, 0)
            s = tt // (IH // IH_TASK)
            ln = lane
            chs = [ln, ln + 64]
            ip = ld_pos(s)
            rc = ld_f32(rsrc(rope_cos), ip * (ROPE_DIM // 2) + ln // 2)
            rs2 = ld_f32(rsrc(rope_sin), ip * (ROPE_DIM // 2) + ln // 2)
            for k in range_constexpr(IH_TASK // WAVES):
                ihead = (tt % (IH // IH_TASK)) * IH_TASK + wave + k * WAVES
                base_i = (s * IH + ihead) * IHD
                v = getf_many([(mb("i_q_raw"), base_i + c) for c in chs])
                partner = xshfl(v[1], 1)
                even = ln % 2 == 0
                v[1] = bf16_round(even.select(v[1] * rc - partner * rs2, partner * rs2 + v[1] * rc))
                v[0] = bf16_round(v[0])
                h0, h1 = had_pair(v[0], v[1], ln)
                o0, o1 = fp4_block(bf16_round(h0))[0], fp4_block(bf16_round(h1))[0]
                put(mb("i_q"), base_i + chs[0], o0)
                put(mb("i_q"), base_i + chs[1], o1)
            stamp("i_q", tt, 4)

    # ===== 3c. weights_proj: the per-head weight of the score's head sum (bf16, no MFMA)
    if const_expr(IHD):
        for tt in range(start("i_wp"), S * IH // IH_TASK, G):
            tt = fx.Int32(tt)
            stamp("i_wp", tt, 0)
            r_iw = rsrc(i_w)
            sw = tt // (IH // IH_TASK)  # the sample
            h0 = (tt % (IH // IH_TASK)) * IH_TASK  # its first head

            def ld_hw(sks, sw=sw):
                # count=1 below: `sw` is the sample, the pair's `s` is always 0
                if const_expr(HC > 1):
                    vals = poll([(mb("xin"), (sw * HIDDEN + k) // 2, 2) for _s, k in sks])
                    res = []
                    for i in range_constexpr(len(sks)):
                        a0, a1 = bf2_f32(vals[i][0])
                        b0, b1 = bf2_f32(vals[i][1])
                        res.append([a0, a1, b0, b1])
                    return res
                res = []
                for _s, k in sks:
                    w = fx.Vector(bo.buffer_load(r_h, (sw * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            # qkv_a's normed input, recomputed rather than republished
            ks, act = _rmsnorm_tail_ks(HIDDEN)
            gs, xv = load_x_rmsnorm(ld_hw, HIDDEN, g_in, count=1)
            ssq = fx.Float32(0.0)
            for i in range_constexpr(len(ks)):
                for a in xv[i]:
                    term = a * a
                    if const_expr(act is not None and i == len(ks) - 1):
                        term = act.select(term, fx.Float32(0.0))
                    ssq = ssq + term
            rstd = rsq(block_sums([ssq])[0] * (1.0 / HIDDEN) + EPS)
            stamp("i_wp", tt, 2)
            parts = []
            for hh in range_constexpr(IH_TASK):
                acc = fx.Float32(0.0)
                for i in range_constexpr(len(ks)):
                    wv = (
                        fx.Vector(bo.buffer_load(r_iw, ((h0 + hh) * HIDDEN + ks[i]) // 2, vec_width=2, dtype=T.i32))
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    for j in range_constexpr(4):
                        term = xv[i][j] * rstd * gs[i][j] * wv[j]
                        if const_expr(act is not None and i == len(ks) - 1):
                            term = act.select(term, fx.Float32(0.0))
                        acc = acc + term
                parts.append(acc)
            tots = block_sums(parts)
            sc = float(IHD) ** -0.5 * float(IH) ** -0.5
            for hh in range_constexpr(IH_TASK):
                if tid == 0:
                    put(mb("i_wp"), sw * IH + h0 + hh, bf16_round(tots[hh]) * sc)
            stamp("i_wp", tt, 4)
    return {}


@traced
def index_select_stages(ctx):
    """3d-3e. Indexer scores over every compressed entry and the exact top-k (CSA only)."""
    CR = ctx["CR"]
    G = ctx["G"]
    IC_BLK_WORDS = ctx["IC_BLK_WORDS"]
    IC_GRP_WORDS = ctx["IC_GRP_WORDS"]
    IC_S_BLK = ctx["IC_S_BLK"]
    IH = ctx["IH"]
    IHD = ctx["IHD"]
    K_PB = ctx["K_PB"]
    N_COMP = ctx["N_COMP"]
    N_IHG = ctx["N_IHG"]
    N_INDEX = ctx["N_INDEX"]
    N_ISEL = ctx["N_ISEL"]
    S = ctx["S"]
    TK_BC = ctx["TK_BC"]
    TK_BINS = ctx["TK_BINS"]
    TK_PARTS = ctx["TK_PARTS"]
    TK_PER = ctx["TK_PER"]
    TK_REP = ctx["TK_REP"]
    TK_TRIPS = ctx["TK_TRIPS"]
    TOK = ctx["TOK"]
    _other_parts = ctx["_other_parts"]
    block_excl_scan = ctx["block_excl_scan"]
    bt_block = ctx["bt_block"]
    comp_row = ctx["comp_row"]
    get2_many = ctx["get2_many"]
    getf = ctx["getf"]
    getf_many = ctx["getf_many"]
    hist = ctx["hist"]
    i_cache = ctx["i_cache"]
    i_cache_s = ctx["i_cache_s"]
    lane = ctx["lane"]
    ld_pos = ctx["ld_pos"]
    ld_slot = ctx["ld_slot"]
    mb = ctx["mb"]
    part_keys = ctx["part_keys"]
    pl = ctx["pl"]
    poll = ctx["poll"]
    put = ctx["put"]
    red = ctx["red"]
    st_ic = ctx["st_ic"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    v4f = ctx["v4f"]
    wave = ctx["wave"]
    xs = ctx["xs"]
    if const_expr(IHD):
        # ===== 3d. score every compressed entry: score[c] = sum_h relu(q[h] . k[c]) * w[h]
        # only (sample, tile) tasks up to the batch's largest live-tile count run
        ISC_L = fx.Int32(0)
        for s_ in range_constexpr(S):
            nl = fx.min((ld_pos(s_) + 1) // CR, fx.Int32(N_COMP))
            ISC_L = fx.max(ISC_L, (nl > N_INDEX).select((nl + SCORE_TILE - 1) // SCORE_TILE, fx.Int32(0)))
        for tt in range(start("i_score"), S * ISC_L, G):
            tt = fx.Int32(tt)
            stamp("i_score", tt, 0)
            s = tt // ISC_L
            blk = tt % ISC_L
            # skip <= N_INDEX live entries (all kept, see i_topk) and tiles past the live ones
            n_live_s = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
            if (n_live_s > N_INDEX) & (blk * SCORE_TILE < n_live_s):
                # the query as f32, then as bf16 pairs at QBF (the MFMA B operand; exact)
                QBF = IH * IHD
                NQW = IH * IHD // 2  # element pairs of the query
                qws = [fx.min(tid + i * THREADS, NQW - 1) for i in range((NQW + THREADS - 1) // THREADS)]
                qv = get2_many([(mb("i_q"), s * IH * IHD + 2 * w2) for w2 in qws])
                for i in range_constexpr(len(qws)):
                    q0, q1 = qv[i]
                    lds_st(xs, 2 * qws[i], q0)  # clamped duplicates rewrite the same value
                    lds_st(xs, 2 * qws[i] + 1, q1)
                    lds_st(xs, QBF + qws[i], bf16_pair(q0, q1))
                if tid < IH:  # each head's weight polled once, read through LDS
                    lds_st(pl, tid, getf(mb("i_wp"), s * IH + tid))
                gpu.barrier()
                stamp("i_score", tt, 2)
                c = blk * SCORE_TILE + tid
                sp = ld_pos(s)
                n_live = (sp + 1) // CR
                r_ic2 = rsrc(i_cache)
                r_ics = rsrc(i_cache_s)
                icb = ld_slot(s, st_ic)  # bytes; the scale pool's base is 1/16 of it
                j_tok = s % TOK
                # MFMA: a wave's 64 entries as 4 x 16 A rows, the heads as B columns, 16 per group
                hn = lane % 16
                wcols, qbs = [], []
                for hg in range_constexpr(N_IHG):
                    hd = hg * 16 + hn  # this lane's head in group hg
                    wcol = lds_ld(pl, fx.min(hd, fx.Int32(IH - 1)))
                    wcols.append((hd < IH).select(wcol, fx.Float32(0.0)))
                    qbs.append(QBF + (fx.min(hd, fx.Int32(IH - 1)) * IHD + (lane // 16) * 8) // 2)
                for g in range_constexpr(4):
                    ec = fx.min(blk * SCORE_TILE + wave * 64 + g * 16 + hn, fx.Int32(N_COMP - 1))
                    blk_e = bt_block(s, ec)
                    sl_e = ec % K_PB
                    db = icb // 4 + blk_e * IC_BLK_WORDS + sl_e * 4 + lane // 16
                    sdb = icb // 64 + blk_e * (IC_S_BLK // 4) + sl_e % 16
                    kds = [
                        bo.buffer_load(r_ic2, db + kb * IC_GRP_WORDS, vec_width=1, dtype=T.i32)
                        for kb in range(IHD // 32)
                    ]
                    sds = [
                        bo.buffer_load(r_ics, sdb + kb * (K_PB // 4), vec_width=1, dtype=T.i32)
                        for kb in range(IHD // 32)
                    ]
                    # the keys decode once and serve every head group
                    ka = []
                    for kb in range_constexpr(IHD // 32):
                        bsc = (((fx.Int32(sds[kb]) >> ((sl_e // 16) * 8)) & 0xFF) << 23).bitcast(fx.Float32)
                        ka.append(mxfp4_to_bf16x8(fx.Int32(kds[kb]), bsc))
                    vs = [fx.Float32(0.0)] * 4
                    for hg in range_constexpr(N_IHG):
                        acc = fx.Vector.filled(4, 0.0, fx.Float32)
                        for kb in range_constexpr(IHD // 32):
                            b = fx.ptr_load(xs + (qbs[hg] + kb * 16), result_type=v4f).bitcast(fx.BFloat16)
                            acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [ka[kb], b, acc]))
                        # C[entry 4 * (lane // 16) + i][head lane % 16]: relu, weight
                        vs = [vs[i] + fx.max(acc[i], fx.Float32(0.0)) * wcols[hg] for i in range(4)]
                    # ... then sum the heads across the 16 lanes
                    for i in range_constexpr(4):
                        v = vs[i]
                        for off in (1, 2, 4, 8):
                            v = xred(v, off, lambda x, y: x + y)
                        if hn == i:
                            lds_st(red, wave * 64 + g * 16 + 4 * (lane // 16) + i, v)
                gpu.barrier()
                # entries this launch wrote may not be visible in the cache: their wave
                # rescores them from i_cnew (token s - d wrote entry (sp - d) // CR)
                for d in range_constexpr(TOK):
                    sd = s - fx.min(fx.Int32(d), j_tok)
                    pd = sp - d
                    ne = fx.max(pd, fx.Int32(0)) // CR
                    has_new = (fx.Int32(d) <= j_tok) & ((pd + 1) % CR == 0) & (blk == ne // SCORE_TILE)
                    if has_new & (wave == (ne - blk * SCORE_TILE) // 64):
                        nv = getf_many([(mb("i_cnew"), sd * IHD + lane + 64 * hf) for hf in range(IHD // 64)])
                        sc_n = fx.Float32(0.0)
                        for hh in range_constexpr(IH):
                            part = fx.Float32(0.0)
                            for hf in range_constexpr(IHD // 64):
                                part = part + nv[hf] * lds_ld(xs, hh * IHD + lane + 64 * hf)
                            sc_n = sc_n + fx.max(wave_sum(part), fx.Float32(0.0)) * lds_ld(pl, hh)
                        if lane == 0:
                            lds_st(red, ne - blk * SCORE_TILE, sc_n)
                gpu.barrier()
                sc_t = lds_ld(red, tid)
                live = (c < n_live) & (c < N_COMP)
                stamp("i_score", tt, 3)
                if c < N_COMP:
                    put(mb("i_score"), s * N_COMP + c, live.select(sc_t, fx.Float32(NEG)))
            stamp("i_score", tt, 4)

    # ============ 3e. top-k: which compressed entries the attention gathers
    # Exact radix select (8-bit digits, MSB first) over scores every rank computes
    # identically (the indexer is replicated), so every rank picks the same set.
    if const_expr(IHD):
        for tt in range(start("i_topk"), S * TK_PARTS, G):
            tt = fx.Int32(tt)
            stamp("i_topk", tt, 0)
            s = tt // TK_PARTS
            part = tt % TK_PARTS
            n_live = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
            k_want = fx.min(n_live, fx.Int32(N_INDEX))
            sbase = s * N_COMP
            # <= N_INDEX live: take all; while they fit one part, part 0 selects alone
            n_parts = (n_live <= TK_TRIPS * THREADS * TK_PER).select(fx.Int32(1), fx.Int32(TK_PARTS))
            if (n_live > N_INDEX) & (part < n_parts):
                tk_cbs, tk_keys = part_keys(sbase, part, n_live, n_parts)

                def trip_live(j):
                    """Whether trip ``j`` of this part holds any live candidate (CTA-uniform)."""
                    return (fx.Int32(j) * n_parts + part) * (THREADS * TK_PER) < n_live

                stamp("i_topk", tt, 2)

                pfx = fx.Int32(0)  # the digits already fixed, in the unsigned domain
                gt = fx.Int32(0)  # candidates ranking strictly above that prefix
                # of those (and, after the last digit, of the ties), how many earlier parts hold
                gt_b = fx.Int32(0)
                eq_b = fx.Int32(0)
                for d in range_constexpr(4):
                    sh = 24 - 8 * d
                    # the bits above this digit, folded at trace time (no 32-wide shift)
                    mk = (~((1 << (sh + 8)) - 1)) & 0xFFFFFFFF
                    hi = fx.Int32(mk - (1 << 32) if mk >= (1 << 31) else mk)
                    for z in range_constexpr(-(-(TK_BC + 4) // THREADS)):
                        zi = fx.Int32(tid) + z * THREADS
                        if zi < TK_BC + 4:
                            lds_st(hist, zi, fx.Int32(0))
                    gpu.barrier()
                    for j in range_constexpr(TK_TRIPS):
                        if trip_live(j):
                            for q in range_constexpr(TK_PER):
                                c = tk_cbs[j] + q
                                ok = c < n_live
                                u = ok.select(tk_keys[j * TK_PER + q], fx.Int32(0))
                                if ok & (((u ^ pfx) & hi) == 0):
                                    fx.atomic_add(
                                        hist + ((u >> sh) & (TK_BINS - 1)) * TK_REP + (tid & (TK_REP - 1)),
                                        fx.Int32(1),
                                        syncscope=fx.rocdl.SyncScope.Workgroup,
                                    )
                    gpu.barrier()
                    # thread t takes bin TK_BINS - 1 - t: the exclusive scan counts those above it
                    need = k_want - gt
                    cnt = fx.Int32(0)
                    bn = TK_BINS - 1 - tid
                    if tid < TK_BINS:
                        for r in range_constexpr(TK_REP):
                            cnt = cnt + lds_ld(hist, bn * TK_REP + r)
                    if const_expr(TK_PARTS > 1):
                        # the parts trade bins so all pick the same digit
                        if (tid < TK_BINS) & (n_parts > 1):
                            put(mb("tk_hist"), ((s * 4 + d) * TK_PARTS + part) * TK_BINS + bn, cnt)
                        tot = cnt
                        before = fx.Int32(0)  # this bin's count in the parts before this one
                        if (tid < TK_BINS) & (n_parts > 1):
                            vs = poll(
                                [
                                    (mb("tk_hist"), ((s * 4 + d) * TK_PARTS + pp) * TK_BINS + bn, 1)
                                    for pp in _other_parts(part)
                                ]
                            )
                            others = _other_parts(part)
                            for k in range_constexpr(TK_PARTS - 1):
                                tot = tot + vs[k][0]
                                before = before + (others[k] < part).select(vs[k][0], fx.Int32(0))
                        cnt = tot
                    above, _tot = block_excl_scan(cnt)
                    if const_expr(TK_PARTS > 1):
                        # at the chosen bin: how many candidates above it earlier parts hold
                        above_b, _tb = block_excl_scan(before)
                    if (tid < TK_BINS) & (above < need) & ((above + cnt) >= need):
                        lds_st(hist, TK_BC, bn)
                        lds_st(hist, TK_BC + 1, above)
                        if const_expr(TK_PARTS > 1):
                            lds_st(hist, TK_BC + 2, above_b)
                            lds_st(hist, TK_BC + 3, before)
                    gpu.barrier()
                    pfx = pfx | (lds_ld(hist, TK_BC) << sh)
                    gt = gt + lds_ld(hist, TK_BC + 1)
                    if const_expr(TK_PARTS > 1):
                        gt_b = gt_b + lds_ld(hist, TK_BC + 2)
                        eq_b = lds_ld(hist, TK_BC + 3)  # the last digit's is the ties'
                    gpu.barrier()
                thr = pfx ^ MIN_I32  # back to the signed-comparable domain

                # Compaction: picks above the threshold go to [gt_b, ..) of [0, gt), ties
                # to [gt + eq_b, ..) below k_want. The gather polls every i_sel slot, so
                # a radix bug that leaves one unwritten is a hang, not a wrong answer.
                stamp("i_topk", tt, 3)
                if tid == 0:
                    lds_st(hist, TK_BC, fx.Int32(0))
                    lds_st(hist, TK_BC + 1, fx.Int32(0))
                gpu.barrier()
                for j in range_constexpr(TK_TRIPS):
                    if trip_live(j):
                        for q in range_constexpr(TK_PER):
                            c = tk_cbs[j] + q
                            ok = c < n_live
                            sk = tk_keys[j * TK_PER + q] ^ MIN_I32
                            if ok & (sk > thr):
                                w = fx.Int32(
                                    fx.atomic_add(hist + TK_BC, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup)
                                )
                                put(mb("i_sel"), s * N_ISEL + gt_b + w, comp_row(s, c))
                            if ok & (sk == thr):
                                w = fx.Int32(
                                    fx.atomic_add(hist + TK_BC + 1, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup)
                                )
                                if (gt + eq_b + w) < k_want:
                                    put(mb("i_sel"), s * N_ISEL + gt + eq_b + w, comp_row(s, c))
                # the next task's first digit re-zeroes these counters
                gpu.barrier()
                # part 0 fills the tail (every part writes only below k_want)
                if part == 0:
                    for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                        o = fx.Int32(tid) + j * THREADS
                        if o < N_ISEL:
                            if o >= k_want:
                                put(mb("i_sel"), s * N_ISEL + o, fx.Int32(-1))
            else:
                if part == 0:
                    for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                        o = fx.Int32(tid) + j * THREADS
                        if o < N_ISEL:
                            row = comp_row(s, fx.max(fx.min(o, n_live - 1), fx.Int32(0)))
                            put(mb("i_sel"), s * N_ISEL + o, (o < n_live).select(row, fx.Int32(-1)))
            stamp("i_topk", tt, 4)
    return {}
