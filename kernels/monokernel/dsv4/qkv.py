# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Attention input projections of the DeepSeek-V4 MonoKernel: q_a / kv, the KV ring and
compressor, and q_b."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.common import pow2_ceil
from kernels.monokernel.dsv4.config import EPS, FP8_MAX, ROPE_DIM
from kernels.monokernel.dsv4.plan import Q_B_TILE, QKV_A_TILE, WAVES, q_b_groups, qkv_a_groups
from kernels.monokernel.helpers import traced
from kernels.monokernel.layout import NEG
from kernels.monokernel.ops import (
    bf2_f32,
    bf16_round,
    exp,
    ld_bf16,
    ld_f32,
    lds_ld,
    lds_st,
    rcp,
    rsq,
    rsrc,
    wave_max,
    xred,
    xshfl,
)


@traced
def qkv_stages(ctx):
    """1-2b. q_a / kv GEMV, KV RMSNorm + RoPE -> sliding-window ring, and the KV compressor."""
    CMP_CHUNK = ctx["CMP_CHUNK"]
    CMP_CHUNK_T = ctx["CMP_CHUNK_T"]
    CR = ctx["CR"]
    CW = ctx["CW"]
    C_COFF = ctx["C_COFF"]
    C_RING = ctx["C_RING"]
    C_ROWS = ctx["C_ROWS"]
    G = ctx["G"]
    HC = ctx["HC"]
    HEAD_DIM = ctx["HEAD_DIM"]
    HIDDEN = ctx["HIDDEN"]
    IW = ctx["IW"]
    KV_FP8 = ctx["KV_FP8"]
    KV_ROW_BYTES = ctx["KV_ROW_BYTES"]
    NOPE_DIM = ctx["NOPE_DIM"]
    N_QKV_A = ctx["N_QKV_A"]
    N_QKV_C = ctx["N_QKV_C"]
    OVERLAP = ctx["OVERLAP"]
    QKV_A_ROWS = ctx["QKV_A_ROWS"]
    Q_LORA = ctx["Q_LORA"]
    S = ctx["S"]
    TOK = ctx["TOK"]
    ape = ctx["ape"]
    block_sum = ctx["block_sum"]
    block_sums = ctx["block_sums"]
    comp_row = ctx["comp_row"]
    emit_out = ctx["emit_out"]
    g_ckv = ctx["g_ckv"]
    g_in = ctx["g_in"]
    g_kv = ctx["g_kv"]
    getf = ctx["getf"]
    getf_many = ctx["getf_many"]
    hint_wait = ctx["hint_wait"]
    kv_cache = ctx["kv_cache"]
    kv_rope = ctx["kv_rope"]
    kv_state = ctx["kv_state"]
    lane = ctx["lane"]
    ld_dest = ctx["ld_dest"]
    ld_pos = ctx["ld_pos"]
    load_x_rmsnorm = ctx["load_x_rmsnorm"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    r_h = ctx["r_h"]
    reduce_rows = ctx["reduce_rows"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    row_rsrc = ctx["row_rsrc"]
    run_units = ctx["run_units"]
    s_qkv_a = ctx["s_qkv_a"]
    score_state = ctx["score_state"]
    slot_rsrc = ctx["slot_rsrc"]
    st_kv = ctx["st_kv"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_fp8 = ctx["unit_fp8"]
    w_qkv_a = ctx["w_qkv_a"]
    w_qkv_c = ctx["w_qkv_c"]
    wave = ctx["wave"]
    # ================================================= 1. q_a / kv GEMV
    r_wqa, r_sqa = rsrc(w_qkv_a), rsrc(s_qkv_a)
    r_wqc = rsrc(w_qkv_c)
    QA_NKC = HIDDEN // 64

    def qkv_stage(name, n_tiles, row0, bf16_w):
        """One fused-GEMV stage over n_tiles 16-row groups from global row row0: the
        FP8 q_a | kv rows (qkv_a) or the compressors' BF16 rows (qkv_c)."""
        QA_R = qkv_a_groups(n_tiles)
        QA_WPR = WAVES // QA_R
        QA_UPW = QA_NKC // QA_WPR
        # units in flight per wave (larger BF16 batches spill VGPRs)
        QA_BATCH = max(1, QA_NKC // WAVES // (4 if bf16_w else 1))
        QA_ROWS = QKV_A_TILE * QA_R
        for t in range(start(name), n_tiles // QA_R, G):
            t = fx.Int32(t)
            stamp(name, t, 0)
            qa_rg = t * QA_R + wave // QA_WPR

            def u_qa(c):
                kc = (wave % QA_WPR) * QA_UPW + c
                if const_expr(bf16_w):
                    return unit_bf16(r_wqc, qa_rg, kc, QA_NKC, (n_sel() * HIDDEN + kc * 64) // 2)
                return unit_fp8(r_wqa, r_sqa, qa_rg, kc, QA_NKC, HIDDEN, 128, (n_sel() * HIDDEN + kc * 64) // 2)

            def ld_h(sks):
                if const_expr(HC > 1):  # hc_pre already contracted the streams
                    vals = poll([(mb("xin"), (s * HIDDEN + k) // 2, 2) for s, k in sks])
                    res = []
                    for i in range_constexpr(len(sks)):
                        a0, a1 = bf2_f32(vals[i][0])
                        b0, b1 = bf2_f32(vals[i][1])
                        res.append([a0, a1, b0, b1])
                    return res
                res = []
                for s, k in sks:
                    w = fx.Vector(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            # input loads go out before the weight stream (loads complete in order)
            h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in)
            pre = [u_qa(c) for c in range(QA_BATCH)]
            stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld)
            gpu.barrier()
            stamp(name, t, 2)
            acc = run_units(u_qa, QA_UPW, QA_BATCH, pre)
            reduce_rows(QA_R, acc, emit_out(QA_ROWS))
            stamp(name, t, 3)
            gpu.barrier()
            if tid < S * QA_ROWS:
                s = tid // QA_ROWS
                row = row0 + t * QA_ROWS + tid % QA_ROWS
                v = lds_ld(outs, tid)
                # column ranges of the fused GEMV (reference.qkv_a_split())
                if row < Q_LORA:
                    put(mb("q_a"), s * Q_LORA + row, v)
                elif row < Q_LORA + HEAD_DIM:
                    put(mb("kv_a"), s * HEAD_DIM + row - Q_LORA, v)
                elif row < Q_LORA + HEAD_DIM + CW:
                    put(mb("c_kv"), s * CW + row - Q_LORA - HEAD_DIM, v)
                elif row < Q_LORA + HEAD_DIM + 2 * CW:
                    put(mb("c_gate"), s * CW + row - Q_LORA - HEAD_DIM - CW, v)
                elif row < Q_LORA + HEAD_DIM + 2 * CW + IW:
                    put(mb("i_kv"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW, v)
                else:
                    put(mb("i_gate"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW - IW, v)
            stamp(name, t, 4)

    qkv_stage("qkv_a", N_QKV_A, 0, False)
    if const_expr(N_QKV_C):
        qkv_stage("qkv_c", N_QKV_C, QKV_A_ROWS, True)

    def kv_quant(nv):
        """This thread's KV channel through the NoPE FP8 round trip (one 64-group per
        wave, ue8m0 scale 2**ceil(log2(amax / 448)) as ATOM) -> (value, FP8 byte, E8M0 byte)."""
        amax = wave_max(fmath.absf(nv))
        sc = pow2_ceil(fx.max(amax, fx.Float32(FP8_MAX * 2.0**-126)) * (1.0 / FP8_MAX))
        q = fx.min(fx.max(nv * rcp(sc), -FP8_MAX), FP8_MAX)  # rcp is exact on a power of two
        word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
        v2 = fx.Vector.make_type(2, fx.Float32)
        d = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
        return d * sc, word & 0xFF, (sc.bitcast(fx.Int32) >> 23) & 0xFF

    def put_kv_row(row, kvn, byte, e8):
        """Write one KV row (thread = channel; ``kvn`` its value, ``byte`` / ``e8``
        from kv_quant). CTA-uniform: the fp8 layout trades scale bytes through LDS."""
        if const_expr(KV_FP8):
            r_nope, r_rope = row_rsrc(kv_cache, row, KV_ROW_BYTES), row_rsrc(kv_rope, row, ROPE_DIM * 2)
            # four lanes' FP8 bytes to one dword, channel tid in byte tid % 4
            wb = byte << ((tid % 4) * 8)
            for off in (1, 2):
                wb = xred(wb, off, lambda a, b: a | b)
            if (tid < NOPE_DIM) & (tid % 4 == 0):
                bo.buffer_store(wb, r_nope, tid // 4)
            if tid >= NOPE_DIM:
                bo.buffer_store(kvn.to(fx.BFloat16), r_rope, tid - NOPE_DIM)
            if (lane == 0) & (tid < NOPE_DIM):
                lds_st(misc, wave, e8.bitcast(fx.Float32))
            gpu.barrier()
            # scale dword k: groups 2k and 2k + 1, each byte twice
            NG = NOPE_DIM // 64
            if tid < (NG + 1) // 2:
                lo = lds_ld(misc, fx.min(2 * tid, fx.Int32(NG - 1))).bitcast(fx.Int32)
                hi = (2 * tid + 1 < NG).select(
                    lds_ld(misc, fx.min(2 * tid + 1, fx.Int32(NG - 1))).bitcast(fx.Int32), fx.Int32(0)
                )
                sw = lo | (lo << 8) | (hi << 16) | (hi << 24)
                bo.buffer_store(sw, r_nope, NOPE_DIM // 4 + tid)
            gpu.barrier()
        else:
            if tid < HEAD_DIM:
                bo.buffer_store(kvn.to(fx.BFloat16), row_rsrc(kv_cache, row, HEAD_DIM * 2), tid)

    # ====== 2. KV RMSNorm + RoPE + FP8 round trip -> sliding-window ring cache
    # the NOPE_DIM lanes are FP8 round-tripped in 64-blocks, matching the checkpoint's QAT
    for t in range(start("cache"), 1, G):
        stamp("cache", t, 0)
        g = ld_bf16(rsrc(g_kv), fx.min(tid, HEAD_DIM - 1))
        ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
        ps = [ld_pos(sx) for sx in range(S)]
        cs = [ld_f32(rsrc(rope_cos), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)]
        sns = [ld_f32(rsrc(rope_sin), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)]
        hint_wait(
            HEAD_DIM // QKV_A_TILE,
            lambda k: (mb("kv_a"), (S - 1) * HEAD_DIM + k * QKV_A_TILE + QKV_A_TILE - 1),
            mark=("cache", t),
        )
        vs = getf_many([(mb("kv_a"), s * HEAD_DIM + fx.min(tid, HEAD_DIM - 1)) for s in range(S)])
        stamp("cache", t, 2)
        live = tid < HEAD_DIM
        ssq = block_sums([live.select(v * v, fx.Float32(0.0)) for v in vs])
        for s in range_constexpr(S):
            nv = vs[s] * rsq(ssq[s] * (1.0 / HEAD_DIM) + EPS) * g
            # rope tail: lane ^ 1 is the other half of this interleaved (2i, 2i+1) pair
            partner = xshfl(nv, 1)
            even = tid % 2 == 0
            rot = even.select(nv * cs[s] - partner * sns[s], partner * sns[s] + nv * cs[s])
            dq, byte, e8 = kv_quant(nv)
            kvn = bf16_round((tid < NOPE_DIM).select(dq, rot))
            put_kv_row(ld_dest(0, s), kvn, byte, e8)
            if live:
                put(mb("kvnew"), s * HEAD_DIM + tid, kvn)
        stamp("cache", t, 4)

    # ============= 2b. KV compressor (HCA): rolling state, pooled on the boundary
    # each CR window's last token emits a per-channel softmax pool, normed / RoPE'd / FP8'd
    def window_pool(p, j_tok, ring, launch, own):
        """Online softmax over the compressor window ending at p -> (den, num). Element i
        comes from the ring, this token (``own``), or, for a token d back in this launch,
        its mailbox (``launch``): its state write may not be visible yet."""

        def fold(acc, svs, kvs):
            m, den, num = acc
            for e in range_constexpr(len(svs)):
                m_new = fx.max(m, svs[e])
                rescale = exp(m - m_new)
                w = exp(svs[e] - m_new)
                den = den * rescale + w
                num = num * rescale + w * kvs[e]
                m = m_new
            return [m, den, num]

        n_ring = C_ROWS if const_expr(TOK == 1) else C_ROWS - TOK
        chunk = CMP_CHUNK if const_expr(TOK == 1) else CMP_CHUNK_T
        for _i, acc in range(0, n_ring // chunk, fx.Int32(1), init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)]):
            ib = fx.Int32(_i) * chunk
            lds_ = [ring(ib + e) for e in range(chunk)]
            res = yield fold(
                [fx.Float32(acc[0]), fx.Float32(acc[1]), fx.Float32(acc[2])],
                [a for a, _ in lds_],
                [b for _, b in lds_],
            )
        acc = [fx.Float32(res[0]), fx.Float32(res[1]), fx.Float32(res[2])]
        if const_expr(TOK > 1):
            svs, kvs = [], []
            for i in range_constexpr(C_ROWS - TOK, C_ROWS):
                d = C_ROWS - 1 - i
                if const_expr(d == 0):
                    sv, kv = own(i)
                else:
                    r_sv, r_kv = ring(fx.Int32(i))
                    l_sv, l_kv = launch(d, i)
                    inl = fx.Int32(d) <= j_tok
                    sv, kv = inl.select(l_sv, r_sv), inl.select(l_kv, r_kv)
                svs.append(sv)
                kvs.append(kv)
            acc = fold(acc, svs, kvs)
        return acc[1], acc[2]

    def ring_row(p, i):
        """State-ring row of window element i (position p + 1 - C_ROWS + i)."""
        return (p + 1 + i + (C_RING - C_ROWS)) % C_RING

    if const_expr(CR):
        for tt in range(start("cmp"), S, G):
            tt = fx.Int32(tt)
            stamp("cmp", tt, 0)
            ch = fx.min(tid, HEAD_DIM - 1)
            live = tid < HEAD_DIM
            # tt is the sample; per-sample state keeps these unordered tasks from racing
            p = ld_pos(tt)
            rs_kv, rs_sc, sb = slot_rsrc(kv_state, tt, st_kv), slot_rsrc(score_state, tt, st_kv), 0
            slot = p % CR
            ap0 = [ld_f32(rsrc(ape), slot * CW + j * HEAD_DIM + ch) for j in range(C_COFF)]
            g = ld_bf16(rsrc(g_ckv), ch)
            # the window's first position (clamped: loaded unconditionally)
            anchor = fx.max(p + 1 - CR, fx.Int32(0))
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            # one rope table per layer: a compressing layer's is on compress_rope_theta for all it rotates
            rc = ld_f32(rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ri)
            rs = ld_f32(rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ri)
            kvv = [getf(mb("c_kv"), (tt * C_COFF + j) * HEAD_DIM + ch) for j in range(C_COFF)]
            gtv = [getf(mb("c_gate"), (tt * C_COFF + j) * HEAD_DIM + ch) for j in range(C_COFF)]
            stamp("cmp", tt, 2)
            if live:
                for j in range_constexpr(C_COFF):
                    w = sb + (p % C_RING) * CW + j * HEAD_DIM + ch
                    bo.buffer_store(kvv[j], rs_kv, w)
                    bo.buffer_store(gtv[j] + ap0[j], rs_sc, w)
            if (p + 1) % CR == 0:  # uniform across the CTA
                j_tok = tt % TOK

                # overlap: previous window's rows from their first half, current from the second
                def c_half(i):
                    return (i >= CR).select(fx.Int32(1), fx.Int32(0)) if OVERLAP else 0

                def c_ring(i):
                    wi = sb + ring_row(p, i) * CW + c_half(i) * HEAD_DIM + ch
                    return ld_f32(rs_sc, wi), ld_f32(rs_kv, wi)

                def c_own(i):
                    h = 1 if (OVERLAP and i >= CR) else 0
                    return gtv[h] + ap0[h], kvv[h]

                def c_launch(d, i):
                    h = 1 if (OVERLAP and i >= CR) else 0
                    sm = tt - fx.min(fx.Int32(d), j_tok)
                    q = fx.max(p - d, fx.Int32(0))
                    apq = ld_f32(rsrc(ape), (q % CR) * CW + h * HEAD_DIM + ch)
                    return (
                        getf(mb("c_gate"), (sm * C_COFF + h) * HEAD_DIM + ch) + apq,
                        getf(mb("c_kv"), (sm * C_COFF + h) * HEAD_DIM + ch),
                    )

                den_, num_ = window_pool(p, j_tok, c_ring, c_launch, c_own)
                pooled = num_ * rcp(den_)
                pooled = bf16_round(pooled)
                ssq = block_sum(live.select(pooled * pooled, fx.Float32(0.0)))
                # the model's norm returns bf16
                nv = bf16_round(pooled * rsq(ssq * (1.0 / HEAD_DIM) + EPS) * g)
                partner = xshfl(nv, 1)
                even = tid % 2 == 0
                rot = even.select(nv * rc - partner * rs, partner * rs + nv * rc)
                dq, byte, e8 = kv_quant(nv)
                cv = bf16_round((tid < NOPE_DIM).select(dq, rot))
                put_kv_row(comp_row(tt, p // CR), cv, byte, e8)
                if live:
                    put(mb("cnew"), tt * HEAD_DIM + tid, cv)
            stamp("cmp", tt, 4)
    return dict(kv_quant=kv_quant, ring_row=ring_row, window_pool=window_pool)


@traced
def q_b_stage(ctx):
    """3. q_a RMSNorm -> q_b (raw f32 query)."""
    G = ctx["G"]
    H = ctx["H"]
    HEAD_DIM = ctx["HEAD_DIM"]
    N_QB = ctx["N_QB"]
    Q_LORA = ctx["Q_LORA"]
    S = ctx["S"]
    emit_out = ctx["emit_out"]
    g_q = ctx["g_q"]
    get2_many = ctx["get2_many"]
    hint_wait = ctx["hint_wait"]
    mb = ctx["mb"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    put = ctx["put"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    s_q_b = ctx["s_q_b"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_fp8 = ctx["unit_fp8"]
    w_q_b = ctx["w_q_b"]
    wave = ctx["wave"]
    # ==================================== 3. q_a RMSNorm -> q_b (raw f32 query)
    r_wqb, r_sqb = rsrc(w_q_b), rsrc(s_q_b)
    QB_NKC = Q_LORA // 64
    QB_R = q_b_groups(N_QB)
    QB_WPR = WAVES // QB_R
    QB_UPW = QB_NKC // QB_WPR
    QB_BATCH = QB_NKC // WAVES
    QB_ROWS = Q_B_TILE * QB_R
    for t in range(start("q_b"), N_QB // QB_R, G):
        t = fx.Int32(t)
        stamp("q_b", t, 0)
        qb_rg = t * QB_R + wave // QB_WPR

        def u_qb(c):
            kc = (wave % QB_WPR) * QB_UPW + c
            return unit_fp8(r_wqb, r_sqb, qb_rg, kc, QB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

        pre = [u_qb(c) for c in range(QB_BATCH)]
        hint_wait(
            Q_LORA // QKV_A_TILE,
            lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
            mark=("q_b", t),
        )

        def ld_qa(sks):
            v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
            return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

        stage_x_rmsnorm(ld_qa, Q_LORA, g_q)
        stamp("q_b", t, 2)
        gpu.barrier()
        acc = run_units(u_qb, QB_UPW, QB_BATCH, pre)
        reduce_rows(QB_R, acc, emit_out(QB_ROWS))
        stamp("q_b", t, 3)
        gpu.barrier()
        # f32: the per-head RMS sees the unrounded output; bf16 rounding comes after RoPE
        if tid < S * QB_ROWS:
            s = tid // QB_ROWS
            row = t * QB_ROWS + tid % QB_ROWS
            put(mb("q_raw"), (s * H + row // HEAD_DIM) * HEAD_DIM + row % HEAD_DIM, lds_ld(outs, tid))
        stamp("q_b", t, 4)
    return {}
