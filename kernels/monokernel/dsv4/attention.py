# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Sparse attention of the DeepSeek-V4 MonoKernel: split softmax, merge and the output projection."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.common import bf16x2_has_nan, wave_any
from kernels.monokernel.dsv4.config import EPS, ROPE_DIM
from kernels.monokernel.dsv4.plan import Q_B_TILE, ROW_TILE, SPLIT_KEYS, THREADS, WAVES, o_a_spt
from kernels.monokernel.helpers import traced
from kernels.monokernel.layout import NEG
from kernels.monokernel.ops import (
    bf2_f32,
    bf16_pair,
    exp,
    fp8_to_bf16x8,
    ld_f32,
    lds_ld,
    lds_st,
    rcp,
    rsq,
    rsrc,
    wave_max,
    wave_sum,
    xshfl,
)


@traced
def attention_stages(ctx):
    """4-7b. Query RMS + RoPE, gather-sparse split softmax, merge, o_a / o_b and the
    attention TP peer reduce."""
    CR = ctx["CR"]
    G = ctx["G"]
    H = ctx["H"]
    HC = ctx["HC"]
    HEAD_DIM = ctx["HEAD_DIM"]
    HIDDEN = ctx["HIDDEN"]
    IHD = ctx["IHD"]
    KS = ctx["KS"]
    KT_OFF = ctx["KT_OFF"]
    KV_FP8 = ctx["KV_FP8"]
    KV_ROW_BYTES = ctx["KV_ROW_BYTES"]
    LIVE_SPLITS = ctx["LIVE_SPLITS"]
    NOPE_DIM = ctx["NOPE_DIM"]
    N_ISEL = ctx["N_ISEL"]
    N_KEYS = ctx["N_KEYS"]
    N_OA = ctx["N_OA"]
    N_ROW_TILES = ctx["N_ROW_TILES"]
    N_SPLIT = ctx["N_SPLIT"]
    N_UV = ctx["N_UV"]
    OA_K = ctx["OA_K"]
    OA_PER_GROUP = ctx["OA_PER_GROUP"]
    OB_K = ctx["OB_K"]
    O_GROUPS = ctx["O_GROUPS"]
    O_LORA = ctx["O_LORA"]
    QB_PER_HEAD = ctx["QB_PER_HEAD"]
    QK_DIM = ctx["QK_DIM"]
    QS = ctx["QS"]
    S = ctx["S"]
    TOK = ctx["TOK"]
    UV_CHUNK = ctx["UV_CHUNK"]
    UV_PER_HEAD = ctx["UV_PER_HEAD"]
    UV_TILE = ctx["UV_TILE"]
    UV_WIDE = ctx["UV_WIDE"]
    attn_sink = ctx["attn_sink"]
    block_max = ctx["block_max"]
    block_sum = ctx["block_sum"]
    comp_row = ctx["comp_row"]
    emit_out = ctx["emit_out"]
    get = ctx["get"]
    get2_many = ctx["get2_many"]
    getf = ctx["getf"]
    hc_post = ctx["hc_post"]
    hc_stage_coef = ctx["hc_stage_coef"]
    hint_wait = ctx["hint_wait"]
    indices = ctx["indices"]
    keys = ctx["keys"]
    ktile = ctx["ktile"]
    kv_cache = ctx["kv_cache"]
    kv_quant = ctx["kv_quant"]
    kv_rope = ctx["kv_rope"]
    lane = ctx["lane"]
    ld_dest = ctx["ld_dest"]
    ld_pos = ctx["ld_pos"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    peer_reduce = ctx["peer_reduce"]
    pl = ctx["pl"]
    poll = ctx["poll"]
    pre_poll = ctx["pre_poll"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    r_h = ctx["r_h"]
    red = ctx["red"]
    reduce_rows = ctx["reduce_rows"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    row_rsrc = ctx["row_rsrc"]
    run_units = ctx["run_units"]
    s_o_a = ctx["s_o_a"]
    s_o_b = ctx["s_o_b"]
    scale = ctx["scale"]
    stage_x_pairs = ctx["stage_x_pairs"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_fp8 = ctx["unit_fp8"]
    v4f = ctx["v4f"]
    w_o_a = ctx["w_o_a"]
    w_o_b = ctx["w_o_b"]
    wave = ctx["wave"]
    window = ctx["window"]
    xs = ctx["xs"]
    # ============== 4. per-head query RMS (no weight) + RoPE -> bf16 query
    for tt in range(start("q_norm"), S * H, G):
        tt = fx.Int32(tt)
        stamp("q_norm", tt, 0)
        s = tt // H
        head = tt % H
        ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
        sp = ld_pos(s)
        c = ld_f32(rsrc(rope_cos), sp * (ROPE_DIM // 2) + ri)
        sn = ld_f32(rsrc(rope_sin), sp * (ROPE_DIM // 2) + ri)
        hint_wait(
            QB_PER_HEAD,
            lambda k: (mb("q_raw"), (s * H + head) * HEAD_DIM + k * Q_B_TILE + Q_B_TILE - 1),
            mark=("q_norm", tt),
        )
        qv = getf(mb("q_raw"), (s * H + head) * HEAD_DIM + fx.min(tid, HEAD_DIM - 1))
        stamp("q_norm", tt, 2)
        live = tid < HEAD_DIM
        ssq = block_sum(live.select(qv * qv, fx.Float32(0.0)))
        nv = qv * rsq(ssq * (1.0 / HEAD_DIM) + EPS)
        partner = xshfl(nv, 1)
        even = tid % 2 == 0
        rot = even.select(nv * c - partner * sn, partner * sn + nv * c)
        if const_expr(KV_FP8):
            # ATOM's fp8 attention takes the NoPE query as FP8 too
            nv = kv_quant(nv)[0]
        qn = (tid < NOPE_DIM).select(nv, rot)
        other = xshfl(qn, 1)
        if live & (tid % 2 == 0):
            put(mb("q"), ((s * H + head) * HEAD_DIM + tid) // 2, bf16_pair(qn, other))
        stamp("q_norm", tt, 4)

    # ============== 5. gather-sparse sliding-window split: 64 keys x H heads
    r_idx = rsrc(indices)
    KPW = SPLIT_KEYS // WAVES
    EPL = HEAD_DIM // 64  # KV elements one lane owns of a key's row
    WPL = EPL // 2

    def split_keys(t, s):
        """Wave 0 writes this split's 64 cache rows to LDS keys (-1 = unwritten); with an
        indexer the compressed half comes from the top-k's i_sel."""
        if wave == 0:
            k_pos = t * SPLIT_KEYS + lane
            if const_expr(LIVE_SPLITS):
                # a tile group can overrun the list; those tiles hold no keys
                k_at = s * N_KEYS + fx.min(k_pos, fx.Int32(N_KEYS - 1))
                kk = fx.Int32(bo.buffer_load(r_idx, k_at, vec_width=1, dtype=T.i32))
                lds_st(keys, lane, (t < N_SPLIT).select(kk, fx.Int32(-1)))
            elif const_expr(IHD):
                if t * SPLIT_KEYS >= window:
                    lds_st(keys, lane, get(mb("i_sel"), s * N_ISEL + k_pos - window))
                else:
                    lds_st(
                        keys,
                        lane,
                        fx.Int32(bo.buffer_load(r_idx, s * N_KEYS + k_pos, vec_width=1, dtype=T.i32)),
                    )
            else:
                lds_st(
                    keys,
                    lane,
                    fx.Int32(bo.buffer_load(r_idx, s * N_KEYS + k_pos, vec_width=1, dtype=T.i32)),
                )

    def gather_old_kv():
        """Each wave copies its KPW keys' KV rows (absolute plane rows) into the tile;
        unwritten slots (-1) are clamped to 0 here and masked in the softmax."""
        krows = [fx.max(lds_ld(keys, wave * KPW + jj), fx.Int32(0)) for jj in range(KPW)]
        for jj in range_constexpr(KPW):
            j = wave * KPW + jj
            if const_expr(KV_FP8):
                # NoPE lanes: 8 FP8 bytes x E8M0 (exact in bf16); the rest the RoPE plane
                r_row = row_rsrc(kv_cache, krows[jj], KV_ROW_BYTES)
                q8 = fx.Vector(
                    bo.buffer_load(r_row, fx.min(lane, fx.Int32(NOPE_DIM // EPL - 1)) * 2, vec_width=2, dtype=T.i32)
                )
                g = fx.min(lane, fx.Int32(NOPE_DIM // EPL - 1)) // (64 // EPL)  # this lane's 64-group
                sw = fx.Int32(bo.buffer_load(r_row, NOPE_DIM // 4 + g // 2, vec_width=1, dtype=T.i32))
                sb = (sw >> ((g % 2) * 16)) & 0xFF
                sc = (sb << 23).bitcast(fx.Float32)
                nope = (fp8_to_bf16x8(q8[0], q8[1]).to(fx.Float32) * sc).to(fx.BFloat16)
                rope = fx.Vector(
                    bo.buffer_load(
                        row_rsrc(kv_rope, krows[jj], ROPE_DIM * 2),
                        fx.max(lane - NOPE_DIM // EPL, fx.Int32(0)) * WPL,
                        vec_width=WPL,
                        dtype=T.i32,
                    )
                )
                nw = nope.bitcast(fx.Int32)
                is_n = lane < NOPE_DIM // EPL
                # ATOM leaves some rows unwritten (0xFF) or NaN; as its attention, mask
                # the key (-1) and zero it. A 0xFF scale is checked on its own (inf x FP8
                # is not NaN); one bad lane masks the whole key.
                words = [is_n.select(nw[m], rope[m]) for m in range(WPL)]
                bad = sb == fx.Int32(0xFF)
                for w_ in words:
                    bad = bad | bf16x2_has_nan(w_)
                kv8 = fx.Vector.from_elements([bad.select(fx.Int32(0), w_) for w_ in words], fx.Int32)
                if wave_any(bad) & (lane == 0):
                    lds_st(keys, j, fx.Int32(-1))
            else:
                kv8 = fx.Vector(
                    bo.buffer_load(row_rsrc(kv_cache, krows[jj], HEAD_DIM * 2), lane * WPL, vec_width=WPL, dtype=T.i32)
                )
            fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * WPL))

    def patch_new_kv(s):
        """Patch the rows this launch wrote (kvnew, and cnew on a boundary) from the
        mailboxes, for sample ``s``'s own sequence only (with TOK > 1, its run's
        tokens up to ``s``). ``kr`` is wave-uniform, so whole waves reach the polls."""
        sp = ld_pos(s)
        j_tok = s % TOK
        for d in range_constexpr(TOK):
            sd = s - fx.min(fx.Int32(d), j_tok)  # clamped onto s; masked by `inl`
            inl = fx.Int32(d) <= j_tok
            pd = sp - d
            w_row = ld_dest(0, sd)
            c_row = comp_row(sd, fx.max(pd, fx.Int32(0)) // CR) if const_expr(CR) else fx.Int32(0)
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kr = lds_ld(keys, j)
                if inl & (kr == w_row):
                    kvp = get2_many([(mb("kvnew"), sd * HEAD_DIM + lane * EPL + m * 2) for m in range(WPL)])
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                    fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * WPL))
                if const_expr(CR):
                    if inl & ((pd + 1) % CR == 0) & (kr == c_row):
                        cvp = get2_many([(mb("cnew"), sd * HEAD_DIM + lane * EPL + m * 2) for m in range(WPL)])
                        # NaN when ATOM's compressor state is: masked (see gather_old_kv)
                        words = [bf16_pair(a0, a1).bitcast(fx.Int32) for a0, a1 in cvp]
                        bad = bf16x2_has_nan(words[0])
                        for w_ in words[1:]:
                            bad = bad | bf16x2_has_nan(w_)
                        w = [bad.select(fx.Int32(0), w_).bitcast(fx.Float32) for w_ in words]
                        fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * WPL))
                        if wave_any(bad) & (lane == 0):
                            lds_st(keys, j, fx.Int32(-1))

    def live_splits(s):
        """Splits of sample ``s`` that can hold a live key: the window's plus the
        ``(pos + 1) // CR`` compressed keys so far (HCA only, else N_SPLIT)."""
        n = N_SPLIT
        if const_expr(LIVE_SPLITS):
            nc = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_KEYS - window))
            n = (window + nc + SPLIT_KEYS - 1) // SPLIT_KEYS
        return n

    # (sample, split) tasks over the batch's largest live-split count
    SPL_L = N_SPLIT
    if const_expr(LIVE_SPLITS):
        SPL_L = fx.Int32(0)
        for s_ in range_constexpr(S):
            SPL_L = fx.max(SPL_L, live_splits(s_))
    # past one round of the grid a task folds TPT tiles (flash-style) into one partial
    TPT = 1
    NG = SPL_L
    if const_expr(LIVE_SPLITS):
        TPT = fx.min((S * SPL_L + G - 1) // G, fx.Int32(N_SPLIT))
        NG = (SPL_L + TPT - 1) // TPT
    NDG = HEAD_DIM // 32 // WAVES  # 32-dim output groups a wave owns in P V
    HPW = H // WAVES  # heads a wave runs the softmax for
    FAC = H * (SPLIT_KEYS // 2)  # words of `p` past P^T: per-head rescale factors

    def load_tile(t, s):
        split_keys(t, s)
        gpu.barrier()
        gather_old_kv()

    def fold_head(carry, hg, h, m, lsum):
        """Fold one head's tile max / sum into the running pair; leave the two
        rescale factors in LDS for the P V accumulator rows of that head."""
        m_run, l_run = fx.Float32(carry[2 * hg]), fx.Float32(carry[2 * hg + 1])
        m_new = fx.max(m_run, m)
        f_old = exp(m_run - m_new)
        f_new = exp(m - m_new)
        if lane == 0:
            lds_st(pl, FAC + 2 * h, f_old)
            lds_st(pl, FAC + 2 * h + 1, f_new)
        return [m_new, l_run * f_old + lsum * f_new]

    def publish_folded(s, t, fin):
        """The folded group's partial, as tile group ``t``."""
        for hg in range_constexpr(HPW):
            if lane == 0:
                h = wave + hg * WAVES
                put(mb("sp_m"), (s * N_SPLIT + t) * H + h, fin[2 * hg])
                put(mb("sp_l"), (s * N_SPLIT + t) * H + h, fin[2 * hg + 1])
        for g in range_constexpr(NDG):
            dw = (wave * NDG + g) * 16 + lane % 16
            if lane < 16 * (H // 4):  # rows (heads) 4 * (lane // 16) + e < H
                for e in range_constexpr(4):
                    hh = (lane // 16) * 4 + e
                    o0 = fin[2 * HPW + (g * 2) * 4 + e]
                    o1 = fin[2 * HPW + (g * 2 + 1) * 4 + e]
                    put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * HEAD_DIM + dw * 2, [o0, o1])

    def run_tiles(s, t):
        """TPT tiles from t * TPT (the first already loaded), folded."""
        init = [fx.Float32(NEG), fx.Float32(0.0)] * HPW + [fx.Float32(0.0)] * (8 * NDG)
        for i_t, carry in range(0, TPT, fx.Int32(1), init=init):
            i_t = fx.Int32(i_t)
            if i_t > 0:
                gpu.barrier()  # the previous tile's P V is done with the tile
                load_tile(t * TPT + i_t, s)
            nxt = compute_tile(s, t, carry, True)
            res = yield nxt
        return [fx.Float32(v) for v in res]

    def compute_tile(s, t, carry, multi):
        """Scores, softmax and P V of the tile in LDS: folded into ``carry`` (running
        max / sum per head + P V accumulator) if ``multi``, else published as tile ``t``."""
        patch_new_kv(s)
        gpu.barrier()
        # scores = K Q^T on MFMA: keys M (4 row groups), HEAD_DIM K (two wave halves), heads N
        hn = fx.min(lane % 16, H - 1)
        rgk = wave % 4
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        for st in range_constexpr(QK_DIM // 32 // 2):
            kst = (wave // 4) * (QK_DIM // 32 // 2) + st
            key = rgk * 16 + lane % 16
            kw = KT_OFF + key * KS + kst * 16
            a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
            b = fx.ptr_load(xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
            c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        # split-local softmax: wave h, lane = key j (score = sum of the two K halves).
        ml = []
        for hg in range_constexpr(HPW):
            h = wave + hg * WAVES
            valid = lds_ld(keys, lane) >= 0
            r16 = lane % 16
            cl = h + 16 * (r16 // 4)
            raw = lds_ld(red, ((lane // 16) * 64 + cl) * 4 + r16 % 4) + lds_ld(
                red, ((lane // 16 + 4) * 64 + cl) * 4 + r16 % 4
            )
            sc_v = valid.select(raw * scale, fx.Float32(NEG))
            m = wave_max(sc_v)
            p = valid.select(exp(sc_v - m), fx.Float32(0.0))
            lsum = wave_sum(p)
            p_n = xshfl(p, 1)
            if lane % 2 == 0:  # P^T bf16 [h][64 keys] (words h * 32 + j / 2)
                lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
            # (trace-time: a conditional expression, since the tracer turns `if` into a branch)
            ml += fold_head(carry, hg, h, m, lsum) if multi else [m, lsum]
            if not multi:  # trace-time; nothing assigned here is used after it
                if lane == 0:  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
        gpu.barrier()
        # O = P V on MFMA: heads M, keys K (2 steps), dims N; V is the K tile
        ov = []
        f_o = [lds_ld(pl, FAC + 2 * ((lane // 16) * 4 + e)) for e in range(4)] if multi else None
        f_n = [lds_ld(pl, FAC + 2 * ((lane // 16) * 4 + e) + 1) for e in range(4)] if multi else None
        for g in range_constexpr(NDG):
            dw = (wave * NDG + g) * 16 + lane % 16
            c0 = fx.Vector.filled(4, 0.0, fx.Float32)
            c1 = fx.Vector.filled(4, 0.0, fx.Float32)
            for js in range_constexpr(SPLIT_KEYS // 32):
                a = fx.ptr_load(pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4), result_type=v4f).bitcast(
                    fx.BFloat16
                )
                ws = [
                    fx.ptr_load(ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)).bitcast(fx.Int32)
                    for i in range(8)
                ]
                w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)]
                b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
            if not multi:
                if lane < 16 * (H // 4):
                    for e in range_constexpr(4):
                        hh = (lane // 16) * 4 + e
                        put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * HEAD_DIM + dw * 2, [c0[e], c1[e]])
            for e in range_constexpr(4):
                k0 = 2 * HPW + (g * 2) * 4 + e
                k1 = 2 * HPW + (g * 2 + 1) * 4 + e
                ov += (
                    [
                        fx.Float32(carry[k0]) * f_o[e] + c0[e] * f_n[e],
                        fx.Float32(carry[k1]) * f_o[e] + c1[e] * f_n[e],
                    ]
                    if multi
                    else [c0[e], c1[e]]
                )
        out = list(ml)
        for g in range_constexpr(NDG):
            out += [ov[(g * 4 + e) * 2] for e in range(4)] + [ov[(g * 4 + e) * 2 + 1] for e in range(4)]
        return out

    for tt in range(start("split"), S * NG, G):
        tt = fx.Int32(tt)
        stamp("split", tt, 0)
        s = tt // NG
        t = tt % NG
        load_tile(t * TPT, s)  # before waiting for q: these rows are from earlier launches
        hint_wait(
            H,
            lambda k: (mb("q"), ((s * H + k) * HEAD_DIM + HEAD_DIM - 2) // 2),
            mark=("split", tt),
        )
        # q of all heads -> bf16 Q[h][HEAD_DIM] at words h * QS + d / 2 (padded stride)
        NQ_TOT = H * HEAD_DIM // 4
        NQ = NQ_TOT // THREADS

        def stage_q(w4, v):
            qw = (w4 // (HEAD_DIM // 4)) * QS + (w4 % (HEAD_DIM // 4)) * 2
            lds_st(xs, qw, v[0].bitcast(fx.Float32))
            lds_st(xs, qw + 1, v[1].bitcast(fx.Float32))

        qv = poll([(mb("q"), (s * H * HEAD_DIM + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ)])
        for i in range_constexpr(NQ):
            stage_q(tid + i * THREADS, qv[i])
        if const_expr(NQ_TOT % THREADS):
            w4 = tid + NQ * THREADS
            if w4 < NQ_TOT:
                stage_q(w4, poll([(mb("q"), (s * H * HEAD_DIM + w4 * 4) // 2, 2)])[0])
        stamp("split", tt, 2)
        if const_expr(LIVE_SPLITS):
            if TPT == 1:
                compute_tile(s, t, None, False)
            else:
                publish_folded(s, t, run_tiles(s, t))
        else:
            compute_tile(s, t, None, False)
        stamp("split", tt, 4)

    # ============ 6. split merge (+ attention sink) + inverse RoPE -> o
    # the sink enters the denominator only; RoPE lanes are de-rotated (V is the RoPE'd K)
    UV_PAIRS = UV_TILE // 2
    for tt in range(start("uv"), S * N_UV, G):
        tt = fx.Int32(tt)
        stamp("uv", tt, 0)
        s = tt // N_UV
        t = tt % N_UV  # UV_TILE-dim tile
        head = t // UV_PER_HEAD
        doff = (t % UV_PER_HEAD) * UV_TILE
        sink = ld_f32(rsrc(attn_sink), head)
        n_sp = (live_splits(s) + TPT - 1) // TPT  # the tile groups written: see the split stage
        hint_wait(n_sp, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head), mark=("uv", tt))
        pre_poll(n_sp, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
        dp = fx.min(tid, UV_PAIRS - 1)
        # one thread per split: a lane of wave 0, or of the block when UV_WIDE
        sp_id = tid if UV_WIDE else lane
        spi = fx.min(sp_id, n_sp - 1)  # clamped so the spare threads read a live slot
        ml = (s * N_SPLIT + spi) * H + head
        got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)], batch=2)
        # per-split weights exp(m - M) / L for this head -> misc[split]
        ok_sp = sp_id < n_sp
        m_sp = ok_sp.select(got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
        l_sp = ok_sp.select(got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
        if UV_WIDE:
            mx = block_max(m_sp)
            w_sp = exp(m_sp - mx)
            den = block_sum(l_sp * w_sp) + exp(sink - mx)
            if ok_sp:
                lds_st(misc, sp_id, w_sp * rcp(den))
        else:
            if wave == 0:
                mx = wave_max(m_sp)
                w_sp = exp(m_sp - mx)
                den = wave_sum(l_sp * w_sp) + exp(sink - mx)
                if ok_sp:
                    lds_st(misc, sp_id, w_sp * rcp(den))
        stamp("uv", tt, 2)
        gpu.barrier()
        # runtime loop over UV_CHUNK splits; past the live ones re-read the last at weight 0
        for _c, acc in range(
            0, (n_sp + UV_CHUNK - 1) // UV_CHUNK, fx.Int32(1), init=[fx.Float32(0.0), fx.Float32(0.0)]
        ):
            cb = fx.Int32(_c) * UV_CHUNK
            gc = poll(
                [
                    (
                        mb("sp_acc"),
                        ((s * N_SPLIT + (fx.min(cb + e, n_sp - 1) if const_expr(LIVE_SPLITS) else cb + e)) * H + head)
                        * (HEAD_DIM // 2)
                        + doff // 2
                        + dp,
                        1,
                    )
                    for e in range(UV_CHUNK)
                ],
                batch=UV_CHUNK,
            )
            o0 = fx.Float32(acc[0])
            o1 = fx.Float32(acc[1])
            for e in range_constexpr(UV_CHUNK):
                wj = lds_ld(misc, cb + e)
                if const_expr(LIVE_SPLITS):
                    wj = (cb + e < n_sp).select(wj, fx.Float32(0.0))
                a0, a1 = bf2_f32(gc[e][0])
                o0 = o0 + a0 * wj
                o1 = o1 + a1 * wj
            res = yield [o0, o1]
        o0 = fx.Float32(res[0])
        o1 = fx.Float32(res[1])
        # de-rotate the RoPE lanes (inverse rotation: sin negated)
        d0 = doff + dp * 2
        ri = fx.max(d0 - NOPE_DIM, fx.Int32(0)) // 2
        sp = ld_pos(s)
        c = ld_f32(rsrc(rope_cos), sp * (ROPE_DIM // 2) + ri)
        sn = ld_f32(rsrc(rope_sin), sp * (ROPE_DIM // 2) + ri)
        rot = d0 >= NOPE_DIM
        v0 = rot.select(o0 * c + o1 * sn, o0)
        v1 = rot.select(o1 * c - o0 * sn, o1)
        if tid < UV_PAIRS:
            put(mb("o"), ((s * H + head) * HEAD_DIM + doff) // 2 + dp, bf16_pair(v0, v1))
        stamp("uv", tt, 4)

    # ================= 7a. o_a: grouped low-rank output projection (group = OA_K slice)
    r_woa, r_soa = rsrc(w_o_a), rsrc(s_o_a)
    OA_NKC = OA_K // 64
    OA_R = ROW_TILE // 16
    OA_WPR = WAVES // OA_R
    OA_SPT = o_a_spt(S, O_GROUPS, O_LORA)
    for tt in range(start("o_a"), N_OA // OA_SPT, G):
        tt = fx.Int32(tt)
        stamp("o_a", tt, 0)
        s = (tt // (O_GROUPS * OA_PER_GROUP)) * OA_SPT
        t = tt % (O_GROUPS * OA_PER_GROUP)
        grp = t // OA_PER_GROUP
        # B column n holds sample s + n (columns past OA_SPT repeat the last)
        oa_col = fx.min(lane % 16, OA_SPT - 1)

        def u_oa(c):
            kc = (wave % OA_WPR) * (OA_NKC // OA_WPR) + c
            return unit_fp8(
                r_woa, r_soa, t * OA_R + wave // OA_WPR, kc, OA_NKC, OA_K, 128, (oa_col * OA_K + kc * 64) // 2
            )

        pre = [u_oa(c) for c in range(OA_NKC // OA_WPR)]
        hint_wait(
            H // O_GROUPS,
            lambda k: (mb("o"), ((s * H + grp * (H // O_GROUPS) + k) * HEAD_DIM + HEAD_DIM - 2) // 2),
            mark=("o_a", tt),
        )
        stage_x_pairs("o", OA_SPT * OA_K, lambda k: ((s + k // OA_K) * H) * HEAD_DIM + grp * OA_K + k % OA_K)
        stamp("o_a", tt, 2)
        gpu.barrier()
        acc = run_units(u_oa, OA_NKC // OA_WPR, OA_NKC // OA_WPR, pre)
        reduce_rows(OA_R, acc, emit_out(ROW_TILE))
        stamp("o_a", tt, 3)
        gpu.barrier()
        if tid < OA_SPT * ROW_TILE // 4:
            n = tid // (ROW_TILE // 4)
            r = tid % (ROW_TILE // 4) * 4
            put_bf(
                mb("o_lora"),
                (s + n) * OB_K + t * ROW_TILE + r,
                [lds_ld(outs, n * ROW_TILE + r + j) for j in range(4)],
            )
        stamp("o_a", tt, 4)

    # ============= 7b. o_b + attention TP peer reduce + residual -> a
    r_wob, r_sob = rsrc(w_o_b), rsrc(s_o_b)
    OB_NKC = OB_K // 64
    OB_R = ROW_TILE // 16
    OB_WPR = WAVES // OB_R
    for t in range(start("o_b"), N_ROW_TILES, G):
        t = fx.Int32(t)
        stamp("o_b", t, 0)

        def u_ob(c):
            kc = (wave % OB_WPR) * (OB_NKC // OB_WPR) + c
            return unit_fp8(
                r_wob, r_sob, t * OB_R + wave // OB_WPR, kc, OB_NKC, OB_K, 128, (n_sel() * OB_K + kc * 64) // 2
            )

        pre = [u_ob(c) for c in range(OB_NKC // OB_WPR)]
        hint_wait(
            S * O_GROUPS * OA_PER_GROUP,
            lambda k: (
                mb("o_lora"),
                (k // (O_GROUPS * OA_PER_GROUP)) * OB_K + (k % (O_GROUPS * OA_PER_GROUP)) * ROW_TILE + ROW_TILE - 1,
            ),
            mark=("o_b", t),
        )
        stage_x_pairs("o_lora", S * OB_K, lambda k: k)
        stamp("o_b", t, 2)
        gpu.barrier()
        acc = run_units(u_ob, OB_NKC // OB_WPR, OB_NKC // OB_WPR, pre)
        reduce_rows(OB_R, acc, emit_out(ROW_TILE))
        stamp("o_b", t, 3)
        gpu.barrier()

        def resid_h(s, row):
            w = fx.Vector.from_elements(
                [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
            )
            v = w.bitcast(fx.BFloat16).to(fx.Float32)
            return v[0], v[1]

        if const_expr(HC > 1):
            hc_stage_coef(0)
            peer_reduce(
                "attn",
                t,
                None,  # hc_post owns the combination
                lambda s, row, v0, v1: hc_post(
                    s,
                    row,
                    v0,
                    v1,
                    lambda s_, j, r_: fx.Int32(
                        bo.buffer_load(r_h, ((s_ * HC + j) * HIDDEN + r_) // 2, vec_width=1, dtype=T.i32)
                    ),
                    lambda s_, k, r_, o0, o1: put_bf(mb("a"), (s_ * HC + k) * HIDDEN + r_, [o0, o1]),
                ),
            )
        else:
            peer_reduce(
                "attn",
                t,
                resid_h,
                lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            )
        stamp("o_b", t, 4)
    return {}
