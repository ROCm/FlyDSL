# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Hyper-connection mixers of the DeepSeek-V4 MonoKernel."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.config import EPS
from kernels.monokernel.dsv4.kernel.plan import ROW_TILE
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import bf2_f32, bf16_round, exp, ld_f32, lds_ld, lds_st, rcp, rsq, rsrc, xred


@traced
def hc_attn_stages(ctx):
    """0. Hyper-connection pre-mix on the attention side; returns the mixers the FFN
    side reuses."""
    G = ctx["G"]
    HC = ctx["HC"]
    HC_COEF = ctx["HC_COEF"]
    HC_COL_OFFS = ctx["HC_COL_OFFS"]
    HC_KSLICE = ctx["HC_KSLICE"]
    HC_MISC = ctx["HC_MISC"]
    HC_MIX = ctx["HC_MIX"]
    HC_NKC = ctx["HC_NKC"]
    HC_NKC_FULL = ctx["HC_NKC_FULL"]
    HC_PW = ctx["HC_PW"]
    HC_RG = ctx["HC_RG"]
    HC_ROWS = ctx["HC_ROWS"]
    HC_ROW_OFFS = ctx["HC_ROW_OFFS"]
    HC_TASKS = ctx["HC_TASKS"]
    HC_TPW = ctx["HC_TPW"]
    HC_VALS = ctx["HC_VALS"]
    HC_WPR = ctx["HC_WPR"]
    HIDDEN = ctx["HIDDEN"]
    N_ROW_TILES = ctx["N_ROW_TILES"]
    S = ctx["S"]
    block_sum = ctx["block_sum"]
    emit_out = ctx["emit_out"]
    getf = ctx["getf"]
    hc_attn_fn = ctx["hc_attn_fn"]
    hc_attn_sb = ctx["hc_attn_sb"]
    hc_eps = ctx["hc_eps"]
    hc_sinkhorn_iters = ctx["hc_sinkhorn_iters"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    r_h = ctx["r_h"]
    red = ctx["red"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    wave = ctx["wave"]
    xs = ctx["xs"]

    # ======== 0. hyper-connection pre-mix: partial dots + partial sum of squares
    def hc_stage_coef(sd):
        """Every sample's pre | post | comb into LDS."""
        if tid < S * HC_COEF:
            s_ = tid // HC_COEF
            lds_st(misc, HC_MISC + tid, getf(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + tid % HC_COEF))
        gpu.barrier()

    def hc_post(s, row, v0, v1, res_word, emit):
        """out[k] = post[k] * x + sum_j comb[j, k] * residual[j], for a row pair.
        ``v0``/``v1`` are rounded to bf16 first, as the model's all-reduce returns bf16."""
        v0 = bf16_round(v0)
        v1 = bf16_round(v1)
        rj = [bf2_f32(res_word(s, j, row)) for j in range_constexpr(HC)]
        for k in range_constexpr(HC):
            pk = lds_ld(misc, HC_MISC + s * HC_COEF + HC + k)
            o0 = pk * v0
            o1 = pk * v1
            for j in range_constexpr(HC):
                cjk = lds_ld(misc, HC_MISC + s * HC_COEF + 2 * HC + j * HC + k)
                o0 = o0 + cjk * rj[j][0]
                o1 = o1 + cjk * rj[j][1]
            emit(s, k, row, o0, o1)

    def hc_pre_stages(side, sd, fn_ptr, sb_ptr, src_word, out_name, contract=True, src_pair=None):
        """hcd (the mixing projection's partials) and, with ``contract``, hcc (the
        contracted input). Returns the coefficient routine. ``src_pair(s, k)``: the
        source's (mailbox, pair), so hcc polls a row's streams in one batch."""
        r_fn = rsrc(fn_ptr)
        r_sb = rsrc(sb_ptr)  # [3 scales | HC_MIX bases]
        for tt in range(start(f"hcd_{side}"), S * HC_TASKS, G):
            tt = fx.Int32(tt)
            stamp(f"hcd_{side}", tt, 0)
            s = tt // HC_TASKS
            t = tt % HC_TASKS
            w = src_word(s, t * HC_KSLICE + tid * 2)
            lds_st(xs, tid, w.bitcast(fx.Float32))
            x0, x1 = bf2_f32(w)
            ssq = block_sum(x0 * x0 + x1 * x1)
            stamp(f"hcd_{side}", tt, 2)
            gpu.barrier()

            # the fp32 mixer is packed as bf16 hi / lo (pack_hc_fn), both into one tile
            HC_UPW = HC_NKC // HC_WPR

            def u_hc(c, t=t):
                lo = 1 if c >= HC_UPW else 0
                kc = (wave % HC_WPR) * HC_UPW + c % HC_UPW
                return unit_bf16(r_fn, wave // HC_WPR + lo * HC_RG, t * HC_NKC + kc, HC_NKC_FULL, kc * 32)

            acc = run_units(u_hc, 2 * HC_UPW, 2 * HC_UPW)
            reduce_rows(HC_RG, acc, emit_out(HC_ROWS))
            stamp(f"hcd_{side}", tt, 3)
            gpu.barrier()
            base_i = ((s * 2 + sd) * HC_TASKS + t) * HC_VALS
            if tid < HC_ROWS:
                put(mb("hc_d"), base_i + tid, lds_ld(outs, tid))
            if tid == HC_ROWS:
                put(mb("hc_d"), base_i + HC_ROWS, ssq)
            stamp(f"hcd_{side}", tt, 4)

        def hc_coefficients(sd, publish):
            """Reduce the hcd partials, take the RMS scale, run the Sinkhorn; redone
            per hcc task (no extra stage), ``publish`` puts them for hc_post."""
            sc0 = ld_f32(r_sb, 0)
            sc1 = ld_f32(r_sb, 1)
            sc2 = ld_f32(r_sb, 2)
            # HC_PW waves poll one batch each; wave 0 adds them in wave order (rank-identical)
            NBLK = (S * HC_VALS + 63) // 64
            for blk in range_constexpr(NBLK):
                idx = lane + blk * 64
                if (wave < HC_PW) & (idx < S * HC_VALS):
                    s_ = idx // HC_VALS
                    j = idx % HC_VALS
                    tasks = [fx.min(wave * HC_TPW + i, HC_TASKS - 1) for i in range(HC_TPW)]
                    parts = poll([(mb("hc_d"), ((s_ * 2 + sd) * HC_TASKS + ti) * HC_VALS + j, 1) for ti in tasks])
                    tot = fx.Float32(0.0)
                    for i in range_constexpr(HC_TPW):
                        ok = (wave * HC_TPW + i) < HC_TASKS
                        tot = tot + ok.select(parts[i][0].bitcast(fx.Float32), fx.Float32(0.0))
                    lds_st(red, (wave * NBLK + blk) * 64 + lane, tot)
            gpu.barrier()
            for blk in range_constexpr(NBLK):
                idx = lane + blk * 64
                if (wave == 0) & (idx < S * HC_VALS):
                    tot = fx.Float32(0.0)
                    for w_ in range_constexpr(HC_PW):
                        tot = tot + lds_ld(red, (w_ * NBLK + blk) * 64 + lane)
                    lds_st(red, idx, tot)
            gpu.barrier()
            # one wave per sample: the samples' Sinkhorn chains run in parallel
            if wave < S:
                s_ = wave
                rstd = rsq(lds_ld(red, s_ * HC_VALS + HC_ROWS) * (1.0 / (HC * HIDDEN)) + EPS)

                def coef(i, s_=s_, rstd=rstd):
                    return lds_ld(red, s_ * HC_VALS + i) * rstd

                if lane < 2 * HC:  # pre then post share these lanes
                    m = coef(fx.min(lane, fx.Int32(HC_MIX - 1)))
                    b = ld_f32(r_sb, 3 + lane)
                    v = (lane < HC).select(
                        rcp(1.0 + exp(-(m * sc0 + b))) + hc_eps,
                        2.0 * rcp(1.0 + exp(-(m * sc1 + b))),
                    )
                    lds_st(misc, HC_MISC + s_ * HC_COEF + lane, v)
                    if publish:
                        put(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + lane, v)
                if lane < HC * HC:
                    cb = coef(2 * HC + lane) * sc2 + ld_f32(r_sb, 3 + 2 * HC + lane)
                    rmax = cb
                    for off in HC_ROW_OFFS:
                        rmax = xred(rmax, off, fx.max)
                    c = exp(cb - rmax)
                    rsum = c
                    for off in HC_ROW_OFFS:
                        rsum = xred(rsum, off, lambda a, b: a + b)
                    c = c * rcp(rsum) + hc_eps
                    csum = c
                    for off in HC_COL_OFFS:
                        csum = xred(csum, off, lambda a, b: a + b)
                    c = c * rcp(csum + hc_eps)
                    for _ in range_constexpr(hc_sinkhorn_iters - 1):
                        rsum = c
                        for off in HC_ROW_OFFS:
                            rsum = xred(rsum, off, lambda a, b: a + b)
                        c = c * rcp(rsum + hc_eps)
                        csum = c
                        for off in HC_COL_OFFS:
                            csum = xred(csum, off, lambda a, b: a + b)
                        c = c * rcp(csum + hc_eps)
                    lds_st(misc, HC_MISC + s_ * HC_COEF + 2 * HC + lane, c)
                    if publish:
                        put(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + 2 * HC + lane, c)
            gpu.barrier()

        if const_expr(not contract):
            return hc_coefficients

        # --- contract the hc_mult streams by `pre` into the single-width input
        for t in range(start(f"hcc_{side}"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp(f"hcc_{side}", t, 0)
            hc_coefficients(sd, t == 0)
            stamp(f"hcc_{side}", t, 2)
            if tid < S * ROW_TILE // 2:
                s_ = tid // (ROW_TILE // 2)
                r = (tid % (ROW_TILE // 2)) * 2
                row = t * ROW_TILE + r
                a0 = fx.Float32(0.0)
                a1 = fx.Float32(0.0)
                if const_expr(src_pair is not None):
                    ws = [v[0] for v in poll([src_pair(s_, j * HIDDEN + row) + (1,) for j in range(HC)])]
                else:
                    ws = [src_word(s_, j * HIDDEN + row) for j in range(HC)]
                for j in range_constexpr(HC):
                    pj = lds_ld(misc, HC_MISC + s_ * HC_COEF + j)
                    x0, x1 = bf2_f32(ws[j])
                    a0 = a0 + pj * x0
                    a1 = a1 + pj * x1
                put_bf(mb(out_name), s_ * HIDDEN + row, [a0, a1])
            stamp(f"hcc_{side}", t, 4)
        return hc_coefficients

    if const_expr(HC > 1):
        hc_pre_stages(
            "a",
            0,
            hc_attn_fn,
            hc_attn_sb,
            lambda s, k: fx.Int32(bo.buffer_load(r_h, (s * HC * HIDDEN + k) // 2, vec_width=1, dtype=T.i32)),
            "xin",
        )
    return dict(hc_post=hc_post, hc_pre_stages=hc_pre_stages, hc_stage_coef=hc_stage_coef)
