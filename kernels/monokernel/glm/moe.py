# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Router, shared/routed gate-up, SiLU, expert down and TP residual stages."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import EPS, HIDDEN, INTER, MOE_SLOTS, N_EXPERTS, SCALE_BM, SHARED_EXPERT, TOP_K
from kernels.monokernel.glm.layout import (
    BLOCKS,
    N_ROUTER,
    N_ROW_TILES,
    NATIVE_UG_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    WAVES,
    XQ_BLOCKS,
    XQ_GROUPS,
    XQ_WAVES,
    down_prefetch_batch,
    sample_wave_batches,
    ug_split,
)
from kernels.monokernel.layout import CM_SYS, THREADS, atom_mxfp4_scale_index
from kernels.monokernel.ops import (
    bpermute_i32,
)
from kernels.monokernel.ops import (
    exp as _exp,
)
from kernels.monokernel.ops import (
    rcp as _rcp,
)
from kernels.monokernel.ops import (
    rsq as _rsq,
)
from kernels.monokernel.ops import (
    rsrc as _rsrc,
)
from kernels.monokernel.ops import (
    uniform as _uniform,
)
from kernels.monokernel.ops import (
    uniform_f32 as _uniform_f32,
)


@flyc.jit
def emit_router(plan, state, tools):
    quant_moe = tools.quant_moe
    DN_UNION_CANDIDATE = plan.DN_UNION_CANDIDATE
    G = plan.G
    S = plan.S
    index_request_width = plan.index_request_width
    indexer_cp = plan.indexer_cp
    native_fp4_mfma = plan.native_fp4_mfma
    g_post = state.g_post
    lane = state.lane
    red = state.red
    tid = state.tid
    w_r = state.w_r
    wave = state.wave
    ld_bf16 = tools.ld_bf16
    lds_ld = tools.lds_ld
    mb = tools.mb
    put = tools.put
    bf2_f32 = tools.bf2_f32
    poll = tools.poll
    hint_wait = tools.hint_wait
    unit_bf16 = tools.unit_bf16
    run_units = tools.run_units
    stage_x_rmsnorm = tools.stage_x_rmsnorm
    publish_moe_quant = tools.publish_moe_quant
    moe_roundtrip = tools.moe_roundtrip
    start = tools.start

    if const_expr(DN_UNION_CANDIDATE and index_request_width == 6 and not indexer_cp):
        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # Two samples per CTA, retaining the original per-sample K-fold and rounding.
        # Reuse weights across two live MFMA columns; 192 tasks retain the original CTA base.
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), (S // 2) * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            router_sample = (tt // N_ROUTER) * 2

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.min(lane % ROUTER_TILE, 1)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(
                    r_wr,
                    t * ROUTER_TILE // 16,
                    kc,
                    R_NKC,
                    (r_ns * HIDDEN + kc * 64) // 2,
                    r_ln,
                )

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (
                    mb("a"),
                    router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                ),
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
            r_gp = _rsrc(g_post)
            x_blk = (wave % XQ_WAVES) * N_ROUTER + t
            x_local_sample = fx.min(wave // XQ_WAVES, 1)
            x_s = router_sample + x_local_sample
            x_ok = (wave < 2 * XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [(mb("a"), ((router_sample + s) * HIDDEN + k) // 2, 2) for s, k in sks]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, count=2)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = (x_local_sample == 0).select(rstds[0], rstds[1])
                a0, a1 = xa[0]
                q0, q1, qs = quant_moe(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                publish_moe_quant(mb("xq"), x_s * HIDDEN + xk, q0, q1)
                d0, d1 = moe_roundtrip(q0, q1)
                qscale = (qs << 23).bitcast(fx.Float32) if const_expr(native_fp4_mfma) else qs
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                    _rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
                if const_expr(native_fp4_mfma):
                    if lane % 16 == 0:
                        put(
                            mb("xqs"),
                            x_s * XQ_GROUPS + x_blk * 4 + lane // 16,
                            qs,
                        )
                else:
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < 2 * ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = tid // ROUTER_TILE
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(
                            red,
                            (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4,
                        )
                put(
                    mb("scores"),
                    (router_sample + n) * N_EXPERTS + t * ROUTER_TILE + r,
                    _rcp(1.0 + _exp(-logit)),
                )

    else:
        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K.
        # S > 1 gets S times as many independent router CTAs instead of serializing
        # every sample's normalization and output columns inside one CTA.
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt if const_expr(S == 1) else tt % N_ROUTER
            router_sample = fx.Int32(0) if const_expr(S == 1) else tt // N_ROUTER

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.Int32(0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(
                    r_wr,
                    t * ROUTER_TILE // 16,
                    kc,
                    R_NKC,
                    (r_ns * HIDDEN + kc * 64) // 2,
                    r_ln,
                )

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (
                    mb("a"),
                    router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                ),
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
            r_gp = _rsrc(g_post)
            x_blk = wave * N_ROUTER + t
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [(mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, count=1)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = rstds[0]
                a0, a1 = xa[0]
                q0, q1, qs = quant_moe(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                publish_moe_quant(mb("xq"), x_s * HIDDEN + xk, q0, q1)
                d0, d1 = moe_roundtrip(q0, q1)
                qscale = (qs << 23).bitcast(fx.Float32) if const_expr(native_fp4_mfma) else qs
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                    _rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
                if const_expr(native_fp4_mfma):
                    if lane % 16 == 0:
                        put(
                            mb("xqs"),
                            x_s * XQ_GROUPS + x_blk * 4 + lane // 16,
                            qs,
                        )
                else:
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = fx.Int32(0)
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(
                            red,
                            (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4,
                        )
                put(
                    mb("scores"),
                    router_sample * N_EXPERTS + t * ROUTER_TILE + r,
                    _rcp(1.0 + _exp(-logit)),
                )


@flyc.jit
def emit_moe_up(plan, state, tools):
    COHORT_ROW_TILES = plan.COHORT_ROW_TILES
    DN_UNION_ALIGNED = plan.DN_UNION_ALIGNED
    DN_UNION_CANDIDATE = plan.DN_UNION_CANDIDATE
    DN_UNION_ROW10 = plan.DN_UNION_ROW10
    G = plan.G
    GLOBAL_UG = plan.GLOBAL_UG
    GLOBAL_UG_WIDTH = plan.GLOBAL_UG_WIDTH
    GROUPED_UG = plan.GROUPED_UG
    MIDQ_WORDS = plan.MIDQ_WORDS
    ROW_TILES = plan.ROW_TILES
    S = plan.S
    SC = plan.SC
    TILE_SCRATCH_BYTES = plan.TILE_SCRATCH_BYTES
    UG_SCALE_ROW = plan.UG_SCALE_ROW
    UG_TASK_ROUNDS = plan.UG_TASK_ROUNDS
    UG_UNION_MAX = plan.UG_UNION_MAX
    UG_X_ROW = plan.UG_X_ROW
    atom_experts = plan.atom_experts
    expert_inter = plan.expert_inter
    expert_mxfp4 = plan.expert_mxfp4
    native_fp4_mfma = plan.native_fp4_mfma
    paired_ug_rows = plan.paired_ug_rows
    total_samples = plan.total_samples
    bid = state.bid
    dnw = state.dnw
    full_bid = state.full_bid
    full_scratch = state.full_scratch
    g_post = state.g_post
    keys = state.keys
    lane = state.lane
    misc = state.misc
    outs = state.outs
    red = state.red
    row_tile = state.row_tile
    s_ug = state.s_ug
    tid = state.tid
    w_ug = state.w_ug
    wave = state.wave
    xs = state.xs
    ld_f32 = tools.ld_f32
    ld_bf16 = tools.ld_bf16
    lds_ld = tools.lds_ld
    lds_st = tools.lds_st
    mb = tools.mb
    put = tools.put
    put2 = tools.put2
    bf2_f32 = tools.bf2_f32
    poll = tools.poll
    hint_wait = tools.hint_wait
    get = tools.get
    get2 = tools.get2
    block_sum = tools.block_sum
    unit_mxfp4 = tools.unit_mxfp4
    unit_native_mxfp4 = tools.unit_native_mxfp4
    mma_units = tools.mma_units
    reduce_rows = tools.reduce_rows
    emit_out = tools.emit_out
    quant_scaled = tools.quant_scaled
    quant_mxfp8 = tools.quant_mxfp8
    publish_moe_quant = tools.publish_moe_quant
    stage_xq = tools.stage_xq
    st_f8 = tools.st_f8
    st_mxfp8 = tools.st_mxfp8
    load_bias = tools.load_bias
    route_top8 = tools.route_top8
    start = tools.start
    n_sel = tools.n_sel
    dn_route = tools.dn_route

    # ================================ 9. expert up/gate + SiLU
    # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
    UG_NKC = HIDDEN // 64
    UG_W_BYTES = 2 * expert_inter * HIDDEN // (2 if expert_mxfp4 else 1)
    UG_S_BYTES = (
        2 * expert_inter * (HIDDEN // 32) if expert_mxfp4 else 2 * expert_inter // SCALE_BM * (HIDDEN // 128) * 4
    )

    if const_expr(S == 1):
        # Each CTA handles one eight-intermediate tile per task round;
        # the first routed slot also computes the shared expert. TP4 has
        # two rounds, TP8 one. All eight waves split the K dimension.
        UG8 = NATIVE_UG_TILE
        UG8_CPW = UG_NKC // WAVES
        for u in range(start("ug"), expert_inter, G):
            u = fx.Int32(u)
            s_u, c = fx.Int32(0), u % (expert_inter // UG8)
            has_sh = u < expert_inter // UG8
            slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))
            # the FP8 activation is computed here from the post-attention state (in
            # parallel with the router): RMSNorm, then per-128 quant with one wave per
            # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
            NB = XQ_BLOCKS // WAVES
            ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
            r_gp = _rsrc(g_post)
            gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]  # issued ahead of the wait
            bs = load_bias()
            w_rg = (
                c if paired_ug_rows else ((lane % 16) // 8) * (expert_inter // 16) + c // 2
            )  # MFMA rows 0-7 gate, 8-15 up
            w_ln = lane if paired_ug_rows else (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (expert_inter // 16) + c // 2  # this lane's output rows

            def u_ug8(cc, e, live=None):  # expert e's weights (loads return 0 unless live)
                nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                r_wug = _rsrc(
                    w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                    nw,
                )
                r_sug = _rsrc(
                    s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                    ns,
                )
                unit = wave * (UG8_CPW // 2) + cc
                kc = unit * 2
                if const_expr(expert_mxfp4):
                    if const_expr(native_fp4_mfma):
                        return unit_native_mxfp4(
                            r_wug,
                            r_sug,
                            w_rg,
                            unit,
                            HIDDEN,
                            unit * 32,
                            8 + unit * 4 + lane // 16,
                            ln=w_ln,
                        )
                    return unit_mxfp4(
                        r_wug,
                        r_sug,
                        w_rg,
                        unit,
                        HIDDEN,
                        unit * 32,
                        lambda: _uniform_f32(lds_ld(misc, 8 + unit)),
                        w_ln,
                    )
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            r_wug,
                            ((w_rg * UG_NKC + kc + h) * 64 + w_ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for h in range(2)
                ]
                sc = ld_f32(r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)
                return (
                    "f8f8",
                    wv,
                    lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                    kc * 16 + (lane // 16) * 4,
                )

            # the shared expert's weights do not depend on routing: prefetch them (the
            # later zero-weight MMAs of the other tasks are cheaper than a branch)
            pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_CPW // 2)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
            )
            # the sum of squares takes the router's element partition and order
            # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
            NQ4 = HIDDEN // (4 * THREADS)
            got = poll(
                [(mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ4)]
                + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
            )
            av = [bf2_f32(w[0]) for w in got[NQ4:]]
            ss = fx.Float32(0.0)
            for w in got[:NQ4]:
                for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                    ss = ss + a * a
            rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
            for j in range_constexpr(NB):
                if const_expr(native_fp4_mfma):
                    q0, q1, qs = quant_mxfp8(
                        av[j][0] * rstd * gps[j][0],
                        av[j][1] * rstd * gps[j][1],
                    )
                    st_mxfp8(ks_[j], q0, q1)
                    if lane % 16 == 0:
                        lds_st(
                            misc,
                            8 + (wave + j * WAVES) * 4 + lane // 16,
                            qs.bitcast(fx.Float32),
                        )
                else:
                    q0, q1, qs = quant_scaled(
                        av[j][0] * rstd * gps[j][0],
                        av[j][1] * rstd * gps[j][1],
                    )
                    st_f8(ks_[j], q0, q1)
                    if lane == 0:
                        lds_st(misc, 8 + wave + j * WAVES, qs)
            if wave == 0:
                e, w = route_top8(s_u, bs=bs)
                if lane == slot - 1:
                    lds_st(keys, 0, e)
                    lds_st(misc, 0, w)
            gpu.barrier()
            e_sel = _uniform(lds_ld(keys, 0))
            post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
            reduce_rows(
                1,
                mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                emit_out(16),
            )
            gpu.barrier()
            reduce_rows(
                1,
                mma_units([fx.Float32(0.0) for _ in range(4)], post),
                lambda rl, n, v: lds_st(outs, 16 + rl, v),
            )
            gpu.barrier()
            if tid < UG8:  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                r = (tid % (UG8 // 2)) * 2
                o = (tid // (UG8 // 2)) * 16
                g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                if has_sh | (tid >= UG8 // 2):
                    put2(
                        mb("mid"),
                        (tid < UG8 // 2).select(fx.Int32(0), slot) * expert_inter + c * UG8 + r,
                        g0 * _rcp(1.0 + _exp(-g0)) * u0,
                        g1 * _rcp(1.0 + _exp(-g1)) * u1,
                    )
            if (c == 0) & (tid == 0):  # routing record (debug / tests)
                put(mb("sel"), slot, e_sel)
                put(mb("prob"), slot, lds_ld(misc, 0))
                if has_sh:
                    put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                    put(mb("prob"), 0, fx.Float32(1.0))
            if const_expr(expert_inter > G):
                gpu.barrier()
    elif const_expr(S > 1):
        # One eight-intermediate tile per CTA and task round. Shared-expert weights
        # feed all sample columns of one MFMA, while routed-expert weights are
        # prefetched one sample ahead.  This avoids the segmented partial
        # tiles and mailbox reduction used by the older S=2/4 schedule.
        UG8 = NATIVE_UG_TILE
        UG8_UNITS = (HIDDEN // 128) // WAVES
        XW = UG_X_ROW

        def ug8_units(e, sample, c, live=None):
            w_rg = c if paired_ug_rows else ((lane % 16) // 8) * (expert_inter // 16) + c // 2
            w_ln = lane if paired_ug_rows else (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (expert_inter // 16) + c // 2
            if const_expr(live is None):
                rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
            else:
                rw = _rsrc(
                    w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                    live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                )
                rs = _rsrc(
                    s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                    live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                )
            sn = n_sel() if sample is None else fx.Int32(sample)
            units = []
            for cc in range_constexpr(UG8_UNITS):
                unit = wave * UG8_UNITS + cc
                kc = unit * 2
                if const_expr(expert_mxfp4):
                    if const_expr(native_fp4_mfma):
                        units.append(
                            unit_native_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XW + unit * 32,
                                8 + sn * UG_SCALE_ROW + unit * 4 + lane // 16,
                                ln=w_ln,
                            )
                        )
                    else:
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XW + unit * 32,
                                lambda unit=unit, sn=sn: lds_ld(misc, 8 + sn * XQ_BLOCKS + unit),
                                w_ln,
                            )
                        )
                    continue
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            rw,
                            ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for j in range(2)
                ]
                sc = ld_f32(rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)
                units.append(
                    (
                        "f8f8",
                        wv,
                        lambda sc=sc, kc=kc, sn=sn: (sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kc // 2)),
                        sn * XW + kc * 16 + (lane // 16) * 4,
                    )
                )
            return units

        dn_route(load_bias())
        gpu.barrier()
        stage_ug_first = atom_experts or (expert_mxfp4 and S > 4)
        if const_expr(stage_ug_first and not GLOBAL_UG):
            stage_xq(list(range(S)))
            gpu.barrier()
        if const_expr(GLOBAL_UG):
            # Publish the original route once per cohort before global work.
            if (bid == 0) & (tid < S * MOE_SLOTS):
                put(mb("sel"), tid, lds_ld(keys, tid))
                put(mb("prob"), tid, lds_ld(dnw, tid))
            gpu.barrier()
            if tid < total_samples * MOE_SLOTS:
                gu_row = tid // MOE_SLOTS
                gu_local = (gu_row % S) * MOE_SLOTS + tid % MOE_SLOTS
                gu_scratch = full_scratch + fx.Int64(gu_row // S) * TILE_SCRATCH_BYTES
                gu_route = get(gu_scratch + fx.Int64(SC["sel"]), gu_local)
                gu_prob = get(gu_scratch + fx.Int64(SC["prob"]), gu_local)
                lds_st(keys, tid, gu_route)
                lds_st(dnw, tid, gu_prob.bitcast(fx.Float32))
            gpu.barrier()
            gu_experts = keys + total_samples * MOE_SLOTS
            gu_lo = gu_experts + N_EXPERTS + 1
            gu_hi = gu_lo + N_EXPERTS + 1
            gu_counts = gu_hi + N_EXPERTS + 1
            gu_selected = gu_counts + N_EXPERTS // 64 + 1
            gu_red = fx.recast_iter(fx.Int32, red)
            for gu_i in range_constexpr(2 * WAVES * N_EXPERTS // THREADS):
                lds_st(gu_red, tid + gu_i * THREADS, fx.Int32(0))
            gpu.barrier()
            for gu_batch in range_constexpr((total_samples + WAVES - 1) // WAVES):
                gu_sample = wave + gu_batch * WAVES
                if (gu_sample < total_samples) & (lane < TOP_K):
                    gu_expert = fx.Int32(lds_ld(keys, gu_sample * MOE_SLOTS + lane + 1))
                    gu_address = (gu_sample // 32) * WAVES * N_EXPERTS + wave * N_EXPERTS + gu_expert
                    lds_st(gu_red, gu_address, lds_ld(gu_red, gu_address) | (fx.Int32(1) << (gu_sample % 32)))
                gpu.barrier()
            gu_mask_lo, gu_mask_hi = fx.Int32(0), fx.Int32(0)
            if tid < N_EXPERTS:
                for gu_w in range_constexpr(WAVES):
                    gu_mask_lo = gu_mask_lo | lds_ld(gu_red, gu_w * N_EXPERTS + tid)
                    gu_mask_hi = gu_mask_hi | lds_ld(gu_red, (WAVES + gu_w) * N_EXPERTS + tid)
            gu_ballot = fx.Int64(rocdl.ballot(T.i64, (gu_mask_lo | gu_mask_hi) != 0))
            gu_below = fx.Int32(fmath.ctpop(gu_ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
            gu_count = fx.Int32(fmath.ctpop(gu_ballot))
            if (tid < N_EXPERTS) & (lane == 0):
                lds_st(gu_counts, wave, gu_count)
            gpu.barrier()
            gu_offset = fx.Int32(0)
            for gu_w in range_constexpr(N_EXPERTS // 64):
                gu_offset = gu_offset + (wave > gu_w).select(fx.Int32(lds_ld(gu_counts, gu_w)), fx.Int32(0))
            if (tid < N_EXPERTS) & ((gu_mask_lo | gu_mask_hi) != 0):
                lds_st(gu_experts, gu_offset + gu_below, tid)
                lds_st(gu_lo, gu_offset + gu_below, gu_mask_lo)
                lds_st(gu_hi, gu_offset + gu_below, gu_mask_hi)
            if tid == N_EXPERTS - 1:
                gu_total = gu_offset + gu_count
                lds_st(gu_experts, gu_total, fx.Int32(SHARED_EXPERT))
                lds_st(gu_lo, gu_total, fx.Int32(-1 if total_samples >= 32 else (1 << total_samples) - 1))
                lds_st(
                    gu_hi,
                    gu_total,
                    fx.Int32(-1 if total_samples >= 64 else (1 << max(0, total_samples - 32)) - 1),
                )
                lds_st(gu_counts, N_EXPERTS // 64, gu_total + 1)
            gpu.barrier()
            gu_nunion = _uniform(lds_ld(gu_counts, N_EXPERTS // 64))
            # One global task contains one <=S-token chunk, including shared.
            # The retired reduction arena holds only prefix-scan totals here.
            gu_chunk_map = gu_selected + total_samples
            gu_prefix_index = fx.min(tid, gu_nunion - 1)
            gu_prefix_bits = fx.Int64(fx.Uint32(lds_ld(gu_lo, gu_prefix_index))) | (
                fx.Int64(fx.Uint32(lds_ld(gu_hi, gu_prefix_index))) << 32
            )
            gu_chunk_count = (tid < gu_nunion).select((fx.Int32(fmath.ctpop(gu_prefix_bits)) + S - 1) // S, fx.Int32(0))
            gu_chunk_scan = gu_chunk_count
            for gu_shift in range_constexpr(6):
                gu_distance = 1 << gu_shift
                gu_previous = bpermute_i32(fx.max(lane - gu_distance, fx.Int32(0)) * 4, gu_chunk_scan)
                gu_chunk_scan = gu_chunk_scan + (lane >= gu_distance).select(gu_previous, fx.Int32(0))
            if lane == 63:
                lds_st(gu_red, wave, gu_chunk_scan)
            gpu.barrier()
            gu_chunk_offset = gu_chunk_scan - gu_chunk_count
            for gu_scan_wave in range_constexpr(WAVES):
                gu_chunk_offset = gu_chunk_offset + (wave > gu_scan_wave).select(
                    lds_ld(gu_red, gu_scan_wave), fx.Int32(0)
                )
            if tid == THREADS - 1:
                lds_st(gu_red, WAVES, gu_chunk_offset + gu_chunk_count)
            for gu_chunk_id in range_constexpr((total_samples + S - 1) // S):
                if gu_chunk_id < gu_chunk_count:
                    lds_st(gu_chunk_map, gu_chunk_offset + gu_chunk_id, tid * ROW_TILES + gu_chunk_id)
            gpu.barrier()
            gu_nchunks = _uniform(lds_ld(gu_red, WAVES))
            gu_tiles = expert_inter // UG8
            for gu_task in range(full_bid, gu_nchunks * (gu_tiles // (GLOBAL_UG_WIDTH // 8)), BLOCKS):
                gu_task = fx.Int32(gu_task)
                gu_chunk_record = _uniform(lds_ld(gu_chunk_map, gu_task // (gu_tiles // (GLOBAL_UG_WIDTH // 8))))
                gu_union = gu_chunk_record // ROW_TILES
                gu_segment = gu_chunk_record % ROW_TILES
                gu_cbase = (gu_task % (gu_tiles // (GLOBAL_UG_WIDTH // 8))) * (GLOBAL_UG_WIDTH // 8)
                gu_expert = _uniform(lds_ld(gu_experts, gu_union))
                gu_bits = fx.Int64(fx.Uint32(_uniform(lds_ld(gu_lo, gu_union)))) | (
                    fx.Int64(fx.Uint32(_uniform(lds_ld(gu_hi, gu_union)))) << 32
                )
                gu_nselected = fx.Int32(fmath.ctpop(gu_bits))
                if tid < total_samples:
                    gu_bit = fx.Int64(1) << fx.Int64(tid)
                    if (gu_bits & gu_bit) != 0:
                        gu_index = fx.Int32(fmath.ctpop(gu_bits & (gu_bit - 1)))
                        lds_st(gu_selected, gu_index, tid)
                gpu.barrier()
                gu_valid = fx.min(fx.Int32(S), gu_nselected - gu_segment * S)
                if gu_valid < S:
                    # Every valid MFMA B column has its own token payload.
                    # Unused columns never publish mid values. Give them
                    # FP8 zero and neutral E8M0 scales, without rereading
                    # the last valid token for every unused column.
                    for gu_j in range(gu_valid):
                        gu_j = fx.Int32(gu_j)
                        gu_xtoken = _uniform(lds_ld(gu_selected, gu_segment * S + gu_j))
                        gu_xbase = full_scratch + fx.Int64(gu_xtoken // S) * TILE_SCRATCH_BYTES
                        gu_live_got = poll(
                            [
                                (
                                    gu_xbase + fx.Int64(SC["xq"]),
                                    (gu_xtoken % S) * (HIDDEN // 4) + tid + gu_i * THREADS,
                                    1,
                                )
                                for gu_i in range(HIDDEN // 4 // THREADS)
                            ]
                            + [
                                (
                                    gu_xbase + fx.Int64(SC["xqs"]),
                                    (gu_xtoken % S) * XQ_GROUPS + fx.min(tid, XQ_GROUPS - 1),
                                    1,
                                )
                            ]
                        )
                        for gu_i in range_constexpr(HIDDEN // 4 // THREADS):
                            lds_st(
                                xs,
                                gu_j * UG_X_ROW + tid + gu_i * THREADS,
                                gu_live_got[gu_i][0].bitcast(fx.Float32),
                            )
                        if tid < XQ_GROUPS:
                            lds_st(
                                misc,
                                8 + gu_j * UG_SCALE_ROW + tid,
                                gu_live_got[HIDDEN // 4 // THREADS][0].bitcast(fx.Float32),
                            )
                    for gu_j in range_constexpr(S):
                        if gu_j >= gu_valid:
                            for gu_i in range_constexpr(HIDDEN // 4 // THREADS):
                                lds_st(xs, gu_j * UG_X_ROW + tid + gu_i * THREADS, fx.Float32(0.0))
                            if tid < XQ_GROUPS:
                                lds_st(misc, 8 + gu_j * UG_SCALE_ROW + tid, fx.Int32(127).bitcast(fx.Float32))
                else:
                    gu_tokens = [
                        _uniform(lds_ld(gu_selected, gu_segment * S + fx.min(fx.Int32(gu_col_j), gu_valid - 1)))
                        for gu_col_j in range(S)
                    ]
                    gu_bases = [full_scratch + fx.Int64(gu_xtoken // S) * TILE_SCRATCH_BYTES for gu_xtoken in gu_tokens]
                    gu_nxw = HIDDEN // 4 // THREADS
                    gu_got = poll(
                        [
                            (
                                gu_xbase + fx.Int64(SC["xq"]),
                                (gu_xtoken % S) * (HIDDEN // 4) + tid + gu_load_i * THREADS,
                                1,
                            )
                            for gu_xtoken, gu_xbase in zip(gu_tokens, gu_bases)
                            for gu_load_i in range(gu_nxw)
                        ]
                        + [
                            (
                                gu_xbase + fx.Int64(SC["xqs"]),
                                (gu_xtoken % S) * XQ_GROUPS + fx.min(tid, XQ_GROUPS - 1),
                                1,
                            )
                            for gu_xtoken, gu_xbase in zip(gu_tokens, gu_bases)
                        ]
                    )
                    for gu_j in range_constexpr(S):
                        for gu_i in range_constexpr(gu_nxw):
                            lds_st(
                                xs,
                                gu_j * UG_X_ROW + tid + gu_i * THREADS,
                                gu_got[gu_j * gu_nxw + gu_i][0].bitcast(fx.Float32),
                            )
                        if tid < XQ_GROUPS:
                            lds_st(
                                misc,
                                8 + gu_j * UG_SCALE_ROW + tid,
                                gu_got[S * gu_nxw + gu_j][0].bitcast(fx.Float32),
                            )
                # All valid operands and defined padding are ready before MFMA.
                gpu.barrier()
                # Reuse selected activations across the configured output tile.
                for gu_pair in range_constexpr(GLOBAL_UG_WIDTH // 16):
                    gu_units = [ug8_units(gu_expert, None, gu_cbase + gu_pair * 2 + gu_part) for gu_part in range(2)]
                    for gu_part in range_constexpr(2):
                        gu_acc = mma_units([fx.Float32(0.0) for _ in range(4)], gu_units[gu_part])
                        fx.ptr_store(
                            fx.Vector.from_elements(gu_acc, fx.Float32),
                            red + ((gu_part * WAVES + wave) * 64 + lane) * 4,
                        )
                    gpu.barrier()
                    if tid < S * 32:
                        gu_part = tid // (S * 16)
                        gu_row = tid % (S * 16)
                        gu_n, gu_r = gu_row // 16, gu_row % 16
                        gu_sum = fx.Float32(0.0)
                        for gu_w in range_constexpr(WAVES):
                            gu_sum = gu_sum + lds_ld(
                                red,
                                ((gu_part * WAVES + gu_w) * 64 + gu_n + 16 * (gu_r // 4)) * 4 + gu_r % 4,
                            )
                        lds_st(outs, tid, gu_sum)
                    gpu.barrier()
                    if tid < gu_valid * UG8:
                        gu_sample = tid // UG8
                        gu_element = (tid % UG8) * 2
                        gu_part, gu_r = gu_element // UG8, gu_element % UG8
                        gu_token = fx.Int32(lds_ld(gu_selected, gu_segment * S + gu_sample))
                        gu_slot = fx.Int32(0)
                        for gu_j in range_constexpr(1, MOE_SLOTS):
                            gu_slot = (fx.Int32(lds_ld(keys, gu_token * MOE_SLOTS + gu_j)) == gu_expert).select(
                                fx.Int32(gu_j), gu_slot
                            )
                        gu_o = gu_part * (S * 16) + gu_sample * 16 + gu_r
                        gu_g0, gu_g1 = lds_ld(outs, gu_o), lds_ld(outs, gu_o + 1)
                        gu_v0, gu_v1 = lds_ld(outs, gu_o + UG8), lds_ld(outs, gu_o + UG8 + 1)
                        gu_dst = full_scratch + fx.Int64(gu_token // S) * TILE_SCRATCH_BYTES + fx.Int64(SC["mid"])
                        put2(
                            gu_dst,
                            ((gu_token % S) * MOE_SLOTS + gu_slot) * expert_inter
                            + (gu_cbase + gu_pair * 2 + gu_part) * UG8
                            + gu_r,
                            gu_g0 * _rcp(1.0 + _exp(-gu_g0)) * gu_v0,
                            gu_g1 * _rcp(1.0 + _exp(-gu_g1)) * gu_v1,
                        )
                    gpu.barrier()
            # Restore this CTA's original cohort-local route layout for down.
            gu_my_key = lds_ld(keys, fx.min(tid, S * MOE_SLOTS - 1) + row_tile * S * MOE_SLOTS)
            gu_my_prob = lds_ld(dnw, fx.min(tid, S * MOE_SLOTS - 1) + row_tile * S * MOE_SLOTS)
            gpu.barrier()
            if tid < S * MOE_SLOTS:
                lds_st(keys, tid, gu_my_key)
                lds_st(dnw, tid, gu_my_prob)
            gpu.barrier()
            if const_expr(DN_UNION_ROW10 or DN_UNION_CANDIDATE or DN_UNION_ALIGNED):
                # Rebuild cohort-local expert metadata expected by ordered down union;
                # wave-local masks only occupy the otherwise-idle reduction workspace.
                union_experts = keys + S * MOE_SLOTS
                union_masks = union_experts + UG_UNION_MAX
                union_counts = union_masks + UG_UNION_MAX
                mask_red = fx.recast_iter(fx.Int32, red)
                for i in range_constexpr(WAVES * N_EXPERTS // THREADS):
                    lds_st(mask_red, tid + i * THREADS, fx.Int32(0))
                gpu.barrier()
                for batch in range_constexpr(sample_wave_batches(S)):
                    sample = wave + batch * WAVES
                    if (sample < S) & (lane < TOP_K):
                        expert = fx.Int32(lds_ld(keys, sample * MOE_SLOTS + lane + 1))
                        address = wave * N_EXPERTS + expert
                        previous = lds_ld(mask_red, address)
                        lds_st(mask_red, address, previous | (fx.Int32(1) << sample))
                    gpu.barrier()
                mask = fx.Int32(0)
                if tid < N_EXPERTS:
                    for w in range_constexpr(WAVES):
                        mask = mask | lds_ld(mask_red, w * N_EXPERTS + tid)
                ballot = fx.Int64(rocdl.ballot(T.i64, mask != 0))
                below = fx.Int32(fmath.ctpop(ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
                count = fx.Int32(fmath.ctpop(ballot))
                if (tid < N_EXPERTS) & (lane == 0):
                    lds_st(union_counts, wave, count)
                gpu.barrier()
                offset = fx.Int32(0)
                for w in range_constexpr(N_EXPERTS // 64):
                    offset = offset + (wave > w).select(fx.Int32(lds_ld(union_counts, w)), fx.Int32(0))
                if (tid < N_EXPERTS) & (mask != 0):
                    lds_st(union_experts, offset + below, tid)
                    lds_st(union_masks, offset + below, mask)
                if tid == N_EXPERTS - 1:
                    total = offset + count
                    lds_st(union_experts, total, fx.Int32(SHARED_EXPERT))
                    lds_st(union_masks, total, fx.Int32((1 << S) - 1))
                    lds_st(union_counts, N_EXPERTS // 64, total + 1)
                gpu.barrier()
        elif const_expr(GROUPED_UG):
            # Reuse the indexer's retired key arena for the compact union and masks;
            # wave-local masks only occupy the otherwise-idle reduction workspace.
            union_experts = keys + S * MOE_SLOTS
            union_masks = union_experts + UG_UNION_MAX
            union_counts = union_masks + UG_UNION_MAX
            mask_red = fx.recast_iter(fx.Int32, red)
            for i in range_constexpr(WAVES * N_EXPERTS // THREADS):
                lds_st(mask_red, tid + i * THREADS, fx.Int32(0))
            gpu.barrier()
            for batch in range_constexpr(sample_wave_batches(S)):
                sample = wave + batch * WAVES
                if (sample < S) & (lane < TOP_K):
                    expert = fx.Int32(lds_ld(keys, sample * MOE_SLOTS + lane + 1))
                    address = wave * N_EXPERTS + expert
                    previous = lds_ld(mask_red, address)
                    lds_st(mask_red, address, previous | (fx.Int32(1) << sample))
                gpu.barrier()
            mask = fx.Int32(0)
            if tid < N_EXPERTS:
                for w in range_constexpr(WAVES):
                    mask = mask | lds_ld(mask_red, w * N_EXPERTS + tid)
            ballot = fx.Int64(rocdl.ballot(T.i64, mask != 0))
            below = fx.Int32(fmath.ctpop(ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
            count = fx.Int32(fmath.ctpop(ballot))
            if (tid < N_EXPERTS) & (lane == 0):
                lds_st(union_counts, wave, count)
            gpu.barrier()
            offset = fx.Int32(0)
            for w in range_constexpr(N_EXPERTS // 64):
                offset = offset + (wave > w).select(fx.Int32(lds_ld(union_counts, w)), fx.Int32(0))
            if (tid < N_EXPERTS) & (mask != 0):
                lds_st(union_experts, offset + below, tid)
                lds_st(union_masks, offset + below, mask)
            if tid == N_EXPERTS - 1:
                total = offset + count
                lds_st(union_experts, total, fx.Int32(SHARED_EXPERT))
                lds_st(union_masks, total, fx.Int32((1 << S) - 1))
                lds_st(union_counts, N_EXPERTS // 64, total + 1)
            gpu.barrier()
            n_union = _uniform(lds_ld(union_counts, N_EXPERTS // 64))

            # This native body writes UG8 elements, unlike the generic UG_TILE=16 path.
            # Cover the complete intermediate row so every down mailbox gets published.
            grouped_tiles = expert_inter // UG8
            # Only populated union entries execute MFMA; preserve per-expert K order.
            if const_expr(S == 12 or COHORT_ROW_TILES):
                assert grouped_tiles % 2 == 0
                ugpair_tiles = grouped_tiles // 2
                for ugpair_task in range(start("ug"), n_union * ugpair_tiles, G):
                    ugpair_task = fx.Int32(ugpair_task)
                    ugpair_union = ugpair_task // ugpair_tiles
                    ugpair_cbase = (ugpair_task % ugpair_tiles) * 2
                    ugpair_expert = _uniform(lds_ld(union_experts, ugpair_union))
                    ugpair_mask = _uniform(lds_ld(union_masks, ugpair_union))
                    ugpair_units = [ug8_units(ugpair_expert, None, ugpair_cbase + p) for p in range(2)]
                    for ugpair_part in range_constexpr(2):
                        ugpair_acc = mma_units(
                            [fx.Float32(0.0) for _ in range(4)],
                            ugpair_units[ugpair_part],
                        )
                        fx.ptr_store(
                            fx.Vector.from_elements(ugpair_acc, fx.Float32),
                            red + ((ugpair_part * WAVES + wave) * 64 + lane) * 4,
                        )
                    gpu.barrier()
                    # Two independent reductions, each retaining the exact original
                    # wave0..7 addition order; one synchronization covers both tiles.
                    if tid < S * 32:
                        ugpair_part = tid // (S * 16)
                        ugpair_row = tid % (S * 16)
                        ugpair_n = ugpair_row // 16
                        ugpair_r = ugpair_row % 16
                        ugpair_total = fx.Float32(0.0)
                        for ugpair_w in range_constexpr(WAVES):
                            ugpair_total = ugpair_total + lds_ld(
                                red,
                                ((ugpair_part * WAVES + ugpair_w) * 64 + ugpair_n + 16 * (ugpair_r // 4)) * 4
                                + ugpair_r % 4,
                            )
                        lds_st(outs, tid, ugpair_total)
                    gpu.barrier()
                    if tid < S * UG8:
                        ugpair_sample = tid // UG8
                        ugpair_element = (tid % UG8) * 2
                        ugpair_part = ugpair_element // UG8
                        ugpair_r = ugpair_element % UG8
                        if ((ugpair_mask >> ugpair_sample) & 1) != 0:
                            ugpair_slot = fx.Int32(0)
                            for ugpair_j in range_constexpr(1, MOE_SLOTS):
                                ugpair_slot = (
                                    fx.Int32(
                                        lds_ld(
                                            keys,
                                            ugpair_sample * MOE_SLOTS + ugpair_j,
                                        )
                                    )
                                    == ugpair_expert
                                ).select(fx.Int32(ugpair_j), ugpair_slot)
                            ugpair_o = ugpair_part * (S * 16) + ugpair_sample * 16 + ugpair_r
                            ugpair_g0, ugpair_g1 = lds_ld(outs, ugpair_o), lds_ld(outs, ugpair_o + 1)
                            ugpair_v0, ugpair_v1 = lds_ld(outs, ugpair_o + UG8), lds_ld(outs, ugpair_o + UG8 + 1)
                            ugpair_c = ugpair_cbase + ugpair_part
                            put2(
                                mb("mid"),
                                (ugpair_sample * MOE_SLOTS + ugpair_slot) * expert_inter + ugpair_c * UG8 + ugpair_r,
                                ugpair_g0 * _rcp(1.0 + _exp(-ugpair_g0)) * ugpair_v0,
                                ugpair_g1 * _rcp(1.0 + _exp(-ugpair_g1)) * ugpair_v1,
                            )
                            if (ugpair_c == 0) & (ugpair_r == 0):
                                put(
                                    mb("sel"),
                                    ugpair_sample * MOE_SLOTS + ugpair_slot,
                                    ugpair_expert,
                                )
                                put(
                                    mb("prob"),
                                    ugpair_sample * MOE_SLOTS + ugpair_slot,
                                    lds_ld(
                                        dnw,
                                        ugpair_sample * MOE_SLOTS + ugpair_slot,
                                    ),
                                )
                    gpu.barrier()
            else:
                for task in range(start("ug"), n_union * grouped_tiles, G):
                    u = fx.Int32(task)
                    c = u % grouped_tiles
                    expert = _uniform(lds_ld(union_experts, u // grouped_tiles))
                    token_mask = _uniform(lds_ld(union_masks, u // grouped_tiles))
                    units = ug8_units(expert, None, c)
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], units),
                        emit_out(16),
                    )
                    gpu.barrier()
                    if tid < S * UG8 // 2:
                        sample = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        if ((token_mask >> sample) & 1) != 0:
                            slot = fx.Int32(0)
                            for j in range_constexpr(1, MOE_SLOTS):
                                slot = (fx.Int32(lds_ld(keys, sample * MOE_SLOTS + j)) == expert).select(
                                    fx.Int32(j), slot
                                )
                            g0, g1 = lds_ld(outs, sample * 16 + r), lds_ld(outs, sample * 16 + r + 1)
                            v0, v1 = lds_ld(outs, sample * 16 + UG8 + r), lds_ld(outs, sample * 16 + UG8 + r + 1)
                            put2(
                                mb("mid"),
                                (sample * MOE_SLOTS + slot) * expert_inter + c * UG8 + r,
                                g0 * _rcp(1.0 + _exp(-g0)) * v0,
                                g1 * _rcp(1.0 + _exp(-g1)) * v1,
                            )
                            if (c == 0) & (r == 0):
                                put(mb("sel"), sample * MOE_SLOTS + slot, expert)
                                put(
                                    mb("prob"),
                                    sample * MOE_SLOTS + slot,
                                    lds_ld(dnw, sample * MOE_SLOTS + slot),
                                )
                    gpu.barrier()
        else:
            u0 = start("ug")
            for task_round in range_constexpr(UG_TASK_ROUNDS):
                u = fx.Int32(u0 + task_round * G)
                c = u % (expert_inter // UG8)
                has_sh = u < expert_inter // UG8
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))

                def ug8_emit(sample, shared):
                    if tid < (S if shared else 1) * UG8 // 2:
                        n = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        g0, g1 = (
                            lds_ld(outs, n * 16 + r),
                            lds_ld(outs, n * 16 + r + 1),
                        )
                        v0 = lds_ld(outs, n * 16 + UG8 + r)
                        v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                        sn = n if shared else fx.Int32(sample)
                        sl = fx.Int32(0) if shared else slot
                        put2(
                            mb("mid"),
                            (sn * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                    if (c == 0) & (tid < S if shared else tid == 0):
                        sn = tid if shared else fx.Int32(sample)
                        sl = fx.Int32(0) if shared else slot
                        put(
                            mb("sel"),
                            sn * MOE_SLOTS + sl,
                            lds_ld(keys, sn * MOE_SLOTS + sl),
                        )
                        put(
                            mb("prob"),
                            sn * MOE_SLOTS + sl,
                            lds_ld(dnw, sn * MOE_SLOTS + sl),
                        )

                shared_pre = ug8_units(fx.Int32(SHARED_EXPERT), None, c, has_sh)
                cur = ug8_units(_uniform(lds_ld(keys, slot)), 0, c)
                if const_expr(not stage_ug_first and task_round == 0):
                    # Native weights can be prefetched while router activations
                    # arrive. ATOM tiles retain their lower-register staging order.
                    stage_xq(list(range(S)))
                    gpu.barrier()
                if has_sh:
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    ug8_emit(0, True)
                for sample in range_constexpr(S):
                    pre = cur
                    if const_expr(sample + 1 < S):
                        cur = ug8_units(
                            _uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)),
                            sample + 1,
                            c,
                        )
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    ug8_emit(sample, False)
                if const_expr(task_round + 1 < UG_TASK_ROUNDS):
                    gpu.barrier()

    if const_expr(GROUPED_UG):
        # One owner wave per 128-value block; all producers remain resident.
        # Preserve UG8 scheduling and original FP32->MXFP8 arithmetic, while
        # removing the same quantization from every down CTA.
        for mid_chunk in range(bid * WAVES + wave, S * MOE_SLOTS * expert_inter // 128, G * WAVES):
            mid_chunk = fx.Int32(mid_chunk)
            a0, a1 = get2(mb("mid"), mid_chunk * 128 + lane * 2)
            q0, q1, mid_exponent = quant_mxfp8(a0, a1)
            publish_moe_quant(mb("ugp"), mid_chunk * 128 + lane * 2, q0, q1)
            if lane % 16 == 0:
                put(
                    mb("ugp"),
                    MIDQ_WORDS + mid_chunk * 4 + lane // 16,
                    mid_exponent,
                )
        gpu.barrier()


@flyc.jit
def emit_moe_down(plan, state, tools):
    DNW_OFF = plan.DNW_OFF
    DN_COHORT_PAIR = plan.DN_COHORT_PAIR
    DN_CONTIG_CONTRIBUTIONS = plan.DN_CONTIG_CONTRIBUTIONS
    DN_FULL_TASKS = plan.DN_FULL_TASKS
    DN_INPUT_REUSE = plan.DN_INPUT_REUSE
    DN_PAIR_FULL = plan.DN_PAIR_FULL
    DN_PAIR_G = plan.DN_PAIR_G
    DN_PAIR_ROWS = plan.DN_PAIR_ROWS
    DN_PAIR_SPLIT = plan.DN_PAIR_SPLIT
    DN_PAIR_WORK = plan.DN_PAIR_WORK
    DN_SCALE_ROW = plan.DN_SCALE_ROW
    DN_SPLIT_UNION_TAIL = plan.DN_SPLIT_UNION_TAIL
    DN_TILE = plan.DN_TILE
    DN_UNION_ALIGNED = plan.DN_UNION_ALIGNED
    DN_UNION_CANDIDATE = plan.DN_UNION_CANDIDATE
    DN_UNION_INPUT_REUSE = plan.DN_UNION_INPUT_REUSE
    DN_UNION_ROW10 = plan.DN_UNION_ROW10
    DN_WORK_ITEMS = plan.DN_WORK_ITEMS
    DN_X_ROW = plan.DN_X_ROW
    DP_CROFF = plan.DP_CROFF
    DP_CWORDS = plan.DP_CWORDS
    DP_DWOFF = plan.DP_DWOFF
    DP_KEYOFF = plan.DP_KEYOFF
    DP_MSOFF = plan.DP_MSOFF
    DP_ROUTE_WORDS = plan.DP_ROUTE_WORDS
    DP_SCROW = plan.DP_SCROW
    DP_SLOTS = plan.DP_SLOTS
    DP_UMAX = plan.DP_UMAX
    DP_XROW = plan.DP_XROW
    DU_CROFF = plan.DU_CROFF
    DU_CWORDS = plan.DU_CWORDS
    DU_DWOFF = plan.DU_DWOFF
    DU_KEYOFF = plan.DU_KEYOFF
    DU_KEY_WORDS = plan.DU_KEY_WORDS
    DU_MSOFF = plan.DU_MSOFF
    DU_ROUTE_WORDS = plan.DU_ROUTE_WORDS
    DU_SCROW = plan.DU_SCROW
    DU_SLOTS = plan.DU_SLOTS
    DU_UMAX = plan.DU_UMAX
    DU_XROW = plan.DU_XROW
    G = plan.G
    GROUPED_DN = plan.GROUPED_DN
    GROUPED_UG = plan.GROUPED_UG
    I_PER_SLOT = plan.I_PER_SLOT
    KEYS_OFF = plan.KEYS_OFF
    MIDQ_WORDS = plan.MIDQ_WORDS
    MID_SCALE_WORDS = plan.MID_SCALE_WORDS
    N_DN_TILES = plan.N_DN_TILES
    N_UG = plan.N_UG
    RED_WORDS = plan.RED_WORDS
    S = plan.S
    SC = plan.SC
    SY = plan.SY
    TILE_PEER_BYTES = plan.TILE_PEER_BYTES
    TILE_SCRATCH_BYTES = plan.TILE_SCRATCH_BYTES
    W = plan.W
    activation_fp4 = plan.activation_fp4
    atom_experts = plan.atom_experts
    base = plan.base
    dcp_size = plan.dcp_size
    expert_inter = plan.expert_inter
    expert_mxfp4 = plan.expert_mxfp4
    index_request_width = plan.index_request_width
    indexer_cp = plan.indexer_cp
    native_fp4_mfma = plan.native_fp4_mfma
    npes = plan.npes
    with_indexer = plan.with_indexer
    dnw = state.dnw
    full_bid = state.full_bid
    full_out = state.full_out
    full_scratch = state.full_scratch
    full_sym = state.full_sym
    keys = state.keys
    lane = state.lane
    lds = state.lds
    misc = state.misc
    outs = state.outs
    peer_dst = state.peer_dst
    peer_slot = state.peer_slot
    rank = state.rank
    red = state.red
    row_tile = state.row_tile
    s_dn = state.s_dn
    tid = state.tid
    v4f = state.v4f
    w_dn = state.w_dn
    wave = state.wave
    x_out = state.x_out
    xs = state.xs
    lds_ld = tools.lds_ld
    lds_st = tools.lds_st
    bf16_pair = tools.bf16_pair
    mb = tools.mb
    put_bf = tools.put_bf
    bf2_f32 = tools.bf2_f32
    poll = tools.poll
    hint_wait = tools.hint_wait
    get = tools.get
    get2_many = tools.get2_many
    unit_f8f8 = tools.unit_f8f8
    unit_mxfp4_bf16 = tools.unit_mxfp4_bf16
    unit_native_mxfp4 = tools.unit_native_mxfp4
    mma_units = tools.mma_units
    run_units = tools.run_units
    reduce_rows = tools.reduce_rows
    quant_scaled = tools.quant_scaled
    quant_mxfp8 = tools.quant_mxfp8
    st_f8 = tools.st_f8
    st_mxfp8 = tools.st_mxfp8
    load_bias = tools.load_bias
    ffn_owner_reduce = tools.ffn_owner_reduce
    peer_reduce = tools.peer_reduce
    start = tools.start
    n_sel = tools.n_sel
    dn_route = tools.dn_route

    # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
    # 2 row groups x (sample tile * 9 slots * 4) chunks: 4 waves per group.
    # S=8 is evaluated as two four-sample groups so its FP8 mid tile and
    # in-flight weight batches fit comfortably in LDS/VGPRs.
    DN_NKC = expert_inter // 64
    DN_R = (DN_TILE + 15) // 16  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
    DN_WPR = WAVES // DN_R
    DN_BATCH = down_prefetch_batch(S, native_fp4_mfma, atom_experts, expert_inter)
    DN_W_BYTES = HIDDEN * expert_inter // (2 if expert_mxfp4 else 1)
    DN_S_BYTES = HIDDEN * (expert_inter // 32) if expert_mxfp4 else HIDDEN // SCALE_BM * (expert_inter // 128) * 4

    def union_inputs_needed(t):
        if const_expr(DN_UNION_INPUT_REUSE):
            return t == start("down")
        return fx.Boolean(True)

    def dn_union_tile(t, output_rows=32):

        hidden, INTER, SLOTS = HIDDEN, 512, DU_SLOTS
        XROW, SCROW, UNION_MAX = DU_XROW, DU_SCROW, DU_UMAX
        MSOFF, KEYOFF, DWOFF = DU_MSOFF, DU_KEYOFF, DU_DWOFF
        CROFF, CWORDS = DU_CROFF, DU_CWORDS
        gu, gw = wave // 4, wave % 4
        tile = t
        arena = lds.work.ptr
        old_keys = fx.recast_iter(fx.Int32, arena + KEYS_OFF)
        old_dnw = arena + DNW_OFF
        xs, misc = arena, arena + MSOFF
        keys = fx.recast_iter(fx.Int32, arena + KEYOFF)
        dnw, contributions = arena + DWOFF, arena + CROFF
        red = contributions

        def ld(p, i):
            return fx.ptr_load(p + i)

        def st(p, i, v):
            fx.ptr_store(v, p + i)

        def contribution_index(token, slot, kc, row):
            if const_expr(DN_CONTIG_CONTRIBUTIONS):
                # Preserve the logical (slot, K, row, token) contribution;
                # keep each lane's four adjacent rows together for float4 I/O.
                return (((slot * 4 + kc) * 8 + row // 4) * S + token) * 4 + row % 4
            return ((slot * 4 + kc) * 32 + row) * S + token

        def unit(expert, slot, kc, coefficient):
            sn = fx.min(lane % 16, S - 1)
            row = sn * SLOTS + slot
            wb = _rsrc(w_dn + fx.Int64(expert) * fx.Int64(hidden * INTER // 2))
            sb = _rsrc(s_dn + fx.Int64(expert) * fx.Int64(hidden * INTER // 32))
            rg = tile * (output_rows // 16) + gu + (DN_FULL_TASKS if output_rows == 16 else 0)
            raw = fx.Vector(
                bo.buffer_load(
                    wb,
                    (rg * 8 + kc * 2) * 128 + lane * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale = fx.Int32(
                bo.buffer_load(
                    sb,
                    atom_mxfp4_scale_index(rg * 16 + lane % 16, kc * 4 + lane // 16, 16),
                    vec_width=1,
                    dtype=T.i8,
                )
            )
            return (
                raw,
                scale,
                row * XROW + kc * 32,
                row * SCROW + kc * 4 + lane // 16,
                coefficient,
            )

        def partial(u):
            raw, scale, bw, sw, coefficient = u
            words = [
                fx.ptr_load(xs + bw + 16 * half + 4 * (lane // 16), result_type=v4f).bitcast(fx.Int32)
                for half in range(2)
            ]
            b = fx.Vector.from_elements(
                [words[half][e] for half in range(2) for e in range(4)],
                fx.Int32,
            )
            if const_expr(activation_fp4):
                b = fx.ptr_load(xs + bw + 4 * (lane // 16), result_type=v4f).bitcast(fx.Int32)
            asc = ld(misc, sw).bitcast(fx.Int32)
            zero = fx.full(4, 0.0, fx.Float32)
            return fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.vec(4, T.f32), [raw, b, zero, 4, 4 if activation_fp4 else 0, 0, scale, 0, asc]
                )
            )

        # Keep union metadata through every output task. Original keys
        # overlap the LUT and are read only before its first construction.
        if union_inputs_needed(t):
            incoming_key = lds_ld(old_keys, fx.min(tid, DU_KEY_WORDS - 1))
            incoming_prob = lds_ld(old_dnw, fx.min(tid, DU_ROUTE_WORDS - 1))
            gpu.barrier()
            if tid < DU_KEY_WORDS:
                st(keys, tid, incoming_key)
            if tid < DU_ROUTE_WORDS:
                st(dnw, tid, incoming_prob)
            gpu.barrier()

        # Contributions, metadata restores, LUT rebuilds and TP do not
        # write this input prefix. Retain it across output task rounds.
        if union_inputs_needed(t):
            MID_PAIR_ROUNDS = (MIDQ_WORDS + 2 * THREADS - 1) // (2 * THREADS)
            for pack_base in range_constexpr(0, MID_PAIR_ROUNDS, 4):
                packed = poll(
                    [
                        (
                            mb("ugp"),
                            fx.min((tid + j * THREADS) * 2, MIDQ_WORDS - 2),
                            2,
                        )
                        for j in range(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS))
                    ]
                )
                for j in range_constexpr(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS)):
                    word = (tid + j * THREADS) * 2
                    if word < MIDQ_WORDS:
                        target_word = word
                        if const_expr(GROUPED_DN):
                            target_word = (word // (expert_inter // 4)) * DN_X_ROW + word % (expert_inter // 4)
                        st(
                            xs,
                            target_word,
                            packed[j - pack_base][0].bitcast(fx.Float32),
                        )
                        st(
                            xs,
                            target_word + 1,
                            packed[j - pack_base][1].bitcast(fx.Float32),
                        )
            for j in range_constexpr((MID_SCALE_WORDS + THREADS - 1) // THREADS):
                index = tid + j * THREADS
                mid_exponent = get(mb("ugp"), MIDQ_WORDS + fx.min(index, MID_SCALE_WORDS - 1))
                if index < MID_SCALE_WORDS:
                    target_scale = index
                    if const_expr(GROUPED_DN):
                        target_scale = (index // (expert_inter // 32)) * DN_SCALE_ROW + index % (expert_inter // 32)
                    st(misc, target_scale, mid_exponent.bitcast(fx.Float32))

        gpu.barrier()

        ue = keys + S * SLOTS
        um = ue + UNION_MAX
        uc = um + UNION_MAX

        slot_lut = fx.recast_iter(fx.Int8, arena + CROFF + CWORDS)
        if union_inputs_needed(t):
            for j in range_constexpr((257 * S + THREADS - 1) // THREADS):
                address = tid + j * THREADS
                if address < 257 * S:
                    st(slot_lut, address, fx.Int8(0))
            gpu.barrier()
            if tid < S * SLOTS:
                lut_expert = ld(keys, tid)
                st(
                    slot_lut,
                    lut_expert * S + tid // SLOTS,
                    fx.Int8(tid % SLOTS),
                )
            gpu.barrier()

        n_union = _uniform(ld(uc, 4))

        sn = fx.min(lane % 16, S - 1)

        BATCH = 4

        if wave < (output_rows // 16) * 4:
            for batch in range((n_union + BATCH - 1) // BATCH):
                units, slots, active_columns = [], [], []
                for part in range_constexpr(BATCH):
                    union_index = fx.min(batch * BATCH + part, n_union - 1)
                    expert = _uniform(ld(ue, union_index))
                    slot = fx.Int32(ld(slot_lut, expert * S + sn))
                    slots.append(slot)
                    active_columns.append((lane % 16 < S) & (ld(keys, sn * SLOTS + slot) == expert))
                    units.append(unit(expert, slot, gw, fx.Float32(1.0)))
                for part in range_constexpr(BATCH):
                    if (batch * BATCH + part < n_union) | (part == 0 and index_request_width == 6):
                        c = partial(units[part])
                        if active_columns[part]:
                            # Adjacent rows use a fixed layout stride.
                            # Share their base so the LDS writes can pair.
                            row = gu * 16 + lane // 16 * 4
                            target = contribution_index(sn, slots[part], gw, row)
                            if const_expr(DN_CONTIG_CONTRIBUTIONS):
                                fx.ptr_store(c, contributions + target)
                            else:
                                stride = S
                                for e in range_constexpr(4):
                                    st(contributions, target + e * stride, c[e])

        gpu.barrier()

        acc = fx.full(4, 0.0, fx.Float32)

        if const_expr(DN_UNION_ROW10):
            # S10 assigns 90 contiguous slot/K units to each wave.
            # Tokens2/7 cross a wave boundary at slot4, K chunk2.
            if lane % 16 < S:
                for slot in range_constexpr(SLOTS):
                    coefficient = ld(dnw, sn * SLOTS + slot)
                    for kc in range_constexpr(4):
                        if ((sn * SLOTS + slot) * 4 + kc) // 90 == gw:
                            if const_expr(DN_CONTIG_CONTRIBUTIONS):
                                vals = fx.ptr_load(
                                    contributions + contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4),
                                    result_type=v4f,
                                )
                            else:
                                vals = [
                                    ld(
                                        contributions,
                                        contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4 + e),
                                    )
                                    for e in range(4)
                                ]
                            acc = fx.Vector.from_elements(
                                [acc[e] + vals[e] * coefficient for e in range(4)], fx.Float32
                            )
        else:
            # Keep the original wave and every slot/K addition order.
            samples_per_wave = S // 4 if DN_UNION_ALIGNED else 3
            if (lane % 16 < S) & (sn // samples_per_wave == gw) & (gu < output_rows // 16):
                for slot in range_constexpr(SLOTS):
                    coefficient = ld(dnw, sn * SLOTS + slot)
                    for kc in range_constexpr(4):
                        if const_expr(DN_CONTIG_CONTRIBUTIONS):
                            vals = fx.ptr_load(
                                contributions + contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4),
                                result_type=v4f,
                            )
                        else:
                            vals = [
                                ld(
                                    contributions,
                                    contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4 + e),
                                )
                                for e in range(4)
                            ]
                        acc = fx.Vector.from_elements(
                            [acc[e] + vals[e] * coefficient for e in range(4)],
                            fx.Float32,
                        )

        gpu.barrier()

        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)

        gpu.barrier()

        if tid < S * output_rows:
            output_row, output_token = tid % output_rows, tid // output_rows
            tot = fx.Float32(0.0)
            for w in range_constexpr(4):
                ww = output_row // 16 * 4 + w
                tot = tot + ld(
                    red,
                    (ww * 64 + output_token + 16 * ((output_row % 16) // 4)) * 4 + output_row % 4,
                )
            lds_st(outs, output_token * output_rows + output_row, tot)

        if const_expr(N_DN_TILES > G and not DN_UNION_INPUT_REUSE):
            # Multiple output tasks reuse this CTA's original route data.
            # The union arena overlaps it, so restore metadata after all
            # contribution reads and before any later task reconstructs it.
            restore_key = ld(keys, fx.min(tid, DU_KEY_WORDS - 1))
            restore_prob = ld(dnw, fx.min(tid, DU_ROUTE_WORDS - 1))
            gpu.barrier()
            if tid < DU_KEY_WORDS:
                st(old_keys, tid, restore_key)
            if tid < DU_ROUTE_WORDS:
                st(old_dnw, tid, restore_prob)
            gpu.barrier()

    def down_inputs_needed(t):
        if const_expr(DN_INPUT_REUSE):
            return t == start("down")
        return fx.Boolean(True)

    pair_row_tile = (row_tile // 2) * 2

    def pair_mid_spec(wire_word):
        source_cohort = pair_row_tile + wire_word // MIDQ_WORDS
        ptr = full_scratch + fx.Int64(source_cohort) * TILE_SCRATCH_BYTES + fx.Int64(SC["ugp"])
        return (ptr, wire_word % MIDQ_WORDS, 2)

    def pair_peer_reduce(t, tile):
        # Original BF16 partial rounding and rank-order FP32 sum, with
        # the unchanged per-cohort symmetric addresses for each token.
        region_base = fx.Int64(SY["ffn"]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
        if wave < W:
            pair_count = DN_PAIR_ROWS * tile // 2
            for batch in range_constexpr((pair_count + 63) // 64):
                pair = lane + batch * 64
                if pair < pair_count:
                    si = pair // (tile // 2)
                    ri = (pair % (tile // 2)) * 2
                    destination = peer_dst + fx.Int64(pair_row_tile + si // S - row_tile) * TILE_PEER_BYTES
                    put_bf(
                        destination + region_base,
                        (rank * S + si % S) * HIDDEN + t * tile + ri,
                        [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
                        CM_SYS,
                    )
        gpu.barrier()
        if tid < DN_PAIR_ROWS * tile // 2:
            si = tid // (tile // 2)
            ri = (tid % (tile // 2)) * 2
            output_row = t * tile + ri
            cohort = pair_row_tile + si // S
            own = full_sym + fx.Int64(cohort) * TILE_PEER_BYTES + region_base
            got = poll([(own, ((src * S + si % S) * HIDDEN + output_row) // 2, 1) for src in range(W)], "one-as")
            parts = [bf2_f32(value[0]) for value in got]
            t0, t1 = fx.Float32(0.0), fx.Float32(0.0)
            for src in range_constexpr(W):
                t0 = t0 + parts[src][0]
                t1 = t1 + parts[src][1]
            residual_ptr = full_scratch + fx.Int64(cohort) * TILE_SCRATCH_BYTES + fx.Int64(SC["a"])
            r0, r1 = bf2_f32(get(residual_ptr, ((si % S) * HIDDEN + output_row) // 2))
            token = pair_row_tile * S + si
            bo.buffer_store(
                fx.Vector.from_elements([r0 + t0, r1 + t1], fx.Float32).to(fx.BFloat16),
                _rsrc(full_out),
                token * HIDDEN + output_row,
            )

    def dn_pair_tile(t, output_rows=32):

        hidden, INTER, SLOTS = HIDDEN, 512, DP_SLOTS
        XROW, SCROW, UNION_MAX = DP_XROW, DP_SCROW, DP_UMAX
        MSOFF, KEYOFF, DWOFF = DP_MSOFF, DP_KEYOFF, DP_DWOFF
        CROFF, CWORDS = DP_CROFF, DP_CWORDS
        gu, gw = wave // 4, wave % 4
        tile = t
        arena = lds.work.ptr
        xs, misc = arena, arena + MSOFF
        keys = fx.recast_iter(fx.Int32, arena + KEYOFF)
        dnw, contributions = arena + DWOFF, arena + CROFF
        red = contributions

        def ld(p, i):
            return fx.ptr_load(p + i)

        def st(p, i, v):
            fx.ptr_store(v, p + i)

        def contribution_index(token, slot, kc, row):
            if const_expr(DN_CONTIG_CONTRIBUTIONS):
                # Preserve the logical (slot, K, row, token) contribution;
                # keep each lane's four adjacent rows together for float4 I/O.
                return (((slot * 4 + kc) * 8 + row // 4) * DN_PAIR_ROWS + token) * 4 + row % 4
            return ((slot * 4 + kc) * 32 + row) * DN_PAIR_ROWS + token

        def unit(expert, slot, kc, coefficient):
            sn = fx.min(lane % 16, DN_PAIR_ROWS - 1)
            row = sn * SLOTS + slot
            wb = _rsrc(w_dn + fx.Int64(expert) * fx.Int64(hidden * INTER // 2))
            sb = _rsrc(s_dn + fx.Int64(expert) * fx.Int64(hidden * INTER // 32))
            rg = tile * (output_rows // 16) + gu + (DN_PAIR_FULL if output_rows == 16 else 0)
            raw = fx.Vector(
                bo.buffer_load(
                    wb,
                    (rg * 8 + kc * 2) * 128 + lane * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale = fx.Int32(
                bo.buffer_load(
                    sb,
                    atom_mxfp4_scale_index(rg * 16 + lane % 16, kc * 4 + lane // 16, 16),
                    vec_width=1,
                    dtype=T.i8,
                )
            )
            return (
                raw,
                scale,
                row * XROW + kc * 16,
                row * SCROW + kc * 4 + lane // 16,
                coefficient,
            )

        def partial(u):
            raw, scale, bw, sw, coefficient = u
            words = [
                fx.ptr_load(xs + bw + 16 * half + 4 * (lane // 16), result_type=v4f).bitcast(fx.Int32)
                for half in range(2)
            ]
            b = fx.Vector.from_elements(
                [words[half][e] for half in range(2) for e in range(4)],
                fx.Int32,
            )
            if const_expr(activation_fp4):
                b = fx.ptr_load(xs + bw + 4 * (lane // 16), result_type=v4f).bitcast(fx.Int32)
            asc = ld(misc, sw).bitcast(fx.Int32)
            zero = fx.full(4, 0.0, fx.Float32)
            return fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.vec(4, T.f32), [raw, b, zero, 4, 4 if activation_fp4 else 0, 0, scale, 0, asc]
                )
            )

        # Contributions, metadata restores, LUT rebuilds and TP do not
        # write this input prefix. Retain it across output task rounds.
        mid_read_words = (DN_PAIR_ROWS * MOE_SLOTS * expert_inter // 4) // 2
        MID_PAIR_ROUNDS = (mid_read_words + 2 * THREADS - 1) // (2 * THREADS)

        def mid_wire_word(compact_word):
            word = fx.min(compact_word, mid_read_words - 2)
            return (word // 16) * 32 + word % 16

        for pack_base in range_constexpr(0, MID_PAIR_ROUNDS, 4):
            packed = poll(
                [
                    pair_mid_spec(mid_wire_word((tid + j * THREADS) * 2))
                    for j in range(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS))
                ]
            )
            for j in range_constexpr(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS)):
                word = (tid + j * THREADS) * 2
                if word < mid_read_words:
                    target_word = word
                    if const_expr(GROUPED_DN):
                        row_words = expert_inter // 8
                        target_word = (word // row_words) * XROW + word % row_words
                    st(
                        xs,
                        target_word,
                        packed[j - pack_base][0].bitcast(fx.Float32),
                    )
                    st(
                        xs,
                        target_word + 1,
                        packed[j - pack_base][1].bitcast(fx.Float32),
                    )
        for j in range_constexpr(((DN_PAIR_ROWS * MOE_SLOTS * expert_inter // 32) + THREADS - 1) // THREADS):
            index = tid + j * THREADS
            scale_index = fx.min(index, DN_PAIR_ROWS * MOE_SLOTS * expert_inter // 32 - 1)
            scale_cohort = pair_row_tile + scale_index // MID_SCALE_WORDS
            scale_ptr = full_scratch + fx.Int64(scale_cohort) * TILE_SCRATCH_BYTES + fx.Int64(SC["ugp"])
            mid_exponent = get(scale_ptr, MIDQ_WORDS + scale_index % MID_SCALE_WORDS)
            if index < (DN_PAIR_ROWS * MOE_SLOTS * expert_inter // 32):
                target_scale = index
                if const_expr(GROUPED_DN):
                    target_scale = (index // (expert_inter // 32)) * DN_SCALE_ROW + index % (expert_inter // 32)
                st(misc, target_scale, mid_exponent.bitcast(fx.Float32))

        gpu.barrier()

        ue = keys + DN_PAIR_ROWS * SLOTS
        um = ue + UNION_MAX
        uc = um + UNION_MAX

        slot_lut = fx.recast_iter(fx.Int8, arena + CROFF + CWORDS)
        for j in range_constexpr((257 * DN_PAIR_ROWS + THREADS - 1) // THREADS):
            address = tid + j * THREADS
            if address < 257 * DN_PAIR_ROWS:
                st(slot_lut, address, fx.Int8(0))
        gpu.barrier()
        if tid < DN_PAIR_ROWS * SLOTS:
            lut_expert = ld(keys, tid)
            st(
                slot_lut,
                lut_expert * DN_PAIR_ROWS + tid // SLOTS,
                fx.Int8(tid % SLOTS),
            )
        gpu.barrier()

        n_union = _uniform(ld(uc, 4))

        sn = fx.min(lane % 16, DN_PAIR_ROWS - 1)

        BATCH = 4

        if wave < (output_rows // 16) * 4:
            for batch in range((n_union + BATCH - 1) // BATCH):
                units, slots, active_columns = [], [], []
                for part in range_constexpr(BATCH):
                    union_index = fx.min(batch * BATCH + part, n_union - 1)
                    expert = _uniform(ld(ue, union_index))
                    slot = fx.Int32(ld(slot_lut, expert * DN_PAIR_ROWS + sn))
                    slots.append(slot)
                    active_columns.append((lane % 16 < DN_PAIR_ROWS) & (ld(keys, sn * SLOTS + slot) == expert))
                    units.append(unit(expert, slot, gw, fx.Float32(1.0)))
                for part in range_constexpr(BATCH):
                    if (batch * BATCH + part < n_union) | (part == 0 and index_request_width == 6):
                        c = partial(units[part])
                        if active_columns[part]:
                            # Adjacent rows use a fixed layout stride.
                            # Share their base so the LDS writes can pair.
                            row = gu * 16 + lane // 16 * 4
                            target = contribution_index(sn, slots[part], gw, row)
                            if const_expr(DN_CONTIG_CONTRIBUTIONS):
                                fx.ptr_store(c, contributions + target)
                            else:
                                stride = DN_PAIR_ROWS
                                for e in range_constexpr(4):
                                    st(contributions, target + e * stride, c[e])

        gpu.barrier()

        acc = fx.full(4, 0.0, fx.Float32)

        if const_expr(DN_UNION_ROW10):
            # S10 assigns 90 contiguous slot/K units to each wave.
            # Tokens2/7 cross a wave boundary at slot4, K chunk2.
            if lane % 16 < DN_PAIR_ROWS:
                for slot in range_constexpr(SLOTS):
                    coefficient = ld(dnw, sn * SLOTS + slot)
                    for kc in range_constexpr(4):
                        if ((sn * SLOTS + slot) * 4 + kc) // 90 == gw:
                            if const_expr(DN_CONTIG_CONTRIBUTIONS):
                                vals = fx.ptr_load(
                                    contributions + contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4),
                                    result_type=v4f,
                                )
                            else:
                                vals = [
                                    ld(
                                        contributions,
                                        contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4 + e),
                                    )
                                    for e in range(4)
                                ]
                            acc = fx.Vector.from_elements(
                                [acc[e] + vals[e] * coefficient for e in range(4)], fx.Float32
                            )
        else:
            # Keep the original wave and every slot/K addition order.
            samples_per_wave = S // 4
            if (lane % 16 < DN_PAIR_ROWS) & ((sn % S) // samples_per_wave == gw) & (gu < output_rows // 16):
                for slot in range_constexpr(SLOTS):
                    coefficient = ld(dnw, sn * SLOTS + slot)
                    for kc in range_constexpr(4):
                        if const_expr(DN_CONTIG_CONTRIBUTIONS):
                            vals = fx.ptr_load(
                                contributions + contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4),
                                result_type=v4f,
                            )
                        else:
                            vals = [
                                ld(
                                    contributions,
                                    contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4 + e),
                                )
                                for e in range(4)
                            ]
                        acc = fx.Vector.from_elements(
                            [acc[e] + vals[e] * coefficient for e in range(4)],
                            fx.Float32,
                        )

        gpu.barrier()

        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)

        gpu.barrier()

        if tid < DN_PAIR_ROWS * output_rows:
            output_row, output_token = tid % output_rows, tid // output_rows
            tot = fx.Float32(0.0)
            for w in range_constexpr(4):
                ww = output_row // 16 * 4 + w
                tot = tot + ld(
                    red,
                    (ww * 64 + output_token + 16 * ((output_row % 16) // 4)) * 4 + output_row % 4,
                )
            lds_st(outs, output_token * output_rows + output_row, tot)

    if const_expr(DN_COHORT_PAIR):
        pair_keys = fx.recast_iter(fx.Int32, lds.work.ptr + DP_KEYOFF)
        pair_probs = lds.work.ptr + DP_DWOFF
        if tid < DP_ROUTE_WORDS:
            token = tid // MOE_SLOTS
            cohort = pair_row_tile + token // S
            ptr = full_scratch + fx.Int64(cohort) * TILE_SCRATCH_BYTES
            item = (token % S) * MOE_SLOTS + tid % MOE_SLOTS
            lds_st(pair_keys, tid, get(ptr + fx.Int64(SC["sel"]), item))
            lds_st(pair_probs, tid, get(ptr + fx.Int64(SC["prob"]), item).bitcast(fx.Float32))
        gpu.barrier()
        # Rebuild cohort-local expert metadata expected by ordered down union;
        # wave-local masks only occupy the otherwise-idle reduction workspace.
        union_experts = pair_keys + DN_PAIR_ROWS * MOE_SLOTS
        union_masks = union_experts + DP_UMAX
        union_counts = union_masks + DP_UMAX
        mask_red = fx.recast_iter(fx.Int32, lds.work.ptr + DP_CROFF)
        for i in range_constexpr(WAVES * N_EXPERTS // THREADS):
            lds_st(mask_red, tid + i * THREADS, fx.Int32(0))
        gpu.barrier()
        for batch in range_constexpr(sample_wave_batches(DN_PAIR_ROWS)):
            sample = wave + batch * WAVES
            if (sample < DN_PAIR_ROWS) & (lane < TOP_K):
                expert = fx.Int32(lds_ld(pair_keys, sample * MOE_SLOTS + lane + 1))
                address = wave * N_EXPERTS + expert
                previous = lds_ld(mask_red, address)
                lds_st(mask_red, address, previous | (fx.Int32(1) << sample))
            gpu.barrier()
        mask = fx.Int32(0)
        if tid < N_EXPERTS:
            for w in range_constexpr(WAVES):
                mask = mask | lds_ld(mask_red, w * N_EXPERTS + tid)
        ballot = fx.Int64(rocdl.ballot(T.i64, mask != 0))
        below = fx.Int32(fmath.ctpop(ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
        count = fx.Int32(fmath.ctpop(ballot))
        if (tid < N_EXPERTS) & (lane == 0):
            lds_st(union_counts, wave, count)
        gpu.barrier()
        offset = fx.Int32(0)
        for w in range_constexpr(N_EXPERTS // 64):
            offset = offset + (wave > w).select(fx.Int32(lds_ld(union_counts, w)), fx.Int32(0))
        if (tid < N_EXPERTS) & (mask != 0):
            lds_st(union_experts, offset + below, tid)
            lds_st(union_masks, offset + below, mask)
        if tid == N_EXPERTS - 1:
            total = offset + count
            lds_st(union_experts, total, fx.Int32(SHARED_EXPERT))
            lds_st(union_masks, total, fx.Int32((1 << DN_PAIR_ROWS) - 1))
            lds_st(union_counts, N_EXPERTS // 64, total + 1)
        gpu.barrier()

        pair_bid = full_bid % DN_PAIR_G
        pair_first = (pair_bid + DN_PAIR_G - base["down"]) & (DN_PAIR_G - 1)
        for pair_task in range(pair_first, DN_PAIR_WORK, DN_PAIR_G):
            pair_task = fx.Int32(pair_task)
            if const_expr(DN_PAIR_SPLIT):
                if pair_task >= DN_PAIR_FULL:
                    dn_pair_tile(pair_task, output_rows=16)
                else:
                    dn_pair_tile(pair_task)
            else:
                dn_pair_tile(pair_task)
            gpu.barrier()
            if const_expr(DN_PAIR_SPLIT):
                if pair_task >= DN_PAIR_FULL:
                    pair_peer_reduce(pair_task + DN_PAIR_FULL, 16)
                else:
                    pair_peer_reduce(pair_task, 32)
            else:
                pair_peer_reduce(pair_task, 32)
            gpu.barrier()
    else:
        for t in range(start("down"), DN_WORK_ITEMS, G):
            t = fx.Int32(t)
            if const_expr(DN_UNION_CANDIDATE or DN_UNION_ROW10 or DN_UNION_ALIGNED):
                if const_expr(DN_SPLIT_UNION_TAIL):
                    if t >= DN_FULL_TASKS:
                        dn_union_tile(t, output_rows=16)
                    else:
                        dn_union_tile(t)
                else:
                    dn_union_tile(t)
            else:
                if down_inputs_needed(t):
                    if const_expr(ug_split(S, expert_inter) is None):  # retain the tuned down-stage route schedule
                        dn_route(load_bias())
                gpu.barrier()
                gu = wave // DN_WPR
                dn_rg = t * DN_TILE // 16
                dn_off = t * DN_TILE % 16
                # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
                # (same cache lines) and are dropped in the output
                dn_lr = gu * 16 + lane % 16 - dn_off
                dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

                DN_NU = S * MOE_SLOTS * DN_NKC // 2
                DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
                DN_BLK = S * MOE_SLOTS * expert_inter // 128

                def u_dn(cc):  # cc: 128-k chunk of this wave
                    qu = (wave % DN_WPR) * DN_UPW + cc
                    live = qu < DN_NU
                    unit = fx.min(qu, DN_NU - 1)
                    q = unit * 2  # 64-k chunk index over (s, slot, kc)
                    s_q = q // (MOE_SLOTS * DN_NKC)
                    slot_q = (q // DN_NKC) % MOE_SLOTS
                    kc = q % DN_NKC
                    e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                    wb = _rsrc(
                        w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                        (None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))),
                    )
                    sb = _rsrc(
                        s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                        (None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))),
                    )

                    def coef():  # mid block scale * route weight, only in this sample's column
                        return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0))

                    if const_expr(expert_mxfp4):

                        def bf16_coef():
                            return (lane % 16 == s_q).select(
                                _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                                fx.Float32(0.0),
                            )

                        if const_expr(native_fp4_mfma):
                            return unit_native_mxfp4(
                                wb,
                                sb,
                                dn_rg + gu,
                                unit % (expert_inter // 128),
                                expert_inter,
                                unit * 32,
                                unit * 4 + lane // 16,
                                bf16_coef,
                                dn_ln,
                            )
                        return unit_mxfp4_bf16(
                            wb,
                            sb,
                            dn_rg + gu,
                            unit % (expert_inter // 128),
                            expert_inter,
                            unit * 64,
                            bf16_coef,
                            dn_ln,
                        )
                    return unit_f8f8(
                        wb,
                        sb,
                        dn_rg + gu,
                        kc,
                        DN_NKC,
                        expert_inter,
                        q * 16,
                        coef,
                        dn_ln,
                    )

                if const_expr(GROUPED_DN and S == 12):
                    # keys/dnw are ready before mid payloads. Build in the disjoint red
                    # arena now, so producer progress can overlap this scheduling work.
                    # Preserve the old contiguous sample partition of each row-group
                    # wave. Reorder only independent token columns, never one token's
                    # nonzero slot/K additions or its final wave-reduction position.
                    assert S % DN_WPR == 0
                    DN_GROUP_SAMPLES = S // DN_WPR
                    DN_GROUP_MAX = DN_GROUP_SAMPLES * MOE_SLOTS
                    DN_COUNT_OFF = DN_WPR * DN_GROUP_MAX * 3
                    assert DN_COUNT_OFF + DN_WPR <= RED_WORDS
                    dsg_dn_table = fx.recast_iter(fx.Int32, red)
                    dsg_count = fx.Int32(0)
                    dsg_mask = fx.Int32(0)
                    if wave < DN_WPR:
                        dsg_first_sample = wave * DN_GROUP_SAMPLES
                        for dsg_build_slot in range_constexpr(MOE_SLOTS):
                            dsg_experts = [
                                _uniform(
                                    lds_ld(
                                        keys,
                                        (dsg_first_sample + j) * MOE_SLOTS + dsg_build_slot,
                                    )
                                )
                                for j in range(DN_GROUP_SAMPLES)
                            ]
                            for j in range_constexpr(DN_GROUP_SAMPLES):
                                dsg_distinct = fx.Int32(1) == 1
                                for dsg_previous in range_constexpr(j):
                                    dsg_distinct = dsg_distinct & (dsg_experts[j] != dsg_experts[dsg_previous])
                                if dsg_distinct:
                                    dsg_mask = fx.Int32(0)
                                    for dsg_member in range_constexpr(DN_GROUP_SAMPLES):
                                        dsg_mask = dsg_mask | (dsg_experts[j] == dsg_experts[dsg_member]).select(
                                            fx.Int32(1) << (dsg_first_sample + dsg_member),
                                            fx.Int32(0),
                                        )
                                    if lane == 0:
                                        dsg_address = (wave * DN_GROUP_MAX + dsg_count) * 3
                                        lds_st(
                                            dsg_dn_table,
                                            dsg_address,
                                            dsg_experts[j],
                                        )
                                        lds_st(
                                            dsg_dn_table,
                                            dsg_address + 1,
                                            fx.Int32(dsg_build_slot),
                                        )
                                        lds_st(dsg_dn_table, dsg_address + 2, dsg_mask)
                                    dsg_count = dsg_count + 1
                        if lane == 0:
                            lds_st(dsg_dn_table, DN_COUNT_OFF + wave, dsg_count)
                    gpu.barrier()
                if const_expr(S <= 4):
                    pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
                # Later output tasks read the same route, packed mid values,
                # and exponents. Their compute/reduction/TP bodies only read
                # xs/misc/keys/dnw, so keep those LDS inputs until down ends.
                if down_inputs_needed(t):
                    hint_wait(
                        N_UG,
                        lambda k: (
                            mb("mid"),
                            (k // (MOE_SLOTS * I_PER_SLOT) * MOE_SLOTS + (k // I_PER_SLOT) % MOE_SLOTS) * expert_inter
                            + (k % I_PER_SLOT) * UG_TILE
                            + UG_TILE
                            - 1,
                        ),
                    )
                    if const_expr(GROUPED_UG):
                        # Poll each tagged payload directly: no new global completion
                        # flag or relaxed data-publication assumption is introduced.
                        MID_PAIR_ROUNDS = (MIDQ_WORDS + 2 * THREADS - 1) // (2 * THREADS)
                        for pack_base in range_constexpr(0, MID_PAIR_ROUNDS, 4):
                            packed = poll(
                                [
                                    (
                                        mb("ugp"),
                                        fx.min((tid + j * THREADS) * 2, MIDQ_WORDS - 2),
                                        2,
                                    )
                                    for j in range(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS))
                                ]
                            )
                            for j in range_constexpr(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS)):
                                word = (tid + j * THREADS) * 2
                                if word < MIDQ_WORDS:
                                    target_word = word
                                    if const_expr(GROUPED_DN):
                                        target_word = (word // (expert_inter // 4)) * DN_X_ROW + word % (
                                            expert_inter // 4
                                        )
                                    lds_st(
                                        xs,
                                        target_word,
                                        packed[j - pack_base][0].bitcast(fx.Float32),
                                    )
                                    lds_st(
                                        xs,
                                        target_word + 1,
                                        packed[j - pack_base][1].bitcast(fx.Float32),
                                    )
                        for j in range_constexpr((MID_SCALE_WORDS + THREADS - 1) // THREADS):
                            index = tid + j * THREADS
                            mid_exponent = get(
                                mb("ugp"),
                                MIDQ_WORDS + fx.min(index, MID_SCALE_WORDS - 1),
                            )
                            if index < MID_SCALE_WORDS:
                                target_scale = index
                                if const_expr(GROUPED_DN):
                                    target_scale = (index // (expert_inter // 32)) * DN_SCALE_ROW + index % (
                                        expert_inter // 32
                                    )
                                lds_st(misc, target_scale, mid_exponent.bitcast(fx.Float32))
                    else:
                        # TP4 with 12 rows has 54 wave-local groups. Quantize bounded batches
                        # so the full set of mid values does not stay live in registers.
                        N_MID_BLOCKS = (DN_BLK + WAVES - 1) // WAVES
                        MID_BATCH = 8 if native_fp4_mfma and expert_inter > INTER and S == 12 else N_MID_BLOCKS
                        for mid_base in range_constexpr(0, N_MID_BLOCKS, MID_BATCH):
                            mids = get2_many(
                                [
                                    (
                                        mb("mid"),
                                        fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2,
                                    )
                                    for b in range(
                                        mid_base,
                                        min(mid_base + MID_BATCH, N_MID_BLOCKS),
                                    )
                                ]
                            )
                            for b in range_constexpr(mid_base, min(mid_base + MID_BATCH, N_MID_BLOCKS)):
                                blk = wave + b * WAVES
                                if blk < DN_BLK:
                                    if const_expr(expert_mxfp4):
                                        if const_expr(native_fp4_mfma):
                                            q0, q1, qs = quant_mxfp8(
                                                mids[b - mid_base][0],
                                                mids[b - mid_base][1],
                                            )
                                            st_mxfp8(blk * 128 + lane * 2, q0, q1)
                                            if lane % 16 == 0:
                                                lds_st(
                                                    misc,
                                                    blk * 4 + lane // 16,
                                                    qs.bitcast(fx.Float32),
                                                )
                                        else:
                                            lds_st(
                                                xs,
                                                blk * 64 + lane,
                                                bf16_pair(
                                                    mids[b - mid_base][0],
                                                    mids[b - mid_base][1],
                                                ),
                                            )
                                    else:
                                        q0, q1, qs = quant_scaled(mids[b - mid_base][0], mids[b - mid_base][1])
                                        st_f8(blk * 128 + lane * 2, q0, q1)
                                        if lane == 0:
                                            lds_st(
                                                misc,
                                                blk,
                                                qs * lds_ld(dnw, blk // (expert_inter // 128)),
                                            )
                gpu.barrier()
                if const_expr(GROUPED_DN):
                    if const_expr(S != 12):
                        # Preserve the old contiguous sample partition of each row-group
                        # wave. Reorder only independent token columns, never one token's
                        # nonzero slot/K additions or its final wave-reduction position.
                        assert S % DN_WPR == 0
                        DN_GROUP_SAMPLES = S // DN_WPR
                        DN_GROUP_MAX = DN_GROUP_SAMPLES * MOE_SLOTS
                        DN_COUNT_OFF = DN_WPR * DN_GROUP_MAX * 3
                        assert DN_COUNT_OFF + DN_WPR <= RED_WORDS
                        dsg_dn_table = fx.recast_iter(fx.Int32, red)
                        dsg_count = fx.Int32(0)
                        dsg_mask = fx.Int32(0)
                        if wave < DN_WPR:
                            dsg_first_sample = wave * DN_GROUP_SAMPLES
                            for dsg_build_slot in range_constexpr(MOE_SLOTS):
                                dsg_experts = [
                                    _uniform(
                                        lds_ld(
                                            keys,
                                            (dsg_first_sample + j) * MOE_SLOTS + dsg_build_slot,
                                        )
                                    )
                                    for j in range(DN_GROUP_SAMPLES)
                                ]
                                for j in range_constexpr(DN_GROUP_SAMPLES):
                                    dsg_distinct = fx.Int32(1) == 1
                                    for dsg_previous in range_constexpr(j):
                                        dsg_distinct = dsg_distinct & (dsg_experts[j] != dsg_experts[dsg_previous])
                                    if dsg_distinct:
                                        dsg_mask = fx.Int32(0)
                                        for dsg_member in range_constexpr(DN_GROUP_SAMPLES):
                                            dsg_mask = dsg_mask | (dsg_experts[j] == dsg_experts[dsg_member]).select(
                                                fx.Int32(1) << (dsg_first_sample + dsg_member),
                                                fx.Int32(0),
                                            )
                                        if lane == 0:
                                            dsg_address = (wave * DN_GROUP_MAX + dsg_count) * 3
                                            lds_st(
                                                dsg_dn_table,
                                                dsg_address,
                                                dsg_experts[j],
                                            )
                                            lds_st(
                                                dsg_dn_table,
                                                dsg_address + 1,
                                                fx.Int32(dsg_build_slot),
                                            )
                                            lds_st(
                                                dsg_dn_table,
                                                dsg_address + 2,
                                                dsg_mask,
                                            )
                                        dsg_count = dsg_count + 1
                            if lane == 0:
                                lds_st(dsg_dn_table, DN_COUNT_OFF + wave, dsg_count)
                        gpu.barrier()
                    dsg_group_wave = wave % DN_WPR
                    dsg_groups = _uniform(lds_ld(dsg_dn_table, DN_COUNT_OFF + dsg_group_wave))
                    dsg_down_acc = fx.full(4, 0.0, fx.Float32)
                    DN_GROUP_BATCH = 2 if S == 12 else 1
                    for dsg_batch_index in range((dsg_groups + DN_GROUP_BATCH - 1) // DN_GROUP_BATCH):
                        dsg_units = []
                        for dsg_part in range_constexpr(DN_GROUP_BATCH):
                            group_index = dsg_batch_index
                            if const_expr(DN_GROUP_BATCH > 1):
                                group_index = fx.min(
                                    dsg_batch_index * DN_GROUP_BATCH + dsg_part,
                                    dsg_groups - 1,
                                )
                            dsg_address = (dsg_group_wave * DN_GROUP_MAX + group_index) * 3
                            dsg_expert = _uniform(lds_ld(dsg_dn_table, dsg_address))
                            dsg_route_slot = _uniform(lds_ld(dsg_dn_table, dsg_address + 1))
                            dsg_token_mask = _uniform(lds_ld(dsg_dn_table, dsg_address + 2))
                            if const_expr(DN_GROUP_BATCH > 1):
                                dsg_token_mask = (dsg_batch_index * DN_GROUP_BATCH + dsg_part < dsg_groups).select(
                                    dsg_token_mask, fx.Int32(0)
                                )
                            dsg_sn = n_sel()
                            dsg_mid_row = dsg_sn * MOE_SLOTS + dsg_route_slot
                            dsg_coefficient = (((dsg_token_mask >> dsg_sn) & 1) != 0).select(
                                lds_ld(dnw, dsg_mid_row), fx.Float32(0.0)
                            )
                            dsg_wb = _rsrc(w_dn + fx.Int64(dsg_expert) * fx.Int64(DN_W_BYTES))
                            dsg_sb = _rsrc(s_dn + fx.Int64(dsg_expert) * fx.Int64(DN_S_BYTES))
                            dsg_units.extend(
                                [
                                    unit_native_mxfp4(
                                        dsg_wb,
                                        dsg_sb,
                                        dn_rg + gu,
                                        kc,
                                        expert_inter,
                                        dsg_mid_row * DN_X_ROW + kc * 32,
                                        dsg_mid_row * DN_SCALE_ROW + kc * 4 + lane // 16,
                                        dsg_coefficient,
                                        dn_ln,
                                    )
                                    for kc in range(expert_inter // 128)
                                ]
                            )
                        dsg_down_acc = fx.Vector.from_elements(
                            mma_units([dsg_down_acc[e] for e in range(4)], dsg_units),
                            fx.Float32,
                        )
                    acc = [dsg_down_acc[e] for e in range(4)]
                    # All waves must finish reading the schedule before reduce_rows
                    # reuses red for per-wave accumulator stores.
                    gpu.barrier()
                else:
                    if const_expr(native_fp4_mfma and npes == 4 and S == 10 and expert_inter == 512):
                        # Bound the unrolled body while preserving every wave's
                        # slot/K order and two-batch software pipeline.
                        DN_LOOP_UNITS = 10
                        assert DN_UPW % DN_LOOP_UNITS == 0 and DN_LOOP_UNITS % DN_BATCH == 0
                        dn_loop_acc = fx.full(4, 0.0, fx.Float32)
                        for dn_loop in range(DN_UPW // DN_LOOP_UNITS):
                            dn_base = dn_loop * DN_LOOP_UNITS
                            dn_cur = [u_dn(dn_base + cc) for cc in range(DN_BATCH)]
                            for dn_begin in range_constexpr(0, DN_LOOP_UNITS, DN_BATCH):
                                dn_next = None
                                if const_expr(dn_begin + DN_BATCH < DN_LOOP_UNITS):
                                    dn_next = [
                                        u_dn(dn_base + cc)
                                        for cc in range(
                                            dn_begin + DN_BATCH,
                                            dn_begin + 2 * DN_BATCH,
                                        )
                                    ]
                                dn_loop_acc = fx.Vector.from_elements(
                                    mma_units([dn_loop_acc[e] for e in range(4)], dn_cur),
                                    fx.Float32,
                                )
                                dn_cur = dn_next
                        acc = [dn_loop_acc[e] for e in range(4)]
                    else:
                        if const_expr(S > 4):
                            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
                        acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

                def emit_dn(rl, n, v):
                    if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                        lds_st(outs, n * DN_TILE + rl - dn_off, v)

                reduce_rows(DN_R, acc, emit_dn)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            def residual_a(s, row):
                return bf2_f32(get(mb("a"), (s * HIDDEN + row) // 2))

            if const_expr(
                native_fp4_mfma
                and atom_experts
                and npes == 4
                and S == 12
                and expert_inter == 512
                and not with_indexer
                and not indexer_cp
                and dcp_size == 1
            ):
                if const_expr(DN_SPLIT_UNION_TAIL):
                    if t >= DN_FULL_TASKS:
                        ffn_owner_reduce(t + DN_FULL_TASKS, residual_a, store_x, tile=16)
                    else:
                        ffn_owner_reduce(t, residual_a, store_x)
                else:
                    ffn_owner_reduce(t, residual_a, store_x)
            else:
                if const_expr(DN_SPLIT_UNION_TAIL):
                    if t >= DN_FULL_TASKS:
                        peer_reduce("ffn", t + DN_FULL_TASKS, residual_a, store_x, tile=16)
                    else:
                        peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
                else:
                    peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
            gpu.barrier()
