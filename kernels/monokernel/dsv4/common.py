# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device helpers shared by every stage of the DeepSeek-V4 MonoKernel."""

from functools import partial

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.monokernel import helpers
from kernels.monokernel.dsv4.config import EPS, FP8_MAX, ROUTE_SCALE
from kernels.monokernel.dsv4.plan import MIN_I32, THREADS, TL_COLS, WAVES
from kernels.monokernel.helpers import SHARED_SOURCE_KEY, LaunchState
from kernels.monokernel.ops import (
    exp,
    f8_word,
    fp8_roundtrip,
    lds_ld,
    lds_st,
    mxfp8_to_bf16x8,
    rcp,
    wave_max,
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
    wave = ctx["wave"]
    xs = ctx["xs"]

    # ------------------------------------------------------------ helpers
    # ---- tagged-pair mailboxes
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

    # ------------------------------------------------ MFMA GEMV machinery
    def mx_rg(c):
        """Up/gate tile ``c`` (8 intermediates: gate rows on lanes r < 8, up rows on
        r >= 8) as this lane's row group of the gate/up bank in ATOM's order."""
        return (c // 2) * 2 + (lane % 16) // 8

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

    launch_state = LaunchState(
        source_key=SHARED_SOURCE_KEY,
        S=S,
        tid=tid,
        lane=lane,
        wave=wave,
        bid=bid,
        G=G,
        base=base,
        scratch=scratch,
        SC=SC,
        tag=tag,
        red=red,
        outs=outs,
        xs=xs,
        bias=bias,
        N_EXPERTS=N_EXPERTS,
        timeline=timeline,
        timeline_buf=timeline_buf,
        first=first,
        tl_cols=TL_COLS,
        rank=rank,
        W=W,
        HIDDEN=HIDDEN,
        peer_dst=peer_dst,
        sym=sym,
        SY=SY,
        eps=EPS,
    )
    launch_state.poll = partial(
        helpers.poll, launch_state, hang=hang, poll_timeout_ticks=POLL_TIMEOUT_TICKS if BOUNDED_POLL else None
    )
    launch_state.stamp = partial(helpers.stamp, launch_state, stamp_fence=True)
    mb = launch_state.mb
    put = launch_state.put
    put2 = launch_state.put2
    put_bf = launch_state.put_bf
    get = launch_state.get
    getf = launch_state.getf
    getf_many = launch_state.getf_many
    get2_many = launch_state.get2_many
    pre_poll = launch_state.pre_poll
    hint_wait = launch_state.hint_wait
    block_sums = launch_state.block_sums
    block_sum = launch_state.block_sum
    unit_fp8 = launch_state.unit_fp8
    unit_f8f8 = launch_state.unit_f8f8
    unit_bf16 = launch_state.unit_bf16
    run_units = launch_state.run_units
    reduce_rows = launch_state.reduce_rows
    emit_out = launch_state.emit_out
    stage_x_pairs = launch_state.stage_x_pairs
    quant_scaled = launch_state.quant_scaled
    st_f8 = launch_state.st_f8
    load_bias = launch_state.load_bias
    start = launch_state.start
    n_sel = launch_state.n_sel
    poll = launch_state.poll
    stamp = launch_state.stamp
    mma_units = launch_state.mma_units
    unit_fp8mx = launch_state.unit_fp8mx
    unit_mxfp4 = launch_state.unit_mxfp4_atom
    peer_reduce = launch_state.peer_reduce
    stage_x_rmsnorm = launch_state.stage_x_rmsnorm
    load_x_rmsnorm = launch_state.load_x_rmsnorm
    _rmsnorm_tail_ks = launch_state.rmsnorm_tail_ks

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
