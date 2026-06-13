# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill (paged, vec_k_col_v) for gfx942 — scheduler variant.

Derived from ``fmha_prefill_fp8_v7.py``. This keeps the diagonal-pair q-tile
mapping and tests a more partial LDS-read schedule around the QK/PV MFMA
clusters, closer to PyISA's partial ``lgkmcnt`` pattern.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

HD = 128
KSTEPS = HD // 16
DT = HD // 32
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
NTHREADS = NWAVES * 64
WAVE_ROWS = 32
TILE_BM = NWAVES * WAVE_ROWS
BM = 2 * TILE_BM
BN = 32
NSLOT = 256
NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_v9_smem")
_K_BYTES = BN * HD
_V_BYTES = HD * BN
_K_OFF = 0
_V_OFF = _K_OFF + 2 * _K_BYTES
_alloc.ptr = _V_OFF + 2 * _V_BYTES


def const_false(x):
    return fx.const_expr(x == 0)


_LGKMCNT0 = 0xC07F


def _wait_lds():
    fx.rocdl.s_waitcnt(_LGKMCNT0)


@flyc.kernel(known_block_size=[NTHREADS, 1, 1])
def attn_kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    Qd: fx.Tensor,
    Kd: fx.Tensor,
    Vd: fx.Tensor,
    LTD: fx.Tensor,
    LTP: fx.Tensor,
    Ps: fx.Tensor,
    O: fx.Tensor,
    sq: fx.Int32,
    sk: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    page_size: fx.Constexpr[int],
    k_page_stride: fx.Int32,
    v_page_stride: fx.Int32,
    sm_scale: fx.Constexpr[float],
    causal: fx.Constexpr[int],
):
    tid = fx.Int32(fx.thread_idx.x)
    wave_id = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)
    blk = fx.Int32(fx.block_idx.x)
    gqa = nq // nk
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)

    num_q_tiles = (sq_i + fx.Int32(TILE_BM - 1)) // fx.Int32(TILE_BM)
    num_pair_tiles = (num_q_tiles + fx.Int32(1)) // fx.Int32(2)
    pair_idx = blk % num_pair_tiles
    tmp = blk // num_pair_tiles
    qhead = tmp % fx.Int32(nq)
    batch = tmp // fx.Int32(nq)
    kvhead = qhead // fx.Int32(gqa)

    q_local = lane % fx.Int32(32)
    half = lane // fx.Int32(32)

    rq = fx.buffer_ops.create_buffer_resource(Q)
    rk = fx.buffer_ops.create_buffer_resource(K)
    rv = fx.buffer_ops.create_buffer_resource(V)
    rqd = fx.buffer_ops.create_buffer_resource(Qd)
    rkd = fx.buffer_ops.create_buffer_resource(Kd)
    rvd = fx.buffer_ops.create_buffer_resource(Vd)
    rltd = fx.buffer_ops.create_buffer_resource(LTD)
    rltp = fx.buffer_ops.create_buffer_resource(LTP)
    ro = fx.buffer_ops.create_buffer_resource(O)

    q_tok_stride = fx.Int32(nq * HD)
    v_tok_stride = fx.Int32(nk * HD)

    page0 = fx.buffer_ops.buffer_load(rltp, batch, vec_width=1, dtype=fx.Int32)
    k_head_off = kvhead * fx.Int32(HD * page_size)
    v_head_off = kvhead * fx.Int32(HD)
    v_descale = fx.buffer_ops.buffer_load(rvd, batch * fx.Int32(nk) + kvhead, vec_width=1, dtype=fx.Float32)

    rps = fx.buffer_ops.create_buffer_resource(Ps)
    p_scale = fx.buffer_ops.buffer_load(rps, batch * fx.Int32(nq) + qhead, vec_width=1, dtype=fx.Float32)
    _ps_raw = p_scale.ir_value() if hasattr(p_scale, "ir_value") else p_scale
    log2_pscale = fx.Float32(fx.math.log2(_ps_raw))

    f32x16 = fx.typing.T.vec(16, fx.typing.T.f32)
    neg_inf = fx.Float32(-3.0e38)
    width64 = fx.Int32(64)
    off32 = fx.Int32(32)

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(2 * _K_BYTES,)).get()
    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(2 * _V_BYTES,)).get()

    pass_slot = []
    pass_valid = []
    pass_kv = []
    pass_cg = []
    for p in fx.range_constexpr(NPASS):
        slot = fx.Int32(p * NTHREADS) + tid
        pass_slot.append(slot)
        pass_valid.append(slot < fx.Int32(NSLOT))
        slot_s = (slot < fx.Int32(NSLOT)).select(slot, fx.Int32(0))
        pass_kv.append(slot_s // fx.Int32(8))
        pass_cg.append(slot_s % fx.Int32(8))

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]
    ps_i = fx.Int32(page_size)
    n_kv_full = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)

    def load_kv_regs(kv0_):
        kc_words = []
        vc_words = []
        for p in fx.range_constexpr(NPASS):
            kvrow = kv0_ + pass_kv[p]
            kvrow_safe = (kvrow < sk_i).select(kvrow, fx.Int32(0))
            slot = page0 + kvrow_safe // ps_i
            phys = fx.buffer_ops.buffer_load(rltd, slot, vec_width=1, dtype=fx.Int32)
            intra = kvrow_safe % ps_i
            kc_off = phys * k_page_stride + k_head_off + pass_cg[p] * (ps_i * fx.Int32(16)) + intra * fx.Int32(16)
            kc_words.append(fx.buffer_ops.buffer_load(rk, kc_off // fx.Int32(4), vec_width=4, dtype=fx.Int32))
            vc_vidx = phys * v_page_stride + intra * v_tok_stride + v_head_off + pass_cg[p] * fx.Int32(16)
            vc_words.append(fx.buffer_ops.buffer_load(rv, vc_vidx // fx.Int32(4), vec_width=4, dtype=fx.Int32))
        return kc_words, vc_words

    def store_kv_to_lds(kc_words, vc_words, kbuf_off, vbuf_off):
        for p in fx.range_constexpr(NPASS):
            if const_expr(NPASS > 1):
                guard = pass_valid[p]
            else:
                guard = None
            kv_row = pass_kv[p]
            cg = pass_cg[p]
            k_dst = kbuf_off + kv_row * fx.Int32(HD) + cg * fx.Int32(16)
            v_d0 = cg * fx.Int32(16)
            if guard is not None:
                if guard:
                    fx.Vector(kc_words[p]).bitcast(fx.Int8).store(k_lds, [fx.Index(k_dst)])
                    vc_i8 = fx.Vector(vc_words[p]).bitcast(fx.Int8)
                    for e in fx.range_constexpr(16):
                        fx.Vector.from_elements([fx.Vector(vc_i8)[e]], fx.Int8).store(
                            vt_lds, [fx.Index(vbuf_off + (v_d0 + fx.Int32(e)) * fx.Int32(BN) + kv_row)]
                        )
            else:
                fx.Vector(kc_words[p]).bitcast(fx.Int8).store(k_lds, [fx.Index(k_dst)])
                vc_i8 = fx.Vector(vc_words[p]).bitcast(fx.Int8)
                for e in fx.range_constexpr(16):
                    fx.Vector.from_elements([fx.Vector(vc_i8)[e]], fx.Int8).store(
                        vt_lds, [fx.Index(vbuf_off + (v_d0 + fx.Int32(e)) * fx.Int32(BN) + kv_row)]
                    )

    kv_local = lane % fx.Int32(32)
    sk_m1 = sk_i - fx.Int32(1)

    def process_qtile(qtile):
        wave_q0 = qtile * fx.Int32(TILE_BM) + wave_id * fx.Int32(WAVE_ROWS)
        qrow = wave_q0 + q_local
        qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))
        q_base = batch * (sq_i * q_tok_stride) + qrow_safe * q_tok_stride + qhead * fx.Int32(HD)

        q_i64 = []
        for ks in fx.range_constexpr(KSTEPS):
            off = q_base + fx.Int32(ks * 16) + half * fx.Int32(8)
            w = fx.buffer_ops.buffer_load(rq, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
            q_i64.append(fx.Vector(w).bitcast(fx.Int64)[0])

        qd_idx = (batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe
        q_descale = fx.buffer_ops.buffer_load(rqd, qd_idx, vec_width=1, dtype=fx.Float32)

        if const_expr(causal == 0):
            n_kv_rt = n_kv_full
        else:
            q_max = qtile * fx.Int32(TILE_BM) + fx.Int32(TILE_BM - 1)
            kv_max = q_max + (sk_i - sq_i)
            n_kv_caus = (kv_max + fx.Int32(BN)) // fx.Int32(BN)
            n_kv_rt = (n_kv_caus < n_kv_full).select(n_kv_caus, n_kv_full)

        if const_expr(causal != 0):
            cb = qrow + (sk_i - sq_i)
            eff_bound = (cb < sk_m1).select(cb, sk_m1)
        else:
            eff_bound = sk_m1

        kc_w0, vc_w0 = load_kv_regs(fx.Int32(0))
        store_kv_to_lds(kc_w0, vc_w0, fx.Int32(0), fx.Int32(0))
        fx.gpu.barrier()
        init_state = [m_run0, l_run0] + o_acc0
        for kt_iv, st in range(fx.Index(0), fx.Index(n_kv_rt), fx.Index(1), init=init_state):
            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]
            kv0 = fx.Int32(kt_iv) * fx.Int32(BN)
            cur_buf = fx.Int32(kt_iv) % fx.Int32(2)
            kbuf = cur_buf * fx.Int32(_K_BYTES)
            vbuf = cur_buf * fx.Int32(_V_BYTES)

            nxt_buf = (fx.Int32(kt_iv) + fx.Int32(1)) % fx.Int32(2)
            kbuf_n = nxt_buf * fx.Int32(_K_BYTES)
            vbuf_n = nxt_buf * fx.Int32(_V_BYTES)
            kc_w_next, vc_w_next = load_kv_regs(kv0 + fx.Int32(BN))

            fx.rocdl.s_setprio(1)
            k_packs = []
            for ks in fx.range_constexpr(KSTEPS):
                k_lds_elem = kbuf + kv_local * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
                kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_lds_elem)])
                k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
            fx.rocdl.sched_dsrd(4)
            acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
            for ks in fx.range_constexpr(KSTEPS):
                a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
                b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
                acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
                if const_expr(ks == 3):
                    fx.rocdl.sched_dsrd(4)
                fx.rocdl.sched_mfma(1)
            sv = fx.Vector(acc_raw)

            kd_row_base = (batch * fx.Int32(nk) + kvhead) * sk_i
            kdv = []
            for g in fx.range_constexpr(4):
                kv_g0 = kv0 + fx.Int32(g * 8) + half * fx.Int32(4)
                kv_g0s = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))
                kdv.append(fx.Vector(fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0s, vec_width=4, dtype=fx.Float32)))

            s_vals = []
            for i in fx.range_constexpr(16):
                kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
                k_descale = kdv[i // 4][i % 4]
                s = sv[i] * (q_descale * k_descale * fx.Float32(sm_scale))
                s_vals.append(fx.Float32((kv <= eff_bound).select(s, neg_inf)))

            m_loc = s_vals[0]
            for i in fx.range_constexpr(15):
                m_loc = m_loc.maximumf(s_vals[i + 1])
            m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
            m_new = m_run.maximumf(m_loc)

            m_is_neg = m_new < fx.Float32(-1.0e38)
            safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
            f32t = fx.typing.T.f32
            _ar = fx.arith.unwrap
            corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
            corr = m_is_neg.select(fx.Float32(0.0), corr)

            l_loc = fx.Float32(0.0)
            p_vals = []
            for i in fx.range_constexpr(16):
                p = fx.Float32(fx.rocdl.exp2(f32t, _ar((s_vals[i] - safe_m) * fx.Float32(LOG2E) + log2_pscale)))
                p_vals.append(p)
                l_loc = l_loc + p
            l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
            l_run = l_run * corr + l_loc

            def _cvt4(v0, v1, v2, v3):
                lo = fx.rocdl.cvt_pk_fp8_f32(
                    fx.typing.T.i32,
                    fx.Float32(v0).ir_value(),
                    fx.Float32(v1).ir_value(),
                    fx.Int32(0).ir_value(),
                    False,
                )
                return fx.rocdl.cvt_pk_fp8_f32(
                    fx.typing.T.i32,
                    fx.Float32(v2).ir_value(),
                    fx.Float32(v3).ir_value(),
                    lo,
                    True,
                )

            is_h0 = half == fx.Int32(0)
            q_byte = q_local * fx.Int32(4)
            q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)
            p_i64_s = []
            for s in fx.range_constexpr(2):
                pack0 = _cvt4(p_vals[s * 8 + 0], p_vals[s * 8 + 1], p_vals[s * 8 + 2], p_vals[s * 8 + 3])
                pack1 = _cvt4(p_vals[s * 8 + 4], p_vals[s * 8 + 5], p_vals[s * 8 + 6], p_vals[s * 8 + 7])
                h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
                h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
                h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
                h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
                _wait_lds()
                w0 = is_h0.select(h0_b0, h0_b1)
                w1 = is_h0.select(h1_b0, h1_b1)
                p_i64_s.append(fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0])

            corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
            for dt in fx.range_constexpr(DT):
                o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

            v_packs = []
            for dt in fx.range_constexpr(DT):
                d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
                for s in fx.range_constexpr(2):
                    v_lds_elem = vbuf + d_col * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                    vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_lds_elem)])
                    v_packs.append(fx.Vector(vv8).bitcast(fx.Int64)[0])
            fx.rocdl.sched_dsrd(4)
            for dt in fx.range_constexpr(DT):
                acc2 = fx.Vector(o_acc[dt]).ir_value()
                for s in fx.range_constexpr(2):
                    v_i64 = v_packs[dt * 2 + s]
                    p_i64 = p_i64_s[s]
                    a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                    b_raw = p_i64.ir_value() if hasattr(p_i64, "ir_value") else p_i64
                    acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
                    if const_expr(dt == 1 and s == 1):
                        fx.rocdl.sched_dsrd(4)
                    fx.rocdl.sched_mfma(1)
                o_acc[dt] = fx.Vector(acc2)
            fx.rocdl.s_setprio(0)

            store_kv_to_lds(kc_w_next, vc_w_next, kbuf_n, vbuf_n)
            fx.gpu.barrier()

            new_state = [m_new, l_run] + [o_acc[d] for d in range(DT)]
            st = yield new_state

        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]

        l_is_zero = l_run < fx.Float32(1.0e-30)
        inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
        scale_o = fx.Float32(v_descale * inv_l)
        scale_vec = fx.Vector.filled(16, scale_o, fx.Float32)
        in_b = qrow < sq_i
        o_row_base = ((batch * sq_i + qrow_safe) * fx.Int32(nq) + qhead) * fx.Int32(HD)
        if in_b:
            for dt in fx.range_constexpr(DT):
                ov = fx.Vector(o_acc[dt]) * scale_vec
                ov_bf16 = fx.Vector(ov).to(fx.BFloat16)
                for j in fx.range_constexpr(4):
                    d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                    v4 = fx.Vector.from_elements([fx.Vector(ov_bf16)[j * 4 + e] for e in range(4)], fx.BFloat16)
                    fx.buffer_ops.buffer_store(v4.ir_value(), ro, (o_row_base + d).ir_value())

    process_qtile(pair_idx)
    mirror_qtile = num_q_tiles - fx.Int32(1) - pair_idx
    if mirror_qtile > pair_idx:
        process_qtile(mirror_qtile)


@flyc.jit
def run_attn(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    Qd: fx.Tensor,
    Kd: fx.Tensor,
    Vd: fx.Tensor,
    LTD: fx.Tensor,
    LTP: fx.Tensor,
    Ps: fx.Tensor,
    O: fx.Tensor,
    sq: fx.Int32,
    sk: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    page_size: fx.Constexpr[int],
    k_page_stride: fx.Int32,
    v_page_stride: fx.Int32,
    sm_scale: fx.Constexpr[float],
    causal: fx.Constexpr[int],
    grid_blocks: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    attn_kernel(
        Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal
    ).launch(grid=(grid_blocks,), block=(NTHREADS,), stream=stream)
