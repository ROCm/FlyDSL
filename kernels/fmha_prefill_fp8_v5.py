# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill (paged, vec_k_col_v) for gfx942 — FlyDSL port.

Port of the hand-written PyISA kernel
``f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv``. Feature parity:

- head_dim 128 (QK and V), FP8 e4m3 **FNUZ** inputs, **bf16** output, causal mask.
- GQA (``nhead_q = gqa * nhead_k``; q head ``h`` uses kv head ``h // gqa``).
- Per-token-per-head Q/K descale (applied to S after the QK MFMA); per-head V descale
  (folded into the output normalization).
- Paged KV with **vec_k_col_v** layout: K pool ``[pages, nhead_k, hd/16, page_size, 16]``,
  V pool row-major ``[pages, page_size, nhead_k, hd]``; addressed via a flat page-id table
  ``LTD`` + per-batch ``kv_indptr`` (``LTP``). Any power-of-2 page_size.
- ``p_scale`` per-(batch, q_head): rescales the fp8 probabilities P for e4m3 range; cancels
  exactly in O/L so it only affects P's quantization precision.

GEMMs use ``mfma_f32_32x32x16_fp8_fp8``. GEMM1 = K@Qᵀ (S as [kv, q]); online softmax over
kv in registers; P is transposed through LDS (q-major fp8) and fed to GEMM2 = Vᵀ@P (O as
[d, q]). One workgroup = one (batch, q_head, BM-row q-tile), NWAVES waves.

Default config: BM=128, 4 waves/256 threads (FMHA_NWAVES sweepable: 8/4/2). BM=128 measured
uniformly fastest on MI308X (better occupancy/latency-hiding than the 256×128/8-wave variant).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

HD = 128
KSTEPS = HD // 16  # GEMM1 K-steps over head_dim
DT = HD // 32  # GEMM2 d-tiles
# NWAVES (waves/workgroup) is sweepable via FMHA_NWAVES to fight CU-starvation at small seqlen:
# smaller BM => more q-tiles => more workgroups => fill all 80 CUs. 8w=512thr/BM256 is the
# large-seq config; 4w/2w give 2x/4x the grid for short sequences.
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
NTHREADS = NWAVES * 64
WAVE_ROWS = 32  # q rows owned by each wave
BM = NWAVES * WAVE_ROWS  # q rows per workgroup
BN = 32  # kv per MFMA subtile
NSUB = 4  # subtiles processed under ONE softmax (cross-GEMM pipeline): 128 kv / softmax
BN128 = BN * NSUB  # 128 kv per outer iteration
# The cooperative K/V load fills NSLOT=256 (32 kv x 8 feature-groups) 16B slots per BN=32 subtile.
# With NTHREADS threads that takes NPASS passes (1 pass when NTHREADS>=256).
NSLOT = 256
NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_v5_smem")
# v5 layout: all NSUB=4 K and 4 V subtiles of the 128-kv group resident at once, so the whole
# group loads with ONE barrier (vs 4 in v4) then runs 4 GEMM1 -> 1 softmax -> 4 GEMM2 barrier-free.
# 32KB total still gives 2 wg/CU on the 64KB SIMD (same occupancy as v4's 24KB). Single-buffered;
# next-group prefetch is register-staged (OPT3 pattern). No P scratch (register-resident P).
_K_BYTES = BN * HD  # 4 KB per K subtile (kv-major [kv x hd])
_V_BYTES = HD * BN  # 4 KB per V subtile (transposed [d x kv])
_K_OFF = 0
_V_OFF = _K_OFF + NSUB * _K_BYTES  # 4 K subtiles = 16 KB
_alloc.ptr = _V_OFF + NSUB * _V_BYTES  # + 4 V subtiles = 32 KB total


def const_false(x):
    return fx.const_expr(x == 0)


# s_waitcnt lgkmcnt(0): wait for all LDS/scalar ops, leave vmcnt/expcnt as don't-care.
# gfx942 encoding: vmcnt[3:0]=0xf, vmcnt[15:14]=0x3 (don't-wait), expcnt[6:4]=0x7, lgkmcnt[13:8]=0.
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
    LTD: fx.Tensor,  # int32 [total_pages] physical page id per slot
    LTP: fx.Tensor,  # int32 [batch+1] kv_indptr
    Ps: fx.Tensor,  # f32 [batch*nq] per-(batch,qhead) p_scale (1.0 = disabled)
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

    num_q_tiles = (sq_i + fx.Int32(BM - 1)) // fx.Int32(BM)
    qtile = blk % num_q_tiles
    tmp = blk // num_q_tiles
    qhead = tmp % fx.Int32(nq)
    batch = tmp // fx.Int32(nq)
    kvhead = qhead // fx.Int32(gqa)

    q_local = lane % fx.Int32(32)
    half = lane // fx.Int32(32)
    # this wave's 32-row q sub-tile within the 256-row workgroup tile
    wave_q0 = qtile * fx.Int32(BM) + wave_id * fx.Int32(WAVE_ROWS)

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
    v_tok_stride = fx.Int32(nk * HD)  # within a V page (row-major [ps, nk, hd])

    # Paging: this batch's page range [page0, page_end) from LTP.
    page0 = fx.buffer_ops.buffer_load(rltp, batch, vec_width=1, dtype=fx.Int32)
    k_head_off = kvhead * fx.Int32(HD * page_size)  # vec_k_col_v: [pages, nk, hd/16, ps, 16]
    v_head_off = kvhead * fx.Int32(HD)  # row-major V: [pages, ps, nk, hd]

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
    v_descale = fx.buffer_ops.buffer_load(rvd, batch * fx.Int32(nk) + kvhead, vec_width=1, dtype=fx.Float32)

    # p_scale: per-(batch,qhead). P is multiplied by p_scale before fp8 cast (more e4m3 range),
    # then divided back out in the epilogue. log2(p_scale) is added to the exp bias.
    rps = fx.buffer_ops.create_buffer_resource(Ps)
    p_scale = fx.buffer_ops.buffer_load(rps, batch * fx.Int32(nq) + qhead, vec_width=1, dtype=fx.Float32)
    _ps_raw = p_scale.ir_value() if hasattr(p_scale, "ir_value") else p_scale
    log2_pscale = fx.Float32(fx.math.log2(_ps_raw))

    f32x16 = fx.typing.T.vec(16, fx.typing.T.f32)
    neg_inf = fx.Float32(-3.0e38)
    width64 = fx.Int32(64)
    off32 = fx.Int32(32)

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(NSUB * _K_BYTES,)).get()
    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(NSUB * _V_BYTES,)).get()
    # V4: 128-bit (16-fp8) cooperative loads — wide global reads like PyISA's buffer_load_dwordx4.
    # NSLOT=256 distinct (kv, group-of-16) slots cover the 32x128 tile. With NTHREADS threads we
    # cover them in NPASS passes; slot p*NTHREADS+tid, guarded slot<NSLOT (extra threads idle).
    # Per-pass slot decomposition (all lists length NPASS):
    pass_slot = []
    pass_valid = []
    pass_kv = []  # slot//8 : kv row 0..31
    pass_cg = []  # slot%8  : feature group 0..7 (16 feats each)
    for p in fx.range_constexpr(NPASS):
        slot = fx.Int32(p * NTHREADS) + tid
        pass_slot.append(slot)
        pass_valid.append(slot < fx.Int32(NSLOT))
        slot_s = (slot < fx.Int32(NSLOT)).select(slot, fx.Int32(0))  # clamp for index math
        pass_kv.append(slot_s // fx.Int32(8))
        pass_cg.append(slot_s % fx.Int32(8))

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]

    ps_i = fx.Int32(page_size)
    # Causal kv upper bound: this workgroup's max q-row is qtile*256+255, which can attend to
    # kv up to q + (sk - sq). Tiles beyond that are fully masked → skip them entirely. This is
    # the dominant algorithmic win for causal (early Q-tiles do far less KV work).
    n_kv_full = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)
    if const_expr(causal == 0):
        n_kv_rt = n_kv_full
    else:
        q_max = qtile * fx.Int32(BM) + fx.Int32(BM - 1)
        kv_max = q_max + (sk_i - sq_i)  # last attendable kv index for this workgroup
        n_kv_caus = (kv_max + fx.Int32(BN)) // fx.Int32(BN)  # ceil((kv_max+1)/BN)
        n_kv_rt = (n_kv_caus < n_kv_full).select(n_kv_caus, n_kv_full)

    # Issue the cooperative global loads for a kv-tile; return the per-thread K/V words
    # (each a vec<2xi32> = one i64). The buffer_loads are async, so calling this for tile
    # i+1 while computing tile i overlaps global-memory latency with MFMA (prefetch).
    def load_kv_regs(kv0_):
        # Returns (kc_words, vc_words): one K/V 128-bit load per pass (length NPASS each).
        kc_words = []
        vc_words = []
        for p in fx.range_constexpr(NPASS):
            kvrow = kv0_ + pass_kv[p]
            kvrow_safe = (kvrow < sk_i).select(kvrow, fx.Int32(0))
            slot = page0 + kvrow_safe // ps_i
            phys = fx.buffer_ops.buffer_load(rltd, slot, vec_width=1, dtype=fx.Int32)
            intra = kvrow_safe % ps_i
            # K: 128-bit load of the full 16-fp8 cg group (no hg split).
            kc_off = phys * k_page_stride + k_head_off + pass_cg[p] * (ps_i * fx.Int32(16)) + intra * fx.Int32(16)
            kc_words.append(fx.buffer_ops.buffer_load(rk, kc_off // fx.Int32(4), vec_width=4, dtype=fx.Int32))
            # V: 128-bit load of 16 contiguous d (pass_cg is the 16-d group).
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
                # idle threads (slot>=NSLOT) must not store; dest already clamped to slot0.
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
    # OPT2: per-lane loop-invariant causal/bounds limit. valid kv iff kv <= eff_bound.
    sk_m1 = sk_i - fx.Int32(1)
    if const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_m1).select(cb, sk_m1)
    else:
        eff_bound = sk_m1
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap
    is_h0 = half == fx.Int32(0)
    q_byte = q_local * fx.Int32(4)  # half0 lane's byte addr (same q)
    q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)  # half1 lane's byte addr

    def _cvt4(v0, v1, v2, v3):
        lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(), fx.Int32(0).ir_value(), False)
        return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(), lo, True)

    # v5 CROSS-GEMM PIPELINE: process NSUB=4 BN=32 subtiles (128 kv) per outer iteration under a
    # SINGLE softmax. Cuts softmax/online-rescale VALU 4x (corr, o_acc rescale, m/l bookkeeping
    # done once per 128 kv instead of per 32) and barriers 4x (2/group vs ~1/subtile).
    kd_row_base = (batch * fx.Int32(nk) + kvhead) * sk_i
    n_grp = (n_kv_rt + fx.Int32(NSUB - 1)) // fx.Int32(NSUB)
    init_state = [m_run0, l_run0] + o_acc0
    for grp_iv, st in range(fx.Index(0), fx.Index(n_grp), fx.Index(1), init=init_state):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        g_kv0 = fx.Int32(grp_iv) * fx.Int32(BN128)

        # cooperative load: all NSUB K + V subtiles of this 128-kv group into LDS (one barrier).
        for s in fx.range_constexpr(NSUB):
            kc_w, vc_w = load_kv_regs(g_kv0 + fx.Int32(s * BN))
            store_kv_to_lds(kc_w, vc_w, fx.Int32(s * _K_BYTES), fx.Int32(s * _V_BYTES))
        fx.gpu.barrier()

        fx.rocdl.s_setprio(1)
        # GEMM1 x NSUB: S_s[kv,q] = K_s @ Q^T. Each leaves 16 f32 in sv_all[s].
        sv_all = []
        for s in fx.range_constexpr(NSUB):
            k_packs = []
            for ks in fx.range_constexpr(KSTEPS):
                k_lds_elem = fx.Int32(s * _K_BYTES) + kv_local * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
                kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_lds_elem)])
                k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
            acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
            for ks in fx.range_constexpr(KSTEPS):
                a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
                b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
                acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
            sv_all.append(fx.Vector(acc_raw))

        # descale + causal mask -> 64 s_vals (NSUB subtiles x 16). k_descale: 4 contiguous-kv
        # 4-wide loads per subtile.
        s_vals = []  # flat list length NSUB*16; s_vals[s*16 + i]
        for s in fx.range_constexpr(NSUB):
            sub_kv0 = g_kv0 + fx.Int32(s * BN)
            kdv = []
            for g in fx.range_constexpr(4):
                kv_g0 = sub_kv0 + fx.Int32(g * 8) + half * fx.Int32(4)
                kv_g0s = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))
                kdv.append(fx.Vector(fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0s, vec_width=4, dtype=fx.Float32)))
            for i in fx.range_constexpr(16):
                kv = sub_kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
                k_descale = kdv[i // 4][i % 4]
                sval = sv_all[s][i] * (q_descale * k_descale * fx.Float32(sm_scale))
                s_vals.append(fx.Float32((kv <= eff_bound).select(sval, neg_inf)))

        # SINGLE softmax over all 64 kv: row max, cross-half, then exp/sum.
        m_loc = s_vals[0]
        for i in fx.range_constexpr(NSUB * 16 - 1):
            m_loc = m_loc.maximumf(s_vals[i + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
        m_new = m_run.maximumf(m_loc)

        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)

        l_loc = fx.Float32(0.0)
        p_vals = []  # flat NSUB*16
        for i in fx.range_constexpr(NSUB * 16):
            p = fx.Float32(fx.rocdl.exp2(f32t, _ar((s_vals[i] - safe_m) * fx.Float32(LOG2E) + log2_pscale)))
            p_vals.append(p)
            l_loc = l_loc + p
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # register-resident P transpose per subtile -> p_i64[s][kstep] (kstep 0..1).
        p_i64 = []
        for s in fx.range_constexpr(NSUB):
            base = s * 16
            ks_pair = []
            for ks in fx.range_constexpr(2):
                pack0 = _cvt4(p_vals[base + ks * 8 + 0], p_vals[base + ks * 8 + 1], p_vals[base + ks * 8 + 2], p_vals[base + ks * 8 + 3])
                pack1 = _cvt4(p_vals[base + ks * 8 + 4], p_vals[base + ks * 8 + 5], p_vals[base + ks * 8 + 6], p_vals[base + ks * 8 + 7])
                h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
                h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
                h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
                h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
                _wait_lds()
                w0 = is_h0.select(h0_b0, h0_b1)
                w1 = is_h0.select(h1_b0, h1_b1)
                ks_pair.append(fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0])
            p_i64.append(ks_pair)

        # rescale running O once per group
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        for dt in fx.range_constexpr(DT):
            o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

        # GEMM2: O[d,q] += sum_s V_s^T @ P_s. Bulk-read all V packs first, then run all MFMAs
        # across the DT independent o_acc chains (NSUB*2 MFMAs accumulated per dt).
        v_packs = []  # v_packs[s][dt*2 + ks]
        for s in fx.range_constexpr(NSUB):
            sp = []
            for dt in fx.range_constexpr(DT):
                d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
                for ks in fx.range_constexpr(2):
                    v_lds_elem = fx.Int32(s * _V_BYTES) + d_col * fx.Int32(BN) + fx.Int32(ks * 16) + half * fx.Int32(8)
                    vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_lds_elem)])
                    sp.append(fx.Vector(vv8).bitcast(fx.Int64)[0])
            v_packs.append(sp)
        for dt in fx.range_constexpr(DT):
            acc2 = fx.Vector(o_acc[dt]).ir_value()
            for s in fx.range_constexpr(NSUB):
                for ks in fx.range_constexpr(2):
                    v_i64 = v_packs[s][dt * 2 + ks]
                    p_op = p_i64[s][ks]
                    a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                    b_raw = p_op.ir_value() if hasattr(p_op, "ir_value") else p_op
                    acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
            o_acc[dt] = fx.Vector(acc2)
        fx.rocdl.s_setprio(0)
        fx.gpu.barrier()  # all V/K LDS reads done before next group overwrites

        new_state = [m_new, l_run] + [o_acc[d] for d in range(DT)]
        st = yield new_state

    # results from the runtime loop's final iteration
    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]

    # epilogue: O[d,q] *= v_descale / l_run, cast bf16, store O[b, qrow, qhead, d]
    # Guard l_run==0 (query row with no valid keys, e.g. sk<sq causal): emit 0, not nan.
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
            # accumulator elements [j*4 .. j*4+3] are 4 contiguous d at dt*32 + j*8 + half*4,
            # so each group of 4 bf16 is a single vectorized store.
            for j in fx.range_constexpr(4):
                d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                v4 = fx.Vector.from_elements([fx.Vector(ov_bf16)[j * 4 + e] for e in range(4)], fx.BFloat16)
                fx.buffer_ops.buffer_store(v4.ir_value(), ro, (o_row_base + d).ir_value())


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
