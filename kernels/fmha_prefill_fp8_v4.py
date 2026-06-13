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
[d, q]). One workgroup = one (batch, q_head, 32-row q-tile); 1 wave (64 threads).

This is the correctness-first config (BM=32, 1 wave). The 256×128 / 8-wave performance
variant and optimization passes are layered on separately.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

HD = 128
KSTEPS = HD // 16  # GEMM1 K-steps over head_dim
DT = HD // 32  # GEMM2 d-tiles
NWAVES = 8  # waves per workgroup (512 threads)
WAVE_ROWS = 32  # q rows owned by each wave
BM = NWAVES * WAVE_ROWS  # 256 q rows per workgroup
BN = 32  # kv per MFMA tile
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_8w_smem")
_P_TILE = WAVE_ROWS * BN  # per-wave P transpose scratch [q*BN + kv] fp8
_P_BYTES = NWAVES * _P_TILE  # 8 KB
_P_OFF = 0
# K/V tiles DOUBLE-BUFFERED (ping-pong) so the next tile's cooperative load overlaps the
# current tile's compute. Buffer b lives at _K_OFF + b*_K_BYTES (and _V_OFF + b*_V_BYTES).
_K_BYTES = BN * HD  # 4 KB per buffer (kv-major K tile [kv x hd])
_V_BYTES = HD * BN  # 4 KB per buffer (V transposed [d x kv])
_K_OFF = _P_OFF + _P_BYTES
_V_OFF = _K_OFF + 2 * _K_BYTES
_alloc.ptr = _V_OFF + 2 * _V_BYTES  # 8 + 8 + 8 = 24 KB total


def const_false(x):
    return fx.const_expr(x == 0)


# s_waitcnt lgkmcnt(0): wait for all LDS/scalar ops, leave vmcnt/expcnt as don't-care.
# gfx942 encoding: vmcnt[3:0]=0xf, vmcnt[15:14]=0x3 (don't-wait), expcnt[6:4]=0x7, lgkmcnt[13:8]=0.
_LGKMCNT0 = 0xC07F


def _wait_lds():
    fx.rocdl.s_waitcnt(_LGKMCNT0)


@flyc.kernel(known_block_size=[512, 1, 1])
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
    wave_p0 = wave_id * fx.Int32(_P_TILE)  # per-wave base into the shared P-LDS scratch

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

    p_lds = SmemPtr(_alloc.get_base(), _P_OFF, fx.typing.T.i8, shape=(_P_BYTES,)).get()
    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(2 * _K_BYTES,)).get()
    # V4: 128-bit (16-fp8) cooperative loads — wide global reads like PyISA's buffer_load_dwordx4.
    # 256 distinct (kv, group-of-16) slots cover the 32x128 tile; tid%256 so 256..511 duplicate
    # 0..255 (idempotent same-load+same-dest, harmless).
    ktid = tid % fx.Int32(256)
    kc_kv = ktid // fx.Int32(8)  # kv row 0..31
    kc_cg = ktid % fx.Int32(8)  # feature group 0..7 (16 feats each) — whole group now
    kc_lds_elem = kc_kv * fx.Int32(HD) + kc_cg * fx.Int32(16)  # 16-contiguous-fp8 dest

    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(2 * _V_BYTES,)).get()
    vc_kv = ktid // fx.Int32(8)  # kv row 0..31
    vc_dg = ktid % fx.Int32(8)  # which 16-d group 0..7
    vc_d0 = vc_dg * fx.Int32(16)  # first d of this thread's 16

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
        kc_kvrow = kv0_ + kc_kv
        kc_kvrow_safe = (kc_kvrow < sk_i).select(kc_kvrow, fx.Int32(0))
        kc_slot = page0 + kc_kvrow_safe // ps_i
        kc_phys = fx.buffer_ops.buffer_load(rltd, kc_slot, vec_width=1, dtype=fx.Int32)
        kc_intra = kc_kvrow_safe % ps_i
        # K: 128-bit load of the full 16-fp8 cg group (no hg split).
        kc_off = kc_phys * k_page_stride + k_head_off + kc_cg * (ps_i * fx.Int32(16)) + kc_intra * fx.Int32(16)
        kc_w_ = fx.buffer_ops.buffer_load(rk, kc_off // fx.Int32(4), vec_width=4, dtype=fx.Int32)
        # V: 128-bit load of 16 contiguous d.
        vc_kvrow = kv0_ + vc_kv
        vc_kvrow_safe = (vc_kvrow < sk_i).select(vc_kvrow, fx.Int32(0))
        vc_slot = page0 + vc_kvrow_safe // ps_i
        vc_phys = fx.buffer_ops.buffer_load(rltd, vc_slot, vec_width=1, dtype=fx.Int32)
        vc_vidx = vc_phys * v_page_stride + (vc_kvrow_safe % ps_i) * v_tok_stride + v_head_off + vc_d0
        vc_w_ = fx.buffer_ops.buffer_load(rv, vc_vidx // fx.Int32(4), vec_width=4, dtype=fx.Int32)
        return kc_w_, vc_w_

    def store_kv_to_lds(kc_w_, vc_w_, kbuf_off, vbuf_off):
        # K: one wide 16B store (contiguous cg group).
        fx.Vector(kc_w_).bitcast(fx.Int8).store(k_lds, [fx.Index(kbuf_off + kc_lds_elem)])
        # V: keep opt5's TRANSPOSED LDS layout (wide reads in GEMM2) — store the 16 loaded d as
        # 16 byte-writes to vt_lds[d*BN + kv]. (Narrow writes, but reads are wide & happen 4x more.)
        vc_i8 = fx.Vector(vc_w_).bitcast(fx.Int8)
        for e in fx.range_constexpr(16):
            fx.Vector.from_elements([fx.Vector(vc_i8)[e]], fx.Int8).store(
                vt_lds, [fx.Index(vbuf_off + (vc_d0 + fx.Int32(e)) * fx.Int32(BN) + vc_kv)]
            )

    kv_local = lane % fx.Int32(32)
    # OPT2: per-lane loop-invariant causal/bounds limit. valid kv iff kv <= eff_bound.
    sk_m1 = sk_i - fx.Int32(1)
    if const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_m1).select(cb, sk_m1)
    else:
        eff_bound = sk_m1
    # Ping-pong: tile N reads/computes from buffer (N&1); its cooperative load was issued
    # and stored during tile N-1. Prologue stores tile 0 into buffer 0.
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

        # OPT3: ISSUE the next tile's global loads NOW (async), but DEFER the LDS store until
        # after the GEMMs — so the ~300cy global-load latency overlaps GEMM1+softmax+GEMM2
        # instead of stalling right before GEMM1.
        nxt_buf = (fx.Int32(kt_iv) + fx.Int32(1)) % fx.Int32(2)
        kbuf_n = nxt_buf * fx.Int32(_K_BYTES)
        vbuf_n = nxt_buf * fx.Int32(_V_BYTES)
        kc_w_next, vc_w_next = load_kv_regs(kv0 + fx.Int32(BN))

        # GEMM1: S[kv,q] = K @ Q^T. OPT5: BULK-read all 8 K packs from LDS first, so the LDS-read
        # latencies overlap each other (and the early MFMAs) instead of each MFMA waiting on its
        # own read. The 8-deep MFMA accumulation chain then runs back-to-back.
        fx.rocdl.s_setprio(1)
        k_packs = []
        for ks in fx.range_constexpr(KSTEPS):
            k_lds_elem = kbuf + kv_local * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
            kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_lds_elem)])
            k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
        acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        for ks in fx.range_constexpr(KSTEPS):
            a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
            b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
            acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
        sv = fx.Vector(acc_raw)
        # (no barrier here: the P-store barrier below already separates this GEMM1
        # K-read from the next iteration's cooperative K-write.)

        # descale + causal mask. This lane's 16 kv split into 4 groups of 4 CONTIGUOUS kv
        # (kv = kv0 + g*8 + half*4 + {0..3}), so load k_descale as 4 vectorized 4-wide loads
        # instead of 16 scalar global loads.
        kd_row_base = (batch * fx.Int32(nk) + kvhead) * sk_i
        kdv = []
        for g in fx.range_constexpr(4):
            kv_g0 = kv0 + fx.Int32(g * 8) + half * fx.Int32(4)
            kv_g0s = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))  # in-bounds 4-vec base
            kdv.append(fx.Vector(fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0s, vec_width=4, dtype=fx.Float32)))
        # OPT2: fold the two mask compares (kv<sk_i AND kv<=qrow+(sk-sq)) into ONE per-element
        # compare against a per-lane LOOP-INVARIANT bound, hoisted out of the kv loop:
        #   eff_bound = causal ? min(qrow+(sk-sq), sk_i-1) : sk_i-1.
        # Valid iff kv <= eff_bound. Halves mask VALU and removes redundant per-tile recompute.
        s_vals = []
        for i in fx.range_constexpr(16):
            kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            k_descale = kdv[i // 4][i % 4]
            s = sv[i] * (q_descale * k_descale * fx.Float32(sm_scale))
            s_vals.append(fx.Float32((kv <= eff_bound).select(s, neg_inf)))

        # row max over 16 kv rows, then across halves
        m_loc = s_vals[0]
        for i in fx.range_constexpr(15):
            m_loc = m_loc.maximumf(s_vals[i + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
        m_new = m_run.maximumf(m_loc)

        # Guard against an all-masked history (m_new == -inf): use 0 in the
        # exponent so masked p=exp2(s-0)=0 (s is -inf) and corr=0 instead of NaN/1.
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        # Fast exp2: rocdl.exp2 = single v_exp_f32 (inputs pre-clamped via safe_m, so the
        # full-range v_exp+v_ldexp pair that Float32.exp2() emits is unnecessary).
        f32t = fx.typing.T.f32
        _ar = fx.arith.unwrap
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)

        # p = 2^((s - m_new)*log2e); l_tile = sum p (cross-half)
        l_loc = fx.Float32(0.0)
        p_vals = []
        for i in fx.range_constexpr(16):
            p = fx.Float32(fx.rocdl.exp2(f32t, _ar((s_vals[i] - safe_m) * fx.Float32(LOG2E) + log2_pscale)))
            p_vals.append(p)
            l_loc = l_loc + p
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # ---- REGISTER-RESIDENT P transpose (no LDS round-trip) ----
        # GEMM2 needs B-operand P[kv, q]; lane needs 8 consecutive kv = s*16 + h*8 + e.
        # GEMM1 left p_vals[i] at kv=(i//4)*8 + h*4 + i%4. Verified mapping: for k-step s,
        # slot_base = s*8 + h*4; lane's own 4 fp8 (slots base..base+3) + peer(lane^32)'s 4,
        # interleaved as [half0 i32, half1 i32]. One cvt + one ds_bpermute per s.
        # Operand byte e for lane (half h, step s): e0-3 from the HALF0 lane, e4-7 from the
        # HALF1 lane, BOTH at slots {s*8 + h*4 + j}. So each lane packs both bases (0 and 4),
        # picks pack_h = (h? pack1 : pack0), and bpermutes pack_h from both the h0 and h1 lanes.
        def _cvt4(v0, v1, v2, v3):
            lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(), fx.Int32(0).ir_value(), False)
            return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(), lo, True)

        is_h0 = half == fx.Int32(0)
        q_byte = q_local * fx.Int32(4)  # half0 lane's byte addr (same q)
        q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)  # half1 lane's byte addr
        p_i64_s = []
        for s in fx.range_constexpr(2):
            pack0 = _cvt4(p_vals[s * 8 + 0], p_vals[s * 8 + 1], p_vals[s * 8 + 2], p_vals[s * 8 + 3])  # base s*8+0
            pack1 = _cvt4(p_vals[s * 8 + 4], p_vals[s * 8 + 5], p_vals[s * 8 + 6], p_vals[s * 8 + 7])  # base s*8+4
            # bpermute BOTH bases from BOTH half-lanes, then pick base by this lane's half h.
            h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
            h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
            h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
            h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
            _wait_lds()
            # operand low i32 = half0 lane's pack at base (h? s*8+4 : s*8+0); high = half1 lane's same.
            w0 = is_h0.select(h0_b0, h0_b1)
            w1 = is_h0.select(h1_b0, h1_b1)
            p_i64_s.append(fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0])

        # rescale running O
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        for dt in fx.range_constexpr(DT):
            o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

        # GEMM2: O[d,q] += V^T @ P. OPT5: bulk-read all 8 V packs first (overlap LDS-read latency),
        # then run the 8 MFMAs across the 4 independent o_acc[dt] chains.
        v_packs = []
        for dt in fx.range_constexpr(DT):
            d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
            for s in fx.range_constexpr(2):
                v_lds_elem = vbuf + d_col * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_lds_elem)])
                v_packs.append(fx.Vector(vv8).bitcast(fx.Int64)[0])
        for dt in fx.range_constexpr(DT):
            acc2 = fx.Vector(o_acc[dt]).ir_value()
            for s in fx.range_constexpr(2):
                v_i64 = v_packs[dt * 2 + s]
                p_i64 = p_i64_s[s]
                a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                b_raw = p_i64.ir_value() if hasattr(p_i64, "ir_value") else p_i64
                acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
            o_acc[dt] = fx.Vector(acc2)
        fx.rocdl.s_setprio(0)
        # OPT3: NOW store the prefetched next tile to LDS (its global load was issued before
        # GEMM1, so by here the data has arrived and the store doesn't stall).
        store_kv_to_lds(kc_w_next, vc_w_next, kbuf_n, vbuf_n)
        fx.gpu.barrier()  # V/K LDS consumed + next-tile stored; safe for next iter

        new_state = [m_new, l_run] + [o_acc[d] for d in range(DT)]
        st = yield new_state

    # results from the runtime loop's final iteration
    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    p_lds = SmemPtr(_alloc.get_base(), _P_OFF, fx.typing.T.i8, shape=(_P_BYTES,)).get()

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
    ).launch(grid=(grid_blocks,), block=(512,), stream=stream)
