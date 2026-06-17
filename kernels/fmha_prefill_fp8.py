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
BM = 32  # q rows per workgroup
BN = 32  # kv per MFMA tile
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_smem")
_P_BYTES = BM * BN  # P transpose [q*BN + kv] fp8 (1 byte each)
_P_OFF = 0
_alloc.ptr = _P_OFF + _P_BYTES


def const_false(x):
    return fx.const_expr(x == 0)


@flyc.kernel
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
    lane = fx.Int32(fx.thread_idx.x)
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

    qrow = qtile * fx.Int32(BM) + q_local
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

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]

    ps_i = fx.Int32(page_size)
    n_kv_rt = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)  # runtime kv-tile count
    init_state = [m_run0, l_run0] + o_acc0
    for kt_iv, st in range(fx.Index(0), fx.Index(n_kv_rt), fx.Index(1), init=init_state):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt_iv) * fx.Int32(BN)
        kv_local = lane % fx.Int32(32)
        kvrow = kv0 + kv_local
        kvrow_safe = (kvrow < sk_i).select(kvrow, fx.Int32(0))
        # paged: slot = page0 + kv//ps; phys = LTD[slot]; intra = kv%ps
        k_slot = page0 + kvrow_safe // ps_i
        k_phys = fx.buffer_ops.buffer_load(rltd, k_slot, vec_width=1, dtype=fx.Int32)
        k_intra = kvrow_safe % ps_i
        # base byte offset of this kv's feature group 0 in the K pool (vec_k_col_v)
        k_tok_base = k_phys * k_page_stride + k_head_off + k_intra * fx.Int32(16)

        # GEMM1: S[kv,q] = K @ Q^T
        acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        for ks in fx.range_constexpr(KSTEPS):
            # group cg=ks (16 features); pick low/high 8 via half. stride ps*16 per group.
            off = k_tok_base + fx.Int32(ks) * (ps_i * fx.Int32(16)) + half * fx.Int32(8)
            kw = fx.buffer_ops.buffer_load(rk, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
            k_i64 = fx.Vector(kw).bitcast(fx.Int64)[0]
            a_raw = k_i64.ir_value() if hasattr(k_i64, "ir_value") else k_i64
            b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
            acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
        sv = fx.Vector(acc_raw)

        # descale + causal mask
        s_vals = []
        for i in fx.range_constexpr(16):
            kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            kvalid = kv < sk_i
            kd_idx = (batch * fx.Int32(nk) + kvhead) * sk_i + kvalid.select(kv, fx.Int32(0))
            k_descale = fx.buffer_ops.buffer_load(rkd, kd_idx, vec_width=1, dtype=fx.Float32)
            s = sv[i] * (q_descale * k_descale * fx.Float32(sm_scale))
            masked = kvalid if const_false(causal) else (kvalid & (kv <= qrow + (sk_i - sq_i)))
            s_vals.append(masked.select(s, neg_inf))

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
        corr = fx.Float32(((m_run - safe_m) * fx.Float32(LOG2E)).exp2())
        corr = m_is_neg.select(fx.Float32(0.0), corr)

        # p = 2^((s - m_new)*log2e); l_tile = sum p (cross-half)
        l_loc = fx.Float32(0.0)
        p_vals = []
        for i in fx.range_constexpr(16):
            # p_scaled = 2^((s-m)*log2e + log2_pscale) = p_scale * 2^((s-m)*log2e)
            p = fx.Float32(((s_vals[i] - safe_m) * fx.Float32(LOG2E) + log2_pscale).exp2())
            p_vals.append(p)
            l_loc = l_loc + p
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # store P -> LDS, q-major fp8: p_lds[q_local*BN + kv_in_tile].
        # Convert the lane's 16 p (f32) to fp8 as one vector, then store each.
        # f32 -> fp8 via ROCDL cvt_pk_fp8_f32 (packs 2 f32 -> 2 fp8 into an i32).
        p_words = []
        for w in fx.range_constexpr(4):
            lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p_vals[w * 4 + 0].ir_value(), p_vals[w * 4 + 1].ir_value(), fx.Int32(0).ir_value(), False)
            hi = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p_vals[w * 4 + 2].ir_value(), p_vals[w * 4 + 3].ir_value(), lo, True)
            p_words.append(hi)
        p_i8_vec = fx.Vector(fx.Vector.from_elements(p_words, fx.Int32)).bitcast(fx.Int8)
        for i in fx.range_constexpr(16):
            kv_in_tile = fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            fx.Vector.from_elements([fx.Vector(p_i8_vec)[i]], fx.Int8).store(
                p_lds, [fx.Index(q_local * fx.Int32(BN) + kv_in_tile)]
            )
        fx.gpu.barrier()

        # rescale running O
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        for dt in fx.range_constexpr(DT):
            o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

        # GEMM2: O[d,q] += V^T @ P. P operand (B) = p_lds[q=col, kv=k]; V operand (A)=V[kv,d].
        # MFMA K=16 over kv (2 steps of 8? BN=32 kv -> need kv as contraction of 32 => 2 MFMA k-steps of 16).
        # Reload P from LDS as B operand: lane holds B[k=(half)*8+e? ...]. Use mfma convention:
        # B[k, col=q=lane%32]; k = half*8 + e. So per k-step s in {0,1}: kv = s*16 + half*8 + e.
        for dt in fx.range_constexpr(DT):
            acc2 = fx.Vector(o_acc[dt]).ir_value()
            for s in fx.range_constexpr(2):  # 2 k-steps of 16 over BN=32 kv
                # B = P: 8 contiguous fp8 = p_lds[q_local*BN + (s*16 + half*8 + e)], e=0..7
                p_base = q_local * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                p_vec8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), p_lds, [fx.Index(p_base)])
                p_i64 = fx.Vector(p_vec8).bitcast(fx.Int64)[0]
                # A = V: V[kv = s*16 + half*8 + e, d = dt*32 + lane%32] (paged row-major
                # [pages, ps, nk, hd]: vidx = v_phys*v_page_stride + (kv%ps)*nk*hd + v_head_off + d)
                d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
                vb = []
                for e in fx.range_constexpr(8):
                    kv = kv0 + fx.Int32(s * 16) + half * fx.Int32(8) + fx.Int32(e)
                    kvalid = kv < sk_i
                    kvv = kvalid.select(kv, fx.Int32(0))
                    v_slot = page0 + kvv // ps_i
                    v_phys = fx.buffer_ops.buffer_load(rltd, v_slot, vec_width=1, dtype=fx.Int32)
                    vidx = v_phys * v_page_stride + (kvv % ps_i) * v_tok_stride + v_head_off + d_col
                    vb.append(kvalid.select(fx.buffer_ops.buffer_load(rv, vidx, vec_width=1, dtype=fx.Int8), fx.Int8(0)))
                vdw = []
                for d2 in fx.range_constexpr(2):
                    w = fx.Int32(0)
                    for j in fx.range_constexpr(4):
                        bj = fx.Int32(vb[d2 * 4 + j]) & fx.Int32(0xFF)
                        w = w | (bj << fx.Int32(8 * j))
                    vdw.append(w)
                v_i64 = fx.Vector.from_elements(vdw, fx.Int32).bitcast(fx.Int64)[0]
                a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                b_raw = p_i64.ir_value() if hasattr(p_i64, "ir_value") else p_i64
                acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
            o_acc[dt] = fx.Vector(acc2)
        fx.gpu.barrier()

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
    for dt in fx.range_constexpr(DT):
        ov = fx.Vector(o_acc[dt]) * scale_vec
        ov_bf16 = fx.Vector(ov).to(fx.BFloat16)
        for i in fx.range_constexpr(16):
            d = fx.Int32(dt * 32) + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            oval = fx.Vector(ov_bf16)[i]
            o_idx = ((batch * sq_i + qrow) * fx.Int32(nq) + qhead) * fx.Int32(HD) + d
            o_idx_s = in_b.select(o_idx, fx.Int32(0))
            fx.buffer_ops.buffer_store(in_b.select(oval, fx.BFloat16(0.0)), ro, o_idx_s)


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
    ).launch(grid=(grid_blocks,), block=(64,), stream=stream)
