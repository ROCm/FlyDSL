# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL Paged Attention Decode with Persistent Scheduling — FP8.

Persistent scheduling (PS) mode:
- Grid = (num_SM, 1, 4) so each CTA handles one 256-token sub-tile of a 1024-token KV page
- Outer work loop iterates over pre-computed worklist from get_pa_metadata_v1
- Inner KV loop iterates pages from kv_page_indices
- Supports split-reduce for load balancing across CUs

Requires: aiter's get_pa_metadata_v1 (module_pa_metadata.so)
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.attention.pa_common import _prefetch_q_chunks
from kernels.attention.pa_decode_swa import compile_pa_decode_sw, compile_pa_decode_sw_reduce
from kernels.attention.pa_decode_tile import pa_decode_tile
from kernels.attention.pa_metadata import compile_pa_decode_metadata
from kernels.common import dpp_utils
from kernels.common.tensor_shim import _run_compiled
from kernels.common.utils import (
    cdiv,
    exp2_f32_fast,
    rcp_f32,
    udiv_const,
    urem_const,
)

# ── Kernel geometry constants ────────────────────────────────────────
KV_BLOCK_SIZE = 1024  # physical page size (matches SP3 kBlockSize)
KV_COMPUTE_BLOCK = 256  # tile size (matches SP3 kTileKV)
# Persistent-grid oversubscription for the metadata decode path: launch
# CU_count * this many workgroups so the HW keeps multiple workgroups resident
# per CU (memory-latency hiding).  1 = original (1 wg/CU).
_PA_METADATA_GRID_OVERSUB = 3
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256
MFMA_N = 16

TOKENS_PER_WARP = KV_COMPUTE_BLOCK // NUM_WARPS  # 64
TLOOP = TOKENS_PER_WARP // MFMA_N  # 4
ROWS_PER_WARP = WARP_SIZE // MFMA_N  # 4
FP8_ELEMS_16B = 16  # 16 FP8 per 16-byte load
QKHE_PER_FETCH = FP8_ELEMS_16B * ROWS_PER_WARP  # 64

VTLOOP = NUM_WARPS  # 4
Q_ELEMS_PER_LANE = 8
Q_CHUNKS_PER_LANE = Q_ELEMS_PER_LANE // 4

# LDS sizes
PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4  # 512
LDS_SCALE_V_PADDING = 4  # break K/V same-bank paired writes
LDS_SCALE_V_OFFSET = KV_COMPUTE_BLOCK + LDS_SCALE_V_PADDING
LDS_SCALE_BYTES = (LDS_SCALE_V_OFFSET + KV_COMPUTE_BLOCK) * 4  # K/V per-token scale staging

FP8_MAX = 240.0
LOG2E = 1.4426950408889634

_PACKED_FP8_QUERY_DTYPES = tuple(
    dtype
    for dtype in (
        torch.uint8,
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e4m3fn", None),
    )
    if dtype is not None
)


def _flatten_v_results(v_results, vhe_loop: int = 2):
    """v_results[vt][vhe] = i64x2 → flat list of 2 * VTLOOP * vhe_loop scalar i64
    values, in the same order ``_unflatten_v_results`` expects.  Used to carry
    V data through scf.for state (which only accepts scalar values)."""
    flat = []
    for vt in range(VTLOOP):
        for vhe in range(vhe_loop):
            v_i64x2 = fx.Vector(v_results[vt][vhe])
            flat.append(v_i64x2[0])
            flat.append(v_i64x2[1])
    return flat


def _unflatten_v_results(v_flat, vhe_loop: int = 2):
    """Inverse of ``_flatten_v_results``: rebuild v_results[vt][vhe] = i64x2."""
    v_results = []
    idx = 0
    for vt in range(VTLOOP):
        vhe_data = []
        for vhe in range(vhe_loop):
            v_i64x2 = vector.from_elements(T.vec(2, T.i64), [v_flat[idx], v_flat[idx + 1]])
            vhe_data.append(v_i64x2)
            idx += 2
        v_results.append(vhe_data)
    return v_results


def _build_pa_thread_invariants(
    warp_id,
    lane16id,
    rowid,
    *,
    per_token_kv,
):
    c_tokens_per_warp = fx.Int32(TOKENS_PER_WARP)
    kv_tok_thread_base = warp_id * c_tokens_per_warp + rowid * 4
    rowid_8x8 = rowid >> fx.Int32(1)
    offset_in_slot = rowid & fx.Int32(1)
    prob_wr_thread_base = (
        warp_id * fx.Int32(4 * MFMA_N * PROB_ROW_STRIDE_BYTES)
        + lane16id * fx.Int32(PROB_ROW_STRIDE_BYTES)
        + rowid_8x8 * fx.Int32(8)
        + offset_in_slot * 4
    )
    pv_prob_read_base = rowid * fx.Int32(MFMA_N * PROB_ROW_STRIDE_BYTES) + lane16id * fx.Int32(PROB_ROW_STRIDE_BYTES)

    sm_lane_wave_base = lane16id * fx.Int32(NUM_WARPS)
    sm_max_off = fx.Index(sm_lane_wave_base + warp_id)
    sm_sum_off = fx.Index(fx.Int32(NUM_WARPS * MFMA_N) + sm_lane_wave_base + warp_id)
    sm_rd_max_offs = [fx.Index(sm_lane_wave_base + fx.Int32(w)) for w in range(NUM_WARPS)]
    sm_rd_sum_offs = [
        fx.Index(fx.Int32(NUM_WARPS * MFMA_N) + sm_lane_wave_base + fx.Int32(w)) for w in range(NUM_WARPS)
    ]

    sm_vmax_wr_off = None
    sm_vmax_rd_offs = None
    if const_expr(per_token_kv):
        sm_vmax_wr_off = fx.Index(fx.Int32(2 * NUM_WARPS * MFMA_N) + sm_lane_wave_base + warp_id)
        sm_vmax_rd_offs = [
            fx.Index(fx.Int32(2 * NUM_WARPS * MFMA_N) + sm_lane_wave_base + fx.Int32(w)) for w in range(NUM_WARPS)
        ]

    return (
        kv_tok_thread_base,
        prob_wr_thread_base,
        pv_prob_read_base,
        sm_max_off,
        sm_sum_off,
        sm_rd_max_offs,
        sm_rd_sum_offs,
        sm_vmax_wr_off,
        sm_vmax_rd_offs,
    )


def _compute_mtp_group_state(
    lane16id,
    local_qhead_idx,
    *,
    mtp_group_idx,
    query_length,
    query_group_size,
):
    g_off = mtp_group_idx * 16
    lane_pair_raw = lane16id + fx.Int32(g_off)
    c_total_pairs = fx.Int32(query_length * query_group_size)
    c_pair_max = fx.Int32(query_length * query_group_size - 1)
    c_ql_m1 = fx.Int32(query_length - 1)

    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        lane_pair = lane_pair_raw
    else:
        lane_pair = arith.select(lane_pair_raw < c_total_pairs, lane_pair_raw, c_pair_max)
    qi_raw = udiv_const(lane_pair, query_group_size)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        qi_val = qi_raw
    else:
        qi_val = arith.select(qi_raw < c_ql_m1, qi_raw, c_ql_m1)
    qhi_pos = urem_const(lane_pair, query_group_size)

    lqh_pair_raw = local_qhead_idx + fx.Int32(g_off)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        lqh_pair = lqh_pair_raw
    else:
        lqh_pair = arith.select(lqh_pair_raw < c_total_pairs, lqh_pair_raw, c_pair_max)
    lqi_raw = udiv_const(lqh_pair, query_group_size)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        qi_for_q = lqi_raw
    else:
        qi_for_q = arith.select(lqi_raw < c_ql_m1, lqi_raw, c_ql_m1)
    local_qhead_idx_for_q = urem_const(lqh_pair, query_group_size)
    return qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q


@flyc.jit
def _finish_q_fragments(
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    q_chunks,
    lane16id,
    rowid,
    local_qhead_idx,
    *,
    head_size: int,
    qkhe_loop: int,
    q_lanes_per_head: int,
):
    # LDS Q layout (compact, per-qhead contiguous):
    #   Q[head=h][hd=d]  at byte offset  h * HEAD_SIZE + d   (FP8 after conversion)
    # Total Q footprint = 16 qheads * HEAD_SIZE bytes, aliased with the later P
    # writes via `logits_lds_i32 / logits_lds_i64` (same base).  For HEAD_SIZE=64,
    # only the first 8 lanes write Q for each qhead.
    #
    # Writer: thread (warp_id W, rowid R', lane16id L') owns qhead = W*4 + R' =
    # `local_qhead_idx`, and within that qhead owns the 8 FP8 elements at
    # head_dim [L'*8 .. L'*8+7].  We therefore write 2 i32 words (= 1 i64 = 8 B)
    # at `local_qhead_idx * HEAD_SIZE + lane16id * 8`.
    #
    # Reader: MFMA lane layout for mfma_f32_16x16x32_fp8_fp8 (B = Q^T, N = qhead,
    # K = head_dim) — reverse-engineered from `_load_k_flat`: thread (rowid R,
    # lane16id L) consumes, for k_step = qkhe*2 + qkr,
    #   Q[head = L][hd = (qkhe*4 + R) * 16 + qkr * 8 + 0..7]
    # i.e. the read byte offset is `L * HEAD_SIZE + qkhe*64 + R*16 + qkr*8`.
    c_head_size = fx.Int32(head_size)
    lds_q_base = local_qhead_idx * c_head_size + lane16id * 8
    abs_mask = fx.Vector.filled(4, 0x7FFFFFFF, fx.Int32)
    c_zero_f = fx.Float32(0.0)
    c_one_f = fx.Float32(1.0)

    q_f32_chunks = []
    local_max = c_zero_f
    for q_src in q_chunks:
        q_f32 = fx.Vector(q_src).to(fx.Float32)
        q_f32_chunks.append(q_f32)
        q_i32 = q_f32.bitcast(fx.Int32)
        q_abs_i32 = q_i32 & abs_mask
        q_abs = q_abs_i32.bitcast(fx.Float32)
        chunk_max = q_abs.reduce("max")
        local_max = fx.maxnumf(local_max, chunk_max)

    for sh in [8, 4, 2, 1]:
        local_max = fx.maxnumf(local_max, dpp_utils.dpp_xor_f32(local_max, sh))
    query_scale_lane = fx.Float32(
        arith.select(
            local_max > c_zero_f,
            local_max * fx.Float32(1.0 / FP8_MAX).ir_value(),
            c_one_f,
        )
    )
    inv_query_scale = rcp_f32(query_scale_lane)
    q_words = []
    for q_f32 in q_f32_chunks:
        p = q_f32 * inv_query_scale
        lo = rocdl.cvt_pk_fp8_f32(T.i32, p[0], p[1], fx.Int32(0), False)
        q_words.append(rocdl.cvt_pk_fp8_f32(T.i32, p[2], p[3], lo, True))
    q_w0, q_w1 = q_words

    if lane16id == fx.Int32(0):
        fx.Vector.from_elements([query_scale_lane], dtype=fx.Float32).store(
            softmax_lds_f32, [fx.Index(local_qhead_idx)]
        )

    v01 = fx.Vector.from_elements([q_w0, q_w1], dtype=fx.Int32)
    lds_q_i32 = lds_q_base >> fx.Int32(2)
    if const_expr(q_lanes_per_head < MFMA_N):
        if lane16id < fx.Int32(q_lanes_per_head):
            v01.store(logits_lds_i32, [fx.Index(lds_q_i32)])
    else:
        v01.store(logits_lds_i32, [fx.Index(lds_q_i32)])

    q_frags = []
    gpu.barrier()
    query_scale_lane = fx.Vector.load(T.vec(1, fx.Float32.ir_type), softmax_lds_f32, [fx.Index(lane16id)])[0].ir_value()
    for qkhe in range_constexpr(qkhe_loop):
        for qkr in range_constexpr(2):
            # See layout comment above. Byte offset:
            #   lane16id * HEAD_SIZE + qkhe*64 + rowid*16 + qkr*8
            lds_rd_byte = lane16id * c_head_size + fx.Int32(qkhe << 6) + (rowid << fx.Int32(4)) + fx.Int32(qkr << 3)
            lds_rd_base = lds_rd_byte >> fx.Int32(3)
            q_v1 = fx.Vector.load(T.vec(1, T.i64), logits_lds_i64, [fx.Index(lds_rd_base)])
            q_frags.append(q_v1[0])
    return q_frags, query_scale_lane


def _prefetch_mtp_group_query(
    q_rsrc,
    batch_idx,
    kv_h,
    stride_q_seq,
    stride_q_head,
    lane16id,
    local_qhead_idx,
    *,
    mtp_group_idx,
    query_length,
    query_group_size,
    query_load_is_bf16,
    q_lanes_per_head,
):
    qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q = _compute_mtp_group_state(
        lane16id,
        local_qhead_idx,
        mtp_group_idx=mtp_group_idx,
        query_length=query_length,
        query_group_size=query_group_size,
    )
    q_row = batch_idx * arith.constant(query_length, type=T.i32) + qi_for_q
    q_base = (
        q_row * stride_q_seq
        + (kv_h * arith.constant(query_group_size, type=T.i32) + local_qhead_idx_for_q) * stride_q_head
    )
    q_chunks = _prefetch_q_chunks(
        q_rsrc,
        q_base,
        lane16id,
        query_load_is_bf16=query_load_is_bf16,
        q_lanes_per_head=q_lanes_per_head,
    )
    return qi_val, qhi_pos, q_chunks


def _finish_mtp_group_q_fragments(
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    mtp_prefetch,
    lane16id,
    rowid,
    local_qhead_idx,
    *,
    head_size: int,
    qkhe_loop: int,
    q_lanes_per_head: int,
):
    qi_val, qhi_pos, q_chunks = mtp_prefetch
    q_frags, query_scale_lane = _finish_q_fragments(
        logits_lds_i32,
        logits_lds_i64,
        softmax_lds_f32,
        q_chunks,
        lane16id,
        rowid,
        local_qhead_idx,
        head_size=head_size,
        qkhe_loop=qkhe_loop,
        q_lanes_per_head=q_lanes_per_head,
    )
    return qi_val, qhi_pos, q_frags, query_scale_lane


def _normalize_pa_output(running_sum, outs, zero_f):
    one_f = fx.Float32(1.0).ir_value()
    safe_sum = arith.select(running_sum > zero_f, running_sum, one_f)
    inv_sum = rcp_f32(safe_sum)
    inv_sum_vec = vector.broadcast(T.f32x4, inv_sum)
    return [out * inv_sum_vec for out in outs]


@flyc.jit
def _make_pa_phase_helpers(
    *,
    per_token_q,
    per_token_kv,
    needs_mask,
    query_length,
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    scale_lds_f32,
    softmax_scale_base,
    softmax_q_scale,
    k_scale_val,
    scale,
    v_scale_val,
    warp_id,
    lane16id,
    rowid,
    kv_tok_thread_base,
    prob_wr_thread_base,
    pv_prob_read_base,
    sm_max_off,
    sm_sum_off,
    sm_rd_max_offs,
    sm_rd_sum_offs,
    sm_vmax_wr_off,
    sm_vmax_rd_offs,
    c_w,
    neg_inf,
    zero_f,
    qkhe_loop: int = 2,
    vhe_loop: int = 2,
):
    apply_causal_mask = needs_mask or query_length > 1
    pv_prob_i64_indices = []
    for vt in range_constexpr(VTLOOP):
        for j in range_constexpr(2):
            p_byte = (
                arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                + pv_prob_read_base
                + arith.constant(j * 8, type=T.i32)
            )
            pv_prob_i64_indices.append(fx.Index(p_byte >> fx.Int32(3)))

    def _scale_row_base(td: int):
        return kv_tok_thread_base + fx.Int32(td * MFMA_N)

    def _load_k_scale_vec(td: int):
        return vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(_scale_row_base(td))])

    def _load_v_scale_vec(td: int):
        return vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + _scale_row_base(td))])

    def _get_k_scale_vec(td: int, k_scale_vecs=None):
        if const_expr(per_token_kv):
            return k_scale_vecs[td]
        return _load_k_scale_vec(td)

    def _get_v_scale_vec(td: int, v_scale_vecs=None):
        if const_expr(per_token_kv):
            return v_scale_vecs[td]
        return _load_v_scale_vec(td)

    def _store_vmax_warp(partition_start, *, seq_end=None, v_scale_vecs=None):
        if const_expr(per_token_kv):
            kv_tok_base = partition_start + kv_tok_thread_base if const_expr(seq_end is not None) else None
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = _get_v_scale_vec(td, v_scale_vecs)
                for i in range_constexpr(4):
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        vs_i = arith.select(kv_tok < seq_end, vs_i, zero_f)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                v_max_warp = fx.maxnumf(v_max_warp, fx.Vector(vs).reduce("max"))
            for sh in [32, 16]:
                v_max_warp = fx.maxnumf(v_max_warp, v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            vector.store(
                fx.Vector.from_elements([v_max_warp], dtype=fx.Float32),
                softmax_lds_f32,
                [sm_vmax_wr_off],
            )

    def _token_vec_i32(kv_tok_base, td: int):
        kv_tok_td_base = kv_tok_base + arith.constant(td * MFMA_N, type=T.i32)
        return fx.Vector.from_elements(
            [kv_tok_td_base + arith.constant(i, type=T.i32) for i in range_constexpr(4)],
            dtype=fx.Int32,
        )

    def _apply_token_mask_vec(logit_vec, td: int, kv_tok_base, causal_bound, false_value):
        tok_vec = _token_vec_i32(kv_tok_base, td)
        if const_expr(apply_causal_mask):
            in_range = tok_vec < causal_bound
            return arith.select(in_range, logit_vec, vector.broadcast(T.f32x4, arith.unwrap(false_value)))
        return logit_vec

    def _qk_and_intra_softmax(
        k_ops,
        partition_start,
        q_frags,
        causal_bound,
        query_scale_lane=None,
        *,
        preloaded_scales=None,
    ):
        if const_expr(preloaded_scales is not None):
            k_scale_vecs, v_scale_vecs = preloaded_scales

        query_scale_vec = None
        if const_expr(per_token_q):
            query_scale_vec = vector.broadcast(T.f32x4, query_scale_lane * softmax_scale_base)
        d_out = []
        for td in range_constexpr(TLOOP):
            acc = arith.constant_vector(0.0, T.f32x4)
            for k_step in range_constexpr(qkhe_loop * 2):
                acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
            if const_expr(per_token_kv):
                k_scale_vec = _get_k_scale_vec(td, k_scale_vecs)
                scale_vec = (
                    k_scale_vec * query_scale_vec
                    if const_expr(per_token_q)
                    else k_scale_vec * vector.broadcast(T.f32x4, softmax_q_scale)
                )
                d_out.append(acc * scale_vec)
            else:
                if const_expr(per_token_q):
                    d_out.append(acc * (query_scale_vec * vector.broadcast(T.f32x4, k_scale_val)))
                else:
                    d_out.append(acc * vector.broadcast(T.f32x4, scale))

        kv_tok_base = partition_start + kv_tok_thread_base if const_expr(apply_causal_mask) else None
        qk_max = neg_inf
        for td in range_constexpr(TLOOP):
            logits_vec = d_out[td]
            if const_expr(kv_tok_base is not None):
                logits_vec = _apply_token_mask_vec(logits_vec, td, kv_tok_base, causal_bound, neg_inf)
                d_out[td] = logits_vec
            qk_max = fx.maxnumf(qk_max, fx.Vector(logits_vec).reduce("max"))
        for sh in [32, 16]:
            qk_max = fx.maxnumf(qk_max, qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
        vector.store(
            fx.Vector.from_elements([qk_max], dtype=fx.Float32),
            softmax_lds_f32,
            [sm_max_off],
        )

        if const_expr(per_token_kv):
            return d_out, v_scale_vecs
        return d_out

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs, v_scale_vecs):
        partition_max = neg_inf
        partition_sum = zero_f
        max_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_rd_max_offs[0]]))
        for w in range_constexpr(NUM_WARPS):
            partition_max = fx.maxnumf(partition_max, max_vec[w])

        new_rmax = fx.maxnumf(rmax, partition_max)
        safe_eff_max = arith.select(partition_max > neg_inf, new_rmax, zero_f) if const_expr(needs_mask) else new_rmax
        local_exp_sum = zero_f
        for td in range_constexpr(TLOOP):
            diff_vec = fx.Vector(d_out[td]) - vector.broadcast(T.f32x4, arith.unwrap(safe_eff_max))
            p_vec = exp2_f32_fast(diff_vec * vector.broadcast(T.f32x4, arith.unwrap(fx.Float32(LOG2E))))
            local_exp_sum = local_exp_sum + fx.Vector(p_vec).reduce("add")
            d_out[td] = p_vec
        for sh in [32, 16]:
            local_exp_sum = local_exp_sum + local_exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
        vector.store(
            fx.Vector.from_elements([local_exp_sum], dtype=fx.Float32),
            softmax_lds_f32,
            [sm_sum_off],
        )
        if const_expr(needs_mask):
            accum_scale = arith.select(
                rmax > neg_inf,
                exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value()),
                zero_f,
            )
        else:
            accum_scale = exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value())

        gpu.barrier()
        sum_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_rd_sum_offs[0]]))
        for w in range_constexpr(NUM_WARPS):
            partition_sum = arith.addf(
                arith.unwrap(partition_sum), arith.unwrap(sum_vec[w]), fastmath=arith.FastMathFlags.contract
            )

        accum_sum = arith.mulf(arith.unwrap(accum_scale), arith.unwrap(rsum), fastmath=arith.FastMathFlags.contract)
        rsum = arith.addf(accum_sum, arith.unwrap(partition_sum), fastmath=arith.FastMathFlags.contract)
        rmax = new_rmax
        accum_scale_vec = vector.broadcast(T.f32x4, arith.unwrap(accum_scale))
        for vhe in range_constexpr(vhe_loop):
            outs[vhe] = outs[vhe] * accum_scale_vec

        if const_expr(per_token_kv):
            v_max_global = zero_f
            vmax_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_vmax_rd_offs[0]]))
            for w in range_constexpr(NUM_WARPS):
                w_vmax = vmax_vec[w]
                v_max_global = fx.maxnumf(v_max_global, w_vmax)
            v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX).ir_value()
            v_max_safe_scaled = v_max_scaled + fx.Float32(1e-8 / FP8_MAX).ir_value()
            norm_factor = rcp_f32(v_max_safe_scaled)
            v_correction = v_max_scaled
            _vec_norm_p = arith.unwrap(norm_factor)
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * (_get_v_scale_vec(td, v_scale_vecs) * vector.broadcast(T.f32x4, _vec_norm_p))
        else:
            v_correction = v_scale_val

        for td in range_constexpr(TLOOP):
            p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
            p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
            p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
            p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
            lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
            pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
            byte_base = prob_wr_thread_base + arith.constant(td * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
            i32_off = byte_base >> fx.Int32(2)
            pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
            vector.store(pk_vec, logits_lds_i32, [fx.Index(i32_off)])
        return rmax, rsum, outs, v_correction

    def _pv_mfma(v_ops, outs, v_correction):
        v_correction = fx.Float32(v_correction).ir_value()
        fm_contract = arith.FastMathFlags.contract
        v_correction_vec = vector.broadcast(T.f32x4, v_correction)

        # ── Batch-load all P_i64 from LDS upfront ──
        # `p_i64` depends only on (vt, j), NOT on vhe, so the previous
        # per-vhe inner LDS load was redundant: VHELOOP × VTLOOP*2 reads
        # of the same VTLOOP*2 LDS slots.  Issue all VTLOOP*2 ds_read_b64
        # ops once at the start so the compiler pipelines them — lgkmcnt
        # drains during the address arithmetic before the MFMA chain.
        p_i64_all = []
        for vt in range_constexpr(VTLOOP):
            for j in range_constexpr(2):
                p_i64_idx = pv_prob_i64_indices[vt * 2 + j]
                p_i64_all.append(fx.Vector.load(T.vec(1, T.i64), logits_lds_i64, [p_i64_idx])[0])

        for vhe in range_constexpr(vhe_loop):
            tmp_out = arith.constant_vector(0.0, T.f32x4)
            for vt in range_constexpr(VTLOOP):
                v_i64x2 = fx.Vector(v_ops[vt][vhe])
                for j in range_constexpr(2):
                    tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4,
                        [
                            v_i64x2[j],
                            p_i64_all[vt * 2 + j],
                            tmp_out,
                            0,
                            0,
                            0,
                        ],
                    )
            outs[vhe] = arith.addf(
                arith.mulf(tmp_out, v_correction_vec, fastmath=fm_contract),
                outs[vhe],
                fastmath=fm_contract,
            )
        return outs

    return (
        _store_vmax_warp,
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
    )


# =====================================================================
# Launch API — Persistent Scheduling mode
# =====================================================================


def get_pa_metadata(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_indptr: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    partition_size: int = KV_COMPUTE_BLOCK,
):
    """Compute PA metadata (worklist, reduce maps) via get_pa_metadata_v1.

    The worklist is now load-balanced at **partition** granularity
    (``partition_size`` tokens, default ``KV_COMPUTE_BLOCK=256``) rather than at
    physical block granularity: ``kv_granularity = partition_size``, so each
    scheduled work unit is one partition and ``work_info.kv_start/kv_end`` are
    cumulative **partition** indices (in ``partition_size``-token units), not
    page indices. The partition↔block relationship for the consumer is:
    ``partition_size > block_size`` → ``partition_size // block_size`` blocks per
    partition; otherwise ``block_size // partition_size`` partitions per block.

    NOTE: the consuming decode kernel must interpret kv_start/kv_end as partition
    indices accordingly.

    Returns a dict with: work_indptr, work_info_flat, reduce_indptr,
    reduce_final_map, reduce_partial_map, num_sm, partial_output,
    partial_lse, stride_po_partial, stride_pl_partial.
    """
    from kernels.attention.pa_metadata import get_pa_metadata_info_v1, get_pa_metadata_v1

    dev = query.device
    batch_size = context_lengths.shape[0]
    query_length = query.shape[0] // batch_size
    head_size = query.shape[-1]

    props = torch.cuda.get_device_properties(dev)
    # Oversubscribe the persistent grid: the decode kernel is memory-latency-bound
    # and only ~3 workgroups/CU fit by VGPR, but the worklist defaults to 1 wg/CU
    # (grid = CU count).  Distributing work across num_cu = CU_count * OVERSUB bins
    # (and launching that many workgroups) lets the HW keep multiple workgroups
    # resident per CU → more waves in flight → better latency hiding.
    base_cu = props.multi_processor_count
    num_sm = base_cu * _PA_METADATA_GRID_OVERSUB
    num_sm = (num_sm // num_kv_heads) * num_kv_heads  # keep divisible by num_kv_heads

    seqlens_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=dev) * query_length

    # Cumulative-partition prefix sum (in partition_size-token units).  The decode
    # kernel needs partition_base[batch] = partition_indptr[batch] to convert a
    # global cumulative partition index (work_info.kv_start/kv_end) into a local
    # within-sequence partition index.
    _parts_per_batch = (context_lengths.to(torch.int32) + (partition_size - 1)) // partition_size
    partition_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
    partition_indptr[1:] = torch.cumsum(_parts_per_batch, dim=0).to(torch.int32)

    block_size = key_cache.shape[-2] if len(key_cache.shape) == 5 else key_cache.shape[-2]

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_pa_metadata_info_v1(batch_size, num_kv_heads, num_cu=num_sm)

    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=dev)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=dev)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type, device=dev)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=dev)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=dev)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=dev)

    get_pa_metadata_v1(
        seqlens_qo_indptr,
        kv_indptr,
        context_lengths,
        num_query_heads // num_kv_heads,
        num_kv_heads,
        True,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=partition_size,
        block_size=block_size,
        max_seqlen_qo=query_length,
        uni_seqlen_qo=query_length,
        fast_mode=True,
        max_split_per_batch=-1,
        num_cu=num_sm,
    )

    # The FlyDSL get_pa_metadata_v1 produces the reduce_* maps natively
    # (faithful to the C++ kernel), so work_info / reduce_* are consumed directly
    # (no post-hoc expansion). work_info.kv_start/kv_end are partition indices and
    # work_info[:,1] (partial_qo_loc) is -1 for direct works or a partition-row
    # offset for split works.
    work_info_flat = work_info.reshape(-1).contiguous()

    # Number of partial slots = reduce_indptr[-1] (= last_reduce_indptr). Each
    # split partial occupies query_length rows in the partial buffer.
    num_partials = int(reduce_indptr[-1].item())
    max_qlen = query_length
    partial_output = torch.empty(
        ((num_partials + 1) * max_qlen, 1, num_query_heads, head_size), dtype=torch.float32, device=dev
    )
    partial_lse = torch.empty(((num_partials + 1) * max_qlen, 1, num_query_heads, 1), dtype=torch.float32, device=dev)

    stride_po_partial = query_length * num_query_heads * head_size
    stride_pl_partial = query_length * num_query_heads
    stride_po_ql = num_query_heads * head_size
    stride_pl_ql = num_query_heads

    return {
        "work_indptr": work_indptr,
        "work_info_flat": work_info_flat,
        "partition_indptr": partition_indptr,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "num_sm": num_sm,
        "partial_output": partial_output,
        "partial_lse": partial_lse,
        "stride_po_partial": stride_po_partial,
        "stride_pl_partial": stride_pl_partial,
        "stride_po_ql": stride_po_ql,
        "stride_pl_ql": stride_pl_ql,
        "query_length": query_length,
    }


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _prepare_scale_tensor(
    name: str,
    scale,
    *,
    device: torch.device,
    is_graph_capturing: bool,
) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        if is_graph_capturing:
            if scale.device != device:
                raise ValueError(
                    f"CUDA graph capture requires `{name}` to already be on {device}, " f"got {scale.device}."
                )
            if scale.dtype != torch.float32:
                raise ValueError(f"CUDA graph capture requires `{name}` to already be float32, " f"got {scale.dtype}.")
            return scale
        return scale.to(device=device, dtype=torch.float32)

    if is_graph_capturing:
        raise ValueError(
            f"CUDA graph capture requires `{name}` to be passed as a pre-created "
            "float32 tensor on the target device."
        )

    return torch.tensor([float(scale or 1.0)], device=device, dtype=torch.float32)


def _get_query_input_dtype(query: torch.Tensor) -> str:
    if query.dtype in _PACKED_FP8_QUERY_DTYPES:
        return "packed_fp8"
    if query.dtype == torch.bfloat16:
        return "bf16"
    if query.dtype == torch.float16:
        return "f16"
    raise ValueError(
        f"Unsupported query dtype for pa_decode_ps_launch: {query.dtype}. " "Expected packed FP8/uint8, bf16, or f16."
    )


def _get_output_dtype_str(output: torch.Tensor) -> str:
    if output.dtype == torch.bfloat16:
        return "bf16"
    if output.dtype == torch.float16:
        return "f16"
    if output.dtype == torch.float32:
        return "f32"
    raise ValueError(
        f"Unsupported output dtype for pa_decode_ps_launch reduce: {output.dtype}. " "Expected bf16, f16, or f32."
    )


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    *,
    sliding_window: int = 0,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
    max_blocks_per_seq: int | None = None,
    block_size: int = 1,
    device: torch.device | None = None,
    target_ctas_per_cu: int = 8,
    min_tiles_per_partition: int = 2,
    tile_tok: int = 256,
) -> int:
    """Recommend ``max_context_partition_num`` for PS partitioned paths.

    For sliding-window PS, this includes the old
    ``get_sw_ps_max_context_partition_num`` token-window calculation. For
    non-sliding PS, this mirrors ``get_recommended_splits`` in
    ``aiter/ops/triton/gluon/pa_decode_gluon.py`` so FlyDSL callers do not need
    to depend on aiter for the host-side split count.
    """
    if sliding_window > 0:
        window_token_count = sliding_window + query_length
        return cdiv(window_token_count - 1, context_partition_size) + 1

    props = torch.cuda.get_device_properties(device or torch.device("cuda"))
    # Reference uses occupancy = 2 (see `get_occupancy()` in the Gluon module).
    occupancy = 2
    num_sm = props.multi_processor_count * occupancy
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = cdiv(num_sm, denom) * split_kv_blocks
    base_np = max(4, min(n, 8))
    if max_blocks_per_seq is None:
        return base_np

    device_cus = props.multi_processor_count
    cu_fill_np = cdiv(target_ctas_per_cu * device_cus, num_sequences * num_kv_heads)
    max_possible_tiles = cdiv(max_blocks_per_seq * block_size, tile_tok)
    cu_starved = (num_sequences * num_kv_heads) < device_cus
    tiles_np_cap = max_possible_tiles if cu_starved else max(1, max_possible_tiles // min_tiles_per_partition)
    return max(1, min(max(base_np, cu_fill_np), tiles_np_cap))


# Small block_size (16/64) is routed through the load-balanced worklist
# (metadata) path: `compile_pa_decode_metadata` gathers 256//block_size physical
# pages per 256-token partition, for both per-tensor and per-token KV quant.
_PA_DECODE_PS_SMALL_BLOCK_SIZES = (16, 64)


def pa_decode_ps_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,  # [total_pages] int32
    kv_indptr: torch.Tensor,  # [num_seqs + 1] int32
    softmax_scale: float,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    sliding_window: int = 0,
    metadata: dict = None,
    block_tables: torch.Tensor = None,  # [num_seqs, max_blocks_per_seq] i32
    max_context_partition_num: int = 0,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    """Launch PA decode with persistent scheduling.

    Args:
        metadata: Pre-computed metadata dict from get_pa_metadata().
                  If None, calls get_pa_metadata() internally.
    """
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    trans_v = len(value_cache.shape) == 5
    query_input_dtype = _get_query_input_dtype(query)

    dev = query.device
    is_graph_capturing = _is_current_stream_capturing()

    key_scale = _prepare_scale_tensor(
        "key_scale",
        key_scale,
        device=dev,
        is_graph_capturing=is_graph_capturing,
    )
    value_scale = _prepare_scale_tensor(
        "value_scale",
        value_scale,
        device=dev,
        is_graph_capturing=is_graph_capturing,
    )
    if query_input_dtype == "packed_fp8":
        raise ValueError(
            "`pa_decode_ps_launch` no longer accepts host query_scale and only supports "
            "bf16/f16 query inputs with kernel-internal query scale computation."
        )

    # Detect per-token vs per-tensor quantization from scale tensor
    # dimensionality: a >1-D scale tensor carries one scale per (block, head,
    # token), which enables the per-token K/V path in the metadata kernel.
    per_token_kv = key_scale.ndim > 1

    query_length = query.shape[0] // context_lengths.shape[0]
    query_group_size = num_query_heads // num_kv_heads

    # Strides for key_scale/value_scale
    if per_token_kv:
        stride_ks_block = key_scale.stride(0)
        stride_ks_head = key_scale.stride(1)
    else:
        stride_ks_block = 0
        stride_ks_head = 0

    s = stream or torch.cuda.current_stream()

    if sliding_window > 0:
        # Launch one CTA per 256-token context partition in the sliding window:
        # grid = (batch, kv_heads, max_context_partition_num).
        batch_size = context_lengths.shape[0]
        head_size = query.shape[-1]
        eqgs = query_length * query_group_size
        context_partition_size = KV_COMPUTE_BLOCK
        if max_context_partition_num == 0:
            max_context_partition_num = get_recommended_splits(
                batch_size,
                num_kv_heads,
                sliding_window=sliding_window,
                context_partition_size=context_partition_size,
                query_length=query_length,
            )
        if is_graph_capturing and (exp_sums is None or max_logits is None or temporary_output is None):
            raise ValueError(
                "CUDA graph capture requires preallocated `exp_sums`, `max_logits`, "
                "and `temporary_output` for the sliding-window path."
            )
        if exp_sums is None:
            exp_sums = torch.zeros(
                batch_size, num_kv_heads, max_context_partition_num, eqgs, device=dev, dtype=torch.float32
            )
        if max_logits is None:
            max_logits = torch.full(
                (batch_size, num_kv_heads, max_context_partition_num, eqgs),
                float("-inf"),
                device=dev,
                dtype=torch.float32,
            )
        if temporary_output is None:
            temporary_output = torch.zeros(
                batch_size, num_kv_heads, max_context_partition_num, eqgs, head_size, device=dev, dtype=torch.bfloat16
            )

        # The fused SW kernel is useful only when there is no real cross-partition
        # parallelism to exploit.  For the 1023-token window case, one CTA would
        # serialize six 256-token partitions and regress badly versus the
        # partitioned main kernel plus reduce.
        fuse_sw_partitions = max_context_partition_num <= 1
        sw_mtp_groups = (eqgs + MFMA_N - 1) // MFMA_N
        sw_grid_y = num_kv_heads * sw_mtp_groups
        output_5d = output.reshape(batch_size, query_length, num_kv_heads, query_group_size, head_size)

        compiled_sw = compile_pa_decode_sw(
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            trans_v=trans_v,
            query_group_size=query_group_size,
            per_token_kv=per_token_kv,
            query_length=query_length,
            query_input_dtype=query_input_dtype,
            fuse_partitions=fuse_sw_partitions,
            head_dim=int(head_size),
        )

        _run_compiled(
            compiled_sw["launch"],
            exp_sums.data_ptr(),
            max_logits.data_ptr(),
            temporary_output.data_ptr(),
            output_5d.data_ptr(),
            query.data_ptr(),
            key_cache.data_ptr(),
            value_cache.data_ptr(),
            block_tables.data_ptr(),
            context_lengths.data_ptr(),
            key_scale.data_ptr(),
            value_scale.data_ptr(),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            block_tables.stride(0),
            stride_ks_block,
            stride_ks_head,
            batch_size,
            sw_grid_y,
            1 if fuse_sw_partitions else max_context_partition_num,
            s,
        )

        if fuse_sw_partitions:
            return "ps_sw_fused_partitioned"

        compiled_sw_reduce = compile_pa_decode_sw_reduce(
            max_context_partition_num=max_context_partition_num,
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=head_size,
            output_dtype_str=_get_output_dtype_str(output),
        )
        _run_compiled(
            compiled_sw_reduce["launch"],
            output_5d.data_ptr(),
            exp_sums.data_ptr(),
            max_logits.data_ptr(),
            temporary_output.data_ptr(),
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            batch_size,
            num_kv_heads,
            s,
        )
        return "ps_sw_partitioned"

    # ── small-block (block_size 16/64) → tile kernel ──
    # Key cache shape is [num_blocks, num_kv_heads, head_size // 16, block_size, 16].
    block_size = key_cache.shape[-2]
    if block_size in _PA_DECODE_PS_SMALL_BLOCK_SIZES:
        if block_tables is None:
            raise ValueError(
                f"pa_decode_ps_launch: block_size={block_size} requires `block_tables` "
                "(per-sequence physical block index table)."
            )
        batch_size = context_lengths.shape[0]
        np_tile = None
        tile_pmax = tile_psum = tile_pout = None
        if is_graph_capturing:
            # Buffer sizes must be fixed ahead of capture and stay
            # identical across every replay, so force the same
            # `max_context_partition_num` heuristic the other PS paths use
            # here (instead of pa_decode_tile's own internal per-call
            # choice) and require the caller to have preallocated
            # exp_sums/max_logits/temporary_output for it, exactly as the
            # other paths already require.
            np_tile = max_context_partition_num
            if np_tile == 0:
                blocks_per_partition = KV_COMPUTE_BLOCK // block_size
                np_tile = get_recommended_splits(batch_size, num_kv_heads, split_kv_blocks=blocks_per_partition)
            if exp_sums is None or max_logits is None or temporary_output is None:
                raise ValueError(
                    "CUDA graph capture requires preallocated `exp_sums`, `max_logits`, "
                    "and `temporary_output` for the tile-backed small-block PS path."
                )
            tile_pmax, tile_psum, tile_pout = max_logits, exp_sums, temporary_output
        # pa_decode_tile requires an exact [num_blocks, num_kv_heads,
        # block_size] per-token scale shape; callers here may pass an extra
        # trailing singleton dim (e.g. from a pertoken-quant helper), which
        # reshape away without changing the strides.
        if per_token_kv:
            num_blocks = key_cache.shape[0]
            key_scale = key_scale.reshape(num_blocks, num_kv_heads, block_size)
            value_scale = value_scale.reshape(num_blocks, num_kv_heads, block_size)
        pa_decode_tile(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            softmax_scale=softmax_scale,
            stream=s,
            num_partitions=np_tile,
            pmax=tile_pmax,
            psum=tile_psum,
            pout=tile_pout,
        )
        return "ps_small_block"

    if metadata is None:
        if is_graph_capturing:
            raise ValueError(
                "CUDA graph capture requires precomputed `metadata`; "
                "call `get_pa_metadata()` before capture and pass it via `metadata=`."
            )
        metadata = get_pa_metadata(query, key_cache, context_lengths, kv_indptr, num_query_heads, num_kv_heads)

    work_indptr = metadata["work_indptr"]
    work_info_flat = metadata["work_info_flat"]
    partition_indptr = metadata["partition_indptr"]
    partial_output = metadata["partial_output"]
    partial_lse = metadata["partial_lse"]
    stride_po_partial = metadata["stride_po_partial"]
    stride_pl_partial = metadata["stride_pl_partial"]
    num_sm = metadata["num_sm"]

    metadata_block_size = key_cache.shape[-2]
    compiled = compile_pa_decode_metadata(
        softmax_scale=softmax_scale,
        trans_v=trans_v,
        query_group_size=query_group_size,
        per_token_kv=per_token_kv,
        query_length=query_length,
        query_input_dtype=query_input_dtype,
        head_dim=int(query.shape[-1]),
        block_size=int(metadata_block_size),
        output_dtype_str=_get_output_dtype_str(output),
    )

    stride_po_ql = metadata.get("stride_po_ql", num_query_heads * query.shape[-1])
    stride_pl_ql = metadata.get("stride_pl_ql", num_query_heads)

    _run_compiled(
        compiled["launch"],
        output.data_ptr(),
        partial_output.data_ptr(),
        partial_lse.data_ptr(),
        query.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        context_lengths.data_ptr(),
        key_scale.data_ptr(),
        value_scale.data_ptr(),
        work_indptr.data_ptr(),
        work_info_flat.data_ptr(),
        kv_page_indices.data_ptr(),
        kv_indptr.data_ptr(),
        partition_indptr.data_ptr(),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        output.stride(0),
        output.stride(1),
        stride_po_partial,
        stride_pl_partial,
        stride_ks_block,
        stride_ks_head,
        stride_po_ql,
        stride_pl_ql,
        num_sm,
        s,
    )

    from kernels.attention.pa_metadata import pa_ps_reduce

    # Deterministic FlyDSL reduce replaces the racy aiter pa_reduce_v1/mla_reduce_v1
    # (root cause of the flaky test_pa NaN). Same partial layout / reduce maps.
    pa_ps_reduce(
        partial_output=partial_output[query_length:],
        partial_lse=partial_lse[query_length:],
        reduce_indptr=metadata["reduce_indptr"],
        reduce_final_map=metadata["reduce_final_map"],
        reduce_partial_map=metadata["reduce_partial_map"],
        max_seqlen_q=query_length,
        final_output=output,
        num_query_heads=num_query_heads,
        head_size=int(query.shape[-1]),
        stream=s,
    )

    return "ps_split_reduce"
