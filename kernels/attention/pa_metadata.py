# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Paged-attention worklist scheduler and persistent block-1024 decode kernel.

The scheduler is specialized for uniform query lengths, causal non-sparse
attention, one query tile, and ``num_splits = num_cu``. It produces
``work_indptr``, ``work_info``, and the three reduction maps without aiter.

``work_info`` layout (8 x int32 per work):
  [0] batch_idx  [1] partial_qo_loc(-1 if no split)  [2] qo_start  [3] qo_end
  [4] kv_start   [5] kv_end                          [6] kv_offset(=0)
  [7] q_head_range = (qhead_end << 16) | (qhead_start & 0xFFFF)
"""

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_common import _compute_block_base_dw_i64
from kernels.common import dpp_utils
from kernels.common.kernels_common import get_warp_size
from kernels.common.tensor_shim import _run_compiled
from kernels.common.utils import (
    exp2_f32_fast,
    is_pow2,
    rcp_f32,
    udiv_const,
    unflatten_k,
    urem_const,
)

_WORK_INFO_FIELDS = 8
_FLAT_BUFFER_ELEMENTS = 1 << 30
_PA_OUTPUT_DTYPES = {
    "f32": fx.Float32,
    "f16": fx.Float16,
    "bf16": fx.BFloat16,
}


def _global_pointer_from_addr(addr, dtype, *, alignment: int):
    ptr_type = fx.PointerType.get(
        elem_ty=dtype.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=alignment,
    )
    return fx.inttoptr(ptr_type, addr)


def _copy_load(source, offset, copy_atom, register):
    fx.copy(copy_atom, fx.slice(source, (None, fx.Int32(offset))), register)
    return fx.memref_load_vec(register)


def _load_global_16b(global_ptr, byte_offset, copy_atom, reg):
    src = fx.make_view(global_ptr + byte_offset, fx.make_layout(16, 1))
    fx.copy(copy_atom, src, reg)
    return fx.memref_load_vec(reg).bitcast(fx.Int64)


def get_pa_metadata_info_v1(batch_size: int, num_head_k: int = 1, num_cu: int = None):
    """Return buffer sizes and dtypes for the PA worklist tensors.

    Returns (shape, dtype) tuples for:
      work_indptr, work_info_set, reduce_indptr,
      reduce_final_map, reduce_partial_map.

    ``num_cu`` overrides the worklist bin count (default = device CU count);
    pass a multiple of the CU count to oversubscribe the persistent grid.
    """
    if num_cu is None:
        gpu = torch.cuda.current_device()
        num_cu = torch.cuda.get_device_properties(gpu).multi_processor_count

    max_work = (batch_size + num_cu - 1) * num_head_k
    max_split_tiles = min(batch_size + num_cu - 1, (num_cu - 1) * 2)

    return (
        ((num_cu + 1,), torch.int32),  # work_indptr
        ((max_work, _WORK_INFO_FIELDS), torch.int32),  # work_info_set
        ((batch_size + 1,), torch.int32),  # reduce_indptr
        ((batch_size, 2), torch.int32),  # reduce_final_map
        ((max_split_tiles,), torch.int32),  # reduce_partial_map
    )


KV_BLOCK_SIZE = 1024  # physical page size (matches SP3 kBlockSize)

KV_COMPUTE_BLOCK = 256  # tile size (matches SP3 kTileKV)

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

PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict

LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240

LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4  # 512

LDS_SCALE_V_PADDING = 4  # break K/V same-bank paired writes

LDS_SCALE_V_OFFSET = KV_COMPUTE_BLOCK + LDS_SCALE_V_PADDING

LDS_SCALE_BYTES = (LDS_SCALE_V_OFFSET + KV_COMPUTE_BLOCK) * 4  # K/V per-token scale staging

FP8_MAX = 240.0

LOG2E = 1.4426950408889634


def _load_k_flat(
    k_global_ptr,
    k_copy_atom,
    k_reg,
    k_block_base_dw_i64,
    tile_token_offset_i32,
    k_tok_thread_base,
    c_tok_stride_dw,
    k_he_off_dw,
    *,
    qkhe_loop: int = 2,
):
    k_flat = []
    tile_tok_base = tile_token_offset_i32 + k_tok_thread_base

    for td in range_constexpr(TLOOP):
        kbo = tile_tok_base + fx.Int32(td * MFMA_N)
        kbo_dw = kbo * c_tok_stride_dw
        for qkhe in range_constexpr(qkhe_loop):
            ka_dw = k_block_base_dw_i64 + fx.Int64(kbo_dw + k_he_off_dw[qkhe])
            k2 = _load_global_16b(k_global_ptr, ka_dw * fx.Int64(4), k_copy_atom, k_reg)
            k2_words = fx.Vector(k2)
            k_flat.append(k2_words[0])
            k_flat.append(k2_words[1])

    return k_flat


def _build_pa_k_thread_invariants(
    warp_id,
    lane16id,
    rowid,
    *,
    qkhe_loop: int = 2,
):
    k_tok_thread_base = warp_id * fx.Int32(TOKENS_PER_WARP) + lane16id
    c_tok_stride_dw = fx.Int32(FP8_ELEMS_16B // 4)
    c_he_stride_dw = fx.Int32(KV_BLOCK_SIZE * FP8_ELEMS_16B // 4)
    k_he_off_dw = [rowid * c_he_stride_dw + fx.Int32(qkhe * 4) * c_he_stride_dw for qkhe in range(qkhe_loop)]
    return k_tok_thread_base, c_tok_stride_dw, k_he_off_dw


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
    logits_base,
    softmax_base,
    q_chunks,
    lane16id,
    rowid,
    local_qhead_idx,
    *,
    head_dim: int,
):
    qkhe_loop = head_dim // QKHE_PER_FETCH
    q_lanes_per_head = head_dim // Q_ELEMS_PER_LANE
    lds_q_base = local_qhead_idx * fx.Int32(head_dim // 4) + lane16id * 2
    abs_mask = fx.Vector.filled(4, 0x7FFFFFFF, fx.Int32)

    q_f32_chunks = []
    local_max = fx.Float32(0.0)
    for q_src in q_chunks:
        q_f32 = fx.Vector(q_src).to(fx.Float32)
        q_f32_chunks.append(q_f32)
        q_abs = (q_f32.bitcast(fx.Int32) & abs_mask).bitcast(fx.Float32)
        local_max = fx.maxnumf(local_max, q_abs.reduce("max"))

    for sh in [8, 4, 2, 1]:
        local_max = fx.maxnumf(local_max, dpp_utils.dpp_xor_f32(local_max, sh))
    query_scale_lane = (local_max > fx.Float32(0.0)).select(
        local_max * fx.Float32(1.0 / FP8_MAX),
        fx.Float32(1.0),
    )
    inv_query_scale = fx.Float32(rcp_f32(query_scale_lane))
    q_words = []
    for q_f32 in q_f32_chunks:
        p = q_f32 * inv_query_scale
        lo = rocdl.cvt_pk_fp8_f32(T.i32, p[0], p[1], fx.Int32(0), False)
        q_words.append(rocdl.cvt_pk_fp8_f32(T.i32, p[2], p[3], lo, True))

    if lane16id == fx.Int32(0):
        fx.ptr_store(
            fx.Vector.from_elements([query_scale_lane], dtype=fx.Float32),
            softmax_base + local_qhead_idx,
        )

    q_vec = fx.Vector.from_elements(q_words, dtype=fx.Int32)
    if const_expr(q_lanes_per_head < MFMA_N):
        if lane16id < fx.Int32(q_lanes_per_head):
            fx.ptr_store(q_vec, logits_base + lds_q_base)
    else:
        fx.ptr_store(q_vec, logits_base + lds_q_base)

    q_frags = []
    gpu.barrier()
    query_scale_lane = fx.ptr_load(softmax_base + lane16id, result_type=fx.Vector.make_type(1, fx.Float32))[0]
    for qkhe in range_constexpr(qkhe_loop):
        for qkr in range_constexpr(2):
            lds_rd = lane16id * fx.Int32(head_dim // 8) + fx.Int32(qkhe * 8) + rowid * fx.Int32(2) + fx.Int32(qkr)
            q_frags.append(
                fx.ptr_load(
                    fx.recast_iter(fx.Int64, logits_base) + lds_rd,
                    result_type=fx.Vector.make_type(1, fx.Int64),
                )[0]
            )
    return q_frags, query_scale_lane


def _prefetch_mtp_group_query(
    q_tiles,
    q_copy_atom,
    q_reg,
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
    q_lanes_per_head,
):
    qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q = _compute_mtp_group_state(
        lane16id,
        local_qhead_idx,
        mtp_group_idx=mtp_group_idx,
        query_length=query_length,
        query_group_size=query_group_size,
    )
    q_row = batch_idx * fx.Int32(query_length) + qi_for_q
    q_base = q_row * stride_q_seq + (kv_h * fx.Int32(query_group_size) + local_qhead_idx_for_q) * stride_q_head
    q_load_lane = lane16id
    if const_expr(q_lanes_per_head < MFMA_N):
        q_load_lane = (lane16id < fx.Int32(q_lanes_per_head)).select(lane16id, fx.Int32(0))
    q_elem = q_base + q_load_lane * fx.Int32(Q_ELEMS_PER_LANE)
    q_tile = q_elem // fx.Int32(4)
    q_chunks = []
    for qwi in range_constexpr(Q_CHUNKS_PER_LANE):
        q_chunks.append(_copy_load(q_tiles, q_tile + fx.Int32(qwi), q_copy_atom, q_reg))
    return qi_val, qhi_pos, q_chunks


def _normalize_pa_output(running_sum, outs, zero_f):
    one_f = fx.Float32(1.0).ir_value()
    safe_sum = arith.select(running_sum > zero_f, running_sum, one_f)
    inv_sum = rcp_f32(safe_sum)
    return [out * fx.Float32(inv_sum) for out in outs]


@flyc.jit
def _make_pa_phase_helpers(
    *,
    trans_v,
    per_token_kv,
    kv_h,
    v_global_ptr,
    v_copy_atom,
    v_reg,
    ks_tiles,
    vs_tiles,
    scale_copy_atom,
    scale_reg,
    logits_base,
    softmax_base,
    scale_base,
    stride_ks_block,
    stride_ks_head,
    softmax_scale,
    k_scale_val,
    v_scale_val,
    warp_id,
    lane16id,
    rowid,
    head_dim: int = 128,
):
    qkhe_loop = head_dim // QKHE_PER_FETCH
    vhe_loop = head_dim // MFMA_N // NUM_WARPS

    vhead_elems = [
        fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + lane16id for vhe in range(vhe_loop)
    ]
    v_tok_thread_off = [fx.Int32(vt * TOKENS_PER_WARP) + rowid * fx.Int32(MFMA_N) for vt in range(VTLOOP)]
    if const_expr(trans_v):
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(FP8_ELEMS_16B // 4) for vhe in range(vhe_loop)]
    else:
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(KV_BLOCK_SIZE // 4) for vhe in range(vhe_loop)]

    kv_tok_thread_base = warp_id * fx.Int32(TOKENS_PER_WARP) + rowid * fx.Int32(4)
    rowid_8x8 = rowid >> fx.Int32(1)
    offset_in_slot = rowid & fx.Int32(1)
    prob_row_i32 = PROB_ROW_STRIDE_BYTES // 4
    prob_row_i64 = PROB_ROW_STRIDE_BYTES // 8
    prob_wr_thread_base = (
        warp_id * fx.Int32(4 * MFMA_N * prob_row_i32)
        + lane16id * fx.Int32(prob_row_i32)
        + rowid_8x8 * fx.Int32(2)
        + offset_in_slot
    )
    pv_prob_read_base = rowid * fx.Int32(MFMA_N * prob_row_i64) + lane16id * fx.Int32(prob_row_i64)

    sm_lane_wave_base = lane16id * fx.Int32(NUM_WARPS)
    sm_max_off = sm_lane_wave_base + warp_id
    sm_sum_off = fx.Int32(NUM_WARPS * MFMA_N) + sm_lane_wave_base + warp_id
    sm_rd_max_offs = [sm_lane_wave_base + fx.Int32(w) for w in range(NUM_WARPS)]
    sm_rd_sum_offs = [fx.Int32(NUM_WARPS * MFMA_N) + sm_lane_wave_base + fx.Int32(w) for w in range(NUM_WARPS)]

    sm_vmax_wr_off = None
    sm_vmax_rd_offs = None
    if const_expr(per_token_kv):
        sm_vmax_wr_off = fx.Int32(2 * NUM_WARPS * MFMA_N) + sm_lane_wave_base + warp_id
        sm_vmax_rd_offs = [fx.Int32(2 * NUM_WARPS * MFMA_N) + sm_lane_wave_base + fx.Int32(w) for w in range(NUM_WARPS)]

    neg_inf = fx.Float32(float("-inf"))
    zero_f = fx.Float32(0.0)

    pv_prob_i64_elems = []
    for vt in range_constexpr(VTLOOP):
        for j in range_constexpr(2):
            p_elem = fx.Int32(vt * 4 * MFMA_N * (PROB_ROW_STRIDE_BYTES // 8)) + pv_prob_read_base + fx.Int32(j)
            pv_prob_i64_elems.append(p_elem)

    def _load_kv_scale_scalars(tile_token_offset_i32, phys_block):
        if const_expr(per_token_kv):
            scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            scale_global_token = tile_token_offset_i32 + scale_stage_token
            scale_offset = fx.Int32(scale_block_base + scale_global_token)
            k_scale_scalar = _copy_load(ks_tiles, scale_offset, scale_copy_atom, scale_reg)[0]
            v_scale_scalar = _copy_load(vs_tiles, scale_offset, scale_copy_atom, scale_reg)[0]
            return k_scale_scalar, v_scale_scalar
        return ()

    def _load_v_and_scales(
        v_block_base_dw,
        tile_token_offset_i32,
        scale_scalars,
    ):
        if const_expr(per_token_kv):
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            k_scale_scalar, v_scale_scalar = scale_scalars
            fx.ptr_store(
                fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32),
                scale_base + scale_stage_token,
            )
            fx.ptr_store(
                fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32),
                scale_base + (fx.Int32(LDS_SCALE_V_OFFSET) + scale_stage_token),
            )
            rocdl.sched_barrier(0)

        v_results = []
        for vt in range_constexpr(VTLOOP):
            vhe_data = []
            for vhe in range_constexpr(vhe_loop):
                v_token_in_block = tile_token_offset_i32 + v_tok_thread_off[vt]
                if const_expr(trans_v):
                    vt_group = v_token_in_block >> fx.Int32(4)
                    va_dw_delta = vt_group * fx.Int32(head_dim * FP8_ELEMS_16B // 4) + vhead_elem_dw[vhe]
                else:
                    va_dw_delta = vhead_elem_dw[vhe] + (v_token_in_block >> fx.Int32(2))
                va_byte = (v_block_base_dw + fx.Int64(va_dw_delta)) * fx.Int64(4)
                v_i64x2 = _load_global_16b(v_global_ptr, va_byte, v_copy_atom, v_reg)
                vhe_data.append(v_i64x2)
            v_results.append(vhe_data)

        if const_expr(per_token_kv):
            gpu.barrier()
            k_scale_vecs = []
            v_scale_vecs = []
            for td in range_constexpr(TLOOP):
                scale_row_base = kv_tok_thread_base + fx.Int32(td * MFMA_N)
                k_scale_vecs.append(
                    fx.ptr_load(scale_base + (scale_row_base), result_type=fx.Vector.make_type(4, fx.Float32))
                )
                v_scale_vecs.append(
                    fx.ptr_load(
                        scale_base + (fx.Int32(LDS_SCALE_V_OFFSET) + scale_row_base),
                        result_type=fx.Vector.make_type(4, fx.Float32),
                    )
                )
            return v_results, k_scale_vecs, v_scale_vecs

        return v_results

    def _store_vmax_warp(partition_start, *, seq_end=None, v_scale_vecs=None):
        if const_expr(per_token_kv):
            kv_tok_base = partition_start + kv_tok_thread_base if const_expr(seq_end is not None) else None
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = fx.Vector(v_scale_vecs[td])
                if const_expr(kv_tok_base is not None):
                    vs = (_token_vec_i32(kv_tok_base, td) < seq_end).select(vs, zero_f)
                v_max_warp = fx.maxnumf(v_max_warp, vs.reduce("max"))
            for sh in [32, 16]:
                v_max_warp = fx.maxnumf(v_max_warp, v_max_warp.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE)))
            fx.ptr_store(
                fx.Vector.from_elements([v_max_warp], dtype=fx.Float32),
                softmax_base + sm_vmax_wr_off,
            )

    def _token_vec_i32(kv_tok_base, td: int):
        kv_tok_td_base = kv_tok_base + fx.Int32(td * MFMA_N)
        return fx.Vector.from_elements(
            [kv_tok_td_base + fx.Int32(i) for i in range_constexpr(4)],
            dtype=fx.Int32,
        )

    def _qk_and_intra_softmax(
        k_ops,
        partition_start,
        q_frags,
        causal_bound,
        query_scale_lane,
        *,
        preloaded_scales=None,
    ):
        if const_expr(per_token_kv):
            k_scale_vecs, v_scale_vecs = preloaded_scales

        query_scale = fx.Float32(query_scale_lane * softmax_scale)
        d_out = []
        for td in range_constexpr(TLOOP):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for k_step in range_constexpr(qkhe_loop * 2):
                acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
            if const_expr(per_token_kv):
                d_out.append(fx.Vector(acc) * (k_scale_vecs[td] * query_scale))
            else:
                d_out.append(fx.Vector(acc) * (query_scale * k_scale_val))

        kv_tok_base = partition_start + kv_tok_thread_base
        qk_max = neg_inf
        for td in range_constexpr(TLOOP):
            logits_vec = (_token_vec_i32(kv_tok_base, td) < causal_bound).select(fx.Vector(d_out[td]), neg_inf)
            d_out[td] = logits_vec
            qk_max = fx.maxnumf(qk_max, fx.Vector(logits_vec).reduce("max"))
        for sh in [32, 16]:
            qk_max = fx.maxnumf(qk_max, qk_max.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE)))
        fx.ptr_store(
            fx.Vector.from_elements([qk_max], dtype=fx.Float32),
            softmax_base + sm_max_off,
        )

        if const_expr(per_token_kv):
            return d_out, v_scale_vecs
        return d_out

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs, v_scale_vecs):
        partition_max = neg_inf
        partition_sum = zero_f
        max_vec = fx.ptr_load(softmax_base + (sm_rd_max_offs[0]), result_type=fx.Vector.make_type(4, fx.Float32))
        for w in range_constexpr(NUM_WARPS):
            partition_max = fx.maxnumf(partition_max, max_vec[w])

        new_rmax = fx.maxnumf(rmax, partition_max)
        safe_eff_max = arith.select(partition_max > neg_inf, new_rmax, zero_f)
        local_exp_sum = zero_f
        for td in range_constexpr(TLOOP):
            diff_vec = fx.Vector(d_out[td]) - fx.Float32(safe_eff_max)
            p_vec = exp2_f32_fast(diff_vec * fx.Float32(LOG2E))
            local_exp_sum = local_exp_sum + fx.Vector(p_vec).reduce("add")
            d_out[td] = p_vec
        for sh in [32, 16]:
            local_exp_sum = local_exp_sum + local_exp_sum.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE))
        fx.ptr_store(
            fx.Vector.from_elements([local_exp_sum], dtype=fx.Float32),
            softmax_base + sm_sum_off,
        )
        accum_scale = arith.select(
            rmax > neg_inf,
            exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value()),
            zero_f,
        )

        gpu.barrier()
        sum_vec = fx.ptr_load(softmax_base + (sm_rd_sum_offs[0]), result_type=fx.Vector.make_type(4, fx.Float32))
        for w in range_constexpr(NUM_WARPS):
            partition_sum = arith.addf(
                arith.unwrap(partition_sum), arith.unwrap(sum_vec[w]), fastmath=arith.FastMathFlags.contract
            )

        accum_sum = arith.mulf(arith.unwrap(accum_scale), arith.unwrap(rsum), fastmath=arith.FastMathFlags.contract)
        rsum = arith.addf(accum_sum, arith.unwrap(partition_sum), fastmath=arith.FastMathFlags.contract)
        rmax = new_rmax
        for vhe in range_constexpr(vhe_loop):
            outs[vhe] = outs[vhe] * fx.Float32(accum_scale)

        if const_expr(per_token_kv):
            v_max_global = zero_f
            vmax_vec = fx.ptr_load(softmax_base + (sm_vmax_rd_offs[0]), result_type=fx.Vector.make_type(4, fx.Float32))
            for w in range_constexpr(NUM_WARPS):
                w_vmax = vmax_vec[w]
                v_max_global = fx.maxnumf(v_max_global, w_vmax)
            v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX).ir_value()
            v_max_safe_scaled = v_max_scaled + fx.Float32(1e-8 / FP8_MAX).ir_value()
            norm_factor = rcp_f32(v_max_safe_scaled)
            v_correction = v_max_scaled
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * (v_scale_vecs[td] * fx.Float32(norm_factor))
        else:
            v_correction = v_scale_val

        for td in range_constexpr(TLOOP):
            pv = fx.Vector(d_out[td])
            lo = rocdl.cvt_pk_fp8_f32(T.i32, pv[0], pv[1], fx.Int32(0), False)
            pk = rocdl.cvt_pk_fp8_f32(T.i32, pv[2], pv[3], lo, True)
            elem_base = prob_wr_thread_base + fx.Int32(td * MFMA_N * (PROB_ROW_STRIDE_BYTES // 4))
            pk_vec = fx.Vector.from_elements([pk], dtype=fx.Int32)
            fx.ptr_store(pk_vec, logits_base + elem_base)
        return rmax, rsum, outs, v_correction

    def _pv_mfma(v_ops, outs, v_correction):
        fm_contract = arith.FastMathFlags.contract
        v_correction_vec = fx.Vector.filled(4, fx.Float32(v_correction), fx.Float32)

        # P depends only on (vt, j); load it once before every VHE MFMA chain.
        p_i64_all = []
        for vt in range_constexpr(VTLOOP):
            for j in range_constexpr(2):
                p_i64_off = pv_prob_i64_elems[vt * 2 + j]
                p_i64_all.append(
                    fx.ptr_load(
                        fx.recast_iter(fx.Int64, logits_base) + (p_i64_off),
                        result_type=fx.Vector.make_type(1, fx.Int64),
                    )[0]
                )

        for vhe in range_constexpr(vhe_loop):
            tmp_out = fx.Vector.filled(4, 0.0, fx.Float32)
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
        _load_kv_scale_scalars,
        _load_v_and_scales,
        _store_vmax_warp,
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
    )


@functools.lru_cache(maxsize=256)
def compile_pa_metadata_v1(
    *,
    num_cu: int,
    num_heads_k: int,
    gqa: int,
    kv_granularity: int,
    query_length: int,
    warp_size: int,
):
    """Compile the single-wave worklist scheduler for a fixed shape config."""
    assert is_pow2(kv_granularity), "kv_granularity must be power of 2"
    assert num_cu % num_heads_k == 0, "num_cu must be divisible by num_heads_k"
    num_splits_per_khead = num_cu // num_heads_k
    _shuffle_offsets = []
    _o = warp_size // 2
    while _o >= 1:
        _shuffle_offsets.append(_o)
        _o //= 2

    @flyc.kernel(known_block_size=(warp_size, 1, 1))
    def pa_metadata_v1_kernel(
        seqlens_qo_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (cumulative qo seqlens)
        context_lens_ptr: fx.Tensor,  # [num_batches] i32
        work_indptr_ptr: fx.Tensor,  # [num_cu + 1] i32   (output)
        work_info_ptr: fx.Tensor,  # [max_work * 8] i32 (output, flattened)
        reduce_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (output)
        reduce_final_map_ptr: fx.Tensor,  # [num_batches * 2] i32 (output, flattened)
        reduce_partial_map_ptr: fx.Tensor,  # [max_split_tiles] i32 (output)
        num_batches: Int32,
    ):
        scalar_layout = fx.make_layout(1, 1)
        work_layout = fx.make_layout(4, 1)
        reduce_final_layout = fx.make_layout(2, 1)

        def _divide_buffer(tensor, tile_layout):
            return fx.logical_divide(fx.rocdl.make_buffer_tensor(tensor), tile_layout)

        sq = _divide_buffer(seqlens_qo_indptr_ptr, scalar_layout)
        context_lens = _divide_buffer(context_lens_ptr, scalar_layout)
        work_indptr = _divide_buffer(work_indptr_ptr, scalar_layout)
        work_info = _divide_buffer(work_info_ptr, work_layout)
        reduce_indptr = _divide_buffer(reduce_indptr_ptr, scalar_layout)
        reduce_final_map = _divide_buffer(reduce_final_map_ptr, reduce_final_layout)
        reduce_partial_map = _divide_buffer(reduce_partial_map_ptr, scalar_layout)

        copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy_i32x2 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        copy_i32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        load_reg = fx.make_rmem_tensor(1, fx.Int32)
        store_reg = fx.make_rmem_tensor(1, fx.Int32)
        store_reg2 = fx.make_rmem_tensor(2, fx.Int32)
        store_reg4 = fx.make_rmem_tensor(4, fx.Int32)

        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        c_qlen = fx.Int32(query_length)
        c_nb = num_batches  # Int32 runtime
        c_nspk = fx.Int32(num_splits_per_khead)
        c_numcu = fx.Int32(num_cu)
        c_ws = fx.Int32(warp_size)
        c_kvg = fx.Int32(kv_granularity)
        lane = fx.Int32(gpu.thread_id("x"))

        def _num_part(batch_idx):
            ctxv = _copy_load(context_lens, batch_idx, copy_i32, load_reg)[0]
            return fx.Int32(arith.ceildivui(ctxv.ir_value(), c_kvg.ir_value()))

        def _store(divided, off, val):
            fx.memref_store_vec(fx.Vector.from_elements([fx.Int32(val)], dtype=fx.Int32), store_reg)
            fx.copy(copy_i32, store_reg, fx.slice(divided, (None, off)))

        _store(work_indptr, 0, 0)
        _store(reduce_indptr, 0, 0)

        # Lane-strided partition count followed by a wave reduction.
        b = lane
        sum_blocks = c0
        while b < c_nb:
            nblk = _num_part(b)
            b = b + c_ws
            sum_blocks = sum_blocks + nblk
        for sh in _shuffle_offsets:
            sum_blocks = sum_blocks + sum_blocks.shuffle_xor(fx.Int32(sh), c_ws)

        average = fx.Int32(arith.divui(sum_blocks.ir_value(), c_nspk.ir_value()))
        reminder = fx.Int32(arith.remui(sum_blocks.ir_value(), c_nspk.ir_value()))

        def _remain_for_cid(cid_val):
            mod = fx.Int32(arith.remui(cid_val.ir_value(), c_nspk.ir_value()))
            return average + (mod < reminder).select(c1, c0)

        cid = c0
        num_works = c0

        for khead in range_constexpr(num_heads_k):
            qh_start = khead * gqa
            qh_end = (khead + 1) * gqa
            qhr_const = (qh_end << 16) | (qh_start & 0xFFFF)  # python int constant

            kvend0 = _num_part(c0)  # partitions in batch 0 (cumulative kv end)
            remain0 = _remain_for_cid(cid)

            cid_ = cid
            batch_ = c0
            kvblk_ = c0
            nsplit_ = c0
            nworks_ = num_works
            pidx_ = c0
            kvbeg_ = c0
            kvend_ = kvend0
            remain_ = remain0
            lri_ = c0
            grt_ = c0

            while (cid_ < c_numcu) & (batch_ < c_nb):
                pages = kvend_ - kvbeg_
                remain_kv = pages - kvblk_
                do_finish = remain_ >= remain_kv  # fx bool

                qo_start = _copy_load(sq, batch_, copy_i32, load_reg)[0]
                qo_end = _copy_load(sq, batch_ + c1, copy_i32, load_reg)[0]
                kv_start = kvbeg_ + kvblk_  # same for both branches

                f_kv_end = kvend_  # min(kv_start + remain_kv, kvend_) == kvend_
                nsplit_pos = nsplit_ > c0
                f_ploc = nsplit_pos.select(pidx_, fx.Int32(-1))
                f_pidx2 = nsplit_pos.select(pidx_ + c_qlen, pidx_)
                f_nworks2 = nworks_ + c1
                f_remain2 = remain_ - remain_kv
                f_batch2 = batch_ + c1
                # Select evaluates both values, so clamp the speculative load.
                nb_in_range = f_batch2 < c_nb
                safe_idx = nb_in_range.select(f_batch2, c0)
                f_new_pages = nb_in_range.select(_num_part(safe_idx), c0)
                f_kvbeg2 = kvend_
                f_kvend2 = kvend_ + f_new_pages

                s_emit = remain_ > c0
                s_kv_end_raw = kv_start + remain_
                s_kv_end = (s_kv_end_raw < kvend_).select(s_kv_end_raw, kvend_)
                s_nworks2 = s_emit.select(nworks_ + c1, nworks_)
                s_pidx2 = s_emit.select(pidx_ + c_qlen, pidx_)
                s_kvblk2 = s_emit.select(kvblk_ + remain_, kvblk_)
                s_nsplit2 = s_emit.select(nsplit_ + c1, nsplit_)
                s_cid2 = cid_ + c1
                s_remain2 = _remain_for_cid(s_cid2)

                w_ploc = do_finish.select(f_ploc, pidx_)
                w_kv_end = do_finish.select(f_kv_end, s_kv_end)
                work_values = [
                    batch_,
                    w_ploc,
                    qo_start,
                    qo_end,
                    kv_start,
                    w_kv_end,
                    c0,
                    fx.Int32(qhr_const),
                ]
                for half in range_constexpr(2):
                    start = half * 4
                    fx.memref_store_vec(
                        fx.Vector.from_elements(
                            [work_values[start + field] for field in range_constexpr(4)],
                            dtype=fx.Int32,
                        ),
                        store_reg4,
                    )
                    fx.copy(
                        copy_i32x4,
                        store_reg4,
                        fx.slice(work_info, (None, nworks_ * fx.Int32(2) + fx.Int32(half))),
                    )

                do_reduce = do_finish & nsplit_pos
                num_splits = nsplit_ + c1
                # Non-reduce writes are overwritten before their slots become visible.
                _store(reduce_indptr, grt_ + c1, lri_ + num_splits)
                fx.memref_store_vec(
                    fx.Vector.from_elements([qo_start, qo_end], dtype=fx.Int32),
                    store_reg2,
                )
                fx.copy(copy_i32x2, store_reg2, fx.slice(reduce_final_map, (None, grt_)))
                rcount = do_reduce.select(num_splits, c0)
                sidx = c0
                while sidx < rcount:
                    val = pidx_ - (nsplit_ - sidx) * c_qlen
                    _store(reduce_partial_map, lri_ + sidx, val)
                    sidx = sidx + c1
                next_num_works = do_finish.select(f_nworks2, s_nworks2)

                # The last same-value-race write before advancing cid is authoritative.
                _store(work_indptr, cid_ + c1, next_num_works)

                (
                    cid_,
                    batch_,
                    kvblk_,
                    nsplit_,
                    nworks_,
                    pidx_,
                    kvbeg_,
                    kvend_,
                    remain_,
                    lri_,
                    grt_,
                ) = (
                    do_finish.select(cid_, s_cid2),
                    do_finish.select(f_batch2, batch_),
                    do_finish.select(c0, s_kvblk2),
                    do_finish.select(c0, s_nsplit2),
                    next_num_works,
                    do_finish.select(f_pidx2, s_pidx2),
                    do_finish.select(f_kvbeg2, kvbeg_),
                    do_finish.select(f_kvend2, kvend_),
                    do_finish.select(f_remain2, s_remain2),
                    lri_ + do_reduce.select(num_splits, c0),
                    grt_ + do_reduce.select(c1, c0),
                )

            cid = cid_
            num_works = nworks_
            last_reduce_indptr = lri_
            global_reduce_tile_idx = grt_

            in_range = cid < c_numcu
            cid = in_range.select(cid + c1, cid)

        it_t = cid
        while it_t <= c_numcu:
            _store(work_indptr, it_t, num_works)
            it_t = it_t + c1

        c_rip_size = c_nb + c1  # reduce_indptr length = num_batches + 1
        it_r = global_reduce_tile_idx
        while it_r < c_rip_size:
            _store(reduce_indptr, it_r, last_reduce_indptr)
            it_r = it_r + c1

    @flyc.jit
    def launch_pa_metadata_v1(
        seqlens_qo_indptr: fx.Tensor,
        context_lens: fx.Tensor,
        work_indptr: fx.Tensor,
        work_info: fx.Tensor,
        reduce_indptr: fx.Tensor,
        reduce_final_map: fx.Tensor,
        reduce_partial_map: fx.Tensor,
        num_batches: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_metadata_v1_kernel(
            seqlens_qo_indptr,
            context_lens,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            num_batches,
        ).launch(grid=(1, 1, 1), block=(warp_size, 1, 1), stream=stream)

    return {"kernel": pa_metadata_v1_kernel, "launch": launch_pa_metadata_v1}


def get_pa_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    *,
    query_group_size: int,
    num_kv_heads: int,
    kv_granularity: int,
    query_length: int,
    num_cu: int = None,
    stream=None,
) -> None:
    """Build the PA-decode worklist and reduction maps.

    ``num_cu`` overrides the worklist bin count (default = device CU count);
    pass a multiple of the CU count to oversubscribe the persistent grid.
    """
    dev = context_lens.device
    if num_cu is None:
        num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
    num_batches = context_lens.shape[0]
    warp_size = get_warp_size(get_rocm_arch())

    compiled = compile_pa_metadata_v1(
        num_cu=num_cu,
        num_heads_k=num_kv_heads,
        gqa=query_group_size,
        kv_granularity=kv_granularity,
        query_length=query_length,
        warp_size=warp_size,
    )

    work_info_flat = work_info.view(-1)
    reduce_final_map_flat = reduce_final_map.view(-1)

    _run_compiled(
        compiled["launch"],
        seqlens_qo_indptr,
        context_lens,
        work_indptr,
        work_info_flat,
        reduce_indptr,
        reduce_final_map_flat,
        reduce_partial_map,
        num_batches,
        stream,
    )


@functools.lru_cache(maxsize=256)
def compile_pa_decode_metadata(
    softmax_scale=None,
    trans_v=False,
    query_group_size=16,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "bf16",
    head_dim: int = 128,
    block_size: int = None,
    output_dtype_str: str = "bf16",
):
    """Compile a PS-mode PA decode kernel.

    This does NOT bake in num_seqs/num_kv_heads/num_partitions because PS mode
    uses dynamic work distribution. Grid = (num_sm, 1, 1).

    The worklist is load-balanced at ``KV_COMPUTE_BLOCK`` (256-token) **partition**
    granularity (see ``get_pa_metadata``): ``work_info.kv_start/kv_end`` are
    cumulative partition indices. Only 1024-token physical pages are supported;
    each page contains four 256-token partitions. ``partial_qo_loc``
    (``work_info[1]``) ``< 0`` writes the final output directly to ``out``, while
    ``>= 0`` writes a partial slot that ``pa_reduce_v1`` later combines.
    """
    if block_size is None:
        block_size = KV_BLOCK_SIZE
    if block_size != KV_BLOCK_SIZE:
        raise ValueError(f"compile_pa_decode_metadata only supports block_size={KV_BLOCK_SIZE}, got {block_size}")
    if head_dim % QKHE_PER_FETCH != 0 or head_dim % (MFMA_N * NUM_WARPS) != 0 or head_dim % Q_ELEMS_PER_LANE != 0:
        raise ValueError(f"Unsupported head_dim={head_dim}; must be a multiple of {MFMA_N * NUM_WARPS}.")
    _QKHELOOP = head_dim // QKHE_PER_FETCH
    _VHELOOP = head_dim // MFMA_N // NUM_WARPS
    _Q_LANES_PER_HEAD = head_dim // Q_ELEMS_PER_LANE
    _N_K_h = TLOOP * _QKHELOOP * 2
    if query_input_dtype not in ("bf16", "f16"):
        raise ValueError(f"`compile_pa_decode_metadata` only supports bf16/f16 queries, got {query_input_dtype!r}")
    _QUERY_DTYPE = fx.BFloat16 if query_input_dtype == "bf16" else fx.Float16
    _OUTPUT_DTYPE = _PA_OUTPUT_DTYPES[output_dtype_str]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    softmax_scale = float(softmax_scale)
    parts_per_block = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK

    # Per-token mode adds one cross-warp vmax slot per lane.
    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0  # 256 or 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    LDS_SCALE_TOTAL = LDS_SCALE_BYTES if const_expr(per_token_kv) else 0
    softmax_off = LDS_LOGITS_BYTES
    scale_off = softmax_off + LDS_SOFTMAX_TOTAL
    _LDS_TOTAL_BYTES = scale_off + LDS_SCALE_TOTAL

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, _LDS_TOTAL_BYTES // 4, 16]

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_metadata_kenrel(
        out_ptr: fx.Int64,  # output [batch, num_q_heads, head_dim]
        partial_out_ptr: fx.Int64,  # partial output [num_partials, 1, nhead, head_dim] fp32
        partial_lse_ptr: fx.Int64,  # partial LSE [num_partials, 1, nhead, 1] fp32
        query_ptr: fx.Int64,  # queries [batch, num_q_heads, head_dim]
        key_cache_ptr: fx.Int64,  # key cache
        value_cache_ptr: fx.Int64,  # value cache
        context_lengths_ptr: fx.Int64,  # [batch] int32
        key_scale_ptr: fx.Int64,
        value_scale_ptr: fx.Int64,
        work_indptr_ptr: fx.Int64,  # [num_sm + 1] int32
        work_info_ptr: fx.Int64,  # [num_work, 8] int32 (flattened to 1D)
        kv_page_indices_ptr: fx.Int64,  # [total_pages] int32
        kv_indptr_ptr: fx.Int64,  # [num_seqs + 1] int32 — prefix sum of pages per seq
        partition_indptr_ptr: fx.Int64,  # [num_seqs + 1] int32 — prefix sum of partitions per seq
        stride_q_seq: Int32,
        stride_q_head: Int32,
        stride_k_block: Int32,
        stride_k_head: Int32,
        stride_v_block: Int32,
        stride_v_head: Int32,
        stride_out_seq: Int32,
        stride_out_head: Int32,
        stride_po_partial: Int32,  # stride for partial_output partial dim (nhead * head_dim)
        stride_pl_partial: Int32,  # stride for partial_lse partial dim (nhead)
        stride_ks_block: Int32,  # key_scale stride for block dim (num_kv_heads * KV_BLOCK_SIZE); 0 for per-tensor
        stride_ks_head: Int32,  # key_scale stride for head dim (KV_BLOCK_SIZE); 0 for per-tensor
        stride_po_ql: Int32,  # stride for partial_output query-length dim (num_query_heads * head_dim)
        stride_pl_ql: Int32,  # stride for partial_lse query-length dim (num_query_heads)
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        cu_id = fx.Int32(gpu.block_id("x"))  # CU index (0..num_sm-1)

        lane16id = tid & fx.Int32(15)
        rowid = (tid >> fx.Int32(4)) & fx.Int32(3)
        warp_id = tid >> fx.Int32(6)

        def _divide_addr(addr, dtype, width):
            ptr = _global_pointer_from_addr(addr, dtype, alignment=width * dtype.width // 8)
            flat = fx.make_view(ptr, fx.make_layout(_FLAT_BUFFER_ELEMENTS, 1))
            return fx.logical_divide(
                fx.rocdl.make_buffer_tensor(flat),
                fx.make_layout(width, 1),
            )

        q_tiles = _divide_addr(query_ptr, _QUERY_DTYPE, 4)
        out_tiles = _divide_addr(out_ptr, _OUTPUT_DTYPE, 4)
        partial_out_tiles = _divide_addr(partial_out_ptr, fx.Float32, 4)
        partial_lse = _divide_addr(partial_lse_ptr, fx.Float32, 1)
        context_lengths = _divide_addr(context_lengths_ptr, fx.Int32, 1)
        work_indptr = _divide_addr(work_indptr_ptr, fx.Int32, 1)
        work_info = _divide_addr(work_info_ptr, fx.Int32, 4)
        kv_page_indices = _divide_addr(kv_page_indices_ptr, fx.Int32, 1)
        kv_indptr = _divide_addr(kv_indptr_ptr, fx.Int32, 1)
        partition_indptr = _divide_addr(partition_indptr_ptr, fx.Int32, 1)
        key_scales = _divide_addr(key_scale_ptr, fx.Float32, 1)
        value_scales = _divide_addr(value_scale_ptr, fx.Float32, 1)

        copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy_i32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_q = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), _QUERY_DTYPE)
        copy_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b() if _OUTPUT_DTYPE.width == 32 else fx.rocdl.BufferCopy64b(),
            _OUTPUT_DTYPE,
        )
        copy_f32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        i32_reg = fx.make_rmem_tensor(1, fx.Int32)
        i32x4_reg = fx.make_rmem_tensor(4, fx.Int32)
        scale_reg = fx.make_rmem_tensor(1, fx.Float32)
        q_reg = fx.make_rmem_tensor(4, _QUERY_DTYPE)
        out_reg = fx.make_rmem_tensor(4, _OUTPUT_DTYPE)
        partial_out_reg = fx.make_rmem_tensor(4, fx.Float32)
        partial_lse_reg = fx.make_rmem_tensor(1, fx.Float32)

        k_global_ptr = _global_pointer_from_addr(key_cache_ptr, fx.Uint8, alignment=16)
        v_global_ptr = _global_pointer_from_addr(value_cache_ptr, fx.Uint8, alignment=16)
        global_copy_16b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Uint8)
        global_reg_16b = fx.make_rmem_tensor(16, fx.Uint8)

        if const_expr(per_token_kv):
            k_scale_val = fx.Float32(1.0)
            v_scale_val = fx.Float32(1.0)
        else:
            k_scale_val = _copy_load(key_scales, 0, copy_f32, scale_reg)[0]
            v_scale_val = _copy_load(value_scales, 0, copy_f32, scale_reg)[0]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        logits_base = lds.buf.ptr
        softmax_base = fx.recast_iter(fx.Float32, fx.add_offset(lds.buf.ptr, fx.Int32(softmax_off // 4)))
        scale_base = None
        if const_expr(per_token_kv):
            scale_base = fx.recast_iter(fx.Float32, fx.add_offset(lds.buf.ptr, fx.Int32(scale_off // 4)))

        local_qhead_idx = warp_id * fx.Int32(4) + rowid
        _k_tok_thread_base, _c_tok_stride_dw, _k_he_off_dw = _build_pa_k_thread_invariants(
            warp_id,
            lane16id,
            rowid,
            qkhe_loop=_QKHELOOP,
        )

        work_start = _copy_load(work_indptr, cu_id, copy_i32, i32_reg)[0]
        work_end = _copy_load(work_indptr, cu_id + fx.Int32(1), copy_i32, i32_reg)[0]

        for work_idx in range(work_start, work_end, fx.Int32(1)):
            # Two naturally aligned dwordx4 loads cover the eight work fields.
            info_tile = work_idx * fx.Int32(2)
            wi_lo_v = _copy_load(work_info, info_tile, copy_i32x4, i32x4_reg)
            batch_idx = wi_lo_v[0]
            partial_idx = wi_lo_v[1]
            qo_start = wi_lo_v[2]
            wi_hi_v = _copy_load(work_info, info_tile + fx.Int32(1), copy_i32x4, i32x4_reg)
            kv_start = wi_hi_v[0]
            kv_end = wi_hi_v[1]
            q_head_range = wi_hi_v[3]

            # Convert cumulative partition indices to sequence-local page indices.
            kv_part_base = _copy_load(partition_indptr, batch_idx, copy_i32, i32_reg)[0]
            kv_page_base = _copy_load(kv_indptr, batch_idx, copy_i32, i32_reg)[0]
            local_part_start = kv_start - kv_part_base

            q_head_start = q_head_range & fx.Int32(0xFFFF)
            kv_h = udiv_const(q_head_start, query_group_size)

            context_len = _copy_load(context_lengths, batch_idx, copy_i32, i32_reg)[0]
            _k_head_off = kv_h * stride_k_head
            _v_head_off = kv_h * stride_v_head

            (
                _load_kv_scale_scalars,
                _load_v_and_scales,
                _store_vmax_warp,
                _qk_and_intra_softmax,
                _cross_warp_softmax_and_prob_pack,
                _pv_mfma,
            ) = _make_pa_phase_helpers(
                trans_v=trans_v,
                per_token_kv=per_token_kv,
                kv_h=kv_h,
                v_global_ptr=v_global_ptr,
                v_copy_atom=global_copy_16b,
                v_reg=global_reg_16b,
                ks_tiles=key_scales,
                vs_tiles=value_scales,
                scale_copy_atom=copy_f32,
                scale_reg=scale_reg,
                logits_base=logits_base,
                softmax_base=softmax_base,
                scale_base=scale_base,
                stride_ks_block=stride_ks_block,
                stride_ks_head=stride_ks_head,
                softmax_scale=fx.Float32(softmax_scale),
                k_scale_val=k_scale_val,
                v_scale_val=v_scale_val,
                warp_id=warp_id,
                lane16id=lane16id,
                rowid=rowid,
                head_dim=head_dim,
            )

            # Q stays in registers while K/V are reused across all MTP groups.
            def _unwrap(v):
                return v.ir_value() if hasattr(v, "ir_value") else v

            # Negative partial_idx writes final output; split rows reserve the first QL slots.
            _is_direct = partial_idx < fx.Int32(0)
            _po_row_base = partial_idx + fx.Int32(query_length)

            num_parts_in_work = kv_end - kv_start
            last_part_idx_val = num_parts_in_work - fx.Int32(1)

            _mtp_groups = math.ceil(query_length * query_group_size / 16)

            q_frags_per_mtp = []
            qi_per_mtp = []
            qhi_per_mtp = []
            qscale_per_mtp = []
            for _mtp_g in range_constexpr(_mtp_groups):
                if const_expr(_mtp_g > 0):
                    gpu.barrier()
                _qi, _qhi, q_chunks = _prefetch_mtp_group_query(
                    q_tiles,
                    copy_q,
                    q_reg,
                    batch_idx,
                    kv_h,
                    stride_q_seq,
                    stride_q_head,
                    lane16id,
                    local_qhead_idx,
                    mtp_group_idx=_mtp_g,
                    query_length=query_length,
                    query_group_size=query_group_size,
                    q_lanes_per_head=_Q_LANES_PER_HEAD,
                )
                _qfrags, _qscale = _finish_q_fragments(
                    logits_base,
                    softmax_base,
                    q_chunks,
                    lane16id,
                    rowid,
                    local_qhead_idx,
                    head_dim=head_dim,
                )
                qi_per_mtp.append(_qi)
                qhi_per_mtp.append(_qhi)
                q_frags_per_mtp.append(_qfrags)
                qscale_per_mtp.append(_qscale)
            gpu.barrier()

            causal_bound_per_mtp = [
                context_len + fx.Int32(1 - query_length) + qi_per_mtp[_mtp_g] for _mtp_g in range(_mtp_groups)
            ]

            local_last_part = local_part_start + last_part_idx_val
            first_phys_block = _copy_load(
                kv_page_indices,
                kv_page_base + udiv_const(local_last_part, parts_per_block),
                copy_i32,
                i32_reg,
            )[0]
            first_tile_tok = urem_const(local_last_part, parts_per_block) * fx.Int32(KV_COMPUTE_BLOCK)
            first_k_base = _compute_block_base_dw_i64(first_phys_block, stride_k_block, _k_head_off)
            scale_scalars0 = _load_kv_scale_scalars(first_tile_tok, first_phys_block)
            k_flat0 = _load_k_flat(
                k_global_ptr,
                global_copy_16b,
                global_reg_16b,
                first_k_base,
                first_tile_tok,
                _k_tok_thread_base,
                _c_tok_stride_dw,
                _k_he_off_dw,
                qkhe_loop=_QKHELOOP,
            )

            state_width = 2 + _VHELOOP

            def _pack_states_kv(states, k_flat, scale_scalars):
                flat = [_unwrap(value) for state in states for value in state]
                flat.extend(_unwrap(v) for v in k_flat)
                flat.extend(_unwrap(v) for v in scale_scalars)
                return flat

            def _unpack_states_kv(flat):
                state_end = state_width * _mtp_groups
                k_end = state_end + _N_K_h
                states = [tuple(flat[state_width * i : state_width * (i + 1)]) for i in range(_mtp_groups)]
                return states, list(flat[state_end:k_end]), tuple(flat[k_end:])

            init_states = [
                tuple(
                    [fx.Float32(float("-inf")), fx.Float32(0.0)]
                    + [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(_VHELOOP)]
                )
                for _ in range(_mtp_groups)
            ]

            for ib, state in range(
                fx.Int32(0),
                num_parts_in_work,
                fx.Int32(1),
                init=_pack_states_kv(init_states, k_flat0, scale_scalars0),
            ):
                cur_states, k_flat, scale_scalars = _unpack_states_kv(state)
                # Process partition zero last for online-softmax stability.
                rel_part = last_part_idx_val - ib
                lp = local_part_start + rel_part
                next_rel = rel_part - fx.Int32(1)
                next_rel_clamped = (next_rel >= fx.Int32(0)).select(next_rel, fx.Int32(0))
                next_lp = local_part_start + next_rel_clamped

                k_ops = unflatten_k(k_flat, qkhe_loop=_QKHELOOP)
                partition_start = lp * fx.Int32(KV_COMPUTE_BLOCK)

                phys_block = _copy_load(
                    kv_page_indices,
                    kv_page_base + udiv_const(lp, parts_per_block),
                    copy_i32,
                    i32_reg,
                )[0]
                tile_token_offset = urem_const(lp, parts_per_block) * fx.Int32(KV_COMPUTE_BLOCK)
                v_base = _compute_block_base_dw_i64(phys_block, stride_v_block, _v_head_off)
                if const_expr(per_token_kv):
                    v_ops, k_scale_vecs, v_scale_vecs = _load_v_and_scales(
                        v_base,
                        tile_token_offset,
                        scale_scalars,
                    )
                else:
                    v_ops = _load_v_and_scales(
                        v_base,
                        tile_token_offset,
                        scale_scalars,
                    )
                new_states = []
                for _mtp_g in range_constexpr(_mtp_groups):
                    if const_expr(_mtp_g > 0):
                        gpu.barrier()
                    state = cur_states[_mtp_g]
                    rmax, rsum = state[0], state[1]
                    outs = [state[2 + vhe] for vhe in range_constexpr(_VHELOOP)]

                    if const_expr(per_token_kv):
                        d_out, v_scales = _qk_and_intra_softmax(
                            k_ops,
                            partition_start,
                            q_frags_per_mtp[_mtp_g],
                            causal_bound_per_mtp[_mtp_g],
                            query_scale_lane=qscale_per_mtp[_mtp_g],
                            preloaded_scales=(k_scale_vecs, v_scale_vecs),
                        )
                    else:
                        d_out = _qk_and_intra_softmax(
                            k_ops,
                            partition_start,
                            q_frags_per_mtp[_mtp_g],
                            causal_bound_per_mtp[_mtp_g],
                            query_scale_lane=qscale_per_mtp[_mtp_g],
                        )
                        v_scales = None

                    # Stage vmax before the cross-wave normalization reads it.
                    if const_expr(per_token_kv):
                        _store_vmax_warp(partition_start, seq_end=context_len, v_scale_vecs=v_scales)

                    gpu.barrier()
                    rmax, rsum, outs, v_correction = _cross_warp_softmax_and_prob_pack(
                        d_out, rmax, rsum, outs, v_scales
                    )
                    gpu.barrier()
                    outs = _pv_mfma(v_ops, outs, v_correction)
                    new_states.append(tuple([rmax, rsum] + outs))

                next_phys_block = _copy_load(
                    kv_page_indices,
                    kv_page_base + udiv_const(next_lp, parts_per_block),
                    copy_i32,
                    i32_reg,
                )[0]
                next_tile_tok = urem_const(next_lp, parts_per_block) * fx.Int32(KV_COMPUTE_BLOCK)
                next_k_base = _compute_block_base_dw_i64(next_phys_block, stride_k_block, _k_head_off)
                next_scale_scalars = _load_kv_scale_scalars(next_tile_tok, next_phys_block)
                k_next_flat = _load_k_flat(
                    k_global_ptr,
                    global_copy_16b,
                    global_reg_16b,
                    next_k_base,
                    next_tile_tok,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                    qkhe_loop=_QKHELOOP,
                )

                results = yield _pack_states_kv(new_states, k_next_flat, next_scale_scalars)

            final_states, _, _ = _unpack_states_kv(results)

            for _mtp_g in range_constexpr(_mtp_groups):
                final_state = final_states[_mtp_g]
                rmax_raw, rsum_raw = final_state[0], final_state[1]
                outs_raw = [final_state[2 + vhe] for vhe in range_constexpr(_VHELOOP)]
                running_max = fx.Float32(rmax_raw)
                running_sum = fx.Float32(rsum_raw)
                outs = [fx.Vector(out_raw) for out_raw in outs_raw]
                outelems_norm = _normalize_pa_output(running_sum, outs, fx.Float32(0.0))
                qi_val_mg = qi_per_mtp[_mtp_g]
                qhi_pos_mg = qhi_per_mtp[_mtp_g]
                qhead = kv_h * fx.Int32(query_group_size) + qhi_pos_mg

                if _is_direct:
                    out_row = qo_start + qi_val_mg
                    for vhe in range_constexpr(_VHELOOP):
                        hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * fx.Int32(4)
                        out_off = out_row * stride_out_seq + qhead * stride_out_head + hs_base
                        fx.memref_store_vec(outelems_norm[vhe].to(_OUTPUT_DTYPE), out_reg)
                        fx.copy(copy_out, out_reg, fx.slice(out_tiles, (None, out_off // fx.Int32(4))))
                else:
                    _po_row = _po_row_base + qi_val_mg
                    for vhe in range_constexpr(_VHELOOP):
                        hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * fx.Int32(4)
                        po_off = _po_row * stride_po_ql + qhead * fx.Int32(head_dim) + hs_base
                        fx.memref_store_vec(outelems_norm[vhe], partial_out_reg)
                        fx.copy(
                            copy_f32x4,
                            partial_out_reg,
                            fx.slice(partial_out_tiles, (None, po_off // fx.Int32(4))),
                        )

                    safe_sum_lse = (running_sum > fx.Float32(0.0)).select(running_sum, fx.Float32(1.0))
                    log_sum = fmath.log(safe_sum_lse, fastmath=arith.FastMathFlags.fast)
                    lse_val = running_max + log_sum
                    pl_off = _po_row * stride_pl_ql + qhead
                    fx.memref_store_vec(fx.Vector.from_elements([lse_val], dtype=fx.Float32), partial_lse_reg)
                    fx.copy(copy_f32, partial_lse_reg, fx.slice(partial_lse, (None, pl_off)))

    @flyc.jit
    def launch_pa_decode_metadata(
        out: fx.Int64,
        po: fx.Int64,
        pl: fx.Int64,
        q: fx.Int64,
        kc: fx.Int64,
        vc: fx.Int64,
        cl: fx.Int64,
        ks: fx.Int64,
        vs: fx.Int64,
        work_indptr: fx.Int64,
        work_info: fx.Int64,
        kv_page_indices: fx.Int64,
        kv_indptr: fx.Int64,
        partition_indptr: fx.Int64,
        s_q_seq,
        s_q_head,
        s_k_block,
        s_k_head,
        s_v_block,
        s_v_head,
        s_out_seq,
        s_out_head,
        s_po_partial,
        s_pl_partial,
        s_ks_block,
        s_ks_head,
        s_po_ql,
        s_pl_ql,
        num_sm,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_decode_metadata_kenrel(
            out,
            po,
            pl,
            q,
            kc,
            vc,
            cl,
            ks,
            vs,
            work_indptr,
            work_info,
            kv_page_indices,
            kv_indptr,
            partition_indptr,
            s_q_seq,
            s_q_head,
            s_k_block,
            s_k_head,
            s_v_block,
            s_v_head,
            s_out_seq,
            s_out_head,
            s_po_partial,
            s_pl_partial,
            s_ks_block,
            s_ks_head,
            s_po_ql,
            s_pl_ql,
        ).launch(grid=(num_sm, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        "launch": launch_pa_decode_metadata,
        "kernel": pa_decode_metadata_kenrel,
    }


# One thread per output element serially combines splits with online softmax.
@functools.lru_cache(maxsize=64)
def compile_pa_metadata_reduce(
    *,
    query_length: int,
    num_query_heads: int,
    head_dim: int,
    output_dtype_str: str,
):
    block_threads = head_dim
    assert 0 < block_threads <= 1024, "head_dim must fit in one workgroup"
    output_dtype = _PA_OUTPUT_DTYPES[output_dtype_str]

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_metadata_reduce_kernel(
        final_output_ptr: fx.Tensor,
        partial_output_ptr: fx.Tensor,
        partial_lse_ptr: fx.Tensor,
        reduce_indptr_ptr: fx.Tensor,
        reduce_final_map_ptr: fx.Tensor,
        reduce_partial_map_ptr: fx.Tensor,
        stride_out_seq: Int32,
        stride_out_head: Int32,
        stride_po_row: Int32,  # num_query_heads * head_dim
        stride_pl_row: Int32,  # num_query_heads
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        g = fx.Int32(gpu.block_id("x"))
        qhead = fx.Int32(gpu.block_id("y"))
        qi = fx.Int32(gpu.block_id("z"))

        scalar_layout = fx.make_layout(1, 1)
        flat_layout = fx.make_layout(_FLAT_BUFFER_ELEMENTS, 1)

        def _flat_divide(tensor):
            buffer_tensor = fx.rocdl.make_buffer_tensor(tensor)
            flat = fx.make_view(fx.get_iter(buffer_tensor), flat_layout)
            return fx.logical_divide(flat, scalar_layout)

        final_output = _flat_divide(final_output_ptr)
        partial_output = _flat_divide(partial_output_ptr)
        partial_lse = _flat_divide(partial_lse_ptr)
        reduce_indptr = _flat_divide(reduce_indptr_ptr)
        reduce_final_map = _flat_divide(reduce_final_map_ptr)
        reduce_partial_map = _flat_divide(reduce_partial_map_ptr)

        copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_output = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if output_dtype.width == 32 else fx.rocdl.BufferCopy16b(),
            output_dtype,
        )
        reg_i32 = fx.make_rmem_tensor(1, fx.Int32)
        reg_f32 = fx.make_rmem_tensor(1, fx.Float32)
        reg_output = fx.make_rmem_tensor(1, output_dtype)

        lo = _copy_load(reduce_indptr, g, copy_i32, reg_i32)[0]
        hi = _copy_load(reduce_indptr, g + fx.Int32(1), copy_i32, reg_i32)[0]

        if hi > lo:
            qo_start = _copy_load(reduce_final_map, g * fx.Int32(2), copy_i32, reg_i32)[0]
            out_row = qo_start + qi

            for slot, st in range(
                lo,
                hi,
                fx.Int32(1),
                init=[fx.Float32(float("-inf")), fx.Float32(0.0), fx.Float32(0.0)],
            ):
                m = fx.Float32(st[0])
                denom = fx.Float32(st[1])
                acc = fx.Float32(st[2])
                prow = _copy_load(reduce_partial_map, slot, copy_i32, reg_i32)[0] + qi
                lse = _copy_load(partial_lse, prow * stride_pl_row + qhead, copy_f32, reg_f32)[0]
                v = _copy_load(
                    partial_output,
                    prow * stride_po_row + qhead * fx.Int32(head_dim) + tid,
                    copy_f32,
                    reg_f32,
                )[0]
                m_new = m.maximumf(lse)
                scale_old = exp2_f32_fast((m - m_new) * fx.Float32(LOG2E))
                w = exp2_f32_fast((lse - m_new) * fx.Float32(LOG2E))
                denom_new = denom * scale_old + w
                acc_new = acc * scale_old + v * w
                results = yield [m_new, denom_new, acc_new]

            denom_f = fx.Float32(results[1])
            acc_f = fx.Float32(results[2])
            safe_denom = (denom_f > fx.Float32(0.0)).select(denom_f, fx.Float32(1.0))
            out_acc = acc_f * rcp_f32(safe_denom)

            out_off = out_row * stride_out_seq + qhead * stride_out_head + tid
            output_val = fx.Float32(out_acc).to(output_dtype)
            fx.memref_store_vec(fx.Vector.from_elements([output_val], dtype=output_dtype), reg_output)
            fx.copy(copy_output, reg_output, fx.slice(final_output, (None, out_off)))

    @flyc.jit
    def launch_pa_metadata_reduce(
        final_output,
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        stride_out_seq,
        stride_out_head,
        stride_po_row,
        stride_pl_row,
        num_groups,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_metadata_reduce_kernel(
            final_output,
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            stride_out_seq,
            stride_out_head,
            stride_po_row,
            stride_pl_row,
        ).launch(
            grid=(num_groups, num_query_heads, query_length),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return {"launch": launch_pa_metadata_reduce, "kernel": pa_metadata_reduce_kernel}


_PA_PS_REDUCE_DTYPE_STR = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def pa_metadata_reduce(
    *,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    num_query_heads: int,
    head_dim: int,
    stream=None,
) -> None:
    """Deterministic drop-in replacement for ``aiter.pa_reduce_v1`` on the PS path.

    ``partial_output`` / ``partial_lse`` must already be the ``[query_length:]``
    slices (same as the old pa_reduce_v1 call site). One reduce group is launched
    per batch tile; empty groups (``reduce_indptr`` delta 0 — direct outputs) skip
    in-kernel, so passing ``reduce_indptr.numel() - 1`` groups needs no host sync.
    """
    num_groups = reduce_indptr.numel() - 1
    stride_po_row = num_query_heads * head_dim
    stride_pl_row = num_query_heads
    out_dtype_str = _PA_PS_REDUCE_DTYPE_STR[final_output.dtype]
    compiled = compile_pa_metadata_reduce(
        query_length=int(max_seqlen_q),
        num_query_heads=int(num_query_heads),
        head_dim=int(head_dim),
        output_dtype_str=out_dtype_str,
    )
    _run_compiled(
        compiled["launch"],
        final_output,
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        int(final_output.stride(0)),
        int(final_output.stride(1)),
        stride_po_row,
        stride_pl_row,
        num_groups,
        stream,
    )
