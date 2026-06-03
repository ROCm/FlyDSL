# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL sliding-window paged attention decode kernel."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr import math as fly_math
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels import dpp_utils

# ── Kernel geometry constants ────────────────────────────────────────
QUERY_GROUP_SIZE = 16
HEAD_SIZE = 128
KV_BLOCK_SIZE = 1024  # physical page size (matches SP3 kBlockSize)
KV_COMPUTE_BLOCK = 256  # tile size (matches SP3 kTileKV)
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256
MFMA_N = 16
MFMA_K = 32

TOKENS_PER_WARP = KV_COMPUTE_BLOCK // NUM_WARPS  # 64
TLOOP = TOKENS_PER_WARP // MFMA_N  # 4
ROWS_PER_WARP = WARP_SIZE // MFMA_N  # 4
FP8_ELEMS_16B = 16  # 16 FP8 per 16-byte load
QKHE_PER_FETCH = FP8_ELEMS_16B * ROWS_PER_WARP  # 64
QKHELOOP = HEAD_SIZE // QKHE_PER_FETCH  # 2

VHELOOP = HEAD_SIZE // MFMA_N // NUM_WARPS  # 2
VTLOOP = NUM_WARPS  # 4
Q_ELEMS_PER_LANE = 8
Q_CHUNKS_PER_LANE = Q_ELEMS_PER_LANE // 4
Q_LANES_PER_HEAD = HEAD_SIZE // Q_ELEMS_PER_LANE

# LDS sizes
PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4  # 512
LDS_SCALE_V_PADDING = 4  # break K/V same-bank paired writes
LDS_SCALE_V_OFFSET = KV_COMPUTE_BLOCK + LDS_SCALE_V_PADDING
LDS_SCALE_BYTES = (LDS_SCALE_V_OFFSET + KV_COMPUTE_BLOCK) * 4  # K/V per-token scale staging

FP8_MAX = 240.0
LOG2E = 1.4426950408889634

# Number of loop-carried K values (i64)
_N_K = TLOOP * QKHELOOP * 2  # 16
# Number of loop-carried V values (i64)
_N_V = VHELOOP * VTLOOP * 2  # 16

# Tiles per block (1024 tokens / 256 tokens per tile = 4, matches SP3 kNumBlockTiles)
TILES_PER_BLOCK = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK  # 4


def _cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


def _get_sw_mtp_group_count(query_length: int, query_group_size: int) -> int:
    return _cdiv(query_length * query_group_size, MFMA_N)


def _get_sw_mtp_pair_offset(mtp_group_idx: int, mtp_subgroup_idx: int = 0) -> int:
    return mtp_group_idx * MFMA_N + mtp_subgroup_idx * MFMA_N


def _pow2_shift(value: int) -> int:
    assert value > 0 and (value & (value - 1)) == 0
    return value.bit_length() - 1


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _udiv_pow2(value, divisor: int):
    return value >> fx.Int32(_pow2_shift(divisor))


def _urem_pow2(value, divisor: int):
    return value & fx.Int32(divisor - 1)


def _udiv_const(value, divisor: int):
    if const_expr(_is_pow2(divisor)):
        return _udiv_pow2(value, divisor)
    return value // fx.Int32(divisor)


def _urem_const(value, divisor: int):
    if const_expr(_is_pow2(divisor)):
        return _urem_pow2(value, divisor)
    return value % fx.Int32(divisor)


def _compute_block_base_dw_i64(phys_block, block_stride, head_offset):
    phys_block_i64 = fx.Int64(phys_block)
    block_stride_i64 = fx.Int64(block_stride)
    head_offset_i64 = fx.Int64(head_offset)
    return (phys_block_i64 * block_stride_i64 + head_offset_i64) >> fx.Int64(2)


def _extract_global_ptr(tensor):
    from flydsl._mlir.dialects import fly as _fly

    raw = tensor.ir_value() if hasattr(tensor, "ir_value") and not isinstance(tensor, ir.Value) else tensor
    ptr_type = ir.Type.parse("!llvm.ptr<1>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, raw)


def _global_load_i64x2(global_ptr, byte_offset_i64):
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=fx.Int64(byte_offset_i64), elem_type=T.i8)
    return llvm.LoadOp(T.i64x2, ptr, alignment=16).result


def _global_load_i32(global_ptr, elem_offset_i32):
    byte_offset_i64 = fx.Int64(elem_offset_i32) * fx.Int64(4)
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=byte_offset_i64, elem_type=T.i8)
    return llvm.LoadOp(T.i32, ptr, alignment=4).result


def _rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def _exp2_f32_fast(value):
    return fly_math.exp2(value, fastmath=arith.FastMathFlags.fast)


def _load_k_flat(
    k_global_ptr,
    k_block_base_dw_i64,
    tile_token_offset_i32,
    k_tok_thread_base,
    c_tok_stride_dw,
    k_he_off_dw,
    *,
    sched_vmem_after_load=True,
):
    k_flat = []
    tile_tok_base = tile_token_offset_i32 + k_tok_thread_base

    for td in range_constexpr(TLOOP):
        kbo = tile_tok_base + fx.Int32(td * MFMA_N)
        kbo_dw = kbo * c_tok_stride_dw
        for qkhe in range_constexpr(QKHELOOP):
            ka_dw = k_block_base_dw_i64 + fx.Int64(kbo_dw + k_he_off_dw[qkhe])
            k2 = _global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
            if const_expr(sched_vmem_after_load):
                rocdl.sched_barrier(rocdl.mask_vmem_rd)
            k2_words = fx.Vector(k2)
            k_flat.append(k2_words[0])
            k_flat.append(k2_words[1])

    return k_flat


def _unflatten_k(k_flat):
    return [[k_flat[td * (QKHELOOP * 2) + j] for j in range(QKHELOOP * 2)] for td in range(TLOOP)]


def _build_pa_thread_invariants(
    warp_id,
    lane16id,
    rowid,
    *,
    trans_v,
    per_token_kv,
):
    c_tokens_per_warp = fx.Int32(TOKENS_PER_WARP)
    c_mfma_n = fx.Int32(MFMA_N)
    k_tok_thread_base = warp_id * c_tokens_per_warp + lane16id
    c_tok_stride_dw = fx.Int32(FP8_ELEMS_16B // 4)
    c_he_stride_dw = fx.Int32(KV_BLOCK_SIZE * FP8_ELEMS_16B // 4)
    k_he_off_dw = [rowid * c_he_stride_dw + fx.Int32(qkhe * 4) * c_he_stride_dw for qkhe in range(QKHELOOP)]

    vhead_elems = [fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * c_mfma_n + lane16id for vhe in range(VHELOOP)]
    v_tok_thread_off = [fx.Int32(vt * TOKENS_PER_WARP) + rowid * c_mfma_n for vt in range(VTLOOP)]
    if const_expr(trans_v):
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(FP8_ELEMS_16B // 4) for vhe in range(VHELOOP)]
    else:
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(KV_BLOCK_SIZE // 4) for vhe in range(VHELOOP)]

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
        k_tok_thread_base,
        c_tok_stride_dw,
        k_he_off_dw,
        v_tok_thread_off,
        vhead_elem_dw,
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


def _compute_sw_mtp_group_state(
    lane16id,
    local_qhead_idx,
    *,
    mtp_group_idx,
    mtp_subgroup_idx=0,
    query_length,
    query_group_size,
):
    g_off = _get_sw_mtp_pair_offset(mtp_group_idx, mtp_subgroup_idx)
    lane_pair_raw = lane16id + fx.Int32(g_off)
    c_total_pairs = fx.Int32(query_length * query_group_size)
    c_pair_max = fx.Int32(query_length * query_group_size - 1)
    c_ql_m1 = fx.Int32(query_length - 1)

    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        lane_pair = lane_pair_raw
    else:
        lane_pair = arith.select(lane_pair_raw < c_total_pairs, lane_pair_raw, c_pair_max)
    qi_raw = _udiv_const(lane_pair, query_group_size)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        qi_val = qi_raw
    else:
        qi_val = arith.select(qi_raw < c_ql_m1, qi_raw, c_ql_m1)
    qhi_pos = _urem_const(lane_pair, query_group_size)

    lqh_pair_raw = local_qhead_idx + fx.Int32(g_off)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        lqh_pair = lqh_pair_raw
    else:
        lqh_pair = arith.select(lqh_pair_raw < c_total_pairs, lqh_pair_raw, c_pair_max)
    lqi_raw = _udiv_const(lqh_pair, query_group_size)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        qi_for_q = lqi_raw
    else:
        qi_for_q = arith.select(lqi_raw < c_ql_m1, lqi_raw, c_ql_m1)
    local_qhead_idx_for_q = _urem_const(lqh_pair, query_group_size)
    return qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q


@flyc.jit
def _prefetch_q_chunks(
    q_rsrc,
    q_base,
    lane16id,
    *,
    query_load_is_bf16,
):
    # bf16/f16 + in-kernel query_scale path.  Each lane owns 8 Q elements,
    # loaded as 2 × vec_width=4 buffer loads (4 bf16/f16 elems per load = 8 B,
    # element offset += 4 per iter).  After FP8 packing each load produces
    # one i32 word, so the per-lane store is `vec<2, i32>` = 8 B = 1 i64.
    q_load_lane = lane16id
    if const_expr(Q_LANES_PER_HEAD < MFMA_N):
        q_load_lane = arith.select(lane16id < fx.Int32(Q_LANES_PER_HEAD), lane16id, fx.Int32(0))
    q_elem = q_base + q_load_lane * fx.Int32(Q_ELEMS_PER_LANE)
    q_chunks = []
    for qwi in range_constexpr(Q_CHUNKS_PER_LANE):
        q_chunks.append(
            buffer_ops.buffer_load(
                q_rsrc,
                q_elem + fx.Int32(qwi * 4),
                vec_width=4,
                dtype=fx.BFloat16 if query_load_is_bf16 else fx.Float16,
            )
        )
    return q_chunks


@flyc.jit
def _finish_q_fragments(
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    q_chunks,
    lane16id,
    rowid,
    local_qhead_idx,
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
    c_head_size = fx.Int32(HEAD_SIZE)
    lds_q_base = local_qhead_idx * c_head_size + lane16id * 8
    abs_mask = fx.Vector.filled(4, 0x7FFFFFFF, fx.Int32)
    c_zero_f = fx.Float32(0.0)
    c_one_f = fx.Float32(1.0)
    fx.Float32(FP8_MAX)
    q_f32_chunks = []
    local_max = c_zero_f
    for q_src in q_chunks:
        q_f32 = fx.Vector(q_src).to(fx.Float32)
        q_f32_chunks.append(q_f32)
        q_i32 = q_f32.bitcast(fx.Int32)
        q_abs_i32 = q_i32 & abs_mask
        q_abs = q_abs_i32.bitcast(fx.Float32)
        chunk_max = q_abs.reduce("max")
        local_max = local_max.maximumf(chunk_max)

    for sh in [8, 4, 2, 1]:
        local_max = local_max.maximumf(dpp_utils.dpp_xor_f32(local_max, sh))
    query_scale_lane = fx.Float32(
        arith.select(
            local_max > c_zero_f,
            local_max * fx.Float32(1.0 / FP8_MAX).ir_value(),
            c_one_f,
        )
    )
    inv_query_scale = _rcp_f32(query_scale_lane)
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
    if const_expr(Q_LANES_PER_HEAD < MFMA_N):
        if lane16id < fx.Int32(Q_LANES_PER_HEAD):
            v01.store(logits_lds_i32, [fx.Index(lds_q_i32)])
    else:
        v01.store(logits_lds_i32, [fx.Index(lds_q_i32)])

    q_frags = []
    gpu.barrier()
    query_scale_lane = fx.Vector.load(T.vec(1, fx.Float32.ir_type), softmax_lds_f32, [fx.Index(lane16id)])[0].ir_value()
    for qkhe in range_constexpr(QKHELOOP):
        for qkr in range_constexpr(2):
            # See layout comment above. Byte offset:
            #   lane16id * HEAD_SIZE + qkhe*64 + rowid*16 + qkr*8
            lds_rd_byte = lane16id * c_head_size + fx.Int32(qkhe << 6) + (rowid << fx.Int32(4)) + fx.Int32(qkr << 3)
            lds_rd_base = lds_rd_byte >> fx.Int32(3)
            q_v1 = fx.Vector.load(T.vec(1, T.i64), logits_lds_i64, [fx.Index(lds_rd_base)])
            q_frags.append(q_v1[0])
    return q_frags, query_scale_lane


def _prefetch_sw_mtp_group_queries(
    q_rsrc,
    batch_idx,
    kv_h,
    stride_q_seq,
    stride_q_head,
    lane16id,
    local_qhead_idx,
    *,
    mtp_group_idx,
    mtp_subgroup_count,
    query_length,
    query_group_size,
    query_load_is_bf16,
):
    mtp_prefetches = []
    c_query_length = arith.constant(query_length, type=T.i32)
    c_query_group_size = arith.constant(query_group_size, type=T.i32)
    for mtp_subgroup_idx in range_constexpr(mtp_subgroup_count):
        qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q = _compute_sw_mtp_group_state(
            lane16id,
            local_qhead_idx,
            mtp_group_idx=mtp_group_idx,
            mtp_subgroup_idx=mtp_subgroup_idx,
            query_length=query_length,
            query_group_size=query_group_size,
        )
        q_row = batch_idx * c_query_length + qi_for_q
        q_base = q_row * stride_q_seq + (kv_h * c_query_group_size + local_qhead_idx_for_q) * stride_q_head
        q_chunks = _prefetch_q_chunks(
            q_rsrc,
            q_base,
            lane16id,
            query_load_is_bf16=query_load_is_bf16,
        )
        mtp_prefetches.append((qi_val, qhi_pos, q_chunks))
    return mtp_prefetches


def _finish_sw_mtp_subgroup_q_fragments(
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    mtp_prefetches,
    lane16id,
    rowid,
    local_qhead_idx,
    *,
    mtp_subgroup_idx,
):
    qi_val, qhi_pos, q_chunks = mtp_prefetches[mtp_subgroup_idx]
    q_frags, query_scale_lane = _finish_q_fragments(
        logits_lds_i32,
        logits_lds_i64,
        softmax_lds_f32,
        q_chunks,
        lane16id,
        rowid,
        local_qhead_idx,
    )
    return qi_val, qhi_pos, q_frags, query_scale_lane


def _normalize_pa_output(running_sum, outs, zero_f):
    one_f = fx.Float32(1.0).ir_value()
    safe_sum = arith.select(running_sum > zero_f, running_sum, one_f)
    inv_sum = _rcp_f32(safe_sum)
    normalized_outs = []
    for vhe in range_constexpr(VHELOOP):
        normalized_outs.append(outs[vhe] * vector.broadcast(T.f32x4, inv_sum))
    return normalized_outs


def _make_pa_phase_helpers(
    *,
    trans_v,
    per_token_q,
    per_token_kv,
    needs_mask,
    query_length,
    kv_h,
    v_global_ptr,
    ks_rsrc,
    vs_rsrc,
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    scale_lds_f32,
    stride_ks_block,
    stride_ks_head,
    softmax_scale_base,
    softmax_q_scale,
    k_scale_val,
    scale,
    v_scale_val,
    warp_id,
    lane16id,
    rowid,
    k_tok_thread_base,
    v_tok_thread_off,
    vhead_elem_dw,
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
):
    # Sliding-window decode always needs an upper-bound mask: even for a
    # single query, the tail block can contain tokens beyond context_len.
    pv_prob_i64_indices = []
    for vt in range_constexpr(VTLOOP):
        for j in range_constexpr(2):
            p_byte = (
                arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                + pv_prob_read_base
                + arith.constant(j * 8, type=T.i32)
            )
            pv_prob_i64_indices.append(fx.Index(p_byte >> fx.Int32(3)))

    def _load_kv_scale_scalars(tile_token_offset_i32, phys_block):
        if const_expr(per_token_kv):
            scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            scale_global_token = tile_token_offset_i32 + scale_stage_token
            k_scale_scalar = buffer_ops.buffer_load(
                ks_rsrc,
                scale_block_base + scale_global_token,
                vec_width=1,
                dtype=fx.Float32,
            )
            v_scale_scalar = buffer_ops.buffer_load(
                vs_rsrc,
                scale_block_base + scale_global_token,
                vec_width=1,
                dtype=fx.Float32,
            )
            return k_scale_scalar, v_scale_scalar
        return None

    def _load_v_and_scales(
        v_block_base_dw,
        tile_token_offset_i32,
        *,
        preloaded_scale_scalars=None,
    ):
        if const_expr(per_token_kv):
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            k_scale_scalar, v_scale_scalar = preloaded_scale_scalars
            fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32).store(
                scale_lds_f32,
                [fx.Index(scale_stage_token)],
            )
            fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32).store(
                scale_lds_f32,
                [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + scale_stage_token)],
            )
            rocdl.sched_barrier(rocdl.mask_vmem_rd)

        v_results = []
        for vt in range_constexpr(VTLOOP):
            vhe_data = []
            for vhe in range_constexpr(VHELOOP):
                v_token_in_block = tile_token_offset_i32 + v_tok_thread_off[vt]
                if const_expr(trans_v):
                    vt_group = v_token_in_block >> fx.Int32(4)
                    va_dw_delta = (
                        vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + vhead_elem_dw[vhe]
                    )
                else:
                    va_dw_delta = vhead_elem_dw[vhe] + (v_token_in_block >> fx.Int32(2))
                va_byte = (v_block_base_dw + fx.Int64(va_dw_delta)) * fx.Int64(4)
                v_i64x2 = _global_load_i64x2(v_global_ptr, va_byte)
                rocdl.sched_barrier(rocdl.mask_vmem_rd)
                vhe_data.append(v_i64x2)
            v_results.append(vhe_data)

        return v_results

    def _scale_row_base(td: int):
        return kv_tok_thread_base + fx.Int32(td * MFMA_N)

    def _load_k_scale_vec(td: int):
        return vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(_scale_row_base(td))])

    def _load_v_scale_vec(td: int):
        return vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + _scale_row_base(td))])

    def _store_vmax_warp(partition_start, *, seq_end=None):
        if const_expr(per_token_kv):
            kv_tok_base = partition_start + kv_tok_thread_base if const_expr(seq_end is not None) else None
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = _load_v_scale_vec(td)
                for i in range_constexpr(4):
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        vs_i = arith.select(kv_tok < seq_end, vs_i, zero_f)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                v_max_warp = v_max_warp.maximumf(fx.Vector(vs).reduce("max"))
            for sh in [32, 16]:
                v_max_warp = v_max_warp.maximumf(v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
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

    def _apply_token_mask_vec(logit_vec, td: int, kv_tok_base, causal_bound, seq_start, false_value):
        tok_vec = _token_vec_i32(kv_tok_base, td)
        if const_expr(needs_mask and seq_start is not None):
            in_range = (tok_vec < causal_bound) & (tok_vec >= seq_start)
        elif const_expr(needs_mask):
            in_range = tok_vec < causal_bound
        else:
            in_range = tok_vec >= seq_start
        return arith.select(in_range, logit_vec, vector.broadcast(T.f32x4, arith.unwrap(false_value)))

    def _qk_and_intra_softmax(
        k_ops,
        partition_start,
        q_frags,
        causal_bound,
        query_scale_lane=None,
        *,
        seq_start=None,
    ):

        query_scale_vec = None
        if const_expr(per_token_q):
            query_scale_vec = vector.broadcast(T.f32x4, query_scale_lane * softmax_scale_base)
        d_out = []
        for td in range_constexpr(TLOOP):
            acc = arith.constant_vector(0.0, T.f32x4)
            for k_step in range_constexpr(QKHELOOP * 2):
                acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
            if const_expr(per_token_kv):
                k_scale_vec = _load_k_scale_vec(td)
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

        apply_range_mask = seq_start is not None
        kv_tok_base = partition_start + kv_tok_thread_base if const_expr(needs_mask or apply_range_mask) else None
        qk_max = neg_inf
        for td in range_constexpr(TLOOP):
            logits_vec = d_out[td]
            if const_expr(kv_tok_base is not None):
                logits_vec = _apply_token_mask_vec(logits_vec, td, kv_tok_base, causal_bound, seq_start, neg_inf)
                d_out[td] = logits_vec
            qk_max = qk_max.maximumf(fx.Vector(logits_vec).reduce("max"))
        for sh in [32, 16]:
            qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
        vector.store(
            fx.Vector.from_elements([qk_max], dtype=fx.Float32),
            softmax_lds_f32,
            [sm_max_off],
        )

        exp_sum = zero_f
        safe_qk_max = arith.select(qk_max > neg_inf, qk_max, zero_f) if const_expr(kv_tok_base is not None) else qk_max
        for td in range_constexpr(TLOOP):
            diff_vec = fx.Vector(d_out[td]) - vector.broadcast(T.f32x4, arith.unwrap(safe_qk_max))
            p_vec = _exp2_f32_fast(diff_vec * vector.broadcast(T.f32x4, arith.unwrap(fx.Float32(LOG2E))))
            exp_sum = exp_sum + fx.Vector(p_vec).reduce("add")
            d_out[td] = p_vec
        for sh in [32, 16]:
            exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
        vector.store(
            fx.Vector.from_elements([exp_sum], dtype=fx.Float32),
            softmax_lds_f32,
            [sm_sum_off],
        )

        return d_out

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs):
        partition_max = neg_inf
        partition_sum = zero_f
        warp_rescale_factors = []
        max_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_rd_max_offs[0]]))
        for w in range_constexpr(NUM_WARPS):
            w_max = max_vec[w]
            partition_max = partition_max.maximumf(w_max)
            warp_rescale_factors.append(w_max)
        sum_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_rd_sum_offs[0]]))
        for w in range_constexpr(NUM_WARPS):
            diff_w = warp_rescale_factors[w] - partition_max
            if const_expr(needs_mask):
                diff_w = arith.select(partition_max > neg_inf, diff_w, zero_f)
            wf = _exp2_f32_fast(diff_w * fx.Float32(LOG2E).ir_value())
            w_sum = sum_vec[w]
            wf_sum = arith.mulf(arith.unwrap(w_sum), arith.unwrap(wf), fastmath=arith.FastMathFlags.contract)
            partition_sum = arith.addf(arith.unwrap(partition_sum), wf_sum, fastmath=arith.FastMathFlags.contract)
            warp_rescale_factors[w] = wf

        my_warp_rescale = warp_rescale_factors[0]
        for w in range_constexpr(1, NUM_WARPS):
            my_warp_rescale = arith.select(
                warp_id == arith.constant(w, type=T.i32),
                warp_rescale_factors[w],
                my_warp_rescale,
            )

        new_rmax = rmax.maximumf(partition_max)
        if const_expr(needs_mask):
            accum_scale = arith.select(
                rmax > neg_inf,
                _exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value()),
                zero_f,
            )
            part_to_new = arith.select(
                partition_max > neg_inf,
                _exp2_f32_fast((partition_max - new_rmax) * fx.Float32(LOG2E).ir_value()),
                zero_f,
            )
        else:
            accum_scale = _exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value())
            part_to_new = _exp2_f32_fast((partition_max - new_rmax) * fx.Float32(LOG2E).ir_value())

        accum_sum = arith.mulf(arith.unwrap(accum_scale), arith.unwrap(rsum), fastmath=arith.FastMathFlags.contract)
        partition_sum_scaled = arith.mulf(
            arith.unwrap(partition_sum),
            arith.unwrap(part_to_new),
            fastmath=arith.FastMathFlags.contract,
        )
        rsum = arith.addf(accum_sum, partition_sum_scaled, fastmath=arith.FastMathFlags.contract)
        rmax = new_rmax
        accum_scale_vec = vector.broadcast(T.f32x4, arith.unwrap(accum_scale))
        for vhe in range_constexpr(VHELOOP):
            outs[vhe] = outs[vhe] * accum_scale_vec

        if const_expr(per_token_kv):
            v_max_global = zero_f
            vmax_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_vmax_rd_offs[0]]))
            for w in range_constexpr(NUM_WARPS):
                w_vmax = vmax_vec[w]
                v_max_global = v_max_global.maximumf(w_vmax)
            v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX).ir_value()
            v_max_safe_scaled = v_max_scaled + fx.Float32(1e-8 / FP8_MAX).ir_value()
            norm_factor = _rcp_f32(v_max_safe_scaled)
            prob_scale = my_warp_rescale
            v_correction = v_max_scaled * part_to_new
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * (
                    _load_v_scale_vec(td) * vector.broadcast(T.f32x4, arith.unwrap(prob_scale * norm_factor))
                )
        else:
            prob_scale = my_warp_rescale * part_to_new
            v_correction = v_scale_val
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * vector.broadcast(T.f32x4, arith.unwrap(prob_scale))

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
        for vhe in range_constexpr(VHELOOP):
            tmp_out = arith.constant_vector(0.0, T.f32x4)
            for vt in range_constexpr(VTLOOP):
                v_i64x2 = fx.Vector(v_ops[vt][vhe])
                for j in range_constexpr(2):
                    p_i64_idx = pv_prob_i64_indices[vt * 2 + j]
                    p_i64 = fx.Vector.load(T.vec(1, T.i64), logits_lds_i64, [p_i64_idx])[0]
                    tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4,
                        [
                            v_i64x2[j],
                            p_i64,
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


def get_sw_max_context_partition_num(
    sliding_window: int,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    if sliding_window <= 0:
        return 0
    window_token_count = sliding_window + query_length
    return _cdiv(window_token_count - 1, context_partition_size) + 1


@functools.lru_cache(maxsize=256)
def compile_pa_decode_sw_reduce(
    *,
    max_context_partition_num: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    output_dtype_str: str,
):
    block_threads = head_size
    assert block_threads > 0, "head_size must be positive"
    assert block_threads <= 1024, "head_size must fit in one workgroup"
    reduce_width = 1 if max_context_partition_num <= 1 else 1 << ((max_context_partition_num - 1).bit_length())
    reduce_shuffle_offsets = [off for off in [32, 16, 8, 4, 2, 1] if off < reduce_width]
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)
    arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_sw_reduce_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_off + red_slots * 4
    part_weights_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = part_weights_off + max_context_partition_num * 4

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_decode_sw_reduce_kernel(
        output_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        logits_ptr: fx.Tensor,
        stride_output_bs: Int32,
        stride_output_len: Int32,
        stride_output_kv_head: Int32,
        stride_output_group_size: Int32,
        stride_exp_sums_seq: Int32,
        stride_exp_sums_head: Int32,
        stride_exp_sums_part: Int32,
        stride_logits_seq: Int32,
        stride_logits_head: Int32,
        stride_logits_part: Int32,
        stride_logits_group: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        batch_idx = fx.Int32(gpu.block_id("x"))
        kv_head_idx = fx.Int32(gpu.block_id("y"))
        eqgs_idx = fx.Int32(gpu.block_id("z"))

        smem_base = allocator.get_base()
        red_scratch = SmemPtr(smem_base, red_off, T.f32, shape=(red_slots,))
        red_scratch.get()
        if const_expr(max_context_partition_num > WARP_SIZE):
            part_weights_lds = SmemPtr(smem_base, part_weights_off, T.f32, shape=(max_context_partition_num,))
            part_weights_lds.get()

        out_rsrc = buffer_ops.create_buffer_resource(output_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        logits_rsrc = buffer_ops.create_buffer_resource(logits_ptr, max_size=True)

        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        fm_fast = arith.FastMathFlags.fast

        c_w = fx.Int32(WARP_SIZE)
        c_wave_mask = fx.Int32(WARP_SIZE - 1)
        c_red_slots = fx.Int32(red_slots)
        lane = tid & c_wave_mask
        wave = fx.Int32(tid >> fx.Int32(6))

        def _wave_reduce_max_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.maximumf(red.shuffle_xor(fx.Int32(sh), c_w))
            return red

        def _wave_reduce_sum_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.addf(
                    red.shuffle_xor(fx.Int32(sh), c_w),
                    fastmath=fm_fast,
                )
            return red

        def _block_reduce(val, mode):
            if const_expr(red_slots == 1):
                return _wave_reduce_max_full(val) if const_expr(mode == "max") else _wave_reduce_sum_full(val)

            neutral = c_neg_inf if const_expr(mode == "max") else c_zero_f
            w = _wave_reduce_max_full(val) if const_expr(mode == "max") else _wave_reduce_sum_full(val)

            if lane == 0:
                wave_idx = fx.Index(wave)
                red_scratch.store(w, [wave_idx])
            gpu.barrier()

            if wave == 0:
                in_range = lane < c_red_slots
                lane_safe = arith.select(in_range, lane, 0)
                lane_safe_idx = fx.Index(lane_safe)
                red_val = red_scratch.load([lane_safe_idx])
                red_val = arith.select(in_range, red_val, neutral)
                red_val = (
                    _wave_reduce_max_full(red_val) if const_expr(mode == "max") else _wave_reduce_sum_full(red_val)
                )
                if lane == 0:
                    red_scratch.store(red_val, [fx.Index(0)])
            gpu.barrier()

            return red_scratch.load([fx.Index(0)])

        if const_expr(max_context_partition_num <= WARP_SIZE):
            c_part_num = fx.Int32(max_context_partition_num)
            c_reduce_width = fx.Int32(reduce_width)

            def _wave_reduce_max(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.maximumf(red.shuffle_xor(fx.Int32(sh), c_w))
                return red

            def _wave_reduce_sum(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.addf(
                        red.shuffle_xor(fx.Int32(sh), c_w),
                        fastmath=fm_fast,
                    )
                return red

            lane_in_range = lane < c_part_num
            lane_in_reduce = lane < c_reduce_width
            part_sum = c_zero_f
            part_max = c_neg_inf
            if lane_in_reduce:
                part_i32 = arith.select(lane_in_range, lane, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                part_max_raw = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                part_sum = arith.select(lane_in_range, part_sum_raw, c_zero_f)
                part_max = arith.select(lane_in_range, part_max_raw, c_neg_inf)

            global_max = _wave_reduce_max(part_max)
            part_scale = arith.select(
                lane_in_range,
                _exp2_f32_fast((part_max - global_max) * c_log2e),
                c_zero_f,
            )
            scaled_sum = part_sum * part_scale
            global_exp_sum = _wave_reduce_sum(scaled_sum)
            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )
            inv_global_exp_sum = _rcp_f32(safe_global_exp_sum)
            weight_local = scaled_sum * inv_global_exp_sum
            weight_local_i32 = arith.bitcast(T.i32, arith.unwrap(weight_local))

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = fx.Int32(part_idx)
                bcast_addr = part_i32 * 4
                weight_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(weight_local_i32))
                weight = arith.bitcast(T.f32, weight_i32)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_bf16 = buffer_ops.buffer_load(logits_rsrc, logits_off, vec_width=1, dtype=fx.BFloat16)
                part_logits = fx.Float32(part_logits_bf16)
                acc = acc + part_logits * weight
        else:
            # Fallback for unusually large sliding-window partition counts.
            global_max = c_neg_inf
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_max_raw = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=fx.Float32)
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                chunk_max = _block_reduce(part_max, "max")
                global_max = global_max.maximumf(chunk_max)

            global_exp_sum = c_zero_f
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                part_max_raw = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                part_sum = arith.select(in_chunk, part_sum_raw, c_zero_f)
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                part_scale = arith.select(
                    in_chunk,
                    _exp2_f32_fast((part_max - global_max) * c_log2e),
                    c_zero_f,
                )
                chunk_sum = _block_reduce(part_sum * part_scale, "sum")
                global_exp_sum = global_exp_sum + chunk_sum

            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )
            inv_global_exp_sum = _rcp_f32(safe_global_exp_sum)

            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                part_max_raw = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                if in_chunk:
                    part_sum = part_sum_raw
                    part_max = part_max_raw
                    part_scale = _exp2_f32_fast((part_max - global_max) * c_log2e)
                    weight = part_sum * part_scale * inv_global_exp_sum
                    part_idx_idx = fx.Index(part_i32)
                    part_weights_lds.store(weight, [part_idx_idx])

            gpu.barrier()

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = fx.Int32(part_idx)
                part_idx_idx = fx.Index(part_idx)
                weight = part_weights_lds.load([part_idx_idx])
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_bf16 = buffer_ops.buffer_load(logits_rsrc, logits_off, vec_width=1, dtype=fx.BFloat16)
                part_logits = fx.Float32(part_logits_bf16)
                acc = acc + part_logits * weight

        query_idx = _udiv_const(eqgs_idx, query_group_size)
        group_idx = _urem_const(eqgs_idx, query_group_size)
        out_off = (
            batch_idx * stride_output_bs
            + query_idx * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
            + tid
        )
        if const_expr(output_dtype_str == "f32"):
            out_val = acc
        elif const_expr(output_dtype_str == "f16"):
            out_val = acc.to(fx.Float16)
        else:
            out_val = acc.to(fx.BFloat16)
        buffer_ops.buffer_store(out_val, out_rsrc, out_off)

    @flyc.jit
    def launch_pa_decode_sw_reduce(
        output,
        exp_sums,
        max_logits,
        logits,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_seq,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_seq,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        batch_size,
        num_kv_heads,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_sw_reduce_kernel(
            output,
            exp_sums,
            max_logits,
            logits,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
        ).launch(
            grid=(batch_size, num_kv_heads, query_seq_len * query_group_size),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_sw_reduce,
        "kernel": pa_decode_sw_reduce_kernel,
        "allocator": allocator,
    }


# =====================================================================
# =====================================================================
# compile_pa_decode_sw — Sliding Window kernel with one CTA per 256-token tile
# Grid = (batch_size, num_kv_heads, max_context_partition_num)
# Each block handles one 256-token context partition. `partition_idx` is decoded
# into (physical_block, 256-token sub-tile) after applying the sliding-window offset.
# Uses block_tables for physical block lookup instead of kv_page_indices.
# Output: exp_sums, max_logits, temporary_output -> reduced by a separate kernel.
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_sw(
    sliding_window: int,  # required > 0 -- baked as compile-time constant
    softmax_scale=None,
    trans_v=False,
    query_group_size=QUERY_GROUP_SIZE,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "bf16",
    fuse_partitions: bool = False,
):
    """Compile a Gluon-style partitioned PA decode kernel for sliding window.

    Grid = (batch_size, num_kv_heads * mtp_groups, max_context_partition_num).
    Each GPU block processes one 256-token partition selected from the visible KV
    region: the sliding tail window.
    sliding_window is a compile-time constant.
    """
    assert sliding_window > 0, "compile_pa_decode_sw requires sliding_window > 0"
    arch = get_hip_arch()
    if query_input_dtype not in ("bf16", "f16"):
        raise ValueError("`compile_pa_decode_sw` only supports bf16/f16 query inputs.")
    query_load_is_bf16 = query_input_dtype == "bf16"
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE**0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024
    _max_context_partition_num = get_sw_max_context_partition_num(
        sliding_window,
        KV_COMPUTE_BLOCK,
        query_length,
    )
    _mtp_groups = _get_sw_mtp_group_count(query_length, query_group_size)

    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    LDS_SCALE_TOTAL = LDS_SCALE_BYTES if const_expr(per_token_kv) else 0
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_sw_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL
    scale_off = allocator.ptr
    allocator.ptr += LDS_SCALE_TOTAL

    @flyc.kernel
    def pa_decode_sw_kernel(
        exp_sums_ptr: fx.Tensor,  # [batch, kv_heads, max_parts, eqgs] f32
        max_logits_ptr: fx.Tensor,  # [batch, kv_heads, max_parts, eqgs] f32
        tmp_out_ptr: fx.Tensor,  # [batch, kv_heads, max_parts, eqgs, head_size] bf16
        out_ptr: fx.Tensor,  # [batch, query_length, kv_heads, query_group_size, head_size] bf16
        query_ptr: fx.Tensor,
        key_cache_ptr: fx.Tensor,
        value_cache_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,  # [batch, max_blocks_per_seq] i32
        context_lengths_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        stride_q_seq: Int32,
        stride_q_head: Int32,
        stride_k_block: Int32,
        stride_k_head: Int32,
        stride_v_block: Int32,
        stride_v_head: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_es_part: Int32,
        stride_to_seq: Int32,
        stride_to_head: Int32,
        stride_to_part: Int32,
        stride_to_group: Int32,
        stride_out_bs: Int32,
        stride_out_len: Int32,
        stride_out_kv_head: Int32,
        stride_out_group_size: Int32,
        stride_bt_seq: Int32,
        stride_ks_block: Int32,
        stride_ks_head: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        batch_idx = fx.Int32(gpu.block_id("x"))
        grid_y = fx.Int32(gpu.block_id("y"))
        kv_h = _udiv_const(grid_y, _mtp_groups)
        mtp_group_from_grid = _urem_const(grid_y, _mtp_groups)
        partition_idx = fx.Int32(gpu.block_id("z"))
        cl_global_ptr = _extract_global_ptr(context_lengths_ptr)
        context_len = _global_load_i32(cl_global_ptr, batch_idx)
        lane16id = tid & 15
        rowid = (tid >> 4) & 3
        warp_id = fx.Int32(tid >> fx.Int32(6))

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_global_ptr = _extract_global_ptr(key_cache_ptr)
        v_global_ptr = _extract_global_ptr(value_cache_ptr)

        bt_global_ptr = _extract_global_ptr(block_tables_ptr)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        to_rsrc = buffer_ops.create_buffer_resource(tmp_out_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)

        q_scale_val = 1.0
        if const_expr(per_token_kv):
            k_scale_val = 1.0
            v_scale_val = 1.0
        else:
            k_scale_val = buffer_ops.buffer_load(ks_rsrc, 0, vec_width=1)
            v_scale_val = buffer_ops.buffer_load(vs_rsrc, 0, vec_width=1)

        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_TOTAL // 4,)).get()
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()
        scale_lds_f32 = None
        if const_expr(per_token_kv):
            scale_lds_f32 = SmemPtr(smem_base, scale_off, T.f32, shape=(LDS_SCALE_BYTES // 4,)).get()

        _softmax_scale_const = arith.constant(_softmax_scale, type=T.f32)
        _softmax_q_scale = _softmax_scale_const * q_scale_val
        _scale = _softmax_q_scale * k_scale_val  # per-tensor only; per-token uses per-token k_scale
        c_w = fx.Int32(WARP_SIZE)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)
        c_cps = fx.Int32(KV_COMPUTE_BLOCK)
        c_bs = fx.Int32(_bs)

        local_qhead_idx = warp_id * 4 + rowid
        (
            _k_tok_thread_base,
            _c_tok_stride_dw,
            _k_he_off_dw,
            _v_tok_thread_off,
            _vhead_elem_dw,
            _kv_tok_thread_base,
            _prob_wr_thread_base,
            _pv_prob_read_base,
            _sm_max_off,
            _sm_sum_off,
            _sm_rd_max_offs,
            _sm_rd_sum_offs,
            _sm_vmax_wr_off,
            _sm_vmax_rd_offs,
        ) = _build_pa_thread_invariants(
            warp_id,
            lane16id,
            rowid,
            trans_v=trans_v,
            per_token_kv=per_token_kv,
        )

        # ── Context length and partition mapping ──
        # Visible tiles cover the union of all per-query sliding windows.

        _c_sw = fx.Int32(sliding_window)
        _c_query_len = fx.Int32(query_length)
        num_tiles_for_seq = (context_len + c_cps - 1) >> fx.Int32(8)
        seq_start_global = context_len - _c_query_len - _c_sw
        seq_start_global = arith.select(seq_start_global > 0, seq_start_global, 0)
        tail_start_tile = seq_start_global >> fx.Int32(8)
        visible_tile_count = num_tiles_for_seq - tail_start_tile
        tile_partition_idx_raw = tail_start_tile + partition_idx

        _is_valid = partition_idx < visible_tile_count

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
            per_token_q=True,
            per_token_kv=per_token_kv,
            needs_mask=True,
            query_length=query_length,
            kv_h=kv_h,
            v_global_ptr=v_global_ptr,
            ks_rsrc=ks_rsrc,
            vs_rsrc=vs_rsrc,
            logits_lds_i32=logits_lds_i32,
            logits_lds_i64=logits_lds_i64,
            softmax_lds_f32=softmax_lds_f32,
            scale_lds_f32=scale_lds_f32,
            stride_ks_block=stride_ks_block,
            stride_ks_head=stride_ks_head,
            softmax_scale_base=_softmax_scale_const,
            softmax_q_scale=_softmax_q_scale,
            k_scale_val=k_scale_val,
            scale=_scale,
            v_scale_val=v_scale_val,
            warp_id=warp_id,
            lane16id=lane16id,
            rowid=rowid,
            k_tok_thread_base=_k_tok_thread_base,
            v_tok_thread_off=_v_tok_thread_off,
            vhead_elem_dw=_vhead_elem_dw,
            kv_tok_thread_base=_kv_tok_thread_base,
            prob_wr_thread_base=_prob_wr_thread_base,
            pv_prob_read_base=_pv_prob_read_base,
            sm_max_off=_sm_max_off,
            sm_sum_off=_sm_sum_off,
            sm_rd_max_offs=_sm_rd_max_offs,
            sm_rd_sum_offs=_sm_rd_sum_offs,
            sm_vmax_wr_off=_sm_vmax_wr_off,
            sm_vmax_rd_offs=_sm_vmax_rd_offs,
            c_w=c_w,
            neg_inf=NEG_INF,
            zero_f=ZERO_F,
        )

        def _process_block_split(
            rmax,
            rsum,
            outs,
            k_ops,
            preloaded_v_and_scales,
            q_frags,
            causal_bound,
            query_scale_lane,
            seq_start,
            partition_start,
        ):
            """Process one 256-token tile inside the selected physical block."""
            v0_ops = preloaded_v_and_scales
            d_out_0 = _qk_and_intra_softmax(
                k_ops,
                partition_start,
                q_frags,
                causal_bound,
                query_scale_lane=query_scale_lane,
                seq_start=seq_start,
            )
            gpu.barrier()
            rmax, rsum, outs, vc0 = _cross_warp_softmax_and_prob_pack(d_out_0, rmax, rsum, outs)
            gpu.barrier()
            outs = _pv_mfma(v0_ops, outs, vc0)
            return rmax, rsum, outs

        def _f32_bits_as_i32(value):
            return fx.Float32(value).ir_value().bitcast(fx.Int32.ir_type)

        def _store_partition_results(eqgs_lane, running_sum, running_max, outelems_norm):
            for vhe in range_constexpr(VHELOOP):
                hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * 4
                to_off = (
                    batch_idx * stride_to_seq
                    + kv_h * stride_to_head
                    + partition_idx * stride_to_part
                    + eqgs_lane * stride_to_group
                    + hs_base
                )
                out_i32 = fx.Vector(outelems_norm[vhe]).to(fx.BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(out_i32, to_rsrc, to_off * 2, offset_is_bytes=True)

            es_off = batch_idx * stride_es_seq + kv_h * stride_es_head + partition_idx * stride_es_part + eqgs_lane
            es_i32 = _f32_bits_as_i32(running_sum)
            ml_i32 = _f32_bits_as_i32(running_max)
            buffer_ops.buffer_store(es_i32, es_rsrc, es_off * 4, offset_is_bytes=True)
            buffer_ops.buffer_store(ml_i32, ml_rsrc, es_off * 4, offset_is_bytes=True)

        def _store_group_results(qi_val, qhi_pos, running_sum, running_max, outs):
            outelems_norm = _normalize_pa_output(running_sum, outs, ZERO_F)
            eqgs_lane = qi_val * fx.Int32(query_group_size) + qhi_pos
            _store_partition_results(eqgs_lane, running_sum, running_max, outelems_norm)

        def _store_fused_group_results(qi_val, qhi_pos, running_sum, outs):
            outelems_norm = _normalize_pa_output(running_sum, outs, ZERO_F)
            for vhe in range_constexpr(VHELOOP):
                hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * 4
                out_off = (
                    batch_idx * stride_out_bs
                    + qi_val * stride_out_len
                    + kv_h * stride_out_kv_head
                    + qhi_pos * stride_out_group_size
                    + hs_base
                )
                out_i32 = fx.Vector(outelems_norm[vhe]).to(fx.BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(out_i32, out_rsrc, out_off * 2, offset_is_bytes=True)

        def _write_empty_partition():
            zero_output = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(VHELOOP)]
            qi_val, qhi_pos, _, _ = _compute_sw_mtp_group_state(
                lane16id,
                local_qhead_idx,
                mtp_group_idx=mtp_group_from_grid,
                mtp_subgroup_idx=0,
                query_length=query_length,
                query_group_size=query_group_size,
            )
            eqgs_lane = qi_val * fx.Int32(query_group_size) + qhi_pos
            _store_partition_results(eqgs_lane, ZERO_F, NEG_INF, zero_output)

        def _run_valid_partition():
            def _get_tile_metadata(tile_partition_idx_value, tile_valid):
                if const_expr(tile_valid):
                    safe_tile_partition_idx = tile_partition_idx_value
                    tile_context_len = context_len
                else:
                    safe_tile_partition_idx = arith.select(tile_valid, tile_partition_idx_value, 0)
                    tile_context_len = arith.select(tile_valid, context_len, 0)
                tile_seq_partition_idx = safe_tile_partition_idx >> fx.Int32(2)
                tile_block_split_idx = safe_tile_partition_idx & fx.Int32(TILES_PER_BLOCK - 1)
                tile_token_offset_local = tile_block_split_idx * c_cps
                tile_kv_seq_start = tile_seq_partition_idx * c_bs + tile_token_offset_local
                tile_bt_off = batch_idx * stride_bt_seq + tile_seq_partition_idx
                tile_phys_block = _global_load_i32(bt_global_ptr, tile_bt_off)
                return tile_token_offset_local, tile_kv_seq_start, tile_context_len, tile_phys_block

            def _load_tile(tile_metadata, tile_scale_scalars):
                tile_token_offset_local, tile_kv_seq_start, tile_context_len, tile_phys_block = tile_metadata
                tile_k_base = _compute_block_base_dw_i64(tile_phys_block, stride_k_block, _k_head_off)

                tile_k_flat = _load_k_flat(
                    k_global_ptr,
                    tile_k_base,
                    tile_token_offset_local,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                )

                tile_v_base = _compute_block_base_dw_i64(tile_phys_block, stride_v_block, _v_head_off)
                tile_v_ops = _load_v_and_scales(
                    tile_v_base,
                    tile_token_offset_local,
                    preloaded_scale_scalars=tile_scale_scalars,
                )
                _store_vmax_warp(tile_kv_seq_start, seq_end=tile_context_len)
                return (
                    _unflatten_k(tile_k_flat),
                    tile_v_ops,
                    tile_kv_seq_start,
                    tile_context_len,
                )

            mtp_prefetches = _prefetch_sw_mtp_group_queries(
                q_rsrc,
                batch_idx,
                kv_h,
                stride_q_seq,
                stride_q_head,
                lane16id,
                local_qhead_idx,
                mtp_group_idx=mtp_group_from_grid,
                mtp_subgroup_count=1,
                query_length=query_length,
                query_group_size=query_group_size,
                query_load_is_bf16=query_load_is_bf16,
            )
            if const_expr(fuse_partitions):
                tile_valid = fx.Int32(0) < visible_tile_count
                prefetched_tile_metadata = _get_tile_metadata(tail_start_tile, tile_valid)
            else:
                prefetched_tile_metadata = _get_tile_metadata(tile_partition_idx_raw, True)
            prefetched_tile_scale_scalars = _load_kv_scale_scalars(
                prefetched_tile_metadata[0],
                prefetched_tile_metadata[3],
            )
            qi_val, qhi_pos, q_frags, query_scale_lane = _finish_sw_mtp_subgroup_q_fragments(
                logits_lds_i32,
                logits_lds_i64,
                softmax_lds_f32,
                mtp_prefetches,
                lane16id,
                rowid,
                local_qhead_idx,
                mtp_subgroup_idx=0,
            )
            if const_expr(fuse_partitions):
                running_max = NEG_INF
                running_sum = ZERO_F
                outs = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(VHELOOP)]
                (
                    tile_k_ops,
                    tile_v_and_scales,
                    tile_kv_seq_start,
                    tile_context_len,
                ) = _load_tile(prefetched_tile_metadata, prefetched_tile_scale_scalars)
                causal_bound = tile_context_len + fx.Int32(1 - query_length) + qi_val
                seq_start = tile_context_len - fx.Int32(query_length + sliding_window) + qi_val
                running_max, running_sum, outs = _process_block_split(
                    running_max,
                    running_sum,
                    outs,
                    tile_k_ops,
                    tile_v_and_scales,
                    q_frags,
                    causal_bound,
                    query_scale_lane,
                    seq_start,
                    tile_kv_seq_start,
                )
                _store_fused_group_results(qi_val, qhi_pos, running_sum, outs)
            else:
                (
                    k_ops,
                    preloaded_v_and_scales,
                    tile_kv_seq_start,
                    _,
                ) = _load_tile(prefetched_tile_metadata, prefetched_tile_scale_scalars)
                causal_bound = context_len + fx.Int32(1 - query_length) + qi_val
                seq_start = context_len - fx.Int32(query_length + sliding_window) + qi_val
                outs = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(VHELOOP)]
                running_max, running_sum, outs = _process_block_split(
                    NEG_INF,
                    ZERO_F,
                    outs,
                    k_ops,
                    preloaded_v_and_scales,
                    q_frags,
                    causal_bound,
                    query_scale_lane,
                    seq_start,
                    tile_kv_seq_start,
                )
                _store_group_results(qi_val, qhi_pos, running_sum, running_max, outs)

        if const_expr(fuse_partitions):
            _run_valid_partition()
        else:
            if _is_valid:
                _run_valid_partition()
            else:
                _write_empty_partition()

    @flyc.jit
    def launch_pa_decode_sw(
        es: fx.Tensor,
        ml: fx.Tensor,
        to: fx.Tensor,
        out: fx.Tensor,
        q: fx.Tensor,
        kc: fx.Tensor,
        vc: fx.Tensor,
        bt: fx.Tensor,
        cl: fx.Tensor,
        ks: fx.Tensor,
        vs: fx.Tensor,
        s_q_seq: Int32,
        s_q_head: Int32,
        s_k_block: Int32,
        s_k_head: Int32,
        s_v_block: Int32,
        s_v_head: Int32,
        s_es_seq: Int32,
        s_es_head: Int32,
        s_es_part: Int32,
        s_to_seq: Int32,
        s_to_head: Int32,
        s_to_part: Int32,
        s_to_group: Int32,
        s_out_bs: Int32,
        s_out_len: Int32,
        s_out_kv_head: Int32,
        s_out_group_size: Int32,
        s_bt_seq: Int32,
        s_ks_block: Int32,
        s_ks_head: Int32,
        gx: Int32,
        gy: Int32,
        gz: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_sw_kernel(
            es,
            ml,
            to,
            out,
            q,
            kc,
            vc,
            bt,
            cl,
            ks,
            vs,
            s_q_seq,
            s_q_head,
            s_k_block,
            s_k_head,
            s_v_block,
            s_v_head,
            s_es_seq,
            s_es_head,
            s_es_part,
            s_to_seq,
            s_to_head,
            s_to_part,
            s_to_group,
            s_out_bs,
            s_out_len,
            s_out_kv_head,
            s_out_group_size,
            s_bt_seq,
            s_ks_block,
            s_ks_head,
        ).launch(grid=(gx, gy, gz), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        "launch": launch_pa_decode_sw,
        "kernel": pa_decode_sw_kernel,
        "allocator": allocator,
    }
