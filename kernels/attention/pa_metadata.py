# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""FlyDSL implementation of aiter's ``get_pa_metadata_v1`` worklist scheduler.

This replaces the ``aiter.ops.attention.get_pa_metadata_v1`` C++/CUDA dependency
(``module_pa_metadata.so``) with a FlyDSL device kernel, so paged-attention
decode (``kernels/pa_decode_fp8.py``) can build its CU worklist without aiter.

Scope — PA-decode-specialized port of
``aiter/csrc/kernels/mla/metadata/v1_2_pa_device.cuh::kn_get_pa_metadata_v1_2``.
The following invariants always hold for the PA decode use and are baked in:

* ``kQoSplits == False`` — ``packed_qo_len = query_length * gqa`` is small
  (<= ~32) so it never exceeds ``kPackedQoLenPerWg=128`` ⇒ ``num_qo_tiles = 1``,
  ``qo_tile_size = query_length``.
* uniform qo length across batches (``uni_seqlen_qo = query_length``).
* causal, non-sparse (``topk = -1``), ``qk_batch_ratio = 1``,
  ``num_splits = num_cu`` (``max_split_per_batch = -1``).

All six outputs are produced as a faithful drop-in for the C++ kernel:
``work_metadata_ptrs``, ``work_indptr``, ``work_info`` (8 fields),
``reduce_indptr``, ``reduce_final_map`` and ``reduce_partial_map`` — each
verified element-for-element against aiter. The caller consumes them directly
(no post-hoc expansion).

work_info layout (8 x int32 per work), matching ``PaWorkInfo``:
  [0] batch_idx  [1] partial_qo_loc(-1 if no split)  [2] qo_start  [3] qo_end
  [4] kv_start   [5] kv_end                          [6] kv_offset(=0)
  [7] q_head_range = (qhead_end << 16) | (qhead_start & 0xFFFF)

The kernel is launched single-thread (grid=block=(1,1,1)); the scheduler is a
serial algorithm (warp reductions / lane-parallel fills in the original collapse
to serial loops). It runs once per shape and the result is cached, so single-
thread is the correct, simplest model.
"""

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_common import _compute_block_base_dw_i64, _prefetch_q_chunks
from kernels.common import dpp_utils
from kernels.common.kernels_common import get_warp_size
from kernels.common.utils import (
    exp2_f32_fast,
    global_load_i64x2,
    global_ptr_from_addr,
    is_pow2,
    rcp_f32,
    udiv_const,
    unflatten_k,
    urem_const,
)

_WORK_INFO_FIELDS = 8


def get_pa_metadata_info_v1(batch_size: int, num_head_k: int = 1, num_cu: int = None):
    """Buffer sizes/dtypes, matching ``aiter.get_pa_metadata_info_v1``.

    Returns (shape, dtype) tuples for:
      work_metadata_ptrs, work_indptr, work_info_set,
      reduce_indptr, reduce_final_map, reduce_partial_map.

    ``num_cu`` overrides the worklist bin count (default = device CU count);
    pass a multiple of the CU count to oversubscribe the persistent grid.
    """
    if num_cu is None:
        gpu = torch.cuda.current_device()
        num_cu = torch.cuda.get_device_properties(gpu).multi_processor_count
    cu_num = num_cu

    tile_cnt = batch_size
    max_work = (tile_cnt + cu_num - 1) * num_head_k
    max_split_tiles = min(batch_size + cu_num - 1, (cu_num - 1) * 2)

    return (
        ((2,), torch.uint64),  # work_metadata_ptrs
        ((cu_num + 1,), torch.int32),  # work_indptr
        ((max_work, _WORK_INFO_FIELDS), torch.int32),  # work_info_set
        ((tile_cnt + 1,), torch.int32),  # reduce_indptr
        ((tile_cnt, 2), torch.int32),  # reduce_final_map
        ((max_split_tiles,), torch.int32),  # reduce_partial_map
    )


# ── PA decode geometry + helpers (shared math lives in kernels/utils.py) ──
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


# ── LDS (shared memory) raw-pointer access helpers ───────────────────
# The shared-storage API hands out `!fly.memref` views that only accept scalar
# fx.memref_load/store. This kernel needs vectorized LDS traffic, so we
# reinterpret each region's base pointer as an LDS (address space 3) llvm.ptr and
# issue vector llvm.load/store at explicit byte offsets. `base_i64` is the raw
# integer address of a shared-storage region (via fx.ptrtoint); `byte_off` is a
# byte offset into that region.
def _lds_ptr(base_i64, byte_off):
    return buffer_ops.create_llvm_ptr(fx.Int64(base_i64) + fx.Int64(byte_off), address_space=3)


def _lds_store_vec(base_i64, byte_off, vec_value, align):
    llvm.StoreOp(vec_value, _lds_ptr(base_i64, byte_off), alignment=align)


def _lds_load_vec(base_i64, byte_off, vec_ir_type, align):
    return fx.Vector(llvm.LoadOp(vec_ir_type, _lds_ptr(base_i64, byte_off), alignment=align).result)


def _load_k_flat(
    k_global_ptr,
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
            k2 = global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
            k2_words = fx.Vector(k2)
            k_flat.append(k2_words[0])
            k_flat.append(k2_words[1])

    return k_flat


def _build_pa_thread_invariants(
    warp_id,
    lane16id,
    rowid,
    *,
    trans_v,
    per_token_kv,
    qkhe_loop: int = 2,
    vhe_loop: int = 2,
):
    c_tokens_per_warp = fx.Int32(TOKENS_PER_WARP)
    c_mfma_n = fx.Int32(MFMA_N)
    k_tok_thread_base = warp_id * c_tokens_per_warp + lane16id
    c_tok_stride_dw = fx.Int32(FP8_ELEMS_16B // 4)
    c_he_stride_dw = fx.Int32(KV_BLOCK_SIZE * FP8_ELEMS_16B // 4)
    k_he_off_dw = [rowid * c_he_stride_dw + fx.Int32(qkhe * 4) * c_he_stride_dw for qkhe in range(qkhe_loop)]

    vhead_elems = [fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * c_mfma_n + lane16id for vhe in range(vhe_loop)]
    v_tok_thread_off = [fx.Int32(vt * TOKENS_PER_WARP) + rowid * c_mfma_n for vt in range(VTLOOP)]
    if const_expr(trans_v):
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(FP8_ELEMS_16B // 4) for vhe in range(vhe_loop)]
    else:
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(KV_BLOCK_SIZE // 4) for vhe in range(vhe_loop)]

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

    # Softmax-LDS element offsets (f32 indices); callers convert to byte offsets.
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
    head_size: int,
    qkhe_loop: int,
    q_lanes_per_head: int,
):
    # LDS Q layout (compact, per-qhead contiguous):
    #   Q[head=h][hd=d]  at byte offset  h * HEAD_SIZE + d   (FP8 after conversion)
    # Total Q footprint = 16 qheads * HEAD_SIZE bytes, aliased with the later P
    # writes via the logits/Q region base (addressed as i32 and i64).  For
    # HEAD_SIZE=64, only the first 8 lanes write Q for each qhead.
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
        _lds_store_vec(
            softmax_base,
            local_qhead_idx * 4,
            fx.Vector.from_elements([query_scale_lane], dtype=fx.Float32),
            align=4,
        )

    v01 = fx.Vector.from_elements([q_w0, q_w1], dtype=fx.Int32)
    # `lds_q_base` is already a byte offset into the logits/Q region.
    if const_expr(q_lanes_per_head < MFMA_N):
        if lane16id < fx.Int32(q_lanes_per_head):
            _lds_store_vec(logits_base, lds_q_base, v01, align=8)
    else:
        _lds_store_vec(logits_base, lds_q_base, v01, align=8)

    q_frags = []
    gpu.barrier()
    query_scale_lane = _lds_load_vec(softmax_base, lane16id * 4, T.vec(1, fx.Float32.ir_type), align=4)[0].ir_value()
    for qkhe in range_constexpr(qkhe_loop):
        for qkr in range_constexpr(2):
            # See layout comment above. Byte offset:
            #   lane16id * HEAD_SIZE + qkhe*64 + rowid*16 + qkr*8
            lds_rd_byte = lane16id * c_head_size + fx.Int32(qkhe << 6) + (rowid << fx.Int32(4)) + fx.Int32(qkr << 3)
            q_v1 = _lds_load_vec(logits_base, lds_rd_byte, T.vec(1, T.i64), align=8)
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
    logits_base,
    softmax_base,
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
        logits_base,
        softmax_base,
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
    trans_v,
    per_token_q,
    per_token_kv,
    needs_mask,
    query_length,
    kv_h,
    v_global_ptr,
    ks_rsrc,
    vs_rsrc,
    logits_base,
    softmax_base,
    scale_base,
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
    cache_scale_vecs=False,
    head_size: int = 128,
    qkhe_loop: int = 2,
    vhe_loop: int = 2,
):
    apply_causal_mask = needs_mask or query_length > 1
    pv_prob_i64_bytes = []
    for vt in range_constexpr(VTLOOP):
        for j in range_constexpr(2):
            p_byte = (
                arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                + pv_prob_read_base
                + arith.constant(j * 8, type=T.i32)
            )
            pv_prob_i64_bytes.append(p_byte)

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
        phys_block,
        preloaded_scale_scalars=None,
    ):
        if const_expr(per_token_kv):
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            if const_expr(preloaded_scale_scalars is None):
                preloaded_scale_scalars = _load_kv_scale_scalars(tile_token_offset_i32, phys_block)
            k_scale_scalar, v_scale_scalar = preloaded_scale_scalars
            _lds_store_vec(
                scale_base,
                scale_stage_token * 4,
                fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32),
                align=4,
            )
            _lds_store_vec(
                scale_base,
                (fx.Int32(LDS_SCALE_V_OFFSET) + scale_stage_token) * 4,
                fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32),
                align=4,
            )
            rocdl.sched_barrier(0)

        v_results = []
        for vt in range_constexpr(VTLOOP):
            vhe_data = []
            for vhe in range_constexpr(vhe_loop):
                v_token_in_block = tile_token_offset_i32 + v_tok_thread_off[vt]
                if const_expr(trans_v):
                    vt_group = v_token_in_block >> fx.Int32(4)
                    va_dw_delta = (
                        vt_group * arith.constant(head_size * FP8_ELEMS_16B // 4, type=T.i32) + vhead_elem_dw[vhe]
                    )
                else:
                    va_dw_delta = vhead_elem_dw[vhe] + (v_token_in_block >> fx.Int32(2))
                va_byte = (v_block_base_dw + fx.Int64(va_dw_delta)) * fx.Int64(4)
                v_i64x2 = global_load_i64x2(v_global_ptr, va_byte)
                vhe_data.append(v_i64x2)
            v_results.append(vhe_data)

        if const_expr(per_token_kv):
            gpu.barrier()
            if const_expr(cache_scale_vecs):
                k_scale_vecs = []
                v_scale_vecs = []
                for td in range_constexpr(TLOOP):
                    scale_row_base = kv_tok_thread_base + fx.Int32(td * MFMA_N)
                    k_scale_vecs.append(_lds_load_vec(scale_base, scale_row_base * 4, T.f32x4, align=16))
                    v_scale_vecs.append(
                        _lds_load_vec(
                            scale_base,
                            (fx.Int32(LDS_SCALE_V_OFFSET) + scale_row_base) * 4,
                            T.f32x4,
                            align=16,
                        )
                    )
                return v_results, k_scale_vecs, v_scale_vecs

        return v_results

    def _scale_row_base(td: int):
        return kv_tok_thread_base + fx.Int32(td * MFMA_N)

    def _load_k_scale_vec(td: int):
        return _lds_load_vec(scale_base, _scale_row_base(td) * 4, T.f32x4, align=16)

    def _load_v_scale_vec(td: int):
        return _lds_load_vec(scale_base, (fx.Int32(LDS_SCALE_V_OFFSET) + _scale_row_base(td)) * 4, T.f32x4, align=16)

    def _get_k_scale_vec(td: int, k_scale_vecs=None):
        if const_expr(cache_scale_vecs):
            return k_scale_vecs[td]
        return _load_k_scale_vec(td)

    def _get_v_scale_vec(td: int, v_scale_vecs=None):
        if const_expr(cache_scale_vecs):
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
            _lds_store_vec(
                softmax_base,
                sm_vmax_wr_off * 4,
                fx.Vector.from_elements([v_max_warp], dtype=fx.Float32),
                align=4,
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
            if const_expr(cache_scale_vecs and per_token_kv):
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
                if const_expr(cache_scale_vecs and per_token_kv):
                    k_scale_vec = _get_k_scale_vec(td, k_scale_vecs)
                else:
                    k_scale_vec = _get_k_scale_vec(td)
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
        _lds_store_vec(
            softmax_base,
            sm_max_off * 4,
            fx.Vector.from_elements([qk_max], dtype=fx.Float32),
            align=4,
        )

        if const_expr(cache_scale_vecs and per_token_kv):
            return d_out, v_scale_vecs
        return d_out

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs, v_scale_vecs):
        partition_max = neg_inf
        partition_sum = zero_f
        max_vec = _lds_load_vec(softmax_base, sm_rd_max_offs[0] * 4, T.f32x4, align=16)
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
        _lds_store_vec(
            softmax_base,
            sm_sum_off * 4,
            fx.Vector.from_elements([local_exp_sum], dtype=fx.Float32),
            align=4,
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
        sum_vec = _lds_load_vec(softmax_base, sm_rd_sum_offs[0] * 4, T.f32x4, align=16)
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
            vmax_vec = _lds_load_vec(softmax_base, sm_vmax_rd_offs[0] * 4, T.f32x4, align=16)
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
            pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
            _lds_store_vec(logits_base, byte_base, pk_vec, align=4)
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
                p_i64_byte = pv_prob_i64_bytes[vt * 2 + j]
                p_i64_all.append(_lds_load_vec(logits_base, p_i64_byte, T.vec(1, T.i64), align=8)[0])

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
        _load_kv_scale_scalars,
        _load_v_and_scales,
        _store_vmax_warp,
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
    )


_PA_DECODE_PS_SMALL_BLOCK_SIZES = (16, 64)


@flyc.jit
def _pa_small_block_load_k_flat(
    k_global_ptr,
    kv_h_i32,
    stride_k_block_i32,
    stride_k_head_i32,
    lane16id_i32,
    rowid_i32,
    *,
    block_size: int,
    phys_blocks,
    qkhe_loop: int = 2,
):
    """Load K data for one warp's 64-token slice of a 256-token partition.

    Returns ``k_flat`` (a list of ``TLOOP * qkhe_loop * 2`` i64 scalars) compatible
    with ``unflatten_k`` and downstream MFMA invocations.
    """
    c_he_stride_dw = fx.Int32(block_size * FP8_ELEMS_16B // 4)
    c_tok_stride_dw = fx.Int32(FP8_ELEMS_16B // 4)
    k_he_off_dw = [rowid_i32 * c_he_stride_dw + fx.Int32(qkhe * 4) * c_he_stride_dw for qkhe in range(qkhe_loop)]
    k_head_off = kv_h_i32 * stride_k_head_i32

    k_flat = []
    if const_expr(block_size == 64):
        # Each warp owns exactly one physical block (64 tokens).
        phys_block = phys_blocks
        k_block_base_dw = _compute_block_base_dw_i64(phys_block, stride_k_block_i32, k_head_off)
        for td in range_constexpr(TLOOP):
            within_block_token = fx.Int32(td * MFMA_N) + lane16id_i32
            kbo_dw = within_block_token * c_tok_stride_dw
            for qkhe in range_constexpr(qkhe_loop):
                ka_dw = k_block_base_dw + fx.Int64(kbo_dw + k_he_off_dw[qkhe])
                k2 = global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
                k2_words = fx.Vector(k2)
                k_flat.append(k2_words[0])
                k_flat.append(k2_words[1])
    else:
        # block_size == 16: each warp spans 4 blocks (one MFMA tile per block).
        within_block_token = lane16id_i32
        kbo_dw = within_block_token * c_tok_stride_dw
        for td in range_constexpr(TLOOP):
            phys_block = phys_blocks[td]
            k_block_base_dw = _compute_block_base_dw_i64(phys_block, stride_k_block_i32, k_head_off)
            for qkhe in range_constexpr(qkhe_loop):
                ka_dw = k_block_base_dw + fx.Int64(kbo_dw + k_he_off_dw[qkhe])
                k2 = global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
                rocdl.sched_barrier(rocdl.mask_vmem_rd)
                k2_words = fx.Vector(k2)
                k_flat.append(k2_words[0])
                k_flat.append(k2_words[1])
    return k_flat


@flyc.jit
def _pa_small_block_load_v_trans(
    v_global_ptr,
    kv_h_i32,
    stride_v_block_i32,
    stride_v_head_i32,
    warp_id_i32,
    lane16id_i32,
    rowid_i32,
    v_phys_blocks,
    *,
    block_size: int,
    head_size: int = 128,
    vhe_loop: int = 2,
):
    """Load V tiles for one CTA's 256-token partition (``trans_v=True``).

    Returns ``v_results[vt][vhe]`` (i64x2) indexed exactly as the reference
    ``_load_v_and_scales`` so it can be passed as ``preloaded_v_and_scales``.
    """
    v_head_off = kv_h_i32 * stride_v_head_i32
    vhead_elems = [
        fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id_i32 * fx.Int32(MFMA_N) + lane16id_i32 for vhe in range(vhe_loop)
    ]
    vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(FP8_ELEMS_16B // 4) for vhe in range(vhe_loop)]
    c_subblock_dw = fx.Int32(head_size * FP8_ELEMS_16B // 4)

    v_results = []
    for vt in range_constexpr(VTLOOP):
        phys_block = v_phys_blocks[vt]
        if const_expr(block_size == 64):
            # vt selects the physical block (4 blocks per partition); rowid
            # selects the 16-token sub-block within that physical block.
            sub_block_idx = rowid_i32
        else:
            # block_size == 16: (vt * 4 + rowid) selects the block; only one
            # 16-token sub-block per physical block, so sub_block_idx == 0.
            sub_block_idx = fx.Int32(0)
        v_block_base_dw = _compute_block_base_dw_i64(phys_block, stride_v_block_i32, v_head_off)
        vhe_data = []
        for vhe in range_constexpr(vhe_loop):
            va_dw_delta = sub_block_idx * c_subblock_dw + vhead_elem_dw[vhe]
            va_byte = (v_block_base_dw + fx.Int64(va_dw_delta)) * fx.Int64(4)
            v_i64x2 = global_load_i64x2(v_global_ptr, va_byte)
            vhe_data.append(v_i64x2)
        v_results.append(vhe_data)
    return v_results


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
    """Compile the FlyDSL worklist scheduler for a fixed device/shape config.

    ``num_batches`` stays a runtime kernel argument; everything else is baked.

    Launched as a single warp (``grid=block=(warp_size,1,1)``), matching aiter's
    ``<<<grid, warp_size, ...>>>``: Phase-1 (per-batch block counts + sum) is
    warp-parallel (lane-strided + warp reduce), while the serial CU x batch
    scheduler runs uniformly on all lanes — exactly as the original, whose
    work_indptr / work_info writes are not lane-guarded (benign same-value
    races). The original's lane-divided / lane-0-guarded work is only for the
    reduce_* maps, which the caller recomputes and we therefore omit.
    """
    assert is_pow2(kv_granularity), "kv_granularity must be power of 2"
    assert num_cu % num_heads_k == 0, "num_cu must be divisible by num_heads_k"
    num_splits_per_khead = num_cu // num_heads_k
    # warp-reduce shuffle offsets: warp_size/2, ..., 1
    _shuffle_offsets = []
    _o = warp_size // 2
    while _o >= 1:
        _shuffle_offsets.append(_o)
        _o //= 2

    @flyc.kernel(known_block_size=(warp_size, 1, 1))
    def pa_metadata_v1_kernel(
        seqlens_qo_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (cumulative qo seqlens)
        pages_kv_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (cumulative pages)
        context_lens_ptr: fx.Tensor,  # [num_batches] i32
        work_indptr_ptr: fx.Tensor,  # [num_cu + 1] i32   (output)
        work_info_ptr: fx.Tensor,  # [max_work * 8] i32 (output, flattened)
        reduce_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (output)
        reduce_final_map_ptr: fx.Tensor,  # [num_batches * 2] i32 (output, flattened)
        reduce_partial_map_ptr: fx.Tensor,  # [max_split_tiles] i32 (output)
        num_batches: Int32,
    ):
        i32 = T.i32
        sq_rsrc = buffer_ops.create_buffer_resource(seqlens_qo_indptr_ptr, max_size=True)
        ctx_rsrc = buffer_ops.create_buffer_resource(context_lens_ptr, max_size=True)
        wi_rsrc = buffer_ops.create_buffer_resource(work_indptr_ptr, max_size=True)
        winfo_rsrc = buffer_ops.create_buffer_resource(work_info_ptr, max_size=True)
        rip_rsrc = buffer_ops.create_buffer_resource(reduce_indptr_ptr, max_size=True)
        rfm_rsrc = buffer_ops.create_buffer_resource(reduce_final_map_ptr, max_size=True)
        rpm_rsrc = buffer_ops.create_buffer_resource(reduce_partial_map_ptr, max_size=True)
        # pages_kv_indptr_ptr is accepted for signature compat but unused: the
        # work unit is a `partition_size`-token partition, so both the load
        # balance and the kv ranges are counted as ceil(context_len/kv_gran)
        # partitions (kv_granularity == partition_size). This matches the
        # original kernel's kLdsBatchInfo=true path where curr_kv_pages is the
        # per-batch num_blocks, not the pages_kv_indptr delta.

        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        c_qlen = fx.Int32(query_length)
        c_nb = num_batches  # Int32 runtime
        c_nspk = fx.Int32(num_splits_per_khead)
        c_numcu = fx.Int32(num_cu)
        c_ws = fx.Int32(warp_size)
        c_kvg = fx.Int32(kv_granularity)
        lane = fx.Int32(gpu.thread_id("x"))

        def _load(rsrc, off):
            return fx.Int32(buffer_ops.buffer_load(rsrc, fx.Int32(off).ir_value(), vec_width=1, dtype=i32))

        def _num_part(batch_idx):
            # number of partition_size-token partitions for this batch =
            # ceil(context_len[batch_idx] / kv_granularity)
            ctxv = _load(ctx_rsrc, batch_idx)
            return fx.Int32(arith.ceildivui(ctxv.ir_value(), c_kvg.ir_value()))

        def _store(rsrc, off, val):
            # NOTE: no masked stores — masked buffer_store sets OOB offset
            # (0x7FFFFFFF) expecting HW bounds-check to drop it, but our
            # max_size resources disable bounds-checking, so a masked store
            # faults. All stores here are unconditional + overwrite-safe.
            buffer_ops.buffer_store(fx.Int32(val).ir_value(), rsrc, fx.Int32(off).ir_value())

        def _sel(cond_b, a, b):
            return fx.Int32(arith.select(cond_b.ir_value(), fx.Int32(a).ir_value(), fx.Int32(b).ir_value()))

        # work_indptr[0] = 0 ; reduce_indptr[0] = 0
        _store(wi_rsrc, 0, 0)
        _store(rip_rsrc, 0, 0)

        # ---- Phase 1: sum_blocks = Sum_b ceil(context_lens[b] / kv_granularity) ----
        # (causal + uniform + tiny qo  =>  effective_kv == context_lens[b])
        # warp-parallel: each lane sums batches {lane, lane+ws, lane+2ws, ...},
        # then a warp reduce-add gives the total in every lane (matches aiter).
        b = lane
        sum_blocks = c0
        while b < c_nb:
            nblk = _num_part(b)
            b = b + c_ws
            sum_blocks = sum_blocks + nblk
        for sh in _shuffle_offsets:
            sum_blocks = sum_blocks + sum_blocks.shuffle_xor(arith.constant(sh, type=i32), c_ws.ir_value())

        average = fx.Int32(arith.divui(sum_blocks.ir_value(), c_nspk.ir_value()))
        reminder = fx.Int32(arith.remui(sum_blocks.ir_value(), c_nspk.ir_value()))

        def _remain_for_cid(cid_val):
            # remain = average + (1 if (cid % num_splits_per_khead) < reminder else 0)
            mod = fx.Int32(arith.remui(cid_val.ir_value(), c_nspk.ir_value()))
            return average + _sel(mod < reminder, 1, 0)

        # ---- Phase 2: per khead, flattened CU x batch scheduler ----
        # cid and num_works persist across kheads; the rest reset per khead.
        cid = c0
        num_works = c0

        for khead in range_constexpr(num_heads_k):
            qh_start = khead * gqa
            qh_end = (khead + 1) * gqa
            qhr_const = (qh_end << 16) | (qh_start & 0xFFFF)  # python int constant

            kvend0 = _num_part(c0)  # partitions in batch 0 (cumulative kv end)
            remain0 = _remain_for_cid(cid)

            # State (11 i32), loop-carried through the scf.while emitted from the
            # Python `while` below:
            #  0 cid, 1 batch, 2 kvblk, 3 nsplit, 4 num_works, 5 pidx,
            #  6 kvbeg, 7 kvend, 8 remain, 9 last_reduce_indptr, 10 global_reduce_tile_idx
            # cid + num_works persist across kheads; lri + grt reset per khead.
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

                # qo_start/qo_end from seqlens_qo_indptr (the QoState array path,
                # matching C++). batch_ < num_batches in the loop, so batch_+1 <=
                # num_batches indexes the valid last element (no OOB). For uniform
                # qo (sq = arange*query_length) this equals query_length*batch.
                qo_start = _load(sq_rsrc, batch_)
                qo_end = _load(sq_rsrc, batch_ + c1)
                kv_start = kvbeg_ + kvblk_  # same for both branches

                # ---- finish branch (CU completes this batch) ----
                f_kv_end = kvend_  # min(kv_start + remain_kv, kvend_) == kvend_
                nsplit_pos = nsplit_ > c0
                f_ploc = _sel(nsplit_pos, pidx_, -1)
                f_pidx2 = _sel(nsplit_pos, pidx_ + c_qlen, pidx_)
                f_nworks2 = nworks_ + c1
                f_remain2 = remain_ - remain_kv
                f_batch2 = batch_ + c1
                # next batch kv window (in partition units). max_size buffer rsrc
                # disables HW bounds checking, so clamp the index before loading
                # context_lens (f_batch2 can equal num_batches → OOB on last batch;
                # the result is unused after the loop exits anyway).
                nb_in_range = f_batch2 < c_nb
                safe_idx = _sel(nb_in_range, f_batch2, 0)
                f_new_pages = _sel(nb_in_range, _num_part(safe_idx), 0)
                f_kvbeg2 = kvend_
                f_kvend2 = kvend_ + f_new_pages

                # ---- split branch (CU does a partial; close cid) ----
                s_emit = remain_ > c0
                s_kv_end_raw = kv_start + remain_
                s_kv_end = _sel(s_kv_end_raw < kvend_, s_kv_end_raw, kvend_)
                s_nworks2 = _sel(s_emit, nworks_ + c1, nworks_)
                s_pidx2 = _sel(s_emit, pidx_ + c_qlen, pidx_)
                s_kvblk2 = _sel(s_emit, kvblk_ + remain_, kvblk_)
                s_nsplit2 = _sel(s_emit, nsplit_ + c1, nsplit_)
                s_cid2 = cid_ + c1
                s_remain2 = _remain_for_cid(s_cid2)

                # ---- emit work entry at slot nworks_ ----
                # Overwrite-safe: if this step does not emit, nworks_ is unchanged
                # so the slot is reused by the next emit (or lies beyond valid_work).
                w_ploc = _sel(do_finish, f_ploc, pidx_)
                w_kv_end = _sel(do_finish, f_kv_end, s_kv_end)
                base = nworks_ * fx.Int32(_WORK_INFO_FIELDS)
                _store(winfo_rsrc, base + fx.Int32(0), batch_)
                _store(winfo_rsrc, base + fx.Int32(1), w_ploc)
                _store(winfo_rsrc, base + fx.Int32(2), qo_start)
                _store(winfo_rsrc, base + fx.Int32(3), qo_end)
                _store(winfo_rsrc, base + fx.Int32(4), kv_start)
                _store(winfo_rsrc, base + fx.Int32(5), w_kv_end)
                _store(winfo_rsrc, base + fx.Int32(6), c0)
                _store(winfo_rsrc, base + fx.Int32(7), fx.Int32(qhr_const))

                # ---- reduce maps: only when finishing a SPLIT batch (nsplit>0) ----
                # This batch was split across (nsplit_+1) CUs and this CU finishes
                # it, forming one reduce group. Faithful to the C++ kernel
                # (kQoSplits=False path).
                do_reduce = do_finish & nsplit_pos
                num_splits = nsplit_ + c1
                # reduce_indptr[grt+1] = lri + num_splits ; reduce_final_map[grt] =
                # (qo_start, qo_end). Unconditional + overwrite-safe (same argument
                # as work_indptr: grt only advances on do_reduce, so non-do_reduce
                # writes to grt+1 are overwritten by the next do_reduce or the tail;
                # reduce_final_map[grt*2..] beyond the final grt is never read).
                _store(rip_rsrc, grt_ + c1, lri_ + num_splits)
                _store(rfm_rsrc, grt_ * fx.Int32(2), qo_start)
                _store(rfm_rsrc, grt_ * fx.Int32(2) + c1, qo_end)
                # reduce_partial_map[lri + s] = pidx - (nsplit - s)*qlen, s in [0,num_splits)
                # nested loop runs num_splits times when do_reduce, else 0 times.
                rcount = _sel(do_reduce, num_splits, 0)
                sidx = c0
                while sidx < rcount:
                    val = pidx_ - (nsplit_ - sidx) * c_qlen
                    _store(rpm_rsrc, lri_ + sidx, val)
                    sidx = sidx + c1
                n_lri = lri_ + _sel(do_reduce, num_splits, 0)
                n_grt = grt_ + _sel(do_reduce, 1, 0)

                # ---- new state via select(do_finish, finish, split) ----
                n_cid = _sel(do_finish, cid_, s_cid2)
                n_batch = _sel(do_finish, f_batch2, batch_)
                n_kvblk = _sel(do_finish, 0, s_kvblk2)
                n_nsplit = _sel(do_finish, 0, s_nsplit2)
                n_nworks = _sel(do_finish, f_nworks2, s_nworks2)
                n_pidx = _sel(do_finish, f_pidx2, s_pidx2)
                n_kvbeg = _sel(do_finish, f_kvbeg2, kvbeg_)
                n_kvend = _sel(do_finish, f_kvend2, kvend_)
                n_remain = _sel(do_finish, f_remain2, s_remain2)

                # ---- work_indptr[cid_+1] = n_nworks (unconditional, overwrite-safe) ----
                # In the finish branch cid_ does not advance, so repeated writes to
                # work_indptr[cid_+1] keep updating until the cid closes; the last
                # write (before cid advances in the split branch, or loop exit) holds
                # the correct running num_works for that cid. cid_+1 <= num_cu (loop
                # guard cid_ < num_cu) so the index is always in-bounds.
                _store(wi_rsrc, cid_ + c1, n_nworks)

                # reassign loop-carried state (becomes the scf.while yield)
                cid_ = n_cid
                batch_ = n_batch
                kvblk_ = n_kvblk
                nsplit_ = n_nsplit
                nworks_ = n_nworks
                pidx_ = n_pidx
                kvbeg_ = n_kvbeg
                kvend_ = n_kvend
                remain_ = n_remain
                lri_ = n_lri
                grt_ = n_grt

            cid = cid_
            num_works = nworks_
            last_reduce_indptr = lri_
            global_reduce_tile_idx = grt_

            # ---- post-khead close: advance cid past the last processed cid so
            # the next khead (and the tail) start fresh. The loop already wrote
            # work_indptr[last_cid+1]=num_works on its final iteration, so no
            # store is needed here — only the cid advance.
            in_range = cid < c_numcu
            cid = _sel(in_range, cid + c1, cid)

        # ---- tail: work_indptr[i] = num_works for i in [cid, num_cu] ----
        it_t = cid
        while it_t <= c_numcu:
            _store(wi_rsrc, it_t, num_works)
            it_t = it_t + c1

        # ---- tail: reduce_indptr[i] = last_reduce_indptr for i in [grt, num_batches] ----
        # (reduce_indptr has num_batches+1 entries; uses the final khead's grt/lri,
        # matching the original which resets grt per khead and fills the tail once.)
        c_rip_size = c_nb + c1  # reduce_indptr length = num_batches + 1
        it_r = global_reduce_tile_idx
        while it_r < c_rip_size:
            _store(rip_rsrc, it_r, last_reduce_indptr)
            it_r = it_r + c1

    @flyc.jit
    def launch_pa_metadata_v1(
        seqlens_qo_indptr: fx.Tensor,
        pages_kv_indptr: fx.Tensor,
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
            pages_kv_indptr,
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
    pages_kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    kv_granularity: int = 16,
    block_size: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
    num_cu: int = None,
) -> None:
    """Drop-in replacement for ``aiter.ops.attention.get_pa_metadata_v1``.

    PA-decode-specialized: requires causal, non-sparse, uniform qo. Fills
    ``work_indptr`` and ``work_info`` in-place; ``reduce_*`` are left untouched
    (the caller recomputes them from work_indptr/work_info).

    ``num_cu`` overrides the worklist bin count (default = device CU count);
    pass a multiple of the CU count to oversubscribe the persistent grid.
    """
    assert is_causal, "FlyDSL pa_metadata only supports causal"
    assert topk == -1, "FlyDSL pa_metadata does not support sparse (topk)"
    assert uni_seqlen_qo >= 1, "FlyDSL pa_metadata requires uniform qo length"

    dev = pages_kv_indptr.device
    if num_cu is None:
        num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
    num_batches = context_lens.shape[0]
    query_length = uni_seqlen_qo
    warp_size = get_warp_size(get_rocm_arch())

    compiled = compile_pa_metadata_v1(
        num_cu=num_cu,
        num_heads_k=num_heads_k,
        gqa=num_heads_per_head_k,
        kv_granularity=kv_granularity,
        query_length=query_length,
        warp_size=warp_size,
    )

    # work_metadata_ptrs[0/1] = device addresses of work_indptr / work_info,
    # matching the C++ kernel (which writes them in-kernel via reinterpret_cast).
    # These are exactly the tensors' data_ptr() values, so writing them host-side
    # produces identical bytes.
    if work_metadata_ptrs is not None and work_metadata_ptrs.numel() >= 2:
        work_metadata_ptrs[0] = work_indptr.data_ptr()
        work_metadata_ptrs[1] = work_info.data_ptr()

    # work_info [max_work, 8] and reduce_final_map [num_batches, 2] are written
    # flattened by the kernel.
    work_info_flat = work_info.view(-1)
    reduce_final_map_flat = reduce_final_map.view(-1)

    compiled["launch"](
        seqlens_qo_indptr,
        pages_kv_indptr,
        context_lens,
        work_indptr,
        work_info_flat,
        reduce_indptr,
        reduce_final_map_flat,
        reduce_partial_map,
        num_batches,
    )


# =====================================================================
# compile_pa_decode_metadata — Persistent Scheduling PA decode kernel
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_metadata(
    softmax_scale=None,
    trans_v=False,
    needs_mask=True,
    query_group_size=16,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "packed_fp8",
    head_dim: int = 128,
    block_size: int = None,
    output_dtype_str: str = "bf16",
):
    """Compile a PS-mode PA decode kernel.

    This does NOT bake in num_seqs/num_kv_heads/num_partitions because PS mode
    uses dynamic work distribution. Grid = (num_sm, 1, 1).

    The worklist is load-balanced at ``KV_COMPUTE_BLOCK`` (256-token) **partition**
    granularity (see ``get_pa_metadata``): ``work_info.kv_start/kv_end`` are
    cumulative partition indices.  Each work item is decoded as a range of
    256-token partitions; for ``block_size < 256`` each partition gathers
    ``256 // block_size`` physical pages, and for ``block_size > 256`` (1024) each
    partition is a 256-token sub-tile of one physical page.  ``partial_qo_loc``
    (``work_info[1]``) ``< 0`` writes the final output directly to ``out``;
    ``>= 0`` writes a partial slot that ``pa_reduce_v1`` later combines.
    """
    if block_size is None:
        block_size = KV_BLOCK_SIZE
    if head_dim % QKHE_PER_FETCH != 0 or head_dim % (MFMA_N * NUM_WARPS) != 0 or head_dim % Q_ELEMS_PER_LANE != 0:
        raise ValueError(f"Unsupported head_dim={head_dim}; must be a multiple of {MFMA_N * NUM_WARPS}.")
    _HEAD = head_dim
    _QKHELOOP = head_dim // QKHE_PER_FETCH
    _VHELOOP = head_dim // MFMA_N // NUM_WARPS
    _Q_LANES_PER_HEAD = head_dim // Q_ELEMS_PER_LANE
    _N_K_h = TLOOP * _QKHELOOP * 2
    query_packed_fp8 = query_input_dtype == "packed_fp8"
    query_load_is_bf16 = query_input_dtype == "bf16"
    query_scale_in_kernel = not query_packed_fp8
    cache_scale_vecs = True
    if const_expr(query_packed_fp8):
        raise ValueError(
            "`compile_pa_decode_metadata` only supports bf16/f16 queries with kernel-internal query scale."
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    _softmax_scale = float(softmax_scale)
    _block_size = int(block_size)
    # A partition is KV_COMPUTE_BLOCK (256) tokens.  For small blocks each
    # partition gathers ``_blocks_per_partition`` physical pages; for block_size
    # >= 256 each physical page holds ``_parts_per_block`` partitions (sub-tiles).
    _is_small_block = _block_size < KV_COMPUTE_BLOCK
    _blocks_per_partition = KV_COMPUTE_BLOCK // _block_size if _is_small_block else 1
    _parts_per_block = _block_size // KV_COMPUTE_BLOCK if not _is_small_block else 1
    if _is_small_block:
        if _block_size not in _PA_DECODE_PS_SMALL_BLOCK_SIZES:
            raise ValueError(
                f"compile_pa_decode_metadata: unsupported small block_size={_block_size}; "
                f"expected one of {_PA_DECODE_PS_SMALL_BLOCK_SIZES} or >= {KV_COMPUTE_BLOCK}."
            )
        if per_token_kv:
            raise NotImplementedError(
                "compile_pa_decode_metadata: per_token_kv=True is not supported for "
                "small block_size 16/64; small blocks use compile_pa_decode_ps."
            )
        if not trans_v:
            raise NotImplementedError(
                "compile_pa_decode_metadata: trans_v=False is not supported for small block_size 16/64."
            )

    # LDS allocation
    # Extra LDS for cross-warp v_scale_max reduction (per_token_kv only):
    # NUM_WARPS floats per lane16id slot, aligned to same layout as softmax data.
    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0  # 256 or 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    LDS_SCALE_TOTAL = LDS_SCALE_BYTES if const_expr(per_token_kv) else 0
    # One contiguous shared-storage region carved into sub-regions by byte offset
    # (matches the prior single-buffer layout, incl. logits/Q i32↔i64 aliasing).
    logits_off = 0
    softmax_off = LDS_LOGITS_BYTES
    scale_off = softmax_off + LDS_SOFTMAX_TOTAL
    # Phys-block staging LDS for the small-block path (cross-warp visibility of
    # the per-warp page indices so V can read all blocks of a partition).
    bt_off = scale_off + LDS_SCALE_TOTAL
    _LDS_TOTAL_BYTES = bt_off + (NUM_WARPS * TLOOP * 4 if _is_small_block else 0)

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, _LDS_TOTAL_BYTES // 4, 16]

    # ── @flyc.kernel ─────────────────────────────────────────────────
    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_metadata_kenrel(
        # Raw-pointer kernargs: bare i64 data_ptr() (strides are explicit args).
        out_ptr: fx.Int64,  # output [batch, num_q_heads, head_size]
        partial_out_ptr: fx.Int64,  # partial output [num_partials, 1, nhead, head_dim] fp32
        partial_lse_ptr: fx.Int64,  # partial LSE [num_partials, 1, nhead, 1] fp32
        query_ptr: fx.Int64,  # queries [batch, num_q_heads, head_size]
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
        stride_po_ql: Int32,  # stride for partial_output query-length dim (num_query_heads * head_size)
        stride_pl_ql: Int32,  # stride for partial_lse query-length dim (num_query_heads)
    ):
        tid = gpu.thread_idx.x
        cu_id = gpu.block_idx.x  # CU index (0..num_sm-1)

        # ── Thread decomposition ──
        lane16id = tid & arith.constant(15, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)

        # ── Buffer resources ──
        q_rsrc = buffer_ops.create_buffer_resource_from_addr(query_ptr)
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(out_ptr)
        k_global_ptr = global_ptr_from_addr(key_cache_ptr)
        v_global_ptr = global_ptr_from_addr(value_cache_ptr)
        po_rsrc = buffer_ops.create_buffer_resource_from_addr(partial_out_ptr)
        pl_rsrc = buffer_ops.create_buffer_resource_from_addr(partial_lse_ptr)
        cl_rsrc = buffer_ops.create_buffer_resource_from_addr(context_lengths_ptr)
        wi_rsrc = buffer_ops.create_buffer_resource_from_addr(work_indptr_ptr)
        winfo_rsrc = buffer_ops.create_buffer_resource_from_addr(work_info_ptr)
        kpi_rsrc = buffer_ops.create_buffer_resource_from_addr(kv_page_indices_ptr)
        kvindptr_rsrc = buffer_ops.create_buffer_resource_from_addr(kv_indptr_ptr)
        pip_rsrc = buffer_ops.create_buffer_resource_from_addr(partition_indptr_ptr)
        ks_rsrc = buffer_ops.create_buffer_resource_from_addr(key_scale_ptr)
        vs_rsrc = buffer_ops.create_buffer_resource_from_addr(value_scale_ptr)

        q_scale_val = arith.constant(1.0, type=T.f32)
        if const_expr(per_token_kv):
            k_scale_val = arith.constant(1.0, type=T.f32)
            v_scale_val = arith.constant(1.0, type=T.f32)
        else:
            k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
            v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        # ── LDS region base addresses ──
        # Built ONCE at the top (before any runtime scf.if/for) so they dominate
        # all child regions; downstream LDS traffic uses these raw i64 addresses
        # via the pointer-based _lds_store_vec / _lds_load_vec helpers.  The
        # logits/Q region is addressed as both i32 and i64 off the same base.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _lds_base = fx.Int64(fx.ptrtoint(lds.buf.ptr))
        logits_base = _lds_base + fx.Int64(logits_off)
        softmax_base = _lds_base + fx.Int64(softmax_off)
        scale_base = None
        if const_expr(per_token_kv):
            scale_base = _lds_base + fx.Int64(scale_off)
        bt_base = None
        if const_expr(_is_small_block):
            bt_base = _lds_base + fx.Int64(bt_off)

        # ── Constants ──
        c_kb = stride_k_block
        c_kh = stride_k_head
        c_vb = stride_v_block
        c_vh = stride_v_head

        _softmax_scale_const = arith.constant(_softmax_scale, type=T.f32)
        _softmax_q_scale = _softmax_scale_const * q_scale_val
        _scale = _softmax_q_scale * k_scale_val  # per-tensor only; per-token uses per-token k_scale
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)  # 256-token partition
        c_one = arith.constant(1, type=T.i32)

        local_qhead_idx = warp_id * arith.constant(4, type=T.i32) + rowid
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
            qkhe_loop=_QKHELOOP,
            vhe_loop=_VHELOOP,
        )

        # ── Work loop bounds ──
        # wi[cu_id] and wi[cu_id+1] are adjacent int32; load both in one vec2 load.
        work_bounds = buffer_ops.buffer_load(wi_rsrc, cu_id, vec_width=2, dtype=T.i32)
        work_start = vector.extract(work_bounds, static_position=[0], dynamic_position=[])
        work_end = vector.extract(work_bounds, static_position=[1], dynamic_position=[])

        # Outer work loop — each work item = one (batch, kv_head_range, kv_page_range)
        _work_start_idx = fx.Index(arith.unwrap(work_start))
        _work_end_idx = fx.Index(arith.unwrap(work_end))
        _work_step = arith.index(1)

        for _wi in range(_work_start_idx, _work_end_idx, _work_step):
            work_idx = arith.index_cast(T.i32, _wi)

            # ── Load work_info[work_idx] — 8 × int32, as 2 × vec4 loads ──
            # info_base is a multiple of 8, so both dwordx4 loads are naturally
            # aligned (info_base @ 32 B, info_base+4 @ 16 B).  Fields 3 and 6 are
            # currently unused and simply not extracted.
            info_base = work_idx * arith.constant(8, type=T.i32)
            wi_lo = buffer_ops.buffer_load(winfo_rsrc, info_base, vec_width=4, dtype=T.i32)
            wi_hi = buffer_ops.buffer_load(
                winfo_rsrc, info_base + arith.constant(4, type=T.i32), vec_width=4, dtype=T.i32
            )
            batch_idx = vector.extract(wi_lo, static_position=[0], dynamic_position=[])
            partial_idx = vector.extract(wi_lo, static_position=[1], dynamic_position=[])
            qo_start = vector.extract(wi_lo, static_position=[2], dynamic_position=[])
            kv_start = vector.extract(wi_hi, static_position=[0], dynamic_position=[])
            kv_end = vector.extract(wi_hi, static_position=[1], dynamic_position=[])
            q_head_range = vector.extract(wi_hi, static_position=[3], dynamic_position=[])

            # work_info.kv_start/kv_end are cumulative partition indices (256-token
            # units, summed across batches).  partition_indptr[batch] gives the
            # cumulative-partition base for this sequence (→ local partition index),
            # kv_indptr[batch] gives the physical-page base into kv_page_indices.
            kv_part_base = buffer_ops.buffer_load(pip_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            # kv_indptr[batch] / kv_indptr[batch+1] in one dwordx2 load: this
            # sequence's physical-page base and end in the flat kv_page_indices
            # array.  kv_page_end clamps small-block page-gather reads so the last
            # (partial) partition never reads past the sequence.
            _kvind2 = buffer_ops.buffer_load(kvindptr_rsrc, batch_idx, vec_width=2, dtype=T.i32)
            kv_page_base = vector.extract(_kvind2, static_position=[0], dynamic_position=[])
            kv_page_end = vector.extract(_kvind2, static_position=[1], dynamic_position=[])
            local_part_start = kv_start - kv_part_base

            # Derive kv_head from q_head_range
            q_head_start = q_head_range & arith.constant(0xFFFF, type=T.i32)
            kv_h = udiv_const(q_head_start, query_group_size)

            # Context length for this sequence
            context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            # Head offsets for K and V cache
            _k_head_off = kv_h * c_kh
            _v_head_off = kv_h * c_vh

            (
                _load_kv_scale_scalars,
                _load_v_and_scales,
                _store_vmax_warp,
                _qk_and_intra_softmax,
                _cross_warp_softmax_and_prob_pack,
                _pv_mfma,
            ) = _make_pa_phase_helpers(
                trans_v=trans_v,
                per_token_q=query_scale_in_kernel,
                per_token_kv=per_token_kv,
                needs_mask=needs_mask,
                query_length=query_length,
                kv_h=kv_h,
                v_global_ptr=v_global_ptr,
                ks_rsrc=ks_rsrc,
                vs_rsrc=vs_rsrc,
                logits_base=logits_base,
                softmax_base=softmax_base,
                scale_base=scale_base,
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
                cache_scale_vecs=cache_scale_vecs,
                head_size=_HEAD,
                qkhe_loop=_QKHELOOP,
                vhe_loop=_VHELOOP,
            )

            # Inner KV loop — one CTA processes one 256-token sub-tile across all
            # 1024-token physical blocks.  MTP groups loop is nested INSIDE so K/V
            # load once per physical block and are reused; Q is hoisted out (loaded
            # once per work item, kept in registers).
            def _unwrap(v):
                return v.ir_value() if hasattr(v, "ir_value") else v

            c_ql = arith.constant(query_length, type=T.i32)
            c_zero_i32 = arith.constant(0, type=T.i32)
            c_bpp = arith.constant(_blocks_per_partition, type=T.i32)

            # Output target: partial_qo_loc (work_info[1]) < 0 → write the final
            # output directly; >= 0 → write a partial slot (combined later by
            # pa_reduce_v1).  The partial buffer reserves the first `query_length`
            # rows (pa_reduce_v1 runs on partial_output[query_length:]), so the
            # partial row base is `partial_idx + query_length`.  qo_start
            # (work_info[2]) is the final-output row base for direct works.
            _is_direct = partial_idx < c_zero_i32
            _po_row_base = partial_idx + c_ql

            # Loop over the work item's partitions: [kv_start, kv_end) cumulative
            # partition indices → num_parts local 256-token partitions, in reverse
            # (sink-prone partition 0 processed last for online-softmax stability).
            num_parts_in_work = kv_end - kv_start
            last_part_idx_val = num_parts_in_work - c_one
            _loop_start_g = arith.index(0)
            _loop_stop_g = fx.Index(arith.unwrap(num_parts_in_work))
            _loop_step_g = arith.index(1)

            _mtp_groups = math.ceil(query_length * query_group_size / 16)

            # ── Small-block (16/64) physical-page gather helpers ──
            # A 256-token partition spans `_blocks_per_partition` physical pages.
            # Each warp loads its own K page(s); the per-warp page indices are
            # staged to LDS so every warp can read all pages for the V load.
            # (Only used when `_is_small_block`; for block_size >= 256 a partition
            # is a 256-token sub-tile of a single physical page.)
            _kpi_last = kv_page_end - c_one  # last in-bounds page index for this seq

            def _meta_load_phys_clamped(elem_idx):
                # Clamp to the sequence's page range so the last (partial) partition
                # never reads past the flat kv_page_indices window (kpi_rsrc has no
                # HW bounds check).  Out-of-range lanes map to tokens >= context_len,
                # which softmax masks to 0, so the clamped block's content is unused.
                safe = arith.select(elem_idx < kv_page_end, elem_idx, _kpi_last)
                return buffer_ops.buffer_load(kpi_rsrc, safe, vec_width=1, dtype=T.i32)

            def _meta_stage_phys(local_part):
                page_base = kv_page_base + local_part * c_bpp
                if const_expr(_block_size == 64):
                    return _meta_load_phys_clamped(page_base + warp_id)
                wbase = page_base + warp_id * arith.constant(TLOOP, type=T.i32)
                elems = [_meta_load_phys_clamped(wbase + arith.constant(td, type=T.i32)) for td in range(TLOOP)]
                return fx.Vector.from_elements(elems, dtype=fx.Int32)

            def _meta_store_phys_to_lds(phys_vec):
                if (lane16id | rowid) == c_zero_i32:
                    if const_expr(_block_size == 64):
                        _lds_store_vec(
                            bt_base,
                            warp_id * fx.Int32(4),
                            fx.Vector.from_elements([phys_vec], dtype=fx.Int32),
                            align=4,
                        )
                    else:
                        _lds_store_vec(bt_base, warp_id * arith.constant(TLOOP * 4, type=T.i32), phys_vec, align=16)

            def _meta_load_v_phys_from_lds():
                v_phys_blocks = []
                if const_expr(_block_size == 64):
                    phys_block_vec = _lds_load_vec(bt_base, 0, T.vec(VTLOOP, T.i32), align=16)
                    for vt in range_constexpr(VTLOOP):
                        v_phys_blocks.append(phys_block_vec[vt])
                else:
                    for vt in range_constexpr(VTLOOP):
                        bt_lds_byte = arith.constant(vt * TLOOP * 4, type=T.i32) + rowid * fx.Int32(4)
                        v_phys_blocks.append(_lds_load_vec(bt_base, bt_lds_byte, T.vec(1, T.i32), align=4)[0])
                return v_phys_blocks

            # ── Pre-load Q for every MTP group ONCE per work item.  Each
            # group's q_frags / qi / qhi / qscale stay in registers across
            # the entire KV loop, so we pay the Q-load cost (global → LDS →
            # registers) exactly once per work-item regardless of how many
            # blocks the work item spans.
            q_frags_per_mtp = []
            qi_per_mtp = []
            qhi_per_mtp = []
            qscale_per_mtp = []
            for _mtp_g in range_constexpr(_mtp_groups):
                if const_expr(_mtp_g > 0):
                    gpu.barrier()
                mtp_prefetch = _prefetch_mtp_group_query(
                    q_rsrc,
                    batch_idx,
                    kv_h,
                    stride_q_seq,
                    stride_q_head,
                    lane16id,
                    local_qhead_idx,
                    mtp_group_idx=_mtp_g,
                    query_length=query_length,
                    query_group_size=query_group_size,
                    query_load_is_bf16=query_load_is_bf16,
                    q_lanes_per_head=_Q_LANES_PER_HEAD,
                )
                _qi, _qhi, _qfrags, _qscale = _finish_mtp_group_q_fragments(
                    logits_base,
                    softmax_base,
                    mtp_prefetch,
                    lane16id,
                    rowid,
                    local_qhead_idx,
                    head_size=_HEAD,
                    qkhe_loop=_QKHELOOP,
                    q_lanes_per_head=_Q_LANES_PER_HEAD,
                )
                qi_per_mtp.append(_qi)
                qhi_per_mtp.append(_qhi)
                q_frags_per_mtp.append(_qfrags)
                qscale_per_mtp.append(_qscale)
            gpu.barrier()

            # MTP causal bound per group (depends only on qi, computed once).
            causal_bound_per_mtp = [
                context_len + arith.constant(1 - query_length, type=T.i32) + qi_per_mtp[_mtp_g]
                for _mtp_g in range(_mtp_groups)
            ]

            # ── K init: load the reverse-start (last) partition's K (loop-carried) ──
            local_last_part = local_part_start + last_part_idx_val
            if const_expr(_is_small_block):
                _first_phys_blocks = _meta_stage_phys(local_last_part)
                k_flat0 = _pa_small_block_load_k_flat(
                    k_global_ptr,
                    kv_h,
                    c_kb,
                    c_kh,
                    lane16id,
                    rowid,
                    block_size=_block_size,
                    phys_blocks=_first_phys_blocks,
                    qkhe_loop=_QKHELOOP,
                )
                scale_scalars0 = None
            else:
                _first_phys_block = buffer_ops.buffer_load(
                    kpi_rsrc, kv_page_base + udiv_const(local_last_part, _parts_per_block), vec_width=1, dtype=T.i32
                )
                _first_tile_tok = urem_const(local_last_part, _parts_per_block) * c_cps
                first_k_base = _compute_block_base_dw_i64(_first_phys_block, c_kb, _k_head_off)
                scale_scalars0 = _load_kv_scale_scalars(_first_tile_tok, _first_phys_block)
                k_flat0 = _load_k_flat(
                    k_global_ptr,
                    first_k_base,
                    _first_tile_tok,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                    qkhe_loop=_QKHELOOP,
                )

            # Multi-MTP state packing: (rmax, rsum, outs...) per MTP group,
            # + _N_K_h K values, + 2 scale scalars (per_token_kv only).
            state_width = 2 + _VHELOOP

            def _pack_states_kv(states, k_flat, scale_scalars=None):
                flat = []
                for st in states:
                    rmax, rsum = st[0], st[1]
                    outs = [st[2 + vhe] for vhe in range_constexpr(_VHELOOP)]
                    flat.extend([_unwrap(rmax), _unwrap(rsum)])
                    flat.extend(_unwrap(out) for out in outs)
                flat.extend(_unwrap(v) for v in k_flat)
                if const_expr(cache_scale_vecs and per_token_kv):
                    flat.extend(_unwrap(v) for v in scale_scalars)
                return flat

            def _unpack_states_kv(flat):
                base = state_width * _mtp_groups
                states = [tuple(flat[state_width * i + j] for j in range(state_width)) for i in range(_mtp_groups)]
                k_flat = list(flat[base : base + _N_K_h])
                if const_expr(cache_scale_vecs and per_token_kv):
                    scale_scalars = tuple(flat[base + _N_K_h : base + _N_K_h + 2])
                else:
                    scale_scalars = None
                return states, k_flat, scale_scalars

            init_states = [
                tuple([NEG_INF, ZERO_F] + [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(_VHELOOP)])
                for _ in range(_mtp_groups)
            ]

            # KV outer loop over physical blocks (MTP processing nested inside).
            for ib, state in range(
                _loop_start_g,
                _loop_stop_g,
                _loop_step_g,
                init=_pack_states_kv(init_states, k_flat0, scale_scalars0),
            ):
                cur_states, k_flat, scale_scalars = _unpack_states_kv(state)
                # Reverse iteration: scf.for walks ib forward (0..N-1); remap to
                # the local partition index lp = N-1..0 so the sink-prone first
                # partition is processed last.
                rel_part = last_part_idx_val - arith.index_cast(T.i32, ib)
                lp = local_part_start + rel_part
                next_rel = rel_part - c_one
                next_rel_clamped = arith.select(next_rel >= c_zero_i32, next_rel, c_zero_i32)
                next_lp = local_part_start + next_rel_clamped

                k_ops = unflatten_k(k_flat, qkhe_loop=_QKHELOOP)
                partition_start = lp * c_cps  # within-sequence token offset of this 256-tile

                # Load V (and per-token scales if applicable) ONCE per partition;
                # reused across all MTP groups below.
                if const_expr(_is_small_block):
                    _meta_store_phys_to_lds(_meta_stage_phys(lp))
                    gpu.barrier()
                    v_ops = _pa_small_block_load_v_trans(
                        v_global_ptr,
                        kv_h,
                        c_vb,
                        c_vh,
                        warp_id,
                        lane16id,
                        rowid,
                        _meta_load_v_phys_from_lds(),
                        block_size=_block_size,
                        head_size=_HEAD,
                        vhe_loop=_VHELOOP,
                    )
                else:
                    phys_block = buffer_ops.buffer_load(
                        kpi_rsrc, kv_page_base + udiv_const(lp, _parts_per_block), vec_width=1, dtype=T.i32
                    )
                    tile_token_offset = urem_const(lp, _parts_per_block) * c_cps
                    v_base = _compute_block_base_dw_i64(phys_block, c_vb, _v_head_off)
                    if const_expr(cache_scale_vecs and per_token_kv):
                        v_ops, k_scale_vecs, v_scale_vecs = _load_v_and_scales(
                            v_base,
                            tile_token_offset,
                            phys_block=phys_block,
                            preloaded_scale_scalars=scale_scalars,
                        )
                    else:
                        v_ops = _load_v_and_scales(
                            v_base,
                            tile_token_offset,
                            phys_block=phys_block,
                            preloaded_scale_scalars=scale_scalars,
                        )
                new_states = []
                for _mtp_g in range_constexpr(_mtp_groups):
                    if const_expr(_mtp_g > 0):
                        gpu.barrier()
                    state = cur_states[_mtp_g]
                    rmax, rsum = state[0], state[1]
                    outs = [state[2 + vhe] for vhe in range_constexpr(_VHELOOP)]

                    if const_expr(cache_scale_vecs and per_token_kv):
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

                    # Bugfix: per_token_kv path needs v_max staged to LDS so
                    # _cross_warp_softmax_and_prob_pack can read it for
                    # norm_factor.  Without this write the read sees stale/
                    # uninitialized LDS and produces NaN.
                    if const_expr(per_token_kv):
                        _store_vmax_warp(partition_start, seq_end=context_len, v_scale_vecs=v_scales)

                    gpu.barrier()
                    rmax, rsum, outs, v_correction = _cross_warp_softmax_and_prob_pack(
                        d_out, rmax, rsum, outs, v_scales
                    )
                    gpu.barrier()
                    outs = _pv_mfma(v_ops, outs, v_correction)
                    new_states.append(tuple([rmax, rsum] + outs))

                # Prefetch next partition's K (once per iter, after all MTP groups)
                if const_expr(_is_small_block):
                    k_next_flat = _pa_small_block_load_k_flat(
                        k_global_ptr,
                        kv_h,
                        c_kb,
                        c_kh,
                        lane16id,
                        rowid,
                        block_size=_block_size,
                        phys_blocks=_meta_stage_phys(next_lp),
                        qkhe_loop=_QKHELOOP,
                    )
                    next_scale_scalars = None
                else:
                    next_phys_block = buffer_ops.buffer_load(
                        kpi_rsrc, kv_page_base + udiv_const(next_lp, _parts_per_block), vec_width=1, dtype=T.i32
                    )
                    next_tile_tok = urem_const(next_lp, _parts_per_block) * c_cps
                    next_k_base = _compute_block_base_dw_i64(next_phys_block, c_kb, _k_head_off)
                    next_scale_scalars = _load_kv_scale_scalars(next_tile_tok, next_phys_block)
                    k_next_flat = _load_k_flat(
                        k_global_ptr,
                        next_k_base,
                        next_tile_tok,
                        _k_tok_thread_base,
                        _c_tok_stride_dw,
                        _k_he_off_dw,
                        qkhe_loop=_QKHELOOP,
                    )

                results = yield _pack_states_kv(new_states, k_next_flat, next_scale_scalars)

            # ── Normalize + store one slot per MTP group ──
            # partial_qo_loc (work_info[1]) < 0 → write the fully-normalized output
            # directly to `out` at row qo_start+qi; >= 0 → write a partial slot
            # (+LSE) at row partial_idx+query_length+qi for pa_reduce_v1.
            final_states, _, _ = _unpack_states_kv(results)
            from flydsl._mlir.dialects import math as _mlir_math

            def _store_out_vec(vec_f32x4, elem_off):
                if const_expr(output_dtype_str == "f32"):
                    buffer_ops.buffer_store(vec_f32x4, out_rsrc, elem_off)
                elif const_expr(output_dtype_str == "f16"):
                    buffer_ops.buffer_store(fx.Vector(vec_f32x4).to(fx.Float16), out_rsrc, elem_off)
                else:
                    buffer_ops.buffer_store(fx.Vector(vec_f32x4).to(fx.BFloat16), out_rsrc, elem_off)

            for _mtp_g in range_constexpr(_mtp_groups):
                final_state = final_states[_mtp_g]
                rmax_raw, rsum_raw = final_state[0], final_state[1]
                outs_raw = [final_state[2 + vhe] for vhe in range_constexpr(_VHELOOP)]
                running_max = fx.Float32(rmax_raw)
                running_sum = fx.Float32(rsum_raw)
                outs = [fx.Vector(out_raw) for out_raw in outs_raw]
                outelems_norm = _normalize_pa_output(running_sum, outs, ZERO_F)
                qi_val_mg = qi_per_mtp[_mtp_g]
                qhi_pos_mg = qhi_per_mtp[_mtp_g]
                qhead = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos_mg

                if _is_direct:
                    out_row = qo_start + qi_val_mg
                    for vhe in range_constexpr(_VHELOOP):
                        hs_base = (
                            arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                            + warp_id * arith.constant(MFMA_N, type=T.i32)
                            + rowid * arith.constant(4, type=T.i32)
                        )
                        out_off = out_row * stride_out_seq + qhead * stride_out_head + hs_base
                        _store_out_vec(outelems_norm[vhe], out_off)
                else:
                    _po_row = _po_row_base + qi_val_mg
                    for vhe in range_constexpr(_VHELOOP):
                        hs_base = (
                            arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                            + warp_id * arith.constant(MFMA_N, type=T.i32)
                            + rowid * arith.constant(4, type=T.i32)
                        )
                        po_off = _po_row * stride_po_ql + qhead * arith.constant(_HEAD, type=T.i32) + hs_base
                        buffer_ops.buffer_store(
                            outelems_norm[vhe], po_rsrc, po_off * arith.constant(4, type=T.i32), offset_is_bytes=True
                        )

                    # LSE (split partials only)
                    safe_sum_lse = arith.select(running_sum > ZERO_F, running_sum, arith.constant(1.0, type=T.f32))
                    log_sum = _mlir_math.log(safe_sum_lse, fastmath=arith.FastMathFlags.fast)
                    lse_val = running_max + log_sum
                    pl_off = _po_row * stride_pl_ql + qhead
                    lse_as_i32 = arith.bitcast(T.i32, arith.unwrap(lse_val))
                    buffer_ops.buffer_store(
                        lse_as_i32, pl_rsrc, pl_off * arith.constant(4, type=T.i32), offset_is_bytes=True
                    )

    # ── @flyc.jit launch wrapper ─────────────────────────────────────
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
            # value_attrs=_mfma_agpr_value_attrs(),
        ).launch(grid=(num_sm, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    # launch_pa_decode_metadata.compile_hints["llvm_options"] = PA_MFMA_AGPR_LLVM_OPTIONS

    return {
        "launch": launch_pa_decode_metadata,
        "kernel": pa_decode_metadata_kenrel,
    }


# Deterministic split-partition reduce; drop-in replacement for the racy aiter
# ``pa_reduce_v1`` / ``mla_reduce_v1`` on the PS decode path (root cause of the
# flaky ``test_pa`` NaN). Each output element is combined by exactly one thread,
# serially via online softmax (out = Σ_s w_s·o_s / Σ_s w_s, w_s = exp(lse_s − max)).
# The per-split LSE is a per-(row, head) scalar shared by all head_size lanes, so
# the combine is thread-independent — no atomics/cross-thread reduction → bit-exact.
#
# Inputs (from compile_pa_decode_metadata / get_pa_metadata; caller passes the
# ``[query_length:]`` slices, mirroring the old pa_reduce_v1 call):
#   partial_output f32 : row [num_query_heads, head_size], stride stride_po_row;
#                        each split already normalized by its local denom.
#   partial_lse    f32 : row [num_query_heads], stride stride_pl_row; = max + log(sum).
#   reduce_indptr [batch+1]: group g splits = slots [indptr[g], indptr[g+1]).
#   reduce_final_map [batch,2]: group g → final output row range.
#   reduce_partial_map: slot → partial row base (in the sliced buffer).
# Direct (non-split) outputs have empty groups (indptr delta 0) and are skipped.
@functools.lru_cache(maxsize=64)
def compile_pa_ps_reduce(
    *,
    query_length: int,
    num_query_heads: int,
    head_size: int,
    output_dtype_str: str,
):
    block_threads = head_size
    assert 0 < block_threads <= 1024, "head_size must fit in one workgroup"

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_ps_reduce_kernel(
        final_output_ptr: fx.Tensor,
        partial_output_ptr: fx.Tensor,
        partial_lse_ptr: fx.Tensor,
        reduce_indptr_ptr: fx.Tensor,
        reduce_final_map_ptr: fx.Tensor,
        reduce_partial_map_ptr: fx.Tensor,
        stride_out_seq: Int32,
        stride_out_head: Int32,
        stride_po_row: Int32,  # num_query_heads * head_size
        stride_pl_row: Int32,  # num_query_heads
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        g = fx.Int32(gpu.block_id("x"))
        qhead = fx.Int32(gpu.block_id("y"))
        qi = fx.Int32(gpu.block_id("z"))

        out_rsrc = buffer_ops.create_buffer_resource(final_output_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_output_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_lse_ptr, max_size=True)
        rip_rsrc = buffer_ops.create_buffer_resource(reduce_indptr_ptr, max_size=True)
        rfm_rsrc = buffer_ops.create_buffer_resource(reduce_final_map_ptr, max_size=True)
        rpm_rsrc = buffer_ops.create_buffer_resource(reduce_partial_map_ptr, max_size=True)

        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        c_head = fx.Int32(head_size)

        lo = buffer_ops.buffer_load(rip_rsrc, g, vec_width=1, dtype=T.i32)
        hi = buffer_ops.buffer_load(rip_rsrc, g + fx.Int32(1), vec_width=1, dtype=T.i32)

        # Empty group (direct / padding) → nothing to combine; leave row untouched.
        if hi > lo:
            qo_start = buffer_ops.buffer_load(rfm_rsrc, g * fx.Int32(2), vec_width=1, dtype=T.i32)
            out_row = qo_start + qi

            # Online-softmax combine over the group's splits (dynamic count).
            # State = (running_max, running_denom, running_acc), per-thread.
            for _s, st in fx.range(
                fx.Index(arith.unwrap(lo)),
                fx.Index(arith.unwrap(hi)),
                1,
                init=[c_neg_inf, c_zero_f, c_zero_f],
            ):
                m = fx.Float32(st[0])
                denom = fx.Float32(st[1])
                acc = fx.Float32(st[2])
                slot = fx.Int32(_s)
                prow = buffer_ops.buffer_load(rpm_rsrc, slot, vec_width=1, dtype=T.i32) + qi
                lse = buffer_ops.buffer_load(pl_rsrc, prow * stride_pl_row + qhead, vec_width=1, dtype=T.f32)
                v = buffer_ops.buffer_load(
                    po_rsrc, prow * stride_po_row + qhead * c_head + tid, vec_width=1, dtype=T.f32
                )
                m_new = m.maximumf(lse)
                scale_old = exp2_f32_fast((m - m_new) * c_log2e)
                w = exp2_f32_fast((lse - m_new) * c_log2e)
                denom_new = denom * scale_old + w
                acc_new = acc * scale_old + v * w
                results = yield [m_new, denom_new, acc_new]

            denom_f = fx.Float32(results[1])
            acc_f = fx.Float32(results[2])
            safe_denom = arith.select(denom_f > c_zero_f, denom_f, c_one_f)
            out_acc = acc_f * rcp_f32(safe_denom)

            out_off = out_row * stride_out_seq + qhead * stride_out_head + tid
            if const_expr(output_dtype_str == "f32"):
                out_val = out_acc
            elif const_expr(output_dtype_str == "f16"):
                out_val = out_acc.to(fx.Float16)
            else:
                out_val = out_acc.to(fx.BFloat16)
            buffer_ops.buffer_store(out_val, out_rsrc, out_off)

    @flyc.jit
    def launch_pa_ps_reduce(
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
        pa_ps_reduce_kernel(
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

    return {"launch": launch_pa_ps_reduce, "kernel": pa_ps_reduce_kernel}


_PA_PS_REDUCE_DTYPE_STR = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def pa_ps_reduce(
    *,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    num_query_heads: int,
    head_size: int,
    stream=None,
) -> None:
    """Deterministic drop-in replacement for ``aiter.pa_reduce_v1`` on the PS path.

    ``partial_output`` / ``partial_lse`` must already be the ``[query_length:]``
    slices (same as the old pa_reduce_v1 call site). One reduce group is launched
    per batch tile; empty groups (``reduce_indptr`` delta 0 — direct outputs) skip
    in-kernel, so passing ``reduce_indptr.numel() - 1`` groups needs no host sync.
    """
    num_groups = reduce_indptr.numel() - 1
    stride_po_row = num_query_heads * head_size
    stride_pl_row = num_query_heads
    out_dtype_str = _PA_PS_REDUCE_DTYPE_STR[final_output.dtype]
    compiled = compile_pa_ps_reduce(
        query_length=int(max_seqlen_q),
        num_query_heads=int(num_query_heads),
        head_size=int(head_size),
        output_dtype_str=out_dtype_str,
    )
    kwargs = {}
    if stream is not None:
        kwargs["stream"] = stream
    compiled["launch"](
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
        **kwargs,
    )
