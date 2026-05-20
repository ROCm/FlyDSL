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

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr import math as fly_math
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.env import runtime as flydsl_runtime_env
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels import dpp_utils
from kernels.pa_decode_swa import compile_pa_decode_sw, compile_pa_decode_sw_reduce

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

# LDS sizes
PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4  # 512
LDS_SCALE_V_PADDING = 4  # break K/V same-bank paired writes
LDS_SCALE_V_OFFSET = KV_COMPUTE_BLOCK + LDS_SCALE_V_PADDING
LDS_SCALE_BYTES = (LDS_SCALE_V_OFFSET + KV_COMPUTE_BLOCK) * 4  # K/V per-token scale staging

FP8_MAX = 240.0
LOG2E = 1.4426950408889634

# Match the Gluon PA decode kernel's AGPR allocation:
# .amdhsa_accum_offset 200, .amdhsa_next_free_vgpr 248 => 48 AGPRs,
# with FP8 MFMA using up to a[44:47].
PA_MFMA_AGPR_ALLOC = "48,48"
PA_MFMA_AGPR_LLVM_OPTIONS = {"amdgpu-mfma-vgpr-form": False}

# Number of loop-carried K values (i64)
_N_K = TLOOP * QKHELOOP * 2  # 16
# Number of loop-carried V values (i64)
_N_V = VHELOOP * VTLOOP * 2  # 16

# Tiles per block (1024 tokens / 256 tokens per tile = 4, matches SP3 kNumBlockTiles)
TILES_PER_BLOCK = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK  # 4

_PACKED_FP8_QUERY_DTYPES = tuple(
    dtype
    for dtype in (
        torch.uint8,
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e4m3fn", None),
    )
    if dtype is not None
)


def _cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


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


def _rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def _exp2_f32_fast(value):
    return fly_math.exp2(value, fastmath=arith.FastMathFlags.fast)


def _mfma_agpr_value_attrs():
    return {"passthrough": [["amdgpu-agpr-alloc", PA_MFMA_AGPR_ALLOC]]}


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
    q_elem = q_base + lane16id * 8
    q_chunks = []
    for qwi in range_constexpr(2):
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
    # Total Q footprint = 16 qheads * 128 B = 2048 B, aliased with the later P
    # writes via `logits_lds_i32 / logits_lds_i64` (same base).
    #
    # Writer: thread (warp_id W, rowid R', lane16id L') owns qhead = W*4 + R' =
    # `local_qhead_idx`, and within that qhead owns the 8 FP8 elements at
    # head_dim [L'*8 .. L'*8+7].  We therefore write 2 i32 words (= 1 i64 = 8 B)
    # at `local_qhead_idx * 128 + lane16id * 8`.
    #
    # Reader: MFMA lane layout for mfma_f32_16x16x32_fp8_fp8 (B = Q^T, N = qhead,
    # K = head_dim) — reverse-engineered from `_load_k_flat`: thread (rowid R,
    # lane16id L) consumes, for k_step = qkhe*2 + qkr,
    #   Q[head = L][hd = (qkhe*4 + R) * 16 + qkr * 8 + 0..7]
    # i.e. the read byte offset is `L * 128 + qkhe*64 + R*16 + qkr*8`.
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
    v01.store(logits_lds_i32, [fx.Index(lds_q_i32)])

    q_frags = []
    gpu.barrier()
    query_scale_lane = fx.Vector.load(T.vec(1, fx.Float32.ir_type), softmax_lds_f32, [fx.Index(lane16id)])[0].ir_value()
    for qkhe in range_constexpr(QKHELOOP):
        for qkr in range_constexpr(2):
            # See layout comment above. Byte offset:
            #   lane16id * HEAD_SIZE + qkhe*64 + rowid*16 + qkr*8
            lds_rd_byte = (lane16id << fx.Int32(7)) + fx.Int32(qkhe << 6) + (rowid << fx.Int32(4)) + fx.Int32(qkr << 3)
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
    )
    return qi_val, qhi_pos, q_frags, query_scale_lane


def _normalize_pa_output(running_sum, out0, out1, zero_f):
    one_f = fx.Float32(1.0).ir_value()
    safe_sum = arith.select(running_sum > zero_f, running_sum, one_f)
    inv_sum = _rcp_f32(safe_sum)
    return [
        out0 * vector.broadcast(T.f32x4, inv_sum),
        out1 * vector.broadcast(T.f32x4, inv_sum),
    ]


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
    cache_scale_vecs=False,
    sched_vmem_after_load=True,
    preload_pv_operands=False,
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
            fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32).store(
                scale_lds_f32,
                [fx.Index(scale_stage_token)],
            )
            fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32).store(
                scale_lds_f32,
                [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + scale_stage_token)],
            )
            if const_expr(sched_vmem_after_load):
                rocdl.sched_barrier(rocdl.mask_vmem_rd)
            else:
                rocdl.sched_barrier(0)

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
                if const_expr(sched_vmem_after_load):
                    rocdl.sched_barrier(rocdl.mask_vmem_rd)
                vhe_data.append(v_i64x2)
            v_results.append(vhe_data)

        if const_expr(per_token_kv):
            gpu.barrier()
            if const_expr(cache_scale_vecs):
                k_scale_vecs = []
                v_scale_vecs = []
                for td in range_constexpr(TLOOP):
                    scale_row_base = kv_tok_thread_base + fx.Int32(td * MFMA_N)
                    k_scale_vecs.append(vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(scale_row_base)]))
                    v_scale_vecs.append(
                        vector.load_op(
                            T.f32x4,
                            scale_lds_f32,
                            [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + scale_row_base)],
                        )
                    )
                return v_results, k_scale_vecs, v_scale_vecs

        return v_results

    def _scale_row_base(td: int):
        return kv_tok_thread_base + fx.Int32(td * MFMA_N)

    def _load_k_scale_vec(td: int):
        return vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(_scale_row_base(td))])

    def _load_v_scale_vec(td: int):
        return vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + _scale_row_base(td))])

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
        if const_expr(apply_causal_mask and seq_start is not None):
            in_range = (tok_vec < causal_bound) & (tok_vec >= seq_start)
        elif const_expr(apply_causal_mask):
            in_range = tok_vec < causal_bound
        else:
            in_range = tok_vec >= seq_start
        return arith.select(in_range, logit_vec, vector.broadcast(T.f32x4, arith.unwrap(false_value)))

    def _qk_and_intra_softmax(
        k_ops,
        partition_start,
        v_block_base_dw,
        tile_token_offset_i32,
        q_frags,
        causal_bound,
        query_scale_lane=None,
        *,
        phys_block,
        preloaded_v_and_scales=None,
        seq_start=None,
        precomputed_vmax=False,
    ):
        if const_expr(preloaded_v_and_scales is not None):
            if const_expr(cache_scale_vecs and per_token_kv):
                v_results, k_scale_vecs, v_scale_vecs = preloaded_v_and_scales
            else:
                v_results = preloaded_v_and_scales
        else:
            loaded_v_and_scales = _load_v_and_scales(
                v_block_base_dw,
                tile_token_offset_i32,
                phys_block=phys_block,
            )
            if const_expr(cache_scale_vecs and per_token_kv):
                v_results, k_scale_vecs, v_scale_vecs = loaded_v_and_scales
            else:
                v_results = loaded_v_and_scales

        query_scale_vec = None
        if const_expr(per_token_q):
            query_scale_vec = vector.broadcast(T.f32x4, query_scale_lane * softmax_scale_base)
        d_out = []
        for td in range_constexpr(TLOOP):
            acc = arith.constant_vector(0.0, T.f32x4)
            for k_step in range_constexpr(QKHELOOP * 2):
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

        apply_range_mask = seq_start is not None
        kv_tok_base = (
            partition_start + kv_tok_thread_base if const_expr(apply_causal_mask or apply_range_mask) else None
        )
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

        if const_expr(per_token_kv and not precomputed_vmax):
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                if const_expr(cache_scale_vecs and per_token_kv):
                    vs = _get_v_scale_vec(td, v_scale_vecs)
                else:
                    vs = _get_v_scale_vec(td)
                for i in range_constexpr(4):
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        if const_expr(apply_causal_mask and apply_range_mask):
                            in_range = (kv_tok < causal_bound) & (kv_tok >= seq_start)
                            vs_i = arith.select(in_range, vs_i, zero_f)
                        elif const_expr(apply_causal_mask):
                            vs_i = arith.select(kv_tok < causal_bound, vs_i, zero_f)
                        elif const_expr(apply_range_mask):
                            vs_i = arith.select(kv_tok >= seq_start, vs_i, zero_f)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                v_max_warp = v_max_warp.maximumf(fx.Vector(vs).reduce("max"))
            for sh in [32, 16]:
                v_max_warp = v_max_warp.maximumf(v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            vector.store(
                fx.Vector.from_elements([v_max_warp], dtype=fx.Float32),
                softmax_lds_f32,
                [sm_vmax_wr_off],
            )
        if const_expr(cache_scale_vecs and per_token_kv):
            return d_out, v_results, v_scale_vecs
        return d_out, v_results

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, o0, o1, v_scale_vecs=None):
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
        o0 = o0 * vector.broadcast(T.f32x4, accum_scale)
        o1 = o1 * vector.broadcast(T.f32x4, accum_scale)

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
                    _get_v_scale_vec(td, v_scale_vecs) * vector.broadcast(T.f32x4, prob_scale * norm_factor)
                )
        else:
            prob_scale = my_warp_rescale * part_to_new
            v_correction = v_scale_val
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * vector.broadcast(T.f32x4, prob_scale)

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
        return rmax, rsum, o0, o1, v_correction

    def _pv_mfma(v_ops, o0, o1, v_correction):
        v_correction = fx.Float32(v_correction).ir_value()
        fm_contract = arith.FastMathFlags.contract
        v_correction_vec = vector.broadcast(T.f32x4, v_correction)
        if const_expr(preload_pv_operands):
            v_i64s = []
            for vhe in range_constexpr(VHELOOP):
                for vt in range_constexpr(VTLOOP):
                    v_i64x2 = fx.Vector(v_ops[vt][vhe])
                    for j in range_constexpr(2):
                        v_i64s.append(v_i64x2[j])
            p_i64s = []
            for vt in range_constexpr(VTLOOP):
                for j in range_constexpr(2):
                    p_i64_idx = pv_prob_i64_indices[vt * 2 + j]
                    p_i64s.append(fx.Vector.load(T.vec(1, T.i64), logits_lds_i64, [p_i64_idx])[0])
            for vhe in range_constexpr(VHELOOP):
                tmp_out = arith.constant_vector(0.0, T.f32x4)
                for vt in range_constexpr(VTLOOP):
                    for j in range_constexpr(2):
                        tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4,
                            [
                                v_i64s[vhe * VTLOOP * 2 + vt * 2 + j],
                                p_i64s[vt * 2 + j],
                                tmp_out,
                                0,
                                0,
                                0,
                            ],
                        )
                if const_expr(vhe == 0):
                    o0 = arith.addf(
                        arith.mulf(tmp_out, v_correction_vec, fastmath=fm_contract),
                        o0,
                        fastmath=fm_contract,
                    )
                else:
                    o1 = arith.addf(
                        arith.mulf(tmp_out, v_correction_vec, fastmath=fm_contract),
                        o1,
                        fastmath=fm_contract,
                    )
        else:
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
                if const_expr(vhe == 0):
                    o0 = arith.addf(
                        arith.mulf(tmp_out, v_correction_vec, fastmath=fm_contract),
                        o0,
                        fastmath=fm_contract,
                    )
                else:
                    o1 = arith.addf(
                        arith.mulf(tmp_out, v_correction_vec, fastmath=fm_contract),
                        o1,
                        fastmath=fm_contract,
                    )
        return o0, o1

    def _finalize_block_split_group(
        d_out,
        v_ops,
        partition_start,
        causal_bound,
        rmax,
        rsum,
        o0,
        o1,
        *,
        seq_start=None,
    ):
        apply_range_mask = seq_start is not None

        kv_tok_base = (
            partition_start + kv_tok_thread_base if const_expr(apply_causal_mask or apply_range_mask) else None
        )
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

        if const_expr(per_token_kv):
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = _load_v_scale_vec(td)
                for i in range_constexpr(4):
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        if const_expr(apply_causal_mask and apply_range_mask):
                            in_range = (kv_tok < causal_bound) & (kv_tok >= seq_start)
                            vs_i = arith.select(in_range, vs_i, zero_f)
                        elif const_expr(apply_causal_mask):
                            vs_i = arith.select(kv_tok < causal_bound, vs_i, zero_f)
                        elif const_expr(apply_range_mask):
                            vs_i = arith.select(kv_tok >= seq_start, vs_i, zero_f)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                v_max_warp = v_max_warp.maximumf(fx.Vector(vs).reduce("max"))
            for sh in [32, 16]:
                v_max_warp = v_max_warp.maximumf(v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            vector.store(
                fx.Vector.from_elements([v_max_warp], dtype=fx.Float32),
                softmax_lds_f32,
                [sm_vmax_wr_off],
            )

        gpu.barrier()
        rmax, rsum, o0, o1, v_correction = _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, o0, o1)
        gpu.barrier()
        o0, o1 = _pv_mfma(v_ops, o0, o1, v_correction)
        return rmax, rsum, o0, o1

    return (
        _load_kv_scale_scalars,
        _load_v_and_scales,
        _store_vmax_warp,
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
        _finalize_block_split_group,
    )


def _expand_pa_metadata_for_block_splits(
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    query_length: int,
    *,
    block_split_factor: int = TILES_PER_BLOCK,
):
    """Expand PA metadata so each 1024-token work tile reduces 4 block-split partials.

    `get_pa_metadata_v1()` only materializes split partials and uses `partial_idx=-1`
    for direct tiles that write final output directly. With `grid_z=4`, every work item
    becomes four partials, so direct tiles must also participate in the reduce stage.
    """

    dev = work_info.device
    valid_work = int(work_indptr[-1].item())
    work_info_cpu = work_info[:valid_work].cpu()

    if valid_work == 0:
        empty_reduce_indptr = torch.zeros(1, dtype=torch.int32, device=dev)
        empty_reduce_final_map = torch.empty((0, 2), dtype=torch.int32, device=dev)
        empty_reduce_partial_map = torch.empty((0,), dtype=torch.int32, device=dev)
        return work_info[:0].contiguous(), empty_reduce_indptr, empty_reduce_final_map, empty_reduce_partial_map

    group_order = []
    group_slot_keys = {}
    group_slot_seen = {}
    row_slot_keys = []

    for wi in range(valid_work):
        row = work_info_cpu[wi]
        q_start = int(row[2].item())
        q_end = int(row[3].item())
        orig_partial_idx = int(row[1].item())
        group_key = (q_start, q_end)
        if group_key not in group_slot_keys:
            group_order.append(group_key)
            group_slot_keys[group_key] = []
            group_slot_seen[group_key] = set()

        if orig_partial_idx >= 0:
            slot_key = ("split", orig_partial_idx)
        else:
            slot_key = ("direct", q_start, q_end)

        if slot_key not in group_slot_seen[group_key]:
            group_slot_seen[group_key].add(slot_key)
            group_slot_keys[group_key].append(slot_key)
        row_slot_keys.append(slot_key)

    slot_id_by_key = {}
    next_slot_id = 0
    for group_key in group_order:
        for slot_key in group_slot_keys[group_key]:
            if slot_key not in slot_id_by_key:
                slot_id_by_key[slot_key] = next_slot_id
                next_slot_id += 1

    for wi, slot_key in enumerate(row_slot_keys):
        work_info_cpu[wi, 1] = slot_id_by_key[slot_key] * query_length

    reduce_indptr_cpu = torch.zeros(len(group_order) + 1, dtype=torch.int32)
    reduce_final_map_cpu = torch.empty((len(group_order), 2), dtype=torch.int32)
    reduce_partial_map_entries = []
    running = 0

    for group_idx, group_key in enumerate(group_order):
        q_start, q_end = group_key
        reduce_final_map_cpu[group_idx, 0] = q_start
        reduce_final_map_cpu[group_idx, 1] = q_end
        for slot_key in group_slot_keys[group_key]:
            slot_id = slot_id_by_key[slot_key]
            base_row = slot_id * query_length * block_split_factor
            for block_split_idx in range(block_split_factor):
                reduce_partial_map_entries.append(base_row + block_split_idx * query_length)
                running += 1
        reduce_indptr_cpu[group_idx + 1] = running

    work_info_out = work_info_cpu.to(device=dev).contiguous()
    reduce_indptr = reduce_indptr_cpu.to(device=dev)
    reduce_final_map = reduce_final_map_cpu.to(device=dev)
    reduce_partial_map = torch.tensor(reduce_partial_map_entries, dtype=torch.int32, device=dev)
    return work_info_out, reduce_indptr, reduce_final_map, reduce_partial_map


# =====================================================================
# compile_pa_decode_ps — Persistent Scheduling PA decode kernel
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_ps(
    softmax_scale=None,
    trans_v=False,
    needs_mask=True,
    query_group_size=QUERY_GROUP_SIZE,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "packed_fp8",
):
    """Compile a PS-mode PA decode kernel.

    This does NOT bake in num_seqs/num_kv_heads/num_partitions because PS mode
    uses dynamic work distribution. Grid = (num_sm, 1, 4).
    """
    arch = get_hip_arch()
    query_packed_fp8 = query_input_dtype == "packed_fp8"
    query_load_is_bf16 = query_input_dtype == "bf16"
    query_scale_in_kernel = not query_packed_fp8
    cache_scale_vecs = True
    if const_expr(query_packed_fp8):
        raise ValueError("`compile_pa_decode_ps` only supports bf16/f16 queries with kernel-internal query scale.")
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE**0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024 for PS mode (matches SP3 kBlockSize)

    # LDS allocation
    # Extra LDS for cross-warp v_scale_max reduction (per_token_kv only):
    # NUM_WARPS floats per lane16id slot, aligned to same layout as softmax data.
    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0  # 256 or 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    LDS_SCALE_TOTAL = LDS_SCALE_BYTES if const_expr(per_token_kv) else 0
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL
    scale_off = softmax_off + LDS_SOFTMAX_TOTAL
    allocator.ptr += LDS_SCALE_TOTAL

    # ── @flyc.kernel ─────────────────────────────────────────────────
    @flyc.kernel
    def pa_decode_ps_kernel(
        out_ptr: fx.Tensor,  # output [batch, num_q_heads, head_size]
        partial_out_ptr: fx.Tensor,  # partial output [num_partials, 1, nhead, head_dim] fp32
        partial_lse_ptr: fx.Tensor,  # partial LSE [num_partials, 1, nhead, 1] fp32
        query_ptr: fx.Tensor,  # queries [batch, num_q_heads, head_size]
        key_cache_ptr: fx.Tensor,  # key cache
        value_cache_ptr: fx.Tensor,  # value cache
        context_lengths_ptr: fx.Tensor,  # [batch] int32
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        work_indptr_ptr: fx.Tensor,  # [num_sm + 1] int32
        work_info_ptr: fx.Tensor,  # [num_work, 8] int32 (flattened to 1D)
        kv_page_indices_ptr: fx.Tensor,  # [total_pages] int32
        kv_indptr_ptr: fx.Tensor,  # [num_seqs + 1] int32 — prefix sum of pages per seq
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
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_global_ptr = _extract_global_ptr(key_cache_ptr)
        v_global_ptr = _extract_global_ptr(value_cache_ptr)
        po_rsrc = buffer_ops.create_buffer_resource(partial_out_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_lse_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        wi_rsrc = buffer_ops.create_buffer_resource(work_indptr_ptr, max_size=True)
        winfo_rsrc = buffer_ops.create_buffer_resource(work_info_ptr, max_size=True)
        kpi_rsrc = buffer_ops.create_buffer_resource(kv_page_indices_ptr, max_size=True)
        kvindptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)

        q_scale_val = arith.constant(1.0, type=T.f32)
        if const_expr(per_token_kv):
            k_scale_val = arith.constant(1.0, type=T.f32)
            v_scale_val = arith.constant(1.0, type=T.f32)
        else:
            k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
            v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        # ── LDS views ──
        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_TOTAL // 4,)).get()
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()
        scale_lds_f32 = None
        if const_expr(per_token_kv):
            scale_lds_f32 = SmemPtr(smem_base, scale_off, T.f32, shape=(LDS_SCALE_BYTES // 4,)).get()

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
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_bs = arith.constant(_bs, type=T.i32)
        c_tpb = arith.constant(TILES_PER_BLOCK, type=T.i32)

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
        )

        # ── Work loop bounds ──
        work_start = buffer_ops.buffer_load(wi_rsrc, cu_id, vec_width=1, dtype=T.i32)
        work_end = buffer_ops.buffer_load(wi_rsrc, cu_id + c_one, vec_width=1, dtype=T.i32)

        # ════════════════════════════════════════════════════════════
        # Outer work loop — iterate over assigned work items
        # Each work item = one (batch, kv_head_range, kv_page_range)
        # ════════════════════════════════════════════════════════════
        _work_start_idx = fx.Index(arith.unwrap(work_start))
        _work_end_idx = fx.Index(arith.unwrap(work_end))
        _work_step = arith.index(1)

        for _wi in range(_work_start_idx, _work_end_idx, _work_step):
            work_idx = arith.index_cast(T.i32, _wi)

            # ── Load work_info[work_idx] — 8 × int32 ──
            info_base = work_idx * arith.constant(8, type=T.i32)
            batch_idx = buffer_ops.buffer_load(winfo_rsrc, info_base, vec_width=1, dtype=T.i32)
            partial_idx = buffer_ops.buffer_load(winfo_rsrc, info_base + c_one, vec_width=1, dtype=T.i32)
            kv_start = buffer_ops.buffer_load(
                winfo_rsrc, info_base + arith.constant(4, type=T.i32), vec_width=1, dtype=T.i32
            )
            kv_end = buffer_ops.buffer_load(
                winfo_rsrc, info_base + arith.constant(5, type=T.i32), vec_width=1, dtype=T.i32
            )
            q_head_range = buffer_ops.buffer_load(
                winfo_rsrc, info_base + arith.constant(7, type=T.i32), vec_width=1, dtype=T.i32
            )

            # Absolute token offset for the first page of this work item within its sequence.
            # kv_start is an absolute index into kv_page_indices; kv_indptr[batch_idx] is
            # the page index where this sequence starts.  Their difference * KV_BLOCK_SIZE
            # gives the token offset from sequence start to the first token we process.
            kv_indptr_batch = buffer_ops.buffer_load(kvindptr_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            kv_start_abs_tok = (kv_start - kv_indptr_batch) * c_bs

            # Derive kv_head from q_head_range
            q_head_start = q_head_range & arith.constant(0xFFFF, type=T.i32)
            kv_h = _udiv_const(q_head_start, query_group_size)

            # Context length for this sequence
            context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            # ── Prologue: load first block's tile 0 K data ──
            first_phys_block = buffer_ops.buffer_load(kpi_rsrc, kv_start, vec_width=1, dtype=T.i32)
            # Head offsets for K and V cache
            _k_head_off = kv_h * c_kh
            _v_head_off = kv_h * c_vh

            (
                _load_kv_scale_scalars,
                _load_v_and_scales,
                _store_vmax_warp_unused,
                _qk_and_intra_softmax,
                _cross_warp_softmax_and_prob_pack,
                _pv_mfma,
                _finalize_block_split_group_unused,
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
                cache_scale_vecs=cache_scale_vecs,
                sched_vmem_after_load=False,
                preload_pv_operands=True,
            )

            # ════════════════════════════════════════════════════════
            # Inner KV loop — one CTA processes one 256-token sub-tile
            # across all 1024-token physical blocks in the work item.
            # ════════════════════════════════════════════════════════
            def _unwrap(v):
                return v.ir_value() if hasattr(v, "ir_value") else v

            def _pack_state(rmax, rsum, o0, o1, k_flat, scale_scalars=None):
                state = [rmax, rsum, o0, o1] + k_flat
                if const_expr(cache_scale_vecs and per_token_kv):
                    state += list(scale_scalars)
                return [_unwrap(v) for v in state]

            def _unpack_state(state):
                k_flat = list(state[4 : 4 + _N_K])
                if const_expr(per_token_kv):
                    scale_scalars = tuple(state[4 + _N_K : 6 + _N_K])
                else:
                    scale_scalars = None
                return state[0], state[1], state[2], state[3], k_flat, scale_scalars

            def _process_block_split(
                phys_block,
                block_idx_in_work,
                rmax,
                rsum,
                o0,
                o1,
                tile_token_offset_i32,
                k_ops,
                scale_scalars,
                next_phys_block=None,
                next_k_base=None,
            ):
                """Process one 256-token block split inside a 1024-token KV page."""
                partition_start = kv_start_abs_tok + block_idx_in_work * c_bs + tile_token_offset_i32
                v_base = _compute_block_base_dw_i64(phys_block, c_vb, _v_head_off)
                preloaded_v_and_scales = _load_v_and_scales(
                    v_base,
                    tile_token_offset_i32,
                    phys_block=phys_block,
                    preloaded_scale_scalars=scale_scalars,
                )
                if const_expr(per_token_kv):
                    d_out, v_ops, v_scales = _qk_and_intra_softmax(
                        k_ops,
                        partition_start,
                        v_base,
                        tile_token_offset_i32,
                        q_frags,
                        causal_bound,
                        query_scale_lane=query_scale_lane,
                        phys_block=phys_block,
                        preloaded_v_and_scales=preloaded_v_and_scales,
                    )
                else:
                    d_out, v_ops = _qk_and_intra_softmax(
                        k_ops,
                        partition_start,
                        v_base,
                        tile_token_offset_i32,
                        q_frags,
                        causal_bound,
                        query_scale_lane=query_scale_lane,
                        phys_block=phys_block,
                        preloaded_v_and_scales=preloaded_v_and_scales,
                    )
                    v_scales = None

                gpu.barrier()
                rmax, rsum, o0, o1, v_correction = _cross_warp_softmax_and_prob_pack(
                    d_out, rmax, rsum, o0, o1, v_scales
                )
                if const_expr(next_k_base is not None):
                    next_scale_scalars = _load_kv_scale_scalars(tile_token_offset_i32, next_phys_block)
                    k_next_flat = _load_k_flat(
                        k_global_ptr,
                        next_k_base,
                        tile_token_offset_i32,
                        _k_tok_thread_base,
                        _c_tok_stride_dw,
                        _k_he_off_dw,
                        sched_vmem_after_load=False,
                    )
                else:
                    k_next_flat = None
                    next_scale_scalars = None

                gpu.barrier()
                o0, o1 = _pv_mfma(v_ops, o0, o1, v_correction)
                return rmax, rsum, o0, o1, k_next_flat, next_scale_scalars

            # Metadata remaps every work tile into a partial slot shared across q-head ranges.
            # grid_z then expands each slot into 4 block-split partials.
            c_ql = arith.constant(query_length, type=T.i32)
            c_zero_i32 = arith.constant(0, type=T.i32)
            block_split_idx = gpu.block_idx.z
            tile_token_offset = block_split_idx * c_cps
            _partial_ge_zero = partial_idx >= c_zero_i32
            _po_row_base = arith.select(
                _partial_ge_zero,
                partial_idx * c_tpb + block_split_idx * c_ql + c_ql,
                c_zero_i32,
            )

            # Unified loop bounds (shared across mtp_g passes — blocks don't change per mtp_g)
            num_blocks_in_work = kv_end - kv_start
            last_block_idx_val = num_blocks_in_work - c_one
            _loop_start_g = arith.index(0)
            _loop_stop_g = fx.Index(arith.unwrap(num_blocks_in_work))
            _loop_step_g = arith.index(1)

            # ── MTP groups: Python compile-time loop — one MLIR KV-loop per group ──
            # Use range_constexpr so AST rewriter keeps this as a plain Python loop
            _mtp_groups = math.ceil(query_length * query_group_size / 16)
            next_mtp_prefetch = None
            for _mtp_g in range_constexpr(_mtp_groups):
                # Between passes: barrier ensures prev pass's LDS prob-reads are done
                if const_expr(_mtp_g > 0):
                    gpu.barrier()

                if const_expr(_mtp_g == 0):
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
                    )
                else:
                    mtp_prefetch = next_mtp_prefetch

                qi_val, qhi_pos, q_frags, query_scale_lane = _finish_mtp_group_q_fragments(
                    logits_lds_i32,
                    logits_lds_i64,
                    softmax_lds_f32,
                    mtp_prefetch,
                    lane16id,
                    rowid,
                    local_qhead_idx,
                )

                if const_expr(_mtp_g + 1 < _mtp_groups):
                    next_mtp_prefetch = _prefetch_mtp_group_query(
                        q_rsrc,
                        batch_idx,
                        kv_h,
                        stride_q_seq,
                        stride_q_head,
                        lane16id,
                        local_qhead_idx,
                        mtp_group_idx=_mtp_g + 1,
                        query_length=query_length,
                        query_group_size=query_group_size,
                        query_load_is_bf16=query_load_is_bf16,
                    )

                gpu.barrier()

                # MTP causal bound for this lane's qi_val token
                causal_bound = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val

                # ── K init: load this CTA's 256-token block split for the first block ──
                first_k_base = _compute_block_base_dw_i64(first_phys_block, c_kb, _k_head_off)
                scale_scalars = _load_kv_scale_scalars(tile_token_offset, first_phys_block)
                k_flat = _load_k_flat(
                    k_global_ptr,
                    first_k_base,
                    tile_token_offset,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                    sched_vmem_after_load=False,
                )

                init_state = _pack_state(
                    NEG_INF,
                    ZERO_F,
                    arith.constant_vector(0.0, T.f32x4),
                    arith.constant_vector(0.0, T.f32x4),
                    k_flat,
                    scale_scalars,
                )

                for ib, state in range(_loop_start_g, _loop_stop_g, _loop_step_g, init=init_state):
                    running_max, running_sum, out0, out1, k_flat, scale_scalars = _unpack_state(state)
                    block_idx = arith.index_cast(T.i32, ib)

                    phys_block = buffer_ops.buffer_load(kpi_rsrc, kv_start + block_idx, vec_width=1, dtype=T.i32)
                    next_idx_raw = block_idx + c_one
                    next_idx_clamped = arith.select(next_idx_raw < num_blocks_in_work, next_idx_raw, last_block_idx_val)
                    next_phys_block = buffer_ops.buffer_load(
                        kpi_rsrc, kv_start + next_idx_clamped, vec_width=1, dtype=T.i32
                    )
                    next_k_base = _compute_block_base_dw_i64(next_phys_block, c_kb, _k_head_off)

                    k_ops = _unflatten_k(k_flat)

                    running_max, running_sum, out0, out1, k_next_flat, next_scale_scalars = _process_block_split(
                        phys_block,
                        block_idx,
                        running_max,
                        running_sum,
                        out0,
                        out1,
                        tile_token_offset,
                        k_ops,
                        scale_scalars,
                        next_phys_block=next_phys_block,
                        next_k_base=next_k_base,
                    )

                    results = yield _pack_state(
                        running_max,
                        running_sum,
                        out0,
                        out1,
                        k_next_flat,
                        next_scale_scalars,
                    )

                running_max, running_sum, out0, out1, _, _ = _unpack_state(results)

                # ── Normalize output ──
                outelems_norm = _normalize_pa_output(running_sum, out0, out1, ZERO_F)

                for vhe in range_constexpr(VHELOOP):
                    hs_base = (
                        arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                        + warp_id * arith.constant(MFMA_N, type=T.i32)
                        + rowid * arith.constant(4, type=T.i32)
                    )
                    # qhi_pos: mtp_g-based head position within kv_head group
                    qhead = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos
                    _po_row = _po_row_base + qi_val
                    po_off = _po_row * stride_po_ql + qhead * arith.constant(HEAD_SIZE, type=T.i32) + hs_base

                    # pa_reduce_v1 expects normalized partial output from every block split.
                    buffer_ops.buffer_store(
                        outelems_norm[vhe], po_rsrc, po_off * arith.constant(4, type=T.i32), offset_is_bytes=True
                    )

                # ── LSE ──
                safe_sum_lse = arith.select(running_sum > ZERO_F, running_sum, arith.constant(1.0, type=T.f32))
                from flydsl._mlir.dialects import math as _mlir_math

                log_sum = _mlir_math.log(safe_sum_lse, fastmath=arith.FastMathFlags.fast)
                lse_val = running_max + log_sum
                qhead_lse = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos
                _po_row_lse = _po_row_base + qi_val
                pl_off = _po_row_lse * stride_pl_ql + qhead_lse
                lse_as_i32 = arith.bitcast(T.i32, lse_val)
                buffer_ops.buffer_store(
                    lse_as_i32, pl_rsrc, pl_off * arith.constant(4, type=T.i32), offset_is_bytes=True
                )

    # ── @flyc.jit launch wrapper ─────────────────────────────────────
    @flyc.jit
    def launch_pa_decode_ps(
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
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_ps_kernel(
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
            value_attrs=_mfma_agpr_value_attrs(),
        ).launch(grid=(num_sm, 1, TILES_PER_BLOCK), block=(BLOCK_THREADS, 1, 1), stream=stream)

    launch_pa_decode_ps.compile_hints["llvm_options"] = PA_MFMA_AGPR_LLVM_OPTIONS

    return {
        "launch": launch_pa_decode_ps,
        "kernel": pa_decode_ps_kernel,
        "allocator": allocator,
    }


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
):
    """Compute PA metadata (worklist, reduce maps) via get_pa_metadata_v1.

    Then expand each 1024-token work tile into 4 block-split partials so the PS
    kernel can launch with `grid=(num_sm, 1, 4)` and still reuse `pa_reduce_v1`.

    Returns a dict with: work_indptr, work_info_flat, reduce_indptr,
    reduce_final_map, reduce_partial_map, num_sm, partial_output,
    partial_lse, stride_po_partial, stride_pl_partial.
    """
    from aiter.ops.attention import get_pa_metadata_info_v1, get_pa_metadata_v1

    dev = query.device
    batch_size = context_lengths.shape[0]
    query_length = query.shape[0] // batch_size
    head_size = query.shape[-1]

    props = torch.cuda.get_device_properties(dev)
    num_sm = props.multi_processor_count

    seqlens_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=dev) * query_length

    block_size = key_cache.shape[-2] if len(key_cache.shape) == 5 else key_cache.shape[-2]

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_pa_metadata_info_v1(batch_size, num_kv_heads)

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
        kv_granularity=max(block_size, 16),
        block_size=block_size,
        max_seqlen_qo=query_length,
        uni_seqlen_qo=query_length,
        fast_mode=True,
        max_split_per_batch=-1,
    )

    work_info, reduce_indptr, reduce_final_map, reduce_partial_map = _expand_pa_metadata_for_block_splits(
        work_indptr, work_info, query_length, block_split_factor=TILES_PER_BLOCK
    )
    work_info_flat = work_info.reshape(-1).contiguous()

    num_partials = reduce_partial_map.size(0)
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


def get_sw_ps_max_context_partition_num(
    sliding_window: int,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    if sliding_window <= 0:
        return 0
    window_token_count = sliding_window + query_length
    return _cdiv(window_token_count - 1, context_partition_size) + 1


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
    if is_graph_capturing and not flydsl_runtime_env.enable_cache:
        raise ValueError(
            "CUDA graph capture for `pa_decode_ps_launch` requires "
            "`FLYDSL_RUNTIME_ENABLE_CACHE=1` so compiled launch artifacts stay alive."
        )
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

    # Detect per-token vs per-tensor quantization from scale tensor dimensionality
    per_token_kv = key_scale.ndim > 1  # per-tensor: shape [1]; per-token: shape [blocks, heads, block_size, 1]

    if metadata is None:
        if is_graph_capturing:
            raise ValueError(
                "CUDA graph capture requires precomputed `metadata`; "
                "call `get_pa_metadata()` before capture and pass it via `metadata=`."
            )
        metadata = get_pa_metadata(query, key_cache, context_lengths, kv_indptr, num_query_heads, num_kv_heads)

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
            max_context_partition_num = get_sw_ps_max_context_partition_num(
                sliding_window,
                context_partition_size,
                query_length,
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
        )

        compiled_sw["launch"](
            exp_sums,
            max_logits,
            temporary_output,
            output_5d,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
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
        compiled_sw_reduce["launch"](
            output_5d,
            exp_sums,
            max_logits,
            temporary_output,
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

    work_indptr = metadata["work_indptr"]
    work_info_flat = metadata["work_info_flat"]
    partial_output = metadata["partial_output"]
    partial_lse = metadata["partial_lse"]
    stride_po_partial = metadata["stride_po_partial"]
    stride_pl_partial = metadata["stride_pl_partial"]
    num_sm = metadata["num_sm"]

    compiled = compile_pa_decode_ps(
        softmax_scale=softmax_scale,
        trans_v=trans_v,
        query_group_size=query_group_size,
        per_token_kv=per_token_kv,
        query_length=query_length,
        query_input_dtype=query_input_dtype,
    )

    stride_po_ql = metadata.get("stride_po_ql", num_query_heads * query.shape[-1])
    stride_pl_ql = metadata.get("stride_pl_ql", num_query_heads)

    compiled["launch"](
        output,
        partial_output,
        partial_lse,
        query,
        key_cache,
        value_cache,
        context_lengths,
        key_scale,
        value_scale,
        work_indptr,
        work_info_flat,
        kv_page_indices,
        kv_indptr,
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

    from aiter.ops.attention import pa_reduce_v1

    pa_reduce_v1(
        partial_output[query_length:],
        partial_lse[query_length:],
        metadata["reduce_indptr"],
        metadata["reduce_final_map"],
        metadata["reduce_partial_map"],
        query_length,  # max_qlen
        output,
        None,
    )

    return "ps_split_reduce"
