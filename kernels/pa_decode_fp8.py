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
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels import dpp_utils
from kernels.pa_decode_swa import compile_pa_decode_sw, compile_pa_decode_sw_reduce

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
MFMA_K = 32

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

# Match the Gluon PA decode kernel's AGPR allocation:
# .amdhsa_accum_offset 200, .amdhsa_next_free_vgpr 248 => 48 AGPRs,
# with FP8 MFMA using up to a[44:47].
PA_MFMA_AGPR_ALLOC = "48,48"
PA_MFMA_AGPR_LLVM_OPTIONS = {"amdgpu-mfma-vgpr-form": False}

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


def _global_load_i32(global_ptr, elem_offset_i32):
    byte_offset_i64 = fx.Int64(elem_offset_i32) * fx.Int64(4)
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=byte_offset_i64, elem_type=T.i8)
    return llvm.LoadOp(T.i32, ptr, alignment=4).result


def _rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def _exp2_amdgcn_scalar(scalar_value):
    """Direct ``llvm.amdgcn.exp2.f32`` intrinsic call on one f32 scalar.

    The default ``fly_math.exp2`` lowering routes through ``__ocml_exp2_f32``,
    which (for full IEEE range/subnormal correctness) expands at codegen time
    into ``v_exp_f32 + v_ldexp_f32`` per element.  The amdgcn intrinsic compiles
    to a single ``v_exp_f32``, matching what Gluon emits for its softmax.
    Skipping ldexp is safe here because softmax inputs are pre-clamped via
    `safe_qk_max`/`safe_partition_max` so the operand is in the fast-range.
    """
    from flydsl._mlir.ir import F32Type

    raw = (
        arith.unwrap(scalar_value)
        if hasattr(scalar_value, "ir_value") or hasattr(scalar_value, "type")
        else scalar_value
    )
    f32_ty = F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def _exp2_f32_fast(value):
    """Compute 2^value (elementwise for vectors), using the amdgcn intrinsic
    to avoid the ``v_exp_f32 + v_ldexp_f32`` pair OCML lowering produces."""
    from flydsl._mlir.dialects import vector as _vector_dialect
    from flydsl._mlir.ir import VectorType

    raw = arith.unwrap(value) if hasattr(value, "ir_value") or hasattr(value, "type") else value
    ty = raw.type
    if isinstance(ty, VectorType):
        n = ty.shape[0]
        elems = []
        for i in range(n):
            scalar = _vector_dialect.extract(raw, static_position=[i], dynamic_position=[])
            elems.append(_exp2_amdgcn_scalar(scalar))
        return _vector_dialect.from_elements(ty, elems)
    return _exp2_amdgcn_scalar(raw)


def _mfma_agpr_value_attrs():
    return {"passthrough": [["amdgpu-agpr-alloc", PA_MFMA_AGPR_ALLOC]]}


def _maxnumf(a, b):
    """Non-NaN-propagating max, equivalent to ``a.maximumf(b)`` for non-NaN
    inputs but lowers to a single ``v_max_f32`` instead of the
    ``v_max_f32 + v_cmp_o_f32 + s_nop 1 + v_cndmask`` chain that
    ``arith.maximumf`` emits for IEEE 754 NaN-propagation semantics.

    Safe for PA softmax: inputs are either finite or -inf (from masking),
    never NaN.  Each call site saves ~3 instructions + a 1-cycle VCC→VALU
    s_nop hazard in the cross-warp max chain.
    """
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))


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
            k2 = _global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
            k2_words = fx.Vector(k2)
            k_flat.append(k2_words[0])
            k_flat.append(k2_words[1])

    return k_flat


def _unflatten_k(k_flat, qkhe_loop: int = 2):
    return [[k_flat[td * (qkhe_loop * 2) + j] for j in range(qkhe_loop * 2)] for td in range(TLOOP)]


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
    q_lanes_per_head,
):
    # bf16/f16 + in-kernel query_scale path.  Each lane owns 8 Q elements,
    # loaded as 2 × vec_width=4 buffer loads (4 bf16/f16 elems per load = 8 B,
    # element offset += 4 per iter).  After FP8 packing each load produces
    # one i32 word, so the per-lane store is `vec<2, i32>` = 8 B = 1 i64.
    q_load_lane = lane16id
    if const_expr(q_lanes_per_head < MFMA_N):
        q_load_lane = arith.select(lane16id < fx.Int32(q_lanes_per_head), lane16id, fx.Int32(0))
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
        local_max = _maxnumf(local_max, chunk_max)

    for sh in [8, 4, 2, 1]:
        local_max = _maxnumf(local_max, dpp_utils.dpp_xor_f32(local_max, sh))
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
    inv_sum = _rcp_f32(safe_sum)
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
    head_size: int = 128,
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
                v_i64x2 = _global_load_i64x2(v_global_ptr, va_byte)
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
                v_max_warp = _maxnumf(v_max_warp, fx.Vector(vs).reduce("max"))
            for sh in [32, 16]:
                v_max_warp = _maxnumf(v_max_warp, v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
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
            qk_max = _maxnumf(qk_max, fx.Vector(logits_vec).reduce("max"))
        for sh in [32, 16]:
            qk_max = _maxnumf(qk_max, qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
        vector.store(
            fx.Vector.from_elements([qk_max], dtype=fx.Float32),
            softmax_lds_f32,
            [sm_max_off],
        )

        if const_expr(cache_scale_vecs and per_token_kv):
            return d_out, v_scale_vecs
        return d_out

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs, v_scale_vecs):
        partition_max = neg_inf
        partition_sum = zero_f
        max_vec = fx.Vector(vector.load_op(T.f32x4, softmax_lds_f32, [sm_rd_max_offs[0]]))
        for w in range_constexpr(NUM_WARPS):
            partition_max = _maxnumf(partition_max, max_vec[w])

        new_rmax = _maxnumf(rmax, partition_max)
        safe_eff_max = arith.select(partition_max > neg_inf, new_rmax, zero_f) if const_expr(needs_mask) else new_rmax
        local_exp_sum = zero_f
        for td in range_constexpr(TLOOP):
            diff_vec = fx.Vector(d_out[td]) - vector.broadcast(T.f32x4, arith.unwrap(safe_eff_max))
            p_vec = _exp2_f32_fast(diff_vec * vector.broadcast(T.f32x4, arith.unwrap(fx.Float32(LOG2E))))
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
                _exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value()),
                zero_f,
            )
        else:
            accum_scale = _exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value())

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
                v_max_global = _maxnumf(v_max_global, w_vmax)
            v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX).ir_value()
            v_max_safe_scaled = v_max_scaled + fx.Float32(1e-8 / FP8_MAX).ir_value()
            norm_factor = _rcp_f32(v_max_safe_scaled)
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
        _load_kv_scale_scalars,
        _load_v_and_scales,
        _store_vmax_warp,
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
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
    block_size: int = KV_BLOCK_SIZE,
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
    arch = get_hip_arch()
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
    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"pa_ps_smem_bs{_block_size}")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL
    scale_off = softmax_off + LDS_SOFTMAX_TOTAL
    allocator.ptr += LDS_SCALE_TOTAL
    # Phys-block staging LDS for the small-block path (cross-warp visibility of
    # the per-warp page indices so V can read all blocks of a partition).
    bt_off = scale_off + LDS_SCALE_TOTAL
    if _is_small_block:
        allocator.ptr += NUM_WARPS * TLOOP * 4

    # ── @flyc.kernel ─────────────────────────────────────────────────
    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_metadata_kenrel(
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
        partition_indptr_ptr: fx.Tensor,  # [num_seqs + 1] int32 — prefix sum of partitions per seq
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
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        k_global_ptr = _extract_global_ptr(key_cache_ptr)
        v_global_ptr = _extract_global_ptr(value_cache_ptr)
        po_rsrc = buffer_ops.create_buffer_resource(partial_out_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_lse_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        wi_rsrc = buffer_ops.create_buffer_resource(work_indptr_ptr, max_size=True)
        winfo_rsrc = buffer_ops.create_buffer_resource(work_info_ptr, max_size=True)
        kpi_rsrc = buffer_ops.create_buffer_resource(kv_page_indices_ptr, max_size=True)
        kvindptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr_ptr, max_size=True)
        pip_rsrc = buffer_ops.create_buffer_resource(partition_indptr_ptr, max_size=True)
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
        bt_lds_i32 = None
        if const_expr(_is_small_block):
            bt_lds_i32 = SmemPtr(smem_base, bt_off, T.i32, shape=(NUM_WARPS * TLOOP,)).get()

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

        # ════════════════════════════════════════════════════════════
        # Outer work loop — iterate over assigned work items
        # Each work item = one (batch, kv_head_range, kv_page_range)
        # ════════════════════════════════════════════════════════════
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
            kv_h = _udiv_const(q_head_start, query_group_size)

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
                head_size=_HEAD,
                qkhe_loop=_QKHELOOP,
                vhe_loop=_VHELOOP,
            )

            # ════════════════════════════════════════════════════════
            # Inner KV loop — one CTA processes one 256-token sub-tile
            # across all 1024-token physical blocks in the work item.
            # Below: MTP groups loop is nested INSIDE the KV loop so that
            # K and V are loaded once per physical block and reused across
            # all MTP groups.  Q is hoisted out of the KV loop (loaded once
            # per work item, kept in registers).
            # ════════════════════════════════════════════════════════
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
                        fx.Vector.from_elements([phys_vec], dtype=fx.Int32).store(bt_lds_i32, [fx.Index(warp_id)])
                    else:
                        phys_vec.store(bt_lds_i32, [fx.Index(warp_id * arith.constant(TLOOP, type=T.i32))])

            def _meta_load_v_phys_from_lds():
                v_phys_blocks = []
                if const_expr(_block_size == 64):
                    phys_block_vec = fx.Vector.load(T.vec(VTLOOP, T.i32), bt_lds_i32, [fx.Index(0)])
                    for vt in range_constexpr(VTLOOP):
                        v_phys_blocks.append(phys_block_vec[vt])
                else:
                    for vt in range_constexpr(VTLOOP):
                        bt_lds_off = arith.constant(vt * TLOOP, type=T.i32) + rowid
                        v_phys_blocks.append(fx.Vector.load(T.vec(1, T.i32), bt_lds_i32, [fx.Index(bt_lds_off)])[0])
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
                    logits_lds_i32,
                    logits_lds_i64,
                    softmax_lds_f32,
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
                    kpi_rsrc, kv_page_base + _udiv_const(local_last_part, _parts_per_block), vec_width=1, dtype=T.i32
                )
                _first_tile_tok = _urem_const(local_last_part, _parts_per_block) * c_cps
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

            # ════════════════════════════════════════════════════════
            # KV outer loop — iterate over physical blocks in this work
            # item.  MTP processing happens INSIDE the loop body so that
            # K and V are loaded once per block and reused across all
            # MTP groups.
            # ════════════════════════════════════════════════════════
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

                k_ops = _unflatten_k(k_flat, qkhe_loop=_QKHELOOP)
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
                        kpi_rsrc, kv_page_base + _udiv_const(lp, _parts_per_block), vec_width=1, dtype=T.i32
                    )
                    tile_token_offset = _urem_const(lp, _parts_per_block) * c_cps
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
                        kpi_rsrc, kv_page_base + _udiv_const(next_lp, _parts_per_block), vec_width=1, dtype=T.i32
                    )
                    next_tile_tok = _urem_const(next_lp, _parts_per_block) * c_cps
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
        num_sm,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
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
    from kernels.pa_metadata import get_pa_metadata_info_v1, get_pa_metadata_v1

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
        return _cdiv(window_token_count - 1, context_partition_size) + 1

    props = torch.cuda.get_device_properties(torch.device("cuda"))
    # Reference uses occupancy = 2 (see `get_occupancy()` in the Gluon module).
    occupancy = 2
    num_sm = props.multi_processor_count * occupancy
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = _cdiv(num_sm, denom) * split_kv_blocks
    return max(4, min(n, 8))


# Small block_size (16/64) is routed through the load-balanced worklist
# (metadata) path: `compile_pa_decode_metadata` gathers 256//block_size physical
# pages per 256-token partition, for both per-tensor and per-token KV quant.
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
    with ``_unflatten_k`` and downstream MFMA invocations.
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
                k2 = _global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
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
                k2 = _global_load_i64x2(k_global_ptr, ka_dw * fx.Int64(4))
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
            v_i64x2 = _global_load_i64x2(v_global_ptr, va_byte)
            vhe_data.append(v_i64x2)
        v_results.append(vhe_data)
    return v_results


@functools.lru_cache(maxsize=256)
def compile_pa_decode_ps(
    *,
    block_size: int,
    max_context_partition_num: int,
    softmax_scale: float = None,
    trans_v: bool = True,
    query_group_size: int = 16,
    per_token_kv: bool = False,
    query_length: int = 1,
    query_input_dtype: str = "bf16",
    head_dim: int = 128,
):
    """Compile the small-block partition kernel.  See module-level comment."""
    if block_size not in _PA_DECODE_PS_SMALL_BLOCK_SIZES:
        raise ValueError(
            f"compile_pa_decode_ps: unsupported block_size={block_size}; "
            f"expected one of {_PA_DECODE_PS_SMALL_BLOCK_SIZES}."
        )
    if query_input_dtype not in ("bf16", "f16"):
        raise ValueError("compile_pa_decode_ps currently expects bf16/f16 query inputs.")
    if not trans_v:
        raise NotImplementedError("compile_pa_decode_ps: trans_v=False not yet supported.")
    if head_dim % QKHE_PER_FETCH != 0 or head_dim % (MFMA_N * NUM_WARPS) != 0 or head_dim % Q_ELEMS_PER_LANE != 0:
        raise ValueError(f"Unsupported head_dim={head_dim}; must be a multiple of {MFMA_N * NUM_WARPS}.")
    _HEAD = head_dim
    _QKHELOOP = head_dim // QKHE_PER_FETCH
    _VHELOOP = head_dim // MFMA_N // NUM_WARPS
    _Q_LANES_PER_HEAD = head_dim // Q_ELEMS_PER_LANE
    _N_K_h = TLOOP * _QKHELOOP * 2
    _N_V_FLAT_h = 2 * VTLOOP * _VHELOOP

    arch = get_hip_arch()
    query_load_is_bf16 = query_input_dtype == "bf16"
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    _softmax_scale = float(softmax_scale)
    _block_size = block_size
    _blocks_per_partition = KV_COMPUTE_BLOCK // _block_size

    _mtp_groups = max(1, math.ceil(query_length * query_group_size / 16))

    # LDS allocation — same layout as compile_pa_decode_metadata's small-block
    # path.  per_token_kv adds a cross-warp v_scale_max region (appended to the
    # softmax block) and a K/V per-token scale staging region.
    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    LDS_SCALE_TOTAL = LDS_SCALE_BYTES if const_expr(per_token_kv) else 0
    # Unique global symbol per compile to avoid module-level symbol clashes
    # when multiple compiled artifacts are loaded into the same GPU context.
    _smem_sym_name = (
        f"pa_ps_smallblk_smem_bs{block_size}_ql{query_length}"
        f"_qgs{query_group_size}_tv{int(trans_v)}_qd{query_input_dtype}"
        f"_ptkv{int(per_token_kv)}"
    )
    allocator = SmemAllocator(None, arch=arch, global_sym_name=_smem_sym_name)
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL
    # K/V per-token scale staging LDS (per_token_kv only).
    scale_off_ps = softmax_off + LDS_SOFTMAX_TOTAL
    allocator.ptr += LDS_SCALE_TOTAL
    bt_off = scale_off_ps + LDS_SCALE_TOTAL
    allocator.ptr += NUM_WARPS * TLOOP * 4

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_ps_kernel(
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        tmp_out_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        key_cache_ptr: fx.Tensor,
        value_cache_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
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
        stride_bt_seq: Int32,
        # Per-token K/V scale strides (per_token_kv only), metadata layout
        # `[num_blocks, num_kv_heads, block_size]`:
        #   stride_ks_block = num_kv_heads * block_size
        #   stride_ks_head  = block_size
        # Both 0 for per-tensor.
        stride_ks_block: Int32,
        stride_ks_head: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        batch_idx = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        partition_idx = fx.Int32(gpu.block_id("z"))

        cl_global_ptr = _extract_global_ptr(context_lengths_ptr)
        context_len = _global_load_i32(cl_global_ptr, batch_idx)

        lane16id = tid & fx.Int32(15)
        rowid = (tid >> fx.Int32(4)) & fx.Int32(3)
        warp_id = tid >> fx.Int32(6)

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_global_ptr = _extract_global_ptr(key_cache_ptr)
        v_global_ptr = _extract_global_ptr(value_cache_ptr)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=False)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        to_rsrc = buffer_ops.create_buffer_resource(tmp_out_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)

        q_scale_val = arith.constant(1.0, type=T.f32)
        # Per-tensor K/V scales are loaded from index 0; per_token_kv uses
        # per-token scales staged to LDS (see _stage_small_block_kv_scales).
        if const_expr(per_token_kv):
            k_scale_val = arith.constant(1.0, type=T.f32)
            v_scale_val = arith.constant(1.0, type=T.f32)
        else:
            k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
            v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_TOTAL // 4,)).get()
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()
        bt_lds_i32 = SmemPtr(smem_base, bt_off, T.i32, shape=(NUM_WARPS * TLOOP,)).get()
        if const_expr(per_token_kv):
            scale_lds_f32 = SmemPtr(smem_base, scale_off_ps, T.f32, shape=(LDS_SCALE_BYTES // 4,)).get()
        else:
            scale_lds_f32 = None

        _softmax_scale_const = arith.constant(_softmax_scale, type=T.f32)
        _softmax_q_scale = _softmax_scale_const * q_scale_val
        _scale = _softmax_q_scale * k_scale_val
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_query_group_size = arith.constant(query_group_size, type=T.i32)

        local_qhead_idx = warp_id * arith.constant(4, type=T.i32) + rowid

        (
            _k_tok_thread_base_unused,
            _c_tok_stride_dw_unused,
            _k_he_off_dw_unused,
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

        (
            _load_kv_scale_scalars_unused,
            _load_v_and_scales_unused,
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
            stride_ks_block=arith.constant(0, type=T.i32),
            stride_ks_head=arith.constant(0, type=T.i32),
            softmax_scale_base=_softmax_scale_const,
            softmax_q_scale=_softmax_q_scale,
            k_scale_val=k_scale_val,
            scale=_scale,
            v_scale_val=v_scale_val,
            warp_id=warp_id,
            lane16id=lane16id,
            rowid=rowid,
            k_tok_thread_base=_k_tok_thread_base_unused,
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
            cache_scale_vecs=per_token_kv,
            head_size=_HEAD,
            qkhe_loop=_QKHELOOP,
            vhe_loop=_VHELOOP,
        )

        def _store_partition_results(eqgs_lane, running_sum, running_max, outs_norm):
            for vhe in range_constexpr(_VHELOOP):
                hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * fx.Int32(4)
                to_off = (
                    batch_idx * stride_to_seq
                    + kv_h * stride_to_head
                    + partition_idx * stride_to_part
                    + eqgs_lane * stride_to_group
                    + hs_base
                )
                out_bf16 = fx.Vector(outs_norm[vhe]).to(fx.BFloat16)
                buffer_ops.buffer_store(out_bf16, to_rsrc, to_off)
            es_off = batch_idx * stride_es_seq + kv_h * stride_es_head + partition_idx * stride_es_part + eqgs_lane
            buffer_ops.buffer_store(fx.Float32(running_sum), es_rsrc, es_off)
            buffer_ops.buffer_store(fx.Float32(running_max), ml_rsrc, es_off)

        # Slot covers one or more contiguous 256-token sub-partitions.  The
        # inner scf.for loop walks those sub-partitions with online-softmax
        # loop-carried state, mirroring the Gluon `for sequence_partition_idx`
        # loop in `paged_attention_decode_ps`.
        c_max_parts = arith.constant(max_context_partition_num, type=T.i32)
        num_total_partitions = (context_len + c_cps - fx.Int32(1)) >> fx.Int32(8)
        page_size_partitions = (num_total_partitions + c_max_parts - fx.Int32(1)) // c_max_parts
        local_partition_start = partition_idx * page_size_partitions
        local_partition_end_raw = (partition_idx + fx.Int32(1)) * page_size_partitions
        local_partition_end = arith.select(
            local_partition_end_raw < num_total_partitions,
            local_partition_end_raw,
            num_total_partitions,
        )

        def _unwrap(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        # Pack/unpack loop state.  State is `_mtp_groups` accumulators, each a
        # tuple of (rmax, rsum, outs...), plus the current sub-partition's K
        # and V tiles (k_flat: _N_K i64 scalars, v_flat: 2 * _N_V i64 scalars
        # — each V element is i64x2, flattened to two scalars).  Both K and V
        # are loop-carried so the body can use them while we prefetch the
        # NEXT iteration's K and V (ping-pong).
        state_width = 2 + _VHELOOP

        def _pack_states(states, k_flat, v_flat):
            flat = []
            for st in states:
                rmax, rsum = st[0], st[1]
                outs = [st[2 + vhe] for vhe in range_constexpr(_VHELOOP)]
                flat.extend([_unwrap(rmax), _unwrap(rsum)])
                flat.extend(_unwrap(out) for out in outs)
            flat.extend(_unwrap(v) for v in k_flat)
            flat.extend(_unwrap(v) for v in v_flat)
            return flat

        def _unpack_states(flat):
            base = state_width * _mtp_groups
            states = [
                tuple(flat[state_width * i + j] for j in range_constexpr(state_width))
                for i in range_constexpr(_mtp_groups)
            ]
            k_flat = list(flat[base : base + _N_K_h])
            v_flat = list(flat[base + _N_K_h : base + _N_K_h + _N_V_FLAT_h])
            return states, k_flat, v_flat

        init_states = [
            tuple([NEG_INF, ZERO_F] + [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(_VHELOOP)])
            for _ in range(_mtp_groups)
        ]

        loop_start = fx.Index(arith.unwrap(local_partition_start))
        loop_end = fx.Index(arith.unwrap(local_partition_end))
        loop_step = arith.index(1)
        last_partition_idx = local_partition_end - fx.Int32(1)

        def _ptr8_to_v4i32(ptr8_val):
            """Convert a `ptr addrspace(8)` buffer descriptor to `<4 x i32>`
            via ptrtoint(i128) + bitcast.  Both forms are 128-bit type-puns
            of the same SGPR-resident descriptor — the LLVM backend emits
            zero instructions for this conversion (descriptor stays in
            SGPRs)."""
            from flydsl._mlir import ir as _ir
            from flydsl._mlir.dialects import llvm as _llvm

            i128_ty = _ir.IntegerType.get_signless(128)
            v4i32_ty = _ir.VectorType.get([4], _ir.IntegerType.get_signless(32))
            i128_val = _llvm.ptrtoint(i128_ty, ptr8_val)
            return _llvm.bitcast(v4i32_ty, i128_val)

        bt_rsrc_v4 = _ptr8_to_v4i32(bt_rsrc)

        def _s_buffer_load(soffset_bytes_i32, vec_width: int):
            """Scalar buffer load — emits s_buffer_load_dword[x4] via the LLVM
            intrinsic, returning an SGPR value.  Requires `soffset_bytes_i32`
            to be wave-uniform; the result is shared across all 64 lanes.

            Saves the vmcnt(0) drain + readfirstlane that the VMEM
            buffer_load path imposes (result lands in SGPR directly, frees
            VMEM queue slots for V/K loads)."""
            from flydsl._mlir import ir as _ir
            from flydsl._mlir.dialects import llvm as _llvm
            from flydsl.expr.rocdl import _to_ir as _rocdl_to_ir

            i32_ty = _ir.IntegerType.get_signless(32)
            if const_expr(vec_width == 1):
                result_type = i32_ty
                suffix = "i32"
            elif const_expr(vec_width == 4):
                result_type = _ir.VectorType.get([4], i32_ty)
                suffix = "v4i32"
            else:
                raise ValueError(f"_s_buffer_load: unsupported vec_width={vec_width}")
            cache_policy = arith.constant(0, type=T.i32)
            return _llvm.call_intrinsic(
                result_type,
                f"llvm.amdgcn.s.buffer.load.{suffix}",
                [
                    _rocdl_to_ir(bt_rsrc_v4),
                    _rocdl_to_ir(soffset_bytes_i32),
                    _rocdl_to_ir(cache_policy),
                ],
                [],
                [],
            )

        def _pa_small_block_stage_phys_blocks(partition_block_base):
            # bt offset is wave-uniform (batch_idx and warp_id are constant
            # per wave, partition_block_base is workgroup-uniform).  Use
            # s_buffer_load to route through SMEM cache and land the result
            # in SGPRs directly — eliminates the vmcnt(0) drain (was 25% of
            # all kernel stalls) and the downstream readfirstlane.
            if const_expr(block_size == 64):
                bt_elem_off = batch_idx * stride_bt_seq + partition_block_base + warp_id
                phys_blocks = _s_buffer_load(bt_elem_off * fx.Int32(4), vec_width=1)
            else:
                bt_elem_off = batch_idx * stride_bt_seq + partition_block_base + warp_id * fx.Int32(TLOOP)
                phys_blocks = _s_buffer_load(bt_elem_off * fx.Int32(4), vec_width=TLOOP)
            return phys_blocks

        def _pa_small_block_store_phys_blocks_to_lds(phys_block_vec):
            if (lane16id | rowid) == fx.Int32(0):
                if const_expr(block_size == 64):
                    # block_size=64: `_stage_phys_blocks` returned vec_width=1
                    # → scalar i32, not a Vector.  Wrap in a 1-element
                    # Vector so we can use the LDS `.store(...)` API.
                    # Each warp writes 1 i32 to bt_lds_i32[warp_id];
                    # `_load_v_phys_blocks_from_lds` reads back the 4-elem
                    # vec starting at offset 0.
                    fx.Vector.from_elements([phys_block_vec], dtype=fx.Int32).store(
                        bt_lds_i32,
                        [fx.Index(warp_id)],
                    )
                else:
                    phys_block_vec.store(
                        bt_lds_i32,
                        [fx.Index(warp_id * fx.Int32(TLOOP))],
                    )

        def _pa_small_block_load_v_phys_blocks_from_lds():
            v_phys_blocks = []
            if const_expr(block_size == 64):
                phys_block_vec = fx.Vector.load(T.vec(VTLOOP, T.i32), bt_lds_i32, [fx.Index(0)])
                for vt in range_constexpr(VTLOOP):
                    v_phys_blocks.append(phys_block_vec[vt])
            else:
                for vt in range_constexpr(VTLOOP):
                    bt_lds_off = fx.Int32(vt * TLOOP) + rowid
                    phys_block = fx.Vector.load(T.vec(1, T.i32), bt_lds_i32, [fx.Index(bt_lds_off)])[0]
                    v_phys_blocks.append(phys_block)
            return v_phys_blocks

        # Pre-load the FIRST (== reverse-order start = last partition) sub-
        # partition's block-table entries before Q setup so the dependent K
        # prefetch below does not also pay the table latency.
        # Empty-slot guard: when num_total_partitions < max_context_partition_num,
        # CTAs with partition_idx >= num_total_partitions get
        # local_partition_start >= num_total_partitions and the inner loop runs
        # 0 iters.  But the prologue still issues block-table + K reads using
        # `last_partition_idx`; clamp to 0 so all reads stay in-bounds (the
        # results are unused since the loop never executes).
        _safe_init_partition = arith.select(
            local_partition_start < num_total_partitions,
            last_partition_idx,
            arith.constant(0, type=T.i32),
        )
        first_block_base = _safe_init_partition * fx.Int32(_blocks_per_partition)
        first_phys_blocks = _pa_small_block_stage_phys_blocks(first_block_base)

        # Pre-load Q for every MTP group ONCE before the KV loop.  Each group's
        # q_frags / qi / qhi / qscale are kept in registers across the entire
        # KV loop, so we pay the Q-load cost (global → LDS → registers) exactly
        # once per CTA regardless of how many sub-partitions the slot covers.

        q_frags_per_mtp = []
        qi_per_mtp = []
        qhi_per_mtp = []
        qscale_per_mtp = []
        for _mtp_g in range_constexpr(_mtp_groups):
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
            qi_val, qhi_pos, q_frags, query_scale_lane = _finish_mtp_group_q_fragments(
                logits_lds_i32,
                logits_lds_i64,
                softmax_lds_f32,
                mtp_prefetch,
                lane16id,
                rowid,
                local_qhead_idx,
                head_size=_HEAD,
                qkhe_loop=_QKHELOOP,
                q_lanes_per_head=_Q_LANES_PER_HEAD,
            )
            q_frags_per_mtp.append(q_frags)
            qi_per_mtp.append(qi_val)
            qhi_per_mtp.append(qhi_pos)
            qscale_per_mtp.append(query_scale_lane)

        _pa_small_block_store_phys_blocks_to_lds(first_phys_blocks)

        # ── Per-token K/V scale staging (per_token_kv only) ──
        # A 256-token partition spans `_blocks_per_partition` physical pages
        # whose indices are staged in bt_lds_i32.  Each thread stages its own
        # LDS slot ``t`` (= partition-local token) by reading that token's K/V
        # scale from the page it belongs to, using the metadata scale layout
        # `[num_blocks, num_kv_heads, block_size]`.  Mirrors the metadata kernel
        # so the shared `_make_pa_phase_helpers` readers work unchanged.
        def _stage_small_block_kv_scales():
            t = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            part_page = _udiv_const(t, _block_size)
            tok_in_page = _urem_const(t, _block_size)
            phys = fx.Vector.load(T.vec(1, T.i32), bt_lds_i32, [fx.Index(part_page)])[0]
            scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + tok_in_page
            k_scale_scalar = buffer_ops.buffer_load(ks_rsrc, scale_idx, vec_width=1, dtype=fx.Float32)
            v_scale_scalar = buffer_ops.buffer_load(vs_rsrc, scale_idx, vec_width=1, dtype=fx.Float32)
            fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32).store(scale_lds_f32, [fx.Index(t)])
            fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32).store(
                scale_lds_f32, [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + t)]
            )

        def _load_small_block_scale_vecs():
            k_scale_vecs = []
            v_scale_vecs = []
            for td in range_constexpr(TLOOP):
                row = _kv_tok_thread_base + arith.constant(td * MFMA_N, type=T.i32)
                k_scale_vecs.append(vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(row)]))
                v_scale_vecs.append(
                    vector.load_op(T.f32x4, scale_lds_f32, [fx.Index(fx.Int32(LDS_SCALE_V_OFFSET) + row)])
                )
            return k_scale_vecs, v_scale_vecs

        # Pre-load the FIRST sub-partition's K so the loop body can issue the
        # next sub-partition's K prefetch in parallel with the current K's QK
        # MFMA.  For empty slots (loop_start == loop_end), this k_flat0 is
        # computed using local_partition_start but never used because the
        # loop runs 0 iterations — the block_table buffer_load is bounded so
        # any OOB lookup safely returns 0 (→ block 0, then masked out by the
        # softmax causal bound).
        k_flat0 = _pa_small_block_load_k_flat(
            k_global_ptr,
            kv_h,
            stride_k_block,
            stride_k_head,
            lane16id,
            rowid,
            block_size=_block_size,
            phys_blocks=first_phys_blocks,
            qkhe_loop=_QKHELOOP,
        )
        gpu.barrier()
        # ── Prologue V load ──
        # V is cross-iter prefetched (ping-pong with K).  Issue iter 0's V
        # load here so the loop body can issue iter N+1's V at the END of
        # iter N alongside K, both hidden behind the next iter's QK MFMA.
        # `_pa_small_block_load_v_phys_blocks_from_lds` reads the LDS-staged
        # first_phys_blocks written above; the barrier guarantees visibility.
        _v_phys_blocks0 = _pa_small_block_load_v_phys_blocks_from_lds()
        _v_results0 = _pa_small_block_load_v_trans(
            v_global_ptr,
            kv_h,
            stride_v_block,
            stride_v_head,
            warp_id,
            lane16id,
            rowid,
            _v_phys_blocks0,
            block_size=_block_size,
            head_size=_HEAD,
            vhe_loop=_VHELOOP,
        )
        v_flat0 = _flatten_v_results(_v_results0, vhe_loop=_VHELOOP)
        # Note: no runtime `if _is_valid:` wrapping this loop.  See earlier
        # commit log — the `ReplaceIfWithDispatch` rewriter copies the if-body
        # into a synthetic Python function, and since the body contains
        # `ast.Yield` from this scf.for, that function becomes a generator
        # (returns a generator object without executing) — leaving the scf.if
        # then-region empty.  Running the loop unconditionally is the correct
        # workaround; empty slots iterate 0 times and yield the init state
        # (NEG_INF, 0, 0, 0), which is the right empty-slot semantics.
        for sub_part_ib, state in range(
            loop_start,
            loop_end,
            loop_step,
            init=_pack_states(init_states, k_flat0, v_flat0),
        ):
            cur_states, k_flat, v_flat = _unpack_states(state)
            # Reverse iteration: scf.for walks sub_part_ib forward over
            # [local_partition_start, local_partition_end); remap to walk
            # sub_part_i32 from last_partition_idx down to local_partition_start
            # so the sink-prone partition 0 is processed last.
            _sub_raw_i32 = arith.index_cast(T.i32, sub_part_ib)
            sub_part_i32 = last_partition_idx - (_sub_raw_i32 - local_partition_start)
            sub_token_start = sub_part_i32 * c_cps

            # Both K and V come from the loop-carried state (prefetched at the
            # END of the previous iteration).  K's VMEM latency overlaps prev
            # iter's PV MFMA; V's latency overlaps the entire next iter QK +
            # softmax compute before PV consumes it.
            k_ops = _unflatten_k(k_flat, qkhe_loop=_QKHELOOP)
            v_results = _unflatten_v_results(v_flat, vhe_loop=_VHELOOP)

            # ── Per-token K/V scale staging (per_token_kv only) ──
            # bt_lds_i32 holds this partition's physical pages (staged in the
            # prologue / previous iter's tail, barrier-visible).  Stage the
            # per-token K/V scales to LDS once per partition and read the cached
            # f32x4 vecs, reused across all MTP groups.
            if const_expr(per_token_kv):
                _stage_small_block_kv_scales()
                gpu.barrier()
                k_scale_vecs, v_scale_vecs = _load_small_block_scale_vecs()

            # Compute the NEXT sub-partition's K base address (clamped to
            # local_partition_start so the prefetch on the final loop
            # iteration doesn't walk before the block_table window —
            # k_next_flat is yielded out but never consumed since the loop
            # terminates).  Reverse iteration: next == sub_part_i32 - 1.
            next_part_i32 = sub_part_i32 - fx.Int32(1)
            next_safe_part = arith.select(next_part_i32 >= local_partition_start, next_part_i32, local_partition_start)
            next_block_base = next_safe_part * fx.Int32(_blocks_per_partition)

            new_states = []
            k_next_flat = None
            for _mtp_g in range_constexpr(_mtp_groups):
                state = cur_states[_mtp_g]
                rmax, rsum = state[0], state[1]
                outs = [state[2 + vhe] for vhe in range_constexpr(_VHELOOP)]
                causal_bound = context_len + arith.constant(1 - query_length, type=T.i32) + qi_per_mtp[_mtp_g]

                if const_expr(per_token_kv):
                    d_out, v_scales = _qk_and_intra_softmax(
                        k_ops,
                        sub_token_start,
                        q_frags_per_mtp[_mtp_g],
                        causal_bound,
                        query_scale_lane=qscale_per_mtp[_mtp_g],
                        preloaded_scales=(k_scale_vecs, v_scale_vecs),
                    )
                else:
                    d_out = _qk_and_intra_softmax(
                        k_ops,
                        sub_token_start,
                        q_frags_per_mtp[_mtp_g],
                        causal_bound,
                        query_scale_lane=qscale_per_mtp[_mtp_g],
                    )
                    v_scales = None

                if const_expr(_mtp_g == _mtp_groups - 1):
                    next_phys_blocks = _pa_small_block_stage_phys_blocks(next_block_base)

                # per_token_kv needs the cross-warp v_scale_max staged to LDS so
                # _cross_warp_softmax_and_prob_pack can read it for norm_factor.
                if const_expr(per_token_kv):
                    _store_vmax_warp(sub_token_start, seq_end=context_len, v_scale_vecs=v_scales)

                gpu.barrier()

                rmax, rsum, outs, v_correction = _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs, v_scales)

                # Issue the next sub-partition's K prefetch on the LAST MTP
                # iter, after cross_warp_softmax_and_prob_pack but BEFORE
                # _pv_mfma — same hoist as in pa_decode_metadata_kenrel's
                # _process_block_split.  This lets the K VMEM load latency
                # overlap with the upcoming PV MFMA compute.
                if const_expr(_mtp_g == _mtp_groups - 1):
                    _pa_small_block_store_phys_blocks_to_lds(next_phys_blocks)
                    k_next_flat = _pa_small_block_load_k_flat(
                        k_global_ptr,
                        kv_h,
                        stride_k_block,
                        stride_k_head,
                        lane16id,
                        rowid,
                        block_size=_block_size,
                        phys_blocks=next_phys_blocks,
                        qkhe_loop=_QKHELOOP,
                    )
                gpu.barrier()
                outs = _pv_mfma(v_results, outs, v_correction)
                new_states.append(tuple([rmax, rsum] + outs))

            # ── Cross-iter V prefetch (ping-pong) ──
            # Issue NEXT iter's V load AFTER PV MFMA: the current iter's V
            # vgprs are now consumed and can be reused.  V phys_blocks come
            # from the LDS-staged `next_phys_blocks` written above (the
            # barrier after K prefetch ensures cross-warp visibility).  The
            # V VMEM latency is hidden behind next iter's QK MFMA + softmax.
            _v_phys_blocks_next = _pa_small_block_load_v_phys_blocks_from_lds()
            _v_next_results = _pa_small_block_load_v_trans(
                v_global_ptr,
                kv_h,
                stride_v_block,
                stride_v_head,
                warp_id,
                lane16id,
                rowid,
                _v_phys_blocks_next,
                block_size=_block_size,
                head_size=_HEAD,
                vhe_loop=_VHELOOP,
            )
            v_next_flat = _flatten_v_results(_v_next_results, vhe_loop=_VHELOOP)

            results = yield _pack_states(new_states, k_next_flat, v_next_flat)

        # Normalize and store one output slot per MTP group.
        final_states, _final_k_flat, _final_v_flat = _unpack_states(results)
        for _mtp_g in range_constexpr(_mtp_groups):
            final_state = final_states[_mtp_g]
            rmax_raw, rsum_raw = final_state[0], final_state[1]
            outs_raw = [final_state[2 + vhe] for vhe in range_constexpr(_VHELOOP)]
            running_max = fx.Float32(rmax_raw)
            running_sum = fx.Float32(rsum_raw)
            outs = [fx.Vector(out_raw) for out_raw in outs_raw]
            outs_norm = _normalize_pa_output(running_sum, outs, ZERO_F)
            eqgs_lane = qi_per_mtp[_mtp_g] * c_query_group_size + qhi_per_mtp[_mtp_g]
            _store_partition_results(eqgs_lane, running_sum, running_max, outs_norm)

    @flyc.jit
    def launch_pa_decode_ps_small_block(
        exp_sums: fx.Tensor,
        max_logits: fx.Tensor,
        tmp_out: fx.Tensor,
        query: fx.Tensor,
        key_cache: fx.Tensor,
        value_cache: fx.Tensor,
        block_tables: fx.Tensor,
        context_lengths: fx.Tensor,
        key_scale: fx.Tensor,
        value_scale: fx.Tensor,
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
        pa_decode_ps_kernel(
            exp_sums,
            max_logits,
            tmp_out,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
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
            s_bt_seq,
            s_ks_block,
            s_ks_head,
        ).launch(grid=(gx, gy, gz), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        "launch": launch_pa_decode_ps_small_block,
        "kernel": pa_decode_ps_kernel,
        "allocator": allocator,
        "mtp_groups": _mtp_groups,
    }


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

    # ── small-block (block_size 16/64) → grid partition kernel + reduce ──
    # Key cache shape is [num_blocks, num_kv_heads, head_size // 16, block_size, 16].
    block_size = key_cache.shape[-2]
    if block_size in _PA_DECODE_PS_SMALL_BLOCK_SIZES:
        if block_tables is None:
            raise ValueError(
                f"pa_decode_ps_launch: block_size={block_size} requires `block_tables` "
                "(per-sequence physical block index table)."
            )
        batch_size = context_lengths.shape[0]
        head_size = query.shape[-1]
        eqgs = query_length * query_group_size
        context_partition_size = KV_COMPUTE_BLOCK
        blocks_per_partition = context_partition_size // block_size
        if max_context_partition_num == 0:
            max_context_partition_num = get_recommended_splits(
                batch_size,
                num_kv_heads,
                split_kv_blocks=blocks_per_partition,
            )
        if is_graph_capturing and (exp_sums is None or max_logits is None or temporary_output is None):
            raise ValueError(
                "CUDA graph capture requires preallocated `exp_sums`, `max_logits`, "
                "and `temporary_output` for the small-block PS path."
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
        compiled_small = compile_pa_decode_ps(
            block_size=block_size,
            max_context_partition_num=max_context_partition_num,
            softmax_scale=softmax_scale,
            trans_v=trans_v,
            query_group_size=query_group_size,
            per_token_kv=per_token_kv,
            query_length=query_length,
            query_input_dtype=query_input_dtype,
            head_dim=int(head_size),
        )
        output_5d = output.reshape(batch_size, query_length, num_kv_heads, query_group_size, head_size)
        compiled_small["launch"](
            exp_sums,
            max_logits,
            temporary_output,
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
            block_tables.stride(0),
            stride_ks_block,
            stride_ks_head,
            batch_size,
            num_kv_heads,
            max_context_partition_num,
            s,
        )
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
        partition_indptr,
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
        partial_output=partial_output[query_length:],
        partial_lse=partial_lse[query_length:],
        reduce_indptr=metadata["reduce_indptr"],
        reduce_final_map=metadata["reduce_final_map"],
        reduce_partial_map=metadata["reduce_partial_map"],
        max_seqlen_q=query_length,
        num_kv_splits=0,
        final_output=output,
        final_lse=None,
    )

    return "ps_split_reduce"
