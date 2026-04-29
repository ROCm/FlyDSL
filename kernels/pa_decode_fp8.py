"""FlyDSL Paged Attention Decode with Persistent Scheduling — FP8.

Extends pa_decode_sw_fp8.py with persistent scheduling (PS) mode:
- Grid = (num_SM, 1, 4) so each CTA handles one 256-token sub-tile of a 1024-token KV page
- Outer work loop iterates over pre-computed worklist from get_pa_metadata_v1
- Inner KV loop iterates pages from kv_page_indices instead of block_tables
- Supports split-reduce for load balancing across CUs

Requires: aiter's get_pa_metadata_v1 (module_pa_metadata.so)
"""

from __future__ import annotations
import math
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, range_constexpr, const_expr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.env import runtime as flydsl_runtime_env
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir.dialects import scf

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

FP8_MAX = 240.0
LOG2E = 1.4426950408889634

# Number of loop-carried K values (i64)
_N_K = TLOOP * QKHELOOP * 2  # 16
# Number of loop-carried V values (i64)
_N_V = VHELOOP * VTLOOP * 2  # 16

# Tiles per block (1024 tokens / 256 tokens per tile = 4, matches SP3 kNumBlockTiles)
TILES_PER_BLOCK = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK  # 4

# Sliding-window decode can fuse two 16-wide execution subgroups into one
# 24-wide logical MTP group for the qgs=6, mtp=4 case.
SW_MTP_EXEC_GROUP_SIZE = MFMA_N
SW_LOGICAL_MTP_GROUP_SIZE = 24
SW_MAX_MTP_SUBGROUPS_PER_GROUP = 2

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


def _get_sw_mtp_group_count(query_length: int, query_group_size: int) -> int:
    return _cdiv(query_length * query_group_size, SW_LOGICAL_MTP_GROUP_SIZE)


def _get_sw_mtp_subgroup_count(
    query_length: int,
    query_group_size: int,
    mtp_group_idx: int,
) -> int:
    remaining_pairs = query_length * query_group_size - mtp_group_idx * SW_LOGICAL_MTP_GROUP_SIZE
    if remaining_pairs <= 0:
        return 0
    return min(
        SW_MAX_MTP_SUBGROUPS_PER_GROUP,
        _cdiv(remaining_pairs, SW_MTP_EXEC_GROUP_SIZE),
    )


def _get_sw_mtp_pair_offset(mtp_group_idx: int, mtp_subgroup_idx: int = 0) -> int:
    return mtp_group_idx * SW_LOGICAL_MTP_GROUP_SIZE + mtp_subgroup_idx * SW_MTP_EXEC_GROUP_SIZE


def _pack_i32_pair_to_i64(a_i32, b_i32):
    return fx.Vector.from_elements([a_i32, b_i32], dtype=fx.Int32).bitcast(fx.Int64)[0]


def _widen_nonnegative_i32_to_i64(value):
    value = arith.unwrap(value)
    if const_expr(value.type == T.i64):
        return value
    return arith.ArithValue(value).extui(T.i64)


def _compute_block_base_dw(phys_block, block_stride, head_offset):
    # Keep the final per-load offset in signed i32 range by chunking the buffer
    # resource base, while still computing the block base in i64 first.
    phys_block_i64 = _widen_nonnegative_i32_to_i64(phys_block)
    block_stride_i64 = _widen_nonnegative_i32_to_i64(block_stride)
    head_offset_i64 = _widen_nonnegative_i32_to_i64(head_offset)
    base_elem_i64 = phys_block_i64 * block_stride_i64 + head_offset_i64
    base_dw_i64 = base_elem_i64 // fx.Int64(4)
    return arith.ArithValue(base_dw_i64).trunci(T.i32)


def _make_shifted_buffer_resource(memref_val, byte_offset_i64, num_records_bytes_i64):
    return buffer_ops.create_buffer_resource(
        memref_val,
        max_size=True,
        num_records_bytes=num_records_bytes_i64,
        base_byte_offset=byte_offset_i64,
    )


def _chunk_buffer_resource_for_block(memref_val, phys_block, block_stride):
    # `buffer_load` multiplies the i32 logical offset by element bytes before
    # issuing the raw buffer instruction, so each descriptor window must stay
    # within signed-i32 byte reach.
    max_chunk_bytes_i64 = fx.Int64(0x7FFF0000)
    one_i64 = fx.Int64(1)
    phys_block_i64 = _widen_nonnegative_i32_to_i64(phys_block)
    block_stride_i64 = _widen_nonnegative_i32_to_i64(block_stride)
    chunk_blocks_i64 = arith.select(
        block_stride_i64 < max_chunk_bytes_i64,
        max_chunk_bytes_i64 // block_stride_i64,
        one_i64,
    )
    chunk_span_bytes_i64 = chunk_blocks_i64 * block_stride_i64
    chunk_idx_i64 = phys_block_i64 // chunk_blocks_i64
    local_phys_block_i64 = phys_block_i64 % chunk_blocks_i64
    chunk_byte_offset_i64 = chunk_idx_i64 * chunk_span_bytes_i64
    rsrc = _make_shifted_buffer_resource(memref_val, chunk_byte_offset_i64, chunk_span_bytes_i64)
    local_phys_block_i32 = arith.ArithValue(local_phys_block_i64).trunci(T.i32)
    return rsrc, local_phys_block_i32


def _load_q_words(q_rsrc, q_base_elem, lane16id, *, query_packed_fp8, query_load_is_bf16):
    q_elem = q_base_elem + lane16id * fx.Int32(FP8_ELEMS_16B)
    if const_expr(query_packed_fp8):
        q_vec = buffer_ops.buffer_load(
            q_rsrc,
            q_elem // fx.Int32(4),
            vec_width=4,
            dtype=T.i32,
        )
        q_words = fx.Vector(q_vec)
        return [q_words[i] for i in range(4)]

    q_words = []
    for qwi in range_constexpr(4):
        q_src = buffer_ops.buffer_load(
            q_rsrc,
            q_elem + fx.Int32(qwi * 4),
            vec_width=4,
            dtype=T.bf16 if const_expr(query_load_is_bf16) else T.f16,
        )
        q_f32 = fx.Vector(q_src).to(fx.Float32)
        p0 = q_f32[0]
        p1 = q_f32[1]
        p2 = q_f32[2]
        p3 = q_f32[3]
        lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, fx.Int32(0), False)
        pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
        q_words.append(pk)
    return q_words


def _load_k_flat(
    k_rsrc,
    k_block_base_dw,
    tile_token_offset_i32,
    k_tok_thread_base,
    c_tok_stride_dw,
    k_he_off_dw,
):
    k_flat = []
    tile_tok_base = tile_token_offset_i32 + k_tok_thread_base

    for td in range_constexpr(TLOOP):
        kbo = tile_tok_base + fx.Int32(td * MFMA_N)
        kbo_dw = k_block_base_dw + kbo * c_tok_stride_dw
        for qkhe in range_constexpr(QKHELOOP):
            ka_dw = kbo_dw + k_he_off_dw[qkhe]
            k4 = buffer_ops.buffer_load(k_rsrc, ka_dw, vec_width=4, dtype=T.i32)
            k4_words = fx.Vector(k4)
            k_flat.append(_pack_i32_pair_to_i64(k4_words[0], k4_words[1]))
            k_flat.append(_pack_i32_pair_to_i64(k4_words[2], k4_words[3]))
    return k_flat


def _unflatten_k(k_flat):
    return [[k_flat[td * (QKHELOOP * 2) + j] for j in range(QKHELOOP * 2)] for td in range(TLOOP)]


def _build_pa_thread_invariants(
    warp_id,
    lane16id,
    rowid,
    *,
    c_four,
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

    kv_tok_thread_base = warp_id * c_tokens_per_warp + rowid * c_four
    rowid_8x8 = rowid // fx.Int32(2)
    offset_in_slot = rowid % fx.Int32(2)
    prob_wr_thread_base = (
        warp_id * fx.Int32(4 * MFMA_N * PROB_ROW_STRIDE_BYTES)
        + lane16id * fx.Int32(PROB_ROW_STRIDE_BYTES)
        + rowid_8x8 * fx.Int32(8)
        + offset_in_slot * c_four
    )
    pv_prob_read_base = rowid * fx.Int32(MFMA_N * PROB_ROW_STRIDE_BYTES) + lane16id * fx.Int32(PROB_ROW_STRIDE_BYTES)

    sm_max_off = arith.index_cast(T.index, warp_id * c_mfma_n + lane16id)
    sm_sum_off = arith.index_cast(
        T.index,
        fx.Int32(NUM_WARPS * MFMA_N) + warp_id * c_mfma_n + lane16id,
    )
    sm_rd_max_offs = [arith.index_cast(T.index, fx.Int32(w * MFMA_N) + lane16id) for w in range(NUM_WARPS)]
    sm_rd_sum_offs = [
        arith.index_cast(
            T.index,
            fx.Int32(NUM_WARPS * MFMA_N + w * MFMA_N) + lane16id,
        )
        for w in range(NUM_WARPS)
    ]

    sm_vmax_wr_off = None
    sm_vmax_rd_offs = None
    if const_expr(per_token_kv):
        sm_vmax_wr_off = arith.index_cast(
            T.index,
            fx.Int32(2 * NUM_WARPS * MFMA_N) + warp_id * c_mfma_n + lane16id,
        )
        sm_vmax_rd_offs = [
            arith.index_cast(
                T.index,
                fx.Int32(2 * NUM_WARPS * MFMA_N + w * MFMA_N) + lane16id,
            )
            for w in range(NUM_WARPS)
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
    c_query_group_size = fx.Int32(query_group_size)
    lane_pair_raw = lane16id + fx.Int32(g_off)
    c_total_pairs = fx.Int32(query_length * query_group_size)
    c_pair_max = fx.Int32(query_length * query_group_size - 1)
    c_ql_m1 = fx.Int32(query_length - 1)

    lane_pair = arith.select(lane_pair_raw < c_total_pairs, lane_pair_raw, c_pair_max)
    qi_raw = lane_pair // c_query_group_size
    qi_val = arith.select(qi_raw < c_ql_m1, qi_raw, c_ql_m1)
    qhi_pos = lane_pair % c_query_group_size

    lqh_pair_raw = local_qhead_idx + fx.Int32(g_off)
    lqh_pair = arith.select(lqh_pair_raw < c_total_pairs, lqh_pair_raw, c_pair_max)
    lqi_raw = lqh_pair // c_query_group_size
    qi_for_q = arith.select(lqi_raw < c_ql_m1, lqi_raw, c_ql_m1)
    local_qhead_idx_for_q = lqh_pair % c_query_group_size
    return qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q


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
    c_query_group_size = fx.Int32(query_group_size)
    lane_pair_raw = lane16id + fx.Int32(g_off)
    c_total_pairs = fx.Int32(query_length * query_group_size)
    c_pair_max = fx.Int32(query_length * query_group_size - 1)
    c_ql_m1 = fx.Int32(query_length - 1)

    lane_pair = arith.select(lane_pair_raw < c_total_pairs, lane_pair_raw, c_pair_max)
    qi_raw = lane_pair // c_query_group_size
    qi_val = arith.select(qi_raw < c_ql_m1, qi_raw, c_ql_m1)
    qhi_pos = lane_pair % c_query_group_size

    lqh_pair_raw = local_qhead_idx + fx.Int32(g_off)
    lqh_pair = arith.select(lqh_pair_raw < c_total_pairs, lqh_pair_raw, c_pair_max)
    lqi_raw = lqh_pair // c_query_group_size
    qi_for_q = arith.select(lqi_raw < c_ql_m1, lqi_raw, c_ql_m1)
    local_qhead_idx_for_q = lqh_pair % c_query_group_size
    return qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q


def _load_q_fragments(
    q_rsrc,
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    q_base,
    lane16id,
    lane4id,
    rowid,
    local_qhead_idx,
    *,
    query_packed_fp8,
    query_load_is_bf16,
    compute_query_scale_in_kernel=False,
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
    #
    # This replaces the previous swizzled layout
    #   write: (lane16id//4)*2048 + (lane16id%4)*512 + local_qhead_idx*32
    #   read : qkhe*2048 + rowid*512 + lane16id*32 + qkr*8
    # which implicitly required every lane to load 16 elements of Q and therefore
    # read `lane16id*16` elements past `q_base`.  For bf16/f16 (2 B/elem) lane16id
    # ∈ {8..15} crossed the qhead boundary and faulted once the query tensor
    # ended mid-block.
    lds_q_base = local_qhead_idx * arith.constant(HEAD_SIZE, type=T.i32) + lane16id * arith.constant(8, type=T.i32)
    query_scale_lane = None
    if const_expr(query_packed_fp8 or not compute_query_scale_in_kernel):
        # NB: _load_q_words() is unreachable under compile_pswa_decode_sw today
        # (that entry point rejects packed_fp8 and always forces
        # compute_query_scale_in_kernel=True).  Keep the call site intact for
        # any out-of-tree callers but only consume the first two i32 words so
        # the per-lane footprint matches the new 8-B LDS slot.
        q_words = _load_q_words(
            q_rsrc,
            q_base,
            lane16id,
            query_packed_fp8=query_packed_fp8,
            query_load_is_bf16=query_load_is_bf16,
        )
        q_w0, q_w1 = q_words[0], q_words[1]
    else:
        # bf16/f16 + in-kernel query_scale path.  Each lane owns 8 Q elements,
        # loaded as 2 × vec_width=4 buffer loads (4 bf16/f16 elems per load = 8 B,
        # element offset += 4 per iter).  After FP8 packing each load produces
        # one i32 word, so the per-lane store is `vec<2, i32>` = 8 B = 1 i64.
        q_elem = q_base + lane16id * arith.constant(8, type=T.i32)

        i32_vec_ty = T.i32x4
        abs_mask = arith.constant_vector(0x7FFFFFFF, i32_vec_ty)
        c_zero_f = arith.constant(0.0, type=T.f32)
        c_one_f = arith.constant(1.0, type=T.f32)
        c_fp8_max = arith.constant(FP8_MAX, type=T.f32)
        c_wave16 = arith.constant(16, type=T.i32)
        q_f32_chunks = []
        local_max = c_zero_f
        for qwi in range_constexpr(2):
            q_src = buffer_ops.buffer_load(
                q_rsrc,
                q_elem + arith.constant(qwi * 4, type=T.i32),
                vec_width=4,
                dtype=T.bf16 if const_expr(query_load_is_bf16) else T.f16,
            )
            q_f32 = fx.Vector(q_src).to(fx.Float32)
            q_f32_chunks.append(q_f32)
            q_i32 = q_f32.bitcast(fx.Int32)
            q_abs_i32 = arith.andi(q_i32, abs_mask)
            q_abs = fx.Vector(q_abs_i32).bitcast(fx.Float32)
            chunk_max = q_abs.reduce("max")
            local_max = local_max.maximumf(chunk_max)

        for sh in [8, 4, 2, 1]:
            local_max = local_max.maximumf(local_max.shuffle_xor(arith.constant(sh, type=T.i32), c_wave16))
        query_scale_lane = arith.select(
            local_max > c_zero_f,
            local_max / c_fp8_max,
            c_one_f,
        )
        inv_query_scale = c_one_f / query_scale_lane
        q_words = []
        for q_f32 in q_f32_chunks:
            p0 = q_f32[0] * inv_query_scale
            p1 = q_f32[1] * inv_query_scale
            p2 = q_f32[2] * inv_query_scale
            p3 = q_f32[3] * inv_query_scale
            lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, fx.Int32(0), False)
            q_words.append(rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True))
        q_w0, q_w1 = q_words

        lane0 = arith.cmpi(
            arith.CmpIPredicate.eq,
            lane16id,
            arith.constant(0, type=T.i32),
        )
        lane0_if = scf.IfOp(lane0)
        with ir.InsertionPoint(lane0_if.then_block):
            vector.store(
                fx.Vector.from_elements([query_scale_lane], dtype=fx.Float32),
                softmax_lds_f32,
                [arith.index_cast(T.index, local_qhead_idx)],
            )
            scf.YieldOp([])

    v01 = fx.Vector.from_elements([q_w0, q_w1], dtype=fx.Int32)
    lds_q_i32 = lds_q_base // arith.constant(4, type=T.i32)
    vector.store(v01, logits_lds_i32, [arith.index_cast(T.index, lds_q_i32)])

    q_frags = []
    gpu.barrier()
    if const_expr(compute_query_scale_in_kernel):
        query_scale_lane = fx.Vector(
            vector.load_op(
                T.vec(1, T.f32),
                softmax_lds_f32,
                [arith.index_cast(T.index, lane16id)],
            )
        )[0].ir_value()
    c_head_size = arith.constant(HEAD_SIZE, type=T.i32)
    for qkhe in range_constexpr(QKHELOOP):
        for qkr in range_constexpr(2):
            # See layout comment above. Byte offset:
            #   lane16id * HEAD_SIZE + qkhe*64 + rowid*16 + qkr*8
            lds_rd_byte = (
                lane16id * c_head_size
                + arith.constant(qkhe * 64, type=T.i32)
                + rowid * arith.constant(16, type=T.i32)
                + arith.constant(qkr * 8, type=T.i32)
            )
            lds_rd_base = lds_rd_byte // arith.constant(8, type=T.i32)
            q_v1 = vector.load_op(
                T.vec(1, T.i64),
                logits_lds_i64,
                [arith.index_cast(T.index, lds_rd_base)],
            )
            q_frags.append(fx.Vector(q_v1)[0])
    return q_frags, query_scale_lane


def _load_sw_mtp_group_q_fragments(
    q_rsrc,
    logits_lds_i32,
    logits_lds_i64,
    softmax_lds_f32,
    batch_idx,
    kv_h,
    stride_q_seq,
    stride_q_head,
    lane16id,
    lane4id,
    rowid,
    local_qhead_idx,
    *,
    mtp_group_idx,
    mtp_subgroup_count,
    query_length,
    query_group_size,
    query_packed_fp8,
    query_load_is_bf16,
    compute_query_scale_in_kernel=False,
):
    mtp_states = []
    c_query_length = arith.constant(query_length, type=T.i32)
    c_query_group_size = arith.constant(query_group_size, type=T.i32)
    for mtp_subgroup_idx in range_constexpr(mtp_subgroup_count):
        if const_expr(mtp_subgroup_idx > 0):
            gpu.barrier()
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
        query_scale_lane = None
        q_frags, kernel_query_scale_lane = _load_q_fragments(
            q_rsrc,
            logits_lds_i32,
            logits_lds_i64,
            softmax_lds_f32,
            q_base,
            lane16id,
            lane4id,
            rowid,
            local_qhead_idx,
            query_packed_fp8=query_packed_fp8,
            query_load_is_bf16=query_load_is_bf16,
            compute_query_scale_in_kernel=compute_query_scale_in_kernel,
        )
        if const_expr(compute_query_scale_in_kernel):
            query_scale_lane = kernel_query_scale_lane
        mtp_states.append((qi_val, qhi_pos, q_frags, query_scale_lane))
    return mtp_states


def _normalize_pa_output(running_sum, out0, out1, zero_f):
    safe_sum = arith.select(running_sum > zero_f, running_sum, arith.constant(1.0, type=T.f32))
    inv_sum = arith.constant(1.0, type=T.f32) / safe_sum
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
    v_rsrc,
    ks_rsrc,
    vs_rsrc,
    logits_lds_i32,
    softmax_lds_f32,
    stride_ks_block,
    stride_ks_head,
    softmax_scale_base,
    softmax_q_scale,
    k_scale_val,
    scale,
    v_scale_val,
    warp_id,
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
    c_four,
    c_w,
    neg_inf,
    zero_f,
):
    apply_causal_mask = needs_mask or query_length > 1

    def _qk_and_intra_softmax(
        k_ops,
        partition_start,
        v_rsrc_cur,
        v_block_base_dw,
        tile_token_offset_i32,
        q_frags,
        causal_bound,
        query_scale_lane=None,
        *,
        phys_block,
        seq_start=None,
    ):
        v_results = []
        for vt in range_constexpr(VTLOOP):
            vhe_data = []
            for vhe in range_constexpr(VHELOOP):
                v_token_in_block = tile_token_offset_i32 + v_tok_thread_off[vt]
                if const_expr(trans_v):
                    vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                    va_dw = (
                        v_block_base_dw
                        + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32)
                        + vhead_elem_dw[vhe]
                    )
                else:
                    va_dw = v_block_base_dw + vhead_elem_dw[vhe] + v_token_in_block // c_four
                v_4xi32 = buffer_ops.buffer_load(v_rsrc_cur, va_dw, vec_width=4, dtype=T.i32)
                vhe_data.append(v_4xi32)
            v_results.append(vhe_data)

        if const_expr(per_token_kv):
            scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
            scale_tok_base_pt = tile_token_offset_i32 + k_tok_thread_base
            scale_src_lane_base = rowid * arith.constant(20, type=T.i32)
            k_scale_vecs = []
            v_scale_vecs = []
            for td in range_constexpr(TLOOP):
                tok_off = scale_tok_base_pt + arith.constant(td * MFMA_N, type=T.i32)
                k_scale_lane = buffer_ops.buffer_load(ks_rsrc, scale_block_base + tok_off, vec_width=1, dtype=T.f32)
                v_scale_lane = buffer_ops.buffer_load(vs_rsrc, scale_block_base + tok_off, vec_width=1, dtype=T.f32)
                k_scale_i32 = arith.bitcast(T.i32, k_scale_lane)
                v_scale_i32 = arith.bitcast(T.i32, v_scale_lane)
                k_scale_vals = []
                v_scale_vals = []
                for i in range_constexpr(4):
                    bcast_addr = (scale_src_lane_base + arith.constant(i, type=T.i32)) * c_four
                    sk_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(k_scale_i32))
                    sv_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(v_scale_i32))
                    k_scale_vals.append(arith.bitcast(T.f32, sk_i32))
                    v_scale_vals.append(arith.bitcast(T.f32, sv_i32))
                k_scale_vecs.append(vector.from_elements(T.f32x4, k_scale_vals))
                v_scale_vecs.append(vector.from_elements(T.f32x4, v_scale_vals))
        else:
            v_scale_vecs = None

        d_out = []
        query_scale_vec = None
        if const_expr(per_token_q):
            query_scale_vec = vector.broadcast(T.f32x4, query_scale_lane * softmax_scale_base)
        for td in range_constexpr(TLOOP):
            acc = arith.constant_vector(0.0, T.f32x4)
            for k_step in range_constexpr(QKHELOOP * 2):
                acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
            if const_expr(per_token_kv):
                scale_vec = (
                    k_scale_vecs[td] * query_scale_vec
                    if const_expr(per_token_q)
                    else k_scale_vecs[td] * vector.broadcast(T.f32x4, softmax_q_scale)
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
            for i in range_constexpr(4):
                s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                if const_expr(kv_tok_base is not None):
                    kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                    if const_expr(apply_causal_mask):
                        s = arith.select(kv_tok < causal_bound, s, neg_inf)
                    if const_expr(apply_range_mask):
                        s = arith.select(kv_tok >= seq_start, s, neg_inf)
                qk_max = qk_max.maximumf(s)
        for sh in [32, 16]:
            qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
        vector.store(
            vector.from_elements(T.vec(1, T.f32), [qk_max]),
            softmax_lds_f32,
            [sm_max_off],
        )

        exp_sum = zero_f
        for td in range_constexpr(TLOOP):
            for i in range_constexpr(4):
                s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                diff = s - qk_max
                p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                if const_expr(kv_tok_base is not None):
                    kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                    if const_expr(apply_causal_mask):
                        p = arith.select(kv_tok < causal_bound, p, zero_f)
                    if const_expr(apply_range_mask):
                        p = arith.select(kv_tok >= seq_start, p, zero_f)
                exp_sum = exp_sum + p
                d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
        for sh in [32, 16]:
            exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
        vector.store(
            vector.from_elements(T.vec(1, T.f32), [exp_sum]),
            softmax_lds_f32,
            [sm_sum_off],
        )

        if const_expr(per_token_kv):
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = v_scale_vecs[td]
                for i in range_constexpr(4):
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        if const_expr(apply_causal_mask):
                            vs_i = arith.select(kv_tok < causal_bound, vs_i, zero_f)
                        if const_expr(apply_range_mask):
                            vs_i = arith.select(kv_tok >= seq_start, vs_i, zero_f)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                v_max_warp = v_max_warp.maximumf(vector.reduction(T.f32, "maxnumf", vs))
            for sh in [32, 16]:
                v_max_warp = v_max_warp.maximumf(v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            vector.store(
                vector.from_elements(T.vec(1, T.f32), [v_max_warp]),
                softmax_lds_f32,
                [sm_vmax_wr_off],
            )
        return d_out, v_results, v_scale_vecs

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, o0, o1, v_scale_vecs=None):
        partition_max = neg_inf
        partition_sum = zero_f
        warp_rescale_factors = []
        for w in range_constexpr(NUM_WARPS):
            w_max = vector.extract(
                vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [sm_rd_max_offs[w]]),
                static_position=[0],
            )
            partition_max = partition_max.maximumf(w_max)
            warp_rescale_factors.append(w_max)
        for w in range_constexpr(NUM_WARPS):
            diff_w = warp_rescale_factors[w] - partition_max
            if const_expr(needs_mask):
                diff_w = arith.select(partition_max > neg_inf, diff_w, zero_f)
            wf = (diff_w * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
            w_sum = vector.extract(
                vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [sm_rd_sum_offs[w]]),
                static_position=[0],
            )
            partition_sum = partition_sum + w_sum * wf
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
                ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                zero_f,
            )
            part_to_new = arith.select(
                partition_max > neg_inf,
                ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(
                    fastmath=arith.FastMathFlags.fast
                ),
                zero_f,
            )
        else:
            accum_scale = ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(
                fastmath=arith.FastMathFlags.fast
            )
            part_to_new = ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(
                fastmath=arith.FastMathFlags.fast
            )

        rsum = accum_scale * rsum + partition_sum * part_to_new
        rmax = new_rmax
        o0 = o0 * vector.broadcast(T.f32x4, accum_scale)
        o1 = o1 * vector.broadcast(T.f32x4, accum_scale)

        if const_expr(per_token_kv and v_scale_vecs is not None):
            v_max_global = zero_f
            for w in range_constexpr(NUM_WARPS):
                w_vmax = vector.extract(
                    vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [sm_vmax_rd_offs[w]]),
                    static_position=[0],
                )
                v_max_global = v_max_global.maximumf(w_vmax)
            v_max_safe = v_max_global + arith.constant(1e-8, type=T.f32)
            c_fp8_max = arith.constant(FP8_MAX, type=T.f32)
            norm_factor = c_fp8_max / v_max_safe
            prob_scale = my_warp_rescale
            v_correction = v_max_global / c_fp8_max * part_to_new
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * (v_scale_vecs[td] * vector.broadcast(T.f32x4, prob_scale * norm_factor))
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
            i32_off = byte_base // c_four
            pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
            vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])
        return rmax, rsum, o0, o1, v_correction

    def _pv_mfma(v_ops, o0, o1, v_correction):
        c_one = arith.constant(1, type=T.i32)
        pv_results = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(VHELOOP)]
        v_i64s = []
        p_i64s = []
        for vhe in range_constexpr(VHELOOP):
            for vt in range_constexpr(VTLOOP):
                v_4xi32 = v_ops[vt][vhe]
                for j in range_constexpr(2):
                    v_i64 = _pack_i32_pair_to_i64(
                        vector.extract(v_4xi32, static_position=[j * 2]),
                        vector.extract(v_4xi32, static_position=[j * 2 + 1]),
                    )
                    v_i64s.append(v_i64)
                    p_byte = (
                        arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                        + pv_prob_read_base
                        + arith.constant(j * 8, type=T.i32)
                    )
                    p_i32_idx = p_byte // c_four
                    pw0 = vector.extract(
                        vector.load_op(
                            T.vec(1, T.i32),
                            logits_lds_i32,
                            [arith.index_cast(T.index, p_i32_idx)],
                        ),
                        static_position=[0],
                    )
                    pw1 = vector.extract(
                        vector.load_op(
                            T.vec(1, T.i32),
                            logits_lds_i32,
                            [arith.index_cast(T.index, p_i32_idx + c_one)],
                        ),
                        static_position=[0],
                    )
                    p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                    p_i64s.append(p_i64)
        for vhe in range_constexpr(VHELOOP):
            tmp_out = arith.constant_vector(0.0, T.f32x4)
            for vt in range_constexpr(VTLOOP):
                for j in range_constexpr(2):
                    tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4,
                        [
                            v_i64s[vhe * VTLOOP * 2 + vt * 2 + j],
                            p_i64s[vhe * VTLOOP * 2 + vt * 2 + j],
                            tmp_out,
                            0,
                            0,
                            0,
                        ],
                    )
                    pv_results[vhe] = tmp_out
        o0 = o0 + pv_results[0] * vector.broadcast(T.f32x4, v_correction)
        o1 = o1 + pv_results[1] * vector.broadcast(T.f32x4, v_correction)
        return o0, o1

    def _prepare_block_split_pair(
        k_ops,
        v_rsrc_cur,
        v_block_base_dw,
        tile_token_offset_i32,
        q_frags_0,
        q_frags_1,
        query_scale_lane_0,
        query_scale_lane_1,
        *,
        phys_block,
    ):
        va_dws = []
        for vt in range_constexpr(VTLOOP):
            vhe_data = []
            for vhe in range_constexpr(VHELOOP):
                v_token_in_block = tile_token_offset_i32 + v_tok_thread_off[vt]
                if const_expr(trans_v):
                    vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                    va_dw = (
                        v_block_base_dw
                        + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32)
                        + vhead_elem_dw[vhe]
                    )
                else:
                    va_dw = v_block_base_dw + vhead_elem_dw[vhe] + v_token_in_block // c_four
                vhe_data.append(va_dw)
            va_dws.append(vhe_data)

        if const_expr(per_token_kv):
            scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
            scale_tok_base_pt = tile_token_offset_i32 + k_tok_thread_base
            scale_src_lane_base = rowid * arith.constant(20, type=T.i32)
            k_scale_vecs = []
            v_scale_vecs = []
            for td in range_constexpr(TLOOP):
                tok_off = scale_tok_base_pt + arith.constant(td * MFMA_N, type=T.i32)
                k_scale_lane = buffer_ops.buffer_load(ks_rsrc, scale_block_base + tok_off, vec_width=1, dtype=T.f32)
                v_scale_lane = buffer_ops.buffer_load(vs_rsrc, scale_block_base + tok_off, vec_width=1, dtype=T.f32)
                k_scale_i32 = arith.bitcast(T.i32, k_scale_lane)
                v_scale_i32 = arith.bitcast(T.i32, v_scale_lane)
                k_scale_vals = []
                v_scale_vals = []
                for i in range_constexpr(4):
                    bcast_addr = (scale_src_lane_base + arith.constant(i, type=T.i32)) * c_four
                    sk_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(k_scale_i32))
                    sv_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(v_scale_i32))
                    k_scale_vals.append(arith.bitcast(T.f32, sk_i32))
                    v_scale_vals.append(arith.bitcast(T.f32, sv_i32))
                k_scale_vecs.append(vector.from_elements(T.f32x4, k_scale_vals))
                v_scale_vecs.append(vector.from_elements(T.f32x4, v_scale_vals))
        else:
            k_scale_vecs = None
            v_scale_vecs = None

        d_out_0 = []
        d_out_1 = []
        v_results = []
        query_scale_vec_0 = None
        query_scale_vec_1 = None
        if const_expr(per_token_q):
            query_scale_vec_0 = vector.broadcast(T.f32x4, query_scale_lane_0 * softmax_scale_base)
            query_scale_vec_1 = vector.broadcast(T.f32x4, query_scale_lane_1 * softmax_scale_base)
        for td in range_constexpr(TLOOP):
            vhe_data = []
            acc_0 = arith.constant_vector(0.0, T.f32x4)
            acc_1 = arith.constant_vector(0.0, T.f32x4)
            for k_step in range_constexpr(QKHELOOP * 2):
                if const_expr(k_step % 2 == 0):
                    v_4xi32 = buffer_ops.buffer_load(v_rsrc_cur, va_dws[td][k_step // 2], vec_width=4, dtype=T.i32)
                    vhe_data.append(v_4xi32)
                acc_0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                    T.f32x4,
                    [k_ops[td][k_step], q_frags_0[k_step], acc_0, 0, 0, 0],
                )
                acc_1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                    T.f32x4,
                    [k_ops[td][k_step], q_frags_1[k_step], acc_1, 0, 0, 0],
                )
            v_results.append(vhe_data)
            if const_expr(per_token_kv):
                if const_expr(per_token_q):
                    scale_vec_0 = k_scale_vecs[td] * query_scale_vec_0
                    scale_vec_1 = k_scale_vecs[td] * query_scale_vec_1
                else:
                    scale_vec_0 = k_scale_vecs[td] * vector.broadcast(T.f32x4, softmax_q_scale)
                    scale_vec_1 = scale_vec_0
                d_out_0.append(acc_0 * scale_vec_0)
                d_out_1.append(acc_1 * scale_vec_1)
            else:
                if const_expr(per_token_q):
                    d_out_0.append(acc_0 * (query_scale_vec_0 * vector.broadcast(T.f32x4, k_scale_val)))
                    d_out_1.append(acc_1 * (query_scale_vec_1 * vector.broadcast(T.f32x4, k_scale_val)))
                else:
                    d_out_0.append(acc_0 * vector.broadcast(T.f32x4, scale))
                    d_out_1.append(acc_1 * vector.broadcast(T.f32x4, scale))
        return d_out_0, d_out_1, v_results, v_scale_vecs

    def _finalize_block_split_group(
        d_out,
        v_ops,
        v_scale_vecs,
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
            for i in range_constexpr(4):
                s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                if const_expr(kv_tok_base is not None):
                    kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                    if const_expr(apply_causal_mask):
                        s = arith.select(kv_tok < causal_bound, s, neg_inf)
                    if const_expr(apply_range_mask):
                        s = arith.select(kv_tok >= seq_start, s, neg_inf)
                qk_max = qk_max.maximumf(s)
        for sh in [32, 16]:
            qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
        vector.store(
            vector.from_elements(T.vec(1, T.f32), [qk_max]),
            softmax_lds_f32,
            [sm_max_off],
        )

        exp_sum = zero_f
        for td in range_constexpr(TLOOP):
            for i in range_constexpr(4):
                s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                diff = s - qk_max
                p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                if const_expr(kv_tok_base is not None):
                    kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                    if const_expr(apply_causal_mask):
                        p = arith.select(kv_tok < causal_bound, p, zero_f)
                    if const_expr(apply_range_mask):
                        p = arith.select(kv_tok >= seq_start, p, zero_f)
                exp_sum = exp_sum + p
                d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
        for sh in [32, 16]:
            exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
        vector.store(
            vector.from_elements(T.vec(1, T.f32), [exp_sum]),
            softmax_lds_f32,
            [sm_sum_off],
        )

        if const_expr(per_token_kv and v_scale_vecs is not None):
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = v_scale_vecs[td]
                for i in range_constexpr(4):
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        if const_expr(apply_causal_mask):
                            vs_i = arith.select(kv_tok < causal_bound, vs_i, zero_f)
                        if const_expr(apply_range_mask):
                            vs_i = arith.select(kv_tok >= seq_start, vs_i, zero_f)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                v_max_warp = v_max_warp.maximumf(vector.reduction(T.f32, "maxnumf", vs))
            for sh in [32, 16]:
                v_max_warp = v_max_warp.maximumf(v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            vector.store(
                vector.from_elements(T.vec(1, T.f32), [v_max_warp]),
                softmax_lds_f32,
                [sm_vmax_wr_off],
            )

        gpu.barrier()
        rmax, rsum, o0, o1, v_correction = _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, o0, o1, v_scale_vecs)
        gpu.barrier()
        o0, o1 = _pv_mfma(v_ops, o0, o1, v_correction)
        return rmax, rsum, o0, o1

    return (
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
        _prepare_block_split_pair,
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

    Unlike compile_pa_decode_sw, this does NOT bake in num_seqs/num_kv_heads/num_partitions
    because PS mode uses dynamic work distribution. Grid = (num_sm, 1, 4).
    """
    arch = get_hip_arch()
    query_packed_fp8 = query_input_dtype == "packed_fp8"
    query_load_is_bf16 = query_input_dtype == "bf16"
    query_scale_in_kernel = not query_packed_fp8
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
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL

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
        lane4id = tid & arith.constant(3, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)

        # ── Buffer resources ──
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
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
        c_four = arith.constant(4, type=T.i32)

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
            c_four=c_four,
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
        _work_start_idx = arith.index_cast(T.index, arith.unwrap(work_start))
        _work_end_idx = arith.index_cast(T.index, arith.unwrap(work_end))
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
            kv_h = q_head_start // arith.constant(query_group_size, type=T.i32)

            # Context length for this sequence
            context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            # ── Prologue: load first block's tile 0 K data ──
            first_phys_block = buffer_ops.buffer_load(kpi_rsrc, kv_start, vec_width=1, dtype=T.i32)
            # Head offsets for K and V cache
            _k_head_off = kv_h * c_kh
            _v_head_off = kv_h * c_vh

            (
                _qk_and_intra_softmax,
                _cross_warp_softmax_and_prob_pack,
                _pv_mfma,
                _prepare_block_split_pair_unused,
                _finalize_block_split_group_unused,
            ) = _make_pa_phase_helpers(
                trans_v=trans_v,
                per_token_q=query_scale_in_kernel,
                per_token_kv=per_token_kv,
                needs_mask=needs_mask,
                query_length=query_length,
                kv_h=kv_h,
                v_rsrc=v_rsrc,
                ks_rsrc=ks_rsrc,
                vs_rsrc=vs_rsrc,
                logits_lds_i32=logits_lds_i32,
                softmax_lds_f32=softmax_lds_f32,
                stride_ks_block=stride_ks_block,
                stride_ks_head=stride_ks_head,
                softmax_scale_base=_softmax_scale_const,
                softmax_q_scale=_softmax_q_scale,
                k_scale_val=k_scale_val,
                scale=_scale,
                v_scale_val=v_scale_val,
                warp_id=warp_id,
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
                c_four=c_four,
                c_w=c_w,
                neg_inf=NEG_INF,
                zero_f=ZERO_F,
            )

            # ════════════════════════════════════════════════════════
            # Inner KV loop — one CTA processes one 256-token sub-tile
            # across all 1024-token physical blocks in the work item.
            # ════════════════════════════════════════════════════════
            def _unwrap(v):
                return v.ir_value() if hasattr(v, "ir_value") else v

            def _pack_state(rmax, rsum, o0, o1, k_flat):
                return [_unwrap(v) for v in [rmax, rsum, o0, o1] + k_flat]

            def _unpack_state(state):
                return state[0], state[1], state[2], state[3], list(state[4 : 4 + _N_K])

            def _process_block_split(
                phys_block,
                block_idx_in_work,
                rmax,
                rsum,
                o0,
                o1,
                tile_token_offset_i32,
                k_ops,
                next_k_rsrc=None,
                next_k_base=None,
            ):
                """Process one 256-token block split inside a 1024-token KV page."""
                partition_start = kv_start_abs_tok + block_idx_in_work * c_bs + tile_token_offset_i32
                v_block_rsrc, v_local_phys_block = _chunk_buffer_resource_for_block(value_cache_ptr, phys_block, c_vb)
                v_base = _compute_block_base_dw(v_local_phys_block, c_vb, _v_head_off)
                d_out, v_ops, v_scales = _qk_and_intra_softmax(
                    k_ops,
                    partition_start,
                    v_block_rsrc,
                    v_base,
                    tile_token_offset_i32,
                    q_frags,
                    causal_bound,
                    query_scale_lane=query_scale_lane,
                    phys_block=phys_block,
                )

                gpu.barrier()
                rmax, rsum, o0, o1, v_correction = _cross_warp_softmax_and_prob_pack(
                    d_out, rmax, rsum, o0, o1, v_scales
                )
                if const_expr(next_k_rsrc is not None and next_k_base is not None):
                    k_next_flat = _load_k_flat(
                        next_k_rsrc,
                        next_k_base,
                        tile_token_offset_i32,
                        _k_tok_thread_base,
                        _c_tok_stride_dw,
                        _k_he_off_dw,
                    )
                else:
                    k_next_flat = None

                gpu.barrier()
                o0, o1 = _pv_mfma(v_ops, o0, o1, v_correction)
                return rmax, rsum, o0, o1, k_next_flat

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
            _loop_stop_g = arith.index_cast(T.index, arith.unwrap(num_blocks_in_work))
            _loop_step_g = arith.index(1)

            # ── MTP groups: Python compile-time loop — one MLIR KV-loop per group ──
            # Use range_constexpr so AST rewriter keeps this as a plain Python loop
            _mtp_groups = math.ceil(query_length * query_group_size / 16)
            for _mtp_g in range_constexpr(_mtp_groups):
                qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q = _compute_mtp_group_state(
                    lane16id,
                    local_qhead_idx,
                    mtp_group_idx=_mtp_g,
                    query_length=query_length,
                    query_group_size=query_group_size,
                )
                # MTP causal bound for this lane's qi_val token
                causal_bound = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val

                # ── Q load into LDS for this mtp_g pass ──
                # Between passes: barrier ensures prev pass's LDS prob-reads are done
                if const_expr(_mtp_g > 0):
                    gpu.barrier()
                q_row = batch_idx * arith.constant(query_length, type=T.i32) + qi_for_q
                q_base = (
                    q_row * stride_q_seq
                    + (kv_h * arith.constant(query_group_size, type=T.i32) + local_qhead_idx_for_q) * stride_q_head
                )
                q_frags, query_scale_lane = _load_q_fragments(
                    q_rsrc,
                    logits_lds_i32,
                    logits_lds_i64,
                    softmax_lds_f32,
                    q_base,
                    lane16id,
                    lane4id,
                    rowid,
                    local_qhead_idx,
                    query_packed_fp8=query_packed_fp8,
                    query_load_is_bf16=query_load_is_bf16,
                    compute_query_scale_in_kernel=query_scale_in_kernel,
                )

                # ── K init: load this CTA's 256-token block split for the first block ──
                first_k_rsrc, first_k_local_phys_block = _chunk_buffer_resource_for_block(
                    key_cache_ptr, first_phys_block, c_kb
                )
                first_k_base = _compute_block_base_dw(first_k_local_phys_block, c_kb, _k_head_off)
                k_flat = _load_k_flat(
                    first_k_rsrc,
                    first_k_base,
                    tile_token_offset,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                )

                init_state = _pack_state(
                    NEG_INF,
                    ZERO_F,
                    arith.constant_vector(0.0, T.f32x4),
                    arith.constant_vector(0.0, T.f32x4),
                    k_flat,
                )

                for ib, state in range(_loop_start_g, _loop_stop_g, _loop_step_g, init=init_state):
                    running_max, running_sum, out0, out1, k_flat = _unpack_state(state)
                    block_idx = arith.index_cast(T.i32, ib)

                    phys_block = buffer_ops.buffer_load(kpi_rsrc, kv_start + block_idx, vec_width=1, dtype=T.i32)
                    next_idx_raw = block_idx + c_one
                    next_idx_clamped = arith.select(next_idx_raw < num_blocks_in_work, next_idx_raw, last_block_idx_val)
                    next_phys_block = buffer_ops.buffer_load(
                        kpi_rsrc, kv_start + next_idx_clamped, vec_width=1, dtype=T.i32
                    )
                    next_k_rsrc, next_k_local_phys_block = _chunk_buffer_resource_for_block(
                        key_cache_ptr, next_phys_block, c_kb
                    )
                    next_k_base = _compute_block_base_dw(next_k_local_phys_block, c_kb, _k_head_off)

                    k_ops = _unflatten_k(k_flat)

                    running_max, running_sum, out0, out1, k_next_flat = _process_block_split(
                        phys_block,
                        block_idx,
                        running_max,
                        running_sum,
                        out0,
                        out1,
                        tile_token_offset,
                        k_ops,
                        next_k_rsrc=next_k_rsrc,
                        next_k_base=next_k_base,
                    )

                    results = yield _pack_state(running_max, running_sum, out0, out1, k_next_flat)

                running_max, running_sum, out0, out1, _ = _unpack_state(results)

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
        ).launch(grid=(num_sm, 1, TILES_PER_BLOCK), block=(BLOCK_THREADS, 1, 1), stream=stream)

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
    import aiter

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
    ) = aiter.get_pa_metadata_info_v1(batch_size, num_kv_heads)

    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=dev)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=dev)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type, device=dev)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=dev)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=dev)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=dev)

    aiter.get_pa_metadata_v1(
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
        tid = gpu.thread_idx.x
        batch_idx = gpu.block_idx.x
        kv_head_idx = gpu.block_idx.y
        eqgs_idx = gpu.block_idx.z

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

        c_zero_f = arith.constant(0.0, type=T.f32)
        c_one_f = arith.constant(1.0, type=T.f32)
        c_neg_inf = arith.constant(float("-inf"), type=T.f32)
        c_log2e = arith.constant(LOG2E, type=T.f32)
        fm_fast = arith.FastMathFlags.fast
        c_zero_i = arith.constant(0, type=T.i32)
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        c_wave_mask = arith.constant(WARP_SIZE - 1, type=T.i32)
        c_wave_shift = arith.constant(6, type=T.i32)
        c_red_slots = arith.constant(red_slots, type=T.i32)
        lane = tid & c_wave_mask
        wave = tid >> c_wave_shift

        def _wave_reduce_max_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.maximumf(red.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            return red

        def _wave_reduce_sum_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.addf(
                    red.shuffle_xor(arith.constant(sh, type=T.i32), c_w),
                    fastmath=fm_fast,
                )
            return red

        def _block_reduce(val, mode):
            if const_expr(red_slots == 1):
                return _wave_reduce_max_full(val) if const_expr(mode == "max") else _wave_reduce_sum_full(val)

            neutral = c_neg_inf if const_expr(mode == "max") else c_zero_f
            w = _wave_reduce_max_full(val) if const_expr(mode == "max") else _wave_reduce_sum_full(val)

            if lane == c_zero_i:
                wave_idx = arith.index_cast(T.index, wave)
                red_scratch.store(w, [wave_idx])
            gpu.barrier()

            if wave == c_zero_i:
                in_range = lane < c_red_slots
                lane_safe = arith.select(in_range, lane, c_zero_i)
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                red_val = red_scratch.load([lane_safe_idx])
                red_val = arith.select(in_range, red_val, neutral)
                red_val = (
                    _wave_reduce_max_full(red_val) if const_expr(mode == "max") else _wave_reduce_sum_full(red_val)
                )
                if lane == c_zero_i:
                    red_scratch.store(red_val, [arith.constant(0, index=True)])
            gpu.barrier()

            return red_scratch.load([arith.constant(0, index=True)])

        if const_expr(max_context_partition_num <= WARP_SIZE):
            c_part_num = arith.constant(max_context_partition_num, type=T.i32)
            c_reduce_width = arith.constant(reduce_width, type=T.i32)
            c_four = arith.constant(4, type=T.i32)

            def _wave_reduce_max(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.maximumf(red.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                return red

            def _wave_reduce_sum(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.addf(
                        red.shuffle_xor(arith.constant(sh, type=T.i32), c_w),
                        fastmath=fm_fast,
                    )
                return red

            lane_in_range = lane < c_part_num
            lane_in_reduce = lane < c_reduce_width
            part_sum = c_zero_f
            part_max = c_neg_inf
            if lane_in_reduce:
                part_i32 = arith.select(lane_in_range, lane, c_zero_i)
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
                ((part_max - global_max) * c_log2e).exp2(fastmath=fm_fast),
                c_zero_f,
            )
            scaled_sum = part_sum * part_scale
            global_exp_sum = _wave_reduce_sum(scaled_sum)
            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )
            weight_local = scaled_sum / safe_global_exp_sum
            weight_local_i32 = arith.bitcast(T.i32, arith.unwrap(weight_local))

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = arith.constant(part_idx, type=T.i32)
                bcast_addr = part_i32 * c_four
                weight_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(weight_local_i32))
                weight = arith.bitcast(T.f32, weight_i32)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_bf16 = buffer_ops.buffer_load(logits_rsrc, logits_off, vec_width=1, dtype=T.bf16)
                part_logits = fx.Float32(part_logits_bf16)
                acc = acc + part_logits * weight
        else:
            # Fallback for unusually large sliding-window partition counts.
            global_max = c_neg_inf
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = arith.constant(chunk_size, type=T.i32)
                c_chunk_base = arith.constant(chunk_base, type=T.i32)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, c_zero_i)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_max_raw = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                chunk_max = _block_reduce(part_max, "max")
                global_max = global_max.maximumf(chunk_max)

            global_exp_sum = c_zero_f
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = arith.constant(chunk_size, type=T.i32)
                c_chunk_base = arith.constant(chunk_base, type=T.i32)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, c_zero_i)
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
                    ((part_max - global_max) * c_log2e).exp2(fastmath=fm_fast),
                    c_zero_f,
                )
                chunk_sum = _block_reduce(part_sum * part_scale, "sum")
                global_exp_sum = global_exp_sum + chunk_sum

            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )

            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = arith.constant(chunk_size, type=T.i32)
                c_chunk_base = arith.constant(chunk_base, type=T.i32)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, c_zero_i)
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
                    part_scale = ((part_max - global_max) * c_log2e).exp2(fastmath=fm_fast)
                    weight = (part_sum * part_scale) / safe_global_exp_sum
                    part_idx_idx = arith.index_cast(T.index, part_i32)
                    part_weights_lds.store(weight, [part_idx_idx])

            gpu.barrier()

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = arith.constant(part_idx, type=T.i32)
                part_idx_idx = arith.constant(part_idx, index=True)
                weight = part_weights_lds.load([part_idx_idx])
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_bf16 = buffer_ops.buffer_load(logits_rsrc, logits_off, vec_width=1, dtype=T.bf16)
                part_logits = fx.Float32(part_logits_bf16)
                acc = acc + part_logits * weight

        c_qgs = arith.constant(query_group_size, type=T.i32)
        query_idx = eqgs_idx // c_qgs
        group_idx = eqgs_idx % c_qgs
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
            out_val = arith.trunc_f(T.f16, acc)
        else:
            out_val = arith.trunc_f(T.bf16, acc)
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

        compiled_sw = compile_pa_decode_sw(
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            trans_v=trans_v,
            query_group_size=query_group_size,
            per_token_kv=per_token_kv,
            query_length=query_length,
            query_input_dtype=query_input_dtype,
        )

        compiled_sw["launch"](
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

        head_size = query.shape[-1]
        output_5d = output.reshape(batch_size, query_length, num_kv_heads, query_group_size, head_size)
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


# =====================================================================
# =====================================================================
# compile_pa_decode_sw — Sliding Window kernel with one CTA per 256-token tile
# Grid = (batch_size, num_kv_heads, max_context_partition_num)
# Each block handles one 256-token context partition. `partition_idx` is decoded
# into (physical_block, 256-token sub-tile) after applying the sliding-window offset.
# Uses block_tables for physical block lookup instead of kv_page_indices.
# Output: exp_sums, max_logits, temporary_output -> reduced by a separate kernel.
# =====================================================================
def compile_pa_decode_sw(
    sliding_window: int,  # required > 0 -- baked as compile-time constant
    softmax_scale=None,
    trans_v=False,
    query_group_size=QUERY_GROUP_SIZE,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "packed_fp8",
):
    """Compile a Gluon-style partitioned PA decode kernel for sliding window.

    Grid = (batch_size, num_kv_heads, max_context_partition_num).
    Each GPU block processes one 256-token partition selected from the visible KV
    region: the sliding tail window.
    sliding_window is a compile-time constant.
    """
    assert sliding_window > 0, "Use compile_pa_decode_ps for sliding_window=0"
    arch = get_hip_arch()
    query_packed_fp8 = query_input_dtype == "packed_fp8"
    query_load_is_bf16 = query_input_dtype == "bf16"
    query_scale_in_kernel = not query_packed_fp8
    if const_expr(query_packed_fp8):
        raise ValueError("`compile_pa_decode_sw` only supports bf16/f16 queries with kernel-internal query scale.")
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE**0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024

    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_sw_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL

    @flyc.kernel
    def pa_decode_sw_kernel(
        exp_sums_ptr: fx.Tensor,  # [batch, kv_heads, max_parts, eqgs] f32
        max_logits_ptr: fx.Tensor,  # [batch, kv_heads, max_parts, eqgs] f32
        tmp_out_ptr: fx.Tensor,  # [batch, kv_heads, max_parts, eqgs, head_size] bf16
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
        stride_bt_seq: Int32,
        stride_ks_block: Int32,
        stride_ks_head: Int32,
    ):
        tid = gpu.thread_idx.x
        batch_idx = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        partition_idx = gpu.block_idx.z

        lane16id = tid & arith.constant(15, type=T.i32)
        lane4id = tid & arith.constant(3, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        to_rsrc = buffer_ops.create_buffer_resource(tmp_out_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)

        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        q_scale_val = arith.constant(1.0, type=T.f32)
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
        c_four = arith.constant(4, type=T.i32)
        c_tpb = arith.constant(TILES_PER_BLOCK, type=T.i32)
        c_zero = arith.constant(0, type=T.i32)

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
            c_four=c_four,
            trans_v=trans_v,
            per_token_kv=per_token_kv,
        )

        # ── Context length and partition mapping ──
        # Visible tiles cover the union of all per-query sliding windows.
        context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)
        _c_sw = arith.constant(sliding_window, type=T.i32)
        _c_query_len = arith.constant(query_length, type=T.i32)
        num_tiles_for_seq = (context_len + c_cps - c_one) // c_cps
        seq_start_global = context_len - _c_query_len - _c_sw
        seq_start_global = arith.select(seq_start_global > c_zero, seq_start_global, c_zero)
        tail_start_tile = seq_start_global // c_cps
        visible_tile_count = num_tiles_for_seq - tail_start_tile
        tile_partition_idx_raw = tail_start_tile + partition_idx

        _is_valid = partition_idx < visible_tile_count
        # Clamp for safe memory access; invalid CTAs take a zero-fill fast path below.
        tile_partition_idx = arith.select(_is_valid, tile_partition_idx_raw, c_zero)
        seq_partition_idx = tile_partition_idx // c_tpb
        block_split_idx = tile_partition_idx % c_tpb
        tile_token_offset = block_split_idx * c_cps
        kv_seq_start = seq_partition_idx * c_bs + tile_token_offset
        # For invalid partitions, set context_len to 0 so all tokens get masked
        context_len = arith.select(_is_valid, context_len, arith.constant(0, type=T.i32))

        # Look up physical block (clamped index is always safe)
        bt_off = batch_idx * stride_bt_seq + seq_partition_idx
        phys_block = buffer_ops.buffer_load(bt_rsrc, bt_off, vec_width=1, dtype=T.i32)

        _k_head_off = kv_h * c_kh
        _v_head_off = kv_h * c_vh

        (
            _qk_and_intra_softmax,
            _cross_warp_softmax_and_prob_pack,
            _pv_mfma,
            _prepare_block_split_pair,
            _finalize_block_split_group,
        ) = _make_pa_phase_helpers(
            trans_v=trans_v,
            per_token_q=query_scale_in_kernel,
            per_token_kv=per_token_kv,
            needs_mask=True,
            query_length=query_length,
            kv_h=kv_h,
            v_rsrc=v_rsrc,
            ks_rsrc=ks_rsrc,
            vs_rsrc=vs_rsrc,
            logits_lds_i32=logits_lds_i32,
            softmax_lds_f32=softmax_lds_f32,
            stride_ks_block=stride_ks_block,
            stride_ks_head=stride_ks_head,
            softmax_scale_base=_softmax_scale_const,
            softmax_q_scale=_softmax_q_scale,
            k_scale_val=k_scale_val,
            scale=_scale,
            v_scale_val=v_scale_val,
            warp_id=warp_id,
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
            c_four=c_four,
            c_w=c_w,
            neg_inf=NEG_INF,
            zero_f=ZERO_F,
        )

        def _process_block_split(
            rmax,
            rsum,
            o0,
            o1,
            tile_token_offset_i32,
            k_ops,
            q_frags,
            causal_bound,
            query_scale_lane,
            seq_start,
        ):
            """Process one 256-token tile inside the selected physical block."""
            v_block_rsrc, v_local_phys_block = _chunk_buffer_resource_for_block(value_cache_ptr, phys_block, c_vb)
            v_base = _compute_block_base_dw(v_local_phys_block, c_vb, _v_head_off)
            d_out_0, v0_ops, vs0 = _qk_and_intra_softmax(
                k_ops,
                kv_seq_start,
                v_block_rsrc,
                v_base,
                tile_token_offset_i32,
                q_frags,
                causal_bound,
                query_scale_lane=query_scale_lane,
                phys_block=phys_block,
                seq_start=seq_start,
            )
            gpu.barrier()
            rmax, rsum, o0, o1, vc0 = _cross_warp_softmax_and_prob_pack(d_out_0, rmax, rsum, o0, o1, vs0)
            gpu.barrier()
            o0, o1 = _pv_mfma(v0_ops, o0, o1, vc0)
            return rmax, rsum, o0, o1

        def _process_block_split_pair(
            rmax_0,
            rsum_0,
            o0_0,
            o1_0,
            rmax_1,
            rsum_1,
            o0_1,
            o1_1,
            tile_token_offset_i32,
            k_ops,
            q_frags_0,
            q_frags_1,
            causal_bound_0,
            causal_bound_1,
            query_scale_lane_0,
            query_scale_lane_1,
            seq_start_0,
            seq_start_1,
        ):
            v_block_rsrc, v_local_phys_block = _chunk_buffer_resource_for_block(value_cache_ptr, phys_block, c_vb)
            v_base = _compute_block_base_dw(v_local_phys_block, c_vb, _v_head_off)
            d_out_0, d_out_1, v0_ops, vs0 = _prepare_block_split_pair(
                k_ops,
                v_block_rsrc,
                v_base,
                tile_token_offset_i32,
                q_frags_0,
                q_frags_1,
                query_scale_lane_0,
                query_scale_lane_1,
                phys_block=phys_block,
            )
            rmax_0, rsum_0, o0_0, o1_0 = _finalize_block_split_group(
                d_out_0,
                v0_ops,
                vs0,
                kv_seq_start,
                causal_bound_0,
                rmax_0,
                rsum_0,
                o0_0,
                o1_0,
                seq_start=seq_start_0,
            )
            gpu.barrier()
            rmax_1, rsum_1, o0_1, o1_1 = _finalize_block_split_group(
                d_out_1,
                v0_ops,
                vs0,
                kv_seq_start,
                causal_bound_1,
                rmax_1,
                rsum_1,
                o0_1,
                o1_1,
                seq_start=seq_start_1,
            )
            return rmax_0, rsum_0, o0_0, o1_0, rmax_1, rsum_1, o0_1, o1_1

        _mtp_groups = _get_sw_mtp_group_count(query_length, query_group_size)

        def _store_partition_results(eqgs_lane, running_sum, running_max, outelems_norm):
            for vhe in range_constexpr(VHELOOP):
                hs_base = (
                    arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                    + warp_id * arith.constant(MFMA_N, type=T.i32)
                    + rowid * arith.constant(4, type=T.i32)
                )
                to_off = (
                    batch_idx * stride_to_seq
                    + kv_h * stride_to_head
                    + partition_idx * stride_to_part
                    + eqgs_lane * stride_to_group
                    + hs_base
                )
                out_bf16 = arith.trunc_f(T.vec(4, T.bf16), outelems_norm[vhe])
                out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                buffer_ops.buffer_store(out_i32, to_rsrc, to_off * arith.constant(2, type=T.i32), offset_is_bytes=True)

            es_off = batch_idx * stride_es_seq + kv_h * stride_es_head + partition_idx * stride_es_part + eqgs_lane
            es_i32 = arith.bitcast(T.i32, running_sum)
            ml_i32 = arith.bitcast(T.i32, running_max)
            buffer_ops.buffer_store(es_i32, es_rsrc, es_off * arith.constant(4, type=T.i32), offset_is_bytes=True)
            buffer_ops.buffer_store(ml_i32, ml_rsrc, es_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

        def _store_group_results(qi_val, qhi_pos, running_sum, running_max, out0, out1):
            outelems_norm = _normalize_pa_output(running_sum, out0, out1, ZERO_F)
            eqgs_lane = qi_val * arith.constant(query_group_size, type=T.i32) + qhi_pos
            _store_partition_results(eqgs_lane, running_sum, running_max, outelems_norm)

        def _write_empty_partition():
            zero_output = [
                arith.constant_vector(0.0, T.f32x4),
                arith.constant_vector(0.0, T.f32x4),
            ]
            for _mtp_g in range_constexpr(_mtp_groups):
                _mtp_subgroups = _get_sw_mtp_subgroup_count(query_length, query_group_size, _mtp_g)
                for _mtp_subgroup in range_constexpr(_mtp_subgroups):
                    qi_val, qhi_pos, _, _ = _compute_sw_mtp_group_state(
                        lane16id,
                        local_qhead_idx,
                        mtp_group_idx=_mtp_g,
                        mtp_subgroup_idx=_mtp_subgroup,
                        query_length=query_length,
                        query_group_size=query_group_size,
                    )
                    eqgs_lane = qi_val * arith.constant(query_group_size, type=T.i32) + qhi_pos
                    _store_partition_results(eqgs_lane, ZERO_F, NEG_INF, zero_output)

        def _run_valid_partition():
            for _mtp_g in range_constexpr(_mtp_groups):
                _mtp_subgroups = _get_sw_mtp_subgroup_count(query_length, query_group_size, _mtp_g)
                if const_expr(_mtp_g > 0):
                    gpu.barrier()
                mtp_states = _load_sw_mtp_group_q_fragments(
                    q_rsrc,
                    logits_lds_i32,
                    logits_lds_i64,
                    softmax_lds_f32,
                    batch_idx,
                    kv_h,
                    stride_q_seq,
                    stride_q_head,
                    lane16id,
                    lane4id,
                    rowid,
                    local_qhead_idx,
                    mtp_group_idx=_mtp_g,
                    mtp_subgroup_count=_mtp_subgroups,
                    query_length=query_length,
                    query_group_size=query_group_size,
                    query_packed_fp8=query_packed_fp8,
                    query_load_is_bf16=query_load_is_bf16,
                    compute_query_scale_in_kernel=query_scale_in_kernel,
                )
                gpu.barrier()

                # ── Load K for the single block ──
                k_block_rsrc, k_local_phys_block = _chunk_buffer_resource_for_block(key_cache_ptr, phys_block, c_kb)
                k_base = _compute_block_base_dw(k_local_phys_block, c_kb, _k_head_off)
                k0_flat = _load_k_flat(
                    k_block_rsrc,
                    k_base,
                    tile_token_offset,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                )
                k0_ops = _unflatten_k(k0_flat)

                if const_expr(_mtp_subgroups == 1):
                    qi_val, qhi_pos, q_frags, query_scale_lane = mtp_states[0]
                    causal_bound = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val
                    seq_start = context_len - arith.constant(query_length + sliding_window, type=T.i32) + qi_val

                    running_max, running_sum, out0, out1 = _process_block_split(
                        NEG_INF,
                        ZERO_F,
                        arith.constant_vector(0.0, T.f32x4),
                        arith.constant_vector(0.0, T.f32x4),
                        tile_token_offset,
                        k0_ops,
                        q_frags,
                        causal_bound,
                        query_scale_lane,
                        seq_start,
                    )
                    _store_group_results(qi_val, qhi_pos, running_sum, running_max, out0, out1)
                else:
                    qi_val_0, qhi_pos_0, q_frags_0, query_scale_lane_0 = mtp_states[0]
                    qi_val_1, qhi_pos_1, q_frags_1, query_scale_lane_1 = mtp_states[1]
                    causal_bound_0 = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val_0
                    causal_bound_1 = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val_1
                    seq_start_0 = context_len - arith.constant(query_length + sliding_window, type=T.i32) + qi_val_0
                    seq_start_1 = context_len - arith.constant(query_length + sliding_window, type=T.i32) + qi_val_1

                    (
                        running_max_0,
                        running_sum_0,
                        out0_0,
                        out1_0,
                        running_max_1,
                        running_sum_1,
                        out0_1,
                        out1_1,
                    ) = _process_block_split_pair(
                        NEG_INF,
                        ZERO_F,
                        arith.constant_vector(0.0, T.f32x4),
                        arith.constant_vector(0.0, T.f32x4),
                        NEG_INF,
                        ZERO_F,
                        arith.constant_vector(0.0, T.f32x4),
                        arith.constant_vector(0.0, T.f32x4),
                        tile_token_offset,
                        k0_ops,
                        q_frags_0,
                        q_frags_1,
                        causal_bound_0,
                        causal_bound_1,
                        query_scale_lane_0,
                        query_scale_lane_1,
                        seq_start_0,
                        seq_start_1,
                    )
                    _store_group_results(
                        qi_val_0,
                        qhi_pos_0,
                        running_sum_0,
                        running_max_0,
                        out0_0,
                        out1_0,
                    )
                    _store_group_results(
                        qi_val_1,
                        qhi_pos_1,
                        running_sum_1,
                        running_max_1,
                        out0_1,
                        out1_1,
                    )

        if partition_idx >= visible_tile_count:
            _write_empty_partition()

        if _is_valid:
            _run_valid_partition()

    @flyc.jit
    def launch_pa_decode_sw(
        es: fx.Tensor,
        ml: fx.Tensor,
        to: fx.Tensor,
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
            s_bt_seq,
            s_ks_block,
            s_ks_head,
        ).launch(grid=(gx, gy, gz), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        "launch": launch_pa_decode_sw,
        "kernel": pa_decode_sw_kernel,
        "allocator": allocator,
    }
