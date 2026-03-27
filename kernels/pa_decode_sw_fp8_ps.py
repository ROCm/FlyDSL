"""FlyDSL Paged Attention Decode with Persistent Scheduling — FP8.

Extends pa_decode_sw_fp8.py with persistent scheduling (PS) mode:
- Grid = (num_SM, 1, 1) instead of (batch_size, num_kv_heads, 1)
- Outer work loop iterates over pre-computed worklist from get_pa_metadata_v1
- Inner KV loop iterates pages from kv_page_indices instead of block_tables
- Supports split-reduce for load balancing across CUs

Requires: aiter's get_pa_metadata_v1 (module_pa_metadata.so)
"""

from __future__ import annotations
import math
import torch
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
try:
    from flydsl.autotune import autotune, Config
except ImportError:
    pass
from flydsl._mlir.dialects import arith as _mlir_arith

# ── Kernel geometry constants ────────────────────────────────────────
QUERY_GROUP_SIZE = 16
HEAD_SIZE = 128
KV_BLOCK_SIZE = 1024     # physical page size (matches SP3 kBlockSize)
KV_COMPUTE_BLOCK = 256   # tile size (matches SP3 kTileKV)
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256
MFMA_N = 16
MFMA_K = 32

TOKENS_PER_WARP = KV_COMPUTE_BLOCK // NUM_WARPS  # 64
TLOOP = TOKENS_PER_WARP // MFMA_N                # 4
ROWS_PER_WARP = WARP_SIZE // MFMA_N              # 4
FP8_ELEMS_16B = 16                                # 16 FP8 per 16-byte load
QKHE_PER_FETCH = FP8_ELEMS_16B * ROWS_PER_WARP   # 64
QKHELOOP = HEAD_SIZE // QKHE_PER_FETCH           # 2

VHELOOP = HEAD_SIZE // MFMA_N // NUM_WARPS        # 2
VTLOOP = NUM_WARPS                                # 4

# LDS sizes
PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4     # 512

FP8_MAX = 240.0
LOG2E = 1.4426950408889634

# Number of loop-carried K values (i64)
_N_K = TLOOP * QKHELOOP * 2  # 16
# Number of loop-carried V values (i64)
_N_V = VHELOOP * VTLOOP * 2  # 16

# Tiles per block (1024 tokens / 256 tokens per tile = 4, matches SP3 kNumBlockTiles)
TILES_PER_BLOCK = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK  # 4


def _pack_i32_pair_to_i64(a_i32, b_i32):
    v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
    v1 = vector.bitcast(T.vec(1, T.i64), v)
    return vector.extract(v1, static_position=[0])


# =====================================================================
# compile_pa_decode_ps — Persistent Scheduling PA decode kernel
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_ps(
    softmax_scale=None,
    trans_v=False,
    needs_mask=True,
):
    """Compile a PS-mode PA decode kernel.

    Unlike compile_pa_decode_sw, this does NOT bake in num_seqs/num_kv_heads/num_partitions
    because PS mode uses dynamic work distribution. Grid = (num_sm, 1, 1).
    """
    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024 for PS mode (matches SP3 kBlockSize)

    # Note: waves_per_eu=4 causes agpr=0 regression on current build (0c1805f).
    # Leave empty to let LLVM decide — gets agpr=128, vgpr=96, ~203us.
    CompilationContext._compile_hints.data = {}

    # LDS allocation
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_BYTES

    # ── @flyc.kernel ─────────────────────────────────────────────────
    @flyc.kernel
    def pa_decode_ps_kernel(
        out_ptr: fx.Tensor,           # output [batch, num_q_heads, head_size]
        partial_out_ptr: fx.Tensor,   # partial output [num_partials, 1, nhead, head_dim] fp32
        partial_lse_ptr: fx.Tensor,   # partial LSE [num_partials, 1, nhead, 1] fp32
        query_ptr: fx.Tensor,         # queries [batch, num_q_heads, head_size]
        key_cache_ptr: fx.Tensor,     # key cache
        value_cache_ptr: fx.Tensor,   # value cache
        context_lengths_ptr: fx.Tensor,  # [batch] int32
        query_scale_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        work_indptr_ptr: fx.Tensor,   # [num_sm + 1] int32
        work_info_ptr: fx.Tensor,     # [num_work, 8] int32 (flattened to 1D)
        kv_page_indices_ptr: fx.Tensor,  # [total_pages] int32
        stride_q_seq: Int32,
        stride_q_head: Int32,
        stride_k_block: Int32,
        stride_k_head: Int32,
        stride_v_block: Int32,
        stride_v_head: Int32,
        stride_out_seq: Int32,
        stride_out_head: Int32,
        stride_po_partial: Int32,     # stride for partial_output partial dim (nhead * head_dim)
        stride_pl_partial: Int32,     # stride for partial_lse partial dim (nhead)
    ):
        tid = gpu.thread_idx.x
        cu_id = gpu.block_idx.x  # CU index (0..num_sm-1)

        # ── Thread decomposition ──
        lane16id = tid & arith.constant(15, type=T.i32)
        lane4id = tid & arith.constant(3, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)
        laneid = tid & arith.constant(63, type=T.i32)

        # ── Buffer resources ──
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_out_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_lse_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        wi_rsrc = buffer_ops.create_buffer_resource(work_indptr_ptr, max_size=True)
        winfo_rsrc = buffer_ops.create_buffer_resource(work_info_ptr, max_size=True)
        kpi_rsrc = buffer_ops.create_buffer_resource(kv_page_indices_ptr, max_size=True)

        qs_rsrc = buffer_ops.create_buffer_resource(query_scale_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        q_scale_val = buffer_ops.buffer_load(qs_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        # ── LDS views ──
        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_BYTES // 4,)).get()
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()

        # ── Constants ──
        c_kb = stride_k_block
        c_kh = stride_k_head
        c_vb = stride_v_block
        c_vh = stride_v_head

        _scale = arith.constant(_softmax_scale, type=T.f32) * q_scale_val * k_scale_val
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_bs = arith.constant(_bs, type=T.i32)
        c_tpb = arith.constant(TILES_PER_BLOCK, type=T.i32)

        local_qhead_idx = warp_id * arith.constant(4, type=T.i32) + rowid
        row_head_elem = rowid * arith.constant(FP8_ELEMS_16B, type=T.i32)

        # ── Work loop bounds ──
        work_start = buffer_ops.buffer_load(wi_rsrc, cu_id, vec_width=1, dtype=T.i32)
        work_end = buffer_ops.buffer_load(wi_rsrc, cu_id + c_one, vec_width=1, dtype=T.i32)

        # ════════════════════════════════════════════════════════════
        # Outer work loop — iterate over assigned work items
        # Each work item = one (batch, kv_head_range, kv_page_range)
        # Uses a dummy loop-carried value since scf.for requires at least one
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
            # q_start = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(2, type=T.i32), vec_width=1, dtype=T.i32)
            # q_end = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(3, type=T.i32), vec_width=1, dtype=T.i32)
            kv_start = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(4, type=T.i32), vec_width=1, dtype=T.i32)
            kv_end = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(5, type=T.i32), vec_width=1, dtype=T.i32)
            q_head_range = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(7, type=T.i32), vec_width=1, dtype=T.i32)

            # Derive kv_head from q_head_range
            q_head_start = q_head_range & arith.constant(0xFFFF, type=T.i32)
            kv_h = q_head_start // arith.constant(QUERY_GROUP_SIZE, type=T.i32)

            # Context length for this sequence
            context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)

            # Head offsets for K and V cache
            _k_head_off = kv_h * c_kh
            _v_head_off = kv_h * c_vh

            # ── Q load for this work item ──
            q_base = batch_idx * stride_q_seq + (kv_h * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + local_qhead_idx) * stride_q_head
            q_off = q_base + lane16id * arith.constant(FP8_ELEMS_16B, type=T.i32)
            q_vec = buffer_ops.buffer_load(q_rsrc, q_off // arith.constant(4, type=T.i32), vec_width=4, dtype=T.i32)
            offset1 = lane16id // arith.constant(4, type=T.i32)
            lds_q_base = (offset1 * arith.constant(2048, type=T.i32)
                          + lane4id * arith.constant(512, type=T.i32)
                          + local_qhead_idx * arith.constant(32, type=T.i32))
            q_w0 = vector.extract(q_vec, static_position=[0], dynamic_position=[])
            q_w1 = vector.extract(q_vec, static_position=[1], dynamic_position=[])
            q_w2 = vector.extract(q_vec, static_position=[2], dynamic_position=[])
            q_w3 = vector.extract(q_vec, static_position=[3], dynamic_position=[])
            v01 = vector.from_elements(T.vec(2, T.i32), [q_w0, q_w1])
            v23 = vector.from_elements(T.vec(2, T.i32), [q_w2, q_w3])
            lds_q_i32 = lds_q_base // arith.constant(4, type=T.i32)
            vector.store(v01, logits_lds_i32, [arith.index_cast(T.index, lds_q_i32)])
            vector.store(v23, logits_lds_i32,
                [arith.index_cast(T.index, lds_q_i32 + arith.constant(2, type=T.i32))])

            q_frags = []
            gpu.barrier()
            for qkhe in range_constexpr(QKHELOOP):
                for qkr in range_constexpr(2):
                    lds_rd_byte = (arith.constant(qkhe * 2048, type=T.i32)
                                   + rowid * arith.constant(512, type=T.i32)
                                   + lane16id * arith.constant(32, type=T.i32)
                                   + arith.constant(qkr * 8, type=T.i32))
                    lds_rd_base = lds_rd_byte // arith.constant(8, type=T.i32)
                    q_v1 = vector.load_op(T.vec(1, T.i64), logits_lds_i64,
                        [arith.index_cast(T.index, lds_rd_base)])
                    q_frags.append(vector.extract(q_v1, static_position=[0]))

            # ── Tiles per block constant for tile-based loop (SP3-matching) ──
            c_four = arith.constant(4, type=T.i32)

            # ── Pre-computed thread-invariant offset components ──
            # K: thread base for token index (constant across all K loads)
            _k_tok_thread_base = warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32) + lane16id
            # K: stride constants in dwords (compile-time)
            _c_tok_stride_dw = arith.constant(FP8_ELEMS_16B // 4, type=T.i32)  # 4
            _c_he_stride_dw = arith.constant(_bs * FP8_ELEMS_16B // 4, type=T.i32)  # 4096
            # K: per-qkhe head element offset (only depends on rowid, compile-time qkhe)
            _k_he_off_dw = [rowid * _c_he_stride_dw + arith.constant(qkhe * 4, type=T.i32) * _c_he_stride_dw
                            for qkhe in range(QKHELOOP)]

            # V: thread-invariant head element for each vhe (constant across all V loads)
            _vhead_elems = [arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                            + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id
                            for vhe in range(VHELOOP)]
            # V: thread-invariant token offsets for each vt (only rowid varies)
            _v_tok_thread_off = [arith.constant(vt * TOKENS_PER_WARP, type=T.i32)
                                 + rowid * arith.constant(MFMA_N, type=T.i32)
                                 for vt in range(VTLOOP)]
            # V: pre-computed per-vhe head element dword offsets (for trans_v)
            if trans_v:
                _vhead_elem_dw = [_vhead_elems[vhe] * arith.constant(FP8_ELEMS_16B // 4, type=T.i32)
                                  for vhe in range(VHELOOP)]
            else:
                _vhead_elem_dw = [_vhead_elems[vhe] * arith.constant(_bs // 4, type=T.i32)
                                  for vhe in range(VHELOOP)]

            # Softmax: thread-invariant kv_tok base (partition_start added at call site)
            _kv_tok_thread_base = warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32) + rowid * c_four

            # Prob pack: thread-invariant LDS write base
            _rowid_8x8 = rowid // arith.constant(2, type=T.i32)
            _offset_in_slot = rowid % arith.constant(2, type=T.i32)
            _prob_wr_thread_base = (warp_id * arith.constant(4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                    + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                    + _rowid_8x8 * arith.constant(8, type=T.i32)
                                    + _offset_in_slot * c_four)

            # PV prob read: thread-invariant LDS read base for j=0 and j=1
            # j=0: offset_raw=rowid*4, p_off1=0, p_off2=rowid → base = rowid*640 + lane16id*40
            # j=1: offset_raw=rowid*4+2, p_off1=1, p_off2=rowid → base = rowid*640 + lane16id*40 + 8
            _pv_prob_read_base = (rowid * arith.constant(MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                  + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32))

            # LDS softmax offsets (thread-invariant)
            _sm_max_off = arith.index_cast(T.index,
                warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            _sm_sum_off = arith.index_cast(T.index,
                arith.constant(NUM_WARPS * MFMA_N, type=T.i32) + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            # Cross-warp softmax read offsets (compile-time warp index)
            _sm_rd_max_offs = [arith.index_cast(T.index,
                arith.constant(w * MFMA_N, type=T.i32) + lane16id) for w in range(NUM_WARPS)]
            _sm_rd_sum_offs = [arith.index_cast(T.index,
                arith.constant(NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id) for w in range(NUM_WARPS)]

            # ════════════════════════════════════════════════════════
            # Helper: load K data for one tile using pre-computed block base
            # k_block_base_dw = (phys_block * c_kb + _k_head_off) // 4
            # tile_token_offset_i32 = token offset within block (0, 256, 512, 768)
            # ════════════════════════════════════════════════════════
            def _load_k_flat_ps(k_block_base_dw, tile_token_offset_i32):
                """Load K data for one tile using pre-computed block base address."""
                k_flat = []
                tile_tok_base = tile_token_offset_i32 + _k_tok_thread_base

                for td in range_constexpr(TLOOP):
                    kbo = tile_tok_base + arith.constant(td * MFMA_N, type=T.i32)
                    kbo_dw = k_block_base_dw + kbo * _c_tok_stride_dw
                    for qkhe in range_constexpr(QKHELOOP):
                        ka_dw = kbo_dw + _k_he_off_dw[qkhe]
                        k4 = buffer_ops.buffer_load(k_rsrc, ka_dw,
                                                    vec_width=4, dtype=T.i32)
                        k_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(k4, static_position=[0]),
                            vector.extract(k4, static_position=[1])))
                        k_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(k4, static_position=[2]),
                            vector.extract(k4, static_position=[3])))
                return k_flat

            def _unflatten_k(k_flat):
                return [[k_flat[td * (QKHELOOP * 2) + j]
                          for j in range(QKHELOOP * 2)]
                         for td in range(TLOOP)]

            def _unpack_i64_to_i32_pair(i64_val):
                """Reverse of _pack_i32_pair_to_i64."""
                v1 = vector.from_elements(T.vec(1, T.i64), [i64_val])
                v2 = vector.bitcast(T.vec(2, T.i32), v1)
                return vector.extract(v2, static_position=[0]), vector.extract(v2, static_position=[1])

            # ════════════════════════════════════════════════════════
            # Helper: load V data for one tile — returns vec4<i32> directly (no i64)
            # ════════════════════════════════════════════════════════
            def _load_v_ops(v_block_base_dw, tile_token_offset_i32):
                """Load V data for one tile using pre-computed block base and vhead_elems."""
                result = []
                for vhe in range_constexpr(VHELOOP):
                    vhe_data = []
                    for vt in range_constexpr(VTLOOP):
                        v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                        if trans_v:
                            vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                            va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                        else:
                            va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                        v_4xi32 = buffer_ops.buffer_load(
                            v_rsrc, va_dw,
                            vec_width=4, dtype=T.i32)
                        vhe_data.append(v_4xi32)
                    result.append(vhe_data)
                return result

            def _load_v_flat_ps(v_block_base_dw, tile_token_offset_i32):
                """Load V data for one tile as flat i64 list using pre-computed block base."""
                v_flat = []
                for vhe in range_constexpr(VHELOOP):
                    for vt in range_constexpr(VTLOOP):
                        v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                        if trans_v:
                            vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                            va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                        else:
                            va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                        v_4xi32 = buffer_ops.buffer_load(
                            v_rsrc, va_dw,
                            vec_width=4, dtype=T.i32)
                        v_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(v_4xi32, static_position=[0]),
                            vector.extract(v_4xi32, static_position=[1])))
                        v_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(v_4xi32, static_position=[2]),
                            vector.extract(v_4xi32, static_position=[3])))
                return v_flat

            def _unflatten_v(v_flat):
                """Convert flat i64 list back to v_data[vhe][vt] = vec4<i32>."""
                result = []
                fi = 0
                for _vhe in [0, 1]:  # VHELOOP=2, use literal to avoid DSL range()
                    vhe_data = []
                    for _vt in [0, 1, 2, 3]:  # VTLOOP=4
                        lo0, lo1 = _unpack_i64_to_i32_pair(v_flat[fi])
                        hi0, hi1 = _unpack_i64_to_i32_pair(v_flat[fi + 1])
                        v_4xi32 = vector.from_elements(T.vec(4, T.i32), [lo0, lo1, hi0, hi1])
                        vhe_data.append(v_4xi32)
                        fi = fi + 2
                    result.append(vhe_data)
                return result

            # ════════════════════════════════════════════════════════
            # Helper: compute one tile body (QK → softmax → PV)
            # phys_block_i32 = physical block index (from kv_page_indices)
            # tile_token_offset_i32 = token offset within block (0, 256, 512, 768)
            # partition_start = global token offset for masking
            # ════════════════════════════════════════════════════════
            def _compute_partition_ps(phys_block_i32, tile_token_offset_i32,
                                      partition_start, k_ops_all, v_ops_all,
                                      running_max, running_sum, out0, out1,
                                      prefetch_k_block=None, prefetch_k_tile_off=None,
                                      prefetch_v_block=None, prefetch_v_tile_off=None):

                # V data passed in as v_ops_all[vhe][vt] = vec4<i32>
                v_data_prefetched = v_ops_all
                # ── K prefetch for next tile (after prob pack, overlaps PV) ──
                if prefetch_k_block is not None:
                    k_prefetched = _load_k_flat_ps(prefetch_k_block, prefetch_k_tile_off)
                else:
                    k_prefetched = None
                
                # ── QK MFMA ──
                d_out = []
                for td in range_constexpr(TLOOP):
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for k_step in range_constexpr(QKHELOOP * 2):
                        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [k_ops_all[td][k_step], q_frags[k_step], acc, 0, 0, 0])
                    d_out.append(acc * vector.broadcast(T.f32x4, _scale))

                # ── Intra-warp softmax ──
                qk_max = NEG_INF
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        kv_tok = (partition_start
                                  + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                                  + rowid * arith.constant(4, type=T.i32)
                                  + arith.constant(td * MFMA_N + i, type=T.i32))
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        s = arith.select(kv_tok < context_len, s, NEG_INF)
                        qk_max = qk_max.maximumf(s)
                for sh in [32, 16]:
                    qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))

                sm_max_off = arith.index_cast(T.index,
                    warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
                vector.store(vector.from_elements(T.vec(1, T.f32), [qk_max]), softmax_lds_f32, [sm_max_off])
                exp_sum = ZERO_F
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        kv_tok = (partition_start
                                  + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                                  + rowid * arith.constant(4, type=T.i32)
                                  + arith.constant(td * MFMA_N + i, type=T.i32))
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        diff = s - qk_max
                        p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                        p = arith.select(kv_tok < context_len, p, ZERO_F)
                        exp_sum = exp_sum + p
                        d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
                for sh in [32, 16]:
                    exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)

                # ── Cross-warp via LDS ──
                sm_sum_off = arith.index_cast(T.index,
                    arith.constant(NUM_WARPS * MFMA_N, type=T.i32) + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
                vector.store(vector.from_elements(T.vec(1, T.f32), [exp_sum]), softmax_lds_f32, [sm_sum_off])

                partition_max = NEG_INF
                partition_sum = ZERO_F
                warp_rescale_factors = []
                rd_max_offs = []
                for w in range_constexpr(NUM_WARPS):
                    rd_max_off = arith.index_cast(T.index,
                        arith.constant(w * MFMA_N, type=T.i32) + lane16id)
                    rd_max_offs.append(rd_max_off)
                gpu.barrier()
                for w in range_constexpr(NUM_WARPS):
                    w_max = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [rd_max_offs[w]]), static_position=[0])
                    partition_max = partition_max.maximumf(w_max)
                    warp_rescale_factors.append(w_max)
                for w in range_constexpr(NUM_WARPS):
                    diff_w = warp_rescale_factors[w] - partition_max
                    diff_w = arith.select(partition_max > NEG_INF, diff_w, ZERO_F)
                    wf = (diff_w * arith.constant(LOG2E, type=T.f32)).exp2(
                        fastmath=arith.FastMathFlags.fast)
                    rd_sum_off = arith.index_cast(T.index,
                        arith.constant(NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id)
                    w_sum = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [rd_sum_off]), static_position=[0])
                    partition_sum = partition_sum + w_sum * wf
                    warp_rescale_factors[w] = wf
                my_warp_rescale = warp_rescale_factors[0]
                for w in range_constexpr(1, NUM_WARPS):
                    my_warp_rescale = arith.select(
                        warp_id == arith.constant(w, type=T.i32),
                        warp_rescale_factors[w], my_warp_rescale)

                # ── Online softmax update ──
                new_running_max = running_max.maximumf(partition_max)
                accum_scale = arith.select(running_max > NEG_INF,
                    ((running_max - new_running_max) * arith.constant(LOG2E, type=T.f32)).exp2(
                        fastmath=arith.FastMathFlags.fast),
                    ZERO_F)
                part_to_new = arith.select(partition_max > NEG_INF,
                    ((partition_max - new_running_max) * arith.constant(LOG2E, type=T.f32)).exp2(
                        fastmath=arith.FastMathFlags.fast),
                    ZERO_F)
                running_sum = accum_scale * running_sum + partition_sum * part_to_new
                running_max = new_running_max
                out0 = out0 * vector.broadcast(T.f32x4, accum_scale)
                out1 = out1 * vector.broadcast(T.f32x4, accum_scale)

                # ── Prob FP8 pack + LDS store ──
                prob_scale = my_warp_rescale * part_to_new
                for td in range_constexpr(TLOOP):
                    d_out[td] = d_out[td] * vector.broadcast(T.f32x4, prob_scale)
                for td in range_constexpr(TLOOP):
                    p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
                    p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
                    p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
                    p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
                    lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
                    pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                    rowid_8x8 = rowid // arith.constant(2, type=T.i32)
                    offset_in_slot = rowid % arith.constant(2, type=T.i32)
                    byte_base = (warp_id * arith.constant(4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                 + arith.constant(td * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                 + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                 + rowid_8x8 * arith.constant(8, type=T.i32)
                                 + offset_in_slot * arith.constant(4, type=T.i32))
                    i32_off = byte_base // arith.constant(4, type=T.i32)
                    pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
                    vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])



                # ── V prefetch for next tile (flat i64 for loop state) ──
                if prefetch_v_block is not None:
                    v_prefetched = _load_v_flat_ps(prefetch_v_block, prefetch_v_tile_off)
                else:
                    v_prefetched = None
                
                # ── PV MFMA ──
                pv_results = [arith.constant_vector(0.0, T.f32x4)
                              for _ in range_constexpr(VHELOOP)]
                v_i64s = []
                p_i64s = []
                for vhe in range_constexpr(VHELOOP):
                    for vt in range_constexpr(VTLOOP):
                        v_4xi32 = v_data_prefetched[vhe][vt]
                        for j in range_constexpr(2):
                            v_i64 = _pack_i32_pair_to_i64(
                                vector.extract(v_4xi32, static_position=[j * 2]),
                                vector.extract(v_4xi32, static_position=[j * 2 + 1]))
                            v_i64s.append(v_i64)
                            offset_raw = rowid * arith.constant(4, type=T.i32) + arith.constant(j * 2, type=T.i32)
                            p_off1 = (offset_raw % arith.constant(ROWS_PER_WARP, type=T.i32)) // arith.constant(2, type=T.i32)
                            p_off2 = offset_raw // arith.constant(ROWS_PER_WARP, type=T.i32)
                            p_byte = (arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + p_off2 * arith.constant(MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + p_off1 * arith.constant(8, type=T.i32))
                            p_i32_idx = p_byte // arith.constant(4, type=T.i32)
                            if vhe == 0 and vt == 0 and j == 0:
                                gpu.barrier()
                            pw0 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx)]), static_position=[0])
                            pw1 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx + arith.constant(1, type=T.i32))]), static_position=[0])
                            p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                            p_i64s.append(p_i64)
                for vhe in range_constexpr(VHELOOP):
                    tmp_out = arith.constant_vector(0.0, T.f32x4)
                    for vt in range_constexpr(VTLOOP):
                        for j in range_constexpr(2):
                            tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [v_i64s[vhe * VTLOOP * 2 + vt * 2 + j], p_i64s[vhe * VTLOOP * 2 + vt * 2 + j], tmp_out, 0, 0, 0])
                            pv_results[vhe] = tmp_out
                out0 = out0 + pv_results[0] * vector.broadcast(T.f32x4, v_scale_val)
                out1 = out1 + pv_results[1] * vector.broadcast(T.f32x4, v_scale_val)
                
                return running_max, running_sum, out0, out1, k_prefetched, v_prefetched

            # ════════════════════════════════════════════════════════
            # Inner KV loop — 1 tile per iteration (4x longer loop)
            # Each iteration: QK → barrier → cross-warp softmax + prob pack
            #                 → barrier → PV MFMA
            # 2 barriers per tile
            # ════════════════════════════════════════════════════════
            def _unwrap(v):
                return v.ir_value() if hasattr(v, 'ir_value') else v

            def _pack_state_1t(rmax, rsum, o0, o1, k_flat, v_flat):
                return [_unwrap(v) for v in [rmax, rsum, o0, o1] + k_flat + v_flat]

            def _unpack_state_1t(state):
                return (state[0], state[1], state[2], state[3],
                        list(state[4:4 + _N_K]),
                        list(state[4 + _N_K:4 + _N_K + _N_V]))

            # ── Prologue: load first tile's K + V data ──
            first_phys_block = buffer_ops.buffer_load(kpi_rsrc, kv_start,
                                                       vec_width=1, dtype=T.i32)
            c_zero = arith.constant(0, type=T.i32)

            first_k_base = (first_phys_block * c_kb + _k_head_off) // c_four
            k_init_flat = _load_k_flat_ps(first_k_base, c_zero)

            first_v_base = (first_phys_block * c_vb + _v_head_off) // c_four
            v_init_flat = _load_v_flat_ps(first_v_base, c_zero)

            init_state = _pack_state_1t(
                NEG_INF, ZERO_F,
                arith.constant_vector(0.0, T.f32x4),
                arith.constant_vector(0.0, T.f32x4),
                k_init_flat, v_init_flat)

            # Loop over tiles: total_tiles = num_blocks * TILES_PER_BLOCK
            num_blocks_in_work = kv_end - kv_start
            num_tiles_in_work = num_blocks_in_work * c_tpb
            last_tile_idx_val = num_tiles_in_work - c_one
            _loop_start = arith.index(0)
            _loop_stop = arith.index_cast(T.index, arith.unwrap(num_tiles_in_work))
            _loop_step = arith.index(1)

            for it, state in range(_loop_start, _loop_stop, _loop_step,
                                   init=init_state):
                running_max, running_sum, out0, out1, k_flat, v_flat = _unpack_state_1t(state)
                tile_idx = arith.index_cast(T.i32, it)

                # Unflatten prefetched K and V data
                k_ops = _unflatten_k(k_flat)
                v_ops = _unflatten_v(v_flat)  # [vhe][vt]

                # Partition start = global tile index * KV_COMPUTE_BLOCK
                partition_start = tile_idx * c_cps

                # Next tile prefetch addresses
                next_tile_idx_raw = tile_idx + c_one
                next_tile_idx_clamped = arith.select(
                    next_tile_idx_raw < num_tiles_in_work, next_tile_idx_raw, last_tile_idx_val)
                next_block_idx = next_tile_idx_clamped // c_tpb
                next_tile_in_block = next_tile_idx_clamped % c_tpb
                next_phys_block = buffer_ops.buffer_load(kpi_rsrc,
                    kv_start + next_block_idx, vec_width=1, dtype=T.i32)
                next_k_base = (next_phys_block * c_kb + _k_head_off) // c_four
                next_v_base = (next_phys_block * c_vb + _v_head_off) // c_four
                next_tile_token_offset = next_tile_in_block * c_cps

                # Full tile: QK → barrier → cross-warp softmax + prob pack
                #          → K/V prefetch → barrier → PV MFMA
                running_max, running_sum, out0, out1, k_next_flat, v_next_flat = \
                    _compute_partition_ps(
                        None, None, partition_start, k_ops, v_ops,
                        running_max, running_sum, out0, out1,
                        prefetch_k_block=next_k_base,
                        prefetch_k_tile_off=next_tile_token_offset,
                        prefetch_v_block=next_v_base,
                        prefetch_v_tile_off=next_tile_token_offset)

                results = yield _pack_state_1t(running_max, running_sum, out0, out1,
                                               k_next_flat, v_next_flat)

            # All tiles processed
            running_max, running_sum, out0, out1, _, _ = _unpack_state_1t(results)

            # ── Output: branch on partial_idx ──
            is_direct = partial_idx == arith.constant(-1, type=T.i32)
            is_partial = partial_idx >= c_zero
            # Offset partial_idx by 1: slot 0 is scratch for direct items
            po_idx = partial_idx + c_one  # -1 → 0 (scratch), 0 → 1, 1 → 2, ...

            # --- Normalize output ---
            safe_sum = arith.select(running_sum > ZERO_F, running_sum,
                                    arith.constant(1.0, type=T.f32))
            inv_sum = arith.constant(1.0, type=T.f32) / safe_sum
            out0_norm = out0 * vector.broadcast(T.f32x4, inv_sum)
            out1_norm = out1 * vector.broadcast(T.f32x4, inv_sum)
            outelems_norm = [out0_norm, out1_norm]
            outelems_raw = [out0, out1]

            for vhe in range_constexpr(VHELOOP):
                hs_base = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                           + warp_id * arith.constant(MFMA_N, type=T.i32)
                           + rowid * arith.constant(4, type=T.i32))
                qgs_pos = lane16id
                qhead = kv_h * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + qgs_pos

                out_base = (batch_idx * stride_out_seq
                            + qhead * stride_out_head
                            + hs_base)
                # Partial output offset uses po_idx (shifted by 1)
                po_off = (po_idx * stride_po_partial
                          + qhead * arith.constant(HEAD_SIZE, type=T.i32)
                          + hs_base)

                # -- ALWAYS write normalized bf16 to output (for debug) --
                out_bf16 = arith.trunc_f(T.vec(4, T.bf16), outelems_norm[vhe])
                out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                buffer_ops.buffer_store(out_i32, out_rsrc,
                    out_base * arith.constant(2, type=T.i32), offset_is_bytes=True)

                # -- Partial: NORMALIZED fp32 to partial output buffer --
                # pa_reduce_v1 expects normalized output (like SP3: R = R/L before store)
                buffer_ops.buffer_store(outelems_norm[vhe], po_rsrc,
                    po_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

            # -- LSE --
            safe_sum_lse = arith.select(running_sum > ZERO_F, running_sum,
                                         arith.constant(1.0, type=T.f32))
            from flydsl._mlir.dialects import math as _mlir_math
            log_sum = _mlir_math.log(safe_sum_lse, fastmath=arith.FastMathFlags.fast)
            lse_val = running_max + log_sum
            qhead_lse = kv_h * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + lane16id
            pl_off = po_idx * stride_pl_partial + qhead_lse
            lse_as_i32 = arith.bitcast(T.i32, lse_val)
            buffer_ops.buffer_store(lse_as_i32, pl_rsrc,
                pl_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

    # ── @flyc.jit launch wrapper ─────────────────────────────────────
    @flyc.jit
    def launch_pa_decode_ps(out, po, pl, q, kc, vc, cl,
                            qs, ks, vs,
                            work_indptr, work_info, kv_page_indices,
                            s_q_seq, s_q_head,
                            s_k_block, s_k_head,
                            s_v_block, s_v_head,
                            s_out_seq, s_out_head,
                            s_po_partial, s_pl_partial,
                            num_sm,
                            stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_ps_kernel(
            out, po, pl, q, kc, vc, cl, qs, ks, vs,
            work_indptr, work_info, kv_page_indices,
            s_q_seq, s_q_head, s_k_block, s_k_head,
            s_v_block, s_v_head,
            s_out_seq, s_out_head,
            s_po_partial, s_pl_partial,
        ).launch(
            grid=(num_sm, 1, 1),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        'launch': launch_pa_decode_ps,
        'kernel': pa_decode_ps_kernel,
        'allocator': allocator,
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

    Returns a dict with: work_indptr, work_info_flat, reduce_indptr,
    reduce_final_map, reduce_partial_map, num_sm, partial_output,
    partial_lse, stride_po_partial, stride_pl_partial.
    """
    import aiter

    dev = query.device
    batch_size = query.shape[0]
    head_size = query.shape[-1]

    props = torch.cuda.get_device_properties(dev)
    num_sm = props.multi_processor_count

    seqlens_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=dev)

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
        seqlens_qo_indptr, kv_indptr, context_lengths,
        num_query_heads // num_kv_heads, num_kv_heads, True,
        work_metadata_ptrs, work_indptr, work_info,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        kv_granularity=max(block_size, 16), block_size=block_size,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=True, max_split_per_batch=-1,
    )

    work_info_flat = work_info.reshape(-1).contiguous()

    num_partials = reduce_partial_map.size(0)
    max_qlen = 1
    partial_output = torch.empty(
        ((num_partials + 1) * max_qlen, 1, num_query_heads, head_size),
        dtype=torch.float32, device=dev)
    partial_lse = torch.empty(
        ((num_partials + 1) * max_qlen, 1, num_query_heads, 1),
        dtype=torch.float32, device=dev)

    stride_po_partial = num_query_heads * head_size
    stride_pl_partial = num_query_heads

    return {
        'work_indptr': work_indptr,
        'work_info_flat': work_info_flat,
        'reduce_indptr': reduce_indptr,
        'reduce_final_map': reduce_final_map,
        'reduce_partial_map': reduce_partial_map,
        'num_sm': num_sm,
        'partial_output': partial_output,
        'partial_lse': partial_lse,
        'stride_po_partial': stride_po_partial,
        'stride_pl_partial': stride_pl_partial,
    }


def pa_decode_ps_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,    # [total_pages] int32
    kv_indptr: torch.Tensor,          # [num_seqs + 1] int32
    softmax_scale: float,
    query_scale: torch.Tensor = None,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    metadata: dict = None,
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

    dev = query.device
    if not isinstance(query_scale, torch.Tensor):
        query_scale = torch.tensor([float(query_scale or 1.0)], device=dev, dtype=torch.float32)
    if not isinstance(key_scale, torch.Tensor):
        key_scale = torch.tensor([float(key_scale or 1.0)], device=dev, dtype=torch.float32)
    if not isinstance(value_scale, torch.Tensor):
        value_scale = torch.tensor([float(value_scale or 1.0)], device=dev, dtype=torch.float32)

    if metadata is None:
        metadata = get_pa_metadata(
            query, key_cache, context_lengths, kv_indptr,
            num_query_heads, num_kv_heads)

    work_indptr = metadata['work_indptr']
    work_info_flat = metadata['work_info_flat']
    partial_output = metadata['partial_output']
    partial_lse = metadata['partial_lse']
    stride_po_partial = metadata['stride_po_partial']
    stride_pl_partial = metadata['stride_pl_partial']
    num_sm = metadata['num_sm']

    compiled = compile_pa_decode_ps(
        softmax_scale=softmax_scale, trans_v=trans_v)

    s = stream or torch.cuda.current_stream()

    compiled['launch'](
        output, partial_output, partial_lse,
        query, key_cache, value_cache,
        context_lengths, query_scale, key_scale, value_scale,
        work_indptr, work_info_flat, kv_page_indices,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(1),
        output.stride(0), output.stride(1),
        stride_po_partial, stride_pl_partial,
        num_sm, s)

    from aiter.ops.attention import pa_reduce_v1
    pa_reduce_v1(
        partial_output[1:],
        partial_lse[1:],
        metadata['reduce_indptr'],
        metadata['reduce_final_map'],
        metadata['reduce_partial_map'],
        1,  # max_qlen
        output,
        None,
    )

    return "ps_split_reduce"
