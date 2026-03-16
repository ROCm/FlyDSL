"""FlyDSL Paged Attention Decode — FP8, block_size=1024.

Port of Gluon paged_attention_decode_sliding_window_head_1 to FlyDSL.
Each CTA processes multiple partitions via scf.for loop with online softmax,
matching the Gluon split_idx decomposition and page-based partition assignment.

MFMA layout: mfma(K_lhs, Q_rhs) — hardware convention.
  lane16id (tid%16) = N = QGS head position
  rowid ((tid>>4)&3) = M groups offset (rowid*4)
  4 f32 elements = 4 KV token positions (rowid*4 + elem within tile)

Online softmax across partitions within each CTA.
PV: mfma(V_lhs, P_rhs) with CK LDS transpose for prob operand.
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
from flydsl.autotune import autotune, Config
from flydsl._mlir.dialects import arith as _mlir_arith

# ── Kernel geometry constants ────────────────────────────────────────
QUERY_GROUP_SIZE = 16
HEAD_SIZE = 128
KV_COMPUTE_BLOCK = 256   # partition size (CPS)
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
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * 4 * 8  # 8192
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4     # 512

FP8_MAX = 240.0
LOG2E = 1.4426950408889634


def _pack_i32_pair_to_i64(a_i32, b_i32):
    v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
    v1 = vector.bitcast(T.vec(1, T.i64), v)
    return vector.extract(v1, static_position=[0])


# =====================================================================
# compile_pa_decode_sw  — PA decode kernel (block_size=1024, FP8)
# =====================================================================
@functools.lru_cache(maxsize=1024)
def compile_pa_decode_sw(
    num_seqs,
    num_kv_heads,
    num_partitions,
    sliding_window,
    max_blocks_per_seq=256,
    softmax_scale=None,
    kv_block_size=1024,
    trans_v=False,
    one_shot=False,
):
    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _softmax_scale = float(softmax_scale)
    _sliding_window = int(sliding_window)
    _bs = kv_block_size

    # CONTEXT_PARTITION_SIZE_PER_BLOCK: how many CPS partitions fit in one KV block
    _cpb = math.ceil(kv_block_size / KV_COMPUTE_BLOCK)  # e.g. 4
    _grid_z = 1 if one_shot else num_partitions
    _out_num_parts = _grid_z
    _direct_output = one_shot

    # LDS allocation
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_sw_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_BYTES

    # ── @flyc.kernel ─────────────────────────────────────────────────
    @flyc.kernel
    def pa_decode_sw_kernel(
        out_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        key_cache_ptr: fx.Tensor,
        value_cache_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        context_lengths_ptr: fx.Tensor,
        query_scale_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        stride_q_seq: Int32,
        stride_q_head: Int32,
        stride_k_block: Int32,
        stride_k_head: Int32,
        stride_v_block: Int32,
        stride_v_head: Int32,
        stride_bt_seq: Int32,
        stride_out_seq: Int32,
        stride_out_head: Int32,
        stride_out_part: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_ml_seq: Int32,
    ):
        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        part_z = gpu.block_idx.z

        lane16id = tid & arith.constant(15, type=T.i32)
        lane4id = tid & arith.constant(3, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)
        laneid = tid & arith.constant(63, type=T.i32)

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        context_len = buffer_ops.buffer_load(cl_rsrc, seq, vec_width=1, dtype=T.i32)

        qs_rsrc = buffer_ops.create_buffer_resource(query_scale_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        q_scale_val = buffer_ops.buffer_load(qs_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_BYTES // 4,)).get()

        c_kb = stride_k_block
        c_kh = stride_k_head
        c_vb = stride_v_block
        c_vh = stride_v_head
        c_bt = stride_bt_seq

        _scale = arith.constant(_softmax_scale, type=T.f32) * q_scale_val * k_scale_val
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_cpb = arith.constant(_cpb, type=T.i32)

        local_qhead_idx = warp_id * arith.constant(4, type=T.i32) + rowid
        _k_head_off = kv_h * c_kh
        _v_head_off = kv_h * c_vh
        row_head_elem = rowid * arith.constant(FP8_ELEMS_16B, type=T.i32)

        # ════════════════════════════════════════════════════════════
        # Stage 1: Q load — global → LDS → Qlocal registers
        # ════════════════════════════════════════════════════════════
        q_base = seq * stride_q_seq + (kv_h * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + local_qhead_idx) * stride_q_head
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
        gpu.barrier()

        q_frags = []
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()
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

        # ════════════════════════════════════════════════════════════
        # split_idx decomposition (Gluon pattern)
        # ════════════════════════════════════════════════════════════
        sequence_split_idx = part_z // c_cpb
        block_split_idx = part_z % c_cpb

        grid_z_val = arith.constant(_grid_z, type=T.i32)
        num_seq_splits = grid_z_val // c_cpb
        page_size = (context_len + num_seq_splits - c_one) // num_seq_splits
        seq_start = page_size * sequence_split_idx
        seq_end_raw = page_size * (sequence_split_idx + c_one)
        seq_end = arith.select(seq_end_raw < context_len, seq_end_raw, context_len)

        part_start = seq_start // c_cps + block_split_idx
        part_end = (seq_end + c_cps - c_one) // c_cps + block_split_idx

        # ════════════════════════════════════════════════════════════
        # Helper: load K data for a partition index → flat list of _N_K i64
        # ════════════════════════════════════════════════════════════
        _N_K = TLOOP * QKHELOOP * 2  # 16

        def _load_k_flat(part_idx_i32):
            pstart = part_idx_i32 * c_cps
            k_flat = []
            for td in range_constexpr(TLOOP):
                klocal = (warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                          + arith.constant(td * MFMA_N, type=T.i32) + lane16id)
                kglobal = pstart + klocal
                kbi = kglobal // arith.constant(_bs, type=T.i32)
                kbo = kglobal % arith.constant(_bs, type=T.i32)
                kphys = buffer_ops.buffer_load(bt_rsrc, seq * c_bt + kbi,
                                               vec_width=1, dtype=T.i32)
                for qkhe in range_constexpr(QKHELOOP):
                    he = row_head_elem + arith.constant(qkhe * QKHE_PER_FETCH, type=T.i32)
                    o1 = he // arith.constant(FP8_ELEMS_16B, type=T.i32)
                    o2 = he % arith.constant(FP8_ELEMS_16B, type=T.i32)
                    ka = (kphys * c_kb + _k_head_off
                          + o1 * arith.constant(_bs * FP8_ELEMS_16B, type=T.i32)
                          + kbo * arith.constant(FP8_ELEMS_16B, type=T.i32) + o2)
                    k4 = buffer_ops.buffer_load(k_rsrc, ka // arith.constant(4, type=T.i32),
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

        # ════════════════════════════════════════════════════════════
        # Helper: compute one partition body (QK → softmax → PV)
        # Returns (running_max, running_sum, out0, out1)
        # ════════════════════════════════════════════════════════════
        def _compute_partition(partition_start, k_ops_all,
                               running_max, running_sum, out0, out1):
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

            exp_sum = ZERO_F
            for td in range_constexpr(TLOOP):
                for i in range_constexpr(4):
                    kv_tok = (partition_start
                              + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                              + rowid * arith.constant(4, type=T.i32)
                              + arith.constant(td * MFMA_N + i, type=T.i32))
                    s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                    diff = s - qk_max
                    p = arith.select(kv_tok < context_len,
                                     (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                                     ZERO_F)
                    exp_sum = exp_sum + p
                    d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
            for sh in [32, 16]:
                exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)

            # ── Cross-warp via LDS ──
            gpu.barrier()
            sm_max_off = arith.index_cast(T.index,
                warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            sm_sum_off = arith.index_cast(T.index,
                arith.constant(NUM_WARPS * MFMA_N, type=T.i32) + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            vector.store(vector.from_elements(T.vec(1, T.f32), [qk_max]), softmax_lds_f32, [sm_max_off])
            vector.store(vector.from_elements(T.vec(1, T.f32), [exp_sum]), softmax_lds_f32, [sm_sum_off])
            gpu.barrier()

            partition_max = NEG_INF
            partition_sum = ZERO_F
            warp_rescale_factors = []
            for w in range_constexpr(NUM_WARPS):
                rd_max_off = arith.index_cast(T.index,
                    arith.constant(w * MFMA_N, type=T.i32) + lane16id)
                w_max = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [rd_max_off]), static_position=[0])
                partition_max = partition_max.maximumf(w_max)
                warp_rescale_factors.append(w_max)
            for w in range_constexpr(NUM_WARPS):
                diff_w = warp_rescale_factors[w] - partition_max
                safe_diff = arith.select(partition_max > arith.constant(float("-inf"), type=T.f32),
                                         diff_w, ZERO_F)
                wf = (safe_diff * arith.constant(LOG2E, type=T.f32)).exp2(
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
            accum_scale = arith.select(
                running_max > arith.constant(float("-inf"), type=T.f32),
                ((running_max - new_running_max) * arith.constant(LOG2E, type=T.f32)).exp2(
                    fastmath=arith.FastMathFlags.fast),
                ZERO_F)
            part_to_new = arith.select(
                partition_max > arith.constant(float("-inf"), type=T.f32),
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
            gpu.barrier()
            for td in range_constexpr(TLOOP):
                p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
                p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
                p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
                p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
                lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
                pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                rowid_8x8 = rowid // arith.constant(2, type=T.i32)
                offset_in_slot = rowid % arith.constant(2, type=T.i32)
                byte_base = (warp_id * arith.constant(4 * MFMA_N * 4 * 8, type=T.i32)
                             + arith.constant(td * MFMA_N * 4 * 8, type=T.i32)
                             + lane16id * arith.constant(4 * 8, type=T.i32)
                             + rowid_8x8 * arith.constant(8, type=T.i32)
                             + offset_in_slot * arith.constant(4, type=T.i32))
                i32_off = byte_base // arith.constant(4, type=T.i32)
                pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
                vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])
            gpu.barrier()

            # ── V load + PV MFMA ──
            pv_results = [arith.constant_vector(0.0, T.f32x4)
                          for _ in range_constexpr(VHELOOP)]
            for vhe in range_constexpr(VHELOOP):
                vhead_elem = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                              + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
                tmp_out = arith.constant_vector(0.0, T.f32x4)
                for vt in range_constexpr(VTLOOP):
                    vtoken_base = (partition_start
                                   + arith.constant(vt * TOKENS_PER_WARP, type=T.i32)
                                   + rowid * arith.constant(MFMA_N, type=T.i32))
                    vblock_idx = vtoken_base // arith.constant(_bs, type=T.i32)
                    vblock_off = vtoken_base % arith.constant(_bs, type=T.i32)
                    vphys = buffer_ops.buffer_load(bt_rsrc,
                        seq * c_bt + vblock_idx, vec_width=1, dtype=T.i32)
                    if trans_v:
                        _vb = (vphys * c_vb + _v_head_off
                               + (vblock_off // arith.constant(FP8_ELEMS_16B, type=T.i32))
                                 * arith.constant(HEAD_SIZE * FP8_ELEMS_16B, type=T.i32)
                               + vhead_elem * arith.constant(FP8_ELEMS_16B, type=T.i32))
                    else:
                        _vb = (vphys * c_vb + _v_head_off
                               + vhead_elem * arith.constant(_bs, type=T.i32) + vblock_off)
                    v_4xi32 = buffer_ops.buffer_load(
                        v_rsrc, _vb // arith.constant(4, type=T.i32),
                        vec_width=4, dtype=T.i32)
                    for j in range_constexpr(2):
                        v_i64 = _pack_i32_pair_to_i64(
                            vector.extract(v_4xi32, static_position=[j * 2]),
                            vector.extract(v_4xi32, static_position=[j * 2 + 1]))
                        offset_raw = rowid * arith.constant(4, type=T.i32) + arith.constant(j * 2, type=T.i32)
                        p_off1 = (offset_raw % arith.constant(ROWS_PER_WARP, type=T.i32)) // arith.constant(2, type=T.i32)
                        p_off2 = offset_raw // arith.constant(ROWS_PER_WARP, type=T.i32)
                        p_byte = (arith.constant(vt * 4 * MFMA_N * 4 * 8, type=T.i32)
                                  + p_off2 * arith.constant(MFMA_N * 4 * 8, type=T.i32)
                                  + lane16id * arith.constant(4 * 8, type=T.i32)
                                  + p_off1 * arith.constant(8, type=T.i32))
                        p_i32_idx = p_byte // arith.constant(4, type=T.i32)
                        pw0 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                            [arith.index_cast(T.index, p_i32_idx)]), static_position=[0])
                        pw1 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                            [arith.index_cast(T.index, p_i32_idx + arith.constant(1, type=T.i32))]), static_position=[0])
                        p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                        tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [v_i64, p_i64, tmp_out, 0, 0, 0])
                pv_results[vhe] = tmp_out
            out0 = out0 + pv_results[0] * vector.broadcast(T.f32x4, v_scale_val)
            out1 = out1 + pv_results[1] * vector.broadcast(T.f32x4, v_scale_val)
            return running_max, running_sum, out0, out1

        # ════════════════════════════════════════════════════════════
        # scf.for loop with K prefetch (prologue + loop + epilogue)
        # Loop-carried state: [rmax, rsum, out0, out1, k_flat[_N_K]]
        # ════════════════════════════════════════════════════════════
        def _unwrap(v):
            return v.ir_value() if hasattr(v, 'ir_value') else v

        def _pack_state(rmax, rsum, o0, o1, k_flat):
            return [_unwrap(v) for v in [rmax, rsum, o0, o1] + k_flat]

        def _unpack_state(state):
            return state[0], state[1], state[2], state[3], list(state[4:4 + _N_K])

        # ── Prologue: load first partition's K data ──
        k0_flat = _load_k_flat(part_start)

        init_state = _pack_state(
            NEG_INF, ZERO_F,
            arith.constant_vector(0.0, T.f32x4),
            arith.constant_vector(0.0, T.f32x4),
            k0_flat)

        # Loop runs from part_start to part_end - _cpb (exclusive).
        # Each iteration: compute current partition + prefetch next K.
        # Epilogue handles the last partition.
        _loop_start = arith.index_cast(T.index, arith.unwrap(part_start))
        _loop_stop = arith.index_cast(T.index, arith.unwrap(part_end - c_cpb))
        _loop_step = arith.index(int(_cpb))

        for iv, state in range(_loop_start, _loop_stop, _loop_step,
                               init=init_state):
            running_max, running_sum, out0, out1, k_flat = _unpack_state(state)
            k_ops_all = _unflatten_k(k_flat)
            part_idx = arith.index_cast(T.i32, iv)
            partition_start = part_idx * c_cps

            running_max, running_sum, out0, out1 = _compute_partition(
                partition_start, k_ops_all,
                running_max, running_sum, out0, out1)

            # Prefetch next partition's K (overlaps with PV MFMA latency)
            next_part = part_idx + c_cpb
            k_next_flat = _load_k_flat(next_part)

            results = yield _pack_state(running_max, running_sum, out0, out1, k_next_flat)

        # ── Epilogue: compute last partition from loop results ──
        running_max, running_sum, out0, out1, k_last_flat = _unpack_state(results)
        k_ops_last = _unflatten_k(k_last_flat)
        last_part_idx = part_end - c_cpb
        last_partition_start = last_part_idx * c_cps

        running_max, running_sum, out0, out1 = _compute_partition(
            last_partition_start, k_ops_last,
            running_max, running_sum, out0, out1)

        # Normalize by running_sum
        safe_sum = arith.select(running_sum > ZERO_F, running_sum,
                                arith.constant(1.0, type=T.f32))
        inv_sum = arith.constant(1.0, type=T.f32) / safe_sum
        out0 = out0 * vector.broadcast(T.f32x4, inv_sum)
        out1 = out1 * vector.broadcast(T.f32x4, inv_sum)
        outelems = [out0, out1]

        # ── Stage 8: Output ──
        if _direct_output:
            for vhe in range_constexpr(VHELOOP):
                hs_base = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                           + warp_id * arith.constant(MFMA_N, type=T.i32)
                           + rowid * arith.constant(4, type=T.i32))
                qgs_pos = lane16id
                out_base = (seq * stride_out_seq
                            + (kv_h * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + qgs_pos) * stride_out_head
                            + hs_base)
                out_bf16 = arith.trunc_f(T.vec(4, T.bf16), outelems[vhe])
                out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                buffer_ops.buffer_store(out_i32, out_rsrc,
                    out_base * arith.constant(2, type=T.i32), offset_is_bytes=True)
        else:
            buffer_ops.buffer_store(running_max, ml_rsrc,
                seq * stride_ml_seq + kv_h * stride_es_head
                + part_z * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + lane16id)
            buffer_ops.buffer_store(running_sum, es_rsrc,
                seq * stride_es_seq + kv_h * stride_es_head
                + part_z * arith.constant(QUERY_GROUP_SIZE, type=T.i32) + lane16id)

            for vhe in range_constexpr(VHELOOP):
                hs_base = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                           + warp_id * arith.constant(MFMA_N, type=T.i32)
                           + rowid * arith.constant(4, type=T.i32))
                qgs_pos = lane16id
                out_base = (seq * stride_out_seq + kv_h * stride_out_head
                            + part_z * stride_out_part
                            + qgs_pos * arith.constant(HEAD_SIZE, type=T.i32) + hs_base)
                out_bf16 = arith.trunc_f(T.vec(4, T.bf16), outelems[vhe])
                out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                buffer_ops.buffer_store(out_i32, out_rsrc,
                    out_base * arith.constant(2, type=T.i32), offset_is_bytes=True)

    # ── @flyc.jit launch wrapper ─────────────────────────────────────
    _cache_tag = (num_seqs, num_kv_heads, _grid_z, _sliding_window)

    @autotune(
        configs=[Config(waves_per_eu=1), Config(waves_per_eu=2), Config(waves_per_eu=3), Config(waves_per_eu=4)],
        key=['gy', 'gz'], warmup=3, rep=10,
    )
    @flyc.jit
    def launch_pa_decode_sw(out, es, ml, q, kc, vc, bt, cl,
                            qs, ks, vs,
                            s_q_seq, s_q_head,
                            s_k_block, s_k_head,
                            s_v_block, s_v_head,
                            s_bt_seq,
                            s_out_seq, s_out_head, s_out_part,
                            s_es_seq, s_es_head, s_ml_seq,
                            gx, gy, gz,
                            stream: fx.Stream = fx.Stream(None)):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_sw_kernel(
            out, es, ml, q, kc, vc, bt, cl, qs, ks, vs,
            s_q_seq, s_q_head, s_k_block, s_k_head,
            s_v_block, s_v_head, s_bt_seq,
            s_out_seq, s_out_head, s_out_part,
            s_es_seq, s_es_head, s_ml_seq,
        ).launch(
            grid=(gx, gy, gz),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        'launch': launch_pa_decode_sw,
        'kernel': pa_decode_sw_kernel,
        'allocator': allocator,
        'grid_z': _grid_z,
        'direct_output': _direct_output,
        'out_num_parts': _out_num_parts,
    }


# =====================================================================
# Partition-sum reduce kernel
# =====================================================================
def _ext_f(target_type, value):
    return _mlir_arith.ExtFOp(target_type, value).result


@functools.lru_cache(maxsize=256)
def compile_ps_reduce(num_seqs, num_kv_heads, num_parts):
    QG = QUERY_GROUP_SIZE
    HS = HEAD_SIZE
    THREADS = BLOCK_THREADS

    _po_stride_seq = num_kv_heads * num_parts * QG * HS
    _po_stride_kv_h = num_parts * QG * HS
    _po_stride_part = QG * HS
    _es_stride_seq = num_kv_heads * num_parts * QG
    _es_stride_kv_h = num_parts * QG
    _es_stride_part = QG
    _out_stride_seq = num_kv_heads * QG * HS
    _out_stride_kv_h = QG * HS

    @flyc.kernel
    def ps_reduce_kernel(
        output_ptr: fx.Tensor,
        partial_out_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
    ):
        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        qg = tid >> 4
        hs_chunk = tid & 15

        out_rsrc = buffer_ops.create_buffer_resource(output_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        es_base = (seq * arith.constant(_es_stride_seq, type=T.i32)
                   + kv_h * arith.constant(_es_stride_kv_h, type=T.i32) + qg)
        po_base_bf16 = (seq * arith.constant(_po_stride_seq, type=T.i32)
                        + kv_h * arith.constant(_po_stride_kv_h, type=T.i32)
                        + qg * arith.constant(HS, type=T.i32)
                        + hs_chunk * arith.constant(8, type=T.i32))

        NEG_INF_C = arith.constant(float("-inf"), type=T.f32)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        ONE_F = arith.constant(1.0, type=T.f32)

        global_max = NEG_INF_C
        for p in range_constexpr(num_parts):
            ml_off = es_base + arith.constant(p * _es_stride_part, type=T.i32)
            ml_val = buffer_ops.buffer_load(ml_rsrc, ml_off, vec_width=1)
            global_max = global_max.maximumf(ml_val)

        total_sum = ZERO_F
        acc = arith.constant_vector(0.0, T.vec(8, T.f32))
        for p in range_constexpr(num_parts):
            es_off = es_base + arith.constant(p * _es_stride_part, type=T.i32)
            es_val = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1)
            ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1)
            w = es_val * ((ml_val - global_max) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
            total_sum = total_sum + w
            po_off_i32 = (po_base_bf16 + arith.constant(p * _po_stride_part, type=T.i32)) // arith.constant(2, type=T.i32)
            data_4xi32 = buffer_ops.buffer_load(po_rsrc, po_off_i32, vec_width=4, dtype=T.i32)
            data_8xbf16 = vector.bitcast(T.vec(8, T.bf16), data_4xi32)
            data_8xf32 = _ext_f(T.vec(8, T.f32), data_8xbf16)
            w_vec = vector.broadcast(T.vec(8, T.f32), w)
            acc = acc + data_8xf32 * w_vec

        rcp = ONE_F / total_sum
        result_8xf32 = acc * vector.broadcast(T.vec(8, T.f32), rcp)
        result_8xbf16 = arith.trunc_f(T.vec(8, T.bf16), result_8xf32)
        result_4xi32 = vector.bitcast(T.vec(4, T.i32), result_8xbf16)

        out_off_i32 = (seq * arith.constant(_out_stride_seq, type=T.i32)
                       + kv_h * arith.constant(_out_stride_kv_h, type=T.i32)
                       + qg * arith.constant(HS, type=T.i32)
                       + hs_chunk * arith.constant(8, type=T.i32)) // arith.constant(2, type=T.i32)
        buffer_ops.buffer_store(result_4xi32, out_rsrc, out_off_i32)

    @flyc.jit
    def launch_ps_reduce(out, partial_out, exp_sums, max_logits,
                         gx, gy, stream: fx.Stream = fx.Stream(None)):
        ps_reduce_kernel(out, partial_out, exp_sums, max_logits).launch(
            grid=(gx, gy, 1), block=(THREADS, 1, 1), stream=stream)

    return launch_ps_reduce


# =====================================================================
# Launch API
# =====================================================================
def pa_decode_sw_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    query_scale: torch.Tensor = None,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    sliding_window: int = 0,
    max_context_partition_num: int = 0,
    kv_block_size: int = 0,
    query_length: int = 1,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    num_query_heads = query.shape[1]
    head_size = query.shape[-1]
    batch_size = query.shape[0] // query_length
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads
    if kv_block_size == 0:
        kv_block_size = key_cache.shape[-2]
    blocks_per_seq = block_tables.shape[1]
    trans_v = len(value_cache.shape) == 5

    dev = query.device
    if not isinstance(query_scale, torch.Tensor):
        query_scale = torch.tensor([float(query_scale or 1.0)], device=dev, dtype=torch.float32)
    if not isinstance(key_scale, torch.Tensor):
        key_scale = torch.tensor([float(key_scale or 1.0)], device=dev, dtype=torch.float32)
    if not isinstance(value_scale, torch.Tensor):
        value_scale = torch.tensor([float(value_scale or 1.0)], device=dev, dtype=torch.float32)

    assert max_context_partition_num > 0
    if sliding_window > 0:
        sw_val = sliding_window
    else:
        sw_val = blocks_per_seq * kv_block_size

    one_shot = max_context_partition_num <= 1
    num_parts = 1 if one_shot else max_context_partition_num

    compiled = compile_pa_decode_sw(
        batch_size, num_kv_heads, num_parts,
        sliding_window=sw_val,
        max_blocks_per_seq=blocks_per_seq,
        softmax_scale=softmax_scale,
        kv_block_size=kv_block_size, trans_v=trans_v,
        one_shot=one_shot)

    grid_z = compiled['grid_z']
    direct = compiled['direct_output']
    s = stream or torch.cuda.current_stream()

    s_q_seq = query.stride(0)
    s_q_head = query.stride(1)
    s_k_block = key_cache.stride(0)
    s_k_head = key_cache.stride(1)
    s_v_block = value_cache.stride(0)
    s_v_head = value_cache.stride(1)
    s_bt_seq = block_tables.stride(0)

    output_5d = output

    if direct:
        if exp_sums is None:
            exp_sums = torch.empty(1, device=dev, dtype=torch.float32)
        if max_logits is None:
            max_logits = torch.empty(1, device=dev, dtype=torch.float32)
        compiled['launch'](
            output_5d, exp_sums, max_logits,
            query, key_cache, value_cache, block_tables,
            context_lengths, query_scale, key_scale, value_scale,
            s_q_seq, s_q_head, s_k_block, s_k_head,
            s_v_block, s_v_head, s_bt_seq,
            output_5d.stride(0), output_5d.stride(1), 0,
            0, 0, 0,
            batch_size, num_kv_heads, grid_z, s)
    else:
        eqgs = query_length * query_group_size
        if exp_sums is None:
            exp_sums = torch.empty(batch_size, num_kv_heads, num_parts, eqgs,
                                    device=dev, dtype=torch.float32)
        if max_logits is None:
            max_logits = torch.empty(batch_size, num_kv_heads, num_parts, eqgs,
                                      device=dev, dtype=torch.float32)
        if temporary_output is None:
            temporary_output = torch.empty(
                batch_size, num_kv_heads, num_parts, eqgs, head_size,
                device=dev, dtype=torch.bfloat16)

        compiled['launch'](
            temporary_output, exp_sums, max_logits,
            query, key_cache, value_cache, block_tables,
            context_lengths, query_scale, key_scale, value_scale,
            s_q_seq, s_q_head, s_k_block, s_k_head,
            s_v_block, s_v_head, s_bt_seq,
            temporary_output.stride(0), temporary_output.stride(1),
            temporary_output.stride(2),
            exp_sums.stride(0), exp_sums.stride(1), max_logits.stride(0),
            batch_size, num_kv_heads, grid_z, s)

        reduce_fn = compile_ps_reduce(batch_size, num_kv_heads, num_parts)
        reduce_fn(output_5d, temporary_output, exp_sums, max_logits,
                  batch_size, num_kv_heads, s)

    return "sw_one_shot" if one_shot else f"sw_partitioned({grid_z})"
