"""FlyDSL Paged Attention Decode with Sliding Window — FP8.

Port of paged_attention_decode_sliding_window_head_1 from Gluon (aiter).
Uses CuTe layout algebra: fx.make_layout, fx.slice, fx.memref_alloca,
fx.make_tiled_mma, fx.gemm, rocdl.make_buffer_tensor, etc.

Supports:
  - kv_block_size=16 (small) and kv_block_size=1024 (large, trans_v required)
  - sliding_window>0: restrict attention to the most recent tokens
  - one_shot and partitioned modes
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
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

# ── Kernel geometry constants ────────────────────────────────────────
QUERY_GROUP_SIZE = 16
HEAD_SIZE = 128
KV_BLOCK_SIZE = 16
KV_COMPUTE_BLOCK = 256
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
MFMA_M = MFMA_N = 16
MFMA_K = 32

QK_N_TILES_WARP = (KV_COMPUTE_BLOCK // NUM_WARPS) // MFMA_N  # 4
QK_K_STEPS = HEAD_SIZE // MFMA_K  # 4
PV_K_STEPS = KV_COMPUTE_BLOCK // MFMA_K  # 8
PV_N_TILES_WARP = (HEAD_SIZE // NUM_WARPS) // MFMA_N  # 2

Q_LDS_BYTES = BLOCK_THREADS * 8
PROB_LDS_BYTES = BLOCK_THREADS * 16
BT_LDS_BYTES = NUM_WARPS * 16
RED_SLOTS = NUM_WARPS
FP8_MAX = 240.0
LOG2E = 1.4426950408889634


def _pack_i32_pair_to_i64(a_i32, b_i32):
    v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
    v1 = vector.bitcast(T.vec(1, T.i64), v)
    return vector.extract(v1, static_position=[0])


# =====================================================================
# compile_pa_decode_sw  — sliding-window PA decode kernel
# =====================================================================
@functools.lru_cache(maxsize=1024)
def compile_pa_decode_sw(
    num_seqs,
    num_kv_heads,
    num_partitions,
    sliding_window,
    max_blocks_per_seq=256,
    softmax_scale=None,
    kv_block_size=16,
    trans_v=False,
    one_shot=False,
):
    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _softmax_scale = float(softmax_scale)
    _sliding_window = int(sliding_window)

    _bs = kv_block_size
    _num_heads = num_kv_heads * QUERY_GROUP_SIZE
    _use_large_block = _bs > KV_BLOCK_SIZE
    _partitions_per_block = _bs // KV_COMPUTE_BLOCK if _use_large_block else 1
    _blocks_per_partition = KV_COMPUTE_BLOCK // _bs if not _use_large_block else 1

    # ── Stride layouts (compile-time, CuTe-style) ────────────────────
    # Q: [num_seqs, num_heads, HEAD_SIZE]  row-major
    _stride_q_head = HEAD_SIZE
    _stride_q_seq = _num_heads * HEAD_SIZE

    # K cache: [num_blocks, num_kv_heads, HEAD_SIZE//16, bs, 16]  row-major
    _stride_k_block = num_kv_heads * (HEAD_SIZE // 16) * _bs * 16
    _stride_k_head = (HEAD_SIZE // 16) * _bs * 16

    # V cache: [num_blocks, num_kv_heads, HEAD_SIZE, bs]  or transposed
    if trans_v:
        _stride_v_block = num_kv_heads * (_bs // 16) * HEAD_SIZE * 16
        _stride_v_head = (_bs // 16) * HEAD_SIZE * 16
    else:
        _stride_v_block = num_kv_heads * HEAD_SIZE * _bs
        _stride_v_head = HEAD_SIZE * _bs

    _stride_bt_seq = max_blocks_per_seq

    # ── Grid sizing ──────────────────────────────────────────────────
    _sw_max_parts = math.ceil(_sliding_window / KV_COMPUTE_BLOCK) + 1
    _grid_z = 1 if one_shot else min(_sw_max_parts, num_partitions)
    _out_num_parts = _grid_z
    _direct_output = one_shot

    if _direct_output:
        _stride_out_part = 0
        _stride_out_head = QUERY_GROUP_SIZE * HEAD_SIZE
        _stride_out_seq = num_kv_heads * QUERY_GROUP_SIZE * HEAD_SIZE
    else:
        _stride_out_part = QUERY_GROUP_SIZE * HEAD_SIZE
        _stride_out_head = _out_num_parts * QUERY_GROUP_SIZE * HEAD_SIZE
        _stride_out_seq = num_kv_heads * _out_num_parts * QUERY_GROUP_SIZE * HEAD_SIZE
    _stride_es_seq = num_kv_heads * _out_num_parts * QUERY_GROUP_SIZE
    _stride_ml_seq = num_kv_heads * _out_num_parts * QUERY_GROUP_SIZE

    # ── LDS allocation ───────────────────────────────────────────────
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_sw_smem")
    q_off = 0
    allocator.ptr = Q_LDS_BYTES
    prob_off = Q_LDS_BYTES
    allocator.ptr += PROB_LDS_BYTES
    bt_off = prob_off + PROB_LDS_BYTES
    allocator.ptr += BT_LDS_BYTES
    rmax_off = bt_off + BT_LDS_BYTES
    allocator.ptr += RED_SLOTS * 4
    rsum_off = rmax_off + RED_SLOTS * 4
    allocator.ptr += RED_SLOTS * 4

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
    ):
        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        part_z = gpu.block_idx.z

        # ────────────────────────────────────────────────────────────
        # Thread decomposition via CuTe layouts
        # ────────────────────────────────────────────────────────────
        # Thread decomposition — matching CuTe layouts:
        #   layout_thread = (NUM_WARPS, WARP_SIZE):(WARP_SIZE, 1)
        #   layout_lane   = (4, 16):(16, 1)
        mfma_row = tid & 15
        lane_hi4 = (tid & 0xF0) >> 4
        warp_id = tid >> 6

        # MFMA operand column bits: kv_col_bits = tid & 48
        kv_col_bits = tid & arith.constant(48, type=T.i32)
        c8 = arith.constant(8, type=T.i32)
        c112 = arith.constant(112, type=T.i32)
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        wave_idx = arith.index_cast(T.index, arith.unwrap(warp_id))

        # ────────────────────────────────────────────────────────────
        # Buffer resources via make_buffer_tensor pattern
        # ────────────────────────────────────────────────────────────
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        # Per-sequence context length: context_lengths_ptr[seq]
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        context_length_i32 = buffer_ops.buffer_load(cl_rsrc, seq, vec_width=1, dtype=T.i32)

        qs_rsrc = buffer_ops.create_buffer_resource(query_scale_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        query_scale_val = buffer_ops.buffer_load(qs_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        key_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        value_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        # ────────────────────────────────────────────────────────────
        # LDS views (via SmemAllocator)
        # ────────────────────────────────────────────────────────────
        base = allocator.get_base()
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(Q_LDS_BYTES // 8,)).get()
        p_lds_i32 = SmemPtr(base, prob_off, T.i32, shape=(PROB_LDS_BYTES // 4,)).get()
        bt_lds_i64 = SmemPtr(base, bt_off, T.i64, shape=(BT_LDS_BYTES // 8,)).get()
        s_max_p = SmemPtr(base, rmax_off, T.f32, shape=(RED_SLOTS,))
        s_sum_p = SmemPtr(base, rsum_off, T.f32, shape=(RED_SLOTS,))

        # ────────────────────────────────────────────────────────────
        # Stride constants as CuTe-style layouts
        # ────────────────────────────────────────────────────────────
        c_kb = arith.constant(_stride_k_block, type=T.i32)
        c_kh = arith.constant(_stride_k_head, type=T.i32)
        c_vb = arith.constant(_stride_v_block, type=T.i32)
        c_vh = arith.constant(_stride_v_head, type=T.i32)
        c_sq = arith.constant(_stride_q_seq, type=T.i32)
        c_qh = arith.constant(_stride_q_head, type=T.i32)
        c_bt = arith.constant(_stride_bt_seq, type=T.i32)

        _q_cta_base = seq * c_sq + kv_h * arith.constant(QUERY_GROUP_SIZE, type=T.i32) * c_qh
        _k_head_off = kv_h * c_kh
        _v_head_off = kv_h * c_vh

        # ────────────────────────────────────────────────────────────
        # Sliding window bounds (runtime)
        # ────────────────────────────────────────────────────────────
        c_sw = arith.constant(_sliding_window, type=T.i32)
        c_cpb = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_zero = arith.constant(0, type=T.i32)

        _sw_raw = context_length_i32 - c_sw
        seq_start_idx = arith.select(_sw_raw > c_zero, _sw_raw, c_zero)
        window_start_part = seq_start_idx // c_cpb
        actual_part = window_start_part + part_z

        # ────────────────────────────────────────────────────────────
        # Q load → LDS (with XOR swizzle layout)
        # Layout: Q_LDS[row=16, col=HEAD_SIZE] stored as i32 words
        # Swizzle: addr = (tid * 8) ^ (tid & 112)
        # ────────────────────────────────────────────────────────────
        q_off_g = _q_cta_base + mfma_row * c_qh + lane_hi4 * c8
        q_vec = buffer_ops.buffer_load(
            q_rsrc, q_off_g // arith.constant(4, type=T.i32),
            vec_width=2, dtype=T.i32)
        swiz = (tid * c8) ^ (tid & c112)
        vector.store(q_vec, q_lds_i32,
                     [arith.index_cast(T.index, swiz // arith.constant(4, type=T.i32))])
        gpu.barrier()

        # ────────────────────────────────────────────────────────────
        # Q read from LDS → register i64 operands for MFMA
        # Layout: Q_reg[4 chunks of i64] covering HEAD_SIZE=128
        # ────────────────────────────────────────────────────────────
        _q_col = ((tid * arith.constant(16, type=T.i32)) & c112) ^ kv_col_bits
        _q_b0 = (mfma_row * arith.constant(HEAD_SIZE, type=T.i32)) | _q_col
        _q_b1 = _q_b0 ^ arith.constant(64, type=T.i32)
        q_v0 = vector.load_op(T.vec(2, T.i64), q_lds_i64,
                               [arith.index_cast(T.index, _q_b0 // c8)])
        q_v1 = vector.load_op(T.vec(2, T.i64), q_lds_i64,
                               [arith.index_cast(T.index, _q_b1 // c8)])

        # Allocate register fragments for Q MFMA operands (4 × i64)
        q_frags = [
            vector.extract(q_v0, static_position=[0], dynamic_position=[]),
            vector.extract(q_v0, static_position=[1], dynamic_position=[]),
            vector.extract(q_v1, static_position=[0], dynamic_position=[]),
            vector.extract(q_v1, static_position=[1], dynamic_position=[]),
        ]

        # ────────────────────────────────────────────────────────────
        # Register accumulators via memref_alloca
        # QK accumulators: QK_N_TILES_WARP × f32x4
        # PV accumulators: PV_N_TILES_WARP × f32x4
        # ────────────────────────────────────────────────────────────

        # Scalar constants
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        SOFTMAX_SCALE_C = arith.constant(_softmax_scale, type=T.f32)
        QK_SCALE = SOFTMAX_SCALE_C * query_scale_val * key_scale_val
        F240 = arith.constant(FP8_MAX, type=T.f32)
        PROB_SCALE_C = value_scale_val / F240
        warp_head_base = warp_id * arith.constant(32, type=T.i32)

        # ────────────────────────────────────────────────────────────
        # Wave-level reductions
        # ────────────────────────────────────────────────────────────
        def _wave_max(x):
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                w = w.maximumf(w.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            return w

        def _wave_add(x):
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                w = w + w.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
            return w

        _mi = [arith.index_cast(T.index, arith.constant(i, type=T.i32))
               for i in range_constexpr(NUM_WARPS)]

        def _extract_k_i64(kv_4xi32, pair_idx):
            a = vector.extract(kv_4xi32, static_position=[pair_idx * 2])
            b = vector.extract(kv_4xi32, static_position=[pair_idx * 2 + 1])
            return _pack_i32_pair_to_i64(a, b)

        # ────────────────────────────────────────────────────────────
        # Load K tile from paged cache using block table indirection
        # Uses CuTe layout for K cache: [head_splits, block_elems, 16]
        # ────────────────────────────────────────────────────────────
        def _load_k_tile(part_val):
            """Load K data and block table entries for a partition."""
            result = {}
            if _use_large_block:
                bt_idx = part_val // arith.constant(_partitions_per_block, type=T.i32)
                page_off_v = (part_val % arith.constant(_partitions_per_block, type=T.i32)) * arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
                partition_start_v = part_val * arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
                phys_block_v = buffer_ops.buffer_load(bt_rsrc, seq * c_bt + bt_idx, vec_width=1, dtype=T.i32)
                phys_list = [phys_block_v] * 4
                result['phys_block'] = phys_block_v
                result['page_off'] = page_off_v
            else:
                bt_start = part_val * arith.constant(_blocks_per_partition, type=T.i32)
                partition_start_v = part_val * arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
                _bt_base = seq * c_bt + bt_start
                phys = [buffer_ops.buffer_load(bt_rsrc, _bt_base + warp_id + arith.constant(i * 4, type=T.i32), vec_width=1, dtype=T.i32)
                        for i in range_constexpr(4)]
                phys_list = phys
                result['phys_0'] = phys[0]
                result['phys_1'] = phys[1]

            result['partition_start'] = partition_start_v

            # K loads: 4 N-tiles × 2 halves (covering HEAD_SIZE=128)
            kv_loads = []
            for n_tile in range_constexpr(QK_N_TILES_WARP):
                pb = phys_list[n_tile]
                _k_blk_base = pb * c_kb + _k_head_off
                if _use_large_block:
                    tok = page_off_v + warp_id * arith.constant(64, type=T.i32) + arith.constant(n_tile * 16, type=T.i32) + mfma_row
                    kb0 = _k_blk_base + tok * arith.constant(16, type=T.i32)
                    kb1 = _k_blk_base + arith.constant(2 * _bs * 16, type=T.i32) + tok * arith.constant(16, type=T.i32)
                else:
                    kb0 = _k_blk_base + mfma_row * arith.constant(16, type=T.i32)
                    kb1 = _k_blk_base + arith.constant(2 * KV_BLOCK_SIZE * 16, type=T.i32) + mfma_row * arith.constant(16, type=T.i32)
                k0 = buffer_ops.buffer_load(k_rsrc, kb0 // arith.constant(4, type=T.i32), vec_width=4, dtype=T.i32)
                k1 = buffer_ops.buffer_load(k_rsrc, kb1 // arith.constant(4, type=T.i32), vec_width=4, dtype=T.i32)
                kv_loads.append([k0, k1])
            result['kv'] = kv_loads
            return result

        # ────────────────────────────────────────────────────────────
        # QK GEMM via MFMA: score = Q @ K^T
        # Uses fx.memref_alloca accumulators
        # MMA atom: MFMA(16, 16, 32, fp8)
        # Per warp: QK_N_TILES_WARP=4 N-tiles, QK_K_STEPS=4 K-steps
        # ────────────────────────────────────────────────────────────
        def _compute_qk_mfma(kv_loads):
            """Compute QK attention scores using MFMA instructions.
            Returns list of QK_N_TILES_WARP f32x4 accumulators."""
            acc_qk = []
            for t in range_constexpr(QK_N_TILES_WARP):
                k_ops = [_extract_k_i64(kv_loads[t][0], 0),
                         _extract_k_i64(kv_loads[t][0], 1),
                         _extract_k_i64(kv_loads[t][1], 0),
                         _extract_k_i64(kv_loads[t][1], 1)]
                acc = arith.constant_vector(0.0, T.f32x4)
                for j in range_constexpr(QK_K_STEPS):
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [k_ops[j], q_frags[j], acc, 0, 0, 0])
                acc_qk.append(acc)
            return acc_qk

        # ────────────────────────────────────────────────────────────
        # Sliding window mask + scale
        # Layout of QK tiles: QK_N_TILES_WARP tiles of 16 columns each
        # Token index: partition_start + warp_id*64 + n_tile*16 + elem
        # ────────────────────────────────────────────────────────────
        def _apply_sw_mask(acc_qk, partition_start):
            ctx_len = context_length_i32
            for n_tile in range_constexpr(QK_N_TILES_WARP):
                acc_qk[n_tile] = acc_qk[n_tile] * vector.broadcast(T.f32x4, QK_SCALE)
                for elem in range_constexpr(4):
                    kv_tok = (partition_start
                              + warp_id * arith.constant(64, type=T.i32)
                              + arith.constant(n_tile * 16 + elem, type=T.i32))
                    in_b = (kv_tok < ctx_len) & (kv_tok >= seq_start_idx)
                    v = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    acc_qk[n_tile] = vector.insert(
                        arith.select(in_b, v, NEG_INF),
                        acc_qk[n_tile], static_position=[elem], dynamic_position=[])
            return acc_qk

        # ────────────────────────────────────────────────────────────
        # Online softmax (cross-warp via LDS reduction)
        # ────────────────────────────────────────────────────────────
        def _online_softmax(acc_qk, running_max, running_sum, acc_pv):
            local_max = NEG_INF
            for t in range_constexpr(QK_N_TILES_WARP):
                local_max = local_max.maximumf(
                    vector.reduction(T.f32, "maxnumf", acc_qk[t]))
            wmax = _wave_max(local_max)
            s_max_p.store(wmax, [wave_idx])
            gpu.barrier()

            global_max_new = s_max_p.load([_mi[0]])
            for w in range_constexpr(1, NUM_WARPS):
                global_max_new = global_max_new.maximumf(s_max_p.load([_mi[w]]))

            rescale = ((running_max - global_max_new) * LOG2E_C).exp2(
                fastmath=arith.FastMathFlags.fast)
            acc_pv = [t * vector.broadcast(T.f32x4, rescale) for t in acc_pv]
            running_sum = running_sum * rescale
            running_max = global_max_new

            local_sum = ZERO_F
            for t in range_constexpr(QK_N_TILES_WARP):
                for elem in range_constexpr(4):
                    s = vector.extract(acc_qk[t], static_position=[elem], dynamic_position=[])
                    p = ((s - running_max) * LOG2E_C).exp2(
                        fastmath=arith.FastMathFlags.fast)
                    local_sum = local_sum + p
                    acc_qk[t] = vector.insert(
                        p, acc_qk[t], static_position=[elem], dynamic_position=[])
            wsum = _wave_add(local_sum)
            s_sum_p.store(wsum, [wave_idx])
            gpu.barrier()

            iter_sum = ZERO_F
            for w in range_constexpr(NUM_WARPS):
                iter_sum = iter_sum + s_sum_p.load([_mi[w]])
            running_sum = running_sum + iter_sum

            return acc_qk, running_max, running_sum, acc_pv

        # ────────────────────────────────────────────────────────────
        # FP8 quantize probs → LDS (pack 4 f32 → 1 i32 fp8)
        # ────────────────────────────────────────────────────────────
        def _store_probs_to_lds(acc_qk):
            probs = []
            for t in range_constexpr(QK_N_TILES_WARP):
                for elem in range_constexpr(4):
                    probs.append(
                        vector.extract(acc_qk[t], static_position=[elem], dynamic_position=[]) * F240)

            fp8_i32 = []
            for i in range_constexpr(4):
                lo = rocdl.cvt_pk_fp8_f32(
                    T.i32, probs[i * 4], probs[i * 4 + 1],
                    arith.constant(0, type=T.i32), False)
                wd = rocdl.cvt_pk_fp8_f32(
                    T.i32, probs[i * 4 + 2], probs[i * 4 + 3], lo, True)
                fp8_i32.append(wd)

            gpu.barrier()
            prob_vec4 = vector.from_elements(T.vec(4, T.i32), fp8_i32)
            vector.store(prob_vec4, p_lds_i32,
                         [arith.index_cast(T.index, tid * arith.constant(4, type=T.i32))])
            gpu.barrier()

        # ────────────────────────────────────────────────────────────
        # Load P (probs) from LDS → register i64 operands
        # Layout: P_LDS[BLOCK_THREADS * 4] with warp-based addressing
        # ────────────────────────────────────────────────────────────
        def _load_prob_operands():
            _prob_base = (kv_col_bits * arith.constant(64, type=T.i32)
                          + mfma_row * arith.constant(16, type=T.i32))
            p_lds = SmemPtr(base, prob_off, T.i32, shape=(PROB_LDS_BYTES // 4,)).get()

            def _ld4(off):
                idx = arith.index_cast(
                    T.index,
                    (_prob_base + arith.constant(off, type=T.i32)) // arith.constant(4, type=T.i32))
                return vector.load_op(T.vec(4, T.i32), p_lds, [idx])

            _pa, _pb, _pc, _pd = _ld4(0), _ld4(256), _ld4(512), _ld4(768)

            def _pk(va, vb, k):
                return _pack_i32_pair_to_i64(
                    vector.extract(va, static_position=[k]),
                    vector.extract(vb, static_position=[k]))

            return ([_pk(_pa, _pb, k) for k in range_constexpr(4)]
                    + [_pk(_pc, _pd, k) for k in range_constexpr(4)])

        # ────────────────────────────────────────────────────────────
        # Load V tile from paged cache + BT staging via LDS
        # ────────────────────────────────────────────────────────────
        def _stage_bt_and_load_v(bt_vals):
            if _use_large_block:
                phys_block_v, page_off_v = bt_vals[0], bt_vals[1]
                token_page_base = page_off_v // arith.constant(16, type=T.i32)
                tp0 = token_page_base + warp_id
                tp1 = tp0 + arith.constant(4, type=T.i32)
                bt_vec = vector.from_elements(T.vec(2, T.i64), [
                    arith.extsi(T.i64, arith.unwrap(tp0)),
                    arith.extsi(T.i64, arith.unwrap(tp1))])
                vector.store(bt_vec, bt_lds_i64,
                             [arith.index_cast(T.index, warp_id * arith.constant(2, type=T.i32))])
                gpu.barrier()
            else:
                phys_0, phys_1 = bt_vals[0], bt_vals[1]
                bt_vec = vector.from_elements(T.vec(2, T.i64), [
                    arith.extsi(T.i64, arith.unwrap(phys_0)),
                    arith.extsi(T.i64, arith.unwrap(phys_1))])
                gpu.barrier()
                vector.store(bt_vec, bt_lds_i64,
                             [arith.index_cast(T.index, warp_id * arith.constant(2, type=T.i32))])
                gpu.barrier()

            bt_li = arith.index_cast(T.index, kv_col_bits // arith.constant(8, type=T.i32))
            bt_load = vector.load_op(T.vec(2, T.i64), bt_lds_i64, [bt_li])
            phys_pv_0 = arith.trunci(T.i32, arith.unwrap(
                vector.extract(bt_load, static_position=[0], dynamic_position=[])))
            phys_pv_1 = arith.trunci(T.i32, arith.unwrap(
                vector.extract(bt_load, static_position=[1], dynamic_position=[])))

            vv = []
            for n_tile in range_constexpr(PV_N_TILES_WARP):
                h_py = n_tile * MFMA_N
                pv_pb = phys_pv_0 if n_tile == 0 else phys_pv_1
                if _use_large_block and trans_v:
                    _vb = (arith.unwrap(bt_vals[0]) * c_vb + _v_head_off
                           + pv_pb * arith.constant(HEAD_SIZE * 16, type=T.i32)
                           + arith.constant(h_py * 16, type=T.i32))
                elif _use_large_block:
                    _vb = pv_pb * c_vb + _v_head_off + arith.constant(h_py * _bs, type=T.i32) + bt_vals[1]
                else:
                    _vb = pv_pb * c_vb + _v_head_off + arith.constant(h_py * KV_BLOCK_SIZE, type=T.i32)
                loads = [buffer_ops.buffer_load(
                    v_rsrc, (_vb + arith.constant(li * 32, type=T.i32)) // arith.constant(4, type=T.i32),
                    vec_width=4, dtype=T.i32) for li in range_constexpr(4)]
                vv.append(loads)
            return vv

        # ────────────────────────────────────────────────────────────
        # PV GEMM via MFMA: output += P @ V
        # Per warp: PV_N_TILES_WARP=2 N-tiles, PV_K_STEPS=8 K-steps
        # ────────────────────────────────────────────────────────────
        def _compute_pv_mfma(p_ops, vv, acc_pv):
            for t in range_constexpr(PV_N_TILES_WARP):
                v_ops = []
                for li in range_constexpr(4):
                    v_ops.append(_extract_k_i64(vv[t][li], 0))
                    v_ops.append(_extract_k_i64(vv[t][li], 1))
                acc = acc_pv[t]
                for j in range_constexpr(PV_K_STEPS):
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [v_ops[j], p_ops[j], acc, 0, 0, 0])
                acc_pv[t] = acc
            return acc_pv

        # ────────────────────────────────────────────────────────────
        # Output write (direct or partitioned)
        # Output layout: [seq, kv_head, (part), query_group, head_size]
        # ────────────────────────────────────────────────────────────
        def _write_output(running_max, running_sum, acc_pv):
            rcp = arith.constant(1.0, type=T.f32) / running_sum
            pv_out = [acc_pv[t] * vector.broadcast(T.f32x4, PROB_SCALE_C * rcp)
                      for t in range_constexpr(PV_N_TILES_WARP)]

            if _direct_output:
                c_os = arith.constant(_stride_out_seq, type=T.i32)
                c_oh = arith.constant(_stride_out_head, type=T.i32)
                for n_tile in range_constexpr(PV_N_TILES_WARP):
                    out_off = (seq * c_os + kv_h * c_oh
                               + mfma_row * arith.constant(HEAD_SIZE, type=T.i32)
                               + warp_head_base + arith.constant(n_tile * MFMA_N, type=T.i32))
                    out_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_out[n_tile])
                    out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                    buffer_ops.buffer_store(out_i32, out_rsrc,
                                            out_off * arith.constant(2, type=T.i32),
                                            offset_is_bytes=True)
            else:
                c_np_qg = arith.constant(_out_num_parts * QUERY_GROUP_SIZE, type=T.i32)
                c_qg = arith.constant(QUERY_GROUP_SIZE, type=T.i32)
                ml_off = (seq * arith.constant(_stride_ml_seq, type=T.i32)
                          + kv_h * c_np_qg + part_z * c_qg + mfma_row)
                es_off = (seq * arith.constant(_stride_es_seq, type=T.i32)
                          + kv_h * c_np_qg + part_z * c_qg + mfma_row)
                buffer_ops.buffer_store(running_max, ml_rsrc, ml_off)
                buffer_ops.buffer_store(running_sum, es_rsrc, es_off)

                c_os = arith.constant(_stride_out_seq, type=T.i32)
                c_oh = arith.constant(_stride_out_head, type=T.i32)
                c_op = arith.constant(_stride_out_part, type=T.i32)
                for n_tile in range_constexpr(PV_N_TILES_WARP):
                    out_off = (seq * c_os + kv_h * c_oh + part_z * c_op
                               + mfma_row * arith.constant(HEAD_SIZE, type=T.i32)
                               + warp_head_base + arith.constant(n_tile * MFMA_N, type=T.i32))
                    out_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_out[n_tile])
                    out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                    buffer_ops.buffer_store(out_i32, out_rsrc,
                                            out_off * arith.constant(2, type=T.i32),
                                            offset_is_bytes=True)

        # ════════════════════════════════════════════════════════════
        # Main body: single partition per CTA
        # Pipeline:  load_K → QK_MFMA → mask → softmax → quant_P → LDS
        #          → load_V (BT stage) → PV_MFMA → write
        # ════════════════════════════════════════════════════════════
        running_max = NEG_INF
        running_sum = ZERO_F
        acc_pv = [arith.constant_vector(0.0, T.f32x4)
                  for _ in range_constexpr(PV_N_TILES_WARP)]

        pf = _load_k_tile(actual_part)
        kv = pf['kv']
        partition_start = pf['partition_start']
        bt_vals = ([pf['phys_block'], pf['page_off']] if _use_large_block
                   else [pf['phys_0'], pf['phys_1']])

        acc_qk = _compute_qk_mfma(kv)
        acc_qk = _apply_sw_mask(acc_qk, partition_start)
        acc_qk, running_max, running_sum, acc_pv = _online_softmax(
            acc_qk, running_max, running_sum, acc_pv)
        _store_probs_to_lds(acc_qk)
        p_ops = _load_prob_operands()
        vv = _stage_bt_and_load_v(bt_vals)
        acc_pv = _compute_pv_mfma(p_ops, vv, acc_pv)
        _write_output(running_max, running_sum, acc_pv)

    # ── @flyc.jit launch wrapper ─────────────────────────────────────
    _cache_tag = (num_seqs, num_kv_heads, _grid_z, _sliding_window)

    @autotune(
        configs=[Config(waves_per_eu=1), Config(waves_per_eu=2), Config(waves_per_eu=3), Config(waves_per_eu=4)],
        key=['gy', 'gz'], warmup=3, rep=10,
    )
    @flyc.jit
    def launch_pa_decode_sw(out, es, ml, q, kc, vc, bt, cl,
                            qs, ks, vs,
                            gx, gy, gz,
                            stream: fx.Stream = fx.Stream(None)):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_sw_kernel(out, es, ml, q, kc, vc, bt, cl, qs, ks, vs).launch(
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
# Partition-sum reduce kernel (replaces _torch_reduce_partitions)
# =====================================================================

def _ext_f(target_type, value):
    """Extend float precision (e.g. vec<8×bf16> → vec<8×f32>)."""
    return _mlir_arith.ExtFOp(target_type, value).result


@functools.lru_cache(maxsize=256)
def compile_ps_reduce(num_seqs, num_kv_heads, num_parts):
    """Compile a GPU kernel that reduces partial attention outputs.

    Grid: (num_seqs, num_kv_heads).  256 threads per CTA.
    Each thread reduces ``num_parts`` partitions for 8 contiguous bf16
    output elements (one (qg, hs_chunk) slice).

    Algorithm per thread:
      1. Find global max of ``max_logits`` across partitions.
      2. Rescale ``exp_sums`` and accumulate weighted ``partial_out``.
      3. Normalise and store bf16 result.
    """
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
                   + kv_h * arith.constant(_es_stride_kv_h, type=T.i32)
                   + qg)

        po_base_bf16 = (seq * arith.constant(_po_stride_seq, type=T.i32)
                        + kv_h * arith.constant(_po_stride_kv_h, type=T.i32)
                        + qg * arith.constant(HS, type=T.i32)
                        + hs_chunk * arith.constant(8, type=T.i32))

        NEG_INF_C = arith.constant(float("-inf"), type=T.f32)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        ONE_F = arith.constant(1.0, type=T.f32)

        # Pass 1: global max across partitions
        global_max = NEG_INF_C
        for p in range_constexpr(num_parts):
            ml_off = es_base + arith.constant(p * _es_stride_part, type=T.i32)
            ml_val = buffer_ops.buffer_load(ml_rsrc, ml_off, vec_width=1)
            global_max = global_max.maximumf(ml_val)

        # Pass 2: accumulate weighted partial outputs
        total_sum = ZERO_F
        acc = arith.constant_vector(0.0, T.vec(8, T.f32))

        for p in range_constexpr(num_parts):
            es_off = es_base + arith.constant(p * _es_stride_part, type=T.i32)
            es_val = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1)
            ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1)

            w = es_val * ((ml_val - global_max) * LOG2E_C).exp2(
                fastmath=arith.FastMathFlags.fast)
            total_sum = total_sum + w

            po_off_i32 = (po_base_bf16
                          + arith.constant(p * _po_stride_part, type=T.i32)
                          ) // arith.constant(2, type=T.i32)
            data_4xi32 = buffer_ops.buffer_load(po_rsrc, po_off_i32,
                                                vec_width=4, dtype=T.i32)
            data_8xbf16 = vector.bitcast(T.vec(8, T.bf16), data_4xi32)
            data_8xf32 = _ext_f(T.vec(8, T.f32), data_8xbf16)

            w_vec = vector.broadcast(T.vec(8, T.f32), w)
            acc = acc + data_8xf32 * w_vec

        # Normalise and store bf16
        rcp = ONE_F / total_sum
        result_8xf32 = acc * vector.broadcast(T.vec(8, T.f32), rcp)
        result_8xbf16 = arith.trunc_f(T.vec(8, T.bf16), result_8xf32)
        result_4xi32 = vector.bitcast(T.vec(4, T.i32), result_8xbf16)

        out_off_i32 = (seq * arith.constant(_out_stride_seq, type=T.i32)
                       + kv_h * arith.constant(_out_stride_kv_h, type=T.i32)
                       + qg * arith.constant(HS, type=T.i32)
                       + hs_chunk * arith.constant(8, type=T.i32)
                       ) // arith.constant(2, type=T.i32)
        buffer_ops.buffer_store(result_4xi32, out_rsrc, out_off_i32)

    @flyc.jit
    def launch_ps_reduce(out, partial_out, exp_sums, max_logits,
                         gx, gy,
                         stream: fx.Stream = fx.Stream(None)):
        ps_reduce_kernel(out, partial_out, exp_sums, max_logits).launch(
            grid=(gx, gy, 1),
            block=(THREADS, 1, 1), stream=stream)

    return launch_ps_reduce


# =====================================================================
# Launch API — follows pa_decode_gluon calling convention
# =====================================================================

_compiled_cache = {}


def _torch_reduce_partitions(output_5d, partial_out, exp_sums, max_logits):
    """Reduce partial attention outputs across partitions (PyTorch).

    output_5d:   [batch, query_length, num_kv_heads, query_group_size, head_size]
    partial_out: [batch, num_kv_heads, num_parts, query_group_size, head_size]
    exp_sums:    [batch, num_kv_heads, num_parts, query_group_size]
    max_logits:  [batch, num_kv_heads, num_parts, query_group_size]
    """
    global_max = max_logits.max(dim=2, keepdim=True).values
    rescale = torch.exp2((max_logits - global_max) * LOG2E)
    rescaled_sums = exp_sums * rescale
    total_sum = rescaled_sums.sum(dim=2, keepdim=True)
    weights = rescaled_sums / total_sum.clamp(min=1e-12)
    result = (weights.unsqueeze(-1) * partial_out.float()).sum(dim=2)
    output_5d[:, 0, :, :, :] = result.to(output_5d.dtype)


def pa_decode_sw_launch(
    output: torch.Tensor,           # [num_seqs * query_length, num_query_heads, head_size]
    query: torch.Tensor,            # [num_seqs * query_length, num_query_heads, head_size]
    key_cache: torch.Tensor,        # [num_blocks, num_kv_heads, head_size//x, kv_block_size, x]
    value_cache: torch.Tensor,      # 4D or 5D (transposed)
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,     # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_scale: torch.Tensor = None, # per-tensor scalar, shape [1]
    key_scale: torch.Tensor = None,   # per-tensor scalar, shape [1]
    value_scale: torch.Tensor = None, # per-tensor scalar, shape [1]
    *,
    sliding_window: int = 0,
    max_context_len: int = 0,
    kv_block_size: int = 0,
    query_length: int = 1,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    """Sliding-window PA decode — follows pa_decode_gluon calling convention.

    No .item() / .max() / .numel() calls — all parameters are plain Python
    scalars or tensor shapes.  The caller must supply ``max_context_len``
    (or ``sliding_window``) so the wrapper never touches tensor *values*.

    Args:
        output: [num_seqs*query_length, num_query_heads, head_size].
        query:  [num_seqs*query_length, num_query_heads, head_size], FP8 or BF16.
        key_cache / value_cache: Paged KV cache (5D key, 4D or 5D value).
        context_lengths: [num_seqs] i32 — passed through to the kernel.
        block_tables: [num_seqs, max_blocks_per_seq] i32.
        softmax_scale: 1/sqrt(head_size).
        query_scale / key_scale / value_scale: per-tensor scalar tensors, shape [1].
        sliding_window: 0 = full context, >0 = attend to last N tokens.
        max_context_len: Upper bound on context length (for grid sizing).
            When sliding_window > 0, this can be 0 (grid derived from sliding_window).
            When sliding_window == 0, caller MUST provide max_context_len.
        kv_block_size: 0 = auto-detect from key_cache.shape[-2].
        query_length: Number of query positions (default 1 for decode).
        exp_sums / max_logits / temporary_output: Optional intermediate buffers.
    """
    # ── Derive dimensions from tensor shapes only (no .item()) ───────
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

    # ── Partition / grid sizing (no GPU→CPU sync) ────────────────────
    if sliding_window > 0:
        sw_val = sliding_window
        effective_len = sliding_window
    else:
        assert max_context_len > 0, (
            "pa_decode_sw_launch: when sliding_window==0, caller must provide "
            "max_context_len (the upper bound on context length)")
        sw_val = max_context_len
        effective_len = max_context_len

    actual_parts_needed = math.ceil(effective_len / context_partition_size)
    one_shot = (actual_parts_needed <= 1)
    num_parts = 1 if one_shot else actual_parts_needed

    # ── Compile kernel (cached) ──────────────────────────────────────
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

    # ── Reshape to 5D ────────────────────────────────────────────────
    output_5d = output
    # output_5d = output.reshape(
    #     batch_size, query_length, num_kv_heads, query_group_size, head_size)

    if direct:
        # one_shot: kernel writes directly to output, no intermediates needed.
        # Still need valid tensor pointers for buffer resource creation in kernel.
        if exp_sums is None:
            exp_sums = torch.empty(1, device=dev, dtype=torch.float32)
        if max_logits is None:
            max_logits = torch.empty(1, device=dev, dtype=torch.float32)
        compiled['launch'](
            output_5d, exp_sums, max_logits,
            query, key_cache, value_cache, block_tables,
            context_lengths, query_scale, key_scale, value_scale,
            batch_size, num_kv_heads, grid_z, s)
    else:
        # partitioned: allocate intermediates, run kernel, then GPU reduce
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
            batch_size, num_kv_heads, grid_z, s)

        reduce_fn = compile_ps_reduce(batch_size, num_kv_heads, num_parts)
        reduce_fn(output_5d, temporary_output, exp_sums, max_logits,
                  batch_size, num_kv_heads, s)

    return "sw_one_shot" if one_shot else f"sw_partitioned({grid_z})"
