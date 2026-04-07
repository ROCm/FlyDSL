# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
CDNA paged-attention FP8 decode kernel using MFMA instructions (wave64).

Supports kv_block_size=16 (original) and kv_block_size=1024 (trans_v required).
Reduce kernels are in pa_common.py (shared with RDNA).
"""

from __future__ import annotations
import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir

from kernels.pa_common import (
    QUERY_GROUP_SIZE, HEAD_SIZE, KV_BLOCK_SIZE, KV_COMPUTE_BLOCK,
    FP8_MAX, LOG2E, compute_pa_strides,
    build_ps_reduce_kernel, build_v2_reduce_kernel,  # re-export
)

NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
MFMA_M = MFMA_N = 16
MFMA_K = 32
QK_N_TILES_WARP = (KV_COMPUTE_BLOCK // NUM_WARPS) // MFMA_N  # 4
PV_K_STEPS = KV_COMPUTE_BLOCK // MFMA_K  # 8
PV_N_TILES_WARP = (HEAD_SIZE // NUM_WARPS) // MFMA_N  # 2

Q_LDS_BYTES = BLOCK_THREADS * 8
PROB_LDS_BYTES = BLOCK_THREADS * 16
BT_LDS_BYTES = NUM_WARPS * 16
RED_SLOTS = NUM_WARPS


def _vsplat_mul(vec, scalar):
    s = scalar.ir_value() if hasattr(scalar, 'ir_value') else scalar
    return vec * vector.broadcast(T.f32x4, s)


def _pack_i32_pair_to_i64(a_i32, b_i32):
    """Pack two i32 values into one i64 for MFMA operand."""
    v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
    v1 = vector.bitcast(T.vec(1, T.i64), v)
    return vector.extract(v1, static_position=[0])


allocator = None


def build_pa_decode_module(
    num_seqs,
    num_kv_heads,
    num_partitions,
    max_blocks_per_seq=256,
    softmax_scale=None,
    query_scale=1.0,
    key_scale=1.0,
    value_scale=1.0,
    kv_block_size=16,
    trans_v=False,
    one_shot=False,
    ps_num_splits=0,
):
    global allocator
    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE**0.5)
    _qk_scale = float(softmax_scale * query_scale * key_scale)
    _prob_scale = float(value_scale / FP8_MAX)

    _bs = kv_block_size
    _S = compute_pa_strides(num_kv_heads, num_partitions, max_blocks_per_seq,
                            kv_block_size=kv_block_size, trans_v=trans_v,
                            one_shot=one_shot, ps_num_splits=ps_num_splits)
    _stride_q_seq = _S["stride_q_seq"]
    _stride_q_head = _S["stride_q_head"]
    _stride_k_block = _S["stride_k_block"]
    _stride_k_head = _S["stride_k_head"]
    _stride_bt_seq = _S["stride_bt_seq"]
    _stride_v_block = _S["stride_v_block"]
    _stride_v_head = _S["stride_v_head"]
    _stride_out_part = _S["stride_out_part"]
    _stride_out_head = _S["stride_out_head"]
    _stride_out_seq = _S["stride_out_seq"]
    _stride_es_seq = _S["stride_es_seq"]
    _stride_ml_seq = _S["stride_ml_seq"]
    _use_large_block = _S["use_large_block"]
    _partitions_per_block = _S["partitions_per_block"]
    _blocks_per_partition = _S["blocks_per_partition"]
    _ps_mode = _S["ps_mode"]
    _max_pps = _S["max_pps"]

    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_smem")
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

    @flyc.kernel
    def pa_decode_dot_kernel(
        out_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        key_cache_ptr: fx.Tensor,
        value_cache_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        context_length_i32: Int32,
    ):
        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        part = gpu.block_idx.z

        mfma_row = tid & 15
        lane_hi4 = (tid & 0xF0) >> 4
        warp_id = tid >> 6
        kv_col_bits = tid & 48
        lane_iw = tid % WARP_SIZE
        c8 = fx.Int32(8)
        c112 = fx.Int32(112)
        c_w = fx.Int32(WARP_SIZE)

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        base = allocator.get_base()
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(Q_LDS_BYTES // 8,)).get()
        p_lds_i32 = SmemPtr(base, prob_off, T.i32, shape=(PROB_LDS_BYTES // 4,)).get()
        bt_lds_i64 = SmemPtr(base, bt_off, T.i64, shape=(BT_LDS_BYTES // 8,)).get()
        s_max_p = SmemPtr(base, rmax_off, T.f32, shape=(RED_SLOTS,))
        s_sum_p = SmemPtr(base, rsum_off, T.f32, shape=(RED_SLOTS,))

        c_kb = fx.Int32(_stride_k_block)
        c_kh = fx.Int32(_stride_k_head)
        c_vb = fx.Int32(_stride_v_block)
        c_vh = fx.Int32(_stride_v_head)
        c_sq = fx.Int32(_stride_q_seq)
        c_qh = fx.Int32(_stride_q_head)
        c_bt = fx.Int32(_stride_bt_seq)
        wave_idx = arith.index_cast(T.index, warp_id)

        _q_cta_base = seq * c_sq + kv_h * fx.Int32(QUERY_GROUP_SIZE) * c_qh
        _k_head_off = kv_h * c_kh
        _v_head_off = kv_h * c_vh

        part_z = gpu.block_idx.z

        # -- STEP 1: Q -> LDS --
        q_off_g = _q_cta_base + mfma_row * c_qh + lane_hi4 * c8
        q_vec = buffer_ops.buffer_load(q_rsrc, q_off_g // 4, vec_width=2, dtype=T.i32)
        swiz = (tid * c8) ^ (tid & c112)
        vector.store(q_vec, q_lds_i32, [arith.index_cast(T.index, swiz // 4)])

        # -- STEP 4: barrier for Q LDS --
        gpu.barrier()

        # -- STEP 5: Q from LDS --
        _q_col = ((tid * 16) & c112) ^ kv_col_bits
        _q_b0 = (mfma_row * HEAD_SIZE) | _q_col
        _q_b1 = _q_b0 ^ 64
        q_v0 = vector.load_op(T.vec(2, T.i64), q_lds_i64, [arith.index_cast(T.index, _q_b0 // c8)])
        q_v1 = vector.load_op(T.vec(2, T.i64), q_lds_i64, [arith.index_cast(T.index, _q_b1 // c8)])

        q_a0 = vector.extract(q_v0, static_position=[0], dynamic_position=[])
        q_a1 = vector.extract(q_v0, static_position=[1], dynamic_position=[])
        q_a2 = vector.extract(q_v1, static_position=[0], dynamic_position=[])
        q_a3 = vector.extract(q_v1, static_position=[1], dynamic_position=[])

        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = fx.Float32(0.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)
        F240 = arith.constant(FP8_MAX, type=T.f32)
        PROB_SCALE_C = arith.constant(_prob_scale, type=T.f32)
        warp_head_base = warp_id * 32

        from flydsl.expr.utils.arith import int_to_int as _int_cast
        from flydsl.expr.numeric import Int32 as _Int32, Int64 as _Int64

        def _wave_max(x):
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = w.shuffle_xor(fx.Int32(sh), c_w)
                w = w.maximumf(peer)
            return w

        def _wave_add(x):
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = w.shuffle_xor(fx.Int32(sh), c_w)
                w = w + peer
            return w

        _mi0 = arith.index_cast(T.index, fx.Int32(0))
        _mi1 = arith.index_cast(T.index, fx.Int32(1))
        _mi2 = arith.index_cast(T.index, fx.Int32(2))
        _mi3 = arith.index_cast(T.index, fx.Int32(3))

        # ================================================================
        # Helper: issue BT + K loads for a given partition index
        # Returns dict with: kv, partition_start, phys_block, page_off,
        #                     phys_0, phys_1 (for BT staging later)
        # ================================================================
        def _issue_bt_k_loads(part_val):
            result = {}
            if _use_large_block:
                bt_idx = part_val // _partitions_per_block
                page_off_v = (part_val % _partitions_per_block) * KV_COMPUTE_BLOCK
                partition_start_v = part_val * KV_COMPUTE_BLOCK
                _bt_seq_base = seq * c_bt + bt_idx
                phys_block_v = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base, vec_width=1, dtype=T.i32)
                phys_list = [phys_block_v, phys_block_v, phys_block_v, phys_block_v]
                result['phys_block'] = phys_block_v
                result['page_off'] = page_off_v
            else:
                bt_start = part_val * _blocks_per_partition
                partition_start_v = part_val * KV_COMPUTE_BLOCK
                _bt_seq_base = seq * c_bt + bt_start
                phys_0_v = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base + warp_id, vec_width=1, dtype=T.i32)
                phys_1_v = buffer_ops.buffer_load(
                    bt_rsrc, _bt_seq_base + warp_id + 4, vec_width=1, dtype=T.i32
                )
                phys_2_v = buffer_ops.buffer_load(
                    bt_rsrc, _bt_seq_base + warp_id + 8, vec_width=1, dtype=T.i32
                )
                phys_3_v = buffer_ops.buffer_load(
                    bt_rsrc, _bt_seq_base + warp_id + 12, vec_width=1, dtype=T.i32
                )
                phys_list = [phys_0_v, phys_1_v, phys_2_v, phys_3_v]
                result['phys_0'] = phys_0_v
                result['phys_1'] = phys_1_v

            result['partition_start'] = partition_start_v

            # K loads via buffer_load (4xi32 = 16 bytes = global_load_dwordx4)
            kv_loads = []
            for n_tile in [0, 1, 2, 3]:
                pb = phys_list[n_tile]
                _k_blk_base = pb * c_kb + _k_head_off
                if _use_large_block:
                    tok_in_blk = page_off_v + warp_id * 64 + fx.Int32(n_tile * 16) + mfma_row
                    kb0 = _k_blk_base + tok_in_blk * 16
                    kb1 = _k_blk_base + fx.Int32(2 * _bs * 16) + tok_in_blk * 16
                else:
                    kb0 = _k_blk_base + mfma_row * 16
                    kb1 = _k_blk_base + fx.Int32(2 * KV_BLOCK_SIZE * 16) + mfma_row * 16
                # Load 4xi32 (16 bytes) via buffer resource instead of raw pointer
                k0_4xi32 = buffer_ops.buffer_load(k_rsrc, kb0 // 4, vec_width=4, dtype=T.i32)
                k1_4xi32 = buffer_ops.buffer_load(k_rsrc, kb1 // 4, vec_width=4, dtype=T.i32)
                kv_loads.append([k0_4xi32, k1_4xi32])
            result['kv'] = kv_loads
            return result

        def _extract_k_i64(kv_4xi32, pair_idx):
            """Extract i64 MFMA operand from 4xi32 K load. pair_idx=0 -> elements [0,1], pair_idx=1 -> elements [2,3]."""
            a = vector.extract(kv_4xi32, static_position=[pair_idx * 2])
            b = vector.extract(kv_4xi32, static_position=[pair_idx * 2 + 1])
            return _pack_i32_pair_to_i64(a, b)

        # -- Online softmax state (persists across partition iterations) --
        running_max = NEG_INF
        running_sum = ZERO_F
        acc_pv_running = [arith.constant_vector(0.0, T.f32x4) for _ in [0, 1]]

        # -- Partition loop --
        for _pi in range(int(_max_pps)):
            # Compute partition index for this iteration
            if _ps_mode:
                _pi_i32 = arith.index_cast(T.i32, _pi)
                part = part_z * int(_max_pps) + _pi_i32
            else:
                part = part_z

            # Issue BT + K loads for this iteration
            pf = _issue_bt_k_loads(part)
            kv = pf['kv']
            partition_start = pf['partition_start']
            if _use_large_block:
                phys_block = pf['phys_block']
                page_off = pf['page_off']
            else:
                phys_0 = pf['phys_0']
                phys_1 = pf['phys_1']

            # -- STEP 6: QK MFMAs (4 tiles x 4 K-chunks) --
            q_vecs = [q_a0, q_a1, q_a2, q_a3]
            zero = arith.constant_vector(0.0, T.f32x4)
            acc_qk = []
            for t in [0, 1, 2, 3]:
                k_t = [
                    _extract_k_i64(kv[t][0], 0),
                    _extract_k_i64(kv[t][0], 1),
                    _extract_k_i64(kv[t][1], 0),
                    _extract_k_i64(kv[t][1], 1),
                ]
                acc = zero
                for j in [0, 1, 2, 3]:
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_t[j], q_vecs[j], acc, 0, 0, 0])
                acc_qk.append(acc)

            # -- Masking out-of-bounds tokens --
            ctx_len = context_length_i32
            for n_tile in [0, 1, 2, 3]:
                acc_qk[n_tile] = _vsplat_mul(acc_qk[n_tile], QK_SCALE)
                for elem in [0, 1, 2, 3]:
                    kv_tok = partition_start + warp_id * 64 + fx.Int32(n_tile * 16 + elem)
                    in_b = kv_tok < ctx_len
                    v = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    acc_qk[n_tile] = vector.insert(
                        in_b.select(v, NEG_INF), acc_qk[n_tile], static_position=[elem], dynamic_position=[]
                    )

            # -- STEP 7: BT LDS staging (for V loads) --
            if _use_large_block:
                token_page_base = page_off // 16
                tp0 = token_page_base + warp_id
                tp1 = token_page_base + warp_id + 4
                tp0_i64 = _int_cast(tp0, _Int64, signed=True)
                tp1_i64 = _int_cast(tp1, _Int64, signed=True)
                bt_si = arith.index_cast(T.index, warp_id * 2)
                bt_vec = vector.from_elements(T.vec(2, T.i64), [tp0_i64, tp1_i64])
                vector.store(bt_vec, bt_lds_i64, [bt_si])
                gpu.barrier()
                bt_li = arith.index_cast(T.index, kv_col_bits // 8)
                bt_load = vector.load_op(T.vec(2, T.i64), bt_lds_i64, [bt_li])
                phys_pv_0 = _int_cast(vector.extract(bt_load, static_position=[0], dynamic_position=[]), _Int32)
                phys_pv_1 = _int_cast(vector.extract(bt_load, static_position=[1], dynamic_position=[]), _Int32)
            else:
                gpu.barrier()
                p0_i64 = _int_cast(phys_0, _Int64, signed=True)
                p1_i64 = _int_cast(phys_1, _Int64, signed=True)
                bt_si = arith.index_cast(T.index, warp_id * 2)
                bt_vec = vector.from_elements(T.vec(2, T.i64), [p0_i64, p1_i64])
                vector.store(bt_vec, bt_lds_i64, [bt_si])
                gpu.barrier()
                bt_li = arith.index_cast(T.index, kv_col_bits // 8)
                bt_load = vector.load_op(T.vec(2, T.i64), bt_lds_i64, [bt_li])
                phys_pv_0 = _int_cast(vector.extract(bt_load, static_position=[0], dynamic_position=[]), _Int32)
                phys_pv_1 = _int_cast(vector.extract(bt_load, static_position=[1], dynamic_position=[]), _Int32)

            # -- STEP 8: V batch loads (via buffer_load) --
            vv = []
            for n_tile in [0, 1]:
                h_py = n_tile * MFMA_N
                pv_pb = phys_pv_0 if n_tile == 0 else phys_pv_1
                if _use_large_block and trans_v:
                    _v_blk_base = (
                        phys_block * c_vb
                        + _v_head_off
                        + pv_pb * fx.Int32(HEAD_SIZE * 16)
                        + fx.Int32(h_py * 16)
                    )
                elif _use_large_block:
                    _v_blk_base = pv_pb * c_vb + _v_head_off + fx.Int32(h_py * _bs) + page_off
                else:
                    _v_blk_base = pv_pb * c_vb + _v_head_off + fx.Int32(h_py * KV_BLOCK_SIZE)
                nt_loads = []
                for load_i in [0, 1, 2, 3]:
                    v_off = _v_blk_base + fx.Int32(load_i * 32)
                    v_4xi32 = buffer_ops.buffer_load(v_rsrc, v_off // 4, vec_width=4, dtype=T.i32)
                    nt_loads.append(v_4xi32)
                vv.append(nt_loads)

            # -- STEP 9: Online Softmax --
            local_max = NEG_INF
            for n_tile in [0, 1, 2, 3]:
                local_max = local_max.maximumf(vector.reduction(T.f32, "maxnumf", acc_qk[n_tile]))
            wmax = _wave_max(local_max)
            s_max_p.store(wmax, [wave_idx])
            gpu.barrier()
            global_max_new = (
                s_max_p.load([_mi0])
                .maximumf(s_max_p.load([_mi1]))
                .maximumf(s_max_p.load([_mi2]))
                .maximumf(s_max_p.load([_mi3]))
            )

            if _ps_mode:
                rescale = ((running_max - global_max_new) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                acc_pv_running = [_vsplat_mul(t, rescale) for t in acc_pv_running]
                running_sum = running_sum * rescale
                running_max = global_max_new
            else:
                running_max = global_max_new

            local_sum = ZERO_F
            for n_tile in [0, 1, 2, 3]:
                for elem in [0, 1, 2, 3]:
                    s = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    p = ((s - running_max) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    local_sum = local_sum + p
                    acc_qk[n_tile] = vector.insert(p, acc_qk[n_tile], static_position=[elem], dynamic_position=[])
            wsum = _wave_add(local_sum)
            s_sum_p.store(wsum, [wave_idx])
            gpu.barrier()
            iter_sum = s_sum_p.load([_mi0]) + s_sum_p.load([_mi1]) + s_sum_p.load([_mi2]) + s_sum_p.load([_mi3])
            running_sum = running_sum + iter_sum

            # -- STEP 10: FP8 pack + prob -> LDS --
            probs = []
            for n_tile in [0, 1, 2, 3]:
                for elem in [0, 1, 2, 3]:
                    pf_v = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    probs.append(pf_v * F240)

            fp8_i32 = []
            for i in [0, 1, 2, 3]:
                lo = rocdl.cvt_pk_fp8_f32(T.i32, probs[i * 4], probs[i * 4 + 1], fx.Int32(0), False)
                wd = rocdl.cvt_pk_fp8_f32(T.i32, probs[i * 4 + 2], probs[i * 4 + 3], lo, True)
                fp8_i32.append(wd)

            gpu.barrier()
            prob_vec4 = vector.from_elements(T.vec(4, T.i32), fp8_i32)
            vector.store(prob_vec4, p_lds_i32, [arith.index_cast(T.index, tid * 4)])
            gpu.barrier()

            # -- STEP 11: P from LDS -> 8 i64 --
            _prob_base = kv_col_bits * 64 + mfma_row * 16
            p_lds_i32b = SmemPtr(base, prob_off, T.i32, shape=(PROB_LDS_BYTES // 4,)).get()

            def _load_p4i32(byte_off):
                idx = arith.index_cast(T.index, (_prob_base + fx.Int32(byte_off)) // 4)
                return vector.load_op(T.vec(4, T.i32), p_lds_i32b, [idx])

            _pa = _load_p4i32(0)
            _pb = _load_p4i32(256)
            _pc = _load_p4i32(512)
            _pd = _load_p4i32(768)

            def _pack(vec_a, vec_b, k):
                a = vector.extract(vec_a, static_position=[k])
                b = vector.extract(vec_b, static_position=[k])
                return _pack_i32_pair_to_i64(a, b)

            p_ops = [_pack(_pa, _pb, k) for k in [0, 1, 2, 3]] + [_pack(_pc, _pd, k) for k in [0, 1, 2, 3]]

            # -- STEP 12+13: PV MFMAs (2 tiles x 8 V-chunks) --
            # V data is 4xi32 from buffer_load; extract i64 pairs for MFMA
            for t in [0, 1]:
                # vv[t] has 4 loads of 4xi32 each → 8 i64 operands
                v_t = []
                for load_idx in [0, 1, 2, 3]:
                    v_t.append(_extract_k_i64(vv[t][load_idx], 0))  # elements [0,1]
                    v_t.append(_extract_k_i64(vv[t][load_idx], 1))  # elements [2,3]
                acc = acc_pv_running[t]
                for j in [0, 1, 2, 3, 4, 5, 6, 7]:
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v_t[j], p_ops[j], acc, 0, 0, 0])
                acc_pv_running[t] = acc

            # -- STEP 14: Output --
            _is_last_iter = (_pi == int(_max_pps) - 1)
            if not _ps_mode or _is_last_iter:
                if one_shot or _ps_mode:
                    rcp = fx.Float32(1.0) / running_sum
                    pv_out = [
                        _vsplat_mul(_vsplat_mul(acc_pv_running[0], PROB_SCALE_C), rcp),
                        _vsplat_mul(_vsplat_mul(acc_pv_running[1], PROB_SCALE_C), rcp),
                    ]

                    c_os = fx.Int32(_stride_out_seq)
                    c_oh = fx.Int32(_stride_out_head)
                    for n_tile in [0, 1]:
                        h_py = n_tile * MFMA_N
                        out_off = (
                            seq * c_os
                            + kv_h * c_oh
                            + mfma_row * HEAD_SIZE
                            + warp_head_base
                            + h_py
                        )
                        out_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_out[n_tile])
                        out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                        buffer_ops.buffer_store(out_i32, out_rsrc, out_off * 2, offset_is_bytes=True)
                else:
                    rcp = fx.Float32(1.0) / running_sum
                    pv_out = [
                        _vsplat_mul(_vsplat_mul(acc_pv_running[0], PROB_SCALE_C), rcp),
                        _vsplat_mul(_vsplat_mul(acc_pv_running[1], PROB_SCALE_C), rcp),
                    ]

                    c_np_qg = fx.Int32(num_partitions * QUERY_GROUP_SIZE)
                    c_qg = fx.Int32(QUERY_GROUP_SIZE)
                    ml_off = seq * fx.Int32(_stride_ml_seq) + kv_h * c_np_qg + part * c_qg + mfma_row
                    es_off = seq * fx.Int32(_stride_es_seq) + kv_h * c_np_qg + part * c_qg + mfma_row
                    buffer_ops.buffer_store(running_max, ml_rsrc, ml_off)
                    buffer_ops.buffer_store(running_sum, es_rsrc, es_off)

                    c_os = fx.Int32(_stride_out_seq)
                    c_oh = fx.Int32(_stride_out_head)
                    c_op = fx.Int32(_stride_out_part)
                    for n_tile in [0, 1]:
                        h_py = n_tile * MFMA_N
                        out_off = (
                            seq * c_os
                            + kv_h * c_oh
                            + part * c_op
                            + mfma_row * HEAD_SIZE
                            + warp_head_base
                            + h_py
                        )
                        out_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_out[n_tile])
                        out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                        buffer_ops.buffer_store(out_i32, out_rsrc, out_off * 2, offset_is_bytes=True)

    return pa_decode_dot_kernel
