# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
RDNA4 paged-attention FP8 decode kernel using WMMA instructions.

Ported from CDNA MFMA-based kernel (pa_decode_fp8.py) to RDNA4 wave32 WMMA.

Architecture:
  - 8 warps x 32 lanes = 256 threads per workgroup
  - WMMA f32_16x16x16_fp8_fp8 for both QK and PV dot products
  - QK: each warp handles 2 N-tiles (32 KV tokens), 8 K-steps (128 head dim)
  - PV: each warp handles 1 N-tile (16 head dims), 16 K-steps (256 tokens)
  - Softmax P staged through LDS as f32 for PV consumption

WMMA 16x16x16 FP8 lane mapping (wave32):
  - A operand (v2i32 = 8 FP8 bytes): lane16 selects M-row, klane selects K-half
  - B operand (v2i32 = 8 FP8 bytes): lane16 selects N-col, klane selects K-half
  - C/D accumulator (v8f32): element i -> row = klane*8+i, col = lane16

Current scope (minimal supported subset for bring-up):
  - kv_block_size=16 only
  - trans_v=False only
  - one_shot=False only
  - ps_num_splits=0 only
"""

from __future__ import annotations

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

WMMA_M = WMMA_N = WMMA_K = 16
WARP_SIZE = 32
NUM_WARPS = 8
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
QK_N_TILES_WARP = 2
QK_K_STEPS = HEAD_SIZE // WMMA_K
PV_K_STEPS = KV_COMPUTE_BLOCK // WMMA_K
BLOCKS_PER_PARTITION = KV_COMPUTE_BLOCK // KV_BLOCK_SIZE

P_LDS_F32_COUNT = BLOCKS_PER_PARTITION * QUERY_GROUP_SIZE * KV_BLOCK_SIZE
P_LDS_BYTES = P_LDS_F32_COUNT * 4
RED_SLOTS = NUM_WARPS


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
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _qk_scale = float(softmax_scale * query_scale * key_scale)
    _prob_scale = float(value_scale / FP8_MAX)

    assert kv_block_size == KV_BLOCK_SIZE, "Only kv_block_size=16 supported for RDNA"
    assert not trans_v, "trans_v not yet supported for RDNA"

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
    _blocks_per_partition = _S["blocks_per_partition"]
    _ps_mode = _S["ps_mode"]
    _max_pps = _S["max_pps"]

    # -- LDS allocation --
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_smem")
    p_off = 0
    allocator.ptr = P_LDS_BYTES
    rmax_off = P_LDS_BYTES
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
        v8f32 = T.vec(8, T.f32)
        v2i32 = T.i32x2

        def _vsplat_mul_8(vec, scalar):
            s = scalar.ir_value() if hasattr(scalar, "ir_value") else scalar
            return vec * vector.broadcast(v8f32, s)

        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        part_z = gpu.block_idx.z

        warp_id = tid >> 5
        lane = tid & 31
        lane16 = lane & 15
        klane = lane >> 4
        c_w = fx.Int32(WARP_SIZE)

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        base = allocator.get_base()
        p_smem = SmemPtr(base, p_off, T.f32, shape=(P_LDS_F32_COUNT,))
        p_lds_f32 = p_smem.get()
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

        _q_byte_base = seq * c_sq + kv_h * fx.Int32(QUERY_GROUP_SIZE) * c_qh
        _k_head_off = kv_h * c_kh
        _v_head_off = kv_h * c_vh

        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = fx.Float32(0.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)
        F_FP8MAX = arith.constant(FP8_MAX, type=T.f32)
        PROB_SCALE_C = arith.constant(_prob_scale, type=T.f32)

        _mi0 = arith.index_cast(T.index, fx.Int32(0))
        _mi1 = arith.index_cast(T.index, fx.Int32(1))
        _mi2 = arith.index_cast(T.index, fx.Int32(2))
        _mi3 = arith.index_cast(T.index, fx.Int32(3))
        _mi4 = arith.index_cast(T.index, fx.Int32(4))
        _mi5 = arith.index_cast(T.index, fx.Int32(5))
        _mi6 = arith.index_cast(T.index, fx.Int32(6))
        _mi7 = arith.index_cast(T.index, fx.Int32(7))

        # -- wave-level reductions for 32-lane waves --
        def _wave_max(x):
            w = x
            for sh in [16, 8, 4, 2, 1]:
                w = w.maximumf(w.shuffle_xor(fx.Int32(sh), c_w))
            return w

        def _wave_add(x):
            w = x
            for sh in [16, 8, 4, 2, 1]:
                w = w + w.shuffle_xor(fx.Int32(sh), c_w)
            return w

        def _load_q(rk):
            """Q WMMA-A operand for K-step rk.  v2i32 = 8 FP8 bytes."""
            off = _q_byte_base // 4 + lane16 * fx.Int32(HEAD_SIZE // 4) + fx.Int32(rk * (WMMA_K // 4)) + klane * fx.Int32(2)
            return buffer_ops.buffer_load(q_rsrc, off, vec_width=2, dtype=T.i32)

        def _load_k(phys_block, rk):
            """K WMMA-B operand.  v2i32 = 8 FP8 bytes."""
            k_base = (phys_block * c_kb + _k_head_off) // 4
            off = k_base + fx.Int32(rk * (_bs * 16 // 4)) + lane16 * fx.Int32(16 // 4) + klane * fx.Int32(2)
            return buffer_ops.buffer_load(k_rsrc, off, vec_width=2, dtype=T.i32)

        def _load_v(phys_block, warp_hd_base):
            """V WMMA-B operand.  V cache is [head_dim, token] within (block,head)."""
            v_base = (phys_block * c_vb + _v_head_off) // 4
            off = v_base + (warp_hd_base + lane16) * fx.Int32(_bs // 4) + klane * fx.Int32(2)
            return buffer_ops.buffer_load(v_rsrc, off, vec_width=2, dtype=T.i32)

        def _bt_load(part_val, idx):
            """Load block-table entry idx for partition part_val."""
            bt_start = part_val * fx.Int32(_blocks_per_partition)
            return buffer_ops.buffer_load(bt_rsrc, seq * c_bt + bt_start + idx, vec_width=1, dtype=T.i32)

        def _f32x8_to_fp8(vals):
            """Pack 8 scaled-f32 values into v2i32 (WMMA FP8 operand)."""
            lo = rocdl.cvt_pk_fp8_f32(T.i32, vals[0], vals[1], fx.Int32(0), False)
            lo = rocdl.cvt_pk_fp8_f32(T.i32, vals[2], vals[3], lo, True)
            hi = rocdl.cvt_pk_fp8_f32(T.i32, vals[4], vals[5], fx.Int32(0), False)
            hi = rocdl.cvt_pk_fp8_f32(T.i32, vals[6], vals[7], hi, True)
            return vector.from_elements(v2i32, [lo, hi])

        # -- online softmax state --
        running_max = NEG_INF
        running_sum = ZERO_F
        acc_pv = arith.constant_vector(0.0, v8f32)

        # -- partition loop --
        for _pi in range(int(_max_pps)):
            if _ps_mode:
                _pi_i32 = arith.index_cast(T.i32, _pi)
                part = part_z * int(_max_pps) + _pi_i32
            else:
                part = part_z

            partition_start = part * fx.Int32(KV_COMPUTE_BLOCK)

            # == QK: Q[16,128] x K^T[128,256] -> S[16,256] ==
            zero_acc = arith.constant_vector(0.0, v8f32)
            acc_qk = [zero_acc, zero_acc]

            for nt in [0, 1]:
                bt_idx = warp_id * fx.Int32(2) + fx.Int32(nt)
                phys_block = _bt_load(part, bt_idx)
                acc = zero_acc
                for rk in [0, 1, 2, 3, 4, 5, 6, 7]:
                    q_op = _load_q(rk)
                    k_op = _load_k(phys_block, rk)
                    acc = rocdl.wmma_f32_16x16x16_fp8_fp8(
                        v8f32, q_op, k_op, arith.unwrap(acc)
                    ).result
                acc_qk[nt] = acc

            # -- scale + mask --
            ctx_len = context_length_i32
            for nt in [0, 1]:
                acc_qk[nt] = _vsplat_mul_8(acc_qk[nt], QK_SCALE)
                kv_tok = partition_start + warp_id * fx.Int32(32) + fx.Int32(nt * 16) + lane16
                in_b = kv_tok < ctx_len
                for elem in [0, 1, 2, 3, 4, 5, 6, 7]:
                    v = vector.extract(acc_qk[nt], static_position=[elem], dynamic_position=[])
                    acc_qk[nt] = vector.insert(
                        in_b.select(v, NEG_INF), acc_qk[nt],
                        static_position=[elem], dynamic_position=[],
                    )

            # == online softmax ==
            local_max = NEG_INF
            for nt in [0, 1]:
                local_max = local_max.maximumf(vector.reduction(T.f32, "maxnumf", acc_qk[nt]))
            wmax = _wave_max(local_max)
            s_max_p.store(wmax, [wave_idx])
            gpu.barrier()
            global_max_new = (
                s_max_p.load([_mi0]).maximumf(s_max_p.load([_mi1]))
                .maximumf(s_max_p.load([_mi2])).maximumf(s_max_p.load([_mi3]))
                .maximumf(s_max_p.load([_mi4])).maximumf(s_max_p.load([_mi5]))
                .maximumf(s_max_p.load([_mi6])).maximumf(s_max_p.load([_mi7]))
            )

            if _ps_mode:
                rescale = ((running_max - global_max_new) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                acc_pv = _vsplat_mul_8(acc_pv, rescale)
                running_sum = running_sum * rescale
                running_max = global_max_new
            else:
                running_max = global_max_new

            local_sum = ZERO_F
            for nt in [0, 1]:
                for elem in [0, 1, 2, 3, 4, 5, 6, 7]:
                    s = vector.extract(acc_qk[nt], static_position=[elem], dynamic_position=[])
                    p = ((s - running_max) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    local_sum = local_sum + p
                    acc_qk[nt] = vector.insert(p, acc_qk[nt], static_position=[elem], dynamic_position=[])
            wsum = _wave_add(local_sum)
            s_sum_p.store(wsum, [wave_idx])
            gpu.barrier()
            iter_sum = (
                s_sum_p.load([_mi0]) + s_sum_p.load([_mi1])
                + s_sum_p.load([_mi2]) + s_sum_p.load([_mi3])
                + s_sum_p.load([_mi4]) + s_sum_p.load([_mi5])
                + s_sum_p.load([_mi6]) + s_sum_p.load([_mi7])
            )
            running_sum = running_sum + iter_sum

            # == store P (f32) to LDS ==
            # Layout: P_lds[tile_abs][query][token_in_tile], f32 index =
            #   tile_abs*256 + query*16 + token_in_tile
            # tile_abs = warp_id*2+nt, query = klane*8+elem, token = lane16
            gpu.barrier()
            for nt in [0, 1]:
                tile_abs = warp_id * fx.Int32(2) + fx.Int32(nt)
                for elem in [0, 1, 2, 3, 4, 5, 6, 7]:
                    q_idx = klane * fx.Int32(8) + fx.Int32(elem)
                    p_idx = tile_abs * fx.Int32(256) + q_idx * fx.Int32(16) + lane16
                    p_val = vector.extract(acc_qk[nt], static_position=[elem], dynamic_position=[])
                    p_smem.store(p_val, [arith.index_cast(T.index, p_idx)])
            gpu.barrier()

            # == PV: P[16,256] x V[256,128] -> O[16,128] ==
            warp_hd_base = warp_id * fx.Int32(WMMA_N)
            for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                # P from LDS: 8 contiguous f32 -> query=lane16, tokens=klane*8..+7
                p_lds_idx = fx.Int32(s * 256) + lane16 * fx.Int32(16) + klane * fx.Int32(8)
                p_f32 = vector.load_op(v8f32, p_lds_f32, [arith.index_cast(T.index, p_lds_idx)])
                p_scaled = [
                    vector.extract(p_f32, static_position=[i], dynamic_position=[]) * F_FP8MAX
                    for i in [0, 1, 2, 3, 4, 5, 6, 7]
                ]
                p_fp8 = _f32x8_to_fp8(p_scaled)

                phys_v = _bt_load(part, fx.Int32(s))
                v_fp8 = _load_v(phys_v, warp_hd_base)

                acc_pv = rocdl.wmma_f32_16x16x16_fp8_fp8(
                    v8f32, p_fp8, v_fp8, arith.unwrap(acc_pv)
                ).result

            # == output ==
            _is_last_iter = _pi == int(_max_pps) - 1
            if not _ps_mode or _is_last_iter:
                rcp = fx.Float32(1.0) / running_sum
                pv_out = _vsplat_mul_8(_vsplat_mul_8(acc_pv, PROB_SCALE_C), rcp)

                if one_shot or _ps_mode:
                    c_os = fx.Int32(_stride_out_seq)
                    c_oh = fx.Int32(_stride_out_head)
                    for elem in [0, 1, 2, 3, 4, 5, 6, 7]:
                        q = klane * fx.Int32(8) + fx.Int32(elem)
                        out_off = seq * c_os + kv_h * c_oh + q * fx.Int32(HEAD_SIZE) + warp_hd_base + lane16
                        val = arith.trunc_f(T.bf16, vector.extract(pv_out, static_position=[elem], dynamic_position=[]))
                        buffer_ops.buffer_store(val, out_rsrc, out_off)
                else:
                    c_np_qg = fx.Int32(num_partitions * QUERY_GROUP_SIZE)
                    c_qg = fx.Int32(QUERY_GROUP_SIZE)
                    for elem in [0, 1, 2, 3, 4, 5, 6, 7]:
                        q = klane * fx.Int32(8) + fx.Int32(elem)
                        ml_off = seq * fx.Int32(_stride_ml_seq) + kv_h * c_np_qg + part * c_qg + q
                        es_off = seq * fx.Int32(_stride_es_seq) + kv_h * c_np_qg + part * c_qg + q
                        buffer_ops.buffer_store(running_max, ml_rsrc, ml_off)
                        buffer_ops.buffer_store(running_sum, es_rsrc, es_off)

                    c_os = fx.Int32(_stride_out_seq)
                    c_oh = fx.Int32(_stride_out_head)
                    c_op = fx.Int32(_stride_out_part)
                    for elem in [0, 1, 2, 3, 4, 5, 6, 7]:
                        q = klane * fx.Int32(8) + fx.Int32(elem)
                        out_off = (
                            seq * c_os + kv_h * c_oh + part * c_op
                            + q * fx.Int32(HEAD_SIZE) + warp_hd_base + lane16
                        )
                        val = arith.trunc_f(T.bf16, vector.extract(pv_out, static_position=[elem], dynamic_position=[]))
                        buffer_ops.buffer_store(val, out_rsrc, out_off)

    return pa_decode_dot_kernel

