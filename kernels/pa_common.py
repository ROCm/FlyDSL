# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared code for paged-attention FP8 decode kernels (CDNA + RDNA).

Contains:
  - Constants (QUERY_GROUP_SIZE, HEAD_SIZE, etc.)
  - Stride computation (compute_pa_strides)
  - Reduce kernels (build_ps_reduce_kernel, build_v2_reduce_kernel)
"""

from __future__ import annotations
import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, buffer_ops
from flydsl.expr.typing import T, Int32

# ============================================================================
# Shared constants
# ============================================================================

QUERY_GROUP_SIZE = 16
HEAD_SIZE = 128
KV_BLOCK_SIZE = 16
KV_COMPUTE_BLOCK = 256
FP8_MAX = 240.0
LOG2E = 1.4426950408889634


# ============================================================================
# Stride computation
# ============================================================================

def compute_pa_strides(
    num_kv_heads,
    num_partitions,
    max_blocks_per_seq,
    kv_block_size=16,
    trans_v=False,
    one_shot=False,
    ps_num_splits=0,
):
    """Compute all strides needed by the PA decode dot kernel.

    Returns a dict with stride values and derived parameters.
    """
    _bs = kv_block_size
    _num_heads = num_kv_heads * QUERY_GROUP_SIZE

    s = dict(
        stride_q_seq=_num_heads * HEAD_SIZE,
        stride_q_head=HEAD_SIZE,
        stride_k_block=num_kv_heads * (HEAD_SIZE // 16) * _bs * 16,
        stride_k_head=(HEAD_SIZE // 16) * _bs * 16,
        stride_bt_seq=max_blocks_per_seq,
    )

    if trans_v:
        s["stride_v_block"] = num_kv_heads * (_bs // 16) * HEAD_SIZE * 16
        s["stride_v_head"] = (_bs // 16) * HEAD_SIZE * 16
    else:
        s["stride_v_block"] = num_kv_heads * HEAD_SIZE * _bs
        s["stride_v_head"] = HEAD_SIZE * _bs

    _direct_output = one_shot or (ps_num_splits > 0)
    if _direct_output:
        s["stride_out_part"] = 0
        s["stride_out_head"] = QUERY_GROUP_SIZE * HEAD_SIZE
        s["stride_out_seq"] = num_kv_heads * QUERY_GROUP_SIZE * HEAD_SIZE
    else:
        s["stride_out_part"] = QUERY_GROUP_SIZE * HEAD_SIZE
        s["stride_out_head"] = num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE
        s["stride_out_seq"] = num_kv_heads * num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE

    s["stride_es_seq"] = num_kv_heads * num_partitions * QUERY_GROUP_SIZE
    s["stride_ml_seq"] = num_kv_heads * num_partitions * QUERY_GROUP_SIZE

    # Derived
    s["use_large_block"] = _bs > KV_BLOCK_SIZE
    s["partitions_per_block"] = _bs // KV_COMPUTE_BLOCK if s["use_large_block"] else 1
    s["blocks_per_partition"] = KV_COMPUTE_BLOCK // _bs if not s["use_large_block"] else 1
    s["ps_mode"] = ps_num_splits > 0
    s["max_pps"] = _math.ceil(num_partitions / ps_num_splits) if s["ps_mode"] else 1

    return s


# ============================================================================
# Reduce kernels (architecture-independent scalar code)
# ============================================================================

NEG_INF_VAL = float("-inf")


def _exp_f32(x, log2e_const):
    """Compute exp(x) = exp2(x * LOG2E) using hardware exp2."""
    return (x * log2e_const).exp2(fastmath=arith.FastMathFlags.fast)


def build_ps_reduce_kernel(
    head_size: int,
    query_group_size: int,
    query_seq_len: int,
    max_context_partition_num: int,
    use_sinks: bool = False,
):
    """Build the ps_reduce kernel (fixed MAX_CONTEXT_PARTITION_NUM, single-pass)."""
    qg_total = query_seq_len * query_group_size

    @flyc.kernel
    def ps_reduce_kernel(
        output_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        partial_output_ptr: fx.Tensor,
        sink_token_ptr: fx.Tensor,
        context_partition_num: Int32,
        stride_output_bs: Int32,
        stride_output_len: Int32,
        stride_output_kv_head: Int32,
        stride_output_group: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_es_part: Int32,
        stride_po_seq: Int32,
        stride_po_head: Int32,
        stride_po_part: Int32,
        stride_po_group: Int32,
    ):
        seq_idx = gpu.block_idx.x
        kv_head_idx = gpu.block_idx.y

        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_output_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(output_ptr, max_size=True)

        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)

        tid = gpu.thread_idx.x

        for qg in range(qg_total):
            qg_i32 = fx.Int32(qg)
            ql_idx = fx.Int32(qg // query_group_size)
            gr_idx = fx.Int32(qg % query_group_size)

            global_max = NEG_INF
            for p in range(max_context_partition_num):
                p_i32 = fx.Int32(p)
                p_valid = p_i32 < context_partition_num
                es_off = seq_idx * stride_es_seq + kv_head_idx * stride_es_head + p_i32 * stride_es_part + qg_i32
                ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                ml_val = p_valid.select(ml_val, NEG_INF)
                global_max = global_max.maximumf(ml_val)

            total_exp_sum = ZERO_F
            for p in range(max_context_partition_num):
                p_i32 = fx.Int32(p)
                p_valid = p_i32 < context_partition_num
                es_off = seq_idx * stride_es_seq + kv_head_idx * stride_es_head + p_i32 * stride_es_part + qg_i32
                ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                ml_val = p_valid.select(ml_val, NEG_INF)
                es_val = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                es_val = p_valid.select(es_val, ZERO_F)
                rescaled = es_val * _exp_f32(ml_val - global_max, LOG2E_C)
                total_exp_sum = total_exp_sum + rescaled

            if use_sinks:
                sink_rsrc = buffer_ops.create_buffer_resource(sink_token_ptr, max_size=True)
                sink_off = kv_head_idx * fx.Int32(query_group_size) + gr_idx
                sink_val = buffer_ops.buffer_load(sink_rsrc, sink_off, vec_width=1, dtype=T.f32)
                sink_contrib = _exp_f32(sink_val - global_max, LOG2E_C)
                total_exp_sum = total_exp_sum + sink_contrib

            for h in range(head_size):
                h_i32 = fx.Int32(h)
                acc = ZERO_F
                for p in range(max_context_partition_num):
                    p_i32 = fx.Int32(p)
                    p_valid = p_i32 < context_partition_num
                    es_off = seq_idx * stride_es_seq + kv_head_idx * stride_es_head + p_i32 * stride_es_part + qg_i32
                    ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                    ml_val = p_valid.select(ml_val, NEG_INF)
                    es_val = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                    es_val = p_valid.select(es_val, ZERO_F)
                    rescaled = es_val * _exp_f32(ml_val - global_max, LOG2E_C)
                    attn_prob = rescaled / total_exp_sum
                    po_off = (
                        seq_idx * stride_po_seq
                        + kv_head_idx * stride_po_head
                        + p_i32 * stride_po_part
                        + qg_i32 * stride_po_group
                        + h_i32
                    )
                    po_val = buffer_ops.buffer_load(po_rsrc, po_off, vec_width=1, dtype=T.f32)
                    po_val = p_valid.select(po_val, ZERO_F)
                    acc = acc + po_val * attn_prob

                out_off = (
                    seq_idx * stride_output_bs
                    + ql_idx * stride_output_len
                    + kv_head_idx * stride_output_kv_head
                    + gr_idx * stride_output_group
                    + h_i32
                )
                buffer_ops.buffer_store(acc, out_rsrc, out_off)

    @flyc.jit
    def launch_ps_reduce(
        output: fx.Tensor,
        exp_sums: fx.Tensor,
        max_logits: fx.Tensor,
        partial_output: fx.Tensor,
        sink_token: fx.Tensor,
        context_partition_num: Int32,
        stride_output_bs: Int32,
        stride_output_len: Int32,
        stride_output_kv_head: Int32,
        stride_output_group: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_es_part: Int32,
        stride_po_seq: Int32,
        stride_po_head: Int32,
        stride_po_part: Int32,
        stride_po_group: Int32,
        num_seqs: Int32,
        num_kv_heads: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ps_reduce_kernel(
            output, exp_sums, max_logits, partial_output, sink_token,
            context_partition_num,
            stride_output_bs, stride_output_len,
            stride_output_kv_head, stride_output_group,
            stride_es_seq, stride_es_head, stride_es_part,
            stride_po_seq, stride_po_head, stride_po_part, stride_po_group,
        ).launch(
            grid=(num_seqs, num_kv_heads),
            block=(1,),
            stream=stream,
        )

    return ps_reduce_kernel, launch_ps_reduce


def build_v2_reduce_kernel(
    head_size: int,
    query_group_size: int,
    query_seq_len: int,
    context_partition_size: int,
    max_chunk_size: int = 16,
    use_sinks: bool = False,
):
    """Build the v2_reduce kernel (dynamic partition count, two-pass loop)."""
    qg_total = query_seq_len * query_group_size

    @flyc.kernel
    def v2_reduce_kernel(
        output_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        partial_output_ptr: fx.Tensor,
        context_lengths_ptr: fx.Tensor,
        sink_token_ptr: fx.Tensor,
        stride_output_bs: Int32,
        stride_output_len: Int32,
        stride_output_kv_head: Int32,
        stride_output_group: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_es_part: Int32,
        stride_po_seq: Int32,
        stride_po_head: Int32,
        stride_po_part: Int32,
        stride_po_group: Int32,
    ):
        seq_idx = gpu.block_idx.x
        kv_head_idx = gpu.block_idx.y

        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        ctx_len = buffer_ops.buffer_load(cl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        cps_const = fx.Int32(context_partition_size)
        context_partition_num = (ctx_len + cps_const - 1) / cps_const

        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_output_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(output_ptr, max_size=True)

        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        CHUNK = fx.Int32(max_chunk_size)

        for qg in range(qg_total):
            qg_i32 = fx.Int32(qg)
            ql_idx = fx.Int32(qg // query_group_size)
            gr_idx = fx.Int32(qg % query_group_size)

            global_max = NEG_INF
            global_exp_sum = ZERO_F

            for chunk_base in range(0, max_chunk_size * 64, max_chunk_size):
                chunk_base_i32 = fx.Int32(chunk_base)
                chunk_active = chunk_base_i32 < context_partition_num
                prev_global_max = global_max

                for p_in_chunk in range(max_chunk_size):
                    p_i32 = chunk_base_i32 + fx.Int32(p_in_chunk)
                    p_valid = p_i32 < context_partition_num
                    es_off = seq_idx * stride_es_seq + kv_head_idx * stride_es_head + p_i32 * stride_es_part + qg_i32
                    ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                    ml_val = p_valid.select(ml_val, NEG_INF)
                    global_max = global_max.maximumf(ml_val)

                update_scale = _exp_f32(prev_global_max - global_max, LOG2E_C)
                global_exp_sum = global_exp_sum * update_scale

                for p_in_chunk in range(max_chunk_size):
                    p_i32 = chunk_base_i32 + fx.Int32(p_in_chunk)
                    p_valid = p_i32 < context_partition_num
                    es_off = seq_idx * stride_es_seq + kv_head_idx * stride_es_head + p_i32 * stride_es_part + qg_i32
                    ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                    ml_val = p_valid.select(ml_val, NEG_INF)
                    es_val = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                    es_val = p_valid.select(es_val, ZERO_F)
                    rescaled = es_val * _exp_f32(ml_val - global_max, LOG2E_C)
                    global_exp_sum = global_exp_sum + rescaled

            if use_sinks:
                sink_rsrc = buffer_ops.create_buffer_resource(sink_token_ptr, max_size=True)
                sink_off = kv_head_idx * fx.Int32(query_seq_len * query_group_size) + qg_i32
                sink_val = buffer_ops.buffer_load(sink_rsrc, sink_off, vec_width=1, dtype=T.f32)
                sink_contrib = _exp_f32(sink_val - global_max, LOG2E_C)
                global_exp_sum = global_exp_sum + sink_contrib

            for h in range(head_size):
                h_i32 = fx.Int32(h)
                acc = ZERO_F

                for chunk_base in range(0, max_chunk_size * 64, max_chunk_size):
                    chunk_base_i32 = fx.Int32(chunk_base)
                    for p_in_chunk in range(max_chunk_size):
                        p_i32 = chunk_base_i32 + fx.Int32(p_in_chunk)
                        p_valid = p_i32 < context_partition_num
                        es_off = (
                            seq_idx * stride_es_seq + kv_head_idx * stride_es_head + p_i32 * stride_es_part + qg_i32
                        )
                        ml_val = buffer_ops.buffer_load(ml_rsrc, es_off, vec_width=1, dtype=T.f32)
                        ml_val = p_valid.select(ml_val, NEG_INF)
                        es_val = buffer_ops.buffer_load(es_rsrc, es_off, vec_width=1, dtype=T.f32)
                        es_val = p_valid.select(es_val, ZERO_F)
                        rescaled = es_val * _exp_f32(ml_val - global_max, LOG2E_C)
                        attn_prob = rescaled / global_exp_sum
                        po_off = (
                            seq_idx * stride_po_seq
                            + kv_head_idx * stride_po_head
                            + p_i32 * stride_po_part
                            + qg_i32 * stride_po_group
                            + h_i32
                        )
                        po_val = buffer_ops.buffer_load(po_rsrc, po_off, vec_width=1, dtype=T.f32)
                        po_val = p_valid.select(po_val, ZERO_F)
                        acc = acc + po_val * attn_prob

                out_off = (
                    seq_idx * stride_output_bs
                    + ql_idx * stride_output_len
                    + kv_head_idx * stride_output_kv_head
                    + gr_idx * stride_output_group
                    + h_i32
                )
                buffer_ops.buffer_store(acc, out_rsrc, out_off)

    @flyc.jit
    def launch_v2_reduce(
        output: fx.Tensor,
        exp_sums: fx.Tensor,
        max_logits: fx.Tensor,
        partial_output: fx.Tensor,
        context_lengths: fx.Tensor,
        sink_token: fx.Tensor,
        stride_output_bs: Int32,
        stride_output_len: Int32,
        stride_output_kv_head: Int32,
        stride_output_group: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_es_part: Int32,
        stride_po_seq: Int32,
        stride_po_head: Int32,
        stride_po_part: Int32,
        stride_po_group: Int32,
        num_seqs: Int32,
        num_kv_heads: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        v2_reduce_kernel(
            output, exp_sums, max_logits, partial_output, context_lengths, sink_token,
            stride_output_bs, stride_output_len,
            stride_output_kv_head, stride_output_group,
            stride_es_seq, stride_es_head, stride_es_part,
            stride_po_seq, stride_po_head, stride_po_part, stride_po_group,
        ).launch(
            grid=(num_seqs, num_kv_heads),
            block=(1,),
            stream=stream,
        )

    return v2_reduce_kernel, launch_v2_reduce
