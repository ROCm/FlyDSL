# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Single-launch Kimi-K3 KDA + latent-MoE decode layer kernel.

One cooperative launch performs both AttnRes mixers, KDA, router and latent/shared
projections, MXFP4 experts, both TP8 reductions, and the final residual update.
Intermediate values use launch-tagged scratch mailboxes so dependent CTAs can
make progress without a grid barrier.

The resident-grid schedule, tagged-pair mailbox protocol, and phase-overlaid
shared-memory arena follow the TileRT/GLM MonoKernel design documented at
``kernels/monokernel/glm/kernel.py`` and its upstream reference:
https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
The ordered KDA snapshot pipeline and its state-resident MTP recurrence are
Kimi-K3-specific extensions.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.common.act import sigmoid_batch
from kernels.monokernel.config import (
    EPS,
    FP8_MAX,
    MAX_LAYERS_PER_STEP,
    ConvStateLayout,
    conv_state_offset,
)
from kernels.monokernel.layout import CM_DEV, CM_SYS
from kernels.monokernel.ops import (
    exp,
    fp8_to_bf16x8,
    mxfp4_to_bf16x8,
    mxfp8_to_bf16x8,
    rcp,
    rsq,
    rsrc,
    uniform,
    uniform_f32,
    xred,
    xshfl,
)

_BLOCKS = 256
_THREADS = 512
_WAVE_SIZE = 64
_WAVES = _THREADS // _WAVE_SIZE
_HEADS = 12
_HEAD_DIM = 128
_HIDDEN = 7168
_PROJECTION = _HEADS * _HEAD_DIM
_FUSED_WIDTH = 4 * _PROJECTION + _HEADS + _HEAD_DIM
_FUSED_PAD = 6400
_INPUT_ROW_GROUPS = 4
_INPUT_SPLIT_WAVES = 2
_OUTPUT_ROW_GROUPS = _WAVES // 2
_OUTPUT_SPLIT_WAVES = 2
_ROUTER_ROW_GROUPS = 1
_ROUTER_SPLIT_WAVES = _WAVES
_INPUT_ROW_TILE = _INPUT_ROW_GROUPS * 16
_OUTPUT_ROW_TILE = _OUTPUT_ROW_GROUPS * 16
_ROUTER_ROW_TILE = _ROUTER_ROW_GROUPS * 16
_INPUT_TASKS = _FUSED_PAD // _INPUT_ROW_TILE
_OUTPUT_TASKS = _HIDDEN // _OUTPUT_ROW_TILE
_CONV_CHANNELS = 3 * _PROJECTION
_CONV_STATE_LENGTH = 3
_CONV_KERNEL_WIDTH = 4
_K_LANES = 8
_V_LANES = _WAVE_SIZE // _K_LANES
_VALUES_PER_THREAD = 4
_K_TILE = _K_LANES * _VALUES_PER_THREAD
_K_ITERS = _HEAD_DIM // _K_TILE
_V_TILE = _WAVES * _V_LANES
_V_ITERS = _HEAD_DIM // _V_TILE
_Q_SCALE = _HEAD_DIM**-0.5
_GATE_LOWER_BOUND = -5.0
_MTP_SPLITS = 2
_ATTN_RES_CTAS = 4
_ATTN_RES_STATS = 19
_N_EXPERTS = 896
_TOP_K = 16
_ROUTED_HIDDEN = 3584
_INTER = 384
_SHARED_INTER = 768
_DENSE_INTER = 33792 // 8
_HIDDEN_SHARD = _HIDDEN // 8
_MLA_Q_LORA = 1536
_MLA_KV_LORA = 512
_MLA_PE = 64
_MLA_Q_HEAD = 192
_MLA_CACHE_ROW = _MLA_KV_LORA + _MLA_PE
_MLA_VALUE_CHUNKS = _MLA_KV_LORA // _WAVE_SIZE
_MLA_NORM_EPS = 1.0e-6


def mtp_conv_waits_for_previous(*, agentic_batch_size: int, token):
    """Return the true data dependency for legacy sequential MTP convolution."""

    if agentic_batch_size:
        return False
    return token > 0


def agentic_conv_writeback_requires_all(token):
    """Return whether this q8 token publishes the rolled-back conv window."""

    return token == 7


def agentic_recurrence_tokens_per_cta(*, agentic_batch_size: int) -> int:
    """Keep enough q8 state resident without under-filling the GPU."""

    return 4 if agentic_batch_size else 2


def symmetric_mailbox_epoch(step_value, *, launches_per_step: int, layer):
    """Return the launch epoch shared by symmetric mailbox tags and slots."""

    return step_value * launches_per_step + layer


def monokernel_layout(
    samples: int,
    *,
    fuse_attn_res: bool = False,
    fuse_moe: bool = False,
    mtp: bool = False,
    mla: bool = False,
    dense_ffn: bool = False,
) -> dict[str, int]:
    """Return byte offsets for the MonoKernel's tagged mailboxes."""

    supported = (
        {8, 16, 32}
        if mla
        else ({1, 2, 4, 8, 16, 32, 64} if mtp else {1, 2, 4, 8})
    )
    if samples not in supported:
        raise ValueError(
            f"samples must be one of {sorted(supported)}, got {samples}"
        )
    offsets = {
        "pre": 0,
        "pre_ready": 0,
        "moe": 0,
        "moe_ready": 0,
        "mxfp8": 0,
        "mxfp8_scale": 0,
        "input": 0,
        "norm": 0,
        "norm_packed": 0,
        "attention": 0,
        "mtp_state_handoff": 0,
    }
    offset = 0
    if fuse_attn_res:
        offsets["pre"] = offset
        offset += samples * _HIDDEN * 2
        offsets["pre_ready"] = offset
        offset += samples * _ATTN_RES_CTAS * 8
        offsets["moe"] = offset
        offset += samples * _HIDDEN * 2
        offsets["moe_ready"] = offset
        offset += samples * _ATTN_RES_CTAS * 8
        if fuse_moe:
            offsets["mxfp8"] = offset
            offsets["mxfp8_scale"] = offset
    offsets["input"] = offset
    offset += samples * _FUSED_PAD * 4
    offsets["norm"] = offset
    offset += samples * _PROJECTION * (4 if (mtp or mla) else 2)
    if mtp:
        offsets["norm_packed"] = offset
        offset += samples * _PROJECTION * 2
    else:
        offsets["norm_packed"] = offsets["norm"]
    offsets["norm_ready"] = offset
    offset += samples * _HEADS * 8
    if fuse_attn_res:
        offsets["attention"] = offset
        offset += samples * _HIDDEN * 4
        offsets["pre_stats"] = offset
        offset += samples * _ATTN_RES_CTAS * _ATTN_RES_STATS * 8
        offsets["post_stats"] = offset
        offset += samples * _ATTN_RES_CTAS * _ATTN_RES_STATS * 8
    if fuse_moe:
        ffn_inter = _DENSE_INTER if dense_ffn else _SHARED_INTER
        routed_tiles = _ROUTED_HIDDEN // 16
        for name, size in (
            ("router", samples * _N_EXPERTS * 4),
            ("router_ready", samples * (_N_EXPERTS // 16) * 8),
            ("latent", samples * _ROUTED_HIDDEN * 2),
            ("latent_ready", samples * routed_tiles * 8),
            ("shared_gu", samples * (2 * ffn_inter) * 2),
            ("shared_gu_ready", samples * ((2 * ffn_inter) // 16) * 8),
            ("shared_mid", samples * ffn_inter * 4),
            ("selection_id", samples * _TOP_K * 8),
            ("selection_weight", samples * _TOP_K * 8),
            ("expert_mid", samples * _TOP_K * _INTER * 2),
            ("expert_mid_ready", samples * _TOP_K * (_INTER // 16) * 8),
            ("routed", samples * _ROUTED_HIDDEN * 2),
            ("routed_stats", samples * routed_tiles * 8),
            ("routed_inv", samples * 8),
        ):
            offsets[name] = offset
            offset += size
    if mtp:
        mtp_splits = 4
        offsets["mtp_qkvg"] = offset
        offset += samples * 4 * _PROJECTION * 2
        offsets["mtp_conv_ready"] = offset
        offset += samples * _HEADS * 8
        offsets["mtp_state_ready"] = offset
        offset += samples * _HEADS * mtp_splits * 8
        offsets["mtp_state_handoff"] = offset
        offset += (samples // 8) * _HEADS * _HEAD_DIM * _HEAD_DIM * 4
        offsets["mtp_norm_ready"] = offset
        offset += samples * _HEADS * 8
    if mla:
        for name, elements, element_bytes in (
            ("mla_qkv", samples * (_MLA_Q_LORA + _MLA_CACHE_ROW), 4),
            ("mla_qnorm", samples * _MLA_Q_LORA, 4),
            ("mla_q", samples * _HEADS * _MLA_Q_HEAD, 4),
            ("mla_fresh", samples * _MLA_CACHE_ROW, 4),
            ("mla_qlat", samples * _HEADS * _MLA_KV_LORA, 4),
            ("mla_dense_acc", samples * _HEADS * _MLA_KV_LORA, 4),
            ("mla_gate", samples * _PROJECTION, 4),
        ):
            offsets[name] = offset
            offset += elements * element_bytes
    offsets["_bytes"] = offset
    return offsets


def monokernel_scratch_nbytes(
    samples: int,
    *,
    fuse_attn_res: bool = False,
    fuse_moe: bool = False,
    mtp: bool = False,
    mla: bool = False,
    dense_ffn: bool = False,
) -> int:
    """Bytes required by tagged BF16-pair projection mailboxes."""

    return monokernel_layout(
        samples,
        fuse_attn_res=fuse_attn_res,
        fuse_moe=fuse_moe,
        mtp=mtp,
        mla=mla,
        dense_ffn=dense_ffn,
    )["_bytes"]


@functools.cache
def build_kimi_k3_monokernel(
    samples: int,
    npes: int = 8,
    launches_per_step: int = MAX_LAYERS_PER_STEP,
    attn_res_blocks: int = -1,
    block_write_idx: int = -1,
    fuse_moe: bool = False,
    mtp: bool = False,
    agentic_batch_size: int = 0,
    state_dtype: torch.dtype = torch.float32,
    conv_state_layout: ConvStateLayout = ConvStateLayout.CHANNEL_MAJOR,
    atom_expert_layout: bool = False,
    mla: bool = False,
    dense_ffn: bool = False,
):
    """Build the fixed-shape single-launch Kimi-K3 decode MonoKernel."""

    if dense_ffn and (mla or not fuse_moe):
        raise ValueError("dense FFN requires the fused KDA layer path")
    if mla:
        if (
            samples not in (8, 16, 32)
            or agentic_batch_size
            or mtp
            or state_dtype is not torch.float32
        ):
            raise ValueError(
                "Kimi dense MLA MonoKernel requires rows in {8,16,32} "
                "without KDA recurrent state"
            )
    elif agentic_batch_size:
        if (
            agentic_batch_size not in (1, 2, 4, 8)
            or not mtp
            or samples != agentic_batch_size * 8
            or state_dtype is not torch.float16
            or conv_state_layout is not ConvStateLayout.TIME_MAJOR
        ):
            raise ValueError(
                "Kimi Agentic MonoKernel requires B in {1,2,4,8}, "
                "q=8, MTP, FP16 recurrent state, and time-major conv state"
            )
    elif samples not in {1, 2, 4, 8}:
        raise ValueError(
            f"samples must be one of {{1, 2, 4, 8}}, got {samples}"
        )
    elif state_dtype is not torch.float32:
        raise ValueError("legacy Kimi MonoKernel state must use FP32")
    if npes != 8:
        raise ValueError(f"Kimi-K3 MonoKernel requires TP8, got TP{npes}")
    if not isinstance(conv_state_layout, ConvStateLayout):
        raise TypeError(f"conv_state_layout must be ConvStateLayout, got {conv_state_layout!r}")
    if not 1 <= launches_per_step <= MAX_LAYERS_PER_STEP:
        raise ValueError(f"launches_per_step must be in [1, {MAX_LAYERS_PER_STEP}], got {launches_per_step}")
    fuse_attn_res = attn_res_blocks >= 0
    if fuse_moe and not fuse_attn_res:
        raise ValueError("the KDA + MoE MonoKernel requires fused AttnRes")
    if attn_res_blocks < -1:
        raise ValueError(f"attn_res_blocks must be >= -1, got {attn_res_blocks}")
    if block_write_idx >= 0 and block_write_idx != attn_res_blocks:
        raise ValueError("the pre-attention block write must append at attn_res_blocks")
    state_fp16 = state_dtype is torch.float16
    state_slot_bytes = (
        _HEADS * _HEAD_DIM * _HEAD_DIM * (2 if state_fp16 else 4)
    )
    conv_state_length = 10 if agentic_batch_size else _CONV_STATE_LENGTH
    conv_slot_bytes = _CONV_CHANNELS * conv_state_length * 2
    latent_projection_waves = 3 if samples <= 4 else 4
    shared_projection_waves = 3 if samples <= 4 else 6
    ffn_inter = _DENSE_INTER if dense_ffn else _SHARED_INTER
    mtp_splits = 4 if mtp else _MTP_SPLITS
    mtp_rows_per_split = _HEAD_DIM // mtp_splits
    mtp_v_lanes = mtp_rows_per_split // _WAVES
    mtp_k_lanes = _WAVE_SIZE // mtp_v_lanes
    mtp_k_tile = mtp_k_lanes * _VALUES_PER_THREAD
    mtp_k_iters = _HEAD_DIM // mtp_k_tile
    staged_samples = min(samples, 4)
    sample_groups = (samples + staged_samples - 1) // staged_samples
    output_staged_samples = min(staged_samples, 2) if mtp and samples <= 4 else staged_samples
    output_sample_groups = (samples + output_staged_samples - 1) // output_staged_samples
    input_row_groups = 2 if samples <= 4 else _INPUT_ROW_GROUPS
    input_split_waves = _WAVES // input_row_groups
    input_row_tile = input_row_groups * 16
    input_row_tasks = _FUSED_PAD // input_row_tile
    up_scale_groups = (
        (_ROUTED_HIDDEN // 32 + 7) // 8 * 8
        if atom_expert_layout
        else _ROUTED_HIDDEN // 32
    )
    down_scale_groups = (
        (_INTER // 32 + 7) // 8 * 8
        if atom_expert_layout
        else _INTER // 32
    )
    up_expert_scale_bytes = 2 * _INTER * up_scale_groups
    down_expert_scale_bytes = _ROUTED_HIDDEN * down_scale_groups

    pre_mailbox_offset = 0
    pre_ready_offset = samples * _HIDDEN * 2 if fuse_attn_res else 0
    moe_mailbox_offset = pre_ready_offset + samples * _ATTN_RES_CTAS * 8 if fuse_attn_res else 0
    moe_ready_offset = moe_mailbox_offset + samples * _HIDDEN * 2 if fuse_attn_res else 0
    input_mailbox_offset = moe_ready_offset + samples * _ATTN_RES_CTAS * 8 if fuse_attn_res else 0
    norm_mailbox_offset = input_mailbox_offset + samples * _FUSED_PAD * 4
    norm_packed_offset = norm_mailbox_offset + samples * _PROJECTION * (
        4 if (mtp or mla) else 2
    )
    norm_ready_offset = norm_packed_offset + (samples * _PROJECTION * 2 if mtp else 0)
    attention_mailbox_offset = norm_ready_offset + samples * _HEADS * 8
    pre_stats_offset = attention_mailbox_offset + samples * _HIDDEN * 4
    post_stats_offset = pre_stats_offset + samples * _ATTN_RES_CTAS * _ATTN_RES_STATS * 8
    moe_base = post_stats_offset + samples * _ATTN_RES_CTAS * _ATTN_RES_STATS * 8
    router_offset = moe_base
    router_ready_offset = router_offset + samples * _N_EXPERTS * 4
    latent_offset = router_ready_offset + samples * (_N_EXPERTS // 16) * 8
    latent_ready_offset = latent_offset + samples * _ROUTED_HIDDEN * 2
    shared_gu_offset = latent_ready_offset + samples * (_ROUTED_HIDDEN // 16) * 8
    shared_gu_ready_offset = shared_gu_offset + samples * (2 * ffn_inter) * 2
    shared_mid_offset = shared_gu_ready_offset + samples * ((2 * ffn_inter) // 16) * 8
    selection_id_offset = shared_mid_offset + samples * ffn_inter * 4
    selection_weight_offset = selection_id_offset + samples * _TOP_K * 8
    expert_mid_offset = selection_weight_offset + samples * _TOP_K * 8
    expert_mid_ready_offset = expert_mid_offset + samples * _TOP_K * _INTER * 2
    routed_offset = expert_mid_ready_offset + samples * _TOP_K * (_INTER // 16) * 8
    routed_stats_offset = routed_offset + samples * _ROUTED_HIDDEN * 2
    routed_inv_offset = routed_stats_offset + samples * (_ROUTED_HIDDEN // 16) * 8
    layout = monokernel_layout(
        samples,
        fuse_attn_res=fuse_attn_res,
        fuse_moe=fuse_moe,
        mtp=mtp,
        mla=mla,
        dense_ffn=dense_ffn,
    )
    mtp_qkvg_offset = layout.get("mtp_qkvg", 0)
    mtp_conv_ready_offset = layout.get("mtp_conv_ready", 0)
    mtp_state_ready_offset = layout.get("mtp_state_ready", 0)
    mtp_state_handoff_offset = layout.get("mtp_state_handoff", 0)
    mtp_norm_ready_offset = layout.get("mtp_norm_ready", 0)
    mla_qkv_offset = layout.get("mla_qkv", 0)
    mla_qnorm_offset = layout.get("mla_qnorm", 0)
    mla_q_offset = layout.get("mla_q", 0)
    mla_fresh_offset = layout.get("mla_fresh", 0)
    mla_qlat_offset = layout.get("mla_qlat", 0)
    mla_dense_acc_offset = layout.get("mla_dense_acc", 0)
    mla_gate_offset = layout.get("mla_gate", 0)
    max_pairs = samples * _HIDDEN // 2
    slot_bytes = npes * max_pairs * 8

    # The S=8 kernel benefits from the TileRT/GLM-style phase overlay below.
    # At S<=4, retaining dedicated stage-local views preserves the faster LDS
    # bank/address placement even though it uses slightly more shared memory.
    if samples <= 4:

        @fx.struct
        class SharedStorage:
            x: fx.Array[fx.Float32, staged_samples * _HIDDEN // 2, 16]
            reduction: fx.Array[fx.Float32, _WAVES * _WAVE_SIZE * 4, 16]
            output: fx.Array[fx.Float32, staged_samples * _OUTPUT_ROW_TILE // 2, 16]
            query: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
            key: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
            value: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
            gate: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
            f_a: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
            norm_sums: fx.Array[fx.Float32, 2 * _WAVES, 16]
            attn_values: fx.Array[fx.Float32, 16, 16]

    else:

        @fx.struct
        class SharedStorage:
            x: fx.Array[fx.Float32, staged_samples * _HIDDEN // 2, 16]
            reduction: fx.Array[fx.Float32, _WAVES * _WAVE_SIZE * 4, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def kimi_k3_monokernel(
        hidden_states: Int64,
        output: Int64,
        block_residual: Int64,
        self_res_norm: Int64,
        self_res_qk: Int64,
        input_norm: Int64,
        mlp_res_norm: Int64,
        mlp_res_qk: Int64,
        post_norm: Int64,
        pre_updated: Int64,
        pre_output: Int64,
        updated_prefix: Int64,
        moe_input: Int64,
        quantized_moe_input: Int64,
        quantized_moe_scale: Int64,
        block_stride: Int32,
        packed_router_weight: Int64,
        correction_bias: Int64,
        packed_latent_weight: Int64,
        latent_weight_scale: Int64,
        packed_shared_up: Int64,
        shared_up_scale: Int64,
        packed_expert_up: Int64,
        expert_up_scale: Int64,
        packed_expert_down: Int64,
        expert_down_scale: Int64,
        latent_gain: Int64,
        packed_shared_down: Int64,
        shared_down_scale: Int64,
        packed_latent_up: Int64,
        latent_up_scale: Int64,
        moe_symmetric: Int64,
        moe_peers: Int64,
        final_output: Int64,
        packed_input_weight: Int64,
        gate_weight: Int64,
        conv_weight: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        packed_output_weight: Int64,
        state_indices: Int64,
        num_accepted_tokens: Int64,
        conv_state: Int64,
        recurrent_state: Int64,
        scratch: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        timeline: Int64,
        mla_positions: Int64,
        mla_slot_mapping: Int64,
        mla_batch_ids: Int64,
        mla_context_lens: Int64,
        mla_block_tables: Int64,
        mla_block_table_stride: Int32,
        mla_block_size: Int32,
        mla_block_ratio: Int32,
        mla_cache: Int64,
        mla_cache_scale: Int64,
        mla_qkv_weight: Int64,
        mla_q_norm: Int64,
        mla_kv_norm: Int64,
        mla_q_b_weight: Int64,
        mla_uk_weight: Int64,
        mla_uv_weight: Int64,
        mla_gate_weight: Int64,
        rank: Int32,
        layer: Int32,
        advance_epoch: Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        wave = tid // _WAVE_SIZE
        lane = tid % _WAVE_SIZE
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        x = storage.x.ptr
        reduction = storage.reduction.ptr
        if const_expr(samples <= 4):
            output_values = storage.output.ptr
            shared_query = storage.query.ptr
            shared_key = storage.key.ptr
            shared_value = storage.value.ptr
            shared_gate = storage.gate.ptr
            shared_f_a = storage.f_a.ptr
            norm_sums = storage.norm_sums.ptr
            attn_values = storage.attn_values.ptr
        else:
            # TileRT/GLM-style phase overlay: these small stage-local views reuse
            # the activation arena, while the MFMA reduction arena stays disjoint
            # because its consumers can still have activation loads in flight.
            output_values = x
            shared_bf16 = fx.recast_iter(fx.BFloat16, x)
            shared_query = shared_bf16
            shared_key = shared_bf16 + _HEAD_DIM
            shared_value = shared_bf16 + 2 * _HEAD_DIM
            shared_gate = shared_bf16 + 3 * _HEAD_DIM
            shared_f_a = shared_bf16 + 4 * _HEAD_DIM
            norm_sums = x + (5 * _HEAD_DIM * 2) // 4
            attn_values = norm_sums + 2 * _WAVES

        hidden_rsrc = rsrc(hidden_states)
        blocks_rsrc = rsrc(block_residual)
        input_weight_rsrc = rsrc(packed_input_weight)
        gate_weight_rsrc = rsrc(gate_weight)
        conv_weight_rsrc = rsrc(conv_weight)
        a_log_rsrc = rsrc(a_log)
        dt_bias_rsrc = rsrc(dt_bias)
        norm_weight_rsrc = rsrc(norm_weight)
        output_weight_rsrc = rsrc(packed_output_weight)
        indices_rsrc = rsrc(state_indices)
        accepted_rsrc = rsrc(num_accepted_tokens)
        output_rsrc = rsrc(output)
        input_mailbox_rsrc = rsrc(scratch + fx.Int64(input_mailbox_offset))
        norm_mailbox_rsrc = rsrc(scratch + fx.Int64(norm_mailbox_offset))
        norm_packed_rsrc = rsrc(scratch + fx.Int64(norm_packed_offset))
        norm_ready_rsrc = rsrc(scratch + fx.Int64(norm_ready_offset))
        mtp_qkvg_rsrc = rsrc(scratch + fx.Int64(mtp_qkvg_offset))
        mtp_conv_ready_rsrc = rsrc(scratch + fx.Int64(mtp_conv_ready_offset))
        mtp_state_ready_rsrc = rsrc(scratch + fx.Int64(mtp_state_ready_offset))
        mtp_state_handoff_rsrc = rsrc(
            scratch + fx.Int64(mtp_state_handoff_offset)
        )
        mtp_norm_ready_rsrc = rsrc(scratch + fx.Int64(mtp_norm_ready_offset))
        quantized_moe_rsrc = rsrc(quantized_moe_input)
        quantized_moe_scale_rsrc = rsrc(quantized_moe_scale)
        pre_mailbox_rsrc = rsrc(scratch + fx.Int64(pre_mailbox_offset))
        pre_ready_rsrc = rsrc(scratch + fx.Int64(pre_ready_offset))
        attention_mailbox_rsrc = rsrc(scratch + fx.Int64(attention_mailbox_offset))
        moe_mailbox_rsrc = rsrc(scratch + fx.Int64(moe_mailbox_offset))
        moe_ready_rsrc = rsrc(scratch + fx.Int64(moe_ready_offset))
        pre_stats_rsrc = rsrc(scratch + fx.Int64(pre_stats_offset))
        post_stats_rsrc = rsrc(scratch + fx.Int64(post_stats_offset))
        router_mailbox_rsrc = rsrc(scratch + fx.Int64(router_offset))
        router_ready_rsrc = rsrc(scratch + fx.Int64(router_ready_offset))
        latent_mailbox_rsrc = rsrc(scratch + fx.Int64(latent_offset))
        latent_ready_rsrc = rsrc(scratch + fx.Int64(latent_ready_offset))
        shared_gu_mailbox_rsrc = rsrc(scratch + fx.Int64(shared_gu_offset))
        shared_gu_ready_rsrc = rsrc(scratch + fx.Int64(shared_gu_ready_offset))
        shared_mid_mailbox_rsrc = rsrc(scratch + fx.Int64(shared_mid_offset))
        selection_id_rsrc = rsrc(scratch + fx.Int64(selection_id_offset))
        selection_weight_rsrc = rsrc(scratch + fx.Int64(selection_weight_offset))
        expert_mid_mailbox_rsrc = rsrc(scratch + fx.Int64(expert_mid_offset))
        expert_mid_ready_rsrc = rsrc(scratch + fx.Int64(expert_mid_ready_offset))
        routed_mailbox_rsrc = rsrc(scratch + fx.Int64(routed_offset))
        routed_stats_rsrc = rsrc(scratch + fx.Int64(routed_stats_offset))
        routed_inv_rsrc = rsrc(scratch + fx.Int64(routed_inv_offset))
        mla_qkv_rsrc = rsrc(scratch + fx.Int64(mla_qkv_offset))
        mla_qnorm_rsrc = rsrc(scratch + fx.Int64(mla_qnorm_offset))
        mla_q_rsrc = rsrc(scratch + fx.Int64(mla_q_offset))
        mla_fresh_rsrc = rsrc(scratch + fx.Int64(mla_fresh_offset))
        mla_qlat_rsrc = rsrc(scratch + fx.Int64(mla_qlat_offset))
        mla_dense_acc_rsrc = rsrc(scratch + fx.Int64(mla_dense_acc_offset))
        mla_gate_rsrc = rsrc(scratch + fx.Int64(mla_gate_offset))

        step_value = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        launch_epoch = symmetric_mailbox_epoch(
            step_value,
            launches_per_step=launches_per_step,
            layer=layer,
        )
        tag = launch_epoch + 1
        slot = launch_epoch & 1
        symmetric_base = fx.Int64(slot) * fx.Int64(slot_bytes)

        def state_slots(sample):
            if const_expr(agentic_batch_size > 0):
                request = sample // 8
                token = sample % 8
                accepted = uniform(
                    bo.buffer_load(
                        accepted_rsrc,
                        request,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                input_column = (token == 0).select(
                    fx.max(fx.Int32(0), fx.min(accepted - 1, fx.Int32(7))),
                    token - 1,
                )
                input_slot = uniform(
                    bo.buffer_load(
                        indices_rsrc,
                        request * 8 + input_column,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                output_slot = uniform(
                    bo.buffer_load(
                        indices_rsrc,
                        request * 8 + token,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                acceptance_valid = (accepted >= 1) & (accepted <= 8)
                return input_slot, output_slot, acceptance_valid
            input_slot = uniform(
                bo.buffer_load(
                    indices_rsrc,
                    sample,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            output_slot = input_slot
            if const_expr(mtp):
                output_slot = uniform(
                    bo.buffer_load(
                        indices_rsrc,
                        sample + 1,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            return input_slot, output_slot, input_slot == input_slot

        def stamp(index):
            if (bid == 0) & (tid == 0):
                now = fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                bo.buffer_store(now, rsrc(timeline), index, cache_modifier=CM_DEV)

        stamp(0)

        def bf16_pair(a, b):
            return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

        def bf16_round(value):
            return fx.Float32(fx.Float32(value).to(fx.BFloat16))

        def lds_load(pointer, index):
            return fx.ptr_load(pointer + index)

        def lds_store(pointer, index, value):
            fx.ptr_store(value, pointer + index)

        def wave_sum(value):
            for offset in (32, 16, 8, 4, 2, 1):
                value = xred(value, offset, lambda lhs, rhs: lhs + rhs)
            return value

        def block_sum(value):
            value = wave_sum(value)
            if lane == 0:
                lds_store(norm_sums, wave, value)
            gpu.barrier()
            total = lds_load(norm_sums, 0)
            for source_wave in range_constexpr(1, _WAVES):
                total = total + lds_load(norm_sums, source_wave)
            gpu.barrier()
            return total

        def block_sums(lhs, rhs):
            lhs_wave = wave_sum(lhs)
            rhs_wave = wave_sum(rhs)
            if lane == 0:
                lds_store(norm_sums, wave, lhs_wave)
                lds_store(norm_sums, _WAVES + wave, rhs_wave)
            gpu.barrier()
            lhs_total = lds_load(norm_sums, 0)
            rhs_total = lds_load(norm_sums, _WAVES)
            for source_wave in range_constexpr(1, _WAVES):
                lhs_total = lhs_total + lds_load(norm_sums, source_wave)
                rhs_total = rhs_total + lds_load(norm_sums, _WAVES + source_wave)
            gpu.barrier()
            return lhs_total, rhs_total

        def load_pair(mailbox_rsrc, pair):
            def load_once():
                return fx.Vector(
                    bo.buffer_load(
                        mailbox_rsrc,
                        pair * 2,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            words = load_once()
            while words[1] != tag:
                rocdl.s_nop(0)
                words = load_once()
            return words[0]

        def store_pair(mailbox_rsrc, pair, value_low, value_high):
            packed = bf16_pair(value_low, value_high).bitcast(fx.Int32)
            bo.buffer_store(
                fx.Vector.from_elements([packed, tag], fx.Int32),
                mailbox_rsrc,
                pair * 2,
                cache_modifier=CM_DEV,
            )

        def load_f32(mailbox_rsrc, index):
            def load_once():
                return fx.Vector(
                    bo.buffer_load(
                        mailbox_rsrc,
                        index * 2,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            words = load_once()
            while words[1] != tag:
                rocdl.s_nop(0)
                words = load_once()
            return words[0].bitcast(fx.Float32)

        def store_f32(mailbox_rsrc, index, value):
            bo.buffer_store(
                fx.Vector.from_elements([fx.Float32(value).bitcast(fx.Int32), tag], fx.Int32),
                mailbox_rsrc,
                index * 2,
                cache_modifier=CM_DEV,
            )

        def load_i32(mailbox_rsrc, index):
            def load_once():
                return fx.Vector(
                    bo.buffer_load(
                        mailbox_rsrc,
                        index * 2,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            words = load_once()
            while words[1] != tag:
                rocdl.s_nop(0)
                words = load_once()
            return words[0]

        def store_i32(mailbox_rsrc, index, value):
            bo.buffer_store(
                fx.Vector.from_elements([fx.Int32(value), tag], fx.Int32),
                mailbox_rsrc,
                index * 2,
                cache_modifier=CM_DEV,
            )

        def load_raw_pair(mailbox_rsrc, pair):
            return fx.Int32(
                bo.buffer_load(
                    mailbox_rsrc,
                    pair,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )

        def store_raw_pair(mailbox_rsrc, pair, value_low, value_high):
            bo.buffer_store(
                bf16_pair(value_low, value_high).bitcast(fx.Int32),
                mailbox_rsrc,
                pair,
                cache_modifier=CM_DEV,
            )

        def pair_values(mailbox_rsrc, pair):
            packed = load_pair(mailbox_rsrc, pair)
            return (
                (packed << 16).bitcast(fx.Float32),
                (packed & fx.Int32(-65536)).bitcast(fx.Float32),
            )

        def tagged_bf16(mailbox_rsrc, index):
            low, high = pair_values(mailbox_rsrc, index // 2)
            return (index % 2 == 0).select(low, high)

        def load_raw_f32(mailbox_rsrc, index):
            return fx.Int32(
                bo.buffer_load(
                    mailbox_rsrc,
                    index,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            ).bitcast(fx.Float32)

        def store_raw_f32(mailbox_rsrc, index, value):
            bo.buffer_store(
                fx.Float32(value),
                mailbox_rsrc,
                index,
                cache_modifier=CM_DEV,
            )

        def put_input_pair(sample, row, value_low, value_high):
            pair = (sample * _FUSED_PAD + row) // 2
            store_pair(input_mailbox_rsrc, pair, value_low, value_high)

        def get_input(sample, row):
            packed = load_pair(input_mailbox_rsrc, (sample * _FUSED_PAD + row) // 2)
            low = (packed << 16).bitcast(fx.Float32)
            high = (packed & fx.Int32(-65536)).bitcast(fx.Float32)
            return (row & 1).select(high, low)

        def put_norm_pair(sample, row, value_low, value_high):
            pair = (sample * _PROJECTION + row) // 2
            store_raw_pair(norm_mailbox_rsrc, pair, value_low, value_high)

        def run_attn_res(
            sample,
            prefix_address,
            delta_address,
            norm_address,
            qk_address,
            output_norm_address,
            updated_address,
            output_address,
            output_mailbox_rsrc,
            num_blocks,
            has_delta,
            tagged_prefix,
            tagged_delta,
            write_block,
            quantize,
        ):
            prefix_rsrc = rsrc(prefix_address)
            delta_rsrc = rsrc(delta_address)
            norm_rsrc = rsrc(norm_address)
            qk_rsrc = rsrc(qk_address)
            output_norm_rsrc = rsrc(output_norm_address)
            updated_rsrc = rsrc(updated_address)
            result_rsrc = rsrc(output_address)
            quantized_rsrc = rsrc(quantized_moe_input)
            quantized_scale_rsrc = rsrc(quantized_moe_scale)
            pair_rounds = _HIDDEN // (2 * _THREADS)
            num_sources = num_blocks + 1

            def load_updated(pair_in_row):
                pair = sample * (_HIDDEN // 2) + pair_in_row
                if const_expr(tagged_prefix):
                    prefix_word = load_pair(attention_mailbox_rsrc, pair)
                else:
                    prefix_word = fx.Int32(
                        bo.buffer_load(
                            prefix_rsrc,
                            pair,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                prefix_low = (prefix_word << 16).bitcast(fx.Float32)
                prefix_high = (prefix_word & fx.Int32(-65536)).bitcast(fx.Float32)
                if const_expr(has_delta):
                    if const_expr(tagged_delta):
                        delta_word = load_pair(attention_mailbox_rsrc, pair)
                    else:
                        delta_word = fx.Int32(
                            bo.buffer_load(
                                delta_rsrc,
                                pair,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                    prefix_low = prefix_low + (delta_word << 16).bitcast(fx.Float32)
                    prefix_high = prefix_high + (delta_word & fx.Int32(-65536)).bitcast(fx.Float32)
                return (
                    fx.Float32(prefix_low.to(fx.BFloat16)),
                    fx.Float32(prefix_high.to(fx.BFloat16)),
                )

            updated_pairs = []
            for pair_round in range_constexpr(pair_rounds):
                pair_in_row = tid + pair_round * _THREADS
                updated_low, updated_high = load_updated(pair_in_row)
                updated_pairs.append((updated_low, updated_high))
                updated_word = bf16_pair(updated_low, updated_high).bitcast(fx.Int32)
                bo.buffer_store(
                    updated_word,
                    updated_rsrc,
                    sample * (_HIDDEN // 2) + pair_in_row,
                )
                if const_expr(write_block >= 0):
                    block_pair = (sample * block_stride + write_block) * (_HIDDEN // 2) + pair_in_row
                    bo.buffer_store(updated_word, blocks_rsrc, block_pair)

            logits = []
            for source in range_constexpr(num_sources):
                square_sum = fx.Float32(0.0)
                weighted_sum = fx.Float32(0.0)
                for pair_round in range_constexpr(pair_rounds):
                    pair_in_row = tid + pair_round * _THREADS
                    if const_expr(source < num_blocks):
                        block_pair = (sample * block_stride + source) * (_HIDDEN // 2) + pair_in_row
                        source_word = fx.Int32(bo.buffer_load(blocks_rsrc, block_pair, vec_width=1, dtype=T.i32))
                        value_low = (source_word << 16).bitcast(fx.Float32)
                        value_high = (source_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    else:
                        value_low, value_high = updated_pairs[pair_round]
                    norm_word = fx.Int32(bo.buffer_load(norm_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                    qk_word = fx.Int32(bo.buffer_load(qk_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                    norm_low = (norm_word << 16).bitcast(fx.Float32)
                    norm_high = (norm_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    qk_low = (qk_word << 16).bitcast(fx.Float32)
                    qk_high = (qk_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    square_sum = square_sum + value_low * value_low + value_high * value_high
                    weighted_sum = weighted_sum + value_low * norm_low * qk_low + value_high * norm_high * qk_high
                total_square, total_weighted = block_sums(square_sum, weighted_sum)
                logits.append(total_weighted * rsq(total_square * (1.0 / _HIDDEN) + EPS))

            max_logit = logits[0]
            for source in range_constexpr(1, num_sources):
                max_logit = fx.max(max_logit, logits[source])
            probabilities = [exp(logit - max_logit) for logit in logits]
            probability_sum = probabilities[0]
            for source in range_constexpr(1, num_sources):
                probability_sum = probability_sum + probabilities[source]
            inverse_probability_sum = rcp(probability_sum)
            probabilities = [probability * inverse_probability_sum for probability in probabilities]

            mixed_pairs = []
            mixed_square_sum = fx.Float32(0.0)
            for pair_round in range_constexpr(pair_rounds):
                pair_in_row = tid + pair_round * _THREADS
                mixed_low = fx.Float32(0.0)
                mixed_high = fx.Float32(0.0)
                for source in range_constexpr(num_sources):
                    if const_expr(source < num_blocks):
                        block_pair = (sample * block_stride + source) * (_HIDDEN // 2) + pair_in_row
                        source_word = fx.Int32(bo.buffer_load(blocks_rsrc, block_pair, vec_width=1, dtype=T.i32))
                        value_low = (source_word << 16).bitcast(fx.Float32)
                        value_high = (source_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    else:
                        value_low, value_high = updated_pairs[pair_round]
                    mixed_low = mixed_low + probabilities[source] * value_low
                    mixed_high = mixed_high + probabilities[source] * value_high
                mixed_pairs.append((mixed_low, mixed_high))
                mixed_square_sum = mixed_square_sum + mixed_low * mixed_low + mixed_high * mixed_high

            total_mixed_square = block_sum(mixed_square_sum)
            output_inverse_rms = rsq(total_mixed_square * (1.0 / _HIDDEN) + EPS)
            for pair_round in range_constexpr(pair_rounds):
                pair_in_row = tid + pair_round * _THREADS
                output_weight_word = fx.Int32(bo.buffer_load(output_norm_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                weight_low = (output_weight_word << 16).bitcast(fx.Float32)
                weight_high = (output_weight_word & fx.Int32(-65536)).bitcast(fx.Float32)
                mixed_low, mixed_high = mixed_pairs[pair_round]
                value_low = mixed_low * output_inverse_rms * weight_low
                value_high = mixed_high * output_inverse_rms * weight_high
                packed_output = bf16_pair(value_low, value_high).bitcast(fx.Int32)
                pair = sample * (_HIDDEN // 2) + pair_in_row
                bo.buffer_store(packed_output, result_rsrc, pair)
                store_pair(output_mailbox_rsrc, pair, value_low, value_high)

                if const_expr(quantize):
                    rounded = (
                        fx.Vector.from_elements([value_low, value_high], fx.Float32).to(fx.BFloat16).to(fx.Float32)
                    )
                    absolute_max = fx.max(
                        fx.max(rounded[0], -rounded[0]),
                        fx.max(rounded[1], -rounded[1]),
                    )
                    for offset in (8, 4, 2, 1):
                        absolute_max = xred(absolute_max, offset, fx.max)
                    raw_scale = absolute_max * fx.Float32(1.0 / FP8_MAX)
                    bits = raw_scale.bitcast(fx.Int32)
                    exponent = bits.shrui(fx.Int32(23)) & fx.Int32(0xFF)
                    round_up = ((bits & fx.Int32(0x400000)) != 0) & (
                        ((bits & fx.Int32(0x200000)) != 0) | ((bits & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
                    )
                    exponent = exponent + round_up.select(fx.Int32(1), fx.Int32(0))
                    nonzero = absolute_max > fx.Float32(0.0)
                    scale = nonzero.select(
                        (exponent << fx.Int32(23)).bitcast(fx.Float32),
                        fx.Float32(1.0),
                    )
                    inverse = nonzero.select(rcp(scale), fx.Float32(1.0))
                    q_low = fx.min(fx.max(rounded[0] * inverse, -FP8_MAX), FP8_MAX)
                    q_high = fx.min(fx.max(rounded[1] * inverse, -FP8_MAX), FP8_MAX)
                    packed_fp8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q_low, q_high, fx.Int32(0), False)) & fx.Int32(
                        0xFFFF
                    )
                    neighbor = xshfl(packed_fp8, 1)
                    if lane % 2 == 0:
                        bo.buffer_store(
                            packed_fp8 | (neighbor << fx.Int32(16)),
                            quantized_rsrc,
                            sample * (_HIDDEN // 4) + pair_in_row // 2,
                            cache_modifier=CM_DEV,
                        )
                    if lane % 16 == 0:
                        scale_column = pair_in_row // 16
                        scale_offset = (
                            (scale_column // 8) * 256
                            + (scale_column % 4) * 64
                            + sample * 4
                            + ((scale_column // 4) % 2) * 2
                        )
                        bo.buffer_store(
                            exponent.to(fx.Uint8),
                            quantized_scale_rsrc,
                            scale_offset,
                            cache_modifier=CM_DEV,
                            offset_is_bytes=True,
                        )

        def run_attn_res_chunk(
            sample,
            chunk,
            prefix_address,
            delta_address,
            norm_address,
            qk_address,
            output_norm_address,
            updated_address,
            output_address,
            output_mailbox_rsrc,
            output_ready_rsrc,
            stats_rsrc,
            num_blocks,
            has_delta,
            tagged_prefix,
            tagged_delta,
            write_block,
            quantize,
        ):
            prefix_rsrc = rsrc(prefix_address)
            delta_rsrc = rsrc(delta_address)
            norm_rsrc = rsrc(norm_address)
            qk_rsrc = rsrc(qk_address)
            output_norm_rsrc = rsrc(output_norm_address)
            updated_rsrc = rsrc(updated_address)
            result_rsrc = rsrc(output_address)
            quantized_rsrc = rsrc(quantized_moe_input)
            quantized_scale_rsrc = rsrc(quantized_moe_scale)
            pairs_per_chunk = (_HIDDEN // 2) // _ATTN_RES_CTAS
            pair_rounds = (pairs_per_chunk + _THREADS - 1) // _THREADS
            pair_begin = chunk * pairs_per_chunk
            num_sources = num_blocks + 1
            stats_base = (sample * _ATTN_RES_CTAS + chunk) * _ATTN_RES_STATS

            def load_updated(pair_in_row):
                pair = sample * (_HIDDEN // 2) + pair_in_row
                if const_expr(tagged_prefix):
                    prefix_word = load_pair(attention_mailbox_rsrc, pair)
                else:
                    prefix_word = fx.Int32(bo.buffer_load(prefix_rsrc, pair, vec_width=1, dtype=T.i32))
                prefix_low = (prefix_word << 16).bitcast(fx.Float32)
                prefix_high = (prefix_word & fx.Int32(-65536)).bitcast(fx.Float32)
                if const_expr(has_delta):
                    if const_expr(tagged_delta):
                        delta_word = load_pair(attention_mailbox_rsrc, pair)
                    else:
                        delta_word = fx.Int32(bo.buffer_load(delta_rsrc, pair, vec_width=1, dtype=T.i32))
                    prefix_low = prefix_low + (delta_word << 16).bitcast(fx.Float32)
                    prefix_high = prefix_high + (delta_word & fx.Int32(-65536)).bitcast(fx.Float32)
                return (
                    fx.Float32(prefix_low.to(fx.BFloat16)),
                    fx.Float32(prefix_high.to(fx.BFloat16)),
                )

            updated_pairs = []
            for pair_round in range_constexpr(pair_rounds):
                local_pair = tid + pair_round * _THREADS
                valid = local_pair < pairs_per_chunk
                pair_in_row = fx.min(pair_begin + local_pair, _HIDDEN // 2 - 1)
                updated_low, updated_high = load_updated(pair_in_row)
                updated_pairs.append((updated_low, updated_high))
                if valid:
                    updated_word = bf16_pair(updated_low, updated_high).bitcast(fx.Int32)
                    pair = sample * (_HIDDEN // 2) + pair_in_row
                    bo.buffer_store(updated_word, updated_rsrc, pair)
                    if const_expr(write_block >= 0):
                        block_pair = (sample * block_stride + write_block) * (_HIDDEN // 2) + pair_in_row
                        bo.buffer_store(updated_word, blocks_rsrc, block_pair)

            for source in range_constexpr(num_sources):
                square_sum = fx.Float32(0.0)
                weighted_sum = fx.Float32(0.0)
                for pair_round in range_constexpr(pair_rounds):
                    local_pair = tid + pair_round * _THREADS
                    valid = local_pair < pairs_per_chunk
                    pair_in_row = fx.min(pair_begin + local_pair, _HIDDEN // 2 - 1)
                    if const_expr(source < num_blocks):
                        block_pair = (sample * block_stride + source) * (_HIDDEN // 2) + pair_in_row
                        source_word = fx.Int32(bo.buffer_load(blocks_rsrc, block_pair, vec_width=1, dtype=T.i32))
                        value_low = (source_word << 16).bitcast(fx.Float32)
                        value_high = (source_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    else:
                        value_low, value_high = updated_pairs[pair_round]
                    norm_word = fx.Int32(bo.buffer_load(norm_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                    qk_word = fx.Int32(bo.buffer_load(qk_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                    norm_low = (norm_word << 16).bitcast(fx.Float32)
                    norm_high = (norm_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    qk_low = (qk_word << 16).bitcast(fx.Float32)
                    qk_high = (qk_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    square_sum = square_sum + valid.select(
                        value_low * value_low + value_high * value_high,
                        fx.Float32(0.0),
                    )
                    weighted_sum = weighted_sum + valid.select(
                        value_low * norm_low * qk_low + value_high * norm_high * qk_high,
                        fx.Float32(0.0),
                    )
                total_square, total_weighted = block_sums(square_sum, weighted_sum)
                if tid == 0:
                    store_f32(stats_rsrc, stats_base + source * 2, total_square)
                    store_f32(stats_rsrc, stats_base + source * 2 + 1, total_weighted)

            if tid == 0:
                logits = []
                for source in range_constexpr(num_sources):
                    total_square = fx.Float32(0.0)
                    total_weighted = fx.Float32(0.0)
                    for source_chunk in range_constexpr(_ATTN_RES_CTAS):
                        source_base = (sample * _ATTN_RES_CTAS + source_chunk) * _ATTN_RES_STATS
                        total_square = total_square + load_f32(stats_rsrc, source_base + source * 2)
                        total_weighted = total_weighted + load_f32(stats_rsrc, source_base + source * 2 + 1)
                    logits.append(total_weighted * rsq(total_square * (1.0 / _HIDDEN) + EPS))
                max_logit = logits[0]
                for source in range_constexpr(1, num_sources):
                    max_logit = fx.max(max_logit, logits[source])
                probabilities = [exp(logit - max_logit) for logit in logits]
                probability_sum = probabilities[0]
                for source in range_constexpr(1, num_sources):
                    probability_sum = probability_sum + probabilities[source]
                inverse_probability_sum = rcp(probability_sum)
                for source in range_constexpr(num_sources):
                    lds_store(
                        attn_values,
                        source,
                        probabilities[source] * inverse_probability_sum,
                    )
            gpu.barrier()

            mixed_pairs = []
            mixed_square_sum = fx.Float32(0.0)
            for pair_round in range_constexpr(pair_rounds):
                local_pair = tid + pair_round * _THREADS
                valid = local_pair < pairs_per_chunk
                pair_in_row = fx.min(pair_begin + local_pair, _HIDDEN // 2 - 1)
                mixed_low = fx.Float32(0.0)
                mixed_high = fx.Float32(0.0)
                for source in range_constexpr(num_sources):
                    if const_expr(source < num_blocks):
                        block_pair = (sample * block_stride + source) * (_HIDDEN // 2) + pair_in_row
                        source_word = fx.Int32(bo.buffer_load(blocks_rsrc, block_pair, vec_width=1, dtype=T.i32))
                        value_low = (source_word << 16).bitcast(fx.Float32)
                        value_high = (source_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    else:
                        value_low, value_high = updated_pairs[pair_round]
                    probability = lds_load(attn_values, source)
                    mixed_low = mixed_low + probability * value_low
                    mixed_high = mixed_high + probability * value_high
                mixed_pairs.append((mixed_low, mixed_high))
                mixed_square_sum = mixed_square_sum + valid.select(
                    mixed_low * mixed_low + mixed_high * mixed_high,
                    fx.Float32(0.0),
                )

            total_mixed_square = block_sum(mixed_square_sum)
            if tid == 0:
                store_f32(stats_rsrc, stats_base + 2 * num_sources, total_mixed_square)
                full_square = fx.Float32(0.0)
                for source_chunk in range_constexpr(_ATTN_RES_CTAS):
                    source_base = (sample * _ATTN_RES_CTAS + source_chunk) * _ATTN_RES_STATS
                    full_square = full_square + load_f32(stats_rsrc, source_base + 2 * num_sources)
                lds_store(attn_values, num_sources, rsq(full_square * (1.0 / _HIDDEN) + EPS))
            gpu.barrier()
            output_inverse_rms = lds_load(attn_values, num_sources)

            for pair_round in range_constexpr(pair_rounds):
                local_pair = tid + pair_round * _THREADS
                if local_pair < pairs_per_chunk:
                    pair_in_row = pair_begin + local_pair
                    output_weight_word = fx.Int32(
                        bo.buffer_load(output_norm_rsrc, pair_in_row, vec_width=1, dtype=T.i32)
                    )
                    weight_low = (output_weight_word << 16).bitcast(fx.Float32)
                    weight_high = (output_weight_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    mixed_low, mixed_high = mixed_pairs[pair_round]
                    value_low = mixed_low * output_inverse_rms * weight_low
                    value_high = mixed_high * output_inverse_rms * weight_high
                    pair = sample * (_HIDDEN // 2) + pair_in_row
                    bo.buffer_store(
                        bf16_pair(value_low, value_high).bitcast(fx.Int32),
                        result_rsrc,
                        pair,
                    )
                    store_raw_pair(output_mailbox_rsrc, pair, value_low, value_high)

                    if const_expr(quantize):
                        rounded = (
                            fx.Vector.from_elements([value_low, value_high], fx.Float32).to(fx.BFloat16).to(fx.Float32)
                        )
                        absolute_max = fx.max(
                            fx.max(rounded[0], -rounded[0]),
                            fx.max(rounded[1], -rounded[1]),
                        )
                        for offset in (8, 4, 2, 1):
                            absolute_max = xred(absolute_max, offset, fx.max)
                        raw_scale = absolute_max * fx.Float32(1.0 / FP8_MAX)
                        bits = raw_scale.bitcast(fx.Int32)
                        exponent = bits.shrui(fx.Int32(23)) & fx.Int32(0xFF)
                        round_up = ((bits & fx.Int32(0x400000)) != 0) & (
                            ((bits & fx.Int32(0x200000)) != 0) | ((bits & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
                        )
                        exponent = exponent + round_up.select(fx.Int32(1), fx.Int32(0))
                        nonzero = absolute_max > fx.Float32(0.0)
                        scale = nonzero.select(
                            (exponent << fx.Int32(23)).bitcast(fx.Float32),
                            fx.Float32(1.0),
                        )
                        inverse = nonzero.select(rcp(scale), fx.Float32(1.0))
                        q_low = fx.min(fx.max(rounded[0] * inverse, -FP8_MAX), FP8_MAX)
                        q_high = fx.min(fx.max(rounded[1] * inverse, -FP8_MAX), FP8_MAX)
                        packed_fp8 = fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, q_low, q_high, fx.Int32(0), False)
                        ) & fx.Int32(0xFFFF)
                        neighbor = xshfl(packed_fp8, 1)
                        if lane % 2 == 0:
                            bo.buffer_store(
                                packed_fp8 | (neighbor << fx.Int32(16)),
                                quantized_rsrc,
                                sample * (_HIDDEN // 4) + pair_in_row // 2,
                                cache_modifier=CM_DEV,
                            )
                        if lane % 16 == 0:
                            scale_column = pair_in_row // 16
                            scale_offset = (
                                (scale_column // 8) * 256
                                + (scale_column % 4) * 64
                                + sample * 4
                                + ((scale_column // 4) % 2) * 2
                            )
                            bo.buffer_store(
                                exponent.to(fx.Uint8),
                                quantized_scale_rsrc,
                                scale_offset,
                                cache_modifier=CM_DEV,
                                offset_is_bytes=True,
                            )

            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                store_i32(
                    output_ready_rsrc,
                    sample * _ATTN_RES_CTAS + chunk,
                    1,
                )

        def wait_attn_res_chunks(ready_rsrc, sample_base, sample_count):
            ready_count = sample_count * _ATTN_RES_CTAS
            for ready_round in range_constexpr((ready_count + _THREADS - 1) // _THREADS):
                ready = tid + ready_round * _THREADS
                if ready < ready_count:
                    local_sample = ready // _ATTN_RES_CTAS
                    chunk = ready % _ATTN_RES_CTAS
                    load_i32(
                        ready_rsrc,
                        (sample_base + local_sample) * _ATTN_RES_CTAS + chunk,
                    )
            gpu.barrier()

        def stage_hidden(sample_base, sample_count):
            if const_expr(fuse_attn_res):
                wait_attn_res_chunks(pre_ready_rsrc, sample_base, sample_count)
            pairs = sample_count * _HIDDEN // 2
            for load_round in range_constexpr((pairs + _THREADS - 1) // _THREADS):
                pair = tid + load_round * _THREADS
                if pair < pairs:
                    local_sample = pair // (_HIDDEN // 2)
                    pair_in_sample = pair % (_HIDDEN // 2)
                    global_pair = (sample_base + local_sample) * (_HIDDEN // 2) + pair_in_sample
                    if const_expr(fuse_attn_res):
                        word = load_raw_pair(pre_mailbox_rsrc, global_pair)
                    else:
                        word = fx.Int32(bo.buffer_load(hidden_rsrc, global_pair, vec_width=1, dtype=T.i32))
                    lds_store(x, pair, word.bitcast(fx.Float32))

        def stage_moe_hidden(sample_base, sample_count):
            wait_attn_res_chunks(moe_ready_rsrc, sample_base, sample_count)
            pairs = sample_count * _HIDDEN // 2
            for load_round in range_constexpr((pairs + _THREADS - 1) // _THREADS):
                pair = tid + load_round * _THREADS
                if pair < pairs:
                    local_sample = pair // (_HIDDEN // 2)
                    pair_in_sample = pair % (_HIDDEN // 2)
                    global_pair = (sample_base + local_sample) * (_HIDDEN // 2) + pair_in_sample
                    word = load_raw_pair(moe_mailbox_rsrc, global_pair)
                    lds_store(x, pair, word.bitcast(fx.Float32))

        def stage_mxfp8_hidden(sample_base, sample_count):
            wait_attn_res_chunks(moe_ready_rsrc, sample_base, sample_count)
            words = sample_count * _HIDDEN // 4
            for load_round in range_constexpr((words + _THREADS - 1) // _THREADS):
                word = tid + load_round * _THREADS
                if word < words:
                    local_sample = word // (_HIDDEN // 4)
                    word_in_sample = word % (_HIDDEN // 4)
                    global_word = (sample_base + local_sample) * (_HIDDEN // 4) + word_in_sample
                    packed = fx.Int32(
                        bo.buffer_load(
                            quantized_moe_rsrc,
                            global_word,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )
                    lds_store(x, word, packed.bitcast(fx.Float32))

        def stage_norm(sample_base, sample_count):
            ready_count = sample_count * _HEADS
            if const_expr(not mla):
                for ready_round in range_constexpr((ready_count + _THREADS - 1) // _THREADS):
                    ready = tid + ready_round * _THREADS
                    if ready < ready_count:
                        if const_expr(mtp):
                            local_sample = ready // _HEADS
                            head = ready % _HEADS
                            load_i32(
                                mtp_norm_ready_rsrc,
                                (sample_base + local_sample) * _HEADS + head,
                            )
                        else:
                            local_sample = ready // _HEADS
                            head = ready % _HEADS
                            load_i32(
                                norm_ready_rsrc,
                                (sample_base + local_sample) * _HEADS + head,
                            )
            gpu.barrier()
            pairs = sample_count * _PROJECTION // 2
            for load_round in range_constexpr((pairs + _THREADS - 1) // _THREADS):
                pair = tid + load_round * _THREADS
                if pair < pairs:
                    local_sample = pair // (_PROJECTION // 2)
                    pair_in_sample = pair % (_PROJECTION // 2)
                    global_pair = (sample_base + local_sample) * (_PROJECTION // 2) + pair_in_sample
                    if const_expr(mla):
                        packed = load_pair(
                            norm_mailbox_rsrc,
                            global_pair,
                        )
                    elif const_expr(mtp):
                        packed = load_raw_pair(norm_packed_rsrc, global_pair)
                    else:
                        packed = load_raw_pair(norm_mailbox_rsrc, global_pair)
                    lds_store(x, pair, packed.bitcast(fx.Float32))

        def bf16_mfma(
            weight_rsrc,
            first_row_group,
            k_size,
            row_groups,
            split_waves,
            batch_size,
            sample_count,
        ):
            chunks = k_size // 64
            chunks_per_wave = chunks // split_waves
            accumulator = [fx.Float32(0.0) for _ in range(4)]
            sample = fx.min(lane % 16, sample_count - 1)
            if wave < row_groups * split_waves:
                row_group = first_row_group + wave // split_waves
                split = wave % split_waves

                def load_unit(local_chunk):
                    chunk = split * chunks_per_wave + local_chunk
                    return [
                        fx.Vector(
                            bo.buffer_load(
                                weight_rsrc,
                                (((row_group * chunks + chunk) * 2 + step_index) * _WAVE_SIZE + lane) * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for step_index in range_constexpr(2)
                    ]

                starts = list(range(0, chunks_per_wave, batch_size))
                current = [load_unit(chunk) for chunk in range(0, min(batch_size, chunks_per_wave))]
                for batch_index in range_constexpr(len(starts)):
                    following = None
                    if const_expr(batch_index + 1 < len(starts)):
                        next_start = starts[batch_index + 1]
                        following = [
                            load_unit(chunk)
                            for chunk in range(next_start, min(next_start + batch_size, chunks_per_wave))
                        ]
                    for unit_index in range_constexpr(len(current)):
                        local_chunk = starts[batch_index] + unit_index
                        chunk = split * chunks_per_wave + local_chunk
                        weights = current[unit_index]
                        for step_index in range_constexpr(2):
                            lhs = weights[step_index].bitcast(fx.BFloat16)
                            rhs = fx.ptr_load(
                                x + (sample * k_size + chunk * 64) // 2 + (lane // 16) * 4 + step_index * 16,
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            ).bitcast(fx.BFloat16)
                            accumulator = list(
                                fx.Vector(
                                    rocdl.mfma_f32_16x16x32_bf16(
                                        T.vec(4, T.f32),
                                        [lhs, rhs, fx.Vector.from_elements(accumulator, fx.Float32)],
                                    )
                                )
                            )
                    current = following
            return accumulator

        mxfp8_scale_atoms = [
            fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    fx.Float8E4M3FN,
                    opsel_a=opsel,
                    opsel_b=opsel,
                )
            )
            for opsel in (0, 2)
        ]

        def mxfp8_scaled_mfma(
            weight_rsrc,
            scale_rsrc,
            row_tile,
            sample_base,
            sample_count,
        ):
            k_chunks = _HIDDEN // 64
            k_scale_chunks = _HIDDEN // 256
            lane_div16 = lane // 16
            lane_mod16 = lane % 16
            accumulator = fx.make_rmem_tensor(4, fx.Float32)
            accumulator.store(fx.Vector.filled(4, 0.0, fx.Float32))
            valid_sample = lane_mod16 < sample_count
            sample = fx.min(lane_mod16, sample_count - 1)
            scale_lane = lane_div16 * 16 + lane_mod16
            for k256 in range_constexpr(k_scale_chunks):
                activation_scale = fx.Int32(
                    bo.buffer_load(
                        quantized_moe_scale_rsrc,
                        k256 * 64 + lane_div16 * 16 + sample_base + sample,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                ) & fx.Int32(0x00FF00FF)
                weight_scale = fx.Int32(
                    bo.buffer_load(
                        scale_rsrc,
                        ((row_tile // 2) * k_scale_chunks + k256) * 64 + scale_lane,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                weight_scale = ((row_tile % 2) != 0).select(
                    weight_scale.shrui(fx.Int32(8)),
                    weight_scale,
                )
                for k128_half in range_constexpr(2):
                    k128 = k256 * 2 + k128_half
                    k_base = k128 * 128 + lane_div16 * 16
                    activation_halves = []
                    for k64_half in range_constexpr(2):
                        loaded = fx.Vector(
                            fx.ptr_load(
                                x + (sample * _HIDDEN + k_base + k64_half * 64) // 4,
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            )
                        ).bitcast(fx.Int32)
                        activation_halves.append(
                            fx.Vector.from_elements(
                                [valid_sample.select(loaded[index], fx.Int32(0)) for index in range(4)],
                                fx.Int32,
                            )
                        )
                    activation_fragment = fx.make_rmem_tensor(8, fx.Int32)
                    activation_fragment.store(activation_halves[0].shuffle(activation_halves[1], list(range(8))))
                    weight_halves = []
                    for k64_half in range_constexpr(2):
                        k64 = k128 * 2 + k64_half
                        weight_halves.append(
                            fx.Vector(
                                bo.buffer_load(
                                    weight_rsrc,
                                    (((row_tile * k_chunks + k64) * 4 + lane_div16) * 16 + lane_mod16) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                        )
                    weight_fragment = fx.make_rmem_tensor(8, fx.Int32)
                    weight_fragment.store(weight_halves[0].shuffle(weight_halves[1], list(range(8))))
                    fx.gemm(
                        mxfp8_scale_atoms[k128_half],
                        accumulator,
                        activation_fragment,
                        weight_fragment,
                        accumulator,
                        scale_a=activation_scale,
                        scale_b=weight_scale,
                    )
            return accumulator.load()

        def publish_mxfp8_tile(
            values,
            row_tile,
            rows,
            mailbox_rsrc,
            sample_base,
            sample_count,
        ):
            lane_mod16 = lane % 16
            output_sample_base = (lane // 16) * 4
            for element in range_constexpr(4):
                local_sample = output_sample_base + element
                value = fx.Float32(values[element])
                neighbor = xshfl(value, 1)
                if (local_sample < sample_count) & (lane_mod16 % 2 == 0):
                    sample = sample_base + local_sample
                    row = row_tile * 16 + lane_mod16
                    store_pair(
                        mailbox_rsrc,
                        (sample * rows + row) // 2,
                        value,
                        neighbor,
                    )

        def publish_raw_mxfp8_tile(
            values,
            row_tile,
            rows,
            mailbox_rsrc,
            ready_rsrc,
            sample_base,
            sample_count,
        ):
            lane_mod16 = lane % 16
            output_sample_base = (lane // 16) * 4
            for element in range_constexpr(4):
                local_sample = output_sample_base + element
                value = fx.Float32(values[element])
                neighbor = xshfl(value, 1)
                if (local_sample < sample_count) & (lane_mod16 % 2 == 0):
                    sample = sample_base + local_sample
                    row = row_tile * 16 + lane_mod16
                    store_raw_pair(
                        mailbox_rsrc,
                        (sample * rows + row) // 2,
                        value,
                        neighbor,
                    )
            rocdl.s_waitcnt(vmcnt=0)
            if lane == 0:
                for local_sample in range_constexpr(sample_count):
                    store_i32(
                        ready_rsrc,
                        (sample_base + local_sample) * (rows // 16) + row_tile,
                        1,
                    )

        def stage_mailbox_vector(mailbox_rsrc, pair_base, pairs):
            for load_round in range_constexpr((pairs + _THREADS - 1) // _THREADS):
                pair = tid + load_round * _THREADS
                if pair < pairs:
                    packed = load_pair(mailbox_rsrc, pair_base + pair)
                    lds_store(x, pair, packed.bitcast(fx.Float32))

        def stage_raw_vector(
            mailbox_rsrc,
            ready_rsrc,
            ready_base,
            ready_count,
            pair_base,
            pairs,
        ):
            for ready_round in range_constexpr((ready_count + _THREADS - 1) // _THREADS):
                ready = tid + ready_round * _THREADS
                if ready < ready_count:
                    load_i32(ready_rsrc, ready_base + ready)
            gpu.barrier()
            for load_round in range_constexpr((pairs + _THREADS - 1) // _THREADS):
                pair = tid + load_round * _THREADS
                if pair < pairs:
                    packed = load_raw_pair(mailbox_rsrc, pair_base + pair)
                    lds_store(x, pair, packed.bitcast(fx.Float32))

        def mxfp4_fragment(weight_rsrc, scale_rsrc, row_group, k_chunk, k_size):
            if const_expr(atom_expert_layout):
                # Read ATOM/AITER's preshuffled 16-row weights and
                # 32-row-by-8-group E8M0 scales without repacking checkpoints.
                row_in_group = lane % 16
                lane_word = lane // 16
                row = row_group * 16 + row_in_group
                raw_values = []
                scale_values = []
                padded_groups = ((k_size // 32 + 7) // 8) * 8
                for step in range_constexpr(4):
                    group = k_chunk * 4 + step
                    k64 = group // 2
                    half = group % 2
                    weight_offset = (
                        (((row_group * (k_size // 64) + k64) * 2 + half) * 16 + row_in_group) * 4
                        + lane_word
                    )
                    raw_values.append(
                        fx.Int32(
                            bo.buffer_load(
                                weight_rsrc,
                                weight_offset,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                    )
                    scale_offset = (
                        ((row // 32) * (padded_groups // 8) + group // 8) * 256
                        + (group % 4) * 64
                        + (row % 16) * 4
                        + ((group % 8) // 4) * 2
                        + (row % 32) // 16
                    )
                    scale_values.append(
                        fx.Int32(
                            bo.buffer_load(
                                scale_rsrc,
                                scale_offset,
                                vec_width=1,
                                dtype=T.i8,
                            )
                        )
                    )
                raw = fx.Vector.from_elements(raw_values, fx.Int32)
                scales = [
                    ((value & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                    for value in scale_values
                ]
                return raw, scales
            raw = fx.Vector(
                bo.buffer_load(
                    weight_rsrc,
                    ((row_group * (k_size // 128) + k_chunk) * _WAVE_SIZE + lane) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            row = row_group * 16 + lane % 16
            packed_scale = fx.Int32(
                bo.buffer_load(
                    scale_rsrc,
                    row * (k_size // 128) + k_chunk,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            scales = [
                ((packed_scale.shrui(fx.Int32(step * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                for step in range_constexpr(4)
            ]
            return raw, scales

        def mxfp4_apply(accumulator, fragment, input_word, coefficient=None):
            raw, scales = fragment
            for step in range_constexpr(4):
                lhs = mxfp4_to_bf16x8(raw[step], scales[step])
                rhs = fx.ptr_load(
                    x + input_word + (lane // 16) * 4 + step * 16,
                    result_type=fx.Vector.make_type(4, fx.Float32),
                ).bitcast(fx.BFloat16)
                partial = fx.Vector.filled(4, 0.0, fx.Float32)
                partial = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [lhs, rhs, partial]))
                if const_expr(coefficient is None):
                    accumulator = [accumulator[item] + partial[item] for item in range_constexpr(4)]
                else:
                    accumulator = [accumulator[item] + partial[item] * coefficient for item in range_constexpr(4)]
            return accumulator

        def mxfp8_bf16_accumulate(
            weight_rsrc,
            scale_rsrc,
            activation_word_base,
            row_tile,
            k_dim,
            split_wave,
            split_waves,
        ):
            k_chunks = k_dim // 64
            chunks_per_wave = k_chunks // split_waves
            accumulator = fx.Vector.filled(4, 0.0, fx.Float32)
            for local_chunk in range_constexpr(chunks_per_wave):
                chunk = split_wave * chunks_per_wave + local_chunk
                for step_index in range_constexpr(2):
                    atom_group = step_index * 2 + (lane // 16) // 2
                    weight = fx.Vector(
                        bo.buffer_load(
                            weight_rsrc,
                            (((row_tile * k_chunks + chunk) * 4 + atom_group) * 16 + lane % 16) * 4
                            + ((lane // 16) % 2) * 2,
                            vec_width=2,
                            dtype=T.i32,
                        )
                    )
                    scale_group = chunk * 2 + step_index
                    scale_word = fx.Int32(
                        bo.buffer_load(
                            scale_rsrc,
                            (
                                ((row_tile // 2) * (k_dim // 256) + scale_group // 8) * 64
                                + (scale_group % 4) * 16
                                + lane % 16
                            ),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    scale_byte_index = ((scale_group % 8) // 4) * 2 + row_tile % 2
                    scale_byte = scale_word.shrui(fx.Int32(scale_byte_index * 8)) & fx.Int32(0xFF)
                    scale = (scale_byte << fx.Int32(23)).bitcast(fx.Float32)
                    lhs = mxfp8_to_bf16x8(weight[0], weight[1], scale)
                    rhs = fx.ptr_load(
                        x + activation_word_base + (chunk * 64) // 2 + (lane // 16) * 4 + step_index * 16,
                        result_type=fx.Vector.make_type(4, fx.Float32),
                    ).bitcast(fx.BFloat16)
                    accumulator = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(4, T.f32),
                            [lhs, rhs, accumulator],
                        )
                    )
            return list(accumulator)

        def moe_peer_reduce(local_pairs, pair_base, local_values, region, emit):
            moe_max_pairs = samples * _HIDDEN // 2
            moe_slot_bytes = npes * moe_max_pairs * 8
            region_base = fx.Int64(region * 2 * moe_slot_bytes) + fx.Int64(slot) * fx.Int64(moe_slot_bytes)
            peer_rounds = (npes + _WAVES - 1) // _WAVES
            for peer_round in range_constexpr(peer_rounds):
                peer = wave + peer_round * _WAVES
                if peer < npes:
                    peer_words = fx.Vector(bo.buffer_load(rsrc(moe_peers), peer * 2, vec_width=2, dtype=T.i32))
                    peer_address = (fx.Int64(uniform(peer_words[1])) << 32) | fx.Int64(
                        fx.Uint32(uniform(peer_words[0]))
                    )
                    peer_rsrc = rsrc(peer_address + region_base)
                    if lane < local_pairs:
                        global_pair = pair_base + lane
                        packed = lds_load(local_values, lane).bitcast(fx.Int32)
                        mailbox = rank * moe_max_pairs + global_pair
                        bo.buffer_store(
                            fx.Vector.from_elements([packed, tag], fx.Int32),
                            peer_rsrc,
                            mailbox * 2,
                            cache_modifier=CM_SYS,
                        )
            gpu.barrier()
            if tid < local_pairs:
                global_pair = pair_base + tid
                local_rsrc = rsrc(moe_symmetric + region_base)

                def load_peers():
                    words = []
                    for source_rank in range_constexpr(npes):
                        mailbox = source_rank * moe_max_pairs + global_pair
                        value_tag = fx.Vector(
                            bo.buffer_load(
                                local_rsrc,
                                mailbox * 2,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        words += [value_tag[0], value_tag[1]]
                    return fx.Vector.from_elements(words, fx.Int32)

                peer_values = load_peers()
                pending = peer_values[1] != tag
                for source_rank in range_constexpr(1, npes):
                    pending = pending | (peer_values[source_rank * 2 + 1] != tag)
                while pending:
                    rocdl.s_nop(0)
                    peer_values = load_peers()
                    pending = peer_values[1] != tag
                    for source_rank in range_constexpr(1, npes):
                        pending = pending | (peer_values[source_rank * 2 + 1] != tag)
                sum_low = fx.Float32(0.0)
                sum_high = fx.Float32(0.0)
                for source_rank in range_constexpr(npes):
                    packed = peer_values[source_rank * 2]
                    sum_low = sum_low + (packed << 16).bitcast(fx.Float32)
                    sum_high = sum_high + (packed & fx.Int32(-65536)).bitcast(fx.Float32)
                emit(tid, sum_low, sum_high)
            gpu.barrier()

        def publish_mfma_pairs(
            accumulator,
            rows,
            waves_per_row,
            emit,
            sample_base,
            sample_count,
        ):
            fx.ptr_store(
                fx.Vector.from_elements(accumulator, fx.Float32),
                reduction + (wave * _WAVE_SIZE + lane) * 4,
            )
            gpu.barrier()
            pair_count = rows * sample_count // 2
            for output_round in range_constexpr((pair_count + _THREADS - 1) // _THREADS):
                item = tid + output_round * _THREADS
                if item < pair_count:
                    local_sample = item // (rows // 2)
                    local_row = (item % (rows // 2)) * 2
                    pair_values = []
                    for pair_element in range_constexpr(2):
                        row = local_row + pair_element
                        row_in_group = row % 16
                        first_source_wave = (row // 16) * waves_per_row
                        value = fx.Float32(0.0)
                        for source_offset in range_constexpr(waves_per_row):
                            source_wave = first_source_wave + source_offset
                            source_index = (
                                source_wave * _WAVE_SIZE + local_sample + 16 * (row_in_group // 4)
                            ) * 4 + row_in_group % 4
                            value = value + lds_load(reduction, source_index)
                        pair_values.append(value)
                    emit(
                        local_row,
                        sample_base + local_sample,
                        pair_values[0],
                        pair_values[1],
                    )
            gpu.barrier()

        # Stage 0: the MonoKernel specialization folds pre-attention AttnRes
        # into the same launch and publishes its normalized BF16 output.
        if const_expr(fuse_attn_res):
            if bid < samples * _ATTN_RES_CTAS:
                run_attn_res_chunk(
                    bid // _ATTN_RES_CTAS,
                    bid % _ATTN_RES_CTAS,
                    hidden_states,
                    hidden_states,
                    self_res_norm,
                    self_res_qk,
                    input_norm,
                    pre_updated,
                    pre_output,
                    pre_mailbox_rsrc,
                    pre_ready_rsrc,
                    pre_stats_rsrc,
                    attn_res_blocks,
                    False,
                    False,
                    False,
                    block_write_idx,
                    False,
                )

        # Dense Kimi MLA frontend.  The KDA and MLA mixers share Stage 0 and
        # Stage 3 onward; this branch publishes the same 1536-wide per-head
        # value mailbox consumed by the common output projection and K3 tail.
        if const_expr(mla):
            wait_attn_res_chunks(pre_ready_rsrc, 0, samples)

            def hidden_value(sample, k):
                return fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(
                            pre_mailbox_rsrc,
                            sample * _HIDDEN + k,
                            vec_width=1,
                            dtype=T.bf16,
                            cache_modifier=CM_DEV,
                        )
                    )
                )

            def raw_bf16(resource, index):
                return fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(
                            resource,
                            index,
                            vec_width=1,
                            dtype=T.bf16,
                            cache_modifier=CM_DEV,
                        )
                    )
                )

            def project_hidden(weight_address, output_rsrc, rows):
                weight = rsrc(weight_address)
                pair_task = bid * _THREADS + tid
                pair_count = samples * (rows // 2)
                while pair_task < pair_count:
                    sample = pair_task // (rows // 2)
                    row = (pair_task % (rows // 2)) * 2
                    low = fx.Float32(0.0)
                    high = fx.Float32(0.0)
                    k = fx.Int32(0)
                    while k < fx.Int32(_HIDDEN):
                        value = hidden_value(sample, k)
                        low = low + value * raw_bf16(weight, row * _HIDDEN + k)
                        high = high + value * raw_bf16(
                            weight,
                            (row + 1) * _HIDDEN + k,
                        )
                        k = k + 1
                    store_pair(
                        output_rsrc,
                        sample * (rows // 2) + row // 2,
                        low,
                        high,
                    )
                    pair_task = pair_task + _BLOCKS * _THREADS

            project_hidden(
                mla_qkv_weight,
                mla_qkv_rsrc,
                _MLA_Q_LORA + _MLA_CACHE_ROW,
            )
            project_hidden(mla_gate_weight, mla_gate_rsrc, _PROJECTION)

            if bid < samples:
                sample = bid
                q_square = fx.Float32(0.0)
                q_pair = tid
                while q_pair < _MLA_Q_LORA // 2:
                    q0, q1 = pair_values(
                        mla_qkv_rsrc,
                        sample * ((_MLA_Q_LORA + _MLA_CACHE_ROW) // 2)
                        + q_pair,
                    )
                    q_square = q_square + q0 * q0 + q1 * q1
                    q_pair = q_pair + _THREADS
                q_inv = rsq(
                    block_sum(q_square) * (1.0 / _MLA_Q_LORA)
                    + _MLA_NORM_EPS
                )
                q_pair = tid
                q_gain_rsrc = rsrc(mla_q_norm)
                while q_pair < _MLA_Q_LORA // 2:
                    q0, q1 = pair_values(
                        mla_qkv_rsrc,
                        sample * ((_MLA_Q_LORA + _MLA_CACHE_ROW) // 2)
                        + q_pair,
                    )
                    store_pair(
                        mla_qnorm_rsrc,
                        sample * (_MLA_Q_LORA // 2) + q_pair,
                        q0 * q_inv * raw_bf16(q_gain_rsrc, q_pair * 2),
                        q1 * q_inv
                        * raw_bf16(q_gain_rsrc, q_pair * 2 + 1),
                    )
                    q_pair = q_pair + _THREADS

                kv_square = fx.Float32(0.0)
                kv_pair = tid
                qkv_pairs = (_MLA_Q_LORA + _MLA_CACHE_ROW) // 2
                while kv_pair < _MLA_KV_LORA // 2:
                    kv0, kv1 = pair_values(
                        mla_qkv_rsrc,
                        sample * qkv_pairs + _MLA_Q_LORA // 2 + kv_pair,
                    )
                    kv_square = kv_square + kv0 * kv0 + kv1 * kv1
                    kv_pair = kv_pair + _THREADS
                kv_inv = rsq(
                    block_sum(kv_square) * (1.0 / _MLA_KV_LORA)
                    + _MLA_NORM_EPS
                )
                kv_gain_rsrc = rsrc(mla_kv_norm)
                kv_pair = tid
                while kv_pair < _MLA_KV_LORA // 2:
                    kv0, kv1 = pair_values(
                        mla_qkv_rsrc,
                        sample * qkv_pairs + _MLA_Q_LORA // 2 + kv_pair,
                    )
                    store_pair(
                        mla_fresh_rsrc,
                        sample * (_MLA_CACHE_ROW // 2) + kv_pair,
                        kv0
                        * kv_inv
                        * raw_bf16(kv_gain_rsrc, kv_pair * 2),
                        kv1
                        * kv_inv
                        * raw_bf16(kv_gain_rsrc, kv_pair * 2 + 1),
                    )
                    kv_pair = kv_pair + _THREADS

                pe_pair = tid
                while pe_pair < _MLA_PE // 2:
                    pe0, pe1 = pair_values(
                        mla_qkv_rsrc,
                        sample * qkv_pairs
                        + (_MLA_Q_LORA + _MLA_KV_LORA) // 2
                        + pe_pair,
                    )
                    store_pair(
                        mla_fresh_rsrc,
                        sample * (_MLA_CACHE_ROW // 2)
                        + _MLA_KV_LORA // 2
                        + pe_pair,
                        pe0,
                        pe1,
                    )
                    pe_pair = pe_pair + _THREADS
                gpu.barrier()

                batch_id = uniform(
                    bo.buffer_load(
                        rsrc(mla_batch_ids),
                        sample,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                physical_slot = uniform(
                    bo.buffer_load(
                        rsrc(mla_slot_mapping),
                        sample,
                        vec_width=1,
                        dtype=T.i64,
                    )
                )
                cache_live = (batch_id >= 0) & (physical_slot >= 0)
                if (tid < _MLA_CACHE_ROW // 4) & cache_live:
                    d = tid * 4
                    a0, a1 = pair_values(
                        mla_fresh_rsrc,
                        sample * (_MLA_CACHE_ROW // 2) + d // 2,
                    )
                    a2, a3 = pair_values(
                        mla_fresh_rsrc,
                        sample * (_MLA_CACHE_ROW // 2) + d // 2 + 1,
                    )
                    descale = fx.Float32(
                        bo.buffer_load(
                            rsrc(mla_cache_scale),
                            0,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                    inverse = rcp(descale)
                    q0 = fx.max(
                        fx.Float32(-FP8_MAX),
                        fx.min(fx.Float32(FP8_MAX), a0 * inverse),
                    )
                    q1 = fx.max(
                        fx.Float32(-FP8_MAX),
                        fx.min(fx.Float32(FP8_MAX), a1 * inverse),
                    )
                    q2 = fx.max(
                        fx.Float32(-FP8_MAX),
                        fx.min(fx.Float32(FP8_MAX), a2 * inverse),
                    )
                    q3 = fx.max(
                        fx.Float32(-FP8_MAX),
                        fx.min(fx.Float32(FP8_MAX), a3 * inverse),
                    )
                    low = fx.Int32(
                        rocdl.cvt_pk_fp8_f32(
                            T.i32,
                            q0,
                            q1,
                            fx.Int32(0),
                            False,
                        )
                    ) & fx.Int32(0xFFFF)
                    high = fx.Int32(
                        rocdl.cvt_pk_fp8_f32(
                            T.i32,
                            q2,
                            q3,
                            fx.Int32(0),
                            False,
                        )
                    ) & fx.Int32(0xFFFF)
                    bo.buffer_store(
                        low | (high << 16),
                        rsrc(mla_cache),
                        fx.Int64(physical_slot)
                        * (_MLA_CACHE_ROW // 4)
                        + tid,
                        cache_modifier=CM_DEV,
                    )

            # q_b: normalized low-rank query -> 12 x (128 no-PE + 64 PE).
            q_pair_task = bid * _THREADS + tid
            q_pairs = _HEADS * _MLA_Q_HEAD // 2
            q_weight_rsrc = rsrc(mla_q_b_weight)
            while q_pair_task < samples * q_pairs:
                sample = q_pair_task // q_pairs
                row = (q_pair_task % q_pairs) * 2
                low = fx.Float32(0.0)
                high = fx.Float32(0.0)
                k = fx.Int32(0)
                while k < fx.Int32(_MLA_Q_LORA):
                    value = tagged_bf16(
                        mla_qnorm_rsrc,
                        sample * _MLA_Q_LORA + k,
                    )
                    low = low + value * raw_bf16(
                        q_weight_rsrc,
                        row * _MLA_Q_LORA + k,
                    )
                    high = high + value * raw_bf16(
                        q_weight_rsrc,
                        (row + 1) * _MLA_Q_LORA + k,
                    )
                    k = k + 1
                store_pair(
                    mla_q_rsrc,
                    sample * q_pairs + row // 2,
                    low,
                    high,
                )
                q_pair_task = q_pair_task + _BLOCKS * _THREADS

            # W_UK absorbs the non-positional 128 query dimensions.
            uk_pair_task = bid * _THREADS + tid
            uk_pairs = _HEADS * _MLA_KV_LORA // 2
            uk_weight_rsrc = rsrc(mla_uk_weight)
            while uk_pair_task < samples * uk_pairs:
                sample = uk_pair_task // uk_pairs
                row = (uk_pair_task % uk_pairs) * 2
                head = row // _MLA_KV_LORA
                low = fx.Float32(0.0)
                high = fx.Float32(0.0)
                k = fx.Int32(0)
                while k < fx.Int32(_HEAD_DIM):
                    value = tagged_bf16(
                        mla_q_rsrc,
                        sample * (_HEADS * _MLA_Q_HEAD)
                        + head * _MLA_Q_HEAD
                        + k,
                    )
                    low = low + value * raw_bf16(
                        uk_weight_rsrc,
                        row * _HEAD_DIM + k,
                    )
                    high = high + value * raw_bf16(
                        uk_weight_rsrc,
                        (row + 1) * _HEAD_DIM + k,
                    )
                    k = k + 1
                store_pair(
                    mla_qlat_rsrc,
                    sample * uk_pairs + row // 2,
                    low,
                    high,
                )
                uk_pair_task = uk_pair_task + _BLOCKS * _THREADS

            # Dense paged MLA.  Each CTA owns a 64-value chunk and walks every
            # visible logical key through the request's physical block table.
            dense_task = bid
            dense_tasks = samples * _HEADS * _MLA_VALUE_CHUNKS
            cache_descale = fx.Float32(
                bo.buffer_load(
                    rsrc(mla_cache_scale),
                    0,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            while dense_task < dense_tasks:
                sample = dense_task // (_HEADS * _MLA_VALUE_CHUNKS)
                head_chunk = dense_task % (
                    _HEADS * _MLA_VALUE_CHUNKS
                )
                head = head_chunk // _MLA_VALUE_CHUNKS
                value_chunk = head_chunk % _MLA_VALUE_CHUNKS
                batch_id = uniform(
                    bo.buffer_load(
                        rsrc(mla_batch_ids),
                        sample,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                position = uniform(
                    bo.buffer_load(
                        rsrc(mla_positions),
                        sample,
                        vec_width=1,
                        dtype=T.i64,
                    )
                )
                context = (batch_id >= 0).select(
                    uniform(
                        bo.buffer_load(
                            rsrc(mla_context_lens),
                            fx.max(batch_id, fx.Int32(0)),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                    fx.Int32(0),
                )
                visible = fx.min(context, fx.Int32(position + 1))
                running_max = fx.Float32(-3.402823466e38)
                running_sum = fx.Float32(0.0)
                value_acc = fx.Float32(0.0)
                logical = fx.Int32(0)
                while logical < visible:
                    logical_block = logical // mla_block_size
                    block_offset = logical % mla_block_size
                    physical_block = uniform(
                        bo.buffer_load(
                            rsrc(mla_block_tables),
                            batch_id * mla_block_table_stride
                            + logical_block,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    physical_slot = physical_block * mla_block_ratio + block_offset
                    fresh_row = fx.Int32(-1)
                    fresh_scan = fx.Int32(0)
                    while fresh_scan < fx.Int32(samples):
                        fresh_batch = uniform(
                            bo.buffer_load(
                                rsrc(mla_batch_ids),
                                fresh_scan,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        fresh_slot = uniform(
                            bo.buffer_load(
                                rsrc(mla_slot_mapping),
                                fresh_scan,
                                vec_width=1,
                                dtype=T.i64,
                            )
                        )
                        matched = (fresh_batch >= 0) & (
                            fresh_slot == fx.Int64(physical_slot)
                        )
                        fresh_row = matched.select(fresh_scan, fresh_row)
                        fresh_scan = fresh_scan + 1

                    local_score = fx.Float32(0.0)
                    if tid < _MLA_CACHE_ROW // 8:
                        d0 = tid * 8
                        words = fx.Vector(
                            bo.buffer_load(
                                rsrc(mla_cache),
                                physical_slot * (_MLA_CACHE_ROW // 4)
                                + tid * 2,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        cached = fp8_to_bf16x8(words[0], words[1]).to(
                            fx.Float32
                        )
                        for j in range_constexpr(8):
                            d = d0 + j
                            query = fx.Float32(0.0)
                            if d < _MLA_KV_LORA:
                                query = tagged_bf16(
                                    mla_qlat_rsrc,
                                    (sample * _HEADS + head)
                                    * _MLA_KV_LORA
                                    + d,
                                )
                            else:
                                query = tagged_bf16(
                                    mla_q_rsrc,
                                    (sample * _HEADS + head)
                                    * _MLA_Q_HEAD
                                    + _HEAD_DIM
                                    + d
                                    - _MLA_KV_LORA,
                                )
                            key = fx.Float32(cached[j]) * cache_descale
                            if fresh_row >= 0:
                                key = tagged_bf16(
                                    mla_fresh_rsrc,
                                    fresh_row * _MLA_CACHE_ROW + d,
                                )
                            local_score = local_score + query * key
                    score = block_sum(local_score) * (
                        _MLA_Q_HEAD**-0.5
                    )

                    value = fx.Float32(0.0)
                    if (wave == 0) & (lane < _WAVE_SIZE):
                        value_d = value_chunk * _WAVE_SIZE + lane
                        value_words = fx.Vector(
                            bo.buffer_load(
                                rsrc(mla_cache),
                                physical_slot * (_MLA_CACHE_ROW // 4)
                                + (value_d // 8) * 2,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        cache_values = fp8_to_bf16x8(
                            value_words[0],
                            value_words[1],
                        ).to(fx.Float32)
                        cached_value = fx.Float32(cache_values[0])
                        for j in range_constexpr(1, 8):
                            cached_value = (
                                value_d % 8 == j
                            ).select(
                                fx.Float32(cache_values[j]),
                                cached_value,
                            )
                        value = cached_value * cache_descale
                        if fresh_row >= 0:
                            value = tagged_bf16(
                                mla_fresh_rsrc,
                                fresh_row * _MLA_CACHE_ROW + value_d,
                            )

                    next_max = fx.max(running_max, score)
                    old_scale = exp(running_max - next_max)
                    new_scale = exp(score - next_max)
                    value_acc = value_acc * old_scale + value * new_scale
                    running_sum = running_sum * old_scale + new_scale
                    running_max = next_max
                    logical = logical + 1
                result0 = (running_sum > 0.0).select(
                    value_acc * rcp(running_sum),
                    fx.Float32(0.0),
                )
                result1 = xshfl(result0, 1)
                if (wave == 0) & (lane % 2 == 0):
                    store_pair(
                        mla_dense_acc_rsrc,
                        (sample * _HEADS + head)
                        * (_MLA_KV_LORA // 2)
                        + value_chunk * (_WAVE_SIZE // 2)
                        + lane // 2,
                        result0,
                        result1,
                    )
                dense_task = dense_task + _BLOCKS

            # W_UV and the Kimi attention output gate publish the common
            # 1536-wide Stage-3 input mailbox.
            uv_pair_task = bid * _THREADS + tid
            uv_pairs = _PROJECTION // 2
            uv_weight_rsrc = rsrc(mla_uv_weight)
            while uv_pair_task < samples * uv_pairs:
                sample = uv_pair_task // uv_pairs
                row = (uv_pair_task % uv_pairs) * 2
                head = row // _HEAD_DIM
                low = fx.Float32(0.0)
                high = fx.Float32(0.0)
                k = fx.Int32(0)
                while k < fx.Int32(_MLA_KV_LORA):
                    value = tagged_bf16(
                        mla_dense_acc_rsrc,
                        (sample * _HEADS + head) * _MLA_KV_LORA + k,
                    )
                    low = low + value * raw_bf16(
                        uv_weight_rsrc,
                        row * _MLA_KV_LORA + k,
                    )
                    high = high + value * raw_bf16(
                        uv_weight_rsrc,
                        (row + 1) * _MLA_KV_LORA + k,
                    )
                    k = k + 1
                gate0, gate1 = pair_values(
                    mla_gate_rsrc,
                    sample * uv_pairs + row // 2,
                )
                low = low * rcp(fx.Float32(1.0) + exp(-gate0))
                high = high * rcp(fx.Float32(1.0) + exp(-gate1))
                store_pair(
                    norm_mailbox_rsrc,
                    sample * uv_pairs + row // 2,
                    low,
                    high,
                )
                uv_pair_task = uv_pair_task + _BLOCKS * _THREADS

        # Stage 1: BF16 7168 -> 6400 input projection.  One wave owns one
        # 16-row group and accumulates the complete K dimension, preserving the
        # non-split-K numerical order needed by the recurrent state update.
        input_tasks = 0 if mla else sample_groups * input_row_tasks
        input_task = bid
        while input_task < input_tasks:
            if const_expr(sample_groups == 1):
                input_row_task = input_task
                sample_base = 0
            else:
                sample_group = input_task // input_row_tasks
                input_row_task = input_task % input_row_tasks
                sample_base = sample_group * staged_samples
            stage_hidden(sample_base, staged_samples)
            gpu.barrier()
            input_accumulator = bf16_mfma(
                input_weight_rsrc,
                input_row_task * input_row_groups,
                _HIDDEN,
                input_row_groups,
                input_split_waves,
                14,
                staged_samples,
            )

            def emit_input(local_row, sample, value_low, value_high):
                row = input_row_task * input_row_tile + local_row
                if row < _FUSED_WIDTH:
                    put_input_pair(sample, row, value_low, value_high)

            publish_mfma_pairs(
                input_accumulator,
                input_row_tile,
                input_split_waves,
                emit_input,
                sample_base,
                staged_samples,
            )
            input_task = input_task + _BLOCKS

        def prepare_kda_gate(sample, head):
            if tid < _HEAD_DIM:
                f_a_value = get_input(sample, 4 * _PROJECTION + _HEADS + tid)
                fx.ptr_store(f_a_value.to(fx.BFloat16), shared_f_a + tid)
            gpu.barrier()

            if tid < 4 * _HEAD_DIM:
                gate_row = tid // 4
                gate_split = tid % 4
                gate_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                for feature_group in range_constexpr(0, _HEAD_DIM, 4 * _VALUES_PER_THREAD):
                    feature_base = feature_group + gate_split * _VALUES_PER_THREAD
                    features = fx.Vector(
                        fx.ptr_load(
                            shared_f_a + feature_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                    weights = fx.Vector(
                        bo.buffer_load(
                            gate_weight_rsrc,
                            (head * _HEAD_DIM + gate_row) * _HEAD_DIM + feature_base,
                            vec_width=_VALUES_PER_THREAD,
                            dtype=T.bf16,
                        )
                    ).to(fx.Float32)
                    gate_parts = fx.math.fma(features, weights, gate_parts)
                gate_value = gate_parts.reduce(fx.ReductionOp.ADD)
                for offset in (2, 1):
                    gate_value = gate_value + xshfl(gate_value, offset)
                if gate_split == 0:
                    fx.ptr_store(gate_value.to(fx.BFloat16), shared_gate + gate_row)
            gpu.barrier()

        def prepare_kda_conv(sample, head, conv_state_rsrc, conv_state_out_rsrc):
            def convolve(channel):
                if const_expr(agentic_batch_size > 0):
                    request = sample // 8
                    token = sample % 8
                    accepted = uniform(
                        bo.buffer_load(
                            accepted_rsrc,
                            request,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )

                    def load_old(row):
                        return fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    conv_state_rsrc,
                                    row * _CONV_CHANNELS + channel,
                                    vec_width=1,
                                    dtype=T.bf16,
                                    cache_modifier=CM_DEV,
                                )
                            )
                        )

                    old_m1 = load_old(accepted - 1)
                    old_0 = load_old(accepted)
                    old_p1 = load_old(accepted + 1)
                    request_start = request * 8
                    draft_m3 = get_input(
                        request_start + fx.max(token - 3, fx.Int32(0)),
                        channel,
                    )
                    draft_m2 = get_input(
                        request_start + fx.max(token - 2, fx.Int32(0)),
                        channel,
                    )
                    draft_m1 = get_input(
                        request_start + fx.max(token - 1, fx.Int32(0)),
                        channel,
                    )
                    state0 = (token == 0).select(
                        old_m1,
                        (token == 1).select(
                            old_0,
                            (token == 2).select(old_p1, draft_m3),
                        ),
                    )
                    state1 = (token == 0).select(
                        old_0,
                        (token == 1).select(old_p1, draft_m2),
                    )
                    state2 = (token == 0).select(old_p1, draft_m1)
                else:
                    state_offsets = (
                        conv_state_offset(
                            conv_state_layout,
                            channel,
                            0,
                            _CONV_CHANNELS,
                            _CONV_STATE_LENGTH,
                        ),
                        conv_state_offset(
                            conv_state_layout,
                            channel,
                            1,
                            _CONV_CHANNELS,
                            _CONV_STATE_LENGTH,
                        ),
                        conv_state_offset(
                            conv_state_layout,
                            channel,
                            2,
                            _CONV_CHANNELS,
                            _CONV_STATE_LENGTH,
                        ),
                    )
                    state0 = fx.Float32(
                        fx.BFloat16(
                            bo.buffer_load(
                                conv_state_rsrc,
                                state_offsets[0],
                                vec_width=1,
                                dtype=T.bf16,
                                cache_modifier=CM_DEV,
                            )
                        )
                    )
                    state1 = fx.Float32(
                        fx.BFloat16(
                            bo.buffer_load(
                                conv_state_rsrc,
                                state_offsets[1],
                                vec_width=1,
                                dtype=T.bf16,
                                cache_modifier=CM_DEV,
                            )
                        )
                    )
                    state2 = fx.Float32(
                        fx.BFloat16(
                            bo.buffer_load(
                                conv_state_rsrc,
                                state_offsets[2],
                                vec_width=1,
                                dtype=T.bf16,
                                cache_modifier=CM_DEV,
                            )
                        )
                    )
                current = get_input(sample, channel)
                weights = fx.Vector(
                    bo.buffer_load(
                        conv_weight_rsrc,
                        channel * _CONV_KERNEL_WIDTH,
                        vec_width=_CONV_KERNEL_WIDTH,
                        dtype=T.bf16,
                    )
                ).to(fx.Float32)
                values = fx.Vector.from_elements(
                    [state0, state1, state2, current],
                    fx.Float32,
                )
                convolution = (values * weights).reduce(fx.ReductionOp.ADD)
                activated = convolution * sigmoid_batch([convolution])[0]
                if const_expr(agentic_batch_size > 0):
                    if token == 7:
                        bo.buffer_store(
                            old_0.to(fx.BFloat16),
                            conv_state_out_rsrc,
                            channel,
                            cache_modifier=CM_DEV,
                        )
                        bo.buffer_store(
                            old_p1.to(fx.BFloat16),
                            conv_state_out_rsrc,
                            _CONV_CHANNELS + channel,
                            cache_modifier=CM_DEV,
                        )
                        for draft_token in range_constexpr(8):
                            bo.buffer_store(
                                get_input(
                                    request_start + draft_token,
                                    channel,
                                ).to(fx.BFloat16),
                                conv_state_out_rsrc,
                                (draft_token + 2) * _CONV_CHANNELS + channel,
                                cache_modifier=CM_DEV,
                            )
                else:
                    bo.buffer_store(
                        state1.to(fx.BFloat16),
                        conv_state_out_rsrc,
                        state_offsets[0],
                        cache_modifier=CM_DEV,
                    )
                    bo.buffer_store(
                        state2.to(fx.BFloat16),
                        conv_state_out_rsrc,
                        state_offsets[1],
                        cache_modifier=CM_DEV,
                    )
                    bo.buffer_store(
                        current.to(fx.BFloat16),
                        conv_state_out_rsrc,
                        state_offsets[2],
                        cache_modifier=CM_DEV,
                    )
                return activated.to(fx.BFloat16)

            if tid < _HEAD_DIM:
                channel = head * _HEAD_DIM + tid
                fx.ptr_store(convolve(channel), shared_query + tid)
            elif tid < 2 * _HEAD_DIM:
                channel_in_head = tid - _HEAD_DIM
                channel = _PROJECTION + head * _HEAD_DIM + channel_in_head
                fx.ptr_store(convolve(channel), shared_key + channel_in_head)
            elif tid < 3 * _HEAD_DIM:
                channel_in_head = tid - 2 * _HEAD_DIM
                channel = 2 * _PROJECTION + head * _HEAD_DIM + channel_in_head
                fx.ptr_store(convolve(channel), shared_value + channel_in_head)
            gpu.barrier()

        # Stage 2a: ordinary decode uses one CTA per independent (sample, head).
        # Ordered MTP recurrence is handled by the pipeline below.
        recurrence_task = bid
        recurrence_tasks = 0 if (mtp or mla) else samples * _HEADS
        if recurrence_task < recurrence_tasks:
            stamp(1)
            sample = recurrence_task // _HEADS
            head = recurrence_task % _HEADS
            input_slot, output_slot, acceptance_valid = state_slots(sample)
            state_rsrc = rsrc(
                recurrent_state
                + fx.Int64(input_slot) * fx.Int64(state_slot_bytes)
            )
            state_out_rsrc = state_rsrc
            conv_state_rsrc = rsrc(
                conv_state + fx.Int64(input_slot) * fx.Int64(conv_slot_bytes)
            )
            conv_state_out_rsrc = conv_state_rsrc

            if (
                (input_slot >= 0)
                & (output_slot >= 0)
                & acceptance_valid
            ):
                prepare_kda_gate(sample, head)
                prepare_kda_conv(sample, head, conv_state_rsrc, conv_state_out_rsrc)

                k_lane = lane % _K_LANES
                v_lane = lane // _K_LANES
                exp_a_log = exp(fx.Float32(bo.buffer_load(a_log_rsrc, head, vec_width=1, dtype=T.f32)))
                beta_logit = get_input(sample, 4 * _PROJECTION + head)
                beta_value = sigmoid_batch([beta_logit])[0]

                query_vectors = [None] * _K_ITERS
                key_vectors = [None] * _K_ITERS
                decay_vectors = [None] * _K_ITERS
                query_square = fx.Float32(0.0)
                key_square = fx.Float32(0.0)
                for k_iter in range_constexpr(_K_ITERS):
                    k_base = k_lane * _VALUES_PER_THREAD + k_iter * _K_TILE
                    query_vector = fx.Vector(
                        fx.ptr_load(
                            shared_query + k_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                    key_vector = fx.Vector(
                        fx.ptr_load(
                            shared_key + k_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                    gate_vector = fx.Vector(
                        fx.ptr_load(
                            shared_gate + k_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                    dt_vector = fx.Vector(
                        bo.buffer_load(
                            dt_bias_rsrc,
                            head * _HEAD_DIM + k_base,
                            vec_width=_VALUES_PER_THREAD,
                            dtype=T.bf16,
                        )
                    ).to(fx.Float32)
                    query_vectors[k_iter] = query_vector
                    key_vectors[k_iter] = key_vector
                    query_square = query_square + (query_vector * query_vector).reduce(fx.ReductionOp.ADD)
                    key_square = key_square + (key_vector * key_vector).reduce(fx.ReductionOp.ADD)
                    gate_sigmoid = sigmoid_batch(
                        [
                            exp_a_log * (gate_vector[item] + dt_vector[item])
                            for item in range_constexpr(_VALUES_PER_THREAD)
                        ]
                    )
                    decay_vectors[k_iter] = fx.Vector.from_elements(
                        [
                            exp(fx.Float32(_GATE_LOWER_BOUND) * gate_sigmoid[item])
                            for item in range_constexpr(_VALUES_PER_THREAD)
                        ],
                        fx.Float32,
                    )

                def subgroup_sum(value):
                    for offset in (4, 2, 1):
                        value = value + xshfl(value, offset)
                    return value

                query_inverse_norm = rsq(subgroup_sum(query_square) + fx.Float32(1.0e-6))
                key_inverse_norm = rsq(subgroup_sum(key_square) + fx.Float32(1.0e-6))
                for k_iter in range_constexpr(_K_ITERS):
                    query_vectors[k_iter] = query_vectors[k_iter] * fx.Vector.filled(
                        _VALUES_PER_THREAD,
                        query_inverse_norm * fx.Float32(_Q_SCALE),
                        fx.Float32,
                    )
                    key_vectors[k_iter] = key_vectors[k_iter] * fx.Vector.filled(
                        _VALUES_PER_THREAD,
                        key_inverse_norm,
                        fx.Float32,
                    )

                dot_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                for k_iter in range_constexpr(_K_ITERS):
                    dot_parts = fx.math.fma(key_vectors[k_iter], query_vectors[k_iter], dot_parts)
                dot_key_query = subgroup_sum(dot_parts.reduce(fx.ReductionOp.ADD))

                state_vectors = [None] * (_V_ITERS * _K_ITERS)
                results = [None] * _V_ITERS
                for v_iter in range_constexpr(_V_ITERS):
                    value_index = wave * _V_LANES + v_lane + v_iter * _V_TILE
                    for k_iter in range_constexpr(_K_ITERS):
                        k_base = k_lane * _VALUES_PER_THREAD + k_iter * _K_TILE
                        state_offset = (head * _HEAD_DIM + value_index) * _HEAD_DIM + k_base
                        state_vectors[v_iter * _K_ITERS + k_iter] = fx.Vector(
                            bo.buffer_load(
                                state_rsrc,
                                state_offset,
                                vec_width=_VALUES_PER_THREAD,
                                dtype=T.f32,
                                cache_modifier=CM_DEV,
                            )
                        )

                for v_iter in range_constexpr(_V_ITERS):
                    value_index = wave * _V_LANES + v_lane + v_iter * _V_TILE
                    state_key_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                    state_query_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                    for k_iter in range_constexpr(_K_ITERS):
                        index = v_iter * _K_ITERS + k_iter
                        decayed = state_vectors[index] * decay_vectors[k_iter]
                        state_vectors[index] = decayed
                        state_key_parts = fx.math.fma(decayed, key_vectors[k_iter], state_key_parts)
                        state_query_parts = fx.math.fma(decayed, query_vectors[k_iter], state_query_parts)
                    state_key = subgroup_sum(state_key_parts.reduce(fx.ReductionOp.ADD))
                    state_query = subgroup_sum(state_query_parts.reduce(fx.ReductionOp.ADD))
                    value_input = fx.Float32(fx.ptr_load(shared_value + value_index))
                    value_new = (value_input - state_key) * beta_value
                    value_new_vector = fx.Vector.filled(_VALUES_PER_THREAD, value_new, fx.Float32)
                    for k_iter in range_constexpr(_K_ITERS):
                        index = v_iter * _K_ITERS + k_iter
                        state_vectors[index] = fx.math.fma(key_vectors[k_iter], value_new_vector, state_vectors[index])
                    results[v_iter] = state_query + value_new * dot_key_query

                for v_iter in range_constexpr(_V_ITERS):
                    value_index = wave * _V_LANES + v_lane + v_iter * _V_TILE
                    for k_iter in range_constexpr(_K_ITERS):
                        k_base = k_lane * _VALUES_PER_THREAD + k_iter * _K_TILE
                        state_offset = (head * _HEAD_DIM + value_index) * _HEAD_DIM + k_base
                        bo.buffer_store(
                            state_vectors[v_iter * _K_ITERS + k_iter],
                            state_out_rsrc,
                            state_offset,
                            cache_modifier=CM_DEV,
                        )

                square_sum = fx.Float32(0.0)
                if k_lane == 0:
                    for v_iter in range_constexpr(_V_ITERS):
                        square_sum = square_sum + results[v_iter] * results[v_iter]
                square_sum = wave_sum(square_sum)
                if lane == 0:
                    lds_store(norm_sums, wave, square_sum)
                gpu.barrier()
                total_square = lds_load(norm_sums, 0)
                for source_wave in range_constexpr(1, _WAVES):
                    total_square = total_square + lds_load(norm_sums, source_wave)
                inverse_rms = rsq(total_square * fx.Float32(1.0 / _HEAD_DIM) + fx.Float32(EPS))
                if k_lane == 0:
                    for v_iter in range_constexpr(_V_ITERS):
                        value_index = wave * _V_LANES + v_lane + v_iter * _V_TILE
                        output_gate = get_input(sample, 3 * _PROJECTION + head * _HEAD_DIM + value_index)
                        gain = fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    norm_weight_rsrc,
                                    value_index,
                                    vec_width=1,
                                    dtype=T.bf16,
                                )
                            )
                        )
                        gated = results[v_iter] * inverse_rms * gain * sigmoid_batch([output_gate])[0]
                        lds_store(reduction, value_index, bf16_round(gated))
                gpu.barrier()
                if tid < _HEAD_DIM // 2:
                    value_index = tid * 2
                    put_norm_pair(
                        sample,
                        head * _HEAD_DIM + value_index,
                        lds_load(reduction, value_index),
                        lds_load(reduction, value_index + 1),
                    )
            else:
                if tid < _HEAD_DIM // 2:
                    put_norm_pair(
                        sample,
                        head * _HEAD_DIM + tid * 2,
                        fx.Float32(0.0),
                        fx.Float32(0.0),
                    )
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                store_i32(norm_ready_rsrc, sample * _HEADS + head, 1)

        # True MTP uses a producer/consumer KDA pipeline.  Per-token convolution
        # and recurrence CTAs advance disjoint snapshot chains, while grouped
        # norm CTAs remove the cross-split reduction from the recurrence path.
        if const_expr(mtp):
            conv_task = bid
            resident_producer_task = fx.Int32(-1)
            conv_tasks = samples * _HEADS
            while conv_task < conv_tasks:
                sample = conv_task // _HEADS
                head = conv_task % _HEADS
                input_slot, output_slot, acceptance_valid = state_slots(sample)
                valid_state = (
                    (input_slot >= 0)
                    & (output_slot >= 0)
                    & acceptance_valid
                )
                if valid_state:
                    prepare_kda_gate(sample, head)
                token = sample
                if const_expr(agentic_batch_size > 0):
                    token = sample % 8
                if const_expr(agentic_batch_size > 0):
                    if agentic_conv_writeback_requires_all(token):
                        if tid == 0:
                            for prior_token in range_constexpr(7):
                                load_i32(
                                    mtp_conv_ready_rsrc,
                                    (sample - 7 + prior_token) * _HEADS + head,
                                )
                        gpu.barrier()
                else:
                    if mtp_conv_waits_for_previous(
                        agentic_batch_size=agentic_batch_size,
                        token=token,
                    ):
                        if tid == 0:
                            load_i32(mtp_conv_ready_rsrc, (sample - 1) * _HEADS + head)
                        gpu.barrier()

                qkvg_base = (sample * _HEADS + head) * 4 * _HEAD_DIM

                def publish_mtp_component(component, values, first_row=0):
                    first_thread = component * (_HEAD_DIM // 2) + first_row // 2
                    pair_count = (_HEAD_DIM - first_row) // 2
                    if (tid >= first_thread) & (tid < first_thread + pair_count):
                        row = first_row + (tid - first_thread) * 2
                        store_raw_pair(
                            mtp_qkvg_rsrc,
                            (qkvg_base + component * _HEAD_DIM + row) // 2,
                            fx.Float32(fx.ptr_load(values + row)),
                            fx.Float32(fx.ptr_load(values + row + 1)),
                        )

                if valid_state:
                    conv_input_slot = input_slot
                    conv_output_slot = output_slot
                    if const_expr(agentic_batch_size > 0):
                        conv_input_slot = uniform(
                            bo.buffer_load(
                                indices_rsrc,
                                (sample // 8) * 8,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        conv_output_slot = conv_input_slot
                    conv_state_rsrc = rsrc(
                        conv_state
                        + fx.Int64(conv_input_slot) * fx.Int64(conv_slot_bytes)
                    )
                    conv_state_out_rsrc = rsrc(
                        conv_state
                        + fx.Int64(conv_output_slot) * fx.Int64(conv_slot_bytes)
                    )
                    prepare_kda_conv(sample, head, conv_state_rsrc, conv_state_out_rsrc)
                    publish_mtp_component(0, shared_query)
                    publish_mtp_component(1, shared_key)
                    publish_mtp_component(
                        2,
                        shared_value,
                        mtp_rows_per_split if samples <= 4 else 0,
                    )
                    publish_mtp_component(3, shared_gate)
                elif tid < 2 * _HEAD_DIM:
                    component = tid // (_HEAD_DIM // 2)
                    row = (tid % (_HEAD_DIM // 2)) * 2
                    store_raw_pair(
                        mtp_qkvg_rsrc,
                        (qkvg_base + component * _HEAD_DIM + row) // 2,
                        fx.Float32(0.0),
                        fx.Float32(0.0),
                    )
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid == 0:
                    store_i32(mtp_conv_ready_rsrc, sample * _HEADS + head, 1)
                resident_producer_task = conv_task
                conv_task = conv_task + _BLOCKS

            stamp(1)

            def run_mtp_recurrence(sample, head, value_split):
                local_qkvg = (
                    sample * _HEADS + head == resident_producer_task
                )
                if tid == 0:
                    if not local_qkvg:
                        load_i32(mtp_conv_ready_rsrc, sample * _HEADS + head)
                    lds_store(
                        norm_sums,
                        _WAVES,
                        fx.Float32(bo.buffer_load(a_log_rsrc, head, vec_width=1, dtype=T.f32)),
                    )
                    lds_store(norm_sums, _WAVES + 1, get_input(sample, 4 * _PROJECTION + head))
                gpu.barrier()

                input_slot, output_slot, acceptance_valid = state_slots(sample)
                valid_state = (
                    (input_slot >= 0)
                    & (output_slot >= 0)
                    & acceptance_valid
                )
                result = fx.Float32(0.0)
                value_index = value_split * mtp_rows_per_split + wave * mtp_v_lanes + lane // mtp_k_lanes
                if valid_state:
                    state_rsrc = rsrc(
                        recurrent_state
                        + fx.Int64(input_slot) * fx.Int64(state_slot_bytes)
                    )
                    state_out_rsrc = rsrc(
                        recurrent_state
                        + fx.Int64(output_slot) * fx.Int64(state_slot_bytes)
                    )
                    qkvg_base = (sample * _HEADS + head) * 4 * _HEAD_DIM
                    k_lane = lane % mtp_k_lanes
                    exp_a_log = exp(lds_load(norm_sums, _WAVES))
                    beta_logit = lds_load(norm_sums, _WAVES + 1)
                    beta_value = sigmoid_batch([beta_logit])[0]

                    query_vectors = [None] * mtp_k_iters
                    key_vectors = [None] * mtp_k_iters
                    decay_vectors = [None] * mtp_k_iters
                    query_square = fx.Float32(0.0)
                    key_square = fx.Float32(0.0)
                    for k_iter in range_constexpr(mtp_k_iters):
                        k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                        query_vector = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                        key_vector = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                        gate_vector = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                        if local_qkvg:
                            query_vector = fx.Vector(
                                fx.ptr_load(
                                    shared_query + k_base,
                                    result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                                )
                            ).to(fx.Float32)
                            key_vector = fx.Vector(
                                fx.ptr_load(
                                    shared_key + k_base,
                                    result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                                )
                            ).to(fx.Float32)
                            gate_vector = fx.Vector(
                                fx.ptr_load(
                                    shared_gate + k_base,
                                    result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                                )
                            ).to(fx.Float32)
                        else:
                            query_vector = fx.Vector(
                                bo.buffer_load(
                                    mtp_qkvg_rsrc,
                                    qkvg_base + k_base,
                                    vec_width=_VALUES_PER_THREAD,
                                    dtype=T.bf16,
                                    cache_modifier=CM_DEV,
                                )
                            ).to(fx.Float32)
                            key_vector = fx.Vector(
                                bo.buffer_load(
                                    mtp_qkvg_rsrc,
                                    qkvg_base + _HEAD_DIM + k_base,
                                    vec_width=_VALUES_PER_THREAD,
                                    dtype=T.bf16,
                                    cache_modifier=CM_DEV,
                                )
                            ).to(fx.Float32)
                            gate_vector = fx.Vector(
                                bo.buffer_load(
                                    mtp_qkvg_rsrc,
                                    qkvg_base + 3 * _HEAD_DIM + k_base,
                                    vec_width=_VALUES_PER_THREAD,
                                    dtype=T.bf16,
                                    cache_modifier=CM_DEV,
                                )
                            ).to(fx.Float32)
                        dt_vector = fx.Vector(
                            bo.buffer_load(
                                dt_bias_rsrc,
                                head * _HEAD_DIM + k_base,
                                vec_width=_VALUES_PER_THREAD,
                                dtype=T.bf16,
                            )
                        ).to(fx.Float32)
                        query_vectors[k_iter] = query_vector
                        key_vectors[k_iter] = key_vector
                        query_square = query_square + (query_vector * query_vector).reduce(fx.ReductionOp.ADD)
                        key_square = key_square + (key_vector * key_vector).reduce(fx.ReductionOp.ADD)
                        gate_sigmoid = sigmoid_batch(
                            [
                                exp_a_log * (gate_vector[item] + dt_vector[item])
                                for item in range_constexpr(_VALUES_PER_THREAD)
                            ]
                        )
                        decay_vectors[k_iter] = fx.Vector.from_elements(
                            [
                                exp(fx.Float32(_GATE_LOWER_BOUND) * gate_sigmoid[item])
                                for item in range_constexpr(_VALUES_PER_THREAD)
                            ],
                            fx.Float32,
                        )

                    def mtp_subgroup_sum(value):
                        for offset in ((8, 4, 2, 1) if mtp_k_lanes == 16 else (4, 2, 1)):
                            value = value + xshfl(value, offset)
                        return value

                    query_inverse_norm = rsq(mtp_subgroup_sum(query_square) + fx.Float32(1.0e-6))
                    key_inverse_norm = rsq(mtp_subgroup_sum(key_square) + fx.Float32(1.0e-6))
                    for k_iter in range_constexpr(mtp_k_iters):
                        query_vectors[k_iter] = query_vectors[k_iter] * fx.Vector.filled(
                            _VALUES_PER_THREAD,
                            query_inverse_norm * fx.Float32(_Q_SCALE),
                            fx.Float32,
                        )
                        key_vectors[k_iter] = key_vectors[k_iter] * fx.Vector.filled(
                            _VALUES_PER_THREAD,
                            key_inverse_norm,
                            fx.Float32,
                        )

                    dot_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                    for k_iter in range_constexpr(mtp_k_iters):
                        dot_parts = fx.math.fma(key_vectors[k_iter], query_vectors[k_iter], dot_parts)
                    dot_key_query = mtp_subgroup_sum(dot_parts.reduce(fx.ReductionOp.ADD))

                    state_vectors = [None] * mtp_k_iters
                    state_key_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                    state_query_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                    for k_iter in range_constexpr(mtp_k_iters):
                        k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                        state_offset = (head * _HEAD_DIM + value_index) * _HEAD_DIM + k_base
                        state_vector = fx.Vector(
                            bo.buffer_load(
                                state_rsrc,
                                state_offset,
                                vec_width=_VALUES_PER_THREAD,
                                dtype=T.f32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        state_vector = state_vector * decay_vectors[k_iter]
                        state_vectors[k_iter] = state_vector
                        state_key_parts = fx.math.fma(state_vector, key_vectors[k_iter], state_key_parts)
                        state_query_parts = fx.math.fma(state_vector, query_vectors[k_iter], state_query_parts)
                    state_key = mtp_subgroup_sum(state_key_parts.reduce(fx.ReductionOp.ADD))
                    state_query = mtp_subgroup_sum(state_query_parts.reduce(fx.ReductionOp.ADD))
                    value_input = fx.Float32(0.0)
                    if local_qkvg:
                        value_input = fx.Float32(fx.ptr_load(shared_value + value_index))
                    else:
                        value_input = fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    mtp_qkvg_rsrc,
                                    qkvg_base + 2 * _HEAD_DIM + value_index,
                                    vec_width=1,
                                    dtype=T.bf16,
                                    cache_modifier=CM_DEV,
                                )
                            )
                        )
                    value_new = (value_input - state_key) * beta_value
                    value_new_vector = fx.Vector.filled(_VALUES_PER_THREAD, value_new, fx.Float32)
                    for k_iter in range_constexpr(mtp_k_iters):
                        k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                        state_offset = (head * _HEAD_DIM + value_index) * _HEAD_DIM + k_base
                        state_vectors[k_iter] = fx.math.fma(
                            key_vectors[k_iter], value_new_vector, state_vectors[k_iter]
                        )
                        bo.buffer_store(
                            state_vectors[k_iter],
                            state_out_rsrc,
                            state_offset,
                            cache_modifier=CM_DEV,
                        )
                    result = state_query + value_new * dot_key_query

                k_lane = lane % mtp_k_lanes
                if k_lane == 0:
                    store_raw_f32(
                        norm_mailbox_rsrc,
                        sample * _PROJECTION + head * _HEAD_DIM + value_index,
                        result,
                    )
                square_sum = (k_lane == 0).select(result * result, fx.Float32(0.0))
                square_sum = wave_sum(square_sum)
                if lane == 0:
                    lds_store(norm_sums, wave, square_sum)
                gpu.barrier()
                partial_square = lds_load(norm_sums, 0)
                for source_wave in range_constexpr(1, _WAVES):
                    partial_square = partial_square + lds_load(norm_sums, source_wave)
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid == 0:
                    store_f32(
                        mtp_state_ready_rsrc,
                        (sample * _HEADS + head) * mtp_splits + value_split,
                        partial_square,
                    )

            mtp_tokens_per_cta = agentic_recurrence_tokens_per_cta(
                agentic_batch_size=agentic_batch_size
            )

            def run_mtp_pair(pair, head, value_split):
                sample_base = pair * mtp_tokens_per_cta
                token_base = sample_base
                request = fx.Int32(0)
                if const_expr(agentic_batch_size > 0):
                    token_base = sample_base % 8
                    request = sample_base // 8
                if tid == 0:
                    if token_base > 0:
                        load_f32(
                            mtp_state_ready_rsrc,
                            ((sample_base - 1) * _HEADS + head) * mtp_splits + value_split,
                        )
                    lds_store(
                        norm_sums,
                        _WAVES,
                        fx.Float32(bo.buffer_load(a_log_rsrc, head, vec_width=1, dtype=T.f32)),
                    )
                gpu.barrier()

                k_lane = lane % mtp_k_lanes
                value_index = value_split * mtp_rows_per_split + wave * mtp_v_lanes + lane // mtp_k_lanes
                request_sample = sample_base
                if const_expr(agentic_batch_size > 0):
                    request_sample = request * 8
                input_slot, _, acceptance_valid = state_slots(request_sample)
                state_vectors = [
                    fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32) for _ in range_constexpr(mtp_k_iters)
                ]
                if (input_slot >= 0) & acceptance_valid:
                    for k_iter in range_constexpr(mtp_k_iters):
                        k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                        state_offset = (head * _HEAD_DIM + value_index) * _HEAD_DIM + k_base
                        if token_base == 0:
                            state_rsrc = rsrc(
                                recurrent_state
                                + fx.Int64(input_slot) * fx.Int64(state_slot_bytes)
                            )
                            state_vectors[k_iter] = fx.Vector(
                                bo.buffer_load(
                                    state_rsrc,
                                    state_offset,
                                    vec_width=_VALUES_PER_THREAD,
                                    dtype=T.f16 if state_fp16 else T.f32,
                                    cache_modifier=CM_DEV,
                                )
                            ).to(fx.Float32)
                        else:
                            state_vectors[k_iter] = fx.Vector(
                                bo.buffer_load(
                                    mtp_state_handoff_rsrc,
                                    request * (_HEADS * _HEAD_DIM * _HEAD_DIM)
                                    + state_offset,
                                    vec_width=_VALUES_PER_THREAD,
                                    dtype=T.f32,
                                    cache_modifier=CM_DEV,
                                )
                            )

                exp_a_log = exp(lds_load(norm_sums, _WAVES))
                dt_vectors = [None] * mtp_k_iters
                for k_iter in range_constexpr(mtp_k_iters):
                    k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                    dt_vectors[k_iter] = fx.Vector(
                        bo.buffer_load(
                            dt_bias_rsrc,
                            head * _HEAD_DIM + k_base,
                            vec_width=_VALUES_PER_THREAD,
                            dtype=T.bf16,
                        )
                    ).to(fx.Float32)

                for token_offset, loop_args in range(
                    fx.Int32(0),
                    fx.Int32(mtp_tokens_per_cta),
                    fx.Int32(1),
                    init=state_vectors,
                ):
                    sample = sample_base + token_offset
                    local_qkvg = (
                        sample * _HEADS + head == resident_producer_task
                    )
                    if tid == 0:
                        if not local_qkvg:
                            load_i32(mtp_conv_ready_rsrc, sample * _HEADS + head)
                        lds_store(norm_sums, _WAVES + 1, get_input(sample, 4 * _PROJECTION + head))
                    gpu.barrier()

                    _, output_slot, token_acceptance_valid = state_slots(sample)
                    valid_state = (
                        (input_slot >= 0)
                        & (output_slot >= 0)
                        & token_acceptance_valid
                    )
                    next_state_vectors = [loop_args[k_iter] for k_iter in range_constexpr(mtp_k_iters)]
                    result = fx.Float32(0.0)
                    if valid_state:
                        state_out_rsrc = rsrc(
                            recurrent_state
                            + fx.Int64(output_slot)
                            * fx.Int64(state_slot_bytes)
                        )
                        qkvg_base = (sample * _HEADS + head) * 4 * _HEAD_DIM
                        beta_logit = lds_load(norm_sums, _WAVES + 1)
                        beta_value = sigmoid_batch([beta_logit])[0]

                        query_vectors = [None] * mtp_k_iters
                        key_vectors = [None] * mtp_k_iters
                        decay_vectors = [None] * mtp_k_iters
                        query_square = fx.Float32(0.0)
                        key_square = fx.Float32(0.0)
                        for k_iter in range_constexpr(mtp_k_iters):
                            k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                            query_vector = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                            key_vector = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                            gate_vector = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                            if local_qkvg:
                                query_vector = fx.Vector(
                                    fx.ptr_load(
                                        shared_query + k_base,
                                        result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                                    )
                                ).to(fx.Float32)
                                key_vector = fx.Vector(
                                    fx.ptr_load(
                                        shared_key + k_base,
                                        result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                                    )
                                ).to(fx.Float32)
                                gate_vector = fx.Vector(
                                    fx.ptr_load(
                                        shared_gate + k_base,
                                        result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                                    )
                                ).to(fx.Float32)
                            else:
                                query_vector = fx.Vector(
                                    bo.buffer_load(
                                        mtp_qkvg_rsrc,
                                        qkvg_base + k_base,
                                        vec_width=_VALUES_PER_THREAD,
                                        dtype=T.bf16,
                                        cache_modifier=CM_DEV,
                                    )
                                ).to(fx.Float32)
                                key_vector = fx.Vector(
                                    bo.buffer_load(
                                        mtp_qkvg_rsrc,
                                        qkvg_base + _HEAD_DIM + k_base,
                                        vec_width=_VALUES_PER_THREAD,
                                        dtype=T.bf16,
                                        cache_modifier=CM_DEV,
                                    )
                                ).to(fx.Float32)
                                gate_vector = fx.Vector(
                                    bo.buffer_load(
                                        mtp_qkvg_rsrc,
                                        qkvg_base + 3 * _HEAD_DIM + k_base,
                                        vec_width=_VALUES_PER_THREAD,
                                        dtype=T.bf16,
                                        cache_modifier=CM_DEV,
                                    )
                                ).to(fx.Float32)
                            query_vectors[k_iter] = query_vector
                            key_vectors[k_iter] = key_vector
                            query_square = query_square + (query_vector * query_vector).reduce(fx.ReductionOp.ADD)
                            key_square = key_square + (key_vector * key_vector).reduce(fx.ReductionOp.ADD)
                            gate_sigmoid = sigmoid_batch(
                                [
                                    exp_a_log * (gate_vector[item] + dt_vectors[k_iter][item])
                                    for item in range_constexpr(_VALUES_PER_THREAD)
                                ]
                            )
                            decay_vectors[k_iter] = fx.Vector.from_elements(
                                [
                                    exp(fx.Float32(_GATE_LOWER_BOUND) * gate_sigmoid[item])
                                    for item in range_constexpr(_VALUES_PER_THREAD)
                                ],
                                fx.Float32,
                            )

                        def mtp_pair_subgroup_sum(value):
                            for offset in (8, 4, 2, 1):
                                value = value + xshfl(value, offset)
                            return value

                        query_inverse_norm = rsq(mtp_pair_subgroup_sum(query_square) + fx.Float32(1.0e-6))
                        key_inverse_norm = rsq(mtp_pair_subgroup_sum(key_square) + fx.Float32(1.0e-6))
                        for k_iter in range_constexpr(mtp_k_iters):
                            query_vectors[k_iter] = query_vectors[k_iter] * fx.Vector.filled(
                                _VALUES_PER_THREAD,
                                query_inverse_norm * fx.Float32(_Q_SCALE),
                                fx.Float32,
                            )
                            key_vectors[k_iter] = key_vectors[k_iter] * fx.Vector.filled(
                                _VALUES_PER_THREAD,
                                key_inverse_norm,
                                fx.Float32,
                            )

                        dot_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                        for k_iter in range_constexpr(mtp_k_iters):
                            dot_parts = fx.math.fma(key_vectors[k_iter], query_vectors[k_iter], dot_parts)
                        dot_key_query = mtp_pair_subgroup_sum(dot_parts.reduce(fx.ReductionOp.ADD))

                        state_key_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                        state_query_parts = fx.Vector.filled(_VALUES_PER_THREAD, 0.0, fx.Float32)
                        for k_iter in range_constexpr(mtp_k_iters):
                            next_state_vectors[k_iter] = loop_args[k_iter] * decay_vectors[k_iter]
                            state_key_parts = fx.math.fma(
                                next_state_vectors[k_iter],
                                key_vectors[k_iter],
                                state_key_parts,
                            )
                            state_query_parts = fx.math.fma(
                                next_state_vectors[k_iter],
                                query_vectors[k_iter],
                                state_query_parts,
                            )
                        state_key = mtp_pair_subgroup_sum(state_key_parts.reduce(fx.ReductionOp.ADD))
                        state_query = mtp_pair_subgroup_sum(state_query_parts.reduce(fx.ReductionOp.ADD))
                        value_input = fx.Float32(0.0)
                        if local_qkvg:
                            value_input = fx.Float32(fx.ptr_load(shared_value + value_index))
                        else:
                            value_input = fx.Float32(
                                fx.BFloat16(
                                    bo.buffer_load(
                                        mtp_qkvg_rsrc,
                                        qkvg_base + 2 * _HEAD_DIM + value_index,
                                        vec_width=1,
                                        dtype=T.bf16,
                                        cache_modifier=CM_DEV,
                                    )
                                )
                            )
                        value_new = (value_input - state_key) * beta_value
                        value_new_vector = fx.Vector.filled(_VALUES_PER_THREAD, value_new, fx.Float32)
                        for k_iter in range_constexpr(mtp_k_iters):
                            k_base = k_lane * _VALUES_PER_THREAD + k_iter * mtp_k_tile
                            state_offset = (head * _HEAD_DIM + value_index) * _HEAD_DIM + k_base
                            next_state_vectors[k_iter] = fx.math.fma(
                                key_vectors[k_iter],
                                value_new_vector,
                                next_state_vectors[k_iter],
                            )
                            state_value = next_state_vectors[k_iter]
                            if token_offset == mtp_tokens_per_cta - 1:
                                bo.buffer_store(
                                    state_value,
                                    mtp_state_handoff_rsrc,
                                    request * (_HEADS * _HEAD_DIM * _HEAD_DIM)
                                    + state_offset,
                                    cache_modifier=CM_DEV,
                                )
                            if const_expr(state_fp16):
                                state_value = state_value.to(fx.Float16)
                            bo.buffer_store(
                                state_value,
                                state_out_rsrc,
                                state_offset,
                                cache_modifier=CM_DEV,
                            )
                        result = state_query + value_new * dot_key_query

                    if k_lane == 0:
                        store_raw_f32(
                            norm_mailbox_rsrc,
                            sample * _PROJECTION + head * _HEAD_DIM + value_index,
                            result,
                        )
                    square_sum = (k_lane == 0).select(result * result, fx.Float32(0.0))
                    square_sum = wave_sum(square_sum)
                    if lane == 0:
                        lds_store(norm_sums, wave, square_sum)
                    gpu.barrier()
                    partial_square = lds_load(norm_sums, 0)
                    for source_wave in range_constexpr(1, _WAVES):
                        partial_square = partial_square + lds_load(norm_sums, source_wave)
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    if tid == 0:
                        store_f32(
                            mtp_state_ready_rsrc,
                            (sample * _HEADS + head) * mtp_splits + value_split,
                            partial_square,
                        )

                    _ = yield next_state_vectors

            mtp_recurrence_tasks = samples * _HEADS
            if const_expr(samples <= 4):
                if bid < mtp_splits * mtp_recurrence_tasks:
                    value_split = bid // mtp_recurrence_tasks
                    mtp_recurrence_task = bid % mtp_recurrence_tasks
                    sample = mtp_recurrence_task // _HEADS
                    head = mtp_recurrence_task % _HEADS
                    if tid == 0:
                        if sample > 0:
                            load_f32(
                                mtp_state_ready_rsrc,
                                ((sample - 1) * _HEADS + head) * mtp_splits + value_split,
                            )
                    gpu.barrier()
                    run_mtp_recurrence(sample, head, value_split)
            else:
                mtp_pair_tasks = (samples // mtp_tokens_per_cta) * _HEADS
                mtp_task = bid
                while mtp_task < mtp_splits * mtp_pair_tasks:
                    if const_expr(agentic_batch_size > 0):
                        conv_sample = mtp_task // _HEADS
                        pair = conv_sample // mtp_tokens_per_cta
                        head = mtp_task % _HEADS
                        value_split = conv_sample % mtp_tokens_per_cta
                    else:
                        primary_task = mtp_task < mtp_recurrence_tasks
                        conv_sample = mtp_task // _HEADS
                        local_pair = conv_sample // 2
                        local_split = conv_sample % 2
                        remote_task = mtp_task - mtp_recurrence_tasks
                        remote_split = 2 + remote_task // mtp_pair_tasks
                        remote_pair_task = remote_task % mtp_pair_tasks
                        pair = primary_task.select(
                            local_pair,
                            remote_pair_task // _HEADS,
                        )
                        head = primary_task.select(
                            mtp_task % _HEADS,
                            remote_pair_task % _HEADS,
                        )
                        value_split = primary_task.select(
                            local_split,
                            remote_split,
                        )
                    run_mtp_pair(pair, head, value_split)
                    mtp_task = mtp_task + _BLOCKS

            mtp_norm_head_groups = _HEADS // 4
            mtp_norm_tasks = samples * mtp_norm_head_groups
            mtp_norm_task = bid
            if mtp_norm_task < mtp_norm_tasks:
                sample = mtp_norm_task // mtp_norm_head_groups
                head_base = (mtp_norm_task % mtp_norm_head_groups) * 4
                if tid < 4:
                    head = head_base + tid
                    total_square = load_f32(
                        mtp_state_ready_rsrc,
                        (sample * _HEADS + head) * mtp_splits,
                    )
                    for source_split in range_constexpr(1, mtp_splits):
                        total_square = total_square + load_f32(
                            mtp_state_ready_rsrc,
                            (sample * _HEADS + head) * mtp_splits + source_split,
                        )
                    lds_store(norm_sums, tid, total_square)
                gpu.barrier()
                local_head = tid // _HEAD_DIM
                head = head_base + local_head
                value_index = tid % _HEAD_DIM
                inverse_rms = rsq(lds_load(norm_sums, local_head) * fx.Float32(1.0 / _HEAD_DIM) + fx.Float32(EPS))
                raw_result = load_raw_f32(
                    norm_mailbox_rsrc,
                    sample * _PROJECTION + head * _HEAD_DIM + value_index,
                )
                output_gate = get_input(
                    sample,
                    3 * _PROJECTION + head * _HEAD_DIM + value_index,
                )
                gain = fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(
                            norm_weight_rsrc,
                            value_index,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                )
                gated = raw_result * inverse_rms * gain * sigmoid_batch([output_gate])[0]
                gated_neighbor = xshfl(gated, 1)
                if value_index % 2 == 0:
                    packed_pair = (sample * _PROJECTION + head * _HEAD_DIM + value_index) // 2
                    store_raw_pair(norm_packed_rsrc, packed_pair, gated, gated_neighbor)
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid < 4:
                    store_i32(
                        mtp_norm_ready_rsrc,
                        sample * _HEADS + head_base + tid,
                        1,
                    )

        # Stage 3: BF16 1536 -> 7168 output projection followed by the tagged
        # TP8 reduction.  Each wave again owns a complete-K 16-row group.
        output_row_tasks = _OUTPUT_TASKS
        output_tasks = output_sample_groups * output_row_tasks
        output_task = bid
        while output_task < output_tasks:
            if const_expr(output_sample_groups == 1):
                output_row_task = output_task
                sample_base = 0
            else:
                sample_group = output_task // output_row_tasks
                output_row_task = output_task % output_row_tasks
                sample_base = sample_group * output_staged_samples
            stamp(2)
            stage_norm(sample_base, output_staged_samples)
            gpu.barrier()
            output_accumulator = bf16_mfma(
                output_weight_rsrc,
                output_row_task * _OUTPUT_ROW_GROUPS,
                _PROJECTION,
                _OUTPUT_ROW_GROUPS,
                _OUTPUT_SPLIT_WAVES,
                6,
                output_staged_samples,
            )

            def emit_output(local_row, sample, value_low, value_high):
                lds_store(
                    output_values,
                    ((sample - sample_base) * _OUTPUT_ROW_TILE + local_row) // 2,
                    bf16_pair(value_low, value_high),
                )

            publish_mfma_pairs(
                output_accumulator,
                _OUTPUT_ROW_TILE,
                _OUTPUT_SPLIT_WAVES,
                emit_output,
                sample_base,
                output_staged_samples,
            )

            pair_count = output_staged_samples * _OUTPUT_ROW_TILE // 2
            send_rounds = (pair_count + _WAVE_SIZE - 1) // _WAVE_SIZE
            peer_rounds = (npes + _WAVES - 1) // _WAVES
            for peer_round in range_constexpr(peer_rounds):
                peer = wave + peer_round * _WAVES
                if peer < npes:
                    peer_words = fx.Vector(bo.buffer_load(rsrc(peers), peer * 2, vec_width=2, dtype=T.i32))
                    peer_address = (fx.Int64(uniform(peer_words[1])) << 32) | fx.Int64(
                        fx.Uint32(uniform(peer_words[0]))
                    )
                    peer_rsrc = rsrc(peer_address + symmetric_base)
                    for send_round in range_constexpr(send_rounds):
                        local_pair = lane + send_round * _WAVE_SIZE
                        if local_pair < pair_count:
                            local_sample = local_pair // (_OUTPUT_ROW_TILE // 2)
                            sample = sample_base + local_sample
                            row_pair = local_pair % (_OUTPUT_ROW_TILE // 2)
                            local_row = row_pair * 2
                            row = output_row_task * _OUTPUT_ROW_TILE + local_row
                            packed = lds_load(
                                output_values,
                                (local_sample * _OUTPUT_ROW_TILE + local_row) // 2,
                            ).bitcast(fx.Int32)
                            global_pair = (sample * _HIDDEN + row) // 2
                            mailbox = rank * max_pairs + global_pair
                            bo.buffer_store(
                                fx.Vector.from_elements([packed, tag], fx.Int32),
                                peer_rsrc,
                                mailbox * 2,
                                cache_modifier=CM_SYS,
                            )
            gpu.barrier()

            local_rsrc = rsrc(symmetric + symmetric_base)
            pair_rounds = (pair_count + _THREADS - 1) // _THREADS
            for pair_round in range_constexpr(pair_rounds):
                local_pair = tid + pair_round * _THREADS
                if local_pair < pair_count:
                    local_sample = local_pair // (_OUTPUT_ROW_TILE // 2)
                    sample = sample_base + local_sample
                    row_pair = local_pair % (_OUTPUT_ROW_TILE // 2)
                    row = output_row_task * _OUTPUT_ROW_TILE + row_pair * 2
                    global_pair = (sample * _HIDDEN + row) // 2

                    def load_peers():
                        words = []
                        for source_rank in range_constexpr(npes):
                            mailbox = source_rank * max_pairs + global_pair
                            value_tag = fx.Vector(
                                bo.buffer_load(
                                    local_rsrc,
                                    mailbox * 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                            words += [value_tag[0], value_tag[1]]
                        return fx.Vector.from_elements(words, fx.Int32)

                    peer_values = load_peers()
                    pending = peer_values[1] != tag
                    for source_rank in range_constexpr(1, npes):
                        pending = pending | (peer_values[source_rank * 2 + 1] != tag)
                    while pending:
                        rocdl.s_nop(0)
                        peer_values = load_peers()
                        pending = peer_values[1] != tag
                        for source_rank in range_constexpr(1, npes):
                            pending = pending | (peer_values[source_rank * 2 + 1] != tag)

                    sum_low = fx.Float32(0.0)
                    sum_high = fx.Float32(0.0)
                    for source_rank in range_constexpr(npes):
                        packed = peer_values[source_rank * 2]
                        sum_low = sum_low + (packed << 16).bitcast(fx.Float32)
                        sum_high = sum_high + (packed & fx.Int32(-65536)).bitcast(fx.Float32)
                    packed_sum = (
                        fx.Vector.from_elements([sum_low, sum_high], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    )
                    bo.buffer_store(packed_sum, output_rsrc, global_pair, cache_modifier=CM_DEV)
                    if const_expr(fuse_attn_res):
                        store_pair(
                            attention_mailbox_rsrc,
                            global_pair,
                            sum_low,
                            sum_high,
                        )
            output_task = output_task + _BLOCKS
        stamp(3)

        # Stage 4: post-attention AttnRes and MXFP8 activation quantization for
        # the latent/router/shared projection stage that follows this kernel.
        if const_expr(fuse_attn_res):
            if bid < samples * _ATTN_RES_CTAS:
                if const_expr(block_write_idx >= 0):
                    post_prefix = output
                    post_delta = output
                    post_has_delta = False
                else:
                    post_prefix = hidden_states
                    post_delta = output
                    post_has_delta = True
                run_attn_res_chunk(
                    bid // _ATTN_RES_CTAS,
                    bid % _ATTN_RES_CTAS,
                    post_prefix,
                    post_delta,
                    mlp_res_norm,
                    mlp_res_qk,
                    post_norm,
                    updated_prefix,
                    moe_input,
                    moe_mailbox_rsrc,
                    moe_ready_rsrc,
                    post_stats_rsrc,
                    attn_res_blocks + int(block_write_idx >= 0),
                    post_has_delta,
                    block_write_idx >= 0,
                    block_write_idx < 0,
                    -1,
                    True,
                )
        stamp(4)

        if const_expr(dense_ffn):
            # Layer 0 uses the checkpoint's BF16 dense SiTU FFN.  Keep its
            # gate/up, activation, down projection, TP reduction and residual
            # update in this same application launch.
            dense_row_groups = 2
            dense_split_waves = 4
            dense_row_tile = dense_row_groups * 16
            dense_row_tasks = (2 * _DENSE_INTER) // dense_row_tile
            dense_tasks = sample_groups * dense_row_tasks
            dense_task = bid
            while dense_task < dense_tasks:
                if const_expr(sample_groups == 1):
                    row_task = dense_task
                    sample_base = 0
                else:
                    sample_group = dense_task // dense_row_tasks
                    row_task = dense_task % dense_row_tasks
                    sample_base = sample_group * staged_samples
                stage_moe_hidden(sample_base, staged_samples)
                gpu.barrier()
                row_base = row_task * dense_row_groups
                dense_accumulator = bf16_mfma(
                    rsrc(packed_shared_up),
                    row_base,
                    _HIDDEN,
                    dense_row_groups,
                    dense_split_waves,
                    14,
                    staged_samples,
                )

                def emit_dense_up(local_row, sample, value_low, value_high):
                    row = row_task * dense_row_tile + local_row
                    store_raw_pair(
                        shared_gu_mailbox_rsrc,
                        (sample * (2 * _DENSE_INTER) + row) // 2,
                        value_low,
                        value_high,
                    )

                publish_mfma_pairs(
                    dense_accumulator,
                    dense_row_tile,
                    dense_split_waves,
                    emit_dense_up,
                    sample_base,
                    staged_samples,
                )
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid < dense_row_groups:
                    for local_sample in range_constexpr(staged_samples):
                        store_i32(
                            shared_gu_ready_rsrc,
                            (sample_base + local_sample)
                            * ((2 * _DENSE_INTER) // 16)
                            + row_task * dense_row_groups
                            + tid,
                            1,
                        )
                dense_task = dense_task + _BLOCKS

            if bid < samples:
                dense_pairs = _DENSE_INTER // 2
                dense_tiles = (2 * _DENSE_INTER) // 16
                for ready_round in range_constexpr(
                    (dense_tiles + _THREADS - 1) // _THREADS
                ):
                    ready = tid + ready_round * _THREADS
                    if ready < dense_tiles:
                        load_i32(
                            shared_gu_ready_rsrc,
                            bid * dense_tiles + ready,
                        )
                gpu.barrier()
                for pair_round in range_constexpr(
                    (dense_pairs + _THREADS - 1) // _THREADS
                ):
                    pair_in_row = tid + pair_round * _THREADS
                    if pair_in_row < dense_pairs:
                        gate_word = load_raw_pair(
                            shared_gu_mailbox_rsrc,
                            (bid * (2 * _DENSE_INTER)) // 2 + pair_in_row,
                        )
                        up_word = load_raw_pair(
                            shared_gu_mailbox_rsrc,
                            (
                                bid * (2 * _DENSE_INTER)
                                + _DENSE_INTER
                            )
                            // 2
                            + pair_in_row,
                        )
                        gate_values = fx.Vector.from_elements(
                            [gate_word], fx.Int32
                        ).bitcast(fx.BFloat16).to(fx.Float32)
                        up_values = fx.Vector.from_elements(
                            [up_word], fx.Int32
                        ).bitcast(fx.BFloat16).to(fx.Float32)
                        mids = []
                        for item in range_constexpr(2):
                            gate_value = gate_values[item]
                            up_value = up_values[item]
                            gate_tanh = fx.Float32(2.0) * rcp(
                                fx.Float32(1.0)
                                + exp(fx.Float32(-0.5) * gate_value)
                            ) - fx.Float32(1.0)
                            gate_sigmoid = rcp(
                                fx.Float32(1.0) + exp(-gate_value)
                            )
                            up_tanh = fx.Float32(2.0) * rcp(
                                fx.Float32(1.0)
                                + exp(fx.Float32(-0.08) * up_value)
                            ) - fx.Float32(1.0)
                            mids.append(
                                fx.Float32(4.0)
                                * gate_tanh
                                * gate_sigmoid
                                * fx.Float32(25.0)
                                * up_tanh
                            )
                        store_pair(
                            shared_mid_mailbox_rsrc,
                            bid * dense_pairs + pair_in_row,
                            mids[0],
                            mids[1],
                        )
            stamp(5)

            dense_output_row_groups = 4
            dense_output_split_waves = 2
            dense_output_row_tile = dense_output_row_groups * 16
            dense_output_tasks = samples * (
                _HIDDEN // dense_output_row_tile
            )
            dense_output_task = bid
            while dense_output_task < dense_output_tasks:
                sample = dense_output_task // (
                    _HIDDEN // dense_output_row_tile
                )
                row_task = dense_output_task % (
                    _HIDDEN // dense_output_row_tile
                )
                dense_pairs = _DENSE_INTER // 2
                for load_round in range_constexpr(
                    (dense_pairs + _THREADS - 1) // _THREADS
                ):
                    pair = tid + load_round * _THREADS
                    if pair < dense_pairs:
                        packed = load_pair(
                            shared_mid_mailbox_rsrc,
                            sample * dense_pairs + pair,
                        )
                        lds_store(x, pair, packed.bitcast(fx.Float32))
                gpu.barrier()
                dense_down_accumulator = bf16_mfma(
                    rsrc(packed_shared_down),
                    row_task * dense_output_row_groups,
                    _DENSE_INTER,
                    dense_output_row_groups,
                    dense_output_split_waves,
                    11,
                    1,
                )
                fx.ptr_store(
                    fx.Vector.from_elements(
                        dense_down_accumulator, fx.Float32
                    ),
                    reduction + (wave * _WAVE_SIZE + lane) * 4,
                )
                gpu.barrier()
                if tid < dense_output_row_tile // 2:
                    local_row = tid * 2
                    values = []
                    for pair_element in range_constexpr(2):
                        row = local_row + pair_element
                        source_lane = 16 * (row % 16 // 4)
                        first_wave = (
                            row // 16
                        ) * dense_output_split_waves
                        value = fx.Float32(0.0)
                        for source_offset in range_constexpr(
                            dense_output_split_waves
                        ):
                            source_index = (
                                (
                                    first_wave + source_offset
                                )
                                * _WAVE_SIZE
                                + source_lane
                            ) * 4 + row % 4
                            value = value + lds_load(
                                reduction, source_index
                            )
                        values.append(value)
                    lds_store(
                        output_values,
                        tid,
                        bf16_pair(values[0], values[1]),
                    )
                gpu.barrier()
                pair_base = (
                    sample * (_HIDDEN // 2)
                    + row_task * (dense_output_row_tile // 2)
                )

                def emit_dense_final(
                    local_pair, value_low, value_high
                ):
                    residual_word = fx.Int32(
                        bo.buffer_load(
                            rsrc(updated_prefix),
                            pair_base + local_pair,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    residual_low = (
                        residual_word << 16
                    ).bitcast(fx.Float32)
                    residual_high = (
                        residual_word & fx.Int32(-65536)
                    ).bitcast(fx.Float32)
                    bo.buffer_store(
                        bf16_pair(
                            residual_low + value_low,
                            residual_high + value_high,
                        ).bitcast(fx.Int32),
                        rsrc(final_output),
                        pair_base + local_pair,
                        cache_modifier=CM_DEV,
                    )

                moe_peer_reduce(
                    dense_output_row_tile // 2,
                    pair_base,
                    output_values,
                    1,
                    emit_dense_final,
                )
                dense_output_task = dense_output_task + _BLOCKS
            stamp(9)

        if const_expr(fuse_moe and not dense_ffn):
            # Stage 5: the BF16 router and the two production MXFP8 dense
            # projections.  All consume the post-AttnRes result; the dense
            # branches directly reuse the quantized activation emitted there.
            router_row_tasks = _N_EXPERTS // _ROUTER_ROW_TILE
            router_tasks = sample_groups * router_row_tasks
            latent_tiles = _ROUTED_HIDDEN // 16
            shared_tiles = (2 * _SHARED_INTER) // 16
            latent_blocks = (latent_tiles + latent_projection_waves - 1) // latent_projection_waves
            shared_blocks = (shared_tiles + shared_projection_waves - 1) // shared_projection_waves
            latent_tasks = sample_groups * latent_blocks
            shared_tasks = sample_groups * shared_blocks
            projection_tasks = router_tasks + latent_tasks + shared_tasks
            projection_task = bid
            while projection_task < projection_tasks:
                if projection_task < router_tasks:
                    if const_expr(sample_groups == 1):
                        router_row_task = projection_task
                        sample_base = 0
                    else:
                        sample_group = projection_task // router_row_tasks
                        router_row_task = projection_task % router_row_tasks
                        sample_base = sample_group * staged_samples
                    stage_moe_hidden(sample_base, staged_samples)
                    gpu.barrier()
                    row_base = router_row_task * _ROUTER_ROW_GROUPS
                    projection_accumulator = bf16_mfma(
                        rsrc(packed_router_weight),
                        row_base,
                        _HIDDEN,
                        _ROUTER_ROW_GROUPS,
                        _ROUTER_SPLIT_WAVES,
                        14,
                        staged_samples,
                    )

                    def emit_router(local_row, sample, value_low, value_high):
                        row = router_row_task * _ROUTER_ROW_TILE + local_row
                        logit_low = bf16_round(value_low)
                        logit_high = bf16_round(value_high)
                        store_raw_f32(
                            router_mailbox_rsrc,
                            sample * _N_EXPERTS + row,
                            rcp(fx.Float32(1.0) + exp(-logit_low)),
                        )
                        store_raw_f32(
                            router_mailbox_rsrc,
                            sample * _N_EXPERTS + row + 1,
                            rcp(fx.Float32(1.0) + exp(-logit_high)),
                        )

                    publish_mfma_pairs(
                        projection_accumulator,
                        _ROUTER_ROW_TILE,
                        _ROUTER_SPLIT_WAVES,
                        emit_router,
                        sample_base,
                        staged_samples,
                    )
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    if tid == 0:
                        for local_sample in range_constexpr(staged_samples):
                            store_i32(
                                router_ready_rsrc,
                                (sample_base + local_sample) * router_row_tasks + router_row_task,
                                1,
                            )
                elif projection_task < router_tasks + latent_tasks:
                    latent_task = projection_task - router_tasks
                    if const_expr(sample_groups == 1):
                        latent_block = latent_task
                        sample_base = 0
                    else:
                        sample_group = latent_task // latent_blocks
                        latent_block = latent_task % latent_blocks
                        sample_base = sample_group * staged_samples
                    stage_mxfp8_hidden(sample_base, staged_samples)
                    gpu.barrier()
                    latent_tile = latent_block * latent_projection_waves + wave
                    if wave < latent_projection_waves:
                        projection_values = mxfp8_scaled_mfma(
                            rsrc(packed_latent_weight),
                            rsrc(latent_weight_scale),
                            fx.min(latent_tile, latent_tiles - 1),
                            sample_base,
                            staged_samples,
                        )
                        if latent_tile < latent_tiles:
                            publish_raw_mxfp8_tile(
                                projection_values,
                                latent_tile,
                                _ROUTED_HIDDEN,
                                latent_mailbox_rsrc,
                                latent_ready_rsrc,
                                sample_base,
                                staged_samples,
                            )
                else:
                    shared_task = projection_task - router_tasks - latent_tasks
                    if const_expr(sample_groups == 1):
                        shared_block = shared_task
                        sample_base = 0
                    else:
                        sample_group = shared_task // shared_blocks
                        shared_block = shared_task % shared_blocks
                        sample_base = sample_group * staged_samples
                    stage_mxfp8_hidden(sample_base, staged_samples)
                    gpu.barrier()
                    shared_tile = shared_block * shared_projection_waves + wave
                    if wave < shared_projection_waves:
                        projection_values = mxfp8_scaled_mfma(
                            rsrc(packed_shared_up),
                            rsrc(shared_up_scale),
                            fx.min(shared_tile, shared_tiles - 1),
                            sample_base,
                            staged_samples,
                        )
                        if shared_tile < shared_tiles:
                            publish_raw_mxfp8_tile(
                                projection_values,
                                shared_tile,
                                2 * _SHARED_INTER,
                                shared_gu_mailbox_rsrc,
                                shared_gu_ready_rsrc,
                                sample_base,
                                staged_samples,
                            )
                projection_task = projection_task + _BLOCKS

            # One selector CTA handles four samples, one sample per wave.
            selector_tasks = (samples + _WAVES - 1) // _WAVES
            if bid < selector_tasks:
                sample = bid * _WAVES + wave
                if sample < samples:
                    if lane < router_row_tasks:
                        load_i32(
                            router_ready_rsrc,
                            sample * router_row_tasks + lane,
                        )
                gpu.barrier()
                if sample < samples:
                    scores = []
                    biases = []
                    for value_index in range_constexpr(_N_EXPERTS // _WAVE_SIZE):
                        expert = lane + value_index * _WAVE_SIZE
                        scores.append(
                            load_raw_f32(
                                router_mailbox_rsrc,
                                sample * _N_EXPERTS + expert,
                            )
                        )
                        biases.append(
                            fx.Float32(
                                fx.BFloat16(
                                    bo.buffer_load(
                                        rsrc(correction_bias),
                                        expert,
                                        vec_width=1,
                                        dtype=T.bf16,
                                    )
                                )
                            )
                        )
                    corrected = [scores[index] + biases[index] for index in range_constexpr(_N_EXPERTS // _WAVE_SIZE)]
                    selected_sum = fx.Float32(0.0)
                    for selected_index in range_constexpr(_TOP_K):
                        best_score = corrected[0]
                        best_id = fx.Int32(lane)
                        for value_index in range_constexpr(1, _N_EXPERTS // _WAVE_SIZE):
                            candidate_id = fx.Int32(lane + value_index * _WAVE_SIZE)
                            candidate_score = corrected[value_index]
                            take = (candidate_score > best_score) | (
                                (ArithValue(candidate_score) == ArithValue(best_score)) & (candidate_id < best_id)
                            )
                            best_score = take.select(candidate_score, best_score)
                            best_id = take.select(candidate_id, best_id)
                        for offset in (32, 16, 8, 4, 2, 1):
                            peer_score = xshfl(best_score, offset)
                            peer_id = xshfl(best_id, offset)
                            take = (peer_score > best_score) | (
                                (ArithValue(peer_score) == ArithValue(best_score)) & (peer_id < best_id)
                            )
                            best_score = take.select(peer_score, best_score)
                            best_id = take.select(peer_id, best_id)
                        best_bias = fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    rsrc(correction_bias),
                                    best_id,
                                    vec_width=1,
                                    dtype=T.bf16,
                                )
                            )
                        )
                        best_raw = best_score - best_bias
                        selected_sum = selected_sum + best_raw
                        if lane == 0:
                            route = sample * _TOP_K + selected_index
                            store_i32(selection_id_rsrc, route, best_id)
                            lds_store(output_values, wave * _TOP_K + selected_index, best_raw)
                        for value_index in range_constexpr(_N_EXPERTS // _WAVE_SIZE):
                            expert = fx.Int32(lane + value_index * _WAVE_SIZE)
                            corrected[value_index] = (expert == best_id).select(
                                fx.Float32(float("-inf")), corrected[value_index]
                            )
                    if lane == 0:
                        inverse_sum = rcp(selected_sum)
                        for selected_index in range_constexpr(_TOP_K):
                            route = sample * _TOP_K + selected_index
                            store_f32(
                                selection_weight_rsrc,
                                route,
                                lds_load(output_values, wave * _TOP_K + selected_index) * inverse_sum,
                            )

            # Shared SiTU activation is cheap enough to run as one CTA/sample.
            if bid < samples:
                shared_pairs = _SHARED_INTER // 2
                shared_tiles = (2 * _SHARED_INTER) // 16
                if tid < shared_tiles:
                    load_i32(
                        shared_gu_ready_rsrc,
                        bid * shared_tiles + tid,
                    )
                gpu.barrier()
                for pair_round in range_constexpr((shared_pairs + _THREADS - 1) // _THREADS):
                    pair_in_row = tid + pair_round * _THREADS
                    if pair_in_row < shared_pairs:
                        gate_word = load_raw_pair(
                            shared_gu_mailbox_rsrc,
                            (bid * (2 * _SHARED_INTER)) // 2 + pair_in_row,
                        )
                        up_word = load_raw_pair(
                            shared_gu_mailbox_rsrc,
                            (bid * (2 * _SHARED_INTER) + _SHARED_INTER) // 2 + pair_in_row,
                        )
                        gate_values = fx.Vector.from_elements([gate_word], fx.Int32).bitcast(fx.BFloat16).to(fx.Float32)
                        up_values = fx.Vector.from_elements([up_word], fx.Int32).bitcast(fx.BFloat16).to(fx.Float32)
                        mids = []
                        for item in range_constexpr(2):
                            gate_value = gate_values[item]
                            up_value = up_values[item]
                            gate_tanh = fx.Float32(2.0) * rcp(
                                fx.Float32(1.0) + exp(fx.Float32(-0.5) * gate_value)
                            ) - fx.Float32(1.0)
                            gate_sigmoid = rcp(fx.Float32(1.0) + exp(-gate_value))
                            up_tanh = fx.Float32(2.0) * rcp(
                                fx.Float32(1.0) + exp(fx.Float32(-0.08) * up_value)
                            ) - fx.Float32(1.0)
                            mids.append(fx.Float32(4.0) * gate_tanh * gate_sigmoid * fx.Float32(25.0) * up_tanh)
                        store_pair(
                            shared_mid_mailbox_rsrc,
                            bid * shared_pairs + pair_in_row,
                            mids[0],
                            mids[1],
                        )
            stamp(5)

            # Stage 6: direct top-16 routed expert up/gate.  Avoid sorting at
            # decode scale: each task owns one 16-row intermediate tile for one
            # selected route and reads the selected expert directly.
            up_tiles = _INTER // 16
            up_tasks = samples * _TOP_K * up_tiles
            up_task = bid
            while up_task < up_tasks:
                sample = up_task // (_TOP_K * up_tiles)
                route_in_sample = (up_task // up_tiles) % _TOP_K
                row_group = up_task % up_tiles
                route = sample * _TOP_K + route_in_sample
                expert = uniform(load_i32(selection_id_rsrc, route))
                expert_weight_bytes = 2 * _INTER * (_ROUTED_HIDDEN // 2)
                up_weight_rsrc = rsrc(
                    packed_expert_up
                    + fx.Int64(expert) * fx.Int64(expert_weight_bytes)
                )
                up_scale_rsrc = rsrc(
                    expert_up_scale
                    + fx.Int64(expert) * fx.Int64(up_expert_scale_bytes)
                )
                stage_raw_vector(
                    latent_mailbox_rsrc,
                    latent_ready_rsrc,
                    sample * (_ROUTED_HIDDEN // 16),
                    _ROUTED_HIDDEN // 16,
                    sample * (_ROUTED_HIDDEN // 2),
                    _ROUTED_HIDDEN // 2,
                )
                gpu.barrier()

                accumulator = [fx.Float32(0.0) for _ in range(4)]
                split = wave % 4
                selected_row_group = (wave < 4).select(
                    row_group,
                    row_group + _INTER // 16,
                )
                chunks_per_wave = (_ROUTED_HIDDEN // 128) // 4
                for local_chunk in range_constexpr(chunks_per_wave):
                    k_chunk = split * chunks_per_wave + local_chunk
                    fragment = mxfp4_fragment(
                        up_weight_rsrc,
                        up_scale_rsrc,
                        selected_row_group,
                        k_chunk,
                        _ROUTED_HIDDEN,
                    )
                    accumulator = mxfp4_apply(
                        accumulator,
                        fragment,
                        k_chunk * 64,
                    )

                fx.ptr_store(
                    fx.Vector.from_elements(accumulator, fx.Float32),
                    reduction + (wave * _WAVE_SIZE + lane) * 4,
                )
                gpu.barrier()
                if tid < 16 // 2:
                    local_row = tid * 2
                    activated = []
                    for pair_element in range_constexpr(2):
                        row = local_row + pair_element
                        source_lane = 16 * (row // 4)
                        gate_value = fx.Float32(0.0)
                        up_value = fx.Float32(0.0)
                        for source_wave in range_constexpr(4):
                            source_index = (source_wave * _WAVE_SIZE + source_lane) * 4 + row % 4
                            gate_value = gate_value + lds_load(reduction, source_index)
                            up_index = ((source_wave + 4) * _WAVE_SIZE + source_lane) * 4 + row % 4
                            up_value = up_value + lds_load(reduction, up_index)
                        gate_value = bf16_round(gate_value)
                        up_value = bf16_round(up_value)
                        gate_tanh = fx.Float32(2.0) * rcp(
                            fx.Float32(1.0) + exp(fx.Float32(-0.5) * gate_value)
                        ) - fx.Float32(1.0)
                        gate_sigmoid = rcp(fx.Float32(1.0) + exp(-gate_value))
                        up_tanh = fx.Float32(2.0) * rcp(
                            fx.Float32(1.0) + exp(fx.Float32(-0.08) * up_value)
                        ) - fx.Float32(1.0)
                        activated.append(fx.Float32(4.0) * gate_tanh * gate_sigmoid * fx.Float32(25.0) * up_tanh)
                    pair = ((sample * _TOP_K + route_in_sample) * _INTER + row_group * 16 + local_row) // 2
                    store_raw_pair(
                        expert_mid_mailbox_rsrc,
                        pair,
                        activated[0],
                        activated[1],
                    )
                gpu.barrier()
                if tid == 0:
                    rocdl.s_waitcnt(vmcnt=0)
                    store_i32(
                        expert_mid_ready_rsrc,
                        (sample * _TOP_K + route_in_sample) * up_tiles + row_group,
                        1,
                    )
                up_task = up_task + _BLOCKS
            stamp(6)

            # Stage 7: expert down, route weighting, TP reduction, and per-tile
            # squared-norm publication for the latent RMSNorm.
            routed_tiles = _ROUTED_HIDDEN // 16
            down_tasks = samples * routed_tiles
            down_task = bid
            while down_task < down_tasks:
                sample = down_task // routed_tiles
                row_group = down_task % routed_tiles
                mid_pairs_per_route = _INTER // 2
                staged_pairs = _TOP_K * mid_pairs_per_route
                stage_raw_vector(
                    expert_mid_mailbox_rsrc,
                    expert_mid_ready_rsrc,
                    sample * _TOP_K * up_tiles,
                    _TOP_K * up_tiles,
                    sample * staged_pairs,
                    staged_pairs,
                )
                gpu.barrier()

                accumulator = [fx.Float32(0.0) for _ in range(4)]
                for route_round in range_constexpr(_TOP_K // _WAVES):
                    route_in_sample = wave + route_round * _WAVES
                    route = sample * _TOP_K + route_in_sample
                    expert = uniform(load_i32(selection_id_rsrc, route))
                    route_weight = uniform_f32(load_f32(selection_weight_rsrc, route))
                    expert_weight_bytes = _ROUTED_HIDDEN * (_INTER // 2)
                    down_weight_rsrc = rsrc(
                        packed_expert_down
                        + fx.Int64(expert) * fx.Int64(expert_weight_bytes)
                    )
                    down_scale_rsrc = rsrc(
                        expert_down_scale
                        + fx.Int64(expert) * fx.Int64(down_expert_scale_bytes)
                    )
                    route_accumulator = [fx.Float32(0.0) for _ in range(4)]
                    for k_chunk in range_constexpr(_INTER // 128):
                        fragment = mxfp4_fragment(
                            down_weight_rsrc,
                            down_scale_rsrc,
                            row_group,
                            k_chunk,
                            _INTER,
                        )
                        route_accumulator = mxfp4_apply(
                            route_accumulator,
                            fragment,
                            route_in_sample * mid_pairs_per_route + k_chunk * 64,
                        )
                    accumulator = [
                        accumulator[item] + route_accumulator[item] * route_weight for item in range_constexpr(4)
                    ]

                fx.ptr_store(
                    fx.Vector.from_elements(accumulator, fx.Float32),
                    reduction + (wave * _WAVE_SIZE + lane) * 4,
                )
                gpu.barrier()
                if tid < 16 // 2:
                    local_row = tid * 2
                    values = []
                    for pair_element in range_constexpr(2):
                        row = local_row + pair_element
                        source_lane = 16 * (row // 4)
                        value = fx.Float32(0.0)
                        for source_wave in range_constexpr(_WAVES):
                            source_index = (source_wave * _WAVE_SIZE + source_lane) * 4 + row % 4
                            value = value + lds_load(reduction, source_index)
                        values.append(value)
                    lds_store(
                        output_values,
                        tid,
                        bf16_pair(values[0], values[1]),
                    )
                gpu.barrier()

                pair_base = sample * (_ROUTED_HIDDEN // 2) + row_group * (16 // 2)

                def emit_routed(local_pair, value_low, value_high):
                    reduced_low = bf16_round(value_low)
                    reduced_high = bf16_round(value_high)
                    store_raw_pair(
                        routed_mailbox_rsrc,
                        pair_base + local_pair,
                        reduced_low,
                        reduced_high,
                    )
                    lds_store(
                        output_values,
                        local_pair,
                        reduced_low * reduced_low + reduced_high * reduced_high,
                    )

                moe_peer_reduce(16 // 2, pair_base, output_values, 0, emit_routed)
                square_part = (tid < 16 // 2).select(
                    lds_load(output_values, fx.min(tid, 16 // 2 - 1)),
                    fx.Float32(0.0),
                )
                square_sum = block_sum(square_part)
                if tid == 0:
                    rocdl.s_waitcnt(vmcnt=0)
                    store_f32(
                        routed_stats_rsrc,
                        sample * routed_tiles + row_group,
                        square_sum,
                    )
                down_task = down_task + _BLOCKS
            stamp(7)

            # One CTA per sample collapses the routed norm partials.  Tail
            # tasks poll this inverse RMS while loading their latent input.
            if bid < samples:
                square_part = fx.Float32(0.0)
                if tid < routed_tiles:
                    square_part = load_f32(
                        routed_stats_rsrc,
                        bid * routed_tiles + tid,
                    )
                total_square = block_sum(square_part)
                if tid == 0:
                    store_f32(
                        routed_inv_rsrc,
                        bid,
                        rsq(total_square * (1.0 / _ROUTED_HIDDEN) + EPS),
                    )
            stamp(8)

            # Stage 8: shared-down and rank-local latent-up run together, then
            # the final TP reduction adds the post-AttnRes residual in place.
            hidden_tiles = _HIDDEN // 16
            tail_tasks = samples * hidden_tiles
            tail_task = bid
            while tail_task < tail_tasks:
                sample = tail_task // hidden_tiles
                row_group = tail_task % hidden_tiles
                inverse_rms = uniform_f32(load_f32(routed_inv_rsrc, sample))

                shared_pairs = _SHARED_INTER // 2
                for load_round in range_constexpr((shared_pairs + _THREADS - 1) // _THREADS):
                    pair = tid + load_round * _THREADS
                    if pair < shared_pairs:
                        packed = load_pair(
                            shared_mid_mailbox_rsrc,
                            sample * shared_pairs + pair,
                        )
                        lds_store(x, pair, packed.bitcast(fx.Float32))

                latent_pairs = _ROUTED_HIDDEN // 2
                gain_rsrc = rsrc(latent_gain)
                for load_round in range_constexpr((latent_pairs + _THREADS - 1) // _THREADS):
                    pair = tid + load_round * _THREADS
                    if pair < latent_pairs:
                        packed = load_raw_pair(
                            routed_mailbox_rsrc,
                            sample * latent_pairs + pair,
                        )
                        values = fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.BFloat16).to(fx.Float32)
                        gain_word = fx.Int32(bo.buffer_load(gain_rsrc, pair, vec_width=1, dtype=T.i32))
                        gains = fx.Vector.from_elements([gain_word], fx.Int32).bitcast(fx.BFloat16).to(fx.Float32)
                        lds_store(
                            x,
                            shared_pairs + pair,
                            bf16_pair(
                                values[0] * inverse_rms * gains[0],
                                values[1] * inverse_rms * gains[1],
                            ),
                        )
                gpu.barrier()

                accumulator = [fx.Float32(0.0) for _ in range(4)]
                if wave < 4:
                    accumulator = mxfp8_bf16_accumulate(
                        rsrc(packed_shared_down),
                        rsrc(shared_down_scale),
                        0,
                        row_group,
                        _SHARED_INTER,
                        wave,
                        4,
                    )
                else:
                    first_local_row = rank * _HIDDEN_SHARD
                    global_row = row_group * 16
                    latent_live = (global_row >= first_local_row) & (global_row < first_local_row + _HIDDEN_SHARD)
                    local_row_group = fx.max(
                        fx.Int32(0),
                        (global_row - first_local_row) // 16,
                    )
                    split_wave = wave - 4
                    if latent_live:
                        accumulator = mxfp8_bf16_accumulate(
                            rsrc(packed_latent_up),
                            rsrc(latent_up_scale),
                            shared_pairs,
                            local_row_group,
                            _ROUTED_HIDDEN,
                            split_wave,
                            4,
                        )

                fx.ptr_store(
                    fx.Vector.from_elements(accumulator, fx.Float32),
                    reduction + (wave * _WAVE_SIZE + lane) * 4,
                )
                gpu.barrier()
                latent_live = (row_group * 16 >= rank * _HIDDEN_SHARD) & (row_group * 16 < (rank + 1) * _HIDDEN_SHARD)
                if tid < 16 // 2:
                    local_row = tid * 2
                    values = []
                    for pair_element in range_constexpr(2):
                        row = local_row + pair_element
                        source_lane = 16 * (row // 4)
                        shared_result = fx.Float32(0.0)
                        latent_value = fx.Float32(0.0)
                        for source_wave in range_constexpr(4):
                            source_index = (source_wave * _WAVE_SIZE + source_lane) * 4 + row % 4
                            shared_result = shared_result + lds_load(reduction, source_index)
                        for source_wave in range_constexpr(4):
                            latent_index = ((source_wave + 4) * _WAVE_SIZE + source_lane) * 4 + row % 4
                            latent_value = latent_value + lds_load(reduction, latent_index)
                        shared_result = bf16_round(shared_result)
                        latent_value = latent_live.select(bf16_round(latent_value), fx.Float32(0.0))
                        values.append(bf16_round(shared_result + latent_value))
                    lds_store(
                        output_values,
                        tid,
                        bf16_pair(values[0], values[1]),
                    )
                gpu.barrier()

                pair_base = sample * (_HIDDEN // 2) + row_group * (16 // 2)

                def emit_final(local_pair, value_low, value_high):
                    residual_word = fx.Int32(
                        bo.buffer_load(
                            rsrc(updated_prefix),
                            pair_base + local_pair,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    residual_low = (residual_word << 16).bitcast(fx.Float32)
                    residual_high = (residual_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    final_word = bf16_pair(
                        residual_low + value_low,
                        residual_high + value_high,
                    ).bitcast(fx.Int32)
                    if const_expr(mla):
                        batch_id = uniform(
                            bo.buffer_load(
                                rsrc(mla_batch_ids),
                                sample,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        final_word = (batch_id >= 0).select(
                            final_word,
                            fx.Int32(0),
                        )
                    bo.buffer_store(
                        final_word,
                        rsrc(final_output),
                        pair_base + local_pair,
                        cache_modifier=CM_DEV,
                    )

                moe_peer_reduce(16 // 2, pair_base, output_values, 1, emit_final)
                tail_task = tail_task + _BLOCKS
            stamp(9)
        if (bid == 0) & (tid == 0) & (advance_epoch != 0):
            bo.buffer_store(
                step_value + fx.Int32(1),
                rsrc(step),
                0,
                cache_modifier=CM_DEV,
            )

    @flyc.jit
    def launch(
        hidden_states: Int64,
        output: Int64,
        block_residual: Int64,
        self_res_norm: Int64,
        self_res_qk: Int64,
        input_norm: Int64,
        mlp_res_norm: Int64,
        mlp_res_qk: Int64,
        post_norm: Int64,
        pre_updated: Int64,
        pre_output: Int64,
        updated_prefix: Int64,
        moe_input: Int64,
        quantized_moe_input: Int64,
        quantized_moe_scale: Int64,
        block_stride: Int32,
        packed_router_weight: Int64,
        correction_bias: Int64,
        packed_latent_weight: Int64,
        latent_weight_scale: Int64,
        packed_shared_up: Int64,
        shared_up_scale: Int64,
        packed_expert_up: Int64,
        expert_up_scale: Int64,
        packed_expert_down: Int64,
        expert_down_scale: Int64,
        latent_gain: Int64,
        packed_shared_down: Int64,
        shared_down_scale: Int64,
        packed_latent_up: Int64,
        latent_up_scale: Int64,
        moe_symmetric: Int64,
        moe_peers: Int64,
        final_output: Int64,
        packed_input_weight: Int64,
        gate_weight: Int64,
        conv_weight: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        packed_output_weight: Int64,
        state_indices: Int64,
        num_accepted_tokens: Int64,
        conv_state: Int64,
        recurrent_state: Int64,
        scratch: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        timeline: Int64,
        mla_positions: Int64,
        mla_slot_mapping: Int64,
        mla_batch_ids: Int64,
        mla_context_lens: Int64,
        mla_block_tables: Int64,
        mla_block_table_stride: Int32,
        mla_block_size: Int32,
        mla_block_ratio: Int32,
        mla_cache: Int64,
        mla_cache_scale: Int64,
        mla_qkv_weight: Int64,
        mla_q_norm: Int64,
        mla_kv_norm: Int64,
        mla_q_b_weight: Int64,
        mla_uk_weight: Int64,
        mla_uv_weight: Int64,
        mla_gate_weight: Int64,
        rank: Int32,
        layer: Int32,
        advance_epoch: Int32,
        stream: Stream = Stream(None),
    ):
        kimi_k3_monokernel(
            hidden_states,
            output,
            block_residual,
            self_res_norm,
            self_res_qk,
            input_norm,
            mlp_res_norm,
            mlp_res_qk,
            post_norm,
            pre_updated,
            pre_output,
            updated_prefix,
            moe_input,
            quantized_moe_input,
            quantized_moe_scale,
            block_stride,
            packed_router_weight,
            correction_bias,
            packed_latent_weight,
            latent_weight_scale,
            packed_shared_up,
            shared_up_scale,
            packed_expert_up,
            expert_up_scale,
            packed_expert_down,
            expert_down_scale,
            latent_gain,
            packed_shared_down,
            shared_down_scale,
            packed_latent_up,
            latent_up_scale,
            moe_symmetric,
            moe_peers,
            final_output,
            packed_input_weight,
            gate_weight,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            packed_output_weight,
            state_indices,
            num_accepted_tokens,
            conv_state,
            recurrent_state,
            scratch,
            symmetric,
            peers,
            step,
            timeline,
            mla_positions,
            mla_slot_mapping,
            mla_batch_ids,
            mla_context_lens,
            mla_block_tables,
            mla_block_table_stride,
            mla_block_size,
            mla_block_ratio,
            mla_cache,
            mla_cache_scale,
            mla_qkv_weight,
            mla_q_norm,
            mla_kv_norm,
            mla_q_b_weight,
            mla_uk_weight,
            mla_uv_weight,
            mla_gate_weight,
            rank,
            layer,
            advance_epoch,
        ).launch(grid=(_BLOCKS,), block=(_THREADS,), stream=stream)

    block_tag = f"m{-attn_res_blocks}" if attn_res_blocks < 0 else str(attn_res_blocks)
    write_tag = f"m{-block_write_idx}" if block_write_idx < 0 else str(block_write_idx)
    launch.func.__name__ = (
        f"kimi_k3_monokernel_s{samples}_p{npes}_l{launches_per_step}"
        f"_b{block_tag}_w{write_tag}_f{int(fuse_moe)}_m{int(mtp)}"
        f"_bq{agentic_batch_size}_h{2 if state_fp16 else 4}"
        f"_c{conv_state_layout.value}_a{int(atom_expert_layout)}"
        f"_mla{int(mla)}"
    )
    return launch
