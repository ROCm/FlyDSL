# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused low-token BF16 router projection and sigmoid top-k selection."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.mla_moe_layer.kernel_common import exp, rcp, rsrc, uniform, xshfl
from kernels.mla_moe_layer.kernel_layout import CM_DEV, LAYER_SLOTS

_THREADS = 512
_WAVE_SIZE = 64
_WAVES = _THREADS // _WAVE_SIZE
_EXPERT_TILE = 8
_SAMPLES_PER_CTA = 4
_ROUTING_TILE = 16


@functools.cache
def build_router_projection(hidden: int, num_experts: int, topk: int, samples: int, moe_elements: int):
    """Build one co-resident launch for projection, top-k, and MoE dispatch.

    Projection CTAs compute eight expert rows and reuse each packed router-weight
    tile across up to four samples.  A final CTA polls tagged scores and selects
    top-k. The final CTA also emits the padded sorting contract consumed by the
    A16W4 expert kernels, so the graph path avoids the intermediate BF16 logits
    tensor and two standalone kernel launches.
    """

    if hidden <= 0 or hidden % 64:
        raise ValueError(f"hidden must be a positive multiple of 64, got {hidden}")
    if num_experts <= 0 or num_experts % _WAVE_SIZE:
        raise ValueError(f"num_experts must be a positive multiple of {_WAVE_SIZE}, got {num_experts}")
    if not 0 < topk <= 32:
        raise ValueError(f"topk must be in [1, 32], got {topk}")
    if samples not in {1, 2, 4, 8}:
        raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")
    if moe_elements <= 0 or moe_elements % 8:
        raise ValueError(f"moe_elements must be a positive multiple of 8, got {moe_elements}")

    sample_group = min(samples, _SAMPLES_PER_CTA)
    sample_groups = (samples + sample_group - 1) // sample_group
    expert_tiles = num_experts // _EXPERT_TILE
    projection_tasks = sample_groups * expert_tiles
    grid = projection_tasks + 1
    if grid > 256:
        raise ValueError(f"co-resident router grid must fit 256 CUs, got {grid} blocks")

    k_chunks = hidden // 64
    if k_chunks % (_WAVES * 2):
        raise ValueError(f"hidden/64 must be divisible by {_WAVES * 2}, got {k_chunks}")
    chunks_per_fold = k_chunks // _WAVES // 2
    values_per_lane = num_experts // _WAVE_SIZE

    x_words = sample_group * hidden // 2
    reduction_words = sample_group * _WAVES * _WAVE_SIZE * 4
    route_count = samples * topk
    block_scan = fx.coop.BlockScan[fx.Int32, _THREADS]

    @fx.struct
    class SharedStorage:
        x: fx.Array[fx.Float32, x_words, 16]
        reduction: fx.Array[fx.Float32, reduction_words, 16]
        route_ids: fx.Array[fx.Int32, route_count, 16]
        route_weights: fx.Array[fx.Float32, route_count, 16]
        cumsum: fx.Array[fx.Int32, num_experts + 1, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def router_projection_kernel(
        hidden_states: Int64,
        packed_weight: Int64,
        correction_bias: Int64,
        score_mailbox: Int64,
        scores_out: Int64,
        ids_out: Int64,
        weights_out: Int64,
        sorted_token_ids: Int64,
        sorted_weights: Int64,
        sorted_expert_ids: Int64,
        num_valid_ids: Int64,
        moe_buf: Int64,
        step: Int64,
        layer: Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE

        hidden_rsrc = rsrc(hidden_states)
        weight_rsrc = rsrc(packed_weight)
        bias_rsrc = rsrc(correction_bias)
        mailbox_rsrc = rsrc(score_mailbox)
        scores_rsrc = rsrc(scores_out)
        ids_rsrc = rsrc(ids_out)
        weights_rsrc = rsrc(weights_out)
        sorted_ids_rsrc = rsrc(sorted_token_ids)
        sorted_weights_rsrc = rsrc(sorted_weights)
        sorted_experts_rsrc = rsrc(sorted_expert_ids)
        num_valid_rsrc = rsrc(num_valid_ids)
        moe_buf_rsrc = rsrc(moe_buf)
        step_value = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1

        allocator = fx.SharedAllocator()
        shared = allocator.allocate(SharedStorage).peek()
        x = shared.x.ptr
        reduction = shared.reduction.ptr
        route_ids = shared.route_ids.ptr
        route_weights = shared.route_weights.ptr
        cumsum = shared.cumsum.ptr
        scan_storage = allocator.allocate(block_scan.SharedStorage).peek()
        v4f = fx.Vector.make_type(4, fx.Float32)

        if bid < fx.Int32(projection_tasks):
            expert_tile = bid % expert_tiles
            sample_base = (bid // expert_tiles) * sample_group

            loads_per_row = (hidden + 4 * _THREADS - 1) // (4 * _THREADS)
            for local_sample in range_constexpr(sample_group):
                sample = sample_base + local_sample
                for load_index in range_constexpr(loads_per_row):
                    k = (tid + load_index * _THREADS) * 4
                    if k < hidden:
                        words = fx.Vector(
                            bo.buffer_load(
                                hidden_rsrc,
                                (sample * hidden + k) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        fx.ptr_store(words[0].bitcast(fx.Float32), x + (local_sample * hidden + k) // 2)
                        fx.ptr_store(words[1].bitcast(fx.Float32), x + (local_sample * hidden + k) // 2 + 1)
            gpu.barrier()

            row_subtile = expert_tile * _EXPERT_TILE % 16
            weight_lane = (lane & -16) | (row_subtile + lane % _EXPERT_TILE)
            fold = (lane % 16) // _EXPERT_TILE
            accumulators = [[fx.Float32(0.0) for _ in range(4)] for _ in range(sample_group)]

            for chunk_index in range_constexpr(chunks_per_fold):
                chunk = wave * (k_chunks // _WAVES) + fold * chunks_per_fold + chunk_index
                row_group = expert_tile * _EXPERT_TILE // 16
                weight_vectors = [
                    fx.Vector(
                        bo.buffer_load(
                            weight_rsrc,
                            (((row_group * k_chunks + chunk) * 2 + step_index) * _WAVE_SIZE + weight_lane) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for step_index in range_constexpr(2)
                ]
                for local_sample in range_constexpr(sample_group):
                    result = fx.Vector.filled(4, 0.0, fx.Float32)
                    for step_index in range_constexpr(2):
                        lhs = weight_vectors[step_index].bitcast(fx.BFloat16)
                        rhs = fx.ptr_load(
                            x + (local_sample * hidden + chunk * 64) // 2 + (lane // 16) * 4 + step_index * 16,
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        result = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [lhs, rhs, result]))
                    accumulators[local_sample] = [
                        accumulators[local_sample][element] + result[element] for element in range(4)
                    ]

            for local_sample in range_constexpr(sample_group):
                base = (local_sample * _WAVES * _WAVE_SIZE + wave * _WAVE_SIZE + lane) * 4
                fx.ptr_store(fx.Vector.from_elements(accumulators[local_sample], fx.Float32), reduction + base)
            gpu.barrier()

            outputs = sample_group * _EXPERT_TILE
            if tid < outputs:
                local_sample = tid // _EXPERT_TILE
                row = tid % _EXPERT_TILE
                sample = sample_base + local_sample
                logit = fx.Float32(0.0)
                for source_wave in range_constexpr(_WAVES):
                    for source_fold in range_constexpr(2):
                        matrix_row = source_fold * _EXPERT_TILE + row
                        source_lane = source_fold * _EXPERT_TILE + 16 * (matrix_row // 4)
                        source_index = (
                            local_sample * _WAVES * _WAVE_SIZE + source_wave * _WAVE_SIZE + source_lane
                        ) * 4 + matrix_row % 4
                        logit = logit + fx.ptr_load(reduction + source_index)
                # The production contract is BF16 router GEMM output followed by
                # FP32 sigmoid/top-k. Preserve that handoff so near-tied K3
                # routes match the original rocBLAS path.
                logit = fx.Float32(logit.to(fx.BFloat16))
                score = rcp(fx.Float32(1.0) + exp(-logit))
                output_offset = sample * num_experts + expert_tile * _EXPERT_TILE + row
                bo.buffer_store(score, scores_rsrc, output_offset)
                bo.buffer_store(
                    fx.Vector.from_elements([score.bitcast(fx.Int32), tag], fx.Int32),
                    mailbox_rsrc,
                    output_offset * 2,
                    cache_modifier=CM_DEV,
                )

        if bid == fx.Int32(projection_tasks):
            zero4 = fx.Vector.filled(4, 0, fx.Int32)
            moe_vectors = moe_elements // 8
            for zero_index in range_constexpr((moe_vectors + _THREADS - 1) // _THREADS):
                vector_index = tid + zero_index * _THREADS
                if vector_index < moe_vectors:
                    bo.buffer_store(zero4, moe_buf_rsrc, vector_index * 4)

            if wave < samples:
                raw_scores = []
                corrected_scores = []
                expert_ids = []
                for value_index in range_constexpr(values_per_lane):
                    expert = lane + value_index * _WAVE_SIZE
                    offset = wave * num_experts + expert

                    pair = fx.Vector(
                        bo.buffer_load(mailbox_rsrc, offset * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV)
                    )
                    while pair[1] != tag:
                        rocdl.s_nop(0)
                        pair = fx.Vector(
                            bo.buffer_load(
                                mailbox_rsrc,
                                offset * 2,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                    score = pair[0].bitcast(fx.Float32)
                    bias = fx.Float32(fx.BFloat16(bo.buffer_load(bias_rsrc, expert, vec_width=1, dtype=T.bf16)))
                    raw_scores.append(score)
                    corrected_scores.append(score + bias)
                    expert_ids.append(fx.Int32(expert))

                selected_scores = []
                selected_ids = []
                selected_sum = fx.Float32(0.0)
                negative_infinity = fx.Float32(float("-inf"))
                for _ in range_constexpr(topk):
                    best_score = corrected_scores[0]
                    best_raw = raw_scores[0]
                    best_id = expert_ids[0]
                    for value_index in range_constexpr(1, values_per_lane):
                        candidate_score = corrected_scores[value_index]
                        candidate_id = expert_ids[value_index]
                        take = (candidate_score > best_score) | (
                            (ArithValue(candidate_score) == ArithValue(best_score)) & (candidate_id < best_id)
                        )
                        best_score = take.select(candidate_score, best_score)
                        best_raw = take.select(raw_scores[value_index], best_raw)
                        best_id = take.select(candidate_id, best_id)

                    for shuffle_offset in (32, 16, 8, 4, 2, 1):
                        peer_score = xshfl(best_score, shuffle_offset)
                        peer_raw = xshfl(best_raw, shuffle_offset)
                        peer_id = xshfl(best_id, shuffle_offset)
                        take = (peer_score > best_score) | (
                            (ArithValue(peer_score) == ArithValue(best_score)) & (peer_id < best_id)
                        )
                        best_score = take.select(peer_score, best_score)
                        best_raw = take.select(peer_raw, best_raw)
                        best_id = take.select(peer_id, best_id)

                    selected_scores.append(best_raw)
                    selected_ids.append(best_id)
                    selected_sum = selected_sum + best_raw
                    for value_index in range_constexpr(values_per_lane):
                        corrected_scores[value_index] = (expert_ids[value_index] == best_id).select(
                            negative_infinity, corrected_scores[value_index]
                        )

                if lane == 0:
                    inverse_sum = rcp(selected_sum)
                    for selected_index in range_constexpr(topk):
                        output_offset = wave * topk + selected_index
                        selected_weight = selected_scores[selected_index] * inverse_sum
                        bo.buffer_store(selected_ids[selected_index], ids_rsrc, output_offset)
                        bo.buffer_store(selected_weight, weights_rsrc, output_offset)
                        fx.ptr_store(selected_ids[selected_index], route_ids + output_offset)
                        fx.ptr_store(selected_weight, route_weights + output_offset)

            gpu.barrier()

            fx.ptr_store(fx.Int32(0), cumsum)
            for expert_iteration in range_constexpr((num_experts + _THREADS - 1) // _THREADS):
                expert = tid + expert_iteration * _THREADS
                if expert < num_experts:
                    count = fx.Int32(0)
                    for route_index in range_constexpr(route_count):
                        count = count + (fx.ptr_load(route_ids + route_index) == expert).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                    padded = (count > 0).select(
                        ((count + _ROUTING_TILE - 1) // _ROUTING_TILE) * _ROUTING_TILE,
                        fx.Int32(0),
                    )
                    fx.ptr_store(padded, cumsum + expert + 1)
            gpu.barrier()

            for chunk in range_constexpr((num_experts + _THREADS - 1) // _THREADS):
                expert = chunk * _THREADS + tid
                valid = expert < num_experts
                value = valid.select(fx.ptr_load(cumsum + expert + 1), fx.Int32(0))
                inclusive = block_scan.inclusive(value, fx.ReductionOp.ADD, storage=scan_storage)
                base = fx.Int32(0) if chunk == 0 else fx.ptr_load(cumsum + chunk * _THREADS)
                if valid:
                    fx.ptr_store(base + inclusive, cumsum + expert + 1)
                gpu.barrier()

            total_padded = fx.ptr_load(cumsum + num_experts)
            if tid == 0:
                bo.buffer_store(total_padded, num_valid_rsrc, 0)
                bo.buffer_store(fx.Int32(samples), num_valid_rsrc, 1)

            sentinel = fx.Int32((topk << 24) | samples)
            for expert_iteration in range_constexpr((num_experts + _THREADS - 1) // _THREADS):
                expert = tid + expert_iteration * _THREADS
                if expert < num_experts:
                    start = fx.ptr_load(cumsum + expert)
                    end = fx.ptr_load(cumsum + expert + 1)
                    active = end > start
                    count = fx.Int32(0)
                    for route_index in range_constexpr(route_count):
                        count = count + (fx.ptr_load(route_ids + route_index) == expert).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                    if active:
                        bo.buffer_store(expert, sorted_experts_rsrc, start // _ROUTING_TILE)
                        for padding_index in range_constexpr(_ROUTING_TILE):
                            if padding_index >= count:
                                slot = start + padding_index
                                bo.buffer_store(sentinel, sorted_ids_rsrc, slot)
                                bo.buffer_store(fx.Float32(0.0), sorted_weights_rsrc, slot)

            if tid < route_count:
                expert = fx.ptr_load(route_ids + tid)
                position = fx.ptr_load(cumsum + expert)
                for prior_index in range_constexpr(route_count):
                    is_prior = prior_index < tid
                    same_expert = fx.ptr_load(route_ids + prior_index) == expert
                    position = position + (is_prior & same_expert).select(fx.Int32(1), fx.Int32(0))
                token = tid // topk
                topk_slot = tid % topk
                bo.buffer_store((topk_slot << 24) | token, sorted_ids_rsrc, position)
                bo.buffer_store(fx.ptr_load(route_weights + tid), sorted_weights_rsrc, position)

    @flyc.jit
    def launch(
        hidden_states: Int64,
        packed_weight: Int64,
        correction_bias: Int64,
        score_mailbox: Int64,
        scores_out: Int64,
        ids_out: Int64,
        weights_out: Int64,
        sorted_token_ids: Int64,
        sorted_weights: Int64,
        sorted_expert_ids: Int64,
        num_valid_ids: Int64,
        moe_buf: Int64,
        step: Int64,
        layer: Int32,
        stream: Stream = Stream(None),
    ):
        router_projection_kernel(
            hidden_states,
            packed_weight,
            correction_bias,
            score_mailbox,
            scores_out,
            ids_out,
            weights_out,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            step,
            layer,
            value_attrs={"rocdl.flat_work_group_size": f"{_THREADS},{_THREADS}"},
        ).launch(grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"router_projection_h{hidden}_e{num_experts}_k{topk}_s{samples}"
    return launch


class FusedRouterProjection:
    """Torch adapter for the graph-safe fused router projection kernel."""

    def __init__(self, hidden: int, num_experts: int, topk: int, samples: int, moe_elements: int) -> None:
        self.launch = build_router_projection(hidden, num_experts, topk, samples, moe_elements)
        self.hidden = hidden
        self.num_experts = num_experts
        self.topk = topk
        self.samples = samples

    def __call__(
        self,
        hidden_states: torch.Tensor,
        packed_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        score_mailbox: torch.Tensor,
        scores_out: torch.Tensor,
        ids_out: torch.Tensor,
        weights_out: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        moe_buf: torch.Tensor,
        step: torch.Tensor,
        layer: int,
    ) -> None:
        if hidden_states.shape != (self.samples, self.hidden) or hidden_states.dtype != torch.bfloat16:
            raise ValueError("hidden_states must be contiguous BF16 [samples, hidden]")
        if correction_bias.shape != (self.num_experts,) or correction_bias.dtype != torch.bfloat16:
            raise ValueError("correction_bias must be BF16 [experts]")
        if score_mailbox.numel() != self.samples * self.num_experts * 2 or score_mailbox.dtype != torch.int32:
            raise ValueError("score_mailbox must be int32 storage for tagged FP32 scores")
        if scores_out.shape != (self.samples, self.num_experts) or scores_out.dtype != torch.float32:
            raise ValueError("scores_out must be FP32 [samples, experts]")
        if ids_out.shape != (self.samples, self.topk) or ids_out.dtype != torch.int32:
            raise ValueError("ids_out must be int32 [samples, topk]")
        if weights_out.shape != ids_out.shape or weights_out.dtype != torch.float32:
            raise ValueError("weights_out must be FP32 [samples, topk]")
        if step.shape != (1,) or step.dtype != torch.int32:
            raise ValueError("step must be int32[1]")
        tensors = (
            hidden_states,
            packed_weight,
            correction_bias,
            score_mailbox,
            scores_out,
            ids_out,
            weights_out,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            step,
        )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("router inputs, scratch, and outputs must be contiguous")
        self.launch(
            hidden_states.data_ptr(),
            packed_weight.data_ptr(),
            correction_bias.data_ptr(),
            score_mailbox.data_ptr(),
            scores_out.data_ptr(),
            ids_out.data_ptr(),
            weights_out.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            moe_buf.data_ptr(),
            step.data_ptr(),
            layer,
            stream=torch.cuda.current_stream(),
        )
