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
_EXPERT_TILE = 16
_PROJECTION_WAVES = 4
_SAMPLES_PER_CTA = 4
_ROUTING_TILE = 16


@functools.cache
def build_router_projection(
    hidden: int,
    num_experts: int,
    topk: int,
    samples: int,
    moe_elements: int,
    latent_rows: int,
    shared_rows: int,
    situ_beta: float,
    situ_linear_beta: float,
    include_router: bool = True,
    include_latent: bool = True,
    include_shared: bool = True,
):
    """Build one co-resident launch for router and independent MXFP8 projections.

    Projection CTAs compute eight expert rows and reuse each packed router-weight
    tile across up to four samples. The same persistent CTA set concurrently
    computes the latent and shared up/gate projections from MXFP8 weights. A
    reserved CTA polls tagged router scores and emits the padded sorting contract
    consumed by the MXFP4 expert kernels.
    """

    if hidden <= 0 or hidden % 128:
        raise ValueError(f"hidden must be a positive multiple of 128, got {hidden}")
    if num_experts <= 0 or num_experts % _WAVE_SIZE:
        raise ValueError(f"num_experts must be a positive multiple of {_WAVE_SIZE}, got {num_experts}")
    if not 0 < topk <= 32:
        raise ValueError(f"topk must be in [1, 32], got {topk}")
    if samples not in {1, 2, 4, 8}:
        raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")
    if moe_elements <= 0 or moe_elements % 8:
        raise ValueError(f"moe_elements must be a positive multiple of 8, got {moe_elements}")
    if latent_rows <= 0 or latent_rows % _EXPERT_TILE:
        raise ValueError(f"latent_rows must be a positive multiple of {_EXPERT_TILE}, got {latent_rows}")
    if shared_rows <= 0 or shared_rows % _EXPERT_TILE:
        raise ValueError(f"shared_rows must be a positive multiple of {_EXPERT_TILE}, got {shared_rows}")

    sample_group = 2 if samples == 4 else min(samples, _SAMPLES_PER_CTA)
    sample_groups = (samples + sample_group - 1) // sample_group
    expert_tiles = num_experts // _EXPERT_TILE
    latent_tiles = latent_rows // _EXPERT_TILE
    shared_tiles = shared_rows // _EXPERT_TILE
    shared_half_tiles = shared_tiles // 2
    if not (include_router or include_latent or include_shared):
        raise ValueError("at least one projection stage must be enabled")
    router_tasks = sample_groups * expert_tiles if include_router else 0
    latent_blocks = (latent_tiles + _PROJECTION_WAVES - 1) // _PROJECTION_WAVES if include_latent else 0
    shared_blocks = (shared_half_tiles + _PROJECTION_WAVES - 1) // _PROJECTION_WAVES if include_shared else 0
    latent_base = router_tasks
    shared_base = latent_base + latent_blocks
    selector_bid = shared_base + shared_blocks
    grid = selector_bid + int(include_router)

    k_chunks = hidden // 64
    k_scale_chunks = hidden // 256
    if include_router and k_chunks % _WAVES:
        raise ValueError(f"hidden/64 must be divisible by {_WAVES}, got {k_chunks}")
    router_chunks_per_wave = k_chunks // _WAVES
    values_per_lane = num_experts // _WAVE_SIZE

    x_words = max(sample_group * hidden // 2, samples * hidden // 4)
    reduction_words = _WAVES * _WAVE_SIZE * 4
    route_count = samples * topk
    block_scan = fx.coop.BlockScan[fx.Int32, _THREADS]

    @fx.struct
    class SharedStorage:
        x: fx.Array[fx.Float32, x_words, 16]
        reduction: fx.Array[fx.Float32, reduction_words, 16]
        route_ids: fx.Array[fx.Int32, route_count, 16]
        route_weights: fx.Array[fx.Float32, route_count, 16]
        route_positions: fx.Array[fx.Int32, route_count, 16]
        cumsum: fx.Array[fx.Int32, num_experts + 1, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def router_projection_kernel(
        hidden_states: Int64,
        quantized_hidden: Int64,
        quantized_hidden_scale: Int64,
        packed_router_weight: Int64,
        packed_latent_weight: Int64,
        latent_weight_scale: Int64,
        packed_shared_weight: Int64,
        shared_weight_scale: Int64,
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
        latent_out: Int64,
        shared_out: Int64,
        shared_mid_out: Int64,
        step: Int64,
        layer: Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE

        hidden_rsrc = rsrc(hidden_states)
        quantized_hidden_rsrc = rsrc(quantized_hidden)
        quantized_hidden_scale_rsrc = rsrc(quantized_hidden_scale)
        router_weight_rsrc = rsrc(packed_router_weight)
        latent_weight_rsrc = rsrc(packed_latent_weight)
        latent_scale_rsrc = rsrc(latent_weight_scale)
        shared_weight_rsrc = rsrc(packed_shared_weight)
        shared_scale_rsrc = rsrc(shared_weight_scale)
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
        latent_out_rsrc = rsrc(latent_out)
        shared_out_rsrc = rsrc(shared_out)
        shared_mid_rsrc = rsrc(shared_mid_out)
        step_value = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1

        allocator = fx.SharedAllocator()
        shared = allocator.allocate(SharedStorage).peek()
        x = shared.x.ptr
        reduction = shared.reduction.ptr
        route_ids = shared.route_ids.ptr
        route_weights = shared.route_weights.ptr
        route_positions = shared.route_positions.ptr
        cumsum = shared.cumsum.ptr
        scan_storage = allocator.allocate(block_scan.SharedStorage).peek()

        if include_router:
            if bid < fx.Int32(router_tasks):
                router_task = bid
                sample_base = (router_task // expert_tiles) * sample_group
                expert_tile = router_task % expert_tiles
                group_elements = sample_group * hidden
                loads_per_group = (group_elements + 4 * _THREADS - 1) // (4 * _THREADS)
                for load_index in range_constexpr(loads_per_group):
                    element = (tid + load_index * _THREADS) * 4
                    if element < group_elements:
                        words = fx.Vector(
                            bo.buffer_load(
                                hidden_rsrc,
                                (sample_base * hidden + element) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        fx.ptr_store(words[0].bitcast(fx.Float32), x + element // 2)
                        fx.ptr_store(words[1].bitcast(fx.Float32), x + element // 2 + 1)
                gpu.barrier()
                local_sample = fx.min(lane % 16, sample_group - 1)
                sample = sample_base + local_sample
                accumulator = fx.Vector.filled(4, 0.0, fx.Float32)

                for chunk_index in range_constexpr(router_chunks_per_wave):
                    chunk = wave * router_chunks_per_wave + chunk_index
                    weight_vectors = [
                        fx.Vector(
                            bo.buffer_load(
                                router_weight_rsrc,
                                (((expert_tile * k_chunks + chunk) * 2 + step_index) * _WAVE_SIZE + lane) * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for step_index in range_constexpr(2)
                    ]
                    for step_index in range_constexpr(2):
                        lhs = weight_vectors[step_index].bitcast(fx.BFloat16)
                        rhs = fx.ptr_load(
                            x + (local_sample * hidden + chunk * 64) // 2 + (lane // 16) * 4 + step_index * 16,
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        ).bitcast(fx.BFloat16)
                        accumulator = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [lhs, rhs, accumulator]))

                fx.ptr_store(accumulator, reduction + (wave * _WAVE_SIZE + lane) * 4)
                gpu.barrier()

                outputs = sample_group * _EXPERT_TILE
                if tid < outputs:
                    local_sample = tid // _EXPERT_TILE
                    row = tid % _EXPERT_TILE
                    sample = sample_base + local_sample
                    logit = fx.Float32(0.0)
                    for source_wave in range_constexpr(_WAVES):
                        source_lane = local_sample + 16 * (row // 4)
                        source_index = (source_wave * _WAVE_SIZE + source_lane) * 4 + row % 4
                        logit = logit + fx.ptr_load(reduction + source_index)
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

        if include_latent or include_shared:
            if (bid >= fx.Int32(latent_base)) & (bid < fx.Int32(grid)):
                lane_div16 = lane // 16
                lane_mod16 = lane % 16
                quantized_words = samples * hidden // 4
                quantized_loads = (quantized_words + 4 * _THREADS - 1) // (4 * _THREADS)
                for load_index in range_constexpr(quantized_loads):
                    word = (tid + load_index * _THREADS) * 4
                    if word < quantized_words:
                        values = fx.Vector(
                            bo.buffer_load(
                                quantized_hidden_rsrc,
                                word,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        fx.ptr_store(values.bitcast(fx.Float32), x + word)
                gpu.barrier()
                scale_atoms = [
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

                def dense_accumulate(weight_rsrc, scale_rsrc, row_tile):
                    accumulator = fx.make_rmem_tensor(4, fx.Float32)
                    accumulator.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    valid_sample = lane_mod16 < samples
                    sample = fx.min(lane_mod16, samples - 1)
                    scale_lane = lane_div16 * 16 + lane_mod16
                    for k256 in range_constexpr(k_scale_chunks):
                        activation_scale = fx.Int32(
                            bo.buffer_load(
                                quantized_hidden_scale_rsrc,
                                k256 * 64 + scale_lane,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
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
                                        x + (sample * hidden + k_base + k64_half * 64) // 4,
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
                            activation_fragment.store(
                                activation_halves[0].shuffle(activation_halves[1], list(range(8)))
                            )
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
                                scale_atoms[k128_half],
                                accumulator,
                                activation_fragment,
                                weight_fragment,
                                accumulator,
                                scale_a=activation_scale,
                                scale_b=weight_scale,
                            )
                    return accumulator.load()

                if include_latent:
                    if (bid >= fx.Int32(latent_base)) & (bid < fx.Int32(shared_base)):
                        row_tile = (bid - latent_base) * _PROJECTION_WAVES + wave
                        if (wave < _PROJECTION_WAVES) & (row_tile < latent_tiles):
                            values = dense_accumulate(latent_weight_rsrc, latent_scale_rsrc, row_tile)
                            output_sample_base = lane_div16 * 4
                            for element in range_constexpr(4):
                                output_sample = output_sample_base + element
                                if output_sample < samples:
                                    bo.buffer_store(
                                        values[element].to(fx.BFloat16),
                                        latent_out_rsrc,
                                        output_sample * latent_rows + row_tile * _EXPERT_TILE + lane_mod16,
                                    )

                if include_shared:
                    if (bid >= fx.Int32(shared_base)) & (bid < fx.Int32(selector_bid)):
                        row_tile = (bid - shared_base) * _PROJECTION_WAVES + wave
                        if (wave < _PROJECTION_WAVES) & (row_tile < shared_half_tiles):
                            gate_values = dense_accumulate(shared_weight_rsrc, shared_scale_rsrc, row_tile)
                            up_values = dense_accumulate(
                                shared_weight_rsrc,
                                shared_scale_rsrc,
                                row_tile + shared_half_tiles,
                            )
                            output_sample_base = lane_div16 * 4
                            for element in range_constexpr(4):
                                output_sample = output_sample_base + element
                                if output_sample < samples:
                                    gate_bf16 = gate_values[element].to(fx.BFloat16)
                                    up_bf16 = up_values[element].to(fx.BFloat16)
                                    output_offset = output_sample * shared_rows + row_tile * _EXPERT_TILE + lane_mod16
                                    bo.buffer_store(gate_bf16, shared_out_rsrc, output_offset)
                                    bo.buffer_store(
                                        up_bf16,
                                        shared_out_rsrc,
                                        output_offset + shared_rows // 2,
                                    )
                                    gate = fx.Float32(gate_bf16)
                                    up = fx.Float32(up_bf16)
                                    gate_tanh = fx.Float32(2.0) * rcp(
                                        fx.Float32(1.0) + exp(fx.Float32(-2.0 / situ_beta) * gate)
                                    ) - fx.Float32(1.0)
                                    gate_sigmoid = rcp(fx.Float32(1.0) + exp(-gate))
                                    up_tanh = fx.Float32(2.0) * rcp(
                                        fx.Float32(1.0) + exp(fx.Float32(-2.0 / situ_linear_beta) * up)
                                    ) - fx.Float32(1.0)
                                    value = (
                                        fx.Float32(situ_beta)
                                        * gate_tanh
                                        * gate_sigmoid
                                        * fx.Float32(situ_linear_beta)
                                        * up_tanh
                                    )
                                    bo.buffer_store(
                                        value.to(fx.BFloat16),
                                        shared_mid_rsrc,
                                        output_sample * (shared_rows // 2) + row_tile * _EXPERT_TILE + lane_mod16,
                                    )

        if include_router and bid == fx.Int32(selector_bid):
            zero4 = fx.Vector.filled(4, 0, fx.Int32)
            moe_vectors = moe_elements // 8
            for zero_index in range_constexpr((moe_vectors + _THREADS - 1) // _THREADS):
                vector_index = tid + zero_index * _THREADS
                if vector_index < moe_vectors:
                    bo.buffer_store(zero4, moe_buf_rsrc, vector_index * 4)

            if wave < samples:
                corrected_scores = []
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
                    corrected_scores.append(score + bias)

                selected_sum = fx.Float32(0.0)
                negative_infinity = fx.Float32(float("-inf"))
                for selected_index in range_constexpr(topk):
                    best_score = corrected_scores[0]
                    best_id = fx.Int32(lane)
                    for value_index in range_constexpr(1, values_per_lane):
                        candidate_score = corrected_scores[value_index]
                        candidate_id = fx.Int32(lane + value_index * _WAVE_SIZE)
                        take = (candidate_score > best_score) | (
                            (ArithValue(candidate_score) == ArithValue(best_score)) & (candidate_id < best_id)
                        )
                        best_score = take.select(candidate_score, best_score)
                        best_id = take.select(candidate_id, best_id)

                    for shuffle_offset in (32, 16, 8, 4, 2, 1):
                        peer_score = xshfl(best_score, shuffle_offset)
                        peer_id = xshfl(best_id, shuffle_offset)
                        take = (peer_score > best_score) | (
                            (ArithValue(peer_score) == ArithValue(best_score)) & (peer_id < best_id)
                        )
                        best_score = take.select(peer_score, best_score)
                        best_id = take.select(peer_id, best_id)

                    best_bias = fx.Float32(fx.BFloat16(bo.buffer_load(bias_rsrc, best_id, vec_width=1, dtype=T.bf16)))
                    best_raw = best_score - best_bias
                    selected_sum = selected_sum + best_raw
                    if lane == 0:
                        output_offset = wave * topk + selected_index
                        bo.buffer_store(best_id, ids_rsrc, output_offset)
                        bo.buffer_store(best_raw, weights_rsrc, output_offset)
                        fx.ptr_store(best_id, route_ids + output_offset)
                    for value_index in range_constexpr(values_per_lane):
                        expert = fx.Int32(lane + value_index * _WAVE_SIZE)
                        corrected_scores[value_index] = (expert == best_id).select(
                            negative_infinity, corrected_scores[value_index]
                        )

                if lane == 0:
                    inverse_sum = rcp(selected_sum)
                    for selected_index in range_constexpr(topk):
                        output_offset = wave * topk + selected_index
                        selected_weight = (
                            fx.Float32(
                                bo.buffer_load(
                                    weights_rsrc,
                                    output_offset,
                                    vec_width=1,
                                    dtype=T.f32,
                                )
                            )
                            * inverse_sum
                        )
                        bo.buffer_store(selected_weight, weights_rsrc, output_offset)
                        fx.ptr_store(selected_weight, route_weights + output_offset)

            gpu.barrier()

            for expert_iteration in range_constexpr((num_experts + _THREADS - 1) // _THREADS):
                expert = tid + expert_iteration * _THREADS
                if expert < num_experts:
                    fx.ptr_store(fx.Int32(0), cumsum + expert + 1)
            if tid == 0:
                fx.ptr_store(fx.Int32(0), cumsum)
            gpu.barrier()

            if tid < route_count:
                expert = fx.ptr_load(route_ids + tid)
                position = fx.atomic_add(
                    cumsum + expert + 1,
                    fx.Int32(1),
                    syncscope=fx.rocdl.SyncScope.Workgroup,
                )
                fx.ptr_store(position, route_positions + tid)
            gpu.barrier()

            for expert_iteration in range_constexpr((num_experts + _THREADS - 1) // _THREADS):
                expert = tid + expert_iteration * _THREADS
                if expert < num_experts:
                    active = fx.ptr_load(cumsum + expert + 1) != 0
                    fx.ptr_store(
                        active.select(fx.Int32(_ROUTING_TILE), fx.Int32(0)),
                        cumsum + expert + 1,
                    )
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
                    if active:
                        bo.buffer_store(expert, sorted_experts_rsrc, start // _ROUTING_TILE)
                        for padding_index in range_constexpr(_ROUTING_TILE):
                            slot = start + padding_index
                            bo.buffer_store(sentinel, sorted_ids_rsrc, slot)
                            bo.buffer_store(fx.Float32(0.0), sorted_weights_rsrc, slot)

            # Padding writers and route writers can target the same expert tile.
            # Make the real route entries the final stores deterministically.
            gpu.barrier()

            if tid < route_count:
                expert = fx.ptr_load(route_ids + tid)
                position = fx.ptr_load(cumsum + expert) + fx.ptr_load(route_positions + tid)
                token = tid // topk
                topk_slot = tid % topk
                bo.buffer_store((topk_slot << 24) | token, sorted_ids_rsrc, position)
                bo.buffer_store(fx.ptr_load(route_weights + tid), sorted_weights_rsrc, position)

    @flyc.jit
    def launch(
        hidden_states: Int64,
        quantized_hidden: Int64,
        quantized_hidden_scale: Int64,
        packed_router_weight: Int64,
        packed_latent_weight: Int64,
        latent_weight_scale: Int64,
        packed_shared_weight: Int64,
        shared_weight_scale: Int64,
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
        latent_out: Int64,
        shared_out: Int64,
        shared_mid_out: Int64,
        step: Int64,
        layer: Int32,
        stream: Stream = Stream(None),
    ):
        router_projection_kernel(
            hidden_states,
            quantized_hidden,
            quantized_hidden_scale,
            packed_router_weight,
            packed_latent_weight,
            latent_weight_scale,
            packed_shared_weight,
            shared_weight_scale,
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
            latent_out,
            shared_out,
            shared_mid_out,
            step,
            layer,
            value_attrs={"rocdl.flat_work_group_size": f"{_THREADS},{_THREADS}"},
        ).launch(grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    launch.func.__name__ = (
        f"router_projection_h{hidden}_e{num_experts}_k{topk}_s{samples}"
        f"_r{int(include_router)}l{int(include_latent)}s{int(include_shared)}"
    )
    return launch


class FusedRouterProjection:
    """Torch adapter for the fused router, latent, and shared projections."""

    def __init__(
        self,
        hidden: int,
        num_experts: int,
        topk: int,
        samples: int,
        moe_elements: int,
        latent_rows: int,
        shared_rows: int,
        situ_beta: float,
        situ_linear_beta: float,
        include_router: bool = True,
        include_latent: bool = True,
        include_shared: bool = True,
    ) -> None:
        self.launch = build_router_projection(
            hidden,
            num_experts,
            topk,
            samples,
            moe_elements,
            latent_rows,
            shared_rows,
            situ_beta,
            situ_linear_beta,
            include_router,
            include_latent,
            include_shared,
        )
        self.hidden = hidden
        self.num_experts = num_experts
        self.topk = topk
        self.samples = samples
        self.latent_rows = latent_rows
        self.shared_rows = shared_rows

    def __call__(
        self,
        hidden_states: torch.Tensor,
        quantized_hidden: torch.Tensor,
        quantized_hidden_scale: torch.Tensor,
        packed_router_weight: torch.Tensor,
        packed_latent_weight: torch.Tensor,
        latent_weight_scale: torch.Tensor,
        packed_shared_weight: torch.Tensor,
        shared_weight_scale: torch.Tensor,
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
        latent_out: torch.Tensor,
        shared_out: torch.Tensor,
        shared_mid_out: torch.Tensor,
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
        if quantized_hidden.shape != (32, self.hidden) or quantized_hidden.dtype != torch.uint8:
            raise ValueError("quantized_hidden must be uint8 [32, hidden]")
        if quantized_hidden_scale.numel() != 32 * (self.hidden // 32):
            raise ValueError("quantized_hidden_scale has the wrong packed size")
        if quantized_hidden_scale.dtype != torch.uint8:
            raise ValueError("quantized_hidden_scale must use uint8 E8M0 storage")
        if latent_weight_scale.numel() != self.latent_rows * (self.hidden // 32):
            raise ValueError("latent_weight_scale has the wrong packed size")
        if shared_weight_scale.numel() != self.shared_rows * (self.hidden // 32):
            raise ValueError("shared_weight_scale has the wrong packed size")
        if latent_weight_scale.dtype != torch.uint8 or shared_weight_scale.dtype != torch.uint8:
            raise ValueError("MXFP8 weight scales must use uint8 E8M0 storage")
        if latent_out.shape != (self.samples, self.latent_rows) or latent_out.dtype != torch.bfloat16:
            raise ValueError("latent_out must be BF16 [samples, latent_rows]")
        if shared_out.shape != (self.samples, self.shared_rows) or shared_out.dtype != torch.bfloat16:
            raise ValueError("shared_out must be BF16 [samples, shared_rows]")
        if shared_mid_out.shape != (self.samples, self.shared_rows // 2) or shared_mid_out.dtype != torch.bfloat16:
            raise ValueError("shared_mid_out must be BF16 [samples, shared_rows / 2]")
        tensors = (
            hidden_states,
            quantized_hidden,
            quantized_hidden_scale,
            packed_router_weight,
            packed_latent_weight,
            latent_weight_scale,
            packed_shared_weight,
            shared_weight_scale,
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
            latent_out,
            shared_out,
            shared_mid_out,
            step,
        )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("router inputs, scratch, and outputs must be contiguous")
        self.launch(
            hidden_states.data_ptr(),
            quantized_hidden.data_ptr(),
            quantized_hidden_scale.data_ptr(),
            packed_router_weight.data_ptr(),
            packed_latent_weight.data_ptr(),
            latent_weight_scale.data_ptr(),
            packed_shared_weight.data_ptr(),
            shared_weight_scale.data_ptr(),
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
            latent_out.data_ptr(),
            shared_out.data_ptr(),
            shared_mid_out.data_ptr(),
            step.data_ptr(),
            layer,
            stream=torch.cuda.current_stream(),
        )
