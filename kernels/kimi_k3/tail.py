# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused MXFP8 Kimi-K3 shared/latent tail and final TP reduction."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.common.fused_layer_config import EPS
from kernels.common.fused_layer_layout import CM_DEV, CM_SYS, LAYER_SLOTS
from kernels.common.fused_layer_ops import mxfp8_to_bf16x8, rsq, rsrc, uniform, xred

_THREADS = 512
_WAVE_SIZE = 64
_WAVES = _THREADS // _WAVE_SIZE
_ROW_TILE = 16
_TILES_PER_TASK = 2
_WAVES_PER_TILE = _WAVES // _TILES_PER_TASK
_SAMPLES_PER_CTA = 8
_BLOCKS = 224


@functools.cache
def build_kimi_k3_tail(
    samples: int,
    hidden: int,
    routed_hidden: int,
    shared_inter: int,
    npes: int,
    max_pairs: int,
):
    """Build the shared-down + latent-up + final all-reduce launcher."""

    if samples not in {1, 2, 4, 8}:
        raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")
    if npes != _WAVES:
        raise ValueError(f"fused Kimi-K3 tail requires {_WAVES} peers, got {npes}")
    if hidden % (_ROW_TILE * npes):
        raise ValueError(f"hidden must be divisible by {_ROW_TILE * npes}, got {hidden}")
    if shared_inter % 64 or routed_hidden % 64:
        raise ValueError("MXFP8 tail K dimensions must be multiples of 64")
    if max_pairs < samples * hidden // 2:
        raise ValueError("symmetric mailbox is too small for the final reduction")

    sample_group = min(samples, _SAMPLES_PER_CTA)
    sample_groups = (samples + sample_group - 1) // sample_group
    row_tiles = hidden // _ROW_TILE
    row_tile_pairs = row_tiles // _TILES_PER_TASK
    hidden_shard = hidden // npes
    shard_tiles = hidden_shard // _ROW_TILE
    shared_chunks = shared_inter // 64
    latent_chunks = routed_hidden // 64
    shared_chunks_per_wave = (shared_chunks + _WAVES_PER_TILE - 1) // _WAVES_PER_TILE
    latent_chunks_per_wave = (latent_chunks + _WAVES_PER_TILE - 1) // _WAVES_PER_TILE
    output_values = _TILES_PER_TASK * sample_group * _ROW_TILE
    output_pairs = output_values // 2
    task_rounds = (row_tile_pairs + _BLOCKS - 1) // _BLOCKS
    routed_pairs_per_row = routed_hidden // 2
    routed_blocks_per_row = (routed_pairs_per_row + _THREADS - 1) // _THREADS
    routed_blocks = samples * routed_blocks_per_row
    launch_blocks = _BLOCKS + routed_blocks
    if launch_blocks > 256:
        raise ValueError(f"fused tail requires a co-resident grid, got {launch_blocks} blocks")
    slot_bytes = npes * max_pairs * 8
    region_base = 2 * slot_bytes

    @fx.struct
    class SharedStorage:
        shared_x: fx.Array[fx.Float32, sample_group * shared_inter // 2, 16]
        latent_x: fx.Array[fx.Float32, sample_group * routed_hidden // 2, 16]
        reduction: fx.Array[fx.Float32, _WAVES * _WAVE_SIZE * 4, 16]
        shared_values: fx.Array[fx.Float32, output_values, 16]
        local_values: fx.Array[fx.Float32, output_values, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def kimi_k3_tail_kernel(
        routed_partial: Int64,
        routed_reduced: Int64,
        routed_gain: Int64,
        norm_scratch: Int64,
        shared_mid: Int64,
        latent_norm: Int64,
        packed_shared_down: Int64,
        shared_down_scale: Int64,
        packed_latent_up: Int64,
        latent_up_scale: Int64,
        residual: Int64,
        shared_partial: Int64,
        tail: Int64,
        final_partial: Int64,
        moe_delta: Int64,
        output: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE

        routed_partial_rsrc = rsrc(routed_partial)
        routed_reduced_rsrc = rsrc(routed_reduced)
        routed_gain_rsrc = rsrc(routed_gain)
        norm_scratch_rsrc = rsrc(norm_scratch)
        shared_mid_rsrc = rsrc(shared_mid)
        latent_norm_rsrc = rsrc(latent_norm)
        shared_weight_rsrc = rsrc(packed_shared_down)
        shared_scale_rsrc = rsrc(shared_down_scale)
        latent_up_weight_rsrc = rsrc(packed_latent_up)
        latent_scale_rsrc = rsrc(latent_up_scale)
        residual_rsrc = rsrc(residual)
        shared_partial_rsrc = rsrc(shared_partial)
        tail_rsrc = rsrc(tail)
        final_partial_rsrc = rsrc(final_partial)
        moe_delta_rsrc = rsrc(moe_delta)
        output_rsrc = rsrc(output)

        step_value = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1
        slot = (step_value * LAYER_SLOTS + layer) & 1
        routed_base = fx.Int64(slot) * fx.Int64(slot_bytes)
        final_base = fx.Int64(region_base) + fx.Int64(slot) * fx.Int64(slot_bytes)
        peer_words = fx.Vector(bo.buffer_load(rsrc(peers), wave * 2, vec_width=2, dtype=T.i32))
        peer_base = (fx.Int64(uniform(peer_words[1])) << 32) | fx.Int64(fx.Uint32(uniform(peer_words[0])))
        routed_peer_rsrc = rsrc(peer_base + routed_base)
        routed_local_rsrc = rsrc(symmetric + routed_base)
        peer_rsrc = rsrc(peer_base + final_base)
        local_rsrc = rsrc(symmetric + final_base)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        shared_x = storage.shared_x.ptr
        latent_x = storage.latent_x.ptr
        reduction = storage.reduction.ptr
        shared_values = storage.shared_values.ptr
        local_values = storage.local_values.ptr

        # Extra co-resident CTAs reduce and normalize the routed branch while
        # the regular tail CTAs compute the independent shared projection.
        if bid >= fx.Int32(_BLOCKS):
            reduce_bid = bid - fx.Int32(_BLOCKS)
            sample = reduce_bid // fx.Int32(routed_blocks_per_row)
            block_in_row = reduce_bid % fx.Int32(routed_blocks_per_row)
            pair_in_row = block_in_row * fx.Int32(_THREADS) + tid
            valid = pair_in_row < fx.Int32(routed_pairs_per_row)
            pair = sample * fx.Int32(routed_pairs_per_row) + pair_in_row

            if valid:
                value = fx.Int32(bo.buffer_load(routed_partial_rsrc, pair, vec_width=1, dtype=T.i32))
                mailbox = rank * max_pairs + pair
                bo.buffer_store(
                    fx.Vector.from_elements([value, tag], fx.Int32),
                    routed_peer_rsrc,
                    mailbox * 2,
                    cache_modifier=CM_SYS,
                )
            gpu.barrier()

            reduced_lo = fx.Float32(0.0)
            reduced_hi = fx.Float32(0.0)
            if (rank == wave) & valid:

                def load_routed_peers():
                    words = []
                    for source_rank in range_constexpr(npes):
                        mailbox = source_rank * max_pairs + pair
                        value_tag = fx.Vector(
                            bo.buffer_load(
                                routed_local_rsrc,
                                mailbox * 2,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        words += [value_tag[0], value_tag[1]]
                    return fx.Vector.from_elements(words, fx.Int32)

                values = load_routed_peers()
                pending = values[1] != tag
                for source_rank in range_constexpr(1, npes):
                    pending = pending | (values[source_rank * 2 + 1] != tag)
                while pending:
                    rocdl.s_nop(0)
                    values = load_routed_peers()
                    pending = values[1] != tag
                    for source_rank in range_constexpr(1, npes):
                        pending = pending | (values[source_rank * 2 + 1] != tag)

                sum_lo = fx.Float32(0.0)
                sum_hi = fx.Float32(0.0)
                for source_rank in range_constexpr(npes):
                    word = values[source_rank * 2]
                    sum_lo = sum_lo + (word << 16).bitcast(fx.Float32)
                    sum_hi = sum_hi + (word & fx.Int32(-65536)).bitcast(fx.Float32)
                packed = fx.Vector.from_elements([sum_lo, sum_hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                fx.ptr_store(packed.bitcast(fx.Float32), reduction + tid)
            gpu.barrier()

            result_tag = tag + fx.Int32(1 << 30)
            owned_pair_in_row = block_in_row * fx.Int32(_THREADS) + rank * _WAVE_SIZE + lane
            owned_valid = owned_pair_in_row < fx.Int32(routed_pairs_per_row)
            if (wave < npes) & owned_valid:
                owned_pair = sample * fx.Int32(routed_pairs_per_row) + owned_pair_in_row
                packed = fx.ptr_load(reduction + rank * _WAVE_SIZE + lane).bitcast(fx.Int32)
                mailbox = rank * max_pairs + owned_pair
                bo.buffer_store(
                    fx.Vector.from_elements([packed, result_tag], fx.Int32),
                    routed_peer_rsrc,
                    mailbox * 2,
                    cache_modifier=CM_SYS,
                )
            gpu.barrier()

            if valid:
                mailbox = wave * max_pairs + pair

                def load_routed_reduced():
                    return fx.Vector(
                        bo.buffer_load(
                            routed_local_rsrc,
                            mailbox * 2,
                            vec_width=2,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )

                reduced = load_routed_reduced()
                while reduced[1] != result_tag:
                    rocdl.s_nop(0)
                    reduced = load_routed_reduced()
                packed = reduced[0]
                bo.buffer_store(packed, routed_reduced_rsrc, pair, cache_modifier=CM_DEV)
                reduced_lo = (packed << 16).bitcast(fx.Float32)
                reduced_hi = (packed & fx.Int32(-65536)).bitcast(fx.Float32)

            square_sum = reduced_lo * reduced_lo + reduced_hi * reduced_hi
            for offset in (32, 16, 8, 4, 2, 1):
                square_sum = xred(square_sum, offset, lambda lhs, rhs: lhs + rhs)
            if lane == 0:
                fx.ptr_store(square_sum, reduction + wave)
            gpu.barrier()
            block_square_sum = fx.ptr_load(reduction)
            for source_wave in range_constexpr(1, _WAVES):
                block_square_sum = block_square_sum + fx.ptr_load(reduction + source_wave)
            if tid == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([block_square_sum.bitcast(fx.Int32), tag], fx.Int32),
                    norm_scratch_rsrc,
                    reduce_bid * 2,
                    cache_modifier=CM_DEV,
                )
            gpu.barrier()

            if tid < routed_blocks_per_row:
                partial_index = sample * fx.Int32(routed_blocks_per_row) + tid

                def load_norm_partial():
                    return fx.Vector(
                        bo.buffer_load(
                            norm_scratch_rsrc,
                            partial_index * 2,
                            vec_width=2,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )

                partial = load_norm_partial()
                while partial[1] != tag:
                    rocdl.s_nop(0)
                    partial = load_norm_partial()
                fx.ptr_store(partial[0].bitcast(fx.Float32), reduction + tid)
            gpu.barrier()

            total_square = fx.ptr_load(reduction)
            for source_block in range_constexpr(1, routed_blocks_per_row):
                total_square = total_square + fx.ptr_load(reduction + source_block)
            inverse_rms = rsq(total_square * (1.0 / routed_hidden) + EPS)
            if valid:
                gain_word = fx.Int32(bo.buffer_load(routed_gain_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                gain_lo = (gain_word << 16).bitcast(fx.Float32)
                gain_hi = (gain_word & fx.Int32(-65536)).bitcast(fx.Float32)
                normalized_word = (
                    fx.Vector.from_elements(
                        [reduced_lo * inverse_rms * gain_lo, reduced_hi * inverse_rms * gain_hi],
                        fx.Float32,
                    )
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                bo.buffer_store(normalized_word, rsrc(latent_norm), pair, cache_modifier=CM_DEV)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                bo.buffer_store(tag, norm_scratch_rsrc, routed_blocks * 2 + reduce_bid, cache_modifier=CM_DEV)

        def stage_bf16(source_rsrc, source_base, destination, elements):
            loads = (elements + 4 * _THREADS - 1) // (4 * _THREADS)
            for load_index in range_constexpr(loads):
                element = (tid + load_index * _THREADS) * 4
                if element < elements:
                    words = fx.Vector(
                        bo.buffer_load(source_rsrc, (source_base + element) // 2, vec_width=2, dtype=T.i32)
                    )
                    fx.ptr_store(words[0].bitcast(fx.Float32), destination + element // 2)
                    fx.ptr_store(words[1].bitcast(fx.Float32), destination + element // 2 + 1)

        def dense_accumulate(
            weight_rsrc,
            scale_rsrc,
            activation,
            k_dim,
            k_chunks,
            chunks_per_wave,
            row_tile_base,
            sample,
        ):
            tile_in_task = wave // _WAVES_PER_TILE
            wave_in_tile = wave % _WAVES_PER_TILE
            row_tile = row_tile_base + tile_in_task
            accumulator = fx.Vector.filled(4, 0.0, fx.Float32)
            for chunk_index in range_constexpr(chunks_per_wave):
                chunk = wave_in_tile * chunks_per_wave + chunk_index
                if chunk < k_chunks:
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
                        scale = ((scale_byte & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                        lhs = mxfp8_to_bf16x8(weight[0], weight[1], scale)
                        rhs = fx.ptr_load(
                            activation + (sample * k_dim + chunk * 64) // 2 + (lane // 16) * 4 + step_index * 16,
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        ).bitcast(fx.BFloat16)
                        accumulator = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [lhs, rhs, accumulator]))
            return accumulator

        def reduce_value(tile_in_task, local_sample, output_row):
            value = fx.Float32(0.0)
            for source_wave_index in range_constexpr(_WAVES_PER_TILE):
                source_wave = tile_in_task * _WAVES_PER_TILE + source_wave_index
                source_lane = local_sample + 16 * (output_row // 4)
                source_index = (source_wave * _WAVE_SIZE + source_lane) * 4 + output_row % 4
                value = value + fx.ptr_load(reduction + source_index)
            return value.to(fx.BFloat16)

        for sample_group_index in range_constexpr(sample_groups):
            sample_base = sample_group_index * sample_group
            stage_bf16(
                shared_mid_rsrc,
                sample_base * shared_inter,
                shared_x,
                sample_group * shared_inter,
            )
            gpu.barrier()

            for task_round in range_constexpr(task_rounds):
                row_tile_pair = bid + task_round * _BLOCKS
                if row_tile_pair < row_tile_pairs:
                    row_tile_base = row_tile_pair * _TILES_PER_TASK
                    sample = fx.min(lane % 16, sample_group - 1)
                    shared_accumulator = dense_accumulate(
                        shared_weight_rsrc,
                        shared_scale_rsrc,
                        shared_x,
                        shared_inter,
                        shared_chunks,
                        shared_chunks_per_wave,
                        row_tile_base,
                        sample,
                    )
                    fx.ptr_store(shared_accumulator, reduction + (wave * _WAVE_SIZE + lane) * 4)
                    gpu.barrier()

                    if tid < output_values:
                        tile_in_task = tid // (sample_group * _ROW_TILE)
                        tile_offset = tid % (sample_group * _ROW_TILE)
                        local_sample = tile_offset // _ROW_TILE
                        output_row = tile_offset % _ROW_TILE
                        row_tile = row_tile_base + tile_in_task
                        sample_out = sample_base + local_sample
                        shared_value = reduce_value(tile_in_task, local_sample, output_row)
                        bo.buffer_store(
                            shared_value,
                            shared_partial_rsrc,
                            sample_out * hidden + row_tile * _ROW_TILE + output_row,
                            cache_modifier=CM_DEV,
                        )
                        fx.ptr_store(fx.Float32(shared_value), shared_values + tid)
                    gpu.barrier()

                    owner_rank = row_tile_base // shard_tiles
                    owner_words = fx.Vector(bo.buffer_load(rsrc(peers), owner_rank * 2, vec_width=2, dtype=T.i32))
                    owner_base = (fx.Int64(uniform(owner_words[1])) << 32) | fx.Int64(
                        fx.Uint32(uniform(owner_words[0]))
                    )
                    owner_rsrc = rsrc(owner_base + final_base)
                    for output_batch in range_constexpr((output_pairs + _WAVE_SIZE - 1) // _WAVE_SIZE):
                        output_pair = lane + output_batch * _WAVE_SIZE
                        if (wave == 0) & (output_pair < output_pairs):
                            tile_in_task = output_pair // (sample_group * (_ROW_TILE // 2))
                            pair_offset = output_pair % (sample_group * (_ROW_TILE // 2))
                            local_sample = pair_offset // (_ROW_TILE // 2)
                            row_pair = pair_offset % (_ROW_TILE // 2)
                            row_tile = row_tile_base + tile_in_task
                            sample_out = sample_base + local_sample
                            value_index = (
                                tile_in_task * sample_group * _ROW_TILE + local_sample * _ROW_TILE + row_pair * 2
                            )
                            shared_word = (
                                fx.Vector.from_elements(
                                    [
                                        fx.ptr_load(shared_values + value_index),
                                        fx.ptr_load(shared_values + value_index + 1),
                                    ],
                                    fx.Float32,
                                )
                                .to(fx.BFloat16)
                                .bitcast(fx.Int32)[0]
                            )
                            pair = sample_out * (hidden // 2) + row_tile * (_ROW_TILE // 2) + row_pair
                            bo.buffer_store(shared_word, final_partial_rsrc, pair, cache_modifier=CM_DEV)
                            mailbox = rank * max_pairs + pair
                            bo.buffer_store(
                                fx.Vector.from_elements([shared_word, tag], fx.Int32),
                                owner_rsrc,
                                mailbox * 2,
                                cache_modifier=CM_SYS,
                            )
                    gpu.barrier()

                    if tid < routed_blocks:

                        def load_norm_done():
                            return fx.Int32(
                                bo.buffer_load(
                                    norm_scratch_rsrc,
                                    routed_blocks * 2 + tid,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )

                        done = load_norm_done()
                        while done != tag:
                            rocdl.s_nop(0)
                            done = load_norm_done()
                    gpu.barrier()

                    stage_bf16(
                        latent_norm_rsrc,
                        sample_base * routed_hidden,
                        latent_x,
                        sample_group * routed_hidden,
                    )
                    gpu.barrier()

                    local_start_tile = rank * shard_tiles
                    is_local = (row_tile_base >= local_start_tile) & (row_tile_base < local_start_tile + shard_tiles)
                    if is_local:
                        tail_row_tile_base = row_tile_base - local_start_tile
                        tail_accumulator = dense_accumulate(
                            latent_up_weight_rsrc,
                            latent_scale_rsrc,
                            latent_x,
                            routed_hidden,
                            latent_chunks,
                            latent_chunks_per_wave,
                            tail_row_tile_base,
                            sample,
                        )
                        fx.ptr_store(tail_accumulator, reduction + (wave * _WAVE_SIZE + lane) * 4)
                        gpu.barrier()
                        if tid < output_values:
                            tile_in_task = tid // (sample_group * _ROW_TILE)
                            tile_offset = tid % (sample_group * _ROW_TILE)
                            local_sample = tile_offset // _ROW_TILE
                            output_row = tile_offset % _ROW_TILE
                            tail_row_tile = tail_row_tile_base + tile_in_task
                            sample_out = sample_base + local_sample
                            tail_value = reduce_value(tile_in_task, local_sample, output_row)
                            bo.buffer_store(
                                tail_value,
                                tail_rsrc,
                                sample_out * hidden_shard + tail_row_tile * _ROW_TILE + output_row,
                                cache_modifier=CM_DEV,
                            )
                            fx.ptr_store(fx.Float32(tail_value), local_values + tid)
                    gpu.barrier()

                    if (rank == owner_rank) & (tid < output_pairs):
                        tile_in_task = tid // (sample_group * (_ROW_TILE // 2))
                        pair_offset = tid % (sample_group * (_ROW_TILE // 2))
                        local_sample = pair_offset // (_ROW_TILE // 2)
                        row_pair = pair_offset % (_ROW_TILE // 2)
                        row_tile = row_tile_base + tile_in_task
                        sample_out = sample_base + local_sample
                        pair = sample_out * (hidden // 2) + row_tile * (_ROW_TILE // 2) + row_pair

                        def load_all():
                            words = []
                            for source_rank in range_constexpr(npes):
                                mailbox = source_rank * max_pairs + pair
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

                        values = load_all()
                        pending = values[1] != tag
                        for source_rank in range_constexpr(1, npes):
                            pending = pending | (values[source_rank * 2 + 1] != tag)
                        while pending:
                            rocdl.s_nop(0)
                            values = load_all()
                            pending = values[1] != tag
                            for source_rank in range_constexpr(1, npes):
                                pending = pending | (values[source_rank * 2 + 1] != tag)

                        sum_lo = fx.Float32(0.0)
                        sum_hi = fx.Float32(0.0)
                        for source_rank in range_constexpr(npes):
                            word = values[source_rank * 2]
                            sum_lo = sum_lo + (word << 16).bitcast(fx.Float32)
                            sum_hi = sum_hi + (word & fx.Int32(-65536)).bitcast(fx.Float32)
                        value_index = tile_in_task * sample_group * _ROW_TILE + local_sample * _ROW_TILE + row_pair * 2
                        tail_lo = fx.ptr_load(local_values + value_index)
                        tail_hi = fx.ptr_load(local_values + value_index + 1)
                        local_word = (
                            fx.Vector.from_elements(
                                [
                                    fx.ptr_load(shared_values + value_index) + tail_lo,
                                    fx.ptr_load(shared_values + value_index + 1) + tail_hi,
                                ],
                                fx.Float32,
                            )
                            .to(fx.BFloat16)
                            .bitcast(fx.Int32)[0]
                        )
                        bo.buffer_store(local_word, final_partial_rsrc, pair, cache_modifier=CM_DEV)
                        sum_lo = sum_lo + tail_lo
                        sum_hi = sum_hi + tail_hi
                        reduced_word = (
                            fx.Vector.from_elements([sum_lo, sum_hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                        )
                        fx.ptr_store(reduced_word.bitcast(fx.Float32), local_values + tid)
                    gpu.barrier()

                    result_tag = tag + fx.Int32(1 << 30)
                    if (rank == owner_rank) & (wave < npes):
                        for output_batch in range_constexpr((output_pairs + _WAVE_SIZE - 1) // _WAVE_SIZE):
                            output_pair = lane + output_batch * _WAVE_SIZE
                            if output_pair < output_pairs:
                                tile_in_task = output_pair // (sample_group * (_ROW_TILE // 2))
                                pair_offset = output_pair % (sample_group * (_ROW_TILE // 2))
                                local_sample = pair_offset // (_ROW_TILE // 2)
                                row_pair = pair_offset % (_ROW_TILE // 2)
                                row_tile = row_tile_base + tile_in_task
                                sample_out = sample_base + local_sample
                                pair = sample_out * (hidden // 2) + row_tile * (_ROW_TILE // 2) + row_pair
                                reduced_word = fx.ptr_load(local_values + output_pair).bitcast(fx.Int32)
                                mailbox = owner_rank * max_pairs + pair
                                bo.buffer_store(
                                    fx.Vector.from_elements([reduced_word, result_tag], fx.Int32),
                                    peer_rsrc,
                                    mailbox * 2,
                                    cache_modifier=CM_SYS,
                                )
                    gpu.barrier()

                    if tid < output_pairs:
                        tile_in_task = tid // (sample_group * (_ROW_TILE // 2))
                        pair_offset = tid % (sample_group * (_ROW_TILE // 2))
                        local_sample = pair_offset // (_ROW_TILE // 2)
                        row_pair = pair_offset % (_ROW_TILE // 2)
                        row_tile = row_tile_base + tile_in_task
                        sample_out = sample_base + local_sample
                        pair = sample_out * (hidden // 2) + row_tile * (_ROW_TILE // 2) + row_pair
                        mailbox = owner_rank * max_pairs + pair

                        def load_reduced():
                            return fx.Vector(
                                bo.buffer_load(
                                    local_rsrc,
                                    mailbox * 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )

                        reduced = load_reduced()
                        while reduced[1] != result_tag:
                            rocdl.s_nop(0)
                            reduced = load_reduced()
                        reduced_word = reduced[0]
                        bo.buffer_store(reduced_word, moe_delta_rsrc, pair, cache_modifier=CM_DEV)
                        sum_lo = (reduced_word << 16).bitcast(fx.Float32)
                        sum_hi = (reduced_word & fx.Int32(-65536)).bitcast(fx.Float32)
                        residual_word = fx.Int32(bo.buffer_load(residual_rsrc, pair, vec_width=1, dtype=T.i32))
                        residual_lo = (residual_word << 16).bitcast(fx.Float32)
                        residual_hi = (residual_word & fx.Int32(-65536)).bitcast(fx.Float32)
                        output_word = (
                            fx.Vector.from_elements([residual_lo + sum_lo, residual_hi + sum_hi], fx.Float32)
                            .to(fx.BFloat16)
                            .bitcast(fx.Int32)[0]
                        )
                        bo.buffer_store(output_word, output_rsrc, pair, cache_modifier=CM_DEV)
                    gpu.barrier()

    @flyc.jit
    def launch(
        routed_partial: Int64,
        routed_reduced: Int64,
        routed_gain: Int64,
        norm_scratch: Int64,
        shared_mid: Int64,
        latent_norm: Int64,
        packed_shared_down: Int64,
        shared_down_scale: Int64,
        packed_latent_up: Int64,
        latent_up_scale: Int64,
        residual: Int64,
        shared_partial: Int64,
        tail: Int64,
        final_partial: Int64,
        moe_delta: Int64,
        output: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: Stream = Stream(None),
    ):
        kimi_k3_tail_kernel(
            routed_partial,
            routed_reduced,
            routed_gain,
            norm_scratch,
            shared_mid,
            latent_norm,
            packed_shared_down,
            shared_down_scale,
            packed_latent_up,
            latent_up_scale,
            residual,
            shared_partial,
            tail,
            final_partial,
            moe_delta,
            output,
            symmetric,
            peers,
            step,
            rank,
            layer,
            value_attrs={"rocdl.flat_work_group_size": f"{_THREADS},{_THREADS}"},
        ).launch(grid=(launch_blocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"kimi_k3_tail_s{samples}_h{hidden}_r{routed_hidden}_i{shared_inter}"
    return launch


class FusedKimiK3Tail:
    """Torch adapter for the graph-safe fused Kimi-K3 output tail."""

    def __init__(
        self,
        samples: int,
        hidden: int,
        routed_hidden: int,
        shared_inter: int,
        rank: int,
        npes: int,
        max_pairs: int,
    ) -> None:
        self.samples = samples
        self.hidden = hidden
        self.routed_hidden = routed_hidden
        self.shared_inter = shared_inter
        self.rank = rank
        self.npes = npes
        self.launch = build_kimi_k3_tail(samples, hidden, routed_hidden, shared_inter, npes, max_pairs)

    def __call__(
        self,
        routed_partial: torch.Tensor,
        routed_reduced: torch.Tensor,
        routed_gain: torch.Tensor,
        norm_scratch: torch.Tensor,
        shared_mid: torch.Tensor,
        latent_norm: torch.Tensor,
        packed_shared_down: torch.Tensor,
        shared_down_scale: torch.Tensor,
        packed_latent_up: torch.Tensor,
        latent_up_scale: torch.Tensor,
        residual: torch.Tensor,
        shared_partial: torch.Tensor,
        tail: torch.Tensor,
        final_partial: torch.Tensor,
        moe_delta: torch.Tensor,
        output: torch.Tensor,
        symmetric: int,
        peers: torch.Tensor,
        step: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        expected = {
            "routed_partial": (routed_partial, (self.samples, self.routed_hidden), torch.bfloat16),
            "routed_reduced": (routed_reduced, (self.samples, self.routed_hidden), torch.bfloat16),
            "routed_gain": (routed_gain, (self.routed_hidden,), torch.bfloat16),
            "shared_mid": (shared_mid, (self.samples, self.shared_inter), torch.bfloat16),
            "latent_norm": (latent_norm, (self.samples, self.routed_hidden), torch.bfloat16),
            "residual": (residual, (self.samples, self.hidden), torch.bfloat16),
            "shared_partial": (shared_partial, (self.samples, self.hidden), torch.bfloat16),
            "tail": (tail, (self.samples, self.hidden // self.npes), torch.bfloat16),
            "final_partial": (final_partial, (self.samples, self.hidden), torch.bfloat16),
            "moe_delta": (moe_delta, (self.samples, self.hidden), torch.bfloat16),
            "output": (output, (self.samples, self.hidden), torch.bfloat16),
        }
        for name, (tensor, shape, dtype) in expected.items():
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous {dtype} {list(shape)}")
        shared_scale_rows = (self.hidden + 255) // 256 * 256
        latent_scale_rows = (self.hidden // self.npes + 255) // 256 * 256
        if shared_down_scale.numel() != shared_scale_rows * (self.shared_inter // 32):
            raise ValueError("shared_down_scale has the wrong shape")
        if latent_up_scale.numel() != latent_scale_rows * (self.routed_hidden // 32):
            raise ValueError("latent_up_scale has the wrong shape")
        if shared_down_scale.dtype != torch.uint8 or latent_up_scale.dtype != torch.uint8:
            raise ValueError("MXFP8 scales must use uint8 E8M0 storage")
        routed_blocks = self.samples * ((self.routed_hidden // 2 + _THREADS - 1) // _THREADS)
        if norm_scratch.dtype != torch.int32 or norm_scratch.numel() < routed_blocks * 3:
            raise ValueError("norm_scratch has the wrong size or dtype")
        tensors = (
            norm_scratch,
            packed_shared_down,
            shared_down_scale,
            packed_latent_up,
            latent_up_scale,
            peers,
            step,
        )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("packed weights, scales, peers, and step must be contiguous")
        self.launch(
            routed_partial.data_ptr(),
            routed_reduced.data_ptr(),
            routed_gain.data_ptr(),
            norm_scratch.data_ptr(),
            shared_mid.data_ptr(),
            latent_norm.data_ptr(),
            packed_shared_down.data_ptr(),
            shared_down_scale.data_ptr(),
            packed_latent_up.data_ptr(),
            latent_up_scale.data_ptr(),
            residual.data_ptr(),
            shared_partial.data_ptr(),
            tail.data_ptr(),
            final_partial.data_ptr(),
            moe_delta.data_ptr(),
            output.data_ptr(),
            symmetric,
            peers.data_ptr(),
            step.data_ptr(),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        return output
