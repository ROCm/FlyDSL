# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Small graph-safe BF16 all-reduce using tagged symmetric peer mailboxes."""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.mla_moe_layer.kernel_common import rsrc, uniform
from kernels.mla_moe_layer.kernel_layout import (
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    symmetric_allreduce_nbytes,
)
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer

THREADS = 512
WAVE_SIZE = 64


def build_symmetric_bf16_allreduce(
    numel: int,
    npes: int,
    max_pairs: int,
    region: int,
):
    """Build one tagged-mailbox all-reduce launcher for a fixed BF16 shape."""

    if numel <= 0 or numel % 2:
        raise ValueError(f"numel must be positive and even, got {numel}")
    if npes not in {2, 4, 8}:
        raise ValueError(f"npes must be one of {{2, 4, 8}}, got {npes}")
    pairs = numel // 2
    if pairs > max_pairs:
        raise ValueError(f"pairs={pairs} exceeds max_pairs={max_pairs}")
    blocks = (pairs + THREADS - 1) // THREADS
    slot_bytes = npes * max_pairs * 8
    region_base = region * 2 * slot_bytes

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def symmetric_bf16_allreduce(
        source: Int64,
        output: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WAVE_SIZE
        wave = tid // WAVE_SIZE

        step_value = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1
        slot = (step_value * LAYER_SLOTS + layer) & 1
        base = fx.Int64(region_base) + fx.Int64(slot) * fx.Int64(slot_bytes)

        peer_words = fx.Vector(bo.buffer_load(rsrc(peers), fx.min(wave, npes - 1) * 2, vec_width=2, dtype=T.i32))
        peer_base = (fx.Int64(uniform(peer_words[1])) << 32) | fx.Int64(fx.Uint32(uniform(peer_words[0])))
        source_rsrc = rsrc(source)

        if wave < npes:
            peer_rsrc = rsrc(peer_base + base)
            for batch in range_constexpr(THREADS // WAVE_SIZE):
                pair = bid * THREADS + lane + batch * WAVE_SIZE
                if pair < pairs:
                    value = fx.Int32(bo.buffer_load(source_rsrc, pair, vec_width=1, dtype=T.i32))
                    mailbox = rank * max_pairs + pair
                    bo.buffer_store(
                        fx.Vector.from_elements([value, tag], fx.Int32),
                        peer_rsrc,
                        mailbox * 2,
                        cache_modifier=CM_SYS,
                    )
        gpu.barrier()

        pair = bid * THREADS + tid
        if pair < pairs:
            local_rsrc = rsrc(symmetric + base)

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
            packed = fx.Vector.from_elements([sum_lo, sum_hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            bo.buffer_store(packed[0], rsrc(output), pair, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(
        source: Int64,
        output: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: Stream = Stream(None),
    ):
        symmetric_bf16_allreduce(
            source,
            output,
            symmetric,
            peers,
            step,
            rank,
            layer,
            value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"},
        ).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"symmetric_bf16_allreduce_n{numel}_w{npes}_r{region}"
    return launch


def build_symmetric_bf16_final_reduce(
    numel: int,
    hidden: int,
    shard_width: int,
    npes: int,
    max_pairs: int,
    region: int,
):
    """Build ``shared + rank-local tail -> all-reduce -> residual`` in one launch."""

    if numel <= 0 or numel % 2 or numel % hidden:
        raise ValueError(f"numel must be a positive even multiple of hidden={hidden}, got {numel}")
    if shard_width <= 0 or shard_width * npes != hidden:
        raise ValueError(f"shard_width*npes must equal hidden, got {shard_width}*{npes} != {hidden}")
    pairs = numel // 2
    if pairs > max_pairs:
        raise ValueError(f"pairs={pairs} exceeds max_pairs={max_pairs}")
    blocks = (pairs + THREADS - 1) // THREADS
    slot_bytes = npes * max_pairs * 8
    region_base = region * 2 * slot_bytes

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def symmetric_bf16_final_reduce(
        shared_source: Int64,
        tail: Int64,
        residual: Int64,
        local_partial: Int64,
        reduced: Int64,
        output: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WAVE_SIZE
        wave = tid // WAVE_SIZE

        step_value = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1
        slot = (step_value * LAYER_SLOTS + layer) & 1
        base = fx.Int64(region_base) + fx.Int64(slot) * fx.Int64(slot_bytes)

        peer_words = fx.Vector(bo.buffer_load(rsrc(peers), fx.min(wave, npes - 1) * 2, vec_width=2, dtype=T.i32))
        peer_base = (fx.Int64(uniform(peer_words[1])) << 32) | fx.Int64(fx.Uint32(uniform(peer_words[0])))
        shared_rsrc = rsrc(shared_source)
        tail_rsrc = rsrc(tail)
        local_partial_rsrc = rsrc(local_partial)

        if wave < npes:
            peer_rsrc = rsrc(peer_base + base)
            for batch in range_constexpr(THREADS // WAVE_SIZE):
                pair = bid * THREADS + lane + batch * WAVE_SIZE
                if pair < pairs:
                    element = pair * 2
                    sample = element // hidden
                    column = element % hidden
                    local_start = rank * shard_width
                    is_local = (column >= local_start) & (column < local_start + shard_width)
                    tail_pair = (sample * shard_width + column - local_start) // 2
                    safe_tail_pair = is_local.select(tail_pair, fx.Int32(0))

                    shared_word = fx.Int32(bo.buffer_load(shared_rsrc, pair, vec_width=1, dtype=T.i32))
                    tail_word = fx.Int32(bo.buffer_load(tail_rsrc, safe_tail_pair, vec_width=1, dtype=T.i32))
                    shared_lo = (shared_word << 16).bitcast(fx.Float32)
                    shared_hi = (shared_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    tail_lo = is_local.select((tail_word << 16).bitcast(fx.Float32), fx.Float32(0.0))
                    tail_hi = is_local.select((tail_word & fx.Int32(-65536)).bitcast(fx.Float32), fx.Float32(0.0))
                    packed = (
                        fx.Vector.from_elements([shared_lo + tail_lo, shared_hi + tail_hi], fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)[0]
                    )
                    if wave == 0:
                        bo.buffer_store(packed, local_partial_rsrc, pair, cache_modifier=CM_DEV)
                    mailbox = rank * max_pairs + pair
                    bo.buffer_store(
                        fx.Vector.from_elements([packed, tag], fx.Int32),
                        peer_rsrc,
                        mailbox * 2,
                        cache_modifier=CM_SYS,
                    )
        gpu.barrier()

        pair = bid * THREADS + tid
        if pair < pairs:
            local_rsrc = rsrc(symmetric + base)

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

            reduced_word = fx.Vector.from_elements([sum_lo, sum_hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
            bo.buffer_store(reduced_word, rsrc(reduced), pair, cache_modifier=CM_DEV)
            reduced_lo = (reduced_word << 16).bitcast(fx.Float32)
            reduced_hi = (reduced_word & fx.Int32(-65536)).bitcast(fx.Float32)

            residual_word = fx.Int32(bo.buffer_load(rsrc(residual), pair, vec_width=1, dtype=T.i32))
            residual_lo = (residual_word << 16).bitcast(fx.Float32)
            residual_hi = (residual_word & fx.Int32(-65536)).bitcast(fx.Float32)
            output_word = (
                fx.Vector.from_elements([residual_lo + reduced_lo, residual_hi + reduced_hi], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            bo.buffer_store(output_word, rsrc(output), pair, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(
        shared_source: Int64,
        tail: Int64,
        residual: Int64,
        local_partial: Int64,
        reduced: Int64,
        output: Int64,
        symmetric: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: Stream = Stream(None),
    ):
        symmetric_bf16_final_reduce(
            shared_source,
            tail,
            residual,
            local_partial,
            reduced,
            output,
            symmetric,
            peers,
            step,
            rank,
            layer,
            value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"},
        ).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"symmetric_bf16_final_reduce_n{numel}_w{npes}_r{region}"
    return launch




class SymmetricBf16Allreduce:
    """Own symmetric storage and fixed-shape launchers for small BF16 reductions."""

    def __init__(
        self,
        sizes: tuple[int, ...],
        rank: int,
        npes: int,
        group=None,
        *,
        final_hidden: int | None = None,
        final_shard_width: int | None = None,
    ) -> None:
        nbytes = symmetric_allreduce_nbytes(sizes, npes)
        self.max_pairs = max(sizes) // 2
        self.peer_buffer = SymmetricPeerBuffer(nbytes, rank=rank, npes=npes, group=group)
        self.launches = tuple(
            build_symmetric_bf16_allreduce(size, npes, self.max_pairs, region) for region, size in enumerate(sizes)
        )
        self.final_launch = None
        if final_hidden is not None and final_shard_width is not None:
            self.final_launch = build_symmetric_bf16_final_reduce(
                sizes[-1],
                final_hidden,
                final_shard_width,
                npes,
                self.max_pairs,
                len(sizes) - 1,
            )
        self.rank = rank

    def reduce(
        self,
        region: int,
        source: torch.Tensor,
        output: torch.Tensor,
        step: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Reduce one fixed-shape contiguous BF16 tensor into ``output``."""

        if source.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("symmetric all-reduce requires BF16 source and output")
        if not source.is_contiguous() or not output.is_contiguous():
            raise ValueError("symmetric all-reduce requires contiguous tensors")
        if source.numel() != output.numel():
            raise ValueError("symmetric all-reduce source/output sizes must match")
        launch = self.launches[region]
        launch(
            source.data_ptr(),
            output.data_ptr(),
            self.peer_buffer.local_address,
            self.peer_buffer.addresses.data_ptr(),
            step.data_ptr(),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        return output

    def reduce_final(
        self,
        shared_source: torch.Tensor,
        tail: torch.Tensor,
        residual: torch.Tensor,
        local_partial: torch.Tensor,
        reduced: torch.Tensor,
        output: torch.Tensor,
        step: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Fuse final local composition, TP reduction, and residual update."""

        if self.final_launch is None:
            raise ValueError("final fused reduction was not configured")
        tensors = (shared_source, tail, residual, local_partial, reduced, output)
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise ValueError("fused final reduction requires BF16 tensors")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("fused final reduction requires contiguous tensors")
        if shared_source.shape != residual.shape or reduced.shape != residual.shape or output.shape != residual.shape:
            raise ValueError("shared, residual, reduced, and output shapes must match")
        if tail.numel() * self.peer_buffer.npes != shared_source.numel():
            raise ValueError("tail must contain one rank-local hidden shard")
        self.final_launch(
            shared_source.data_ptr(),
            tail.data_ptr(),
            residual.data_ptr(),
            local_partial.data_ptr(),
            reduced.data_ptr(),
            output.data_ptr(),
            self.peer_buffer.local_address,
            self.peer_buffer.addresses.data_ptr(),
            step.data_ptr(),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        return output

    def close(self) -> None:
        self.peer_buffer.close()
