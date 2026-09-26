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


class SymmetricBf16Allreduce:
    """Own symmetric storage and fixed-shape launchers for small BF16 reductions."""

    def __init__(self, sizes: tuple[int, ...], rank: int, npes: int, group=None) -> None:
        nbytes = symmetric_allreduce_nbytes(sizes, npes)
        self.max_pairs = max(sizes) // 2
        self.peer_buffer = SymmetricPeerBuffer(nbytes, rank=rank, npes=npes, group=group)
        self.launches = tuple(
            build_symmetric_bf16_allreduce(size, npes, self.max_pairs, region) for region, size in enumerate(sizes)
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

    def close(self) -> None:
        self.peer_buffer.close()
