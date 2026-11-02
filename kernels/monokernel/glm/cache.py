# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Deferred KV/PE commit after all readers retire."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import KV_LORA, PE_DIM
from kernels.monokernel.glm.context import StageContext
from kernels.monokernel.glm.interfaces import KvCache
from kernels.monokernel.glm.layout import BLOCKS, fp8_kv_upper_pair_lane, fp8_pe_upper_pair_lane
from kernels.monokernel.ops import bpermute_i32
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import xshfl as _xshfl


@flyc.jit
def cache_commit(cache: KvCache, ctx: StageContext):
    plan = ctx.plan
    CACHE_READERS_JOINED = plan.CACHE_READERS_JOINED
    DEFER_OFF_CACHE = plan.DEFER_OFF_CACHE
    QK_DIM = plan.QK_DIM
    ROW_TILES = plan.ROW_TILES
    S = plan.S
    TILE_SCRATCH_BYTES = plan.TILE_SCRATCH_BYTES
    cache_fp8 = plan.cache_fp8
    total_samples = plan.total_samples
    bid = ctx.execution.thread.block_id
    full_bid = ctx.execution.thread.global_block_id
    full_scratch = ctx.execution.workspace.all_scratch
    full_slots = ctx.execution.tokens.all_slots
    kv_buffer = cache.kv_buffer
    lane = ctx.execution.thread.lane_id
    rope_buffer = cache.rope_buffer
    row_offset = ctx.execution.thread.row_offset
    tid = ctx.execution.thread.thread_id
    row_slot = ctx.ops.memory.row_slot
    mb = ctx.ops.memory.mb
    put = ctx.ops.memory.put
    get = ctx.ops.memory.get
    getf = ctx.ops.memory.getf
    get2 = ctx.ops.memory.get2

    if const_expr(DEFER_OFF_CACHE):
        if const_expr(not CACHE_READERS_JOINED):
            # Every reader finishes before persistent cache storage is overwritten.
            # This second barrier has a distinct slot at the same invocation tag.
            fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
            gpu.barrier()
            commit_arrivals = full_scratch + fx.Int64(ROW_TILES * TILE_SCRATCH_BYTES + (BLOCKS + 1) * 8)
            if tid == 0:
                put(commit_arrivals, full_bid, fx.Int32(1))
            if full_bid == 0:
                if tid < BLOCKS:
                    get(commit_arrivals, tid)
                    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                gpu.barrier()
                if tid == 0:
                    put(commit_arrivals, BLOCKS, fx.Int32(1))
            if tid == 0:
                get(commit_arrivals, BLOCKS)
            gpu.barrier()
            fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
        if bid < S:
            commit_slot = row_slot(bid)
            later_slot = fx.Int32(
                bo.buffer_load(_rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32)
            )
            overwritten = (lane > row_offset + bid) & (lane < total_samples) & (later_slot == commit_slot)
            overwritten_mask = fx.Int64(rocdl.ballot(T.i64, overwritten))
            commit_owner = (commit_slot >= 0) & (overwritten_mask == 0)
            if commit_owner:
                # Buffer instructions use a 32-bit byte offset. Rebase
                # each cache row with 64-bit arithmetic before storing.
                cache_element_bytes = 1 if const_expr(cache_fp8) else 2
                commit_kv = kv_buffer + fx.Int64(commit_slot) * (QK_DIM * cache_element_bytes)
                commit_pe = rope_buffer + fx.Int64(commit_slot) * (QK_DIM * cache_element_bytes)
                committed_kv = getf(mb("kvnew"), bid * KV_LORA + tid)
                if const_expr(cache_fp8):
                    kv_pair8 = fx.Int32(
                        rocdl.cvt_pk_fp8_f32(T.i32, committed_kv, _xshfl(committed_kv, 1), fx.Int32(0), False)
                    ) & fx.Int32(0xFFFF)
                    kv_upper8 = bpermute_i32(fp8_kv_upper_pair_lane(lane) * 4, kv_pair8)
                    if lane % 4 == 0:
                        bo.buffer_store(
                            kv_pair8 | (kv_upper8 << 16),
                            _rsrc(commit_kv),
                            tid // 4,
                        )
                else:
                    bo.buffer_store(committed_kv.to(fx.BFloat16), _rsrc(commit_kv), tid)
                if tid < PE_DIM // 2:
                    committed_pe0, committed_pe1 = get2(mb("penew"), bid * PE_DIM + tid * 2)
                    if const_expr(cache_fp8):
                        pe_pair8 = fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, committed_pe0, committed_pe1, fx.Int32(0), False)
                        ) & fx.Int32(0xFFFF)
                        pe_upper8 = bpermute_i32(fp8_pe_upper_pair_lane(lane) * 4, pe_pair8)
                        if lane % 2 == 0:
                            bo.buffer_store(
                                pe_pair8 | (pe_upper8 << 16),
                                _rsrc(commit_pe),
                                KV_LORA // 4 + tid // 2,
                            )
                    else:
                        bo.buffer_store(
                            fx.Vector.from_elements([committed_pe0, committed_pe1], fx.Float32).to(fx.BFloat16),
                            _rsrc(commit_pe),
                            KV_LORA + tid * 2,
                        )
