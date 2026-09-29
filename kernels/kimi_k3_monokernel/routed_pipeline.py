# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Experimental tile pipeline: SiTU UG producers -> down consumers -> TP.

The grid is resident (256 CTAs, or a validated 512-CTA configuration). A subset first produces tagged BF16 intermediate
tiles; output-owner CTAs consume each ready K fragment. After producing their
assigned UG tiles, producers become consumers too. When the output grid has
fewer than 256 tiles, CTAs without output tiles are assigned production first.
Each output tile owns its route reduction and performs TP immediately, without
a global atomic scatter or grid barrier. Static output ownership requires no
atomic work queue and keeps cross-rank waits independent of task-claim order.
Every rank visits a CTA's output tiles in the same increasing order, so TP
cannot form a task-order cycle even when a CTA owns multiple output tiles.
No consumer can block an unfinished UG task assigned to itself. The same
instance must run on one stream, with a distinct (step, layer) per invocation.
"""

import functools
import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl._mlir.dialects import llvm

from kernels.common import buffer_ops as bo
from kernels.common.act import gate_up_act, situ_params
from kernels.monokernel.layout import CM_DEV, CM_SYS, LAYER_SLOTS
from kernels.monokernel.ops import mxfp4_to_bf16x8, rsrc, uniform, uniform_f32
from kernels.monokernel.packing import pack_mxfp4
from kernels.common.tensor_shim import _run_compiled
from kernels.glm5_monokernel.primitives import mem_realtime, spin_pause
from kernels.moe.mxfp_moe.mxfp4_gemm_common import global_typed_ptr

H, INTER, TOPK = 3584, 384, 16
THREADS, WAVES, GRID = 512, 8, 256
SOFT_FIELDS = (
    "total_ticks",
    "ug_ticks",
    "down_ticks",
    "pack_ticks",
    "tp_push_ticks",
    "tp_wait_reduce_ticks",
    "mid_poll_ticks",
    "entry_tick",
    "ug_end_tick",
    "down_begin_tick",
    "down_end_tick",
    "tp_end_tick",
    "epoch_tag",
    "producer",
    "clock_pair_ticks",
    "consumer",
)


@functools.cache
def build_routed_pipeline(
    samples,
    npes,
    max_pairs,
    producers,
    up_tile,
    prefetch,
    trace,
    handoff,
    sample_group_override,
    tp_mode,
    grid,
    down_tile_override=0,
    up_order="sample",
    soft_profile=0,
    route_prefetch=False,
):
    assert 1 <= samples <= 8 and npes in (1, 8)
    assert 8 <= producers < grid and producers % 8 == 0
    assert up_tile in (16, 32) and prefetch in (1, 2, 4, 8, 16)
    sample_group = min(samples, sample_group_override or 4)
    groups = (samples + sample_group - 1) // sample_group
    assert down_tile_override in (0, 16, 32, 64)
    down_tile = down_tile_override or (16 if groups == 1 or sample_group_override else 32)
    down_rows = down_tile // 16
    down_waves = WAVES // down_rows
    down_chunks = (sample_group * TOPK * 3 + down_waves - 1) // down_waves
    down_tasks = groups * H // down_tile
    assert sample_group_override in (0, 1, 2, 4)
    assert handoff in ("direct", "wave")
    assert tp_mode in ("complete", "push", "reduce_scatter")
    assert up_order in ("sample", "interleaved")
    assert soft_profile in (0, 1, 2)
    up_groups = up_tile // 16
    up_waves = WAVES // (2 * up_groups)
    up_chunks = (H // 128) // up_waves
    assert (H // 128) % up_waves == 0
    up_tasks = samples * TOPK * (INTER // up_tile)
    if route_prefetch:
        assert (samples, sample_group, down_tile, up_tile, prefetch, handoff) == (4, 4, 16, 32, 4, "wave")
    if grid == 768:
        assert route_prefetch and (npes, producers, soft_profile, trace, tp_mode, up_order) == (
            8,
            544,
            0,
            False,
            "complete",
            "interleaved",
        )
    down_batch = 3 if route_prefetch else min(prefetch, down_chunks)
    if down_chunks % down_batch:
        down_batch = math.gcd(down_chunks, down_batch)
    slot_bytes = npes * max_pairs * 8

    @fx.struct
    class SharedStorage:
        x: fx.Array[fx.Float32, H // 2, 16]
        red: fx.Array[fx.Float32, THREADS * 4, 16]

    @flyc.kernel(
        name=f"kimi_routed_pipeline_g{grid}_s{samples}_p{producers}_u{up_tile}_d{down_tile}_f{prefetch}_tp{npes}_trace{int(trace)}_{handoff}_sg{sample_group}_{tp_mode}_{up_order}_{'routeprefetch3_' if route_prefetch else ''}soft{soft_profile}",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        activation: fx.Int64,
        up_weight: fx.Int64,
        up_scale: fx.Int64,
        down_weight: fx.Int64,
        down_scale: fx.Int64,
        ids: fx.Int64,
        weights: fx.Int64,
        mid: fx.Int64,
        partial: fx.Int64,
        reduced: fx.Int64,
        symmetric: fx.Int64,
        peers: fx.Int64,
        step: fx.Int64,
        rank: fx.Int32,
        layer: fx.Int32,
        timeline: fx.Int64,
        durations: fx.Int64,
    ):
        bid = fx.Int32(gpu.block_idx.x)
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid % 64
        wave = uniform(tid // 64)
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        x, red = storage.x.ptr, storage.red.ptr
        mid_r = rsrc(mid)
        tag = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32)) * LAYER_SLOTS + layer + 1
        slot = (tag - 1) & 1
        params = situ_params(
            fx.Float32(4.0), fx.Float32(0.25), fx.Float32(25.0), fx.Float32(0.04), fx.Float32(float("inf"))
        )

        # Coarse mode records wave 0 of each CTA. Detailed mode records all
        # waves of every 31st CTA, spanning XCDs while limiting observer cost.
        soft_active = (wave == 0) if soft_profile == 1 else (bid % 31 == 0)

        def soft_clock():
            value = fx.Int64(0)
            if const_expr(soft_profile != 0):
                if soft_active:
                    value = mem_realtime()
            return value

        def soft_put(column, value):
            if const_expr(soft_profile != 0):
                if soft_active & (lane == 0):
                    fx.ptr_store(
                        fx.Int64(value),
                        global_typed_ptr(durations, T.i64, align=8) + (bid * WAVES + wave) * len(SOFT_FIELDS) + column,
                    )

        soft_entry = soft_clock()
        soft_calibration = soft_clock()
        soft_put(7, soft_entry)
        soft_put(12, tag)
        soft_put(14, soft_calibration - soft_entry)

        def stamp(task, column):
            if const_expr(trace):
                if tid == 0:
                    fx.ptr_store(mem_realtime(), global_typed_ptr(timeline, T.i64, align=8) + task * 6 + column)

        def fragment(w, scale, row_group, k_chunk, k_dim):
            raw = fx.Vector(
                bo.buffer_load(w, ((row_group * (k_dim // 128) + k_chunk) * 64 + lane) * 4, vec_width=4, dtype=T.i32)
            )
            row = row_group * 16 + lane % 16
            scales = fx.Int32(bo.buffer_load(scale, row * (k_dim // 128) + k_chunk, vec_width=1, dtype=T.i32))
            return raw, scales

        def unpack_weight(raw, scales, part):
            scale = ((scales.shrui(fx.Int32(part * 8)) & 255) << 23).bitcast(fx.Float32)
            return mxfp4_to_bf16x8(raw[part], scale)

        def poll_mid(pair):
            # Four (packed BF16 pair, tag) words. The loaded payload is the data
            # consumed by MFMA, not a separate readiness indication.
            def load():
                a = fx.Vector(bo.buffer_load(mid_r, pair * 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
                b = fx.Vector(bo.buffer_load(mid_r, pair * 2 + 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
                return fx.Vector.from_elements([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]], fx.Int32)

            data = load()
            pending = (data[1] != tag) | (data[3] != tag) | (data[5] != tag) | (data[7] != tag)
            while pending:
                spin_pause()
                data = load()
                pending = (data[1] != tag) | (data[3] != tag) | (data[5] != tag) | (data[7] != tag)
            return fx.Vector.from_elements([data[0], data[2], data[4], data[6]], fx.Int32).bitcast(fx.BFloat16)

        def stage_mid_chunk(pair_base, data):
            def load():
                return fx.Vector(
                    bo.buffer_load(mid_r, (pair_base + lane) * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV)
                )

            wait_start = fx.Int64(0)
            wait_ticks = fx.Int64(0)
            if const_expr(soft_profile == 2):
                wait_start = soft_clock()
            while data[1] != tag:
                spin_pause()
                data = load()
            if const_expr(soft_profile == 2):
                wait_ticks = soft_clock() - wait_start
            fx.ptr_store(data[0].bitcast(fx.Float32), x + wave * 64 + lane)
            llvm.call_intrinsic(None, "llvm.amdgcn.wave.barrier", [], [], [])
            rocdl.s_waitcnt(lgkmcnt=0)
            return wait_ticks

        producer_id = (bid + max(0, grid - down_tasks)) % grid
        if producer_id < producers:
            for task in range(producer_id, up_tasks, producers):
                task = fx.Int32(task)
                stamp(task, 0)
                route = task // (INTER // up_tile)
                if const_expr(up_order == "interleaved"):
                    # Publish the first route of every consumer wave early.
                    # S=4/D16: waves start at slots 0 and 8 for every sample.
                    route_slot = route // samples
                    route = (route % samples) * TOPK + (route_slot % 2) * (TOPK // 2) + route_slot // 2
                sample = route // TOPK
                tile = task % (INTER // up_tile)
                expert = uniform(bo.buffer_load(rsrc(ids), route, vec_width=1, dtype=T.i32))
                w = rsrc(up_weight + fx.Int64(expert) * (2 * INTER * H // 2))
                sc = rsrc(up_scale + fx.Int64(expert) * (2 * INTER * H // 32))
                gpu.barrier()
                for batch in range_constexpr((H // 2 + THREADS - 1) // THREADS):
                    pair = tid + batch * THREADS
                    if pair < H // 2:
                        word = fx.Int32(
                            bo.buffer_load(rsrc(activation), sample * (H // 2) + pair, vec_width=1, dtype=T.i32)
                        )
                        fx.ptr_store(word.bitcast(fx.Float32), x + pair)
                gpu.barrier()
                row_group = tile * up_groups + (wave // up_waves) % up_groups + (wave // (WAVES // 2)) * (INTER // 16)
                first_k = (wave % up_waves) * up_chunks
                acc = fx.Vector.filled(4, 0.0, fx.Float32)

                def up_prefetch(begin, count):
                    return [fragment(w, sc, row_group, first_k + begin + j, H) for j in range_constexpr(count)]

                cur = up_prefetch(0, min(prefetch, up_chunks))
                for begin in range_constexpr(0, up_chunks, prefetch):
                    nxt = None
                    if const_expr(begin + prefetch < up_chunks):
                        nxt = up_prefetch(begin + prefetch, min(prefetch, up_chunks - begin - prefetch))
                    for j in range_constexpr(min(prefetch, up_chunks - begin)):
                        raw, scales = cur[j]
                        for part in range_constexpr(4):
                            lhs = unpack_weight(raw, scales, part)
                            rhs = fx.ptr_load(
                                x + (first_k + begin + j) * 64 + (lane // 16) * 4 + part * 16,
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            ).bitcast(fx.BFloat16)
                            acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [lhs, rhs, acc]))
                    cur = nxt
                fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
                gpu.barrier()
                if tid < up_tile // 2:
                    gates, ups = [], []
                    for element in range_constexpr(2):
                        row = tid * 2 + element
                        row16 = row % 16
                        source_lane = 16 * (row16 // 4)
                        g, u = fx.Float32(0.0), fx.Float32(0.0)
                        for kw in range_constexpr(up_waves):
                            source_wave = (row // 16) * up_waves + kw
                            g = g + fx.ptr_load(red + (source_wave * 64 + source_lane) * 4 + row16 % 4)
                            u = u + fx.ptr_load(red + ((source_wave + WAVES // 2) * 64 + source_lane) * 4 + row16 % 4)
                        gates.append(g)
                        ups.append(u)
                    value = (
                        fx.Vector.from_elements(gate_up_act("situv2", gates, ups, params), fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)[0]
                    )
                    pair = route * (INTER // 2) + tile * (up_tile // 2) + tid
                    bo.buffer_store(
                        fx.Vector.from_elements([value, tag], fx.Int32), mid_r, pair * 2, cache_modifier=CM_DEV
                    )
                if const_expr(trace):
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                stamp(task, 2)

        soft_up_end = soft_clock()
        soft_put(8, soft_up_end)
        soft_put(1, (producer_id < producers).select(soft_up_end - soft_entry, fx.Int64(0)))
        soft_put(13, (producer_id < producers).select(fx.Int64(1), fx.Int64(0)))

        def consume_output(job):
            soft_down_begin = soft_clock()
            soft_mid_wait = fx.Int64(0)
            soft_put(9, soft_down_begin)
            soft_put(15, fx.Int64(1))
            stamp(up_tasks + job, 0)
            group = job // (H // down_tile)
            row_tile = job % (H // down_tile)
            row_group = row_tile * down_rows + wave // down_waves

            def down_prefetch(begin, count):
                result = []
                if const_expr(route_prefetch):
                    # All three K128 chunks belong to the same route. Keep its
                    # descriptors and coefficient while issuing the fragments;
                    # readiness and math remain per chunk.
                    unit = (wave % down_waves) * down_chunks + begin
                    local_sample = unit // (TOPK * 3)
                    sample = fx.min(group * sample_group + local_sample, fx.Int32(samples - 1))
                    route = sample * TOPK + (unit // 3) % TOPK
                    expert = uniform(bo.buffer_load(rsrc(ids), route, vec_width=1, dtype=T.i32))
                    weight = uniform_f32(bo.buffer_load(rsrc(weights), route, vec_width=1, dtype=T.f32))
                    w = rsrc(down_weight + fx.Int64(expert) * (H * INTER // 2))
                    sc = rsrc(down_scale + fx.Int64(expert) * (H * INTER // 32))
                for j in range_constexpr(count):
                    if const_expr(route_prefetch):
                        kc = fx.Int32(j)
                    else:
                        unit = (wave % down_waves) * down_chunks + begin + j
                        local_sample = unit // (TOPK * 3)
                        sample = fx.min(group * sample_group + local_sample, fx.Int32(samples - 1))
                        route = sample * TOPK + (unit // 3) % TOPK
                        kc = unit % 3
                        expert = uniform(bo.buffer_load(rsrc(ids), route, vec_width=1, dtype=T.i32))
                        weight = uniform_f32(bo.buffer_load(rsrc(weights), route, vec_width=1, dtype=T.f32))
                        w = rsrc(down_weight + fx.Int64(expert) * (H * INTER // 2))
                        sc = rsrc(down_scale + fx.Int64(expert) * (H * INTER // 32))
                    initial = fx.Vector.filled(2, 0, fx.Int32)
                    if const_expr(handoff == "wave"):
                        initial = fx.Vector(
                            bo.buffer_load(
                                mid_r,
                                (route * (INTER // 2) + kc * 64 + lane) * 2,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                    result.append((fragment(w, sc, row_group, kc, INTER), route, kc, local_sample, weight, initial))
                return result

            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            cur = down_prefetch(fx.Int32(0), down_batch)
            for begin in range(0, down_chunks, down_batch):
                next_begin = fx.min(fx.Int32(begin) + down_batch, fx.Int32(down_chunks - down_batch))
                nxt = down_prefetch(next_begin, down_batch)
                for j in range_constexpr(down_batch):
                    frag, route, kc, local_sample, weight, initial = cur[j]
                    raw, scales = frag
                    if const_expr(handoff == "wave"):
                        chunk_wait = stage_mid_chunk(route * (INTER // 2) + kc * 64, initial)
                        if const_expr(soft_profile == 2):
                            soft_mid_wait = soft_mid_wait + chunk_wait
                    chunk_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for part in range_constexpr(4):
                        lhs = unpack_weight(raw, scales, part)
                        pair = route * (INTER // 2) + kc * 64 + (lane // 16) * 4 + part * 16
                        if const_expr(handoff == "wave"):
                            rhs = fx.ptr_load(
                                x + wave * 64 + (lane // 16) * 4 + part * 16,
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            ).bitcast(fx.BFloat16)
                        else:
                            rhs = poll_mid(pair)
                        chunk_acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [lhs, rhs, chunk_acc]))
                    if const_expr(handoff == "wave"):
                        llvm.call_intrinsic(None, "llvm.amdgcn.wave.barrier", [], [], [])
                    coefficient = (lane % 16 == local_sample).select(weight, fx.Float32(0.0))
                    acc = fx.Vector.from_elements(
                        [acc[e] + chunk_acc[e] * coefficient for e in range_constexpr(4)], fx.Float32
                    )
                    if const_expr(j == 0):
                        if begin == 0:
                            stamp(up_tasks + job, 2)
                cur = nxt
            fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            soft_down_end = soft_clock()
            soft_put(2, soft_down_end - soft_down_begin)
            soft_put(6, soft_mid_wait)
            soft_put(10, soft_down_end)
            if tid < sample_group * down_tile // 2:
                sample = tid // (down_tile // 2)
                row = (tid % (down_tile // 2)) * 2
                sums = []
                for element in range_constexpr(2):
                    r = row + element
                    row16 = r % 16
                    source_lane = sample + 16 * (row16 // 4)
                    total = fx.Float32(0.0)
                    for kw in range_constexpr(down_waves):
                        source_wave = (r // 16) * down_waves + kw
                        total = total + fx.ptr_load(red + (source_wave * 64 + source_lane) * 4 + row16 % 4)
                    sums.append(total)
                packed = fx.Vector.from_elements(sums, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                fx.ptr_store(packed.bitcast(fx.Float32), x + tid)
                global_pair = (group * sample_group + sample) * (H // 2) + row_tile * (down_tile // 2) + row // 2
                if group * sample_group + sample < samples:
                    bo.buffer_store(packed, rsrc(partial), global_pair, cache_modifier=CM_DEV)
            gpu.barrier()
            stamp(up_tasks + job, 3)
            stamp(up_tasks + job, 4)
            soft_pack_end = soft_clock()
            soft_put(3, soft_pack_end - soft_down_end)
            if const_expr(npes == 8):
                # Each wave pushes this completed output tile to one peer.
                peer_word = fx.Vector(bo.buffer_load(rsrc(peers), wave * 2, vec_width=2, dtype=T.i32))
                peer = (fx.Int64(uniform(peer_word[1])) << 32) | fx.Int64(fx.Uint32(uniform(peer_word[0])))
                for batch in range_constexpr((sample_group * down_tile // 2 + 63) // 64):
                    pair = lane + batch * 64
                    sample = pair // (down_tile // 2)
                    send = (pair < sample_group * down_tile // 2) & (group * sample_group + sample < samples)
                    if const_expr(tp_mode in ("push", "reduce_scatter")):
                        owner = (row_tile * (down_tile // 2) // 64) % npes
                        send = send & (wave == owner)
                    if send:
                        word = fx.ptr_load(x + pair).bitcast(fx.Int32)
                        global_pair = (
                            (group * sample_group + sample) * (H // 2)
                            + row_tile * (down_tile // 2)
                            + pair % (down_tile // 2)
                        )
                        bo.buffer_store(
                            fx.Vector.from_elements([word, tag], fx.Int32),
                            rsrc(peer + fx.Int64(slot) * slot_bytes),
                            (rank * max_pairs + global_pair) * 2,
                            cache_modifier=CM_SYS,
                        )
                gpu.barrier()
            soft_push_end = soft_clock()
            soft_put(4, soft_push_end - soft_pack_end)
            if const_expr(npes == 8 and tp_mode == "reduce_scatter"):
                # One owner reduces each cache-line region, then broadcasts the
                # final BF16 result. All ranks visit the same static tile order.
                owner = (row_tile * (down_tile // 2) // 64) % npes
                result_tag = tag + fx.Int32(1 << 30)
                if rank == owner:
                    if tid < sample_group * down_tile // 2:
                        sample = tid // (down_tile // 2)
                        if group * sample_group + sample < samples:
                            pair = (
                                (group * sample_group + sample) * (H // 2)
                                + row_tile * (down_tile // 2)
                                + tid % (down_tile // 2)
                            )

                            def load_parts():
                                words = []
                                for p in range_constexpr(npes):
                                    v = fx.Vector(
                                        bo.buffer_load(
                                            rsrc(symmetric + fx.Int64(slot) * slot_bytes),
                                            (p * max_pairs + pair) * 2,
                                            vec_width=2,
                                            dtype=T.i32,
                                            cache_modifier=CM_DEV,
                                        )
                                    )
                                    words.extend([v[0], v[1]])
                                return fx.Vector.from_elements(words, fx.Int32)

                            vals = load_parts()
                            pending = vals[1] != tag
                            for p in range_constexpr(1, npes):
                                pending = pending | (vals[p * 2 + 1] != tag)
                            while pending:
                                spin_pause()
                                vals = load_parts()
                                pending = vals[1] != tag
                                for p in range_constexpr(1, npes):
                                    pending = pending | (vals[p * 2 + 1] != tag)
                            lo, hi = fx.Float32(0.0), fx.Float32(0.0)
                            for p in range_constexpr(npes):
                                lo = lo + (vals[p * 2] << 16).bitcast(fx.Float32)
                                hi = hi + (vals[p * 2] & fx.Int32(-65536)).bitcast(fx.Float32)
                            packed = fx.Vector.from_elements([lo, hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                            fx.ptr_store(packed.bitcast(fx.Float32), x + tid)
                    gpu.barrier()
                    for batch in range_constexpr((sample_group * down_tile // 2 + 63) // 64):
                        pair_in_tile = lane + batch * 64
                        sample = pair_in_tile // (down_tile // 2)
                        if (pair_in_tile < sample_group * down_tile // 2) & (group * sample_group + sample < samples):
                            pair = (
                                (group * sample_group + sample) * (H // 2)
                                + row_tile * (down_tile // 2)
                                + pair_in_tile % (down_tile // 2)
                            )
                            word = fx.ptr_load(x + pair_in_tile).bitcast(fx.Int32)
                            bo.buffer_store(
                                fx.Vector.from_elements([word, result_tag], fx.Int32),
                                rsrc(peer + fx.Int64(slot) * slot_bytes),
                                (rank * max_pairs + pair) * 2,
                                cache_modifier=CM_SYS,
                            )
                if tid < sample_group * down_tile // 2:
                    sample = tid // (down_tile // 2)
                    if group * sample_group + sample < samples:
                        pair = (
                            (group * sample_group + sample) * (H // 2)
                            + row_tile * (down_tile // 2)
                            + tid % (down_tile // 2)
                        )

                        def load_result():
                            return fx.Vector(
                                bo.buffer_load(
                                    rsrc(symmetric + fx.Int64(slot) * slot_bytes),
                                    (owner * max_pairs + pair) * 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )

                        value = load_result()
                        while value[1] != result_tag:
                            spin_pause()
                            value = load_result()
                        bo.buffer_store(value[0], rsrc(reduced), pair, cache_modifier=CM_DEV)
            if const_expr(npes == 1 or tp_mode == "complete"):
                if tid < sample_group * down_tile // 2:
                    sample = tid // (down_tile // 2)
                    if group * sample_group + sample < samples:
                        pair = (
                            (group * sample_group + sample) * (H // 2)
                            + row_tile * (down_tile // 2)
                            + tid % (down_tile // 2)
                        )
                        if const_expr(npes == 8):

                            def load_peers():
                                words = []
                                for p in range_constexpr(npes):
                                    v = fx.Vector(
                                        bo.buffer_load(
                                            rsrc(symmetric + fx.Int64(slot) * slot_bytes),
                                            (p * max_pairs + pair) * 2,
                                            vec_width=2,
                                            dtype=T.i32,
                                            cache_modifier=CM_DEV,
                                        )
                                    )
                                    words.extend([v[0], v[1]])
                                return fx.Vector.from_elements(words, fx.Int32)

                            vals = load_peers()
                            pending = vals[1] != tag
                            for p in range_constexpr(1, npes):
                                pending = pending | (vals[p * 2 + 1] != tag)
                            while pending:
                                spin_pause()
                                vals = load_peers()
                                pending = vals[1] != tag
                                for p in range_constexpr(1, npes):
                                    pending = pending | (vals[p * 2 + 1] != tag)
                            lo, hi = fx.Float32(0.0), fx.Float32(0.0)
                            for p in range_constexpr(npes):
                                word = vals[p * 2]
                                lo = lo + (word << 16).bitcast(fx.Float32)
                                hi = hi + (word & fx.Int32(-65536)).bitcast(fx.Float32)
                            packed = fx.Vector.from_elements([lo, hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                        else:
                            packed = fx.ptr_load(x + tid).bitcast(fx.Int32)
                        bo.buffer_store(packed, rsrc(reduced), pair, cache_modifier=CM_DEV)
            if const_expr(trace):
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
            stamp(up_tasks + job, 5)
            soft_tp_end = soft_clock()
            soft_put(5, soft_tp_end - soft_push_end)
            soft_put(11, soft_tp_end)

        if const_expr(down_tasks <= grid):
            if bid < down_tasks:
                consume_output(bid)
        else:
            for output_job in range(bid, down_tasks, grid):
                consume_output(fx.Int32(output_job))
        soft_put(0, soft_clock() - soft_entry)

    @flyc.jit
    def launch(
        activation: fx.Int64,
        up_weight: fx.Int64,
        up_scale: fx.Int64,
        down_weight: fx.Int64,
        down_scale: fx.Int64,
        ids: fx.Int64,
        weights: fx.Int64,
        mid: fx.Int64,
        partial: fx.Int64,
        reduced: fx.Int64,
        symmetric: fx.Int64,
        peers: fx.Int64,
        step: fx.Int64,
        rank: fx.Int32,
        layer: fx.Int32,
        timeline: fx.Int64,
        durations: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            activation,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
            ids,
            weights,
            mid,
            partial,
            reduced,
            symmetric,
            peers,
            step,
            rank,
            layer,
            timeline,
            durations,
        ).launch(grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


class RoutedTilePipeline:
    def __init__(
        self,
        samples,
        ug,
        ug_scale,
        dn,
        dn_scale,
        *,
        npes=1,
        rank=0,
        max_pairs=None,
        producers=None,
        up_tile=32,
        prefetch=None,
        trace=False,
        handoff="wave",
        sample_group=0,
        tp_mode="complete",
        grid=256,
        down_tile=0,
        up_order="sample",
        soft_profile=0,
        route_prefetch=False,
    ):
        if samples not in (1, 2, 4, 8) or npes not in (1, 8) or not 0 <= rank < npes:
            raise ValueError("routed pipeline supports S=1/2/4/8 and TP1/8 with a valid rank")
        if max_pairs is not None and max_pairs < samples * H // 2:
            raise ValueError("routed pipeline mailbox is too small")
        for name, tensor, shape in (
            ("ug", ug, (896, 2 * INTER, H // 2)),
            ("ug_scale", ug_scale, (896, 2 * INTER, H // 32)),
            ("dn", dn, (896, H, INTER // 2)),
            ("dn_scale", dn_scale, (896, H, INTER // 32)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
                or tensor.device != ug.device
            ):
                raise ValueError(f"{name} must be contiguous uint8 {shape} on {ug.device}")
        self.npes, self.device = npes, ug.device
        if soft_profile not in (0, 1, 2):
            raise ValueError("soft_profile must be 0 (off), 1 (coarse), or 2 (sampled wave detail)")
        if soft_profile and (
            samples != 4 or sample_group or down_tile not in (0, 16) or handoff != "wave" or tp_mode != "complete"
        ):
            raise ValueError(
                "soft profiling currently supports S4, D16, default sample grouping, wave handoff, complete TP"
            )
        self.soft_profile = soft_profile
        properties = torch.cuda.get_device_properties(ug.device)
        if not properties.gcnArchName.startswith("gfx950") or properties.multi_processor_count < 256:
            raise ValueError("the resident tile pipeline currently requires gfx950")
        producers = producers if producers is not None else {1: 192, 2: 192, 4: 192, 8: 224}[samples]
        prefetch = prefetch if prefetch is not None else (4 if samples == 4 else 8)
        if up_order not in ("sample", "interleaved"):
            raise ValueError("up_order must be sample or interleaved")
        if (
            grid == 512
            and (up_order != "sample" or tp_mode != "complete")
            and (samples, producers, prefetch) != (4, 256, 4)
        ):
            raise ValueError("grid 512 reordered/owner TP is validated only for S4/P256/F4")
        if grid == 512 and (
            (samples, producers, prefetch) not in {(2, 384, 8), (4, 256, 4), (4, 384, 4), (4, 448, 4), (8, 448, 4)}
            or trace
            or handoff != "wave"
            or up_tile != 32
            or sample_group != 0
            or tp_mode not in ("complete", "reduce_scatter")
            or down_tile != 0
        ):
            raise ValueError(
                "grid 512 requires a validated resident configuration: S2/P384/F8, S4/P256-or-384-or-448/F4, "
                "or S8/P448/F4, with wave handoff, U32, default sample grouping/down tile and no trace"
            )
        if route_prefetch and (
            samples != 4
            or up_tile != 32
            or prefetch != 4
            or handoff != "wave"
            or sample_group != 0
            or down_tile != 0
            or tp_mode != "complete"
            or up_order != "interleaved"
            or trace
            or (grid, producers) not in ((512, 256), (768, 544))
        ):
            raise ValueError(
                "route prefetch requires S4/U32/D16/F4, default sample grouping, interleaved wave handoff, "
                "complete TP, no trace, and grid/producers 512/256 or 768/544"
            )
        if grid == 768 and (not route_prefetch or npes != 8 or soft_profile != 0):
            raise ValueError("grid 768 requires TP8 route prefetch without soft profiling (3 resident CTA/CU)")
        self.producers, self.prefetch, self.handoff = producers, prefetch, handoff
        self.route_prefetch = route_prefetch
        self.tp_mode = tp_mode
        self.up_order = up_order
        self.grid = grid
        if grid not in (256, 512, 768):
            raise ValueError("the experimental grid must be 256, 512 or 768")
        self.up_tasks = samples * TOPK * (INTER // up_tile)
        group = min(samples, sample_group or 4)
        groups = (samples + group - 1) // group
        if down_tile not in (0, 16, 32, 64):
            raise ValueError("down_tile must be 0, 16, 32 or 64")
        self.down_tile = down_tile or (16 if groups == 1 or sample_group else 32)
        self.down_tasks = groups * H // self.down_tile
        self.samples, self.rank = samples, rank
        self.ug, self.dn = pack_mxfp4(ug), pack_mxfp4(dn)
        self.ug_scale, self.dn_scale = ug_scale, dn_scale
        self.mid = torch.zeros(samples, TOPK, INTER // 2, 2, dtype=torch.int32, device=ug.device)
        self.timeline = torch.zeros(self.up_tasks + self.down_tasks, 6, dtype=torch.int64, device=ug.device)
        self.soft_durations = (
            torch.zeros(grid, WAVES, len(SOFT_FIELDS), dtype=torch.int64, device=ug.device) if soft_profile else None
        )
        self.launch = build_routed_pipeline(
            samples,
            npes,
            max_pairs or samples * H // 2,
            producers,
            up_tile,
            prefetch,
            trace,
            handoff,
            sample_group,
            tp_mode,
            grid,
            down_tile,
            up_order,
            soft_profile,
            route_prefetch,
        )

    def __call__(self, activation, ids, weights, partial, reduced, step, layer, *, symmetric=0, peers=None):
        if not 0 <= layer < LAYER_SLOTS:
            raise ValueError(f"layer must be in [0, {LAYER_SLOTS}) and unique within each step")
        if self.npes == 8 and (not symmetric or peers is None):
            raise ValueError("TP8 pipeline requires symmetric peer mailboxes")
        for name, tensor, shape, dtype in (
            ("activation", activation, (self.samples, H), torch.bfloat16),
            ("ids", ids, (self.samples, TOPK), torch.int32),
            ("weights", weights, (self.samples, TOPK), torch.float32),
            ("partial", partial, (self.samples, H), torch.bfloat16),
            ("reduced", reduced, (self.samples, H), torch.bfloat16),
            ("step", step, (1,), torch.int32),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.device != self.device
            ):
                raise ValueError(f"{name} must be contiguous {dtype} {shape} on {self.device}")
        _run_compiled(
            self.launch,
            activation.data_ptr(),
            self.ug.data_ptr(),
            self.ug_scale.data_ptr(),
            self.dn.data_ptr(),
            self.dn_scale.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
            self.mid.data_ptr(),
            partial.data_ptr(),
            reduced.data_ptr(),
            symmetric,
            peers.data_ptr() if peers is not None else 0,
            step.data_ptr(),
            self.rank,
            layer,
            self.timeline.data_ptr(),
            self.soft_durations.data_ptr() if self.soft_durations is not None else 0,
            torch.cuda.current_stream(),
        )
        return reduced
