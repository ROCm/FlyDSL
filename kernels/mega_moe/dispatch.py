# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""Compact dispatch path for MegaMoE v2 stage1."""

from enum import IntEnum

import mori.ir.flydsl as mori_shmem

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from kernels.comm import communication_ops_utils as comm_ops
from kernels.common import buffer_ops

from .gemm_util import wait_all
from .mxfp4_smoothquant import (
    copy_mxfp4_row_and_compute_smoothquant_scale_wave,
    fused_prepare_bf16_smoothquant_row_wave,
)


class DispatchSlot(IntEnum):
    PAIR_BASE = 0
    P2P_TOKEN = 1
    P2P_SCALE = 2
    P2P_WEIGHT = 3
    P2P_SRCMAP = 4
    SORTED_EXPERT = 5
    TILE_ROW_BASE = 6
    NUM_VALID = 7
    SRCMAP = 8
    LOCAL_HIST = 9
    COUNT_MATRIX = 10
    P2P_COUNT_MATRIX = 11
    COUNT_DONE = 12
    P2P_COUNT_DONE = 13
    TASK_ROW_BASE = 14
    LOCAL_CURSOR = 15
    P2P_PAYLOAD_READY = 16
    PAIR_ORDER = 17
    P2P_TASK_ROW_BASE = 18
    P2P_PLAN_READY = 19
    PLAN_READY = 20
    PAIR_READY = 21
    ENTRY_COUNT = 22
    EPOCH_GATE = 23
    PAIR_ORDER_READY = 24
    WORK_HEAD = 25
    GROUP_DONE = 26
    RUNNING = 27
    P2P_RUNNING = 28
    LAUNCH_READY = 29
    P2P_LAUNCH_READY = 30
    MAX_EXPERT_TILES = 31
    PAYLOAD_CHUNK_DONE = 32
    TILE_READY = 33
    P2P_TILE_READY = 34
    TILE_EXPECTED = 35
    ACTIVE_PAYLOAD_BLOCKS = 36
    PAYLOAD_READY_ROWS = 37
    P2P_PAYLOAD_READY_ROWS = 38
    PAYLOAD_BLOCKS_PER_DESTINATION = 39
    TOTAL_RECV = 40
    DEST_COUNTER = 41
    RECV_NUM = 42
    P2P_RECV_NUM = 43
    ROW_SCALE = 44
    P2P_ROW_SCALE = 45


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


@flyc.jit
def _wave_inclusive_scan_i32(value, lane):
    value_raw = value.ir_value()
    zero_raw = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.rocdl.update_dpp(T.i32, zero_raw, value_raw, dpp, 0xF, 0xF, True)
        value = (lane >= fx.Int32(shift)).select(value + fx.Int32(remote), value)
        value_raw = value.ir_value()
    source16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(T.i32, source16 * fx.Int32(4), value)
    value = (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)
    source32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
    remote32 = fx.rocdl.ds_bpermute(T.i32, source32 * fx.Int32(4), value)
    return (lane >= fx.Int32(32)).select(value + fx.Int32(remote32), value)


@flyc.jit
def _wave_reduce_max_i32(value, lane):
    for distance in (1, 2, 4, 8, 16, 32):
        peer = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane ^ fx.Int32(distance)) * fx.Int32(4), value))
        value = (peer > value).select(peer, value)
    return value


@flyc.jit
def _increment_i32(rsrc, index):
    value = buffer_ops.buffer_load(rsrc, index, vec_width=1, dtype=fx.Int32)
    buffer_ops.buffer_store(value + fx.Int32(1), rsrc, index)


# fmt: off
@flyc.jit
def _configure_payload_blocks(
    addr_block_counts, addr_active_blocks, lane, *, fz_npes,
    dispatch_blocks, payload_block_cap,
):
# fmt: on
    crfa = buffer_ops.create_buffer_resource_from_addr
    block_counts = crfa(addr_block_counts)
    if lane == fx.Int32(0):
        active_blocks = fx.Int32(
            min(payload_block_cap, dispatch_blocks // fz_npes)
        )
        for destination in range_constexpr(fz_npes):
            buffer_ops.buffer_store(active_blocks, block_counts, fx.Int32(destination))
        buffer_ops.buffer_store(
            active_blocks * fx.Int32(fz_npes),
            crfa(addr_active_blocks),
            fx.Int32(0),
        )


# fmt: off
@flyc.jit
def _store_expert_metadata(
    addr_sorted_expert, addr_tile_row_base, addr_srcmap, ge, local_row_base, total_count, num_tiles,
    padded_rows, *, fz_tile_m, invalid_source,
):
# fmt: on
    crfa = buffer_ops.create_buffer_resource_from_addr
    sorted_expert = crfa(addr_sorted_expert)
    tile_row_base = crfa(addr_tile_row_base)
    srcmap = crfa(addr_srcmap)
    base_tile = local_row_base // fx.Int32(fz_tile_m)
    for tile in range(fx.Int32(0), num_tiles, 1):
        metadata_index = base_tile + tile
        buffer_ops.buffer_store(ge, sorted_expert, metadata_index)
        buffer_ops.buffer_store(local_row_base + tile * fx.Int32(fz_tile_m), tile_row_base, metadata_index)
    padding = padded_rows - total_count
    for pad in range(fx.Int32(0), padding, 1):
        buffer_ops.buffer_store(fx.Int32(invalid_source), srcmap, local_row_base + total_count + pad)


@flyc.jit
def _copy_token_row(source_rsrc, destination_rsrc, lane, *, fz_safe_end_i32, fz_n_i32):
    lane_offset = lane * fx.Int32(4)
    if const_expr(fz_safe_end_i32 > 0):
        for column in range(lane_offset, fz_safe_end_i32, 512):
            value0 = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
            value1 = buffer_ops.buffer_load(source_rsrc, column + fx.Int32(256), vec_width=4, dtype=fx.Int32)
            buffer_ops.buffer_store(value0, destination_rsrc, column)
            buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
    if const_expr(fz_safe_end_i32 < fz_n_i32):
        for column in range(lane_offset + fz_safe_end_i32, fz_n_i32, 256):
            value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
            buffer_ops.buffer_store(value, destination_rsrc, column)


@flyc.jit
def fused_prepare(
    source_addr,
    destination_addr,
    smooth_rsrc,
    smooth_scale_rsrc,
    source_row,
    destination_row,
    expert,
    lane,
    *,
    smoothquant_mode,
    model_dim,
    payload_row_bytes,
    fz_safe_end_i32,
    fz_n_i32,
):
    """Prepare one dispatch row in its final remote payload layout.

    ``none`` preserves the existing pre-quantized row copy. ``bf16_route``
    consumes token-major BF16 and fuses expert-aware SmoothQuant with the
    prepare copy, returning the row's dynamic INT8 scale.
    """
    crfa = buffer_ops.create_buffer_resource_from_addr
    if const_expr(smoothquant_mode == "bf16_route"):
        return fused_prepare_bf16_smoothquant_row_wave(
            crfa(source_addr),
            smooth_rsrc,
            smooth_scale_rsrc,
            crfa(destination_addr),
            source_row,
            destination_row,
            expert,
            cols=model_dim,
        )

    source_row_rsrc = crfa(
        source_addr + fx.Int64(source_row) * fx.Int64(payload_row_bytes)
    )
    destination_row_rsrc = crfa(
        destination_addr
        + fx.Int64(destination_row) * fx.Int64(payload_row_bytes)
    )
    _copy_token_row(
        source_row_rsrc,
        destination_row_rsrc,
        lane,
        fz_safe_end_i32=fz_safe_end_i32,
        fz_n_i32=fz_n_i32,
    )
    return fx.Float32(0.0)




@flyc.jit
def _publish_tile_range(p_tile_ready, destination, destination_base, row_begin, row_end, rows_per_tile):
    if row_end > row_begin:
        crfa = buffer_ops.create_buffer_resource_from_addr
        comm_ops.fence_system_release()
        remote_tile_ready = buffer_ops.buffer_load(crfa(p_tile_ready), destination, vec_width=1, dtype=fx.Int64)
        first_tile = (destination_base + row_begin) // rows_per_tile
        last_tile = (destination_base + row_end - fx.Int32(1)) // rows_per_tile
        for tile in range(first_tile, last_tile + fx.Int32(1), 1):
            comm_ops.atomic_add_system(remote_tile_ready + fx.Int64(tile) * fx.Int64(4), fx.Int32(1))


# fmt: off


# fmt: off
@flyc.jit
def emit_exact_total_recv(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_total_experts, fz_rank, addr_disp,
    i32_cur_tok, addr_in_idx,
):
# fmt: on
    """Publish the exact number of distinct source tokens received by this rank.

    Each source token builds one destination bitmask in O(topk), then contributes
    at most once to every destination.  Peers exchange those exact counts and
    overwrite TOTAL_RECV, so the host never needs a pre-zero/fill launch.
    """
    assert fz_npes <= 8, "destination mask is encoded in one i32"
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    a_dest_counter = dp(DispatchSlot.DEST_COUNTER)
    a_recv_num = dp(DispatchSlot.RECV_NUM)
    p_recv_num = dp(DispatchSlot.P2P_RECV_NUM)
    a_total_recv = dp(DispatchSlot.TOTAL_RECV)
    r_idx = crfa(addr_in_idx)
    tid = fx.thread_idx.x
    total_threads = fx.Int32(num_waves * 64)

    for token in range(tid, i32_cur_tok, total_threads):
        destination_mask = fx.Int32(0)
        for slot in range_constexpr(fz_k):
            expert = buffer_ops.buffer_load(
                r_idx, token * fx.Int32(fz_k) + fx.Int32(slot),
                vec_width=1, dtype=fx.Int32,
            )
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
            safe_expert = valid.select(expert, fx.Int32(0))
            destination = safe_expert // fx.Int32(fz_epr)
            bit = fx.Int32(1) << destination
            destination_mask = valid.select(destination_mask | bit, destination_mask)
        for destination in range_constexpr(fz_npes):
            if (destination_mask & fx.Int32(1 << destination)) != fx.Int32(0):
                comm_ops.atomic_add_agent(
                    a_dest_counter + fx.Int64(destination * 4), fx.Int32(1)
                )
    wait_all()
    fx.barrier()

    if tid < fx.Int32(fz_npes):
        destination = tid
        count = buffer_ops.buffer_load(
            crfa(a_dest_counter), destination, vec_width=1, dtype=fx.Int32
        )
        remote_recv = buffer_ops.buffer_load(
            crfa(p_recv_num), destination, vec_width=1, dtype=fx.Int64
        )
        comm_ops.fence_system_release()
        comm_ops.store_i32_system(remote_recv, fx.Int32(fz_rank), count)
    fx.barrier()

    if tid < fx.Int32(fz_npes):
        source = tid
        mori_shmem.int32_wait_until_greater_than(
            a_recv_num + fx.Int64(source) * fx.Int64(4), fx.Int32(-1)
        )
    fx.barrier()
    comm_ops.fence_system_acquire()

    if tid == fx.Int32(0):
        total_recv = fx.Int32(0)
        for source in range_constexpr(fz_npes):
            total_recv = total_recv + buffer_ops.buffer_load(
                crfa(a_recv_num), fx.Int32(source), vec_width=1, dtype=fx.Int32
            )
            buffer_ops.buffer_store(fx.Int32(-1), crfa(a_recv_num), fx.Int32(source))
        buffer_ops.buffer_store(total_recv, crfa(a_total_recv), fx.Int32(0))
        for destination in range_constexpr(fz_npes):
            buffer_ops.buffer_store(
                fx.Int32(0), crfa(a_dest_counter), fx.Int32(destination)
            )
    wait_all()
    fx.barrier()


# fmt: off
@flyc.jit
def emit_direct_fixed_slot_payload(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_cap, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32,
    fz_scale_n_i32, fz_enable_scales, fz_route_payload, smoothquant_mode,
    smooth_rsrc, smooth_scale_rsrc, model_dim,
    addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    i32_cur_tok, dispatch_blocks, producer_slot, parity, expected,
    destination_groups,
):
# fmt: on
    """Allocate and publish routes directly into destination fixed slots."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    p_running = dp(DispatchSlot.P2P_RUNNING)
    p_source_done = dp(DispatchSlot.P2P_COUNT_DONE)
    a_producer_done = dp(DispatchSlot.GROUP_DONE)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    # Expert-aware SmoothQuant already has one producer wave for every route
    # at the decode geometry.  Let those waves own routes directly instead of
    # making two destination-parity groups scan the route list independently.
    # The remote running counter still selects the destination fixed slot, and
    # the single completion leader publishes all destinations only after every
    # active producer has drained its payload stores.
    assert destination_groups in (1, 2, 4, 8), (
        "direct fixed-slot destination groups must be 1, 2, 4, or 8"
    )
    assert dispatch_blocks % destination_groups == 0, "direct fixed-slot dispatch needs even producer groups"
    producers_per_group = dispatch_blocks // destination_groups
    completion_cohort = 8
    completion_cohorts = (
        producers_per_group + completion_cohort - 1
    ) // completion_cohort
    producer_group = producer_slot % fx.Int32(destination_groups)
    group_slot = producer_slot // fx.Int32(destination_groups)
    route = group_slot * fx.Int32(num_waves) + warp
    route_stride = fx.Int32(producers_per_group * num_waves)
    route_limit = i32_cur_tok * fx.Int32(fz_k)
    r_idx = crfa(addr_in_idx)
    r_wts = crfa(addr_in_wts)
    r_scales = crfa(addr_in_sc)

    def process_route(wk):
        source_token = wk // fx.Int32(fz_k)
        topk_slot = wk - source_token * fx.Int32(fz_k)
        global_expert_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            global_expert_lane = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
        global_expert = fx.Int32(fx.rocdl.readfirstlane(T.i32, global_expert_lane))
        valid_expert = (global_expert >= fx.Int32(0)) & (
            global_expert < fx.Int32(fz_total_experts)
        )
        safe_expert = valid_expert.select(global_expert, fx.Int32(0))
        destination = safe_expert // fx.Int32(fz_epr)
        local_expert = safe_expert - destination * fx.Int32(fz_epr)
        offset_lane = fx.Int32(0)
        assigned = valid_expert & (
            destination % fx.Int32(destination_groups) == producer_group
        )
        if lane == fx.Int32(0):
            if assigned:
                remote_running = buffer_ops.buffer_load(
                    crfa(p_running), destination, vec_width=1, dtype=fx.Int64
                )
                offset_lane = fx.Int32(
                    comm_ops.atomic_add_system(
                        remote_running + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
        expert_offset = fx.Int32(fx.rocdl.readlane(T.i32, offset_lane, 0))
        publish = assigned & (expert_offset < fx.Int32(fz_cap))
        payload_row = local_expert * fx.Int32(fz_cap) + expert_offset

        if publish:
            remote_token = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
            source_row = (
                source_token
                if const_expr(smoothquant_mode == "bf16_route")
                else wk
                if const_expr(fz_route_payload)
                else source_token
            )
            row_scale = fused_prepare(
                addr_in_tok,
                remote_token,
                smooth_rsrc,
                smooth_scale_rsrc,
                source_row,
                payload_row,
                global_expert,
                lane,
                smoothquant_mode=smoothquant_mode,
                model_dim=model_dim,
                payload_row_bytes=fz_nbytes,
                fz_safe_end_i32=(fz_n_i32 // 512) * 512,
                fz_n_i32=fz_n_i32,
            )

            if const_expr(fz_enable_scales):
                if const_expr(smoothquant_mode == "bf16_route"):
                    if lane == fx.Int32(0):
                        remote_scale = buffer_ops.buffer_load(
                            crfa(p_sc), destination, vec_width=1, dtype=fx.Int64
                        )
                        buffer_ops.buffer_store(
                            row_scale,
                            crfa(remote_scale),
                            payload_row,
                        )
                elif lane < fx.Int32(fz_scale_n_i32):
                    scale = buffer_ops.buffer_load(
                        r_scales, source_row * fx.Int32(fz_scale_n_i32) + lane,
                        vec_width=1, dtype=fx.Int32,
                    )
                    remote_scale = buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(scale, crfa(remote_scale), payload_row * fx.Int32(fz_scale_n_i32) + lane)

            if lane == fx.Int32(0):
                weight = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                remote_weights = buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64)
                remote_srcmap = buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64)
                buffer_ops.buffer_store(weight_bits, crfa(remote_weights), payload_row)
                buffer_ops.buffer_store(source_encoding, crfa(remote_srcmap), payload_row)

    for wk in range(route, route_limit, route_stride):
        process_route(wk)

    wait_all()
    fx.barrier()
    if tid == fx.Int32(0):
        active_producers = (
            route_limit + fx.Int32(num_waves - 1)
        ) // fx.Int32(num_waves)
        active_producers = (active_producers < fx.Int32(producers_per_group)).select(
            active_producers, fx.Int32(producers_per_group)
        )
        producer_active = group_slot < active_producers
        if producer_active:
            comm_ops.store_i32_system(
                a_producer_done, producer_slot, expected
            )
        cohort = group_slot // fx.Int32(completion_cohort)
        cohort_leader = producer_active & (
            group_slot == cohort * fx.Int32(completion_cohort)
        )
        if cohort_leader:
            for member in range_constexpr(completion_cohort):
                member_group_slot = (
                    cohort * fx.Int32(completion_cohort) + fx.Int32(member)
                )
                if member_group_slot < active_producers:
                    member_slot = (
                        producer_group
                        + member_group_slot * fx.Int32(destination_groups)
                    )
                    mori_shmem.int32_wait_until_equals(
                        a_producer_done + fx.Int64(member_slot) * fx.Int64(4),
                        expected,
                    )
            comm_ops.fence_system_acquire()
            summary_slot = (
                fx.Int32(dispatch_blocks)
                + producer_group * fx.Int32(completion_cohorts)
                + cohort
            )
            comm_ops.store_i32_system(a_producer_done, summary_slot, expected)
        if group_slot == fx.Int32(0):
            active_cohorts = (
                active_producers + fx.Int32(completion_cohort - 1)
            ) // fx.Int32(completion_cohort)
            for cohort_index in range(fx.Int32(0), active_cohorts, 1):
                summary_slot = (
                    fx.Int32(dispatch_blocks)
                    + producer_group * fx.Int32(completion_cohorts)
                    + cohort_index
                )
                mori_shmem.int32_wait_until_equals(
                    a_producer_done + fx.Int64(summary_slot) * fx.Int64(4),
                    expected,
                )
            comm_ops.fence_system_acquire()
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            for destination in range_constexpr(fz_npes):
                if producer_group == fx.Int32(destination % destination_groups):
                    remote_done = buffer_ops.buffer_load(
                        crfa(p_source_done), fx.Int32(destination), vec_width=1, dtype=fx.Int64
                    )
                    comm_ops.store_i32_system(remote_done, done_index, expected)


@flyc.jit
def emit_direct_fixed_slot_finalize(
    *, fz_npes, fz_epr, fz_cap, fz_mtpr, fz_rank, fz_tile_m, addr_disp, parity, expected,
):
    """Finalize local fixed slots as soon as every source publishes this destination."""
    assert 0 < fz_epr <= 64, "direct fixed-slot finalize requires 1..64 experts per rank"
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_running = dp(DispatchSlot.RUNNING)
    a_source_done = dp(DispatchSlot.COUNT_DONE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    if warp == fx.Int32(0):
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            mori_shmem.int32_wait_until_equals(a_source_done + fx.Int64(done_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()

        valid_expert = lane < fx.Int32(fz_epr)
        safe_expert = valid_expert.select(lane, fx.Int32(0))
        count = buffer_ops.buffer_load(crfa(a_running), safe_expert, vec_width=1, dtype=fx.Int32)
        count = valid_expert.select(count, fx.Int32(0))
        overflow_flag = (count > fx.Int32(fz_cap)).select(fx.Int32(1), fx.Int32(0))
        overflow_prefix = _wave_inclusive_scan_i32(overflow_flag, lane)
        overflow_count = fx.Int32(fx.rocdl.readlane(T.i32, overflow_prefix, fz_epr - 1))
        no_overflow = overflow_count == fx.Int32(0)
        safe_count = (count <= fx.Int32(fz_cap)).select(count, fx.Int32(0))
        num_expert_tiles = (safe_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
        max_expert_tiles = _wave_reduce_max_i32(num_expert_tiles, lane)
        inclusive_tiles = _wave_inclusive_scan_i32(num_expert_tiles, lane)
        metadata_base = inclusive_tiles - num_expert_tiles
        total_tiles = fx.Int32(fx.rocdl.readlane(T.i32, inclusive_tiles, fz_epr - 1))

        if valid_expert:
            if no_overflow:
                global_expert = fx.Int32(fz_rank * fz_epr) + safe_expert
                payload_base = safe_expert * fx.Int32(fz_cap)
                for tile in range(fx.Int32(0), num_expert_tiles, 1):
                    metadata_index = metadata_base + tile
                    buffer_ops.buffer_store(global_expert, crfa(a_se), metadata_index)
                    buffer_ops.buffer_store(payload_base + tile * fx.Int32(fz_tile_m), crfa(a_trb), metadata_index)
                padded_rows = num_expert_tiles * fx.Int32(fz_tile_m)
                for pad in range(fx.Int32(0), padded_rows - safe_count, 1):
                    buffer_ops.buffer_store(fx.Int32(fz_npes * fz_mtpr), crfa(a_sm), payload_base + safe_count + pad)
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_running), safe_expert)

        # Every expert lane must finish invalidating its padded source rows
        # before lane 0 publishes work to the consumer CTAs.
        wait_all()
        comm_ops.fence_system_release()
        if lane == fx.Int32(0):
            num_valid = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            buffer_ops.buffer_store(num_valid, crfa(a_nv), fx.Int32(0))
            # num_valid[1] is a device-visible overflow status.
            buffer_ops.buffer_store(overflow_count, crfa(a_nv), fx.Int32(1))
            buffer_ops.buffer_store(
                max_expert_tiles, crfa(a_max_expert_tiles), fx.Int32(0)
            )

        wait_all()
        comm_ops.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = buffer_ops.buffer_load(crfa(p_plan_ready), source, vec_width=1, dtype=fx.Int64)
            ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.store_i32_system(remote_ready, ready_index, expected)
    fx.barrier()


# fmt: off
@flyc.jit
def emit_dispatch_plan(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_mtpr, fz_rank, fz_tile_m, fz_total_experts, addr_disp,
    i32_cur_tok, addr_in_idx, parity, expected, external_grouping, external_counting,
    dispatch_blocks, payload_chunk_rows=0, payload_tile_ready=False, payload_block_cap=4,
    lds_grouping=False, lds_group_blocks=0, count_allgather_plan=False,
    fused_total_recv=False,
):
# fmt: on
    """Build a destination-owned compact plan in one producer-only CTA."""
    if fused_total_recv:
        assert fz_npes <= 8, "destination mask is encoded in one i32"
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_bc = dp(DispatchSlot.COUNT_MATRIX)
    p_bc = dp(DispatchSlot.P2P_COUNT_MATRIX)
    a_cd = dp(DispatchSlot.COUNT_DONE)
    p_cd = dp(DispatchSlot.P2P_COUNT_DONE)
    a_lc = dp(DispatchSlot.LOCAL_CURSOR)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_mb = dp(DispatchSlot.TASK_ROW_BASE)
    p_mb = dp(DispatchSlot.P2P_TASK_ROW_BASE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_group_done = dp(DispatchSlot.GROUP_DONE)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)
    a_tile_ready = dp(DispatchSlot.TILE_READY)
    a_tile_expected = dp(DispatchSlot.TILE_EXPECTED)
    a_active_payload_blocks = dp(DispatchSlot.ACTIVE_PAYLOAD_BLOCKS)
    a_payload_blocks_per_destination = dp(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
    a_dest_counter = dp(DispatchSlot.DEST_COUNTER)
    a_recv_num = dp(DispatchSlot.RECV_NUM)
    p_recv_num = dp(DispatchSlot.P2P_RECV_NUM)
    a_total_recv = dp(DispatchSlot.TOTAL_RECV)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    block_threads = num_waves * 64

    gtid = tid
    gnt = fx.Int32(block_threads)
    wl = i32_cur_tok * fx.Int32(fz_k)
    r_idx = crfa(addr_in_idx)
    r_lh = crfa(a_lh)
    r_bc = crfa(a_bc)
    r_pair_base = crfa(a_pair_base)
    r_pair = crfa(a_pair_order)
    r_lc = crfa(a_lc)
    if const_expr(external_counting):
        if tid == fx.Int32(0):
            count_blocks = lds_group_blocks if lds_grouping else dispatch_blocks
            mori_shmem.int32_wait_until_equals(a_group_done, fx.Int32(count_blocks))
            comm_ops.fence_agent_acquire()
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_group_done), fx.Int32(0))
            wait_all()
            comm_ops.fence_agent_release()
    else:
        if const_expr(fused_total_recv):
            # Count expert routes and unique destination-token traffic in one
            # pass over top-k.  TOTAL_RECV used to rescan the same input and
            # complete a separate cross-rank handshake before planning.
            for token in range(gtid, i32_cur_tok, gnt):
                destination_mask = fx.Int32(0)
                for slot in range_constexpr(fz_k):
                    wk = token * fx.Int32(fz_k) + fx.Int32(slot)
                    expert = buffer_ops.buffer_load(
                        r_idx, wk, vec_width=1, dtype=fx.Int32
                    )
                    valid = (expert >= fx.Int32(0)) & (
                        expert < fx.Int32(fz_total_experts)
                    )
                    safe_expert = valid.select(expert, fx.Int32(0))
                    if valid:
                        comm_ops.atomic_add_agent(
                            a_lh + fx.Int64(safe_expert) * fx.Int64(4),
                            fx.Int32(1),
                        )
                    destination = safe_expert // fx.Int32(fz_epr)
                    destination_mask = valid.select(
                        destination_mask | (fx.Int32(1) << destination),
                        destination_mask,
                    )
                for destination in range_constexpr(fz_npes):
                    if (
                        destination_mask & fx.Int32(1 << destination)
                    ) != fx.Int32(0):
                        comm_ops.atomic_add_agent(
                            a_dest_counter + fx.Int64(destination * 4),
                            fx.Int32(1),
                        )
        elif const_expr(num_waves >= 8):
            for wk0 in range(gtid, wl, gnt * fx.Int32(2)):
                wk1 = wk0 + gnt
                valid_wk1 = wk1 < wl
                safe_wk1 = valid_wk1.select(wk1, fx.Int32(0))
                expert0 = buffer_ops.buffer_load(r_idx, wk0, vec_width=1, dtype=fx.Int32)
                expert1 = buffer_ops.buffer_load(r_idx, safe_wk1, vec_width=1, dtype=fx.Int32)
                valid0 = (expert0 >= fx.Int32(0)) & (expert0 < fx.Int32(fz_total_experts))
                valid1 = valid_wk1 & (expert1 >= fx.Int32(0)) & (expert1 < fx.Int32(fz_total_experts))
                if valid0:
                    comm_ops.atomic_add_agent(a_lh + fx.Int64(expert0) * fx.Int64(4), fx.Int32(1))
                if valid1:
                    comm_ops.atomic_add_agent(a_lh + fx.Int64(expert1) * fx.Int64(4), fx.Int32(1))
        else:
            for wk in range(gtid, wl, gnt):
                expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    comm_ops.atomic_add_agent(a_lh + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
    wait_all()
    fx.barrier()
    comm_ops.fence_agent_acquire()

    if const_expr(fused_total_recv):
        # Publish the unique-token count before COUNT_DONE.  That existing
        # system-scope signal now covers both the histogram and TOTAL_RECV, so
        # there is no second recv_num rendezvous.
        if tid < fx.Int32(fz_npes):
            destination = tid
            count = buffer_ops.buffer_load(
                crfa(a_dest_counter), destination,
                vec_width=1, dtype=fx.Int32,
            )
            remote_recv = buffer_ops.buffer_load(
                crfa(p_recv_num), destination,
                vec_width=1, dtype=fx.Int64,
            )
            comm_ops.store_i32_system(
                remote_recv, fx.Int32(fz_rank), count
            )
        wait_all()
        fx.barrier()

    if const_expr(payload_tile_ready and dispatch_blocks > 32):
        if warp == fx.Int32(0):
            _configure_payload_blocks(
                a_payload_blocks_per_destination, a_active_payload_blocks,
                lane, fz_npes=fz_npes, dispatch_blocks=dispatch_blocks,
                payload_block_cap=payload_block_cap,
            )
        wait_all()
        fx.barrier()
        comm_ops.fence_agent_release()

    if const_expr(count_allgather_plan):
        # Broadcast the tiny local [global_expert] histogram to every rank.
        # Once [source][global_expert] is resident everywhere, each sender can
        # derive its own remote row bases and no receiver->sender offset return
        # trip is needed.
        matrix_items = fz_npes * fz_total_experts
        for matrix_item in range(gtid, matrix_items, gnt):
            peer = matrix_item // fx.Int32(fz_total_experts)
            ge = matrix_item - peer * fx.Int32(fz_total_experts)
            remote_bigcnt = buffer_ops.buffer_load(
                crfa(p_bc), peer, vec_width=1, dtype=fx.Int64
            )
            count = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
            buffer_ops.buffer_store(
                count,
                crfa(remote_bigcnt),
                fx.Int32(fz_rank * fz_total_experts) + ge,
            )
    else:
        # Legacy destination-owned matrix: each receiver sees only counts for
        # its local experts and must return TASK_ROW_BASE to every sender.
        for ge in range(gtid, fz_total_experts, gnt):
            destination = ge // fx.Int32(fz_epr)
            local_expert = ge - destination * fx.Int32(fz_epr)
            remote_bigcnt = buffer_ops.buffer_load(
                crfa(p_bc), destination, vec_width=1, dtype=fx.Int64
            )
            count = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
            buffer_ops.buffer_store(
                count,
                crfa(remote_bigcnt),
                fx.Int32(fz_rank * fz_epr) + local_expert,
            )
    wait_all()
    fx.barrier()

    if const_expr(count_allgather_plan):
        # One wave owns one destination.  The count matrix makes all eight
        # plans independent, so there is no reason to serialize destinations
        # in warp 0 or to return offsets from receiver to sender.
        if warp == fx.Int32(0):
            comm_ops.fence_system_release()
            for peer in range(lane, fz_npes, 64):
                remote_done = buffer_ops.buffer_load(
                    crfa(p_cd), peer, vec_width=1, dtype=fx.Int64
                )
                done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                comm_ops.store_i32_system(remote_done, done_index, expected)
            for source in range(lane, fz_npes, 64):
                done_index = parity * fx.Int32(fz_npes) + source
                mori_shmem.int32_wait_until_equals(
                    a_cd + fx.Int64(done_index) * fx.Int64(4), expected
                )
        fx.barrier()
        comm_ops.fence_system_acquire()

        if const_expr(fused_total_recv):
            if (warp == fx.Int32(0)) & (lane == fx.Int32(0)):
                total_recv = fx.Int32(0)
                for source in range_constexpr(fz_npes):
                    total_recv = total_recv + buffer_ops.buffer_load(
                        crfa(a_recv_num), fx.Int32(source),
                        vec_width=1, dtype=fx.Int32,
                    )
                    buffer_ops.buffer_store(
                        fx.Int32(-1), crfa(a_recv_num), fx.Int32(source)
                    )
                buffer_ops.buffer_store(
                    total_recv, crfa(a_total_recv), fx.Int32(0)
                )
                for destination_reset in range_constexpr(fz_npes):
                    buffer_ops.buffer_store(
                        fx.Int32(0), crfa(a_dest_counter),
                        fx.Int32(destination_reset),
                    )
            wait_all()
            fx.barrier()

        destination = warp
        r_nv = crfa(a_nv)
        r_mb = crfa(a_mb)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        local_destination = destination == fx.Int32(fz_rank)
        for expert_chunk in range_constexpr((fz_epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(fz_epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            ge = destination * fx.Int32(fz_epr) + local_expert
            safe_ge = destination * fx.Int32(fz_epr) + safe_expert
            source_counts = []
            total_count = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                source_count = buffer_ops.buffer_load(
                    r_bc,
                    fx.Int32(source * fz_total_experts) + safe_ge,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                source_count = valid_expert.select(source_count, fx.Int32(0))
                source_counts.append(source_count)
                total_count = total_count + source_count
            num_tiles = (total_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
            chunk_max = _wave_reduce_max_i32(num_tiles, lane)
            max_expert_tiles = (chunk_max > max_expert_tiles).select(
                chunk_max, max_expert_tiles
            )
            padded_rows = num_tiles * fx.Int32(fz_tile_m)
            inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
            local_row_base = row_carry + inclusive_rows - padded_rows

            sender_prefix = fx.Int32(0)
            local_sender_prefix = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                if source == fz_rank:
                    local_sender_prefix = sender_prefix
                sender_prefix = sender_prefix + source_counts[source]
            if valid_expert:
                buffer_ops.buffer_store(
                    local_row_base + local_sender_prefix, r_mb, ge
                )

            if local_destination & valid_expert:
                if const_expr(payload_tile_ready):
                    base_tile = local_row_base // fx.Int32(fz_tile_m)
                    for tile in range(fx.Int32(0), num_tiles, 1):
                        tile_index = base_tile + tile
                        buffer_ops.buffer_store(
                            fx.Int32(0), crfa(a_tile_ready), tile_index
                        )
                        buffer_ops.buffer_store(
                            fx.Int32(1), crfa(a_tile_expected), tile_index
                        )
                    sender_prefix = fx.Int32(0)
                    for source in range_constexpr(fz_npes):
                        source_count = source_counts[source]
                        source_active = source_count > fx.Int32(0)
                        source_boundary = source_active & (sender_prefix > fx.Int32(0))
                        source_boundary = source_boundary & (
                            sender_prefix % fx.Int32(fz_tile_m) != fx.Int32(0)
                        )
                        if source_boundary:
                            tile_index = base_tile + sender_prefix // fx.Int32(fz_tile_m)
                            _increment_i32(crfa(a_tile_expected), tile_index)
                        for chunk_offset in range(
                            fx.Int32(payload_chunk_rows), source_count, payload_chunk_rows
                        ):
                            boundary = sender_prefix + chunk_offset
                            boundary_unaligned = boundary % fx.Int32(fz_tile_m) != fx.Int32(0)
                            if boundary_unaligned:
                                tile_index = base_tile + boundary // fx.Int32(fz_tile_m)
                                _increment_i32(crfa(a_tile_expected), tile_index)
                        sender_prefix = sender_prefix + source_count
                _store_expert_metadata(
                    a_se, a_trb, a_sm, ge, local_row_base, total_count,
                    num_tiles, padded_rows, fz_tile_m=fz_tile_m,
                    invalid_source=fz_npes * fz_mtpr,
                )

            last_lane = min(63, fz_epr - expert_chunk * 64 - 1)
            row_carry = row_carry + fx.Int32(
                fx.rocdl.readlane(T.i32, inclusive_rows, last_lane)
            )

        if (lane == fx.Int32(0)) & local_destination:
            buffer_ops.buffer_store(row_carry, r_nv, fx.Int32(0))
            buffer_ops.buffer_store(
                max_expert_tiles, crfa(a_max_expert_tiles), fx.Int32(0)
            )
        wait_all()
        comm_ops.fence_system_release()
        if lane == fx.Int32(0):
            ready_index = parity * fx.Int32(fz_npes) + destination
            comm_ops.store_i32_system(a_plan_ready, ready_index, expected)
        wait_all()
    else:
        # Legacy receiver-owned plan.  Warp 0 returns every source's offset and
        # publishes PLAN_READY back to those sources.
        # Seed values before the dynamic warp predicate so FlyDSL does not
        # infer optional scf.if results when this compile-time branch is used.
        r_nv = crfa(a_nv)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        if warp == fx.Int32(0):
            comm_ops.fence_system_release()
            for peer in range(lane, fz_npes, 64):
                remote_done = buffer_ops.buffer_load(
                    crfa(p_cd), peer, vec_width=1, dtype=fx.Int64
                )
                done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                comm_ops.store_i32_system(remote_done, done_index, expected)
            for source in range(lane, fz_npes, 64):
                done_index = parity * fx.Int32(fz_npes) + source
                mori_shmem.int32_wait_until_equals(
                    a_cd + fx.Int64(done_index) * fx.Int64(4), expected
                )
            comm_ops.fence_system_acquire()

            if const_expr(fused_total_recv):
                if lane == fx.Int32(0):
                    total_recv = fx.Int32(0)
                    for source in range_constexpr(fz_npes):
                        total_recv = total_recv + buffer_ops.buffer_load(
                            crfa(a_recv_num), fx.Int32(source),
                            vec_width=1, dtype=fx.Int32,
                        )
                        buffer_ops.buffer_store(
                            fx.Int32(-1), crfa(a_recv_num), fx.Int32(source)
                        )
                    buffer_ops.buffer_store(
                        total_recv, crfa(a_total_recv), fx.Int32(0)
                    )
                    for destination_reset in range_constexpr(fz_npes):
                        buffer_ops.buffer_store(
                            fx.Int32(0), crfa(a_dest_counter),
                            fx.Int32(destination_reset),
                        )
                wait_all()

            for expert_chunk in range_constexpr((fz_epr + 63) // 64):
                local_expert = fx.Int32(expert_chunk * 64) + lane
                valid_expert = local_expert < fx.Int32(fz_epr)
                safe_expert = valid_expert.select(local_expert, fx.Int32(0))
                ge = fx.Int32(fz_rank * fz_epr) + local_expert
                source_counts = []
                total_count = fx.Int32(0)
                for source in range_constexpr(fz_npes):
                    source_count = buffer_ops.buffer_load(
                        r_bc,
                        fx.Int32(source * fz_epr) + safe_expert,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                    source_count = valid_expert.select(source_count, fx.Int32(0))
                    source_counts.append(source_count)
                    total_count = total_count + source_count
                num_tiles = (total_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
                chunk_max = _wave_reduce_max_i32(num_tiles, lane)
                max_expert_tiles = (chunk_max > max_expert_tiles).select(
                    chunk_max, max_expert_tiles
                )
                padded_rows = num_tiles * fx.Int32(fz_tile_m)
                inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
                local_row_base = row_carry + inclusive_rows - padded_rows

                sender_prefix = fx.Int32(0)
                for source in range_constexpr(fz_npes):
                    if valid_expert:
                        remote_my_base = buffer_ops.buffer_load(
                            crfa(p_mb), fx.Int32(source), vec_width=1, dtype=fx.Int64
                        )
                        buffer_ops.buffer_store(
                            local_row_base + sender_prefix, crfa(remote_my_base), ge
                        )
                    sender_prefix = sender_prefix + source_counts[source]

                if valid_expert:
                    if const_expr(payload_tile_ready):
                        base_tile = local_row_base // fx.Int32(fz_tile_m)
                        for tile in range(fx.Int32(0), num_tiles, 1):
                            tile_index = base_tile + tile
                            buffer_ops.buffer_store(fx.Int32(0), crfa(a_tile_ready), tile_index)
                            buffer_ops.buffer_store(fx.Int32(1), crfa(a_tile_expected), tile_index)
                        sender_prefix = fx.Int32(0)
                        for source in range_constexpr(fz_npes):
                            source_count = source_counts[source]
                            source_active = source_count > fx.Int32(0)
                            source_boundary = source_active & (sender_prefix > fx.Int32(0))
                            source_boundary = source_boundary & (
                                sender_prefix % fx.Int32(fz_tile_m) != fx.Int32(0)
                            )
                            if source_boundary:
                                tile_index = base_tile + sender_prefix // fx.Int32(fz_tile_m)
                                _increment_i32(crfa(a_tile_expected), tile_index)
                            for chunk_offset in range(
                                fx.Int32(payload_chunk_rows), source_count, payload_chunk_rows
                            ):
                                boundary = sender_prefix + chunk_offset
                                boundary_unaligned = boundary % fx.Int32(fz_tile_m) != fx.Int32(0)
                                if boundary_unaligned:
                                    tile_index = base_tile + boundary // fx.Int32(fz_tile_m)
                                    _increment_i32(crfa(a_tile_expected), tile_index)
                            sender_prefix = sender_prefix + source_count
                    _store_expert_metadata(
                        a_se, a_trb, a_sm, ge, local_row_base, total_count,
                        num_tiles, padded_rows, fz_tile_m=fz_tile_m,
                        invalid_source=fz_npes * fz_mtpr,
                    )

                last_lane = min(63, fz_epr - expert_chunk * 64 - 1)
                row_carry = row_carry + fx.Int32(
                    fx.rocdl.readlane(T.i32, inclusive_rows, last_lane)
                )

            if lane == fx.Int32(0):
                buffer_ops.buffer_store(row_carry, r_nv, fx.Int32(0))
                buffer_ops.buffer_store(
                    max_expert_tiles, crfa(a_max_expert_tiles), fx.Int32(0)
                )
            wait_all()
            comm_ops.fence_system_release()
            for source in range(lane, fz_npes, 64):
                remote_ready = buffer_ops.buffer_load(
                    crfa(p_plan_ready), source, vec_width=1, dtype=fx.Int64
                )
                ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                comm_ops.store_i32_system(remote_ready, ready_index, expected)
            wait_all()

    if warp == fx.Int32(1):
        # Build the global-expert exclusive prefix cooperatively.
        pairs_per_lane = (fz_total_experts + 63) // 64
        lane_base = lane * fx.Int32(pairs_per_lane)
        lane_total = fx.Int32(0)
        lane_counts = []
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            safe_ge = valid_ge.select(ge, fx.Int32(0))
            source_count = buffer_ops.buffer_load(r_lh, safe_ge, vec_width=1, dtype=fx.Int32)
            source_count = valid_ge.select(source_count, fx.Int32(0))
            lane_counts.append(source_count)
            lane_total = lane_total + source_count
        lane_prefix = _wave_inclusive_scan_i32(lane_total, lane) - lane_total
        source_prefix = lane_prefix
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            if valid_ge:
                buffer_ops.buffer_store(source_prefix, r_pair_base, ge)
                buffer_ops.buffer_store(source_prefix, r_lc, ge)
            source_prefix = source_prefix + lane_counts[item]
        wait_all()
        comm_ops.fence_agent_release()
        if lane == fx.Int32(0):
            comm_ops.store_i32_system(a_pair_ready, parity, expected)

    if const_expr(not external_grouping):
        # Warp 1 groups immediately; later waves wait for its prefix.
        if warp > fx.Int32(0):
            if warp > fx.Int32(1):
                if lane == fx.Int32(0):
                    mori_shmem.int32_wait_until_equals(a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected)
                    comm_ops.fence_agent_acquire()
            group_tid = (warp - fx.Int32(1)) * fx.Int32(64) + lane
            group_threads = fx.Int32((num_waves - 1) * 64)
            for wk in range(group_tid, wl, group_threads):
                expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    position = fx.Int32(
                        comm_ops.atomic_add_agent(a_lc + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
                    )
                    buffer_ops.buffer_store(wk, r_pair, position)

    wait_all()
    fx.barrier()
    if tid == fx.Int32(0):
        if const_expr(external_grouping):
            active_group_blocks = fx.Int32(dispatch_blocks)
            if const_expr(payload_tile_ready and dispatch_blocks > 32):
                active_group_blocks = buffer_ops.buffer_load(
                    crfa(a_active_payload_blocks), fx.Int32(0), vec_width=1, dtype=fx.Int32
                )
            mori_shmem.int32_wait_until_equals(a_group_done, active_group_blocks)
            comm_ops.fence_agent_acquire()
        comm_ops.fence_agent_release()
        comm_ops.store_i32_system(a_pair_order_ready, parity, expected)


# fmt: off
@flyc.jit
def emit_dispatch_group(
    *, num_waves, fz_k, fz_total_experts, addr_disp, i32_cur_tok, addr_in_idx, dispatch_blocks,
    producer_slot, parity, expected, external_counting, adaptive_grouping=False,
    lds_grouping=False, lds_group_blocks=0, group_scratch=None,
):
# fmt: on
    """Count and group disjoint route spans across payload producer CTAs."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_local_hist = dp(DispatchSlot.LOCAL_HIST)
    a_local_cursor = dp(DispatchSlot.LOCAL_CURSOR)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_group_done = dp(DispatchSlot.GROUP_DONE)
    a_active_payload_blocks = dp(DispatchSlot.ACTIVE_PAYLOAD_BLOCKS)
    r_idx = crfa(addr_in_idx)
    r_pair = crfa(a_pair_order)
    tid = fx.thread_idx.x
    block_threads = fx.Int32(num_waves * 64)
    group_tid = producer_slot * block_threads + tid
    group_threads = fx.Int32(dispatch_blocks) * block_threads
    route_limit = i32_cur_tok * fx.Int32(fz_k)

    def _lds_atomic_add(index, value):
        byte_address = fx.Int64(fx.ptrtoint(group_scratch)) + fx.Int64(index) * fx.Int64(4)
        pointer = buffer_ops.create_llvm_ptr(byte_address, address_space=3)
        return fx.Int32(
            _llvm_d.AtomicRMWOp(
                _llvm_d.AtomicBinOp.add,
                pointer,
                value.ir_value(),
                _llvm_d.AtomicOrdering.monotonic,
                syncscope=fx.rocdl.SyncScope.Workgroup,
                alignment=4,
            ).res
        )

    if const_expr(external_counting):
        if const_expr(lds_grouping):
            count_active = producer_slot < fx.Int32(lds_group_blocks)
            if count_active:
                count_tid = producer_slot * block_threads + tid
                count_threads = fx.Int32(lds_group_blocks) * block_threads
                for expert in range(tid, fx.Int32(fz_total_experts), block_threads):
                    fx.ptr_store(
                        fx.Vector.from_elements([fx.Int32(0)], fx.Int32),
                        group_scratch + expert,
                    )
                fx.barrier()
                for route in range(count_tid, route_limit, count_threads):
                    expert = buffer_ops.buffer_load(r_idx, route, vec_width=1, dtype=fx.Int32)
                    valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                    if valid:
                        _lds_atomic_add(expert, fx.Int32(1))
                fx.barrier()
                for expert in range(tid, fx.Int32(fz_total_experts), block_threads):
                    block_count = fx.Int32(fx.ptr_load(group_scratch + expert))
                    if block_count > fx.Int32(0):
                        comm_ops.atomic_add_agent(
                            a_local_hist + fx.Int64(expert) * fx.Int64(4),
                            block_count,
                        )
                wait_all()
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_agent_release()
                    comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
        else:
            for route in range(group_tid, route_limit, group_threads):
                expert = buffer_ops.buffer_load(r_idx, route, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    comm_ops.atomic_add_agent(a_local_hist + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
            wait_all()
            fx.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
    if tid == fx.Int32(0):
        mori_shmem.int32_wait_until_equals(a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected)
    fx.barrier()
    # Every grouping thread consumes planner-owned cursors below.  The wait is
    # block-uniform control, but its acquire must cover each consuming wave.
    comm_ops.fence_agent_acquire()

    active_group_blocks = fx.Int32(dispatch_blocks)
    if const_expr(lds_grouping):
        active_group_blocks = fx.Int32(lds_group_blocks)
    elif const_expr(adaptive_grouping and dispatch_blocks > 32):
        active_group_blocks = buffer_ops.buffer_load(
            crfa(a_active_payload_blocks), fx.Int32(0), vec_width=1, dtype=fx.Int32
        )
    group_active = producer_slot < active_group_blocks
    if group_active:
        active_group_tid = producer_slot * block_threads + tid
        active_group_threads = active_group_blocks * block_threads
        if const_expr(lds_grouping):
            group_base = group_scratch + fx.Int32(fz_total_experts)
            for expert in range(tid, fx.Int32(fz_total_experts), block_threads):
                block_count = fx.Int32(fx.ptr_load(group_scratch + expert))
                reserved_base = fx.Int32(0)
                if block_count > fx.Int32(0):
                    reserved_base = fx.Int32(
                        comm_ops.atomic_add_agent(
                            a_local_cursor + fx.Int64(expert) * fx.Int64(4),
                            block_count,
                        )
                    )
                fx.ptr_store(
                    fx.Vector.from_elements([reserved_base], fx.Int32),
                    group_base + expert,
                )
                fx.ptr_store(
                    fx.Vector.from_elements([fx.Int32(0)], fx.Int32),
                    group_scratch + expert,
                )
            wait_all()
            fx.barrier()
            for route in range(active_group_tid, route_limit, active_group_threads):
                expert = buffer_ops.buffer_load(r_idx, route, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    local_position = _lds_atomic_add(expert, fx.Int32(1))
                    reserved_base = fx.Int32(fx.ptr_load(group_base + expert))
                    buffer_ops.buffer_store(route, r_pair, reserved_base + local_position)
        else:
            for route in range(active_group_tid, route_limit, active_group_threads):
                expert = buffer_ops.buffer_load(r_idx, route, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    position = fx.Int32(
                        comm_ops.atomic_add_agent(a_local_cursor + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
                    )
                    buffer_ops.buffer_store(route, r_pair, position)
        wait_all()
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
            mori_shmem.int32_wait_until_equals(a_pair_order_ready + fx.Int64(parity) * fx.Int64(4), expected)
    fx.barrier()
    if group_active:
        # Pair-order entries are produced by all grouping CTAs and consumed by
        # lane 0 of every payload-copy wave, then broadcast with readfirstlane.
        if (tid & fx.Int32(63)) == fx.Int32(0):
            comm_ops.fence_agent_acquire()
    fx.barrier()


# fmt: off


# fmt: off
@flyc.jit
def emit_dispatch_payload(
    *, num_waves, fz_epr, fz_k, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32, fz_safe_end_i32,
    fz_scale_n_i32, fz_enable_scales, fz_route_payload, fz_copy_payload,
    fz_mxfp4_smooth_pload, smoothquant_mode, smooth_rsrc, model_dim,
    addr_disp, addr_in_tok, addr_in_wts, addr_in_sc, dispatch_blocks,
    producer_slot, parity, expected, producers_per_destination,
    payload_chunk_rows=0,
    payload_tile_ready=False,
):
# fmt: on
    """Produce independently publishable expert payloads from a compact plan."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_mb = dp(DispatchSlot.TASK_ROW_BASE)
    p_payload_ready = dp(DispatchSlot.P2P_PAYLOAD_READY)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_chunk_done = dp(DispatchSlot.PAYLOAD_CHUNK_DONE)
    p_tile_ready = dp(DispatchSlot.P2P_TILE_READY)
    p_payload_ready_rows = dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS)
    p_row_scale = dp(DispatchSlot.P2P_ROW_SCALE)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    r_pair_base = crfa(a_pair_base)
    r_lh = crfa(a_lh)
    r_mb = crfa(a_mb)
    r_pair = crfa(a_pair_order)
    r_wts = crfa(addr_in_wts)
    r_chunk_done = crfa(a_chunk_done)
    row0 = warp
    row_stride = fx.Int32(num_waves)

    def _publish_task(destination, local_expert, ge):
        comm_ops.fence_system_release()
        ready_remote = buffer_ops.buffer_load(crfa(p_payload_ready), destination, vec_width=1, dtype=fx.Int64)
        ready_index = parity * fx.Int32(fz_epr) + local_expert
        comm_ops.atomic_add_system(ready_remote + fx.Int64(ready_index) * fx.Int64(4), fx.Int32(1))
        buffer_ops.buffer_store(fx.Int32(0), r_lh, ge)

    def _finish_task(destination, local_expert, ge, num_chunks):
        if const_expr(payload_chunk_rows > 0):
            comm_ops.fence_system_release()
            completed = fx.Int32(
                comm_ops.atomic_add_agent(a_chunk_done + fx.Int64(ge) * fx.Int64(4), fx.Int32(1))
            )
            if completed == num_chunks - fx.Int32(1):
                comm_ops.fence_agent_acquire()
                buffer_ops.buffer_store(fx.Int32(0), r_chunk_done, ge)
                _publish_task(destination, local_expert, ge)
        else:
            _publish_task(destination, local_expert, ge)

    num_destinations = fz_total_experts // fz_epr
    local_expert = fx.Int32(0)
    chunk_id = fx.Int32(0)
    task_index = fx.Int32(0)
    task_limit = fx.Int32(0)
    task_stride = fx.Int32(1)
    task_active = fx.Int32(0) != fx.Int32(0)
    if const_expr(payload_chunk_rows > 0):
        assert dispatch_blocks % num_destinations == 0
        # Keep each expert's chunks consecutive on one producer.  The former
        # chunk-major rotation delayed expert zero's final chunk behind every
        # other expert and made the first consumer tile the compact-path
        # critical tail at large MTPR.
        producer_round = producer_slot // fx.Int32(num_destinations)
        local_expert = producer_round
        task_active = local_expert < fx.Int32(fz_epr)
    else:
        task_limit = fx.Int32(fz_total_experts)
        task_index = producer_slot
        task_stride = fx.Int32(dispatch_blocks)
        task_active = task_index < task_limit
    hoist_remote_resources = fz_mtpr >= 1024
    producer_destination = producer_slot % fx.Int32(num_destinations)
    ready_index = parity * fx.Int32(num_destinations) + producer_destination
    if tid == fx.Int32(0):
        mori_shmem.int32_wait_until_equals(a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()
    destination_ready_rows = fx.Int32(0)
    if const_expr(payload_tile_ready):
        if tid == fx.Int32(0):
            remote_ready_rows = buffer_ops.buffer_load(
                crfa(p_payload_ready_rows), producer_destination, vec_width=1, dtype=fx.Int64
            )
            destination_ready_rows = buffer_ops.buffer_load(
                crfa(remote_ready_rows), fx.Int32(0), vec_width=1, dtype=fx.Int32
            )
    fx.barrier()
    while task_active:
        if const_expr(payload_chunk_rows == 0):
            chunk_id = fx.Int32(0)
            local_expert = task_index // fx.Int32(num_destinations)
        destination = producer_destination
        ge = destination * fx.Int32(fz_epr) + local_expert
        source_count_lane = fx.Int32(0)
        source_base_lane = fx.Int32(0)
        destination_base_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            source_count_lane = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
            source_base_lane = buffer_ops.buffer_load(r_pair_base, ge, vec_width=1, dtype=fx.Int32)
            destination_base_lane = buffer_ops.buffer_load(r_mb, ge, vec_width=1, dtype=fx.Int32)
        source_count = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_count_lane))
        source_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_base_lane))
        destination_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, destination_base_lane))
        if const_expr(payload_chunk_rows > 0):
            num_chunks = (source_count + fx.Int32(payload_chunk_rows - 1)) // fx.Int32(
                payload_chunk_rows
            )
            num_chunks = (num_chunks > fx.Int32(0)).select(num_chunks, fx.Int32(1))
            chunk_active = chunk_id < num_chunks
            chunk_begin = chunk_id * fx.Int32(payload_chunk_rows)
            chunk_limit = chunk_begin + fx.Int32(payload_chunk_rows)
            chunk_end = (source_count < chunk_limit).select(source_count, chunk_limit)
            row_begin = chunk_active.select(chunk_begin, fx.Int32(0))
            row_end = chunk_active.select(chunk_end, fx.Int32(0))
        else:
            num_chunks = fx.Int32(1)
            chunk_active = fx.Int32(0) == fx.Int32(0)
            row_begin = fx.Int32(0)
            row_end = source_count
        if const_expr(hoist_remote_resources):
            wts_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64))
            srcmap_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64))
            token_remote = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
            if const_expr(fz_enable_scales and fz_copy_payload):
                scale_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64))
            if const_expr(fz_mxfp4_smooth_pload):
                row_scale_remote_rsrc = crfa(
                    buffer_ops.buffer_load(
                        crfa(p_row_scale), destination, vec_width=1, dtype=fx.Int64
                    )
                )
        for row in range(row_begin + row0, row_end, row_stride):
            wk_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                wk_lane = buffer_ops.buffer_load(r_pair, source_base + row, vec_width=1, dtype=fx.Int32)
            wk = fx.Int32(fx.rocdl.readfirstlane(T.i32, wk_lane))
            source_token = wk // fx.Int32(fz_k)
            topk_slot = wk % fx.Int32(fz_k)
            source_row = (
                source_token
                if const_expr(smoothquant_mode == "bf16_route")
                else wk
                if const_expr(fz_route_payload)
                else source_token
            )
            destination_row = destination_base + row

            def _copy_route_header():
                weight = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                if const_expr(hoist_remote_resources):
                    buffer_ops.buffer_store(weight_bits, wts_remote_rsrc, destination_row)
                    buffer_ops.buffer_store(source_encoding, srcmap_remote_rsrc, destination_row)
                else:
                    wts_remote = buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(weight_bits, crfa(wts_remote), destination_row)
                    srcmap_remote = buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(source_encoding, crfa(srcmap_remote), destination_row)

            if lane == fx.Int32(0):
                _copy_route_header()

            if const_expr(
                fz_enable_scales
                and fz_copy_payload
                and smoothquant_mode == "none"
            ):
                scale_lane = lane
                if const_expr(fz_scale_n_i32 % 4 == 0):
                    scale_offset = scale_lane * fx.Int32(4)
                    if scale_offset < fx.Int32(fz_scale_n_i32):
                        scale = buffer_ops.buffer_load(
                            crfa(addr_in_sc), source_row * fx.Int32(fz_scale_n_i32) + scale_offset,
                            vec_width=4, dtype=fx.Int32,
                        )
                        if const_expr(hoist_remote_resources):
                            buffer_ops.buffer_store(
                                scale, scale_remote_rsrc, destination_row * fx.Int32(fz_scale_n_i32) + scale_offset
                            )
                        else:
                            row_scale_remote = crfa(
                                buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                            )
                            buffer_ops.buffer_store(
                                scale, row_scale_remote, destination_row * fx.Int32(fz_scale_n_i32) + scale_offset
                            )
                elif scale_lane < fx.Int32(fz_scale_n_i32):
                    scale = buffer_ops.buffer_load(
                        crfa(addr_in_sc), source_row * fx.Int32(fz_scale_n_i32) + scale_lane,
                        vec_width=1, dtype=fx.Int32,
                    )
                    if const_expr(hoist_remote_resources):
                        buffer_ops.buffer_store(
                            scale, scale_remote_rsrc, destination_row * fx.Int32(fz_scale_n_i32) + scale_lane
                        )
                    else:
                        row_scale_remote = crfa(
                            buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                        )
                        buffer_ops.buffer_store(
                            scale, row_scale_remote, destination_row * fx.Int32(fz_scale_n_i32) + scale_lane
                        )
            if const_expr(fz_copy_payload):
                if const_expr(hoist_remote_resources):
                    destination_token_addr = token_remote
                else:
                    destination_token_addr = buffer_ops.buffer_load(
                        crfa(p_rx), destination, vec_width=1, dtype=fx.Int64
                    )
                if const_expr(smoothquant_mode == "bf16_route"):
                    row_scale = fused_prepare(
                        addr_in_tok,
                        destination_token_addr,
                        smooth_rsrc,
                        smooth_rsrc,
                        source_token,
                        destination_row,
                        ge,
                        lane,
                        smoothquant_mode=smoothquant_mode,
                        model_dim=model_dim,
                        payload_row_bytes=fz_nbytes,
                        fz_safe_end_i32=fz_safe_end_i32,
                        fz_n_i32=fz_n_i32,
                    )
                    if lane == fx.Int32(0):
                        if const_expr(hoist_remote_resources):
                            buffer_ops.buffer_store(
                                row_scale, scale_remote_rsrc, destination_row
                            )
                        else:
                            remote_scale = buffer_ops.buffer_load(
                                crfa(p_sc),
                                destination,
                                vec_width=1,
                                dtype=fx.Int64,
                            )
                            buffer_ops.buffer_store(
                                row_scale, crfa(remote_scale), destination_row
                            )
                elif const_expr(fz_mxfp4_smooth_pload):
                    source_rsrc = crfa(
                        addr_in_tok
                        + fx.Int64(source_row) * fx.Int64(fz_nbytes)
                    )
                    destination_rsrc = crfa(
                        destination_token_addr
                        + fx.Int64(destination_row) * fx.Int64(fz_nbytes)
                    )
                    row_scale = copy_mxfp4_row_and_compute_smoothquant_scale_wave(
                        source_rsrc,
                        destination_rsrc,
                        crfa(addr_in_sc),
                        smooth_rsrc,
                        source_row,
                        ge,
                        cols=fz_nbytes * 2,
                    )
                    if lane == fx.Int32(0):
                        if const_expr(hoist_remote_resources):
                            buffer_ops.buffer_store(
                                row_scale, row_scale_remote_rsrc, destination_row
                            )
                        else:
                            remote_row_scale = crfa(
                                buffer_ops.buffer_load(
                                    crfa(p_row_scale),
                                    destination,
                                    vec_width=1,
                                    dtype=fx.Int64,
                                )
                            )
                            buffer_ops.buffer_store(
                                row_scale, remote_row_scale, destination_row
                            )
                else:
                    fused_prepare(
                        addr_in_tok,
                        destination_token_addr,
                        smooth_rsrc,
                        smooth_rsrc,
                        source_row,
                        destination_row,
                        ge,
                        lane,
                        smoothquant_mode=smoothquant_mode,
                        model_dim=model_dim,
                        payload_row_bytes=fz_nbytes,
                        fz_safe_end_i32=fz_safe_end_i32, fz_n_i32=fz_n_i32,
                    )

        if chunk_active:
            wait_all()
            fx.barrier()
            if tid == fx.Int32(0):
                if const_expr(payload_tile_ready):
                    _publish_tile_range(
                        p_tile_ready, destination, destination_base, row_begin, row_end, destination_ready_rows
                    )
                _finish_task(destination, local_expert, ge, num_chunks)
            fx.barrier()

        if const_expr(payload_chunk_rows > 0):
            next_chunk = chunk_id + fx.Int32(1)
            advance_expert = next_chunk >= num_chunks
            local_expert = advance_expert.select(
                local_expert + fx.Int32(producers_per_destination),
                local_expert,
            )
            chunk_id = advance_expert.select(fx.Int32(0), next_chunk)
            task_active = local_expert < fx.Int32(fz_epr)
        else:
            task_index = task_index + task_stride
            task_active = task_index < task_limit
