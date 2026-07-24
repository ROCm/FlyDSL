# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Single compact task-channel dispatch path for MegaMoE v2."""

import mori.ir.flydsl as mori_shmem

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, range_constexpr

from ..utils import epk
from .planner import DispatchSlot, SmallFixedSlot


@flyc.jit
# fmt: off
def emit_dispatch_plan(*, num_waves, fz_npes, fz_epr, fz_k, fz_mtpr, fz_rank,
    fz_tile_m, fz_total_experts, fz_epoch_increment, addr_disp, i32_cur_tok, addr_in_idx,
    dispatch_blocks, addr_parity, addr_expected):
# fmt: on
    """Build a destination-owned compact plan in one producer-only CTA."""
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
    p_mb = dp(DispatchSlot.P2P_TASK_ROW_BASE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_pair_ready = dp(DispatchSlot.PAIR_READY)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    flat = fx.Int32(fx.block_idx.x)
    block_threads = num_waves * 64
    gtid = flat * fx.Int32(block_threads) + tid
    gnt = fx.Int32(dispatch_blocks * block_threads)
    gwid = flat * fx.Int32(num_waves) + warp
    wl = i32_cur_tok * fx.Int32(fz_k)
    r_idx = crfa(addr_in_idx)
    r_lh = crfa(a_lh)
    r_bc = crfa(a_bc)
    r_pair_base = crfa(a_pair_base)
    r_pair = crfa(a_pair_order)
    r_lc = crfa(a_lc)
    parity_rsrc = crfa(addr_parity)
    expected_rsrc = crfa(addr_expected)

    # The producer-only planner owns the next overlapped launch's epoch and
    # publishes expected before parity.
    if flat == fx.Int32(0):
        if tid == fx.Int32(0):
            old_parity = buffer_ops.buffer_load(parity_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)
            next_parity = old_parity ^ fx.Int32(1)
            previous_expected = buffer_ops.buffer_load(
                expected_rsrc, next_parity, vec_width=1, dtype=fx.Int32
            )
            next_expected = previous_expected + fx.Int32(fz_epoch_increment)
            buffer_ops.buffer_store(next_expected, expected_rsrc, next_parity)
            fx.rocdl.s_waitcnt(0)
            epk.fence_agent_release()
            buffer_ops.buffer_store(next_parity, parity_rsrc, fx.Int32(0))
        fx.barrier()
    parity = buffer_ops.buffer_load(parity_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)
    expected = buffer_ops.buffer_load(expected_rsrc, parity, vec_width=1, dtype=fx.Int32)

    # Count pass: direct agent atomics into the production local_hist buffer.
    if const_expr(num_waves >= 8):
        for wk0 in range(gtid, wl, gnt * fx.Int32(2)):
            wk1 = wk0 + gnt
            valid_wk1 = wk1 < wl
            safe_wk1 = valid_wk1.select(wk1, fx.Int32(0))
            expert0 = buffer_ops.buffer_load(r_idx, wk0, vec_width=1, dtype=fx.Int32)
            expert1 = buffer_ops.buffer_load(r_idx, safe_wk1, vec_width=1, dtype=fx.Int32)
            valid0 = (expert0 >= fx.Int32(0)) & (expert0 < fx.Int32(fz_total_experts))
            valid1 = (
                valid_wk1
                & (expert1 >= fx.Int32(0))
                & (expert1 < fx.Int32(fz_total_experts))
            )
            if valid0:
                epk.atomic_add_agent(a_lh + fx.Int64(expert0) * fx.Int64(4), fx.Int32(1))
            if valid1:
                epk.atomic_add_agent(a_lh + fx.Int64(expert1) * fx.Int64(4), fx.Int32(1))
    else:
        for wk in range(gtid, wl, gnt):
            expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
            if valid:
                epk.atomic_add_agent(a_lh + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    epk.fence_agent_acquire()

    # Warp 0 exchanges destination-local counts, builds compact bases/tile
    # metadata, and publishes each destination plan independently.
    if gwid == fx.Int32(0):
        epk.fence_system_release()
        for destination in range_constexpr(fz_npes):
            remote_bigcnt = buffer_ops.buffer_load(
                crfa(p_bc), destination, vec_width=1, dtype=fx.Int64
            )
            remote_bigcnt_rsrc = crfa(remote_bigcnt)
            for local_expert in range(lane, fz_epr, 64):
                ge = fx.Int32(destination * fz_epr) + local_expert
                count = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
                buffer_ops.buffer_store(
                    count,
                    remote_bigcnt_rsrc,
                    fx.Int32(fz_rank * fz_epr) + local_expert,
                )
        fx.rocdl.s_waitcnt(0)
        epk.fence_system_release()
        for peer in range(lane, fz_npes, 64):
            remote_done = buffer_ops.buffer_load(crfa(p_cd), peer, vec_width=1, dtype=fx.Int64)
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            epk.store_i32_system(remote_done, done_index, expected)
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            mori_shmem.int32_wait_until_equals(
                a_cd + fx.Int64(done_index) * fx.Int64(4),
                expected,
            )
        epk.fence_system_acquire()

        # This destination owns compact placement. One lane owns each source
        # rank; wave shuffles form totals/source prefixes while all lanes
        # cooperate on tile metadata.
        valid_source = lane < fx.Int32(fz_npes)
        remote_my_base = fx.Int64(0)
        if valid_source:
            remote_my_base = buffer_ops.buffer_load(crfa(p_mb), lane, vec_width=1, dtype=fx.Int64)
        tile_count = fx.Int32(0)
        local_row_base = fx.Int32(0)
        r_se = crfa(a_se)
        r_trb = crfa(a_trb)
        r_nv = crfa(a_nv)
        r_sm = crfa(a_sm)
        for local_expert in range_constexpr(fz_epr):
            ge = fx.Int32(fz_rank * fz_epr + local_expert)
            lane_count = fx.Int32(0)
            if valid_source:
                lane_count = buffer_ops.buffer_load(
                    r_bc,
                    lane * fx.Int32(fz_epr) + fx.Int32(local_expert),
                    vec_width=1,
                    dtype=fx.Int32,
                )
            total_count = fx.Int32(0)
            sender_prefix = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                source_count = fx.Int32(epk._readlane(lane_count, source))
                total_count = total_count + source_count
                sender_prefix = (fx.Int32(source) < lane).select(
                    sender_prefix + source_count,
                    sender_prefix,
                )
            if valid_source:
                buffer_ops.buffer_store(local_row_base + sender_prefix, crfa(remote_my_base), ge)
            num_tiles = (
                total_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            for tile in range(lane, num_tiles, 64):
                metadata_index = tile_count + tile
                buffer_ops.buffer_store(ge, r_se, metadata_index)
                buffer_ops.buffer_store(
                    local_row_base + tile * fx.Int32(fz_tile_m),
                    r_trb,
                    metadata_index,
                )
            padding = num_tiles * fx.Int32(fz_tile_m) - total_count
            for pad in range(lane, padding, 64):
                buffer_ops.buffer_store(
                    fx.Int32(fz_npes * fz_mtpr),
                    r_sm,
                    local_row_base + total_count + pad,
                )
            tile_count = tile_count + num_tiles
            local_row_base = local_row_base + num_tiles * fx.Int32(fz_tile_m)
        if lane == fx.Int32(0):
            num_valid = tile_count * fx.Int32(fz_tile_m)
            buffer_ops.buffer_store(num_valid, r_nv, fx.Int32(0))
            buffer_ops.buffer_store(num_valid, r_nv, fx.Int32(1))
        fx.rocdl.s_waitcnt(0)
        epk.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = buffer_ops.buffer_load(
                crfa(p_plan_ready), source, vec_width=1, dtype=fx.Int64
            )
            ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            epk.store_i32_system(remote_ready, ready_index, expected)
        fx.rocdl.s_waitcnt(0)
        for destination in range(lane, fz_npes, 64):
            ready_index = parity * fx.Int32(fz_npes) + destination
            mori_shmem.int32_wait_until_equals(
                a_plan_ready + fx.Int64(ready_index) * fx.Int64(4),
                expected,
            )
        epk.fence_system_acquire()
    elif gwid == fx.Int32(1):
        # Build the global-expert exclusive prefix cooperatively: each lane
        # owns a contiguous span, then a wave scan prefixes the lane totals.
        pairs_per_lane = (fz_total_experts + 63) // 64
        lane_base = lane * fx.Int32(pairs_per_lane)
        lane_total = fx.Int32(0)
        lane_counts = []
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            safe_ge = valid_ge.select(ge, fx.Int32(0))
            source_count = buffer_ops.buffer_load(
                r_lh,
                safe_ge,
                vec_width=1,
                dtype=fx.Int32,
            )
            source_count = valid_ge.select(source_count, fx.Int32(0))
            lane_counts.append(source_count)
            lane_total = lane_total + source_count
        lane_prefix = fx.Int32(0)
        for source_lane in range_constexpr(64):
            source_total = fx.Int32(epk._readlane(lane_total, source_lane))
            lane_prefix = (fx.Int32(source_lane) < lane).select(
                lane_prefix + source_total,
                lane_prefix,
            )
        source_prefix = lane_prefix
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            if valid_ge:
                buffer_ops.buffer_store(source_prefix, r_pair_base, ge)
                buffer_ops.buffer_store(source_prefix, r_lc, ge)
            source_prefix = source_prefix + lane_counts[item]
        if lane == fx.Int32(0):
            epk.store_i32_system(a_pair_ready, parity, expected)

    # Warp 1 can group immediately after publishing the pair prefix. Remaining
    # non-communication waves wait only on that local epoch, so grouping stays
    # overlapped with warp 0's cross-rank planning without a CTA-wide barrier.
    if warp > fx.Int32(0):
        if warp > fx.Int32(1):
            if lane == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(
                    a_pair_ready + fx.Int64(parity) * fx.Int64(4),
                    expected,
                )
                epk.fence_agent_acquire()
        group_tid = (warp - fx.Int32(1)) * fx.Int32(64) + lane
        group_threads = fx.Int32((num_waves - 1) * 64)
        for wk in range(group_tid, wl, group_threads):
            expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
            if valid:
                position = fx.Int32(
                    epk.atomic_add_agent(a_lc + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
                )
                buffer_ops.buffer_store(wk, r_pair, position)

    # Every wave drains its own global operations. Kernel completion provides
    # the cross-wave boundary before the same-stream main-kernel launch, so a
    # final CTA barrier would only make completed waves wait for the slowest.
    fx.rocdl.s_waitcnt(0)


@flyc.jit
# fmt: off
def emit_small_fixedslot(*, num_waves, fz_npes, fz_epr, fz_k, fz_mtpr, fz_rank,
    fz_total_experts, fz_cap, fz_tile_m, fz_nbytes, fz_n_i32, fz_safe_end_i32,
    fz_scale_n_i32, fz_enable_scales, addr_small, i32_cur_tok, addr_in_tok,
    addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected):
# fmt: on
    """Graph-safe direct fixed-slot dispatch followed by one global metadata epoch."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rtab = crfa(addr_small)

    def sp(i):
        return buffer_ops.buffer_load(rtab, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    a_running = sp(SmallFixedSlot.RUNNING)
    p_running = sp(SmallFixedSlot.P2P_RUNNING)
    p_rx = sp(SmallFixedSlot.P2P_TOKEN)
    p_sc = sp(SmallFixedSlot.P2P_SCALE)
    p_wts = sp(SmallFixedSlot.P2P_WEIGHT)
    p_sm = sp(SmallFixedSlot.P2P_SRCMAP)
    a_cnt = sp(SmallFixedSlot.EXPERT_COUNT)
    a_se = sp(SmallFixedSlot.SORTED_EXPERT)
    a_trb = sp(SmallFixedSlot.TILE_ROW_BASE)
    a_nv = sp(SmallFixedSlot.NUM_VALID)
    a_done = sp(SmallFixedSlot.ROUTE_DONE)
    a_leader = sp(SmallFixedSlot.LEADER_CLAIM)
    a_meta = sp(SmallFixedSlot.META_READY)
    a_source_done = sp(SmallFixedSlot.SOURCE_DONE)
    p_source_done = sp(SmallFixedSlot.P2P_SOURCE_DONE)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    flat = fx.Int32(fx.block_idx.x)
    wl = i32_cur_tok * fx.Int32(fz_k)
    parity = buffer_ops.buffer_load(crfa(addr_parity), fx.Int32(0), vec_width=1, dtype=fx.Int32)
    expected = buffer_ops.buffer_load(crfa(addr_expected), parity, vec_width=1, dtype=fx.Int32)
    r_idx = crfa(addr_in_idx)
    r_wts = crfa(addr_in_wts)

    warp = tid >> fx.Int32(6)
    is_leader = (wl == fx.Int32(0)) & (flat == fx.Int32(0)) & (
        warp == fx.Int32(0)
    )
    task = flat * fx.Int32(num_waves) + warp
    if task < wl:
        source_token = task // fx.Int32(fz_k)
        topk_slot = task % fx.Int32(fz_k)
        expert = buffer_ops.buffer_load(r_idx, task, vec_width=1, dtype=fx.Int32)
        valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
        safe_expert = valid.select(expert, fx.Int32(0))
        destination = safe_expert // fx.Int32(fz_epr)
        local_expert = safe_expert % fx.Int32(fz_epr)

        slot_in_expert = fx.Int32(0)
        if lane == fx.Int32(0):
            remote_running = buffer_ops.buffer_load(
                crfa(p_running), destination, vec_width=1, dtype=fx.Int64
            )
            if valid:
                slot_in_expert = fx.Int32(
                    epk.atomic_add_system(
                        remote_running + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
        slot_in_expert = fx.Int32(epk._readlane0(slot_in_expert))
        in_range = slot_in_expert < fx.Int32(fz_cap)
        publish = valid & in_range
        slot = local_expert * fx.Int32(fz_cap) + slot_in_expert

        if lane == fx.Int32(0):
            if publish:
                weight = buffer_ops.buffer_load(r_wts, task, vec_width=1, dtype=fx.Float32)
                source_encoding = (
                    fx.Int32(fz_rank * fz_mtpr) + source_token
                ) | (topk_slot << fx.Int32(24))
                wts_remote = buffer_ops.buffer_load(
                    crfa(p_wts), destination, vec_width=1, dtype=fx.Int64
                )
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                buffer_ops.buffer_store(weight_bits, crfa(wts_remote), slot)
                srcmap_remote = buffer_ops.buffer_load(
                    crfa(p_sm), destination, vec_width=1, dtype=fx.Int64
                )
                buffer_ops.buffer_store(source_encoding, crfa(srcmap_remote), slot)

        if const_expr(fz_enable_scales):
            if lane < fx.Int32(fz_scale_n_i32):
                if publish:
                    scale = buffer_ops.buffer_load(
                        crfa(addr_in_sc),
                        source_token * fx.Int32(fz_scale_n_i32) + lane,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                    scale_remote = buffer_ops.buffer_load(
                        crfa(p_sc), destination, vec_width=1, dtype=fx.Int64
                    )
                    buffer_ops.buffer_store(
                        scale,
                        crfa(scale_remote),
                        slot * fx.Int32(fz_scale_n_i32) + lane,
                    )

        token_remote = buffer_ops.buffer_load(
            crfa(p_rx), destination, vec_width=1, dtype=fx.Int64
        )
        source_rsrc = crfa(
            addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes)
        )
        destination_rsrc = crfa(
            token_remote + fx.Int64(slot) * fx.Int64(fz_nbytes)
        )
        lane_offset = lane * fx.Int32(4)
        if const_expr(fz_n_i32 >= 512 and fz_safe_end_i32 > 0):
            copy_end = publish.select(fx.Int32(fz_safe_end_i32), lane_offset)
            for column in range(lane_offset, copy_end, 512):
                value0 = buffer_ops.buffer_load(
                    source_rsrc, column, vec_width=4, dtype=fx.Int32
                )
                value1 = buffer_ops.buffer_load(
                    source_rsrc,
                    column + fx.Int32(256),
                    vec_width=4,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(value0, destination_rsrc, column)
                buffer_ops.buffer_store(
                    value1, destination_rsrc, column + fx.Int32(256)
                )
        if const_expr(fz_safe_end_i32 < fz_n_i32):
            copy_end = publish.select(fx.Int32(fz_n_i32), lane_offset)
            for column in range(
                lane_offset + fx.Int32(fz_safe_end_i32), copy_end, 256
            ):
                value = buffer_ops.buffer_load(
                    source_rsrc, column, vec_width=4, dtype=fx.Int32
                )
                buffer_ops.buffer_store(value, destination_rsrc, column)

        fx.rocdl.s_waitcnt(0)
        done_before = fx.Int32(0)
        if lane == fx.Int32(0):
            epk.fence_system_release()
            done_before = fx.Int32(epk.atomic_add_agent(a_done, fx.Int32(1)))
        done_before = fx.Int32(epk._readlane0(done_before))
        is_leader = is_leader | (done_before == (wl - fx.Int32(1)))

    if (lane == fx.Int32(0)) & is_leader:
        epk.fence_agent_acquire()
        epk.fence_system_release()
        for destination in range_constexpr(fz_npes):
            remote_done = buffer_ops.buffer_load(
                crfa(p_source_done),
                fx.Int32(destination),
                vec_width=1,
                dtype=fx.Int64,
            )
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            epk.store_i32_system(remote_done, done_index, expected)
        for source in range_constexpr(fz_npes):
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(source)
            mori_shmem.int32_wait_until_equals(
                a_source_done + fx.Int64(done_index) * fx.Int64(4),
                expected,
            )
        epk.fence_system_acquire()
        epk.fence_agent_release()
        epk.store_i32_system(a_leader, parity, expected)

    if flat == fx.Int32(0):
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(
                a_leader + fx.Int64(parity) * fx.Int64(4),
                expected,
            )
            epk.fence_agent_acquire()
        fx.barrier()

        r_running = crfa(a_running)
        r_cnt = crfa(a_cnt)
        r_se = crfa(a_se)
        r_trb = crfa(a_trb)
        r_nv = crfa(a_nv)
        local_srcmap = buffer_ops.buffer_load(
            crfa(p_sm), fx.Int32(fz_rank), vec_width=1, dtype=fx.Int64
        )
        r_srcmap = crfa(local_srcmap)
        block_threads = fx.Int32(num_waves * 64)

        for local_expert in range(tid, fz_epr, block_threads):
            count = buffer_ops.buffer_load(
                r_running,
                local_expert,
                vec_width=1,
                dtype=fx.Int32,
            )
            safe_count = (count <= fx.Int32(fz_cap)).select(
                count, fx.Int32(fz_cap)
            )
            buffer_ops.buffer_store(
                safe_count, r_cnt, local_expert
            )
            buffer_ops.buffer_store(fx.Int32(0), r_running, local_expert)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()

        for local_expert in range(tid, fz_epr, block_threads):
            safe_count = buffer_ops.buffer_load(
                r_cnt,
                local_expert,
                vec_width=1,
                dtype=fx.Int32,
            )
            tile_count = fx.Int32(0)
            for previous_expert in range(fx.Int32(0), local_expert, 1):
                previous_count = buffer_ops.buffer_load(
                    r_cnt,
                    previous_expert,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                tile_count = tile_count + (
                    previous_count + fx.Int32(fz_tile_m - 1)
                ) // fx.Int32(fz_tile_m)
            num_tiles = (
                safe_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            for tile in range(fx.Int32(0), num_tiles, 1):
                metadata_index = tile_count + tile
                buffer_ops.buffer_store(
                    fx.Int32(fz_rank * fz_epr + local_expert),
                    r_se,
                    metadata_index,
                )
                buffer_ops.buffer_store(
                    fx.Int32(local_expert * fz_cap)
                    + tile * fx.Int32(fz_tile_m),
                    r_trb,
                    metadata_index,
                )
            padded_count = num_tiles * fx.Int32(fz_tile_m)
            for padding_row in range(safe_count, padded_count, 1):
                fixed_row = (
                    fx.Int32(local_expert * fz_cap) + padding_row
                )
                buffer_ops.buffer_store(
                    fx.Int32(fz_npes * fz_mtpr),
                    r_srcmap,
                    fixed_row,
                )
        fx.rocdl.s_waitcnt(0)
        fx.barrier()

        if tid == fx.Int32(0):
            tile_count = fx.Int32(0)
            for local_expert in range(fx.Int32(0), fx.Int32(fz_epr), 1):
                count = buffer_ops.buffer_load(
                    r_cnt,
                    local_expert,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                tile_count = tile_count + (
                    count + fx.Int32(fz_tile_m - 1)
                ) // fx.Int32(fz_tile_m)
            num_valid = tile_count * fx.Int32(fz_tile_m)
            buffer_ops.buffer_store(num_valid, r_nv, fx.Int32(0))
            buffer_ops.buffer_store(num_valid, r_nv, fx.Int32(1))
            fx.rocdl.s_waitcnt(0)
            epk.fence_system_release()
            epk.fence_agent_release()
            buffer_ops.buffer_store(expected, crfa(a_meta), parity)

    if tid == fx.Int32(0):
        mori_shmem.int32_wait_until_equals(
            a_meta + fx.Int64(parity) * fx.Int64(4),
            expected,
        )
        epk.fence_agent_acquire()
    fx.barrier()


@flyc.jit
# fmt: off
def emit_dispatch_payload(*, num_waves, fz_epr, fz_k, fz_mtpr, fz_rank,
    fz_total_experts, fz_nbytes, fz_n_i32, fz_safe_end_i32, fz_scale_n_i32, fz_enable_scales,
    addr_disp, addr_in_tok, addr_in_wts, addr_in_sc, dispatch_blocks, addr_parity,
    wave_task_producer):
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
    a_lc = dp(DispatchSlot.LOCAL_CURSOR)
    p_payload_ready = dp(DispatchSlot.P2P_PAYLOAD_READY)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    flat = fx.Int32(fx.block_idx.x)
    r_pair_base = crfa(a_pair_base)
    r_lh = crfa(a_lh)
    r_mb = crfa(a_mb)
    r_lc = crfa(a_lc)
    r_pair = crfa(a_pair_order)
    r_wts = crfa(addr_in_wts)
    parity_rsrc = crfa(addr_parity)
    parity = buffer_ops.buffer_load(parity_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)

    if const_expr(wave_task_producer):
        task0 = flat * fx.Int32(num_waves) + warp
        task_stride = fx.Int32(dispatch_blocks * num_waves)
        row0 = fx.Int32(0)
        row_stride = fx.Int32(1)
    else:
        task0 = flat
        task_stride = fx.Int32(dispatch_blocks)
        row0 = warp
        row_stride = fx.Int32(num_waves)

    def _publish_task(destination, local_expert, ge):
        epk.fence_system_release()
        ready_remote = buffer_ops.buffer_load(
            crfa(p_payload_ready), destination, vec_width=1, dtype=fx.Int64
        )
        ready_index = parity * fx.Int32(fz_epr) + local_expert
        epk.atomic_add_system(
            ready_remote + fx.Int64(ready_index) * fx.Int64(4),
            fx.Int32(1),
        )
        buffer_ops.buffer_store(fx.Int32(0), r_lh, ge)
        buffer_ops.buffer_store(fx.Int32(0), r_lc, ge)

    # Sparse configurations assign one expert task per wave; dense
    # configurations retain CTA-cooperative row striping.
    num_destinations = fz_total_experts // fz_epr
    for task in range(task0, fz_total_experts, task_stride):
        # Match the consumer's local-expert-major order while spreading each
        # producer round across destinations. Destination-major global expert
        # IDs create an all-sources-to-one-rank incast.
        local_expert = task // fx.Int32(num_destinations)
        destination = task % fx.Int32(num_destinations)
        ge = destination * fx.Int32(fz_epr) + local_expert
        source_end = buffer_ops.buffer_load(
            r_lc,
            ge,
            vec_width=1,
            dtype=fx.Int32,
        )
        source_base = buffer_ops.buffer_load(r_pair_base, ge, vec_width=1, dtype=fx.Int32)
        source_count = source_end - source_base
        destination_base = buffer_ops.buffer_load(r_mb, ge, vec_width=1, dtype=fx.Int32)
        for row in range(row0, source_count, row_stride):
            wk = buffer_ops.buffer_load(r_pair, source_base + row, vec_width=1, dtype=fx.Int32)
            source_token = wk // fx.Int32(fz_k)
            topk_slot = wk % fx.Int32(fz_k)
            destination_row = destination_base + row
            if lane == fx.Int32(0):
                weight = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                source_encoding = (
                    fx.Int32(fz_rank * fz_mtpr) + source_token
                ) | (topk_slot << fx.Int32(24))
                wts_remote = buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64)
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                buffer_ops.buffer_store(weight_bits, crfa(wts_remote), destination_row)
                srcmap_remote = buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64)
                buffer_ops.buffer_store(source_encoding, crfa(srcmap_remote), destination_row)
            if const_expr(fz_enable_scales):
                if lane < fx.Int32(fz_scale_n_i32):
                    scale = buffer_ops.buffer_load(
                        crfa(addr_in_sc),
                        source_token * fx.Int32(fz_scale_n_i32) + lane,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                    scale_remote = buffer_ops.buffer_load(
                        crfa(p_sc), destination, vec_width=1, dtype=fx.Int64
                    )
                    buffer_ops.buffer_store(
                        scale,
                        crfa(scale_remote),
                        destination_row * fx.Int32(fz_scale_n_i32) + lane,
                    )
            token_remote = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
            source_rsrc = crfa(addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes))
            destination_rsrc = crfa(token_remote + fx.Int64(destination_row) * fx.Int64(fz_nbytes))
            lane_offset = lane * fx.Int32(4)
            if const_expr(fz_n_i32 >= 512 and fz_safe_end_i32 > 0):
                for column in range(lane_offset, fz_safe_end_i32, 512):
                    value0 = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    value1 = buffer_ops.buffer_load(
                        source_rsrc,
                        column + fx.Int32(256),
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                    buffer_ops.buffer_store(value0, destination_rsrc, column)
                    buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
            if const_expr(fz_safe_end_i32 < fz_n_i32):
                for column in range(lane_offset + fz_safe_end_i32, fz_n_i32, 256):
                    value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    buffer_ops.buffer_store(value, destination_rsrc, column)
            elif const_expr(fz_n_i32 < 512):
                for column in range(lane_offset, fz_n_i32, 256):
                    value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    buffer_ops.buffer_store(value, destination_rsrc, column)
        fx.rocdl.s_waitcnt(0)
        if const_expr(wave_task_producer):
            if lane == fx.Int32(0):
                _publish_task(destination, local_expert, ge)
        else:
            fx.barrier()
            if tid == fx.Int32(0):
                _publish_task(destination, local_expert, ge)
            fx.barrier()


@flyc.jit
def wait_expert_payload(addr_payload_ready, local_expert, epoch_parity, epoch_expected, fz_epr):
    """Acquire one expert's payload immediately before its GEMM tile."""
    if fx.thread_idx.x == fx.Int32(0):
        ready_index = epoch_parity * fx.Int32(fz_epr) + local_expert
        mori_shmem.int32_wait_until_greater_than(
            addr_payload_ready + fx.Int64(ready_index) * fx.Int64(4),
            epoch_expected - fx.Int32(1),
        )
        epk.fence_system_acquire()
    fx.barrier()
