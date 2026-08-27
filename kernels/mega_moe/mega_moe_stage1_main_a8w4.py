# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused stage1 with low-ID dispatch producers and oversubscribed FP8xFP4 grouped-GEMM1 consumers."""

import functools

import mori.ir.flydsl as mori_shmem
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from kernels.comm import communication_ops_utils as comm_ops
from kernels.common.tensor_shim import _run_compiled
from .dispatch import (
    DispatchSlot,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_payload,
    emit_dispatch_plan,
)
from .gemm1 import _LdsF32View, build_fused_gemm1
from .gemm_util import (
    _buffer_load,
    _buffer_store,
    _make_buffer,
    _make_buffer_from_addr,
)
from .mega_moe_stage1 import A8W4_ENTRY_COUNT_SHARDS

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32


def ceildiv(a, b):
    return (a + b - 1) // b


def _use_direct_fixed_slot(
    enabled, npes, experts_per_rank, max_tokens_per_rank, cap, tile_m
):
    if not enabled or tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    return npes == 8 and experts_per_rank == 48 and cap == required_cap


def _validate_dispatch_capacity(
    batch_size,
    npes,
    experts_per_rank,
    topk,
    tile_m,
    row_bytes,
    output_row_bytes,
    use_tile_resource,
):
    max_rows = npes * batch_size * topk + experts_per_rank * tile_m
    if not use_tile_resource and max_rows * row_bytes >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError(
            "MegaMoE v2 stage1 payload exceeds the 32-bit buffer-resource ABI"
        )
    if (
        not use_tile_resource
        and max_rows * output_row_bytes >= _BUFFER_OFFSET_ABI_BYTES
    ):
        raise ValueError(
            "MegaMoE v2 stage1 output exceeds the 32-bit buffer-resource ABI"
        )


def _validate_bounded_consumer_mapping(launch_grid_x, role_prefix_blocks):
    control_blocks = role_prefix_blocks + 1
    initial_consumers = launch_grid_x - control_blocks
    if initial_consumers <= 0:
        raise ValueError("bounded dispatch must admit an initial consumer")
    ordinals = [
        ticket - control_blocks
        if ticket >= control_blocks
        else initial_consumers + ticket
        for ticket in range(launch_grid_x)
    ]
    if sorted(ordinals) != list(range(launch_grid_x)):
        raise ValueError("bounded consumer mapping is not bijective")
    return initial_consumers


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fixed_slot_dispatch: bool, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    swiglu_limit: float = 0.0,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage1 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    NUM_WAVES = int(num_waves)
    assert NUM_WAVES > 1, "planner needs one communication wave and at least one grouping wave"
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    dispatch_blocks = int(num_dispatch_cu)
    # Native A8W4 has one immutable decode configuration per operator instance,
    # so all graph generations can reuse one sharded admission counter.
    grid_epoch_slot = 0
    assert 0 < dispatch_blocks < num_cu, "num_dispatch_cu must be in [1, num_cu)"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    planner_blocks = 1
    role_prefix_blocks = dispatch_blocks
    bounded_compact_roles = not fixed_slot_dispatch
    if planner_blocks + role_prefix_blocks >= num_cu:
        raise ValueError(
            "planner and producer roles must leave at least one resident consumer CU"
        )
    # Bound compact launches to the configured persistent grid.  Their
    # finite planner/producer prefix joins the GEMM queue after dispatch, so a
    # shape change cannot leave a replacement cohort on a stale generation.
    # Fixed-slot dispatch retains replacement CTAs for its retired roles.
    if bounded_compact_roles:
        replacement_blocks = 0
        launch_grid_x = num_cu * grid_mult
        initial_consumer_blocks = _validate_bounded_consumer_mapping(
            launch_grid_x, role_prefix_blocks
        )
    else:
        replacement_blocks = planner_blocks + role_prefix_blocks
        if replacement_blocks <= 0:
            raise ValueError("retired roles require a positive replacement cohort")
        launch_grid_x = num_cu * grid_mult + replacement_blocks
        initial_consumer_blocks = 0
    grid_x = launch_grid_x - planner_blocks - role_prefix_blocks
    assert grid_x > 0, "consumer grid must remain positive"
    assert launch_grid_x <= num_cu * 33 + 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2
    assert TILE_K_BYTES % 128 == 0
    A_K_STEP_BYTES = tile_k
    assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    if fz_npes * fz_mtpr > 1 << 24:
        raise ValueError("MegaMoE v2 source-token encoding exceeds 24 bits")
    if fz_k > 1 << 8:
        raise ValueError("MegaMoE v2 top-k slot encoding exceeds 8 bits")
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    direct_fixed_slot = _use_direct_fixed_slot(
        fixed_slot_dispatch, fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m
    )
    # Direct fixed-slot dispatch publishes PLAN_READY only after its complete
    # payload is visible, so consumers use a deterministic ordinal/stride
    # mapping. Tile-ready publication is reserved for chunked compact payloads.
    fz_total_experts = fz_npes * fz_epr
    # Small batches stream B; large batches cache it across M tiles.
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fz_mtpr <= 512 else 0)
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if direct_fixed_slot and fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512
    _validate_dispatch_capacity(
        fz_mtpr, fz_npes, fz_epr, fz_k, fz_tile_m, fz_nbytes, inter_dim, use_tile_resource
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    role_suffix = (
        f"rb{replacement_blocks}"
        f"bc{int(bounded_compact_roles)}"
    )
    kernel_name = (
        f"megamoe_stage1_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_r{fz_rank}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}"
        f"{role_suffix}"
        f"{swiglu_suffix}_dualslot1_norelayagent1_nestedexpert1"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_pair_order_ready = _disp_ptr(DispatchSlot.PAIR_ORDER_READY)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
        p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_local_hist = _disp_ptr(DispatchSlot.LOCAL_HIST)

        block_index, _, _ = fx.block_idx
        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            entry_shard = block_index & fx.Int32(A8W4_ENTRY_COUNT_SHARDS - 1)
            shard_population = (
                fx.Int32(launch_grid_x - 1) - entry_shard
            ) // fx.Int32(A8W4_ENTRY_COUNT_SHARDS) + fx.Int32(1)
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(
                    a_entry_count
                    + fx.Int64(
                        (grid_epoch_slot * A8W4_ENTRY_COUNT_SHARDS) * 8
                    )
                    + fx.Int64(entry_shard) * fx.Int64(8),
                    fx.Int64(1),
                )
            )
            entry_value64 = ticket64 // fx.Int64(shard_population)
            fx.ptr_store(Vec.from_elements([entry_value64], fx.Int64), ticket_scratch)
        fx.barrier()
        entry_value = Vec(ticket_view.load())[0]
        generation = entry_value
        ticket = block_index
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        compact_owner = ticket == fx.Int32(0)
        compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)

        if compact_owner:
            next_parity_lane = fx.Int32(0)
            launch_epoch_lane = fx.Int32(0)
            if tid == fx.Int32(0):
                old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                next_parity_lane = old_parity ^ fx.Int32(1)
                previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
                next_expected = previous_expected + fx.Int32(fz_npes)
                _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
                launch_epoch_lane = (
                    (next_expected // fx.Int32(fz_npes)) * fx.Int32(2) - next_parity_lane
                )
            next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
            launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
            if tid < fx.Int32(fz_npes):
                peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                comm_ops.fence_system_release()
                launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
                remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
                comm_ops.store_i32_system(remote_launch_ready, fx.Int32(fz_rank), launch_epoch)
                mori_shmem.int32_wait_until_greater_than(
                    a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
                )
                comm_ops.fence_system_acquire()
            if tid == fx.Int32(0):
                if const_expr(direct_fixed_slot):
                    group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                    for destination in range_constexpr(fz_npes):
                        _buffer_store(group_done_rsrc, fx.Int32(destination), fx.Int32(0), fx.Int32)
            if const_expr(not direct_fixed_slot):
                local_hist_rsrc = _make_buffer_from_addr(a_local_hist, fx.Int32)
                for expert in range(tid, fz_total_experts, TOTAL_THREADS):
                    _buffer_store(local_hist_rsrc, expert, fx.Int32(0), fx.Int32)
                fx.rocdl.s_waitcnt(0)
            fx.barrier()
            if tid == fx.Int32(0):
                fx.rocdl.s_waitcnt(0)
                comm_ops.fence_agent_release()
                _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                fx.rocdl.s_waitcnt(0)
                comm_ops.fence_agent_release()
                comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
        else:
            if tid == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
                comm_ops.fence_agent_acquire()
            fx.barrier()

        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)

        if compact_owner:  # noqa: SIM102 - keep the device and compile-time branches separate.
            if const_expr(not direct_fixed_slot):
                emit_dispatch_plan(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr,
                    fz_rank=fz_rank, fz_tile_m=fz_tile_m, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, parity=payload_parity,
                    expected=payload_expected,
                    external_grouping=False, external_counting=False,
                    dispatch_blocks=dispatch_blocks,
                )

        if compact_producer:
            if const_expr(direct_fixed_slot):
                emit_direct_fixed_slot_payload(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales,
                    fz_route_payload=False, smoothquant_mode="none",
                    smooth_rsrc=None, smooth_scale_rsrc=None,
                    model_dim=model_dim, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                    parity=payload_parity, expected=payload_expected,
                )
            else:
                if tid == fx.Int32(0):
                    mori_shmem.int32_wait_until_equals(
                        a_pair_order_ready + fx.Int64(payload_parity) * fx.Int64(4), payload_expected)
                    comm_ops.fence_agent_acquire()
                fx.barrier()
                producers_per_destination = fx.Int32(dispatch_blocks // fz_npes)
                emit_dispatch_payload(
                    num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32, fz_safe_end_i32=fz_safe_end_i32,
                    fz_scale_n_i32=fz_scale_n_i32,
                    fz_enable_scales=fz_enable_scales,
                    fz_route_payload=False, fz_copy_payload=True,
                    fz_mxfp4_smooth_pload=False,
                    smoothquant_mode="none", smooth_rsrc=None,
                    model_dim=model_dim, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok,
                    addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    dispatch_blocks=dispatch_blocks,
                    producer_slot=producer_slot, parity=payload_parity,
                    expected=payload_expected,
                    producers_per_destination=producers_per_destination,
                )
        if const_expr(direct_fixed_slot):
            if compact_owner:
                emit_direct_fixed_slot_finalize(
                    fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_tile_m=fz_tile_m, addr_disp=addr_disp, parity=payload_parity,
                    expected=payload_expected,
                )
        # Compact bounded roles perform finite dispatch work, then reuse the
        # resident CTA as a GEMM consumer.  Other paths retain role retirement.
        consumer_active = (
            fx.Int32(1) == fx.Int32(1)
            if const_expr(bounded_compact_roles)
            else ticket > fx.Int32(role_prefix_blocks)
        )
        if consumer_active:
            if const_expr(not direct_fixed_slot):
                payload_table = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.P2P_PAYLOAD_READY)), fx.Int64)
                addr_payload_ready = _buffer_load(
                    _make_buffer_from_addr(payload_table, fx.Int64), fx.Int32(fz_rank), fx.Int64
                )
            wave_id = fx.thread_idx.x // 64

            w_rsrc = _make_buffer(w, fx.Int32, 4)
            sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
            sw_rsrc = _make_buffer(scale_w, fx.Int32)
            trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
            expert_rsrc = _make_buffer(expert_ids, fx.Int32)
            nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
            scale_cols = (inter_dim // 32 + 7) // 8 * 8
            os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
            if const_expr(use_tile_resource):
                out_rsrc = None
            else:
                out_nbytes = tokens * fx.Int32(inter_dim)
                out_rsrc = _make_buffer(out, fx.Int16, max_size=False, num_records_bytes=out_nbytes)
            os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

            expert_of_flat, _do_scheduled_tile = build_fused_gemm1(
                x_tensor=x, w_rsrc=w_rsrc,
                sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
                trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
                a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
                model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
                tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
                m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
                total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
                n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
                swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
                async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
                swiglu_limit=swiglu_limit,
            )

            if tid == fx.Int32(0):
                local_plan_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.PLAN_READY)), fx.Int64)
                ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                mori_shmem.int32_wait_until_equals(
                    local_plan_ready + fx.Int64(ready_index) * fx.Int64(4), payload_expected)
                comm_ops.fence_agent_acquire()
            fx.barrier()

            num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
            num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
            total_work = num_m_tiles * fx.Int32(N_TILES)

            def _wait_payload(flat):
                pe = expert_of_flat(flat)
                pe_index = payload_parity * fx.Int32(fz_epr) + pe
                mori_shmem.int32_wait_until_equals(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), payload_expected
                )

            if const_expr(direct_fixed_slot):
                work = ticket - fx.Int32(role_prefix_blocks + 1)
                while work < total_work:
                    _do_scheduled_tile(work)
                    work = work + fx.Int32(grid_x)
            else:
                # Initially resident consumers own the low ordinals.  The
                # finite owner/producer prefix joins with disjoint tail
                # ordinals after communication, and every CTA strides by the
                # complete bounded launch population.
                control_blocks = fx.Int32(role_prefix_blocks + 1)
                initial_consumers = fx.Int32(initial_consumer_blocks)
                ordinary_consumer = ticket >= control_blocks
                work = ordinary_consumer.select(
                    ticket - control_blocks,
                    initial_consumers + ticket,
                )
                while work < total_work:
                    if tid == fx.Int32(0):
                        _wait_payload(work)
                    fx.barrier()
                    comm_ops.fence_system_acquire()
                    _do_scheduled_tile(work)
                    work = work + fx.Int32(launch_grid_x)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, swiglu_limit=0.0):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, stream,
    )
# fmt: on
