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
from kernels.common import buffer_ops
from kernels.common.tensor_shim import _run_compiled
from kernels.gemm.fp8_gemm_utils import ceildiv

from .dispatch import (
    DispatchSlot,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_group,
    emit_dispatch_payload,
    emit_dispatch_plan,
    emit_exact_total_recv,
)
from .gemm1_role import _LdsF32View, build_fused_gemm1, build_fused_int8_gemm1
from .gemm_util_role import (
    _buffer_load,
    _buffer_store,
    _make_buffer,
    _make_buffer_from_addr,
    wait_all,
)
from .mxfp4_smoothquant import (
    convert_mxfp4_smoothquant_row,
    convert_mxfp4_smoothquant_row_wavewise,
)

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32


def _stage1_quant_traits(quant_mode: str, inter_dim: int, tile_n: int):
    """Return compile-time Stage1 geometry for a quant mode."""
    if quant_mode not in ("a8w4", "a8w4smooth", "w8a8smooth"):
        raise ValueError(f"unsupported Stage1 quant_mode={quant_mode!r}")
    is_int8 = quant_mode in ("a8w4smooth", "w8a8smooth")
    gemm_n = int(inter_dim) if is_int8 else 2 * int(inter_dim)
    if tile_n <= 0 or gemm_n % int(tile_n):
        raise ValueError(
            f"Stage1 N={gemm_n} must tile evenly by tile_n={tile_n}"
        )
    return {
        "is_int8": is_int8,
        "packed_int4": quant_mode == "a8w4smooth",
        "gemm_n": gemm_n,
        "n_tiles": gemm_n // int(tile_n),
        "uses_mx_scale_lds": not is_int8,
    }


def _use_direct_fixed_slot(enabled, npes, experts_per_rank, max_tokens_per_rank, cap, tile_m):
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
    flat_output=False,
):
    max_rows = npes * batch_size * topk + experts_per_rank * tile_m
    if not use_tile_resource and max_rows * row_bytes >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE v2 stage1 payload exceeds the 32-bit buffer-resource ABI")
    if not use_tile_resource and not flat_output and max_rows * output_row_bytes >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE v2 stage1 output exceeds the 32-bit buffer-resource ABI")


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fixed_slot_dispatch: bool, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    work_shards: int | None = None, external_grouping: bool | None = None,
    external_counting: bool | None = None, payload_chunk_rows: int = 0, payload_tile_ready: bool = False,
    swiglu_limit: float = 0.0, quant_mode: str = "a8w4", mxfp4_transport: bool = False,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage1 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    traits = _stage1_quant_traits(quant_mode, inter_dim, tile_n)
    is_int8 = traits["is_int8"]
    packed_int4 = traits["packed_int4"]
    contract = (
        model_dim,
        inter_dim,
        experts_per_rank,
        fuse_npes,
        fuse_topk,
        fuse_mtpr,
        quant_mode,
        bool(mxfp4_transport),
        bool(fixed_slot_dispatch),
    )
    expected_contract = (
        3584, 1280, 48, 8, 8, 32768, "w8a8smooth", True, False
    )
    if contract != expected_contract:
        raise ValueError(
            "role-specialized MXFP4 Stage1 is restricted to M13 t4096 "
            f"prefill, got {contract!r}"
        )
    if int(num_cu) != 256:
        raise ValueError("role-specialized MXFP4 Stage1 requires 256 CUs")

    # Frozen, previously validated t4096 configuration.  Keeping these as
    # Python booleans/integers makes every branch compile-time and prevents the
    # role path from changing native A8W4 or standalone A8W4Smooth kernels.
    int8_pipe_weights = False
    debug_skip_conversion = False
    debug_skip_gemm = False
    debug_skip_total_recv = False
    debug_skip_payload_copy = False
    debug_no_consumer_work = False
    fused_mxfp4_aload = False
    int8_direct_output = False
    m_tile_consumer = False
    static_rxq_requested = False
    static_consumer_work = False
    role_specialized_rxq = True
    ready_tile_queue = False
    role_rxq_converter_blocks = 255
    dual_row_conversion = False
    static_rxq_pipeline = False
    static_rxq_converter_blocks = 128
    static_rxq_gemm_blocks = num_cu
    static_rxq_interleaved = False
    wavewise_conversion = True
    last_converter_gemm = False
    lds_grouping = False
    count_allgather_plan = False
    fused_count_total_recv = False
    NUM_WAVES = int(num_waves)
    assert NUM_WAVES > 1, "planner needs one communication wave and at least one grouping wave"
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    N_TILES = traits["n_tiles"]
    GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    assert grid_mult in GRID_MULT_VALUES, "grid_mult out of range"
    grid_epoch_slot = GRID_MULT_VALUES.index(grid_mult)
    dispatch_blocks = int(num_dispatch_cu)
    payload_chunk_rows = int(payload_chunk_rows)
    assert 0 < dispatch_blocks < num_cu, "num_dispatch_cu must be in [1, num_cu)"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    payload_block_cap = 4
    assert 1 <= payload_block_cap <= dispatch_blocks // fuse_npes
    lds_group_blocks = fuse_npes * min(
        payload_block_cap, dispatch_blocks // fuse_npes
    )
    if payload_chunk_rows:
        assert not fixed_slot_dispatch and payload_chunk_rows % sort_block_m == 0
    assert not payload_tile_ready or payload_chunk_rows > 0
    planner_blocks = 1
    # The established persistent path keeps the whole fused grid on an exact CU
    # multiple.  The static experiment mirrors Primus-Turbo's role-specialized
    # launch instead: planner/producers are a low-ticket prefix that retires,
    # followed by an oversubscribed, statically mapped consumer grid.  Queued
    # consumers backfill CUs as producers finish without contending on work_head.
    # Each static CTA owns exactly one work item; this is important because a
    # strided persistent loop can create cyclic qcount dependencies between
    # resident and queued N-tile peers.
    if role_specialized_rxq:
        # Queue static GEMM work behind a complete converter prefix.  The prefix
        # may be wider than the dispatch cohort: those extra CTAs keep GEMM work
        # out of the first resident wave and retire after planning.  Dispatch
        # CTAs decode/requantize after their PUSH task, then retire; hardware
        # admits GEMM-only CTAs without a work-head atomic.
        requested_consumers = 1600
        grid_x = (
            (requested_consumers + N_TILES - 1) // N_TILES
        ) * N_TILES
    elif static_rxq_pipeline:
        # Put the complete converter cohort before an over-applied GEMM cohort.
        # With one 128-KiB CTA resident per CU, interleaving one converter with
        # N_TILES waiting GEMM CTAs starves conversion.  The fixed cohorts keep
        # every converter resident while a first GEMM wave overlaps on the
        # remaining CUs; queued GEMM CTAs automatically backfill producer and
        # converter CUs as those roles retire.  Both cohorts stride, so there
        # is no work-head atomic and no load-imbalance-dependent M-tile bound.
        grid_x = static_rxq_converter_blocks + static_rxq_gemm_blocks
    elif static_consumer_work:
        # Never truncate an M tile: every conversion partition waits for all
        # N-tile peers before GEMM, so a partial tail group would deadlock.
        static_target = num_cu * grid_mult
        grid_x = ((static_target + N_TILES - 1) // N_TILES) * N_TILES
    else:
        grid_x = num_cu * grid_mult - planner_blocks - dispatch_blocks
    assert grid_x > 0, "consumer grid must remain positive"
    role_prefix_blocks = (
        role_rxq_converter_blocks if role_specialized_rxq else dispatch_blocks
    )
    launch_grid_x = planner_blocks + role_prefix_blocks + grid_x
    assert launch_grid_x <= (1 << 31) - 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k if is_int8 else tile_k // 2
    assert TILE_K_BYTES % (64 if is_int8 else 128) == 0
    A_K_STEP_BYTES = tile_k
    assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    WORK_SHARDS = 4 if work_shards is None and int(fuse_mtpr) >= 8192 else 8
    if work_shards is not None:
        WORK_SHARDS = int(work_shards)
    assert WORK_SHARDS in (1, 2, 4, 8)

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n if is_int8 else tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = (
        2 * a_lds_size
        if int8_direct_output
        else max(2 * a_lds_size, cs_size * 4)
    )
    # Smooth INT8 has one f32 scale per row, consumed once in the epilogue.
    # Do not reserve or populate the MX per-K scale LDS.
    n_scale_bytes = 16 if is_int8 else sort_block_m * (model_dim // 32)

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    if fz_npes * fz_mtpr > 1 << 24:
        raise ValueError("MegaMoE v2 source-token encoding exceeds 24 bits")
    if fz_k > 1 << 8:
        raise ValueError("MegaMoE v2 top-k slot encoding exceeds 8 bits")
    if external_grouping is None:
        external_grouping = fz_mtpr >= 2048 and fz_npes == 8 and fz_epr == 48
    if external_counting is None:
        external_counting = external_grouping and fz_mtpr >= 8192
    assert not external_counting or external_grouping
    if fused_count_total_recv and external_counting:
        raise ValueError(
            "fused count/TOTAL_RECV requires planner-owned counting"
        )
    if count_allgather_plan and NUM_WAVES != fz_npes:
        raise ValueError(
            "count-allgather planning requires one wave per rank"
        )
    if lds_grouping and not external_counting:
        raise ValueError("LDS grouping requires external_counting")
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    direct_fixed_slot = _use_direct_fixed_slot(
        fixed_slot_dispatch, fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m
    )
    fz_total_experts = fz_npes * fz_epr
    # Small batches stream B; large batches cache it across M tiles.
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fz_mtpr <= 512 else 0)
    fz_n_i32 = model_dim // (8 if mxfp4_transport else 4)
    fz_nbytes = model_dim // (2 if mxfp4_transport else 1)
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if direct_fixed_slot and fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512
    int8_output_nbytes = fz_npes * fz_mtpr * fz_k * inter_dim * 2
    int8_flat_output = is_int8 and int8_output_nbytes >= _BUFFER_OFFSET_ABI_BYTES
    _validate_dispatch_capacity(
        fz_mtpr,
        fz_npes,
        fz_epr,
        fz_k,
        fz_tile_m,
        fz_nbytes,
        inter_dim * (2 if is_int8 else 1),
        use_tile_resource,
        flat_output=int8_flat_output,
    )
    quant_count_slots = (
        (fz_npes * fz_mtpr * fz_k + fz_epr * fz_tile_m + fz_tile_m - 1) // fz_tile_m
        if not fixed_slot_dispatch
        else (fz_epr * fz_cap + 256 + fz_tile_m - 1) // fz_tile_m
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    kernel_name = (
        f"megamoe_stage1_{quant_mode}_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}_ws{WORK_SHARDS}"
        f"_pc{payload_chunk_rows}"
        f"_ptr{int(payload_tile_ready)}"
        f"_mx4t{int(mxfp4_transport)}_qcount_reset_release"
        f"_mxa{int(fused_mxfp4_aload)}"
        f"_depi{int(int8_direct_output)}"
        f"_ipw{int(int8_pipe_weights)}"
        f"_dbgcv{int(debug_skip_conversion)}g{int(debug_skip_gemm)}"
        f"tr{int(debug_skip_total_recv)}"
        f"pcopy{int(not debug_skip_payload_copy)}"
        f"cw{int(not debug_no_consumer_work)}"
        f"scw{int(static_consumer_work)}"
        f"rsrxq{int(role_specialized_rxq)}"
        f"rxc{role_rxq_converter_blocks if role_specialized_rxq else 0}"
        f"rtq{int(ready_tile_queue)}"
        f"dr{int(dual_row_conversion)}"
        f"wcv{int(wavewise_conversion)}"
        f"rxq{int(static_rxq_pipeline)}"
        f"i{int(static_rxq_interleaved)}"
        f"c{static_rxq_converter_blocks if static_rxq_pipeline else 0}"
        f"g{static_rxq_gemm_blocks if static_rxq_pipeline else 0}"
        f"_lcg{int(last_converter_gemm)}"
        f"_mtc{int(m_tile_consumer)}"
        f"_lg{int(lds_grouping)}"
        f"_cagp{int(count_allgather_plan)}"
        f"_fctr{int(fused_count_total_recv)}"
        f"_pbc{payload_block_cap}"
        f"_qp{int(packed_int4)}{swiglu_suffix}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        compact_src: fx.Tensor, compact_experts: fx.Tensor, compact_weights: fx.Tensor,
        qscale_w: fx.Tensor, qzero_w: fx.Tensor, transport_smooth: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, addr_quant_count: fx.Int64,
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
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        if const_expr(ready_tile_queue):
            a_work_tail = _disp_ptr(DispatchSlot.WORK_TAIL)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
        if const_expr(payload_tile_ready):
            a_payload_chunks_per_destination = _disp_ptr(
                DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION
            )
        a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
        p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_payload_ready_rows = _disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)
        transport_smooth_rsrc = buffer_ops.create_buffer_resource(
            transport_smooth, max_size=True
        )

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(
                    a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1)
                )
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid_x)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        compact_owner = ticket == fx.Int32(0)
        compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)
        role_rxq_converter = compact_producer
        role_rxq_converter_slot = producer_slot

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
            if const_expr(payload_tile_ready):
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m))
                    comm_ops.fence_system_release()
                fx.barrier()
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
                work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                for shard in range_constexpr(8):
                    _buffer_store(work_head_rsrc, fx.Int32(shard * 16), fx.Int32(0), fx.Int32)
                if const_expr(ready_tile_queue):
                    _buffer_store(
                        _make_buffer_from_addr(a_work_tail, fx.Int32),
                        fx.Int32(0),
                        fx.Int32(0),
                        fx.Int32,
                    )
                if const_expr(external_grouping or direct_fixed_slot):
                    group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                    # Direct completion slots carry the epoch value and do not
                    # need a reset. Compact external grouping retains its
                    # rank-local counter protocol in slot 0.
                    if const_expr(not direct_fixed_slot):
                        _buffer_store(
                            group_done_rsrc,
                            fx.Int32(0),
                            fx.Int32(0),
                            fx.Int32,
                        )
            if const_expr(
                mxfp4_transport and not fused_mxfp4_aload and not m_tile_consumer
            ):
                quant_count_rsrc = _make_buffer_from_addr(addr_quant_count, fx.Int32)
                for base in range_constexpr(0, quant_count_slots, TOTAL_THREADS):
                    count_index = tid + fx.Int32(base)
                    if count_index < fx.Int32(quant_count_slots):
                        _buffer_store(
                            quant_count_rsrc,
                            count_index,
                            fx.Int32(0),
                            fx.Int32,
                        )
                # Every lane owns a slice of the counter reset.  A wait/fence in
                # tid0 cannot drain VMEM stores issued by sibling waves; publishing
                # the launch gate before those stores complete lets consumers see
                # a previous replay's N_TILES value and enter GEMM before all rows
                # of the current tile have been converted.
                wait_all()
                comm_ops.fence_agent_release()
            fx.barrier()
            if tid == fx.Int32(0):
                wait_all()
                comm_ops.fence_agent_release()
                _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                wait_all()
                comm_ops.fence_agent_release()
                comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
            wait_all()
            fx.barrier()
        else:
            if tid == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
                if const_expr(direct_fixed_slot):
                    comm_ops.fence_system_acquire()
                else:
                    comm_ops.fence_agent_acquire()
            fx.barrier()

        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)

        if compact_owner:  # noqa: SIM102 - keep the device and compile-time branches separate.
            if const_expr(
                is_int8
                and not debug_skip_total_recv
                and not fused_count_total_recv
            ):
                emit_exact_total_recv(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr,
                    fz_k=fz_k, fz_total_experts=fz_total_experts, fz_rank=fz_rank,
                    addr_disp=addr_disp, i32_cur_tok=i32_cur_tok,
                    addr_in_idx=addr_in_idx,
                )
            if const_expr(not direct_fixed_slot):
                emit_dispatch_plan(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr,
                    fz_rank=fz_rank, fz_tile_m=fz_tile_m, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, parity=payload_parity,
                    expected=payload_expected, external_grouping=external_grouping,
                    external_counting=external_counting,
                    dispatch_blocks=dispatch_blocks, payload_chunk_rows=payload_chunk_rows,
                    payload_tile_ready=payload_tile_ready,
                    payload_block_cap=payload_block_cap,
                    lds_grouping=lds_grouping,
                    lds_group_blocks=lds_group_blocks,
                    count_allgather_plan=count_allgather_plan,
                    fused_total_recv=(
                        is_int8
                        and not debug_skip_total_recv
                        and fused_count_total_recv
                    ),
                )

        if compact_producer:
            if const_expr(direct_fixed_slot):
                emit_direct_fixed_slot_payload(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales,
                    fz_route_payload=is_int8 and not mxfp4_transport, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                    parity=payload_parity, expected=payload_expected,
                )
            else:
                if const_expr(external_grouping):
                    emit_dispatch_group(
                        num_waves=NUM_WAVES, fz_k=fz_k, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        external_counting=external_counting, adaptive_grouping=payload_tile_ready,
                        lds_grouping=lds_grouping,
                        lds_group_blocks=lds_group_blocks,
                        group_scratch=fx.recast_iter(fx.Int32, a_buf.ptr),
                    )
                else:
                    if tid == fx.Int32(0):
                        mori_shmem.int32_wait_until_equals(
                            a_pair_order_ready + fx.Int64(payload_parity) * fx.Int64(4), payload_expected)
                        comm_ops.fence_agent_acquire()
                    fx.barrier()
                producers_per_destination = fx.Int32(dispatch_blocks // fz_npes)
                chunks_per_destination = fx.Int32(1)
                if const_expr(payload_chunk_rows > 0):
                    chunks_per_destination = fx.Int32(
                        (fz_mtpr + payload_chunk_rows - 1) // payload_chunk_rows
                    )
                payload_active = fx.Int32(0) == fx.Int32(0)
                if const_expr(payload_tile_ready and dispatch_blocks > 32):
                    producer_destination = producer_slot % fx.Int32(fz_npes)
                    producer_round = producer_slot // fx.Int32(fz_npes)
                    producers_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_blocks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    chunks_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_chunks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    payload_active = producer_round < producers_per_destination
                if payload_active:
                    emit_dispatch_payload(
                        num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                        fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                        fz_safe_end_i32=fz_safe_end_i32, fz_scale_n_i32=fz_scale_n_i32,
                        fz_enable_scales=fz_enable_scales,
                        fz_route_payload=is_int8 and not mxfp4_transport,
                        fz_copy_payload=not debug_skip_payload_copy,
                        fz_mxfp4_smooth_pload=fused_mxfp4_aload,
                        smoothquant_mode="none",
                        smooth_rsrc=transport_smooth_rsrc,
                        model_dim=model_dim,
                        addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                        addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        producers_per_destination=producers_per_destination,
                        payload_chunk_rows=payload_chunk_rows,
                        payload_tile_ready=payload_tile_ready,
                    )
        if const_expr(direct_fixed_slot):
            if compact_owner:
                emit_direct_fixed_slot_finalize(
                    fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_tile_m=fz_tile_m, n_tiles=N_TILES, addr_disp=addr_disp, parity=payload_parity,
                    expected=payload_expected,
                )
        else:
            payload_table = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.P2P_PAYLOAD_READY)), fx.Int64)
            addr_payload_ready = _buffer_load(
                _make_buffer_from_addr(payload_table, fx.Int64), fx.Int32(fz_rank), fx.Int64
            )
            addr_tile_ready = _disp_ptr(DispatchSlot.TILE_READY)
            addr_tile_expected = _disp_ptr(DispatchSlot.TILE_EXPECTED)
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        if const_expr(is_int8):
            sx_rsrc = _make_buffer(scale_x, fx.Float32)
            sw_rsrc = _make_buffer(scale_w, fx.Float32)
            if const_expr(int8_flat_output):
                out_rsrc = None
            else:
                out_rsrc = _make_buffer(
                    out,
                    fx.Float16,
                    max_size=False,
                    num_records_bytes=fx.Int32(int8_output_nbytes),
                )
            qscale_rsrc = _make_buffer(qscale_w, fx.Int32)
            qzero_rsrc = _make_buffer(qzero_w, fx.Int32)
            compact_src_rsrc = _make_buffer(
                compact_src,
                fx.Int32,
                max_size=False,
                num_records_bytes=tokens * fx.Int32(4),
            )
            compact_expert_rsrc = _make_buffer(
                compact_experts,
                fx.Int32,
                max_size=False,
                num_records_bytes=ceildiv(tokens, fx.Int32(sort_block_m))
                * fx.Int32(4),
            )
            compact_weight_rsrc = _make_buffer(
                compact_weights,
                fx.Float32,
                max_size=False,
                num_records_bytes=tokens * fx.Int32(4),
            )
            srcmap_rsrc = _make_buffer_from_addr(_disp_ptr(DispatchSlot.SRCMAP), fx.Int32)
            p2p_weight_rsrc = _make_buffer_from_addr(
                _disp_ptr(DispatchSlot.P2P_WEIGHT), fx.Int64
            )
            weight_rsrc = _make_buffer_from_addr(
                _buffer_load(p2p_weight_rsrc, fx.Int32(fz_rank), fx.Int64),
                fx.Float32,
            )
            if const_expr(mxfp4_transport):
                p2p_token_rsrc = _make_buffer_from_addr(
                    _disp_ptr(DispatchSlot.P2P_TOKEN), fx.Int64
                )
                p2p_scale_rsrc = _make_buffer_from_addr(
                    _disp_ptr(DispatchSlot.P2P_SCALE), fx.Int64
                )
                transport_payload_addr = _buffer_load(
                    p2p_token_rsrc, fx.Int32(fz_rank), fx.Int64
                )
                transport_scale_addr = _buffer_load(
                    p2p_scale_rsrc, fx.Int32(fz_rank), fx.Int64
                )
                transport_payload_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    transport_payload_addr.ir_value()
                )
                transport_scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    transport_scale_addr.ir_value()
                )
                if const_expr(fused_mxfp4_aload):
                    transport_row_scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
                        _disp_ptr(DispatchSlot.ROW_SCALE).ir_value()
                    )
                    sx_rsrc = _make_buffer_from_addr(
                        _disp_ptr(DispatchSlot.ROW_SCALE), fx.Float32
                    )
                else:
                    converted_x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
                    converted_scale_rsrc = buffer_ops.create_buffer_resource(scale_x, max_size=True)
                    quant_reduction_scratch = fx.recast_iter(fx.Float32, a_buf.ptr)
        else:
            sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
            sw_rsrc = _make_buffer(scale_w, fx.Int32)
            scale_cols = (inter_dim // 32 + 7) // 8 * 8
            os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
            if const_expr(use_tile_resource):
                out_rsrc = None
            else:
                out_nbytes = tokens * fx.Int32(inter_dim)
                out_rsrc = _make_buffer(out, fx.Int16, max_size=False, num_records_bytes=out_nbytes)
            os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

        if const_expr(is_int8):
            expert_of_flat, _do_scheduled_tile = build_fused_int8_gemm1(
                x_tensor=x, w_rsrc=w_rsrc, qscale_rsrc=qscale_rsrc,
                qzero_rsrc=qzero_rsrc, sx_rsrc=sx_rsrc, sw_rsrc=sw_rsrc,
                out_rsrc=out_rsrc, trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc,
                srcmap_rsrc=srcmap_rsrc, weight_rsrc=weight_rsrc,
                compact_src_rsrc=compact_src_rsrc,
                compact_expert_rsrc=compact_expert_rsrc,
                compact_weight_rsrc=compact_weight_rsrc, a_buf=a_buf, c_tile=c_tile,
                model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
                tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave,
                wave_id=wave_id, m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N,
                total_threads=TOTAL_THREADS, k_iters=K_ITERS, n_tiles=N_TILES,
                expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
                swizzle_a=swizzle_a, packed_int4=packed_int4,
                atom_tokens=fz_npes * fz_mtpr, topk=fz_k,
                async_a_copy=async_a_copy,
                pipe_weights=int8_pipe_weights,
                swiglu_limit=swiglu_limit,
                out_tensor=out if int8_flat_output else None,
                mxfp4_a_load=fused_mxfp4_aload,
                transport_payload_rsrc=(
                    transport_payload_rsrc if mxfp4_transport else None
                ),
                transport_scale_rsrc=(
                    transport_scale_rsrc if mxfp4_transport else None
                ),
                transport_smooth_rsrc=(
                    transport_smooth_rsrc if mxfp4_transport else None
                ),
                transport_row_scale_rsrc=(
                    transport_row_scale_rsrc if fused_mxfp4_aload else None
                ),
                direct_output=int8_direct_output,
            )
        else:
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
        fx.barrier()
        # All consumer waves read planner-owned tile/expert metadata.
        if const_expr(direct_fixed_slot):
            comm_ops.fence_system_acquire()
        else:
            comm_ops.fence_agent_acquire()

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = (
            fx.Int32(0)
            if const_expr(debug_no_consumer_work)
            else num_m_tiles * fx.Int32(N_TILES)
        )
        consumer_work_limit = (
            num_m_tiles if const_expr(m_tile_consumer) else total_work
        )

        def _wait_tile_payload(flat):
            if const_expr(payload_tile_ready):
                tile_index = flat // fx.Int32(N_TILES)
                expected_tiles = _buffer_load(
                    _make_buffer_from_addr(addr_tile_expected, fx.Int32), tile_index, fx.Int32
                )
                mori_shmem.int32_wait_until_equals(
                    addr_tile_ready + fx.Int64(tile_index) * fx.Int64(4), expected_tiles
                )
            else:
                pe = expert_of_flat(flat)
                pe_index = payload_parity * fx.Int32(fz_epr) + pe
                mori_shmem.int32_wait_until_equals(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), payload_expected
                )

        # Persistent mode lets every control CTA join the dynamic work pool.
        # Static mode leaves planner/producers retired and assigns each consumer
        # one deterministic work item, matching the Primus role split.
        consumer_active = (
            ticket > fx.Int32(role_prefix_blocks)
            if const_expr(role_specialized_rxq)
            else ticket > fx.Int32(dispatch_blocks)
            if const_expr(static_consumer_work)
            else fx.Int32(1) == fx.Int32(1)
        )
        static_work_item = ticket - fx.Int32(
            planner_blocks + role_prefix_blocks
        )
        rxq_static_is_converter = fx.Int32(0) != fx.Int32(0)
        if const_expr(static_rxq_pipeline):
            rxq_static_role = static_work_item
            if const_expr(static_rxq_interleaved):
                gemm_per_converter = (
                    static_rxq_gemm_blocks // static_rxq_converter_blocks
                )
                role_group_size = gemm_per_converter + 1
                role_group = rxq_static_role // fx.Int32(role_group_size)
                role_in_group = rxq_static_role - role_group * fx.Int32(
                    role_group_size
                )
                rxq_static_is_converter = role_in_group == fx.Int32(0)
                static_work_item = rxq_static_is_converter.select(
                    role_group,
                    role_group * fx.Int32(gemm_per_converter)
                    + role_in_group
                    - fx.Int32(1),
                )
            else:
                rxq_static_is_converter = rxq_static_role < fx.Int32(
                    static_rxq_converter_blocks
                )
                static_work_item = rxq_static_is_converter.select(
                    rxq_static_role,
                    rxq_static_role - fx.Int32(static_rxq_converter_blocks),
                )
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        work_shard = ticket & fx.Int32(WORK_SHARDS - 1)
        if const_expr(role_specialized_rxq):
            # Dispatch CTAs become receiver-side FP4 decode/SmoothQuant workers
            # after their PUSH task.  Producer-slot striding is deterministic,
            # needs no atomic/LDS control queue, and gives each partition to
            # exactly one CTA.  Producers never execute MFMA.
            conversion_work = role_rxq_converter_slot
            conversion_active = role_rxq_converter & (
                conversion_work < num_m_tiles
            )
            while conversion_active:
                if tid == fx.Int32(0):
                    if const_expr(not debug_skip_conversion):
                        _wait_tile_payload(
                            conversion_work * fx.Int32(N_TILES)
                        )
                if const_expr(not debug_skip_conversion):
                    fx.barrier()
                if const_expr(not debug_skip_conversion):
                    comm_ops.fence_system_acquire()
                    m_tile = conversion_work
                    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
                    local_expert = expert_of_flat(
                        m_tile * fx.Int32(N_TILES)
                    )
                    if const_expr(wavewise_conversion):
                        for wave_batch in range_constexpr(
                            (sort_block_m + NUM_WAVES - 1) // NUM_WAVES
                        ):
                            row_offset = wave_id + fx.Int32(
                                wave_batch * NUM_WAVES
                            )
                            if row_offset < fx.Int32(sort_block_m):
                                convert_mxfp4_smoothquant_row_wavewise(
                                    transport_payload_rsrc,
                                    transport_scale_rsrc,
                                    transport_smooth_rsrc,
                                    converted_x_rsrc,
                                    converted_scale_rsrc,
                                    tile_row_base + row_offset,
                                    local_expert
                                    + fx.Int32(fz_rank * fz_epr),
                                    cols=model_dim,
                                )
                    else:
                        for row_offset in range(
                            fx.Int32(0), fx.Int32(sort_block_m), 2
                        ):
                            convert_mxfp4_smoothquant_row_pair(
                                transport_payload_rsrc,
                                transport_scale_rsrc,
                                transport_smooth_rsrc,
                                converted_x_rsrc,
                                converted_scale_rsrc,
                                quant_reduction_scratch,
                                tile_row_base + fx.Int32(row_offset),
                                tile_row_base + fx.Int32(row_offset + 1),
                                fx.Int32(0) == fx.Int32(0),
                                fx.Int32(0) == fx.Int32(0),
                                local_expert + fx.Int32(fz_rank * fz_epr),
                                cols=model_dim,
                            )
                    wait_all()
                    comm_ops.fence_agent_release()
                    fx.barrier()
                    if tid == fx.Int32(0):
                        if const_expr(ready_tile_queue):
                            # Publish completed M tiles in completion order.  A
                            # converter performs one atomic reservation per M
                            # tile; GEMM CTAs remain statically assigned and do
                            # not contend on a dynamic work head.  This removes
                            # the mismatch between producer-strided conversion
                            # order and the old consecutive-M consumer order.
                            ready_slot = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    a_work_tail, fx.Int32(1)
                                )
                            )
                            comm_ops.atomic_xchg_global_at(
                                addr_quant_count
                                + fx.Int64(ready_slot) * fx.Int64(4),
                                m_tile + fx.Int32(1),
                            )
                        else:
                            comm_ops.atomic_add_agent(
                                addr_quant_count
                                + fx.Int64(m_tile) * fx.Int64(4),
                                fx.Int32(1),
                            )
                conversion_work = conversion_work + fx.Int32(dispatch_blocks)
                conversion_active = conversion_work < num_m_tiles
        while consumer_active:
            if tid == fx.Int32(0):
                if const_expr(static_consumer_work or role_specialized_rxq):
                    work = static_work_item
                else:
                    local_work = fx.Int32(
                        comm_ops.atomic_add_agent(
                            a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)
                        )
                    )
                    work = work_shard + local_work * fx.Int32(WORK_SHARDS)
                fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            fx.barrier()
            work = Vec(work_scratch_view.load())[0]
            if tid == fx.Int32(0):
                if const_expr(static_rxq_pipeline):
                    converter_has_work = work < num_m_tiles
                    gemm_has_work = work < total_work
                    has_work = rxq_static_is_converter.select(
                        converter_has_work, gemm_has_work
                    ).select(fx.Int32(1), fx.Int32(0))
                    payload_work = rxq_static_is_converter.select(
                        work * fx.Int32(N_TILES), work
                    )
                else:
                    has_work = (work < consumer_work_limit).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    payload_work = (
                        work * fx.Int32(N_TILES)
                        if const_expr(m_tile_consumer)
                        else work
                    )
                if has_work != fx.Int32(0):  # noqa: SIM102 - keep the device and compile-time branches separate.
                    if const_expr(
                        not direct_fixed_slot and not role_specialized_rxq
                    ):
                        # In the static RXQ pipeline only converter CTAs touch
                        # peer-written FP4 payload.  GEMM CTAs wait on the
                        # converter's agent-scope tile flag instead of also
                        # traversing the expert payload-ready protocol.
                        if const_expr(static_rxq_pipeline):
                            if rxq_static_is_converter:
                                _wait_tile_payload(payload_work)
                        else:
                            _wait_tile_payload(payload_work)
                fx.ptr_store(Vec.from_elements([has_work], fx.Int32), work_scratch)
            fx.barrier()
            has_work = Vec(work_scratch_view.load())[0]
            if has_work != fx.Int32(0):
                if const_expr(not direct_fixed_slot):
                    if const_expr(static_rxq_pipeline):
                        if rxq_static_is_converter:
                            comm_ops.fence_system_acquire()
                    else:
                        comm_ops.fence_system_acquire()
                if const_expr(m_tile_consumer):
                    m_tile = work
                    if const_expr(not debug_skip_conversion):
                        tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
                        local_expert = expert_of_flat(
                            m_tile * fx.Int32(N_TILES)
                        )
                        for row_offset in range(
                            fx.Int32(0), fx.Int32(sort_block_m), 2
                        ):
                            convert_mxfp4_smoothquant_row_pair(
                                transport_payload_rsrc,
                                transport_scale_rsrc,
                                transport_smooth_rsrc,
                                converted_x_rsrc,
                                converted_scale_rsrc,
                                quant_reduction_scratch,
                                tile_row_base + fx.Int32(row_offset),
                                tile_row_base + fx.Int32(row_offset + 1),
                                fx.Int32(0) == fx.Int32(0),
                                fx.Int32(0) == fx.Int32(0),
                                local_expert + fx.Int32(fz_rank * fz_epr),
                                cols=model_dim,
                            )
                        wait_all()
                        comm_ops.fence_agent_release()
                        fx.barrier()
                    if const_expr(not debug_skip_gemm):
                        flat_base = m_tile * fx.Int32(N_TILES)
                        for n_tile in range_constexpr(N_TILES):
                            _do_scheduled_tile(flat_base + fx.Int32(n_tile))
                elif const_expr(role_specialized_rxq):
                    static_m_tile = work // fx.Int32(N_TILES)
                    n_tile = work - static_m_tile * fx.Int32(N_TILES)
                    m_tile = static_m_tile
                    if const_expr(not debug_skip_conversion):
                        if tid == fx.Int32(0):
                            if const_expr(ready_tile_queue):
                                ready_value = (
                                    mori_shmem.int32_wait_until_greater_than(
                                        addr_quant_count
                                        + fx.Int64(static_m_tile) * fx.Int64(4),
                                        fx.Int32(0),
                                    )
                                )
                                work = (
                                    ready_value - fx.Int32(1)
                                ) * fx.Int32(N_TILES) + n_tile
                                fx.ptr_store(
                                    Vec.from_elements([work], fx.Int32),
                                    work_scratch,
                                )
                            else:
                                mori_shmem.int32_wait_until_equals(
                                    addr_quant_count
                                    + fx.Int64(m_tile) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                            comm_ops.fence_agent_acquire()
                        fx.barrier()
                        if const_expr(ready_tile_queue):
                            work = Vec(work_scratch_view.load())[0]
                    if const_expr(not debug_skip_gemm):
                        _do_scheduled_tile(work)
                elif const_expr(static_rxq_pipeline):
                    # Type the dynamic-role branch locals explicitly.  FlyDSL
                    # otherwise treats their pre-branch value as None and
                    # cannot build the scf.if result list.
                    m_tile = fx.Int32(0)
                    flat_base = fx.Int32(0)
                    tile_row_base = fx.Int32(0)
                    local_expert = fx.Int32(0)
                    if rxq_static_is_converter:
                        m_tile = work
                        flat_base = m_tile * fx.Int32(N_TILES)
                        if const_expr(not debug_skip_conversion):
                            tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
                            local_expert = expert_of_flat(flat_base)
                            # One fixed converter CTA owns the whole M tile.
                            # Each wave converts one row without CTA-wide
                            # reductions, so 8 waves retire 8 rows per pass.
                            # A single publication replaces the old five-way
                            # N-tile quant_count rendezvous.
                            for wave_batch in range_constexpr(
                                (sort_block_m + NUM_WAVES - 1) // NUM_WAVES
                            ):
                                row_offset = wave_id + fx.Int32(
                                    wave_batch * NUM_WAVES
                                )
                                if row_offset < fx.Int32(sort_block_m):
                                    convert_mxfp4_smoothquant_row_wavewise(
                                        transport_payload_rsrc,
                                        transport_scale_rsrc,
                                        transport_smooth_rsrc,
                                        converted_x_rsrc,
                                        converted_scale_rsrc,
                                        tile_row_base + row_offset,
                                        local_expert
                                        + fx.Int32(fz_rank * fz_epr),
                                        cols=model_dim,
                                    )
                            wait_all()
                            comm_ops.fence_agent_release()
                            fx.barrier()
                            if tid == fx.Int32(0):
                                comm_ops.atomic_xchg_global_at(
                                    addr_quant_count
                                    + fx.Int64(m_tile) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                    else:
                        m_tile = work // fx.Int32(N_TILES)
                        if const_expr(not debug_skip_conversion):
                            if tid == fx.Int32(0):
                                mori_shmem.int32_wait_until_equals(
                                    addr_quant_count + fx.Int64(m_tile) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                                comm_ops.fence_agent_acquire()
                            fx.barrier()
                        if const_expr(not debug_skip_gemm):
                            _do_scheduled_tile(work)
                elif const_expr(
                    mxfp4_transport
                    and not fused_mxfp4_aload
                    and not debug_skip_conversion
                ):
                    m_tile = work // fx.Int32(N_TILES)
                    n_tile = work - m_tile * fx.Int32(N_TILES)
                    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
                    local_expert = expert_of_flat(work)
                    if const_expr(wavewise_conversion):
                        rows_per_partition = (
                            sort_block_m + N_TILES - 1
                        ) // N_TILES
                        wave_batches = (
                            rows_per_partition + NUM_WAVES - 1
                        ) // NUM_WAVES
                        for wave_batch in range_constexpr(wave_batches):
                            partition_row = wave_id + fx.Int32(
                                wave_batch * NUM_WAVES
                            )
                            row_offset = (
                                n_tile + partition_row * fx.Int32(N_TILES)
                            )
                            if row_offset < fx.Int32(sort_block_m):
                                convert_mxfp4_smoothquant_row_wavewise(
                                    transport_payload_rsrc,
                                    transport_scale_rsrc,
                                    transport_smooth_rsrc,
                                    converted_x_rsrc,
                                    converted_scale_rsrc,
                                    tile_row_base + row_offset,
                                    local_expert + fx.Int32(fz_rank * fz_epr),
                                    cols=model_dim,
                                )
                    elif const_expr(dual_row_conversion):
                        row0 = n_tile
                        while row0 < fx.Int32(sort_block_m):
                            row1 = row0 + fx.Int32(N_TILES)
                            row1_valid = row1 < fx.Int32(sort_block_m)
                            safe_row1 = row1_valid.select(row1, row0)
                            convert_mxfp4_smoothquant_row_pair(
                                transport_payload_rsrc,
                                transport_scale_rsrc,
                                transport_smooth_rsrc,
                                converted_x_rsrc,
                                converted_scale_rsrc,
                                quant_reduction_scratch,
                                tile_row_base + row0,
                                tile_row_base + safe_row1,
                                fx.Int32(0) == fx.Int32(0),
                                row1_valid,
                                local_expert + fx.Int32(fz_rank * fz_epr),
                                cols=model_dim,
                            )
                            row0 = row0 + fx.Int32(2 * N_TILES)
                    else:
                        for row_offset in range_constexpr(sort_block_m):
                            if n_tile == fx.Int32(row_offset % N_TILES):
                                convert_mxfp4_smoothquant_row(
                                    transport_payload_rsrc,
                                    transport_scale_rsrc,
                                    transport_smooth_rsrc,
                                    converted_x_rsrc,
                                    converted_scale_rsrc,
                                    quant_reduction_scratch,
                                    tile_row_base + fx.Int32(row_offset),
                                    local_expert + fx.Int32(fz_rank * fz_epr),
                                    cols=model_dim,
                                    total_threads=TOTAL_THREADS,
                                )
                    wait_all()
                    comm_ops.fence_agent_release()
                    fx.barrier()
                    if tid == fx.Int32(0):
                        previous_count = fx.Int32(comm_ops.atomic_add_agent(
                            addr_quant_count + fx.Int64(m_tile) * fx.Int64(4),
                            fx.Int32(1),
                        ))
                        if const_expr(last_converter_gemm):
                            final_converter = (
                                previous_count == fx.Int32(N_TILES - 1)
                            ).select(fx.Int32(1), fx.Int32(0))
                            fx.ptr_store(
                                Vec.from_elements([final_converter], fx.Int32),
                                work_scratch,
                            )
                    fx.barrier()
                    if const_expr(last_converter_gemm):
                        final_converter = Vec(work_scratch_view.load())[0]
                        flat_base = m_tile * fx.Int32(N_TILES)
                        if final_converter != fx.Int32(0):
                            comm_ops.fence_agent_acquire()
                            if const_expr(not debug_skip_gemm):
                                for final_n in range_constexpr(N_TILES):
                                    _do_scheduled_tile(
                                        flat_base + fx.Int32(final_n)
                                    )
                    else:
                        if tid == fx.Int32(0):
                            mori_shmem.int32_wait_until_equals(
                                addr_quant_count + fx.Int64(m_tile) * fx.Int64(4),
                                fx.Int32(N_TILES),
                            )
                            comm_ops.fence_agent_acquire()
                        fx.barrier()
                if const_expr(
                    not static_rxq_pipeline
                    and not role_specialized_rxq
                    and not m_tile_consumer
                    and not debug_skip_gemm
                    and not last_converter_gemm
                ):
                    _do_scheduled_tile(work)
            if const_expr(static_rxq_pipeline):
                static_stride = rxq_static_is_converter.select(
                    fx.Int32(static_rxq_converter_blocks),
                    fx.Int32(static_rxq_gemm_blocks),
                )
                static_work_item = static_work_item + static_stride
                consumer_active = rxq_static_is_converter.select(
                    static_work_item < num_m_tiles,
                    static_work_item < total_work,
                )
            elif const_expr(static_consumer_work and fused_mxfp4_aload):
                static_work_item = static_work_item + fx.Int32(grid_x)
                consumer_active = static_work_item < total_work
            else:
                consumer_active = (
                    fx.Int32(0) != fx.Int32(0)
                    if const_expr(static_consumer_work or role_specialized_rxq)
                    else has_work != fx.Int32(0)
                )

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        compact_src: fx.Tensor, compact_experts: fx.Tensor, compact_weights: fx.Tensor,
        qscale_w: fx.Tensor, qzero_w: fx.Tensor, transport_smooth: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, addr_quant_count: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
            compact_src, compact_experts, compact_weights, qscale_w, qzero_w, transport_smooth, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            addr_quant_count,
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
    b_nt=-1, work_shards=None, external_grouping=None, external_counting=None,
    payload_chunk_rows=0, payload_tile_ready=False, swiglu_limit=0.0,
    quant_mode="a8w4", compact_src=None, compact_experts=None, compact_weights=None,
    qscale_w=None, qzero_w=None, mxfp4_transport=False, transport_smooth=None,
    addr_quant_count=0):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, work_shards=work_shards, external_grouping=external_grouping,
        external_counting=external_counting, payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
        swiglu_limit=swiglu_limit,
        quant_mode=quant_mode,
        mxfp4_transport=mxfp4_transport,
    )
    # These tensors are compile-time dead on the established a8w4 branch.
    compact_src = sorted_token_ids if compact_src is None else compact_src
    compact_experts = expert_ids if compact_experts is None else compact_experts
    compact_weights = scale_w if compact_weights is None else compact_weights
    qscale_w = scale_w if qscale_w is None else qscale_w
    qzero_w = scale_w if qzero_w is None else qzero_w
    transport_smooth = scale_w if transport_smooth is None else transport_smooth
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        compact_src, compact_experts, compact_weights, qscale_w, qzero_w, transport_smooth, tokens,
        addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, fx.Int64(addr_quant_count), stream,
    )
# fmt: on
