# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.common.tensor_shim import _run_compiled

from .gemm_util_role import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    Int8AS2RLoader,
    Int8BWeightLoader,
    MfmaInt8GU,
    MfmaScaleGU,
    SiluF16AtomEpilogue,
    SiluQuantEpilogue,
    TileScheduler,
    _buffer_load,
    _make_buffer,
    wait_lds_barrier,
)
from .mxfp4_smoothquant import MxFp4SmoothInt8ATileLoader


class _LdsF32View:
    def __init__(self, ptr):
        self.ptr = ptr


@flyc.jit
# fmt: off
def do_int8_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, mfma, epi,
    a_buf, A_LDS_I32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES,
    async_a_copy, pipe_weights, mxfp4_a_load, trb_rsrc):
# fmt: on
    """Run one gfx950 INT8 tile with the MX-style next-A pipeline."""
    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
    wave = fx.thread_idx.x // fx.Int32(64)
    wave_n_base = wave * fx.Int32(epi._tile_n // epi._num_waves)
    gate_row = sched.gate_base_row(expert) + n_tile_base + wave_n_base
    up_row = gate_row + fx.Int32(epi._inter_dim)
    if const_expr(mxfp4_a_load):
        a_gather.for_tile(tile_row_base, expert)
    else:
        a_gather.for_tile(tile_row_base)
    gate_acc = [mfma.zero_value for _ in range(M_REPEAT * NUM_ACC_N)]
    up_acc = [mfma.zero_value for _ in range(M_REPEAT * NUM_ACC_N)]

    if const_expr(async_a_copy):
        a_gather.prefetch_to_lds(fx.Int32(0), a_buf, fx.Int32(0))
    else:
        a_gather.store(
            a_buf,
            a_gather.load_regs(fx.Int32(0)),
            fx.Int32(0),
        )
    wait_lds_barrier(0 if async_a_copy else 63)
    acc_count = M_REPEAT * NUM_ACC_N
    b_count = 4 * NUM_ACC_N
    last = fx.Int32(K_ITERS - 1)
    if const_expr(pipe_weights):
        gate_b0, up_b0 = b_loader.load_step(
            gate_row, up_row, fx.Int32(0)
        )
        init = list(gate_acc) + list(up_acc)
        init += [gate_b0[ksub][ni] for ksub in range(4) for ni in range(NUM_ACC_N)]
        init += [up_b0[ksub][ni] for ksub in range(4) for ni in range(NUM_ACC_N)]
        for step_i, state in range(0, K_ITERS - 1, 1, init=init):
            step = fx.Int32(step_i)
            gate_acc = [Vec(value) for value in state[:acc_count]]
            up_acc = [Vec(value) for value in state[acc_count : 2 * acc_count]]
            gate_state = state[2 * acc_count : 2 * acc_count + b_count]
            up_state = state[2 * acc_count + b_count :]
            gate_prev = [
                [Vec(gate_state[ksub * NUM_ACC_N + ni]) for ni in range(NUM_ACC_N)]
                for ksub in range(4)
            ]
            up_prev = [
                [Vec(up_state[ksub * NUM_ACC_N + ni]) for ni in range(NUM_ACC_N)]
                for ksub in range(4)
            ]
            a_off = (step & fx.Int32(1)) * fx.Int32(A_LDS_I32)
            next_off = ((step + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(A_LDS_I32)
            step_next = step + fx.Int32(1)

            def a_load(mi, ksub, _base=a_off):
                return a_s2r.load_operand(a_buf, mi, ksub, _base)

            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    step_next * fx.Int32(A_K_STEP_BYTES), a_buf, next_off
                )
                rocdl.sched_barrier(0)
            else:
                next_a_regs = a_gather.load_regs(
                    step_next * fx.Int32(A_K_STEP_BYTES)
                )

            def load_next(ni, _step=step_next):
                return b_loader.load_projection_pair_ni(
                    gate_row, up_row, ni, _step
                )

            gate_acc, up_acc, gate_next, up_next = mfma.call_pipe(
                a_load,
                gate_prev,
                up_prev,
                gate_acc,
                up_acc,
                load_next,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(0)
            else:
                a_gather.store(a_buf, next_a_regs, next_off)
                wait_lds_barrier()
            state = yield (
                list(gate_acc)
                + list(up_acc)
                + [gate_next[ksub][ni] for ksub in range(4) for ni in range(NUM_ACC_N)]
                + [up_next[ksub][ni] for ksub in range(4) for ni in range(NUM_ACC_N)]
            )

        gate_acc = [Vec(value) for value in state[:acc_count]]
        up_acc = [Vec(value) for value in state[acc_count : 2 * acc_count]]
        gate_state = state[2 * acc_count : 2 * acc_count + b_count]
        up_state = state[2 * acc_count + b_count :]
        gate_prev = [
            [Vec(gate_state[ksub * NUM_ACC_N + ni]) for ni in range(NUM_ACC_N)]
            for ksub in range(4)
        ]
        up_prev = [
            [Vec(up_state[ksub * NUM_ACC_N + ni]) for ni in range(NUM_ACC_N)]
            for ksub in range(4)
        ]
        final_off = (last & fx.Int32(1)) * fx.Int32(A_LDS_I32)

        def final_a_load(mi, ksub, _base=final_off):
            return a_s2r.load_operand(a_buf, mi, ksub, _base)

        gate_acc, up_acc = mfma.call(
            final_a_load, gate_prev, up_prev, gate_acc, up_acc
        )
    else:
        for step_i, state in range(
            0, K_ITERS, 1, init=list(gate_acc) + list(up_acc)
        ):
            step = fx.Int32(step_i)
            gate_acc = [Vec(value) for value in state[:acc_count]]
            up_acc = [Vec(value) for value in state[acc_count:]]
            a_off = (step & fx.Int32(1)) * fx.Int32(A_LDS_I32)
            next_off = ((step + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(A_LDS_I32)
            step_next = (step + fx.Int32(1) < last).select(
                step + fx.Int32(1), last
            )
            gate_b, up_b = b_loader.load_step(gate_row, up_row, step)

            def a_load(mi, ksub, _base=a_off):
                return a_s2r.load_operand(a_buf, mi, ksub, _base)

            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    step_next * fx.Int32(A_K_STEP_BYTES), a_buf, next_off
                )
                rocdl.sched_barrier(0)
            else:
                next_a_regs = a_gather.load_regs(
                    step_next * fx.Int32(A_K_STEP_BYTES)
                )

            gate_acc, up_acc = mfma.call(
                a_load, gate_b, up_b, gate_acc, up_acc
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(0)
            else:
                a_gather.store(a_buf, next_a_regs, next_off)
                wait_lds_barrier()
            state = yield list(gate_acc) + list(up_acc)

        gate_acc = [Vec(value) for value in state[:acc_count]]
        up_acc = [Vec(value) for value in state[acc_count:]]
    wait_lds_barrier()
    epi.store(
        gate_acc,
        up_acc,
        m_tile,
        tile_row_base,
        n_tile_base,
        expert,
    )


# fmt: off
def build_fused_int8_gemm1(*, x_tensor, w_rsrc, qscale_rsrc, qzero_rsrc, sx_rsrc, sw_rsrc,
    out_rsrc, trb_rsrc, expert_rsrc, srcmap_rsrc, weight_rsrc, compact_src_rsrc,
    compact_expert_rsrc, compact_weight_rsrc, a_buf, c_tile, model_dim, inter_dim, sort_block_m, tile_n,
    num_waves, n_per_wave, wave_id, m_repeat, num_acc_n, total_threads, k_iters,
    n_tiles, expert_offset, b_cache_modifier, swizzle_a, packed_int4, atom_tokens,
    topk, async_a_copy, pipe_weights=False, out_tensor=None, swiglu_limit=0.0, mxfp4_a_load=False,
    transport_payload_rsrc=None, transport_scale_rsrc=None,
    transport_smooth_rsrc=None, transport_row_scale_rsrc=None,
    direct_output=False):
# fmt: on
    """Build the independent INT8 Stage1 compute branch."""
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,
    )
    if mxfp4_a_load:
        if async_a_copy:
            raise ValueError("MXFP4 SmoothQuant A-load does not support async copy")
        a_gather = MxFp4SmoothInt8ATileLoader(
            payload_rsrc=transport_payload_rsrc,
            mx_scale_rsrc=transport_scale_rsrc,
            smooth_rsrc=transport_smooth_rsrc,
            row_scale_rsrc=transport_row_scale_rsrc,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            k_step_bytes=256,
            total_threads=total_threads,
            expert_offset=expert_offset,
            swizzle=swizzle_a,
        )
    else:
        a_gather = ATileLoader(
            row_bytes=model_dim,
            sort_block_m=sort_block_m,
            k_step_bytes=256,
            total_threads=total_threads,
            swizzle=swizzle_a,
            x_tensor=x_tensor,
            async_copy=async_a_copy,
            async_elem_ty=fx.Int8,
        )
    a_s2r = Int8AS2RLoader(k_step_bytes=256, swizzle=swizzle_a)
    b_loader = Int8BWeightLoader(
        w_rsrc=w_rsrc,
        qscale_rsrc=qscale_rsrc,
        qzero_rsrc=qzero_rsrc,
        num_acc_n=num_acc_n,
        model_dim=model_dim,
        packed_int4=packed_int4,
        cache_modifier=b_cache_modifier,
    )
    mfma = MfmaInt8GU(m_repeat=m_repeat, num_acc_n=num_acc_n)
    epi = SiluF16AtomEpilogue(
        out_rsrc=out_rsrc,
        sx_rsrc=sx_rsrc,
        sw_rsrc=sw_rsrc,
        srcmap_rsrc=srcmap_rsrc,
        weight_rsrc=weight_rsrc,
        compact_src_rsrc=compact_src_rsrc,
        compact_expert_rsrc=compact_expert_rsrc,
        compact_weight_rsrc=compact_weight_rsrc,
        atom_tokens=atom_tokens,
        topk=topk,
        inter_dim=inter_dim,
        m_repeat=m_repeat,
        num_acc_n=num_acc_n,
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        lds_out=c_tile,
        swiglu_limit=swiglu_limit,
        out_tensor=out_tensor,
        direct_output=direct_output,
    )

    def _decode(flat):
        m_tile = flat // fx.Int32(n_tiles)
        n_tile = flat - m_tile * fx.Int32(n_tiles)
        return m_tile, n_tile

    def expert_of_flat(flat):
        m_tile, _ = _decode(flat)
        return sched.expert_of(m_tile)

    def run_tile(flat):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        do_int8_tile(
            m_tile,
            n_tile_base,
            expert,
            sched,
            a_gather,
            a_s2r,
            b_loader,
            mfma,
            epi,
            a_buf,
            sort_block_m * 256 // 4,
            k_iters,
            m_repeat,
            num_acc_n,
            256,
            async_a_copy,
            pipe_weights,
            mxfp4_a_load,
            trb_rsrc,
        )

    return expert_of_flat, run_tile


# fmt: off
@functools.lru_cache(maxsize=None)
def compile_int8_gemm1(
    *, model_dim: int, inter_dim: int, expert_offset: int, atom_tokens: int, topk: int,
    packed_int4: bool, sort_block_m: int = 32, tile_n: int = 256, tile_k: int = 256,
    num_waves: int = 4, swizzle_a: bool = True, async_a_copy: bool = False,
    waves_per_eu_hint: int = 2, b_cache_modifier: int = 0, swiglu_limit: float = 0.0,
):
# fmt: on
    """Compile the INT8 GEMM1 body without dispatch/planner orchestration.

    This is intentionally the same compute and epilogue used by fused Stage1.
    It provides a hard GEMM-only baseline for overlap analysis and is also the
    consumer kernel used by split producer/consumer experiments.
    """
    num_waves = int(num_waves)
    if num_waves <= 1 or tile_n % num_waves:
        raise ValueError("tile_n must be divisible by num_waves > 1")
    if tile_k != 256 or model_dim % tile_k:
        raise ValueError("INT8 GEMM1 requires tile_k=256 dividing model_dim")
    # One INT8 work item computes gate and up together for the same logical
    # output columns.  The scheduler therefore tiles INTER, not 2*INTER (the
    # latter is only the physical W1 row count).  Using 2*INTER here launched a
    # second set of N tiles past both the epilogue and W1 bounds.
    if inter_dim % tile_n:
        raise ValueError("tile_n must divide inter_dim")
    if not 1 <= waves_per_eu_hint <= 4:
        raise ValueError("waves_per_eu_hint must be in [1, 4]")

    n_per_wave = tile_n // num_waves
    n_tiles = inter_dim // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    if num_acc_n % 2 or m_repeat % 2:
        raise ValueError("INT8 GEMM1 requires even M/N accumulator repeats")
    total_threads = num_waves * 64
    k_iters = model_dim // tile_k
    a_lds_size = sort_block_m * tile_k
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * tile_n * 4)

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor,
        scale_w: fx.Tensor, tile_row_base: fx.Tensor, expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor, srcmap: fx.Tensor, route_weight: fx.Tensor,
        compact_src: fx.Tensor, compact_experts: fx.Tensor,
        compact_weights: fx.Tensor, qscale_w: fx.Tensor, qzero_w: fx.Tensor,
        grid_x: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Float32)
        sw_rsrc = _make_buffer(scale_w, fx.Float32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        srcmap_rsrc = _make_buffer(srcmap, fx.Int32)
        weight_rsrc = _make_buffer(route_weight, fx.Float32)
        compact_src_rsrc = _make_buffer(compact_src, fx.Int32)
        compact_expert_rsrc = _make_buffer(compact_experts, fx.Int32)
        compact_weight_rsrc = _make_buffer(compact_weights, fx.Float32)
        qscale_rsrc = _make_buffer(qscale_w, fx.Int32)
        qzero_rsrc = _make_buffer(qzero_w, fx.Int32)

        _, run_tile = build_fused_int8_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, qscale_rsrc=qscale_rsrc,
            qzero_rsrc=qzero_rsrc, sx_rsrc=sx_rsrc, sw_rsrc=sw_rsrc,
            out_rsrc=None, trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc,
            srcmap_rsrc=srcmap_rsrc, weight_rsrc=weight_rsrc,
            compact_src_rsrc=compact_src_rsrc,
            compact_expert_rsrc=compact_expert_rsrc,
            compact_weight_rsrc=compact_weight_rsrc, a_buf=a_buf, c_tile=c_tile,
            model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves,
            n_per_wave=n_per_wave, wave_id=fx.thread_idx.x // fx.Int32(64),
            m_repeat=m_repeat, num_acc_n=num_acc_n,
            total_threads=total_threads, k_iters=k_iters, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a, packed_int4=packed_int4,
            atom_tokens=atom_tokens, topk=topk, async_a_copy=async_a_copy,
            pipe_weights=False, out_tensor=out, swiglu_limit=swiglu_limit,
        )
        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        total_work = (
            (num_valid + fx.Int32(sort_block_m - 1))
            // fx.Int32(sort_block_m)
        ) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            run_tile(flat)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor,
        scale_w: fx.Tensor, tile_row_base: fx.Tensor, expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor, srcmap: fx.Tensor, route_weight: fx.Tensor,
        compact_src: fx.Tensor, compact_experts: fx.Tensor,
        compact_weights: fx.Tensor, qscale_w: fx.Tensor, qzero_w: fx.Tensor,
        grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids,
            num_valid_ids, srcmap, route_weight, compact_src, compact_experts,
            compact_weights, qscale_w, qzero_w, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(
            grid=(fx.Int64(grid_x), 1, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch


# fmt: off
def int8_gemm1_kernel(
    out, x, w, scale_x, scale_w, tile_row_base, expert_ids, num_valid_ids,
    srcmap, route_weight, compact_src, compact_experts, compact_weights,
    qscale_w, qzero_w, stream, *, model_dim: int, inter_dim: int,
    expert_offset: int, atom_tokens: int, topk: int, packed_int4: bool,
    sort_block_m: int = 32, tile_n: int = 256, tile_k: int = 256,
    num_waves: int = 4, swizzle_a: bool = True, async_a_copy: bool = False,
    waves_per_eu_hint: int = 2, b_cache_modifier: int = 0,
    swiglu_limit: float = 0.0, num_cu: int = 256,
):
# fmt: on
    """Launch standalone INT8 GEMM1 over planner-produced metadata."""
    launch = compile_int8_gemm1(
        model_dim=model_dim, inter_dim=inter_dim,
        expert_offset=expert_offset, atom_tokens=atom_tokens, topk=topk,
        packed_int4=packed_int4, sort_block_m=sort_block_m, tile_n=tile_n,
        tile_k=tile_k, num_waves=num_waves, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, waves_per_eu_hint=waves_per_eu_hint,
        b_cache_modifier=b_cache_modifier, swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, tile_row_base, expert_ids,
        num_valid_ids, srcmap, route_weight, compact_src, compact_experts,
        compact_weights, qscale_w, qzero_w, fx.Int32(num_cu), stream,
    )
    return out


@flyc.jit
# fmt: off
def do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    B_STATE_END = (
        N_ACC + NUM_ACC_N * _PACK
    )
    SB_STATE_END = B_STATE_END + NUM_B_SCALE
    last = fx.Int32(K_ITERS - 1)
    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
    b_row = sched.gate_base_row(expert) + n_tile_base
    a_gather.for_tile(tile_row_base)
    if const_expr(pipe_weights):
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_row_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        b0 = b_loader.load_step(b_row, fx.Int32(0))
        init = [mfma.zero_value for _ in range(N_ACC)]
        init += [h for ni_list in b0 for h in ni_list]
        if const_expr(async_a_copy):
            init += b_scale.load_step(
                b_row,
                fx.Int32(0),
            )
            init += a_scale.load_step(
                a_scale_lds,
                fx.Int32(0),
            )
        for sp_i, state in range(0, K_ITERS - 1, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state[:N_ACC]]
            b_prev = [
                [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)
            ]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = sp + fx.Int32(1)
            if const_expr(async_a_copy):
                sb = [
                    fx.Int32(
                        state[B_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_B_SCALE
                    )
                ]
                sa = [
                    fx.Int32(
                        state[SB_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_A_SCALE
                    )
                ]
            else:
                sb = b_scale.load_step(b_row, sp)
                sa = a_scale.load_step(
                    a_scale_lds,
                    sp,
                )

            def a_load(mi, ks, _base=cur_off):
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
                sb_next = b_scale.load_step(
                    b_row,
                    spn,
                )
                sa_next = a_scale.load_step(
                    a_scale_lds,
                    spn,
                )
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )

            def load_next(ni, _kn=spn):
                return b_loader.load_ni(b_row, ni, _kn)

            call_pipe = (
                mfma.call_pipe_am
                if mfma_amajor
                else mfma.call_pipe
            )
            acc, b_next = call_pipe(
                a_load,
                b_prev,
                acc,
                sa,
                sb,
                load_next,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(
                    NUM_ACC_N * _PACK
                    + NUM_B_SCALE
                )
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            yv = list(acc) + [h for ni_list in b_next for h in ni_list]
            if const_expr(async_a_copy):
                yv += sb_next
                yv += sa_next
            state = yield yv
        acc = [Vec(r) for r in state[:N_ACC]]
        b_prev = [
            [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
            for ni in range(NUM_ACC_N)
        ]
        final_off = (last & fx.Int32(1)) * fx.Int32(a_lds_i32)

        def final_a_load(mi, ks, _base=final_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        if const_expr(async_a_copy):
            sb = [
                fx.Int32(
                    state[B_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_B_SCALE
                )
            ]
            sa = [
                fx.Int32(
                    state[SB_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_A_SCALE
                )
            ]
        else:
            sb = b_scale.load_step(b_row, last)
            sa = a_scale.load_step(
                a_scale_lds,
                last,
            )
        acc = mfma.call_pipe_am_final(
            final_a_load,
            b_prev,
            acc,
            sa,
            sb,
        )
    else:
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_row_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        init = [mfma.zero_value for _ in range(N_ACC)]
        for sp_i, state in range(0, K_ITERS, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = (sp + fx.Int32(1) < last).select(
                sp + fx.Int32(1),
                last,
            )

            def a_load(mi, ks, _base=cur_off):
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            b = b_loader.load_step(b_row, sp)
            sa = a_scale.load_step(a_scale_lds, sp)
            sb = b_scale.load_step(b_row, sp)
            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )
            acc = mfma.call(
                a_load,
                b,
                acc,
                sa,
                sb,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(0)
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            state = yield list(acc)
        acc = [Vec(r) for r in state]
    # The epilogue aliases A_buf as cshuffle LDS after every wave finishes its final A ds_read.
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base)


# fmt: off
def build_fused_gemm1(*, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, expert_rsrc, out_tensor, a_buf, a_scale_lds, c_tile,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource, swiglu_limit=0.0):
    # fmt: on
    """Build the GEMM1 atoms and return its expert resolver and tile runner."""
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)

    # fmt: off
    a_gather = ATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy)
    # fmt: on
    a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a)
    b_loader = BWeightLoader(
        w_rsrc=w_rsrc,
        num_acc_n=num_acc_n,
        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
    )
    b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
    a_scale = AScaleLoader(
        scale_rsrc=sx_rsrc,
        m_repeat=m_repeat,
        model_dim=model_dim,
        sort_block_m=sort_block_m,
        total_threads=total_threads,
    )
    mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n)
    # fmt: off
    epi = SiluQuantEpilogue(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
        inter_dim=inter_dim, m_repeat=m_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
        num_waves=num_waves, lds_out=c_tile, swiglu_limit=swiglu_limit, always_valid=True,
        out_tensor=out_tensor if use_tile_resource else None)
    # fmt: on

    def _decode(flat):
        m_tile = flat // fx.Int32(n_tiles)
        n_tile = flat - m_tile * fx.Int32(n_tiles)
        return m_tile, n_tile

    def expert_of_flat(flat):
        m_tile, _n = _decode(flat)
        return sched.expert_of(m_tile)

    def do_scheduled_tile(flat):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        # fmt: off
        do_tile(m_tile, n_tile_base, expert, sched, a_gather,
            a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
            a_k_step_bytes, pipe_weights, mfma_amajor, async_a_copy,
            trb_rsrc)
        # fmt: on

    return expert_of_flat, do_scheduled_tile


# fmt: off
@functools.lru_cache(maxsize=None)
def compile_gemm1(
    *, model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, async_a_copy: bool = False,
    use_tile_resource: bool = True, waves_per_eu_hint: int = 2, b_cache_modifier: int = 0,
    swiglu_limit: float = 0.0,
):
    # fmt: on
    """Compile standalone group GEMM1 from the fused Stage1 compute body."""
    num_waves = int(num_waves)
    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0

    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0

    a_k_step_bytes = tile_k
    k_iters = model_dim // tile_k
    total_threads = num_waves * 64
    a_lds_size = sort_block_m * a_k_step_bytes
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_rsrc = _make_buffer(
                out, fx.Int16, max_size=False, num_records_bytes=num_valid * fx.Int32(inter_dim)
            )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = _make_buffer(
            out_scale,
            fx.Int8,
            max_size=False,
            num_records_bytes=num_valid * fx.Int32(scale_cols) + fx.Int32(8192),
        )
        wave_id = fx.thread_idx.x // 64

        _, run_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, sw_rsrc=sw_rsrc,
            sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc, trb_rsrc=trb_rsrc,
            expert_rsrc=expert_rsrc, out_tensor=out, a_buf=a_buf,
            a_scale_lds=a_scale_lds, c_tile=c_tile, model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, n_per_wave=n_per_wave,
            wave_id=wave_id, m_repeat=m_repeat, num_acc_n=num_acc_n, a_k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, k_iters=k_iters, a_lds_i32=a_lds_i32, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier, swizzle_a=swizzle_a,
            pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, async_a_copy=async_a_copy,
            use_tile_resource=use_tile_resource, swiglu_limit=swiglu_limit,
        )
        total_work = (num_valid // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            run_tile(flat)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


# fmt: off
def gemm1_kernel(
    out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, stream, *,
    model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 4,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True, waves_per_eu_hint: int = 2,
    num_cu: int = 256, b_cache_modifier: int = 0, swiglu_limit: float = 0.0,
):
    # fmt: on
    """Run standalone MegaMoEV2 group GEMM1 and return ``(out, out_scale)``."""
    num_valid = int(num_valid)
    if num_valid < 0 or num_valid % int(sort_block_m):
        raise ValueError("num_valid must be a non-negative multiple of sort_block_m")
    if num_valid == 0:
        return out, out_scale
    n_tiles = (2 * int(inter_dim)) // int(tile_n)
    total_work = (num_valid // int(sort_block_m)) * n_tiles
    grid_x = min(total_work, int(num_cu) * int(grid_mult))
    launch = compile_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, expert_offset=expert_offset,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, b_cache_modifier=b_cache_modifier,
        swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale,
        fx.Int32(num_valid), fx.Int32(grid_x), stream,
    )
    return out, out_scale
