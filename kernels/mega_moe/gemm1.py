# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1 compute for fused MegaMoE v2 stage1."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from .gemm_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    wait_lds_barrier,
)


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
    tile_row_base = _buffer_ops.buffer_load(trb_rsrc, m_tile, vec_width=1, dtype=fx.Int32)
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
def build_fused_gemm1(*, x_rsrc, x_base_addr, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, expert_rsrc, out_base_addr, a_buf, a_scale_lds, c_tile,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource):
    # fmt: on
    """Build the GEMM1 atoms and return (expert_of_flat, do_scheduled_tile).

    do_scheduled_tile(flat) runs one (m_tile, n_tile) tile; sched is returned so
    the caller can resolve a tile's expert for dispatch-side payload waits.
    """
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)

    # fmt: off
    a_gather = ATileLoader(x_rsrc=x_rsrc, row_bytes=model_dim, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_base_addr=x_base_addr, x_tensor=x_tensor,
        async_copy=async_a_copy)
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
        num_waves=num_waves, lds_out=c_tile, always_valid=True,
        out_base_addr=out_base_addr if use_tile_resource else None)
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
