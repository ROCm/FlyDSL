# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Experimental producer-only fixed-slot EP dispatch for MegaMoE v2."""

import mori.ir.flydsl as mori_shmem

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, range_constexpr

from ..utils import epk


@flyc.jit
def emit_dispatch_prologue(
    *,
    num_waves,
    fz_npes,
    fz_epr,
    fz_k,
    fz_cap,
    fz_mtpr,
    fz_rank,
    fz_tile_m,
    fz_total_experts,
    fz_nbytes,
    fz_n_i32,
    fz_safe_end_i32,
    fz_scale_n_i32,
    fz_enable_scales,
    addr_disp,
    i32_cur_tok,
    addr_in_tok,
    addr_in_idx,
    addr_in_wts,
    addr_in_sc,
    dispatch_blocks,
    addr_ready
):
    """Producer-only fixed-slot EP dispatch for the oversubscribed v2 stage-1 kernel."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    a_tok = addr_in_tok
    a_idx = addr_in_idx
    a_wts = addr_in_wts
    a_sc = addr_in_sc
    a_gb1 = dp(4)
    a_run = dp(5)
    a_cnt = dp(7)
    p_rx = dp(8)
    p_sc = dp(9)
    p_idx = dp(10)
    p_wts = dp(11)
    p_sm = dp(12)
    p_run = dp(13)
    a_se = dp(15)
    a_trb = dp(16)
    a_nv = dp(17)
    a_meta = dp(18)
    a_sm = dp(19)
    a_trecv = dp(21)
    a_dctr = dp(22)
    a_rnum = dp(23)
    p_rnum = dp(24)
    tid = fx.thread_idx.x
    lane = tid & 63
    warp = tid >> 6
    flat = fx.block_idx.x
    nblk = fx.Int32(dispatch_blocks)
    gwid = flat * num_waves + warp
    gwn = nblk * num_waves
    wl = i32_cur_tok * fx.Int32(fz_k)
    c_epr = fx.Int32(fz_epr)
    c_cap = fx.Int32(fz_cap)
    r_idx = crfa(a_idx)
    r_wts = crfa(a_wts)
    bn = fx.Uint64(nblk)

    def _process(wk):
        src_tok = wk // fx.Int32(fz_k)
        k_slot = wk % fx.Int32(fz_k)
        expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
        is_valid = expert < fx.Int32(fz_total_experts)
        dest_pe = expert // c_epr
        le = expert % c_epr
        off_l0 = fx.Int32(0)
        if lane == 0:
            if is_valid:
                run_remote = buffer_ops.buffer_load(crfa(p_run), dest_pe, vec_width=1, dtype=fx.Int64)
                off_l0 = fx.Int32(epk.atomic_add_system(run_remote + fx.Int64(le) * fx.Int64(4), fx.Int32(1)))
        off = fx.Int32(epk._readlane0(off_l0))
        in_range = off < c_cap
        do_pub = is_valid & in_range
        slot = le * c_cap + off
        if lane == 0:
            if do_pub:
                prior_same = fx.Int32(0)
                for kk_d in range_constexpr(fz_k):
                    kk_idx = fx.Int32(kk_d)
                    is_prior = kk_idx < k_slot
                    ee_d = buffer_ops.buffer_load(r_idx, src_tok * fx.Int32(fz_k) + kk_idx, vec_width=1, dtype=fx.Int32)
                    vld_d = ee_d < fx.Int32(fz_total_experts)
                    dpe_d = ee_d // c_epr
                    same_d = dpe_d == dest_pe
                    dup_d = is_prior & vld_d & same_d
                    prior_same = dup_d.select(fx.Int32(1), prior_same)
                is_first_dest = prior_same == fx.Int32(0)
                if is_first_dest:
                    epk.atomic_add_agent(a_dctr + fx.Int64(dest_pe) * fx.Int64(4), fx.Int32(1))
                wt_val = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                src_enc = (fx.Int32(fz_rank * fz_mtpr) + src_tok) | (k_slot << fx.Int32(24))
                idx_remote = buffer_ops.buffer_load(crfa(p_idx), dest_pe, vec_width=1, dtype=fx.Int64)
                buffer_ops.buffer_store(expert, crfa(idx_remote), slot)
                wts_remote = buffer_ops.buffer_load(crfa(p_wts), dest_pe, vec_width=1, dtype=fx.Int64)
                wt_bits = fx.Vector.from_elements([wt_val], fx.Float32).bitcast(fx.Int32)[0]
                buffer_ops.buffer_store(wt_bits, crfa(wts_remote), slot)
                sm_remote = buffer_ops.buffer_load(crfa(p_sm), dest_pe, vec_width=1, dtype=fx.Int64)
                buffer_ops.buffer_store(src_enc, crfa(sm_remote), slot)
        if const_expr(fz_enable_scales):
            if lane < fz_scale_n_i32:
                if do_pub:
                    sc_val = buffer_ops.buffer_load(
                        crfa(a_sc), src_tok * fx.Int32(fz_scale_n_i32) + lane, vec_width=1, dtype=fx.Int32
                    )
                    sc_remote = buffer_ops.buffer_load(crfa(p_sc), dest_pe, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(sc_val, crfa(sc_remote), slot * fx.Int32(fz_scale_n_i32) + lane)
        tok_remote_base = buffer_ops.buffer_load(crfa(p_rx), dest_pe, vec_width=1, dtype=fx.Int64)
        rsrc_src = crfa(a_tok + fx.Int64(src_tok) * fx.Int64(fz_nbytes))
        rsrc_dst = crfa(tok_remote_base + fx.Int64(slot) * fx.Int64(fz_nbytes))
        lane_off = lane * 4
        if const_expr(fz_n_i32 >= 512 and fz_safe_end_i32 > 0):
            ce_main = do_pub.select(fx.Int32(fz_safe_end_i32), lane_off)
            for co in range(lane_off, ce_main, 512):
                va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=fx.Int32)
                vb = buffer_ops.buffer_load(rsrc_src, co + 256, vec_width=4, dtype=fx.Int32)
                buffer_ops.buffer_store(va, rsrc_dst, co)
                buffer_ops.buffer_store(vb, rsrc_dst, co + 256)
        if const_expr(fz_safe_end_i32 < fz_n_i32):
            ce_tail = do_pub.select(fx.Int32(fz_n_i32), lane_off)
            for co in range(lane_off + fz_safe_end_i32, ce_tail, 256):
                va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=fx.Int32)
                buffer_ops.buffer_store(va, rsrc_dst, co)
        elif const_expr(fz_n_i32 < 512):
            ce_small = do_pub.select(fx.Int32(fz_n_i32), lane_off)
            for co in range(lane_off, ce_small, 256):
                va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=fx.Int32)
                buffer_ops.buffer_store(va, rsrc_dst, co)

    for wk in range(gwid, wl, gwn):
        _process(wk)
    fx.barrier()
    if tid == 0:
        epk.fence_agent_release()
        epk.atomic_add_agent(a_gb1, fx.Int64(1))
    if gwid == 0:
        one2b = fx.Uint64(1)
        gb1now = fx.Uint64(buffer_ops.buffer_load(crfa(a_gb1), fx.Int32(0), vec_width=1, dtype=fx.Int64))
        tg2 = ((gb1now - one2b) // bn + one2b) * bn
        mori_shmem.int64_wait_until_equals(a_gb1, tg2)
        epk.fence_agent_acquire()
        epoch_i32 = fx.Int32(tg2 // bn)
        epk.fence_system_release()
        for dpe in range(lane, fz_npes, 64):
            sig = buffer_ops.buffer_load(crfa(a_dctr), dpe, vec_width=1, dtype=fx.Int32) + fx.Int32(1)
            rnum_remote = buffer_ops.buffer_load(crfa(p_rnum), dpe, vec_width=1, dtype=fx.Int64) + fx.Int64(
                fz_rank
            ) * fx.Int64(4)
            mori_shmem.int32_wait_until_equals(rnum_remote, fx.Int32(0))
            epk.store_i32_system(rnum_remote, fx.Int32(0), sig)
        for spe in range(lane, fz_npes, 64):
            rn_src = a_rnum + fx.Int64(spe) * fx.Int64(4)
            sv = fx.Int32(mori_shmem.int32_wait_until_greater_than(rn_src, fx.Int32(0)))
            epk.store_i32_system(rn_src, fx.Int32(0), fx.Int32(0))
            epk.atomic_add_system(a_trecv, sv - fx.Int32(1))
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_dctr), spe)
        epk.fence_system_acquire()
        r_run = crfa(a_run)
        r_se = crfa(a_se)
        r_trb = crfa(a_trb)
        r_nv = crfa(a_nv)
        r_cnt = crfa(a_cnt)
        r_sm = crfa(a_sm)
        ctm = fx.Int32(fz_tile_m)
        ctm1 = fx.Int32(fz_tile_m - 1)
        acc = fx.Int32(0)
        for le2 in range(fz_epr):
            cnt = buffer_ops.buffer_load(r_run, le2, vec_width=1, dtype=fx.Int32)
            ntiles = (cnt + ctm1) // ctm
            se_val = le2 + fx.Int32(fz_rank * fz_epr)
            trb_base = le2 * fx.Int32(fz_cap)
            for t in range(lane, ntiles, 64):
                ci = acc + t
                buffer_ops.buffer_store(se_val, r_se, ci)
                buffer_ops.buffer_store(trb_base + t * ctm, r_trb, ci)
            sentinel = fx.Int32(fz_npes * fz_mtpr)
            for pp in range(cnt + lane, ntiles * ctm, 64):
                buffer_ops.buffer_store(sentinel, r_sm, trb_base + pp)
            acc = acc + ntiles
        nvv = acc * ctm
        buffer_ops.buffer_store(nvv, r_nv, fx.Int32(0))
        buffer_ops.buffer_store(nvv, r_nv, fx.Int32(1))
        for lei in range(lane, fz_epr, 64):
            cnt2 = buffer_ops.buffer_load(r_run, lei, vec_width=1, dtype=fx.Int32)
            buffer_ops.buffer_store(cnt2, r_cnt, lei)
            buffer_ops.buffer_store(fx.Int32(0), r_run, lei)
        epk.fence_system_release()
        epk.fence_agent_release()
        buffer_ops.buffer_store(epoch_i32, crfa(a_meta), fx.Int32(0))
        epk.fence_system_release()
        if lane == 0:
            epk.store_i32_system(addr_ready, fx.Int32(0), fx.Int32(1))


@flyc.jit
def wait_dispatch_ready(addr_ready):
    """Consumer-side acquire for the split producer/consumer stage-1 kernel."""
    if fx.thread_idx.x == 0:
        mori_shmem.int32_wait_until_greater_than(addr_ready, fx.Int32(0))
        epk.fence_system_acquire()
    fx.barrier()
