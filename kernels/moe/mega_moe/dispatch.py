# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused all-to-all EP dispatch prologue for the MoE stage-1 megakernel.

Traced inline at the GEMM call site; runs three mutually-exclusive fixed-slot schemes (fixedslot /
naive-compact / compact-allgather), each guarded by ``const_expr`` and ending in a grid barrier.
Python if/for is lowered to ``scf`` via ``ASTRewriter.transform``.
"""

import mori.ir.flydsl as mori_shmem

import flydsl.expr as fx
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr
from kernels.common.utils import lds_atomic_add, lds_load, lds_store

from .utils import epk


def dsl_rewrite(func):
    """Lower a device helper's Python if/for/while into scf via the AST rewriter."""
    return ASTRewriter.transform(func)


@dsl_rewrite
def emit_dispatch_prologue(
    *,
    # fuse compile-time consts
    num_waves,
    sort_block_m,
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
    # mode flags
    fuse_fs,
    compact,
    static_tiles,
    # LDS pong arena + histogram offset (same arena as GEMM, not re-allocated)
    allocator_pong,
    cnt_lds_off,
    # runtime SSA kernel args
    addr_disp,
    i32_cur_tok,
    addr_in_tok,
    addr_in_idx,
    addr_in_wts,
    addr_in_sc,
):
    # Fused dispatch prologue: decode, handshake-free fixed-slot; same persistent grid as the GEMM.
    # _ctm/_ctm1 hoisted before the const_expr branches (rewriter needs the local set before control flow).
    ctm = arith.constant(sort_block_m)
    ctm1 = arith.constant(sort_block_m - 1)
    # Traced only for the decode fixed-slot scheme; _compact has its own branch below.
    if const_expr(fuse_fs and not compact):
        crfa = buffer_ops.create_buffer_resource_from_addr
        rdisp = crfa(addr_disp)

        def dp(i):
            return buffer_ops.buffer_load(rdisp, arith.constant(i), vec_width=1, dtype=T.i64)

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
        a_meta = dp(18)  # metadata-ready flag (block0 release / consumers acquire)
        # total_recv / dest_ctr / recv_num (disp idx 21-24; distinct-recv counts for stage2 combine)
        a_trecv = dp(21)
        a_dctr = dp(22)
        a_rnum = dp(23)
        p_rnum = dp(24)

        tid = fx.thread_idx.x
        lane = tid & 63
        warp = tid >> 6
        gdx = fx.grid_dim.x
        flat = fx.block_idx.y * gdx + fx.block_idx.x
        gwid = flat * num_waves + warp
        nblk = gdx * fx.grid_dim.y
        gwn = nblk * num_waves
        wl = i32_cur_tok * arith.constant(fz_k)
        c_epr = arith.constant(fz_epr)
        c_cap = arith.constant(fz_cap)
        r_idx = crfa(a_idx)
        r_wts = crfa(a_wts)

        # No initial GB1: payload writes are independent peer atomics; launch epoch derived at the first grid barrier.
        bn = epk._to_i64(nblk)

        # write: each (token,expert) -> fixed slot le*cap + atomic(running[le])
        for wk in range(gwid, wl, gwn):
            src_tok = wk // arith.constant(fz_k)
            k_slot = wk % arith.constant(fz_k)
            expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=T.i32)
            is_valid = arith.cmpi(CmpIPredicate.ult, expert, arith.constant(fz_total_experts))
            dest_pe = expert // c_epr
            le = expert % c_epr
            off_l0 = arith.constant(0)
            if lane == 0:
                if is_valid:
                    run_remote = buffer_ops.buffer_load(crfa(p_run), dest_pe, vec_width=1, dtype=T.i64)
                    off_l0 = epk.atomic_add_system(run_remote + epk._to_i64(le) * 4, arith.constant(1))
            off = epk._readlane0(off_l0)
            in_range = arith.cmpi(CmpIPredicate.ult, off, c_cap)
            do_pub = arith.select(is_valid, in_range, is_valid)
            slot = le * c_cap + off

            if lane == 0:
                if do_pub:
                    # total_recv: count each (src token -> dest PE) once (distinct-dest dedup)
                    prior_same = arith.constant(0, type=T.i32)
                    for kk_d in range_constexpr(fz_k):
                        kk_idx = arith.constant(kk_d)
                        is_prior = arith.cmpi(CmpIPredicate.ult, kk_idx, k_slot)
                        ee_d = buffer_ops.buffer_load(
                            r_idx, src_tok * arith.constant(fz_k) + kk_idx, vec_width=1, dtype=T.i32
                        )
                        vld_d = arith.cmpi(CmpIPredicate.ult, ee_d, arith.constant(fz_total_experts))
                        dpe_d = ee_d // c_epr
                        same_d = arith.cmpi(CmpIPredicate.eq, dpe_d, dest_pe)
                        dup_d = arith.andi(is_prior, arith.andi(vld_d, same_d))
                        prior_same = arith.select(dup_d, arith.constant(1, type=T.i32), prior_same)
                    is_first_dest = arith.cmpi(CmpIPredicate.eq, prior_same, arith.constant(0, type=T.i32))
                    if is_first_dest:
                        epk.atomic_add_agent(a_dctr + epk._to_i64(dest_pe) * 4, arith.constant(1))
                    wt_val = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=T.f32)
                    src_enc = (arith.constant(fz_rank * fz_mtpr) + src_tok) | (k_slot << arith.constant(24))
                    idx_remote = buffer_ops.buffer_load(crfa(p_idx), dest_pe, vec_width=1, dtype=T.i64)
                    buffer_ops.buffer_store(expert, crfa(idx_remote), slot)
                    wts_remote = buffer_ops.buffer_load(crfa(p_wts), dest_pe, vec_width=1, dtype=T.i64)
                    buffer_ops.buffer_store(arith.bitcast(T.i32, wt_val), crfa(wts_remote), slot)
                    sm_remote = buffer_ops.buffer_load(crfa(p_sm), dest_pe, vec_width=1, dtype=T.i64)
                    buffer_ops.buffer_store(src_enc, crfa(sm_remote), slot)

            if const_expr(fz_enable_scales):
                if lane < fz_scale_n_i32:
                    if do_pub:
                        sc_val = buffer_ops.buffer_load(
                            crfa(a_sc), src_tok * arith.constant(fz_scale_n_i32) + lane, vec_width=1, dtype=T.i32
                        )
                        sc_remote = buffer_ops.buffer_load(crfa(p_sc), dest_pe, vec_width=1, dtype=T.i64)
                        buffer_ops.buffer_store(sc_val, crfa(sc_remote), slot * arith.constant(fz_scale_n_i32) + lane)

            # token embedding copy (all lanes, 16B v4i32 chunks)
            tok_remote_base = buffer_ops.buffer_load(crfa(p_rx), dest_pe, vec_width=1, dtype=T.i64)
            rsrc_src = crfa(a_tok + epk._to_i64(src_tok) * fz_nbytes)
            rsrc_dst = crfa(tok_remote_base + epk._to_i64(slot) * fz_nbytes)
            lane_off = lane * 4
            if const_expr(fz_n_i32 >= 512 and fz_safe_end_i32 > 0):
                ce_main = arith.select(do_pub, arith.constant(fz_safe_end_i32), lane_off)
                for co in range(lane_off, ce_main, 512):
                    va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=T.i32)
                    vb = buffer_ops.buffer_load(rsrc_src, co + 256, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(va, rsrc_dst, co)
                    buffer_ops.buffer_store(vb, rsrc_dst, co + 256)
            if const_expr(fz_safe_end_i32 < fz_n_i32):
                ce_tail = arith.select(do_pub, arith.constant(fz_n_i32), lane_off)
                for co in range(lane_off + fz_safe_end_i32, ce_tail, 256):
                    va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(va, rsrc_dst, co)
            elif const_expr(fz_n_i32 < 512):
                ce_small = arith.select(do_pub, arith.constant(fz_n_i32), lane_off)
                for co in range(lane_off, ce_small, 256):
                    va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(va, rsrc_dst, co)

        # Post-write arrival (N->1, only block0 waits; not a symmetric grid barrier): each block arrives via
        # atomic_add gb1 (release); block0 spins (acquire, sees all writes), does cross-PE + post-pass, publishes meta_flag.
        fx.barrier()
        e0_i32 = arith.constant(0)
        if tid == 0:
            e0_i32 = buffer_ops.buffer_load(crfa(a_meta), 0, vec_width=1, dtype=T.i32)  # = L-1
            one2 = arith.constant(1, type=T.i64)
            epk.fence_agent_release()
            epk.atomic_add_agent(a_gb1, one2)  # arrive (no spin here)

        # block0: wait arrivals -> cross-PE recv-count handshake -> post-pass -> publish flag
        if gwid == 0:
            one2b = arith.constant(1, type=T.i64)
            gb1now = buffer_ops.buffer_load(crfa(a_gb1), 0, vec_width=1, dtype=T.i64)
            tg2 = ((arith.ArithValue(gb1now, signed=False) - one2b) // bn + one2b) * bn
            mori_shmem.int64_wait_until_equals(a_gb1, tg2)
            epk.fence_agent_acquire()
            epoch_i32 = arith.trunci(T.i32, arith.unwrap(tg2 // bn))
            # Cross-PE handshake (also the payload/running barrier): signal peer recv_num[rank]=dest_ctr+1
            # (>=1 even for 0 tokens) after fence release; block0 wait+acquire makes peer writes visible
            # before post-pass reads running[]. recv_num self-resets; dest_ctr reset here; total_recv pre-zeroed.
            epk.fence_system_release()
            for dpe in range(lane, fz_npes, 64):
                sig = buffer_ops.buffer_load(crfa(a_dctr), dpe, vec_width=1, dtype=T.i32) + arith.constant(1)
                rnum_remote = (
                    buffer_ops.buffer_load(crfa(p_rnum), dpe, vec_width=1, dtype=T.i64) + epk._to_i64(fz_rank) * 4
                )
                mori_shmem.int32_wait_until_equals(rnum_remote, arith.constant(0))
                epk.store_i32_system(rnum_remote, arith.constant(0), sig)
            for spe in range(lane, fz_npes, 64):
                rn_src = a_rnum + epk._to_i64(spe) * 4
                sv = mori_shmem.int32_wait_until_greater_than(rn_src, arith.constant(0))
                epk.store_i32_system(rn_src, arith.constant(0), arith.constant(0))
                _ = epk.atomic_add_system(a_trecv, sv - arith.constant(1))
                buffer_ops.buffer_store(arith.constant(0), crfa(a_dctr), spe)
            epk.fence_system_acquire()
            # post-pass: emit ONLY occupied tiles compactly + fold count reset
            r_run = crfa(a_run)
            r_se = crfa(a_se)
            r_trb = crfa(a_trb)
            r_nv = crfa(a_nv)
            r_cnt = crfa(a_cnt)
            ctm = arith.constant(fz_tile_m)
            ctm1 = arith.constant(fz_tile_m - 1)
            if const_expr(static_tiles and fz_epr <= 64):
                # P-static: no compact tile-list (se/trb); only num_valid + copy/reset (lane le owns expert le).
                # GEMM derives (expert,k) via an on-the-fly readlane prefix-scan of ll_count.
                le_ok = arith.cmpi(CmpIPredicate.ult, lane, arith.constant(fz_epr))
                run_lane = arith.select(
                    le_ok, buffer_ops.buffer_load(r_run, lane, vec_width=1, dtype=T.i32), arith.constant(0)
                )
                nt_lane = (run_lane + ctm1) // ctm
                tot = arith.constant(0)
                for j in range_constexpr(fz_epr):
                    tot = tot + epk._readlane(nt_lane, j)
                nvv = tot * ctm
                if lane == 0:
                    buffer_ops.buffer_store(nvv, r_nv, arith.constant(0))
                    buffer_ops.buffer_store(nvv, r_nv, arith.constant(1))
                if le_ok:
                    buffer_ops.buffer_store(run_lane, r_cnt, lane)  # copy only; reset deferred past meta
            else:
                acc = arith.constant(0)
                for le2 in range(fz_epr):  # constexpr unroll (epr small)
                    cnt = buffer_ops.buffer_load(r_run, le2, vec_width=1, dtype=T.i32)
                    ntiles = (cnt + ctm1) // ctm
                    se_val = le2 + arith.constant(fz_rank * fz_epr)
                    trb_base = le2 * arith.constant(fz_cap)
                    for t in range(lane, ntiles, 64):
                        ci = acc + t
                        buffer_ops.buffer_store(se_val, r_se, ci)
                        buffer_ops.buffer_store(trb_base + t * ctm, r_trb, ci)
                    acc = acc + ntiles
                nvv = acc * ctm
                buffer_ops.buffer_store(nvv, r_nv, arith.constant(0))
                buffer_ops.buffer_store(nvv, r_nv, arith.constant(1))
                for lei in range(lane, fz_epr, 64):
                    cnt2 = buffer_ops.buffer_load(r_run, lei, vec_width=1, dtype=T.i32)
                    buffer_ops.buffer_store(cnt2, r_cnt, lei)
                    buffer_ops.buffer_store(arith.constant(0), r_run, lei)
            if const_expr(static_tiles and fz_epr <= 64):
                # nv/ll_count are intra-GPU, so agent release suffices for GEMM blocks. running reset is
                # deferred past meta (off critical path); its system fence keeps it cross-PE-visible next launch.
                epk.fence_agent_release()
                buffer_ops.buffer_store(epoch_i32, crfa(a_meta), arith.constant(0))
                le_okr = arith.cmpi(CmpIPredicate.ult, lane, arith.constant(fz_epr))
                if le_okr:
                    buffer_ops.buffer_store(arith.constant(0), r_run, lane)
                epk.fence_system_release()
            else:
                epk.fence_system_release()
                # publish metadata-ready: release-store meta_flag = epoch (1 writer, no atomic storm)
                epk.fence_agent_release()
                buffer_ops.buffer_store(epoch_i32, crfa(a_meta), arith.constant(0))

        # all blocks: wait metadata ready (acquire; N readers spin a cacheline)
        if tid == 0:
            mori_shmem.int32_wait_until_greater_than(a_meta, e0_i32)
            epk.fence_agent_acquire()
        fx.barrier()

    # Naive compact (all-gather, 2 cross-PE rounds): count -> allgather -> my_base+meta -> strict write -> payload.
    if const_expr(fuse_fs and compact):
        crfa = buffer_ops.create_buffer_resource_from_addr
        rdisp = crfa(addr_disp)

        def dpa(i):
            return buffer_ops.buffer_load(rdisp, arith.constant(i), vec_width=1, dtype=T.i64)

        a_tok = addr_in_tok
        a_idx = addr_in_idx
        a_wts = addr_in_wts
        a_sc = addr_in_sc
        a_gb1 = dpa(4)
        a_done2 = dpa(6)
        a_cnt = dpa(7)
        p_rx = dpa(8)
        p_sc = dpa(9)
        p_idx = dpa(10)
        p_wts = dpa(11)
        p_sm = dpa(12)
        p_done2 = dpa(14)
        a_se = dpa(15)
        a_trb = dpa(16)
        a_nv = dpa(17)
        a_meta = dpa(18)
        a_gbc = dpa(21)
        a_meta2 = dpa(24)
        # all-gather extras (disp idx 29..35)
        a_lh = dpa(29)
        a_bc = dpa(30)
        p_bc = dpa(31)
        a_cd = dpa(32)
        p_cd = dpa(33)
        a_mb = dpa(34)
        a_lc = dpa(35)
        # total_recv/dest_ctr/recv_num at disp idx 36-39 (21=gb_cnt, 24=meta2 already used by compact_ag)
        a_trecv = dpa(36)
        a_dctr = dpa(37)
        a_rnum = dpa(38)
        p_rnum = dpa(39)

        tid = fx.thread_idx.x
        lane = tid & 63
        warp = tid >> 6
        gdx = fx.grid_dim.x
        flat = fx.block_idx.y * gdx + fx.block_idx.x
        gwid = flat * num_waves + warp
        nblk = gdx * fx.grid_dim.y
        gwn = nblk * num_waves
        wl = i32_cur_tok * arith.constant(fz_k)
        c_epr = arith.constant(fz_epr)
        r_idx = crfa(a_idx)
        r_wts = crfa(a_wts)
        r_lh = crfa(a_lh)
        bn = epk._to_i64(nblk)
        ctm = arith.constant(fz_tile_m)
        ctm1 = arith.constant(fz_tile_m - 1)
        c_te = arith.constant(fz_total_experts)
        c_te_sz = fz_total_experts

        e0 = arith.constant(0)
        e0m2 = arith.constant(0)
        if tid == 0:
            e0 = buffer_ops.buffer_load(crfa(a_meta), 0, vec_width=1, dtype=T.i32)
            e0m2 = buffer_ops.buffer_load(crfa(a_meta2), 0, vec_width=1, dtype=T.i32)

        # PHASE-0: per-block LDS histogram (ds_add, all threads) -> reduce to global local_hist (1 atomic/expert/block).
        def ix0(i):
            return arith.unwrap(arith.index_cast(T.index, arith.unwrap(i)))

        # Hoist LDS base to an MLIR value once: the rewriter can't carry the SmemAllocator across the if _gwid==0.
        pong_base = allocator_pong.get_base()
        chl = SmemPtr(pong_base, cnt_lds_off, T.i32, shape=(fz_total_experts,)).get()
        c_bt = num_waves * 64  # block threads (Python int)
        gtid = flat * arith.constant(c_bt) + tid  # global thread id
        gnt = nblk * arith.constant(c_bt)  # total threads (runtime)
        # clear LDS histogram
        for e in range(tid, fz_total_experts, c_bt):
            lds_store(arith.constant(0), chl, ix0(e))
        fx.barrier()
        # histogram: all threads, grid-strided over (token,expert) pairs, LDS ds_add
        for wk in range(gtid, wl, gnt):
            expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=T.i32)
            if arith.cmpi(CmpIPredicate.ult, expert, c_te):
                _ = lds_atomic_add(arith.constant(1), chl, ix0(expert))
        fx.barrier()
        # reduce LDS histogram -> global local_hist (1 HBM atomic per non-empty expert)
        for e in range(tid, fz_total_experts, c_bt):
            hv = arith.ArithValue(lds_load(chl, ix0(e)))
            if arith.cmpi(CmpIPredicate.sgt, hv, arith.constant(0)):
                _ = epk.atomic_add_agent(a_lh + epk._to_i64(e) * 4, hv)
        rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == 0:
            epk.fence_agent_release()
            epk.atomic_add_agent(a_gb1, arith.constant(1, type=T.i64))

        # block0: wait count -> all-gather bigcnt -> my_base + metadata -> meta
        if gwid == 0:
            gb1now = buffer_ops.buffer_load(crfa(a_gb1), 0, vec_width=1, dtype=T.i64)
            one2 = arith.constant(1, type=T.i64)
            tg = ((arith.ArithValue(gb1now, signed=False) - one2) // bn + one2) * bn
            mori_shmem.int64_wait_until_equals(a_gb1, tg)
            epk.fence_agent_acquire()
            epoch_i32 = arith.trunci(T.i32, arith.unwrap(tg // bn))
            # PUB: write my local_hist into every peer's bigcnt[rank*te .. ] (all-gather)
            epk.fence_system_release()
            base_row = arith.constant(fz_rank * c_te_sz)
            te4 = (c_te_sz // 4) * 4
            for p in range_constexpr(fz_npes):
                bc_remote = buffer_ops.buffer_load(crfa(p_bc), p, vec_width=1, dtype=T.i64)
                rdst = crfa(bc_remote)
                if const_expr(te4 > 0):
                    for off in range(lane * 4, te4, 256):
                        v4 = buffer_ops.buffer_load(r_lh, off, vec_width=4, dtype=T.i32)
                        buffer_ops.buffer_store(v4, rdst, base_row + off)
                if const_expr(te4 < c_te_sz):
                    for off in range(lane + te4, c_te_sz, 64):
                        v1 = buffer_ops.buffer_load(r_lh, off, vec_width=1, dtype=T.i32)
                        buffer_ops.buffer_store(v1, rdst, base_row + off)
            epk.fence_system_release()
            for p in range(lane, fz_npes, 64):
                cd_remote = buffer_ops.buffer_load(crfa(p_cd), p, vec_width=1, dtype=T.i64)
                epk.store_i32_system(cd_remote, arith.constant(fz_rank), epoch_i32)
            for s in range(lane, fz_npes, 64):
                mori_shmem.int32_wait_until_equals(a_cd + epk._to_i64(s) * 4, epoch_i32)
            epk.fence_system_acquire()
            r_bc = crfa(a_bc)
            r_mb = crfa(a_mb)
            # Reduce bigcnt[npes*te] -> LDS cs[ge] (total) + sp[ge] (sender<rank prefix), parallel over warp0.
            # warp0-only: use s_waitcnt for LDS ordering (fx.barrier would deadlock: other warps are at meta wait).
            csl = SmemPtr(pong_base, cnt_lds_off, T.i32, shape=(2 * fz_total_experts,)).get()
            for ge in range(lane, fz_total_experts, 64):
                cs = arith.constant(0)
                sp = arith.constant(0)
                for s in range_constexpr(fz_npes):
                    v = buffer_ops.buffer_load(r_bc, arith.constant(s * c_te_sz) + ge, vec_width=1, dtype=T.i32)
                    cs = cs + v
                    if const_expr(s < fz_rank):
                        sp = sp + v
                lds_store(cs, csl, ix0(ge))
                lds_store(sp, csl, ix0(ge + arith.constant(fz_total_experts)))
            rocdl.s_waitcnt(0)
            # CMP my_base[ge] = (dest-expert tile-padded base) + sp[ge]; reads cs/sp from LDS
            for d in range(lane, fz_npes, 64):
                eb = arith.constant(0)
                for le in range_constexpr(fz_epr):
                    ge = d * c_epr + arith.constant(le)
                    cs = arith.ArithValue(lds_load(csl, ix0(ge)))
                    sp = arith.ArithValue(lds_load(csl, ix0(ge + arith.constant(fz_total_experts))))
                    buffer_ops.buffer_store(eb + sp, r_mb, ge)
                    eb = eb + ((cs + ctm1) // ctm) * ctm
            # tile metadata for MY local experts + expected (ll_count) + num_valid (cs from LDS)
            r_se = crfa(a_se)
            r_trb = crfa(a_trb)
            r_nv = crfa(a_nv)
            r_cnt = crfa(a_cnt)
            # block0 sentinel-fills only the tile-pad GAP slots of this rank's srcmap (real slots are
            # P2P-written by peers in PHASE-2); GEMM reads srcmap directly (pad = sentinel -> t_ok masks).
            r_sm = crfa(dpa(19))
            ca_sent = arith.constant(fz_npes * fz_mtpr)
            acc = arith.constant(0)
            ebase = arith.constant(0)
            for le2 in range(fz_epr):
                cnt = arith.ArithValue(lds_load(csl, ix0(arith.constant(fz_rank * fz_epr) + le2)))
                ntiles = (cnt + ctm1) // ctm
                se_val = le2 + arith.constant(fz_rank * fz_epr)
                for t in range(lane, ntiles, 64):
                    ci = acc + t
                    buffer_ops.buffer_store(se_val, r_se, ci)
                    buffer_ops.buffer_store(ebase + t * ctm, r_trb, ci)
                if lane == 0:
                    buffer_ops.buffer_store(cnt, r_cnt, le2)
                for pp in range(lane, ntiles * ctm - cnt, 64):
                    buffer_ops.buffer_store(ca_sent, r_sm, ebase + cnt + pp)
                acc = acc + ntiles
                ebase = ebase + ntiles * ctm
            nvv = acc * ctm
            buffer_ops.buffer_store(nvv, r_nv, arith.constant(0))
            buffer_ops.buffer_store(nvv, r_nv, arith.constant(1))
            epk.fence_system_release()
            epk.fence_agent_release()
            buffer_ops.buffer_store(epoch_i32, crfa(a_meta), arith.constant(0))

        if tid == 0:
            mori_shmem.int32_wait_until_greater_than(a_meta, e0)
            epk.fence_agent_acquire()
        fx.barrier()

        # ---- PHASE-2: strict write to my_base[ge] + LOCAL cursor (no remote base read) ----
        r_mb2 = crfa(a_mb)
        for wk in range(gwid, wl, gwn):
            src_tok = wk // arith.constant(fz_k)
            k_slot = wk % arith.constant(fz_k)
            expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=T.i32)
            is_valid = arith.cmpi(CmpIPredicate.ult, expert, c_te)
            dest_pe = expert // c_epr
            le = expert % c_epr
            off_l0 = arith.constant(0)
            mb_l0 = arith.constant(0)
            if lane == 0:
                if is_valid:
                    off_l0 = epk.atomic_add_agent(a_lc + epk._to_i64(expert) * 4, arith.constant(1))
                    mb_l0 = buffer_ops.buffer_load(r_mb2, expert, vec_width=1, dtype=T.i32)
            off = epk._readlane0(off_l0)
            mb = epk._readlane0(mb_l0)
            do_pub = is_valid
            slot = mb + off

            if lane == 0:
                if do_pub:
                    wt_val = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=T.f32)
                    src_enc = (arith.constant(fz_rank * fz_mtpr) + src_tok) | (k_slot << arith.constant(24))
                    idx_remote = buffer_ops.buffer_load(crfa(p_idx), dest_pe, vec_width=1, dtype=T.i64)
                    buffer_ops.buffer_store(expert, crfa(idx_remote), slot)
                    wts_remote = buffer_ops.buffer_load(crfa(p_wts), dest_pe, vec_width=1, dtype=T.i64)
                    buffer_ops.buffer_store(arith.bitcast(T.i32, wt_val), crfa(wts_remote), slot)
                    sm_remote = buffer_ops.buffer_load(crfa(p_sm), dest_pe, vec_width=1, dtype=T.i64)
                    buffer_ops.buffer_store(src_enc, crfa(sm_remote), slot)

            if const_expr(fz_enable_scales):
                if lane < fz_scale_n_i32:
                    if do_pub:
                        sc_val = buffer_ops.buffer_load(
                            crfa(a_sc), src_tok * arith.constant(fz_scale_n_i32) + lane, vec_width=1, dtype=T.i32
                        )
                        sc_remote = buffer_ops.buffer_load(crfa(p_sc), dest_pe, vec_width=1, dtype=T.i64)
                        buffer_ops.buffer_store(sc_val, crfa(sc_remote), slot * arith.constant(fz_scale_n_i32) + lane)

            tok_remote_base = buffer_ops.buffer_load(crfa(p_rx), dest_pe, vec_width=1, dtype=T.i64)
            rsrc_src = crfa(a_tok + epk._to_i64(src_tok) * fz_nbytes)
            rsrc_dst = crfa(tok_remote_base + epk._to_i64(slot) * fz_nbytes)
            lane_off = lane * 4
            if const_expr(fz_n_i32 >= 512 and fz_safe_end_i32 > 0):
                ce_main = arith.select(do_pub, arith.constant(fz_safe_end_i32), lane_off)
                for co in range(lane_off, ce_main, 512):
                    va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=T.i32)
                    vb = buffer_ops.buffer_load(rsrc_src, co + 256, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(va, rsrc_dst, co)
                    buffer_ops.buffer_store(vb, rsrc_dst, co + 256)
            if const_expr(fz_safe_end_i32 < fz_n_i32):
                ce_tail = arith.select(do_pub, arith.constant(fz_n_i32), lane_off)
                for co in range(lane_off + fz_safe_end_i32, ce_tail, 256):
                    va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(va, rsrc_dst, co)
            elif const_expr(fz_n_i32 < 512):
                ce_small = arith.select(do_pub, arith.constant(fz_n_i32), lane_off)
                for co in range(lane_off, ce_small, 256):
                    va = buffer_ops.buffer_load(rsrc_src, co, vec_width=4, dtype=T.i32)
                    buffer_ops.buffer_store(va, rsrc_dst, co)

        # Per-token distinct-dest dedup -> dest_ctr: per-block LDS histogram (ds_add, all threads; each
        # thread dedups its token's k dests via a bitmask) -> one global atomic/dest/block; reuses _chl LDS.
        ckk = arith.constant(fz_k)
        cte2 = arith.constant(fz_total_experts)
        for e2 in range(tid, fz_npes, c_bt):
            lds_store(arith.constant(0), chl, ix0(e2))
        fx.barrier()
        for dt in range(gtid, i32_cur_tok, gnt):
            seen = arith.constant(0)
            for kk in range_constexpr(fz_k):
                ee = buffer_ops.buffer_load(r_idx, dt * ckk + arith.constant(kk), vec_width=1, dtype=T.i32)
                vld2 = arith.cmpi(CmpIPredicate.ult, ee, cte2)
                dpe2 = ee // c_epr
                already = (seen >> dpe2) & arith.constant(1)
                isnew = arith.andi(vld2, arith.cmpi(CmpIPredicate.eq, already, arith.constant(0)))
                if isnew:
                    _ = lds_atomic_add(arith.constant(1), chl, ix0(dpe2))
                seen = arith.select(isnew, seen | (arith.constant(1) << dpe2), seen)
        fx.barrier()
        for e2 in range(tid, fz_npes, c_bt):
            hv2 = arith.ArithValue(lds_load(chl, ix0(e2)))
            if arith.cmpi(CmpIPredicate.sgt, hv2, arith.constant(0)):
                _ = epk.atomic_add_agent(a_dctr + epk._to_i64(e2) * 4, hv2)

        # write grid-arrival (gb_cnt) + cross-PE#2 (done2) + payload-ready flag
        rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == 0:
            epk.fence_agent_release()
            epk.atomic_add_agent(a_gbc, arith.constant(1, type=T.i64))
        if gwid == 0:
            gbcnow = buffer_ops.buffer_load(crfa(a_gbc), 0, vec_width=1, dtype=T.i64)
            one2b = arith.constant(1, type=T.i64)
            tgw = ((arith.ArithValue(gbcnow, signed=False) - one2b) // bn + one2b) * bn
            mori_shmem.int64_wait_until_equals(a_gbc, tgw)
            epk.fence_agent_acquire()
            epochw_i32 = arith.trunci(T.i32, arith.unwrap(tgw // bn))
            epk.fence_system_release()
            for dpe in range(lane, fz_npes, 64):
                d2w = buffer_ops.buffer_load(crfa(p_done2), dpe, vec_width=1, dtype=T.i64)
                epk.store_i32_system(d2w, arith.constant(fz_rank), epochw_i32)
            for spe in range(lane, fz_npes, 64):
                mori_shmem.int32_wait_until_equals(a_done2 + epk._to_i64(spe) * 4, epochw_i32)
            epk.fence_system_acquire()
            # Cross-PE recv-count -> total_recv: signal peer recv_num[rank]=dest_ctr+1, then total_recv += (signal-1).
            # recv_num self-resets (wait==0 before write, 0 after read); dest_ctr reset here.
            epk.fence_system_release()
            for dpe2b in range(lane, fz_npes, 64):
                sig = buffer_ops.buffer_load(crfa(a_dctr), dpe2b, vec_width=1, dtype=T.i32) + arith.constant(1)
                rnum_remote = (
                    buffer_ops.buffer_load(crfa(p_rnum), dpe2b, vec_width=1, dtype=T.i64) + epk._to_i64(fz_rank) * 4
                )
                mori_shmem.int32_wait_until_equals(rnum_remote, arith.constant(0))
                epk.store_i32_system(rnum_remote, arith.constant(0), sig)
            for spe2 in range(lane, fz_npes, 64):
                rn_src = a_rnum + epk._to_i64(spe2) * 4
                sv = mori_shmem.int32_wait_until_greater_than(rn_src, arith.constant(0))
                epk.store_i32_system(rn_src, arith.constant(0), arith.constant(0))
                _ = epk.atomic_add_system(a_trecv, sv - arith.constant(1))
                buffer_ops.buffer_store(arith.constant(0), crfa(a_dctr), spe2)
            epk.fence_system_acquire()
            epk.fence_agent_release()
            buffer_ops.buffer_store(epochw_i32, crfa(a_meta2), arith.constant(0))
        if tid == 0:
            mori_shmem.int32_wait_until_greater_than(a_meta2, e0m2)
            epk.fence_agent_acquire()
        fx.barrier()
