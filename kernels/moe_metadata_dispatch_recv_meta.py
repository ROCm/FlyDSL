# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Metadata-only intranode MoE dispatch (recv_meta rows, no activation payload).

Mirrors slot allocation / barriers of :func:`kernels.dispatch_combine_intranode_kernel.make_dispatch_kernel`
Phase 1 routing loop (including **per-``dest_pe`` ``dest_ctr``** increments before the
post-loop barrier), but writes **32-byte** ``recv_meta`` rows (see :mod:`kernels.recv_meta_a8w4`)
instead of copying ``out_tok`` / separate ``out_wts`` / ``out_idx`` payloads.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, arith
from flydsl.expr import Stream
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from kernels.dispatch_combine_intranode_kernel import (
    _lv_unwrap,
    _to_i32,
    _to_idx,
    atomic_add_global_at,
    ballot_i64,
    readlane,
    store_i32_system,
)
import mori.ir.flydsl as mori_shmem


def make_metadata_dispatch_recv_meta_kernel(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
):
    max_recv = int(npes) * int(max_tok_per_rank)
    _rank = int(rank)
    _npes = int(npes)
    _epr = int(experts_per_rank)
    _ept = int(experts_per_token)
    _mtpr = int(max_tok_per_rank)

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def ep_metadata_dispatch_recv_meta(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_tok_off: fx.Int64,
        addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64,
        addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        addr_p2p_recv_meta: fx.Int64,
        cur_tok: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gw_id = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        limit = cur_tok * experts_per_token

        _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
        _r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
        _r_tok_map = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_map))
        _r_tok_off = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_off))
        _r_dest_ctr = create_buffer_resource_from_addr(_lv_unwrap(addr_dest_ctr))
        _r_disp_bar = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_bar))
        _r_total_rv = create_buffer_resource_from_addr(_lv_unwrap(addr_total_rv))
        _r_p2p_tok_off = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tok_off))
        _r_p2p_recv_num = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_recv_num))
        _r_p2p_recv_meta = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_recv_meta))
        _r_p2p_tis = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tis))

        for i in range(_to_idx(gw_id), _to_idx(limit), _to_idx(gw_num)):
            i = _to_i32(i)
            src_tok = arith.divui(i, experts_per_token)
            j = arith.remui(i, experts_per_token)
            dest_exp = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32())
            safe_lane = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, lane, j), lane, arith.constant(0)
            )
            lane_exp = buffer_load(
                _r_idx, src_tok * experts_per_token + safe_lane, vec_width=1, dtype=T.i32()
            )
            dest_pe = arith.divui(dest_exp, experts_per_rank)
            lane_pe = arith.divui(lane_exp, experts_per_rank)
            dup_per_lane = arith.select(
                arith.cmpi(arith.CmpIPredicate.eq, lane_pe, dest_pe),
                arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, lane, j), lane, arith.constant(64)
                ),
                arith.constant(64),
            )
            dup_ballot = ballot_i64(
                arith.cmpi(arith.CmpIPredicate.ult, dup_per_lane, arith.constant(64))
            )
            is_dup = arith.cmpi(
                arith.CmpIPredicate.ne, dup_ballot, arith.constant(0, type=T.i64())
            )

            from flydsl._mlir.dialects import scf as _scf_d
            from flydsl._mlir.ir import InsertionPoint as _IP

            _i32_ty = T.i32()
            _if_lane0 = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                [_i32_ty],
                has_else=True,
            )
            with _IP(_if_lane0.then_block):
                _if_nodup = _scf_d.IfOp(
                    _lv_unwrap(
                        arith.cmpi(
                            arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())
                        )
                    ),
                    [_i32_ty],
                    has_else=True,
                )
                with _IP(_if_nodup.then_block):
                    _old_tok = atomic_add_global_at(
                        buffer_load(_r_p2p_tok_off, dest_pe, vec_width=1, dtype=T.i64()),
                        arith.constant(1),
                    )
                    _scf_d.YieldOp([_lv_unwrap(_old_tok)])
                with _IP(_if_nodup.else_block):
                    _scf_d.YieldOp([_lv_unwrap(arith.constant(0))])
                _scf_d.YieldOp([_if_nodup.result])
            with _IP(_if_lane0.else_block):
                _scf_d.YieldOp([_lv_unwrap(arith.constant(0))])
            dest_tok_all = readlane(_if_lane0.result, 0)

            src_enc = (
                arith.constant(_rank, type=T.i32()) * arith.constant(_mtpr, type=T.i32())
                + src_tok
            )
            sentinel_val = arith.constant(_npes * max_recv, type=T.i32())
            dtm_val = arith.select(
                is_dup,
                sentinel_val,
                arith.addi(
                    arith.muli(dest_pe, arith.constant(max_recv, type=T.i32())),
                    dest_tok_all,
                ),
            )
            if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                buffer_store(_lv_unwrap(dtm_val), _r_tok_map, i)

            if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                    _r_tis_remote = create_buffer_resource_from_addr(
                        buffer_load(_r_p2p_tis, dest_pe, vec_width=1, dtype=T.i64())
                    )
                    buffer_store(_lv_unwrap(src_enc), _r_tis_remote, dest_tok_all)
                    ctr_addr = addr_dest_ctr + arith.zext_i64(dest_pe) * 4
                    atomic_add_global_at(ctr_addr, arith.constant(1))

            if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(experts_per_token)):
                if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                    wt_src = src_tok * experts_per_token + lane
                    wt_val = buffer_load(_r_wts, wt_src, vec_width=1, dtype=T.f32())
                    ix_val = buffer_load(_r_idx, wt_src, vec_width=1, dtype=T.i32())
                    dst_slot = dest_tok_all * experts_per_token + lane
                    remote_base = buffer_load(
                        _r_p2p_recv_meta, dest_pe, vec_width=1, dtype=T.i64()
                    )
                    row_bytes = arith.muli(dst_slot, arith.constant(32, type=T.i32()))
                    row_addr = remote_base + arith.zext_i64(row_bytes)
                    r_row = create_buffer_resource_from_addr(row_addr)
                    rank_i32 = arith.constant(_rank, type=T.i32())
                    hdr = arith.addi(
                        arith.andi(rank_i32, arith.constant(0xFF, type=T.i32())),
                        arith.shli(
                            arith.andi(lane, arith.constant(0xFF, type=T.i32())),
                            arith.constant(8, type=T.i32()),
                        ),
                    )
                    c0 = arith.constant(0, type=T.i32())
                    c1 = arith.constant(1, type=T.i32())
                    c2 = arith.constant(2, type=T.i32())
                    c3 = arith.constant(3, type=T.i32())
                    c4 = arith.constant(4, type=T.i32())
                    c5 = arith.constant(5, type=T.i32())
                    buffer_store(hdr, r_row, c0)
                    buffer_store(src_tok, r_row, c1)
                    buffer_store(ix_val, r_row, c2)
                    buffer_store(arith.bitcast(T.i32(), wt_val), r_row, c3)
                    buffer_store(src_enc, r_row, c4)
                    buffer_store(dst_slot, r_row, c5)
                    z = arith.constant(0, type=T.i32())
                    buffer_store(z, r_row, arith.constant(6, type=T.i32()))
                    buffer_store(z, r_row, arith.constant(7, type=T.i32()))

        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
            atomic_add_global_at(addr_disp_bar, arith.constant(1))

        rtn_local_off = arith.zext_i64(arith.constant(_rank)) * 4
        for dest_pe in range(_to_idx(lane), _to_idx(_npes), 64):
            dest_pe = _to_i32(dest_pe)
            if arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0)):
                mori_shmem.int32_wait_until_equals(addr_disp_bar, block_num)
                buffer_store(_lv_unwrap(arith.constant(0)), _r_disp_bar, arith.constant(0))
                nsig = buffer_load(_r_dest_ctr, dest_pe, vec_width=1, dtype=T.i32()) + 1
                rtn_remote = (
                    buffer_load(_r_p2p_recv_num, dest_pe, vec_width=1, dtype=T.i64())
                    + rtn_local_off
                )
                mori_shmem.int32_wait_until_equals(rtn_remote, 0)
                store_i32_system(rtn_remote, arith.constant(0), nsig)

        for src_pe in range(_to_idx(lane), _to_idx(_npes), 64):
            src_pe = _to_i32(src_pe)
            if arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0)):
                rtn_src = addr_recv_num + arith.zext_i64(src_pe) * 4
                sig_val = mori_shmem.int32_wait_until_greater_than(rtn_src, 0)
                recv_cnt = sig_val - 1
                store_i32_system(rtn_src, arith.constant(0), arith.constant(0))
                atomic_add_global_at(addr_total_rv, recv_cnt)
                buffer_store(_lv_unwrap(arith.constant(0)), _r_dest_ctr, src_pe)

        if arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0)):
            if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                buffer_store(_lv_unwrap(arith.constant(0)), _r_tok_off, arith.constant(0))

    return ep_metadata_dispatch_recv_meta


def make_metadata_dispatch_recv_meta_jit(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
):
    kernel = make_metadata_dispatch_recv_meta_kernel(
        rank=rank,
        npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )

    @flyc.jit
    def launch_metadata_dispatch(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_tok_off: fx.Int64,
        addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64,
        addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        addr_p2p_recv_meta: fx.Int64,
        cur_tok: fx.Int32,
        stream: Stream = Stream(None),
    ):
        kernel(
            addr_idx,
            addr_wts,
            addr_tok_off,
            addr_recv_num,
            addr_dest_ctr,
            addr_disp_bar,
            addr_tok_map,
            addr_total_rv,
            addr_p2p_tok_off,
            addr_p2p_tis,
            addr_p2p_recv_num,
            addr_p2p_recv_meta,
            cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return launch_metadata_dispatch


__all__ = [
    "make_metadata_dispatch_recv_meta_kernel",
    "make_metadata_dispatch_recv_meta_jit",
]
