# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Internode v1 LL combine kernels (main LL + EpCombineAll) — split from public module."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_HERE, "../python"), "/home/yashao/FlyDSL/python", "/home/yashao/mori/python"]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr import range_constexpr
from flydsl.expr import T
from flydsl.expr.rocdl import readlane
from flydsl.expr.vector import bitcast_i32_to_v2bf16, bitcast_v2bf16_to_i32
from flydsl.expr.buffer_ops import create_buffer_resource_from_addr, buffer_load, buffer_store
from flydsl._mlir.dialects import scf as _scf_d
from flydsl._mlir.ir import InsertionPoint as _IP
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr.extern import ExternFunction

# Match mori v1::CombineInterNodeLLTyped: __threadfence_system() before cross-node barrier atomics
# (see internode_v1.cpp ~1074–1077) so remote peers never see torn chunk-flag state.
_mori_threadfence_system = ExternFunction(
    symbol="mori_shmem_threadfence_system",
    arg_types=[],
    ret_type="int32",
)

from .dispatch_combine_internode_v1ll_kernel import (
    WARP_SZ,
    _broadcast_i64_lane0,
    _ceil_div_u32,
    load_i64_global_monotonic,
    _lv_unwrap,
    _to_idx,
    _to_i32,
    atomic_add_i32_seq_cst_one_as_ret_prev,
    load_i64_system,
    store_i64_global_system,
    store_i32_seq_cst_system,
    store_u32_relaxed_global,
)
import mori.ir.flydsl as mori_shmem


def _send_buf_slot_offset(pe: int, slot_id: int, max_tok_per_rank: int) -> int:
    return pe * max_tok_per_rank + slot_id


def make_combine_internode_v1ll_main_ll_kernel(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    block_num: int,
    rdma_block_num: int,
    warp_num_per_block: int,
    num_qp_per_pe: int,
    staging_slot_bytes: int,
):
    assert hidden_elem_size == 2 and experts_per_token == 2
    n_nodes = npes // gpu_per_node
    my_node = rank // gpu_per_node
    nbytes = hidden_dim * hidden_elem_size
    assert staging_slot_bytes >= nbytes
    max_chunk_num = (max_tok_per_rank + WARP_SZ - 1) // WARP_SZ
    warps_per_token_rdma = 4
    stride_flat = npes * max_tok_per_rank
    xgmi_warp_num = (block_num - rdma_block_num) * warp_num_per_block

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="comb_v1ll_main_smem_p11")
    off_ptrs = allocator._align(allocator.ptr, 8)
    sz_ptrs = warp_num_per_block * experts_per_token * 8
    allocator.ptr = off_ptrs + sz_ptrs

    @flyc.kernel
    def ep_combine_main_ll_v1ll(
        addr_staging: fx.Int64,
        addr_chunk_flag: fx.Int64,
        addr_node_recv: fx.Int64,
        addr_inter_dest: fx.Int64,
        addr_disp_dest: fx.Int64,
        addr_p2p_comb_inp: fx.Int64,
        addr_combine_inp_sym: fx.Int64,
        addr_inter_bar: fx.Int64,
        addr_chunk_flag_combine: fx.Int64,
        addr_cross_flag: fx.Int64,
        addr_xdb_local: fx.Int64,
        addr_p2p_xdb: fx.Int64,
        cur_rank_num_token: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gw_id = bid * warp_num_per_block + warp
        warp_num = warp_num_per_block
        slot_b_i64 = arith.constant(staging_slot_bytes, type=T.i64())

        base_ptr = allocator.get_base()
        _lds_ptrs = SmemPtr(base_ptr, off_ptrs, T.i64(), shape=(warp_num * experts_per_token,))
        _lds_ptrs.get()
        _r_p2p_ci = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_comb_inp))
        # mori uses two routing tables: (1) RDMA part of main LL loads
        # interNodeDispDestTokIdMap[globalTokenId*k+expert] → addr_inter_dest only.
        # (2) XGMI / intra blocks below use dispDestTokIdMap[tokenId*k+expert] → addr_disp_dest.
        # Do not read disp_dest_map in the RDMA loop (wrong layout / intermittent faults on ROCm).
        _r_inter = create_buffer_resource_from_addr(_lv_unwrap(addr_inter_dest))

        fx.barrier()

        _if_rd = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, bid, arith.constant(rdma_block_num))),
            [],
            has_else=True,
        )
        with _IP(_if_rd.then_block):
            rdma_warp_num = rdma_block_num * warp_num
            _r_nrecv = create_buffer_resource_from_addr(_lv_unwrap(addr_node_recv))

            for n_iter in range_constexpr(max(0, n_nodes - 1)):
                node = (my_node + 1 + n_iter) % n_nodes
                node_u64 = buffer_load(_r_nrecv, arith.constant(node), vec_width=1, dtype=T.i64())
                node_count = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ugt, node_u64, arith.constant(0, type=T.i64())),
                    arith.trunci(T.i32(), node_u64 - arith.constant(1, type=T.i64())),
                    arith.constant(0),
                )
                hidden_dim_per_warp = _ceil_div_u32(
                    arith.constant(hidden_dim), arith.constant(warps_per_token_rdma)
                )
                lim = node_count * arith.constant(warps_per_token_rdma)

                for i_idx in range(_to_idx(gw_id), _to_idx(lim), _to_idx(rdma_warp_num)):
                    i = _to_i32(i_idx)
                    token_id = arith.divui(i, arith.constant(warps_per_token_rdma))
                    k_chunk = arith.divui(token_id, arith.constant(WARP_SZ))
                    start_tok = k_chunk * arith.constant(WARP_SZ)
                    cf_addr = addr_chunk_flag + arith.zext_i64(
                        (arith.constant(node) * arith.constant(max_chunk_num) + k_chunk)
                        * arith.constant(8)
                    )
                    chunk_raw = load_i64_system(cf_addr)
                    this_n = arith.select(
                        arith.cmpi(
                            arith.CmpIPredicate.ugt, chunk_raw, arith.constant(0, type=T.i64())
                        ),
                        arith.trunci(T.i32(), chunk_raw - arith.constant(1, type=T.i64())),
                        arith.constant(0),
                    )
                    rel_tok = token_id - start_tok
                    in_part = arith.remui(i, arith.constant(warps_per_token_rdma))
                    h_off = in_part * hidden_dim_per_warp
                    _if_h = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, h_off, arith.constant(hidden_dim))),
                        [T.i32()],
                        has_else=True,
                    )
                    with _IP(_if_h.then_block):
                        rem = arith.constant(hidden_dim) - h_off
                        hs = arith.select(
                            arith.cmpi(arith.CmpIPredicate.ult, rem, hidden_dim_per_warp),
                            rem,
                            hidden_dim_per_warp,
                        )
                        _scf_d.YieldOp([hs])
                    with _IP(_if_h.else_block):
                        _scf_d.YieldOp([arith.constant(0)])
                    h_sz = _if_h.results[0]
                    hn = (h_sz * arith.constant(2) + arith.constant(3)) // arith.constant(4)
                    global_tok_id = arith.constant(_send_buf_slot_offset(node, 0, max_tok_per_rank)) + token_id

                    _if_work = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, rel_tok, this_n)),
                        [],
                        has_else=True,
                    )
                    with _IP(_if_work.then_block):
                        flat0 = buffer_load(
                            _r_inter, global_tok_id * arith.constant(2), vec_width=1, dtype=T.i32()
                        )
                        flat1 = buffer_load(
                            _r_inter,
                            global_tok_id * arith.constant(2) + arith.constant(1),
                            vec_width=1,
                            dtype=T.i32(),
                        )
                        dpe0 = arith.divui(flat0, arith.constant(stride_flat))
                        dpe1 = arith.divui(flat1, arith.constant(stride_flat))
                        dn0 = arith.divui(dpe0, arith.constant(gpu_per_node))
                        dn1 = arith.divui(dpe1, arith.constant(gpu_per_node))

                        for ej in (0, 1):
                            flat = flat0 if ej == 0 else flat1
                            dpe = dpe0 if ej == 0 else dpe1
                            dn = dn0 if ej == 0 else dn1
                            _if_lane = _scf_d.IfOp(
                                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(ej))),
                                [],
                                has_else=True,
                            )
                            with _IP(_if_lane.then_block):
                                _if_same = _scf_d.IfOp(
                                    _lv_unwrap(
                                        arith.cmpi(
                                            arith.CmpIPredicate.eq, dn, arith.constant(my_node)
                                        )
                                    ),
                                    [],
                                    has_else=True,
                                )
                                with _IP(_if_same.then_block):
                                    dpe_ok = arith.cmpi(
                                        arith.CmpIPredicate.ult, dpe, arith.constant(npes)
                                    )
                                    slot_ok = arith.cmpi(
                                        arith.CmpIPredicate.ult,
                                        arith.remui(flat, arith.constant(stride_flat)),
                                        arith.constant(max_tok_per_rank),
                                    )
                                    route_ok = arith.andi(dpe_ok, slot_ok)
                                    _if_dpe = _scf_d.IfOp(_lv_unwrap(route_ok), [], has_else=True)
                                    with _IP(_if_dpe.then_block):
                                        dl = arith.remui(flat, arith.constant(stride_flat))
                                        pe_tbl = buffer_load(
                                            _r_p2p_ci, dpe, vec_width=1, dtype=T.i64()
                                        )
                                        is_self_pe = arith.cmpi(
                                            arith.CmpIPredicate.eq, dpe, arith.constant(rank)
                                        )
                                        pe_b = arith.select(
                                            is_self_pe, addr_combine_inp_sym, pe_tbl
                                        )
                                        ptr_e = (
                                            _lv_unwrap(pe_b)
                                            + arith.muli(
                                                arith.zext_i64(dl), arith.constant(nbytes, type=T.i64())
                                            )
                                            + arith.muli(
                                                arith.zext_i64(h_off), arith.constant(2, type=T.i64())
                                            )
                                        )
                                        _lds_ptrs.store(ptr_e, [_to_idx(warp * experts_per_token + ej)])
                                        _scf_d.YieldOp([])
                                    with _IP(_if_dpe.else_block):
                                        _lds_ptrs.store(
                                            arith.constant(0, type=T.i64()),
                                            [_to_idx(warp * experts_per_token + ej)],
                                        )
                                        _scf_d.YieldOp([])
                                    _scf_d.YieldOp([])
                                with _IP(_if_same.else_block):
                                    _lds_ptrs.store(
                                        arith.constant(0, type=T.i64()),
                                        [_to_idx(warp * experts_per_token + ej)],
                                    )
                                    _scf_d.YieldOp([])
                                _scf_d.YieldOp([])
                            with _IP(_if_lane.else_block):
                                _scf_d.YieldOp([])

                        p0 = _lds_ptrs.load([_to_idx(warp * experts_per_token + 0)])
                        p1 = _lds_ptrs.load([_to_idx(warp * experts_per_token + 1)])
                        v0 = arith.cmpi(arith.CmpIPredicate.ne, p0, arith.constant(0, type=T.i64()))
                        v1 = arith.cmpi(arith.CmpIPredicate.ne, p1, arith.constant(0, type=T.i64()))
                        # Loads must not use a null resource; select masks acc only, not the load.
                        safe_p0 = arith.select(v0, p0, addr_staging)
                        safe_p1 = arith.select(v1, p1, addr_staging)
                        dst_r = create_buffer_resource_from_addr(
                            _lv_unwrap(
                                addr_staging
                                + arith.muli(arith.zext_i64(global_tok_id), slot_b_i64)
                                + arith.muli(
                                    arith.zext_i64(h_off), arith.constant(2, type=T.i64())
                                )
                            )
                        )
                        _r0_rd = create_buffer_resource_from_addr(_lv_unwrap(safe_p0))
                        _r1_rd = create_buffer_resource_from_addr(_lv_unwrap(safe_p1))
                        lane0_rd = arith.muli(_lv_unwrap(lane), arith.constant(4))
                        for ec4_rd in range(_to_idx(lane0_rd), _to_idx(hn), _to_idx(256)):
                            ec4_rd = _to_i32(ec4_rd)
                            t0_rd = buffer_load(_r0_rd, ec4_rd, vec_width=1, dtype=T.i32())
                            t1_rd = buffer_load(_r1_rd, ec4_rd, vec_width=1, dtype=T.i32())
                            a0_rd = bitcast_i32_to_v2bf16(t0_rd).extf(
                                T.VectorType.get([2], T.f32())
                            )
                            a1_rd = bitcast_i32_to_v2bf16(t1_rd).extf(
                                T.VectorType.get([2], T.f32())
                            )
                            z_rd = arith.constant_vector(0.0, T.VectorType.get([2], T.f32()))
                            acc_rd = arith.select(v0, a0_rd, z_rd) + arith.select(
                                v1, a1_rd, z_rd
                            )
                            out_rd = bitcast_v2bf16_to_i32(
                                acc_rd.truncf(T.VectorType.get([2], T.bf16()))
                            )
                            buffer_store(_lv_unwrap(out_rd), dst_r, ec4_rd)

                        _scf_d.YieldOp([])
                    with _IP(_if_work.else_block):
                        _scf_d.YieldOp([])

                    # mori CombineInterNodeLLTyped: atomic on chunk_flag_combine + putmem run for every
                    # loop index i, not only when (tokenId - startTokenIdx) < thisChunkTokenNum (see
                    # internode_v1.cpp lines 1048–1070 — inner if ends before finished/atomic).
                    cf_c = (
                        addr_chunk_flag_combine
                        + arith.zext_i64(
                            (arith.constant(node) * arith.constant(max_chunk_num) + k_chunk)
                            * arith.constant(4)
                        )
                    )
                    _if_l0a = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                        [T.i32()],
                        has_else=True,
                    )
                    with _IP(_if_l0a.then_block):
                        fin = atomic_add_i32_seq_cst_one_as_ret_prev(cf_c, arith.constant(1))
                        _scf_d.YieldOp([fin])
                    with _IP(_if_l0a.else_block):
                        _scf_d.YieldOp([arith.constant(0)])
                    finished = readlane(_if_l0a.results[0], 0)
                    need = arith.constant(warps_per_token_rdma * WARP_SZ)
                    _if_done = _scf_d.IfOp(
                        _lv_unwrap(
                            arith.cmpi(
                                arith.CmpIPredicate.uge,
                                finished + arith.constant(1),
                                need,
                            )
                        ),
                        [],
                        has_else=True,
                    )
                    with _IP(_if_done.then_block):
                        _if_z = _scf_d.IfOp(
                            _lv_unwrap(
                                arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))
                            ),
                            [],
                            has_else=True,
                        )
                        with _IP(_if_z.then_block):
                            store_i64_global_system(cf_addr, arith.constant(0, type=T.i64()))
                            store_i32_seq_cst_system(cf_c, arith.constant(0))
                            _scf_d.YieldOp([])
                        with _IP(_if_z.else_block):
                            _scf_d.YieldOp([])
                        proxy_pe = arith.constant(node * gpu_per_node + (rank % gpu_per_node))
                        qp_id = arith.remui(k_chunk, arith.constant(num_qp_per_pe))
                        src_slot = arith.constant(
                            _send_buf_slot_offset(my_node + n_nodes, 0, max_tok_per_rank)
                        ) + start_tok
                        dst_slot = arith.constant(
                            _send_buf_slot_offset(node, 0, max_tok_per_rank)
                        ) + start_tok
                        src_off = arith.muli(arith.zext_i64(src_slot), slot_b_i64)
                        dst_off_peer = arith.muli(arith.zext_i64(dst_slot), slot_b_i64)
                        put_n = arith.muli(arith.zext_i64(this_n), slot_b_i64)
                        mori_shmem.putmem_nbi_warp(
                            addr_staging + dst_off_peer,
                            addr_staging + src_off,
                            put_n,
                            proxy_pe,
                            qp_id,
                        )
                        mori_shmem.quiet_thread_pe(proxy_pe)
                        _scf_d.YieldOp([])
                    with _IP(_if_done.else_block):
                        _scf_d.YieldOp([])

            _mori_threadfence_system()
            _if_l0b = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                [T.i32(), T.i64()],
                has_else=True,
            )
            with _IP(_if_l0b.then_block):
                fw = atomic_add_i32_seq_cst_one_as_ret_prev(addr_inter_bar, arith.constant(1))
                bf = load_i64_global_monotonic(addr_cross_flag)
                _scf_d.YieldOp([fw, bf])
            with _IP(_if_l0b.else_block):
                _scf_d.YieldOp([arith.constant(0), arith.constant(0, type=T.i64())])
            finished_warp = readlane(_if_l0b.results[0], 0)
            barrier_flag = _broadcast_i64_lane0(_if_l0b.results[1])
            _if_all_rd = _scf_d.IfOp(
                _lv_unwrap(
                    arith.cmpi(
                        arith.CmpIPredicate.eq,
                        finished_warp + arith.constant(1),
                        arith.constant(rdma_block_num * warp_num),
                    )
                ),
                [],
                has_else=True,
            )
            with _IP(_if_all_rd.then_block):
                for li in range_constexpr(n_nodes):
                    _if_li = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li))),
                        [],
                        has_else=True,
                    )
                    with _IP(_if_li.then_block):
                        store_i64_global_system(
                            addr_node_recv + arith.zext_i64(arith.constant(li * 8)),
                            arith.constant(0, type=T.i64()),
                        )
                        _scf_d.YieldOp([])
                    with _IP(_if_li.else_block):
                        _scf_d.YieldOp([])
                for li2 in range_constexpr(n_nodes):
                    amo_ok = arith.andi(
                        arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li2)),
                        arith.cmpi(
                            arith.CmpIPredicate.ne,
                            arith.constant(li2),
                            arith.constant(my_node),
                        ),
                    )
                    _if_am = _scf_d.IfOp(_lv_unwrap(amo_ok), [], has_else=True)
                    with _IP(_if_am.then_block):
                        proxy_pe2 = arith.constant(li2 * gpu_per_node + (rank % gpu_per_node))
                        for qi in range_constexpr(num_qp_per_pe):
                            xb = addr_xdb_local + arith.zext_i64(arith.constant(rank * 8))
                            mori_shmem.uint64_atomic_add_thread(
                                xb,
                                arith.constant(1, type=T.i64()),
                                proxy_pe2,
                                arith.constant(qi),
                            )
                        # mori CombineInterNodeLLTyped: __threadfence_system after xdb AMOs, before
                        # interNodeBlocksBarrier reset + peer wait (internode_v1.cpp ~1096–1107).
                        _mori_threadfence_system()
                        _scf_d.YieldOp([])
                    with _IP(_if_am.else_block):
                        _scf_d.YieldOp([])
                _if_clr = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                    [],
                    has_else=True,
                )
                with _IP(_if_clr.then_block):
                    store_i32_seq_cst_system(addr_inter_bar, arith.constant(0, type=T.i32()))
                    _scf_d.YieldOp([])
                with _IP(_if_clr.else_block):
                    _scf_d.YieldOp([])
                for li3 in range_constexpr(n_nodes):
                    wait_ok = arith.andi(
                        arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li3)),
                        arith.cmpi(
                            arith.CmpIPredicate.ne,
                            arith.constant(li3),
                            arith.constant(my_node),
                        ),
                    )
                    _if_w = _scf_d.IfOp(_lv_unwrap(wait_ok), [], has_else=True)
                    with _IP(_if_w.then_block):
                        proxy_pe3 = arith.constant(li3 * gpu_per_node + (rank % gpu_per_node))
                        peer_ptr = addr_xdb_local + arith.muli(
                            arith.zext_i64(proxy_pe3), arith.constant(8, type=T.i64())
                        )
                        tgt = barrier_flag * arith.constant(num_qp_per_pe, type=T.i64())
                        mori_shmem.uint64_wait_until_equals(peer_ptr, tgt)
                        _scf_d.YieldOp([])
                    with _IP(_if_w.else_block):
                        _scf_d.YieldOp([])
                _scf_d.YieldOp([])
            with _IP(_if_all_rd.else_block):
                _scf_d.YieldOp([])

            _scf_d.YieldOp([])

        with _IP(_if_rd.else_block):
            _if_ztok = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, cur_rank_num_token, arith.constant(0))),
                [],
                has_else=True,
            )
            with _IP(_if_ztok.then_block):
                _scf_d.YieldOp([])
            with _IP(_if_ztok.else_block):
                block_off = arith.constant(rdma_block_num)
                mw_pi = _ceil_div_u32(arith.constant(xgmi_warp_num), cur_rank_num_token)
                dim_pw = _ceil_div_u32(arith.constant(hidden_dim), mw_pi)
                lim_i = cur_rank_num_token * mw_pi
                inner_gw = gw_id - block_off * warp_num
                for i_idx in range(_to_idx(inner_gw), _to_idx(lim_i), _to_idx(xgmi_warp_num)):
                    i = _to_i32(i_idx)
                    token_id = arith.divui(i, mw_pi)
                    in_part = arith.remui(i, mw_pi)
                    h_off = in_part * dim_pw
                    _if_h2 = _scf_d.IfOp(
                        _lv_unwrap(
                            arith.cmpi(arith.CmpIPredicate.ult, h_off, arith.constant(hidden_dim))
                        ),
                        [T.i32()],
                        has_else=True,
                    )
                    with _IP(_if_h2.then_block):
                        rem = arith.constant(hidden_dim) - h_off
                        hs = arith.select(
                            arith.cmpi(arith.CmpIPredicate.ult, rem, dim_pw), rem, dim_pw
                        )
                        _scf_d.YieldOp([hs])
                    with _IP(_if_h2.else_block):
                        _scf_d.YieldOp([arith.constant(0)])
                    h_sz = _if_h2.results[0]
                    hn = (h_sz * arith.constant(2) + arith.constant(3)) // arith.constant(4)
                    slot0 = _send_buf_slot_offset(n_nodes + my_node, 0, max_tok_per_rank)
                    dst_base = (
                        addr_staging
                        + arith.constant(slot0 * staging_slot_bytes, type=T.i64())
                        + arith.muli(arith.zext_i64(token_id), slot_b_i64)
                        + arith.muli(arith.zext_i64(h_off), arith.constant(2, type=T.i64()))
                    )
                    _r_disp = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_dest))
                    did0 = buffer_load(
                        _r_disp, token_id * arith.constant(2), vec_width=1, dtype=T.i32()
                    )
                    did1 = buffer_load(
                        _r_disp,
                        token_id * arith.constant(2) + arith.constant(1),
                        vec_width=1,
                        dtype=T.i32(),
                    )
                    dpe0 = arith.divui(did0, arith.constant(stride_flat))
                    dpe1 = arith.divui(did1, arith.constant(stride_flat))
                    dn0 = arith.divui(dpe0, arith.constant(gpu_per_node))
                    dn1 = arith.divui(dpe1, arith.constant(gpu_per_node))

                    for ej in (0, 1):
                        did = did0 if ej == 0 else did1
                        dpe = dpe0 if ej == 0 else dpe1
                        dn = dn0 if ej == 0 else dn1
                        _if_lane = _scf_d.IfOp(
                            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(ej))),
                            [],
                            has_else=True,
                        )
                        with _IP(_if_lane.then_block):
                            _if_sn = _scf_d.IfOp(
                                _lv_unwrap(
                                    arith.cmpi(
                                        arith.CmpIPredicate.eq, dn, arith.constant(my_node)
                                    )
                                ),
                                [],
                                has_else=True,
                            )
                            with _IP(_if_sn.then_block):
                                dpe_ok_xg = arith.cmpi(
                                    arith.CmpIPredicate.ult, dpe, arith.constant(npes)
                                )
                                slot_ok_xg = arith.cmpi(
                                    arith.CmpIPredicate.ult,
                                    arith.remui(did, arith.constant(stride_flat)),
                                    arith.constant(max_tok_per_rank),
                                )
                                route_ok_xg = arith.andi(dpe_ok_xg, slot_ok_xg)
                                _if_dpex = _scf_d.IfOp(_lv_unwrap(route_ok_xg), [], has_else=True)
                                with _IP(_if_dpex.then_block):
                                    dl = arith.remui(did, arith.constant(stride_flat))
                                    pe_tbl = buffer_load(
                                        _r_p2p_ci, dpe, vec_width=1, dtype=T.i64()
                                    )
                                    is_self_pe_xg = arith.cmpi(
                                        arith.CmpIPredicate.eq, dpe, arith.constant(rank)
                                    )
                                    pe_b = arith.select(
                                        is_self_pe_xg, addr_combine_inp_sym, pe_tbl
                                    )
                                    ptr_e = (
                                        _lv_unwrap(pe_b)
                                        + arith.muli(
                                            arith.zext_i64(dl), arith.constant(nbytes, type=T.i64())
                                        )
                                        + arith.muli(
                                            arith.zext_i64(h_off), arith.constant(2, type=T.i64())
                                        )
                                    )
                                    _lds_ptrs.store(ptr_e, [_to_idx(warp * experts_per_token + ej)])
                                    _scf_d.YieldOp([])
                                with _IP(_if_dpex.else_block):
                                    _lds_ptrs.store(
                                        arith.constant(0, type=T.i64()),
                                        [_to_idx(warp * experts_per_token + ej)],
                                    )
                                    _scf_d.YieldOp([])
                                _scf_d.YieldOp([])
                            with _IP(_if_sn.else_block):
                                _lds_ptrs.store(
                                    arith.constant(0, type=T.i64()),
                                    [_to_idx(warp * experts_per_token + ej)],
                                )
                                _scf_d.YieldOp([])
                            _scf_d.YieldOp([])
                        with _IP(_if_lane.else_block):
                            _scf_d.YieldOp([])

                    p0 = _lds_ptrs.load([_to_idx(warp * experts_per_token + 0)])
                    p1 = _lds_ptrs.load([_to_idx(warp * experts_per_token + 1)])
                    v0 = arith.cmpi(arith.CmpIPredicate.ne, p0, arith.constant(0, type=T.i64()))
                    v1 = arith.cmpi(arith.CmpIPredicate.ne, p1, arith.constant(0, type=T.i64()))
                    safe_p0_xg = arith.select(v0, p0, dst_base)
                    safe_p1_xg = arith.select(v1, p1, dst_base)
                    dst_r = create_buffer_resource_from_addr(_lv_unwrap(dst_base))
                    _r0_xg = create_buffer_resource_from_addr(_lv_unwrap(safe_p0_xg))
                    _r1_xg = create_buffer_resource_from_addr(_lv_unwrap(safe_p1_xg))
                    lane0_xg = arith.muli(_lv_unwrap(lane), arith.constant(4))
                    for ec4_xg in range(_to_idx(lane0_xg), _to_idx(hn), _to_idx(256)):
                        ec4_xg = _to_i32(ec4_xg)
                        t0_xg = buffer_load(_r0_xg, ec4_xg, vec_width=1, dtype=T.i32())
                        t1_xg = buffer_load(_r1_xg, ec4_xg, vec_width=1, dtype=T.i32())
                        a0_xg = bitcast_i32_to_v2bf16(t0_xg).extf(
                            T.VectorType.get([2], T.f32())
                        )
                        a1_xg = bitcast_i32_to_v2bf16(t1_xg).extf(
                            T.VectorType.get([2], T.f32())
                        )
                        z_xg = arith.constant_vector(0.0, T.VectorType.get([2], T.f32()))
                        acc_xg = arith.select(v0, a0_xg, z_xg) + arith.select(
                            v1, a1_xg, z_xg
                        )
                        out_xg = bitcast_v2bf16_to_i32(
                            acc_xg.truncf(T.VectorType.get([2], T.bf16()))
                        )
                        buffer_store(_lv_unwrap(out_xg), dst_r, ec4_xg)

                _scf_d.YieldOp([])
            _scf_d.YieldOp([])

    return ep_combine_main_ll_v1ll, allocator


def make_combine_internode_v1ll_all_kernel(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    mp_count: int,
    warp_num_per_block: int,
    staging_slot_bytes: int,
):
    assert experts_per_token == 2 and hidden_elem_size == 2
    n_nodes = npes // gpu_per_node
    my_node = rank // gpu_per_node
    nbytes = hidden_dim * hidden_elem_size
    assert staging_slot_bytes >= nbytes
    global_warp_num = mp_count * warp_num_per_block

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="comb_v1ll_all_smem_p2")
    off_ptrs = allocator._align(allocator.ptr, 8)
    sz_ptrs = warp_num_per_block * n_nodes * 8
    allocator.ptr = off_ptrs + sz_ptrs

    @flyc.kernel
    def ep_combine_all_v1ll(
        addr_staging: fx.Int64,
        addr_inter_send: fx.Int64,
        addr_token_indices: fx.Int64,
        addr_combine_out: fx.Int64,
        addr_total_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        cur_rank_num_token: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gw_id = bid * warp_num_per_block + warp
        warp_num = warp_num_per_block

        _if_g0 = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0))),
            [],
            has_else=True,
        )
        with _IP(_if_g0.then_block):
            _if_l0 = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                [],
                has_else=True,
            )
            with _IP(_if_l0.then_block):
                store_i32_seq_cst_system(addr_total_recv, arith.constant(0))
                _scf_d.YieldOp([])
            with _IP(_if_l0.else_block):
                _scf_d.YieldOp([])
            for li in range_constexpr(n_nodes):
                _if_li = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li))),
                    [],
                    has_else=True,
                )
                with _IP(_if_li.then_block):
                    store_i32_seq_cst_system(
                        addr_block_flag + arith.zext_i64(arith.constant(li * 4)),
                        arith.constant(0),
                    )
                    _scf_d.YieldOp([])
                with _IP(_if_li.else_block):
                    _scf_d.YieldOp([])
            _scf_d.YieldOp([])
        with _IP(_if_g0.else_block):
            _scf_d.YieldOp([])

        _if_ret = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, cur_rank_num_token, arith.constant(0))),
            [],
            has_else=True,
        )
        with _IP(_if_ret.then_block):
            _scf_d.YieldOp([])
        with _IP(_if_ret.else_block):
            base_ptr = allocator.get_base()
            _lds_nptr = SmemPtr(base_ptr, off_ptrs, T.i64(), shape=(warp_num * n_nodes,))
            _lds_nptr.get()
            mw_pi = _ceil_div_u32(arith.constant(global_warp_num), cur_rank_num_token)
            dim_pw = _ceil_div_u32(arith.constant(hidden_dim), mw_pi)
            lim_i = cur_rank_num_token * mw_pi

            for i_idx in range(_to_idx(gw_id), _to_idx(lim_i), _to_idx(global_warp_num)):
                i = _to_i32(i_idx)
                token_id = arith.divui(i, mw_pi)
                in_part = arith.remui(i, mw_pi)
                h_off = in_part * dim_pw
                _if_h = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, h_off, arith.constant(hidden_dim))),
                    [T.i32()],
                    has_else=True,
                )
                with _IP(_if_h.then_block):
                    rem = arith.constant(hidden_dim) - h_off
                    hs = arith.select(
                        arith.cmpi(arith.CmpIPredicate.ult, rem, dim_pw), rem, dim_pw
                    )
                    _scf_d.YieldOp([hs])
                with _IP(_if_h.else_block):
                    _scf_d.YieldOp([arith.constant(0)])
                h_sz = _if_h.results[0]
                hn = (h_sz * arith.constant(2) + arith.constant(3)) // arith.constant(4)

                _r_send = create_buffer_resource_from_addr(_lv_unwrap(addr_inter_send))

                for nn in range_constexpr(n_nodes):
                    _if_ln = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(nn))),
                        [],
                        has_else=True,
                    )
                    with _IP(_if_ln.then_block):
                        sm = arith.muli(arith.constant(n_nodes), token_id) + arith.constant(nn)
                        remote_tok = buffer_load(_r_send, sm, vec_width=1, dtype=T.i32())
                        mapped = arith.select(
                            arith.cmpi(
                                arith.CmpIPredicate.eq, arith.constant(nn), arith.constant(my_node)
                            ),
                            token_id,
                            remote_tok,
                        )
                        # Clamp (ROCm): garbage/torn remote_tok or races must not OOB staging rows.
                        map_ok = arith.cmpi(
                            arith.CmpIPredicate.ult, mapped, arith.constant(max_tok_per_rank)
                        )
                        mapped_c = arith.select(map_ok, mapped, arith.constant(0))
                        slot = arith.constant(
                            _send_buf_slot_offset(n_nodes + nn, 0, max_tok_per_rank)
                        ) + mapped_c
                        row_b = arith.muli(
                            arith.zext_i64(slot),
                            arith.constant(staging_slot_bytes, type=T.i64()),
                        )
                        ptr_b = (
                            addr_staging
                            + row_b
                            + arith.muli(arith.zext_i64(h_off), arith.constant(2, type=T.i64()))
                        )
                        _lds_nptr.store(ptr_b, [_to_idx(warp * n_nodes + nn)])
                        _scf_d.YieldOp([])
                    with _IP(_if_ln.else_block):
                        _scf_d.YieldOp([])

                fx.barrier()

                p0 = _lds_nptr.load([_to_idx(warp * n_nodes + 0)])
                # n_nodes==1: only nn=0 fills LDS; warp*n_nodes+1 aliases another warp / is unset → fault.
                # Use null second pointer so v1 is false and only the first expert lane contributes.
                if n_nodes == 1:
                    p1 = arith.constant(0, type=T.i64())
                else:
                    p1 = _lds_nptr.load([_to_idx(warp * n_nodes + 1)])
                v0 = arith.cmpi(arith.CmpIPredicate.ne, p0, arith.constant(0, type=T.i64()))
                v1 = arith.cmpi(arith.CmpIPredicate.ne, p1, arith.constant(0, type=T.i64()))
                out_b = (
                    addr_combine_out
                    + arith.muli(arith.zext_i64(token_id), arith.constant(nbytes, type=T.i64()))
                    + arith.muli(arith.zext_i64(h_off), arith.constant(2, type=T.i64()))
                )
                safe_p0_al = arith.select(v0, p0, addr_staging)
                safe_p1_al = arith.select(v1, p1, addr_staging)
                out_r = create_buffer_resource_from_addr(_lv_unwrap(out_b))
                _r0_al = create_buffer_resource_from_addr(_lv_unwrap(safe_p0_al))
                _r1_al = create_buffer_resource_from_addr(_lv_unwrap(safe_p1_al))
                lane0_al = arith.muli(_lv_unwrap(lane), arith.constant(4))
                for ec4_al in range(_to_idx(lane0_al), _to_idx(hn), _to_idx(256)):
                    ec4_al = _to_i32(ec4_al)
                    t0_al = buffer_load(_r0_al, ec4_al, vec_width=1, dtype=T.i32())
                    t1_al = buffer_load(_r1_al, ec4_al, vec_width=1, dtype=T.i32())
                    a0_al = bitcast_i32_to_v2bf16(t0_al).extf(T.VectorType.get([2], T.f32()))
                    a1_al = bitcast_i32_to_v2bf16(t1_al).extf(T.VectorType.get([2], T.f32()))
                    z_al = arith.constant_vector(0.0, T.VectorType.get([2], T.f32()))
                    acc_al = arith.select(v0, a0_al, z_al) + arith.select(v1, a1_al, z_al)
                    out_al = bitcast_v2bf16_to_i32(
                        acc_al.truncf(T.VectorType.get([2], T.bf16()))
                    )
                    buffer_store(_lv_unwrap(out_al), out_r, ec4_al)

            _scf_d.YieldOp([])

    return ep_combine_all_v1ll, allocator
