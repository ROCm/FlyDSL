# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL internode v1 LL combine (bf16, no weights). See mori ``launch.cpp`` InterNodeV1LL order.

``experts_per_token=2``, ``gpu_per_node=1``. Flat token: ``stride = world_size * max_tok_per_rank``.
"""

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
from flydsl.expr.typing import Stream
import torch

import mori.ir.flydsl as mori_shmem

from flydsl.expr import T
from flydsl.expr.rocdl import readlane
from flydsl.expr.buffer_ops import create_buffer_resource_from_addr, buffer_load, buffer_store
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import scf as _scf_d
from flydsl._mlir.ir import InsertionPoint as _IP
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .dispatch_combine_internode_v1ll_kernel import (
    WARP_SZ,
    _broadcast_i64_lane0,
    _ceil_div_u32,
    load_i64_global_monotonic,
    _lv_unwrap,
    _to_idx,
    _to_i32,
    atomic_add_i32_global_ret_prev,
    load_i64_system,
    store_i64_global_system,
    store_i32_seq_cst_system,
    store_u32_relaxed_global,
)


def _send_buf_slot_offset(pe: int, slot_id: int, max_tok_per_rank: int) -> int:
    return pe * max_tok_per_rank + slot_id


def make_combine_internode_v1ll_sync_kernel(
    *,
    sync_grid_blocks: int,
    warp_num_per_block: int,
    hidden_dim: int,
    hidden_elem_size: int,
):
    """mori ``EpCombineSync``: ``gridDim.x`` = ``mp``; ``tokenPerBlock = ceil(totalRecv / gridDim.x)``."""
    n_i32 = (hidden_dim * hidden_elem_size) // 4
    nbytes = hidden_dim * hidden_elem_size

    @flyc.kernel
    def ep_combine_sync_v1ll(
        addr_inp_tok: fx.Int64,
        addr_combine_inp: fx.Int64,
        addr_total_recv: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gw_id = bid * warp_num_per_block + warp
        warp_num = warp_num_per_block
        _r_tot = create_buffer_resource_from_addr(_lv_unwrap(addr_total_recv))
        total_recv = buffer_load(_r_tot, arith.constant(0), vec_width=1, dtype=T.i32())
        token_per_block = _ceil_div_u32(total_recv, arith.constant(sync_grid_blocks))
        start_tok = bid * token_per_block
        end_tok = arith.select(
            arith.cmpi(arith.CmpIPredicate.ult, start_tok + token_per_block, total_recv),
            start_tok + token_per_block,
            total_recv,
        )
        nb_i64 = arith.constant(nbytes, type=T.i64())
        for token_id in range(_to_idx(start_tok + warp), _to_idx(end_tok), _to_idx(warp_num)):
            token_id = _to_i32(token_id)
            src_b = addr_inp_tok + arith.muli(arith.zext_i64(token_id), nb_i64)
            dst_b = addr_combine_inp + arith.muli(arith.zext_i64(token_id), nb_i64)
            _rs = create_buffer_resource_from_addr(_lv_unwrap(src_b))
            _rd = create_buffer_resource_from_addr(_lv_unwrap(dst_b))
            # mori v1::CombineSync uses WarpCopy over full hidden dim; step by 64 lanes like mori warp
            ni = arith.constant(n_i32, type=T.i32())
            for ec4 in range(_to_idx(lane), _to_idx(ni), _to_idx(arith.constant(64))):
                ec4 = _to_i32(ec4)
                v0 = buffer_load(_rs, ec4, vec_width=1, dtype=T.i32())
                buffer_store(v0, _rd, ec4)

    return ep_combine_sync_v1ll


def make_combine_internode_v1ll_sync_barrier_kernel(*, rank: int, gpu_per_node: int):
    my_node = rank // gpu_per_node

    @flyc.kernel
    def ep_combine_sync_barrier_v1ll(
        addr_cross_flag: fx.Int64,
        addr_xdb_local: fx.Int64,
        addr_p2p_xdb: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lane = tid & 63
        _if_l0 = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
            [T.i64()],
            has_else=True,
        )
        with _IP(_if_l0.then_block):
            _scf_d.YieldOp([load_i64_global_monotonic(addr_cross_flag)])
        with _IP(_if_l0.else_block):
            _scf_d.YieldOp([arith.constant(0, type=T.i64())])
        barrier_flag = _broadcast_i64_lane0(_if_l0.results[0])
        _r_p2p = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_xdb))
        eight_i64 = arith.constant(8, type=T.i64())
        # mori EpCombineSyncBarrier_body: if (laneId < gpuPerNode) { ... }
        _if_ln = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(gpu_per_node))),
            [],
            has_else=True,
        )
        with _IP(_if_ln.then_block):
            dest_pe = arith.constant(my_node * gpu_per_node) + lane
            # mori: GetAs<uint64_t*>(destPe)+rank — shmem_ptr_p2p(xdb, rank, rank) may be 0.
            remote_tbl = buffer_load(_r_p2p, dest_pe, vec_width=1, dtype=T.i64())
            remote_base = arith.select(
                _lv_unwrap(
                    arith.cmpi(arith.CmpIPredicate.eq, dest_pe, arith.constant(rank))
                ),
                _lv_unwrap(addr_xdb_local),
                remote_tbl,
            )
            remote_slot = remote_base + arith.muli(
                arith.zext_i64(arith.constant(rank)), eight_i64
            )
            store_i64_global_system(remote_slot, barrier_flag)
            peer_slot = addr_xdb_local + arith.muli(arith.zext_i64(dest_pe), eight_i64)
            mori_shmem.uint64_wait_until_equals(peer_slot, barrier_flag)
            _scf_d.YieldOp([])
        with _IP(_if_ln.else_block):
            _scf_d.YieldOp([])

    return ep_combine_sync_barrier_v1ll


from ._internode_v1ll_combine_impl import (
    make_combine_internode_v1ll_all_kernel,
    make_combine_internode_v1ll_main_ll_kernel,
)

def make_combine_internode_v1ll_sync_jit(
    *,
    mp_count: int,
    warp_num_per_block: int,
    hidden_dim: int,
    data_type,
):
    elem = torch.tensor([], dtype=data_type).element_size()
    kernel = make_combine_internode_v1ll_sync_kernel(
        sync_grid_blocks=mp_count,
        warp_num_per_block=warp_num_per_block,
        hidden_dim=hidden_dim,
        hidden_elem_size=elem,
    )
    _mp, _wpb = mp_count, warp_num_per_block

    @flyc.jit
    def launch(
        addr_inp_tok: fx.Int64,
        addr_combine_inp: fx.Int64,
        addr_total_recv: fx.Int64,
        stream: Stream = Stream(None),
    ):
        _ = (_mp, _wpb)
        kernel(addr_inp_tok, addr_combine_inp, addr_total_recv).launch(
            grid=(_mp, 1, 1), block=(_wpb * 64, 1, 1), stream=stream
        )

    return launch


def make_combine_internode_v1ll_sync_barrier_jit(*, rank: int, gpu_per_node: int):
    kernel = make_combine_internode_v1ll_sync_barrier_kernel(rank=rank, gpu_per_node=gpu_per_node)

    @flyc.jit
    def launch(
        addr_cross_flag: fx.Int64,
        addr_xdb_local: fx.Int64,
        addr_p2p_xdb: fx.Int64,
        stream: Stream = Stream(None),
    ):
        kernel(addr_cross_flag, addr_xdb_local, addr_p2p_xdb).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


def make_combine_internode_v1ll_main_ll_jit(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    block_num: int,
    rdma_block_num: int,
    warp_num_per_block: int,
    num_qp_per_pe: int,
    staging_slot_bytes: int,
    data_type,
):
    elem = torch.tensor([], dtype=data_type).element_size()
    kernel, _smem_main = make_combine_internode_v1ll_main_ll_kernel(
        rank=rank,
        npes=npes,
        gpu_per_node=gpu_per_node,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=elem,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_num_per_block=warp_num_per_block,
        num_qp_per_pe=num_qp_per_pe,
        staging_slot_bytes=staging_slot_bytes,
    )
    _bn, _wpb = block_num, warp_num_per_block

    @flyc.jit
    def launch(
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
        stream: Stream = Stream(None),
    ):
        _ = (_bn, _wpb)
        _smem_main.finalized = False
        _ctx = CompilationContext.get_current()
        with _mlir_ir.InsertionPoint(_ctx.gpu_module_body):
            _smem_main.finalize()
        kernel(
            addr_staging,
            addr_chunk_flag,
            addr_node_recv,
            addr_inter_dest,
            addr_disp_dest,
            addr_p2p_comb_inp,
            addr_combine_inp_sym,
            addr_inter_bar,
            addr_chunk_flag_combine,
            addr_cross_flag,
            addr_xdb_local,
            addr_p2p_xdb,
            cur_rank_num_token,
        ).launch(grid=(_bn, 1, 1), block=(_wpb * 64, 1, 1), stream=stream)

    launch.compile_hints = {
        **launch.compile_hints,
        "internode_v1ll_jit_rev": 14,
    }
    return launch


def make_combine_internode_v1ll_all_jit(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    mp_count: int,
    warp_num_per_block: int,
    staging_slot_bytes: int,
    data_type,
):
    elem = torch.tensor([], dtype=data_type).element_size()
    kernel, _smem_all = make_combine_internode_v1ll_all_kernel(
        rank=rank,
        npes=npes,
        gpu_per_node=gpu_per_node,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=elem,
        max_tok_per_rank=max_tok_per_rank,
        mp_count=mp_count,
        warp_num_per_block=warp_num_per_block,
        staging_slot_bytes=staging_slot_bytes,
    )
    _mp, _wpb = mp_count, warp_num_per_block

    @flyc.jit
    def launch(
        addr_staging: fx.Int64,
        addr_inter_send: fx.Int64,
        addr_token_indices: fx.Int64,
        addr_combine_out: fx.Int64,
        addr_total_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        cur_rank_num_token: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_mp, _wpb)
        _smem_all.finalized = False
        _ctx = CompilationContext.get_current()
        with _mlir_ir.InsertionPoint(_ctx.gpu_module_body):
            _smem_all.finalize()
        kernel(
            addr_staging,
            addr_inter_send,
            addr_token_indices,
            addr_combine_out,
            addr_total_recv,
            addr_block_flag,
            cur_rank_num_token,
        ).launch(grid=(_mp, 1, 1), block=(_wpb * 64, 1, 1), stream=stream)

    launch.compile_hints = {
        **launch.compile_hints,
        "internode_v1ll_jit_rev": 14,
    }
    return launch
