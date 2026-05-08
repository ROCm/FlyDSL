# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL internode v1 low-latency EP dispatch.

Kernel A (``ep_dispatch_copy_to_staging_v1ll``): pack tokens into symmetric staging (``EpDispatchCopyToStaging``).

Kernel B (``ep_dispatch_internode_v1ll_main``): ``DispatchInterNodeLLSend`` → ``DispatchInterNodeLLRecv`` on
RDMA blocks, ``DispatchIntraNode`` on XGMI blocks, then ``DispatchSync``, optional Phase 4
``ConvertDispatchOutput`` when ``enable_std_moe`` (mirrors intranode dispatch Phase 4).

Uses mori shmem device externs only (``putmem_nbi_signal_thread``, atomics, waits, ``quiet_thread_pe``,
``mori_shmem_threadfence_system`` in DispatchSync — requires mori ``shmem_device_api_wrapper.cpp`` with that symbol).
Addr-based P2P mori ops expect **local symmetric heap VAs**; the device subtracts ``heapBase`` and adds
``peerPtrs[pe]`` (see ``ShmemPutMemNbiSignalThreadKernel`` / ``ShmemAtomicSizeNonFetchThreadKernel`` in mori).
Do not pass ``shmem_ptr_p2p`` results there—they are already ``peerPtrs[destPe] + offset`` and would double-map.
(FlyDSL uses raw buffer loads with ``shmem_ptr_p2p`` VAs elsewhere where no such re-mapping applies.)
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
from flydsl.expr.extern import ExternFunction

# mori_shmem_threadfence_system lives in shmem_device_api_wrapper.cpp (may be newer than the
# installed mori.ir.flydsl auto-ops); keep a local ExternFunction so FlyDSL kernels link it.
_mori_threadfence_system = ExternFunction(
    symbol="mori_shmem_threadfence_system",
    arg_types=[],
    ret_type="int32",
)

from flydsl.expr import T
from flydsl.expr.rocdl import readlane
from flydsl.expr.buffer_ops import create_buffer_resource_from_addr, buffer_load, buffer_store
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import scf as _scf_d
from flydsl._mlir.ir import InsertionPoint as _IP
from flydsl._mlir.ir import IntegerAttr as _IntAttr, IntegerType as _IntTy

# mori::core::atomicType — signal op on chunk flags uses AMO_ADD (see mori internode_v1.cpp).
AMO_ADD = 4
WARP_SZ = 64


def _broadcast_i64_lane0(bf_raw):
    """``readlane`` is i32-only in FlyDSL; split i64 for warp broadcast from lane 0."""
    bf_lo = arith.trunci(T.i32(), bf_raw)
    bf_hi = arith.trunci(T.i32(), arith.shrui(bf_raw, arith.constant(32, type=T.i64())))
    blo = readlane(bf_lo, 0)
    bhi = readlane(bf_hi, 0)
    return arith.addi(
        arith.zext_i64(blo),
        arith.shli(arith.zext_i64(bhi), arith.constant(32, type=T.i64())),
    )


def _lv_unwrap(v):
    if isinstance(v, _ir.Value):
        return v
    if hasattr(v, "__fly_values__"):
        vals = v.__fly_values__()
        if len(vals) == 1:
            return vals[0]
        raise ValueError(f"Expected 1 ir.Value, got {len(vals)}")
    if isinstance(v, int):
        _i32 = _IntTy.get_signless(32)
        return _llvm_d.ConstantOp(_i32, _IntAttr.get(_i32, v)).result
    raise TypeError(f"Cannot convert {type(v).__name__} to ir.Value")


def _to_ptr_global(v):
    return _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), _lv_unwrap(v)).result


def store_i32_at_index(addr_i64, idx_i32, val_i32):
    """Store i32 to ``addr + idx * 4`` with system monotonic."""
    base = _lv_unwrap(addr_i64)
    off = _lv_unwrap(idx_i32)
    val_ = _lv_unwrap(val_i32)
    _i64 = _IntTy.get_signless(64)
    _i32 = _IntTy.get_signless(32)
    _nuw = _ir.Attribute.parse("#llvm.overflow<none>")
    off64 = _llvm_d.ZExtOp(_i64, off).res
    byte_off = _llvm_d.MulOp(
        off64, _llvm_d.ConstantOp(_i64, _IntAttr.get(_i64, 4)).result, _nuw).result
    addr = _llvm_d.AddOp(base, byte_off, _nuw).result
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    _llvm_d.StoreOp(val_, gptr, alignment=4,
                    ordering=_llvm_d.AtomicOrdering.monotonic,
                    syncscope="one-as")


def store_u32_relaxed_global(addr_i64, val_u32):
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(_lv_unwrap(val_u32), gptr, alignment=4,
                    ordering=_llvm_d.AtomicOrdering.monotonic)


def load_i64_system(addr_i64):
    gptr = _to_ptr_global(addr_i64)
    _i64 = _IntTy.get_signless(64)
    return _llvm_d.LoadOp(_i64, gptr, alignment=8,
                          ordering=_llvm_d.AtomicOrdering.monotonic,
                          syncscope="one-as").result


def load_i64_global_monotonic(addr_i64):
    """Global i64 load without system scope (matches ``atomic_add_i64_global_ret_prev`` on same VA).

    Use for ``cross_dev_flag`` / device-local buffers updated only with device-scoped atomics.
    ``load_i64_system`` (one-as) on typical HIP device allocations can fault on ROCm.
    """
    gptr = _to_ptr_global(addr_i64)
    _i64 = _IntTy.get_signless(64)
    return _llvm_d.LoadOp(_i64, gptr, alignment=8,
                          ordering=_llvm_d.AtomicOrdering.monotonic).result


def load_i32_seq_cst_system(addr_i64):
    gptr = _to_ptr_global(addr_i64)
    _i32 = _IntTy.get_signless(32)
    return _llvm_d.LoadOp(_i32, gptr, alignment=4,
                          ordering=_llvm_d.AtomicOrdering.seq_cst,
                          syncscope="one-as").result


def atomic_add_i32_global_ret_prev(addr_i64, val_i32):
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, _lv_unwrap(val_i32),
        _llvm_d.AtomicOrdering.monotonic).res


def atomic_add_i32_seq_cst_one_as_ret_prev(addr_i64, val_i32):
    """DispatchSync grid barrier: seq_cst + one-as matches cross-CU visibility needs on ROCm."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        ptr,
        _lv_unwrap(val_i32),
        _llvm_d.AtomicOrdering.seq_cst,
        syncscope="one-as",
    ).res


def atomic_add_i64_global_ret_prev(addr_i64, val_i64):
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, _lv_unwrap(val_i64),
        _llvm_d.AtomicOrdering.monotonic).res


def store_i64_global_system(addr_i64, val_i64):
    """System-scope monotonic i64 store (global)."""
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(_lv_unwrap(val_i64), gptr, alignment=8,
                    ordering=_llvm_d.AtomicOrdering.monotonic,
                    syncscope="one-as")


def atomic_add_global_at(addr_i64, val):
    """Global atomic fetch-and-add (monotonic). Returns old value."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, _lv_unwrap(val),
        _llvm_d.AtomicOrdering.monotonic).res


def store_i32_seq_cst_system(addr_i64, val_i32):
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(_lv_unwrap(val_i32), gptr, alignment=4,
                    ordering=_llvm_d.AtomicOrdering.seq_cst,
                    syncscope="one-as")


def _to_idx(v):
    if isinstance(v, int):
        return arith.index(v)
    return arith.index_cast(T.index(), v)


def _to_i32(v):
    return arith.index_cast(T.i32(), v)


def _ceil_div_u32(num, den):
    return arith.divui(num + den - arith.constant(1), den)


def _emit_poll_chunk_or_node(cf_addr_i64, nf_addr_i64, start_tok_i, lane):
    """Lane 0 spins; returns i64 chunk_raw (>=1 when data ready). Other lanes return 0."""
    _i64_ty = T.i64()
    _i32_ty = T.i32()
    _if_lane0 = _scf_d.IfOp(
        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
        [_i64_ty],
        has_else=True,
    )
    with _IP(_if_lane0.then_block):
        init = arith.constant(0, type=T.i32())
        w = _scf_d.WhileOp([_i32_ty], [_lv_unwrap(init)])
        before = _ir.Block.create_at_start(w.before, [_i32_ty])
        after = _ir.Block.create_at_start(w.after, [_i32_ty])
        with _IP(before):
            _ = before.arguments[0]
            cv = load_i64_system(cf_addr_i64)
            nf = load_i64_system(nf_addr_i64)
            ok_chunk = arith.cmpi(arith.CmpIPredicate.ne, cv, arith.constant(0, type=T.i64()))
            nf_gt = arith.cmpi(arith.CmpIPredicate.ne, nf, arith.constant(0, type=T.i64()))
            start_u64 = arith.zext_i64(start_tok_i)
            ge = arith.cmpi(arith.CmpIPredicate.uge, start_u64, nf - arith.constant(1, type=T.i64()))
            ok_node = arith.andi(nf_gt, ge)
            done = arith.ori(ok_chunk, ok_node)
            keep = arith.xori(done, arith.constant(1, type=T.bool()))
            _scf_d.ConditionOp(_lv_unwrap(keep), [arith.constant(0)])
        with _IP(after):
            _scf_d.YieldOp([arith.constant(0)])

        cv2 = load_i64_system(cf_addr_i64)
        nf2 = load_i64_system(nf_addr_i64)
        ok_c2 = arith.cmpi(arith.CmpIPredicate.ne, cv2, arith.constant(0, type=T.i64()))
        chunk_raw = arith.select(ok_c2, cv2, arith.constant(1, type=T.i64()))
        _scf_d.YieldOp([chunk_raw])
    with _IP(_if_lane0.else_block):
        _scf_d.YieldOp([arith.constant(0, type=T.i64())])
    return _if_lane0.result


def make_copy_to_staging_kernel(
    *,
    rank: int,
    npes: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    copy_grid_blocks: int,
    warp_num_per_block: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    data_type=None,
):
    _is_fp4 = data_type == torch.float4_e2m1fn_x2
    if _is_fp4:
        n_i32_hidden = hidden_dim // 8
        hidden_bytes = hidden_dim // 2
    else:
        n_i32_hidden = (hidden_dim * hidden_elem_size) // 4
        hidden_bytes = hidden_dim * hidden_elem_size
    scale_bytes = scale_dim * scale_type_size
    index_elems = experts_per_token
    weight_elems = experts_per_token
    xfer_i32 = (hidden_bytes + index_elems * 4 + weight_elems * 4 + scale_bytes + 4 + 3) // 4
    enable_scales = scale_bytes > 0
    scale_i32 = (scale_bytes + 3) // 4 if enable_scales else 0
    max_toks_send = npes * max_tok_per_rank

    @flyc.kernel
    def ep_dispatch_copy_to_staging_v1ll(
        addr_inp: fx.Int64,
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_scales: fx.Int64,
        addr_staging: fx.Int64,
        cur_tok: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gw_id = bid * warp_num_per_block + warp
        gw_num = copy_grid_blocks * warp_num_per_block

        _skip = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, cur_tok, arith.constant(0))),
            [], has_else=True)
        with _IP(_skip.then_block):
            _scf_d.YieldOp([])
        with _IP(_skip.else_block):
            warps_per_tok = _ceil_div_u32(gw_num, cur_tok)
            dim_per_warp = _ceil_div_u32(arith.constant(hidden_dim), warps_per_tok)
            limit = cur_tok * warps_per_tok

            _r_inp = create_buffer_resource_from_addr(_lv_unwrap(addr_inp))
            _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
            _r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
            _r_stg = create_buffer_resource_from_addr(_lv_unwrap(addr_staging))
            _r_sc = (
                create_buffer_resource_from_addr(_lv_unwrap(addr_scales)) if enable_scales else None
            )

            for ii in range(_to_idx(gw_id), _to_idx(limit), _to_idx(gw_num)):
                ii = _to_i32(ii)
                tok_id = arith.divui(ii, warps_per_tok)
                part_id = arith.remui(ii, warps_per_tok)
                dim_off = arith.muli(part_id, dim_per_warp)
                rem = arith.subi(arith.constant(hidden_dim), dim_off)
                dim_chunk = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, dim_off, arith.constant(hidden_dim)),
                    arith.select(
                        arith.cmpi(arith.CmpIPredicate.ult, rem, dim_per_warp),
                        rem,
                        dim_per_warp,
                    ),
                    arith.constant(0),
                )
                n_copy_i32 = arith.divui(
                    dim_chunk * arith.constant(hidden_elem_size) + arith.constant(3),
                    arith.constant(4),
                )
                staging_base_i32 = arith.muli(tok_id, arith.constant(xfer_i32))
                hidden_base_i32 = staging_base_i32 + arith.divui(
                    arith.muli(dim_off, arith.constant(hidden_elem_size)), arith.constant(4)
                )
                lane4 = lane * 4
                for ec4 in range(_to_idx(lane4), _to_idx(n_copy_i32), 256):
                    ec4 = _to_i32(ec4)
                    src_w = (
                        tok_id * arith.constant(n_i32_hidden)
                        + arith.divui(
                            arith.muli(dim_off, arith.constant(hidden_elem_size)), arith.constant(4)
                        )
                        + ec4
                    )
                    vec = buffer_load(_r_inp, src_w, vec_width=4, dtype=T.i32())
                    buffer_store(_lv_unwrap(vec), _r_stg, hidden_base_i32 + ec4)

                _only0 = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ne, part_id, arith.constant(0))),
                    [], has_else=True)
                with _IP(_only0.then_block):
                    _scf_d.YieldOp([])
                with _IP(_only0.else_block):
                    meta0 = staging_base_i32 + arith.constant(n_i32_hidden)
                    for exp_i in range_constexpr(experts_per_token):
                        te = arith.addi(
                            arith.muli(tok_id, arith.constant(experts_per_token)),
                            arith.constant(exp_i),
                        )
                        ix = buffer_load(_r_idx, te, vec_width=1, dtype=T.i32())
                        buffer_store(_lv_unwrap(ix), _r_stg, meta0 + arith.constant(exp_i))
                    wt0 = meta0 + arith.constant(experts_per_token)
                    for exp_i in range_constexpr(experts_per_token):
                        te = arith.addi(
                            arith.muli(tok_id, arith.constant(experts_per_token)),
                            arith.constant(exp_i),
                        )
                        wv = buffer_load(_r_wts, te, vec_width=1, dtype=T.f32())
                        buffer_store(
                            _lv_unwrap(arith.bitcast(T.i32(), wv)), _r_stg, wt0 + arith.constant(exp_i)
                        )
                    if enable_scales:
                        sc0 = wt0 + arith.constant(experts_per_token)
                        for si in range_constexpr(scale_i32):
                            ts = arith.addi(
                                arith.muli(tok_id, arith.constant(scale_i32)), arith.constant(si)
                            )
                            sv = buffer_load(_r_sc, ts, vec_width=1, dtype=T.i32())
                            buffer_store(_lv_unwrap(sv), _r_stg, sc0 + arith.constant(si))
                    else:
                        _ = arith.constant(0)
                    flat_src = arith.constant(rank) * arith.constant(max_toks_send) + tok_id
                    tail = staging_base_i32 + arith.constant(xfer_i32 - 1)
                    _if_w = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                        [], has_else=True)
                    with _IP(_if_w.then_block):
                        buffer_store(_lv_unwrap(flat_src), _r_stg, tail)
                        _scf_d.YieldOp([])
                    with _IP(_if_w.else_block):
                        _scf_d.YieldOp([])
                    _scf_d.YieldOp([])
            _scf_d.YieldOp([])

    return ep_dispatch_copy_to_staging_v1ll


def make_dispatch_internode_v1ll_main_kernel(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_rank: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    block_num: int,
    rdma_block_num: int,
    warp_num_per_block: int,
    num_qp_per_pe: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    enable_std_moe: bool = False,
    data_type=None,
):
    n_nodes = npes // gpu_per_node
    my_node = rank // gpu_per_node
    _is_fp4 = data_type == torch.float4_e2m1fn_x2
    if _is_fp4:
        n_i32_hidden = hidden_dim // 8
        hidden_bytes = hidden_dim // 2
    else:
        n_i32_hidden = (hidden_dim * hidden_elem_size) // 4
        hidden_bytes = hidden_dim * hidden_elem_size
    scale_bytes = scale_dim * scale_type_size
    index_bytes = experts_per_token * 4
    weight_bytes = experts_per_token * 4
    xfer_bytes = hidden_bytes + index_bytes + weight_bytes + scale_bytes + 4
    xfer_i32 = xfer_bytes // 4
    enable_scales = scale_bytes > 0
    scale_i32 = (scale_bytes + 3) // 4 if enable_scales else 0
    # Route flat = dest_pe * max_toks_send + slot (slot < max_tok_per_rank).  Using
    # null_flat = max_toks_send alone collides with dest_pe=1,slot=0 when max_toks_send==16.
    max_toks_send = npes * max_tok_per_rank
    max_tok_send_per_rank = max_tok_per_rank
    max_recv = npes * max_tok_per_rank
    null_flat = (npes - 1) * max_toks_send + max_tok_per_rank
    max_chunk_num = (max_tok_send_per_rank + WARP_SZ - 1) // WARP_SZ
    max_ll_iter = max_tok_send_per_rank * experts_per_token * max(1, n_nodes - 1)
    xgmi_blocks = block_num - rdma_block_num
    max_tokens_per_expert = npes * max_tok_per_rank

    @flyc.kernel
    def ep_dispatch_internode_v1ll_main(
        addr_idx: fx.Int64,
        addr_staging: fx.Int64,
        addr_dispatch_inp: fx.Int64,
        addr_chunk_flag: fx.Int64,
        addr_node_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        addr_inter_bar: fx.Int64,
        addr_inter_dest_map: fx.Int64,
        addr_inter_send_map: fx.Int64,
        addr_disp_dest_map: fx.Int64,
        addr_dest_pe_ctr: fx.Int64,
        addr_recv_sym: fx.Int64,
        addr_p2p_recv: fx.Int64,
        addr_total_recv: fx.Int64,
        addr_disp_grid_bar: fx.Int64,
        addr_combine_grid_bar: fx.Int64,
        addr_cross_dev_flag: fx.Int64,
        addr_disp_tok_off_sym: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_out_tok: fx.Int64,
        addr_p2p_out_idx: fx.Int64,
        addr_p2p_out_wts: fx.Int64,
        addr_p2p_out_scales: fx.Int64,
        # StdMoE ConvertDispatchOutput (local symmetric + device packed buffers; unused if not compiled with enable_std_moe)
        addr_out_tok_local: fx.Int64,
        addr_out_idx_local: fx.Int64,
        addr_tis_local: fx.Int64,
        addr_packed_recv_x: fx.Int64,
        addr_packed_recv_count: fx.Int64,
        addr_packed_recv_src_info: fx.Int64,
        addr_disp_tok_map: fx.Int64,
        cur_tok: fx.Int32,
        dev_trace_level: fx.Int32,
        dev_trace_seq: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gw_id = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        warp_num = warp_num_per_block

        _i32r = arith.constant(rank, type=T.i32())
        _if_d0 = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ne, dev_trace_level, arith.constant(0))),
            [],
            has_else=True,
        )
        with _IP(_if_d0.then_block):
            _if_b0 = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, bid, arith.constant(0))),
                [],
                has_else=True,
            )
            with _IP(_if_b0.then_block):
                _if_t0 = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0))),
                    [],
                    has_else=True,
                )
                with _IP(_if_t0.then_block):
                    fx.printf(
                        "[v1ll dev] r={} seq={} ENTER main cur_tok={} bid={} tid={}\n",
                        _i32r,
                        dev_trace_seq,
                        cur_tok,
                        bid,
                        tid,
                    )
                    _scf_d.YieldOp([])
                with _IP(_if_t0.else_block):
                    _scf_d.YieldOp([])
                _scf_d.YieldOp([])
            with _IP(_if_b0.else_block):
                _scf_d.YieldOp([])
            _scf_d.YieldOp([])
        with _IP(_if_d0.else_block):
            _scf_d.YieldOp([])

        _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
        _r_bf = create_buffer_resource_from_addr(_lv_unwrap(addr_block_flag))
        _r_p2p_tok = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_tok))
        _r_p2p_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_idx))
        _r_p2p_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_wts))
        _r_p2p_toff = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tok_off))
        _r_p2p_tis = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tis))
        _r_p2p_recv = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_recv))
        _r_p2p_sc_tbl = (
            create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_scales))
            if enable_scales
            else None
        )

        # ── RDMA path ──────────────────────────────────────────────────────
        _if_rd = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, bid, arith.constant(rdma_block_num))),
            [], has_else=True)

        with _IP(_if_rd.then_block):
            total_chunk_num = _ceil_div_u32(cur_tok, arith.constant(WARP_SZ))
            block_chunk_num = _ceil_div_u32(total_chunk_num, arith.constant(rdma_block_num))
            chunk_start = arith.muli(
                _lv_unwrap(block_chunk_num),
                _lv_unwrap(bid * arith.constant(WARP_SZ)),
            )
            # Match mori DispatchInterNodeLLSend: each RDMA block covers
            # [chunkStart, min(chunkStart + blockChunkNum * warpSize, curTok)), not
            # blockChunkNum * warpSize * rdmaBlockNum (that would OOB staging / flags).
            span = arith.muli(block_chunk_num, arith.constant(WARP_SZ))
            chunk_end = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, cur_tok, chunk_start + span),
                cur_tok,
                chunk_start + span,
            )

            first_tok = chunk_start + lane
            for token_id in range(_to_idx(first_tok), _to_idx(chunk_end), _to_idx(WARP_SZ)):
                token_id = _to_i32(token_id)
                for i_node in range(_to_idx(warp), _to_idx(n_nodes), _to_idx(warp_num)):
                    i_node = _to_i32(i_node)
                    _skipn = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, i_node, arith.constant(my_node))),
                        [], has_else=True)
                    with _IP(_skipn.then_block):
                        _scf_d.YieldOp([])
                    with _IP(_skipn.else_block):
                        proxy_pe = arith.muli(i_node, arith.constant(gpu_per_node)) + arith.constant(
                            rank % gpu_per_node
                        )
                        should_send = arith.constant(0, type=T.bool())
                        for exp_i in range_constexpr(experts_per_token):
                            tex = arith.addi(
                                arith.muli(token_id, arith.constant(experts_per_token)),
                                arith.constant(exp_i),
                            )
                            exp = buffer_load(_r_idx, tex, vec_width=1, dtype=T.i32())
                            dnode = arith.divui(
                                arith.divui(exp, arith.constant(experts_per_rank)),
                                arith.constant(gpu_per_node),
                            )
                            hit = arith.cmpi(arith.CmpIPredicate.eq, dnode, i_node)
                            should_send = arith.ori(should_send, hit)
                            _nul = _scf_d.IfOp(_lv_unwrap(hit), [], has_else=True)
                            with _IP(_nul.then_block):
                                store_i32_seq_cst_system(
                                    addr_disp_dest_map + arith.zext_i64(tex * arith.constant(4)),
                                    arith.constant(null_flat),
                                )
                                _scf_d.YieldOp([])
                            with _IP(_nul.else_block):
                                _scf_d.YieldOp([])

                        _if_l0 = _scf_d.IfOp(
                            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                            [T.i32()],
                            has_else=True,
                        )
                        with _IP(_if_l0.then_block):
                            prev = atomic_add_i32_global_ret_prev(
                                addr_block_flag + arith.zext_i64(i_node * 4), arith.constant(1)
                            )
                            _scf_d.YieldOp([prev])
                        with _IP(_if_l0.else_block):
                            _scf_d.YieldOp([arith.constant(0)])
                        flag_slot = readlane(_if_l0.result, 0)

                        dest_tok_off = arith.muli(flag_slot, arith.constant(WARP_SZ))
                        dest_tok_id = dest_tok_off + lane
                        remote_idx = arith.constant(my_node) * arith.constant(max_tok_send_per_rank) + dest_tok_id

                        _if_put = _scf_d.IfOp(
                            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                            [], has_else=True,
                        )
                        with _IP(_if_put.then_block):
                            tmax = arith.select(
                                arith.cmpi(
                                    arith.CmpIPredicate.ult,
                                    token_id + arith.constant(WARP_SZ),
                                    chunk_end,
                                ),
                                token_id + arith.constant(WARP_SZ),
                                chunk_end,
                            )
                            token_num = tmax - token_id
                            staging_off = arith.muli(token_id, arith.constant(xfer_bytes))
                            dest_byte = arith.zext_i64(remote_idx * arith.constant(xfer_bytes))
                            put_dest = addr_dispatch_inp + dest_byte
                            src_ptr = addr_staging + arith.zext_i64(staging_off)
                            sig_off = (
                                arith.constant(my_node) * arith.constant(max_chunk_num) + flag_slot
                            ) * arith.constant(8)
                            sig_ptr = addr_chunk_flag + arith.zext_i64(sig_off)
                            qp_id = arith.remui(
                                arith.divui(token_id, arith.constant(WARP_SZ)),
                                arith.constant(num_qp_per_pe),
                            )
                            mori_shmem.putmem_nbi_signal_thread(
                                put_dest,
                                src_ptr,
                                arith.zext_i64(token_num * arith.constant(xfer_bytes)),
                                sig_ptr,
                                arith.zext_i64(token_num + arith.constant(1)),
                                arith.constant(AMO_ADD),
                                proxy_pe,
                                qp_id,
                            )
                            _scf_d.YieldOp([])
                        with _IP(_if_put.else_block):
                            _scf_d.YieldOp([])

                        _if_ss = _scf_d.IfOp(_lv_unwrap(should_send), [], has_else=True)
                        with _IP(_if_ss.then_block):
                            sm_addr = addr_inter_send_map + arith.zext_i64(
                                arith.constant(n_nodes) * token_id + i_node) * 4
                            store_i32_seq_cst_system(sm_addr, dest_tok_id)
                            _scf_d.YieldOp([])
                        with _IP(_if_ss.else_block):
                            _scf_d.YieldOp([])
                        _scf_d.YieldOp([])

            bar1 = addr_inter_bar + arith.constant(4)
            _if_l0b = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                [T.i32()],
                has_else=True,
            )
            with _IP(_if_l0b.then_block):
                prev_b = atomic_add_i32_global_ret_prev(bar1, arith.constant(1))
                _scf_d.YieldOp([_lv_unwrap(prev_b)])
            with _IP(_if_l0b.else_block):
                _scf_d.YieldOp([arith.constant(0)])
            finished_warp = readlane(_if_l0b.result, 0)
            _if_done = _scf_d.IfOp(
                _lv_unwrap(
                    arith.cmpi(
                        arith.CmpIPredicate.eq,
                        finished_warp + arith.constant(1),
                        arith.constant(rdma_block_num * warp_num),
                    )
                ),
                [], has_else=True,
            )
            with _IP(_if_done.then_block):
                for li in range_constexpr(n_nodes):
                    _if_ln = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li))),
                        [], has_else=True,
                    )
                    with _IP(_if_ln.then_block):
                        proxy_pe = arith.constant(li * gpu_per_node + (rank % gpu_per_node))
                        cnt = buffer_load(_r_bf, arith.constant(li), vec_width=1, dtype=T.i32())
                        num_sig = arith.muli(cnt, arith.constant(WARP_SZ)) + arith.constant(1)
                        node_recv_slot = addr_node_recv + arith.zext_i64(
                            arith.constant(my_node * 8)
                        )
                        mori_shmem.uint64_atomic_add_thread(
                            node_recv_slot,
                            arith.zext_i64(num_sig),
                            proxy_pe,
                            arith.constant(0),
                        )
                        _scf_d.YieldOp([])
                    with _IP(_if_ln.else_block):
                        _scf_d.YieldOp([])
                _if_zb = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                    [], has_else=True,
                )
                with _IP(_if_zb.then_block):
                    store_u32_relaxed_global(bar1, arith.constant(0, type=T.i32()))
                    _scf_d.YieldOp([])
                with _IP(_if_zb.else_block):
                    _scf_d.YieldOp([])
                _scf_d.YieldOp([])
            with _IP(_if_done.else_block):
                _scf_d.YieldOp([])

            # LL recv
            for ii in range(
                _to_idx(gw_id), _to_idx(max_ll_iter), _to_idx(rdma_block_num * warp_num)
            ):
                ii = _to_i32(ii)
                expert_id = arith.remui(ii, arith.constant(experts_per_token))
                token_id = arith.remui(
                    arith.divui(ii, arith.constant(experts_per_token)),
                    arith.constant(max_tok_send_per_rank),
                )
                node_id = arith.divui(
                    arith.divui(ii, arith.constant(experts_per_token)),
                    arith.constant(max_tok_send_per_rank),
                )
                node = arith.remui(
                    arith.constant(my_node + 1) + node_id, arith.constant(n_nodes))
                k = arith.divui(token_id, arith.constant(WARP_SZ))
                start_tok_i = arith.muli(k, arith.constant(WARP_SZ))

                cf_a = addr_chunk_flag + arith.zext_i64((node * max_chunk_num + k) * 8)
                nf_a = addr_node_recv + arith.zext_i64(node * 8)
                chunk_raw = _emit_poll_chunk_or_node(cf_a, nf_a, start_tok_i, lane)
                chunk_raw_w = readlane(arith.trunci(T.i32(), _lv_unwrap(chunk_raw)), 0)
                n_tok = arith.subi(chunk_raw_w, arith.constant(1))
                end_tok = start_tok_i + n_tok
                _skip_t = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.uge, token_id, end_tok)),
                    [], has_else=True,
                )
                with _IP(_skip_t.then_block):
                    _scf_d.YieldOp([])
                with _IP(_skip_t.else_block):
                    gtok = node * arith.constant(max_tok_send_per_rank) + token_id
                    ptr_frame = addr_dispatch_inp + arith.zext_i64(gtok * arith.constant(xfer_bytes))
                    _r_fr = create_buffer_resource_from_addr(_lv_unwrap(ptr_frame))

                    lane_pe = arith.constant(-1, type=T.i32())
                    for le in range_constexpr(experts_per_token):
                        ix_l = buffer_load(
                            _r_fr,
                            arith.constant(n_i32_hidden + le),
                            vec_width=1,
                            dtype=T.i32(),
                        )
                        lp_l = arith.divui(ix_l, arith.constant(experts_per_rank))
                        lane_pe = arith.select(
                            arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(le)),
                            lp_l,
                            lane_pe,
                        )

                    src_flat = buffer_load(
                        _r_fr, arith.constant(xfer_i32 - 1), vec_width=1, dtype=T.i32()
                    )

                    dest_pe = readlane(lane_pe, expert_id)
                    dest_node = arith.divui(dest_pe, arith.constant(gpu_per_node))
                    # mori: __any((laneId < expertId) && (destPe == lanePe)) — same as OR over
                    # le < expertId of (readlane(lane_pe, le) == dest_pe); ballot path was incorrect.
                    is_dup = arith.constant(0, type=T.bool())
                    for le in range_constexpr(experts_per_token):
                        prev_pe = readlane(lane_pe, arith.constant(le))
                        le_lt_e = arith.cmpi(arith.CmpIPredicate.ult, arith.constant(le), expert_id)
                        same_dest = arith.cmpi(arith.CmpIPredicate.eq, prev_pe, dest_pe)
                        is_dup = arith.ori(is_dup, arith.andi(le_lt_e, same_dest))
                    off_node = arith.cmpi(
                        arith.CmpIPredicate.ne, dest_node, arith.constant(my_node))
                    should_skip = arith.ori(off_node, is_dup)

                    _if_sk = _scf_d.IfOp(_lv_unwrap(should_skip), [], has_else=True)
                    with _IP(_if_sk.then_block):
                        _if_w = _scf_d.IfOp(
                            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                            [], has_else=True,
                        )
                        with _IP(_if_w.then_block):
                            store_i32_seq_cst_system(
                                addr_inter_dest_map
                                + arith.zext_i64((gtok * experts_per_token + expert_id) * arith.constant(4)),
                                arith.constant(null_flat),
                            )
                            _scf_d.YieldOp([])
                        with _IP(_if_w.else_block):
                            _scf_d.YieldOp([])
                        _scf_d.YieldOp([])
                    with _IP(_if_sk.else_block):
                        _if_dt = _scf_d.IfOp(
                            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                            [T.i32()],
                            has_else=True,
                        )
                        with _IP(_if_dt.then_block):
                            remote_off = buffer_load(
                                _r_p2p_toff, dest_pe, vec_width=1, dtype=T.i64())
                            prev = atomic_add_i32_global_ret_prev(
                                remote_off, arith.constant(1))
                            flat_tok = dest_pe * arith.constant(max_toks_send) + prev
                            store_i32_seq_cst_system(
                                addr_inter_dest_map
                                + arith.zext_i64((gtok * experts_per_token + expert_id) * arith.constant(4)),
                                flat_tok,
                            )
                            remote_tis = buffer_load(
                                _r_p2p_tis, dest_pe, vec_width=1, dtype=T.i64())
                            store_i32_at_index(remote_tis, prev, src_flat)
                            _scf_d.YieldOp([prev])
                        with _IP(_if_dt.else_block):
                            _scf_d.YieldOp([arith.constant(0)])
                        dest_tok = readlane(_if_dt.result, 0)

                        dst_tok_base = buffer_load(_r_p2p_tok, dest_pe, vec_width=1, dtype=T.i64()) + (
                            arith.zext_i64(dest_tok * arith.constant(hidden_bytes))
                        )
                        lane4 = lane * 4
                        _safe = (n_i32_hidden // 512) * 512
                        if n_i32_hidden >= 512 and _safe > 0:
                            for ec4 in range(_to_idx(lane4), _to_idx(_safe), 512):
                                ec4 = _to_i32(ec4)
                                v0 = buffer_load(_r_fr, ec4, vec_width=4, dtype=T.i32())
                                v1 = buffer_load(_r_fr, ec4 + 256, vec_width=4, dtype=T.i32())
                                _rd = create_buffer_resource_from_addr(_lv_unwrap(dst_tok_base))
                                buffer_store(_lv_unwrap(v0), _rd, ec4)
                                buffer_store(_lv_unwrap(v1), _rd, ec4 + 256)
                        else:
                            _ = arith.constant(0)
                        if _safe < n_i32_hidden:
                            for ec4 in range(_to_idx(lane4 + arith.constant(_safe)), _to_idx(n_i32_hidden), 256):
                                ec4 = _to_i32(ec4)
                                v0 = buffer_load(_r_fr, ec4, vec_width=4, dtype=T.i32())
                                _rd = create_buffer_resource_from_addr(_lv_unwrap(dst_tok_base))
                                buffer_store(_lv_unwrap(v0), _rd, ec4)
                        elif n_i32_hidden < 512:
                            for ec4 in range(_to_idx(lane4), _to_idx(n_i32_hidden), 256):
                                ec4 = _to_i32(ec4)
                                v0 = buffer_load(_r_fr, ec4, vec_width=4, dtype=T.i32())
                                _rd = create_buffer_resource_from_addr(_lv_unwrap(dst_tok_base))
                                buffer_store(_lv_unwrap(v0), _rd, ec4)
                        else:
                            _ = arith.constant(0)

                        dst_ix_base = buffer_load(_r_p2p_idx, dest_pe, vec_width=1, dtype=T.i64()) + (
                            arith.zext_i64(dest_tok * arith.constant(experts_per_token * 4))
                        )
                        _rdx = create_buffer_resource_from_addr(_lv_unwrap(dst_ix_base))
                        for le in range_constexpr(experts_per_token):
                            ixv = buffer_load(
                                _r_fr, arith.constant(n_i32_hidden + le), vec_width=1, dtype=T.i32()
                            )
                            buffer_store(_lv_unwrap(ixv), _rdx, arith.constant(le))

                        wt_off_i32 = arith.constant(n_i32_hidden + experts_per_token)
                        dst_wt_base = buffer_load(_r_p2p_wts, dest_pe, vec_width=1, dtype=T.i64()) + (
                            arith.zext_i64(dest_tok * arith.constant(experts_per_token * 4))
                        )
                        _rdw = create_buffer_resource_from_addr(_lv_unwrap(dst_wt_base))
                        for le in range_constexpr(experts_per_token):
                            w32 = buffer_load(
                                _r_fr, wt_off_i32 + arith.constant(le), vec_width=1, dtype=T.i32()
                            )
                            buffer_store(_lv_unwrap(w32), _rdw, arith.constant(le))

                        if enable_scales:
                            sc_off = wt_off_i32 + arith.constant(experts_per_token)
                            psc = buffer_load(
                                _r_p2p_sc_tbl, dest_pe, vec_width=1, dtype=T.i64()
                            ) + arith.zext_i64(dest_tok * arith.constant(scale_i32 * 4))
                            _rds = create_buffer_resource_from_addr(_lv_unwrap(psc))
                            for si in range_constexpr(scale_i32):
                                sv = buffer_load(
                                    _r_fr, sc_off + arith.constant(si), vec_width=1, dtype=T.i32()
                                )
                                buffer_store(_lv_unwrap(sv), _rds, arith.constant(si))
                        else:
                            _ = arith.constant(0)

                        _if_hit_pe = _scf_d.IfOp(
                            _lv_unwrap(
                                arith.cmpi(
                                    arith.CmpIPredicate.eq,
                                    arith.remui(dest_pe, arith.constant(gpu_per_node)),
                                    lane,
                                )
                            ),
                            [], has_else=True,
                        )
                        with _IP(_if_hit_pe.then_block):
                            # mori DispatchInterNodeLLRecv: atomicAdd(destPeTokenCounter +
                            #   myNode * gpuPerNode + laneId, ...), not routing dest_pe.
                            ctr_slot_pe = arith.muli(
                                arith.constant(my_node), arith.constant(gpu_per_node)
                            ) + lane
                            atomic_add_i32_global_ret_prev(
                                addr_dest_pe_ctr
                                + arith.zext_i64(ctr_slot_pe * arith.constant(4)),
                                arith.constant(1),
                            )
                            _scf_d.YieldOp([])
                        with _IP(_if_hit_pe.else_block):
                            _scf_d.YieldOp([])
                        _scf_d.YieldOp([])
                    _scf_d.YieldOp([])

            _scf_d.YieldOp([])

        # ── XGMI intranode path ────────────────────────────────────────────
        with _IP(_if_rd.else_block):
            _if_xg = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ugt, arith.constant(xgmi_blocks), arith.constant(0))),
                [], has_else=True,
            )
            with _IP(_if_xg.then_block):
                b_off = arith.constant(rdma_block_num)
                tok_per_b = _ceil_div_u32(cur_tok, arith.constant(xgmi_blocks))
                start_t = (bid - b_off) * tok_per_b
                end_t = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, cur_tok, start_t + tok_per_b),
                    cur_tok,
                    start_t + tok_per_b,
                )
                span_t = end_t - start_t
                limit_ix = span_t * arith.constant(experts_per_token)
                for ixp in range(_to_idx(warp), _to_idx(limit_ix), _to_idx(warp_num)):
                    ixp = _to_i32(ixp)
                    tok_loc = arith.divui(ixp, arith.constant(experts_per_token)) + start_t
                    in_exp = arith.remui(ixp, arith.constant(experts_per_token))
                    dest_exp = buffer_load(
                        _r_idx,
                        start_t * experts_per_token + ixp,
                        vec_width=1,
                        dtype=T.i32(),
                    )
                    dest_pe = arith.divui(dest_exp, arith.constant(experts_per_rank))
                    dest_node = arith.divui(dest_pe, arith.constant(gpu_per_node))

                    lane_pe = arith.constant(-1, type=T.i32())
                    for le in range_constexpr(experts_per_token):
                        lane_pe = arith.select(
                            arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(le)),
                            arith.divui(
                                buffer_load(
                                    _r_idx,
                                    tok_loc * experts_per_token + arith.constant(le),
                                    vec_width=1,
                                    dtype=T.i32(),
                                ),
                                arith.constant(experts_per_rank),
                            ),
                            lane_pe,
                        )

                    is_dup2 = arith.constant(0, type=T.bool())
                    for le in range_constexpr(experts_per_token):
                        prev_pe = readlane(lane_pe, arith.constant(le))
                        le_lt_e = arith.cmpi(arith.CmpIPredicate.ult, arith.constant(le), in_exp)
                        same_dest = arith.cmpi(arith.CmpIPredicate.eq, prev_pe, dest_pe)
                        is_dup2 = arith.ori(is_dup2, arith.andi(le_lt_e, same_dest))
                    same_n = arith.cmpi(arith.CmpIPredicate.eq, dest_node, arith.constant(my_node))
                    _if_local = _scf_d.IfOp(_lv_unwrap(same_n), [], has_else=True)
                    with _IP(_if_local.then_block):
                        _if_dedup = _scf_d.IfOp(_lv_unwrap(is_dup2), [], has_else=True)
                        with _IP(_if_dedup.then_block):
                            _if_w0 = _scf_d.IfOp(
                                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                                [], has_else=True,
                            )
                            with _IP(_if_w0.then_block):
                                store_i32_seq_cst_system(
                                    addr_disp_dest_map
                                    + arith.zext_i64((start_t * experts_per_token + ixp) * arith.constant(4)),
                                    arith.constant(null_flat),
                                )
                                _scf_d.YieldOp([])
                            with _IP(_if_w0.else_block):
                                _scf_d.YieldOp([])
                            _scf_d.YieldOp([])
                        with _IP(_if_dedup.else_block):
                            _if_dt2 = _scf_d.IfOp(
                                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                                [T.i32()],
                                has_else=True,
                            )
                            with _IP(_if_dt2.then_block):
                                r_off = buffer_load(_r_p2p_toff, dest_pe, vec_width=1, dtype=T.i64())
                                p2 = atomic_add_i32_global_ret_prev(r_off, arith.constant(1))
                                fl = dest_pe * arith.constant(max_toks_send) + p2
                                store_i32_seq_cst_system(
                                    addr_disp_dest_map
                                    + arith.zext_i64((tok_loc * experts_per_token + in_exp) * arith.constant(4)),
                                    fl,
                                )
                                rt = buffer_load(_r_p2p_tis, dest_pe, vec_width=1, dtype=T.i64())
                                sf = arith.constant(rank) * arith.constant(max_toks_send) + tok_loc
                                store_i32_at_index(rt, p2, sf)
                                _scf_d.YieldOp([p2])
                            with _IP(_if_dt2.else_block):
                                _scf_d.YieldOp([arith.constant(0)])
                            dt2 = readlane(_if_dt2.result, 0)

                            src_b = addr_staging + arith.zext_i64(tok_loc * arith.constant(xfer_bytes))
                            _r_sf = create_buffer_resource_from_addr(_lv_unwrap(src_b))
                            dst_tb = buffer_load(_r_p2p_tok, dest_pe, vec_width=1, dtype=T.i64()) + (
                                arith.zext_i64(dt2 * arith.constant(hidden_bytes))
                            )
                            lane4 = lane * 4
                            _safe2 = (n_i32_hidden // 512) * 512
                            if n_i32_hidden >= 512 and _safe2 > 0:
                                for ec4 in range(_to_idx(lane4), _to_idx(_safe2), 512):
                                    ec4 = _to_i32(ec4)
                                    v0 = buffer_load(_r_sf, ec4, vec_width=4, dtype=T.i32())
                                    v1 = buffer_load(_r_sf, ec4 + 256, vec_width=4, dtype=T.i32())
                                    _rdt = create_buffer_resource_from_addr(_lv_unwrap(dst_tb))
                                    buffer_store(_lv_unwrap(v0), _rdt, ec4)
                                    buffer_store(_lv_unwrap(v1), _rdt, ec4 + 256)
                            else:
                                _ = arith.constant(0)
                            if _safe2 < n_i32_hidden:
                                for ec4 in range(
                                    _to_idx(lane4 + arith.constant(_safe2)), _to_idx(n_i32_hidden), 256
                                ):
                                    ec4 = _to_i32(ec4)
                                    v0 = buffer_load(_r_sf, ec4, vec_width=4, dtype=T.i32())
                                    _rdt = create_buffer_resource_from_addr(_lv_unwrap(dst_tb))
                                    buffer_store(_lv_unwrap(v0), _rdt, ec4)
                            elif n_i32_hidden < 512:
                                for ec4 in range(_to_idx(lane4), _to_idx(n_i32_hidden), 256):
                                    ec4 = _to_i32(ec4)
                                    v0 = buffer_load(_r_sf, ec4, vec_width=4, dtype=T.i32())
                                    _rdt = create_buffer_resource_from_addr(_lv_unwrap(dst_tb))
                                    buffer_store(_lv_unwrap(v0), _rdt, ec4)
                            else:
                                _ = arith.constant(0)

                            dst_ixb = buffer_load(_r_p2p_idx, dest_pe, vec_width=1, dtype=T.i64()) + (
                                arith.zext_i64(dt2 * arith.constant(experts_per_token * 4))
                            )
                            _rix = create_buffer_resource_from_addr(_lv_unwrap(dst_ixb))
                            w0_i32 = arith.constant(n_i32_hidden + experts_per_token)
                            for le in range_constexpr(experts_per_token):
                                ixv = buffer_load(
                                    _r_sf, arith.constant(n_i32_hidden + le), vec_width=1, dtype=T.i32()
                                )
                                buffer_store(_lv_unwrap(ixv), _rix, arith.constant(le))
                            dst_wb = buffer_load(_r_p2p_wts, dest_pe, vec_width=1, dtype=T.i64()) + (
                                arith.zext_i64(dt2 * arith.constant(experts_per_token * 4))
                            )
                            _rw = create_buffer_resource_from_addr(_lv_unwrap(dst_wb))
                            for le in range_constexpr(experts_per_token):
                                w32 = buffer_load(
                                    _r_sf, w0_i32 + arith.constant(le), vec_width=1, dtype=T.i32()
                                )
                                buffer_store(_lv_unwrap(w32), _rw, arith.constant(le))

                            if enable_scales:
                                sc0 = w0_i32 + arith.constant(experts_per_token)
                                psb = buffer_load(
                                    _r_p2p_sc_tbl, dest_pe, vec_width=1, dtype=T.i64()
                                ) + arith.zext_i64(dt2 * arith.constant(scale_i32 * 4))
                                _rs = create_buffer_resource_from_addr(_lv_unwrap(psb))
                                for si in range_constexpr(scale_i32):
                                    sv = buffer_load(
                                        _r_sf, sc0 + arith.constant(si), vec_width=1, dtype=T.i32()
                                    )
                                    buffer_store(_lv_unwrap(sv), _rs, arith.constant(si))
                            else:
                                _ = arith.constant(0)

                            _if_hit_ix = _scf_d.IfOp(
                                _lv_unwrap(
                                    arith.cmpi(
                                        arith.CmpIPredicate.eq,
                                        arith.remui(dest_pe, arith.constant(gpu_per_node)),
                                        lane,
                                    )
                                ),
                                [], has_else=True,
                            )
                            with _IP(_if_hit_ix.then_block):
                                ctr_slot_pe = arith.muli(
                                    arith.constant(my_node), arith.constant(gpu_per_node)
                                ) + lane
                                atomic_add_i32_global_ret_prev(
                                    addr_dest_pe_ctr
                                    + arith.zext_i64(ctr_slot_pe * arith.constant(4)),
                                    arith.constant(1),
                                )
                                _scf_d.YieldOp([])
                            with _IP(_if_hit_ix.else_block):
                                _scf_d.YieldOp([])
                            _scf_d.YieldOp([])
                        _scf_d.YieldOp([])
                    with _IP(_if_local.else_block):
                        _scf_d.YieldOp([])

                _scf_d.YieldOp([])
            with _IP(_if_xg.else_block):
                _scf_d.YieldOp([])
            _scf_d.YieldOp([])

        # ── DispatchSync (full grid; global atomic only — no block barrier) ─
        _if_lb = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
            [T.i32()],
            has_else=True,
        )
        with _IP(_if_lb.then_block):
            pw = atomic_add_i32_seq_cst_one_as_ret_prev(
                addr_disp_grid_bar, arith.constant(1)
            )
            _scf_d.YieldOp([_lv_unwrap(pw)])
        with _IP(_if_lb.else_block):
            _scf_d.YieldOp([arith.constant(0)])
        fin_w = readlane(_if_lb.result, 0)
        _if_all = _scf_d.IfOp(
            _lv_unwrap(
                arith.cmpi(
                    arith.CmpIPredicate.eq,
                    fin_w + arith.constant(1),
                    arith.constant(gw_num),
                )
            ),
            [], has_else=True,
        )
        with _IP(_if_all.then_block):
            node_pe0 = arith.constant(my_node * gpu_per_node)
            for li in range_constexpr(gpu_per_node):
                _if_sig = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li))),
                    [], has_else=True,
                )
                with _IP(_if_sig.then_block):
                    dpe3 = node_pe0 + arith.constant(li)
                    ctr = load_i32_seq_cst_system(addr_dest_pe_ctr + arith.zext_i64(dpe3 * arith.constant(4)))
                    ns = ctr + arith.constant(1)
                    # mori: GetAs<index_t*>(destPe) + myPe. For destPe == rank this is local
                    # recv_tok_num[rank]; shmem_ptr_p2p(local, r, r) may be 0 (see mori api docs),
                    # so never use the P2P table for the local-PE case.
                    _if_loc_sig = _scf_d.IfOp(
                        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, dpe3, arith.constant(rank))),
                        [], has_else=True,
                    )
                    with _IP(_if_loc_sig.then_block):
                        wloc = addr_recv_sym + arith.zext_i64(arith.constant(rank * 4))
                        store_i32_seq_cst_system(wloc, ns)
                        _scf_d.YieldOp([])
                    with _IP(_if_loc_sig.else_block):
                        remote = buffer_load(_r_p2p_recv, dpe3, vec_width=1, dtype=T.i64())
                        wptr = remote + arith.zext_i64(arith.constant(rank * 4))
                        store_i32_seq_cst_system(wptr, ns)
                        _scf_d.YieldOp([])
                    _scf_d.YieldOp([])
                with _IP(_if_sig.else_block):
                    _scf_d.YieldOp([])
            _if_clr = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                [], has_else=True,
            )
            with _IP(_if_clr.then_block):
                store_i32_seq_cst_system(addr_disp_grid_bar, arith.constant(0, type=T.i32()))
                _scf_d.YieldOp([])
            with _IP(_if_clr.else_block):
                _scf_d.YieldOp([])

            # Matches mori v1::DispatchSync Phase 2: wait local recv[slot] for
            # slot in [nodePeOffset, nodePeOffset + gpu_per_node); stride warpSize (64).
            for li2 in range_constexpr(gpu_per_node):
                _if_wt = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(li2))),
                    [], has_else=True,
                )
                with _IP(_if_wt.then_block):
                    dpe_w = node_pe0 + arith.constant(li2)
                    sig_addr = addr_recv_sym + arith.zext_i64(dpe_w * arith.constant(4))
                    rv = mori_shmem.int32_wait_until_greater_than(sig_addr, arith.constant(0))
                    recv_n = rv - arith.constant(1)
                    atomic_add_i32_global_ret_prev(addr_total_recv, recv_n)
                    # mori v1::DispatchSync: __threadfence_system() before clearing signal/counter
                    # (not mori_shmem_fence_thread — that also runs ShmemQuietThread()).
                    _mori_threadfence_system()
                    store_i32_seq_cst_system(sig_addr, arith.constant(0))
                    store_i32_seq_cst_system(
                        addr_dest_pe_ctr + arith.zext_i64(dpe_w * arith.constant(4)),
                        arith.constant(0),
                    )
                    _scf_d.YieldOp([])
                with _IP(_if_wt.else_block):
                    _scf_d.YieldOp([])

            _if_fin = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                [], has_else=True,
            )
            with _IP(_if_fin.then_block):
                store_i32_seq_cst_system(addr_disp_tok_off_sym, arith.constant(0))
                atomic_add_i64_global_ret_prev(addr_cross_dev_flag, arith.constant(1, type=T.i64()))
                store_u32_relaxed_global(
                    addr_combine_grid_bar + arith.constant(4), arith.constant(0, type=T.i32())
                )
                _scf_d.YieldOp([])
            with _IP(_if_fin.else_block):
                _scf_d.YieldOp([])

            _scf_d.YieldOp([])
        with _IP(_if_all.else_block):
            _scf_d.YieldOp([])

        for iq in range(_to_idx(gw_id), _to_idx(n_nodes), _to_idx(gw_num)):
            iq = _to_i32(iq)
            px = arith.muli(iq, arith.constant(gpu_per_node)) + arith.constant(rank % gpu_per_node)
            mori_shmem.quiet_thread_pe(px)

        _if_d1 = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ne, dev_trace_level, arith.constant(0))),
            [],
            has_else=True,
        )
        with _IP(_if_d1.then_block):
            _if_b0p = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, bid, arith.constant(0))),
                [],
                has_else=True,
            )
            with _IP(_if_b0p.then_block):
                _if_t0p = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0))),
                    [],
                    has_else=True,
                )
                with _IP(_if_t0p.then_block):
                    fx.printf(
                        "[v1ll dev] r={} seq={} POST_DISPATCH_SYNC+QUIET gw_id={}\n",
                        _i32r,
                        dev_trace_seq,
                        gw_id,
                    )
                    _scf_d.YieldOp([])
                with _IP(_if_t0p.else_block):
                    _scf_d.YieldOp([])
                _scf_d.YieldOp([])
            with _IP(_if_b0p.else_block):
                _scf_d.YieldOp([])
            _scf_d.YieldOp([])
        with _IP(_if_d1.else_block):
            _scf_d.YieldOp([])

        # Phase 4: ConvertDispatchOutput (StdMoE), same structure as intranode dispatch Phase 4.
        # Reuses addr_disp_grid_bar word 0 after DispatchSync cleared it.
        if enable_std_moe:
            n_i32 = n_i32_hidden
            nbytes = hidden_bytes
            _i32_ty = T.i32()

            fx.barrier()
            if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
                atomic_add_global_at(addr_disp_grid_bar, arith.constant(1))
            fx.barrier()
            if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
                mori_shmem.int32_wait_until_equals(addr_disp_grid_bar, block_num)
            fx.barrier()

            _r_out_idx_local = create_buffer_resource_from_addr(_lv_unwrap(addr_out_idx_local))
            _r_tis_local = create_buffer_resource_from_addr(_lv_unwrap(addr_tis_local))
            _r_out_tok_local = create_buffer_resource_from_addr(_lv_unwrap(addr_out_tok_local))
            _r_total_rv = create_buffer_resource_from_addr(_lv_unwrap(addr_total_recv))
            total_recv_moe = buffer_load(_r_total_rv, arith.constant(0), vec_width=1, dtype=T.i32())
            limit_moe = total_recv_moe * experts_per_token

            for ii_idx in range(_to_idx(gw_id), _to_idx(limit_moe), _to_idx(gw_num)):
                ii = _to_i32(ii_idx)
                tok_idx_moe = arith.divui(ii, experts_per_token)

                exp_id = buffer_load(_r_out_idx_local, ii, vec_width=1, dtype=T.i32())
                local_exp = exp_id - arith.constant(rank * experts_per_rank)
                is_local = arith.cmpi(arith.CmpIPredicate.ult, local_exp, arith.constant(experts_per_rank))
                _if_l0 = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                    [_i32_ty], has_else=True)
                with _IP(_if_l0.then_block):
                    _if_loc = _scf_d.IfOp(_lv_unwrap(is_local), [_i32_ty], has_else=True)
                    with _IP(_if_loc.then_block):
                        _cnt_addr = addr_packed_recv_count + arith.zext_i64(local_exp) * 4
                        _old_idx = atomic_add_global_at(_cnt_addr, arith.constant(1))
                        _scf_d.YieldOp([_lv_unwrap(_old_idx)])
                    with _IP(_if_loc.else_block):
                        _scf_d.YieldOp([_lv_unwrap(arith.constant(0))])
                    _scf_d.YieldOp([_if_loc.result])
                with _IP(_if_l0.else_block):
                    _scf_d.YieldOp([_lv_unwrap(arith.constant(0))])
                packed_idx = readlane(_if_l0.result, 0)

                safe_local = arith.select(is_local, local_exp, arith.constant(0))
                linear_idx = safe_local * max_tokens_per_expert + packed_idx
                _slot_val_i64 = arith.select(
                    is_local,
                    arith.zext_i64(linear_idx),
                    arith.constant(-1, type=T.i64()),
                )
                if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                    _slot_addr = addr_disp_tok_map + arith.zext_i64(ii) * 8
                    store_i64_global_system(_slot_addr, _slot_val_i64)

                if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                    if is_local:
                        _src_pos = buffer_load(_r_tis_local, tok_idx_moe, vec_width=1, dtype=T.i32())
                        store_i32_at_index(addr_packed_recv_src_info, linear_idx, _src_pos)

                _src_base = addr_out_tok_local + arith.zext_i64(tok_idx_moe) * nbytes
                _dst_base = addr_packed_recv_x + arith.zext_i64(linear_idx) * nbytes
                _rsrc_s = create_buffer_resource_from_addr(_lv_unwrap(_src_base))
                _rsrc_d = create_buffer_resource_from_addr(_lv_unwrap(_dst_base))
                _lane4 = lane * 4
                _safe_cdo_end = (n_i32 // 512) * 512
                if n_i32 >= 512 and _safe_cdo_end > 0:
                    _copy_end_dual = arith.select(is_local, arith.constant(_safe_cdo_end), _lane4)
                    for ec4 in range(_to_idx(_lane4), _to_idx(_copy_end_dual), 512):
                        ec4 = _to_i32(ec4)
                        _v0 = buffer_load(_rsrc_s, ec4, vec_width=4, dtype=T.i32())
                        _v1 = buffer_load(_rsrc_s, ec4 + 256, vec_width=4, dtype=T.i32())
                        buffer_store(_v0, _rsrc_d, ec4)
                        buffer_store(_v1, _rsrc_d, ec4 + 256)
                if _safe_cdo_end < n_i32:
                    _copy_end_tail = arith.select(is_local, arith.constant(n_i32), _lane4)
                    for ec4 in range(_to_idx(_lane4 + arith.constant(_safe_cdo_end)), _to_idx(_copy_end_tail), 256):
                        ec4 = _to_i32(ec4)
                        _v0 = buffer_load(_rsrc_s, ec4, vec_width=4, dtype=T.i32())
                        buffer_store(_v0, _rsrc_d, ec4)
                elif n_i32 < 512:
                    _copy_end_single = arith.select(is_local, arith.constant(n_i32), _lane4)
                    for ec4 in range(_to_idx(_lane4), _to_idx(_copy_end_single), 256):
                        ec4 = _to_i32(ec4)
                        _v0 = buffer_load(_rsrc_s, ec4, vec_width=4, dtype=T.i32())
                        buffer_store(_v0, _rsrc_d, ec4)

    return ep_dispatch_internode_v1ll_main


def make_copy_staging_jit(
    *,
    rank: int,
    npes: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    copy_grid_blocks: int,
    warp_num_per_block: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    data_type=None,
):
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_copy_to_staging_kernel(
        rank=rank,
        npes=npes,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        copy_grid_blocks=copy_grid_blocks,
        warp_num_per_block=warp_num_per_block,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        data_type=data_type,
    )
    _cg, _wpb = copy_grid_blocks, warp_num_per_block

    @flyc.jit
    def launch_copy(
        addr_inp: fx.Int64,
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_scales: fx.Int64,
        addr_staging: fx.Int64,
        cur_tok: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_cg, _wpb)
        kernel(addr_inp, addr_idx, addr_wts, addr_scales, addr_staging, cur_tok).launch(
            grid=(_cg, 1, 1),
            block=(_wpb * 64, 1, 1),
            stream=stream,
        )

    # Disk cache keys do not include edits under FlyDSL/kernels/ (only flydsl.*); bump when kernel body changes.
    launch_copy.compile_hints = {
        **launch_copy.compile_hints,
        "internode_v1ll_jit_rev": 6,
    }
    return launch_copy


def make_dispatch_internode_v1ll_main_jit(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_rank: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    block_num: int,
    rdma_block_num: int,
    warp_num_per_block: int,
    num_qp_per_pe: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    enable_std_moe: bool = False,
    data_type=None,
):
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_dispatch_internode_v1ll_main_kernel(
        rank=rank,
        npes=npes,
        gpu_per_node=gpu_per_node,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_num_per_block=warp_num_per_block,
        num_qp_per_pe=num_qp_per_pe,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        enable_std_moe=enable_std_moe,
        data_type=data_type,
    )
    _bn, _wpb = block_num, warp_num_per_block

    @flyc.jit
    def launch_main(
        addr_idx: fx.Int64,
        addr_staging: fx.Int64,
        addr_dispatch_inp: fx.Int64,
        addr_chunk_flag: fx.Int64,
        addr_node_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        addr_inter_bar: fx.Int64,
        addr_inter_dest_map: fx.Int64,
        addr_inter_send_map: fx.Int64,
        addr_disp_dest_map: fx.Int64,
        addr_dest_pe_ctr: fx.Int64,
        addr_recv_sym: fx.Int64,
        addr_p2p_recv: fx.Int64,
        addr_total_recv: fx.Int64,
        addr_disp_grid_bar: fx.Int64,
        addr_combine_grid_bar: fx.Int64,
        addr_cross_dev_flag: fx.Int64,
        addr_disp_tok_off_sym: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_out_tok: fx.Int64,
        addr_p2p_out_idx: fx.Int64,
        addr_p2p_out_wts: fx.Int64,
        addr_p2p_out_scales: fx.Int64,
        addr_out_tok_local: fx.Int64,
        addr_out_idx_local: fx.Int64,
        addr_tis_local: fx.Int64,
        addr_packed_recv_x: fx.Int64,
        addr_packed_recv_count: fx.Int64,
        addr_packed_recv_src_info: fx.Int64,
        addr_disp_tok_map: fx.Int64,
        cur_tok: fx.Int32,
        dev_trace_level: fx.Int32,
        dev_trace_seq: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_bn, _wpb)
        kernel(
            addr_idx,
            addr_staging,
            addr_dispatch_inp,
            addr_chunk_flag,
            addr_node_recv,
            addr_block_flag,
            addr_inter_bar,
            addr_inter_dest_map,
            addr_inter_send_map,
            addr_disp_dest_map,
            addr_dest_pe_ctr,
            addr_recv_sym,
            addr_p2p_recv,
            addr_total_recv,
            addr_disp_grid_bar,
            addr_combine_grid_bar,
            addr_cross_dev_flag,
            addr_disp_tok_off_sym,
            addr_p2p_tok_off,
            addr_p2p_tis,
            addr_p2p_out_tok,
            addr_p2p_out_idx,
            addr_p2p_out_wts,
            addr_p2p_out_scales,
            addr_out_tok_local,
            addr_out_idx_local,
            addr_tis_local,
            addr_packed_recv_x,
            addr_packed_recv_count,
            addr_packed_recv_src_info,
            addr_disp_tok_map,
            cur_tok,
            dev_trace_level,
            dev_trace_seq,
        ).launch(grid=(_bn, 1, 1), block=(_wpb * 64, 1, 1), stream=stream)

    launch_main.compile_hints = {
        **launch_main.compile_hints,
        "internode_v1ll_jit_rev": 10,
    }
    return launch_main


def make_dispatch_internode_v1ll_copy_main_fused_jit(
    *,
    rank: int,
    npes: int,
    gpu_per_node: int,
    experts_per_rank: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    block_num: int,
    rdma_block_num: int,
    copy_grid_blocks: int,
    warp_num_per_block: int,
    num_qp_per_pe: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    enable_std_moe: bool = False,
    data_type=None,
):
    """Single ``@jit`` that enqueues copy-to-staging then main (same stream as mori ``_launch_multi``).

    Saves one host round-trip through the FlyDSL launch path vs separate ``copy_jit`` + ``main_jit``.
    """
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    copy_kernel = make_copy_to_staging_kernel(
        rank=rank,
        npes=npes,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        copy_grid_blocks=copy_grid_blocks,
        warp_num_per_block=warp_num_per_block,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        data_type=data_type,
    )
    main_kernel = make_dispatch_internode_v1ll_main_kernel(
        rank=rank,
        npes=npes,
        gpu_per_node=gpu_per_node,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_num_per_block=warp_num_per_block,
        num_qp_per_pe=num_qp_per_pe,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        enable_std_moe=enable_std_moe,
        data_type=data_type,
    )
    _cg, _wpb_c = copy_grid_blocks, warp_num_per_block
    _bn, _wpb_m = block_num, warp_num_per_block

    @flyc.jit
    def launch_copy_main_fused(
        addr_inp: fx.Int64,
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_scales: fx.Int64,
        addr_staging: fx.Int64,
        addr_dispatch_inp: fx.Int64,
        addr_chunk_flag: fx.Int64,
        addr_node_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        addr_inter_bar: fx.Int64,
        addr_inter_dest_map: fx.Int64,
        addr_inter_send_map: fx.Int64,
        addr_disp_dest_map: fx.Int64,
        addr_dest_pe_ctr: fx.Int64,
        addr_recv_sym: fx.Int64,
        addr_p2p_recv: fx.Int64,
        addr_total_recv: fx.Int64,
        addr_disp_grid_bar: fx.Int64,
        addr_combine_grid_bar: fx.Int64,
        addr_cross_dev_flag: fx.Int64,
        addr_disp_tok_off_sym: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_out_tok: fx.Int64,
        addr_p2p_out_idx: fx.Int64,
        addr_p2p_out_wts: fx.Int64,
        addr_p2p_out_scales: fx.Int64,
        addr_out_tok_local: fx.Int64,
        addr_out_idx_local: fx.Int64,
        addr_tis_local: fx.Int64,
        addr_packed_recv_x: fx.Int64,
        addr_packed_recv_count: fx.Int64,
        addr_packed_recv_src_info: fx.Int64,
        addr_disp_tok_map: fx.Int64,
        cur_tok: fx.Int32,
        dev_trace_level: fx.Int32,
        dev_trace_seq: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_cg, _wpb_c, _bn, _wpb_m)
        copy_kernel(addr_inp, addr_idx, addr_wts, addr_scales, addr_staging, cur_tok).launch(
            grid=(_cg, 1, 1),
            block=(_wpb_c * 64, 1, 1),
            stream=stream,
        )
        main_kernel(
            addr_idx,
            addr_staging,
            addr_dispatch_inp,
            addr_chunk_flag,
            addr_node_recv,
            addr_block_flag,
            addr_inter_bar,
            addr_inter_dest_map,
            addr_inter_send_map,
            addr_disp_dest_map,
            addr_dest_pe_ctr,
            addr_recv_sym,
            addr_p2p_recv,
            addr_total_recv,
            addr_disp_grid_bar,
            addr_combine_grid_bar,
            addr_cross_dev_flag,
            addr_disp_tok_off_sym,
            addr_p2p_tok_off,
            addr_p2p_tis,
            addr_p2p_out_tok,
            addr_p2p_out_idx,
            addr_p2p_out_wts,
            addr_p2p_out_scales,
            addr_out_tok_local,
            addr_out_idx_local,
            addr_tis_local,
            addr_packed_recv_x,
            addr_packed_recv_count,
            addr_packed_recv_src_info,
            addr_disp_tok_map,
            cur_tok,
            dev_trace_level,
            dev_trace_seq,
        ).launch(grid=(_bn, 1, 1), block=(_wpb_m * 64, 1, 1), stream=stream)

    launch_copy_main_fused.compile_hints = {
        **launch_copy_main_fused.compile_hints,
        "internode_v1ll_jit_rev": 10,
        "internode_v1ll_fused_copy_main": True,
    }
    return launch_copy_main_fused
