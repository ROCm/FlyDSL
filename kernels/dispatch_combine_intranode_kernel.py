# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL dispatch/combine intranode kernels."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_HERE, "../python"), "/home/yashao/FlyDSL/python",
           "/home/yashao/mori/python"]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr import arith
from flydsl.expr.typing import Stream
import torch

import mori.ir.flydsl as mori_shmem

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr import T

from flydsl.expr.rocdl import ballot_i64, readlane
from flydsl.expr.vector import bitcast_i32_to_v2bf16, bitcast_v2bf16_to_i32
from flydsl.expr.buffer_ops import (
    create_buffer_resource_from_addr,
    buffer_load,
    buffer_store,
)
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.ir import IntegerAttr as _IntAttr, IntegerType as _IntTy


def _lv_unwrap(v):
    """Extract raw ir.Value from DSL wrapper, ir.Value, or Python int."""
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
    """i64 → !llvm.ptr<1> (global address space)."""
    return _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), _lv_unwrap(v)).result


def store_i32_system(addr_i64, offset, val):
    """System-scope monotonic i32 store (global)."""
    base = _lv_unwrap(addr_i64)
    off  = _lv_unwrap(offset)
    val_ = _lv_unwrap(val)
    _i64 = _IntTy.get_signless(64)
    _i32 = _IntTy.get_signless(32)
    _nuw = _ir.Attribute.parse("#llvm.overflow<none>")
    off64 = _llvm_d.ZExtOp(_i64, off).res if off.type == _i32 else off
    byte_off = _llvm_d.MulOp(
        off64, _llvm_d.ConstantOp(_i64, _IntAttr.get(_i64, 4)).result, _nuw).result
    addr = _llvm_d.AddOp(base, byte_off, _nuw).result
    gptr = _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), addr).result
    _llvm_d.StoreOp(val_, gptr, alignment=4,
                    ordering=_llvm_d.AtomicOrdering.monotonic,
                    syncscope="one-as")


def store_i64_global_system(addr_i64, val):
    """System-scope monotonic i64 store (global)."""
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(_lv_unwrap(val), gptr, alignment=8,
                    ordering=_llvm_d.AtomicOrdering.monotonic,
                    syncscope="one-as")


def load_i64_global(addr_i64):
    """Global i64 load (relaxed)."""
    ptr = _to_ptr_global(addr_i64)
    _i64 = _IntTy.get_signless(64)
    return _llvm_d.LoadOp(_i64, ptr, alignment=8).result


def atomic_add_global_at(addr_i64, val):
    """Global atomic fetch-and-add (monotonic). Returns old value."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, _lv_unwrap(val),
        _llvm_d.AtomicOrdering.monotonic).res


def _to_idx(v):
    """ArithValue / Python int → index type."""
    if isinstance(v, int):
        return arith.index(v)
    return arith.index_cast(T.index(), v)

def _to_i32(v):
    """index → i32."""
    return arith.index_cast(T.i32(), v)

def _sel_pe(rem_list, dest_pe):
    """arith.select 链实现运行时动态索引。"""
    result = rem_list[-1]
    for pe in reversed(range(len(rem_list) - 1)):
        result = arith.select(arith.cmpi(arith.CmpIPredicate.eq, dest_pe, arith.constant(pe)), rem_list[pe], result)
    return result


def make_dispatch_kernel(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    enable_std_moe: bool = False,
    data_type=None,
):
    """创建 dispatch intranode @flyc.kernel。"""
    max_recv = npes * max_tok_per_rank
    _is_fp4 = (data_type == torch.float4_e2m1fn_x2)
    if _is_fp4:
        n_i32  = hidden_dim // 8   # 8 fp4 values per i32 (4 bytes)
        nbytes = hidden_dim // 2   # 2 fp4 values per byte
    else:
        n_i32  = (hidden_dim * hidden_elem_size) // 4
        nbytes = hidden_dim * hidden_elem_size
    scale_bytes    = scale_dim * scale_type_size
    scale_n_i32    = (scale_bytes + 3) // 4 if scale_bytes > 0 else 0
    enable_scales  = scale_bytes > 0
    max_tokens_per_expert = npes * max_tok_per_rank  # per-expert bucket capacity

    @flyc.kernel
    def ep_dispatch_intranode(
        addr_inp_tok:  fx.Int64,  # [cur_tok, hidden_dim]  bf16
        addr_idx:      fx.Int64,  # [cur_tok, k]           i32  (token_indices)
        addr_wts:      fx.Int64,  # [cur_tok, k]           f32  (weights_buf)
        addr_out_tok:  fx.Int64,  # shmem_out_tok
        addr_out_wts:  fx.Int64,  # shmem_out_wts
        addr_out_idx:  fx.Int64,  # shmem_out_idx
        addr_tok_off:  fx.Int64,  # shmem_tok_off (i32[1])
        addr_recv_num: fx.Int64,  # recv_tok_num  (i32[npes])
        addr_dest_ctr: fx.Int64,  # dest_pe_ctr   (i32[npes])
        addr_disp_bar: fx.Int64,  # dispatch_bar  (i32[1])
        addr_tok_map:  fx.Int64,  # dest_tok_map  (i32[cur_tok*k])
        addr_tis:      fx.Int64,  # tok_id_to_src (i32[max_recv])
        addr_total_rv: fx.Int64,  # total_recv    (i32[1])
        # P2P 地址数组：预计算的各 shmem buffer 对所有 PE 的远程地址 (i64[npes])
        addr_p2p_tok_off:  fx.Int64,
        addr_p2p_tis:      fx.Int64,
        addr_p2p_out_wts:  fx.Int64,
        addr_p2p_out_idx:  fx.Int64,
        addr_p2p_out_tok:  fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        addr_scales:       fx.Int64,  # 输入 scales buffer
        addr_p2p_out_scales: fx.Int64,  # scales P2P 地址数组 i64[npes]
        # ── StdMoE ConvertDispatchOutput 参数 ──
        addr_packed_recv_x:        fx.Int64,  # expert-major token buffer
        addr_packed_recv_count:    fx.Int64,  # per-expert token count (i32[experts_per_rank])
        addr_packed_recv_src_info: fx.Int64,  # source info (i32[experts_per_rank * max_tok_per_expert])
        addr_disp_tok_map:         fx.Int64,  # slot mapping (i64[max_recv * top_k])
        addr_disp_grid_bar:        fx.Int64,  # grid barrier (i32[1])
        cur_tok:       fx.Int32,  # 动态：本轮实际 token 数
    ):
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63
        warp   = tid >> 6
        gw_id  = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        limit  = cur_tok * experts_per_token
        _r_idx     = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
        _r_wts     = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
        _r_tok_map = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_map))
        _r_tok_off = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_off))
        _r_dest_ctr = create_buffer_resource_from_addr(_lv_unwrap(addr_dest_ctr))
        _r_disp_bar = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_bar))
        _r_total_rv = create_buffer_resource_from_addr(_lv_unwrap(addr_total_rv))
        _r_p2p_tok_off  = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tok_off))
        _r_p2p_tis      = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tis))
        _r_p2p_out_wts  = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_wts))
        _r_p2p_out_idx  = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_idx))
        _r_p2p_out_tok  = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_tok))
        _r_p2p_recv_num = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_recv_num))

        # Phase 1: 发送 token
        for i in range(_to_idx(gw_id), _to_idx(limit), _to_idx(gw_num)):
            i = _to_i32(i)
            src_tok  = arith.divui(i, experts_per_token)
            j        = arith.remui(i, experts_per_token)
            # 两个 idx load 并行发射, divui 延后到两个 load 之后
            dest_exp = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32())
            safe_lane    = arith.select(arith.cmpi(arith.CmpIPredicate.ult, lane, j), lane, arith.constant(0))
            lane_exp     = buffer_load(_r_idx, src_tok * experts_per_token + safe_lane, vec_width=1, dtype=T.i32())
            dest_pe      = arith.divui(dest_exp, experts_per_rank)
            lane_pe      = arith.divui(lane_exp, experts_per_rank)
            dup_per_lane = arith.select(
                arith.cmpi(arith.CmpIPredicate.eq, lane_pe, dest_pe),
                arith.select(arith.cmpi(arith.CmpIPredicate.ult, lane, j), lane, arith.constant(64)),
                arith.constant(64))
            dup_ballot   = ballot_i64(arith.cmpi(arith.CmpIPredicate.ult, dup_per_lane, arith.constant(64)))
            is_dup       = arith.cmpi(arith.CmpIPredicate.ne, dup_ballot, arith.constant(0, type=T.i64()))

            # 原子分配 destTokId: lane0 执行 atomic_add, readlane 广播结果
            from flydsl._mlir.dialects import scf as _scf_d
            from flydsl._mlir.ir import InsertionPoint as _IP
            _i32_ty = T.i32()
            _if_lane0 = _scf_d.IfOp(_lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0))),
                                     [_i32_ty], has_else=True)
            with _IP(_if_lane0.then_block):
                _if_nodup = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64()))),
                    [_i32_ty], has_else=True)
                with _IP(_if_nodup.then_block):
                    _old_tok = atomic_add_global_at(
                        buffer_load(_r_p2p_tok_off, dest_pe, vec_width=1, dtype=T.i64()),
                        arith.constant(1))
                    _scf_d.YieldOp([_lv_unwrap(_old_tok)])
                with _IP(_if_nodup.else_block):
                    _scf_d.YieldOp([_lv_unwrap(arith.constant(0))])
                _scf_d.YieldOp([_if_nodup.result])
            with _IP(_if_lane0.else_block):
                _scf_d.YieldOp([_lv_unwrap(arith.constant(0))])
            dest_tok_all = readlane(_if_lane0.result, 0)

            sentinel_val = npes * max_recv
            dtm_val = arith.select(is_dup, arith.constant(sentinel_val),
                                  dest_pe * max_recv + dest_tok_all)
            if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                buffer_store(_lv_unwrap(dtm_val), _r_tok_map, i)

            if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                    src_enc  = rank * max_tok_per_rank + src_tok
                    _r_tis_remote = create_buffer_resource_from_addr(
                        buffer_load(_r_p2p_tis, dest_pe, vec_width=1, dtype=T.i64()))
                    buffer_store(_lv_unwrap(src_enc), _r_tis_remote, dest_tok_all)
                    ctr_addr = addr_dest_ctr + arith.zext_i64(dest_pe) * 4
                    atomic_add_global_at(ctr_addr, arith.constant(1))

            if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(experts_per_token)):
                if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                    wt_src   = src_tok * experts_per_token + lane
                    wt_val   = buffer_load(_r_wts, wt_src, vec_width=1, dtype=T.f32())
                    ix_val   = buffer_load(_r_idx, wt_src, vec_width=1, dtype=T.i32())
                    dst_slot = dest_tok_all * experts_per_token + lane
                    _r_wts_remote = create_buffer_resource_from_addr(
                        buffer_load(_r_p2p_out_wts, dest_pe, vec_width=1, dtype=T.i64()))
                    buffer_store(_lv_unwrap(arith.bitcast(T.i32(), wt_val)), _r_wts_remote, dst_slot)
                    _r_idx_remote = create_buffer_resource_from_addr(
                        buffer_load(_r_p2p_out_idx, dest_pe, vec_width=1, dtype=T.i64()))
                    buffer_store(ix_val, _r_idx_remote, dst_slot)

            if enable_scales:
                if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(scale_n_i32)):
                    if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                        _r_scales = create_buffer_resource_from_addr(_lv_unwrap(addr_scales))
                        _sc_src_off = src_tok * scale_n_i32 + lane
                        _sc_val = buffer_load(_r_scales, _sc_src_off, vec_width=1, dtype=T.i32())
                        _sc_dst_off = dest_tok_all * scale_n_i32 + lane
                        _r_sc_remote = create_buffer_resource_from_addr(
                            buffer_load(
                                create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_out_scales)),
                                dest_pe, vec_width=1, dtype=T.i64()))
                        buffer_store(_lv_unwrap(_sc_val), _r_sc_remote, _sc_dst_off)

            # 写入 token embedding: is_dup 时 copy_end==lane4 使循环零迭代
            tok_remote = buffer_load(_r_p2p_out_tok, dest_pe, vec_width=1, dtype=T.i64()) + \
                arith.zext_i64(dest_tok_all) * nbytes
            inp_src_b  = addr_inp_tok + arith.zext_i64(src_tok) * nbytes
            rsrc_src = create_buffer_resource_from_addr(_lv_unwrap(inp_src_b))
            rsrc_dst = create_buffer_resource_from_addr(_lv_unwrap(tok_remote))
            lane4    = lane * 4
            _safe_disp_end = (n_i32 // 512) * 512
            if n_i32 >= 512 and _safe_disp_end > 0:
                safe_copy_end = arith.select(is_dup, lane4, arith.constant(_safe_disp_end))
                for ec4 in range(_to_idx(lane4), _to_idx(safe_copy_end), 512):
                    ec4      = _to_i32(ec4)
                    vec4_0   = buffer_load(rsrc_src, ec4, vec_width=4, dtype=T.i32())
                    vec4_1   = buffer_load(rsrc_src, ec4 + 256, vec_width=4, dtype=T.i32())
                    buffer_store(vec4_0, rsrc_dst, ec4)
                    buffer_store(vec4_1, rsrc_dst, ec4 + 256)
            if _safe_disp_end < n_i32:
                tail_copy_end = arith.select(is_dup, lane4, arith.constant(n_i32))
                for ec4 in range(_to_idx(lane4 + arith.constant(_safe_disp_end)), _to_idx(tail_copy_end), 256):
                    ec4      = _to_i32(ec4)
                    vec4_0   = buffer_load(rsrc_src, ec4, vec_width=4, dtype=T.i32())
                    buffer_store(vec4_0, rsrc_dst, ec4)
            elif n_i32 < 512:
                copy_end = arith.select(is_dup, lane4, arith.constant(n_i32))
                for ec4 in range(_to_idx(lane4), _to_idx(copy_end), 256):
                    ec4      = _to_i32(ec4)
                    vec4_0   = buffer_load(rsrc_src, ec4, vec_width=4, dtype=T.i32())
                    buffer_store(vec4_0, rsrc_dst, ec4)

        # Phase 2: 栅栏 + 发送 token 数量信号
        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
            atomic_add_global_at(addr_disp_bar, arith.constant(1))

        rtn_local_off = arith.zext_i64(arith.constant(rank)) * 4
        for dest_pe in range(_to_idx(lane), _to_idx(npes), 64):
            dest_pe = _to_i32(dest_pe)
            if arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0)):
                mori_shmem.int32_wait_until_equals(addr_disp_bar, block_num)
                buffer_store(_lv_unwrap(arith.constant(0)), _r_disp_bar, arith.constant(0))
                nsig       = buffer_load(_r_dest_ctr, dest_pe, vec_width=1, dtype=T.i32()) + 1
                rtn_remote = buffer_load(_r_p2p_recv_num, dest_pe, vec_width=1, dtype=T.i64()) + rtn_local_off
                mori_shmem.int32_wait_until_equals(rtn_remote, 0)
                store_i32_system(rtn_remote, arith.constant(0), nsig)

        # Phase 3: 接收信号，累计 total_recv
        for src_pe in range(_to_idx(lane), _to_idx(npes), 64):
            src_pe = _to_i32(src_pe)
            if arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0)):
                rtn_src  = addr_recv_num + arith.zext_i64(src_pe) * 4
                sig_val  = mori_shmem.int32_wait_until_greater_than(rtn_src, 0)
                recv_cnt = sig_val - 1
                store_i32_system(rtn_src, arith.constant(0), arith.constant(0))
                atomic_add_global_at(addr_total_rv, recv_cnt)
                buffer_store(_lv_unwrap(arith.constant(0)), _r_dest_ctr, src_pe)

        if arith.cmpi(arith.CmpIPredicate.eq, gw_id, arith.constant(0)):
            if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                buffer_store(_lv_unwrap(arith.constant(0)), _r_tok_off, arith.constant(0))

        # Phase 4: ConvertDispatchOutput (StdMoE)
        if enable_std_moe:
            from flydsl._mlir.dialects import scf as _scf_d
            from flydsl._mlir.ir import InsertionPoint as _IP
            _i32_ty = T.i32()

            fx.barrier()
            if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
                atomic_add_global_at(addr_disp_grid_bar, arith.constant(1))
            fx.barrier()
            if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
                mori_shmem.int32_wait_until_equals(addr_disp_grid_bar, block_num)
            fx.barrier()

            _r_out_idx_local = create_buffer_resource_from_addr(_lv_unwrap(addr_out_idx))
            _r_tis_local = create_buffer_resource_from_addr(_lv_unwrap(addr_tis))
            _r_out_tok_local = create_buffer_resource_from_addr(_lv_unwrap(addr_out_tok))
            total_recv_moe = buffer_load(_r_total_rv, arith.constant(0), vec_width=1, dtype=T.i32())
            limit_moe = total_recv_moe * experts_per_token

            for ii_idx in range(_to_idx(gw_id), _to_idx(limit_moe), _to_idx(gw_num)):
                ii = _to_i32(ii_idx)
                tok_idx_moe = arith.divui(ii, experts_per_token)

                exp_id = buffer_load(_r_out_idx_local, ii, vec_width=1, dtype=T.i32())
                local_exp = exp_id - arith.constant(rank * experts_per_rank)
                is_local = arith.cmpi(arith.CmpIPredicate.ult, local_exp,
                                      arith.constant(experts_per_rank))
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
                _slot_val_i64 = arith.select(is_local,
                    arith.zext_i64(linear_idx),
                    arith.constant(-1, type=T.i64()))
                if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                    _slot_addr = addr_disp_tok_map + arith.zext_i64(ii) * 8
                    store_i64_global_system(_slot_addr, _slot_val_i64)

                if arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(0)):
                    if is_local:
                        _src_pos = buffer_load(_r_tis_local, tok_idx_moe,
                                               vec_width=1, dtype=T.i32())
                        store_i32_system(addr_packed_recv_src_info,
                                         linear_idx, _src_pos)

                # WarpCopy token data
                _src_base = addr_out_tok + arith.zext_i64(tok_idx_moe) * nbytes
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

    return ep_dispatch_intranode


def make_combine_kernel(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int = 0,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
    data_type=None,
    enable_weights: bool = False,
    enable_std_moe: bool = False,
    use_p2p_read: bool = False,
):
    """创建 combine intranode @flyc.kernel。"""
    max_recv   = npes * max_tok_per_rank
    _is_fp4 = (data_type == torch.float4_e2m1fn_x2)
    if _is_fp4:
        n_i32  = hidden_dim // 8
        nbytes = hidden_dim // 2
    else:
        n_i32  = (hidden_dim * hidden_elem_size) // 4
        nbytes = hidden_dim * hidden_elem_size
    tok_stride = n_i32 * 4
    if _is_fp4:
        from flydsl._mlir.dialects import rocdl as _rocdl_d

        def _to_accum(i32_val):
            _v2f32 = T.VectorType.get([2], T.f32())
            _v8f32 = T.VectorType.get([8], T.f32())
            src = _lv_unwrap(i32_val)
            scale = _lv_unwrap(arith.constant(1.0, type=T.f32()))
            undef = _llvm_d.UndefOp(_v8f32).res
            vec = undef
            for sel in range(4):
                pair = _rocdl_d.cvt_scalef32_pk_f32_fp4(
                    res=_v2f32, src=src, scale=scale,
                    src_sel_index=sel)
                for q in range(2):
                    e = _llvm_d.ExtractElementOp(pair, _lv_unwrap(arith.constant(q))).res
                    vec = _llvm_d.InsertElementOp(vec, e, _lv_unwrap(arith.constant(sel * 2 + q))).res
            return vec

        def _from_accum(accum_val):
            _i32_ty = _IntTy.get_signless(32)
            src = _lv_unwrap(accum_val)
            scale = _lv_unwrap(arith.constant(1.0, type=T.f32()))
            old = _llvm_d.ConstantOp(_i32_ty, _IntAttr.get(_i32_ty, 0)).result
            for sel in range(4):
                f_a = _llvm_d.ExtractElementOp(src, _lv_unwrap(arith.constant(sel * 2))).res
                f_b = _llvm_d.ExtractElementOp(src, _lv_unwrap(arith.constant(sel * 2 + 1))).res
                old = _rocdl_d.cvt_scalef32_pk_fp4_f32(
                    res=_i32_ty, old_vdst=old, src0=f_a, src1=f_b,
                    scale=scale, dst_sel_index=sel)
            return old

        def _zero_accum():
            return arith.constant_vector(0.0, T.VectorType.get([8], T.f32()))

    elif hidden_elem_size == 2:  # bf16
        def _to_accum(i32_val):
            return bitcast_i32_to_v2bf16(i32_val).extf(
                T.VectorType.get([2], T.f32()))
        def _from_accum(accum_val):
            return bitcast_v2bf16_to_i32(accum_val.truncf(
                T.VectorType.get([2], T.bf16())))
        def _zero_accum():
            return arith.constant_vector(0.0, T.VectorType.get([2], T.f32()))
    elif hidden_elem_size == 4:  # f32
        def _to_accum(i32_val):
            return arith.bitcast(T.f32(), i32_val)
        def _from_accum(accum_val):
            return arith.bitcast(T.i32(), accum_val)
        def _zero_accum():
            return arith.constant(0.0, type=T.f32())
    elif hidden_elem_size == 1:  # fp8
        from flydsl._mlir.dialects import rocdl as _rocdl_d
        from flydsl._mlir.dialects import arith as _am_f8
        from flydsl._mlir.dialects import vector as _vd_f8
        _is_ocp = (data_type == torch.float8_e4m3fn)
        _is_fnuz = (data_type == torch.float8_e4m3fnuz)
        _cvt_pk_f32 = _rocdl_d.cvt_pk_f32_fp8
        _cvt_pk_f8  = _rocdl_d.cvt_pk_fp8_f32

        def _to_accum(i32_val):
            _v2f32 = T.VectorType.get([2], T.f32())
            _v4f32 = T.VectorType.get([4], T.f32())
            src = _lv_unwrap(i32_val)
            lo = _cvt_pk_f32(res=_v2f32, src=src, word_sel=False)
            hi = _cvt_pk_f32(res=_v2f32, src=src, word_sel=True)
            undef = _llvm_d.UndefOp(_v4f32).res
            vec = undef
            for i in range(2):
                e = _llvm_d.ExtractElementOp(lo, arith.constant(i)).res
                vec = _llvm_d.InsertElementOp(vec, e, arith.constant(i)).res
            for i in range(2):
                e = _llvm_d.ExtractElementOp(hi, arith.constant(i)).res
                vec = _llvm_d.InsertElementOp(vec, e, arith.constant(i + 2)).res
            if _is_fnuz:
                _ft = T.f32()
                _hf = _am_f8.ConstantOp(_ft, _ir.FloatAttr.get(_ft, 0.5)).result
                _hv = _vd_f8.BroadcastOp(_v4f32, _hf).result
                vec = _am_f8.MulFOp(vec, _hv).result
            return vec

        def _from_accum(accum_val):
            _i32_ty = _IntTy.get_signless(32)
            _v4f32 = T.VectorType.get([4], T.f32())
            src = _lv_unwrap(accum_val)
            if _is_fnuz:
                _ft = T.f32()
                _tf = _am_f8.ConstantOp(_ft, _ir.FloatAttr.get(_ft, 2.0)).result
                _tv = _vd_f8.BroadcastOp(_v4f32, _tf).result
                src = _am_f8.MulFOp(src, _tv).result
            f0 = _llvm_d.ExtractElementOp(src, arith.constant(0)).res
            f1 = _llvm_d.ExtractElementOp(src, arith.constant(1)).res
            f2 = _llvm_d.ExtractElementOp(src, arith.constant(2)).res
            f3 = _llvm_d.ExtractElementOp(src, arith.constant(3)).res
            zero = arith.constant(0, type=_i32_ty)
            lo = _cvt_pk_f8(res=_i32_ty, src_a=f0, src_b=f1,
                            old=_lv_unwrap(zero), word_sel=False)
            return _cvt_pk_f8(res=_i32_ty, src_a=f2, src_b=f3,
                              old=lo, word_sel=True)

        def _zero_accum():
            return arith.constant_vector(
                0.0, T.VectorType.get([4], T.f32()))
    else:
        raise ValueError(f"Unsupported hidden_elem_size={hidden_elem_size}")

    def _accum_experts(vals, vlds, all_vld):
        """Accumulate expert values → single i32."""
        if all_vld:
            acc = _to_accum(vals[0])
            for j in range(1, len(vals)):
                acc = acc + _to_accum(vals[j])
        else:
            acc = _zero_accum()
            for j in range(len(vals)):
                fa = _to_accum(vals[j])
                z = _zero_accum()
                vld = _lv_unwrap(vlds[j])
                acc = acc + arith.select(vld, fa, z)
        return _from_accum(acc)

    def _weighted_accum_experts(vals, wts, vlds, all_vld):
        """Weighted accumulate: sum(wt[k] * to_accum(val[k])) → i32."""
        from flydsl._mlir.dialects import arith as _am
        from flydsl._mlir.dialects import vector as _vd
        _i32ty = _IntTy.get_signless(32)
        _f32ty = T.f32()
        _zero_f = _am.ConstantOp(_f32ty, _ir.FloatAttr.get(_f32ty, 0.0)).result

        if _is_fp4:  # fp4 → v8f32 accum
            from flydsl._mlir.dialects import rocdl as _rocdl_fp4w
            _v2f32_w = T.VectorType.get([2], T.f32())
            _v8f32_w = T.VectorType.get([8], T.f32())
            acc = _vd.BroadcastOp(_v8f32_w, _zero_f).result
            for j in range(len(vals)):
                vr = _lv_unwrap(vals[j])
                scale_v = _am.ConstantOp(_f32ty, _ir.FloatAttr.get(_f32ty, 1.0)).result
                undef = _llvm_d.UndefOp(_v8f32_w).res
                vec = undef
                for sel in range(4):
                    pair = _rocdl_fp4w.cvt_scalef32_pk_f32_fp4(
                        res=_v2f32_w, src=vr, scale=scale_v,
                        src_sel_index=sel)
                    for q in range(2):
                        e = _llvm_d.ExtractElementOp(pair, _lv_unwrap(arith.constant(q))).res
                        vec = _llvm_d.InsertElementOp(vec, e, _lv_unwrap(arith.constant(sel * 2 + q))).res
                ws = _vd.BroadcastOp(_v8f32_w, _lv_unwrap(wts[j])).result
                w = _am.MulFOp(vec, ws).result
                if all_vld:
                    acc = _am.AddFOp(acc, w).result
                else:
                    z = _vd.BroadcastOp(_v8f32_w, _zero_f).result
                    s = _am.SelectOp(_lv_unwrap(vlds[j]), w, z).result
                    acc = _am.AddFOp(acc, s).result
            scale_v = _am.ConstantOp(_f32ty, _ir.FloatAttr.get(_f32ty, 1.0)).result
            old = _am.ConstantOp(_i32ty, _IntAttr.get(_i32ty, 0)).result
            for sel in range(4):
                f_a = _llvm_d.ExtractElementOp(acc, _lv_unwrap(arith.constant(sel * 2))).res
                f_b = _llvm_d.ExtractElementOp(acc, _lv_unwrap(arith.constant(sel * 2 + 1))).res
                old = _rocdl_fp4w.cvt_scalef32_pk_fp4_f32(
                    res=_i32ty, old_vdst=old, src0=f_a, src1=f_b,
                    scale=scale_v, dst_sel_index=sel)
            return old

        elif hidden_elem_size == 2:  # bf16 → v2f32 accum
            _v2bf16 = T.VectorType.get([2], T.bf16())
            _v2f32 = T.VectorType.get([2], T.f32())
            acc = _vd.BroadcastOp(_v2f32, _zero_f).result
            for j in range(len(vals)):
                vr = _lv_unwrap(vals[j])
                vb = _llvm_d.BitcastOp(_v2bf16, vr).res
                vf = _am.ExtFOp(_v2f32, vb).result
                ws = _vd.BroadcastOp(_v2f32, _lv_unwrap(wts[j])).result
                w = _am.MulFOp(vf, ws).result
                if all_vld:
                    acc = _am.AddFOp(acc, w).result
                else:
                    z = _vd.BroadcastOp(_v2f32, _zero_f).result
                    s = _am.SelectOp(_lv_unwrap(vlds[j]), w, z).result
                    acc = _am.AddFOp(acc, s).result
            return _llvm_d.BitcastOp(_i32ty, _am.TruncFOp(_v2bf16, acc).result).res

        elif hidden_elem_size == 4:  # f32 → f32 accum
            acc = _zero_f
            for j in range(len(vals)):
                vf = _am.BitcastOp(_f32ty, _lv_unwrap(vals[j])).result
                w = _am.MulFOp(vf, _lv_unwrap(wts[j])).result
                if all_vld:
                    acc = _am.AddFOp(acc, w).result
                else:
                    s = _am.SelectOp(_lv_unwrap(vlds[j]), w, _zero_f).result
                    acc = _am.AddFOp(acc, s).result
            return _am.BitcastOp(_i32ty, acc).result

        elif hidden_elem_size == 1:  # fp8 → v4f32 accum
            from flydsl._mlir.dialects import rocdl as _rocdl
            _pk_f32_w = _rocdl.cvt_pk_f32_fp8
            _pk_f8_w  = _rocdl.cvt_pk_fp8_f32
            _v2f32 = T.VectorType.get([2], T.f32())
            _v4f32 = T.VectorType.get([4], T.f32())
            acc = _vd.BroadcastOp(_v4f32, _zero_f).result
            for j in range(len(vals)):
                vr = _lv_unwrap(vals[j])
                lo = _pk_f32_w(res=_v2f32, src=vr, word_sel=False)
                hi = _pk_f32_w(res=_v2f32, src=vr, word_sel=True)
                undef = _llvm_d.UndefOp(_v4f32).res
                vec = undef
                for q in range(2):
                    e = _llvm_d.ExtractElementOp(lo, _lv_unwrap(arith.constant(q))).res
                    vec = _llvm_d.InsertElementOp(vec, e, _lv_unwrap(arith.constant(q))).res
                for q in range(2):
                    e = _llvm_d.ExtractElementOp(hi, _lv_unwrap(arith.constant(q))).res
                    vec = _llvm_d.InsertElementOp(vec, e, _lv_unwrap(arith.constant(q + 2))).res
                if _is_fnuz:
                    _hf_w = _am.ConstantOp(_f32ty, _ir.FloatAttr.get(_f32ty, 0.5)).result
                    _hv_w = _vd.BroadcastOp(_v4f32, _hf_w).result
                    vec = _am.MulFOp(vec, _hv_w).result
                ws = _vd.BroadcastOp(_v4f32, _lv_unwrap(wts[j])).result
                w = _am.MulFOp(vec, ws).result
                if all_vld:
                    acc = _am.AddFOp(acc, w).result
                else:
                    z = _vd.BroadcastOp(_v4f32, _zero_f).result
                    s = _am.SelectOp(_lv_unwrap(vlds[j]), w, z).result
                    acc = _am.AddFOp(acc, s).result
            if _is_fnuz:
                _tf_w = _am.ConstantOp(_f32ty, _ir.FloatAttr.get(_f32ty, 2.0)).result
                _tv_w = _vd.BroadcastOp(_v4f32, _tf_w).result
                acc = _am.MulFOp(acc, _tv_w).result
            f0 = _llvm_d.ExtractElementOp(acc, _lv_unwrap(arith.constant(0))).res
            f1 = _llvm_d.ExtractElementOp(acc, _lv_unwrap(arith.constant(1))).res
            f2 = _llvm_d.ExtractElementOp(acc, _lv_unwrap(arith.constant(2))).res
            f3 = _llvm_d.ExtractElementOp(acc, _lv_unwrap(arith.constant(3))).res
            zi = _lv_unwrap(arith.constant(0))
            lo = _pk_f8_w(res=_i32ty, src_a=f0, src_b=f1, old=zi, word_sel=False)
            return _pk_f8_w(res=_i32ty, src_a=f2, src_b=f3, old=lo, word_sel=True)

    def _log2_if_pow2(v):
        """v 是 2 的幂则返回 log2(v), 否则 None。"""
        if v > 0 and (v & (v - 1)) == 0:
            return v.bit_length() - 1
        return None
    _log2_max_tok = _log2_if_pow2(max_tok_per_rank)
    _log2_max_recv = _log2_if_pow2(max_recv)
    _mask_max_tok = max_tok_per_rank - 1 if _log2_max_tok is not None else None
    _mask_max_recv = max_recv - 1 if _log2_max_recv is not None else None

    _use_compaction = (npes < experts_per_token)

    def _maybe_load(rsrc, offset, vld_flag, **kwargs):
        """Conditional load: returns loaded value or 0 when invalid."""
        if not _use_compaction:
            return buffer_load(rsrc, offset, **kwargs)
        from flydsl._mlir.dialects import scf as _scf_cl
        from flydsl._mlir.ir import InsertionPoint as _IP_cl
        _i32_ty = T.i32()
        _if_v = _scf_cl.IfOp(_lv_unwrap(vld_flag), [_i32_ty], has_else=True)
        with _IP_cl(_if_v.then_block):
            _v = buffer_load(rsrc, offset, **kwargs)
            _scf_cl.YieldOp([_lv_unwrap(_v)])
        with _IP_cl(_if_v.else_block):
            _scf_cl.YieldOp([_lv_unwrap(arith.constant(0))])
        return _if_v.result

    weight_bytes = experts_per_token * 4 if enable_weights else 0
    wt_n_i32     = experts_per_token if enable_weights else 0

    # LDS P2P 基地址表
    allocator = SmemAllocator(None, arch="gfx942")
    p2p_base_offset = allocator._align(allocator.ptr, 8)
    p2p_base_size = npes * 8
    allocator.ptr = p2p_base_offset + p2p_base_size

    if enable_weights:
        p2p_wt_base_offset = allocator._align(allocator.ptr, 8)
        p2p_wt_base_size = npes * 8
        allocator.ptr = p2p_wt_base_offset + p2p_wt_base_size


    @flyc.kernel
    def ep_combine_intranode(
        addr_inp_tok:  fx.Int64,   # inp_tok  基地址（expert 处理后的 token）
        addr_comb_inp: fx.Int64,   # shmem_comb_inp 基地址（symmetric）
        addr_comb_out: fx.Int64,   # shmem_comb_out 基地址（symmetric）
        addr_xdb_mem:  fx.Int64,   # xdev_bar_mem   基地址（u64[npes]）
        addr_xdb_flag: fx.Int64,   # xdev_bar_flag  基地址（u64[1]）
        addr_tok_map:  fx.Int64,   # dest_tok_map   基地址（i32[cur_tok*k]）
        addr_comb_bar: fx.Int64,   # combine_bar    基地址（i32[1]）
        addr_trecv:    fx.Int64,   # total_recv_ptr 基地址（i32[1]）
        addr_tis:      fx.Int64,   # tok_id_to_src  基地址（i32[max_recv]，symmetric）
        addr_p2p_comb_inp: fx.Int64,  # 预计算 P2P 地址数组 i64[npes]
        addr_p2p_xdb_mem:  fx.Int64,  # 预计算 P2P 地址数组 i64[npes]
        addr_wts_buf:  fx.Int64,      # combine 输入权重 float32[max_recv * k]
        addr_comb_inp_wts: fx.Int64,  # shmem 权重 P2P buffer（symmetric）
        addr_comb_out_wts: fx.Int64,  # 累加输出权重 float32[max_tok * k]
        addr_p2p_comb_inp_wts: fx.Int64,  # 权重 P2P 地址数组 i64[npes]
        # ── StdMoE ConvertCombineInput 参数 ──
        addr_packed_recv_x:  fx.Int64,  # expert-major token buffer (post-expert)
        addr_disp_tok_map:   fx.Int64,  # dispTokToEpSlotMap (i64[max_recv * top_k])
        addr_disp_out_wts:   fx.Int64,  # dispatch output weights (f32[max_recv * top_k])
        cur_rank_num_token:  fx.Int32,  # 本 PE 的输出 token 数（Stage 3 用）
    ):
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63
        warp   = tid >> 6
        gw_id  = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        gwtid  = bid * (warp_num_per_block * 64) + tid

        _r_trecv     = create_buffer_resource_from_addr(_lv_unwrap(addr_trecv))
        _r_xdb_flag  = create_buffer_resource_from_addr(_lv_unwrap(addr_xdb_flag))
        _r_tis       = create_buffer_resource_from_addr(_lv_unwrap(addr_tis))
        _r_comb_bar  = create_buffer_resource_from_addr(_lv_unwrap(addr_comb_bar))
        _r_p2p_comb  = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_comb_inp))
        _r_p2p_xdb   = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_xdb_mem))
        _rsrc_tok_map = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_map))

        total_recv_val = buffer_load(_r_trecv, arith.constant(0), vec_width=1, dtype=T.i32())
        cur_flag = buffer_load(_r_xdb_flag, arith.constant(0), vec_width=1, dtype=T.i64())

        # LDS P2P 基地址表
        base_ptr = allocator.get_base()
        _lds_p2p_bases = SmemPtr(base_ptr, p2p_base_offset, T.i64(),
                                 shape=(npes,))
        _lds_p2p_bases.get()

        if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(npes)):
            _p2p_base = buffer_load(_r_p2p_comb, lane, vec_width=1, dtype=T.i64())
            _lds_p2p_bases.store(_p2p_base, [_to_idx(lane)])

        if enable_weights:
            _r_p2p_comb_wt = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_comb_inp_wts))
            _lds_p2p_wt_bases = SmemPtr(base_ptr, p2p_wt_base_offset, T.i64(),
                                        shape=(npes,))
            _lds_p2p_wt_bases.get()
            if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(npes)):
                _p2p_wt_base = buffer_load(_r_p2p_comb_wt, lane, vec_width=1, dtype=T.i64())
                _lds_p2p_wt_bases.store(_p2p_wt_base, [_to_idx(lane)])

        fx.barrier()

        # Stage 1: P2P scatter / ConvertCombineInput
        n_chunks = nbytes // 16

        if enable_std_moe:
            _rsrc_dtm = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_tok_map))
            _rsrc_dow = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_out_wts))
            _all_vld_moe = False

            for tok_i in range(_to_idx(gw_id), _to_idx(total_recv_val), _to_idx(gw_num)):
                tok_i = _to_i32(tok_i)
                dest_enc = buffer_load(_r_tis, tok_i, vec_width=1, dtype=T.i32())
                if _log2_max_tok is not None:
                    dest_pe  = dest_enc >> arith.constant(_log2_max_tok)
                    dest_lid = dest_enc & arith.constant(_mask_max_tok)
                else:
                    dest_pe  = arith.divui(dest_enc, max_tok_per_rank)
                    dest_lid = arith.remui(dest_enc, max_tok_per_rank)

                if use_p2p_read:
                    _dest_off  = arith.zext_i64(tok_i) * nbytes
                    _dest_base = _lv_unwrap(addr_comb_inp) + _dest_off
                else:
                    _pe_base   = _lds_p2p_bases.load([_to_idx(dest_pe)])
                    _dest_off  = arith.zext_i64(arith.constant(rank) * max_tok_per_rank + dest_lid) * nbytes
                    _dest_base = _lv_unwrap(_pe_base) + _dest_off
                _rsrc_dst  = create_buffer_resource_from_addr(_dest_base)

                _exp_rsrcs = []
                _exp_vlds  = []
                _exp_wts   = []
                for _kp in range_constexpr(experts_per_token):
                    _slot_addr = addr_disp_tok_map + arith.zext_i64(tok_i * experts_per_token + _kp) * 8
                    _slot = load_i64_global(_slot_addr)
                    _vld = arith.cmpi(arith.CmpIPredicate.ne, _slot, arith.constant(-1, type=T.i64()))
                    _safe_slot_i64 = arith.select(_vld, _slot, arith.constant(0, type=T.i64()))
                    _exp_base = addr_packed_recv_x + _safe_slot_i64 * nbytes
                    _exp_rsrcs.append(create_buffer_resource_from_addr(_lv_unwrap(_exp_base)))
                    _exp_vlds.append(_vld)
                    _wt = buffer_load(_rsrc_dow, tok_i * experts_per_token + _kp,
                                      vec_width=1, dtype=T.f32())
                    _exp_wts.append(_wt)

                for _ec in range(_to_idx(lane), _to_idx(n_i32), _to_idx(64)):
                    _ec = _to_i32(_ec)
                    _moe_vals = []
                    for _kp2 in range_constexpr(experts_per_token):
                        _moe_vals.append(buffer_load(_exp_rsrcs[_kp2], _ec,
                                                     vec_width=1, dtype=T.i32()))
                    _res_raw = _weighted_accum_experts(_moe_vals, _exp_wts,
                                                       _exp_vlds, _all_vld_moe)
                    buffer_store(_res_raw, _rsrc_dst, _ec)

                if enable_weights:
                    if use_p2p_read:
                        _wt_dest_off = arith.zext_i64(tok_i) * weight_bytes
                        _wt_dest     = _lv_unwrap(addr_comb_inp_wts) + _wt_dest_off
                    else:
                        _wt_pe_base  = _lds_p2p_wt_bases.load([_to_idx(dest_pe)])
                        _wt_dest_off = arith.zext_i64(
                            arith.constant(rank) * max_tok_per_rank + dest_lid) * weight_bytes
                        _wt_dest     = _lv_unwrap(_wt_pe_base) + _wt_dest_off
                    _wt_src      = _lv_unwrap(addr_wts_buf) + arith.zext_i64(tok_i) * weight_bytes
                    _rsrc_wt_s   = create_buffer_resource_from_addr(_wt_src)
                    _rsrc_wt_d   = create_buffer_resource_from_addr(_wt_dest)
                    if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(wt_n_i32)):
                        _wv = buffer_load(_rsrc_wt_s, lane, vec_width=1, dtype=T.i32())
                        buffer_store(_lv_unwrap(_wv), _rsrc_wt_d, lane)

        elif use_p2p_read:
            _safe_dual_end = (n_chunks // 128) * 128
            for tok_i in range(_to_idx(gw_id), _to_idx(total_recv_val), _to_idx(gw_num)):
                tok_i    = _to_i32(tok_i)
                _src_base = addr_inp_tok + arith.zext_i64(tok_i) * nbytes
                _dst_base = addr_comb_inp + arith.zext_i64(tok_i) * nbytes
                _rsrc_src = create_buffer_resource_from_addr(_lv_unwrap(_src_base))
                _rsrc_dst = create_buffer_resource_from_addr(_lv_unwrap(_dst_base))
                if _safe_dual_end >= 128:
                    for cj in range(_to_idx(lane), _to_idx(_safe_dual_end), _to_idx(128)):
                        cj       = _to_i32(cj)
                        cj_elem  = cj * 4
                        cj2_elem = (cj + 64) * 4
                        vec4_a   = buffer_load(_rsrc_src, cj_elem, vec_width=4, dtype=T.i32())
                        vec4_b   = buffer_load(_rsrc_src, cj2_elem, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_a, _rsrc_dst, cj_elem)
                        buffer_store(vec4_b, _rsrc_dst, cj2_elem)
                if _safe_dual_end < n_chunks:
                    for cj in range(_to_idx(lane + arith.constant(_safe_dual_end)), _to_idx(n_chunks), _to_idx(64)):
                        cj      = _to_i32(cj)
                        cj_elem = cj * 4
                        vec4_a  = buffer_load(_rsrc_src, cj_elem, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_a, _rsrc_dst, cj_elem)

            if enable_weights:
                for tok_i in range(_to_idx(gw_id), _to_idx(total_recv_val), _to_idx(gw_num)):
                    tok_i = _to_i32(tok_i)
                    _wt_src = _lv_unwrap(addr_wts_buf) + arith.zext_i64(tok_i) * weight_bytes
                    _wt_dst = _lv_unwrap(addr_comb_inp_wts) + arith.zext_i64(tok_i) * weight_bytes
                    _rsrc_wt_s = create_buffer_resource_from_addr(_wt_src)
                    _rsrc_wt_d = create_buffer_resource_from_addr(_wt_dst)
                    if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(wt_n_i32)):
                        _wv = buffer_load(_rsrc_wt_s, lane, vec_width=1, dtype=T.i32())
                        buffer_store(_lv_unwrap(_wv), _rsrc_wt_d, lane)

        else:
            _safe_dual_end_s1 = (n_chunks // 128) * 128
            for tok_i in range(_to_idx(gw_id), _to_idx(total_recv_val), _to_idx(gw_num)):
                tok_i    = _to_i32(tok_i)
                dest_enc = buffer_load(_r_tis, tok_i, vec_width=1, dtype=T.i32())
                if _log2_max_tok is not None:
                    dest_pe  = dest_enc >> arith.constant(_log2_max_tok)
                    dest_lid = dest_enc & arith.constant(_mask_max_tok)
                else:
                    dest_pe  = arith.divui(dest_enc, max_tok_per_rank)
                    dest_lid = arith.remui(dest_enc, max_tok_per_rank)
                _pe_base   = _lds_p2p_bases.load([_to_idx(dest_pe)])
                _dest_off  = arith.zext_i64(arith.constant(rank) * max_tok_per_rank + dest_lid) * nbytes
                _dest_base = _lv_unwrap(_pe_base) + _dest_off
                _src_base  = addr_inp_tok + arith.zext_i64(tok_i) * nbytes
                _rsrc_src  = create_buffer_resource_from_addr(_lv_unwrap(_src_base))
                _rsrc_dst  = create_buffer_resource_from_addr(_dest_base)
                if _safe_dual_end_s1 >= 128:
                    for cj in range(_to_idx(lane), _to_idx(_safe_dual_end_s1), _to_idx(128)):
                        cj       = _to_i32(cj)
                        cj_elem  = cj * 4
                        cj2_elem = (cj + 64) * 4
                        vec4_a   = buffer_load(_rsrc_src, cj_elem, vec_width=4, dtype=T.i32())
                        vec4_b   = buffer_load(_rsrc_src, cj2_elem, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_a, _rsrc_dst, cj_elem)
                        buffer_store(vec4_b, _rsrc_dst, cj2_elem)
                if _safe_dual_end_s1 < n_chunks:
                    for cj in range(_to_idx(lane + arith.constant(_safe_dual_end_s1)), _to_idx(n_chunks), _to_idx(64)):
                        cj      = _to_i32(cj)
                        cj_elem = cj * 4
                        vec4_a  = buffer_load(_rsrc_src, cj_elem, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_a, _rsrc_dst, cj_elem)

                if enable_weights:
                    _wt_pe_base  = _lds_p2p_wt_bases.load([_to_idx(dest_pe)])
                    _wt_dest_off = arith.zext_i64(
                        arith.constant(rank) * max_tok_per_rank + dest_lid) * weight_bytes
                    _wt_dest     = _lv_unwrap(_wt_pe_base) + _wt_dest_off
                    _wt_src      = _lv_unwrap(addr_wts_buf) + arith.zext_i64(tok_i) * weight_bytes
                    _rsrc_wt_s   = create_buffer_resource_from_addr(_wt_src)
                    _rsrc_wt_d   = create_buffer_resource_from_addr(_wt_dest)
                    if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(wt_n_i32)):
                        _wv = buffer_load(_rsrc_wt_s, lane, vec_width=1, dtype=T.i32())
                        buffer_store(_lv_unwrap(_wv), _rsrc_wt_d, lane)

        # Stage 2: CrossDeviceBarrier
        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
            atomic_add_global_at(addr_comb_bar, arith.constant(1))

        if arith.cmpi(arith.CmpIPredicate.ult, gwtid, arith.constant(npes)):
            mori_shmem.int32_wait_until_equals(addr_comb_bar, block_num)
            buffer_store(_lv_unwrap(arith.constant(0)), _r_comb_bar, arith.constant(0))
            xdb_remote = buffer_load(_r_p2p_xdb, gwtid, vec_width=1, dtype=T.i64()) + \
                arith.zext_i64(arith.constant(rank)) * 8
            store_i64_global_system(xdb_remote, cur_flag)

        if arith.cmpi(arith.CmpIPredicate.eq, gwtid, arith.constant(0)):
            atomic_add_global_at(addr_xdb_flag, arith.constant(1, type=T.i64()))

        if arith.cmpi(arith.CmpIPredicate.ult, tid, arith.constant(npes)):
            peer_slot = addr_xdb_mem + arith.zext_i64(tid) * 8
            mori_shmem.uint64_wait_until_equals(peer_slot, cur_flag)

        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, arith.constant(0)):
            buffer_store(_lv_unwrap(arith.constant(0)), _r_trecv, arith.constant(0))

        # Stage 3: 本地读 + WarpAccum
        from flydsl._mlir.dialects import scf as _scf_d
        from flydsl._mlir.ir import InsertionPoint as _IP
        _SLC      = 2
        _rsrc_out = create_buffer_resource_from_addr(_lv_unwrap(addr_comb_out))

        n_elems = n_i32
        _safe_out = arith.select(
            arith.cmpi(arith.CmpIPredicate.eq, cur_rank_num_token, arith.constant(0)), arith.constant(1), cur_rank_num_token)
        wpt    = (gw_num + _safe_out - 1) // _safe_out
        hpw    = (n_elems + wpt - 1) // wpt
        s3_lim2   = cur_rank_num_token * wpt

        for si in range(_to_idx(gw_id), _to_idx(s3_lim2), _to_idx(gw_num)):
            si      = _to_i32(si)
            tok_id  = arith.divui(si, wpt)
            part_id = arith.remui(si, wpt)
            h_off   = part_id * hpw

            _tm_off_elem = tok_id * experts_per_token
            _tm_vec0     = buffer_load(_rsrc_tok_map, _tm_off_elem, vec_width=4, dtype=T.i32())
            _tm_vec1     = buffer_load(_rsrc_tok_map, _tm_off_elem + 4, vec_width=4, dtype=T.i32())

            _expert_rsrc = []
            _expert_vld  = []
            for j_py in range_constexpr(experts_per_token):
                if j_py < 4:
                    enc_j = _llvm_d.ExtractElementOp(
                        _tm_vec0, arith.constant(j_py)).res
                else:
                    enc_j = _llvm_d.ExtractElementOp(
                        _tm_vec1, arith.constant(j_py - 4)).res
                if _log2_max_recv is not None:
                    dest_pe_j = enc_j >> arith.constant(_log2_max_recv)
                else:
                    dest_pe_j = arith.divui(enc_j, max_recv)
                vld_j     = arith.cmpi(arith.CmpIPredicate.ult, dest_pe_j, arith.constant(npes))
                safe_pe   = arith.select(vld_j, dest_pe_j, arith.constant(rank))
                if use_p2p_read:
                    _dtok_all  = arith.remui(enc_j, max_recv)
                    _safe_ta   = arith.select(vld_j, _dtok_all, arith.constant(0))
                    _pe_base   = _lds_p2p_bases.load([_to_idx(safe_pe)])
                    _tok_off   = arith.zext_i64(_safe_ta) * nbytes
                    _ebase     = _lv_unwrap(_pe_base) + _tok_off
                else:
                    _tok_off  = arith.zext_i64(safe_pe * max_tok_per_rank + tok_id) * nbytes
                    _ebase    = _lv_unwrap(addr_comb_inp + _tok_off)
                _expert_rsrc.append(create_buffer_resource_from_addr(_ebase))
                _expert_vld.append(vld_j)

            _all_vld  = (npes >= experts_per_token)
            _eff_all_vld = _all_vld or _use_compaction

            _use_wide = arith.cmpi(arith.CmpIPredicate.ult, arith.constant(895), hpw)
            _if_wide  = _scf_d.IfOp(_lv_unwrap(_use_wide), [], has_else=True)

            # step=128 路径 (hpw > 895)
            with _IP(_if_wide.then_block):
                _n_rem_128   = arith.constant(n_elems) - h_off
                _adj_end_128 = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, _n_rem_128, hpw), _n_rem_128, hpw)

                if n_i32 % 256 == 0 and warp_num_per_block < 16:
                    _hpw_rem256  = arith.remui(hpw, arith.constant(256))
                    _is_256aln   = arith.cmpi(arith.CmpIPredicate.ult, _hpw_rem256, arith.constant(1))
                    _if_256aln   = _scf_d.IfOp(_lv_unwrap(_is_256aln), [], has_else=True)
                    with _IP(_if_256aln.then_block):
                      _quad_end_128 = _adj_end_128 - arith.constant(192)
                      for ec in range(_to_idx(lane), _to_idx(_quad_end_128), _to_idx(256)):
                        ec       = _to_i32(ec)
                        glob_ec  = h_off + ec
                        _va, _vb, _vc, _vd = [], [], [], []
                        for _j_py in range_constexpr(experts_per_token):
                            _rj = _expert_rsrc[_j_py]
                            _vj = _expert_vld[_j_py]
                            _va.append(_maybe_load(_rj, glob_ec, _vj, vec_width=1, dtype=T.i32(), cache_modifier=_SLC))
                            _vb.append(_maybe_load(_rj, glob_ec, _vj, vec_width=1, dtype=T.i32(), cache_modifier=_SLC, soffset_bytes=256))
                            _vc.append(_maybe_load(_rj, glob_ec, _vj, vec_width=1, dtype=T.i32(), cache_modifier=_SLC, soffset_bytes=512))
                            _vd.append(_maybe_load(_rj, glob_ec, _vj, vec_width=1, dtype=T.i32(), cache_modifier=_SLC, soffset_bytes=768))
                        _i32a = _accum_experts(_va, _expert_vld, _eff_all_vld)
                        _i32b = _accum_experts(_vb, _expert_vld, _eff_all_vld)
                        _i32c = _accum_experts(_vc, _expert_vld, _eff_all_vld)
                        _i32d = _accum_experts(_vd, _expert_vld, _eff_all_vld)
                        _out_elem = tok_id * n_i32 + glob_ec
                        buffer_store(_lv_unwrap(_i32a), _rsrc_out, _out_elem, cache_modifier=_SLC)
                        buffer_store(_lv_unwrap(_i32b), _rsrc_out, _out_elem, cache_modifier=_SLC, soffset_bytes=256)
                        buffer_store(_lv_unwrap(_i32c), _rsrc_out, _out_elem, cache_modifier=_SLC, soffset_bytes=512)
                        buffer_store(_lv_unwrap(_i32d), _rsrc_out, _out_elem, cache_modifier=_SLC, soffset_bytes=768)
                      _scf_d.YieldOp([])
                    with _IP(_if_256aln.else_block):
                      _dual_end_safe = arith.divui(_adj_end_128, arith.constant(128)) * 128
                      for ec in range(_to_idx(lane), _to_idx(_dual_end_safe), _to_idx(128)):
                        ec       = _to_i32(ec)
                        glob_ec  = h_off + ec
                        _va, _vb = [], []
                        for _j_py in range_constexpr(experts_per_token):
                            _rj = _expert_rsrc[_j_py]
                            _vj = _expert_vld[_j_py]
                            _va.append(_maybe_load(_rj, glob_ec, _vj, vec_width=1, dtype=T.i32(), cache_modifier=_SLC))
                            _vb.append(_maybe_load(_rj, glob_ec, _vj, vec_width=1, dtype=T.i32(), cache_modifier=_SLC, soffset_bytes=256))
                        _i32a = _accum_experts(_va, _expert_vld, _eff_all_vld)
                        _i32b = _accum_experts(_vb, _expert_vld, _eff_all_vld)
                        _out_elem = tok_id * n_i32 + glob_ec
                        buffer_store(_lv_unwrap(_i32a), _rsrc_out, _out_elem, cache_modifier=_SLC)
                        buffer_store(_lv_unwrap(_i32b), _rsrc_out, _out_elem, cache_modifier=_SLC, soffset_bytes=256)
                      for ec in range(_to_idx(lane + _dual_end_safe), _to_idx(_adj_end_128), _to_idx(64)):
                        ec      = _to_i32(ec)
                        glob_ec = h_off + ec
                        _vals_tail = []
                        for _j_py in range_constexpr(experts_per_token):
                            _vals_tail.append(_maybe_load(_expert_rsrc[_j_py], glob_ec, _expert_vld[_j_py], vec_width=1, dtype=T.i32(), cache_modifier=_SLC))
                        _i32t = _accum_experts(_vals_tail, _expert_vld, _eff_all_vld)
                        _out_elem_t = tok_id * n_i32 + glob_ec
                        buffer_store(_lv_unwrap(_i32t), _rsrc_out, _out_elem_t, cache_modifier=_SLC)
                      _scf_d.YieldOp([])
                _scf_d.YieldOp([])

            # step=64 路径 (hpw <= 895)
            with _IP(_if_wide.else_block):
                _n_rem_64   = arith.constant(n_elems) - h_off
                _adj_end_64 = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, _n_rem_64, hpw), _n_rem_64, hpw)
                for ec in range(_to_idx(lane), _to_idx(_adj_end_64), _to_idx(64)):
                    ec      = _to_i32(ec)
                    glob_ec = h_off + ec
                    _vals = []
                    for _j_py in range_constexpr(experts_per_token):
                        _vals.append(_maybe_load(_expert_rsrc[_j_py], glob_ec, _expert_vld[_j_py], vec_width=1, dtype=T.i32(), cache_modifier=_SLC))
                    _i32v = _accum_experts(_vals, _expert_vld, _eff_all_vld)
                    _out_elem = tok_id * n_i32 + glob_ec
                    buffer_store(_lv_unwrap(_i32v), _rsrc_out, _out_elem, cache_modifier=_SLC)
                _scf_d.YieldOp([])

        # Stage 3b: Weight accumulation
        if enable_weights:
            _rsrc_out_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_comb_out_wts))
            for wt_tok in range(_to_idx(gw_id), _to_idx(cur_rank_num_token), _to_idx(gw_num)):
                wt_tok = _to_i32(wt_tok)
                _wtm_off  = wt_tok * experts_per_token
                _wtm_vec0 = buffer_load(_rsrc_tok_map, _wtm_off, vec_width=4, dtype=T.i32())
                _wtm_vec1 = buffer_load(_rsrc_tok_map, _wtm_off + 4, vec_width=4, dtype=T.i32())

                if arith.cmpi(arith.CmpIPredicate.ult, lane, arith.constant(experts_per_token)):
                    wt_acc = arith.constant(0.0, type=T.f32())
                    for _wj in range_constexpr(experts_per_token):
                        if _wj < 4:
                            _wenc = _llvm_d.ExtractElementOp(
                                _wtm_vec0, arith.constant(_wj)).res
                        else:
                            _wenc = _llvm_d.ExtractElementOp(
                                _wtm_vec1, arith.constant(_wj - 4)).res
                        if _log2_max_recv is not None:
                            _wpe = _wenc >> arith.constant(_log2_max_recv)
                        else:
                            _wpe = arith.divui(_wenc, max_recv)
                        _wvld = arith.cmpi(arith.CmpIPredicate.ult, _wpe, arith.constant(npes))
                        _wsafe = arith.select(_wvld, _wpe, arith.constant(rank))
                        if use_p2p_read:
                            _wt_tok_all = arith.remui(_wenc, max_recv)
                            _wt_safe_ta = arith.select(_wvld, _wt_tok_all, arith.constant(0))
                            _wt_pe_base = _lds_p2p_wt_bases.load([_to_idx(_wsafe)])
                            _wt_src_off = arith.zext_i64(_wt_safe_ta) * weight_bytes
                            _wt_rsrc = create_buffer_resource_from_addr(
                                _lv_unwrap(_wt_pe_base) + _wt_src_off)
                        else:
                            _wt_src_off = arith.zext_i64(
                                _wsafe * max_tok_per_rank + wt_tok) * weight_bytes
                            _wt_rsrc = create_buffer_resource_from_addr(
                                _lv_unwrap(addr_comb_inp_wts + _wt_src_off))
                        _wv = buffer_load(_wt_rsrc, lane, vec_width=1, dtype=T.f32())
                        if npes >= experts_per_token:
                            wt_acc = wt_acc + _wv
                        else:
                            wt_acc = wt_acc + arith.select(_wvld, _wv,
                                arith.constant(0.0, type=T.f32()))
                    _wt_out_off = wt_tok * experts_per_token + lane
                    buffer_store(_lv_unwrap(wt_acc), _rsrc_out_wts, _wt_out_off)

    ep_combine_intranode._allocator = allocator
    return ep_combine_intranode


def make_dispatch_jit(*, rank, npes, experts_per_rank, experts_per_token,
                      hidden_dim, max_tok_per_rank, block_num,
                      warp_num_per_block, data_type,
                      scale_dim=0, scale_type_size=0,
                      enable_std_moe=False):
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_dispatch_kernel(
        rank=rank, npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        enable_std_moe=enable_std_moe,
        data_type=data_type,
    )

    # JIT cache key 所需闭包变量
    _rank_id, _npes_id, _block_num = rank, npes, block_num
    _wpb, _max_tok, _en_smoe = warp_num_per_block, max_tok_per_rank, enable_std_moe

    @flyc.jit
    def dispatch_launch(
        addr_inp_tok: fx.Int64, addr_idx: fx.Int64, addr_wts: fx.Int64,
        addr_out_tok: fx.Int64, addr_out_wts: fx.Int64, addr_out_idx: fx.Int64,
        addr_tok_off: fx.Int64, addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64, addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64, addr_tis: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_p2p_tok_off: fx.Int64, addr_p2p_tis: fx.Int64,
        addr_p2p_out_wts: fx.Int64, addr_p2p_out_idx: fx.Int64,
        addr_p2p_out_tok: fx.Int64, addr_p2p_recv_num: fx.Int64,
        addr_scales: fx.Int64, addr_p2p_out_scales: fx.Int64,
        addr_packed_recv_x: fx.Int64, addr_packed_recv_count: fx.Int64,
        addr_packed_recv_src_info: fx.Int64, addr_disp_tok_map: fx.Int64,
        addr_disp_grid_bar: fx.Int64,
        cur_tok: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_rank_id, _npes_id, _block_num, _wpb, _max_tok, _en_smoe)
        kernel(addr_inp_tok, addr_idx, addr_wts,
               addr_out_tok, addr_out_wts, addr_out_idx,
               addr_tok_off, addr_recv_num, addr_dest_ctr,
               addr_disp_bar, addr_tok_map, addr_tis,
               addr_total_rv,
               addr_p2p_tok_off, addr_p2p_tis,
               addr_p2p_out_wts, addr_p2p_out_idx,
               addr_p2p_out_tok, addr_p2p_recv_num,
               addr_scales, addr_p2p_out_scales,
               addr_packed_recv_x, addr_packed_recv_count,
               addr_packed_recv_src_info, addr_disp_tok_map,
               addr_disp_grid_bar,
               cur_tok).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return dispatch_launch


def make_combine_jit(*, rank, npes, experts_per_rank=0, experts_per_token,
                     hidden_dim, max_tok_per_rank, block_num,
                     warp_num_per_block, data_type,
                     enable_weights=False, enable_std_moe=False,
                     use_p2p_read=False):
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_combine_kernel(
        rank=rank, npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        data_type=data_type,
        enable_weights=enable_weights,
        enable_std_moe=enable_std_moe,
        use_p2p_read=use_p2p_read,
    )

    # JIT cache key 所需闭包变量
    _rank_id, _npes_id, _block_num = rank, npes, block_num
    _wpb, _max_tok = warp_num_per_block, max_tok_per_rank
    _en_wts, _en_smoe, _p2p_rd = enable_weights, enable_std_moe, use_p2p_read
    _allocator = kernel._allocator

    @flyc.jit
    def combine_launch(
        addr_inp_tok: fx.Int64, addr_comb_inp: fx.Int64,
        addr_comb_out: fx.Int64, addr_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64, addr_tok_map: fx.Int64,
        addr_comb_bar: fx.Int64, addr_trecv: fx.Int64,
        addr_tis: fx.Int64,
        addr_p2p_comb_inp: fx.Int64, addr_p2p_xdb_mem: fx.Int64,
        addr_wts_buf: fx.Int64,
        addr_comb_inp_wts: fx.Int64, addr_comb_out_wts: fx.Int64,
        addr_p2p_comb_inp_wts: fx.Int64,
        addr_packed_recv_x: fx.Int64, addr_disp_tok_map: fx.Int64,
        addr_disp_out_wts: fx.Int64,
        cur_rank_num_token: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_rank_id, _npes_id, _block_num, _wpb, _max_tok, _en_wts, _en_smoe, _p2p_rd)
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        _allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            _allocator.finalize()

        kernel(addr_inp_tok, addr_comb_inp, addr_comb_out,
               addr_xdb_mem, addr_xdb_flag, addr_tok_map,
               addr_comb_bar, addr_trecv, addr_tis,
               addr_p2p_comb_inp, addr_p2p_xdb_mem,
               addr_wts_buf, addr_comb_inp_wts,
               addr_comb_out_wts, addr_p2p_comb_inp_wts,
               addr_packed_recv_x, addr_disp_tok_map,
               addr_disp_out_wts,
               cur_rank_num_token).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return combine_launch
