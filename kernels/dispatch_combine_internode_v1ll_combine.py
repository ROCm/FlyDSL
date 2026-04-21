"""
FlyDSL ``EpCombineSync`` + ``EpCombineAll`` (bf16 + optional fp32 weights) for InterNode V1 / V1LL.

HIP remains for ``EpCombineSyncBarrier`` and ``EpCombineInterNodeV1KernelLowLatency``.

For ``quant_type=fp8_direct_cast``, mori's ``EpCombineSync`` performs bf16→internal fp8
into ``combineInp``; use full HIP combine instead of this module. Standard-MoE
``combine_standard_moe`` also stays on full HIP (``EpCombineSync`` skips hidden copy
when ``ENABLE_STANDARD_MOE_ADAPT`` is on).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_HERE, "../python"), os.path.join(_HERE, "../../FlyDSL/python")]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from flydsl._mlir.dialects import arith as _arith_d
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.ir import BF16Type, F32Type, FloatAttr, IntegerAttr as _IA, IntegerType as _IT_mlir

import flydsl._mlir.ir as _ir

from .dispatch_combine_internode_v1ll_llvm_helpers import (
    _unwrap as _lv_unwrap,
    as_index,
    const_i32,
    const_i64,
    divui,
    icmp_eq_i32,
    icmp_ult_i32,
    idx_to_i32,
    load_f32_global_at,
    load_i32_global,
    load_i32_global_at,
    load_v4i32_global,
    remui,
    select_i32,
    store_i32_global,
    store_i32_global_at,
    store_v4i32_global,
    zext_i32_to_i64,
)


def _i32_add(a, b):
    return _arith_d.AddIOp(_lv_unwrap(a), _lv_unwrap(b)).result


def _i32_mul(a, b):
    return _arith_d.MulIOp(_lv_unwrap(a), _lv_unwrap(b)).result


def _ceil_div_i32(a, b):
    one = const_i32(1)
    num = _arith_d.SubIOp(_arith_d.AddIOp(_lv_unwrap(a), _lv_unwrap(b)).result, _lv_unwrap(one)).result
    return divui(num, b)


def _min_i32(a, b):
    return select_i32(icmp_ult_i32(a, b), a, b)


def _bitcast_f32_to_i32(val):
    return _llvm_d.BitcastOp(_IT_mlir.get_signless(32), _lv_unwrap(val)).result


def _const_f32(v: float):
    return _arith_d.ConstantOp(F32Type.get(), FloatAttr.get(F32Type.get(), v)).result


def _addr_add_i64(base, delta_i64):
    return _llvm_d.AddOp(
        _lv_unwrap(base),
        _lv_unwrap(delta_i64),
        _ir.Attribute.parse("#llvm.overflow<none>"),
    ).result


def _bf16_i16_to_f32(i16v):
    bf = _llvm_d.BitcastOp(BF16Type.get(), i16v).result
    return _arith_d.ExtFOp(F32Type.get(), bf).result


def _f32_to_bf16_i16(fv):
    bf = _arith_d.TruncFOp(BF16Type.get(), fv).result
    return _llvm_d.BitcastOp(_IT_mlir.get_signless(16), bf).result


def _select_f32(cond_i1, a_f32, b_f32):
    return _arith_d.SelectOp(_lv_unwrap(cond_i1), _lv_unwrap(a_f32), _lv_unwrap(b_f32)).result


def make_ep_combine_sync_kernel(
    *,
    hidden_dim: int,
    hidden_elem_size: int,
    experts_per_token: int,
    multiprocessor_count: int,
    warp_num_per_block: int,
    has_weights: bool,
):
    assert hidden_elem_size == 2, "FlyDSL combine_sync: bf16 only"
    n_i32 = (hidden_dim * hidden_elem_size) // 4
    mp_py = multiprocessor_count
    warp_num = warp_num_per_block

    @flyc.kernel
    def ep_combine_sync(
        addr_inp: fx.Int64,
        addr_comb_inp: fx.Int64,
        addr_wts_in: fx.Int64,
        addr_wts_shmem: fx.Int64,
        addr_total_recv: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp = tid >> 6
        lane = tid & 63
        total_recv = load_i32_global_at(addr_total_recv, const_i32(0))
        tpb = _ceil_div_i32(total_recv, const_i32(mp_py))
        start = _i32_mul(idx_to_i32(bid), tpb)
        end_excl = _min_i32(_i32_add(start, tpb), total_recv)

        for token_id in range(as_index(_i32_add(start, idx_to_i32(warp))), as_index(end_excl), as_index(warp_num)):
            token_id = idx_to_i32(token_id)
            row_b = zext_i32_to_i64(_i32_mul(token_id, const_i32(hidden_dim * hidden_elem_size)))
            in_base = addr_inp + row_b
            out_base = addr_comb_inp + row_b
            for c in range(as_index(lane), as_index(n_i32), as_index(64)):
                c = idx_to_i32(c)
                off = zext_i32_to_i64(c) * const_i64(16)
                v = load_v4i32_global(in_base + off)
                store_v4i32_global(v, out_base + off)

        if has_weights:
            for token_id in range(
                as_index(_i32_add(start, idx_to_i32(warp))), as_index(end_excl), as_index(warp_num)
            ):
                token_id = idx_to_i32(token_id)
                if icmp_ult_i32(lane, const_i32(experts_per_token)):
                    off = _i32_add(_i32_mul(token_id, const_i32(experts_per_token)), lane)
                    w = load_f32_global_at(addr_wts_in, off)
                    store_i32_global_at(addr_wts_shmem, off, _bitcast_f32_to_i32(w))

    return ep_combine_sync


def make_ep_combine_all_kernel(
    *,
    rank: int,
    world_size: int,
    gpu_per_node: int,
    max_tok_per_rank: int,
    hidden_dim: int,
    hidden_elem_size: int,
    experts_per_token: int,
    num_experts_per_rank: int,
    multiprocessor_count: int,
    warp_num_per_block: int,
    has_weights: bool,
):
    assert hidden_elem_size == 2, "FlyDSL combine_all: bf16 only"
    n_nodes = world_size // gpu_per_node
    my_node_py = rank // gpu_per_node
    hidden_bytes = hidden_dim * hidden_elem_size
    weight_bytes = experts_per_token * 4 if has_weights else 0
    comb_xfer = hidden_bytes + weight_bytes
    pivot_b = n_nodes * max_tok_per_rank * comb_xfer
    n_pair = (hidden_dim + 1) // 2
    mp_py = multiprocessor_count
    gw_num_py = mp_py * warp_num_per_block
    hd = hidden_dim
    max_tok = max_tok_per_rank
    k_exp = experts_per_token
    ner = num_experts_per_rank

    @flyc.kernel
    def ep_combine_all(
        addr_total_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        addr_send_map: fx.Int64,
        addr_staging: fx.Int64,
        addr_comb_out: fx.Int64,
        addr_comb_wts: fx.Int64,
        addr_indices: fx.Int64,
        cur_tok: fx.Int32,
    ):
        i32_ty = _IT_mlir.get_signless(32)
        i16_ty = _IT_mlir.get_signless(16)
        z_i16 = _llvm_d.ConstantOp(i16_ty, _IA.get(i16_ty, 0)).result
        ner_i = const_i32(ner)
        gpn_i = const_i32(gpu_per_node)
        bid = fx.block_idx.x
        warp = fx.thread_idx.x >> 6
        lane = fx.thread_idx.x & 63
        gw_id = bid * warp_num_per_block + warp
        gw_num = const_i32(gw_num_py)
        my_n = const_i32(my_node_py)

        if icmp_eq_i32(gw_id, const_i32(0)):
            if icmp_eq_i32(lane, const_i32(0)):
                store_i32_global_at(addr_total_recv, const_i32(0), const_i32(0))
            if icmp_ult_i32(lane, const_i32(n_nodes)):
                store_i32_global_at(addr_block_flag, lane, const_i32(0))

        n_work = _i32_mul(cur_tok, const_i32(n_pair))
        xfer = const_i64(comb_xfer)
        piv = const_i64(pivot_b)
        zf = _const_f32(0.0)

        for pi in range(as_index(gw_id), as_index(n_work), as_index(gw_num_py)):
            pi = idx_to_i32(pi)
            token_id = divui(pi, const_i32(n_pair))
            pcol = remui(pi, const_i32(n_pair))
            h0 = _i32_mul(pcol, const_i32(2))
            acc_lo = zf
            acc_hi = zf

            for n_py in range(n_nodes):
                n = const_i32(n_py)
                has_route = const_i32(0)
                for e_py in range(k_exp):
                    ei = const_i32(e_py)
                    exp_idx = load_i32_global_at(
                        addr_indices,
                        _i32_add(_i32_mul(token_id, const_i32(k_exp)), ei),
                    )
                    dnode = divui(divui(exp_idx, ner_i), gpn_i)
                    has_route = select_i32(icmp_eq_i32(dnode, n), const_i32(1), has_route)
                no_route = icmp_eq_i32(has_route, const_i32(0))
                mapped = select_i32(
                    icmp_eq_i32(n, my_n),
                    token_id,
                    load_i32_global_at(
                        addr_send_map,
                        _i32_add(_i32_mul(token_id, const_i32(n_nodes)), n),
                    ),
                )
                slot = _i32_add(_i32_mul(n, const_i32(max_tok)), mapped)
                row_b = zext_i32_to_i64(slot) * xfer
                pair_off = zext_i32_to_i64(_i32_mul(pcol, const_i32(4)))
                src_addr = _addr_add_i64(_addr_add_i64(_addr_add_i64(addr_staging, piv), row_b), pair_off)
                src_word = load_i32_global(src_addr)
                lo16 = _llvm_d.TruncOp(i16_ty, src_word).result
                hi16 = _llvm_d.TruncOp(
                    i16_ty,
                    _arith_d.ShRUIOp(src_word, _arith_d.ConstantOp(i32_ty, _IA.get(i32_ty, 16)).result).result,
                ).result
                fv_lo = _bf16_i16_to_f32(lo16)
                fv_hi = _bf16_i16_to_f32(hi16)
                acc_lo = _arith_d.AddFOp(acc_lo, _select_f32(no_route, zf, fv_lo)).result
                acc_hi = _arith_d.AddFOp(acc_hi, _select_f32(no_route, zf, fv_hi)).result

            has_hi = icmp_ult_i32(_i32_add(h0, const_i32(1)), const_i32(hd))
            lo_st = _f32_to_bf16_i16(acc_lo)
            hi_st = _f32_to_bf16_i16(acc_hi)
            hi_pick = _arith_d.SelectOp(_lv_unwrap(has_hi), _lv_unwrap(hi_st), _lv_unwrap(z_i16)).result
            lo32 = _llvm_d.ZExtOp(i32_ty, lo_st).result
            hi32 = _llvm_d.ShLIOp(_llvm_d.ZExtOp(i32_ty, hi_pick).result, const_i32(16)).result
            out_word = _arith_d.OrIOp(lo32, hi32).result
            wflat = _i32_add(_i32_mul(token_id, const_i32(n_pair)), pcol)
            store_i32_global(addr_comb_out, wflat, out_word)

        if has_weights:
            n_w = _i32_mul(cur_tok, const_i32(k_exp))
            for wi in range(as_index(gw_id), as_index(n_w), as_index(gw_num_py)):
                wi = idx_to_i32(wi)
                token_id = divui(wi, const_i32(k_exp))
                e = remui(wi, const_i32(k_exp))
                aw = zf
                for n_py in range(n_nodes):
                    n = const_i32(n_py)
                    has_route = const_i32(0)
                    for e_py in range(k_exp):
                        ei = const_i32(e_py)
                        exp_idx = load_i32_global_at(
                            addr_indices,
                            _i32_add(_i32_mul(token_id, const_i32(k_exp)), ei),
                        )
                        dnode = divui(divui(exp_idx, ner_i), gpn_i)
                        has_route = select_i32(icmp_eq_i32(dnode, n), const_i32(1), has_route)
                    no_route = icmp_eq_i32(has_route, const_i32(0))
                    mapped = select_i32(
                        icmp_eq_i32(n, my_n),
                        token_id,
                        load_i32_global_at(
                            addr_send_map,
                            _i32_add(_i32_mul(token_id, const_i32(n_nodes)), n),
                        ),
                    )
                    slot = _i32_add(_i32_mul(n, const_i32(max_tok)), mapped)
                    row_b = zext_i32_to_i64(slot) * xfer
                    woff_b = zext_i32_to_i64(_i32_add(const_i32(hidden_dim * 2), _i32_mul(e, const_i32(4))))
                    wi_v = load_i32_global(_addr_add_i64(_addr_add_i64(_addr_add_i64(addr_staging, piv), row_b), woff_b))
                    wf_f = _llvm_d.BitcastOp(F32Type.get(), wi_v).result
                    aw = _arith_d.AddFOp(aw, _select_f32(no_route, zf, wf_f)).result
                wout = _i32_add(_i32_mul(token_id, const_i32(k_exp)), e)
                store_i32_global_at(addr_comb_wts, wout, _bitcast_f32_to_i32(aw))

    return ep_combine_all


def make_ep_combine_sync_jit(
    *,
    hidden_dim: int,
    experts_per_token: int,
    multiprocessor_count: int,
    warp_num_per_block: int,
    has_weights: bool,
    data_type: torch.dtype,
):
    hes = torch.tensor([], dtype=data_type).element_size()
    k = make_ep_combine_sync_kernel(
        hidden_dim=hidden_dim,
        hidden_elem_size=hes,
        experts_per_token=experts_per_token,
        multiprocessor_count=multiprocessor_count,
        warp_num_per_block=warp_num_per_block,
        has_weights=has_weights,
    )

    @flyc.jit
    def launch(
        addr_inp: fx.Int64,
        addr_comb_inp: fx.Int64,
        addr_wts_in: fx.Int64,
        addr_wts_shmem: fx.Int64,
        addr_total_recv: fx.Int64,
    ):
        k(addr_inp, addr_comb_inp, addr_wts_in, addr_wts_shmem, addr_total_recv).launch(
            grid=(multiprocessor_count, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
        )

    return launch


def make_ep_combine_all_jit(
    *,
    rank: int,
    world_size: int,
    gpu_per_node: int,
    max_tok_per_rank: int,
    hidden_dim: int,
    experts_per_token: int,
    num_experts_per_rank: int,
    multiprocessor_count: int,
    warp_num_per_block: int,
    has_weights: bool,
    data_type: torch.dtype,
):
    hes = torch.tensor([], dtype=data_type).element_size()
    k = make_ep_combine_all_kernel(
        rank=rank,
        world_size=world_size,
        gpu_per_node=gpu_per_node,
        max_tok_per_rank=max_tok_per_rank,
        hidden_dim=hidden_dim,
        hidden_elem_size=hes,
        experts_per_token=experts_per_token,
        num_experts_per_rank=num_experts_per_rank,
        multiprocessor_count=multiprocessor_count,
        warp_num_per_block=warp_num_per_block,
        has_weights=has_weights,
    )

    @flyc.jit
    def launch(
        addr_total_recv: fx.Int64,
        addr_block_flag: fx.Int64,
        addr_send_map: fx.Int64,
        addr_staging: fx.Int64,
        addr_comb_out: fx.Int64,
        addr_comb_wts: fx.Int64,
        addr_indices: fx.Int64,
        cur_tok: fx.Int32,
    ):
        k(
            addr_total_recv,
            addr_block_flag,
            addr_send_map,
            addr_staging,
            addr_comb_out,
            addr_comb_wts,
            addr_indices,
            cur_tok,
        ).launch(grid=(multiprocessor_count, 1, 1), block=(warp_num_per_block * 64, 1, 1))

    return launch
