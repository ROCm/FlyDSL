"""
FlyDSL implementation of mori ``EpDispatchCopyToStaging`` (InterNode V1 / V1LL).

This is the first phase of ``InterNodeV1`` / ``InterNodeV1LL`` dispatch: pack each
local token row (hidden, indices, weights, optional scales, flat src id) into the
symmetric RDMA staging buffer. The second phase remains the HIP kernel
``EpDispatchInterNodeV1KernelLowLatency_*`` (RDMA + intra-node routing).

``quant_type=fp8_direct_cast`` is not implemented here: :class:`FlyDSLDispatchCombineInterNodeV1LLOp`
falls back to HIP ``EpDispatchCopyToStaging`` so behaviour matches mori.

Layout per token matches ``internode_v1.cpp`` / ``common.hpp``:
``xferBytes = hiddenBytes + indexBytes + weightBytes + scaleBytes + sizeof(index_t)``.
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
from flydsl._mlir.ir import IntegerType as _IT_mlir
from flydsl.expr.lowlevel import (
    _unwrap as _lv_unwrap,
    as_index,
    const_i32,
    const_i64,
    divui,
    icmp_eq_i32,
    icmp_ult_i32,
    idx_to_i32,
    load_f32_global_at,
    load_i32_global_at,
    load_v4i32_global,
    remui,
    select_i32,
    store_i32_global_at,
    store_v4i32_global,
    zext_i32_to_i64,
)


def _bitcast_f32_to_i32(val):
    return _llvm_d.BitcastOp(_IT_mlir.get_signless(32), _lv_unwrap(val)).result


def _i32_add(a, b):
    return _arith_d.AddIOp(_lv_unwrap(a), _lv_unwrap(b)).result


def _i32_mul(a, b):
    return _arith_d.MulIOp(_lv_unwrap(a), _lv_unwrap(b)).result


def _ceil_div_i32(a, b):
    """``ceil(a / b)`` for strictly positive ``b`` (same as mori CeilDiv)."""
    one = const_i32(1)
    num = _arith_d.SubIOp(_arith_d.AddIOp(_lv_unwrap(a), _lv_unwrap(b)).result, _lv_unwrap(one)).result
    return divui(num, b)


def make_ep_dispatch_copy_to_staging_kernel(
    *,
    rank: int,
    world_size: int,
    max_tok_per_rank: int,
    hidden_dim: int,
    hidden_elem_size: int,
    experts_per_token: int,
    scale_dim: int,
    scale_type_size: int,
    multiprocessor_count: int,
    warp_num_per_block: int,
):
    """Return ``@flyc.kernel`` mirroring ``EpDispatchCopyToStaging_body``."""
    index_bytes = experts_per_token * 4
    weight_bytes = experts_per_token * 4
    scale_bytes = scale_dim * scale_type_size
    hidden_bytes = hidden_dim * hidden_elem_size
    xfer_bytes = hidden_bytes + index_bytes + weight_bytes + scale_bytes + 4
    max_tokens_send = world_size * max_tok_per_rank
    flat_base = rank * max_tokens_send

    gw_num_py = multiprocessor_count * warp_num_per_block

    @flyc.kernel
    def ep_dispatch_copy_to_staging(
        addr_inp_tok: fx.Int64,
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
        gw_id = _i32_add(
            _i32_mul(idx_to_i32(bid), const_i32(warp_num_per_block)),
            idx_to_i32(warp),
        )
        gw_num = const_i32(gw_num_py)
        hd = const_i32(hidden_dim)
        # Avoid div-by-zero in ceil-div when ``cur_tok == 0`` (loop limit is zero anyway).
        safe_tok = select_i32(icmp_eq_i32(cur_tok, const_i32(0)), const_i32(1), cur_tok)
        wpt = _ceil_div_i32(gw_num, safe_tok)
        hpw = _ceil_div_i32(hd, wpt)
        lim = _i32_mul(cur_tok, wpt)

        xfer_b = const_i64(xfer_bytes)
        hb = const_i64(hidden_bytes)
        ib = const_i64(index_bytes)
        wb = const_i64(weight_bytes)
        sb = const_i64(scale_bytes)

        # HIP: ``for (int i = globalWarpId; i < lim; i += globalWarpNum)``.
        # Multi-CTA launches may not lower the dynamic loop start correctly; see
        # ``test_launch_copy_to_staging_multi_warp_smoke`` (xfail).
        for ii in range(as_index(gw_id), as_index(lim), as_index(gw_num_py)):
            ii = idx_to_i32(ii)
            token_id = divui(ii, wpt)
            part_id = remui(ii, wpt)
            h_off = _i32_mul(part_id, hpw)
            n_rem = const_i32(hidden_dim) - h_off
            h_size = select_i32(icmp_ult_i32(n_rem, hpw), n_rem, hpw)
            nbytes = _i32_mul(h_size, const_i32(hidden_elem_size))
            n_chunks = divui(nbytes + const_i32(15), const_i32(16))

            row_off = zext_i32_to_i64(token_id) * xfer_b
            st_row = addr_staging + row_off
            inp_off = zext_i32_to_i64(_i32_mul(token_id, const_i32(hidden_dim * hidden_elem_size)))
            inp_row = addr_inp_tok + inp_off

            h_byte_off = zext_i32_to_i64(_i32_mul(h_off, const_i32(hidden_elem_size)))
            for base_c in range(as_index(0), as_index(n_chunks), as_index(64)):
                base_c_i = idx_to_i32(base_c)
                c = _i32_add(base_c_i, idx_to_i32(lane))
                if icmp_ult_i32(c, n_chunks):
                    off16 = zext_i32_to_i64(c) * const_i64(16)
                    vec = load_v4i32_global(inp_row + h_byte_off + off16)
                    store_v4i32_global(vec, st_row + h_byte_off + off16)

            if icmp_eq_i32(part_id, const_i32(0)):
                st_idx = st_row + hb
                if icmp_ult_i32(lane, const_i32(experts_per_token)):
                    ix_off = _i32_add(_i32_mul(token_id, const_i32(experts_per_token)), lane)
                    ix = load_i32_global_at(addr_idx, ix_off)
                    store_i32_global_at(st_idx, lane, ix)

                st_wt = st_row + hb + ib
                if icmp_ult_i32(lane, const_i32(experts_per_token)):
                    wt_off = _i32_add(_i32_mul(token_id, const_i32(experts_per_token)), lane)
                    wt = load_f32_global_at(addr_wts, wt_off)
                    store_i32_global_at(st_wt, lane, _bitcast_f32_to_i32(wt))

                if scale_bytes > 0:
                    st_sc = st_row + hb + ib + wb
                    sc_row_bytes = scale_dim * scale_type_size
                    n_sch = (sc_row_bytes + 15) // 16
                    sc_byte0 = zext_i32_to_i64(
                        _i32_mul(token_id, const_i32(scale_dim * scale_type_size))
                    )
                    sc_src = addr_scales + sc_byte0
                    n_sch_i32 = const_i32(n_sch)
                    for base_c in range(as_index(0), as_index(n_sch_i32), as_index(64)):
                        base_c_i = idx_to_i32(base_c)
                        c = _i32_add(base_c_i, idx_to_i32(lane))
                        if icmp_ult_i32(c, n_sch_i32):
                            off16 = zext_i32_to_i64(c) * const_i64(16)
                            vec = load_v4i32_global(sc_src + off16)
                            store_v4i32_global(vec, st_sc + off16)

                flat_enc = const_i32(flat_base) + token_id
                if icmp_eq_i32(lane, const_i32(0)):
                    store_i32_global_at(st_row + hb + ib + wb + sb, const_i32(0), flat_enc)

    return ep_dispatch_copy_to_staging


def make_ep_dispatch_copy_to_staging_jit(
    *,
    rank: int,
    world_size: int,
    max_tok_per_rank: int,
    hidden_dim: int,
    experts_per_token: int,
    scale_dim: int,
    scale_type_size: int,
    multiprocessor_count: int,
    warp_num_per_block: int,
    data_type: torch.dtype,
):
    """Build ``@flyc.jit`` launcher for copy-to-staging (matches HIP grid/block)."""
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_ep_dispatch_copy_to_staging_kernel(
        rank=rank,
        world_size=world_size,
        max_tok_per_rank=max_tok_per_rank,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        experts_per_token=experts_per_token,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        multiprocessor_count=multiprocessor_count,
        warp_num_per_block=warp_num_per_block,
    )

    _rank_k = rank
    _ws_k = world_size
    _mp_k = multiprocessor_count
    _wpb_k = warp_num_per_block

    @flyc.jit
    def launch_copy_to_staging(
        addr_inp_tok: fx.Int64,
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_scales: fx.Int64,
        addr_staging: fx.Int64,
        cur_tok: fx.Int32,
    ):
        _ = (_rank_k, _ws_k, _mp_k, _wpb_k)
        kernel(
            addr_inp_tok,
            addr_idx,
            addr_wts,
            addr_scales,
            addr_staging,
            cur_tok,
        ).launch(
            grid=(multiprocessor_count, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
        )

    return launch_copy_to_staging
