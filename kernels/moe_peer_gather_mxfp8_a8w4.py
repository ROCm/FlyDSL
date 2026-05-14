# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device-side peer gather for MX-FP8 activations (uint8 rows) using ``recv_meta`` + P2P bases.

Fills ``out_x[dst_slot, :model_dim]`` from ``base_x_pe[src_pe] + src_token * model_dim``.

``recv_meta`` rows are **32 bytes** (8 × i32); see :mod:`kernels.recv_meta_a8w4`.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T

from kernels.dispatch_combine_intranode_kernel import (
    _lv_unwrap,
    create_buffer_resource_from_addr,
    buffer_load,
    buffer_store,
)


@functools.lru_cache(maxsize=64)
def _compile_peer_gather_mxfp8_impl(*, model_dim: int, max_recv_cap: int):
    md = int(model_dim)
    cap = int(max_recv_cap)

    @flyc.kernel(known_block_size=[1, 1, 1])
    def peer_gather_mxfp8_rows(
        addr_recv_meta: fx.Int64,
        addr_p2p_base_x: fx.Int64,
        addr_out_x: fx.Int64,
        i32_total_recv: fx.Int32,
        i32_model_dim: fx.Int32,
        i32_npes: fx.Int32,
    ):
        recv_slot = gpu.block_id("x")
        recv_slot_i32 = arith.index_cast(T.i32, recv_slot)
        total_recv = arith.index_cast(T.i32, i32_total_recv.ir_value())
        mdim = arith.index_cast(T.i32, i32_model_dim.ir_value())
        npes = arith.index_cast(T.i32, i32_npes.ir_value())

        r_meta = create_buffer_resource_from_addr(_lv_unwrap(addr_recv_meta))
        r_p2p = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_base_x))
        out_base = _lv_unwrap(addr_out_x)

        i32_ty = T.i32
        c0 = arith.constant(0, type=i32_ty)
        c1 = arith.constant(1, type=i32_ty)
        c5 = arith.constant(5, type=i32_ty)
        c8 = arith.constant(8, type=i32_ty)

        ok = arith.cmpi(arith.CmpIPredicate.ult, recv_slot_i32, total_recv)
        _if0 = scf.IfOp(_lv_unwrap(ok))
        with ir.InsertionPoint(_if0.then_block):
            meta_off = recv_slot_i32 * c8
            hdr = buffer_load(r_meta, meta_off + c0, vec_width=1, dtype=T.i32)
            src_tok = buffer_load(r_meta, meta_off + c1, vec_width=1, dtype=T.i32)
            dst_flat = buffer_load(r_meta, meta_off + c5, vec_width=1, dtype=T.i32)
            src_pe = arith.andi(hdr, arith.constant(0xFF, type=T.i32))
            pe_ok = arith.cmpi(arith.CmpIPredicate.ult, src_pe, npes)
            p2p_base = buffer_load(r_p2p, src_pe, vec_width=1, dtype=T.i64)
            row_off = arith.zext_i64(src_tok) * arith.zext_i64(mdim)
            src_row_base = arith.select(
                pe_ok,
                p2p_base + row_off,
                arith.constant(0, type=T.i64),
            )
            dst_row = dst_flat
            dst_row_base = out_base + arith.zext_i64(dst_row) * arith.zext_i64(mdim)
            for col in range_constexpr(md):
                c = arith.constant(col, type=T.i32)
                src_addr = src_row_base + arith.zext_i64(c)
                dst_addr = dst_row_base + arith.zext_i64(c)
                r_src = create_buffer_resource_from_addr(src_addr)
                r_dst = create_buffer_resource_from_addr(dst_addr)
                v = buffer_load(r_src, c0, vec_width=1, dtype=T.i8)
                v_safe = arith.select(pe_ok, v, arith.constant(0, type=T.i8))
                buffer_store(v_safe, r_dst, c0)
            scf.YieldOp([])

    @flyc.jit
    def launch_peer_gather_mxfp8_rows(
        addr_recv_meta: fx.Int64,
        addr_p2p_base_x: fx.Int64,
        addr_out_x: fx.Int64,
        i32_total_recv: fx.Int32,
        i32_model_dim: fx.Int32,
        i32_npes: fx.Int32,
        stream: fx.Stream,
    ):
        gx = max(1, cap)
        peer_gather_mxfp8_rows(
            addr_recv_meta,
            addr_p2p_base_x,
            addr_out_x,
            i32_total_recv,
            i32_model_dim,
            i32_npes,
        ).launch(grid=(gx, 1, 1), block=(1, 1, 1), stream=stream)

    return launch_peer_gather_mxfp8_rows


def compile_peer_gather_mxfp8_rows(*, model_dim: int, max_recv_cap: int = 65536):
    return _compile_peer_gather_mxfp8_impl(
        model_dim=int(model_dim), max_recv_cap=int(max_recv_cap)
    )


__all__ = ["compile_peer_gather_mxfp8_rows"]
