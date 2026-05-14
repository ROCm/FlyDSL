# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL ``moe_mxfp4_sort`` — matches ``tests/kernels/utils/fp4_utils._moe_mxfp4_sort_kernel``."""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, arith, const_expr
from flydsl.expr import Stream
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl._mlir.dialects import scf as _scf_d
from flydsl._mlir.ir import InsertionPoint as _IP

from kernels.dispatch_combine_intranode_kernel import _lv_unwrap


def mxfp4_sort_const_geom(*, max_sorted_rows: int, scale_mx_blocks: int):
    tile_m, tile_n = 32, 8
    bs_m, bs_n = 16, 4
    m_o = int(max_sorted_rows)
    n_i = int(scale_mx_blocks)
    grid_m = (m_o + tile_m - 1) // tile_m
    grid_n = (n_i + tile_n - 1) // tile_n
    out_shape = (grid_m, grid_n, bs_n, bs_m)
    strides = (grid_n * bs_n * bs_m, bs_n * bs_m, bs_m, 1)
    return grid_m, grid_n, bs_m, bs_n, m_o, n_i, out_shape, strides


def mxfp4_sort_flat_numel(*, max_sorted_rows: int, scale_mx_blocks: int) -> int:
    _, _, _, _, _, _, shape, _ = mxfp4_sort_const_geom(
        max_sorted_rows=max_sorted_rows, scale_mx_blocks=scale_mx_blocks
    )
    return int(math.prod(shape))


def _load_scale_byte(
    _r_scale_in,
    _r_sorted_ids,
    row_i32,
    col_i32,
    cur_tok_i32,
    _ept_i32,
    _n_i: int,
    _stride_in_m: int,
    _stride_in_n: int,
    *,
    _scale_rows_per_token: int,
):
    """``_scale_rows_per_token==1``: input is ``[token, scale_mx_blocks]`` (A8W4 stage1)."""
    _c0 = arith.constant(0, type=T.i32())
    _c24 = arith.constant(24, type=T.i32())
    _ff = arith.constant(0xFFFFFF, type=T.i32())
    sid = buffer_load(_r_sorted_ids, row_i32, vec_width=1, dtype=T.i32())
    topk_id = arith.shrui(sid, _c24)
    tok_id = arith.andi(sid, _ff)
    if const_expr(int(_scale_rows_per_token) == 1):
        src_m = tok_id
    else:
        src_m = arith.addi(arith.muli(tok_id, _ept_i32), topk_id)
    in_col = arith.cmpi(arith.CmpIPredicate.ult, col_i32, arith.constant(_n_i, type=T.i32()))
    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok_id, cur_tok_i32)
    _if_ok = _scf_d.IfOp(_lv_unwrap(arith.andi(in_col, tok_ok)), [T.i32()], has_else=True)
    with _IP(_if_ok.then_block):
        off_in = arith.addi(
            arith.muli(src_m, arith.constant(_stride_in_m, type=T.i32())),
            arith.muli(col_i32, arith.constant(_stride_in_n, type=T.i32())),
        )
        b8 = buffer_load(_r_scale_in, off_in, vec_width=1, dtype=T.i8())
        b = arith.extui(T.i32(), b8)
        _scf_d.YieldOp([_lv_unwrap(b)])
    with _IP(_if_ok.else_block):
        _scf_d.YieldOp([_c0])
    return _if_ok.results[0]


def _emit_mxfp4_sort_one_tile(
    *,
    _r_scale_in,
    _r_sorted_ids,
    _r_scale_out,
    cur_tok_i32,
    _ept_i32,
    num_valid,
    prog_m_i32,
    prog_n_i32,
    _bs_m: int,
    _bs_n: int,
    _m_o: int,
    _n_i: int,
    _stride_in_m: int,
    _stride_in_n: int,
    _stride_o0: int,
    _stride_o1: int,
    _stride_o2: int,
    _stride_o3: int,
    _scale_rows_per_token: int,
):
    _c0 = arith.constant(0, type=T.i32())
    _c2 = arith.constant(2, type=T.i32())
    _bs_m_i32 = arith.constant(_bs_m, type=T.i32())
    _bs_n_i32 = arith.constant(_bs_n, type=T.i32())
    pid_m = arith.muli(prog_m_i32, _c2)
    pid_n = arith.muli(prog_n_i32, _c2)
    tile_base_m = arith.muli(pid_m, _bs_m_i32)
    tile_base_i32 = tile_base_m
    in_range = arith.cmpi(arith.CmpIPredicate.ult, tile_base_i32, num_valid)
    _if_tile = _scf_d.IfOp(_lv_unwrap(in_range))
    with _IP(_if_tile.then_block):
        for lm in range(_bs_m):
            lm_i32 = arith.constant(lm, type=T.i32())
            for ln in range(_bs_n):
                ln_i32 = arith.constant(ln, type=T.i32())
                pack = _c0
                for i in range(4):
                    m_off_i32 = arith.constant((i % 2) * _bs_m, type=T.i32())
                    n_off_i32 = arith.constant((i // 2) * _bs_n, type=T.i32())
                    row_i = arith.addi(
                        arith.addi(tile_base_i32, m_off_i32), lm_i32
                    )
                    col_i = arith.addi(
                        arith.addi(
                            arith.muli(pid_n, _bs_n_i32), n_off_i32
                        ),
                        ln_i32,
                    )
                    b = _load_scale_byte(
                        _r_scale_in,
                        _r_sorted_ids,
                        row_i,
                        col_i,
                        cur_tok_i32,
                        _ept_i32,
                        _n_i,
                        _stride_in_m,
                        _stride_in_n,
                        _scale_rows_per_token=_scale_rows_per_token,
                    )
                    sh = arith.constant(i * 8, type=T.i32())
                    pack = arith.addi(pack, arith.shli(b, sh))
                off_o = arith.addi(
                    arith.addi(
                        arith.muli(lm_i32, arith.constant(_stride_o3, type=T.i32())),
                        arith.muli(ln_i32, arith.constant(_stride_o2, type=T.i32())),
                    ),
                    arith.addi(
                        arith.muli(prog_n_i32, arith.constant(_stride_o1, type=T.i32())),
                        arith.muli(prog_m_i32, arith.constant(_stride_o0, type=T.i32())),
                    ),
                )
                buffer_store(_lv_unwrap(pack), _r_scale_out, off_o)
        _scf_d.YieldOp([])


def _emit_moe_mxfp4_sort_parallel(
    *,
    gw_id,
    gw_num,
    _r_scale_in,
    _r_sorted_ids,
    _r_num_valid,
    _r_scale_out,
    cur_tok_i32,
    _ept_i32,
    _grid_m: int,
    _grid_n: int,
    _bs_m: int,
    _bs_n: int,
    _m_o: int,
    _n_i: int,
    _stride_in_m: int,
    _stride_in_n: int,
    _stride_o0: int,
    _stride_o1: int,
    _stride_o2: int,
    _stride_o3: int,
    _scale_rows_per_token: int = 1,
):
    """Deprecated: use per-tile CTA grid in :func:`make_moe_mxfp4_sort_kernel`."""
    _c0 = arith.constant(0, type=T.i32())
    num_valid = buffer_load(_r_num_valid, _c0, vec_width=1, dtype=T.i32())
    for prog_m in range(int(_grid_m)):
        for prog_n in range(int(_grid_n)):
            _emit_mxfp4_sort_one_tile(
                _r_scale_in=_r_scale_in,
                _r_sorted_ids=_r_sorted_ids,
                _r_scale_out=_r_scale_out,
                cur_tok_i32=cur_tok_i32,
                _ept_i32=_ept_i32,
                num_valid=num_valid,
                prog_m_i32=arith.constant(prog_m, type=T.i32()),
                prog_n_i32=arith.constant(prog_n, type=T.i32()),
                _bs_m=_bs_m,
                _bs_n=_bs_n,
                _m_o=_m_o,
                _n_i=_n_i,
                _stride_in_m=_stride_in_m,
                _stride_in_n=_stride_in_n,
                _stride_o0=_stride_o0,
                _stride_o1=_stride_o1,
                _stride_o2=_stride_o2,
                _stride_o3=_stride_o3,
                _scale_rows_per_token=_scale_rows_per_token,
            )


def _emit_moe_mxfp4_sort_serial(
    *,
    _r_scale_in,
    _r_sorted_ids,
    _r_num_valid,
    _r_scale_out,
    cur_tok_i32,
    _ept_i32,
    _grid_m: int,
    _grid_n: int,
    _bs_m: int,
    _bs_n: int,
    _m_o: int,
    _n_i: int,
    _stride_in_m: int,
    _stride_in_n: int,
    _stride_o0: int,
    _stride_o1: int,
    _stride_o2: int,
    _stride_o3: int,
    _scale_rows_per_token: int = 1,
):
    """Serial fallback (lane 0 only); prefer :func:`_emit_moe_mxfp4_sort_parallel`."""
    _c0 = arith.constant(0, type=T.i32())
    num_valid = buffer_load(_r_num_valid, _c0, vec_width=1, dtype=T.i32())
    for prog_m in range(_grid_m):
        for prog_n in range(_grid_n):
            _emit_mxfp4_sort_one_tile(
                _r_scale_in=_r_scale_in,
                _r_sorted_ids=_r_sorted_ids,
                _r_scale_out=_r_scale_out,
                cur_tok_i32=cur_tok_i32,
                _ept_i32=_ept_i32,
                num_valid=num_valid,
                prog_m_i32=arith.constant(prog_m, type=T.i32()),
                prog_n_i32=arith.constant(prog_n, type=T.i32()),
                _bs_m=_bs_m,
                _bs_n=_bs_n,
                _m_o=_m_o,
                _n_i=_n_i,
                _stride_in_m=_stride_in_m,
                _stride_in_n=_stride_in_n,
                _stride_o0=_stride_o0,
                _stride_o1=_stride_o1,
                _stride_o2=_stride_o2,
                _stride_o3=_stride_o3,
                _scale_rows_per_token=_scale_rows_per_token,
            )


def allocate_sorted_scale_x_u8(
    *,
    max_sorted_rows: int,
    scale_mx_blocks: int,
    device,
):
    """Host buffer for device ``moe_mxfp4_sort`` (uint32 tiled → uint8 flat, matches fp4_utils)."""
    import torch

    gm, gn, bm, bn, _, _, shape, _ = mxfp4_sort_const_geom(
        max_sorted_rows=max_sorted_rows, scale_mx_blocks=scale_mx_blocks
    )
    u32 = torch.zeros(shape, dtype=torch.uint32, device=device)
    return u32.view(torch.uint8).reshape(-1).contiguous()


def make_moe_mxfp4_sort_kernel(
    *,
    experts_per_token: int,
    max_sorted_rows: int,
    scale_mx_blocks: int,
):
    (
        _grid_m,
        _grid_n,
        _bs_m,
        _bs_n,
        _m_o,
        _n_i,
        _,
        _strides,
    ) = mxfp4_sort_const_geom(
        max_sorted_rows=max_sorted_rows, scale_mx_blocks=scale_mx_blocks
    )
    _stride_o0, _stride_o1, _stride_o2, _stride_o3 = _strides
    _stride_in_m = scale_mx_blocks
    _stride_in_n = 1
    _ept = int(experts_per_token)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def ep_moe_mxfp4_sort(
        addr_scale_mx_in: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_scale_x_sorted: fx.Int64,
        cur_tok: fx.Int32,
    ):
        tid = fx.thread_idx.x
        prog_m_i32 = arith.index_cast(T.i32(), fx.block_idx.x.ir_value())
        prog_n_i32 = arith.index_cast(T.i32(), fx.block_idx.y.ir_value())
        _c0_tid = arith.constant(0, type=T.i32())
        if arith.cmpi(arith.CmpIPredicate.eq, tid, _c0_tid):
            cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
            _ept_i32 = arith.constant(_ept, type=T.i32())
            _r_scale_in = create_buffer_resource_from_addr(_lv_unwrap(addr_scale_mx_in))
            _r_sorted_ids = create_buffer_resource_from_addr(
                _lv_unwrap(addr_sorted_token_ids)
            )
            _r_num_valid = create_buffer_resource_from_addr(
                _lv_unwrap(addr_num_valid_ids)
            )
            _r_scale_out = create_buffer_resource_from_addr(
                _lv_unwrap(addr_scale_x_sorted)
            )
            num_valid = buffer_load(_r_num_valid, _c0_tid, vec_width=1, dtype=T.i32())
            _emit_mxfp4_sort_one_tile(
                _r_scale_in=_r_scale_in,
                _r_sorted_ids=_r_sorted_ids,
                _r_scale_out=_r_scale_out,
                cur_tok_i32=cur_tok_i32,
                _ept_i32=_ept_i32,
                num_valid=num_valid,
                prog_m_i32=prog_m_i32,
                prog_n_i32=prog_n_i32,
                _bs_m=_bs_m,
                _bs_n=_bs_n,
                _m_o=_m_o,
                _n_i=_n_i,
                _stride_in_m=_stride_in_m,
                _stride_in_n=_stride_in_n,
                _stride_o0=_stride_o0,
                _stride_o1=_stride_o1,
                _stride_o2=_stride_o2,
                _stride_o3=_stride_o3,
                _scale_rows_per_token=1,
            )

    return ep_moe_mxfp4_sort


def make_moe_mxfp4_sort_jit(
    *,
    experts_per_token: int,
    max_tok_per_rank: int,
    num_experts: int,
    sort_block_m: int,
    scale_mx_blocks: int,
):
    from kernels.moe_sorting_flydsl import moe_sorting_const_sizes

    _max_padded, _, _ = moe_sorting_const_sizes(
        max_tok_per_rank=max_tok_per_rank,
        experts_per_token=experts_per_token,
        num_experts=num_experts,
        block_size=sort_block_m,
    )
    kernel = make_moe_mxfp4_sort_kernel(
        experts_per_token=experts_per_token,
        max_sorted_rows=_max_padded,
        scale_mx_blocks=scale_mx_blocks,
    )
    _gm, _gn, _, _, _, _, _, _ = mxfp4_sort_const_geom(
        max_sorted_rows=_max_padded, scale_mx_blocks=scale_mx_blocks
    )

    @flyc.jit
    def launch_moe_mxfp4_sort(
        addr_scale_mx_in: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_scale_x_sorted: fx.Int64,
        cur_tok: fx.Int32,
        stream: Stream = Stream(None),
    ):
        kernel(
            addr_scale_mx_in,
            addr_sorted_token_ids,
            addr_num_valid_ids,
            addr_scale_x_sorted,
            cur_tok,
        ).launch(
            grid=(_gm, _gn, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch_moe_mxfp4_sort


__all__ = [
    "allocate_sorted_scale_x_u8",
    "mxfp4_sort_const_geom",
    "mxfp4_sort_flat_numel",
    "make_moe_mxfp4_sort_jit",
    "_emit_moe_mxfp4_sort_parallel",
    "_emit_moe_mxfp4_sort_serial",
]
