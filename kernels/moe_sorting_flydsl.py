# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL MoE sorting (aiter/CK-compatible layout).

Mirrors ``aiter/op_tests/test_moe_sorting.py::moe_sorting_native``.

Parallel path (default JIT): one CTA per expert, 64 threads scan ``cur_tok * topk`` slots.
Serial path (``_emit_moe_sorting_serial``): metadata inline when ``enable_moe_sorting=True``.
"""

from __future__ import annotations

import functools
from typing import Any

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

from kernels.dispatch_combine_intranode_kernel import (
    _lv_unwrap,
    _to_i32,
    _to_idx,
    atomic_add_global_at,
)
from kernels.moe_mxfp4_sort_flydsl import _emit_mxfp4_sort_one_tile, mxfp4_sort_const_geom

_WARP_THREADS = 64


def _count_tokens_for_expert(_r_idx, limit_i32, expert_id, _ept_i32, _max_slots: int):
    """Count slots where idx[slot] == expert_id (slots < limit)."""
    _c0 = arith.constant(0, type=T.i32())
    _c1 = arith.constant(1, type=T.i32())
    _for = _scf_d.ForOp(
        _c0,
        arith.constant(_max_slots, type=T.i32()),
        _c1,
        [_c0],
    )
    with _IP(_for.body):
        i = _for.induction_variable
        cnt = _for.inner_iter_args[0]
        in_rng = arith.cmpi(arith.CmpIPredicate.ult, i, limit_i32)
        _if_rng = _scf_d.IfOp(_lv_unwrap(in_rng), [T.i32()], has_else=True)
        with _IP(_if_rng.then_block):
            exp_i = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32())
            is_match = arith.cmpi(arith.CmpIPredicate.eq, exp_i, expert_id)
            _if_m = _scf_d.IfOp(_lv_unwrap(is_match), [T.i32()], has_else=True)
            with _IP(_if_m.then_block):
                _scf_d.YieldOp([arith.addi(cnt, _c1)])
            with _IP(_if_m.else_block):
                _scf_d.YieldOp([cnt])
            _scf_d.YieldOp([_lv_unwrap(_if_m.result)])
        with _IP(_if_rng.else_block):
            _scf_d.YieldOp([cnt])
        _scf_d.YieldOp([_lv_unwrap(_if_rng.result)])
    return _for.results[0]


def _ceil_div_i32(a, b):
    _c1 = arith.constant(1, type=T.i32())
    return arith.divui(arith.addi(a, arith.subi(b, _c1)), b)


def _emit_moe_sorting_serial(
    *,
    _r_idx,
    _r_wts,
    _r_sorted_ids,
    _r_sorted_w,
    _r_sorted_expert_ids,
    _r_num_valid,
    cur_tok_i32,
    _ept_i32,
    _num_experts: int,
    _block_size_i32,
    _max_padded: int,
    _max_m_blocks: int,
    _max_slots: int,
):
    """Serial per-expert sort on lane 0 (metadata inline path)."""
    _c0_i32 = arith.constant(0, type=T.i32())
    _c1_i32 = arith.constant(1, type=T.i32())
    _cm1_i32 = arith.constant(-1, type=T.i32())
    _c24_i32 = arith.constant(24, type=T.i32())
    _ff_i32 = arith.constant(0xFFFFFF, type=T.i32())
    limit_i32 = arith.muli(cur_tok_i32, _ept_i32)
    init_val = arith.addi(
        arith.shli(_ept_i32, _c24_i32),
        arith.andi(cur_tok_i32, _ff_i32),
    )

    for off in range(_max_padded):
        buffer_store(_lv_unwrap(init_val), _r_sorted_ids, arith.constant(off, type=T.i32()))
    for off in range(_max_m_blocks):
        buffer_store(_lv_unwrap(_cm1_i32), _r_sorted_expert_ids, arith.constant(off, type=T.i32()))

    sorted_begin = _c0_i32
    expert_block_begin = _c0_i32

    for expert_id in range(_num_experts):
        expert_id_i32 = arith.constant(expert_id, type=T.i32())
        tokens_num = _count_tokens_for_expert(
            _r_idx, limit_i32, expert_id_i32, _ept_i32, _max_slots
        )
        blocks_e = _ceil_div_i32(tokens_num, _block_size_i32)
        pad_tokens = arith.muli(blocks_e, _block_size_i32)
        write_begin = sorted_begin

        local_written = _c0_i32
        _for_w = _scf_d.ForOp(
            _c0_i32,
            arith.constant(_max_slots, type=T.i32()),
            _c1_i32,
            [_c0_i32],
        )
        with _IP(_for_w.body):
            i = _for_w.induction_variable
            written = _for_w.inner_iter_args[0]
            in_rng = arith.cmpi(arith.CmpIPredicate.ult, i, limit_i32)
            _if_rng = _scf_d.IfOp(_lv_unwrap(in_rng), [T.i32()], has_else=True)
            with _IP(_if_rng.then_block):
                src_tok = arith.divui(i, _ept_i32)
                j = arith.remui(i, _ept_i32)
                exp_i = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32())
                is_match = arith.cmpi(arith.CmpIPredicate.eq, exp_i, expert_id_i32)
                _if_m = _scf_d.IfOp(_lv_unwrap(is_match), [T.i32()], has_else=True)
                with _IP(_if_m.then_block):
                    dst_off = arith.addi(write_begin, written)
                    packed = arith.addi(
                        arith.shli(j, _c24_i32),
                        arith.andi(src_tok, _ff_i32),
                    )
                    wt = buffer_load(_r_wts, i, vec_width=1, dtype=T.f32())
                    buffer_store(_lv_unwrap(packed), _r_sorted_ids, dst_off)
                    buffer_store(_lv_unwrap(wt), _r_sorted_w, dst_off)
                    _scf_d.YieldOp([arith.addi(written, _c1_i32)])
                with _IP(_if_m.else_block):
                    _scf_d.YieldOp([written])
                _scf_d.YieldOp([_lv_unwrap(_if_m.result)])
            with _IP(_if_rng.else_block):
                _scf_d.YieldOp([written])
            _scf_d.YieldOp([_lv_unwrap(_if_rng.result)])

        _if_nz = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.sgt, tokens_num, _c0_i32)),
            [T.i32(), T.i32()],
            has_else=True,
        )
        with _IP(_if_nz.then_block):
            for b in range(_max_m_blocks):
                b_i32 = arith.constant(b, type=T.i32())
                in_b = arith.cmpi(arith.CmpIPredicate.ult, b_i32, blocks_e)
                _if_b = _scf_d.IfOp(_lv_unwrap(in_b))
                with _IP(_if_b.then_block):
                    buffer_store(
                        _lv_unwrap(expert_id_i32),
                        _r_sorted_expert_ids,
                        arith.addi(expert_block_begin, b_i32),
                    )
                    _scf_d.YieldOp([])
            sorted_next = arith.addi(sorted_begin, pad_tokens)
            expert_next = arith.addi(expert_block_begin, blocks_e)
            _scf_d.YieldOp([sorted_next, expert_next])
        with _IP(_if_nz.else_block):
            _scf_d.YieldOp([sorted_begin, expert_block_begin])
        sorted_begin = _if_nz.results[0]
        expert_block_begin = _if_nz.results[1]

    buffer_store(_lv_unwrap(sorted_begin), _r_num_valid, _c0_i32)


def _emit_parallel_init(
    *,
    _r_sorted_ids,
    _r_sorted_expert_ids,
    cur_tok_i32,
    _ept_i32,
    _max_padded: int,
    _max_m_blocks: int,
    work_id_i32,
    gw_num_i32,
):
    """Grid-stride init of padding sentinel and expert block ids."""
    _c0_i32 = arith.constant(0, type=T.i32())
    _c1_i32 = arith.constant(1, type=T.i32())
    _cm1_i32 = arith.constant(-1, type=T.i32())
    _c24_i32 = arith.constant(24, type=T.i32())
    _ff_i32 = arith.constant(0xFFFFFF, type=T.i32())
    init_val = arith.addi(
        arith.shli(_ept_i32, _c24_i32),
        arith.andi(cur_tok_i32, _ff_i32),
    )
    _for = _scf_d.ForOp(
        work_id_i32,
        arith.constant(_max_padded, type=T.i32()),
        gw_num_i32,
        [],
    )
    with _IP(_for.body):
        off = _for.induction_variable
        buffer_store(_lv_unwrap(init_val), _r_sorted_ids, off)
        _scf_d.YieldOp([])
    _for2 = _scf_d.ForOp(
        work_id_i32,
        arith.constant(_max_m_blocks, type=T.i32()),
        gw_num_i32,
        [],
    )
    with _IP(_for2.body):
        off = _for2.induction_variable
        buffer_store(_lv_unwrap(_cm1_i32), _r_sorted_expert_ids, off)
        _scf_d.YieldOp([])


def _emit_parallel_ws_init(
    *,
    _r_ws,
    _ws_words: int,
    work_id_i32,
    gw_num_i32,
):
    """Grid-stride zero of count/scatter scratch (``ep_moe_sorting_init`` owns ``addr_workspace``)."""
    _c0_i32 = arith.constant(0, type=T.i32())
    _for = _scf_d.ForOp(
        work_id_i32,
        arith.constant(int(_ws_words), type=T.i32()),
        gw_num_i32,
        [],
    )
    with _IP(_for.body):
        off = _for.induction_variable
        buffer_store(_lv_unwrap(_c0_i32), _r_ws, off)
        _scf_d.YieldOp([])


def _emit_parallel_count_expert(
    *,
    _r_idx,
    expert_id_i32,
    limit_i32,
    _max_slots: int,
    tid_i32,
    gw_num_i32,
    _r_ws,
    ws_tokens_off,
    ws_blocks_off,
    ws_tcnt_off,
    _block_size_i32,
):
    """One CTA per expert: parallel slot scan + per-thread counts in workspace."""
    _c0_i32 = arith.constant(0, type=T.i32())
    _c1_i32 = arith.constant(1, type=T.i32())
    local_cnt = _c0_i32
    _for = _scf_d.ForOp(
        tid_i32,
        arith.constant(_max_slots, type=T.i32()),
        gw_num_i32,
        [_c0_i32],
    )
    with _IP(_for.body):
        i = _for.induction_variable
        cnt = _for.inner_iter_args[0]
        in_rng = arith.cmpi(arith.CmpIPredicate.ult, i, limit_i32)
        _if_rng = _scf_d.IfOp(_lv_unwrap(in_rng), [T.i32()], has_else=True)
        with _IP(_if_rng.then_block):
            exp_i = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32())
            is_match = arith.cmpi(arith.CmpIPredicate.eq, exp_i, expert_id_i32)
            _if_m = _scf_d.IfOp(_lv_unwrap(is_match), [T.i32()], has_else=True)
            with _IP(_if_m.then_block):
                _scf_d.YieldOp([arith.addi(cnt, _c1_i32)])
            with _IP(_if_m.else_block):
                _scf_d.YieldOp([cnt])
            _scf_d.YieldOp([_lv_unwrap(_if_m.result)])
        with _IP(_if_rng.else_block):
            _scf_d.YieldOp([cnt])
        _scf_d.YieldOp([_lv_unwrap(_if_rng.result)])
    local_cnt = _for.results[0]

    buffer_store(_lv_unwrap(local_cnt), _r_ws, arith.addi(ws_tcnt_off, tid_i32))

    _if_t0 = _scf_d.IfOp(
        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid_i32, _c0_i32)),
        has_else=True,
    )
    with _IP(_if_t0.then_block):
        _for_sum = _scf_d.ForOp(
            _c0_i32,
            arith.constant(_WARP_THREADS, type=T.i32()),
            _c1_i32,
            [_c0_i32],
        )
        with _IP(_for_sum.body):
            t_i32 = _for_sum.induction_variable
            acc = _for_sum.inner_iter_args[0]
            tc = buffer_load(
                _r_ws, arith.addi(ws_tcnt_off, t_i32), vec_width=1, dtype=T.i32()
            )
            _scf_d.YieldOp([arith.addi(acc, tc)])
        tokens_num = _for_sum.results[0]
        blocks_e = _ceil_div_i32(tokens_num, _block_size_i32)
        buffer_store(_lv_unwrap(tokens_num), _r_ws, ws_tokens_off)
        buffer_store(_lv_unwrap(blocks_e), _r_ws, ws_blocks_off)
        _scf_d.YieldOp([])
    with _IP(_if_t0.else_block):
        _scf_d.YieldOp([])


def _prefix_pad_before_expert(_r_ws, expert_id_i32, _block_size_i32, _num_experts: int):
    """Exclusive prefix of padded tokens / M-blocks for experts with id < expert_id_i32."""
    _c0_i32 = arith.constant(0, type=T.i32())
    _c1_i32 = arith.constant(1, type=T.i32())
    _ne_i32 = arith.constant(int(_num_experts), type=T.i32())
    _for = _scf_d.ForOp(
        _c0_i32,
        arith.constant(_num_experts, type=T.i32()),
        _c1_i32,
        [_c0_i32, _c0_i32],
    )
    with _IP(_for.body):
        i = _for.induction_variable
        sorted_begin = _for.inner_iter_args[0]
        expert_block_begin = _for.inner_iter_args[1]
        _if = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, i, expert_id_i32)),
            [T.i32(), T.i32()],
            has_else=True,
        )
        with _IP(_if.then_block):
            blk_n = buffer_load(
                _r_ws,
                arith.addi(_ne_i32, i),
                vec_width=1,
                dtype=T.i32(),
            )
            pad_i = arith.muli(blk_n, _block_size_i32)
            _scf_d.YieldOp(
                [
                    arith.addi(sorted_begin, pad_i),
                    arith.addi(expert_block_begin, blk_n),
                ]
            )
        with _IP(_if.else_block):
            _scf_d.YieldOp([sorted_begin, expert_block_begin])
        _scf_d.YieldOp(list(_if.results))
    return _for.results[0], _for.results[1]


def _emit_parallel_scatter_expert(
    *,
    _r_idx,
    _r_wts,
    _r_sorted_ids,
    _r_sorted_w,
    _r_sorted_expert_ids,
    _r_num_valid,
    expert_id_i32,
    _num_experts: int,
    limit_i32,
    _ept_i32,
    _max_slots: int,
    _max_m_blocks: int,
    tid_i32,
    _r_ws,
    ws_tokens_off,
    ws_blocks_off,
    _block_size_i32,
):
    """Per-expert CTA: lane 0 scatters in slot order (matches torch/aiter)."""
    _c0_i32 = arith.constant(0, type=T.i32())
    _c1_i32 = arith.constant(1, type=T.i32())
    _c24_i32 = arith.constant(24, type=T.i32())
    _ff_i32 = arith.constant(0xFFFFFF, type=T.i32())
    _c0_tid = arith.constant(0, type=T.i32())

    _if_lane0 = _scf_d.IfOp(
        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid_i32, _c0_tid)),
        has_else=True,
    )
    with _IP(_if_lane0.then_block):
        tokens_num = buffer_load(_r_ws, ws_tokens_off, vec_width=1, dtype=T.i32())
        blocks_e = buffer_load(_r_ws, ws_blocks_off, vec_width=1, dtype=T.i32())
        sorted_begin, expert_block_begin = _prefix_pad_before_expert(
            _r_ws, expert_id_i32, _block_size_i32, _num_experts
        )
        write_begin = sorted_begin
        _for_w = _scf_d.ForOp(
            _c0_i32,
            arith.constant(_max_slots, type=T.i32()),
            _c1_i32,
            [_c0_i32],
        )
        with _IP(_for_w.body):
            i = _for_w.induction_variable
            written = _for_w.inner_iter_args[0]
            in_rng = arith.cmpi(arith.CmpIPredicate.ult, i, limit_i32)
            _if_rng = _scf_d.IfOp(_lv_unwrap(in_rng), [T.i32()], has_else=True)
            with _IP(_if_rng.then_block):
                src_tok = arith.divui(i, _ept_i32)
                j = arith.remui(i, _ept_i32)
                exp_i = buffer_load(_r_idx, i, vec_width=1, dtype=T.i32())
                is_match = arith.cmpi(arith.CmpIPredicate.eq, exp_i, expert_id_i32)
                _if_m = _scf_d.IfOp(_lv_unwrap(is_match), [T.i32()], has_else=True)
                with _IP(_if_m.then_block):
                    dst_off = arith.addi(write_begin, written)
                    packed = arith.addi(
                        arith.shli(j, _c24_i32),
                        arith.andi(src_tok, _ff_i32),
                    )
                    wt = buffer_load(_r_wts, i, vec_width=1, dtype=T.f32())
                    buffer_store(_lv_unwrap(packed), _r_sorted_ids, dst_off)
                    buffer_store(_lv_unwrap(wt), _r_sorted_w, dst_off)
                    _scf_d.YieldOp([arith.addi(written, _c1_i32)])
                with _IP(_if_m.else_block):
                    _scf_d.YieldOp([written])
                _scf_d.YieldOp([_lv_unwrap(_if_m.result)])
            with _IP(_if_rng.else_block):
                _scf_d.YieldOp([written])
            _scf_d.YieldOp([_lv_unwrap(_if_rng.result)])

        _if_nz = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.sgt, tokens_num, _c0_i32)),
            has_else=True,
        )
        with _IP(_if_nz.then_block):
            for b in range(_max_m_blocks):
                b_i32 = arith.constant(b, type=T.i32())
                in_b = arith.cmpi(arith.CmpIPredicate.ult, b_i32, blocks_e)
                _if_b = _scf_d.IfOp(_lv_unwrap(in_b), has_else=True)
                with _IP(_if_b.then_block):
                    buffer_store(
                        _lv_unwrap(expert_id_i32),
                        _r_sorted_expert_ids,
                        arith.addi(expert_block_begin, b_i32),
                    )
                    _scf_d.YieldOp([])
                with _IP(_if_b.else_block):
                    _scf_d.YieldOp([])
            _scf_d.YieldOp([])
        with _IP(_if_nz.else_block):
            _scf_d.YieldOp([])

        _last_expert_i32 = arith.constant(_num_experts - 1, type=T.i32())
        _if_last = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, expert_id_i32, _last_expert_i32)),
            has_else=True,
        )
        with _IP(_if_last.then_block):
            _for_tot = _scf_d.ForOp(
                _c0_i32,
                arith.constant(_num_experts, type=T.i32()),
                _c1_i32,
                [_c0_i32],
            )
            with _IP(_for_tot.body):
                i = _for_tot.induction_variable
                acc = _for_tot.inner_iter_args[0]
                blk_n = buffer_load(
                    _r_ws,
                    arith.addi(arith.constant(_num_experts, type=T.i32()), i),
                    vec_width=1,
                    dtype=T.i32(),
                )
                _scf_d.YieldOp([arith.addi(acc, arith.muli(blk_n, _block_size_i32))])
            buffer_store(_lv_unwrap(_for_tot.results[0]), _r_num_valid, _c0_i32)
            _scf_d.YieldOp([])
        with _IP(_if_last.else_block):
            _scf_d.YieldOp([])
        _scf_d.YieldOp([])
    with _IP(_if_lane0.else_block):
        _scf_d.YieldOp([])


def moe_sorting_const_sizes(*, max_tok_per_rank: int, experts_per_token: int, num_experts: int, block_size: int):
    ept = int(experts_per_token)
    e = int(num_experts)
    bs = int(block_size)
    mtpr = int(max_tok_per_rank)
    max_padded = mtpr * ept + e * bs - ept
    max_m_blocks = (max_padded + bs - 1) // bs
    max_slots = mtpr * ept
    return int(max_padded), int(max_m_blocks), int(max_slots)


def moe_sorting_workspace_done_off(num_experts: int) -> int:
    """Device int32: number of experts that finished scatter (``== num_experts`` → mxfp4 may run)."""
    e = int(num_experts)
    return 2 * e + _WARP_THREADS * e


def moe_sorting_workspace_ints(num_experts: int) -> int:
    """``ws[0:E)`` tokens, ``ws[E:2E)`` blocks, ``ws[2E:2E+64*E)`` per-thread counts, + done."""
    return moe_sorting_workspace_done_off(num_experts) + 1


def _emit_atomic_inc_workspace_i32(addr_workspace: fx.Int64, done_byte_off: int) -> None:
    """``atomic_add`` on ``workspace[done_byte_off // 4]`` (global monotonic, cross-CTA sync)."""
    addr_done = arith.addi(
        _lv_unwrap(addr_workspace),
        arith.constant(int(done_byte_off), type=T.i64()),
    )
    atomic_add_global_at(addr_done, arith.constant(1, type=T.i32()))


def _emit_spin_until_sort_done(
    _r_ws,
    ws_done_off_i32,
    ne_i32,
    tid_i32,
) -> None:
    """Mxfp4 CTAs wait until all expert scatter CTAs incremented ``workspace[done_off]``."""
    from flydsl._mlir import ir

    _c0_tid = arith.constant(0, type=T.i32())
    _if_lane0 = _scf_d.IfOp(
        _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid_i32, _c0_tid)),
        has_else=True,
    )
    with _IP(_if_lane0.then_block):
        init_done = buffer_load(_r_ws, ws_done_off_i32, vec_width=1, dtype=T.i32())
        w = _scf_d.WhileOp([T.i32()], [init_done])
        wb = ir.Block.create_at_start(w.before, [T.i32()])
        wa = ir.Block.create_at_start(w.after, [T.i32()])
        with _IP(wb):
            cur = wb.arguments[0]
            ready = arith.cmpi(arith.CmpIPredicate.uge, cur, ne_i32)
            _scf_d.ConditionOp(_lv_unwrap(ready), [cur])
        with _IP(wa):
            cur = wa.arguments[0]
            still = arith.cmpi(arith.CmpIPredicate.ult, cur, ne_i32)
            _if_spin = _scf_d.IfOp(_lv_unwrap(still), [T.i32()], has_else=True)
            with _IP(_if_spin.then_block):
                nxt = buffer_load(_r_ws, ws_done_off_i32, vec_width=1, dtype=T.i32())
                _scf_d.YieldOp([nxt])
            with _IP(_if_spin.else_block):
                _scf_d.YieldOp([cur])
            _scf_d.YieldOp([_lv_unwrap(_if_spin.result)])
        _scf_d.YieldOp([])
    with _IP(_if_lane0.else_block):
        _scf_d.YieldOp([])


def make_moe_sorting_kernel_parallel(
    *,
    num_experts: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    sort_block_m: int,
):
    _max_padded, _max_m_blocks, _max_slots = moe_sorting_const_sizes(
        max_tok_per_rank=max_tok_per_rank,
        experts_per_token=experts_per_token,
        num_experts=num_experts,
        block_size=sort_block_m,
    )
    _ne = int(num_experts)
    _ws_words = moe_sorting_workspace_ints(num_experts)
    _init_grid = (max(_max_padded, _max_m_blocks) + _WARP_THREADS - 1) // _WARP_THREADS
    _init_grid = max(
        _init_grid,
        (_ws_words + _WARP_THREADS - 1) // _WARP_THREADS,
    )

    @flyc.kernel(known_block_size=[_WARP_THREADS, 1, 1])
    def ep_moe_sorting_init(
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_workspace: fx.Int64,
        cur_tok: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gw_num = fx.grid_dim.x
        cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        bid_i32 = arith.index_cast(T.i32(), bid.ir_value())
        gw_num_i32 = arith.index_cast(T.i32(), gw_num.ir_value())
        work_id = arith.addi(arith.muli(bid_i32, arith.constant(_WARP_THREADS, type=T.i32())), tid_i32)
        _ept_i32 = arith.constant(int(experts_per_token), type=T.i32())
        _r_sorted_ids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_token_ids))
        _r_sorted_expert_ids = create_buffer_resource_from_addr(
            _lv_unwrap(addr_sorted_expert_ids)
        )
        _r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_workspace))
        _emit_parallel_ws_init(
            _r_ws=_r_ws,
            _ws_words=_ws_words,
            work_id_i32=work_id,
            gw_num_i32=gw_num_i32,
        )
        _emit_parallel_init(
            _r_sorted_ids=_r_sorted_ids,
            _r_sorted_expert_ids=_r_sorted_expert_ids,
            cur_tok_i32=cur_tok_i32,
            _ept_i32=_ept_i32,
            _max_padded=_max_padded,
            _max_m_blocks=_max_m_blocks,
            work_id_i32=work_id,
            gw_num_i32=gw_num_i32,
        )

    @flyc.kernel(known_block_size=[_WARP_THREADS, 1, 1])
    def ep_moe_sorting_count(
        addr_idx: fx.Int64,
        addr_workspace: fx.Int64,
        cur_tok: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        expert_id_i32 = arith.index_cast(T.i32(), bid.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        gw_num_i32 = arith.constant(_WARP_THREADS, type=T.i32())
        cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
        _ept_i32 = arith.constant(int(experts_per_token), type=T.i32())
        _block_size_i32 = arith.constant(int(sort_block_m), type=T.i32())
        limit_i32 = arith.muli(cur_tok_i32, _ept_i32)
        _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
        _r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_workspace))
        ws_tokens_off = expert_id_i32
        ws_blocks_off = arith.addi(expert_id_i32, arith.constant(_ne, type=T.i32()))
        ws_tcnt_base = arith.addi(
            arith.constant(2 * _ne, type=T.i32()),
            arith.muli(expert_id_i32, gw_num_i32),
        )
        _emit_parallel_count_expert(
            _r_idx=_r_idx,
            expert_id_i32=expert_id_i32,
            limit_i32=limit_i32,
            _max_slots=_max_slots,
            tid_i32=tid_i32,
            gw_num_i32=gw_num_i32,
            _r_ws=_r_ws,
            ws_tokens_off=ws_tokens_off,
            ws_blocks_off=ws_blocks_off,
            ws_tcnt_off=ws_tcnt_base,
            _block_size_i32=_block_size_i32,
        )

    @flyc.kernel(known_block_size=[_WARP_THREADS, 1, 1])
    def ep_moe_sorting_scatter(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_workspace: fx.Int64,
        cur_tok: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        expert_id_i32 = arith.index_cast(T.i32(), bid.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        gw_num_i32 = arith.constant(_WARP_THREADS, type=T.i32())
        cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
        _ept_i32 = arith.constant(int(experts_per_token), type=T.i32())
        _block_size_i32 = arith.constant(int(sort_block_m), type=T.i32())
        limit_i32 = arith.muli(cur_tok_i32, _ept_i32)
        _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
        _r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
        _r_sorted_ids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_token_ids))
        _r_sorted_w = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_weights))
        _r_sorted_expert_ids = create_buffer_resource_from_addr(
            _lv_unwrap(addr_sorted_expert_ids)
        )
        _r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        _r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_workspace))
        ws_tokens_off = expert_id_i32
        ws_blocks_off = arith.addi(expert_id_i32, arith.constant(_ne, type=T.i32()))
        _emit_parallel_scatter_expert(
            _r_idx=_r_idx,
            _r_wts=_r_wts,
            _r_sorted_ids=_r_sorted_ids,
            _r_sorted_w=_r_sorted_w,
            _r_sorted_expert_ids=_r_sorted_expert_ids,
            _r_num_valid=_r_num_valid,
            expert_id_i32=expert_id_i32,
            _num_experts=_ne,
            limit_i32=limit_i32,
            _ept_i32=_ept_i32,
            _max_slots=_max_slots,
            _max_m_blocks=_max_m_blocks,
            tid_i32=tid_i32,
            _r_ws=_r_ws,
            ws_tokens_off=ws_tokens_off,
            ws_blocks_off=ws_blocks_off,
            _block_size_i32=_block_size_i32,
        )

    @flyc.kernel(known_block_size=[_WARP_THREADS, 1, 1])
    def ep_moe_sorting_count_scatter(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_workspace: fx.Int64,
        cur_tok: fx.Int32,
    ):
        """One CTA per expert: parallel count then lane-0 scatter (one launch vs count+scatter)."""
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        expert_id_i32 = arith.index_cast(T.i32(), bid.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        gw_num_i32 = arith.constant(_WARP_THREADS, type=T.i32())
        cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
        _ept_i32 = arith.constant(int(experts_per_token), type=T.i32())
        _block_size_i32 = arith.constant(int(sort_block_m), type=T.i32())
        limit_i32 = arith.muli(cur_tok_i32, _ept_i32)
        _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
        _r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
        _r_sorted_ids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_token_ids))
        _r_sorted_w = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_weights))
        _r_sorted_expert_ids = create_buffer_resource_from_addr(
            _lv_unwrap(addr_sorted_expert_ids)
        )
        _r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        _r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_workspace))
        ws_tokens_off = expert_id_i32
        ws_blocks_off = arith.addi(expert_id_i32, arith.constant(_ne, type=T.i32()))
        ws_tcnt_base = arith.addi(
            arith.constant(2 * _ne, type=T.i32()),
            arith.muli(expert_id_i32, gw_num_i32),
        )
        _emit_parallel_count_expert(
            _r_idx=_r_idx,
            expert_id_i32=expert_id_i32,
            limit_i32=limit_i32,
            _max_slots=_max_slots,
            tid_i32=tid_i32,
            gw_num_i32=gw_num_i32,
            _r_ws=_r_ws,
            ws_tokens_off=ws_tokens_off,
            ws_blocks_off=ws_blocks_off,
            ws_tcnt_off=ws_tcnt_base,
            _block_size_i32=_block_size_i32,
        )
        fx.gpu.barrier()
        _emit_parallel_scatter_expert(
            _r_idx=_r_idx,
            _r_wts=_r_wts,
            _r_sorted_ids=_r_sorted_ids,
            _r_sorted_w=_r_sorted_w,
            _r_sorted_expert_ids=_r_sorted_expert_ids,
            _r_num_valid=_r_num_valid,
            expert_id_i32=expert_id_i32,
            _num_experts=_ne,
            limit_i32=limit_i32,
            _ept_i32=_ept_i32,
            _max_slots=_max_slots,
            _max_m_blocks=_max_m_blocks,
            tid_i32=tid_i32,
            _r_ws=_r_ws,
            ws_tokens_off=ws_tokens_off,
            ws_blocks_off=ws_blocks_off,
            _block_size_i32=_block_size_i32,
        )

    return (
        ep_moe_sorting_init,
        ep_moe_sorting_count,
        ep_moe_sorting_scatter,
        ep_moe_sorting_count_scatter,
        _init_grid,
        _ne,
    )


def make_moe_sorting_scatter_mxfp4_kernel(
    *,
    num_experts: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    sort_block_m: int,
    scale_mx_blocks: int,
):
    """One launch: expert scatter CTAs (``bid < E``) then mxfp4 tile CTAs (``bid >= E``)."""
    _max_padded, _max_m_blocks, _max_slots = moe_sorting_const_sizes(
        max_tok_per_rank=max_tok_per_rank,
        experts_per_token=experts_per_token,
        num_experts=num_experts,
        block_size=sort_block_m,
    )
    _ne = int(num_experts)
    _ws_done_off = moe_sorting_workspace_done_off(_ne)
    _gm, _gn, _bs_m, _bs_n, _m_o, _n_i, _, _strides = mxfp4_sort_const_geom(
        max_sorted_rows=_max_padded, scale_mx_blocks=int(scale_mx_blocks)
    )
    _stride_o0, _stride_o1, _stride_o2, _stride_o3 = _strides
    _stride_in_m = int(scale_mx_blocks)
    _stride_in_n = 1
    _ept = int(experts_per_token)
    _mx_ctas = int(_gm) * int(_gn)
    _scatter_mx_grid = _ne + _mx_ctas

    @flyc.kernel(known_block_size=[_WARP_THREADS, 1, 1])
    def ep_moe_sorting_scatter_mxfp4(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_workspace: fx.Int64,
        addr_scale_mx_in: fx.Int64,
        addr_scale_x_sorted: fx.Int64,
        cur_tok: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        bid_i32 = arith.index_cast(T.i32(), bid.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        ne_i32 = arith.constant(_ne, type=T.i32())
        cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
        _ept_i32 = arith.constant(int(experts_per_token), type=T.i32())
        _block_size_i32 = arith.constant(int(sort_block_m), type=T.i32())
        limit_i32 = arith.muli(cur_tok_i32, _ept_i32)
        ws_done_off_i32 = arith.constant(int(_ws_done_off), type=T.i32())
        _r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_workspace))
        _if_expert = _scf_d.IfOp(
            _lv_unwrap(arith.cmpi(arith.CmpIPredicate.ult, bid_i32, ne_i32)),
            has_else=True,
        )
        with _IP(_if_expert.then_block):
            expert_id_i32 = bid_i32
            _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
            _r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
            _r_sorted_ids = create_buffer_resource_from_addr(
                _lv_unwrap(addr_sorted_token_ids)
            )
            _r_sorted_w = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_weights))
            _r_sorted_expert_ids = create_buffer_resource_from_addr(
                _lv_unwrap(addr_sorted_expert_ids)
            )
            _r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
            ws_tokens_off = expert_id_i32
            ws_blocks_off = arith.addi(expert_id_i32, ne_i32)
            _emit_parallel_scatter_expert(
                _r_idx=_r_idx,
                _r_wts=_r_wts,
                _r_sorted_ids=_r_sorted_ids,
                _r_sorted_w=_r_sorted_w,
                _r_sorted_expert_ids=_r_sorted_expert_ids,
                _r_num_valid=_r_num_valid,
                expert_id_i32=expert_id_i32,
                _num_experts=_ne,
                limit_i32=limit_i32,
                _ept_i32=_ept_i32,
                _max_slots=_max_slots,
                _max_m_blocks=_max_m_blocks,
                tid_i32=tid_i32,
                _r_ws=_r_ws,
                ws_tokens_off=ws_tokens_off,
                ws_blocks_off=ws_blocks_off,
                _block_size_i32=_block_size_i32,
            )
            _c0_tid = arith.constant(0, type=T.i32())
            _if_done = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid_i32, _c0_tid)),
                has_else=True,
            )
            with _IP(_if_done.then_block):
                _emit_atomic_inc_workspace_i32(
                    addr_workspace, int(_ws_done_off) * 4
                )
                _scf_d.YieldOp([])
            with _IP(_if_done.else_block):
                _scf_d.YieldOp([])
            _scf_d.YieldOp([])
        with _IP(_if_expert.else_block):
            _emit_spin_until_sort_done(_r_ws, ws_done_off_i32, ne_i32, tid_i32)
            tile_id_i32 = arith.subi(bid_i32, ne_i32)
            gn_i32 = arith.constant(int(_gn), type=T.i32())
            prog_m_i32 = arith.divui(tile_id_i32, gn_i32)
            prog_n_i32 = arith.remui(tile_id_i32, gn_i32)
            _c0_tid = arith.constant(0, type=T.i32())
            _if_mx = _scf_d.IfOp(
                _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, tid_i32, _c0_tid)),
                has_else=True,
            )
            with _IP(_if_mx.then_block):
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
                _c0 = arith.constant(0, type=T.i32())
                num_valid = buffer_load(_r_num_valid, _c0, vec_width=1, dtype=T.i32())
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
                _scf_d.YieldOp([])
            with _IP(_if_mx.else_block):
                _scf_d.YieldOp([])
            _scf_d.YieldOp([])

    return ep_moe_sorting_scatter_mxfp4, _scatter_mx_grid, _ne


def make_moe_sorting_kernel(
    *,
    num_experts: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    sort_block_m: int,
):
    """Legacy serial kernel (metadata inline)."""
    _max_padded, _max_m_blocks, _max_slots = moe_sorting_const_sizes(
        max_tok_per_rank=max_tok_per_rank,
        experts_per_token=experts_per_token,
        num_experts=num_experts,
        block_size=sort_block_m,
    )

    @flyc.kernel(known_block_size=[_WARP_THREADS, 1, 1])
    def ep_moe_sorting(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        cur_tok: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        cur_tok_i32 = arith.index_cast(T.i32(), cur_tok.ir_value())
        _c0_gw = arith.constant(0, type=T.i32())
        _c0_tid = arith.constant(0, type=T.i32())
        bid_i32 = arith.index_cast(T.i32(), bid.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        if arith.cmpi(arith.CmpIPredicate.eq, bid_i32, _c0_gw):
            if arith.cmpi(arith.CmpIPredicate.eq, tid_i32, _c0_tid):
                _r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
                _r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
                _r_sorted_ids = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_sorted_token_ids)
                )
                _r_sorted_w = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_sorted_weights)
                )
                _r_sorted_expert_ids = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_sorted_expert_ids)
                )
                _r_num_valid = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_num_valid_ids)
                )
                _ept_i32 = arith.constant(int(experts_per_token), type=T.i32())
                _block_size_i32 = arith.constant(int(sort_block_m), type=T.i32())
                _emit_moe_sorting_serial(
                    _r_idx=_r_idx,
                    _r_wts=_r_wts,
                    _r_sorted_ids=_r_sorted_ids,
                    _r_sorted_w=_r_sorted_w,
                    _r_sorted_expert_ids=_r_sorted_expert_ids,
                    _r_num_valid=_r_num_valid,
                    cur_tok_i32=cur_tok_i32,
                    _ept_i32=_ept_i32,
                    _num_experts=int(num_experts),
                    _block_size_i32=_block_size_i32,
                    _max_padded=_max_padded,
                    _max_m_blocks=_max_m_blocks,
                    _max_slots=_max_slots,
                )

    return ep_moe_sorting


@functools.lru_cache(maxsize=32)
def _sort_workspace_cache_key(num_experts: int, device_index: int) -> tuple[int, int]:
    return int(num_experts), int(device_index)


def get_moe_sorting_workspace(
    *,
    num_experts: int,
    device: Any,
    min_ints: int = 0,
) -> Any:
    import torch

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    n = max(moe_sorting_workspace_ints(num_experts), int(min_ints))
    return torch.zeros(n, dtype=torch.int32, device=dev)


def make_moe_sorting_jit(
    *,
    num_experts: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    sort_block_m: int,
    scale_mx_blocks: int | None = None,
    fuse_mxfp4_sort: bool = True,
):
    """Parallel MoE sort JIT.

    When ``fuse_mxfp4_sort`` and ``scale_mx_blocks`` are set, ``ep_moe_sorting_scatter_mxfp4``
    replaces separate ``scatter`` + ``moe_mxfp4_sort`` launches (one fewer device launch).
    """
    (
        k_init,
        k_count,
        k_scatter,
        k_count_scatter,
        init_grid,
        ne,
    ) = make_moe_sorting_kernel_parallel(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        max_tok_per_rank=max_tok_per_rank,
        sort_block_m=sort_block_m,
    )
    use_fused_mxfp4 = bool(fuse_mxfp4_sort) and scale_mx_blocks is not None and int(scale_mx_blocks) > 0
    k_mxfp4 = None
    mxfp4_grid = (0, 0)
    if use_fused_mxfp4:
        from kernels.moe_mxfp4_sort_flydsl import make_moe_mxfp4_sort_kernel, mxfp4_sort_const_geom

        _max_padded, _, _ = moe_sorting_const_sizes(
            max_tok_per_rank=max_tok_per_rank,
            experts_per_token=experts_per_token,
            num_experts=num_experts,
            block_size=sort_block_m,
        )
        k_mxfp4 = make_moe_mxfp4_sort_kernel(
            experts_per_token=experts_per_token,
            max_sorted_rows=_max_padded,
            scale_mx_blocks=int(scale_mx_blocks),
        )
        mxfp4_grid = mxfp4_sort_const_geom(
            max_sorted_rows=_max_padded, scale_mx_blocks=int(scale_mx_blocks)
        )[:2]

    def _launch_sort_core(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_workspace: fx.Int64,
        cur_tok: fx.Int32,
        stream: Stream,
    ):
        k_init(
            addr_sorted_token_ids,
            addr_sorted_expert_ids,
            addr_workspace,
            cur_tok,
        ).launch(grid=(init_grid, 1, 1), block=(_WARP_THREADS, 1, 1), stream=stream)
        k_count(addr_idx, addr_workspace, cur_tok).launch(
            grid=(ne, 1, 1), block=(_WARP_THREADS, 1, 1), stream=stream
        )
        k_scatter(
            addr_idx,
            addr_wts,
            addr_sorted_token_ids,
            addr_sorted_weights,
            addr_sorted_expert_ids,
            addr_num_valid_ids,
            addr_workspace,
            cur_tok,
        ).launch(grid=(ne, 1, 1), block=(_WARP_THREADS, 1, 1), stream=stream)

    if use_fused_mxfp4 and k_mxfp4 is not None:

        @flyc.jit
        def launch_moe_sorting(
            addr_idx: fx.Int64,
            addr_wts: fx.Int64,
            addr_sorted_token_ids: fx.Int64,
            addr_sorted_weights: fx.Int64,
            addr_sorted_expert_ids: fx.Int64,
            addr_num_valid_ids: fx.Int64,
            addr_workspace: fx.Int64,
            cur_tok: fx.Int32,
            addr_scale_mx_in: fx.Int64,
            addr_scale_x_sorted: fx.Int64,
            stream: Stream = Stream(None),
        ):
            _launch_sort_core(
                addr_idx,
                addr_wts,
                addr_sorted_token_ids,
                addr_sorted_weights,
                addr_sorted_expert_ids,
                addr_num_valid_ids,
                addr_workspace,
                cur_tok,
                stream,
            )
            k_mxfp4(
                addr_scale_mx_in,
                addr_sorted_token_ids,
                addr_num_valid_ids,
                addr_scale_x_sorted,
                cur_tok,
            ).launch(
                grid=(int(mxfp4_grid[0]), int(mxfp4_grid[1]), 1),
                block=(_WARP_THREADS, 1, 1),
                stream=stream,
            )
    else:

        @flyc.jit
        def launch_moe_sorting(
            addr_idx: fx.Int64,
            addr_wts: fx.Int64,
            addr_sorted_token_ids: fx.Int64,
            addr_sorted_weights: fx.Int64,
            addr_sorted_expert_ids: fx.Int64,
            addr_num_valid_ids: fx.Int64,
            addr_workspace: fx.Int64,
            cur_tok: fx.Int32,
            stream: Stream = Stream(None),
        ):
            _launch_sort_core(
                addr_idx,
                addr_wts,
                addr_sorted_token_ids,
                addr_sorted_weights,
                addr_sorted_expert_ids,
                addr_num_valid_ids,
                addr_workspace,
                cur_tok,
                stream,
            )

    launch_moe_sorting.fuse_mxfp4_sort = use_fused_mxfp4  # type: ignore[attr-defined]
    return launch_moe_sorting


__all__ = [
    "_emit_moe_sorting_serial",
    "moe_sorting_const_sizes",
    "moe_sorting_workspace_done_off",
    "moe_sorting_workspace_ints",
    "get_moe_sorting_workspace",
    "make_moe_sorting_jit",
    "make_moe_sorting_kernel",
    "make_moe_sorting_kernel_parallel",
    "make_moe_sorting_scatter_mxfp4_kernel",
]
