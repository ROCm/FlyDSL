# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Intranode Stage1 A8W4 pull-to-pool kernel."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl._mlir.ir import IntegerType as _IntTy_mlir
from flydsl.expr import T, arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr import Stream
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import T as _FlyT
import mori.ir.flydsl as mori_shmem

from kernels.dispatch_combine_intranode_kernel import (
    _lv_unwrap,
    _to_i32,
    _to_idx,
    atomic_add_global_at,
    ballot_i64,
    readlane,
    store_i32_system,
)
from kernels.moe_mxfp4_sort_flydsl import mxfp4_sort_const_geom
from kernels.moe_sorting_flydsl import moe_sorting_const_sizes


_RECV_META_ROW_BYTES = 32
_POOL_ROW_WORD = 6


def _global_device_ptr_from_i64(addr_i64):
    """i64 -> !llvm.ptr<1> (global address space)."""
    return llvm.IntToPtrOp(
        llvm.PointerType.get(address_space=1), _lv_unwrap(addr_i64)
    ).result


def _i64_cell_addr(addr_base, word_off_i32):
    """addr_base + word_off_i32 * sizeof(i32)."""
    _i64 = _IntTy_mlir.get_signless(64)
    off64 = arith.zext_i64(word_off_i32)
    byte_off = arith.muli(off64, arith.constant(4, type=_i64))
    return arith.addi(_lv_unwrap(addr_base), byte_off)


def _atomic_add_global_acq_rel(addr_i64, val):
    """Global atomic fetch-add with acq_rel semantics."""
    ptr = _global_device_ptr_from_i64(addr_i64)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr, _lv_unwrap(val), llvm.AtomicOrdering.acq_rel
    ).res


def _load_i32_global_acquire_cell(addr_base, word_index_i32):
    """Acquire-load an i32 cell from a global i32 workspace."""
    ptr = _global_device_ptr_from_i64(_i64_cell_addr(addr_base, word_index_i32))
    _i32 = _IntTy_mlir.get_signless(32)
    return llvm.LoadOp(
        _i32, ptr, alignment=4, ordering=llvm.AtomicOrdering.acquire
    ).result


def _store_i32_global_release_cell(addr_base, word_index_i32, val):
    """Release-store an i32 cell into a global i32 workspace."""
    ptr = _global_device_ptr_from_i64(_i64_cell_addr(addr_base, word_index_i32))
    llvm.StoreOp(_lv_unwrap(val), ptr, alignment=4, ordering=llvm.AtomicOrdering.release)


def make_pull_to_pool_a8w4_jit(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    num_experts: int,
    tile_m: int,
    block_num: int,
    warp_num_per_block: int,
    token_row_bytes: int,
    scale_mx_blocks: int,
    fuse_metadata: bool = False,
):
    """JIT: recv_meta -> local expert pool + sorted scheduling metadata."""
    _rank = int(rank)
    _npes = int(npes)
    _epr = int(experts_per_rank)
    _ept = int(experts_per_token)
    _mtpr = int(max_tok_per_rank)
    _ne = int(num_experts)
    _tile_m = int(tile_m)
    _max_recv = int(npes) * int(max_tok_per_rank) * _ept
    _nbytes = int(token_row_bytes)
    _n_i32 = _nbytes // 4
    _scale_bytes = int(scale_mx_blocks)
    _scale_i32_words = (_scale_bytes + 3) // 4
    _fuse_metadata = bool(fuse_metadata)
    _max_recv_payload = int(npes) * int(max_tok_per_rank)
    _recv_meta_words = _max_recv * 8
    _tok_map_words = _mtpr * _ept
    _tis_words = _max_recv_payload
    _reset_sentinel = _npes * _max_recv_payload
    _safe_staging_end = (_n_i32 // 512) * 512
    _max_padded, _max_m_blocks, _ = moe_sorting_const_sizes(
        max_tok_per_rank=_mtpr,
        experts_per_token=_ept,
        num_experts=_ne,
        block_size=_tile_m,
    )
    _mx_gm, _mx_gn, _, _, _, _, _, _mx_strides = mxfp4_sort_const_geom(
        max_sorted_rows=_max_padded,
        scale_mx_blocks=int(scale_mx_blocks),
    )
    _mx_stride_o0, _mx_stride_o1, _mx_stride_o2, _mx_stride_o3 = _mx_strides
    _scale_out_words = int(_mx_gm) * int(_mx_gn) * 4 * 16
    _threads_per_block = int(warp_num_per_block) * 64
    _ws_pool_slot_base = 3 * _ne
    _ws_row_src_base = _ws_pool_slot_base + _max_padded
    _ws_block_row_start_base = _ws_row_src_base + _max_padded
    _ws_block_valid_base = _ws_block_row_start_base + _max_m_blocks
    _ws_arrival_base = _ws_block_valid_base + _max_m_blocks
    _ws_canon_row_base = _ws_arrival_base + _max_m_blocks
    _ws_k2_barrier_base = _ws_canon_row_base + _max_padded
    _ws_k2_barrier_count = _ws_k2_barrier_base
    _ws_k2_barrier_phase = _ws_k2_barrier_base + 1
    _ws_k2_dup_flag = _ws_k2_barrier_base + 2
    # Current formal path copies each pool row directly from p2p_x[src_pe][src_tok].
    # Keep canonical fanout disabled while K12R reset is fused into this kernel:
    # the direct-pull copy phase does not read recv_meta, so reset can safely run
    # after each CTA finishes its copy work without a global phase-5 barrier.
    _can_same_pe_dup = False
    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_pull_to_pool_init_parallel(
        addr_recv_meta: fx.Int64,
        addr_sort_workspace: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())

        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))
        r_eids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_expert_ids))
        r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))

        c0 = arith.constant(0, type=T.i32())
        cm1 = arith.constant(-1, type=T.i32())
        c_row_bytes = arith.constant(_RECV_META_ROW_BYTES, type=T.i32())
        ws_pool_slot_base = arith.constant(_ws_pool_slot_base, type=T.i32())
        ws_row_src_base = arith.constant(_ws_row_src_base, type=T.i32())
        ws_block_row_start_base = arith.constant(_ws_block_row_start_base, type=T.i32())
        ws_block_valid_base = arith.constant(_ws_block_valid_base, type=T.i32())
        ws_arrival_base = arith.constant(_ws_arrival_base, type=T.i32())
        ws_canon_row_base = arith.constant(_ws_canon_row_base, type=T.i32())
        if arith.cmpi(arith.CmpIPredicate.eq, work_id, c0):
            buffer_store(_lv_unwrap(c0), r_num_valid, c0)

        for work_idx in range(_to_idx(work_id), _to_idx(3 * _ne), _to_idx(work_stride)):
            buffer_store(_lv_unwrap(c0), r_ws, _to_i32(work_idx))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_m_blocks), _to_idx(work_stride)):
            b_i32 = _to_i32(work_idx)
            buffer_store(_lv_unwrap(cm1), r_eids, b_i32)
            buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_block_row_start_base, b_i32))
            buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_block_valid_base, b_i32))
            buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_arrival_base, b_i32))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_padded), _to_idx(work_stride)):
            row_i32 = _to_i32(work_idx)
            buffer_store(_lv_unwrap(cm1), r_ws, arith.addi(ws_pool_slot_base, row_i32))
            buffer_store(_lv_unwrap(cm1), r_ws, arith.addi(ws_row_src_base, row_i32))
            buffer_store(_lv_unwrap(cm1), r_ws, arith.addi(ws_canon_row_base, row_i32))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_recv), _to_idx(work_stride)):
            slot_i32 = _to_i32(work_idx)
            row_off = arith.muli(slot_i32, c_row_bytes)
            r_row = create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_recv_meta), arith.zext_i64(row_off))
            )
            buffer_store(_lv_unwrap(cm1), r_row, arith.constant(_POOL_ROW_WORD, type=T.i32()))

    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_pull_to_pool_count_parallel(
        addr_recv_meta: fx.Int64,
        addr_sort_workspace: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())

        c0 = arith.constant(0, type=T.i32())
        c1 = arith.constant(1, type=T.i32())
        c_rank = arith.constant(_rank, type=T.i32())
        c_epr = arith.constant(_epr, type=T.i32())
        c_ne = arith.constant(_ne, type=T.i32())
        c_row_bytes = arith.constant(_RECV_META_ROW_BYTES, type=T.i32())

        def ws_cell_addr(word_i32):
            byte_off = arith.muli(word_i32, arith.constant(4, type=T.i32()))
            return arith.addi(_lv_unwrap(addr_sort_workspace), arith.zext_i64(byte_off))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_recv), _to_idx(work_stride)):
            slot_i32 = _to_i32(work_idx)
            row_off = arith.muli(slot_i32, c_row_bytes)
            r_row = create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_recv_meta), arith.zext_i64(row_off))
            )
            hdr = buffer_load(r_row, c0, vec_width=1, dtype=T.i32())
            src_tok = buffer_load(r_row, c1, vec_width=1, dtype=T.i32())
            expert = buffer_load(r_row, arith.constant(2, type=T.i32()), vec_width=1, dtype=T.i32())
            wt_bits = buffer_load(r_row, arith.constant(3, type=T.i32()), vec_width=1, dtype=T.i32())
            dst_slot = buffer_load(r_row, arith.constant(5, type=T.i32()), vec_width=1, dtype=T.i32())
            any01 = arith.ori(hdr, src_tok)
            any23 = arith.ori(expert, wt_bits)
            any45 = arith.ori(dst_slot, any23)
            non_empty = arith.cmpi(arith.CmpIPredicate.ne, arith.ori(any01, any45), c0)
            expert_in_range = arith.cmpi(arith.CmpIPredicate.ult, expert, c_ne)
            expert_pe = arith.divui(expert, c_epr)
            expert_local = arith.cmpi(arith.CmpIPredicate.eq, expert_pe, c_rank)
            valid = arith.andi(non_empty, arith.andi(expert_in_range, expert_local))
            if valid:
                atomic_add_global_at(ws_cell_addr(expert), c1)

    @flyc.kernel(known_block_size=[1, 1, 1])
    def ep_pull_to_pool_prefix_parallel(
        addr_sort_workspace: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
    ):
        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))
        r_sorted_ids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_token_ids))
        r_eids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_expert_ids))
        r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        r_sorted_w = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_weights))

        c0 = arith.constant(0, type=T.i32())
        c1 = arith.constant(1, type=T.i32())
        c_ept = arith.constant(_ept, type=T.i32())
        c_tile_m = arith.constant(_tile_m, type=T.i32())
        c24 = arith.constant(24, type=T.i32())
        ws_block_row_start_base = arith.constant(_ws_block_row_start_base, type=T.i32())
        ws_block_valid_base = arith.constant(_ws_block_valid_base, type=T.i32())
        ws_arrival_base = arith.constant(_ws_arrival_base, type=T.i32())
        padding_id = arith.shli(c_ept, c24)

        prefix = c0
        block_prefix = c0
        for e_idx in range(_to_idx(0), _to_idx(_ne), _to_idx(1)):
            e_i32 = _to_i32(e_idx)
            cnt = buffer_load(r_ws, e_i32, vec_width=1, dtype=T.i32())
            blocks = arith.divui(arith.addi(cnt, arith.subi(c_tile_m, c1)), c_tile_m)
            padded = arith.muli(blocks, c_tile_m)
            buffer_store(_lv_unwrap(blocks), r_ws, arith.addi(arith.constant(_ne, type=T.i32()), e_i32))
            buffer_store(_lv_unwrap(prefix), r_ws, arith.addi(arith.constant(2 * _ne, type=T.i32()), e_i32))
            buffer_store(_lv_unwrap(c0), r_ws, e_i32)

            for b_idx in range(_to_idx(0), _to_idx(blocks), _to_idx(1)):
                b_i32 = _to_i32(b_idx)
                block_id = arith.addi(_lv_unwrap(block_prefix), b_i32)
                block_row_start = arith.addi(_lv_unwrap(prefix), arith.muli(b_i32, c_tile_m))
                remaining = arith.subi(_lv_unwrap(cnt), arith.muli(b_i32, c_tile_m))
                full_block = arith.cmpi(arith.CmpIPredicate.uge, remaining, c_tile_m)
                block_valid_rows = arith.select(full_block, c_tile_m, remaining)
                buffer_store(_lv_unwrap(e_i32), r_eids, block_id)
                buffer_store(_lv_unwrap(block_row_start), r_ws, arith.addi(ws_block_row_start_base, block_id))
                buffer_store(_lv_unwrap(block_valid_rows), r_ws, arith.addi(ws_block_valid_base, block_id))
                buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_arrival_base, block_id))

            for p_idx in range(_to_idx(cnt), _to_idx(padded), _to_idx(1)):
                p_i32 = _to_i32(p_idx)
                pad_row = arith.addi(_lv_unwrap(prefix), p_i32)
                buffer_store(_lv_unwrap(padding_id), r_sorted_ids, pad_row)
                buffer_store(_lv_unwrap(arith.constant(0.0, type=T.f32())), r_sorted_w, pad_row)

            prefix = arith.addi(_lv_unwrap(prefix), padded)
            block_prefix = arith.addi(_lv_unwrap(block_prefix), blocks)

        buffer_store(_lv_unwrap(prefix), r_num_valid, c0)

    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_pull_to_pool_assign_parallel(
        addr_recv_meta: fx.Int64,
        addr_sort_workspace: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())

        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))
        r_sorted_ids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_token_ids))
        r_sorted_w = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_weights))

        c0 = arith.constant(0, type=T.i32())
        c1 = arith.constant(1, type=T.i32())
        c_rank = arith.constant(_rank, type=T.i32())
        c_epr = arith.constant(_epr, type=T.i32())
        c_ne = arith.constant(_ne, type=T.i32())
        c_row_bytes = arith.constant(_RECV_META_ROW_BYTES, type=T.i32())
        c24 = arith.constant(24, type=T.i32())
        c28 = arith.constant(28, type=T.i32())
        c_ff = arith.constant(0xFF, type=T.i32())
        mask20 = arith.constant(0xFFFFF, type=T.i32())
        mask24 = arith.constant(0xFFFFFF, type=T.i32())
        ws_pool_slot_base = arith.constant(_ws_pool_slot_base, type=T.i32())
        ws_row_src_base = arith.constant(_ws_row_src_base, type=T.i32())

        def ws_cell_addr(word_i32):
            byte_off = arith.muli(word_i32, arith.constant(4, type=T.i32()))
            return arith.addi(_lv_unwrap(addr_sort_workspace), arith.zext_i64(byte_off))

        for slot_idx in range(_to_idx(work_id), _to_idx(_max_recv), _to_idx(work_stride)):
            slot_i32 = _to_i32(slot_idx)
            row_off = arith.muli(slot_i32, c_row_bytes)
            r_row = create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_recv_meta), arith.zext_i64(row_off))
            )
            hdr = buffer_load(r_row, c0, vec_width=1, dtype=T.i32())
            src_tok = buffer_load(r_row, c1, vec_width=1, dtype=T.i32())
            expert = buffer_load(r_row, arith.constant(2, type=T.i32()), vec_width=1, dtype=T.i32())
            wt_bits = buffer_load(r_row, arith.constant(3, type=T.i32()), vec_width=1, dtype=T.i32())
            dst_slot = buffer_load(r_row, arith.constant(5, type=T.i32()), vec_width=1, dtype=T.i32())
            any01 = arith.ori(hdr, src_tok)
            any23 = arith.ori(expert, wt_bits)
            any45 = arith.ori(dst_slot, any23)
            non_empty = arith.cmpi(arith.CmpIPredicate.ne, arith.ori(any01, any45), c0)
            expert_in_range = arith.cmpi(arith.CmpIPredicate.ult, expert, c_ne)
            expert_pe = arith.divui(expert, c_epr)
            expert_local = arith.cmpi(arith.CmpIPredicate.eq, expert_pe, c_rank)
            valid = arith.andi(non_empty, arith.andi(expert_in_range, expert_local))
            if valid:
                cursor = atomic_add_global_at(ws_cell_addr(expert), c1)
                prefix_e = buffer_load(
                    r_ws,
                    arith.addi(arith.constant(2 * _ne, type=T.i32()), expert),
                    vec_width=1,
                    dtype=T.i32(),
                )
                blocks_e = buffer_load(
                    r_ws,
                    arith.addi(arith.constant(_ne, type=T.i32()), expert),
                    vec_width=1,
                    dtype=T.i32(),
                )
                count_cap = arith.muli(blocks_e, arith.constant(_tile_m, type=T.i32()))
                cursor_ok = arith.cmpi(arith.CmpIPredicate.ult, cursor, count_cap)
                pool_row = arith.addi(prefix_e, cursor)
                pool_ok = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    pool_row,
                    arith.constant(_max_padded, type=T.i32()),
                )
                if arith.andi(cursor_ok, pool_ok):
                    src_pe = arith.andi(hdr, c_ff)
                    kp = arith.andi(
                        arith.shrui(hdr, arith.constant(8, type=T.i32())),
                        arith.constant(0xFF, type=T.i32()),
                    )
                    buffer_store(_lv_unwrap(pool_row), r_row, arith.constant(_POOL_ROW_WORD, type=T.i32()))
                    buffer_store(_lv_unwrap(slot_i32), r_ws, arith.addi(ws_pool_slot_base, pool_row))
                    row_src_packed = arith.ori(
                        arith.ori(
                            src_pe,
                            arith.shli(arith.andi(src_tok, mask20), arith.constant(8, type=T.i32())),
                        ),
                        arith.shli(kp, c28),
                    )
                    buffer_store(_lv_unwrap(row_src_packed), r_ws, arith.addi(ws_row_src_base, pool_row))
                    sorted_id = arith.ori(arith.shli(kp, c24), arith.andi(src_tok, mask24))
                    buffer_store(_lv_unwrap(sorted_id), r_sorted_ids, pool_row)
                    wt = arith.bitcast(T.f32(), wt_bits)
                    buffer_store(_lv_unwrap(wt), r_sorted_w, pool_row)

    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_pull_to_pool_mark_canonical(
        addr_recv_meta: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sort_workspace: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())

        r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))

        c0 = arith.constant(0, type=T.i32())
        c1 = arith.constant(1, type=T.i32())
        cm1 = arith.constant(-1, type=T.i32())
        c_ept = arith.constant(_ept, type=T.i32())
        c_max_recv = arith.constant(_max_recv, type=T.i32())
        ws_pool_slot_base = arith.constant(_ws_pool_slot_base, type=T.i32())
        ws_canon_row_base = arith.constant(_ws_canon_row_base, type=T.i32())
        num_valid_i32 = buffer_load(r_num_valid, c0, vec_width=1, dtype=T.i32())

        def recv_meta_row(slot_i32):
            row_off = arith.muli(slot_i32, arith.constant(_RECV_META_ROW_BYTES, type=T.i32()))
            return create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_recv_meta), arith.zext_i64(row_off))
            )

        for work_idx in range(_to_idx(work_id), _to_idx(num_valid_i32), _to_idx(work_stride)):
            pool_row = _to_i32(work_idx)
            slot_i32 = buffer_load(
                r_ws, arith.addi(ws_pool_slot_base, pool_row), vec_width=1, dtype=T.i32()
            )
            canon = pool_row
            found = arith.cmpi(arith.CmpIPredicate.eq, c0, c1)
            slot_ok = arith.cmpi(arith.CmpIPredicate.ne, slot_i32, cm1)
            if slot_ok:
                r_slot = recv_meta_row(slot_i32)
                dst_slot = buffer_load(
                    r_slot, arith.constant(5, type=T.i32()), vec_width=1, dtype=T.i32()
                )
                base_slot = arith.subi(dst_slot, arith.remui(dst_slot, c_ept))
                for lane in range_constexpr(_ept):
                    cand_slot = arith.addi(base_slot, arith.constant(lane, type=T.i32()))
                    cand_slot_ok = arith.cmpi(arith.CmpIPredicate.ult, cand_slot, c_max_recv)
                    if cand_slot_ok:
                        r_cand = recv_meta_row(cand_slot)
                        cand_pool = buffer_load(
                            r_cand,
                            arith.constant(_POOL_ROW_WORD, type=T.i32()),
                            vec_width=1,
                            dtype=T.i32(),
                        )
                        cand_valid = arith.andi(
                            arith.cmpi(arith.CmpIPredicate.ne, cand_pool, cm1),
                            arith.cmpi(arith.CmpIPredicate.ult, cand_pool, num_valid_i32),
                        )
                        take = arith.andi(
                            arith.cmpi(
                                arith.CmpIPredicate.eq,
                                found,
                                arith.cmpi(arith.CmpIPredicate.eq, c0, c1),
                            ),
                            cand_valid,
                        )
                        canon = arith.select(take, cand_pool, _lv_unwrap(canon))
                        found = arith.ori(_lv_unwrap(found), cand_valid)
            buffer_store(_lv_unwrap(canon), r_ws, arith.addi(ws_canon_row_base, pool_row))

    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_pull_to_pool_copy(
        addr_recv_meta: fx.Int64,
        addr_x_staging: fx.Int64,
        addr_p2p_x: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_scale_x: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sort_workspace: fx.Int64,
        pool_base: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())
        pool_base_i32 = arith.index_cast(T.i32(), pool_base.ir_value())

        r_p2p_x = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_x))
        r_p2p_scale = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_scale))
        r_scale_x = create_buffer_resource_from_addr(_lv_unwrap(addr_scale_x))
        r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))

        c0 = arith.constant(0, type=T.i32())
        c1 = arith.constant(1, type=T.i32())
        cm1 = arith.constant(-1, type=T.i32())
        c8 = arith.constant(8, type=T.i32())
        c2 = arith.constant(2, type=T.i32())
        c4 = arith.constant(4, type=T.i32())
        c16 = arith.constant(16, type=T.i32())
        c32 = arith.constant(32, type=T.i32())
        c_ff = arith.constant(0xFF, type=T.i32())
        c_npes = arith.constant(_npes, type=T.i32())
        c_mtpr = arith.constant(_mtpr, type=T.i32())
        mask20 = arith.constant(0xFFFFF, type=T.i32())
        c31 = arith.constant(31, type=T.i32())
        c_token_bytes = arith.constant(_nbytes, type=T.i32())
        x_chunks = _n_i32 // 4
        scale_cols_i32 = arith.constant(_scale_bytes, type=T.i32())
        stride_o0 = arith.constant(_mx_stride_o0, type=T.i32())
        stride_o1 = arith.constant(_mx_stride_o1, type=T.i32())
        stride_o2 = arith.constant(_mx_stride_o2, type=T.i32())
        stride_o3 = arith.constant(_mx_stride_o3, type=T.i32())
        ws_pool_slot_base = arith.constant(_ws_pool_slot_base, type=T.i32())
        ws_row_src_base = arith.constant(_ws_row_src_base, type=T.i32())
        ws_canon_row_base = arith.constant(_ws_canon_row_base, type=T.i32())
        num_valid_i32 = buffer_load(r_num_valid, c0, vec_width=1, dtype=T.i32())
        valid_scale_m_blocks = arith.divui(arith.addi(num_valid_i32, c31), arith.constant(32, type=T.i32()))
        valid_scale_words = arith.muli(
            valid_scale_m_blocks,
            arith.constant(_mx_gn * 64, type=T.i32()),
        )

        p2p_x_base_pe = []
        p2p_scale_base_pe = []
        for pe in range_constexpr(_npes):
            pe_i32 = arith.constant(pe, type=T.i32())
            p2p_x_base_pe.append(buffer_load(r_p2p_x, pe_i32, vec_width=1, dtype=T.i64()))
            p2p_scale_base_pe.append(
                buffer_load(r_p2p_scale, pe_i32, vec_width=1, dtype=T.i64())
            )
        p2p_x_rsrc_pe = []
        p2p_scale_rsrc_pe = []
        for pe in range_constexpr(_npes):
            p2p_x_rsrc_pe.append(create_buffer_resource_from_addr(p2p_x_base_pe[pe]))
            p2p_scale_rsrc_pe.append(create_buffer_resource_from_addr(p2p_scale_base_pe[pe]))

        def p2p_x_rsrc_for(src_pe_i32):
            rsrc = p2p_x_rsrc_pe[0]
            for pe in range_constexpr(1, _npes):
                use_pe = arith.cmpi(
                    arith.CmpIPredicate.eq, src_pe_i32, arith.constant(pe, type=T.i32())
                )
                rsrc = arith.select(use_pe, p2p_x_rsrc_pe[pe], rsrc)
            return rsrc

        def p2p_scale_rsrc_for(src_pe_i32):
            rsrc = p2p_scale_rsrc_pe[0]
            for pe in range_constexpr(1, _npes):
                use_pe = arith.cmpi(
                    arith.CmpIPredicate.eq, src_pe_i32, arith.constant(pe, type=T.i32())
                )
                rsrc = arith.select(use_pe, p2p_scale_rsrc_pe[pe], rsrc)
            return rsrc

        for work_idx in range(_to_idx(work_id), _to_idx(valid_scale_words), _to_idx(work_stride)):
            work_i32 = _to_i32(work_idx)
            buffer_store(_lv_unwrap(c0), r_scale_x, work_i32)

        def copy_x_pool_row(pool_row, lane_i32):
            row_src = buffer_load(r_ws, arith.addi(ws_row_src_base, pool_row), vec_width=1, dtype=T.i32())
            valid = arith.cmpi(arith.CmpIPredicate.ne, row_src, cm1)
            dst_row = arith.addi(pool_base_i32, pool_row)
            dst_row_off = arith.muli(dst_row, c_token_bytes)
            r_dst = create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_x_staging), arith.zext_i64(dst_row_off))
            )
            if valid:
                canon_row = buffer_load(
                    r_ws, arith.addi(ws_canon_row_base, pool_row), vec_width=1, dtype=T.i32()
                )
                is_canon = arith.cmpi(arith.CmpIPredicate.eq, canon_row, pool_row)
                src_pe = arith.andi(row_src, c_ff)
                src_tok = arith.andi(arith.shrui(row_src, c8), mask20)
                src_pe_ok = arith.cmpi(arith.CmpIPredicate.ult, src_pe, c_npes)
                src_tok_ok = arith.cmpi(arith.CmpIPredicate.ult, src_tok, c_mtpr)
                pool_ok = arith.cmpi(arith.CmpIPredicate.ult, pool_row, num_valid_i32)
                if arith.andi(is_canon, arith.andi(pool_ok, arith.andi(src_pe_ok, src_tok_ok))):
                    src_row_i32_off = arith.muli(src_tok, arith.constant(_n_i32, type=T.i32()))
                    r_src = p2p_x_rsrc_for(src_pe)
                    for chunk_base in range_constexpr(0, x_chunks, 64):
                        chunk_i32 = arith.addi(lane_i32, arith.constant(chunk_base, type=T.i32()))
                        in_chunk = arith.cmpi(
                            arith.CmpIPredicate.ult,
                            chunk_i32,
                            arith.constant(x_chunks, type=T.i32()),
                        )
                        if in_chunk:
                            elem_off = arith.muli(chunk_i32, arith.constant(4, type=T.i32()))
                            src_elem_off = arith.addi(src_row_i32_off, elem_off)
                            v = buffer_load(r_src, src_elem_off, vec_width=4, dtype=T.i32())
                            buffer_store(v, r_dst, elem_off)
            else:
                for chunk_base in range_constexpr(0, x_chunks, 64):
                    chunk_i32 = arith.addi(lane_i32, arith.constant(chunk_base, type=T.i32()))
                    in_chunk = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        chunk_i32,
                        arith.constant(x_chunks, type=T.i32()),
                    )
                    if in_chunk:
                        elem_off = arith.muli(chunk_i32, arith.constant(4, type=T.i32()))
                        z = arith.constant_vector(0, T.vector(4, T.i32()))
                        buffer_store(z, r_dst, elem_off)

        def copy_x_work(work_i32):
            pool_row = arith.divui(work_i32, arith.constant(64, type=T.i32()))
            lane_i32 = arith.remui(work_i32, arith.constant(64, type=T.i32()))
            copy_x_pool_row(pool_row, lane_i32)

        for work_idx in range(
            _to_idx(work_id),
            _to_idx(arith.muli(num_valid_i32, arith.constant(64, type=T.i32()))),
            _to_idx(work_stride),
        ):
            copy_x_work(_to_i32(work_idx))

        for work_idx in range(_to_idx(work_id), _to_idx(valid_scale_words), _to_idx(work_stride)):
            work_i32 = _to_i32(work_idx)
            prog_m = arith.divui(work_i32, stride_o0)
            rem0 = arith.remui(work_i32, stride_o0)
            prog_n = arith.divui(rem0, stride_o1)
            rem1 = arith.remui(rem0, stride_o1)
            ln = arith.divui(rem1, stride_o2)
            lm = arith.remui(rem1, stride_o2)
            pack = c0
            for i in range_constexpr(4):
                row_i = arith.addi(
                    arith.muli(prog_m, arith.constant(32, type=T.i32())),
                    arith.addi(arith.constant((i % 2) * 16, type=T.i32()), lm),
                )
                col_i = arith.addi(
                    arith.muli(prog_n, arith.constant(8, type=T.i32())),
                    arith.addi(arith.constant((i // 2) * 4, type=T.i32()), ln),
                )
                row_ok = arith.cmpi(arith.CmpIPredicate.ult, row_i, num_valid_i32)
                col_ok = arith.cmpi(arith.CmpIPredicate.ult, col_i, scale_cols_i32)
                if arith.andi(row_ok, col_ok):
                    row_src = buffer_load(r_ws, arith.addi(ws_row_src_base, row_i), vec_width=1, dtype=T.i32())
                    if arith.cmpi(arith.CmpIPredicate.ne, row_src, cm1):
                        src_pe = arith.andi(row_src, c_ff)
                        src_tok = arith.andi(arith.shrui(row_src, c8), mask20)
                        src_pe_ok = arith.cmpi(arith.CmpIPredicate.ult, src_pe, c_npes)
                        src_tok_ok = arith.cmpi(arith.CmpIPredicate.ult, src_tok, c_mtpr)
                        if arith.andi(src_pe_ok, src_tok_ok):
                            r_src_scale = p2p_scale_rsrc_for(src_pe)
                            src_off = arith.addi(arith.muli(src_tok, scale_cols_i32), col_i)
                            b8 = buffer_load(r_src_scale, src_off, vec_width=1, dtype=T.i8())
                            pack = arith.ori(
                                _lv_unwrap(pack),
                                arith.shli(arith.extui(T.i32(), b8), arith.constant(i * 8, type=T.i32())),
                            )
            buffer_store(_lv_unwrap(pack), r_scale_x, work_i32)

    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_pull_to_pool_copy_duplicates(
        addr_recv_meta: fx.Int64,
        addr_x_staging: fx.Int64,
        addr_scale_x: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sort_workspace: fx.Int64,
        pool_base: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())
        pool_base_i32 = arith.index_cast(T.i32(), pool_base.ir_value())

        r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))

        c0 = arith.constant(0, type=T.i32())
        c4 = arith.constant(4, type=T.i32())
        c_token_bytes = arith.constant(_nbytes, type=T.i32())
        ws_canon_row_base = arith.constant(_ws_canon_row_base, type=T.i32())
        num_valid_i32 = buffer_load(r_num_valid, c0, vec_width=1, dtype=T.i32())
        x_chunks = _n_i32 // 4

        for work_idx in range(
            _to_idx(work_id),
            _to_idx(arith.muli(num_valid_i32, arith.constant(64, type=T.i32()))),
            _to_idx(work_stride),
        ):
            work_i32 = _to_i32(work_idx)
            pool_row = arith.divui(work_i32, arith.constant(64, type=T.i32()))
            lane_i32 = arith.remui(work_i32, arith.constant(64, type=T.i32()))
            canon_row = buffer_load(
                r_ws, arith.addi(ws_canon_row_base, pool_row), vec_width=1, dtype=T.i32()
            )
            is_dup = arith.cmpi(arith.CmpIPredicate.ne, canon_row, pool_row)
            if is_dup:
                src_row = arith.addi(pool_base_i32, canon_row)
                dst_row = arith.addi(pool_base_i32, pool_row)
                src_row_off = arith.muli(src_row, c_token_bytes)
                dst_row_off = arith.muli(dst_row, c_token_bytes)
                r_src = create_buffer_resource_from_addr(
                    arith.addi(_lv_unwrap(addr_x_staging), arith.zext_i64(src_row_off))
                )
                r_dst = create_buffer_resource_from_addr(
                    arith.addi(_lv_unwrap(addr_x_staging), arith.zext_i64(dst_row_off))
                )
                for chunk_base in range_constexpr(0, x_chunks, 64):
                    chunk_i32 = arith.addi(lane_i32, arith.constant(chunk_base, type=T.i32()))
                    in_chunk = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        chunk_i32,
                        arith.constant(x_chunks, type=T.i32()),
                    )
                    if in_chunk:
                        elem_off = arith.muli(chunk_i32, c4)
                        v = buffer_load(r_src, elem_off, vec_width=4, dtype=T.i32())
                        buffer_store(v, r_dst, elem_off)


    @flyc.kernel(known_block_size=[_threads_per_block, 1, 1])
    def ep_stage1_metadata_pull_to_pool_a8w4(
        addr_recv_meta: fx.Int64,
        addr_x_staging: fx.Int64,
        addr_p2p_x: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_scale_x: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sort_workspace: fx.Int64,
        pool_base: fx.Int32,
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_tok_off: fx.Int64,
        addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64,
        addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_tis: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        addr_p2p_recv_meta: fx.Int64,
        addr_inp_tok: fx.Int64,
        addr_pull_row_i64: fx.Int64,
        addr_p2p_pull_row: fx.Int64,
        cur_tok: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        linear = bid * _threads_per_block + tid
        stride = fx.grid_dim.x * _threads_per_block
        work_id = arith.index_cast(T.i32(), linear.ir_value())
        work_stride = arith.index_cast(T.i32(), stride.ir_value())
        tid_i32 = arith.index_cast(T.i32(), tid.ir_value())
        bid_i32 = arith.index_cast(T.i32(), bid.ir_value())
        pool_base_i32 = arith.index_cast(T.i32(), pool_base.ir_value())

        r_ws = create_buffer_resource_from_addr(_lv_unwrap(addr_sort_workspace))
        r_sorted_ids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_token_ids))
        r_eids = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_expert_ids))
        r_num_valid = create_buffer_resource_from_addr(_lv_unwrap(addr_num_valid_ids))
        r_sorted_w = create_buffer_resource_from_addr(_lv_unwrap(addr_sorted_weights))
        r_p2p_x = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_x))
        r_p2p_scale = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_scale))
        r_scale_x = create_buffer_resource_from_addr(_lv_unwrap(addr_scale_x))

        reset_tok_map_words = arith.constant(_tok_map_words, type=T.i32())
        reset_tis_words = arith.constant(_tis_words, type=T.i32())
        reset_sentinel = arith.constant(_reset_sentinel, type=T.i32())

        c0 = arith.constant(0, type=T.i32())
        c1 = arith.constant(1, type=T.i32())
        c2 = arith.constant(2, type=T.i32())
        c4 = arith.constant(4, type=T.i32())
        c8 = arith.constant(8, type=T.i32())
        c16 = arith.constant(16, type=T.i32())
        c24 = arith.constant(24, type=T.i32())
        c28 = arith.constant(28, type=T.i32())
        c31 = arith.constant(31, type=T.i32())
        c32 = arith.constant(32, type=T.i32())
        cm1 = arith.constant(-1, type=T.i32())
        c_ff = arith.constant(0xFF, type=T.i32())
        mask20 = arith.constant(0xFFFFF, type=T.i32())
        mask24 = arith.constant(0xFFFFFF, type=T.i32())
        c_rank = arith.constant(_rank, type=T.i32())
        c_epr = arith.constant(_epr, type=T.i32())
        c_ept = arith.constant(_ept, type=T.i32())
        c_ne = arith.constant(_ne, type=T.i32())
        c_npes = arith.constant(_npes, type=T.i32())
        c_mtpr = arith.constant(_mtpr, type=T.i32())
        c_tile_m = arith.constant(_tile_m, type=T.i32())
        c_row_bytes = arith.constant(_RECV_META_ROW_BYTES, type=T.i32())
        c_token_bytes = arith.constant(_nbytes, type=T.i32())
        scale_cols_i32 = arith.constant(_scale_bytes, type=T.i32())
        ws_pool_slot_base = arith.constant(_ws_pool_slot_base, type=T.i32())
        ws_row_src_base = arith.constant(_ws_row_src_base, type=T.i32())
        ws_block_row_start_base = arith.constant(_ws_block_row_start_base, type=T.i32())
        ws_block_valid_base = arith.constant(_ws_block_valid_base, type=T.i32())
        ws_arrival_base = arith.constant(_ws_arrival_base, type=T.i32())
        ws_canon_row_base = arith.constant(_ws_canon_row_base, type=T.i32())
        barrier_count_word = arith.constant(_ws_k2_barrier_count, type=T.i32())
        barrier_phase_word = arith.constant(_ws_k2_barrier_phase, type=T.i32())
        dup_flag_word = arith.constant(_ws_k2_dup_flag, type=T.i32())
        padding_id = arith.shli(c_ept, c24)
        x_chunks = _n_i32 // 4
        stride_o0 = arith.constant(_mx_stride_o0, type=T.i32())
        stride_o1 = arith.constant(_mx_stride_o1, type=T.i32())
        stride_o2 = arith.constant(_mx_stride_o2, type=T.i32())
        stride_o3 = arith.constant(_mx_stride_o3, type=T.i32())

        is_t0 = arith.cmpi(CmpIPredicate.eq, tid_i32, c0)
        is_b0_t0 = arith.andi(is_t0, arith.cmpi(CmpIPredicate.eq, bid_i32, c0))

        def ws_cell_addr(word_i32):
            return _i64_cell_addr(addr_sort_workspace, word_i32)

        def recv_meta_row(slot_i32):
            row_off = arith.muli(slot_i32, c_row_bytes)
            return create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_recv_meta), arith.zext_i64(row_off))
            )

        def wait_for_phase(target_phase):
            _if_wait = scf.IfOp(is_t0)
            with ir.InsertionPoint(_if_wait.then_block):
                init_cur = _load_i32_global_acquire_cell(addr_sort_workspace, barrier_phase_word)
                w = scf.WhileOp([T.i32()], [init_cur])
                before = ir.Block.create_at_start(w.before, [T.i32()])
                after = ir.Block.create_at_start(w.after, [T.i32()])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.cmpi(CmpIPredicate.ne, cur, target_phase)
                    scf.ConditionOp(need_wait, [cur])
                with ir.InsertionPoint(after):
                    nxt = _load_i32_global_acquire_cell(addr_sort_workspace, barrier_phase_word)
                    scf.YieldOp([nxt])
                scf.YieldOp([])
            gpu.barrier()

        def arrive_and_wait(next_phase):
            gpu.barrier()
            rocdl.s_waitcnt(0)
            _if_arrive = scf.IfOp(is_t0)
            with ir.InsertionPoint(_if_arrive.then_block):
                prev = _atomic_add_global_acq_rel(
                    ws_cell_addr(barrier_count_word),
                    c1,
                )
                is_last = arith.cmpi(
                    CmpIPredicate.eq,
                    prev,
                    arith.constant(block_num - 1, type=T.i32()),
                )
                _if_last = scf.IfOp(is_last)
                with ir.InsertionPoint(_if_last.then_block):
                    _store_i32_global_release_cell(addr_sort_workspace, barrier_count_word, c0)
                    _store_i32_global_release_cell(addr_sort_workspace, barrier_phase_word, next_phase)
                    scf.YieldOp([])
                scf.YieldOp([])
            gpu.barrier()
            wait_for_phase(next_phase)

        if const_expr(_fuse_metadata):
            lane = arith.andi(tid_i32, arith.constant(63, type=T.i32()))
            warp = arith.shrui(tid_i32, arith.constant(6, type=T.i32()))
            gw_id = arith.addi(arith.muli(bid_i32, arith.constant(warp_num_per_block, type=T.i32())), warp)
            gw_num = arith.constant(block_num * warp_num_per_block, type=T.i32())
            limit = arith.muli(arith.index_cast(T.i32(), cur_tok.ir_value()), c_ept)
            grid_staging_workers = block_num * _threads_per_block
            staging_gid_i32 = work_id
            grid_staging_i32 = arith.constant(grid_staging_workers, type=T.i32())

            r_idx = create_buffer_resource_from_addr(_lv_unwrap(addr_idx))
            r_wts = create_buffer_resource_from_addr(_lv_unwrap(addr_wts))
            r_tok_map = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_map))
            r_tok_off = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_off))
            r_dest_ctr = create_buffer_resource_from_addr(_lv_unwrap(addr_dest_ctr))
            r_disp_bar = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_bar))
            r_p2p_tok_off = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tok_off))
            r_p2p_recv_num = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_recv_num))
            r_p2p_recv_meta = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_recv_meta))
            r_p2p_pull_row = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_pull_row))
            r_p2p_tis = create_buffer_resource_from_addr(_lv_unwrap(addr_p2p_tis))

            c3 = arith.constant(3, type=T.i32())
            c5 = arith.constant(5, type=T.i32())
            c64 = arith.constant(64, type=T.i32())
            nbytes_i32 = arith.constant(_nbytes, type=T.i32())
            rank_i32 = arith.constant(_rank, type=T.i32())
            vec4_i32_ty = _FlyT.vec(4, _FlyT.i32)

            def publish_local_token_vec4(src_tok_i32):
                row_off_i64 = arith.muli(arith.zext_i64(src_tok_i32), arith.zext_i64(nbytes_i32))
                in_row = arith.addi(_lv_unwrap(addr_inp_tok), row_off_i64)
                out_row = arith.addi(_lv_unwrap(addr_x_staging), row_off_i64)
                r_in = create_buffer_resource_from_addr(in_row)
                r_out = create_buffer_resource_from_addr(out_row)
                lane4 = arith.muli(lane, c4)

                if const_expr(_n_i32 >= 512 and _safe_staging_end > 0):
                    safe_end = arith.constant(_safe_staging_end, type=T.i32())
                    for ec4 in range(_to_idx(lane4), _to_idx(safe_end), 512):
                        ec4 = _to_i32(ec4)
                        vec4_0 = buffer_load(r_in, ec4, vec_width=4, dtype=T.i32())
                        vec4_1 = buffer_load(r_in, ec4 + 256, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_0, r_out, ec4)
                        buffer_store(vec4_1, r_out, ec4 + 256)
                if const_expr(_safe_staging_end < _n_i32):
                    tail_end = arith.constant(_n_i32, type=T.i32())
                    for ec4 in range(
                        _to_idx(arith.addi(lane4, arith.constant(_safe_staging_end, type=T.i32()))),
                        _to_idx(tail_end),
                        256,
                    ):
                        ec4 = _to_i32(ec4)
                        vec4_0 = buffer_load(r_in, ec4, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_0, r_out, ec4)
                elif const_expr(_n_i32 < 512):
                    copy_end = arith.constant(_n_i32, type=T.i32())
                    for ec4 in range(_to_idx(lane4), _to_idx(copy_end), 256):
                        ec4 = _to_i32(ec4)
                        vec4_0 = buffer_load(r_in, ec4, vec_width=4, dtype=T.i32())
                        buffer_store(vec4_0, r_out, ec4)

            def staging_vec4_worker():
                w = scf.WhileOp([T.i32()], [staging_gid_i32])
                wb = ir.Block.create_at_start(w.before, [T.i32()])
                wa = ir.Block.create_at_start(w.after, [T.i32()])
                with ir.InsertionPoint(wb):
                    tok_w = wb.arguments[0]
                    cont = arith.cmpi(CmpIPredicate.ult, tok_w, arith.index_cast(T.i32(), cur_tok.ir_value()))
                    scf.ConditionOp(_lv_unwrap(cont), [tok_w])
                with ir.InsertionPoint(wa):
                    tok_w = wa.arguments[0]
                    in_rng = arith.cmpi(CmpIPredicate.ult, tok_w, arith.index_cast(T.i32(), cur_tok.ir_value()))
                    if_in = scf.IfOp(_lv_unwrap(in_rng))
                    with ir.InsertionPoint(if_in.then_block):
                        publish_local_token_vec4(tok_w)
                        scf.YieldOp([])
                    next_tok = arith.addi(tok_w, grid_staging_i32)
                    scf.YieldOp([next_tok])

            for i in range(_to_idx(gw_id), _to_idx(limit), _to_idx(gw_num)):
                i = _to_i32(i)
                src_tok = arith.divui(i, c_ept)
                j = arith.remui(i, c_ept)
                dest_exp = buffer_load(r_idx, i, vec_width=1, dtype=T.i32())
                safe_lane = arith.select(
                    arith.cmpi(CmpIPredicate.ult, lane, j),
                    lane,
                    c0,
                )
                lane_exp = buffer_load(r_idx, arith.addi(arith.muli(src_tok, c_ept), safe_lane), vec_width=1, dtype=T.i32())
                dest_pe = arith.divui(dest_exp, c_epr)
                lane_pe = arith.divui(lane_exp, c_epr)
                dup_per_lane = arith.select(
                    arith.cmpi(CmpIPredicate.eq, lane_pe, dest_pe),
                    arith.select(arith.cmpi(CmpIPredicate.ult, lane, j), lane, c64),
                    c64,
                )
                dup_ballot = ballot_i64(arith.cmpi(CmpIPredicate.ult, dup_per_lane, c64))
                is_dup = arith.cmpi(CmpIPredicate.ne, dup_ballot, arith.constant(0, type=T.i64()))

                if const_expr(block_num == 1):
                    if arith.cmpi(CmpIPredicate.eq, j, c0):
                        if arith.cmpi(CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                            publish_local_token_vec4(src_tok)

                dest_tok_all = c0
                src_enc = arith.addi(arith.muli(rank_i32, c_mtpr), src_tok)
                sentinel_val = arith.constant(_npes * _max_recv_payload, type=T.i32())
                dtm_val = arith.select(
                    is_dup,
                    sentinel_val,
                    arith.addi(arith.muli(dest_pe, arith.constant(_max_recv_payload, type=T.i32())), dest_tok_all),
                )
                if arith.cmpi(CmpIPredicate.eq, lane, c0):
                    buffer_store(_lv_unwrap(dtm_val), r_tok_map, i)

                if arith.cmpi(CmpIPredicate.eq, dup_ballot, arith.constant(0, type=T.i64())):
                    zero_i64 = arith.constant(0, type=T.i64())
                    lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0)
                    tok_pe = arith.select(lane0, buffer_load(r_p2p_tok_off, dest_pe, vec_width=1, dtype=T.i64()), zero_i64)
                    recv_pe = arith.select(lane0, buffer_load(r_p2p_recv_meta, dest_pe, vec_width=1, dtype=T.i64()), zero_i64)
                    pull_pe = arith.select(lane0, buffer_load(r_p2p_pull_row, dest_pe, vec_width=1, dtype=T.i64()), zero_i64)
                    tis_pe = arith.select(lane0, buffer_load(r_p2p_tis, dest_pe, vec_width=1, dtype=T.i64()), zero_i64)
                    tok_off_pe = rocdl.readfirstlane(T.i64(), tok_pe)
                    remote_base_pe = rocdl.readfirstlane(T.i64(), recv_pe)
                    pull_remote_pe = rocdl.readfirstlane(T.i64(), pull_pe)
                    tis_remote_pe = rocdl.readfirstlane(T.i64(), tis_pe)

                    if_lane0 = scf.IfOp(_lv_unwrap(lane0), [T.i32()], has_else=True)
                    with ir.InsertionPoint(if_lane0.then_block):
                        old_tok = atomic_add_global_at(tok_off_pe, c1)
                        scf.YieldOp([_lv_unwrap(old_tok)])
                    with ir.InsertionPoint(if_lane0.else_block):
                        scf.YieldOp([_lv_unwrap(c0)])
                    dest_tok_all = readlane(if_lane0.result, 0)

                    if arith.cmpi(CmpIPredicate.eq, lane, c0):
                        r_tis_remote = create_buffer_resource_from_addr(tis_remote_pe)
                        buffer_store(_lv_unwrap(src_enc), r_tis_remote, dest_tok_all)
                        ctr_addr = arith.addi(_lv_unwrap(addr_dest_ctr), arith.muli(arith.zext_i64(dest_pe), arith.constant(4, type=T.i64())))
                        atomic_add_global_at(ctr_addr, c1)

                    if arith.cmpi(CmpIPredicate.ult, lane, c_ept):
                        wt_src = arith.addi(arith.muli(src_tok, c_ept), lane)
                        wt_val = buffer_load(r_wts, wt_src, vec_width=1, dtype=T.f32())
                        ix_val = buffer_load(r_idx, wt_src, vec_width=1, dtype=T.i32())
                        dst_slot = arith.addi(arith.muli(dest_tok_all, c_ept), lane)
                        row_bytes = arith.muli(dst_slot, c32)
                        row_addr = arith.addi(remote_base_pe, arith.zext_i64(row_bytes))
                        r_row = create_buffer_resource_from_addr(row_addr)
                        hdr = arith.addi(
                            arith.andi(rank_i32, c_ff),
                            arith.shli(arith.andi(lane, c_ff), c8),
                        )
                        z = arith.constant(0, type=T.i32())
                        wt_i32 = arith.bitcast(T.i32(), wt_val)
                        row_lo = vector.from_elements(vec4_i32_ty, [hdr, src_tok, ix_val, wt_i32])
                        row_hi = vector.from_elements(vec4_i32_ty, [src_enc, dst_slot, z, z])
                        buffer_store(row_lo, r_row, c0)
                        buffer_store(row_hi, r_row, c4)
                        packed_pr = arith.addi(rank_i32, arith.shli(src_tok, c8))
                        r_pull_remote = create_buffer_resource_from_addr(pull_remote_pe)
                        kp_pr = arith.andi(lane, c_ff)
                        lut_idx = arith.addi(arith.muli(src_tok, c_ept), kp_pr)
                        buffer_store(_lv_unwrap(packed_pr), r_pull_remote, lut_idx)

            fx.barrier()

            if arith.cmpi(CmpIPredicate.eq, tid_i32, c0):
                atomic_add_global_at(addr_disp_bar, c1)

            rtn_local_off = arith.muli(arith.zext_i64(rank_i32), arith.constant(4, type=T.i64()))
            is_hs_gw = arith.cmpi(CmpIPredicate.eq, gw_id, c0)

            def handshake_phase_23():
                if arith.cmpi(CmpIPredicate.eq, lane, c0):
                    mori_shmem.int32_wait_until_equals(addr_disp_bar, block_num)
                    buffer_store(_lv_unwrap(c0), r_disp_bar, c0)

                for dest_pe_h in range(_to_idx(lane), _to_idx(_npes), 64):
                    dest_pe_h = _to_i32(dest_pe_h)
                    nsig = arith.addi(buffer_load(r_dest_ctr, dest_pe_h, vec_width=1, dtype=T.i32()), c1)
                    rtn_remote = arith.addi(buffer_load(r_p2p_recv_num, dest_pe_h, vec_width=1, dtype=T.i64()), rtn_local_off)
                    mori_shmem.int32_wait_until_equals(rtn_remote, 0)
                    store_i32_system(rtn_remote, c0, nsig)

                for src_pe_h in range(_to_idx(lane), _to_idx(_npes), 64):
                    src_pe_h = _to_i32(src_pe_h)
                    rtn_src = arith.addi(_lv_unwrap(addr_recv_num), arith.muli(arith.zext_i64(src_pe_h), arith.constant(4, type=T.i64())))
                    sig_val = mori_shmem.int32_wait_until_greater_than(rtn_src, 0)
                    recv_cnt = arith.subi(sig_val, c1)
                    store_i32_system(rtn_src, c0, c0)
                    atomic_add_global_at(addr_total_rv, recv_cnt)
                    buffer_store(_lv_unwrap(c0), r_dest_ctr, src_pe_h)

            if const_expr(block_num > 1):
                if_hs = scf.IfOp(_lv_unwrap(is_hs_gw), has_else=True)
                with ir.InsertionPoint(if_hs.then_block):
                    handshake_phase_23()
                    scf.YieldOp([])
                with ir.InsertionPoint(if_hs.else_block):
                    staging_vec4_worker()
                    scf.YieldOp([])
            else:
                fx.barrier()
                handshake_phase_23()

            fx.barrier()

            if arith.cmpi(CmpIPredicate.eq, gw_id, c0):
                if arith.cmpi(CmpIPredicate.eq, lane, c0):
                    buffer_store(_lv_unwrap(c0), r_tok_off, c0)

        _if_init_barrier = scf.IfOp(is_b0_t0)
        with ir.InsertionPoint(_if_init_barrier.then_block):
            _store_i32_global_release_cell(addr_sort_workspace, barrier_count_word, c0)
            _store_i32_global_release_cell(addr_sort_workspace, barrier_phase_word, c0)
            scf.YieldOp([])
        wait_for_phase(c0)

        # Phase 0: init_parallel.
        if arith.cmpi(CmpIPredicate.eq, work_id, c0):
            buffer_store(_lv_unwrap(c0), r_num_valid, c0)
            if const_expr(_can_same_pe_dup):
                buffer_store(_lv_unwrap(c0), r_ws, dup_flag_word)

        for work_idx in range(_to_idx(work_id), _to_idx(3 * _ne), _to_idx(work_stride)):
            buffer_store(_lv_unwrap(c0), r_ws, _to_i32(work_idx))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_m_blocks), _to_idx(work_stride)):
            b_i32 = _to_i32(work_idx)
            buffer_store(_lv_unwrap(cm1), r_eids, b_i32)
            buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_block_row_start_base, b_i32))
            buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_block_valid_base, b_i32))
            buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_arrival_base, b_i32))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_padded), _to_idx(work_stride)):
            row_i32 = _to_i32(work_idx)
            buffer_store(_lv_unwrap(cm1), r_ws, arith.addi(ws_pool_slot_base, row_i32))
            buffer_store(_lv_unwrap(cm1), r_ws, arith.addi(ws_row_src_base, row_i32))
            buffer_store(_lv_unwrap(cm1), r_ws, arith.addi(ws_canon_row_base, row_i32))

        for work_idx in range(_to_idx(work_id), _to_idx(_max_recv), _to_idx(work_stride)):
            slot_i32 = _to_i32(work_idx)
            r_row = recv_meta_row(slot_i32)
            buffer_store(_lv_unwrap(cm1), r_row, arith.constant(_POOL_ROW_WORD, type=T.i32()))

        arrive_and_wait(arith.constant(1, type=T.i32()))

        # recv_meta is written densely in topk-blocks (one block per received
        # payload), so the populated extent is exactly [0, total_recv*topk).
        # total_recv is the handshake aggregate (== max payload index + 1), already
        # visible after the phase-1 barrier. Bounding the count/assign scans AND the
        # phase-5 recv_meta reset to this skips the empty tail when max_tok_per_rank
        # exceeds the actually received payloads (e.g. decode serving compiled for a
        # large capacity). The tail beyond scan_limit is never written this iter and
        # never read (every scan is bounded), so leaving it uncleared is safe; each
        # iter clears exactly the region its routing dirtied. When
        # max_tok_per_rank == tokens (right-sized) the bound equals _max_recv, so the
        # dense case never regresses.
        r_total_rv_rd = create_buffer_resource_from_addr(_lv_unwrap(addr_total_rv))
        total_rv_v = buffer_load(r_total_rv_rd, c0, vec_width=1, dtype=T.i32())
        scan_limit_raw = arith.muli(total_rv_v, c_ept)
        c_max_recv_v = arith.constant(_max_recv, type=T.i32())
        scan_limit = arith.select(
            arith.cmpi(CmpIPredicate.ult, scan_limit_raw, c_max_recv_v),
            scan_limit_raw,
            c_max_recv_v,
        )
        scan_limit_words = arith.muli(scan_limit, c8)

        # Phase 1: count_parallel.
        for work_idx in range(_to_idx(work_id), _to_idx(scan_limit), _to_idx(work_stride)):
            slot_i32 = _to_i32(work_idx)
            r_row = recv_meta_row(slot_i32)
            hdr = buffer_load(r_row, c0, vec_width=1, dtype=T.i32())
            src_tok = buffer_load(r_row, c1, vec_width=1, dtype=T.i32())
            expert = buffer_load(r_row, c2, vec_width=1, dtype=T.i32())
            wt_bits = buffer_load(r_row, arith.constant(3, type=T.i32()), vec_width=1, dtype=T.i32())
            dst_slot = buffer_load(r_row, arith.constant(5, type=T.i32()), vec_width=1, dtype=T.i32())
            any01 = arith.ori(hdr, src_tok)
            any23 = arith.ori(expert, wt_bits)
            any45 = arith.ori(dst_slot, any23)
            non_empty = arith.cmpi(CmpIPredicate.ne, arith.ori(any01, any45), c0)
            expert_in_range = arith.cmpi(CmpIPredicate.ult, expert, c_ne)
            expert_pe = arith.divui(expert, c_epr)
            expert_local = arith.cmpi(CmpIPredicate.eq, expert_pe, c_rank)
            valid = arith.andi(non_empty, arith.andi(expert_in_range, expert_local))
            if valid:
                atomic_add_global_at(ws_cell_addr(expert), c1)

        arrive_and_wait(arith.constant(2, type=T.i32()))

        # Phase 2: prefix_parallel, executed by one device thread.
        _if_prefix = scf.IfOp(is_b0_t0)
        with ir.InsertionPoint(_if_prefix.then_block):
            expert_loop = scf.ForOp(
                c0,
                arith.constant(_ne, type=T.i32()),
                c1,
                [c0, c0],
            )
            with ir.InsertionPoint(expert_loop.body):
                e_i32 = expert_loop.induction_variable
                prefix = expert_loop.inner_iter_args[0]
                block_prefix = expert_loop.inner_iter_args[1]
                cnt = buffer_load(r_ws, e_i32, vec_width=1, dtype=T.i32())
                blocks = arith.divui(arith.addi(cnt, arith.subi(c_tile_m, c1)), c_tile_m)
                padded = arith.muli(blocks, c_tile_m)
                buffer_store(_lv_unwrap(blocks), r_ws, arith.addi(arith.constant(_ne, type=T.i32()), e_i32))
                buffer_store(_lv_unwrap(prefix), r_ws, arith.addi(arith.constant(2 * _ne, type=T.i32()), e_i32))
                buffer_store(_lv_unwrap(c0), r_ws, e_i32)

                block_loop = scf.ForOp(c0, blocks, c1)
                with ir.InsertionPoint(block_loop.body):
                    b_i32 = block_loop.induction_variable
                    block_id = arith.addi(block_prefix, b_i32)
                    block_row_start = arith.addi(prefix, arith.muli(b_i32, c_tile_m))
                    remaining = arith.subi(cnt, arith.muli(b_i32, c_tile_m))
                    full_block = arith.cmpi(CmpIPredicate.uge, remaining, c_tile_m)
                    block_valid_rows = arith.select(full_block, c_tile_m, remaining)
                    buffer_store(_lv_unwrap(e_i32), r_eids, block_id)
                    buffer_store(_lv_unwrap(block_row_start), r_ws, arith.addi(ws_block_row_start_base, block_id))
                    buffer_store(_lv_unwrap(block_valid_rows), r_ws, arith.addi(ws_block_valid_base, block_id))
                    buffer_store(_lv_unwrap(c0), r_ws, arith.addi(ws_arrival_base, block_id))
                    scf.YieldOp([])

                pad_loop = scf.ForOp(cnt, padded, c1)
                with ir.InsertionPoint(pad_loop.body):
                    p_i32 = pad_loop.induction_variable
                    pad_row = arith.addi(prefix, p_i32)
                    buffer_store(_lv_unwrap(padding_id), r_sorted_ids, pad_row)
                    buffer_store(_lv_unwrap(arith.constant(0.0, type=T.f32())), r_sorted_w, pad_row)
                    scf.YieldOp([])

                scf.YieldOp([
                    arith.addi(prefix, padded),
                    arith.addi(block_prefix, blocks),
                ])

            buffer_store(_lv_unwrap(expert_loop.results[0]), r_num_valid, c0)
            scf.YieldOp([])

        arrive_and_wait(arith.constant(3, type=T.i32()))

        # Phase 3: assign_parallel.
        for slot_idx in range(_to_idx(work_id), _to_idx(scan_limit), _to_idx(work_stride)):
            slot_i32 = _to_i32(slot_idx)
            r_row = recv_meta_row(slot_i32)
            hdr = buffer_load(r_row, c0, vec_width=1, dtype=T.i32())
            src_tok = buffer_load(r_row, c1, vec_width=1, dtype=T.i32())
            expert = buffer_load(r_row, c2, vec_width=1, dtype=T.i32())
            wt_bits = buffer_load(r_row, arith.constant(3, type=T.i32()), vec_width=1, dtype=T.i32())
            dst_slot = buffer_load(r_row, arith.constant(5, type=T.i32()), vec_width=1, dtype=T.i32())
            any01 = arith.ori(hdr, src_tok)
            any23 = arith.ori(expert, wt_bits)
            any45 = arith.ori(dst_slot, any23)
            non_empty = arith.cmpi(CmpIPredicate.ne, arith.ori(any01, any45), c0)
            expert_in_range = arith.cmpi(CmpIPredicate.ult, expert, c_ne)
            expert_pe = arith.divui(expert, c_epr)
            expert_local = arith.cmpi(CmpIPredicate.eq, expert_pe, c_rank)
            valid = arith.andi(non_empty, arith.andi(expert_in_range, expert_local))
            if const_expr(_can_same_pe_dup):
                is_group_base = arith.cmpi(CmpIPredicate.eq, arith.remui(slot_i32, c_ept), c0)
                if is_group_base:
                    local_count = c0
                    for dup_lane in range_constexpr(_ept):
                        cand_slot = arith.addi(slot_i32, arith.constant(dup_lane, type=T.i32()))
                        cand_slot_ok = arith.cmpi(
                            CmpIPredicate.ult,
                            cand_slot,
                            arith.constant(_max_recv, type=T.i32()),
                        )
                        if cand_slot_ok:
                            r_cand = recv_meta_row(cand_slot)
                            cand_hdr = buffer_load(r_cand, c0, vec_width=1, dtype=T.i32())
                            cand_src_tok = buffer_load(r_cand, c1, vec_width=1, dtype=T.i32())
                            cand_expert = buffer_load(r_cand, c2, vec_width=1, dtype=T.i32())
                            cand_wt_bits = buffer_load(
                                r_cand,
                                arith.constant(3, type=T.i32()),
                                vec_width=1,
                                dtype=T.i32(),
                            )
                            cand_dst_slot = buffer_load(
                                r_cand,
                                arith.constant(5, type=T.i32()),
                                vec_width=1,
                                dtype=T.i32(),
                            )
                            cand_any01 = arith.ori(cand_hdr, cand_src_tok)
                            cand_any23 = arith.ori(cand_expert, cand_wt_bits)
                            cand_any45 = arith.ori(cand_dst_slot, cand_any23)
                            cand_non_empty = arith.cmpi(
                                CmpIPredicate.ne,
                                arith.ori(cand_any01, cand_any45),
                                c0,
                            )
                            cand_in_range = arith.cmpi(CmpIPredicate.ult, cand_expert, c_ne)
                            cand_pe = arith.divui(cand_expert, c_epr)
                            cand_local = arith.cmpi(CmpIPredicate.eq, cand_pe, c_rank)
                            cand_valid = arith.andi(
                                cand_non_empty,
                                arith.andi(cand_in_range, cand_local),
                            )
                            local_count = arith.addi(
                                _lv_unwrap(local_count),
                                arith.select(cand_valid, c1, c0),
                            )
                    if arith.cmpi(CmpIPredicate.uge, local_count, arith.constant(2, type=T.i32())):
                        buffer_store(_lv_unwrap(c1), r_ws, dup_flag_word)
            if valid:
                cursor = atomic_add_global_at(ws_cell_addr(expert), c1)
                prefix_e = buffer_load(
                    r_ws,
                    arith.addi(arith.constant(2 * _ne, type=T.i32()), expert),
                    vec_width=1,
                    dtype=T.i32(),
                )
                blocks_e = buffer_load(
                    r_ws,
                    arith.addi(arith.constant(_ne, type=T.i32()), expert),
                    vec_width=1,
                    dtype=T.i32(),
                )
                count_cap = arith.muli(blocks_e, c_tile_m)
                cursor_ok = arith.cmpi(CmpIPredicate.ult, cursor, count_cap)
                pool_row = arith.addi(prefix_e, cursor)
                pool_ok = arith.cmpi(CmpIPredicate.ult, pool_row, arith.constant(_max_padded, type=T.i32()))
                if arith.andi(cursor_ok, pool_ok):
                    src_pe = arith.andi(hdr, c_ff)
                    kp = arith.andi(arith.shrui(hdr, c8), c_ff)
                    buffer_store(_lv_unwrap(pool_row), r_row, arith.constant(_POOL_ROW_WORD, type=T.i32()))
                    buffer_store(_lv_unwrap(slot_i32), r_ws, arith.addi(ws_pool_slot_base, pool_row))
                    row_src_packed = arith.ori(
                        arith.ori(src_pe, arith.shli(arith.andi(src_tok, mask20), c8)),
                        arith.shli(kp, c28),
                    )
                    buffer_store(_lv_unwrap(row_src_packed), r_ws, arith.addi(ws_row_src_base, pool_row))
                    sorted_id = arith.ori(arith.shli(kp, c24), arith.andi(src_tok, mask24))
                    buffer_store(_lv_unwrap(sorted_id), r_sorted_ids, pool_row)
                    wt = arith.bitcast(T.f32(), wt_bits)
                    buffer_store(_lv_unwrap(wt), r_sorted_w, pool_row)
        arrive_and_wait(arith.constant(4, type=T.i32()))

        # Phase 4: copy. Canonical rows pull P2P X and fan out duplicate X locally.
        num_valid_i32 = buffer_load(r_num_valid, c0, vec_width=1, dtype=T.i32())
        if const_expr(_can_same_pe_dup):
            dup_flag_i32 = buffer_load(r_ws, dup_flag_word, vec_width=1, dtype=T.i32())
            has_dup = arith.cmpi(CmpIPredicate.ne, dup_flag_i32, c0)
        else:
            has_dup = arith.cmpi(CmpIPredicate.eq, c0, c1)
        valid_scale_m_blocks = arith.divui(arith.addi(num_valid_i32, c31), c32)
        valid_scale_words = arith.muli(
            valid_scale_m_blocks,
            arith.constant(_mx_gn * 64, type=T.i32()),
        )

        p2p_x_base_pe = []
        p2p_scale_base_pe = []
        for pe in range_constexpr(_npes):
            pe_i32 = arith.constant(pe, type=T.i32())
            p2p_x_base_pe.append(buffer_load(r_p2p_x, pe_i32, vec_width=1, dtype=T.i64()))
            p2p_scale_base_pe.append(buffer_load(r_p2p_scale, pe_i32, vec_width=1, dtype=T.i64()))
        p2p_x_rsrc_pe = []
        p2p_scale_rsrc_pe = []
        for pe in range_constexpr(_npes):
            p2p_x_rsrc_pe.append(create_buffer_resource_from_addr(p2p_x_base_pe[pe]))
            p2p_scale_rsrc_pe.append(create_buffer_resource_from_addr(p2p_scale_base_pe[pe]))

        def p2p_x_rsrc_for(src_pe_i32):
            rsrc = p2p_x_rsrc_pe[0]
            for pe in range_constexpr(1, _npes):
                use_pe = arith.cmpi(CmpIPredicate.eq, src_pe_i32, arith.constant(pe, type=T.i32()))
                rsrc = arith.select(use_pe, p2p_x_rsrc_pe[pe], rsrc)
            return rsrc

        def p2p_scale_rsrc_for(src_pe_i32):
            rsrc = p2p_scale_rsrc_pe[0]
            for pe in range_constexpr(1, _npes):
                use_pe = arith.cmpi(CmpIPredicate.eq, src_pe_i32, arith.constant(pe, type=T.i32()))
                rsrc = arith.select(use_pe, p2p_scale_rsrc_pe[pe], rsrc)
            return rsrc

        for work_idx in range(_to_idx(work_id), _to_idx(valid_scale_words), _to_idx(work_stride)):
            buffer_store(_lv_unwrap(c0), r_scale_x, _to_i32(work_idx))

        def find_canon_for_pool_row(pool_row):
            slot_i32 = buffer_load(r_ws, arith.addi(ws_pool_slot_base, pool_row), vec_width=1, dtype=T.i32())
            canon = pool_row
            found = arith.cmpi(CmpIPredicate.eq, c0, c1)
            slot_ok = arith.cmpi(CmpIPredicate.ne, slot_i32, cm1)
            if slot_ok:
                r_slot = recv_meta_row(slot_i32)
                dst_slot = buffer_load(r_slot, arith.constant(5, type=T.i32()), vec_width=1, dtype=T.i32())
                base_slot = arith.subi(dst_slot, arith.remui(dst_slot, c_ept))
                for lane in range_constexpr(_ept):
                    cand_slot = arith.addi(base_slot, arith.constant(lane, type=T.i32()))
                    cand_slot_ok = arith.cmpi(CmpIPredicate.ult, cand_slot, arith.constant(_max_recv, type=T.i32()))
                    if cand_slot_ok:
                        r_cand = recv_meta_row(cand_slot)
                        cand_pool = buffer_load(
                            r_cand,
                            arith.constant(_POOL_ROW_WORD, type=T.i32()),
                            vec_width=1,
                            dtype=T.i32(),
                        )
                        cand_valid = arith.andi(
                            arith.cmpi(CmpIPredicate.ne, cand_pool, cm1),
                            arith.cmpi(CmpIPredicate.ult, cand_pool, num_valid_i32),
                        )
                        take = arith.andi(
                            arith.cmpi(
                                CmpIPredicate.eq,
                                found,
                                arith.cmpi(CmpIPredicate.eq, c0, c1),
                            ),
                            cand_valid,
                        )
                        canon = arith.select(take, cand_pool, _lv_unwrap(canon))
                        found = arith.ori(_lv_unwrap(found), cand_valid)
            return canon, slot_i32

        def copy_duplicate_siblings(pool_row, lane_i32, v, base_slot):
            for dup_lane in range_constexpr(_ept):
                cand_slot = arith.addi(base_slot, arith.constant(dup_lane, type=T.i32()))
                cand_slot_ok = arith.cmpi(CmpIPredicate.ult, cand_slot, arith.constant(_max_recv, type=T.i32()))
                if cand_slot_ok:
                    r_cand = recv_meta_row(cand_slot)
                    cand_pool = buffer_load(
                        r_cand,
                        arith.constant(_POOL_ROW_WORD, type=T.i32()),
                        vec_width=1,
                        dtype=T.i32(),
                    )
                    dup_valid = arith.andi(
                        arith.cmpi(CmpIPredicate.ne, cand_pool, cm1),
                        arith.andi(
                            arith.cmpi(CmpIPredicate.ult, cand_pool, num_valid_i32),
                            arith.cmpi(CmpIPredicate.ne, cand_pool, pool_row),
                        ),
                    )
                    if dup_valid:
                        dst_row = arith.addi(pool_base_i32, cand_pool)
                        dst_row_off = arith.muli(dst_row, c_token_bytes)
                        r_dst = create_buffer_resource_from_addr(
                            arith.addi(_lv_unwrap(addr_x_staging), arith.zext_i64(dst_row_off))
                        )
                        elem_off = arith.muli(lane_i32, c4)
                        buffer_store(v, r_dst, elem_off)

        def copy_x_pool_row(pool_row, lane_i32):
            row_src = buffer_load(r_ws, arith.addi(ws_row_src_base, pool_row), vec_width=1, dtype=T.i32())
            valid = arith.cmpi(CmpIPredicate.ne, row_src, cm1)
            dst_row = arith.addi(pool_base_i32, pool_row)
            dst_row_off = arith.muli(dst_row, c_token_bytes)
            r_dst = create_buffer_resource_from_addr(
                arith.addi(_lv_unwrap(addr_x_staging), arith.zext_i64(dst_row_off))
            )
            if valid:
                if const_expr(_can_same_pe_dup):
                    is_canon = arith.cmpi(CmpIPredicate.eq, c0, c0)
                    if has_dup:
                        lane0 = arith.cmpi(CmpIPredicate.eq, lane_i32, c0)
                        canon_local = pool_row
                        if lane0:
                            canon_found, _ = find_canon_for_pool_row(pool_row)
                            canon_local = canon_found
                        canon_row = readlane(canon_local, 0)
                        is_canon = arith.cmpi(CmpIPredicate.eq, canon_row, pool_row)
                else:
                    is_canon = arith.cmpi(CmpIPredicate.eq, c0, c0)
                src_pe = arith.andi(row_src, c_ff)
                src_tok = arith.andi(arith.shrui(row_src, c8), mask20)
                src_pe_ok = arith.cmpi(CmpIPredicate.ult, src_pe, c_npes)
                src_tok_ok = arith.cmpi(CmpIPredicate.ult, src_tok, c_mtpr)
                pool_ok = arith.cmpi(CmpIPredicate.ult, pool_row, num_valid_i32)
                if arith.andi(is_canon, arith.andi(pool_ok, arith.andi(src_pe_ok, src_tok_ok))):
                    src_row_i32_off = arith.muli(src_tok, arith.constant(_n_i32, type=T.i32()))
                    r_src = p2p_x_rsrc_for(src_pe)
                    for chunk_base in range_constexpr(0, x_chunks, 64):
                        chunk_i32 = arith.addi(lane_i32, arith.constant(chunk_base, type=T.i32()))
                        in_chunk = arith.cmpi(
                            CmpIPredicate.ult,
                            chunk_i32,
                            arith.constant(x_chunks, type=T.i32()),
                        )
                        if in_chunk:
                            elem_off = arith.muli(chunk_i32, c4)
                            src_elem_off = arith.addi(src_row_i32_off, elem_off)
                            v = buffer_load(r_src, src_elem_off, vec_width=4, dtype=T.i32())
                            buffer_store(v, r_dst, elem_off)
                            if const_expr(_can_same_pe_dup):
                                if has_dup:
                                    slot_i32 = buffer_load(
                                        r_ws,
                                        arith.addi(ws_pool_slot_base, pool_row),
                                        vec_width=1,
                                        dtype=T.i32(),
                                    )
                                    r_slot = recv_meta_row(slot_i32)
                                    dst_slot = buffer_load(
                                        r_slot,
                                        arith.constant(5, type=T.i32()),
                                        vec_width=1,
                                        dtype=T.i32(),
                                    )
                                    base_slot = arith.subi(dst_slot, arith.remui(dst_slot, c_ept))
                                    copy_duplicate_siblings(pool_row, chunk_i32, v, base_slot)
            else:
                for chunk_base in range_constexpr(0, x_chunks, 64):
                    chunk_i32 = arith.addi(lane_i32, arith.constant(chunk_base, type=T.i32()))
                    in_chunk = arith.cmpi(
                        CmpIPredicate.ult,
                        chunk_i32,
                        arith.constant(x_chunks, type=T.i32()),
                    )
                    if in_chunk:
                        elem_off = arith.muli(chunk_i32, c4)
                        z = arith.constant_vector(0, T.vector(4, T.i32()))
                        buffer_store(z, r_dst, elem_off)

        for work_idx in range(
            _to_idx(work_id),
            _to_idx(arith.muli(num_valid_i32, arith.constant(64, type=T.i32()))),
            _to_idx(work_stride),
        ):
            work_i32 = _to_i32(work_idx)
            pool_row = arith.divui(work_i32, arith.constant(64, type=T.i32()))
            lane_i32 = arith.remui(work_i32, arith.constant(64, type=T.i32()))
            copy_x_pool_row(pool_row, lane_i32)

        for work_idx in range(_to_idx(work_id), _to_idx(valid_scale_words), _to_idx(work_stride)):
            work_i32 = _to_i32(work_idx)
            prog_m = arith.divui(work_i32, stride_o0)
            rem0 = arith.remui(work_i32, stride_o0)
            prog_n = arith.divui(rem0, stride_o1)
            rem1 = arith.remui(rem0, stride_o1)
            ln = arith.divui(rem1, stride_o2)
            lm = arith.remui(rem1, stride_o2)
            pack = c0
            for i in range_constexpr(4):
                row_i = arith.addi(
                    arith.muli(prog_m, c32),
                    arith.addi(arith.constant((i % 2) * 16, type=T.i32()), lm),
                )
                col_i = arith.addi(
                    arith.muli(prog_n, c8),
                    arith.addi(arith.constant((i // 2) * 4, type=T.i32()), ln),
                )
                row_ok = arith.cmpi(CmpIPredicate.ult, row_i, num_valid_i32)
                col_ok = arith.cmpi(CmpIPredicate.ult, col_i, scale_cols_i32)
                if arith.andi(row_ok, col_ok):
                    row_src = buffer_load(r_ws, arith.addi(ws_row_src_base, row_i), vec_width=1, dtype=T.i32())
                    if arith.cmpi(CmpIPredicate.ne, row_src, cm1):
                        src_pe = arith.andi(row_src, c_ff)
                        src_tok = arith.andi(arith.shrui(row_src, c8), mask20)
                        src_pe_ok = arith.cmpi(CmpIPredicate.ult, src_pe, c_npes)
                        src_tok_ok = arith.cmpi(CmpIPredicate.ult, src_tok, c_mtpr)
                        if arith.andi(src_pe_ok, src_tok_ok):
                            r_src_scale = p2p_scale_rsrc_for(src_pe)
                            src_off = arith.addi(arith.muli(src_tok, scale_cols_i32), col_i)
                            b8 = buffer_load(r_src_scale, src_off, vec_width=1, dtype=T.i8())
                            pack = arith.ori(
                                _lv_unwrap(pack),
                                arith.shli(arith.extui(T.i32(), b8), arith.constant(i * 8, type=T.i32())),
                            )
            buffer_store(_lv_unwrap(pack), r_scale_x, work_i32)

        if const_expr(_fuse_metadata):
            r_reset_recv_meta = create_buffer_resource_from_addr(_lv_unwrap(addr_recv_meta))
            r_reset_tok_map = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_map))
            r_reset_tis = create_buffer_resource_from_addr(_lv_unwrap(addr_tis))
            r_reset_recv_num = create_buffer_resource_from_addr(_lv_unwrap(addr_recv_num))
            r_reset_dest_ctr = create_buffer_resource_from_addr(_lv_unwrap(addr_dest_ctr))
            r_reset_tok_off = create_buffer_resource_from_addr(_lv_unwrap(addr_tok_off))
            r_reset_disp_bar = create_buffer_resource_from_addr(_lv_unwrap(addr_disp_bar))
            r_reset_total_rv = create_buffer_resource_from_addr(_lv_unwrap(addr_total_rv))

            for reset_idx in range(
                _to_idx(work_id),
                _to_idx(scan_limit_words),
                _to_idx(work_stride),
            ):
                buffer_store(_lv_unwrap(c0), r_reset_recv_meta, _to_i32(reset_idx))

            for reset_idx in range(
                _to_idx(work_id),
                _to_idx(reset_tok_map_words),
                _to_idx(work_stride),
            ):
                buffer_store(_lv_unwrap(reset_sentinel), r_reset_tok_map, _to_i32(reset_idx))

            for reset_idx in range(
                _to_idx(work_id),
                _to_idx(reset_tis_words),
                _to_idx(work_stride),
            ):
                buffer_store(_lv_unwrap(c0), r_reset_tis, _to_i32(reset_idx))

            for reset_idx in range(_to_idx(work_id), _to_idx(c_npes), _to_idx(work_stride)):
                i = _to_i32(reset_idx)
                buffer_store(_lv_unwrap(c0), r_reset_recv_num, i)
                buffer_store(_lv_unwrap(c0), r_reset_dest_ctr, i)

            if arith.cmpi(CmpIPredicate.eq, work_id, c0):
                buffer_store(_lv_unwrap(c0), r_reset_tok_off, c0)
                buffer_store(_lv_unwrap(c0), r_reset_disp_bar, c0)
                buffer_store(_lv_unwrap(c0), r_reset_total_rv, c0)


    @flyc.jit
    def launch_pull_to_pool_a8w4(
        addr_recv_meta: fx.Int64,
        addr_x_staging: fx.Int64,
        addr_p2p_x: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_scale_x: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sort_workspace: fx.Int64,
        pool_base: fx.Int32,
        stream: Stream = Stream(None),
    ):
        ep_stage1_metadata_pull_to_pool_a8w4(
            addr_recv_meta,
            addr_x_staging,
            addr_p2p_x,
            addr_p2p_scale,
            addr_scale_x,
            addr_sorted_token_ids,
            addr_sorted_expert_ids,
            addr_num_valid_ids,
            addr_sorted_weights,
            addr_sort_workspace,
            pool_base,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int32(0),
        ).launch(
            grid=(block_num, 1, 1),
            block=(_threads_per_block, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_stage1_metadata_pull_to_pool_a8w4(
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_tok_off: fx.Int64,
        addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64,
        addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_tis: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        addr_p2p_recv_meta: fx.Int64,
        addr_inp_tok: fx.Int64,
        addr_x_staging: fx.Int64,
        addr_pull_row_i64: fx.Int64,
        addr_p2p_pull_row: fx.Int64,
        cur_tok: fx.Int32,
        addr_recv_meta: fx.Int64,
        addr_p2p_x: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_scale_x: fx.Int64,
        addr_sorted_token_ids: fx.Int64,
        addr_sorted_expert_ids: fx.Int64,
        addr_num_valid_ids: fx.Int64,
        addr_sorted_weights: fx.Int64,
        addr_sort_workspace: fx.Int64,
        pool_base: fx.Int32,
        stream: Stream = Stream(None),
    ):
        ep_stage1_metadata_pull_to_pool_a8w4(
            addr_recv_meta,
            addr_x_staging,
            addr_p2p_x,
            addr_p2p_scale,
            addr_scale_x,
            addr_sorted_token_ids,
            addr_sorted_expert_ids,
            addr_num_valid_ids,
            addr_sorted_weights,
            addr_sort_workspace,
            pool_base,
            addr_idx,
            addr_wts,
            addr_tok_off,
            addr_recv_num,
            addr_dest_ctr,
            addr_disp_bar,
            addr_tok_map,
            addr_total_rv,
            addr_tis,
            addr_p2p_tok_off,
            addr_p2p_tis,
            addr_p2p_recv_num,
            addr_p2p_recv_meta,
            addr_inp_tok,
            addr_pull_row_i64,
            addr_p2p_pull_row,
            cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=(_threads_per_block, 1, 1),
            stream=stream,
        )

    if _fuse_metadata:
        return launch_stage1_metadata_pull_to_pool_a8w4
    return launch_pull_to_pool_a8w4


def make_stage1_metadata_pull_to_pool_a8w4_jit(**kwargs):
    """JIT: K1 metadata dispatch + K2 pull-to-pool in one device kernel."""
    return make_pull_to_pool_a8w4_jit(**kwargs, fuse_metadata=True)


__all__ = ["make_pull_to_pool_a8w4_jit", "make_stage1_metadata_pull_to_pool_a8w4_jit"]
