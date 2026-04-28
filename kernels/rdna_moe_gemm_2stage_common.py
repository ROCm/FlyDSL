# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared utilities for RDNA4 (gfx120x) MoE 2-stage WMMA kernels.

Helpers that the RDNA4 WMMA MoE path (``rdna_moe_gemm_2stage.py``) pulls in.
"""

from __future__ import annotations

from flydsl.runtime.device import get_rocm_arch as get_hip_arch


def _require_gfx120x() -> None:
    arch = str(get_hip_arch())
    if not arch.startswith("gfx120"):
        raise RuntimeError(f"Expected gfx120x (RDNA4) architecture, got {arch!r}")


def _align_up(v: int, a: int) -> int:
    return ((int(v) + int(a) - 1) // int(a)) * int(a)


def _moe_out_elem_ty(out_dtype: str, T):
    """RDNA4 MoE output element type mapping (f16, bf16, or f32)."""

    out_s = str(out_dtype).strip().lower()
    if out_s in ("f16", "fp16", "half"):
        return T.f16
    if out_s in ("bf16", "bfloat16"):
        return T.bf16
    if out_s in ("f32", "fp32", "float"):
        return T.f32
    raise ValueError(f"Unsupported out_dtype {out_dtype!r}")


def _make_moe_wave_layout(*, m_warp: int, n_warp: int, WAVE_SIZE: int, fx):
    return fx.make_layout(
        (int(m_warp), int(n_warp), 2, 16),
        (int(n_warp) * WAVE_SIZE, WAVE_SIZE, 16, 1),
    )


def _make_wmma_sub_tiles(
    *, wmma_m_rep: int, wmma_n_rep: int, WMMA_M: int, is_fp4: bool = False
) -> list:
    sub_tiles = []
    for wm in range(wmma_m_rep):
        for wn in range(wmma_n_rep):
            if is_fp4:
                for half in range(2):
                    sub_tiles.append(
                        (wm * wmma_n_rep + wn, half * 8, wm * WMMA_M, wn * 2 + half)
                    )
            else:
                sub_tiles.append((wm * wmma_n_rep + wn, 0, wm * WMMA_M, wn))
    return sub_tiles


def _finalize_alloc_and_launch_2d(
    *,
    ctx,
    alloc,
    launcher,
    gx,
    gy,
    block_threads: int,
    stream,
    waves_per_eu,
    ir,
):
    with ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalized = False
        alloc.finalize()
    for op in ctx.gpu_module_body.operations:
        if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
            if waves_per_eu is not None and int(waves_per_eu) >= 1:
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), int(waves_per_eu)
                )
    launcher.launch(
        grid=(gx, gy, 1),
        block=(block_threads, 1, 1),
        stream=stream,
    )


def _fp16_tile_lds_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    num_b_tiles: int = 1,
    lds_pad_a: int = 8,
    lds_pad_b: int = 8,
    elem_bytes: int = 2,
) -> int:
    """Estimate LDS bytes for the RDNA4 stage1/stage2 WMMA tile layout."""

    lds_a_stride = int(tile_k) + int(lds_pad_a)
    lds_b_stride = int(tile_k) + int(lds_pad_b)
    lds_a_elems = int(tile_m) * lds_a_stride + int(lds_pad_a)
    lds_b_elems = int(tile_n) * lds_b_stride + int(lds_pad_b)
    return (lds_a_elems + int(num_b_tiles) * lds_b_elems) * int(elem_bytes)


def _pick_fp16_launch_shape(
    route_tile_m: int,
    route_tile_n: int,
    tile_k: int,
    *,
    max_total_warps: int = 4,
    lds_budget_bytes: int = 60 * 1024,
) -> tuple[int, int, int, int]:
    """Pick a legal launch shape for the RDNA4 fp16/bf16 WMMA MoE path.

    The returned tuple is ``(tile_m, tile_n, m_warp, n_warp)``.
    """

    tile_m = _align_up(int(route_tile_m), 16)
    tile_n = _align_up(int(route_tile_n), 16)

    lds_bytes = _fp16_tile_lds_bytes(tile_m, tile_n, int(tile_k))
    if lds_bytes > int(lds_budget_bytes):
        raise ValueError(
            f"RDNA4 MoE LDS budget exceeded for tile=({tile_m},{tile_n},{tile_k}): "
            f"{lds_bytes} bytes > {lds_budget_bytes} bytes"
        )

    preferred = (
        (2, 2),
        (1, 4),
        (4, 1),
        (2, 1),
        (1, 2),
        (1, 1),
    )
    for mw, nw in preferred:
        if mw * nw > int(max_total_warps):
            continue
        if tile_m % mw != 0 or tile_n % nw != 0:
            continue
        if (tile_m // mw) % 16 != 0 or (tile_n // nw) % 16 != 0:
            continue
        return tile_m, tile_n, mw, nw

    raise ValueError(
        "Cannot find legal RDNA4 WMMA launch shape for "
        f"tile_m={route_tile_m}, tile_n={route_tile_n}, tile_k={tile_k}"
    )


def _emit_stage1_gate_up_epilogue(
    *,
    sub_tiles,
    by,
    tile_m: int,
    route_tile_m: int,
    warp_m_base,
    warp_n_base,
    blk_n,
    lane16,
    lane_kgrp,
    WMMA_N: int,
    i32_tokens_in,
    i32_inter_in,
    topk: int,
    num_valid_i32=None,
    block_row_start=None,
    sorted_rsrc,
    tw_rsrc,
    out_rsrc,
    doweight_stage1: bool,
    out_elem_ty,
    load_gate_up_sub8,
    silu_fn,
    ir,
    fx,
    arith,
    buffer_ops,
    scf,
    vector,
    range_constexpr,
    T,
):
    """RDNA4 stage1 gate/up epilogue (WMMA_K=16 accumulator lane layout)."""

    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    default_block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
    row_base_i32 = block_row_start if block_row_start is not None else default_block_row_start
    for acc_idx, vec_base, m_off, wn in sub_tiles:
        sub8g, sub8u = load_gate_up_sub8(acc_idx, vec_base)
        col = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane16
        col_i32 = arith.index_cast(T.i32, col)
        col_ok = arith.cmpi(arith.CmpIPredicate.ult, col_i32, i32_inter_in)
        for vi in range_constexpr(8):
            row_local = warp_m_base + fx.Index(m_off) + lane_kgrp * fx.Index(8) + fx.Index(vi)
            sorted_row = by * arith.index(int(tile_m)) + row_local
            row_i32 = arith.index_cast(T.i32, row_local)
            sorted_i32 = arith.index_cast(T.i32, sorted_row)
            row_in_route = arith.cmpi(
                arith.CmpIPredicate.ult,
                row_i32,
                arith.constant(int(route_tile_m), type=T.i32),
            )
            if num_valid_i32 is None:
                row_ok_meta = row_in_route
            else:
                row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
                row_ok_meta = arith.andi(row_in_route, row_in_valid)
            sorted_safe = arith.select(row_ok_meta, sorted_i32, row_base_i32)
            fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
            tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
            slot = fused >> arith.constant(24, type=T.i32)
            tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
            slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
            slot_ok1 = arith.cmpi(
                arith.CmpIPredicate.slt, slot, arith.constant(int(topk), type=T.i32)
            )
            row_ok = arith.andi(
                row_ok_meta, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
            )
            tw = (
                buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32)
                if bool(doweight_stage1)
                else arith.constant(1.0, type=T.f32)
            )
            out_ok = arith.andi(row_ok, col_ok)
            _if_out = scf.IfOp(out_ok)
            with ir.InsertionPoint(_if_out.then_block):
                vg = vector.extract(sub8g, static_position=[vi], dynamic_position=[])
                vu = vector.extract(sub8u, static_position=[vi], dynamic_position=[])
                y = silu_fn(vg) * vu
                if bool(doweight_stage1):
                    y = y * tw
                out_v = arith.trunc_f(out_elem_ty, y)
                out_idx = (tok * c_topk_i32 + slot) * i32_inter_in + col_i32
                buffer_ops.buffer_store(out_v, out_rsrc, out_idx)
                scf.YieldOp([])


def _emit_stage2_store_epilogue(
    *,
    sub_tiles,
    by,
    tile_m: int,
    route_tile_m: int,
    warp_m_base,
    warp_n_base,
    blk_n,
    lane16,
    lane_kgrp,
    WMMA_N: int,
    i32_tokens_in,
    i32_n_in,
    topk: int,
    num_valid_i32,
    block_row_start,
    sorted_rsrc,
    tw_rsrc,
    out_rsrc,
    doweight_stage2: bool,
    accumulate: bool,
    out_elem_ty,
    out_is_f32: bool,
    load_sub8,
    ir,
    fx,
    arith,
    buffer_ops,
    scf,
    vector,
    range_constexpr,
    rocdl,
    T,
):
    """RDNA4 stage2 store epilogue (WMMA_K=16 accumulator lane layout)."""

    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    c4_i32 = arith.constant(4, type=T.i32)
    zero_i32 = arith.constant(0, type=T.i32)

    for acc_idx, vec_base, m_off, wn in sub_tiles:
        sub8 = load_sub8(acc_idx, vec_base)
        col = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane16
        col_i32 = arith.index_cast(T.i32, col)
        col_ok = arith.cmpi(arith.CmpIPredicate.ult, col_i32, i32_n_in)
        if bool(accumulate):
            for vi in range_constexpr(8):
                row_local = warp_m_base + fx.Index(m_off) + lane_kgrp * fx.Index(8) + fx.Index(vi)
                sorted_row = by * arith.index(int(tile_m)) + row_local
                row_i32 = arith.index_cast(T.i32, row_local)
                sorted_i32 = arith.index_cast(T.i32, sorted_row)
                row_in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    row_i32,
                    arith.constant(int(route_tile_m), type=T.i32),
                )
                row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
                row_ok = arith.andi(row_in_route, row_in_valid)
                sorted_safe = arith.select(row_ok, sorted_i32, block_row_start)
                fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
                tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                slot = fused >> arith.constant(24, type=T.i32)
                tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
                slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
                row_store_ok = arith.andi(
                    row_ok, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                )
                tw = (
                    buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32)
                    if bool(doweight_stage2)
                    else arith.constant(1.0, type=T.f32)
                )
                out_ok = arith.andi(row_store_ok, col_ok)
                _if_out = scf.IfOp(out_ok)
                with ir.InsertionPoint(_if_out.then_block):
                    v = vector.extract(sub8, static_position=[vi], dynamic_position=[])
                    if bool(doweight_stage2):
                        v = v * tw
                    out_idx = tok * i32_n_in + col_i32
                    byte_off = out_idx * (c4_i32 if bool(out_is_f32) else arith.constant(2, type=T.i32))
                    if bool(out_is_f32):
                        rocdl.raw_ptr_buffer_atomic_fadd(v, out_rsrc, byte_off, zero_i32, zero_i32)
                    else:
                        out_v = arith.trunc_f(out_elem_ty, v)
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            out_v, out_rsrc, byte_off, zero_i32, zero_i32
                        )
                    scf.YieldOp([])
        else:
            for vi in range_constexpr(8):
                row_local = warp_m_base + fx.Index(m_off) + lane_kgrp * fx.Index(8) + fx.Index(vi)
                sorted_row = by * arith.index(int(tile_m)) + row_local
                row_i32 = arith.index_cast(T.i32, row_local)
                sorted_i32 = arith.index_cast(T.i32, sorted_row)
                row_in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    row_i32,
                    arith.constant(int(route_tile_m), type=T.i32),
                )
                row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
                row_ok = arith.andi(row_in_route, row_in_valid)
                sorted_safe = arith.select(row_ok, sorted_i32, block_row_start)
                fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
                tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                slot = fused >> arith.constant(24, type=T.i32)
                tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
                slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
                row_store_ok = arith.andi(
                    row_ok, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                )
                ts = tok * c_topk_i32 + slot
                tw = (
                    buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32)
                    if bool(doweight_stage2)
                    else arith.constant(1.0, type=T.f32)
                )
                out_ok = arith.andi(row_store_ok, col_ok)
                _if_out = scf.IfOp(out_ok)
                with ir.InsertionPoint(_if_out.then_block):
                    v = vector.extract(sub8, static_position=[vi], dynamic_position=[])
                    if bool(doweight_stage2):
                        v = v * tw
                    out_idx = ts * i32_n_in + col_i32
                    out_v = arith.trunc_f(out_elem_ty, v)
                    buffer_ops.buffer_store(out_v, out_rsrc, out_idx)
                    scf.YieldOp([])
