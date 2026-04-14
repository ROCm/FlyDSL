# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


"""Shared utilities for gfx1250 MoE 2-stage kernels.

Common helpers used by both the fp16 WMMA kernels and the mxscale
(fp4/fp8/a8w4) kernels.
"""

from __future__ import annotations

import inspect
from typing import Any

from flydsl.runtime.device import get_rocm_arch as get_hip_arch


def _require_gfx1250() -> None:
    arch = str(get_hip_arch())
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"Expected gfx1250 architecture, got {arch!r}")


def _align_up(v: int, a: int) -> int:
    return ((int(v) + int(a) - 1) // int(a)) * int(a)


def _pick_fp4_warp_shape(tile_m: int, tile_n: int) -> tuple[int, int]:
    """Pick a legal (m_warp, n_warp) for compile_mxfp4_gemm constraints."""
    for m_warp in (4, 2, 1):
        if tile_m % m_warp != 0:
            continue
        warp_tile_m = tile_m // m_warp
        if (warp_tile_m % 16) != 0:
            continue
        for n_warp in (4, 2, 1):
            if tile_n % n_warp != 0:
                continue
            warp_tile_n = tile_n // n_warp
            if (warp_tile_n % 32) == 0:
                return m_warp, n_warp
    raise ValueError(
        f"Cannot find legal (m_warp,n_warp) for FP4 GEMM with tile_m={tile_m}, tile_n={tile_n}. "
        "Need warp_tile_m multiple of 16 and warp_tile_n multiple of 32."
    )


def _pick_fp16_single_launch_shape(route_tile_m: int, route_tile_n: int,
                                    max_total_warps: int = 0) -> tuple[int, int, int, int]:
    """Pick launch shape for fp16 stage1 single-kernel path.

    Single-kernel path should follow route tile size (not backend-expanded 128x*)
    while keeping legal WMMA tile decomposition.
    """
    tile_m = _align_up(int(route_tile_m), 16)
    tile_n = _align_up(int(route_tile_n), 16)
    for mw in (4, 2, 1):
        if tile_m % mw != 0:
            continue
        if (tile_m // mw) % 16 != 0:
            continue
        for nw in (8, 4, 2, 1):
            if max_total_warps > 0 and mw * nw > max_total_warps:
                continue
            if tile_n % nw != 0:
                continue
            if (tile_n // nw) % 16 != 0:
                continue
            return tile_m, tile_n, mw, nw
    raise ValueError(
        f"Cannot find legal single-kernel fp16 shape for tile_m={route_tile_m}, tile_n={route_tile_n}"
    )


def _compile_with_optional_wpe(fn, kwargs: dict[str, Any]):
    sig = inspect.signature(fn)
    if "waves_per_eu" not in sig.parameters:
        kwargs = {k: v for k, v in kwargs.items() if k != "waves_per_eu"}
    return fn(**kwargs)


def _bf16_to_f16_wrapper(fp16_exe, x_arg: int, w_arg: int):
    """Wrap a compiled fp16 kernel to accept bf16 inputs by converting them to fp16 on the host."""
    import torch

    def wrapper(*args, **kwargs):
        args = list(args)
        for idx in (x_arg, w_arg):
            if idx < len(args) and hasattr(args[idx], 'dtype') and args[idx].dtype == torch.bfloat16:
                args[idx] = args[idx].to(torch.float16)
        return fp16_exe(*args, **kwargs)

    for attr in ('mode',):
        if hasattr(fp16_exe, attr):
            setattr(wrapper, attr, getattr(fp16_exe, attr))
    return wrapper


def _pick_mxscale_launch_shape(data_format: str, route_tile_m: int, tile_n: int) -> tuple[int, int, int, int]:
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")
    if data_format == "fp4":
        single_tile_m = _align_up(int(route_tile_m), 16)
        single_tile_n = _align_up(int(tile_n), 32)
        single_m_warp, single_n_warp = _pick_fp4_warp_shape(single_tile_m, single_tile_n)
        return single_tile_m, single_tile_n, single_m_warp, single_n_warp
    return _pick_fp16_single_launch_shape(int(route_tile_m), int(tile_n), max_total_warps=8)


def _make_moe_wave_layout(*, m_warp: int, n_warp: int, WAVE_SIZE: int, fx):
    return fx.make_layout(
        (int(m_warp), int(n_warp), 2, 16),
        (int(n_warp) * WAVE_SIZE, WAVE_SIZE, 16, 1),
    )


def _make_wmma_sub_tiles(
    *, wmma_m_rep: int, wmma_n_rep: int, WMMA_M: int, is_fp4: bool
) -> list[tuple[int, int, int, int]]:
    sub_tiles = []
    for wm in range(wmma_m_rep):
        for wn in range(wmma_n_rep):
            if is_fp4:
                for half in range(2):
                    sub_tiles.append((wm * wmma_n_rep + wn, half * 8, wm * WMMA_M, wn * 2 + half))
            else:
                sub_tiles.append((wm * wmma_n_rep + wn, 0, wm * WMMA_M, wn))
    return sub_tiles


def _moe_out_elem_ty(out_dtype: str, T):
    return T.f16 if out_dtype == "f16" else T.bf16


def _extract_sub8(acc, vec_base: int, *, vector, range_constexpr, ACC_VEC_SIZE: int):
    if ACC_VEC_SIZE == 8:
        return acc
    return vector.shuffle(acc, acc, [vec_base + i for i in range_constexpr(8)])


def _finalize_alloc_and_launch_2d(*, ctx, alloc, launcher, gx, gy, block_threads: int, stream, waves_per_eu, ir,
                                  cluster=None):
    with ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalized = False
        alloc.finalize()
    for op in ctx.gpu_module_body.operations:
        if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
            if waves_per_eu is not None and int(waves_per_eu) >= 1:
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), int(waves_per_eu)
                )
            if cluster is not None:
                op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                    f"{cluster[0]},{cluster[1]},{cluster[2]}")
    launcher.launch(
        grid=(gx, gy, 1),
        block=(block_threads, 1, 1),
        stream=stream,
        cluster=cluster,
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
    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    default_block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
    row_base_i32 = block_row_start if block_row_start is not None else default_block_row_start
    for acc_idx, vec_base, m_off, wn in sub_tiles:
        row_local = warp_m_base + fx.Index(m_off) + lane16
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
        sorted_safe = arith.select(
            row_ok_meta,
            sorted_i32,
            row_base_i32,
        )
        fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
        slot = fused >> arith.constant(24, type=T.i32)
        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.constant(int(topk), type=T.i32))
        row_ok = arith.andi(row_ok_meta, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1)))
        sub8g, sub8u = load_gate_up_sub8(acc_idx, vec_base)
        tw = buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32) if bool(doweight_stage1) else arith.constant(1.0, type=T.f32)
        col_base = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane_kgrp * fx.Index(8)
        for vi in range_constexpr(8):
            col = col_base + fx.Index(vi)
            col_ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, col), i32_inter_in)
            out_ok = arith.andi(row_ok, col_ok)
            _if_out = scf.IfOp(out_ok)
            with ir.InsertionPoint(_if_out.then_block):
                vg = vector.extract(sub8g, static_position=[vi], dynamic_position=[])
                vu = vector.extract(sub8u, static_position=[vi], dynamic_position=[])
                y = silu_fn(vg) * vu
                if bool(doweight_stage1):
                    y = y * tw
                out_v = arith.trunc_f(out_elem_ty, y)
                out_idx = ((tok * c_topk_i32 + slot) * i32_inter_in
                           + arith.index_cast(T.i32, col))
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
    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    c2_i32 = arith.constant(2, type=T.i32)
    zero_i32 = arith.constant(0, type=T.i32)
    mask_even_i32 = arith.constant(0xFFFFFFFE, type=T.i32)

    def atomic_add_x2(val_x2, byte_off_i32):
        rocdl.raw_ptr_buffer_atomic_fadd(val_x2, out_rsrc, byte_off_i32, zero_i32, zero_i32)

    for acc_idx, vec_base, m_off, wn in sub_tiles:
        row_local = warp_m_base + fx.Index(m_off) + lane16
        sorted_row = by * arith.index(int(tile_m)) + row_local
        row_i32 = arith.index_cast(T.i32, row_local)
        sorted_i32 = arith.index_cast(T.i32, sorted_row)
        row_in_route = arith.cmpi(arith.CmpIPredicate.ult, row_i32, arith.constant(int(route_tile_m), type=T.i32))
        row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
        row_ok = arith.andi(row_in_route, row_in_valid)
        sorted_safe = arith.select(row_ok, sorted_i32, block_row_start)
        fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
        slot = fused >> arith.constant(24, type=T.i32)
        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
        row_store_ok = arith.andi(row_ok, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1)))
        ts = tok * c_topk_i32 + slot
        sub8 = load_sub8(acc_idx, vec_base)
        tw = buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32) if bool(doweight_stage2) else arith.constant(1.0, type=T.f32)
        col_base = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane_kgrp * fx.Index(8)
        if bool(accumulate):
            for vpair in range_constexpr(4):
                vi0 = vpair * 2
                vi1 = vi0 + 1
                col0 = col_base + fx.Index(vi0)
                col1 = col_base + fx.Index(vi1)
                col0_i32 = arith.index_cast(T.i32, col0)
                col1_i32 = arith.index_cast(T.i32, col1)
                col0_ok = arith.cmpi(arith.CmpIPredicate.ult, col0_i32, i32_n_in)
                col1_ok = arith.cmpi(arith.CmpIPredicate.ult, col1_i32, i32_n_in)
                out_ok = arith.andi(row_store_ok, col0_ok)
                _if_out = scf.IfOp(out_ok)
                with ir.InsertionPoint(_if_out.then_block):
                    v0 = vector.extract(sub8, static_position=[vi0], dynamic_position=[])
                    v1 = vector.extract(sub8, static_position=[vi1], dynamic_position=[])
                    if bool(doweight_stage2):
                        v0 = v0 * tw
                        v1 = v1 * tw
                    v1 = arith.select(col1_ok, v1, arith.constant(0.0, type=T.f32))
                    out0 = arith.trunc_f(out_elem_ty, v0)
                    out1 = arith.trunc_f(out_elem_ty, v1)
                    frag = vector.from_elements(T.vec(2, out_elem_ty), [out0, out1])
                    idx0 = tok * i32_n_in + col0_i32
                    idx_even = idx0 & mask_even_i32
                    byte_off = idx_even * c2_i32
                    atomic_add_x2(frag, byte_off)
                    scf.YieldOp([])
        else:
            for vi in range_constexpr(8):
                col = col_base + fx.Index(vi)
                col_ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, col), i32_n_in)
                out_ok = arith.andi(row_store_ok, col_ok)
                _if_out = scf.IfOp(out_ok)
                with ir.InsertionPoint(_if_out.then_block):
                    v = vector.extract(sub8, static_position=[vi], dynamic_position=[])
                    if bool(doweight_stage2):
                        v = v * tw
                    col_i32 = arith.index_cast(T.i32, col)
                    out_idx = ts * i32_n_in + col_i32
                    out_v = arith.trunc_f(out_elem_ty, v)
                    buffer_ops.buffer_store(out_v, out_rsrc, out_idx)
                    scf.YieldOp([])


def _pack_stage1_gate_up_tiles(tensor, *, experts: int, inter_dim: int, tile_n: int, cols: int):
    """Pack stage1 gate/up rows into [gate_tile0, up_tile0, gate_tile1, up_tile1, ...]."""
    import torch

    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for stage1 gate/up packing, got {type(tensor)!r}")
    if tensor.numel() == 0:
        return tensor
    elems_per_expert = int(2 * inter_dim) * int(cols)
    if tensor.numel() != int(experts) * elems_per_expert:
        if tensor.numel() % elems_per_expert != 0:
            raise ValueError(
                "Unexpected stage1 tensor size for gate/up packing: "
                f"numel={tensor.numel()} expected={int(experts) * elems_per_expert} "
                f"(experts={experts}, inter_dim={inter_dim}, cols={cols})"
            )
        experts = tensor.numel() // elems_per_expert
    expected_rows = int(experts) * int(2 * inter_dim)
    if int(inter_dim) % int(tile_n) != 0:
        raise ValueError(
            f"Stage1 gate/up packed layout requires inter_dim divisible by tile_n, got {inter_dim} and {tile_n}"
        )

    tensor_3d = tensor.contiguous().view(int(experts), int(2 * inter_dim), int(cols))
    gate = tensor_3d[:, :int(inter_dim), :]
    up = tensor_3d[:, int(inter_dim):, :]
    gate_tiles = gate.view(int(experts), int(inter_dim // tile_n), int(tile_n), int(cols))
    up_tiles = up.view(int(experts), int(inter_dim // tile_n), int(tile_n), int(cols))
    packed = torch.cat((gate_tiles, up_tiles), dim=2)
    return packed.view(expected_rows, int(cols))


class _Stage1GateUpPackedWrapper:
    """Host-side wrapper that repacks stage1 W1 rows to match the merged gate/up TDM layout."""

    def __init__(
        self,
        stage1_exe,
        *,
        experts: int,
        inter_dim: int,
        tile_n: int,
        packed_cols_w: int,
        packed_cols_scale: int,
    ):
        self._stage1_exe = stage1_exe
        self._experts = int(experts)
        self._inter_dim = int(inter_dim)
        self._tile_n = int(tile_n)
        self._packed_cols_w = int(packed_cols_w)
        self._packed_cols_scale = int(packed_cols_scale)
        self._cache = {}

        for attr in ("mode", "compile_hints"):
            if hasattr(stage1_exe, attr):
                setattr(self, attr, getattr(stage1_exe, attr))

    def _get_packed_operands(self, arg_w, arg_scale_w):
        key = (id(arg_w), id(arg_scale_w))
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]

        packed_w = _pack_stage1_gate_up_tiles(
            arg_w,
            experts=self._experts,
            inter_dim=self._inter_dim,
            tile_n=self._tile_n,
            cols=self._packed_cols_w,
        )
        if hasattr(arg_scale_w, "numel") and int(arg_scale_w.numel()) > 0:
            packed_scale_w = _pack_stage1_gate_up_tiles(
                arg_scale_w,
                experts=self._experts,
                inter_dim=self._inter_dim,
                tile_n=self._tile_n,
                cols=self._packed_cols_scale,
            )
        else:
            packed_scale_w = arg_scale_w

        # Store (result, original_refs) — the strong refs to originals
        # prevent id() reuse while the entry is alive.
        self._cache[key] = ((packed_w, packed_scale_w), (arg_w, arg_scale_w))
        return packed_w, packed_scale_w

    def __call__(self, *args, **kwargs):
        args = list(args)
        if len(args) > 4:
            args[2], args[4] = self._get_packed_operands(args[2], args[4])
        return self._stage1_exe(*args, **kwargs)
