# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


"""gfx1250 MoE 2-stage kernels and wrappers.

Target architecture: AMD RDNA4 `gfx1250`.
Supported input dtypes: `fp16`, `fp8`, `fp4`, and `a8w4`.

- `fp16`: stage1/stage2 support single-kernel inline paths
  (route-pack + TDM + WMMA + epilog), with fallback migration modes.
- `fp8`: stage1/stage2 support single-kernel inline paths
  (route-pack + TDM + WMMA_SCALE + epilog), and keep phase1 fallback
  to validated `mxfp8_gemm_gfx1250` backend.
- `a8w4`: stage1/stage2 reuse the fp8 activation path but consume FP4-packed
  weights with E8M0 block scales via the same gfx1250 `WMMA_SCALE` instruction
  family, aligned with `kernels/gemm_fp8fp4_gfx1250.py`.
- `fp4`: stage1/stage2 support single-kernel inline paths
  (route-pack + TDM + WMMA_SCALE + epilog), and keep phase1 fallback
  to validated `mxfp4_gemm_gfx1250` backend
  (`wmma_scale_f32_32x16x128_f4`).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from kernels.moe_gemm_2stage import (
    MoeGemm2Mode,
    compile_moe_gemm1 as _compile_moe_gemm1_base,
    compile_moe_gemm2 as _compile_moe_gemm2_base,
    compile_moe_gemm2_ex as _compile_moe_gemm2_ex_base,
    compile_moe_reduction,
)


def _require_gfx1250() -> None:
    arch = str(get_hip_arch())
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"Expected gfx1250 architecture, got {arch!r}")


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


def _align_up(v: int, a: int) -> int:
    return ((int(v) + int(a) - 1) // int(a)) * int(a)


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


def _mxscale_format_config(data_format: str) -> dict[str, int | bool]:
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")
    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    pack_factor_a = 1 if not is_fp4 else 2
    pack_factor_b = 2 if (is_fp4 or is_a8w4) else 1
    wmma_n_eff = 32 if is_fp4 else 16
    acc_vec_size = 16 if is_fp4 else 8
    ds_loads_per_a_frag = 2 if is_fp4 else 4
    return {
        "is_fp4": is_fp4,
        "is_a8w4": is_a8w4,
        "PACK_FACTOR_A": pack_factor_a,
        "PACK_FACTOR_B": pack_factor_b,
        "WMMA_N_EFF": wmma_n_eff,
        "ACC_VEC_SIZE": acc_vec_size,
        "DS_LOADS_PER_A_FRAG": ds_loads_per_a_frag,
    }


def _mxscale_precompute_preshuffled_b_data_bases(
    *,
    packed_tile_k_b: int,
    warp_tile_n,
    wave_n_idx,
    lane16,
    lane_kgrp,
    wmma_n_rep: int,
    arith,
    range_constexpr,
):
    ngroup_stride = packed_tile_k_b * 16
    n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
    row_off = lane16 * arith.index(16)
    k_tile_off = lane_kgrp * arith.index(256)
    bases = []
    for wn in range_constexpr(wmma_n_rep):
        ngroup_off = n_group_base * arith.index(ngroup_stride) + arith.index(wn * ngroup_stride)
        bases.append(ngroup_off + row_off + k_tile_off)
    return bases


def _mxscale_precompute_a_scale_lane_bases(
    *,
    warp_m_base,
    lane16,
    wmma_m_rep: int,
    interleaved_scale_cols_a: int,
    arith,
):
    warp_lds_row = warp_m_base / arith.index(wmma_m_rep) + lane16
    base = warp_lds_row * arith.index(interleaved_scale_cols_a)
    return [base]


def _mxscale_precompute_b_scale_lane_bases(
    *,
    warp_n_base,
    lane16,
    lane_kgrp,
    b_scale_load_rep: int,
    interleaved_scale_cols_b: int,
    is_fp4: bool,
    is_a8w4: bool,
    arith,
    SCALES_PER_WMMA: int,
):
    warp_lds_row = warp_n_base / arith.index(b_scale_load_rep) + lane16
    base = warp_lds_row * arith.index(interleaved_scale_cols_b)
    if is_fp4 or is_a8w4:
        base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)
    return [base]


def _mxscale_load_scale_b128(
    *,
    lds_buffer,
    scale_base,
    reps: int,
    ks,
    SCALES_PER_WMMA: int,
    _lds_load_b128,
    arith,
    vector,
    range_constexpr,
):
    ks_byte_off = ks * reps * SCALES_PER_WMMA
    eff_base = scale_base if ks_byte_off == 0 else scale_base + arith.index(ks_byte_off)
    num_loads = (reps + 3) // 4
    vecs = []
    for ld in range_constexpr(num_loads):
        off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
        vecs.append(_lds_load_b128(lds_buffer, off))
    results = []
    for i in range_constexpr(reps):
        vi = vector.extract(vecs[i // 4], static_position=[i % 4], dynamic_position=[])
        results.append(vi)
    return results


def _mxscale_load_preshuffled_b_frag(
    *,
    lds_buffer,
    b_lane_bases,
    wn: int,
    ks,
    is_fp4: bool,
    is_a8w4: bool,
    PACK_FACTOR_B: int,
    WMMA_K: int,
    _lds_load_b128,
    arith,
    vector,
):
    num_tiles = WMMA_K // PACK_FACTOR_B // 16
    k_subtile_off = arith.index(ks * num_tiles * 256)
    if is_fp4:
        base0 = b_lane_bases[wn * 2] + k_subtile_off
        base1 = b_lane_bases[wn * 2 + 1] + k_subtile_off
        v0 = _lds_load_b128(lds_buffer, base0)
        v1 = _lds_load_b128(lds_buffer, base0 + arith.index(512))
        v2 = _lds_load_b128(lds_buffer, base1)
        v3 = _lds_load_b128(lds_buffer, base1 + arith.index(512))
        v01 = vector.shuffle(v0, v1, list(range(8)))
        v23 = vector.shuffle(v2, v3, list(range(8)))
        return vector.shuffle(v01, v23, list(range(16)))
    base0 = b_lane_bases[wn] + k_subtile_off
    v0 = _lds_load_b128(lds_buffer, base0)
    v1 = _lds_load_b128(lds_buffer, base0 + arith.index(512))
    if is_a8w4:
        return vector.shuffle(v0, v1, list(range(8)))
    v2 = _lds_load_b128(lds_buffer, base0 + arith.index(1024))
    v3 = _lds_load_b128(lds_buffer, base0 + arith.index(1536))
    v01 = vector.shuffle(v0, v1, list(range(8)))
    v23 = vector.shuffle(v2, v3, list(range(8)))
    return vector.shuffle(v01, v23, list(range(16)))


def _mxscale_load_scale_i32(
    *,
    lds_buffer,
    scale_base,
    ks,
    SCALES_PER_WMMA: int,
    _lds_load_b128,
    llvm_dialect,
    ir,
    arith,
    T,
):
    from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
    from flydsl.expr.arith import _to_raw as _raw
    from flydsl.expr.arith import ArithValue as _AV

    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    raw_memref = arith.unwrap(lds_buffer)
    lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)
    byte_off = scale_base + arith.index(ks * SCALES_PER_WMMA)
    total_byte = _AV(lds_base) + byte_off
    addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
    ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
    return llvm_dialect.load(ir.IntegerType.get_signless(32), ptr_val)


def _mxscale_precompute_a_data_bases(
    *,
    warp_m_base,
    lane16,
    lane_kgrp,
    lds_a_stride_bytes: int,
    wmma_m_rep: int,
    WMMA_M: int,
    is_fp4: bool,
    arith,
    range_constexpr,
):
    row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
    k_half_off = lane_kgrp * arith.index(32 if is_fp4 else 16)
    return [
        row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
        for wm in range_constexpr(wmma_m_rep)
    ]


def _mxscale_precompute_rowmajor_b_data_bases(
    *,
    warp_n_base,
    lane16,
    lane_kgrp,
    lds_b_stride_bytes: int,
    wmma_n_rep: int,
    WMMA_N: int,
    arith,
    range_constexpr,
):
    return [
        (warp_n_base + lane16) * arith.index(lds_b_stride_bytes)
        + lane_kgrp * arith.index(32)
        + arith.index(wnh * WMMA_N * lds_b_stride_bytes)
        for wnh in range_constexpr(wmma_n_rep * 2)
    ]


def _mxscale_precompute_rowmajor_a_scale_lane_bases(
    *,
    warp_m_base,
    lane16,
    scale_k_per_tile: int,
    wmma_m_rep: int,
    WMMA_M: int,
    arith,
    range_constexpr,
):
    return [
        (warp_m_base + lane16) * arith.index(int(scale_k_per_tile))
        + arith.index(wm * WMMA_M * int(scale_k_per_tile))
        for wm in range_constexpr(wmma_m_rep)
    ]


def _mxscale_precompute_rowmajor_b_scale_lane_bases(
    *,
    warp_n_base,
    lane16,
    scale_k_per_tile: int,
    wmma_n_rep: int,
    WMMA_N: int,
    arith,
    range_constexpr,
):
    return [
        (warp_n_base + lane16) * arith.index(int(scale_k_per_tile))
        + arith.index(wn * WMMA_N * int(scale_k_per_tile))
        for wn in range_constexpr(wmma_n_rep * 2)
    ]


def _mxscale_load_data_frag(
    *,
    lds_buffer,
    lane_base,
    ks,
    PACK_FACTOR_A: int,
    WMMA_K: int,
    is_fp4: bool,
    _lds_load_b128,
    arith,
    vector,
):
    byte_off = lane_base + arith.index(ks * WMMA_K // PACK_FACTOR_A)
    v0 = _lds_load_b128(lds_buffer, byte_off)
    if is_fp4:
        v1 = _lds_load_b128(lds_buffer, byte_off + arith.index(16))
        return vector.shuffle(v0, v1, list(range(8)))
    v1 = _lds_load_b128(lds_buffer, byte_off + arith.index(32))
    v2 = _lds_load_b128(lds_buffer, byte_off + arith.index(64))
    v3 = _lds_load_b128(lds_buffer, byte_off + arith.index(96))
    v01 = vector.shuffle(v0, v1, list(range(8)))
    v23 = vector.shuffle(v2, v3, list(range(8)))
    return vector.shuffle(v01, v23, list(range(16)))


def _mxscale_load_rowmajor_b_frag(
    *,
    lds_buffer,
    b_lane_bases,
    wn: int,
    ks,
    PACK_FACTOR_B: int,
    WMMA_K: int,
    _lds_load_b128,
    arith,
    vector,
):
    k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR_B)
    base0 = b_lane_bases[wn * 2] + k_byte_off
    base1 = b_lane_bases[wn * 2 + 1] + k_byte_off
    v0 = _lds_load_b128(lds_buffer, base0)
    v1 = _lds_load_b128(lds_buffer, base0 + arith.index(16))
    v2 = _lds_load_b128(lds_buffer, base1)
    v3 = _lds_load_b128(lds_buffer, base1 + arith.index(16))
    v01 = vector.shuffle(v0, v1, list(range(8)))
    v23 = vector.shuffle(v2, v3, list(range(8)))
    return vector.shuffle(v01, v23, list(range(16)))


def _mxscale_emit_wmma(
    *,
    accs,
    wm: int,
    wn: int,
    a_frag,
    b_frags,
    a_scales,
    b_scales,
    is_fp4: bool,
    is_a8w4: bool,
    use_scale_opsel: bool,
    rocdl,
    T,
):
    idx = wm * len(b_frags) + wn
    if use_scale_opsel:
        a_scale_idx = wm // 2
        a_opsel = wm % 2
    else:
        a_scale_idx = wm
        a_opsel = 0

    if is_fp4:
        accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
            T.vec(16, T.f32),
            b_frags[wn], a_frag, accs[idx],
            b_scales[wn * 2], a_scales[a_scale_idx],
            scaleAType=0,
            scaleBType=a_opsel,
        )
        return

    if use_scale_opsel:
        b_scale_idx = wn // 2
        b_opsel = wn % 2
    else:
        b_scale_idx = wn
        b_opsel = 0
    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
        T.vec(8, T.f32),
        b_frags[wn], a_frag, accs[idx],
        b_scales[b_scale_idx], a_scales[a_scale_idx],
        fmtA=4 if is_a8w4 else 0,
        fmtB=0,
        scaleAType=b_opsel,
        scaleBType=a_opsel,
    )


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


def _make_mxscale_sub_tiles(
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


def _mxscale_extract_sub8(acc, vec_base: int, *, vector, range_constexpr, ACC_VEC_SIZE: int):
    if ACC_VEC_SIZE == 8:
        return acc
    return vector.shuffle(acc, acc, [vec_base + i for i in range_constexpr(8)])


def _finalize_alloc_and_launch_2d(*, ctx, alloc, launcher, gx, gy, block_threads: int, stream, waves_per_eu, ir):
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
    sorted_rsrc,
    tw_rsrc,
    out_rsrc,
    doweight_stage1: bool,
    out_elem_ty,
    load_gate_up_sub8,
    silu_fn,
    fx,
    arith,
    buffer_ops,
    scf,
    vector,
    range_constexpr,
    T,
):
    for acc_idx, vec_base, m_off, wn in sub_tiles:
        row_local = warp_m_base + fx.Index(m_off) + lane16
        sorted_row = by * arith.index(int(tile_m)) + row_local
        row_in_route = arith.cmpi(
            arith.CmpIPredicate.ult,
            arith.index_cast(T.i32, row_local),
            arith.constant(int(route_tile_m), type=T.i32),
        )
        sorted_safe = arith.select(
            row_in_route,
            arith.index_cast(T.i32, sorted_row),
            arith.index_cast(T.i32, by * arith.index(int(route_tile_m))),
        )
        fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
        slot = fused >> arith.constant(24, type=T.i32)
        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.constant(int(topk), type=T.i32))
        row_ok = arith.andi(row_in_route, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1)))
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
                out_idx = ((tok * arith.constant(int(topk), type=T.i32) + slot) * i32_inter_in
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


@functools.lru_cache(maxsize=64)
def _compile_stage1_dense_kernel_impl(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage1: bool,
    out_dtype: str,
    waves_per_eu: int | None,
):
    """Compile dense stage1 single kernel: route-pack + TDM + WMMA + epilog."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
    WAVE_SIZE = 32
    LDS_PAD_A = 8
    LDS_PAD_B = 8
    elem_bytes = 2

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"fp16 stage1 single kernel supports out_dtype in ('f16','bf16'), got {out_dtype!r}")
    if (int(model_dim) % int(tile_k)) != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if (int(tile_m) % WMMA_M) != 0 or (int(tile_n) % WMMA_N) != 0:
        raise ValueError(f"tile_m/tile_n must be multiples of 16, got ({tile_m},{tile_n})")

    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(f"Invalid warp tiling for fp16 single kernel: wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}")

    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = int(model_dim) // int(tile_k)
    k_wmma_steps = int(tile_k) // WMMA_K
    n_total = int(2 * inter_dim)
    _sub_tiles = _make_mxscale_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=False
    )

    lds_a_stride = int(tile_k) + LDS_PAD_A
    lds_b_stride = int(tile_n) + LDS_PAD_B
    lds_a_elems = int(tile_m) * lds_a_stride + LDS_PAD_A
    lds_b_elems = int(tile_k) * lds_b_stride + LDS_PAD_B

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name="moe_fp16_s1_single")
    off_bg = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bg + lds_b_elems * elem_bytes
    off_bu = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bu + lds_b_elems * elem_bytes
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_elems * elem_bytes

    @flyc.kernel
    def moe_fp16_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = (arg_scale_x, arg_scale_w, arg_max_token_ids, i32_k_in)
        llvm_dialect.inline_asm(
            None, [],  # void result, no operands
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # inter tile
        by = gpu.block_id("y")  # expert block

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        inter_idx = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)

        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_nbytes = tokens_idx * arith.index(int(model_dim)) * arith.index(2)
        w_nbytes = arith.index(int(experts * n_total * int(model_dim) * 2))

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        eid_ok = arith.andi(eid_ok0, eid_ok1)

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        base_ptr = alloc.get_base()
        smem_bg = SmemPtr(base_ptr, off_bg, T.f16, shape=(lds_b_elems,))
        smem_bu = SmemPtr(base_ptr, off_bu, T.f16, shape=(lds_b_elems,))
        smem_a = SmemPtr(base_ptr, off_a, T.f16, shape=(lds_a_elems,))
        lds_bg = get_op_result_or_value(smem_bg.get())
        lds_bu = get_op_result_or_value(smem_bu.get())
        lds_a = get_op_result_or_value(smem_a.get())

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        def pack_a_to_lds(k_base):
            total = int(tile_m * tile_k)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(tile_k))
                    col = elem % arith.index(int(tile_k))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, row),
                        arith.constant(int(route_tile_m), type=T.i32),
                    )
                    sorted_row_safe = arith.select(
                        row_in_route,
                        arith.index_cast(T.i32, sorted_row),
                        arith.index_cast(T.i32, by * arith.index(int(route_tile_m))),
                    )
                    fused = buffer_ops.buffer_load(sorted_rsrc, sorted_row_safe, vec_width=1, dtype=T.i32)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    tok_ok0 = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    tok_ok = arith.andi(row_in_route, tok_ok0)
                    x_idx = tok * arith.constant(int(model_dim), type=T.i32) + arith.index_cast(T.i32, k_base + col)
                    x_idx_safe = arith.select(tok_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(
                        tok_ok,
                        buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.f16),
                        arith.constant(0.0, type=T.f16),
                    )
                    lds_idx = row * arith.index(lds_a_stride) + col
                    v1 = vector.from_elements(T.vec(1, T.f16), [x_val])
                    vector.store(v1, lds_a, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def copy_b_to_lds(k_base, lds_memref, up_shift):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_base = eid_idx * arith.index(n_total) + blk_n + arith.index(up_shift)
            total = int(tile_k) * int(tile_n)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    k_local = elem // arith.index(int(tile_n))
                    n_local = elem % arith.index(int(tile_n))
                    w_idx = (n_base + n_local) * arith.index(int(model_dim)) + k_base + k_local
                    w_val = buffer_ops.buffer_load(
                        w_rsrc, arith.index_cast(T.i32, w_idx),
                        vec_width=1, dtype=T.f16,
                    )
                    lds_idx = k_local * arith.index(lds_b_stride) + n_local
                    v1 = vector.from_elements(T.vec(1, T.f16), [w_val])
                    vector.store(v1, lds_memref, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def _precompute_a_lane_bases():
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = row_stride_off + arith.index(wm * WMMA_M * lds_a_stride) + k_lane_off
                bases.append(a_base)
            return bases

        def _precompute_b_lane_bases():
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(lds_b_stride)
            n_lane_off = lane_ngrp * arith.index(8)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = warp_n_base + arith.index(wn * WMMA_N) + n_lane_off
                bases.append(k_lane_off + n_col)
            return bases

        def load_a_frag(a_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            off0 = a_base + arith.index(ks * WMMA_K)
            off1 = a_base + arith.index(ks * WMMA_K + 16)
            v0 = vector.load_op(vec8_ty, lds_a, [off0])
            v1 = vector.load_op(vec8_ty, lds_a, [off1])
            return vector.shuffle(v0, v1, list(range(16)))

        def load_b_frag(lds_buf, b_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride
                elem_off = b_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_buf, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        acc_gate = [acc_zero] * n_accs
        acc_up = [acc_zero] * n_accs

        _if_eid = scf.IfOp(eid_ok)
        with ir.InsertionPoint(_if_eid.then_block):
            a_bases = _precompute_a_lane_bases()
            b_bases = _precompute_b_lane_bases()
            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                pack_a_to_lds(k_base)
                copy_b_to_lds(k_base, lds_bg, 0)
                copy_b_to_lds(k_base, lds_bu, int(inter_dim))
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    b_gate_frags = [load_b_frag(lds_bg, b_bases[wn], ks) for wn in range_constexpr(wmma_n_rep)]
                    b_up_frags = [load_b_frag(lds_bu, b_bases[wn], ks) for wn in range_constexpr(wmma_n_rep)]
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_a_frag(a_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            idx = wm * wmma_n_rep + wn
                            acc_gate[idx] = rocdl.wmma_f32_16x16x32_f16(
                                T.vec(8, T.f32),
                                b_gate_frags[wn],
                                a_frag,
                                acc_gate[idx],
                                signA=False,
                                signB=False,
                                modC=0,
                                reuseA=False,
                                reuseB=False,
                            ).result
                            acc_up[idx] = rocdl.wmma_f32_16x16x32_f16(
                                T.vec(8, T.f32),
                                b_up_frags[wn],
                                a_frag,
                                acc_up[idx],
                                signA=False,
                                signB=False,
                                modC=0,
                                reuseA=False,
                                reuseB=False,
                            ).result
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_gate_up_sub8(acc_idx, _vec_base):
                return acc_gate[acc_idx], acc_up[acc_idx]

            _emit_stage1_gate_up_epilogue(
                sub_tiles=_sub_tiles,
                by=by,
                tile_m=int(tile_m),
                route_tile_m=int(route_tile_m),
                warp_m_base=warp_m_base,
                warp_n_base=warp_n_base,
                blk_n=blk_n,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                WMMA_N=WMMA_N,
                i32_tokens_in=i32_tokens_in,
                i32_inter_in=i32_inter_in,
                topk=int(topk),
                sorted_rsrc=sorted_rsrc,
                tw_rsrc=sw_rsrc,
                out_rsrc=out_rsrc,
                doweight_stage1=bool(doweight_stage1),
                out_elem_ty=out_elem_ty,
                load_gate_up_sub8=_load_gate_up_sub8,
                silu_fn=silu,
                fx=fx,
                arith=arith,
                buffer_ops=buffer_ops,
                scf=scf,
                vector=vector,
                range_constexpr=range_constexpr,
                T=T,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_fp16_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (inter_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_fp16_stage1_single(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        )
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=waves_per_eu,
            ir=ir,
        )

    return launch_fp16_stage1_single


@functools.lru_cache(maxsize=64)
def _compile_stage2_dense_kernel_impl(
    *,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage2: bool,
    out_dtype: str,
    accumulate: bool,
    waves_per_eu: int | None,
):
    """Compile fp16 stage2 single kernel: route-pack + TDM + WMMA + epilog."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
    WAVE_SIZE = 32
    LDS_PAD_A = 8
    LDS_PAD_B = 8
    elem_bytes = 2

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"fp16 stage2 single kernel supports out_dtype in ('f16','bf16'), got {out_dtype!r}")
    if (int(inter_dim) % int(tile_k)) != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")

    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(f"Invalid warp tiling for fp16 stage2 single kernel: wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}")

    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = int(inter_dim) // int(tile_k)
    k_wmma_steps = int(tile_k) // WMMA_K
    _sub_tiles = _make_mxscale_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=False
    )

    lds_a_stride = int(tile_k) + LDS_PAD_A
    lds_b_stride = int(tile_n) + LDS_PAD_B
    lds_a_elems = int(tile_m) * lds_a_stride + LDS_PAD_A
    lds_b_elems = int(tile_k) * lds_b_stride + LDS_PAD_B

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name="moe_fp16_s2_single")
    off_b = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_b + lds_b_elems * elem_bytes
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_elems * elem_bytes

    @flyc.kernel
    def moe_fp16_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = (arg_scale_x, arg_scale_w, i32_k_in)
        llvm_dialect.inline_asm(
            None, [],  # void result, no operands
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # n tile
        by = gpu.block_id("y")  # expert block

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        n_idx = arith.index_cast(T.index, i32_n_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)
        num_valid_i32 = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=True),
            arith.constant(0, type=T.i32),
            vec_width=1,
            dtype=T.i32,
        )

        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_rows = tokens_idx * arith.index(int(topk))
        x_nbytes = x_rows * arith.index(int(inter_dim)) * arith.index(2)
        out_nbytes = tokens_idx * n_idx * arith.index(2)
        if not bool(accumulate):
            out_nbytes = x_rows * n_idx * arith.index(2)

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False, num_records_bytes=out_nbytes)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, block_row_start, num_valid_i32)
        block_ok = arith.andi(block_in_valid, arith.andi(eid_ok0, eid_ok1))

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        base_ptr = alloc.get_base()
        smem_b = SmemPtr(base_ptr, off_b, T.f16, shape=(lds_b_elems,))
        smem_a = SmemPtr(base_ptr, off_a, T.f16, shape=(lds_a_elems,))
        lds_b = get_op_result_or_value(smem_b.get())
        lds_a = get_op_result_or_value(smem_a.get())

        def pack_a_to_lds(k_base):
            total = int(tile_m * tile_k)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(tile_k))
                    col = elem % arith.index(int(tile_k))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_i32 = arith.index_cast(T.i32, row)
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
                    slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.constant(int(topk), type=T.i32))
                    ts = tok * arith.constant(int(topk), type=T.i32) + slot
                    ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                    load_ok = arith.andi(row_ok, ts_ok)
                    x_idx = ts * arith.constant(int(inter_dim), type=T.i32) + arith.index_cast(T.i32, k_base + col)
                    x_idx_safe = arith.select(load_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(
                        load_ok,
                        buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.f16),
                        arith.constant(0.0, type=T.f16),
                    )
                    lds_idx = row * arith.index(lds_a_stride) + col
                    v1 = vector.from_elements(T.vec(1, T.f16), [x_val])
                    vector.store(v1, lds_a, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def copy_b_to_lds(k_base):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_base = eid_idx * n_idx + blk_n
            total = int(tile_k) * int(tile_n)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    k_local = elem // arith.index(int(tile_n))
                    n_local = elem % arith.index(int(tile_n))
                    w_idx = (n_base + n_local) * arith.index(int(inter_dim)) + k_base + k_local
                    w_val = buffer_ops.buffer_load(
                        w_rsrc, arith.index_cast(T.i32, w_idx),
                        vec_width=1, dtype=T.f16,
                    )
                    lds_idx = k_local * arith.index(lds_b_stride) + n_local
                    v1 = vector.from_elements(T.vec(1, T.f16), [w_val])
                    vector.store(v1, lds_b, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def _precompute_a_lane_bases():
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = row_stride_off + arith.index(wm * WMMA_M * lds_a_stride) + k_lane_off
                bases.append(a_base)
            return bases

        def _precompute_b_lane_bases():
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(lds_b_stride)
            n_lane_off = lane_ngrp * arith.index(8)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = warp_n_base + arith.index(wn * WMMA_N) + n_lane_off
                bases.append(k_lane_off + n_col)
            return bases

        def load_a_frag(a_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            off0 = a_base + arith.index(ks * WMMA_K)
            off1 = a_base + arith.index(ks * WMMA_K + 16)
            v0 = vector.load_op(vec8_ty, lds_a, [off0])
            v1 = vector.load_op(vec8_ty, lds_a, [off1])
            return vector.shuffle(v0, v1, list(range(16)))

        def load_b_frag(b_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride
                elem_off = b_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_b, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        acc = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            a_bases = _precompute_a_lane_bases()
            b_bases = _precompute_b_lane_bases()

            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                pack_a_to_lds(k_base)
                copy_b_to_lds(k_base)
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    b_frags = [load_b_frag(b_bases[wn], ks) for wn in range_constexpr(wmma_n_rep)]
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_a_frag(a_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            idx = wm * wmma_n_rep + wn
                            acc[idx] = rocdl.wmma_f32_16x16x32_f16(
                                T.vec(8, T.f32),
                                b_frags[wn],
                                a_frag,
                                acc[idx],
                                signA=False,
                                signB=False,
                                modC=0,
                                reuseA=False,
                                reuseB=False,
                            ).result
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_sub8(acc_idx, _vec_base):
                return acc[acc_idx]

            _emit_stage2_store_epilogue(
                sub_tiles=_sub_tiles,
                by=by,
                tile_m=int(tile_m),
                route_tile_m=int(route_tile_m),
                warp_m_base=warp_m_base,
                warp_n_base=warp_n_base,
                blk_n=blk_n,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                WMMA_N=WMMA_N,
                i32_tokens_in=i32_tokens_in,
                i32_n_in=i32_n_in,
                topk=int(topk),
                num_valid_i32=num_valid_i32,
                block_row_start=block_row_start,
                sorted_rsrc=sorted_rsrc,
                tw_rsrc=sw_rsrc,
                out_rsrc=out_rsrc,
                doweight_stage2=bool(doweight_stage2),
                accumulate=bool(accumulate),
                out_elem_ty=out_elem_ty,
                load_sub8=_load_sub8,
                fx=fx,
                arith=arith,
                buffer_ops=buffer_ops,
                scf=scf,
                vector=vector,
                range_constexpr=range_constexpr,
                rocdl=rocdl,
                T=T,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_fp16_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (n_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_fp16_stage2_single(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        )
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=waves_per_eu,
            ir=ir,
        )

    return launch_fp16_stage2_single


@functools.lru_cache(maxsize=64)
def _compile_stage1_mxscale_kernel_impl(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage1: bool,
    out_dtype: str,
    waves_per_eu: int | None,
    data_format: str = "fp8",
):
    """Compile mxscale stage1 single kernel (route-pack + TDM + WMMA_SCALE + epilog)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    fmt_cfg = _mxscale_format_config(data_format)
    is_fp4 = bool(fmt_cfg["is_fp4"])
    is_a8w4 = bool(fmt_cfg["is_a8w4"])
    PACK_FACTOR_A = int(fmt_cfg["PACK_FACTOR_A"])
    PACK_FACTOR_B = int(fmt_cfg["PACK_FACTOR_B"])
    ACC_VEC_SIZE = int(fmt_cfg["ACC_VEC_SIZE"])
    WMMA_N_EFF = int(fmt_cfg["WMMA_N_EFF"])

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
    SCALE_BLOCK = 32
    SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK
    WAVE_SIZE = 32
    LDS_PAD_A_BYTES = 16
    LDS_PAD_B_BYTES = 16 if is_fp4 else 0

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"mxscale stage1 single kernel supports out_dtype in ('f16','bf16'), got {out_dtype!r}")
    if (int(model_dim) % int(tile_k)) != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if (int(tile_k) % SCALE_BLOCK) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {SCALE_BLOCK}")

    K = int(model_dim)
    N = int(inter_dim)
    K_packed_a = K // PACK_FACTOR_A
    K_packed_b = K // PACK_FACTOR_B
    packed_tile_k_a = int(tile_k) // PACK_FACTOR_A
    packed_tile_k_b = int(tile_k) // PACK_FACTOR_B
    K_scale = K // SCALE_BLOCK
    scale_k_per_tile = int(tile_k) // SCALE_BLOCK
    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    k_wmma_steps = int(tile_k) // WMMA_K
    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = K // int(tile_k)
    b_scale_load_rep = (wmma_n_rep * 2) if is_fp4 else wmma_n_rep
    interleaved_scale_cols_b = b_scale_load_rep * scale_k_per_tile

    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(f"Invalid warp tiling for mxscale stage1 single kernel: wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}")

    lds_a_stride_bytes = int(packed_tile_k_a) + LDS_PAD_A_BYTES
    lds_b_stride_bytes = int(packed_tile_k_b) + LDS_PAD_B_BYTES
    lds_a_data_bytes = int(tile_m) * lds_a_stride_bytes
    lds_b_data_bytes = int(tile_n) * lds_b_stride_bytes
    lds_a_scale_bytes = int(tile_m) * scale_k_per_tile
    lds_b_scale_bytes = int(tile_n) * scale_k_per_tile
    interleaved_scale_cols_a = wmma_m_rep * scale_k_per_tile
    _sub_tiles = _make_mxscale_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=is_fp4
    )

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name=f"moe_mxscale_{data_format}_s1_single")
    off_ag = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_ag + lds_a_data_bytes
    off_bg = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bg + lds_b_data_bytes
    off_as = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_as + lds_a_scale_bytes
    off_bs = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bs + lds_b_scale_bytes
    off_bu = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bu + lds_b_data_bytes
    off_bsu = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bsu + lds_b_scale_bytes

    @flyc.kernel
    def moe_mxscale_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = (arg_max_token_ids, i32_k_in)
        llvm_dialect.inline_asm(
            None, [],
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)
        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_nbytes = tokens_idx * arith.index(K_packed_a)
        sx_nbytes = tokens_idx * arith.index(K_scale)
        w_rows = arith.index(int(experts * (2 * N)))
        w_nbytes = w_rows * arith.index(K_packed_b)
        sw_nbytes = w_rows * arith.index(K_scale)

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False, num_records_bytes=sx_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False, num_records_bytes=sw_nbytes)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        tw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        block_ok = arith.andi(eid_ok0, eid_ok1)

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        base_ptr = alloc.get_base()
        smem_ag = SmemPtr(base_ptr, off_ag, T.i8, shape=(lds_a_data_bytes,))
        smem_bg = SmemPtr(base_ptr, off_bg, T.i8, shape=(lds_b_data_bytes,))
        smem_as = SmemPtr(base_ptr, off_as, T.i8, shape=(lds_a_scale_bytes,))
        smem_bs = SmemPtr(base_ptr, off_bs, T.i8, shape=(lds_b_scale_bytes,))
        smem_bu = SmemPtr(base_ptr, off_bu, T.i8, shape=(lds_b_data_bytes,))
        smem_bsu = SmemPtr(base_ptr, off_bsu, T.i8, shape=(lds_b_scale_bytes,))
        lds_ag = get_op_result_or_value(smem_ag.get())
        lds_bg = get_op_result_or_value(smem_bg.get())
        lds_as = get_op_result_or_value(smem_as.get())
        lds_bs = get_op_result_or_value(smem_bs.get())
        lds_bu = get_op_result_or_value(smem_bu.get())
        lds_bsu = get_op_result_or_value(smem_bsu.get())

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        def make_desc_a(k_base):
            return k_base / arith.index(PACK_FACTOR_A)

        def issue_a_load(k_packed_base):
            total = int(tile_m * packed_tile_k_a)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, elem), arith.constant(total, type=T.i32))
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(packed_tile_k_a))
                    col = elem % arith.index(int(packed_tile_k_a))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_in_route = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, row), arith.constant(int(route_tile_m), type=T.i32))
                    sorted_safe = arith.select(
                        row_in_route,
                        arith.index_cast(T.i32, sorted_row),
                        arith.index_cast(T.i32, by * arith.index(int(route_tile_m))),
                    )
                    fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    ok = arith.andi(row_in_route, tok_ok)
                    x_idx = tok * arith.constant(K_packed_a, type=T.i32) + arith.index_cast(T.i32, k_packed_base + col)
                    x_idx_safe = arith.select(ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(ok, buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.i8), arith.constant(0, type=T.i8))
                    lds_idx = row * arith.index(lds_a_stride_bytes) + col
                    v1 = vector.from_elements(T.vec(1, T.i8), [x_val])
                    vector.store(v1, lds_ag, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def make_desc_as(k_base):
            return k_base / arith.index(SCALE_BLOCK)

        def issue_as_load(k_scale_base):
            total = int(tile_m * scale_k_per_tile)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, elem), arith.constant(total, type=T.i32))
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(scale_k_per_tile))
                    ksc = elem % arith.index(int(scale_k_per_tile))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_in_route = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, row), arith.constant(int(route_tile_m), type=T.i32))
                    sorted_safe = arith.select(
                        row_in_route,
                        arith.index_cast(T.i32, sorted_row),
                        arith.index_cast(T.i32, by * arith.index(int(route_tile_m))),
                    )
                    fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    ok = arith.andi(row_in_route, tok_ok)
                    ksc_off = k_scale_base + ksc
                    sx_idx = tok * arith.constant(K_scale, type=T.i32) + arith.index_cast(T.i32, ksc_off)
                    sx_idx_safe = arith.select(ok, sx_idx, arith.constant(0, type=T.i32))
                    sx_val = arith.select(ok, buffer_ops.buffer_load(sx_rsrc, sx_idx_safe, vec_width=1, dtype=T.i8), arith.constant(127, type=T.i8))
                    if is_fp4:
                        lds_idx = row * arith.index(int(scale_k_per_tile)) + ksc
                    else:
                        warp_row_idx = row / arith.index(warp_tile_m)
                        local_row = row % arith.index(warp_tile_m)
                        lane_row = local_row % arith.index(WMMA_M)
                        local_wm_idx = local_row / arith.index(WMMA_M)
                        global_lds_row = warp_row_idx * arith.index(WMMA_M) + lane_row
                        ksc_blk = ksc / arith.index(SCALES_PER_WMMA)
                        ksc_sub = ksc % arith.index(SCALES_PER_WMMA)
                        lds_idx = (
                            global_lds_row * arith.index(interleaved_scale_cols_a)
                            + ksc_blk * arith.index(wmma_m_rep * SCALES_PER_WMMA)
                            + local_wm_idx * arith.index(SCALES_PER_WMMA)
                            + ksc_sub
                        )
                    v1 = vector.from_elements(T.vec(1, T.i8), [sx_val])
                    vector.store(v1, lds_as, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def make_desc_b(lds_b_mem, n_off, k_base):
            if is_fp4:
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_w, lds_memref=lds_b_mem,
                    global_offset=(n_off, k_base / arith.index(PACK_FACTOR_B)),
                    tensor_shape=(int(tile_n), int(packed_tile_k_b)),
                    strides=(K_packed_b, 1),
                    tile_shape=(int(tile_n), int(packed_tile_k_b)),
                    elem_bytes=1, pad_interval=int(packed_tile_k_b), pad_amount=LDS_PAD_B_BYTES,
                    num_warps=int(m_warp) * int(n_warp), workgroup_mask=0)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(experts * (2 * N) // 16), int(K_packed_b * 16)),
                strides=(K_packed_b * 16, 1),
                tile_shape=(int(tile_n // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=int(m_warp) * int(n_warp),
                workgroup_mask=0)

        def make_desc_bs(lds_bs_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=lds_bs_mem,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=int(m_warp) * int(n_warp), workgroup_mask=0)

        def issue_bs_load(k_base, lds_bs_mem, n_off):
            if is_fp4:
                tdm_ops.tensor_load_2d(make_desc_bs(lds_bs_mem, n_off, k_base))
                return
            total_bs = int(tile_n * scale_k_per_tile)
            rounds_bs = (total_bs + block_threads - 1) // block_threads
            for it in range(rounds_bs):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total_bs, type=T.i32),
                )
                _if_bs = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_bs.then_block):
                    n_local = elem // arith.index(int(scale_k_per_tile))
                    ksc = elem % arith.index(int(scale_k_per_tile))
                    k_scale_base = k_base / arith.index(SCALE_BLOCK)
                    sw_idx = arith.index_cast(T.i32,
                        (n_off + n_local) * arith.index(K_scale) + k_scale_base + ksc)
                    sw_val = buffer_ops.buffer_load(
                        sw_rsrc, sw_idx, vec_width=1, dtype=T.i8)
                    wave_n = n_local / arith.index(warp_tile_n)
                    sub_n = n_local % arith.index(warp_tile_n)
                    wn_idx = sub_n / arith.index(WMMA_N)
                    lane_row = sub_n % arith.index(WMMA_N)
                    ksc_blk = ksc / arith.index(SCALES_PER_WMMA)
                    ksc_sub = ksc % arith.index(SCALES_PER_WMMA)
                    warp_lds_row = wave_n * arith.index(WMMA_N) + lane_row
                    lds_idx = (
                        warp_lds_row * arith.index(interleaved_scale_cols_b)
                        + ksc_blk * arith.index(b_scale_load_rep * SCALES_PER_WMMA)
                        + wn_idx * arith.index(SCALES_PER_WMMA)
                        + ksc_sub
                    )
                    v1 = vector.from_elements(T.vec(1, T.i8), [sw_val])
                    vector.store(v1, lds_bs_mem, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def issue_b_load(k_base, lds_b_mem, lds_bs_mem, up_shift):
            eid_row = arith.index_cast(T.index, eid_i32) * arith.index(int(2 * N))
            n_off = eid_row + blk_n + arith.index(up_shift)
            tdm_ops.tensor_load_2d(make_desc_b(lds_b_mem, n_off, k_base))
            issue_bs_load(k_base, lds_bs_mem, n_off)

        def _lds_load_b128(lds_buffer, byte_offset):
            from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
            from flydsl.expr.arith import _to_raw as _raw
            from flydsl.expr.arith import ArithValue as _AV
            lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
            raw_memref = arith.unwrap(lds_buffer)
            lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)
            total_byte = _AV(lds_base) + byte_offset
            addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
            ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
            vec4_i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
            return llvm_dialect.load(vec4_i32_ty, ptr_val)

        def load_data_frag(lds_buffer, lane_base, ks):
            return _mxscale_load_data_frag(
                lds_buffer=lds_buffer,
                lane_base=lane_base,
                ks=ks,
                PACK_FACTOR_A=PACK_FACTOR_A,
                WMMA_K=WMMA_K,
                is_fp4=is_fp4,
                _lds_load_b128=_lds_load_b128,
                arith=arith,
                vector=vector,
            )

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            if is_fp4:
                return _mxscale_load_rowmajor_b_frag(
                    lds_buffer=lds_buffer,
                    b_lane_bases=b_lane_bases,
                    wn=wn,
                    ks=ks,
                    PACK_FACTOR_B=PACK_FACTOR_B,
                    WMMA_K=WMMA_K,
                    _lds_load_b128=_lds_load_b128,
                    arith=arith,
                    vector=vector,
                )
            if is_a8w4:
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn] + k_subtile_off
                v0 = _lds_load_b128(lds_buffer, base0)
                v1 = _lds_load_b128(lds_buffer, base0 + arith.index(512))
                return vector.shuffle(v0, v1, list(range(8)))
            _num_tiles = WMMA_K // PACK_FACTOR_B // 16
            k_subtile_off = arith.index(ks * _num_tiles * 256)
            base0 = b_lane_bases[wn] + k_subtile_off
            v0 = _lds_load_b128(lds_buffer, base0)
            v1 = _lds_load_b128(lds_buffer, base0 + arith.index(512))
            v2 = _lds_load_b128(lds_buffer, base0 + arith.index(1024))
            v3 = _lds_load_b128(lds_buffer, base0 + arith.index(1536))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def load_scale_i32(lds_buffer, scale_base, ks):
            return _mxscale_load_scale_i32(
                lds_buffer=lds_buffer,
                scale_base=scale_base,
                ks=ks,
                SCALES_PER_WMMA=SCALES_PER_WMMA,
                _lds_load_b128=_lds_load_b128,
                llvm_dialect=llvm_dialect,
                ir=ir,
                arith=arith,
                T=T,
            )

        def _precompute_a_data_bases():
            return _mxscale_precompute_a_data_bases(
                warp_m_base=warp_m_base,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                lds_a_stride_bytes=lds_a_stride_bytes,
                wmma_m_rep=wmma_m_rep,
                WMMA_M=WMMA_M,
                is_fp4=is_fp4,
                arith=arith,
                range_constexpr=range_constexpr,
            )

        def _precompute_b_data_bases():
            if is_fp4:
                return _mxscale_precompute_rowmajor_b_data_bases(
                    warp_n_base=warp_n_base,
                    lane16=lane16,
                    lane_kgrp=lane_kgrp,
                    lds_b_stride_bytes=lds_b_stride_bytes,
                    wmma_n_rep=wmma_n_rep,
                    WMMA_N=WMMA_N,
                    arith=arith,
                    range_constexpr=range_constexpr,
                )
            return _mxscale_precompute_preshuffled_b_data_bases(
                packed_tile_k_b=packed_tile_k_b,
                warp_tile_n=warp_tile_n,
                wave_n_idx=wave_n_idx,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                wmma_n_rep=wmma_n_rep,
                arith=arith,
                range_constexpr=range_constexpr,
            )

        def _precompute_a_scale_lane_bases():
            if is_fp4:
                return _mxscale_precompute_rowmajor_a_scale_lane_bases(
                    warp_m_base=warp_m_base,
                    lane16=lane16,
                    scale_k_per_tile=scale_k_per_tile,
                    wmma_m_rep=wmma_m_rep,
                    WMMA_M=WMMA_M,
                    arith=arith,
                    range_constexpr=range_constexpr,
                )
            return _mxscale_precompute_a_scale_lane_bases(
                warp_m_base=warp_m_base,
                lane16=lane16,
                wmma_m_rep=wmma_m_rep,
                interleaved_scale_cols_a=interleaved_scale_cols_a,
                arith=arith,
            )

        def _precompute_b_scale_lane_bases():
            if is_fp4:
                return _mxscale_precompute_rowmajor_b_scale_lane_bases(
                    warp_n_base=warp_n_base,
                    lane16=lane16,
                    scale_k_per_tile=scale_k_per_tile,
                    wmma_n_rep=wmma_n_rep,
                    WMMA_N=WMMA_N,
                    arith=arith,
                    range_constexpr=range_constexpr,
                )
            return _mxscale_precompute_b_scale_lane_bases(
                warp_n_base=warp_n_base,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                b_scale_load_rep=b_scale_load_rep,
                interleaved_scale_cols_b=interleaved_scale_cols_b,
                is_fp4=is_fp4,
                is_a8w4=is_a8w4,
                arith=arith,
                SCALES_PER_WMMA=SCALES_PER_WMMA,
            )

        def load_scale_b128(lds_buffer, scale_base, reps, ks=0):
            return _mxscale_load_scale_b128(
                lds_buffer=lds_buffer,
                scale_base=scale_base,
                reps=reps,
                ks=ks,
                SCALES_PER_WMMA=SCALES_PER_WMMA,
                _lds_load_b128=_lds_load_b128,
                arith=arith,
                vector=vector,
                range_constexpr=range_constexpr,
            )

        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        acc_g = [acc_zero] * n_accs
        acc_u = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            a_data_bases = _precompute_a_data_bases()
            b_data_bases = _precompute_b_data_bases()
            as_bases = _precompute_a_scale_lane_bases()
            bs_bases = _precompute_b_scale_lane_bases()
            bsu_bases = _precompute_b_scale_lane_bases()

            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                a_desc = make_desc_a(k_base)
                as_desc = make_desc_as(k_base)
                issue_a_load(a_desc)
                issue_as_load(as_desc)
                issue_b_load(k_base, lds_bg, lds_bs, 0)
                issue_b_load(k_base, lds_bu, lds_bsu, int(N))
                tdm_ops.tensor_wait(0)
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    b_g = [load_b_frag(lds_bg, b_data_bases, wn, ks) for wn in range_constexpr(wmma_n_rep)]
                    b_u = [load_b_frag(lds_bu, b_data_bases, wn, ks) for wn in range_constexpr(wmma_n_rep)]
                    if is_fp4:
                        as_v = [load_scale_i32(lds_as, as_bases[wm], ks) for wm in range_constexpr(wmma_m_rep)]
                        bs_g = [load_scale_i32(lds_bs, bs_bases[bi], ks) for bi in range_constexpr(b_scale_load_rep)]
                        bs_u = [load_scale_i32(lds_bsu, bsu_bases[bi], ks) for bi in range_constexpr(b_scale_load_rep)]
                    else:
                        as_v = load_scale_b128(lds_as, as_bases[0], wmma_m_rep, ks)
                        bs_g = load_scale_b128(lds_bs, bs_bases[0], b_scale_load_rep, ks)
                        bs_u = load_scale_b128(lds_bsu, bsu_bases[0], b_scale_load_rep, ks)
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_data_frag(lds_ag, a_data_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            _mxscale_emit_wmma(
                                accs=acc_g,
                                wm=wm,
                                wn=wn,
                                a_frag=a_frag,
                                b_frags=b_g,
                                a_scales=as_v,
                                b_scales=bs_g,
                                is_fp4=is_fp4,
                                is_a8w4=is_a8w4,
                                use_scale_opsel=False,
                                rocdl=rocdl,
                                T=T,
                            )
                            _mxscale_emit_wmma(
                                accs=acc_u,
                                wm=wm,
                                wn=wn,
                                a_frag=a_frag,
                                b_frags=b_u,
                                a_scales=as_v,
                                b_scales=bs_u,
                                is_fp4=is_fp4,
                                is_a8w4=is_a8w4,
                                use_scale_opsel=False,
                                rocdl=rocdl,
                                T=T,
                            )
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_gate_up_sub8(acc_idx, vec_base):
                return (
                    _mxscale_extract_sub8(
                        acc_g[acc_idx], vec_base, vector=vector, range_constexpr=range_constexpr, ACC_VEC_SIZE=ACC_VEC_SIZE
                    ),
                    _mxscale_extract_sub8(
                        acc_u[acc_idx], vec_base, vector=vector, range_constexpr=range_constexpr, ACC_VEC_SIZE=ACC_VEC_SIZE
                    ),
                )

            _emit_stage1_gate_up_epilogue(
                sub_tiles=_sub_tiles,
                by=by,
                tile_m=int(tile_m),
                route_tile_m=int(route_tile_m),
                warp_m_base=warp_m_base,
                warp_n_base=warp_n_base,
                blk_n=blk_n,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                WMMA_N=WMMA_N,
                i32_tokens_in=i32_tokens_in,
                i32_inter_in=i32_inter_in,
                topk=int(topk),
                sorted_rsrc=sorted_rsrc,
                tw_rsrc=tw_rsrc,
                out_rsrc=out_rsrc,
                doweight_stage1=bool(doweight_stage1),
                out_elem_ty=out_elem_ty,
                load_gate_up_sub8=_load_gate_up_sub8,
                silu_fn=silu,
                fx=fx,
                arith=arith,
                buffer_ops=buffer_ops,
                scf=scf,
                vector=vector,
                range_constexpr=range_constexpr,
                T=T,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_mxscale_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (inter_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_mxscale_stage1_single(
            arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights, arg_max_token_ids,
            i32_tokens_in, i32_inter_in, i32_k_in, i32_size_expert_ids_in,
        )
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=waves_per_eu,
            ir=ir,
        )

    return launch_mxscale_stage1_single


@functools.lru_cache(maxsize=64)
def _compile_stage2_mxscale_kernel_impl(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage2: bool,
    out_dtype: str,
    accumulate: bool,
    waves_per_eu: int | None,
    data_format: str = "fp8",
):
    """Compile mxscale stage2 single kernel (route-pack + TDM + WMMA_SCALE + epilog)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    fmt_cfg = _mxscale_format_config(data_format)
    is_fp4 = bool(fmt_cfg["is_fp4"])
    is_a8w4 = bool(fmt_cfg["is_a8w4"])
    PACK_FACTOR_A = int(fmt_cfg["PACK_FACTOR_A"])
    PACK_FACTOR_B = int(fmt_cfg["PACK_FACTOR_B"])
    ACC_VEC_SIZE = int(fmt_cfg["ACC_VEC_SIZE"])
    WMMA_N_EFF = int(fmt_cfg["WMMA_N_EFF"])

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
    SCALE_BLOCK = 32
    SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK
    WAVE_SIZE = 32
    LDS_PAD_A_BYTES = 16
    LDS_PAD_B_BYTES = 16 if is_fp4 else 0

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"mxscale stage2 single kernel supports out_dtype in ('f16','bf16'), got {out_dtype!r}")
    if (int(inter_dim) % int(tile_k)) != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if (int(tile_k) % SCALE_BLOCK) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {SCALE_BLOCK}")

    K = int(inter_dim)
    N_total = int(model_dim)
    K_packed_a = K // PACK_FACTOR_A
    K_packed_b = K // PACK_FACTOR_B
    packed_tile_k_a = int(tile_k) // PACK_FACTOR_A
    packed_tile_k_b = int(tile_k) // PACK_FACTOR_B
    K_scale = K // SCALE_BLOCK
    scale_k_per_tile = int(tile_k) // SCALE_BLOCK
    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    k_wmma_steps = int(tile_k) // WMMA_K
    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = K // int(tile_k)
    b_scale_load_rep = (wmma_n_rep * 2) if is_fp4 else wmma_n_rep
    interleaved_scale_cols_b = b_scale_load_rep * scale_k_per_tile

    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(f"Invalid warp tiling for mxscale stage2 single kernel: wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}")

    lds_a_stride_bytes = int(packed_tile_k_a) + LDS_PAD_A_BYTES
    lds_b_stride_bytes = int(packed_tile_k_b) + LDS_PAD_B_BYTES
    lds_a_data_bytes = int(tile_m) * lds_a_stride_bytes
    lds_b_data_bytes = int(tile_n) * lds_b_stride_bytes
    lds_a_scale_bytes = int(tile_m) * scale_k_per_tile
    lds_b_scale_bytes = int(tile_n) * scale_k_per_tile
    interleaved_scale_cols_a = wmma_m_rep * scale_k_per_tile

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name=f"moe_mxscale_{data_format}_s2_single")
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_data_bytes
    off_b = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_b + lds_b_data_bytes
    off_as = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_as + lds_a_scale_bytes
    off_bs = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bs + lds_b_scale_bytes

    _sub_tiles = _make_mxscale_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=is_fp4
    )

    @flyc.kernel
    def moe_mxscale_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = i32_k_in
        llvm_dialect.inline_asm(
            None, [],
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        n_idx = arith.index_cast(T.index, i32_n_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)
        c_topk_i32 = arith.constant(int(topk), type=T.i32)
        num_valid_i32 = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=True),
            arith.constant(0, type=T.i32),
            vec_width=1,
            dtype=T.i32,
        )

        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_rows = tokens_idx * arith.index(int(topk))
        x_nbytes = x_rows * arith.index(K_packed_a)
        sx_nbytes = x_rows * arith.index(K_scale)
        w_rows = arith.index(int(experts)) * n_idx
        w_nbytes = w_rows * arith.index(K_packed_b)
        sw_nbytes = w_rows * arith.index(K_scale)
        out_nbytes = tokens_idx * n_idx * arith.index(2)
        if not bool(accumulate):
            out_nbytes = x_rows * n_idx * arith.index(2)

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False, num_records_bytes=sx_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False, num_records_bytes=sw_nbytes)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False, num_records_bytes=out_nbytes)
        tw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, block_row_start, num_valid_i32)
        block_ok = arith.andi(block_in_valid, arith.andi(eid_ok0, eid_ok1))

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        base_ptr = alloc.get_base()
        smem_a = SmemPtr(base_ptr, off_a, T.i8, shape=(lds_a_data_bytes,))
        smem_b = SmemPtr(base_ptr, off_b, T.i8, shape=(lds_b_data_bytes,))
        smem_as = SmemPtr(base_ptr, off_as, T.i8, shape=(lds_a_scale_bytes,))
        smem_bs = SmemPtr(base_ptr, off_bs, T.i8, shape=(lds_b_scale_bytes,))
        lds_a = get_op_result_or_value(smem_a.get())
        lds_b = get_op_result_or_value(smem_b.get())
        lds_as = get_op_result_or_value(smem_as.get())
        lds_bs = get_op_result_or_value(smem_bs.get())

        def make_desc_a(k_base):
            return k_base / arith.index(PACK_FACTOR_A)

        def issue_a_load(k_packed_base):
            total = int(tile_m * packed_tile_k_a)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, elem), arith.constant(total, type=T.i32))
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(packed_tile_k_a))
                    col = elem % arith.index(int(packed_tile_k_a))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_i32 = arith.index_cast(T.i32, row)
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
                    ts = tok * c_topk_i32 + slot
                    ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                    load_ok = arith.andi(row_ok, ts_ok)
                    x_idx = ts * arith.constant(K_packed_a, type=T.i32) + arith.index_cast(T.i32, k_packed_base + col)
                    x_idx_safe = arith.select(load_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(load_ok, buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.i8), arith.constant(0, type=T.i8))
                    lds_idx = row * arith.index(lds_a_stride_bytes) + col
                    v1 = vector.from_elements(T.vec(1, T.i8), [x_val])
                    vector.store(v1, lds_a, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def make_desc_as(k_base):
            return k_base / arith.index(SCALE_BLOCK)

        def issue_as_load(k_scale_base):
            total = int(tile_m * scale_k_per_tile)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, elem), arith.constant(total, type=T.i32))
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(scale_k_per_tile))
                    ksc = elem % arith.index(int(scale_k_per_tile))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_i32 = arith.index_cast(T.i32, row)
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
                    ts = tok * c_topk_i32 + slot
                    ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                    load_ok = arith.andi(row_ok, ts_ok)
                    ksc_off = k_scale_base + ksc
                    sx_idx = ts * arith.constant(K_scale, type=T.i32) + arith.index_cast(T.i32, ksc_off)
                    sx_idx_safe = arith.select(load_ok, sx_idx, arith.constant(0, type=T.i32))
                    sx_val = arith.select(load_ok, buffer_ops.buffer_load(sx_rsrc, sx_idx_safe, vec_width=1, dtype=T.i8), arith.constant(127, type=T.i8))
                    if is_fp4:
                        lds_idx = row * arith.index(int(scale_k_per_tile)) + ksc
                    else:
                        warp_row_idx = row / arith.index(warp_tile_m)
                        local_row = row % arith.index(warp_tile_m)
                        lane_row = local_row % arith.index(WMMA_M)
                        local_wm_idx = local_row / arith.index(WMMA_M)
                        global_lds_row = warp_row_idx * arith.index(WMMA_M) + lane_row
                        ksc_blk = ksc / arith.index(SCALES_PER_WMMA)
                        ksc_sub = ksc % arith.index(SCALES_PER_WMMA)
                        lds_idx = (
                            global_lds_row * arith.index(interleaved_scale_cols_a)
                            + ksc_blk * arith.index(wmma_m_rep * SCALES_PER_WMMA)
                            + local_wm_idx * arith.index(SCALES_PER_WMMA)
                            + ksc_sub
                        )
                    v1 = vector.from_elements(T.vec(1, T.i8), [sx_val])
                    vector.store(v1, lds_as, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def make_desc_b(n_off, k_base):
            if is_fp4:
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_w, lds_memref=lds_b,
                    global_offset=(n_off, k_base / arith.index(PACK_FACTOR_B)),
                    tensor_shape=(int(tile_n), int(packed_tile_k_b)),
                    strides=(K_packed_b, 1),
                    tile_shape=(int(tile_n), int(packed_tile_k_b)),
                    elem_bytes=1, pad_interval=int(packed_tile_k_b), pad_amount=LDS_PAD_B_BYTES,
                    num_warps=int(m_warp) * int(n_warp), workgroup_mask=0)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(N_total // 16), int(K_packed_b * 16)),
                strides=(int(K_packed_b * 16), 1),
                tile_shape=(int(tile_n // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=int(m_warp) * int(n_warp),
                workgroup_mask=0)

        def make_desc_bs(n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=lds_bs,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=int(m_warp) * int(n_warp), workgroup_mask=0)

        def issue_bs_load(k_base, n_off):
            if is_fp4:
                tdm_ops.tensor_load_2d(make_desc_bs(n_off, k_base))
                return
            total_bs = int(tile_n * scale_k_per_tile)
            rounds_bs = (total_bs + block_threads - 1) // block_threads
            for it in range(rounds_bs):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total_bs, type=T.i32),
                )
                _if_bs = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_bs.then_block):
                    n_local = elem // arith.index(int(scale_k_per_tile))
                    ksc = elem % arith.index(int(scale_k_per_tile))
                    k_scale_base = k_base / arith.index(SCALE_BLOCK)
                    sw_idx = arith.index_cast(T.i32,
                        (n_off + n_local) * arith.index(K_scale) + k_scale_base + ksc)
                    sw_val = buffer_ops.buffer_load(
                        sw_rsrc, sw_idx, vec_width=1, dtype=T.i8)
                    wave_n = n_local / arith.index(warp_tile_n)
                    sub_n = n_local % arith.index(warp_tile_n)
                    wn_idx = sub_n / arith.index(WMMA_N)
                    lane_row = sub_n % arith.index(WMMA_N)
                    ksc_blk = ksc / arith.index(SCALES_PER_WMMA)
                    ksc_sub = ksc % arith.index(SCALES_PER_WMMA)
                    warp_lds_row = wave_n * arith.index(WMMA_N) + lane_row
                    lds_idx = (
                        warp_lds_row * arith.index(interleaved_scale_cols_b)
                        + ksc_blk * arith.index(b_scale_load_rep * SCALES_PER_WMMA)
                        + wn_idx * arith.index(SCALES_PER_WMMA)
                        + ksc_sub
                    )
                    v1 = vector.from_elements(T.vec(1, T.i8), [sw_val])
                    vector.store(v1, lds_bs, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def issue_b_load(k_base):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_off = eid_idx * n_idx + blk_n
            tdm_ops.tensor_load_2d(make_desc_b(n_off, k_base))
            issue_bs_load(k_base, n_off)

        def _lds_load_b128(lds_buffer, byte_offset):
            from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
            from flydsl.expr.arith import _to_raw as _raw
            from flydsl.expr.arith import ArithValue as _AV
            lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
            raw_memref = arith.unwrap(lds_buffer)
            lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)
            total_byte = _AV(lds_base) + byte_offset
            addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
            ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
            vec4_i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
            return llvm_dialect.load(vec4_i32_ty, ptr_val)

        def load_data_frag(lds_buffer, lane_base, ks):
            return _mxscale_load_data_frag(
                lds_buffer=lds_buffer,
                lane_base=lane_base,
                ks=ks,
                PACK_FACTOR_A=PACK_FACTOR_A,
                WMMA_K=WMMA_K,
                is_fp4=is_fp4,
                _lds_load_b128=_lds_load_b128,
                arith=arith,
                vector=vector,
            )

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            if is_fp4:
                return _mxscale_load_rowmajor_b_frag(
                    lds_buffer=lds_buffer,
                    b_lane_bases=b_lane_bases,
                    wn=wn,
                    ks=ks,
                    PACK_FACTOR_B=PACK_FACTOR_B,
                    WMMA_K=WMMA_K,
                    _lds_load_b128=_lds_load_b128,
                    arith=arith,
                    vector=vector,
                )
            return _mxscale_load_preshuffled_b_frag(
                lds_buffer=lds_buffer,
                b_lane_bases=b_lane_bases,
                wn=wn,
                ks=ks,
                is_fp4=is_fp4,
                is_a8w4=is_a8w4,
                PACK_FACTOR_B=PACK_FACTOR_B,
                WMMA_K=WMMA_K,
                _lds_load_b128=_lds_load_b128,
                arith=arith,
                vector=vector,
            )

        def load_scale_i32(lds_buffer, scale_base, ks):
            return _mxscale_load_scale_i32(
                lds_buffer=lds_buffer,
                scale_base=scale_base,
                ks=ks,
                SCALES_PER_WMMA=SCALES_PER_WMMA,
                _lds_load_b128=_lds_load_b128,
                llvm_dialect=llvm_dialect,
                ir=ir,
                arith=arith,
                T=T,
            )

        def _precompute_a_data_bases():
            return _mxscale_precompute_a_data_bases(
                warp_m_base=warp_m_base,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                lds_a_stride_bytes=lds_a_stride_bytes,
                wmma_m_rep=wmma_m_rep,
                WMMA_M=WMMA_M,
                is_fp4=is_fp4,
                arith=arith,
                range_constexpr=range_constexpr,
            )

        def _precompute_b_data_bases():
            if is_fp4:
                return _mxscale_precompute_rowmajor_b_data_bases(
                    warp_n_base=warp_n_base,
                    lane16=lane16,
                    lane_kgrp=lane_kgrp,
                    lds_b_stride_bytes=lds_b_stride_bytes,
                    wmma_n_rep=wmma_n_rep,
                    WMMA_N=WMMA_N,
                    arith=arith,
                    range_constexpr=range_constexpr,
                )
            return _mxscale_precompute_preshuffled_b_data_bases(
                packed_tile_k_b=packed_tile_k_b,
                warp_tile_n=warp_tile_n,
                wave_n_idx=wave_n_idx,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                wmma_n_rep=wmma_n_rep,
                arith=arith,
                range_constexpr=range_constexpr,
            )

        def _precompute_a_scale_lane_bases():
            if is_fp4:
                return _mxscale_precompute_rowmajor_a_scale_lane_bases(
                    warp_m_base=warp_m_base,
                    lane16=lane16,
                    scale_k_per_tile=scale_k_per_tile,
                    wmma_m_rep=wmma_m_rep,
                    WMMA_M=WMMA_M,
                    arith=arith,
                    range_constexpr=range_constexpr,
                )
            return _mxscale_precompute_a_scale_lane_bases(
                warp_m_base=warp_m_base,
                lane16=lane16,
                wmma_m_rep=wmma_m_rep,
                interleaved_scale_cols_a=interleaved_scale_cols_a,
                arith=arith,
            )

        def _precompute_b_scale_lane_bases():
            if is_fp4:
                return _mxscale_precompute_rowmajor_b_scale_lane_bases(
                    warp_n_base=warp_n_base,
                    lane16=lane16,
                    scale_k_per_tile=scale_k_per_tile,
                    wmma_n_rep=wmma_n_rep,
                    WMMA_N=WMMA_N,
                    arith=arith,
                    range_constexpr=range_constexpr,
                )
            return _mxscale_precompute_b_scale_lane_bases(
                warp_n_base=warp_n_base,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                b_scale_load_rep=b_scale_load_rep,
                interleaved_scale_cols_b=interleaved_scale_cols_b,
                is_fp4=is_fp4,
                is_a8w4=is_a8w4,
                arith=arith,
                SCALES_PER_WMMA=SCALES_PER_WMMA,
            )

        def load_scale_b128(lds_buffer, scale_base, reps, ks=0):
            return _mxscale_load_scale_b128(
                lds_buffer=lds_buffer,
                scale_base=scale_base,
                reps=reps,
                ks=ks,
                SCALES_PER_WMMA=SCALES_PER_WMMA,
                _lds_load_b128=_lds_load_b128,
                arith=arith,
                vector=vector,
                range_constexpr=range_constexpr,
            )

        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        acc = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            a_data_bases = _precompute_a_data_bases()
            b_data_bases = _precompute_b_data_bases()
            as_bases = _precompute_a_scale_lane_bases()
            bs_bases = _precompute_b_scale_lane_bases()

            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                a_desc = make_desc_a(k_base)
                as_desc = make_desc_as(k_base)
                issue_a_load(a_desc)
                issue_as_load(as_desc)
                issue_b_load(k_base)
                tdm_ops.tensor_wait(0)
                gpu.barrier()
                for ks in range_constexpr(k_wmma_steps):
                    b_v = [load_b_frag(lds_b, b_data_bases, wn, ks) for wn in range_constexpr(wmma_n_rep)]
                    if is_fp4:
                        as_v = [load_scale_i32(lds_as, as_bases[wm], ks) for wm in range_constexpr(wmma_m_rep)]
                        bs_v = [load_scale_i32(lds_bs, bs_bases[bi], ks) for bi in range_constexpr(b_scale_load_rep)]
                    else:
                        as_v = load_scale_b128(lds_as, as_bases[0], wmma_m_rep, ks)
                        bs_v = load_scale_b128(lds_bs, bs_bases[0], b_scale_load_rep, ks)
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_data_frag(lds_a, a_data_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            _mxscale_emit_wmma(
                                accs=acc,
                                wm=wm,
                                wn=wn,
                                a_frag=a_frag,
                                b_frags=b_v,
                                a_scales=as_v,
                                b_scales=bs_v,
                                is_fp4=is_fp4,
                                is_a8w4=is_a8w4,
                                use_scale_opsel=False,
                                rocdl=rocdl,
                                T=T,
                            )
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_sub8(acc_idx, vec_base):
                return _mxscale_extract_sub8(
                    acc[acc_idx], vec_base, vector=vector, range_constexpr=range_constexpr, ACC_VEC_SIZE=ACC_VEC_SIZE
                )

            _emit_stage2_store_epilogue(
                sub_tiles=_sub_tiles,
                by=by,
                tile_m=int(tile_m),
                route_tile_m=int(route_tile_m),
                warp_m_base=warp_m_base,
                warp_n_base=warp_n_base,
                blk_n=blk_n,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                WMMA_N=WMMA_N,
                i32_tokens_in=i32_tokens_in,
                i32_n_in=i32_n_in,
                topk=int(topk),
                num_valid_i32=num_valid_i32,
                block_row_start=block_row_start,
                sorted_rsrc=sorted_rsrc,
                tw_rsrc=tw_rsrc,
                out_rsrc=out_rsrc,
                doweight_stage2=bool(doweight_stage2),
                accumulate=bool(accumulate),
                out_elem_ty=out_elem_ty,
                load_sub8=_load_sub8,
                fx=fx,
                arith=arith,
                buffer_ops=buffer_ops,
                scf=scf,
                vector=vector,
                range_constexpr=range_constexpr,
                rocdl=rocdl,
                T=T,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_mxscale_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (n_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_mxscale_stage2_single(
            arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights, arg_num_valid_ids,
            i32_tokens_in, i32_n_in, i32_k_in, i32_size_expert_ids_in,
        )
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=waves_per_eu,
            ir=ir,
        )

    return launch_mxscale_stage2_single



@functools.lru_cache(maxsize=256)
def _compile_moe_stage1_kernel(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    data_format: str,
    out_dtype: str,
    waves_per_eu: int | None,
):
    if data_format not in ("fp16", "bf16", "fp4", "fp8", "a8w4"):
        raise ValueError(f"Unsupported stage1 data_format: {data_format!r}")

    single_tile_m, single_tile_n, single_m_warp, single_n_warp = _pick_mxscale_launch_shape(
        "fp8" if data_format in ("fp16", "bf16") else data_format,
        int(route_tile_m),
        int(tile_n),
    )
    if data_format in ("fp16", "bf16"):
        exe = _compile_stage1_dense_kernel_impl(
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=int(experts),
            topk=int(topk),
            route_tile_m=int(route_tile_m),
            tile_m=int(single_tile_m),
            tile_n=int(single_tile_n),
            tile_k=int(tile_k),
            m_warp=int(single_m_warp),
            n_warp=int(single_n_warp),
            doweight_stage1=bool(doweight_stage1),
            out_dtype=out_dtype,
            waves_per_eu=waves_per_eu,
        )
        if data_format == "bf16":
            return _bf16_to_f16_wrapper(exe, x_arg=1, w_arg=2)
        return exe

    if data_format in ("fp4", "fp8", "a8w4"):
        return _compile_stage1_mxscale_kernel_impl(
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=int(experts),
            topk=int(topk),
            route_tile_m=int(route_tile_m),
            tile_m=int(single_tile_m),
            tile_n=int(single_tile_n),
            tile_k=int(tile_k),
            m_warp=int(single_m_warp),
            n_warp=int(single_n_warp),
            doweight_stage1=bool(doweight_stage1),
            out_dtype=out_dtype,
            waves_per_eu=waves_per_eu,
            data_format=data_format,
        )
    raise ValueError(f"Unsupported stage1 data_format: {data_format!r}")


@functools.lru_cache(maxsize=256)
def _compile_moe_stage2_kernel(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    data_format: str,
    out_dtype: str,
    accumulate: bool,
    waves_per_eu: int | None,
):
    if data_format not in ("fp16", "bf16", "fp4", "fp8", "a8w4"):
        raise ValueError(f"Unsupported stage2 data_format: {data_format!r}")

    single_tile_m, single_tile_n, single_m_warp, single_n_warp = _pick_mxscale_launch_shape(
        "fp8" if data_format in ("fp16", "bf16") else data_format,
        int(route_tile_m),
        int(tile_n),
    )

    if data_format in ("fp16", "bf16"):
        exe = _compile_stage2_dense_kernel_impl(
            inter_dim=int(inter_dim),
            experts=int(experts),
            topk=int(topk),
            route_tile_m=int(route_tile_m),
            tile_m=int(single_tile_m),
            tile_n=int(single_tile_n),
            tile_k=int(tile_k),
            m_warp=int(single_m_warp),
            n_warp=int(single_n_warp),
            doweight_stage2=bool(doweight_stage2),
            out_dtype=out_dtype,
            accumulate=bool(accumulate),
            waves_per_eu=waves_per_eu,
        )
        if data_format == "bf16":
            return _bf16_to_f16_wrapper(exe, x_arg=1, w_arg=2)
        return exe

    if data_format in ("fp4", "fp8", "a8w4"):
        return _compile_stage2_mxscale_kernel_impl(
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=int(experts),
            topk=int(topk),
            route_tile_m=int(route_tile_m),
            tile_m=int(single_tile_m),
            tile_n=int(single_tile_n),
            tile_k=int(tile_k),
            m_warp=int(single_m_warp),
            n_warp=int(single_n_warp),
            doweight_stage2=bool(doweight_stage2),
            out_dtype=out_dtype,
            accumulate=bool(accumulate),
            waves_per_eu=waves_per_eu,
            data_format=data_format,
        )
    raise ValueError(f"Unsupported stage2 data_format: {data_format!r}")


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    in_dtype: str = "fp4",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    waves_per_eu: int | None = None,
):
    _require_gfx1250()
    if waves_per_eu is not None and int(waves_per_eu) < 1:
        raise ValueError(f"waves_per_eu must be >= 1, got {waves_per_eu!r}")

    if in_dtype not in ("fp4", "fp8", "fp16", "bf16", "a8w4"):
        return _compile_with_optional_wpe(
            _compile_moe_gemm1_base,
            dict(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage1=doweight_stage1,
                in_dtype=in_dtype,
                group_size=group_size,
                out_dtype=out_dtype,
                use_cshuffle_epilog=use_cshuffle_epilog,
                waves_per_eu=waves_per_eu,
            ),
        )

    route_tile_m = int(tile_m)
    if in_dtype in ("fp16", "bf16", "fp4", "fp8", "a8w4"):
        return _compile_moe_stage1_kernel(
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=int(experts),
            topk=int(topk),
            route_tile_m=route_tile_m,
            tile_n=int(tile_n),
            tile_k=int(tile_k),
            doweight_stage1=bool(doweight_stage1),
            data_format=in_dtype,
            out_dtype=out_dtype,
            waves_per_eu=waves_per_eu,
        )


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp4",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    accumulate: bool = True,
    waves_per_eu: int | None = None,
):
    _require_gfx1250()
    if waves_per_eu is not None and int(waves_per_eu) < 1:
        raise ValueError(f"waves_per_eu must be >= 1, got {waves_per_eu!r}")

    if in_dtype not in ("fp4", "fp8", "fp16", "bf16", "a8w4"):
        return _compile_with_optional_wpe(
            _compile_moe_gemm2_base,
            dict(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage2=doweight_stage2,
                in_dtype=in_dtype,
                group_size=group_size,
                out_dtype=out_dtype,
                use_cshuffle_epilog=use_cshuffle_epilog,
                accumulate=accumulate,
                waves_per_eu=waves_per_eu,
            ),
        )

    route_tile_m = int(tile_m)
    if in_dtype in ("fp16", "bf16", "fp4", "fp8", "a8w4"):
        return _compile_moe_stage2_kernel(
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=int(experts),
            topk=int(topk),
            route_tile_m=route_tile_m,
            tile_n=int(tile_n),
            tile_k=int(tile_k),
            doweight_stage2=bool(doweight_stage2),
            data_format=in_dtype,
            out_dtype=out_dtype,
            accumulate=bool(accumulate),
            waves_per_eu=waves_per_eu,
        )


def compile_moe_gemm2_ex(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp4",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    waves_per_eu: int | None = None,
    mode: str = MoeGemm2Mode.ATOMIC,
    valid_mask=None,
    zero_intermediate: bool = True,
):
    _require_gfx1250()
    if in_dtype in ("fp4", "fp8", "fp16", "bf16", "a8w4"):
        if mode == MoeGemm2Mode.REDUCE:
            gemm2_exe = compile_moe_gemm2(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage2=doweight_stage2,
                in_dtype=in_dtype,
                group_size=group_size,
                out_dtype=out_dtype,
                use_cshuffle_epilog=use_cshuffle_epilog,
                accumulate=False,
                waves_per_eu=waves_per_eu,
            )
            out_s = str(out_dtype).strip().lower()
            if out_s in ("f16", "fp16", "half"):
                dtype_str = "f16"
            elif out_s in ("bf16", "bfloat16"):
                dtype_str = "bf16"
            else:
                dtype_str = "f32"
            reduce_exe = compile_moe_reduction(
                topk=topk,
                model_dim=model_dim,
                dtype_str=dtype_str,
                use_mask=(valid_mask is not None),
            )
            from kernels.moe_gemm_2stage import _MoeGemm2ReduceWrapper

            return _MoeGemm2ReduceWrapper(
                gemm2_exe=gemm2_exe,
                reduce_exe=reduce_exe,
                topk=topk,
                model_dim=model_dim,
                out_dtype_str=dtype_str,
                use_mask=(valid_mask is not None),
                zero_intermediate=zero_intermediate,
            )
        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=True,
            waves_per_eu=waves_per_eu,
        )

    return _compile_with_optional_wpe(
        _compile_moe_gemm2_ex_base,
        dict(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            waves_per_eu=waves_per_eu,
            mode=mode,
            valid_mask=valid_mask,
            zero_intermediate=zero_intermediate,
        ),
    )


__all__ = [
    "MoeGemm2Mode",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_gemm2_ex",
    "compile_moe_reduction",
]
