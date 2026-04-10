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
        # Use Python object id as key and keep strong refs to originals so
        # the id cannot be reused while cached (data_ptr() suffers from ABA
        # reuse when the allocator recycles GPU addresses).
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
    expert_sched_mode: bool = True,
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

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
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
                ir=ir,
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

    if expert_sched_mode:
        launch_fp16_stage1_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

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
    expert_sched_mode: bool = True,
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

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
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
                ir=ir,
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

    if expert_sched_mode:
        launch_fp16_stage2_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
    DS_LOADS_PER_A_FRAG = int(fmt_cfg["DS_LOADS_PER_A_FRAG"])

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
    if int(num_buffers) not in (1, 2, 3, 4):
        raise ValueError(f"num_buffers must be 1, 2, 3, or 4, got {num_buffers}")
    use_cluster = int(cluster_m) > 1 or int(cluster_n) > 1
    if use_cluster:
        if int(cluster_m) * int(cluster_n) > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    K = int(model_dim)
    N = int(inter_dim)
    _merge_gate_up_tdm = bool((data_format in ("fp8", "a8w4")) and (N % int(tile_n) == 0))
    num_warps_s1 = int(m_warp) * int(n_warp)
    _tdm_loader_waves = 2 if _merge_gate_up_tdm else 4
    if bool(wave_specialized_tdm):
        if num_warps_s1 < _tdm_loader_waves:
            raise ValueError(
                f"wave_specialized_tdm requires at least {_tdm_loader_waves} waves, got {num_warps_s1}")
    tdm_desc_num_warps = 1 if bool(wave_specialized_tdm) else num_warps_s1
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 2

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

    # Pipeline calculations for multi-buffer
    _use_pipeline = int(num_buffers) >= 2
    if _use_pipeline:
        from kernels.gemm_common_gfx1250 import (
            pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
        )
        from kernels.pipeline_utils import make_tail_plan

        pre_loaded = int(num_buffers) - 1
        loop_iters = (num_k_tiles - pre_loaded) // int(num_buffers)
        _tail_start = loop_iters * int(num_buffers)
        extra = num_k_tiles - _tail_start - pre_loaded
        if _merge_gate_up_tdm:
            _B_TDM_PER_STEP = 1 if bool(wave_specialized_tdm) else 2
        else:
            _B_TDM_PER_STEP = 1 if bool(wave_specialized_tdm) else 4
        _A_GATHER_GROUPS = (int(tile_m) + 8 - 1) // 8 if bool(use_tdm_gather) else 0
        if bool(use_tdm_gather) and bool(wave_specialized_tdm):
            _A_GATHER_TDM_PER_STEP = (
                (_A_GATHER_GROUPS + _tdm_loader_waves - 1)
                // _tdm_loader_waves
            )
        else:
            _A_GATHER_TDM_PER_STEP = _A_GATHER_GROUPS
        TDM_PER_STEP = _B_TDM_PER_STEP + _A_GATHER_TDM_PER_STEP
        _fence_outstanding = TDM_PER_STEP * (int(num_buffers) - 2)
        _base_tail_plan = make_tail_plan(int(num_buffers), pre_loaded, extra)
        _tail_plan = [
            (ls, cs, o * TDM_PER_STEP // 2 if o > 0 else o)
            for ls, cs, o in _base_tail_plan
        ]
        if num_k_tiles < int(num_buffers):
            raise ValueError(
                f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
                f"got {num_k_tiles}")
    from kernels.gemm_common_gfx1250 import workgroup_barrier

    alloc = SmemAllocator(
        None,
        arch=str(get_hip_arch()),
        global_sym_name=f"moe_mxscale_{data_format}_s1_single_g{int(bool(use_tdm_gather))}",
    )
    _nb = int(num_buffers)
    off_ag_list, off_as_list = [], []
    off_bg_list, off_bs_list = [], []
    off_bu_list, off_bsu_list = [], []
    off_bg_pair_list, off_bs_pair_list = [], []
    for _buf_i in range(_nb):
        _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_a_data_bytes; off_ag_list.append(_o)
        if _merge_gate_up_tdm:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + 2 * lds_b_data_bytes; off_bg_pair_list.append(_o)
        else:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_data_bytes; off_bg_list.append(_o)
        _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_a_scale_bytes; off_as_list.append(_o)
        if _merge_gate_up_tdm:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + 2 * lds_b_scale_bytes; off_bs_pair_list.append(_o)
        else:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_scale_bytes; off_bs_list.append(_o)
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_data_bytes; off_bu_list.append(_o)
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_scale_bytes; off_bsu_list.append(_o)

    # TDM store epilogue: D output LDS layout (stage1)
    LDS_PAD_D_BYTES_s1 = 16
    elem_bytes_d_s1 = 2  # f16/bf16
    if bool(use_tdm_store):
        from kernels.gemm_common_gfx1250 import (
            store_acc_vec8_to_lds,
        )
        lds_d_row_stride_s1 = warp_tile_n * elem_bytes_d_s1 + LDS_PAD_D_BYTES_s1
        warp_d_bytes_s1 = warp_tile_m * lds_d_row_stride_s1
        total_d_bytes_s1 = num_warps_s1 * warp_d_bytes_s1
        d_output_off_s1 = 0
        _lds_d_stride_elems_s1 = lds_d_row_stride_s1 // 2
        _warp_d_elems_s1 = warp_d_bytes_s1 // 2
        _n_col_d_elems_s1 = WMMA_N * elem_bytes_d_s1 // 2
        d_need_epilogue_fence_s1 = _use_pipeline
        if total_d_bytes_s1 > alloc.ptr:
            alloc.ptr = total_d_bytes_s1

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_mxscale_stage1_single(
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
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = i32_k_in
        if inst_prefetch:
            if arith.cmpi(arith.CmpIPredicate.eq, rocdl.wave_id(),
                          arith.constant(0, type=T.i32)):
                _prefetch_lines = ["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 8, 1), 1"]
                for _pg in range_constexpr(10):
                    _prefetch_lines.append(
                        f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31")
                llvm_dialect.inline_asm(
                    None, [],
                    "\n".join(_prefetch_lines),
                    "", has_side_effects=True,
                )
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

        if use_cluster:
            _local_x, _local_y = gpu.compute_cluster_position()
            _a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                _local_x, _local_y, int(cluster_m), int(cluster_n))
        else:
            b_mcast_mask = 0

        base_ptr = alloc.get_base()
        lds_ag_bufs, lds_as_bufs = [], []
        lds_bg_bufs, lds_bs_bufs = [], []
        lds_bu_bufs, lds_bsu_bufs = [], []
        lds_bg_pair_bufs, lds_bs_pair_bufs = [], []
        for _bi in range_constexpr(_nb):
            lds_ag_bufs.append(get_op_result_or_value(
                SmemPtr(base_ptr, off_ag_list[_bi], T.i8, shape=(lds_a_data_bytes,)).get()))
            lds_as_bufs.append(get_op_result_or_value(
                SmemPtr(base_ptr, off_as_list[_bi], T.i8, shape=(lds_a_scale_bytes,)).get()))
            if _merge_gate_up_tdm:
                lds_bg_pair_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bg_pair_list[_bi], T.i8, shape=(2 * lds_b_data_bytes,)).get()))
                lds_bs_pair_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bs_pair_list[_bi], T.i8, shape=(2 * lds_b_scale_bytes,)).get()))
            else:
                lds_bg_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bg_list[_bi], T.i8, shape=(lds_b_data_bytes,)).get()))
                lds_bs_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bs_list[_bi], T.i8, shape=(lds_b_scale_bytes,)).get()))
                lds_bu_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bu_list[_bi], T.i8, shape=(lds_b_data_bytes,)).get()))
                lds_bsu_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bsu_list[_bi], T.i8, shape=(lds_b_scale_bytes,)).get()))

        if bool(use_tdm_store):
            from kernels.gemm_common_gfx1250 import get_lds_memref
            d_lds_f16_count_s1 = total_d_bytes_s1 // 2
            d_smem_s1 = SmemPtr(base_ptr, d_output_off_s1, T.f16,
                                shape=(d_lds_f16_count_s1,))
            d_lds_buffer_s1 = get_lds_memref(d_smem_s1)
            warp_lds_off_s1 = (
                (wave_m_idx * arith.index(int(n_warp)) + wave_n_idx)
                * arith.index(_warp_d_elems_s1)
            )
            d_lane_base_s1 = (
                warp_lds_off_s1
                + lane16 * arith.index(_lds_d_stride_elems_s1)
                + lane_kgrp * arith.index(4 * elem_bytes_d_s1)
            )
            wave_id_idx_s1 = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr_s1 = (
                wave_id_idx_s1 * arith.index(warp_d_bytes_s1)
                + arith.index(d_output_off_s1)
            )
            warp_m_off_sgpr_s1 = (
                (wave_id_idx_s1 / arith.index(int(n_warp)))
                * arith.index(warp_tile_m)
            )
            warp_n_off_sgpr_s1 = (
                (wave_id_idx_s1 % arith.index(int(n_warp)))
                * arith.index(warp_tile_n)
            )
            # TDM store for MoE stage1 uses gather-store mode because the
            # output rows are not contiguous — each sorted row maps to
            # out[tok * topk + slot, :] which is a scattered layout.
            # d_desc_s1 is built lazily in the epilogue after sorted_ids
            # are decoded (see _emit_tdm_gather_store_s1 below).

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        def make_desc_a(k_base):
            return k_base / arith.index(PACK_FACTOR_A)

        # TDM gather for A data
        _use_tdm_gather_a = bool(use_tdm_gather)

        def issue_a_load(k_packed_base, target_lds):
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
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    load_ok = arith.andi(row_ok, tok_ok)
                    x_idx = tok * arith.constant(K_packed_a, type=T.i32) + arith.index_cast(T.i32, k_packed_base + col)
                    x_idx_safe = arith.select(load_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(load_ok, buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.i8), arith.constant(0, type=T.i8))
                    lds_idx = row * arith.index(lds_a_stride_bytes) + col
                    v1 = vector.from_elements(T.vec(1, T.i8), [x_val])
                    vector.store(v1, target_lds, [lds_idx], alignment=1)
                    scf.YieldOp([])

        # Pre-compute token row indices for ALL tile_m rows (once, outside K-loop).
        # _a_tok_ids[i] = token_id for TDM gather A load
        # _a_out_row_ids[i] = tok * topk + slot for TDM gather store output
        _a_tok_ids = []
        _a_out_row_ids = []
        _a_load_valids = []
        _a_store_valids = []

        def _sum_i32_values(_vals):
            _acc = arith.constant(0, type=T.i32)
            for _vi in range_constexpr(len(_vals)):
                _acc = _acc + _vals[_vi]
            return _acc

        def _precompute_a_row_indices():
            """Load sorted_ids for all tile_m rows and decode token_ids + output row indices."""
            _safe_row = arith.constant(0, type=T.i32)
            _one_i32 = arith.constant(1, type=T.i32)
            _zero_i32 = arith.constant(0, type=T.i32)
            for _ri in range_constexpr(int(tile_m)):
                _sorted_row = by * fx.Index(int(tile_m)) + fx.Index(_ri)
                _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                _row_in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Int32(_ri),
                    fx.Int32(int(route_tile_m)),
                )
                _row_in_valid = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    _sorted_i32,
                    num_valid_i32,
                )
                _row_ok = arith.andi(_row_in_route, _row_in_valid)
                _sorted_safe = arith.select(
                    _row_ok, _sorted_i32,
                    block_row_start,
                )
                _fused = buffer_ops.buffer_load(sorted_rsrc, _sorted_safe, vec_width=1, dtype=T.i32)
                _tok = _fused & fx.Int32((1 << 24) - 1)
                _slot = _fused >> fx.Int32(24)
                _tok_ok = arith.cmpi(arith.CmpIPredicate.ult, _tok, i32_tokens_in)
                _slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, _slot, fx.Int32(0))
                _slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, _slot, c_topk_i32)
                _slot_ok = arith.andi(_slot_ok0, _slot_ok1)
                _row_tok_ok = arith.andi(_row_ok, _tok_ok)
                _load_valid_i32 = arith.select(_row_tok_ok, _one_i32, _zero_i32)
                _a_load_valids.append(rocdl.readfirstlane(T.i32, _load_valid_i32))
                _tok_safe = arith.select(_row_tok_ok, _tok, _safe_row)
                _tok_sgpr = rocdl.readfirstlane(T.i32, _tok_safe)
                _a_tok_ids.append(_tok_sgpr)
                _out_row = _tok * c_topk_i32 + _slot
                _row_fully_ok = arith.andi(_row_tok_ok, _slot_ok)
                _store_valid_i32 = arith.select(_row_fully_ok, _one_i32, _zero_i32)
                _a_store_valids.append(rocdl.readfirstlane(T.i32, _store_valid_i32))
                _out_row_safe = arith.select(
                    _row_fully_ok, _out_row,
                    _safe_row,
                )
                _out_row_sgpr = rocdl.readfirstlane(T.i32, _out_row_safe)
                _a_out_row_ids.append(_out_row_sgpr)

        _TDM_GATHER_CHUNK = 8
        _TDM_GATHER_GROUPS = (int(tile_m) + _TDM_GATHER_CHUNK - 1) // _TDM_GATHER_CHUNK

        _a_tokens_sgpr = None
        _a_tokens_topk_sgpr = None

        def _get_tokens_sgpr():
            nonlocal _a_tokens_sgpr
            if _a_tokens_sgpr is None:
                _tok_i32 = arith.index_cast(T.i32, arith.index_cast(T.index, i32_tokens_in))
                _a_tokens_sgpr = rocdl.readfirstlane(T.i32, _tok_i32)
            return _a_tokens_sgpr

        def _get_tokens_topk_sgpr():
            nonlocal _a_tokens_topk_sgpr
            if _a_tokens_topk_sgpr is None:
                _m_i32 = _get_tokens_sgpr() * c_topk_i32
                _a_tokens_topk_sgpr = rocdl.readfirstlane(T.i32, _m_i32)
            return _a_tokens_topk_sgpr

        def issue_a_load_tdm_gather(k_base, target_lds):
            """Load A data using TDM gather mode — one TDM instruction per 8 rows."""
            k_packed_base = k_base if PACK_FACTOR_A == 1 else k_base // fx.Index(PACK_FACTOR_A)
            _tokens_dim1 = _get_tokens_sgpr()
            _zero_i32 = arith.constant(0, type=T.i32)
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _start = _gi * _TDM_GATHER_CHUNK
                _cnt = min(_TDM_GATHER_CHUNK, int(tile_m) - _start)
                _row_indices = _a_tok_ids[_start:_start + _cnt]
                _valid_count = _sum_i32_values(_a_load_valids[_start:_start + _cnt])
                _lds_off = fx.Index(_start * lds_a_stride_bytes)
                _has_valid = arith.cmpi(arith.CmpIPredicate.sgt, _valid_count, _zero_i32)
                _issue_pred = _has_valid
                if wave_specialized_tdm:
                    _gather_owner = _gi % _tdm_loader_waves
                    _is_gather_loader = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(_gather_owner, type=T.i32),
                    )
                    _issue_pred = arith.andi(_issue_pred, _is_gather_loader)
                _if_issue = scf.IfOp(_issue_pred)
                with ir.InsertionPoint(_if_issue.then_block):
                    desc = tdm_ops.make_tensor_gather_descriptor(
                        global_ptr=arg_x,
                        lds_memref=target_lds,
                        row_indices=_row_indices,
                        row_width=int(packed_tile_k_a),
                        tensor_dim0=K_packed_a,
                        tensor_dim1=_tokens_dim1,
                        stride=K_packed_a,
                        elem_bytes=1,
                        pad_interval=int(packed_tile_k_a) if LDS_PAD_A_BYTES > 0 else 0,
                        pad_amount=LDS_PAD_A_BYTES if LDS_PAD_A_BYTES > 0 else 0,
                        index_size=32,
                        gather_tile_dim1=_valid_count,
                        lds_byte_offset=_lds_off,
                        global_byte_offset=k_packed_base,
                    )
                    tdm_ops.tensor_load_gather(desc)
                    scf.YieldOp([])

        def make_desc_as(k_base):
            return k_base / arith.index(SCALE_BLOCK)

        def issue_as_load(k_scale_base, target_lds):
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
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    load_ok = arith.andi(row_ok, tok_ok)
                    ksc_off = k_scale_base + ksc
                    sx_idx = tok * arith.constant(K_scale, type=T.i32) + arith.index_cast(T.i32, ksc_off)
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
                    vector.store(v1, target_lds, [lds_idx], alignment=1)
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
                    num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(experts * (2 * N) // 16), int(K_packed_b * 16)),
                strides=(K_packed_b * 16, 1),
                tile_shape=(int(tile_n // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_b_pair(lds_b_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(experts * (2 * N) // 16), int(K_packed_b * 16)),
                strides=(K_packed_b * 16, 1),
                tile_shape=(int((2 * tile_n) // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_bs(lds_bs_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=lds_bs_mem,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)

        def make_desc_bs_pair(lds_bs_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=lds_bs_mem,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(2 * tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(2 * tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)

        def _stage1_pair_row_base():
            _eid_row = arith.index_cast(T.index, eid_i32) * arith.index(int(2 * N))
            _tile_idx = blk_n / arith.index(int(tile_n))
            return _eid_row + _tile_idx * arith.index(int(2 * tile_n))

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
            return _mxscale_precompute_rowmajor_b_scale_lane_bases(
                warp_n_base=warp_n_base,
                lane16=lane16,
                scale_k_per_tile=scale_k_per_tile,
                wmma_n_rep=wmma_n_rep,
                WMMA_N=WMMA_N,
                arith=arith,
                range_constexpr=range_constexpr,
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
            if _use_tdm_gather_a or bool(use_tdm_store):
                _precompute_a_row_indices()
            a_data_bases = _precompute_a_data_bases()
            b_data_bases = _precompute_b_data_bases()
            if _merge_gate_up_tdm:
                b_u_data_bases = [
                    _base + arith.index(lds_b_data_bytes)
                    for _base in b_data_bases
                ]
            else:
                b_u_data_bases = b_data_bases
            as_bases = _precompute_a_scale_lane_bases()
            bs_bases = _precompute_b_scale_lane_bases()
            if _merge_gate_up_tdm:
                bsu_bases = [
                    _base + arith.index(lds_b_scale_bytes)
                    for _base in bs_bases
                ]
            else:
                bsu_bases = bs_bases
            _use_scheduled_compute = _use_pipeline and not is_fp4
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm
            _front_wmma = 2 * _front_wm * wmma_n_rep
            _back_wmma = 2 * _back_wm * wmma_n_rep
            _b_frag_ds_loads_per_wn = 2 if is_a8w4 else 4
            _a_scale_ds_loads = wmma_m_rep if is_fp4 else (wmma_m_rep + 3) // 4
            _b_scale_ds_loads = b_scale_load_rep if is_fp4 else wmma_n_rep
            _gate_up_ds_loads = (
                2 * (wmma_n_rep * _b_frag_ds_loads_per_wn + _b_scale_ds_loads)
                + _a_scale_ds_loads
            )

            # ── compute-tile helper (gate + up) ──────────────────────
            def _load_gate_up_b_and_scales(buf_idx, ks):
                if _merge_gate_up_tdm:
                    _gate_b_buf = lds_bg_pair_bufs[buf_idx]
                    _up_b_buf = lds_bg_pair_bufs[buf_idx]
                    _gate_bs_buf = lds_bs_pair_bufs[buf_idx]
                    _up_bs_buf = lds_bs_pair_bufs[buf_idx]
                else:
                    _gate_b_buf = lds_bg_bufs[buf_idx]
                    _up_b_buf = lds_bu_bufs[buf_idx]
                    _gate_bs_buf = lds_bs_bufs[buf_idx]
                    _up_bs_buf = lds_bsu_bufs[buf_idx]

                b_g = [load_b_frag(_gate_b_buf, b_data_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
                b_u = [load_b_frag(_up_b_buf, b_u_data_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
                if is_fp4:
                    as_v = [load_scale_i32(lds_as_bufs[buf_idx], as_bases[wm], ks)
                            for wm in range_constexpr(wmma_m_rep)]
                    bs_gv = [load_scale_i32(_gate_bs_buf, bs_bases[bi], ks)
                             for bi in range_constexpr(b_scale_load_rep)]
                    bs_uv = [load_scale_i32(_up_bs_buf, bsu_bases[bi], ks)
                             for bi in range_constexpr(b_scale_load_rep)]
                else:
                    as_v = load_scale_b128(lds_as_bufs[buf_idx], as_bases[0],
                                           wmma_m_rep, ks)
                    bs_gv = [load_scale_i32(_gate_bs_buf, bs_bases[wn], ks)
                             for wn in range_constexpr(wmma_n_rep)]
                    bs_uv = [load_scale_i32(_up_bs_buf, bsu_bases[wn], ks)
                             for wn in range_constexpr(wmma_n_rep)]
                return b_g, bs_gv, b_u, bs_uv, as_v

            def _emit_rows(acg_in, acu_in, start_wm, a_frags, b_g, b_u, a_scales, bs_g, bs_u):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _mxscale_emit_wmma(
                            accs=acg_in, wm=wm, wn=wn,
                            a_frag=a_frags[frag_i], b_frags=b_g,
                            a_scales=a_scales, b_scales=bs_g,
                            is_fp4=is_fp4, is_a8w4=is_a8w4,
                            use_scale_opsel=False,
                            rocdl=rocdl, T=T,
                        )
                        _mxscale_emit_wmma(
                            accs=acu_in, wm=wm, wn=wn,
                            a_frag=a_frags[frag_i], b_frags=b_u,
                            a_scales=a_scales, b_scales=bs_u,
                            is_fp4=is_fp4, is_a8w4=is_a8w4,
                            use_scale_opsel=False,
                            rocdl=rocdl, T=T,
                        )

            def _compute_k_tile(acg, acu, buf_idx, mid_compute_callback=None):
                _mid_emit_ks = 0
                if k_wmma_steps > 1:
                    _mid_emit_wm = wmma_m_rep - 1
                    _mid_emit_wn = wmma_n_rep - 1
                else:
                    _front_wn = (wmma_n_rep + 1) // 2
                    if wmma_m_rep > 1:
                        _mid_emit_wm = _front_wm - 1
                        _mid_emit_wn = wmma_n_rep - 1
                    else:
                        _mid_emit_wm = 0
                        _mid_emit_wn = _front_wn - 1
                _did_mid = False
                for ks in range_constexpr(k_wmma_steps):
                    b_g, bs_gv, b_u, bs_uv, as_v = _load_gate_up_b_and_scales(buf_idx, ks)
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_data_frag(lds_ag_bufs[buf_idx],
                                                a_data_bases[wm], ks)
                        for wn_raw in range_constexpr(wmma_n_rep):
                            wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                            _mxscale_emit_wmma(
                                accs=acg, wm=wm, wn=wn,
                                a_frag=a_frag, b_frags=b_g,
                                a_scales=as_v, b_scales=bs_gv,
                                is_fp4=is_fp4, is_a8w4=is_a8w4,
                                use_scale_opsel=False,
                                rocdl=rocdl, T=T,
                            )
                            _mxscale_emit_wmma(
                                accs=acu, wm=wm, wn=wn,
                                a_frag=a_frag, b_frags=b_u,
                                a_scales=as_v, b_scales=bs_uv,
                                is_fp4=is_fp4, is_a8w4=is_a8w4,
                                use_scale_opsel=False,
                                rocdl=rocdl, T=T,
                            )
                            if (
                                not _did_mid
                                and mid_compute_callback is not None
                                and ks == _mid_emit_ks
                                and wm == _mid_emit_wm
                                and wn == _mid_emit_wn
                            ):
                                mid_compute_callback()
                                _did_mid = True
                return acg, acu

            def _a_streaming_compute(
                acg,
                acu,
                buf_idx,
                b_g,
                bs_gv,
                b_u,
                bs_uv,
                as_v,
                ks,
                next_bs_info=None,
                mid_compute_callback=None,
            ):
                next_result = None
                a_frags_front = [
                    load_data_frag(lds_ag_bufs[buf_idx], a_data_bases[wm], ks)
                    for wm in range_constexpr(_front_wm)
                ]
                _use_partial_drain = (
                    next_bs_info is not None
                    and _front_wm * wmma_n_rep >= 4
                )

                if _use_partial_drain:
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_gate_up_b_and_scales(_next_buf_idx, _next_ks)
                    rocdl.s_wait_dscnt(_gate_up_ds_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_rows(acg, acu, 0, a_frags_front, b_g, b_u, as_v, bs_gv, bs_uv)

                if mid_compute_callback is not None:
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if _back_wm > 0:
                    a_frags_back = [
                        load_data_frag(
                            lds_ag_bufs[buf_idx],
                            a_data_bases[_front_wm + h],
                            ks,
                        )
                        for h in range_constexpr(_back_wm)
                    ]
                    _back_drain = _gate_up_ds_loads if _use_partial_drain else 0
                    rocdl.s_wait_dscnt(_back_drain)
                    _emit_rows(
                        acg,
                        acu,
                        _front_wm,
                        a_frags_back,
                        b_g,
                        b_u,
                        as_v,
                        bs_gv,
                        bs_uv,
                    )

                if not _use_partial_drain and next_bs_info is not None:
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_gate_up_b_and_scales(_next_buf_idx, _next_ks)
                return acg, acu, next_result

            def _compute_k_tile_scheduled(acg, acu, buf_idx, mid_compute_callback=None):
                current_g = list(acg)
                current_u = list(acu)
                if k_wmma_steps == 1:
                    b_g, bs_gv, b_u, bs_uv, as_v = _load_gate_up_b_and_scales(buf_idx, 0)
                    current_g, current_u, _ = _a_streaming_compute(
                        current_g, current_u, buf_idx,
                        b_g, bs_gv, b_u, bs_uv, as_v, 0,
                        mid_compute_callback=mid_compute_callback,
                    )
                else:
                    b_g, bs_gv, b_u, bs_uv, as_v = _load_gate_up_b_and_scales(buf_idx, 0)
                    for ks in range_constexpr(k_wmma_steps - 1):
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        current_g, current_u, _next = _a_streaming_compute(
                            current_g, current_u, buf_idx,
                            b_g, bs_gv, b_u, bs_uv, as_v, ks,
                            next_bs_info=(buf_idx, ks + 1),
                            mid_compute_callback=_mid_cb,
                        )
                        b_g, bs_gv, b_u, bs_uv, as_v = _next
                    current_g, current_u, _ = _a_streaming_compute(
                        current_g, current_u, buf_idx,
                        b_g, bs_gv, b_u, bs_uv, as_v,
                        k_wmma_steps - 1,
                    )
                return current_g, current_u

            def _hot_loop_scheduler_scheduled():
                if not _use_scheduled_compute:
                    return
                _front_a_loads = _front_wm * DS_LOADS_PER_A_FRAG
                _back_a_loads = _back_wm * DS_LOADS_PER_A_FRAG
                for _ks in range_constexpr(k_wmma_steps):
                    if _ks == 0:
                        rocdl.sched_dsrd(_gate_up_ds_loads + _front_a_loads)
                    else:
                        rocdl.sched_dsrd(_front_a_loads)
                    rocdl.sched_mfma(_front_wmma)
                    if _back_wmma > 0:
                        rocdl.sched_dsrd(_back_a_loads)
                        rocdl.sched_mfma(_back_wmma)
                    if _ks < k_wmma_steps - 1:
                        rocdl.sched_dsrd(_gate_up_ds_loads)
                rocdl.sched_barrier(0)

            if wave_specialized_tdm:
                _tdm_wave_id = rocdl.wave_id()
                _loader_waves = _tdm_loader_waves
                _is_loader_wave = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    _tdm_wave_id,
                    arith.constant(_loader_waves, type=T.i32),
                )
                _tdm_pred = arith.constant(1, type=T.i32)

                def _select_wave_tdm_value(*values):
                    if len(values) != _loader_waves:
                        raise ValueError(
                            f"expected {_loader_waves} wave-specialized TDM values, got {len(values)}"
                        )
                    _selected = values[-1]
                    for _sel_idx in range_constexpr(_loader_waves - 1):
                        _value_idx = _loader_waves - 2 - _sel_idx
                        _is_wave = arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _tdm_wave_id,
                            arith.constant(_value_idx, type=T.i32),
                        )
                        _selected = arith.select(_is_wave, values[_value_idx], _selected)
                    return _selected

                def _tdm_desc_lds_addr(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[1],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_lo(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[2],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_hi(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[3],
                        dynamic_position=[],
                    )

                _zero_k_base = arith.index(0)
                _scale_adv_i32 = arith.constant(scale_k_per_tile, type=T.i32)
                if _merge_gate_up_tdm:
                    _n_pair_init = _stage1_pair_row_base()
                    _data_adv_i32 = arith.constant(packed_tile_k_b * 16, type=T.i32)

                    _stages_b_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_b_pair(
                                lds_bg_pair_bufs[i],
                                _n_pair_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bs_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_bs_pair(
                                lds_bs_pair_bufs[i],
                                _n_pair_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]

                    _desc_b_init = make_desc_b_pair(
                        lds_bg_pair_bufs[0],
                        _n_pair_init,
                        _zero_k_base,
                    )
                    _desc_bs_init = make_desc_bs_pair(
                        lds_bs_pair_bufs[0],
                        _n_pair_init,
                        _zero_k_base,
                    )

                    _active_stage_lds_addr = [
                        _select_wave_tdm_value(
                            _stages_b_lds_addr[i],
                            _stages_bs_lds_addr[i],
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _active_addr_lo = _select_wave_tdm_value(
                        _tdm_desc_addr_lo(_desc_b_init),
                        _tdm_desc_addr_lo(_desc_bs_init),
                    )
                    _active_addr_hi = _select_wave_tdm_value(
                        _tdm_desc_addr_hi(_desc_b_init),
                        _tdm_desc_addr_hi(_desc_bs_init),
                    )
                    _active_dgroup1 = _select_wave_tdm_value(
                        _desc_b_init.dgroup1,
                        _desc_bs_init.dgroup1,
                    )
                    _active_adv_i32 = _select_wave_tdm_value(
                        _data_adv_i32,
                        _scale_adv_i32,
                    )
                else:
                    _eid_row = (
                        arith.index_cast(T.index, eid_i32)
                        * arith.index(int(2 * N))
                    )
                    _n_gate_init = _eid_row + blk_n
                    _n_up_init = _eid_row + blk_n + arith.index(int(N))
                    _data_adv_i32 = arith.constant(
                        packed_tile_k_b if is_fp4 else packed_tile_k_b * 16,
                        type=T.i32,
                    )

                    _stages_bg_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_b(
                                lds_bg_bufs[i],
                                _n_gate_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bu_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_b(
                                lds_bu_bufs[i],
                                _n_up_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bs_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_bs(
                                lds_bs_bufs[i],
                                _n_gate_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bsu_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_bs(
                                lds_bsu_bufs[i],
                                _n_up_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]

                    _desc_bg_init = make_desc_b(
                        lds_bg_bufs[0],
                        _n_gate_init,
                        _zero_k_base,
                    )
                    _desc_bu_init = make_desc_b(
                        lds_bu_bufs[0],
                        _n_up_init,
                        _zero_k_base,
                    )
                    _desc_bs_init = make_desc_bs(
                        lds_bs_bufs[0],
                        _n_gate_init,
                        _zero_k_base,
                    )
                    _desc_bsu_init = make_desc_bs(
                        lds_bsu_bufs[0],
                        _n_up_init,
                        _zero_k_base,
                    )

                    _active_stage_lds_addr = [
                        _select_wave_tdm_value(
                            _stages_bg_lds_addr[i],
                            _stages_bu_lds_addr[i],
                            _stages_bs_lds_addr[i],
                            _stages_bsu_lds_addr[i],
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _active_addr_lo = _select_wave_tdm_value(
                        _tdm_desc_addr_lo(_desc_bg_init),
                        _tdm_desc_addr_lo(_desc_bu_init),
                        _tdm_desc_addr_lo(_desc_bs_init),
                        _tdm_desc_addr_lo(_desc_bsu_init),
                    )
                    _active_addr_hi = _select_wave_tdm_value(
                        _tdm_desc_addr_hi(_desc_bg_init),
                        _tdm_desc_addr_hi(_desc_bu_init),
                        _tdm_desc_addr_hi(_desc_bs_init),
                        _tdm_desc_addr_hi(_desc_bsu_init),
                    )
                    _active_dgroup1 = _select_wave_tdm_value(
                        _desc_bg_init.dgroup1,
                        _desc_bu_init.dgroup1,
                        _desc_bs_init.dgroup1,
                        _desc_bsu_init.dgroup1,
                    )
                    _active_adv_i32 = _select_wave_tdm_value(
                        _data_adv_i32,
                        _data_adv_i32,
                        _scale_adv_i32,
                        _scale_adv_i32,
                    )

                def _issue_active_b_tdm_only(stage_idx, curr_addr_lo):
                    _if_loader = scf.IfOp(_is_loader_wave)
                    with ir.InsertionPoint(_if_loader.then_block):
                        _dg0 = vector.from_elements(T.vec(4, T.i32), [
                            _tdm_pred,
                            _active_stage_lds_addr[stage_idx],
                            curr_addr_lo,
                            _active_addr_hi,
                        ])
                        tdm_ops.tensor_load_2d(
                            tdm_ops.TDMDescriptor2D(_dg0, _active_dgroup1)
                        )
                        scf.YieldOp([])
                    _next_addr_lo = arith.addi(curr_addr_lo, _active_adv_i32)
                    return arith.select(
                        _is_loader_wave,
                        _next_addr_lo,
                        curr_addr_lo,
                    )

            # ── pipeline load helpers ─────────────────────────────────
            def _issue_b_tdm_only(k_base, buf_idx):
                if _merge_gate_up_tdm:
                    _n_pair = _stage1_pair_row_base()
                    tdm_ops.tensor_load_2d(
                        make_desc_b_pair(lds_bg_pair_bufs[buf_idx], _n_pair, k_base))
                    tdm_ops.tensor_load_2d(
                        make_desc_bs_pair(lds_bs_pair_bufs[buf_idx], _n_pair, k_base))
                else:
                    _eid_row = (arith.index_cast(T.index, eid_i32)
                                * arith.index(int(2 * N)))
                    _n_gate = _eid_row + blk_n
                    _n_up = _eid_row + blk_n + arith.index(int(N))
                    tdm_ops.tensor_load_2d(
                        make_desc_b(lds_bg_bufs[buf_idx], _n_gate, k_base))
                    tdm_ops.tensor_load_2d(
                        make_desc_b(lds_bu_bufs[buf_idx], _n_up, k_base))
                    tdm_ops.tensor_load_2d(
                        make_desc_bs(lds_bs_bufs[buf_idx], _n_gate, k_base))
                    tdm_ops.tensor_load_2d(
                        make_desc_bs(lds_bsu_bufs[buf_idx], _n_up, k_base))

            def _issue_scalar_loads(k_base, buf_idx):
                if _use_tdm_gather_a:
                    issue_a_load_tdm_gather(k_base, lds_ag_bufs[buf_idx])
                else:
                    issue_a_load(make_desc_a(k_base), lds_ag_bufs[buf_idx])
                issue_as_load(make_desc_as(k_base), lds_as_bufs[buf_idx])

            def _issue_all_loads(k_base, buf_idx):
                _issue_b_tdm_only(k_base, buf_idx)
                _issue_scalar_loads(k_base, buf_idx)

            def _compute_with_mid_loads(acg, acu, buf_idx, mid_load_callback=None):
                if _use_scheduled_compute:
                    return _compute_k_tile_scheduled(
                        acg, acu, buf_idx,
                        mid_compute_callback=mid_load_callback,
                    )
                return _compute_k_tile(
                    acg, acu, buf_idx,
                    mid_compute_callback=mid_load_callback,
                )

            # ── main K-dimension reduction ────────────────────────────
            if not _use_pipeline:
                if wave_specialized_tdm:
                    active_b_addr_lo = _active_addr_lo
                    for kt in range_constexpr(num_k_tiles):
                        k_base = fx.Index(kt * int(tile_k))
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            0, active_b_addr_lo)
                        _issue_scalar_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc_g, acc_u = _compute_k_tile(acc_g, acc_u, 0)
                        workgroup_barrier(use_cluster=use_cluster)
                else:
                    for kt in range_constexpr(num_k_tiles):
                        k_base = fx.Index(kt * int(tile_k))
                        _issue_all_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc_g, acc_u = _compute_k_tile(acc_g, acc_u, 0)
                        workgroup_barrier(use_cluster=use_cluster)
            else:
                # ── prologue ──
                if wave_specialized_tdm:
                    active_b_addr_lo = _active_addr_lo
                    for _pi in range_constexpr(pre_loaded):
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            _pi, active_b_addr_lo)
                        _issue_scalar_loads(fx.Index(_pi * int(tile_k)), _pi)
                else:
                    for _pi in range_constexpr(pre_loaded):
                        _issue_all_loads(fx.Index(_pi * int(tile_k)), _pi)
                pipeline_fence(outstanding=0, use_cluster=use_cluster)

                # ── main pipelined loop ──
                if loop_iters > 0:
                    if wave_specialized_tdm:
                        _init = list(acc_g) + list(acc_u) + [active_b_addr_lo]
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _ag = list(_st[:n_accs])
                            _au = list(_st[n_accs:2 * n_accs])
                            _cur_b_addr_lo = _st[2 * n_accs]
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _kt * fx.Index(int(tile_k))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)
                                _cur_b_addr_lo = _issue_active_b_tdm_only(
                                    _lb, _cur_b_addr_lo)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if _use_scheduled_compute:
                                    rocdl.sched_barrier(0)
                                _ag, _au = _compute_with_mid_loads(
                                    _ag,
                                    _au,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if _use_scheduled_compute:
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_ag) + list(_au) + [_cur_b_addr_lo]
                        acc_g = list(_res[:n_accs])
                        acc_u = list(_res[n_accs:2 * n_accs])
                        active_b_addr_lo = _res[2 * n_accs]
                    else:
                        _init = list(acc_g) + list(acc_u)
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _ag = list(_st[:n_accs])
                            _au = list(_st[n_accs:2 * n_accs])
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _kt * fx.Index(int(tile_k))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)
                                _issue_b_tdm_only(_kb, _lb)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if _use_scheduled_compute:
                                    rocdl.sched_barrier(0)
                                _ag, _au = _compute_with_mid_loads(
                                    _ag,
                                    _au,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if _use_scheduled_compute:
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_ag) + list(_au)
                        acc_g = list(_res[:n_accs])
                        acc_u = list(_res[n_accs:2 * n_accs])

                # ── post-loop fence ──
                if loop_iters > 0:
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                elif use_cluster:
                    gpu.cluster_barrier()

                # ── tail ──
                _tail_li = 0
                _tail_had_load = False
                for _ls, _cs, _out in _tail_plan:
                    if _out == -1:
                        if _tail_had_load:
                            pipeline_fence(outstanding=0,
                                           use_cluster=use_cluster)
                        if _use_scheduled_compute:
                            rocdl.sched_barrier(0)
                            acc_g, acc_u = _compute_k_tile_scheduled(
                                acc_g, acc_u, _cs)
                            _hot_loop_scheduler_scheduled()
                        else:
                            acc_g, acc_u = _compute_k_tile(
                                acc_g, acc_u, _cs)
                    else:
                        pipeline_fence_signal(outstanding=_out,
                                              use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)
                        if _ls is not None:
                            _tail_had_load = True
                            _tkb = fx.Index(
                                (_tail_start + pre_loaded + _tail_li)
                                * int(tile_k))
                            _tail_li += 1
                            if wave_specialized_tdm:
                                active_b_addr_lo = _issue_active_b_tdm_only(
                                    _ls, active_b_addr_lo)
                            else:
                                _issue_b_tdm_only(_tkb, _ls)

                            def _tail_mid_issue_scalar(_mid_kb=_tkb, _mid_ls=_ls):
                                _issue_scalar_loads(_mid_kb, _mid_ls)

                            if _use_scheduled_compute:
                                rocdl.sched_barrier(0)
                            acc_g, acc_u = _compute_with_mid_loads(
                                acc_g,
                                acc_u,
                                _cs,
                                _tail_mid_issue_scalar,
                            )
                            if _use_scheduled_compute:
                                _hot_loop_scheduler_scheduled()
                        else:
                            if _use_scheduled_compute:
                                rocdl.sched_barrier(0)
                                acc_g, acc_u = _compute_k_tile_scheduled(
                                    acc_g, acc_u, _cs)
                                _hot_loop_scheduler_scheduled()
                            else:
                                acc_g, acc_u = _compute_k_tile(
                                    acc_g, acc_u, _cs)

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            if bool(use_tdm_store):
                # ── TDM store epilogue: silu(gate)*up → LDS → global (contiguous sorted output) ──
                _scale_per_wm_s1 = []
                for _wm in range_constexpr(wmma_m_rep):
                    _m_off_val = _wm * WMMA_M
                    _row_local = warp_m_base + arith.index(_m_off_val) + lane16
                    _sorted_row = by * arith.index(int(tile_m)) + _row_local
                    _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                    _row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, _row_local),
                        arith.constant(int(route_tile_m), type=T.i32))
                    if bool(doweight_stage1):
                        _sorted_safe = arith.select(
                            _row_in_route, _sorted_i32,
                            arith.index_cast(T.i32,
                                by * arith.index(int(route_tile_m))))
                        _tw = buffer_ops.buffer_load(
                            tw_rsrc, _sorted_safe, vec_width=1, dtype=T.f32)
                        _sc = arith.select(
                            _row_in_route, _tw,
                            arith.constant(0.0, type=T.f32))
                    else:
                        _sc = arith.select(
                            _row_in_route,
                            arith.constant(1.0, type=T.f32),
                            arith.constant(0.0, type=T.f32))
                    _scale_per_wm_s1.append(_sc)

                if d_need_epilogue_fence_s1:
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                rocdl.sched_barrier(0)

                for _acc_idx, _vec_base, _m_off, _wn in _sub_tiles:
                    _wm_idx = _m_off // WMMA_M
                    _sc = _scale_per_wm_s1[_wm_idx]
                    _sub8g = _mxscale_extract_sub8(
                        acc_g[_acc_idx], _vec_base,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        ACC_VEC_SIZE=ACC_VEC_SIZE)
                    _sub8u = _mxscale_extract_sub8(
                        acc_u[_acc_idx], _vec_base,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        ACC_VEC_SIZE=ACC_VEC_SIZE)
                    _fused = []
                    for _vi in range_constexpr(8):
                        _vg = vector.extract(
                            _sub8g,
                            static_position=[_vi],
                            dynamic_position=[])
                        _vu = vector.extract(
                            _sub8u,
                            static_position=[_vi],
                            dynamic_position=[])
                        _y = silu(_vg) * _vu * _sc
                        _fused.append(_y)
                    _fused_sub8 = vector.from_elements(
                        T.vec(8, T.f32), _fused)
                    _imm = (_m_off * _lds_d_stride_elems_s1
                            + _wn * _n_col_d_elems_s1)
                    store_acc_vec8_to_lds(
                        d_lds_buffer_s1, d_lane_base_s1, _imm,
                        _fused_sub8, out_elem=out_elem_ty)

                rocdl.s_wait_dscnt(0)
                # TDM gather store: each warp stores its warp_tile_m rows
                # to scattered output positions tok*topk+slot.
                _warp_row_start = arith.index_cast(T.i32, warp_m_base)
                _warp_row_start_py = rocdl.readfirstlane(T.i32, _warp_row_start)
                _d_store_chunk = 8  # 32-bit gather mode
                _d_store_groups = (warp_tile_m + _d_store_chunk - 1) // _d_store_chunk
                _tokens_topk_dim1 = _get_tokens_topk_sgpr()
                for _dsi in range_constexpr(_d_store_groups):
                    _ds_start = _dsi * _d_store_chunk
                    _ds_cnt = min(_d_store_chunk, warp_tile_m - _ds_start)
                    # Global output row indices for this group
                    _ds_start_in_tile = _dsi * _d_store_chunk + rocdl.readfirstlane(
                        T.i32, arith.index_cast(T.i32, warp_m_base))
                    # Can't do runtime add on SGPR easily; use compile-time
                    # warp offset from wave_id. But warp_m_base is runtime.
                    # Instead, index _a_out_row_ids which is tile-global.
                    # warp_m_base = wave_m_idx * warp_tile_m (runtime index)
                    # We need _a_out_row_ids[warp_m_base + _ds_start + i]
                    # Since warp_m_base depends on wave_id, we use scf.if
                    # per warp to select the correct slice.
                    # Simpler: for num_warps_m = m_warp, unroll per warp:
                    _ds_indices = []
                    _ds_valids = []
                    for _wi in range_constexpr(int(m_warp)):
                        _tile_row = _wi * warp_tile_m + _ds_start
                        _warp_indices = _a_out_row_ids[_tile_row:_tile_row + _ds_cnt]
                        _warp_valids = _a_store_valids[_tile_row:_tile_row + _ds_cnt]
                        if _wi == 0:
                            _ds_indices = list(_warp_indices)
                            _ds_valids = list(_warp_valids)
                        else:
                            _is_this_warp = arith.cmpi(
                                arith.CmpIPredicate.eq,
                                rocdl.wave_id() % fx.Int32(int(n_warp * m_warp) // int(n_warp)),
                                fx.Int32(_wi))
                            # Actually wave_m_idx is the M warp index
                            _is_this_warp = arith.cmpi(
                                arith.CmpIPredicate.eq,
                                arith.index_cast(T.i32, wave_m_idx),
                                fx.Int32(_wi))
                            for _ii in range_constexpr(len(_ds_indices)):
                                _ds_indices[_ii] = arith.select(
                                    _is_this_warp,
                                    _warp_indices[_ii],
                                    _ds_indices[_ii])
                                _ds_valids[_ii] = arith.select(
                                    _is_this_warp,
                                    _warp_valids[_ii],
                                    _ds_valids[_ii])
                    # LDS offset within D buffer for this group
                    _ds_lds_off = arith.index(
                        _ds_start * lds_d_row_stride_s1) + d_warp_off_sgpr_s1
                    # Column offset in output
                    _col_byte_off = (blk_n + warp_n_off_sgpr_s1) * arith.index(elem_bytes_d_s1)
                    # For store direction: TDM ignores pad_enable, so we
                    # expand tile_dim0 to include padding so LDS read
                    # addresses align. tensor_dim0 stays at warp_tile_n so
                    # the extra pad elements hit OOB and are dropped.
                    _pad_elems = LDS_PAD_D_BYTES_s1 // elem_bytes_d_s1
                    _store_tile_w = warp_tile_n + _pad_elems
                    _ds_valid_count = _sum_i32_values(_ds_valids)
                    _zero_i32 = arith.constant(0, type=T.i32)
                    _has_store = arith.cmpi(arith.CmpIPredicate.sgt, _ds_valid_count, _zero_i32)
                    _if_store = scf.IfOp(_has_store)
                    with ir.InsertionPoint(_if_store.then_block):
                        _d_store_desc = tdm_ops.make_tensor_gather_descriptor(
                            global_ptr=arg_out,
                            lds_memref=base_ptr,
                            row_indices=_ds_indices,
                            row_width=_store_tile_w,
                            tensor_dim0=warp_tile_n,
                            tensor_dim1=_tokens_topk_dim1,
                            stride=N,
                            elem_bytes=elem_bytes_d_s1,
                            pad_interval=0,
                            pad_amount=0,
                            index_size=32,
                            gather_tile_dim1=_ds_valid_count,
                            lds_byte_offset=_ds_lds_off,
                            global_byte_offset=_col_byte_off,
                        )
                        tdm_ops.tensor_store_gather(_d_store_desc)
                        scf.YieldOp([])
                tdm_ops.tensor_wait(0)
            else:
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
                    num_valid_i32=num_valid_i32,
                    block_row_start=block_row_start,
                    sorted_rsrc=sorted_rsrc,
                    tw_rsrc=tw_rsrc,
                    out_rsrc=out_rsrc,
                    doweight_stage1=bool(doweight_stage1),
                    out_elem_ty=out_elem_ty,
                    load_gate_up_sub8=_load_gate_up_sub8,
                    silu_fn=silu,
                    ir=ir,
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
        arg_num_valid_ids: fx.Tensor,
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
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights, arg_num_valid_ids,
            i32_tokens_in, i32_inter_in, i32_k_in, i32_size_expert_ids_in,
        )
        _cluster_arg = (int(cluster_m), int(cluster_n), 1) if use_cluster else None
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=effective_waves_per_eu,
            ir=ir,
            cluster=_cluster_arg,
        )

    if expert_sched_mode:
        launch_mxscale_stage1_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
    DS_LOADS_PER_A_FRAG = int(fmt_cfg["DS_LOADS_PER_A_FRAG"])

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
    if int(num_buffers) not in (1, 2, 3, 4):
        raise ValueError(f"num_buffers must be 1, 2, 3, or 4, got {num_buffers}")
    if bool(use_tdm_store) and bool(accumulate):
        raise ValueError("use_tdm_store is not compatible with accumulate=True in moe mxscale stage2")
    use_cluster = int(cluster_m) > 1 or int(cluster_n) > 1
    if use_cluster:
        if int(cluster_m) * int(cluster_n) > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    num_warps = int(m_warp) * int(n_warp)
    if bool(wave_specialized_tdm):
        if num_warps < 2:
            raise ValueError(
                f"wave_specialized_tdm requires at least 2 waves (B + B_scale), got {num_warps}")
    _tdm_loader_waves = 2
    tdm_desc_num_warps = 1 if bool(wave_specialized_tdm) else num_warps
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 2

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

    # Pipeline calculations for multi-buffer
    _use_pipeline = int(num_buffers) >= 2
    if _use_pipeline:
        from kernels.gemm_common_gfx1250 import (
            pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
        )
        from kernels.pipeline_utils import make_tail_plan

        pre_loaded = int(num_buffers) - 1
        loop_iters = (num_k_tiles - pre_loaded) // int(num_buffers)
        _tail_start = loop_iters * int(num_buffers)
        extra = num_k_tiles - _tail_start - pre_loaded
        _B_TDM_PER_STEP = 1 if bool(wave_specialized_tdm) else 2
        _A_GATHER_GROUPS = (int(tile_m) + 8 - 1) // 8 if bool(use_tdm_gather) else 0
        if bool(use_tdm_gather) and bool(wave_specialized_tdm):
            _A_GATHER_TDM_PER_STEP = (
                (_A_GATHER_GROUPS + _tdm_loader_waves - 1)
                // _tdm_loader_waves
            )
        else:
            _A_GATHER_TDM_PER_STEP = _A_GATHER_GROUPS
        TDM_PER_STEP = _B_TDM_PER_STEP + _A_GATHER_TDM_PER_STEP
        _fence_outstanding = TDM_PER_STEP * (int(num_buffers) - 2)
        _base_tail_plan = make_tail_plan(int(num_buffers), pre_loaded, extra)
        _tail_plan = [
            (ls, cs, o * TDM_PER_STEP // 2 if o > 0 else o)
            for ls, cs, o in _base_tail_plan
        ]
        if num_k_tiles < int(num_buffers):
            raise ValueError(
                f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
                f"got {num_k_tiles}")
    from kernels.gemm_common_gfx1250 import workgroup_barrier

    alloc = SmemAllocator(
        None,
        arch=str(get_hip_arch()),
        global_sym_name=f"moe_mxscale_{data_format}_s2_single_g{int(bool(use_tdm_gather))}",
    )
    _nb = int(num_buffers)
    off_a_list, off_b_list, off_as_list, off_bs_list = [], [], [], []
    for _buf_i in range(_nb):
        _oa = alloc._align(alloc.ptr, 16)
        alloc.ptr = _oa + lds_a_data_bytes
        off_a_list.append(_oa)
        _ob = alloc._align(alloc.ptr, 16)
        alloc.ptr = _ob + lds_b_data_bytes
        off_b_list.append(_ob)
        _oas = alloc._align(alloc.ptr, 16)
        alloc.ptr = _oas + lds_a_scale_bytes
        off_as_list.append(_oas)
        _obs = alloc._align(alloc.ptr, 16)
        alloc.ptr = _obs + lds_b_scale_bytes
        off_bs_list.append(_obs)

    # TDM store epilogue: D output LDS layout
    LDS_PAD_D_BYTES = 16
    elem_bytes_d = 2  # f16/bf16
    if bool(use_tdm_store):
        from kernels.gemm_common_gfx1250 import (
            store_acc_vec8_to_lds,
        )
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        d_need_epilogue_fence = _use_pipeline
        if total_d_bytes > alloc.ptr:
            alloc.ptr = total_d_bytes

    _sub_tiles = _make_mxscale_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=is_fp4
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
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
        if inst_prefetch:
            if arith.cmpi(arith.CmpIPredicate.eq, rocdl.wave_id(),
                          arith.constant(0, type=T.i32)):
                _prefetch_lines = ["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 8, 1), 1"]
                for _pg in range_constexpr(10):
                    _prefetch_lines.append(
                        f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31")
                llvm_dialect.inline_asm(
                    None, [],
                    "\n".join(_prefetch_lines),
                    "", has_side_effects=True,
                )
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

        if use_cluster:
            _local_x, _local_y = gpu.compute_cluster_position()
            _a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                _local_x, _local_y, int(cluster_m), int(cluster_n))
        else:
            b_mcast_mask = 0

        base_ptr = alloc.get_base()
        lds_a_bufs = []
        lds_b_bufs = []
        lds_as_bufs = []
        lds_bs_bufs = []
        for _bi in range_constexpr(_nb):
            _sa = SmemPtr(base_ptr, off_a_list[_bi], T.i8, shape=(lds_a_data_bytes,))
            _sb = SmemPtr(base_ptr, off_b_list[_bi], T.i8, shape=(lds_b_data_bytes,))
            _sas = SmemPtr(base_ptr, off_as_list[_bi], T.i8, shape=(lds_a_scale_bytes,))
            _sbs = SmemPtr(base_ptr, off_bs_list[_bi], T.i8, shape=(lds_b_scale_bytes,))
            lds_a_bufs.append(get_op_result_or_value(_sa.get()))
            lds_b_bufs.append(get_op_result_or_value(_sb.get()))
            lds_as_bufs.append(get_op_result_or_value(_sas.get()))
            lds_bs_bufs.append(get_op_result_or_value(_sbs.get()))

        if bool(use_tdm_store):
            from kernels.gemm_common_gfx1250 import get_lds_memref
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(base_ptr, d_output_off, T.f16,
                             shape=(d_lds_f16_count,))
            d_lds_buffer = get_lds_memref(d_smem)
            warp_lds_off = (
                (wave_m_idx * arith.index(int(n_warp)) + wave_n_idx)
                * arith.index(_warp_d_elems)
            )
            d_lane_base = (
                warp_lds_off
                + lane16 * arith.index(_lds_d_stride_elems)
                + lane_kgrp * arith.index(4 * elem_bytes_d)
            )
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr = (
                wave_id_idx * arith.index(warp_d_bytes)
                + arith.index(d_output_off)
            )
            warp_m_off_sgpr = (
                (wave_id_idx / arith.index(int(n_warp)))
                * arith.index(warp_tile_m)
            )
            warp_n_off_sgpr = (
                (wave_id_idx % arith.index(int(n_warp)))
                * arith.index(warp_tile_n)
            )
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_out,
                lds_memref=base_ptr,
                global_offset=(
                    by * arith.index(int(tile_m)) + warp_m_off_sgpr,
                    blk_n + warp_n_off_sgpr,
                ),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(N_total, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

        _use_tdm_gather_a = bool(use_tdm_gather)
        _a_row_ids = []
        _a_row_valids = []
        _TDM_GATHER_CHUNK = 8
        _TDM_GATHER_GROUPS = (int(tile_m) + _TDM_GATHER_CHUNK - 1) // _TDM_GATHER_CHUNK
        _tokens_topk_sgpr = None

        def _sum_i32_values(_vals):
            _acc = arith.constant(0, type=T.i32)
            for _vi in range_constexpr(len(_vals)):
                _acc = _acc + _vals[_vi]
            return _acc

        def _get_tokens_topk_sgpr():
            nonlocal _tokens_topk_sgpr
            if _tokens_topk_sgpr is None:
                _m_i32 = arith.index_cast(
                    T.i32,
                    tokens_idx * arith.index(int(topk)),
                )
                _tokens_topk_sgpr = rocdl.readfirstlane(T.i32, _m_i32)
            return _tokens_topk_sgpr

        def _precompute_a_row_indices():
            _safe_row = arith.constant(0, type=T.i32)
            _one_i32 = arith.constant(1, type=T.i32)
            _zero_i32 = arith.constant(0, type=T.i32)
            for _ri in range_constexpr(int(tile_m)):
                _sorted_row = by * fx.Index(int(tile_m)) + fx.Index(_ri)
                _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                _row_in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Int32(_ri),
                    fx.Int32(int(route_tile_m)),
                )
                _row_in_valid = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    _sorted_i32,
                    num_valid_i32,
                )
                _row_ok = arith.andi(_row_in_route, _row_in_valid)
                _sorted_safe = arith.select(_row_ok, _sorted_i32, block_row_start)
                _fused = buffer_ops.buffer_load(sorted_rsrc, _sorted_safe, vec_width=1, dtype=T.i32)
                _tok = _fused & fx.Int32((1 << 24) - 1)
                _slot = _fused >> fx.Int32(24)
                _tok_ok = arith.cmpi(arith.CmpIPredicate.ult, _tok, i32_tokens_in)
                _slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, _slot, fx.Int32(0))
                _slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, _slot, c_topk_i32)
                _ts = _tok * c_topk_i32 + _slot
                _ts_ok = arith.andi(_tok_ok, arith.andi(_slot_ok0, _slot_ok1))
                _row_fully_ok = arith.andi(_row_ok, _ts_ok)
                _row_valid_i32 = arith.select(_row_fully_ok, _one_i32, _zero_i32)
                _a_row_valids.append(rocdl.readfirstlane(T.i32, _row_valid_i32))
                _ts_safe = arith.select(_row_fully_ok, _ts, _safe_row)
                _a_row_ids.append(rocdl.readfirstlane(T.i32, _ts_safe))

        def make_desc_a(k_base):
            return k_base / arith.index(PACK_FACTOR_A)

        def issue_a_load(k_packed_base, target_lds):
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
                    vector.store(v1, target_lds, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def issue_a_load_tdm_gather(k_base, target_lds):
            """Load stage2 A rows via TDM gather using token-slot row ids."""
            k_packed_base = k_base if PACK_FACTOR_A == 1 else k_base // fx.Index(PACK_FACTOR_A)
            _tokens_topk = _get_tokens_topk_sgpr()
            _zero_i32 = arith.constant(0, type=T.i32)
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _start = _gi * _TDM_GATHER_CHUNK
                _cnt = min(_TDM_GATHER_CHUNK, int(tile_m) - _start)
                _row_indices = _a_row_ids[_start:_start + _cnt]
                _valid_count = _sum_i32_values(_a_row_valids[_start:_start + _cnt])
                _lds_off = fx.Index(_start * lds_a_stride_bytes)
                _has_valid = arith.cmpi(arith.CmpIPredicate.sgt, _valid_count, _zero_i32)
                _issue_pred = _has_valid
                if wave_specialized_tdm:
                    _gather_owner = _gi % _tdm_loader_waves
                    _is_gather_loader = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(_gather_owner, type=T.i32),
                    )
                    _issue_pred = arith.andi(_issue_pred, _is_gather_loader)
                _if_issue = scf.IfOp(_issue_pred)
                with ir.InsertionPoint(_if_issue.then_block):
                    desc = tdm_ops.make_tensor_gather_descriptor(
                        global_ptr=arg_x,
                        lds_memref=target_lds,
                        row_indices=_row_indices,
                        row_width=int(packed_tile_k_a),
                        tensor_dim0=K_packed_a,
                        tensor_dim1=_tokens_topk,
                        stride=K_packed_a,
                        elem_bytes=1,
                        pad_interval=int(packed_tile_k_a) if LDS_PAD_A_BYTES > 0 else 0,
                        pad_amount=LDS_PAD_A_BYTES if LDS_PAD_A_BYTES > 0 else 0,
                        index_size=32,
                        gather_tile_dim1=_valid_count,
                        lds_byte_offset=_lds_off,
                        global_byte_offset=k_packed_base,
                    )
                    tdm_ops.tensor_load_gather(desc)
                    scf.YieldOp([])

        def make_desc_as(k_base):
            return k_base / arith.index(SCALE_BLOCK)

        def issue_as_load(k_scale_base, target_lds):
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
                    vector.store(v1, target_lds, [lds_idx], alignment=1)
                    scf.YieldOp([])

        def make_desc_b(n_off, k_base, target_lds):
            if is_fp4:
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_w, lds_memref=target_lds,
                    global_offset=(n_off, k_base / arith.index(PACK_FACTOR_B)),
                    tensor_shape=(int(tile_n), int(packed_tile_k_b)),
                    strides=(K_packed_b, 1),
                    tile_shape=(int(tile_n), int(packed_tile_k_b)),
                    elem_bytes=1, pad_interval=int(packed_tile_k_b), pad_amount=LDS_PAD_B_BYTES,
                    num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=target_lds,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(N_total // 16), int(K_packed_b * 16)),
                strides=(int(K_packed_b * 16), 1),
                tile_shape=(int(tile_n // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_bs(n_off, k_base, target_lds):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=target_lds,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)

        def issue_b_load(k_base, target_lds_b, target_lds_bs):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_off = eid_idx * n_idx + blk_n
            tdm_ops.tensor_load_2d(make_desc_b(n_off, k_base, target_lds_b))
            tdm_ops.tensor_load_2d(make_desc_bs(n_off, k_base, target_lds_bs))

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
            return _mxscale_precompute_rowmajor_b_scale_lane_bases(
                warp_n_base=warp_n_base,
                lane16=lane16,
                scale_k_per_tile=scale_k_per_tile,
                wmma_n_rep=wmma_n_rep,
                WMMA_N=WMMA_N,
                arith=arith,
                range_constexpr=range_constexpr,
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
            if _use_tdm_gather_a:
                _precompute_a_row_indices()
            a_data_bases = _precompute_a_data_bases()
            b_data_bases = _precompute_b_data_bases()
            as_bases = _precompute_a_scale_lane_bases()
            bs_bases = _precompute_b_scale_lane_bases()
            _use_scheduled_compute = _use_pipeline and not is_fp4
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm
            _front_wmma = _front_wm * wmma_n_rep
            _back_wmma = _back_wm * wmma_n_rep
            _b_frag_ds_loads_per_wn = 2 if is_a8w4 else 4
            _a_scale_ds_loads = wmma_m_rep if is_fp4 else (wmma_m_rep + 3) // 4
            _b_scale_ds_loads = b_scale_load_rep if is_fp4 else wmma_n_rep
            _bs_ds_loads = (
                wmma_n_rep * _b_frag_ds_loads_per_wn
                + _b_scale_ds_loads
                + _a_scale_ds_loads
            )

            # ── compute-tile helper ──────────────────────────────────
            def _compute_k_tile(accs_in, buf_idx, mid_compute_callback=None):
                _mid_emit_ks = 0
                if k_wmma_steps > 1:
                    _mid_emit_wm = wmma_m_rep - 1
                    _mid_emit_wn = wmma_n_rep - 1
                else:
                    _front_wm = (wmma_m_rep + 1) // 2
                    _front_wn = (wmma_n_rep + 1) // 2
                    if wmma_m_rep > 1:
                        _mid_emit_wm = _front_wm - 1
                        _mid_emit_wn = wmma_n_rep - 1
                    else:
                        _mid_emit_wm = 0
                        _mid_emit_wn = _front_wn - 1
                _did_mid = False
                for ks in range_constexpr(k_wmma_steps):
                    b_v = [load_b_frag(lds_b_bufs[buf_idx], b_data_bases, wn, ks)
                           for wn in range_constexpr(wmma_n_rep)]
                    if is_fp4:
                        as_v = [load_scale_i32(lds_as_bufs[buf_idx], as_bases[wm], ks)
                                for wm in range_constexpr(wmma_m_rep)]
                        bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[bi], ks)
                                for bi in range_constexpr(b_scale_load_rep)]
                    else:
                        as_v = load_scale_b128(lds_as_bufs[buf_idx], as_bases[0],
                                               wmma_m_rep, ks)
                        bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[wn], ks)
                                for wn in range_constexpr(wmma_n_rep)]
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_data_frag(lds_a_bufs[buf_idx],
                                                a_data_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            _mxscale_emit_wmma(
                                accs=accs_in, wm=wm, wn=wn,
                                a_frag=a_frag, b_frags=b_v,
                                a_scales=as_v, b_scales=bs_v,
                                is_fp4=is_fp4, is_a8w4=is_a8w4,
                                use_scale_opsel=False,
                                rocdl=rocdl, T=T,
                            )
                            if (
                                not _did_mid
                                and mid_compute_callback is not None
                                and ks == _mid_emit_ks
                                and wm == _mid_emit_wm
                                and wn == _mid_emit_wn
                            ):
                                mid_compute_callback()
                                _did_mid = True
                return accs_in

            def _load_b_and_scales(buf_idx, ks):
                b_v = [load_b_frag(lds_b_bufs[buf_idx], b_data_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
                if is_fp4:
                    as_v = [load_scale_i32(lds_as_bufs[buf_idx], as_bases[wm], ks)
                            for wm in range_constexpr(wmma_m_rep)]
                    bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[bi], ks)
                            for bi in range_constexpr(b_scale_load_rep)]
                else:
                    as_v = load_scale_b128(lds_as_bufs[buf_idx], as_bases[0],
                                           wmma_m_rep, ks)
                    bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[wn], ks)
                            for wn in range_constexpr(wmma_n_rep)]
                return b_v, bs_v, as_v

            def _emit_rows(accs_in, start_wm, a_frags, b_frags, a_scales, b_scales):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _mxscale_emit_wmma(
                            accs=accs_in,
                            wm=wm,
                            wn=wn,
                            a_frag=a_frags[frag_i],
                            b_frags=b_frags,
                            a_scales=a_scales,
                            b_scales=b_scales,
                            is_fp4=is_fp4,
                            is_a8w4=is_a8w4,
                            use_scale_opsel=False,
                            rocdl=rocdl,
                            T=T,
                        )

            def _a_streaming_compute(
                accs_in,
                buf_idx,
                b_frags,
                b_scales,
                a_scales,
                ks,
                next_bs_info=None,
                mid_compute_callback=None,
            ):
                current_accs = accs_in
                next_result = None
                a_frags_front = [
                    load_data_frag(lds_a_bufs[buf_idx], a_data_bases[wm], ks)
                    for wm in range_constexpr(_front_wm)
                ]
                _use_partial_drain = (
                    next_bs_info is not None
                    and _front_wm * wmma_n_rep >= 4
                )

                if _use_partial_drain:
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_b_and_scales(_next_buf_idx, _next_ks)
                    rocdl.s_wait_dscnt(_bs_ds_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_rows(current_accs, 0, a_frags_front, b_frags, a_scales, b_scales)

                if mid_compute_callback is not None:
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if _back_wm > 0:
                    a_frags_back = [
                        load_data_frag(
                            lds_a_bufs[buf_idx],
                            a_data_bases[_front_wm + h],
                            ks,
                        )
                        for h in range_constexpr(_back_wm)
                    ]
                    _back_drain = _bs_ds_loads if _use_partial_drain else 0
                    rocdl.s_wait_dscnt(_back_drain)
                    _emit_rows(
                        current_accs,
                        _front_wm,
                        a_frags_back,
                        b_frags,
                        a_scales,
                        b_scales,
                    )

                if _use_partial_drain:
                    return current_accs, next_result
                if next_bs_info is not None:
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_b_and_scales(_next_buf_idx, _next_ks)
                    return current_accs, next_result
                return current_accs

            def _compute_k_tile_scheduled(accs_in, buf_idx, mid_compute_callback=None):
                current_accs = list(accs_in)
                if k_wmma_steps == 1:
                    b_v, bs_v, as_v = _load_b_and_scales(buf_idx, 0)
                    current_accs = _a_streaming_compute(
                        current_accs,
                        buf_idx,
                        b_v,
                        bs_v,
                        as_v,
                        0,
                        mid_compute_callback=mid_compute_callback,
                    )
                else:
                    prev_b, prev_bs, prev_as = _load_b_and_scales(buf_idx, 0)
                    for ks in range_constexpr(k_wmma_steps - 1):
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        current_accs, (prev_b, prev_bs, prev_as) = _a_streaming_compute(
                            current_accs,
                            buf_idx,
                            prev_b,
                            prev_bs,
                            prev_as,
                            ks,
                            next_bs_info=(buf_idx, ks + 1),
                            mid_compute_callback=_mid_cb,
                        )
                    current_accs = _a_streaming_compute(
                        current_accs,
                        buf_idx,
                        prev_b,
                        prev_bs,
                        prev_as,
                        k_wmma_steps - 1,
                    )
                return current_accs

            def _hot_loop_scheduler_scheduled():
                if not _use_scheduled_compute:
                    return
                _front_a_loads = _front_wm * DS_LOADS_PER_A_FRAG
                _back_a_loads = _back_wm * DS_LOADS_PER_A_FRAG
                for _ks in range_constexpr(k_wmma_steps):
                    if _ks == 0:
                        rocdl.sched_dsrd(_bs_ds_loads + _front_a_loads)
                    else:
                        rocdl.sched_dsrd(_front_a_loads)
                    rocdl.sched_mfma(_front_wmma)
                    if _back_wmma > 0:
                        rocdl.sched_dsrd(_back_a_loads)
                        rocdl.sched_mfma(_back_wmma)
                    if _ks < k_wmma_steps - 1:
                        rocdl.sched_dsrd(_bs_ds_loads)
                rocdl.sched_barrier(0)

            if wave_specialized_tdm:
                _tdm_wave_id = rocdl.wave_id()
                _is_loader_wave = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    _tdm_wave_id,
                    arith.constant(_tdm_loader_waves, type=T.i32),
                )
                _tdm_pred = arith.constant(1, type=T.i32)

                def _select_wave_tdm_value(b_value, bs_value):
                    _wave_is_b = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(0, type=T.i32),
                    )
                    return arith.select(_wave_is_b, b_value, bs_value)

                def _tdm_desc_lds_addr(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[1],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_lo(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[2],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_hi(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[3],
                        dynamic_position=[],
                    )

                _eid = arith.index_cast(T.index, eid_i32)
                _n_init = _eid * n_idx + blk_n
                _zero_k_base = arith.index(0)
                _data_adv_i32 = arith.constant(
                    packed_tile_k_b if is_fp4 else packed_tile_k_b * 16,
                    type=T.i32,
                )
                _scale_adv_i32 = arith.constant(scale_k_per_tile, type=T.i32)

                _stages_b_lds_addr = [
                    _tdm_desc_lds_addr(
                        make_desc_b(
                            _n_init,
                            _zero_k_base,
                            lds_b_bufs[i],
                        )
                    )
                    for i in range_constexpr(_nb)
                ]
                _stages_bs_lds_addr = [
                    _tdm_desc_lds_addr(
                        make_desc_bs(
                            _n_init,
                            _zero_k_base,
                            lds_bs_bufs[i],
                        )
                    )
                    for i in range_constexpr(_nb)
                ]

                _desc_b_init = make_desc_b(
                    _n_init,
                    _zero_k_base,
                    lds_b_bufs[0],
                )
                _desc_bs_init = make_desc_bs(
                    _n_init,
                    _zero_k_base,
                    lds_bs_bufs[0],
                )

                _active_stage_lds_addr = [
                    _select_wave_tdm_value(
                        _stages_b_lds_addr[i],
                        _stages_bs_lds_addr[i],
                    )
                    for i in range_constexpr(_nb)
                ]
                _active_addr_lo = _select_wave_tdm_value(
                    _tdm_desc_addr_lo(_desc_b_init),
                    _tdm_desc_addr_lo(_desc_bs_init),
                )
                _active_addr_hi = _select_wave_tdm_value(
                    _tdm_desc_addr_hi(_desc_b_init),
                    _tdm_desc_addr_hi(_desc_bs_init),
                )
                _active_dgroup1 = _select_wave_tdm_value(
                    _desc_b_init.dgroup1,
                    _desc_bs_init.dgroup1,
                )
                _active_adv_i32 = _select_wave_tdm_value(
                    _data_adv_i32,
                    _scale_adv_i32,
                )
                def _issue_active_b_tdm_only(stage_idx, curr_addr_lo):
                    _if_loader = scf.IfOp(_is_loader_wave)
                    with ir.InsertionPoint(_if_loader.then_block):
                        _dg0 = vector.from_elements(T.vec(4, T.i32), [
                            _tdm_pred,
                            _active_stage_lds_addr[stage_idx],
                            curr_addr_lo,
                            _active_addr_hi,
                        ])
                        tdm_ops.tensor_load_2d(
                            tdm_ops.TDMDescriptor2D(_dg0, _active_dgroup1)
                        )
                        scf.YieldOp([])
                    _next_addr_lo = arith.addi(curr_addr_lo, _active_adv_i32)
                    return arith.select(
                        _is_loader_wave,
                        _next_addr_lo,
                        curr_addr_lo,
                    )

            # ── pipeline load helpers ─────────────────────────────────
            def _issue_b_tdm_only(k_base, buf_idx):
                _eid = arith.index_cast(T.index, eid_i32)
                _n = _eid * n_idx + blk_n
                tdm_ops.tensor_load_2d(
                    make_desc_b(_n, k_base, lds_b_bufs[buf_idx]))
                tdm_ops.tensor_load_2d(
                    make_desc_bs(_n, k_base, lds_bs_bufs[buf_idx]))

            def _issue_scalar_loads(k_base, buf_idx):
                if _use_tdm_gather_a:
                    issue_a_load_tdm_gather(k_base, lds_a_bufs[buf_idx])
                else:
                    issue_a_load(make_desc_a(k_base), lds_a_bufs[buf_idx])
                issue_as_load(make_desc_as(k_base), lds_as_bufs[buf_idx])

            def _issue_all_loads(k_base, buf_idx):
                _issue_b_tdm_only(k_base, buf_idx)
                _issue_scalar_loads(k_base, buf_idx)

            def _compute_with_mid_loads(accs_in, buf_idx, mid_load_callback=None):
                if _use_scheduled_compute:
                    return _compute_k_tile_scheduled(
                        accs_in, buf_idx,
                        mid_compute_callback=mid_load_callback,
                    )
                return _compute_k_tile(
                    accs_in, buf_idx,
                    mid_compute_callback=mid_load_callback,
                )

            # ── main K-dimension reduction ────────────────────────────
            if not _use_pipeline:
                # Single-buffer path (num_buffers=1)
                if wave_specialized_tdm:
                    active_b_addr_lo = _active_addr_lo
                    for kt in range_constexpr(num_k_tiles):
                        k_base = fx.Index(kt * int(tile_k))
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            0, active_b_addr_lo)
                        _issue_scalar_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc = _compute_k_tile(acc, 0)
                        workgroup_barrier(use_cluster=use_cluster)
                else:
                    for kt in range_constexpr(num_k_tiles):
                        k_base = fx.Index(kt * int(tile_k))
                        _issue_all_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc = _compute_k_tile(acc, 0)
                        workgroup_barrier(use_cluster=use_cluster)
            else:
                # Multi-buffer pipeline
                # ── prologue: pre-load first `pre_loaded` stages ──
                if wave_specialized_tdm:
                    active_b_addr_lo = _active_addr_lo
                    for _pi in range_constexpr(pre_loaded):
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            _pi, active_b_addr_lo)
                        _issue_scalar_loads(fx.Index(_pi * int(tile_k)), _pi)
                else:
                    for _pi in range_constexpr(pre_loaded):
                        _issue_all_loads(fx.Index(_pi * int(tile_k)), _pi)
                pipeline_fence(outstanding=0, use_cluster=use_cluster)

                # ── main pipelined loop ──
                if loop_iters > 0:
                    if wave_specialized_tdm:
                        _init = list(acc) + [active_b_addr_lo]
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _acc = list(_st[:n_accs])
                            _cur_b_addr_lo = _st[n_accs]
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _kt * fx.Index(int(tile_k))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)

                                _cur_b_addr_lo = _issue_active_b_tdm_only(
                                    _lb, _cur_b_addr_lo)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if _use_scheduled_compute:
                                    rocdl.sched_barrier(0)
                                _acc = _compute_with_mid_loads(
                                    _acc,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if _use_scheduled_compute:
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_acc) + [_cur_b_addr_lo]
                        acc = list(_res[:n_accs])
                        active_b_addr_lo = _res[n_accs]
                    else:
                        _init = list(acc)
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _acc = list(_st[:n_accs])
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _kt * fx.Index(int(tile_k))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)

                                _issue_b_tdm_only(_kb, _lb)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if _use_scheduled_compute:
                                    rocdl.sched_barrier(0)
                                _acc = _compute_with_mid_loads(
                                    _acc,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if _use_scheduled_compute:
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_acc)
                        acc = list(_res[:n_accs])

                # ── post-loop fence ──
                if loop_iters > 0:
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                elif use_cluster:
                    gpu.cluster_barrier()

                # ── tail ──
                _tail_li = 0
                _tail_had_load = False
                for _ls, _cs, _out in _tail_plan:
                    if _out == -1:
                        if _tail_had_load:
                            pipeline_fence(outstanding=0,
                                           use_cluster=use_cluster)
                        if _use_scheduled_compute:
                            rocdl.sched_barrier(0)
                            acc = _compute_k_tile_scheduled(acc, _cs)
                            _hot_loop_scheduler_scheduled()
                        else:
                            acc = _compute_k_tile(acc, _cs)
                    else:
                        pipeline_fence_signal(outstanding=_out,
                                              use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)
                        if _ls is not None:
                            _tail_had_load = True
                            _tkb = fx.Index(
                                (_tail_start + pre_loaded + _tail_li)
                                * int(tile_k))
                            _tail_li += 1

                            if wave_specialized_tdm:
                                active_b_addr_lo = _issue_active_b_tdm_only(
                                    _ls, active_b_addr_lo)
                            else:
                                _issue_b_tdm_only(_tkb, _ls)

                            def _tail_mid_issue_scalar(_mid_kb=_tkb, _mid_ls=_ls):
                                _issue_scalar_loads(_mid_kb, _mid_ls)

                            if _use_scheduled_compute:
                                rocdl.sched_barrier(0)
                            acc = _compute_with_mid_loads(
                                acc,
                                _cs,
                                _tail_mid_issue_scalar,
                            )
                            if _use_scheduled_compute:
                                _hot_loop_scheduler_scheduled()
                        else:
                            if _use_scheduled_compute:
                                rocdl.sched_barrier(0)
                                acc = _compute_k_tile_scheduled(acc, _cs)
                                _hot_loop_scheduler_scheduled()
                            else:
                                acc = _compute_k_tile(acc, _cs)

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            if bool(use_tdm_store):
                # ── TDM store epilogue: acc → LDS → global (contiguous sorted output) ──
                # Pre-compute per-wm row scale (weight × validity mask)
                _scale_per_wm = []
                for _wm in range_constexpr(wmma_m_rep):
                    _m_off_val = _wm * WMMA_M
                    _row_local = warp_m_base + arith.index(_m_off_val) + lane16
                    _sorted_row = by * arith.index(int(tile_m)) + _row_local
                    _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                    _row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, _row_local),
                        arith.constant(int(route_tile_m), type=T.i32))
                    _row_in_valid = arith.cmpi(
                        arith.CmpIPredicate.slt, _sorted_i32, num_valid_i32)
                    _row_ok = arith.andi(_row_in_route, _row_in_valid)
                    if bool(doweight_stage2):
                        _sorted_safe = arith.select(
                            _row_ok, _sorted_i32, block_row_start)
                        _tw = buffer_ops.buffer_load(
                            tw_rsrc, _sorted_safe, vec_width=1, dtype=T.f32)
                        _sc = arith.select(
                            _row_ok, _tw,
                            arith.constant(0.0, type=T.f32))
                    else:
                        _sc = arith.select(
                            _row_ok,
                            arith.constant(1.0, type=T.f32),
                            arith.constant(0.0, type=T.f32))
                    _scale_per_wm.append(_sc)

                if d_need_epilogue_fence:
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                rocdl.sched_barrier(0)

                for _acc_idx, _vec_base, _m_off, _wn in _sub_tiles:
                    _wm_idx = _m_off // WMMA_M
                    _sc = _scale_per_wm[_wm_idx]
                    _sub8 = _mxscale_extract_sub8(
                        acc[_acc_idx], _vec_base,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        ACC_VEC_SIZE=ACC_VEC_SIZE)
                    _scaled = []
                    for _vi in range_constexpr(8):
                        _v = vector.extract(
                            _sub8,
                            static_position=[_vi],
                            dynamic_position=[])
                        _scaled.append(_v * _sc)
                    _scaled_sub8 = vector.from_elements(
                        T.vec(8, T.f32), _scaled)
                    _imm = _m_off * _lds_d_stride_elems + _wn * _n_col_d_elems
                    store_acc_vec8_to_lds(
                        d_lds_buffer, d_lane_base, _imm, _scaled_sub8,
                        out_elem=out_elem_ty)

                rocdl.s_wait_dscnt(0)
                tdm_ops.tensor_store_2d(d_desc)
                tdm_ops.tensor_wait(0)
            else:
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
                    ir=ir,
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
        _cluster_arg = (int(cluster_m), int(cluster_n), 1) if use_cluster else None
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=effective_waves_per_eu,
            ir=ir,
            cluster=_cluster_arg,
        )

    if expert_sched_mode:
        launch_mxscale_stage2_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
            expert_sched_mode=expert_sched_mode,
        )
        if data_format == "bf16":
            return _bf16_to_f16_wrapper(exe, x_arg=1, w_arg=2)
        return exe

    if data_format in ("fp4", "fp8", "a8w4"):
        exe = _compile_stage1_mxscale_kernel_impl(
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
            expert_sched_mode=expert_sched_mode,
            num_buffers=int(num_buffers),
            use_tdm_gather=bool(use_tdm_gather),
            use_tdm_store=bool(use_tdm_store),
            inst_prefetch=bool(inst_prefetch),
            wave_specialized_tdm=bool(wave_specialized_tdm),
            cluster_m=int(cluster_m),
            cluster_n=int(cluster_n),
        )
        if data_format in ("fp8", "a8w4") and (int(inter_dim) % int(single_tile_n) == 0):
            return _Stage1GateUpPackedWrapper(
                exe,
                experts=int(experts),
                inter_dim=int(inter_dim),
                tile_n=int(single_tile_n),
                packed_cols_w=(int(model_dim) // 2) if data_format == "a8w4" else int(model_dim),
                packed_cols_scale=int(model_dim) // 32,
            )
        return exe
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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
            expert_sched_mode=expert_sched_mode,
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
            expert_sched_mode=expert_sched_mode,
            num_buffers=int(num_buffers),
            use_tdm_gather=bool(use_tdm_gather),
            use_tdm_store=bool(use_tdm_store),
            inst_prefetch=bool(inst_prefetch),
            wave_specialized_tdm=bool(wave_specialized_tdm),
            cluster_m=int(cluster_m),
            cluster_n=int(cluster_n),
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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
            expert_sched_mode=expert_sched_mode,
            num_buffers=int(num_buffers),
            use_tdm_gather=bool(use_tdm_gather),
            use_tdm_store=bool(use_tdm_store),
            inst_prefetch=bool(inst_prefetch),
            wave_specialized_tdm=bool(wave_specialized_tdm),
            cluster_m=int(cluster_m),
            cluster_n=int(cluster_n),
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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
            expert_sched_mode=expert_sched_mode,
            num_buffers=int(num_buffers),
            use_tdm_gather=bool(use_tdm_gather),
            use_tdm_store=bool(use_tdm_store),
            inst_prefetch=bool(inst_prefetch),
            wave_specialized_tdm=bool(wave_specialized_tdm),
            cluster_m=int(cluster_m),
            cluster_n=int(cluster_n),
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
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
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
                expert_sched_mode=expert_sched_mode,
                num_buffers=int(num_buffers),
                use_tdm_gather=bool(use_tdm_gather),
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
                cluster_m=int(cluster_m),
                cluster_n=int(cluster_n),
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
            expert_sched_mode=expert_sched_mode,
            num_buffers=int(num_buffers),
            use_tdm_gather=bool(use_tdm_gather),
            use_tdm_store=bool(use_tdm_store),
            inst_prefetch=bool(inst_prefetch),
            wave_specialized_tdm=bool(wave_specialized_tdm),
            cluster_m=int(cluster_m),
            cluster_n=int(cluster_n),
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
