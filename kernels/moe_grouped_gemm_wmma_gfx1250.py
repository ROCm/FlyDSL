# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""m-grouped masked MoE GEMM kernels for gfx1250 (WMMA).

Implements DeepEP-style **masked layout** grouped GEMM for MoE.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
    vector,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity

from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx,
    issue_tdm_loads,
    lds_load_b128_raw,
    lds_transpose_load_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
)
from kernels.pipeline_utils import make_tail_plan

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32
LDS_PAD_A = 8
LDS_PAD_B = 8


def _require_gfx1250() -> None:
    arch = str(get_hip_arch())
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"Expected gfx1250 architecture, got {arch!r}")


def _align_up(value: int, align: int) -> int:
    if value % align == 0:
        return value
    return (value + align - 1) // align * align


def _silu_f32(x):
    t = x * (-1.4426950408889634)
    emu = rocdl.exp2(T.f32, t)
    den = 1.0 + emu
    sig = rocdl.rcp(T.f32, den)
    return x * sig


def _make_tail_plan_scaled(num_buffers, pre_loaded, extra, tdm_loads_per_step):
    base = make_tail_plan(num_buffers, pre_loaded, extra)
    return [
        (ls, cs, o * tdm_loads_per_step // 2 if o > 0 else o)
        for ls, cs, o in base
    ]


@functools.lru_cache(maxsize=64)
def compile_moe_grouped_gemm1_masked(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    max_m: int,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = 64,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    out_dtype: str = "f16",
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    expert_sched_mode: bool = True,
):
    _require_gfx1250()

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {num_buffers}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")

    is_f16 = (in_dtype == "fp16")
    elem_bytes = 2
    elem_bytes_d = 2

    K = int(model_dim)
    N_inter = int(inter_dim)
    N_full = 2 * N_inter
    E = int(experts)
    max_M = int(max_m)

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2, got {tile_k}")
    if N_inter % tile_n != 0:
        raise ValueError(f"inter_dim must be divisible by tile_n={tile_n}, got {N_inter}")

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles>={num_buffers}, "
            f"got {num_k_tiles}"
        )

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    k_wmma_steps = tile_k // WMMA_K

    # NOTE: T.f16 / T.bf16 / T.vec(...) require an MLIR Context to be active,
    # so they must be evaluated inside the kernel body (lazy). Only ODS-built
    # ops like wmma_f32_16x16x32_f16 are safe to capture here.
    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16

    def _elem_ty():
        return T.f16 if is_f16 else T.bf16

    def _out_elem_ty():
        return T.f16 if out_dtype == "f16" else T.bf16

    TDM_LOADS_PER_STEP = 3

    lds_a_stride = tile_k + LDS_PAD_A
    lds_b_stride = tile_n + LDS_PAD_B
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    lds_b_elems = tile_k * lds_b_stride + LDS_PAD_B

    gpu_arch = str(get_hip_arch())
    stage_layout = SmemAllocator(None, arch=gpu_arch, global_sym_name="moe_grp_s1_layout")
    off_bg = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = off_bg + lds_b_elems * elem_bytes
    off_bu = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = off_bu + lds_b_elems * elem_bytes
    off_a = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = off_a + lds_a_elems * elem_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    tail_plan = _make_tail_plan_scaled(
        num_buffers, pre_loaded, extra, TDM_LOADS_PER_STEP
    )

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"moe_grp_s1_{in_dtype}_{out_dtype}_{tile_m}x{tile_n}x{tile_k}_"
            f"{m_warp}x{n_warp}_{num_buffers}buf_arena"
        ),
    )
    arena_alloc.ptr = stage_pitch_bytes * num_buffers
    check_smem_capacity(arena_alloc.ptr, gpu_arch)

    stage_bg_offsets = [i * stage_pitch_bytes + off_bg for i in range(num_buffers)]
    stage_bu_offsets = [i * stage_pitch_bytes + off_bu for i in range(num_buffers)]
    stage_a_offsets = [i * stage_pitch_bytes + off_a for i in range(num_buffers)]

    @flyc.kernel
    def kernel_moe_grp_s1(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_max_m: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_experts: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        _ = (i32_max_m, i32_k_in, i32_experts, i32_inter_in)

        elem_ty = _elem_ty()
        out_elem_ty = _out_elem_ty()

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = gpu.block_id("z")

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        g_idx = bz

        masked_m_rsrc = buffer_ops.create_buffer_resource(arg_masked_m, max_size=True)
        valid_m_g_i32 = buffer_ops.buffer_load(
            masked_m_rsrc, arith.index_cast(T.i32, g_idx),
            vec_width=1, dtype=T.i32,
        )
        valid_m_g_i32 = rocdl.readfirstlane(T.i32, valid_m_g_i32)

        blk_m_i32 = arith.index_cast(T.i32, blk_m)
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, blk_m_i32, valid_m_g_i32)

        x_bytes = arith.index(E * max_M * K * elem_bytes)
        w_bytes = arith.index(E * N_full * K * elem_bytes)
        y_bytes = arith.index(E * max_M * N_inter * elem_bytes_d)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_bytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_bytes)
        y_rsrc = buffer_ops.create_buffer_resource(arg_y, max_size=False, num_records_bytes=y_bytes)
        _ = x_rsrc
        _ = w_rsrc

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1),
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx = fx.get(thr_coord, 0)
        wave_n_idx = fx.get(thr_coord, 1)
        lane_kgrp = fx.get(thr_coord, 2)
        lane16 = fx.get(thr_coord, 3)
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        arena_base_ptr = arena_alloc.get_base()
        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_offsets[i], elem_ty, shape=(lds_a_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_bg = [
            SmemPtr(arena_base_ptr, stage_bg_offsets[i], elem_ty, shape=(lds_b_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_bu = [
            SmemPtr(arena_base_ptr, stage_bu_offsets[i], elem_ty, shape=(lds_b_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_bg_mem = [stages_bg[i].get() for i in range_constexpr(num_buffers)]
        stages_bu_mem = [stages_bu[i].get() for i in range_constexpr(num_buffers)]
        stages_a_idx = [extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)]
        stages_bg_idx = [extract_lds_base_idx(stages_bg[i]) for i in range_constexpr(num_buffers)]
        stages_bu_idx = [extract_lds_base_idx(stages_bu[i]) for i in range_constexpr(num_buffers)]

        x_outer_off = g_idx * arith.index(max_M) + blk_m
        bg_outer_off = g_idx * arith.index(N_full) + blk_n
        bu_outer_off = g_idx * arith.index(N_full) + arith.index(N_inter) + blk_n

        def make_desc_a(lds_a_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_x, lds_memref=lds_a_mem_ref,
                global_offset=(x_outer_off, k_base),
                tensor_shape=(E * max_M, K), strides=(K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps,
            )

        def make_desc_bg(lds_b_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem_ref,
                global_offset=(bg_outer_off, k_base),
                tensor_shape=(E * N_full, K), strides=(K, 1),
                tile_shape=(tile_n, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_B,
                num_warps=num_warps,
            )

        def make_desc_bu(lds_b_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem_ref,
                global_offset=(bu_outer_off, k_base),
                tensor_shape=(E * N_full, K), strides=(K, 1),
                tile_shape=(tile_n, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_B,
                num_warps=num_warps,
            )

        def _precompute_a_lane_bases(lds_base_idx):
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride * elem_bytes)
            k_lane_off = lane_kgrp * arith.index(8 * elem_bytes)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = (
                    row_stride_off
                    + arith.index(wm * WMMA_M * lds_a_stride * elem_bytes)
                    + k_lane_off
                )
                bases.append(a_base)
            return lds_base_idx, bases

        def load_a_frag(a_lds_base_idx, a_lane_base, ks):
            vec8_ty = ir.VectorType.get([8], elem_ty)
            k_byte_off = arith.index(ks * WMMA_K * elem_bytes)
            off0 = a_lane_base + k_byte_off
            off1 = a_lane_base + k_byte_off + arith.index(32)
            raw0 = lds_load_b128_raw(a_lds_base_idx, off0)
            raw1 = lds_load_b128_raw(a_lds_base_idx, off1)
            v0 = vector.bitcast(vec8_ty, raw0)
            v1 = vector.bitcast(vec8_ty, raw1)
            return vector.shuffle(v0, v1, list(range(16)))

        def _precompute_b_lane_bases(lds_base_idx):
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (
                (lane_kgrp * arith.index(8) + lane8)
                * arith.index(lds_b_stride * elem_bytes)
            )
            n_lane_off = lane_ngrp * arith.index(8 * elem_bytes)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = (
                    (warp_n_base + arith.index(wn * WMMA_N))
                    * arith.index(elem_bytes)
                    + n_lane_off
                )
                bases.append(k_lane_off + n_col)
            return lds_base_idx, bases

        def load_b_frag_tr(lds_base_idx, b_lane_base, ks):
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride * elem_bytes
                elem_off = b_lane_base + arith.index(k_row_off)
                v = lds_transpose_load_raw(vec8_ty, lds_base_idx, elem_off)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        def compute_tile(acc_g_in, acc_u_in,
                         lds_a_idx, lds_bg_idx, lds_bu_idx,
                         mid_compute_callback=None):
            cur_g = list(acc_g_in)
            cur_u = list(acc_u_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            bg_buf, b_bases = _precompute_b_lane_bases(lds_bg_idx)
            bu_buf, _ = _precompute_b_lane_bases(lds_bu_idx)

            for ks in range_constexpr(k_wmma_steps):
                bg_frags = [load_b_frag_tr(bg_buf, b_bases[wn], ks)
                            for wn in range_constexpr(wmma_n_rep)]
                bu_frags = [load_b_frag_tr(bu_buf, b_bases[wn], ks)
                            for wn in range_constexpr(wmma_n_rep)]
                if const_expr(mid_compute_callback is not None) and ks == 0:
                    rocdl.sched_barrier(0)
                    mid_compute_callback()
                rocdl.s_wait_dscnt(0)
                for wm in range_constexpr(wmma_m_rep):
                    a_frag = load_a_frag(a_buf, a_bases[wm], ks)
                    rocdl.s_wait_dscnt(0)
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        cur_g[idx] = wmma_op(
                            T.vec(8, T.f32),
                            bg_frags[wn], a_frag, cur_g[idx],
                            signA=False, signB=False, modC=0,
                            reuseA=False, reuseB=False,
                        ).result
                        cur_u[idx] = wmma_op(
                            T.vec(8, T.f32),
                            bu_frags[wn], a_frag, cur_u[idx],
                            signA=False, signB=False, modC=0,
                            reuseA=False, reuseB=False,
                        ).result
            return cur_g, cur_u

        stages_a_lds_addr = []
        stages_bg_lds_addr = []
        stages_bu_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(vector.extract(
                make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))
            stages_bg_lds_addr.append(vector.extract(
                make_desc_bg(stages_bg_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))
            stages_bu_lds_addr.append(vector.extract(
                make_desc_bu(stages_bu_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))

        desc_a_init = make_desc_a(stages_a_mem[0], arith.index(0))
        desc_bg_init = make_desc_bg(stages_bg_mem[0], arith.index(0))
        desc_bu_init = make_desc_bu(stages_bu_mem[0], arith.index(0))

        addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
        addr_lo_bg = vector.extract(desc_bg_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_bg = vector.extract(desc_bg_init.dgroup0, static_position=[3], dynamic_position=[])
        addr_lo_bu = vector.extract(desc_bu_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_bu = vector.extract(desc_bu_init.dgroup0, static_position=[3], dynamic_position=[])
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_bg = desc_bg_init.dgroup1
        dgroup1_bu = desc_bu_init.dgroup1

        adv_a_i32 = arith.constant(tile_k * elem_bytes, type=T.i32)
        adv_b_i32 = arith.constant(tile_k * elem_bytes, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs_g = [acc_zero] * n_accs
        accs_u = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_in_valid)
        with ir.InsertionPoint(_if_blk.then_block):
            cur_lo_a = addr_lo_a
            cur_lo_bg = addr_lo_bg
            cur_lo_bu = addr_lo_bu

            for i in range_constexpr(pre_loaded):
                dg0_a = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_a_lds_addr[i], cur_lo_a, addr_hi_a])
                dg0_bg = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_bg_lds_addr[i], cur_lo_bg, addr_hi_bg])
                dg0_bu = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_bu_lds_addr[i], cur_lo_bu, addr_hi_bu])
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a))
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_bg, dgroup1_bg))
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_bu, dgroup1_bu))
                cur_lo_a = arith.addi(cur_lo_a, adv_a_i32)
                cur_lo_bg = arith.addi(cur_lo_bg, adv_b_i32)
                cur_lo_bu = arith.addi(cur_lo_bu, adv_b_i32)

            pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

            _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

            if const_expr(loop_iters > 0):
                init_args = list(accs_g) + list(accs_u) + [cur_lo_a, cur_lo_bg, cur_lo_bu]
                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    in_g = list(state[:n_accs])
                    in_u = list(state[n_accs:2 * n_accs])
                    cur_a = state[2 * n_accs]
                    cur_bg = state[2 * n_accs + 1]
                    cur_bu = state[2 * n_accs + 2]
                    _ = loop_iter

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers
                        pipeline_fence_signal(outstanding=_fence_outstanding)
                        pipeline_fence_wait()
                        addr_box = [cur_a, cur_bg, cur_bu]

                        def _mid_tdm(_ls=load_stage, _ab=addr_box):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0], addr_hi_a])
                            dg0_bg = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_bg_lds_addr[_ls],
                                _ab[1], addr_hi_bg])
                            dg0_bu = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_bu_lds_addr[_ls],
                                _ab[2], addr_hi_bu])
                            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a))
                            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_bg, dgroup1_bg))
                            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_bu, dgroup1_bu))
                            _ab[0] = arith.addi(_ab[0], adv_a_i32)
                            _ab[1] = arith.addi(_ab[1], adv_b_i32)
                            _ab[2] = arith.addi(_ab[2], adv_b_i32)

                        rocdl.sched_barrier(0)
                        in_g, in_u = compute_tile(
                            in_g, in_u,
                            stages_a_idx[buf_idx],
                            stages_bg_idx[buf_idx],
                            stages_bu_idx[buf_idx],
                            mid_compute_callback=_mid_tdm,
                        )
                        cur_a, cur_bg, cur_bu = addr_box[0], addr_box[1], addr_box[2]
                        rocdl.sched_barrier(0)

                    results = yield list(in_g) + list(in_u) + [cur_a, cur_bg, cur_bu]

                accs_g = list(results[:n_accs])
                accs_u = list(results[n_accs:2 * n_accs])
                cur_lo_a = results[2 * n_accs]
                cur_lo_bg = results[2 * n_accs + 1]
                cur_lo_bu = results[2 * n_accs + 2]

            if const_expr(loop_iters > 0):
                pipeline_fence(outstanding=0)

            _tail_had_load = False
            for _load_stage, _compute_stage, _outstanding in tail_plan:
                if const_expr(_outstanding == -1):
                    if const_expr(_tail_had_load):
                        pipeline_fence(outstanding=0)
                    accs_g, accs_u = compute_tile(
                        accs_g, accs_u,
                        stages_a_idx[_compute_stage],
                        stages_bg_idx[_compute_stage],
                        stages_bu_idx[_compute_stage],
                    )
                else:
                    pipeline_fence_signal(outstanding=_outstanding)
                    pipeline_fence_wait()
                    _tail_mid_cb = None
                    if const_expr(_load_stage is not None):
                        _tail_had_load = True
                        _tail_box = [cur_lo_a, cur_lo_bg, cur_lo_bu]

                        def _tail_mid(_ls=_load_stage, _ab=_tail_box):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0], addr_hi_a])
                            dg0_bg = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_bg_lds_addr[_ls],
                                _ab[1], addr_hi_bg])
                            dg0_bu = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_bu_lds_addr[_ls],
                                _ab[2], addr_hi_bu])
                            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a))
                            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_bg, dgroup1_bg))
                            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_bu, dgroup1_bu))
                            _ab[0] = arith.addi(_ab[0], adv_a_i32)
                            _ab[1] = arith.addi(_ab[1], adv_b_i32)
                            _ab[2] = arith.addi(_ab[2], adv_b_i32)

                        _tail_mid_cb = _tail_mid
                    rocdl.sched_barrier(0)
                    accs_g, accs_u = compute_tile(
                        accs_g, accs_u,
                        stages_a_idx[_compute_stage],
                        stages_bg_idx[_compute_stage],
                        stages_bu_idx[_compute_stage],
                        mid_compute_callback=_tail_mid_cb,
                    )
                    if const_expr(_load_stage is not None):
                        cur_lo_a = _tail_box[0]
                        cur_lo_bg = _tail_box[1]
                        cur_lo_bu = _tail_box[2]

            rocdl.sched_barrier(0)
            valid_m_g_idx = arith.index_cast(T.index, valid_m_g_i32)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    g_vec = accs_g[idx]
                    u_vec = accs_u[idx]
                    row_local = warp_m_base + arith.index(wm * WMMA_M) + lane16
                    row_global_m = blk_m + row_local
                    row_in_valid = arith.cmpi(arith.CmpIPredicate.ult,
                                              row_global_m, valid_m_g_idx)
                    col_base_local = (warp_n_base + arith.index(wn * WMMA_N)
                                      + lane_kgrp * arith.index(8))
                    col_base_global = blk_n + col_base_local
                    col_in_valid = arith.cmpi(arith.CmpIPredicate.ult,
                                              col_base_global,
                                              arith.index(N_inter))
                    out_pred = arith.andi(row_in_valid, col_in_valid)
                    _if_out = scf.IfOp(out_pred)
                    with ir.InsertionPoint(_if_out.then_block):
                        for vi in range_constexpr(8):
                            g_v = vector.extract(g_vec, static_position=[vi], dynamic_position=[])
                            u_v = vector.extract(u_vec, static_position=[vi], dynamic_position=[])
                            y_f32 = _silu_f32(g_v) * u_v
                            y_out = arith.trunc_f(out_elem_ty, y_f32)
                            row_out_full = (g_idx * arith.index(max_M) + row_global_m)
                            col_out = col_base_global + arith.index(vi)
                            y_off = (row_out_full * arith.index(N_inter) + col_out)
                            buffer_ops.buffer_store(
                                y_out, y_rsrc,
                                arith.index_cast(T.i32, y_off),
                            )
                        scf.YieldOp([])
            scf.YieldOp([])

    cache_tag = (in_dtype, out_dtype, K, N_inter, E, max_M,
                 tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers,
                 waves_per_eu, expert_sched_mode)

    @flyc.jit
    def launch_moe_grp_s1(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_max_m: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_experts: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        gx = arith.index(_align_up(max_M, tile_m) // tile_m)
        gy = arith.index(N_inter // tile_n)
        gz = arith.index(E)

        launcher = kernel_moe_grp_s1(
            arg_y, arg_x, arg_w, arg_masked_m,
            i32_max_m, i32_inter_in, i32_k_in, i32_experts,
        )
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                if waves_per_eu is not None and int(waves_per_eu) >= 1:
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), int(waves_per_eu))

        launcher.launch(
            grid=(_raw(gx), _raw(gy), _raw(gz)),
            block=(block_threads, 1, 1),
            stream=stream,
        )

        if expert_sched_mode:
            launch_moe_grp_s1.compile_hints["llvm_options"] = {
                "amdgpu-expert-scheduling-mode": True,
            }

    return launch_moe_grp_s1


@functools.lru_cache(maxsize=64)
def compile_moe_grouped_gemm2_masked(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    max_m: int,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = 64,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    out_dtype: str = "f16",
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    expert_sched_mode: bool = True,
):
    _require_gfx1250()

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {num_buffers}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")

    is_f16 = (in_dtype == "fp16")
    elem_bytes = 2
    elem_bytes_d = 2

    K_out = int(model_dim)
    K = int(inter_dim)
    E = int(experts)
    max_M = int(max_m)

    if K % tile_k != 0:
        raise ValueError(f"inter_dim must be divisible by tile_k={tile_k}, got {K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2, got {tile_k}")
    if K_out % tile_n != 0:
        raise ValueError(f"model_dim must be divisible by tile_n={tile_n}, got {K_out}")

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles>={num_buffers}, "
            f"got {num_k_tiles}"
        )

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    k_wmma_steps = tile_k // WMMA_K

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16

    def _elem_ty():
        return T.f16 if is_f16 else T.bf16

    def _out_elem_ty():
        return T.f16 if out_dtype == "f16" else T.bf16

    TDM_LOADS_PER_STEP = 2

    lds_a_stride = tile_k + LDS_PAD_A
    lds_b_stride = tile_n + LDS_PAD_B
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    lds_b_elems = tile_k * lds_b_stride + LDS_PAD_B

    gpu_arch = str(get_hip_arch())
    stage_layout = SmemAllocator(None, arch=gpu_arch, global_sym_name="moe_grp_s2_layout")
    off_b = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = off_b + lds_b_elems * elem_bytes
    off_a = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = off_a + lds_a_elems * elem_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    tail_plan = _make_tail_plan_scaled(
        num_buffers, pre_loaded, extra, TDM_LOADS_PER_STEP
    )

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"moe_grp_s2_{in_dtype}_{out_dtype}_{tile_m}x{tile_n}x{tile_k}_"
            f"{m_warp}x{n_warp}_{num_buffers}buf_arena"
        ),
    )
    arena_alloc.ptr = stage_pitch_bytes * num_buffers
    check_smem_capacity(arena_alloc.ptr, gpu_arch)

    stage_b_offsets = [i * stage_pitch_bytes + off_b for i in range(num_buffers)]
    stage_a_offsets = [i * stage_pitch_bytes + off_a for i in range(num_buffers)]

    @flyc.kernel
    def kernel_moe_grp_s2(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_max_m: fx.Int32,
        i32_k_out_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_experts: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        _ = (i32_max_m, i32_k_out_in, i32_inter_in, i32_experts)

        elem_ty = _elem_ty()
        out_elem_ty = _out_elem_ty()

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = gpu.block_id("z")

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        g_idx = bz

        masked_m_rsrc = buffer_ops.create_buffer_resource(arg_masked_m, max_size=True)
        valid_m_g_i32 = buffer_ops.buffer_load(
            masked_m_rsrc, arith.index_cast(T.i32, g_idx),
            vec_width=1, dtype=T.i32,
        )
        valid_m_g_i32 = rocdl.readfirstlane(T.i32, valid_m_g_i32)

        blk_m_i32 = arith.index_cast(T.i32, blk_m)
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, blk_m_i32, valid_m_g_i32)

        x_bytes = arith.index(E * max_M * K * elem_bytes)
        w_bytes = arith.index(E * K_out * K * elem_bytes)
        y_bytes = arith.index(E * max_M * K_out * elem_bytes_d)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_bytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_bytes)
        y_rsrc = buffer_ops.create_buffer_resource(arg_y, max_size=False, num_records_bytes=y_bytes)
        _ = x_rsrc
        _ = w_rsrc

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1),
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx = fx.get(thr_coord, 0)
        wave_n_idx = fx.get(thr_coord, 1)
        lane_kgrp = fx.get(thr_coord, 2)
        lane16 = fx.get(thr_coord, 3)
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        arena_base_ptr = arena_alloc.get_base()
        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_offsets[i], elem_ty, shape=(lds_a_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_offsets[i], elem_ty, shape=(lds_b_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_a_idx = [extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i]) for i in range_constexpr(num_buffers)]

        x_outer_off = g_idx * arith.index(max_M) + blk_m
        b_outer_off = g_idx * arith.index(K_out) + blk_n

        def make_desc_a(lds_a_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_x, lds_memref=lds_a_mem_ref,
                global_offset=(x_outer_off, k_base),
                tensor_shape=(E * max_M, K), strides=(K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps,
            )

        def make_desc_b(lds_b_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem_ref,
                global_offset=(b_outer_off, k_base),
                tensor_shape=(E * K_out, K), strides=(K, 1),
                tile_shape=(tile_n, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_B,
                num_warps=num_warps,
            )

        def _precompute_a_lane_bases(lds_base_idx):
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride * elem_bytes)
            k_lane_off = lane_kgrp * arith.index(8 * elem_bytes)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = (
                    row_stride_off
                    + arith.index(wm * WMMA_M * lds_a_stride * elem_bytes)
                    + k_lane_off
                )
                bases.append(a_base)
            return lds_base_idx, bases

        def load_a_frag(a_lds_base_idx, a_lane_base, ks):
            vec8_ty = ir.VectorType.get([8], elem_ty)
            k_byte_off = arith.index(ks * WMMA_K * elem_bytes)
            off0 = a_lane_base + k_byte_off
            off1 = a_lane_base + k_byte_off + arith.index(32)
            raw0 = lds_load_b128_raw(a_lds_base_idx, off0)
            raw1 = lds_load_b128_raw(a_lds_base_idx, off1)
            v0 = vector.bitcast(vec8_ty, raw0)
            v1 = vector.bitcast(vec8_ty, raw1)
            return vector.shuffle(v0, v1, list(range(16)))

        def _precompute_b_lane_bases(lds_base_idx):
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (
                (lane_kgrp * arith.index(8) + lane8)
                * arith.index(lds_b_stride * elem_bytes)
            )
            n_lane_off = lane_ngrp * arith.index(8 * elem_bytes)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = (
                    (warp_n_base + arith.index(wn * WMMA_N))
                    * arith.index(elem_bytes)
                    + n_lane_off
                )
                bases.append(k_lane_off + n_col)
            return lds_base_idx, bases

        def load_b_frag_tr(lds_base_idx, b_lane_base, ks):
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride * elem_bytes
                elem_off = b_lane_base + arith.index(k_row_off)
                v = lds_transpose_load_raw(vec8_ty, lds_base_idx, elem_off)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        def compute_tile(acc_in, lds_a_idx, lds_b_idx, mid_compute_callback=None):
            cur = list(acc_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_idx)
            for ks in range_constexpr(k_wmma_steps):
                b_frags = [load_b_frag_tr(b_buf, b_bases[wn], ks)
                           for wn in range_constexpr(wmma_n_rep)]
                if const_expr(mid_compute_callback is not None) and ks == 0:
                    rocdl.sched_barrier(0)
                    mid_compute_callback()
                rocdl.s_wait_dscnt(0)
                for wm in range_constexpr(wmma_m_rep):
                    a_frag = load_a_frag(a_buf, a_bases[wm], ks)
                    rocdl.s_wait_dscnt(0)
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        cur[idx] = wmma_op(
                            T.vec(8, T.f32),
                            b_frags[wn], a_frag, cur[idx],
                            signA=False, signB=False, modC=0,
                            reuseA=False, reuseB=False,
                        ).result
            return cur

        stages_a_lds_addr = []
        stages_b_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(vector.extract(
                make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))
            stages_b_lds_addr.append(vector.extract(
                make_desc_b(stages_b_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))

        desc_a_init = make_desc_a(stages_a_mem[0], arith.index(0))
        desc_b_init = make_desc_b(stages_b_mem[0], arith.index(0))

        addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
        addr_lo_b = vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_b = vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[])
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1

        adv_a_i32 = arith.constant(tile_k * elem_bytes, type=T.i32)
        adv_b_i32 = arith.constant(tile_k * elem_bytes, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_in_valid)
        with ir.InsertionPoint(_if_blk.then_block):
            cur_lo_a = addr_lo_a
            cur_lo_b = addr_lo_b

            for i in range_constexpr(pre_loaded):
                dg0_a = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_a_lds_addr[i], cur_lo_a, addr_hi_a])
                dg0_b = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_b_lds_addr[i], cur_lo_b, addr_hi_b])
                issue_tdm_loads(
                    tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                    tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                )
                cur_lo_a = arith.addi(cur_lo_a, adv_a_i32)
                cur_lo_b = arith.addi(cur_lo_b, adv_b_i32)

            pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

            _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)
            if const_expr(loop_iters > 0):
                init_args = list(accs) + [cur_lo_a, cur_lo_b]
                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    in_acc = list(state[:n_accs])
                    cur_a = state[n_accs]
                    cur_b = state[n_accs + 1]
                    _ = loop_iter
                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers
                        pipeline_fence_signal(outstanding=_fence_outstanding)
                        pipeline_fence_wait()
                        addr_box = [cur_a, cur_b]

                        def _mid_tdm(_ls=load_stage, _ab=addr_box):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1], addr_hi_b])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            )
                            _ab[0] = arith.addi(_ab[0], adv_a_i32)
                            _ab[1] = arith.addi(_ab[1], adv_b_i32)

                        rocdl.sched_barrier(0)
                        in_acc = compute_tile(
                            in_acc,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            mid_compute_callback=_mid_tdm,
                        )
                        cur_a, cur_b = addr_box[0], addr_box[1]
                        rocdl.sched_barrier(0)

                    results = yield list(in_acc) + [cur_a, cur_b]

                accs = list(results[:n_accs])
                cur_lo_a = results[n_accs]
                cur_lo_b = results[n_accs + 1]

            if const_expr(loop_iters > 0):
                pipeline_fence(outstanding=0)

            _tail_had_load = False
            for _load_stage, _compute_stage, _outstanding in tail_plan:
                if const_expr(_outstanding == -1):
                    if const_expr(_tail_had_load):
                        pipeline_fence(outstanding=0)
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                    )
                else:
                    pipeline_fence_signal(outstanding=_outstanding)
                    pipeline_fence_wait()
                    _tail_mid_cb = None
                    if const_expr(_load_stage is not None):
                        _tail_had_load = True
                        _tail_box = [cur_lo_a, cur_lo_b]

                        def _tail_mid(_ls=_load_stage, _ab=_tail_box):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1], addr_hi_b])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            )
                            _ab[0] = arith.addi(_ab[0], adv_a_i32)
                            _ab[1] = arith.addi(_ab[1], adv_b_i32)

                        _tail_mid_cb = _tail_mid

                    rocdl.sched_barrier(0)
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        mid_compute_callback=_tail_mid_cb,
                    )
                    if const_expr(_load_stage is not None):
                        cur_lo_a = _tail_box[0]
                        cur_lo_b = _tail_box[1]

            rocdl.sched_barrier(0)
            valid_m_g_idx = arith.index_cast(T.index, valid_m_g_i32)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    a_vec = accs[idx]
                    row_local = warp_m_base + arith.index(wm * WMMA_M) + lane16
                    row_global_m = blk_m + row_local
                    row_in_valid = arith.cmpi(arith.CmpIPredicate.ult,
                                              row_global_m, valid_m_g_idx)
                    col_base_local = (warp_n_base + arith.index(wn * WMMA_N)
                                      + lane_kgrp * arith.index(8))
                    col_base_global = blk_n + col_base_local
                    col_in_valid = arith.cmpi(arith.CmpIPredicate.ult,
                                              col_base_global,
                                              arith.index(K_out))
                    out_pred = arith.andi(row_in_valid, col_in_valid)
                    _if_out = scf.IfOp(out_pred)
                    with ir.InsertionPoint(_if_out.then_block):
                        for vi in range_constexpr(8):
                            v = vector.extract(a_vec, static_position=[vi], dynamic_position=[])
                            y_out = arith.trunc_f(out_elem_ty, v)
                            row_out_full = (g_idx * arith.index(max_M) + row_global_m)
                            col_out = col_base_global + arith.index(vi)
                            y_off = (row_out_full * arith.index(K_out) + col_out)
                            buffer_ops.buffer_store(
                                y_out, y_rsrc,
                                arith.index_cast(T.i32, y_off),
                            )
                        scf.YieldOp([])
            scf.YieldOp([])

    cache_tag = (in_dtype, out_dtype, K_out, K, E, max_M,
                 tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers,
                 waves_per_eu, expert_sched_mode)

    @flyc.jit
    def launch_moe_grp_s2(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_max_m: fx.Int32,
        i32_k_out_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_experts: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        gx = arith.index(_align_up(max_M, tile_m) // tile_m)
        gy = arith.index(K_out // tile_n)
        gz = arith.index(E)

        launcher = kernel_moe_grp_s2(
            arg_y, arg_x, arg_w, arg_masked_m,
            i32_max_m, i32_k_out_in, i32_inter_in, i32_experts,
        )
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                if waves_per_eu is not None and int(waves_per_eu) >= 1:
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), int(waves_per_eu))

        launcher.launch(
            grid=(_raw(gx), _raw(gy), _raw(gz)),
            block=(block_threads, 1, 1),
            stream=stream,
        )

        if expert_sched_mode:
            launch_moe_grp_s2.compile_hints["llvm_options"] = {
                "amdgpu-expert-scheduling-mode": True,
            }

    return launch_moe_grp_s2


__all__ = [
    "compile_moe_grouped_gemm1_masked",
    "compile_moe_grouped_gemm2_masked",
]
