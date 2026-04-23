# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RDNA4 (gfx120x) MoE 2-stage fp16/bf16 WMMA kernels.

This path targets the Radeon RDNA4 WMMA ISA (``gfx120x``, including
``gfx1201``) using ``wmma_f32_16x16x16_{f16,bf16}`` and a simple LDS pipeline.
It is intentionally separate from the gfx1250 (MI450 / GFX12) TDM-based WMMA
path in ``moe_gemm_2stage_wmma_gfx1250.py``.

Measured starting points on ``gfx1201``:
- stage1: ``tile_k=128``, ``tile_n=64`` for ``tile_m`` 16/32, ``tile_n=128`` for ``tile_m=64``
- stage2: ``tile_k=128``, ``tile_n=64``
- ``waves_per_eu=2`` often helps stage1, while stage2 remains workload-dependent
"""

from __future__ import annotations

import functools

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from kernels.moe_gemm_2stage import MoeGemm2Mode, make_moe_public_api
from kernels.rdna_moe_gemm_2stage_common import (
    _emit_stage1_gate_up_epilogue,
    _emit_stage2_store_epilogue,
    _finalize_alloc_and_launch_2d,
    _make_moe_wave_layout,
    _make_wmma_sub_tiles,
    _moe_out_elem_ty,
    _pick_fp16_launch_shape,
    _require_gfx120x,
)


def _validate_vectorized_tile(
    tile_rows: int, tile_k: int, block_threads: int, tile_name: str
) -> int:
    load_vec = 8  # 8 bf16/f16 elements = 16 bytes
    total = int(tile_rows) * int(tile_k)
    denom = int(block_threads) * load_vec
    if total % denom != 0:
        raise ValueError(
            f"{tile_name} tile ({tile_rows}x{tile_k}) must be divisible by "
            f"block_threads*{load_vec} ({denom}) for RDNA4 vectorized loads"
        )
    return total // denom


def _set_expert_sched_hint(jit_fn, enabled: bool) -> None:
    if enabled:
        jit_fn.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }


class _MoeGemm2AtomicCastWrapper:
    """Run an internal f32 atomic stage2 kernel and cast back.

    RDNA4 cannot lower scalar f16/bf16 buffer atomics cleanly, so for those
    output dtypes we run an f32 atomic kernel into a temporary buffer and cast
    back to the requested dtype on the host.
    """

    def __init__(self, gemm2_exe, model_dim: int, out_dtype_str: str):
        self._gemm2_exe = gemm2_exe
        self._model_dim = int(model_dim)
        self._out_dtype_str = str(out_dtype_str).strip().lower()
        self._cache = {}
        for attr in ("compile_hints",):
            if hasattr(gemm2_exe, attr):
                setattr(self, attr, getattr(gemm2_exe, attr))

    def _resolve_torch_stream(self, *, arg_out, stream):
        import torch

        device_index = arg_out.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if stream is None:
            return torch.cuda.current_stream(device=device_index)
        if isinstance(stream, int):
            return torch.cuda.ExternalStream(stream, device=device_index)
        return stream

    def _get_tmp(self, arg_out, tokens_in: int, torch_stream):
        import torch

        device_index = arg_out.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (int(device_index), int(tokens_in), int(torch_stream.cuda_stream))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with torch.cuda.device(device_index), torch.cuda.stream(torch_stream):
            tmp = torch.empty(
                int(tokens_in),
                self._model_dim,
                device=arg_out.device,
                dtype=torch.float32,
            )
        tmp.record_stream(torch_stream)
        self._cache[key] = tmp
        return tmp

    def __call__(
        self,
        arg_out,
        arg_x,
        arg_w,
        arg_scale_x,
        arg_scale_w,
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        stream=None,
    ):
        import torch

        torch_stream = self._resolve_torch_stream(arg_out=arg_out, stream=stream)
        device_index = arg_out.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        tmp = self._get_tmp(arg_out, tokens_in, torch_stream)
        with torch.cuda.device(device_index), torch.cuda.stream(torch_stream):
            tmp.zero_()
            self._gemm2_exe(
                tmp,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_sorted_token_ids,
                arg_expert_ids,
                arg_sorted_weights,
                arg_num_valid_ids,
                tokens_in,
                n_in,
                k_in,
                size_expert_ids_in,
                torch_stream,
            )
            arg_out_view = arg_out.view(int(tokens_in), self._model_dim)
            arg_out_view.copy_(tmp)
        tmp.record_stream(torch_stream)

    @property
    def mode(self) -> str:
        return MoeGemm2Mode.ATOMIC


@functools.lru_cache(maxsize=64)
def _compile_stage1_kernel_impl(
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
    in_dtype: str,
    out_dtype: str,
    waves_per_eu: int | None,
    expert_sched_mode: bool = True,
):
    """Compile RDNA4 stage1 single-kernel WMMA MoE path."""

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
    WAVE_SIZE = 32
    LDS_PAD_A = 8
    LDS_PAD_B = 8
    LOAD_VEC = 8
    ELEM_BYTES = 2

    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"RDNA4 stage1 only supports fp16/bf16, got {in_dtype!r}")
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"RDNA4 stage1 only supports f16/bf16 outputs, got {out_dtype!r}")
    if int(model_dim) % int(tile_k) != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_k={tile_k}")
    if int(tile_k) % WMMA_K != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if int(tile_m) % WMMA_M != 0 or int(tile_n) % WMMA_N != 0:
        raise ValueError(f"tile_m/tile_n must be multiples of 16, got ({tile_m},{tile_n})")

    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(
            "Invalid RDNA4 stage1 warp tiling: "
            f"wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}"
        )

    num_k_tiles = int(model_dim) // int(tile_k)
    k_wmma_steps = int(tile_k) // WMMA_K
    n_total = int(2 * inter_dim)
    num_a_loads = _validate_vectorized_tile(tile_m, tile_k, block_threads, "stage1 A")
    num_b_loads = _validate_vectorized_tile(tile_n, tile_k, block_threads, "stage1 B")
    sub_tiles = _make_wmma_sub_tiles(
        wmma_m_rep=wmma_m_rep,
        wmma_n_rep=wmma_n_rep,
        WMMA_M=WMMA_M,
        is_fp4=False,
    )

    lds_a_stride = int(tile_k) + LDS_PAD_A
    lds_b_stride = int(tile_k) + LDS_PAD_B
    lds_a_elems = int(tile_m) * lds_a_stride + LDS_PAD_A
    lds_b_elems = int(tile_n) * lds_b_stride + LDS_PAD_B

    gpu_arch = str(get_hip_arch())
    alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="moe_rdna4_s1")
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_elems * ELEM_BYTES
    off_b = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_b + lds_b_elems * ELEM_BYTES

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_rdna4_stage1(
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
        _ = (arg_scale_x, arg_scale_w, arg_num_valid_ids, i32_k_in)

        in_ir_ty = T.bf16 if in_dtype == "bf16" else T.f16
        v8_in_ty = T.vec(8, in_ir_ty)
        v4f32_ty = T.f32x4
        v8f32_ty = T.vec(8, T.f32)
        v8i16_ty = T.vec(8, T.i16)
        zero_raw = arith.constant_vector(0.0, v4f32_ty)

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
        w_nbytes = arith.index(int(experts * n_total * int(model_dim) * ELEM_BYTES))

        sorted_rsrc = buffer_ops.create_buffer_resource(
            arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes
        )
        eid_rsrc = buffer_ops.create_buffer_resource(
            arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes
        )
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=x_nbytes
        )
        w_rsrc = buffer_ops.create_buffer_resource(
            arg_w, max_size=False, num_records_bytes=w_nbytes
        )
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(
            eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32
        )
        eid_ok0 = arith.cmpi(
            arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32)
        )
        eid_ok1 = arith.cmpi(
            arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32)
        )
        eid_ok = arith.andi(eid_ok0, eid_ok1)

        layout_thr = _make_moe_wave_layout(
            m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))
        model_dim_i32 = arith.constant(int(model_dim), type=T.i32)
        n_total_i32 = arith.constant(int(n_total), type=T.i32)
        c2_i32 = arith.constant(2, type=T.i32)
        base8 = lane_kgrp * fx.Index(8)

        base_ptr = alloc.get_base()
        smem_a = SmemPtr(base_ptr, off_a, in_ir_ty, shape=(lds_a_elems,))
        smem_b = SmemPtr(base_ptr, off_b, in_ir_ty, shape=(lds_b_elems,))
        lds_a = get_op_result_or_value(smem_a.get())
        lds_b = get_op_result_or_value(smem_b.get())

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        def _wmma_op(result_type, a_vec, b_vec, acc):
            if in_dtype == "bf16":
                a_i16 = vector.bitcast(v8i16_ty, a_vec)
                b_i16 = vector.bitcast(v8i16_ty, b_vec)
                return rocdl.wmma_f32_16x16x16_bf16(
                    result_type, a_i16, b_i16, arith.unwrap(acc)
                ).result
            return rocdl.wmma_f32_16x16x16_f16(
                result_type, arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)
            ).result

        a_lds_info = []
        for al in range_constexpr(num_a_loads):
            a_lin = tx * fx.Index(LOAD_VEC) + fx.Index(al * block_threads * LOAD_VEC)
            a_load_row = a_lin // fx.Index(tile_k)
            a_load_col = a_lin % fx.Index(tile_k)
            lds_rel = a_load_row * fx.Index(lds_a_stride) + a_load_col
            a_lds_info.append((a_load_row, a_load_col, lds_rel))

        b_lds_info = []
        for bl in range_constexpr(num_b_loads):
            b_lin = tx * fx.Index(LOAD_VEC) + fx.Index(bl * block_threads * LOAD_VEC)
            b_load_row = b_lin // fx.Index(tile_k)
            b_load_col = b_lin % fx.Index(tile_k)
            lds_rel = b_load_row * fx.Index(lds_b_stride) + b_load_col
            b_lds_info.append((b_load_row, b_load_col, lds_rel))

        def _load_a_tile(k_base):
            raw_data = []
            for al in range_constexpr(num_a_loads):
                a_load_row, a_load_col, _ = a_lds_info[al]
                sorted_row = by * arith.index(int(tile_m)) + a_load_row
                row_i32 = arith.index_cast(T.i32, a_load_row)
                sorted_i32 = arith.index_cast(T.i32, sorted_row)
                row_in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    row_i32,
                    arith.constant(int(route_tile_m), type=T.i32),
                )
                sorted_safe = arith.select(
                    row_in_route,
                    sorted_i32,
                    arith.index_cast(T.i32, by * arith.index(int(route_tile_m))),
                )
                fused = buffer_ops.buffer_load(
                    sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32
                )
                tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                load_ok = arith.andi(row_in_route, tok_ok)
                elem_off = tok * model_dim_i32 + arith.index_cast(T.i32, k_base + a_load_col)
                f32_off = elem_off // c2_i32
                raw_if = scf.IfOp(load_ok, results_=[v4f32_ty], has_else=True)
                with ir.InsertionPoint(raw_if.then_block):
                    scf.YieldOp(
                        [buffer_ops.buffer_load(x_rsrc, f32_off, vec_width=4, dtype=T.f32)]
                    )
                with ir.InsertionPoint(raw_if.else_block):
                    scf.YieldOp([zero_raw])
                raw_data.append(raw_if.results[0])
            return raw_data

        def _load_b_tile(k_base, row_shift: int):
            raw_data = []
            base_row = eid_i32 * n_total_i32 + arith.index_cast(T.i32, blk_n) + arith.constant(
                int(row_shift), type=T.i32
            )
            for bl in range_constexpr(num_b_loads):
                b_load_row, b_load_col, _ = b_lds_info[bl]
                row_i32 = base_row + arith.index_cast(T.i32, b_load_row)
                elem_off = row_i32 * model_dim_i32 + arith.index_cast(T.i32, k_base + b_load_col)
                f32_off = elem_off // c2_i32
                raw_data.append(
                    buffer_ops.buffer_load(w_rsrc, f32_off, vec_width=4, dtype=T.f32)
                )
            return raw_data

        def _store_a_tile(raw_data):
            for al in range_constexpr(num_a_loads):
                _, _, lds_rel = a_lds_info[al]
                a_vec = vector.bitcast(v8_in_ty, raw_data[al])
                vector.store(a_vec, lds_a, [lds_rel])

        def _store_b_tile(raw_data):
            for bl in range_constexpr(num_b_loads):
                _, _, lds_rel = b_lds_info[bl]
                b_vec = vector.bitcast(v8_in_ty, raw_data[bl])
                vector.store(b_vec, lds_b, [lds_rel])

        def _load_a_single_from_lds(rk, rm_val):
            col_base = fx.Index(rk * WMMA_K) + base8
            row = warp_m_base + fx.Index(rm_val * WMMA_M) + lane16
            lds_idx = row * fx.Index(lds_a_stride) + col_base
            return vector.load_op(v8_in_ty, lds_a, [lds_idx])

        def _load_b_from_lds(rk):
            vecs = []
            col_base = fx.Index(rk * WMMA_K) + base8
            for rn in range_constexpr(wmma_n_rep):
                row = warp_n_base + fx.Index(rn * WMMA_N) + lane16
                lds_idx = row * fx.Index(lds_b_stride) + col_base
                vecs.append(vector.load_op(v8_in_ty, lds_b, [lds_idx]))
            return vecs

        def _do_compute_rk(accs_in, rk):
            new_accs = list(accs_in)
            b_vecs = _load_b_from_lds(rk)
            for wm in range_constexpr(wmma_m_rep):
                a_vec = _load_a_single_from_lds(rk, wm)
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    new_accs[idx] = _wmma_op(v8f32_ty, a_vec, b_vecs[wn], new_accs[idx])
            return new_accs

        acc_zero = arith.constant_vector(0.0, v8f32_ty)
        acc_gate = [acc_zero] * (wmma_m_rep * wmma_n_rep)
        acc_up = [acc_zero] * (wmma_m_rep * wmma_n_rep)

        _if_eid = scf.IfOp(eid_ok)
        with ir.InsertionPoint(_if_eid.then_block):
            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                a_data = _load_a_tile(k_base)
                gate_data = _load_b_tile(k_base, 0)
                _store_a_tile(a_data)
                _store_b_tile(gate_data)
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    acc_gate = _do_compute_rk(acc_gate, ks)
                gpu.barrier()

                up_data = _load_b_tile(k_base, int(inter_dim))
                _store_b_tile(up_data)
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    acc_up = _do_compute_rk(acc_up, ks)
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_gate_up_sub8(acc_idx, _vec_base):
                return acc_gate[acc_idx], acc_up[acc_idx]

            _emit_stage1_gate_up_epilogue(
                sub_tiles=sub_tiles,
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
    def launch_stage1(
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
        _ = (arg_num_valid_ids, i32_k_in)
        ctx = CompilationContext.get_current()
        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (inter_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_rdna4_stage1(
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

    _set_expert_sched_hint(launch_stage1, expert_sched_mode)
    return launch_stage1


@functools.lru_cache(maxsize=64)
def _compile_stage2_kernel_impl(
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
    in_dtype: str,
    out_dtype: str,
    accumulate: bool,
    waves_per_eu: int | None,
    expert_sched_mode: bool = True,
):
    """Compile RDNA4 stage2 single-kernel WMMA MoE path."""

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
    WAVE_SIZE = 32
    LDS_PAD_A = 8
    LDS_PAD_B = 8
    LOAD_VEC = 8
    ELEM_BYTES = 2

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"RDNA4 stage2 only supports fp16/bf16, got {in_dtype!r}")
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(f"RDNA4 stage2 only supports f16/bf16/f32 outputs, got {out_dtype!r}")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "RDNA4 compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    if int(inter_dim) % int(tile_k) != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if int(tile_k) % WMMA_K != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if int(tile_m) % WMMA_M != 0 or int(tile_n) % WMMA_N != 0:
        raise ValueError(f"tile_m/tile_n must be multiples of 16, got ({tile_m},{tile_n})")

    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(
            "Invalid RDNA4 stage2 warp tiling: "
            f"wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}"
        )

    num_k_tiles = int(inter_dim) // int(tile_k)
    k_wmma_steps = int(tile_k) // WMMA_K
    num_a_loads = _validate_vectorized_tile(tile_m, tile_k, block_threads, "stage2 A")
    num_b_loads = _validate_vectorized_tile(tile_n, tile_k, block_threads, "stage2 B")
    sub_tiles = _make_wmma_sub_tiles(
        wmma_m_rep=wmma_m_rep,
        wmma_n_rep=wmma_n_rep,
        WMMA_M=WMMA_M,
        is_fp4=False,
    )

    lds_a_stride = int(tile_k) + LDS_PAD_A
    lds_b_stride = int(tile_k) + LDS_PAD_B
    lds_a_elems = int(tile_m) * lds_a_stride + LDS_PAD_A
    lds_b_elems = int(tile_n) * lds_b_stride + LDS_PAD_B

    gpu_arch = str(get_hip_arch())
    alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="moe_rdna4_s2")
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_elems * ELEM_BYTES
    off_b = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_b + lds_b_elems * ELEM_BYTES

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_rdna4_stage2(
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

        in_ir_ty = T.bf16 if in_dtype == "bf16" else T.f16
        v8_in_ty = T.vec(8, in_ir_ty)
        v4f32_ty = T.f32x4
        v8f32_ty = T.vec(8, T.f32)
        v8i16_ty = T.vec(8, T.i16)
        zero_raw = arith.constant_vector(0.0, v4f32_ty)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # N tile
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
        out_elem_bytes = 4 if out_is_f32 else 2
        out_nbytes = tokens_idx * n_idx * arith.index(out_elem_bytes)
        if not bool(accumulate):
            out_nbytes = x_rows * n_idx * arith.index(out_elem_bytes)

        sorted_rsrc = buffer_ops.create_buffer_resource(
            arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes
        )
        eid_rsrc = buffer_ops.create_buffer_resource(
            arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes
        )
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=x_nbytes
        )
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out, max_size=False, num_records_bytes=out_nbytes
        )
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(
            eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32
        )
        eid_ok0 = arith.cmpi(
            arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32)
        )
        eid_ok1 = arith.cmpi(
            arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32)
        )
        block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
        block_in_valid = arith.cmpi(
            arith.CmpIPredicate.slt, block_row_start, num_valid_i32
        )
        block_ok = arith.andi(block_in_valid, arith.andi(eid_ok0, eid_ok1))

        layout_thr = _make_moe_wave_layout(
            m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))
        model_dim_i32 = arith.constant(int(model_dim), type=T.i32)
        inter_dim_i32 = arith.constant(int(inter_dim), type=T.i32)
        topk_i32 = arith.constant(int(topk), type=T.i32)
        c2_i32 = arith.constant(2, type=T.i32)
        base8 = lane_kgrp * fx.Index(8)

        base_ptr = alloc.get_base()
        smem_a = SmemPtr(base_ptr, off_a, in_ir_ty, shape=(lds_a_elems,))
        smem_b = SmemPtr(base_ptr, off_b, in_ir_ty, shape=(lds_b_elems,))
        lds_a = get_op_result_or_value(smem_a.get())
        lds_b = get_op_result_or_value(smem_b.get())

        def _wmma_op(result_type, a_vec, b_vec, acc):
            if in_dtype == "bf16":
                a_i16 = vector.bitcast(v8i16_ty, a_vec)
                b_i16 = vector.bitcast(v8i16_ty, b_vec)
                return rocdl.wmma_f32_16x16x16_bf16(
                    result_type, a_i16, b_i16, arith.unwrap(acc)
                ).result
            return rocdl.wmma_f32_16x16x16_f16(
                result_type, arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)
            ).result

        a_lds_info = []
        for al in range_constexpr(num_a_loads):
            a_lin = tx * fx.Index(LOAD_VEC) + fx.Index(al * block_threads * LOAD_VEC)
            a_load_row = a_lin // fx.Index(tile_k)
            a_load_col = a_lin % fx.Index(tile_k)
            lds_rel = a_load_row * fx.Index(lds_a_stride) + a_load_col
            a_lds_info.append((a_load_row, a_load_col, lds_rel))

        b_lds_info = []
        for bl in range_constexpr(num_b_loads):
            b_lin = tx * fx.Index(LOAD_VEC) + fx.Index(bl * block_threads * LOAD_VEC)
            b_load_row = b_lin // fx.Index(tile_k)
            b_load_col = b_lin % fx.Index(tile_k)
            lds_rel = b_load_row * fx.Index(lds_b_stride) + b_load_col
            b_lds_info.append((b_load_row, b_load_col, lds_rel))

        def _load_a_tile(k_base):
            raw_data = []
            for al in range_constexpr(num_a_loads):
                a_load_row, a_load_col, _ = a_lds_info[al]
                sorted_row = by * arith.index(int(tile_m)) + a_load_row
                row_i32 = arith.index_cast(T.i32, a_load_row)
                sorted_i32 = arith.index_cast(T.i32, sorted_row)
                row_in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    row_i32,
                    arith.constant(int(route_tile_m), type=T.i32),
                )
                row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
                row_ok = arith.andi(row_in_route, row_in_valid)
                sorted_safe = arith.select(row_ok, sorted_i32, block_row_start)
                fused = buffer_ops.buffer_load(
                    sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32
                )
                tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                slot = fused >> arith.constant(24, type=T.i32)
                tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                slot_ok0 = arith.cmpi(
                    arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32)
                )
                slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, topk_i32)
                ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                load_ok = arith.andi(row_ok, ts_ok)
                ts = tok * topk_i32 + slot
                elem_off = ts * inter_dim_i32 + arith.index_cast(T.i32, k_base + a_load_col)
                f32_off = elem_off // c2_i32
                raw_if = scf.IfOp(load_ok, results_=[v4f32_ty], has_else=True)
                with ir.InsertionPoint(raw_if.then_block):
                    scf.YieldOp(
                        [buffer_ops.buffer_load(x_rsrc, f32_off, vec_width=4, dtype=T.f32)]
                    )
                with ir.InsertionPoint(raw_if.else_block):
                    scf.YieldOp([zero_raw])
                raw_data.append(raw_if.results[0])
            return raw_data

        def _load_b_tile(k_base):
            raw_data = []
            base_row = eid_i32 * model_dim_i32 + arith.index_cast(T.i32, blk_n)
            for bl in range_constexpr(num_b_loads):
                b_load_row, b_load_col, _ = b_lds_info[bl]
                row_i32 = base_row + arith.index_cast(T.i32, b_load_row)
                elem_off = row_i32 * inter_dim_i32 + arith.index_cast(T.i32, k_base + b_load_col)
                f32_off = elem_off // c2_i32
                raw_data.append(
                    buffer_ops.buffer_load(w_rsrc, f32_off, vec_width=4, dtype=T.f32)
                )
            return raw_data

        def _store_a_tile(raw_data):
            for al in range_constexpr(num_a_loads):
                _, _, lds_rel = a_lds_info[al]
                a_vec = vector.bitcast(v8_in_ty, raw_data[al])
                vector.store(a_vec, lds_a, [lds_rel])

        def _store_b_tile(raw_data):
            for bl in range_constexpr(num_b_loads):
                _, _, lds_rel = b_lds_info[bl]
                b_vec = vector.bitcast(v8_in_ty, raw_data[bl])
                vector.store(b_vec, lds_b, [lds_rel])

        def _load_a_single_from_lds(rk, rm_val):
            col_base = fx.Index(rk * WMMA_K) + base8
            row = warp_m_base + fx.Index(rm_val * WMMA_M) + lane16
            lds_idx = row * fx.Index(lds_a_stride) + col_base
            return vector.load_op(v8_in_ty, lds_a, [lds_idx])

        def _load_b_from_lds(rk):
            vecs = []
            col_base = fx.Index(rk * WMMA_K) + base8
            for rn in range_constexpr(wmma_n_rep):
                row = warp_n_base + fx.Index(rn * WMMA_N) + lane16
                lds_idx = row * fx.Index(lds_b_stride) + col_base
                vecs.append(vector.load_op(v8_in_ty, lds_b, [lds_idx]))
            return vecs

        def _do_compute_rk(accs_in, rk):
            new_accs = list(accs_in)
            b_vecs = _load_b_from_lds(rk)
            for wm in range_constexpr(wmma_m_rep):
                a_vec = _load_a_single_from_lds(rk, wm)
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    new_accs[idx] = _wmma_op(v8f32_ty, a_vec, b_vecs[wn], new_accs[idx])
            return new_accs

        acc_zero = arith.constant_vector(0.0, v8f32_ty)
        acc = [acc_zero] * (wmma_m_rep * wmma_n_rep)

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                a_data = _load_a_tile(k_base)
                b_data = _load_b_tile(k_base)
                _store_a_tile(a_data)
                _store_b_tile(b_data)
                gpu.barrier()
                for ks in range_constexpr(k_wmma_steps):
                    acc = _do_compute_rk(acc, ks)
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_sub8(acc_idx, _vec_base):
                return acc[acc_idx]

            _emit_stage2_store_epilogue(
                sub_tiles=sub_tiles,
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
                out_is_f32=bool(out_is_f32),
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
    def launch_stage2(
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
        launcher = moe_rdna4_stage2(
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

    _set_expert_sched_hint(launch_stage2, expert_sched_mode)
    return launch_stage2


@functools.lru_cache(maxsize=1024)
def _compile_moe_gemm(
    *,
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    in_dtype: str = "fp16",
    out_dtype: str = "f16",
    accumulate: bool = True,
    waves_per_eu: int | None = None,
    expert_sched_mode: bool = True,
):
    _require_gfx120x()
    if waves_per_eu is not None and int(waves_per_eu) < 1:
        raise ValueError(f"waves_per_eu must be >= 1, got {waves_per_eu!r}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(
            f"Unsupported in_dtype for RDNA4 MoE stage{stage}: {in_dtype!r}; "
            "expected 'fp16' or 'bf16'"
        )

    single_tile_m, single_tile_n, single_m_warp, single_n_warp = _pick_fp16_launch_shape(
        int(tile_m),
        int(tile_n),
        int(tile_k),
        max_total_warps=4,
    )
    common = dict(
        model_dim=int(model_dim),
        inter_dim=int(inter_dim),
        experts=int(experts),
        topk=int(topk),
        route_tile_m=int(tile_m),
        tile_m=int(single_tile_m),
        tile_n=int(single_tile_n),
        tile_k=int(tile_k),
        m_warp=int(single_m_warp),
        n_warp=int(single_n_warp),
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        expert_sched_mode=expert_sched_mode,
    )

    if stage == 1:
        return _compile_stage1_kernel_impl(
            doweight_stage1=bool(doweight),
            **common,
        )
    # RDNA4 stage2 with f16/bf16 atomic accumulate is implemented by running an
    # internal f32-output kernel and casting back on the host; the LLVM backend
    # cannot lower scalar half-atomics cleanly for this path.
    out_s = str(out_dtype).strip().lower()
    if bool(accumulate) and out_s in ("f16", "fp16", "half", "bf16", "bfloat16"):
        f32_exe = _compile_stage2_kernel_impl(
            doweight_stage2=bool(doweight),
            accumulate=True,
            out_dtype="f32",
            **{k: v for k, v in common.items() if k != "out_dtype"},
        )
        return _MoeGemm2AtomicCastWrapper(
            f32_exe,
            model_dim=int(model_dim),
            out_dtype_str=str(out_dtype),
        )
    return _compile_stage2_kernel_impl(
        doweight_stage2=bool(doweight),
        accumulate=bool(accumulate),
        **common,
    )


compile_moe_gemm1, compile_moe_gemm2, compile_moe_gemm2_ex = make_moe_public_api(
    _compile_moe_gemm
)
