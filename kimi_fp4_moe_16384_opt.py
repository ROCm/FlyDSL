"""Experimental Kimi fp4 MoE paths for the fixed M=16384 shape.

This file is intentionally separate from ``kimi_fp4_moe_16384.py``.  It keeps
the known-good FlyDSL GEMMs intact while extracting the relevant mxfp4_moe
orchestration from aiter for direct comparison and incremental migration.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

import aiter
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter import dtypes
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl._mlir.dialects._math_ops_gen import fma as _math_fma
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from aiter.ops.flydsl.kernels.layout_utils import (
    _div_pow2,
    crd2idx,
    idx2crd,
    get as layout_get,
)
from aiter.ops.flydsl.kernels.mfma_epilogues import c_shuffle_epilog
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    swizzle_xor16,
    tile_chunk_coord_i32,
)

from kimi_fp4_moe_16384 import (
    BLOCK_M,
    EXPERTS,
    INTER_DIM,
    MODEL_DIM,
    TOKEN,
    TOPK,
    _ptr_view_safe,
    _run_compiled,
    kimi_fp4_stage1_16384,
    kimi_fp4_stage2_16384,
    kimi_moe_sorting_16384,
)


MXFP4_BLOCK_M = 128
MXFP4_STAGE1_KERNEL = "mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128"
MXFP4_STAGE2_KERNEL = (
    "mxfp4_moe_g2_a4w4_NE385_H7168_E512_BM128_NONATOMIC_MXFP4OUT"
)
MXFP4_SCATTER_THREADS = 128
MXFP4_SCATTER_COLS_PER_THREAD = 8


@dataclass
class Mxfp4SortBuffers:
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    cumsum_tensor: torch.Tensor
    reverse_sorted: torch.Tensor
    sorted_weights: torch.Tensor
    masked_m: torch.Tensor
    m_indices: torch.Tensor
    max_sorted: int
    block_m: int


def _empty_bf16(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=dtypes.bf16, device=device)


def _as_u8_storage(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() == 1 and t.dtype != torch.uint8:
        return t.view(torch.uint8)
    return t


def _ptr_buffer_resource(ptr, num_records_bytes: int):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64,
        num_records_bytes=num_records_bytes,
    )


def _extract_global_ptr(ptr):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), addr_i64)


def _global_load_i32(global_ptr, elem_offset):
    if isinstance(elem_offset, int):
        elem_offset = arith.constant(elem_offset, type=T.i32)
    elem_offset_raw = elem_offset.ir_value() if hasattr(elem_offset, "ir_value") else elem_offset
    if isinstance(elem_offset_raw.type, ir.IndexType):
        elem_offset_i64 = arith.index_cast(T.i64, elem_offset_raw)
    else:
        int_type = ir.IntegerType(elem_offset_raw.type)
        if int_type.width == 64:
            elem_offset_i64 = ArithValue(elem_offset_raw)
        else:
            elem_offset_i64 = ArithValue(arith.ExtSIOp(T.i64, elem_offset_raw).result)
    byte_offset_i64 = elem_offset_i64 * arith.constant(4, type=T.i64)
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=byte_offset_i64, elem_type=T.i8)
    return llvm.LoadOp(T.i32, ptr, alignment=4).result


def _dpp_xor_f32(src, offset: int):
    src_i32 = src.bitcast(T.i32) if hasattr(src, "bitcast") else ArithValue(src).bitcast(T.i32)
    if offset == 1:
        dpp_ctrl = 0xB1
    elif offset == 2:
        dpp_ctrl = 0x4E
    else:
        raise ValueError(f"unsupported DPP xor offset: {offset}")
    out_i32 = llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.update.dpp.i32",
        [
            src_i32,
            src_i32,
            arith.constant(dpp_ctrl, type=T.i32),
            arith.constant(0xF, type=T.i32),
            arith.constant(0xF, type=T.i32),
            arith.constant(False, type=ir.IntegerType.get_signless(1)),
        ],
        [],
        [],
    )
    return out_i32.bitcast(T.f32)


@functools.lru_cache(maxsize=1)
def compile_kimi_mxfp4_scatter_reduce_q_16384():
    """Compile the fixed Kimi mxfp4 scatter/reduce-q kernel.

    This mirrors aiter's
    ``scatter_reduce_mxfp4_kernel<D_HIDDEN=7168, TOPK=9, COLS_PER_THREAD=8>``:
    each thread decodes one 8-column MXFP4 group for one token and reduces the
    nine routed slots into bf16 output.
    """

    cols_per_block = MXFP4_SCATTER_THREADS * MXFP4_SCATTER_COLS_PER_THREAD
    grid_x = (MODEL_DIM + cols_per_block - 1) // cols_per_block
    qcols_i32 = MODEL_DIM // 2 // 4
    scols_i32 = MODEL_DIM // 32 // 4
    out_i32_stride = MODEL_DIM // 2
    q_nbytes = mxfp4_max_sorted_16384() * (MODEL_DIM // 2)
    scale_nbytes = mxfp4_max_sorted_16384() * (MODEL_DIM // 32)
    reverse_nbytes = TOKEN * TOPK * 4
    weights_nbytes = mxfp4_max_sorted_16384() * 4
    out_nbytes = TOKEN * MODEL_DIM * 2
    module_name = "flydsl_kimi_mxfp4_scatter_reduce_q_NE385_H7168_E512_M16384_TOPK9"

    @flyc.kernel(name=module_name)
    def scatter_reduce_q(
        arg_flat_out_q: fx.Pointer,
        arg_flat_out_scale: fx.Pointer,
        arg_reverse_sorted: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_out: fx.Pointer,
    ):
        i32 = T.i32
        f32 = T.f32

        q_rsrc = _ptr_buffer_resource(arg_flat_out_q, q_nbytes)
        scale_rsrc = _ptr_buffer_resource(arg_flat_out_scale, scale_nbytes)
        reverse_rsrc = _ptr_buffer_resource(arg_reverse_sorted, reverse_nbytes)
        weights_rsrc = _ptr_buffer_resource(arg_sorted_weights, weights_nbytes)
        out_rsrc = _ptr_buffer_resource(arg_out, out_nbytes)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        token_idx = gpu.block_id("y")

        col_thread = bx * arith.constant(MXFP4_SCATTER_THREADS, index=True) + tx
        col_base = col_thread * arith.constant(MXFP4_SCATTER_COLS_PER_THREAD, index=True)
        col_base_i32 = arith.index_cast(i32, col_base)
        token_i32 = arith.index_cast(i32, token_idx)

        c1_i32 = arith.constant(1, type=i32)
        c3_i32 = arith.constant(3, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c255_i32 = arith.constant(255, type=i32)
        c_topk_i32 = arith.constant(TOPK, type=i32)
        c_qcols_i32 = arith.constant(qcols_i32, type=i32)
        c_scols_i32 = arith.constant(scols_i32, type=i32)
        c_out_stride_i32 = arith.constant(out_i32_stride, type=i32)

        c0_f32 = arith.constant(0.0, type=f32)

        def _fma_f32(a, b, c):
            return ArithValue(
                _math_fma(
                    arith._to_raw(a),
                    arith._to_raw(b),
                    arith._to_raw(c),
                )
            )

        acc = [c0_f32 for _ in range(MXFP4_SCATTER_COLS_PER_THREAD)]
        blk = col_base_i32 >> arith.constant(5, type=i32)
        scale_word_col = (blk & ~c3_i32) >> arith.constant(2, type=i32)
        q_col_word = col_base_i32 >> c3_i32

        for i in range_constexpr(TOPK):
            reverse_off = token_i32 * c_topk_i32 + arith.constant(i, type=i32)
            sorted_pos = ArithValue(
                buffer_ops.buffer_load(reverse_rsrc, reverse_off, vec_width=1, dtype=i32)
            )
            route_w = buffer_ops.buffer_load(weights_rsrc, sorted_pos, vec_width=1, dtype=f32)

            scale_idx = sorted_pos * c_scols_i32 + scale_word_col
            scale_word = ArithValue(
                buffer_ops.buffer_load(scale_rsrc, scale_idx, vec_width=1, dtype=i32)
            )
            scale_shift = (blk & c3_i32) << c3_i32
            e8 = (scale_word >> scale_shift) & c255_i32
            scale = (e8 << c23_i32).bitcast(f32)

            q_idx = sorted_pos * c_qcols_i32 + q_col_word
            packed = ArithValue(
                buffer_ops.buffer_load(q_rsrc, q_idx, vec_width=1, dtype=i32, cache_modifier=4)
            )
            decoded_pairs = []
            for pair in range_constexpr(MXFP4_SCATTER_COLS_PER_THREAD // 2):
                decoded_pairs.append(
                    llvm.call_intrinsic(
                        T.vec(2, f32),
                        "llvm.amdgcn.cvt.scalef32.pk.f32.fp4",
                        [packed, scale, arith.constant(pair, type=i32)],
                        [],
                        [],
                    )
                )
            for k in range_constexpr(MXFP4_SCATTER_COLS_PER_THREAD):
                decoded = vector.extract(
                    decoded_pairs[k // 2],
                    static_position=[k % 2],
                    dynamic_position=[],
                )
                acc[k] = _fma_f32(decoded, route_w, acc[k])

        packed_words = []
        for k in range_constexpr(MXFP4_SCATTER_COLS_PER_THREAD // 2):
            packed_words.append(rocdl.cvt_pk_bf16_f32(acc[2 * k], acc[2 * k + 1]))
        out_i32_vec = vector.from_elements(T.vec(4, i32), packed_words)
        out_word_off = token_i32 * c_out_stride_i32 + (col_base_i32 >> c1_i32)
        buffer_ops.buffer_store(
            out_i32_vec,
            out_rsrc,
            out_word_off,
            cache_modifier=4,
        )

    @flyc.jit
    def launch_scatter_reduce_q(
        flat_out_q: fx.Pointer,
        flat_out_scale: fx.Pointer,
        reverse_sorted: fx.Pointer,
        sorted_weights: fx.Pointer,
        out: fx.Pointer,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        scatter_reduce_q(
            flat_out_q,
            flat_out_scale,
            reverse_sorted,
            sorted_weights,
            out,
        ).launch(
            grid=(grid_x, TOKEN, 1),
            block=(MXFP4_SCATTER_THREADS, 1, 1),
            stream=stream,
        )

    return launch_scatter_reduce_q


def kimi_mxfp4_scatter_reduce_q_16384(
    flat_out_q: torch.Tensor,
    flat_out_scale: torch.Tensor,
    reverse_sorted: torch.Tensor,
    sorted_weights: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FlyDSL port of aiter's fixed-shape mxfp4 scatter/reduce-q kernel."""
    if tuple(flat_out_q.shape[1:]) != (MODEL_DIM // 2,):
        raise ValueError(
            f"expected flat_out_q second dim {MODEL_DIM // 2}, got {tuple(flat_out_q.shape)}"
        )
    if tuple(flat_out_scale.shape[1:]) != (MODEL_DIM // 32,):
        raise ValueError(
            "expected flat_out_scale second dim "
            f"{MODEL_DIM // 32}, got {tuple(flat_out_scale.shape)}"
        )
    if tuple(reverse_sorted.shape) != (TOKEN * TOPK,):
        raise ValueError(
            f"expected reverse_sorted shape {(TOKEN * TOPK,)}, got {tuple(reverse_sorted.shape)}"
        )
    if out is None:
        out = torch.empty((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device=flat_out_q.device)

    args = (
        _ptr_view_safe(_as_u8_storage(flat_out_q).view(-1)),
        _ptr_view_safe(_as_u8_storage(flat_out_scale).view(-1)),
        _ptr_view_safe(reverse_sorted.view(-1)),
        _ptr_view_safe(sorted_weights.view(-1)),
        _ptr_view_safe(out.view(-1)),
        torch.cuda.current_stream(),
    )
    _run_compiled(compile_kimi_mxfp4_scatter_reduce_q_16384(), args)
    return out


@functools.lru_cache(maxsize=1)
def compile_kimi_mxfp4_gemm1_16384():
    """Compile a fixed-shape FlyDSL candidate for aiter mxfp4 GEMM1.

    Contract matched here:

    ``A_q, A_scale_sorted`` x ``W1`` -> ``inter_sorted_quant`` plus
    ``inter_sorted_shuffled_scale``.

    This is a correctness-first BM128 port of aiter's selected
    ``mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128`` kernel.  It keeps GEMM,
    SiLU*Up, and MXFP4 quantization in one GPU kernel, so replacing aiter GEMM1
    does not change the end-to-end kernel count.
    """

    tile_m = MXFP4_BLOCK_M
    tile_n = 256
    tile_k = 256
    total_threads = 256
    max_sorted = mxfp4_max_sorted_16384()
    num_m_blocks = max_sorted // tile_m
    num_n_blocks = (2 * INTER_DIM) // tile_n

    _scale_pack_m = 2
    _scale_pack_n = 2
    _scale_pack_k = 2
    pack_M = 2
    pack_N = 2
    pack_K = 2
    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    k_unroll = tile_k // 128
    m_repeat_packed = m_repeat // pack_M
    num_acc_n_packed = num_acc_n // pack_N
    num_k_tiles = MODEL_DIM // tile_k

    a_elem_vec_pack = 2
    _eff_lds_stride = tile_k // a_elem_vec_pack
    _eff_tile_k_bytes = tile_k // a_elem_vec_pack
    _single_x_bytes = tile_m * _eff_lds_stride
    _x_slots_bytes = 2 * _single_x_bytes
    _lds_scale_bytes = (tile_m // 32) * num_k_tiles * 256
    _kloop_lds_bytes = _x_slots_bytes + _lds_scale_bytes
    _lds_acc_bytes = tile_m * tile_n * 4

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_mxfp4_gemm1")
    lds_offset = allocator._align(allocator.ptr, 16)
    lds_scale_offset = lds_offset + _x_slots_bytes
    allocator.ptr = lds_offset + max(_kloop_lds_bytes, _lds_acc_bytes)

    x_nbytes = TOKEN * (MODEL_DIM // 2)
    x_scale_nbytes = max_sorted * (MODEL_DIM // 32) * 2
    w_nbytes = EXPERTS * (2 * INTER_DIM) * (MODEL_DIM // 2)
    w_scale_nbytes = EXPERTS * (2 * INTER_DIM) * (MODEL_DIM // 32)
    expert_nbytes = num_m_blocks * 4
    m_indices_nbytes = max_sorted * 4
    numids_nbytes = 4
    out_q_nbytes = max_sorted * (INTER_DIM // 2)
    out_scale_nbytes = max_sorted * 64

    _b_kpack_bytes_s = 16
    _b_kpack_elems_s = _b_kpack_bytes_s
    _b_c_k_s = MODEL_DIM // _scale_pack_k
    _b_c_k0_s = _b_c_k_s // 64
    _b_stride_nlane = _b_kpack_elems_s
    _b_stride_klane = 16 * _b_stride_nlane
    _b_stride_k0 = 4 * _b_stride_klane
    _b_stride_n0 = _b_c_k0_s * _b_stride_k0
    _expert_b_stride = ((2 * INTER_DIM) // 16) * _b_stride_n0

    k_as_per_chunk_dw = ((MODEL_DIM // 32) // 4 // 2) * 64
    k_bs_stride_k0_dw = 64
    k_bs_stride_n0_dw = ((MODEL_DIM // 32) // 4 // 2) * 64
    k_bs_per_expert_dw = (((2 * INTER_DIM) // 16) // 2) * k_bs_stride_n0_dw

    # Output scale layout consumed by GEMM2 for K=INTER_DIM=512.
    k_out_as_per_chunk_dw = ((INTER_DIM // 32) // 4 // 2) * 64

    module_name = "flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v11"

    @flyc.kernel(name=module_name)
    def gemm1(
        arg_inter_sorted_quant: fx.Pointer,
        arg_inter_sorted_scale: fx.Pointer,
        arg_a_quant: fx.Pointer,
        arg_w: fx.Pointer,
        arg_a_scale_sorted: fx.Pointer,
        arg_w_scale: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
    ):
        i32 = T.i32
        i64 = T.i64
        f32 = T.f32
        vec4_f32 = T.vec(4, f32)
        vec2_i64 = T.vec(2, i64)
        vec4_i64 = T.vec(4, i64)
        vec8_i32 = T.vec(8, i32)
        vec16_x = T.vec(16, T.f8)

        out_q_rsrc = _ptr_buffer_resource(arg_inter_sorted_quant, out_q_nbytes)
        out_scale_rsrc = _ptr_buffer_resource(arg_inter_sorted_scale, out_scale_nbytes)
        x_rsrc = _ptr_buffer_resource(arg_a_quant, x_nbytes)
        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)
        sx_rsrc = _ptr_buffer_resource(arg_a_scale_sorted, x_scale_nbytes)
        sw_rsrc = _ptr_buffer_resource(arg_w_scale, w_scale_nbytes)
        expert_ptr = _extract_global_ptr(arg_expert_ids)
        m_indices_ptr = _extract_global_ptr(arg_m_indices)
        numids_ptr = _extract_global_ptr(arg_num_valid_ids)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")
        bx = gpu.block_id("y")
        bx_m = bx * arith.constant(tile_m, index=True)
        bx_m_i32 = arith.index_cast(i32, bx_m)

        num_valid_i32 = _global_load_i32(numids_ptr, 0)
        num_valid_i32 = rocdl.ReadfirstlaneOp(i32, num_valid_i32).res
        blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

        expert_i32 = _global_load_i32(expert_ptr, bx)
        expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(EXPERTS, type=i32))
        do_gemm = arith.andi(blk_valid, exp_valid)

        base_ptr = allocator.get_base()
        lds_x0 = SmemPtr(base_ptr, lds_offset, T.f8, shape=(_single_x_bytes,)).get()
        lds_x1 = SmemPtr(base_ptr, lds_offset + _single_x_bytes, T.f8, shape=(_single_x_bytes,)).get()
        lds_x_slots = [lds_x0, lds_x1]
        lds_scale_i8 = SmemPtr(base_ptr, lds_scale_offset, T.i8, shape=(_lds_scale_bytes,)).get()
        lds_scale_i32 = SmemPtr(
            base_ptr,
            lds_scale_offset,
            T.i32,
            shape=(_lds_scale_bytes // 4,),
        ).get()
        lds_acc = SmemPtr(base_ptr, lds_offset, T.f32, shape=(tile_m * tile_n,)).get()

        shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
        stride_lds = fx.make_stride(_eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)
        k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_wl = idx2crd(tx, layout_tx_wave_lane)
        wave_id = layout_get(coord_wl, 0)
        lane_id = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * arith.constant(16, index=True)
        n_tile_base = wave_id * arith.constant(n_per_wave, index=True)
        by_n = by * arith.constant(tile_n, index=True)
        expert_b_base = expert_idx * arith.constant(_expert_b_stride, index=True)

        n_intra_list = [None] * num_acc_n
        n_blk_list = [None] * num_acc_n
        for ni in range_constexpr(num_acc_n):
            c_offset = arith.constant(ni * 16, index=True)
            global_n = by_n + n_tile_base + c_offset + lane_mod_16
            n_blk_list[ni] = _div_pow2(global_n, 16)
            n_intra_list[ni] = global_n % arith.constant(16, index=True)

        bytes_per_thread_x = tile_m * tile_k // a_elem_vec_pack // total_threads
        x_load_bytes = 16
        num_x_loads = bytes_per_thread_x // x_load_bytes
        chunk_i32 = x_load_bytes // 4
        tile_k_dwords = tile_k // (4 * a_elem_vec_pack)
        layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
        c_chunk_i32 = arith.constant(chunk_i32, index=True)
        tx_i32_base = tx * c_chunk_i32
        c_k_div4 = arith.constant((MODEL_DIM // a_elem_vec_pack) // 4, index=True)

        x_row_base_div4 = []
        x_col_local_i32 = []
        x_row_local = []
        for i in range_constexpr(num_x_loads):
            row_local, col_local_i32 = tile_chunk_coord_i32(
                arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_x_tile_div4,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(row_local)
            x_col_local_i32.append(col_local_i32)
            sorted_row_i = bx_m + row_local
            actual_row = _global_load_i32(m_indices_ptr, sorted_row_i)
            row_ok = arith.cmpi(CmpIPredicate.ult, actual_row, arith.constant(TOKEN, type=i32))
            # aiter lets invalid padded rows read one-past the A buffer; MUBUF
            # OOB returns zero, which keeps padded inter rows zero.
            row_safe = arith.select(row_ok, actual_row, arith.constant(TOKEN, type=i32))
            row_idx = arith.index_cast(ir.IndexType.get(), row_safe)
            x_row_base_div4.append(row_idx * c_k_div4)

        def load_b_packs_k64(k_tile: int, ku: int, ni: int):
            k0 = arith.constant(k_tile * k_unroll + ku, index=True)
            idx_pack = (
                expert_b_base
                + n_blk_list[ni] * arith.constant(_b_stride_n0, index=True)
                + k0 * arith.constant(_b_stride_k0, index=True)
                + lane_div_16 * arith.constant(_b_stride_klane, index=True)
                + n_intra_list[ni] * arith.constant(_b_stride_nlane, index=True)
            )
            b16 = _buffer_load_vec(
                buffer_ops,
                vector,
                w_rsrc,
                idx_pack,
                elem_type=T.i8,
                vec_elems=16,
                elem_bytes=1,
                offset_in_bytes=True,
                cache_modifier=0,
            )
            b_i64x2 = vector.bitcast(vec2_i64, b16)
            return (
                vector.extract(b_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(b_i64x2, static_position=[1], dynamic_position=[]),
            )

        def load_b_tile(k_tile: int):
            b_tile = []
            for ku in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for ni in range_constexpr(num_acc_n):
                    b0, b1 = load_b_packs_k64(k_tile, ku, ni)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        def make_empty_b_tile():
            b_packs0 = []
            b_packs1 = []
            for _ku in range_constexpr(k_unroll):
                b_packs0.append([None] * num_acc_n)
                b_packs1.append([None] * num_acc_n)
            return b_packs0, b_packs1

        def finish_b_tile(b_packs0, b_packs1):
            b_tile = []
            for ku in range_constexpr(k_unroll):
                b_tile.append((b_packs0[ku], b_packs1[ku]))
            return b_tile

        lane_word = lane_div_16 * arith.constant(16, index=True) + lane_mod_16

        def load_a_scale_tile(k_tile: int):
            a_scale_tile = []
            for mi in range_constexpr(m_repeat_packed):
                scale_idx = (
                    arith.constant(mi * k_as_per_chunk_dw, index=True)
                    + arith.constant(k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = vector.extract(
                    vector.load_op(T.vec(1, i32), lds_scale_i32, [scale_idx]),
                    static_position=[0],
                    dynamic_position=[],
                )
                a_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return a_scale_tile

        def load_b_scale_tile(k_tile: int):
            b_scale_tile = []
            mni_base = by * arith.constant(tile_n // 16 // 2, index=True)
            mni_base = mni_base + wave_id * arith.constant(n_per_wave // 16 // 2, index=True)
            for ni in range_constexpr(num_acc_n_packed):
                scale_idx = (
                    expert_idx * arith.constant(k_bs_per_expert_dw, index=True)
                    + (mni_base + arith.constant(ni, index=True))
                    * arith.constant(k_bs_stride_n0_dw, index=True)
                    + arith.constant(k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = buffer_ops.buffer_load(sw_rsrc, scale_idx, vec_width=1, dtype=i32)
                b_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return b_scale_tile

        def lds_load_packs_k64(lds_x_tile, curr_row_a_lds, col_base):
            col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
            loaded_a16 = vector.load_op(vec16_x, lds_x_tile, [idx_a16])
            a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
            return (
                vector.extract(a_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(a_i64x2, static_position=[1], dynamic_position=[]),
            )

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
            return vector.bitcast(vec8_i32, v4)

        c0_i64 = arith.constant(0, type=i64)
        cbsz = 4
        blgp = 4

        def compute_tile(acc_in, lds_x_tile, b_tile_in, a_scale, b_scale):
            acc_list = list(acc_in)
            for k_idx in range_constexpr(k_unroll):
                b_packs0, b_packs1 = b_tile_in[k_idx]
                col_base = col_offset_base + arith.constant(k_idx * 128 // a_elem_vec_pack, index=True)
                for mi in range_constexpr(m_repeat_packed):
                    a_scale_i32 = a_scale[mi]
                    a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                    for ni in range_constexpr(num_acc_n_packed):
                        b_scale_i32 = b_scale[ni]
                        b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi * pack_M + imxdl
                            curr_row_a_lds = row_a_lds + arith.constant(mi_idx * 16, index=True)
                            a0, a1 = lds_load_packs_k64(lds_x_tile, curr_row_a_lds, col_base)
                            a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                            for inxdl in range_constexpr(pack_N):
                                ni_idx = ni * pack_N + inxdl
                                b0 = b_packs0[ni_idx]
                                b1 = b_packs1[ni_idx]
                                b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                                acc_idx = mi_idx * num_acc_n + ni_idx
                                acc_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    vec4_f32,
                                    [
                                        a128,
                                        b128,
                                        acc_list[acc_idx],
                                        cbsz,
                                        blgp,
                                        k_idx * _scale_pack_m + imxdl,
                                        a_scale_val,
                                        k_idx * _scale_pack_n + inxdl,
                                        b_scale_val,
                                    ],
                                )
            return acc_list

        def compute_tile_jmajor(acc_in, lds_x_tile, b_tile_in, a_scale, b_scale):
            acc_list = list(acc_in)
            a_pairs = []
            for k_idx in range_constexpr(k_unroll):
                col_base = col_offset_base + arith.constant(k_idx * 128 // a_elem_vec_pack, index=True)
                for mi_idx in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(lds_x_tile, curr_row_a_lds, col_base)
                    a_pairs.append(pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64))

            for ni in range_constexpr(num_acc_n_packed):
                b_scale_i32 = b_scale[ni]
                b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                for inxdl in range_constexpr(pack_N):
                    ni_idx = ni * pack_N + inxdl
                    for mi in range_constexpr(m_repeat_packed):
                        a_scale_i32 = a_scale[mi]
                        a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi * pack_M + imxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            for k_idx in range_constexpr(k_unroll):
                                b_packs0, b_packs1 = b_tile_in[k_idx]
                                b0 = b_packs0[ni_idx]
                                b1 = b_packs1[ni_idx]
                                b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                                a128 = a_pairs[k_idx * m_repeat + mi_idx]
                                acc_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    vec4_f32,
                                    [
                                        a128,
                                        b128,
                                        acc_list[acc_idx],
                                        cbsz,
                                        blgp,
                                        k_idx * _scale_pack_m + imxdl,
                                        a_scale_val,
                                        k_idx * _scale_pack_n + inxdl,
                                        b_scale_val,
                                    ],
                                )
            return acc_list

        def compute_tile_jmajor_prefetch_a_b(
            acc_in,
            lds_x_tile,
            b_tile_in,
            a_scale,
            b_scale,
            next_a_tile: int,
            next_a_slot: int,
            next_b_tile: int,
        ):
            acc_list = list(acc_in)
            next_b_packs0, next_b_packs1 = make_empty_b_tile()
            a_pairs = []
            for k_idx in range_constexpr(k_unroll):
                col_base = col_offset_base + arith.constant(k_idx * 128 // a_elem_vec_pack, index=True)
                for mi_idx in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(lds_x_tile, curr_row_a_lds, col_base)
                    a_pairs.append(pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64))

            dma_x_tile_to_lds(next_a_tile, lds_x_slots[next_a_slot])

            for ni in range_constexpr(num_acc_n_packed):
                b_scale_i32 = b_scale[ni]
                b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                for inxdl in range_constexpr(pack_N):
                    ni_idx = ni * pack_N + inxdl
                    for mi in range_constexpr(m_repeat_packed):
                        a_scale_i32 = a_scale[mi]
                        a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi * pack_M + imxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            for k_idx in range_constexpr(k_unroll):
                                b_packs0, b_packs1 = b_tile_in[k_idx]
                                b0 = b_packs0[ni_idx]
                                b1 = b_packs1[ni_idx]
                                b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                                a128 = a_pairs[k_idx * m_repeat + mi_idx]
                                acc_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    vec4_f32,
                                    [
                                        a128,
                                        b128,
                                        acc_list[acc_idx],
                                        cbsz,
                                        blgp,
                                        k_idx * _scale_pack_m + imxdl,
                                        a_scale_val,
                                        k_idx * _scale_pack_n + inxdl,
                                        b_scale_val,
                                    ],
                                )

                    rocdl.sched_barrier(0)
                    for k_idx in range_constexpr(k_unroll):
                        nb0, nb1 = load_b_packs_k64(next_b_tile, k_idx, ni_idx)
                        next_b_packs0[k_idx][ni_idx] = nb0
                        next_b_packs1[k_idx][ni_idx] = nb1
                    rocdl.sched_barrier(0)

            b_scale_next = load_b_scale_tile(next_b_tile)
            return acc_list, finish_b_tile(next_b_packs0, next_b_packs1), b_scale_next

        _dma_bytes = 16
        _wave_size = 64

        def dma_x_tile_to_lds(k_tile: int, lds_x_tile):
            c4_idx = arith.index(4)
            base_k_div4 = arith.constant(k_tile * tile_k // a_elem_vec_pack // 4, index=True)
            lds_ptr_i64 = None
            for i in range_constexpr(num_x_loads):
                row_local_i = x_row_local[i]
                col_local_i32_i = x_col_local_i32[i]
                col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                row_k_dw = x_row_base_div4[i] + base_k_div4
                global_byte_idx = row_k_dw * c4_idx + col_local_sw
                global_offset = arith.index_cast(i32, global_byte_idx)
                if const_expr(i == 0):
                    lds_addr = (
                        memref.extract_aligned_pointer_as_index(lds_x_tile)
                        + wave_id * arith.constant(_wave_size * _dma_bytes, index=True)
                    )
                    lds_ptr_i64 = rocdl.readfirstlane(i64, arith.index_cast(i64, lds_addr))
                else:
                    lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=i64)
                lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_i64)
                rocdl.raw_ptr_buffer_load_lds(
                    x_rsrc,
                    lds_ptr,
                    arith.constant(_dma_bytes, type=i32),
                    global_offset,
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                )

        def dma_a_scales_to_lds():
            c4_idx = arith.index(4)
            c16_idx = arith.index(16)
            scale_chunk_bytes = arith.constant(k_as_per_chunk_dw * 4, index=True)
            chunk_base = bx_m / arith.constant(32, index=True)
            lds_scale_base = memref.extract_aligned_pointer_as_index(lds_scale_i8)
            tx_bytes16 = tx * c16_idx
            tx_bytes4 = tx * c4_idx
            for sub in range_constexpr(tile_m // 32):
                sub_idx = arith.constant(sub, index=True)
                global_sub_base = (chunk_base + sub_idx) * scale_chunk_bytes
                lds_sub_base = sub_idx * scale_chunk_bytes

                lds_addr = (
                    lds_scale_base
                    + lds_sub_base
                    + wave_id * arith.constant(1024, index=True)
                )
                lds_ptr_i64 = rocdl.readfirstlane(i64, arith.index_cast(i64, lds_addr))
                lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_i64)
                global_offset = arith.index_cast(i32, global_sub_base + tx_bytes16)
                rocdl.raw_ptr_buffer_load_lds(
                    sx_rsrc,
                    lds_ptr,
                    arith.constant(16, type=i32),
                    global_offset,
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                )

                for tail in range_constexpr(3):
                    byte_off = arith.constant(4096 + tail * 1024, index=True)
                    lds_addr_tail = (
                        lds_scale_base
                        + lds_sub_base
                        + byte_off
                        + wave_id * arith.constant(256, index=True)
                    )
                    lds_ptr_tail_i64 = rocdl.readfirstlane(
                        i64,
                        arith.index_cast(i64, lds_addr_tail),
                    )
                    lds_ptr_tail = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_tail_i64)
                    global_offset_tail = arith.index_cast(
                        i32,
                        global_sub_base + byte_off + tx_bytes4,
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        sx_rsrc,
                        lds_ptr_tail,
                        arith.constant(4, type=i32),
                        global_offset_tail,
                        arith.constant(0, type=i32),
                        arith.constant(0, type=i32),
                        arith.constant(0, type=i32),
                    )

        def _then_body():
            acc = [arith.constant_vector(0.0, vec4_f32)] * (m_repeat * num_acc_n)
            dma_a_scales_to_lds()
            rocdl.s_waitcnt(0)
            gpu.barrier()

            dma_x_tile_to_lds(0, lds_x_slots[0])
            dma_x_tile_to_lds(1, lds_x_slots[1])
            b_tiles = [load_b_tile(0), load_b_tile(1)]
            b_scales = [load_b_scale_tile(0), load_b_scale_tile(1)]
            # Leave both initial B/B-scale tiles outstanding if possible; A0
            # and A1 must be resident in LDS before the first two compute tiles.
            rocdl.s_waitcnt(20)
            gpu.barrier()
            a_scale = load_a_scale_tile(0)

            for k_tile in range_constexpr(num_k_tiles):
                curr_slot = k_tile % 2
                curr_b_slot = k_tile % 2
                next_slot = k_tile % 2
                if const_expr(k_tile + 2 < num_k_tiles):
                    acc, b_tile_next, b_scale_next = compute_tile_jmajor_prefetch_a_b(
                        acc,
                        lds_x_slots[curr_slot],
                        b_tiles[curr_b_slot],
                        a_scale,
                        b_scales[curr_b_slot],
                        k_tile + 2,
                        next_slot,
                        k_tile + 2,
                    )
                else:
                    acc = compute_tile_jmajor(
                        acc,
                        lds_x_slots[curr_slot],
                        b_tiles[curr_b_slot],
                        a_scale,
                        b_scales[curr_b_slot],
                    )

                if const_expr(k_tile + 1 < num_k_tiles):
                    # A and B are both two-ahead.  When a future tile was
                    # issued, keep that tile's 4 A + 10 B/B-scale VMEM loads
                    # outstanding; the next iteration consumes the older tile.
                    if const_expr(k_tile + 2 < num_k_tiles):
                        rocdl.s_waitcnt(14)
                    else:
                        rocdl.s_waitcnt(0)
                    gpu.barrier()
                    a_scale = load_a_scale_tile(k_tile + 1)
                    if const_expr(k_tile + 2 < num_k_tiles):
                        b_tiles[curr_b_slot] = b_tile_next
                        b_scales[curr_b_slot] = b_scale_next

            gpu.barrier()

            c0_f32 = arith.constant(0.0, type=f32)
            c1_f32 = arith.constant(1.0, type=f32)
            c025_f32 = arith.constant(0.25, type=f32)
            neg_log2e = arith.constant(-1.4426950408889634, type=f32)
            c0_i32 = arith.constant(0, type=i32)
            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c3_i32 = arith.constant(3, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c8_i32 = arith.constant(8, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c23_i32 = arith.constant(23, type=i32)
            c254_i32 = arith.constant(254, type=i32)
            c0x200000_i32 = arith.constant(0x200000, type=i32)

            for mi in range_constexpr(m_repeat):
                row_base = arith.constant(mi * 16, index=True) + lane_div_16 * arith.constant(4, index=True)
                for j in range_constexpr(num_acc_n):
                    j_local = j // 2
                    col_local = (
                        wave_id * arith.constant(32, index=True)
                        + arith.constant(j_local * 16, index=True)
                        + lane_mod_16
                    )
                    if const_expr(j % 2 == 1):
                        col_local = col_local + arith.constant(128, index=True)
                    for v_i in range_constexpr(4):
                        row = row_base + arith.constant(v_i, index=True)
                        val = vector.extract(acc[mi * num_acc_n + j], static_position=[v_i], dynamic_position=[])
                        vector.store(
                            vector.from_elements(T.vec(1, f32), [val]),
                            lds_acc,
                            [row * arith.constant(tile_n, index=True) + col_local],
                            alignment=4,
                        )

            gpu.barrier()

            m_lane = tx / arith.constant(16, index=True)
            n_lane = tx % arith.constant(16, index=True)
            wave_grp = n_lane / arith.constant(4, index=True)
            kk = n_lane % arith.constant(4, index=True)
            scales_per_mr = []

            def _silu_mul(g, u):
                t = g * neg_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = c1_f32 + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return g * sig * u

            def _max_u32(a, b):
                return arith.select(arith.cmpi(CmpIPredicate.ugt, b, a), b, a)

            for mr in range_constexpr(m_repeat):
                row_local = arith.constant(mr * 16, index=True) + m_lane
                result_vals = []
                local_max = c0_f32
                for e in range_constexpr(8):
                    col_in_grp = kk * arith.constant(8, index=True) + arith.constant(e, index=True)
                    gate_col = wave_grp * arith.constant(32, index=True) + col_in_grp
                    up_col = gate_col + arith.constant(128, index=True)
                    gate_v = vector.extract(
                        vector.load_op(T.vec(1, f32), lds_acc, [row_local * arith.constant(tile_n, index=True) + gate_col]),
                        static_position=[0],
                        dynamic_position=[],
                    )
                    up_v = vector.extract(
                        vector.load_op(T.vec(1, f32), lds_acc, [row_local * arith.constant(tile_n, index=True) + up_col]),
                        static_position=[0],
                        dynamic_position=[],
                    )
                    res = _silu_mul(gate_v, up_v)
                    result_vals.append(res)
                    abs_v = llvm.call_intrinsic(f32, "llvm.fabs.f32", [res], [], [])
                    local_max = arith.maximumf(local_max, abs_v)

                peer1 = _dpp_xor_f32(local_max, 1)
                local_max = arith.maximumf(local_max, peer1)
                peer2 = _dpp_xor_f32(local_max, 2)
                local_max = arith.maximumf(local_max, peer2)

                amax_i32 = local_max.bitcast(i32)
                quant_scale = (amax_i32 + c0x200000_i32).bitcast(f32) * c025_f32
                sb_raw = quant_scale.bitcast(i32) >> c23_i32
                e8m0 = arith.select(arith.cmpi(CmpIPredicate.ugt, sb_raw, c254_i32), c254_i32, sb_raw)
                scales_per_mr.append(e8m0)

                packed = c0_i32
                for pair in range_constexpr(4):
                    packed = ArithValue(
                        llvm.call_intrinsic(
                            i32,
                            "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                            [
                                packed,
                                result_vals[2 * pair],
                                result_vals[2 * pair + 1],
                                quant_scale,
                                arith.constant(pair, type=i32),
                            ],
                            [],
                            [],
                        )
                    )

                row_i32 = arith.index_cast(i32, bx_m + row_local)
                byte_pos = (
                    by * arith.constant(64, index=True)
                    + wave_grp * arith.constant(16, index=True)
                    + kk * arith.constant(4, index=True)
                )
                byte_pos_i32 = arith.index_cast(i32, byte_pos)
                q_word_off = row_i32 * arith.constant((INTER_DIM // 2) // 4, type=i32) + (byte_pos_i32 >> c2_i32)
                buffer_ops.buffer_store(packed, out_q_rsrc, q_word_off, cache_modifier=4)

            _if_scale = scf.IfOp(arith.cmpi(CmpIPredicate.eq, kk, arith.constant(0, index=True)))
            with ir.InsertionPoint(_if_scale.then_block):
                ku = by >> arith.constant(1, index=True)
                ikxdl = by % arith.constant(2, index=True)
                for sub in range_constexpr(tile_m // 32):
                    chunk = bx * arith.constant(tile_m // 32, index=True) + arith.constant(sub, index=True)
                    dword_off = (
                        chunk * arith.constant(k_out_as_per_chunk_dw, index=True)
                        + ku * arith.constant(64, index=True)
                        + wave_grp * arith.constant(16, index=True)
                        + m_lane
                    )
                    lo = scales_per_mr[sub * 2]
                    hi = scales_per_mr[sub * 2 + 1]
                    pair_i32 = lo | (hi << c8_i32)
                    pair_i16 = arith.TruncIOp(T.i16, pair_i32).result
                    byte_off = arith.index_cast(i32, dword_off * arith.constant(4, index=True) + ikxdl * arith.constant(2, index=True))
                    buffer_ops.buffer_store(
                        pair_i16,
                        out_scale_rsrc,
                        byte_off,
                        cache_modifier=4,
                        offset_is_bytes=True,
                    )
                scf.YieldOp([])

        _if_valid = scf.IfOp(do_gemm)
        with ir.InsertionPoint(_if_valid.then_block):
            _then_body()
            scf.YieldOp([])

    @flyc.jit
    def launch_gemm1(
        inter_sorted_quant: fx.Pointer,
        inter_sorted_scale: fx.Pointer,
        a_quant: fx.Pointer,
        w: fx.Pointer,
        a_scale_sorted: fx.Pointer,
        w_scale: fx.Pointer,
        expert_ids: fx.Pointer,
        m_indices: fx.Pointer,
        num_valid_ids: fx.Pointer,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm1(
            inter_sorted_quant,
            inter_sorted_scale,
            a_quant,
            w,
            a_scale_sorted,
            w_scale,
            expert_ids,
            m_indices,
            num_valid_ids,
        ).launch(
            grid=(num_n_blocks, num_m_blocks, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm1


def kimi_mxfp4_gemm1_16384(
    a_quant: torch.Tensor,
    a_scale_sorted_shuffled: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    m_indices: torch.Tensor,
    cumsum_tensor: torch.Tensor,
    inter_sorted_quant: torch.Tensor | None = None,
    inter_sorted_shuffled_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL candidate for aiter's fixed Kimi mxfp4 GEMM1 kernel."""
    device = a_quant.device
    max_sorted = mxfp4_max_sorted_16384()
    if inter_sorted_quant is None:
        inter_sorted_quant = torch.empty((max_sorted, INTER_DIM // 2), device=device, dtype=torch.uint8)
    if inter_sorted_shuffled_scale is None:
        inter_scale_cols = INTER_DIM // 32
        inter_scale_bytes = max_sorted * (1024 // 64) * 4
        inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
        inter_scale_rows = ((inter_scale_rows + 31) // 32) * 32
        inter_sorted_shuffled_scale = torch.empty(
            (inter_scale_rows, inter_scale_cols),
            device=device,
            dtype=torch.uint8,
        )

    args = (
        _ptr_view_safe(_as_u8_storage(inter_sorted_quant).view(-1)),
        _ptr_view_safe(_as_u8_storage(inter_sorted_shuffled_scale).view(-1)),
        _ptr_view_safe(_as_u8_storage(a_quant).view(-1)),
        _ptr_view_safe(_as_u8_storage(w1).view(-1)),
        _ptr_view_safe(_as_u8_storage(a_scale_sorted_shuffled).view(-1)),
        _ptr_view_safe(_as_u8_storage(w1_scale).view(-1)),
        _ptr_view_safe(sorted_expert_ids.view(-1)),
        _ptr_view_safe(m_indices.view(-1)),
        _ptr_view_safe(cumsum_tensor.view(-1)),
        torch.cuda.current_stream(),
    )
    _run_compiled(compile_kimi_mxfp4_gemm1_16384(), args)
    return inter_sorted_quant, inter_sorted_shuffled_scale


@functools.lru_cache(maxsize=1)
def compile_kimi_mxfp4_gemm2_mxfp4out_16384():
    """Compile a fixed-shape FlyDSL GEMM2 that stages MXFP4 flat output.

    This is the first direct FlyDSL port of aiter's Kimi-only
    ``mxfp4_moe_gemm2_a4w4_mxfp4out`` path.  It keeps the public shape fixed:

    - A: sorted fp4, ``[max_sorted, 512 / 2]``
    - B: preshuffled fp4, ``[385, 7168, 512 / 2]``
    - C: packed fp4 plus e8m0 block scales, sorted-row-major

    The initial version uses a simple 2D CTA grid for debuggability.  The aiter
    kernel uses a persistent 1D grid and AGPR MFMA accumulation; those are the
    next performance-alignment steps once this path is numerically stable.
    """

    tile_m = MXFP4_BLOCK_M
    tile_n = 256
    tile_k = 256
    total_threads = 256
    max_sorted = mxfp4_max_sorted_16384()
    num_m_blocks = max_sorted // tile_m
    num_n_blocks = MODEL_DIM // tile_n

    _scale_pack_m = 2
    _scale_pack_n = 2
    _scale_pack_k = 2
    pack_M = 2
    pack_N = 2
    pack_K = 2
    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    k_unroll = 2
    m_repeat_packed = m_repeat // pack_M
    num_acc_n_packed = num_acc_n // pack_N

    a_elem_bytes = 1
    b_elem_bytes = 1
    a_elem_vec_pack = 2
    tile_k_bytes = tile_k * a_elem_bytes
    _eff_lds_stride = tile_k // a_elem_vec_pack
    _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
    _single_x_bytes = tile_m * _eff_lds_stride * a_elem_bytes
    _input_elems = _single_x_bytes
    _lds_acc_bytes = tile_m * tile_n * 4
    _lds_x_bytes = 2 * _single_x_bytes
    _lds_scale_bytes = 2 * (tile_m // 32) * 256

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_mxfp4_gemm2")
    lds_x0_offset = allocator._align(allocator.ptr, 16)
    lds_x1_offset = lds_x0_offset + _single_x_bytes
    # Overlay the later f32 epilogue tile with the earlier A staging area.
    allocator.ptr = lds_x0_offset + max(_lds_x_bytes, _lds_acc_bytes)
    lds_scale_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_scale_offset + _lds_scale_bytes

    q_nbytes = max_sorted * (MODEL_DIM // 2)
    out_scale_nbytes = max_sorted * (MODEL_DIM // 32)
    x_nbytes = max_sorted * (INTER_DIM // 2)
    x_scale_nbytes = max_sorted * (1024 // 64) * 4
    w_nbytes = EXPERTS * MODEL_DIM * (INTER_DIM // 2)
    w_scale_nbytes = EXPERTS * MODEL_DIM * (INTER_DIM // 32)
    expert_nbytes = num_m_blocks * 4
    numids_nbytes = 4

    _b_kpack_bytes_s = 16
    _b_kpack_elems_s = _b_kpack_bytes_s // b_elem_bytes
    _b_c_k_s = INTER_DIM // _scale_pack_k
    _b_c_k0_s = (_b_c_k_s * b_elem_bytes) // 64
    _b_stride_nlane = _b_kpack_elems_s
    _b_stride_klane = 16 * _b_stride_nlane
    _b_stride_k0 = 4 * _b_stride_klane
    _b_stride_n0 = _b_c_k0_s * _b_stride_k0
    _expert_b_stride = (MODEL_DIM // 16) * _b_stride_n0

    # aiter scale-layout constants for K=512, N=7168, BM=128.
    k_as_per_chunk_dw = 128
    k_bs_stride_k0_dw = 64
    k_bs_stride_n0_dw = 128
    k_bs_per_expert_dw = (MODEL_DIM // 16 // 2) * k_bs_stride_n0_dw

    module_name = "flydsl_kimi_mxfp4_gemm2_mxfp4out_NE385_H7168_E512_BM128_v0"

    @flyc.kernel(name=module_name)
    def gemm2_mxfp4out(
        arg_flat_out_q: fx.Pointer,
        arg_flat_out_scale: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
    ):
        i32 = T.i32
        i64 = T.i64
        f32 = T.f32
        vec4_f32 = T.vec(4, f32)
        vec2_i64 = T.vec(2, i64)
        vec4_i64 = T.vec(4, i64)
        vec8_i32 = T.vec(8, i32)
        vec16_x = T.vec(16, T.f8)

        q_rsrc = _ptr_buffer_resource(arg_flat_out_q, q_nbytes)
        out_scale_rsrc = _ptr_buffer_resource(arg_flat_out_scale, out_scale_nbytes)
        x_rsrc = _ptr_buffer_resource(arg_x, x_nbytes)
        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)
        sx_rsrc = _ptr_buffer_resource(arg_scale_x, x_scale_nbytes)
        sw_rsrc = _ptr_buffer_resource(arg_scale_w, w_scale_nbytes)
        expert_rsrc = _ptr_buffer_resource(arg_expert_ids, expert_nbytes)
        numids_rsrc = _ptr_buffer_resource(arg_num_valid_ids, numids_nbytes)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")
        bx = gpu.block_id("y")
        bx_m = bx * arith.constant(tile_m, index=True)
        bx_m_i32 = arith.index_cast(i32, bx_m)

        num_valid_i32 = buffer_ops.buffer_load(
            numids_rsrc,
            arith.constant(0, index=True),
            vec_width=1,
            dtype=i32,
        )
        num_valid_i32 = rocdl.ReadfirstlaneOp(i32, num_valid_i32).res
        blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

        expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
        expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(EXPERTS, type=i32))
        do_gemm = arith.andi(blk_valid, exp_valid)

        base_ptr = allocator.get_base()
        lds_x0 = SmemPtr(base_ptr, lds_x0_offset, T.f8, shape=(_input_elems,)).get()
        lds_x1 = SmemPtr(base_ptr, lds_x1_offset, T.f8, shape=(_input_elems,)).get()
        lds_out = SmemPtr(base_ptr, lds_x0_offset, T.f32, shape=(tile_m * tile_n,)).get()

        shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
        stride_lds = fx.make_stride(_eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)
        k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_wl = idx2crd(tx, layout_tx_wave_lane)
        wave_id = layout_get(coord_wl, 0)
        lane_id = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * arith.constant(16, index=True)
        n_tile_base = wave_id * arith.constant(n_per_wave, index=True)
        by_n = by * arith.constant(tile_n, index=True)
        expert_b_base = expert_idx * arith.constant(_expert_b_stride, index=True)

        n_intra_list = [None] * num_acc_n
        n_blk_list = [None] * num_acc_n
        for ni in range_constexpr(num_acc_n):
            c_offset = arith.constant(ni * 16, index=True)
            global_n = by_n + n_tile_base + c_offset + lane_mod_16
            n_blk_list[ni] = _div_pow2(global_n, 16)
            n_intra_list[ni] = global_n % arith.constant(16, index=True)

        bytes_per_thread_x = tile_m * tile_k * a_elem_bytes // total_threads
        x_load_bytes = 16
        num_x_loads = bytes_per_thread_x // x_load_bytes
        chunk_i32 = x_load_bytes // 4
        tile_k_dwords = tile_k * a_elem_bytes // (4 * a_elem_vec_pack)
        layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
        c_chunk_i32 = arith.constant(chunk_i32, index=True)
        tx_i32_base = tx * c_chunk_i32
        c_k_div4 = arith.constant((INTER_DIM // a_elem_vec_pack) // 4, index=True)

        x_row_base_div4 = []
        x_col_local_i32 = []
        x_row_local = []
        for i in range_constexpr(num_x_loads):
            row_local, col_local_i32 = tile_chunk_coord_i32(
                arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_x_tile_div4,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(row_local)
            x_col_local_i32.append(col_local_i32)
            x_row_base_div4.append((bx_m + row_local) * c_k_div4)

        def load_b_packs_k64(base_k_packed, ku: int, ni: int):
            k0 = _div_pow2(base_k_packed, 64) + arith.constant(ku, index=True)
            idx_pack = (
                expert_b_base
                + n_blk_list[ni] * arith.constant(_b_stride_n0, index=True)
                + k0 * arith.constant(_b_stride_k0, index=True)
                + lane_div_16 * arith.constant(_b_stride_klane, index=True)
                + n_intra_list[ni] * arith.constant(_b_stride_nlane, index=True)
            )
            b16 = _buffer_load_vec(
                buffer_ops,
                vector,
                w_rsrc,
                idx_pack,
                elem_type=T.i8,
                vec_elems=16,
                elem_bytes=1,
                offset_in_bytes=True,
                cache_modifier=0,
            )
            b_i64x2 = vector.bitcast(vec2_i64, b16)
            return (
                vector.extract(b_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(b_i64x2, static_position=[1], dynamic_position=[]),
            )

        def load_b_tile(base_k_packed):
            b_tile = []
            for ku in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for ni in range_constexpr(num_acc_n):
                    b0, b1 = load_b_packs_k64(base_k_packed, ku, ni)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        def load_a_scale_tile(scale_k_tile: int):
            a_scale_tile = []
            chunk_base = bx_m / arith.constant(32, index=True)
            lane_word = lane_div_16 * arith.constant(16, index=True) + lane_mod_16
            for mi in range_constexpr(m_repeat_packed):
                scale_idx = (
                    (chunk_base + arith.constant(mi, index=True))
                    * arith.constant(k_as_per_chunk_dw, index=True)
                    + arith.constant(scale_k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = buffer_ops.buffer_load(sx_rsrc, scale_idx, vec_width=1, dtype=i32)
                a_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return a_scale_tile

        def load_b_scale_tile(scale_k_tile: int):
            b_scale_tile = []
            mni_base = by * arith.constant(tile_n // 16 // 2, index=True)
            mni_base = mni_base + wave_id * arith.constant(n_per_wave // 16 // 2, index=True)
            lane_word = lane_div_16 * arith.constant(16, index=True) + lane_mod_16
            for ni in range_constexpr(num_acc_n_packed):
                scale_idx = (
                    expert_idx * arith.constant(k_bs_per_expert_dw, index=True)
                    + (mni_base + arith.constant(ni, index=True))
                    * arith.constant(k_bs_stride_n0_dw, index=True)
                    + arith.constant(scale_k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = buffer_ops.buffer_load(sw_rsrc, scale_idx, vec_width=1, dtype=i32)
                b_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return b_scale_tile

        def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
            col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
            loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
            a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
            return (
                vector.extract(a_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(a_i64x2, static_position=[1], dynamic_position=[]),
            )

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
            return vector.bitcast(vec8_i32, v4)

        c0_i64 = arith.constant(0, type=i64)
        cbsz = 4
        blgp = 4

        def compute_tile(acc_in, b_tile_in, lds_buffer, a_scale, b_scale):
            acc_list = list(acc_in)
            for k_idx in range_constexpr(k_unroll):
                ikxdl = k_idx
                b_packs0, b_packs1 = b_tile_in[k_idx]
                col_base = col_offset_base + arith.constant(k_idx * 128 // a_elem_vec_pack, index=True)
                for mi in range_constexpr(m_repeat_packed):
                    a_scale_i32 = a_scale[mi]
                    a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                    for ni in range_constexpr(num_acc_n_packed):
                        b_scale_i32 = b_scale[ni]
                        b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi * pack_M + imxdl
                            curr_row_a_lds = row_a_lds + arith.constant(mi_idx * 16, index=True)
                            a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer)
                            a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                            for inxdl in range_constexpr(pack_N):
                                ni_idx = ni * pack_N + inxdl
                                b0 = b_packs0[ni_idx]
                                b1 = b_packs1[ni_idx]
                                b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                                acc_idx = mi_idx * num_acc_n + ni_idx
                                acc_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    vec4_f32,
                                    [
                                        a128,
                                        b128,
                                        acc_list[acc_idx],
                                        cbsz,
                                        blgp,
                                        ikxdl * _scale_pack_m + imxdl,
                                        a_scale_val,
                                        ikxdl * _scale_pack_n + inxdl,
                                        b_scale_val,
                                    ],
                                )
            return acc_list

        _dma_bytes = 16
        _wave_size = 64

        def dma_x_tile_to_lds(base_k, lds_buffer):
            c4_idx = arith.index(4)
            base_k_div4 = _div_pow2(_div_pow2(base_k, a_elem_vec_pack), 4)
            lds_ptr_i64 = None
            for i in range_constexpr(num_x_loads):
                row_local_i = x_row_local[i]
                col_local_i32_i = x_col_local_i32[i]
                col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                row_k_dw = x_row_base_div4[i] + base_k_div4
                global_byte_idx = row_k_dw * c4_idx + col_local_sw
                global_offset = arith.index_cast(i32, global_byte_idx)
                if const_expr(i == 0):
                    lds_addr = (
                        memref.extract_aligned_pointer_as_index(lds_buffer)
                        + wave_id * arith.constant(_wave_size * _dma_bytes, index=True)
                    )
                    lds_ptr_i64 = rocdl.readfirstlane(i64, arith.index_cast(i64, lds_addr))
                else:
                    lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=i64)
                lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_i64)
                rocdl.raw_ptr_buffer_load_lds(
                    x_rsrc,
                    lds_ptr,
                    arith.constant(_dma_bytes, type=i32),
                    global_offset,
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                )

        def _then_body():
            rocdl.sched_barrier(0)
            dma_x_tile_to_lds(arith.index(0), lds_x0)
            dma_x_tile_to_lds(arith.index(tile_k), lds_x1)
            b_tile0 = load_b_tile(arith.index(0))
            b_tile1 = load_b_tile(arith.index(tile_k // a_elem_vec_pack))
            a_scale0 = load_a_scale_tile(0)
            a_scale1 = load_a_scale_tile(1)
            b_scale0 = load_b_scale_tile(0)
            b_scale1 = load_b_scale_tile(1)
            rocdl.s_waitcnt(0)
            gpu.barrier()

            acc = [arith.constant_vector(0.0, vec4_f32)] * (m_repeat * num_acc_n)
            acc = compute_tile(acc, b_tile0, lds_x0, a_scale0, b_scale0)
            acc = compute_tile(acc, b_tile1, lds_x1, a_scale1, b_scale1)

            c0_f32 = arith.constant(0.0, type=f32)
            c0_i32 = arith.constant(0, type=i32)
            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c3_i32 = arith.constant(3, type=i32)
            c8_i32 = arith.constant(8, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c23_i32 = arith.constant(23, type=i32)
            c255_i32 = arith.constant(255, type=i32)
            c254_i32 = arith.constant(254, type=i32)
            c0x200000_i32 = arith.constant(0x200000, type=i32)

            def write_row_to_lds(
                *,
                mi: int,
                ii: int,
                row_in_tile,
                row,
                row_base_lds,
                col_base_local,
                num_acc_n: int,
                lds_out,
            ):
                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + arith.constant(ni * 16, index=True)
                    acc_idx = mi * num_acc_n + ni
                    v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                    v1 = vector.from_elements(T.vec(1, f32), [v])
                    vector.store(v1, lds_out, [row_base_lds + col_local], alignment=4)

            def precompute_row(*, row_local, row):
                row_i32 = arith.index_cast(i32, row)
                row_valid = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                return (row, row_valid)

            def _max_u32(a, b):
                return arith.select(arith.cmpi(CmpIPredicate.ugt, b, a), b, a)

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                frag_vals = []
                for i in range_constexpr(8):
                    frag_vals.append(vector.extract(frag, static_position=[i], dynamic_position=[]))

                local_max = c0_f32
                for i in range_constexpr(8):
                    abs_v = llvm.call_intrinsic(f32, "llvm.fabs.f32", [frag_vals[i]], [], [])
                    local_max = arith.maximumf(local_max, abs_v)

                # Match aiter: reduce bf16-truncated amax across a 4-lane quad.
                amax_bf16 = local_max.bitcast(i32) >> c16_i32
                peer1 = amax_bf16.shuffle_xor(c1_i32, arith.constant(64, type=i32))
                amax_bf16 = _max_u32(amax_bf16, peer1)
                peer2 = amax_bf16.shuffle_xor(c2_i32, arith.constant(64, type=i32))
                amax_bf16 = _max_u32(amax_bf16, peer2)

                f32bits = amax_bf16 << c16_i32
                bexp = ((f32bits + c0x200000_i32) >> c23_i32) & c255_i32
                e8m0 = bexp - c2_i32
                e8m0 = arith.select(arith.cmpi(CmpIPredicate.slt, e8m0, c0_i32), c0_i32, e8m0)
                e8m0 = arith.select(arith.cmpi(CmpIPredicate.sgt, e8m0, c254_i32), c254_i32, e8m0)
                quant_scale = (e8m0 << c23_i32).bitcast(f32)

                packed = c0_i32
                for pair in range_constexpr(4):
                    packed = ArithValue(
                        llvm.call_intrinsic(
                            i32,
                            "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                            [
                                packed,
                                frag_vals[2 * pair],
                                frag_vals[2 * pair + 1],
                                quant_scale,
                                arith.constant(pair, type=i32),
                            ],
                            [],
                            [],
                        )
                    )

                row_i32 = arith.index_cast(i32, row)
                col_i32 = arith.index_cast(i32, col_g0)
                q_word_off = row_i32 * arith.constant(MODEL_DIM // 8, type=i32) + (col_i32 >> c3_i32)
                buffer_ops.buffer_store(packed, q_rsrc, q_word_off, cache_modifier=4)

                lane_in_scale = (col_i32 >> c3_i32) & c3_i32
                _if_scale = scf.IfOp(arith.cmpi(CmpIPredicate.eq, lane_in_scale, c0_i32))
                with ir.InsertionPoint(_if_scale.then_block):
                    scale_off = row_i32 * arith.constant(MODEL_DIM // 32, type=i32) + (col_i32 >> arith.constant(5, type=i32))
                    e8m0_i8 = arith.TruncIOp(T.i8, e8m0).result
                    buffer_ops.buffer_store(e8m0_i8, out_scale_rsrc, scale_off, cache_modifier=4)
                    scf.YieldOp([])

            c_shuffle_epilog(
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=tile_n,
                e_vec=8,
                cshuffle_nlane=32,
                block_size=total_threads,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lds_out=lds_out,
                frag_elem_type=ir.F32Type.get(),
                write_row_to_lds=write_row_to_lds,
                precompute_row=precompute_row,
                store_pair=store_pair,
            )

        _if_valid = scf.IfOp(do_gemm)
        with ir.InsertionPoint(_if_valid.then_block):
            _then_body()
            scf.YieldOp([])
        gpu.barrier()

    @flyc.jit
    def launch_gemm2_mxfp4out(
        flat_out_q: fx.Pointer,
        flat_out_scale: fx.Pointer,
        x: fx.Pointer,
        w: fx.Pointer,
        scale_x: fx.Pointer,
        scale_w: fx.Pointer,
        expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm2_mxfp4out(
            flat_out_q,
            flat_out_scale,
            x,
            w,
            scale_x,
            scale_w,
            expert_ids,
            num_valid_ids,
        ).launch(
            grid=(num_n_blocks, num_m_blocks, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm2_mxfp4out


def kimi_mxfp4_gemm2_mxfp4out_16384(
    inter_sorted_quant: torch.Tensor,
    inter_sorted_shuffled_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    cumsum_tensor: torch.Tensor,
    flat_out_q: torch.Tensor | None = None,
    flat_out_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL candidate for aiter's fixed Kimi GEMM2 MXFP4-output kernel."""
    max_sorted = inter_sorted_quant.shape[0]
    if max_sorted != mxfp4_max_sorted_16384():
        raise ValueError(f"expected max_sorted={mxfp4_max_sorted_16384()}, got {max_sorted}")
    if tuple(inter_sorted_quant.shape[1:]) != (INTER_DIM // 2,):
        raise ValueError(
            "expected inter_sorted_quant shape "
            f"(*, {INTER_DIM // 2}), got {tuple(inter_sorted_quant.shape)}"
        )
    device = inter_sorted_quant.device
    if flat_out_q is None:
        flat_out_q = torch.empty((max_sorted, MODEL_DIM // 2), dtype=torch.uint8, device=device)
    if flat_out_scale is None:
        flat_out_scale = torch.empty((max_sorted, MODEL_DIM // 32), dtype=torch.uint8, device=device)

    args = (
        _ptr_view_safe(_as_u8_storage(flat_out_q).view(-1)),
        _ptr_view_safe(_as_u8_storage(flat_out_scale).view(-1)),
        _ptr_view_safe(_as_u8_storage(inter_sorted_quant).view(-1)),
        _ptr_view_safe(_as_u8_storage(w2).view(-1)),
        _ptr_view_safe(_as_u8_storage(inter_sorted_shuffled_scale).view(-1)),
        _ptr_view_safe(_as_u8_storage(w2_scale).view(-1)),
        _ptr_view_safe(sorted_expert_ids.view(-1)),
        _ptr_view_safe(cumsum_tensor.view(-1)),
        torch.cuda.current_stream(),
    )
    _run_compiled(compile_kimi_mxfp4_gemm2_mxfp4out_16384(), args)
    return flat_out_q, flat_out_scale


def mxfp4_max_sorted_16384(block_m: int = MXFP4_BLOCK_M) -> int:
    active = min(EXPERTS, TOKEN * TOPK)
    cumsum_max = TOKEN * TOPK + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


def kimi_mxfp4_sort_16384(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    *,
    block_m: int = MXFP4_BLOCK_M,
) -> Mxfp4SortBuffers:
    """Run the fixed mxfp4_moe three-stage sorter for Kimi M=16384."""
    if tuple(topk_ids.shape) != (TOKEN, TOPK):
        raise ValueError(
            f"expected topk_ids shape {(TOKEN, TOPK)}, got {tuple(topk_ids.shape)}"
        )
    if tuple(topk_weight.shape) != (TOKEN, TOPK):
        raise ValueError(
            f"expected topk_weight shape {(TOKEN, TOPK)}, got {tuple(topk_weight.shape)}"
        )

    device = topk_ids.device
    max_sorted = mxfp4_max_sorted_16384(block_m)
    sorted_token_ids = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    sorted_expert_ids = torch.empty(
        (max_sorted // block_m,),
        device=device,
        dtype=dtypes.i32,
    )
    cumsum_tensor = torch.empty((1,), device=device, dtype=dtypes.i32)
    reverse_sorted = torch.empty((TOKEN * TOPK,), device=device, dtype=dtypes.i32)
    sorted_weights = torch.empty((max_sorted,), device=device, dtype=dtypes.fp32)
    masked_m = torch.empty((EXPERTS,), device=device, dtype=dtypes.i32)
    m_indices = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)

    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        masked_m=masked_m,
        m_indices=m_indices,
        bf16_zero_out=_empty_bf16(device),
        bf16_zero_workspace=_empty_bf16(device),
        M_logical=TOKEN,
        NE=EXPERTS,
        TOPK=TOPK,
        D_HIDDEN=MODEL_DIM,
        D_INTER=INTER_DIM,
        MB=block_m,
        prologue=1,
    )
    return Mxfp4SortBuffers(
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        masked_m=masked_m,
        m_indices=m_indices,
        max_sorted=max_sorted,
        block_m=block_m,
    )


def expand_mxfp4_expert_ids_for_flydsl(
    sorted_expert_ids: torch.Tensor,
    *,
    source_block_m: int = MXFP4_BLOCK_M,
    target_block_m: int = BLOCK_M,
) -> torch.Tensor:
    """Expand mxfp4 BM=128 expert ids into the BM=64 format FlyDSL expects."""
    if source_block_m == target_block_m:
        return sorted_expert_ids
    if source_block_m % target_block_m != 0:
        raise ValueError(
            f"cannot expand expert ids from block_m={source_block_m} "
            f"to target_block_m={target_block_m}"
        )
    return torch.repeat_interleave(
        sorted_expert_ids,
        source_block_m // target_block_m,
    )


def run_kimi_fp4_flydsl_mxfp4_sort_16384(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Use mxfp4_moe sorting, then run the existing fixed FlyDSL GEMMs."""
    sort = kimi_mxfp4_sort_16384(topk_ids, topk_weight)
    flydsl_expert_ids = expand_mxfp4_expert_ids_for_flydsl(
        sort.sorted_expert_ids,
        source_block_m=sort.block_m,
        target_block_m=BLOCK_M,
    )
    moe_out = torch.empty(
        (TOKEN, MODEL_DIM),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    a1, a1_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states,
        sorted_ids=sort.sorted_token_ids,
        num_valid_ids=sort.cumsum_tensor,
        token_num=TOKEN,
        topk=TOPK,
        block_size=sort.block_m,
        num_rows=None,
    )
    a2 = kimi_fp4_stage1_16384(
        a1,
        w1,
        w1_scale.view(dtypes.fp8_e8m0),
        a1_scale,
        sort.sorted_token_ids,
        flydsl_expert_ids,
        sort.cumsum_tensor,
    )
    a2_q, a2_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(
        a2.view(-1, INTER_DIM),
        sorted_ids=sort.sorted_token_ids,
        num_valid_ids=sort.cumsum_tensor,
        token_num=TOKEN,
        topk=TOPK,
        block_size=sort.block_m,
        num_rows=None,
    )
    return kimi_fp4_stage2_16384(
        a2_q.view(TOKEN, TOPK, -1),
        w2,
        w2_scale.view(dtypes.fp8_e8m0),
        a2_scale,
        sort.sorted_token_ids,
        sort.sorted_weights,
        flydsl_expert_ids,
        sort.cumsum_tensor,
        moe_out,
    )


def run_kimi_fp4_flydsl_atomic_stage2_16384(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Use the fixed FlyDSL stage1 and generic FlyDSL atomic stage2."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = (
        kimi_moe_sorting_16384(topk_ids, topk_weight, hidden_states.dtype)
    )
    a1, a1_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=TOKEN,
        topk=TOPK,
        block_size=BLOCK_M,
        num_rows=None,
    )
    a2 = kimi_fp4_stage1_16384(
        a1,
        w1,
        w1_scale.view(dtypes.fp8_e8m0),
        a1_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
    )
    a2_q, a2_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(
        a2.view(-1, INTER_DIM),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=TOKEN,
        topk=TOPK,
        block_size=BLOCK_M,
        num_rows=None,
    )
    return flydsl_moe_stage2(
        a2_q.view(TOKEN, TOPK, -1),
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out=moe_out.zero_(),
        topk=TOPK,
        tile_m=64,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=w2_scale.view(dtypes.fp8_e8m0),
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=BLOCK_M,
        persist=True,
        b_nt=0,
        xcd_swizzle=4,
    )


def _run_mxfp4_pipeline_16384(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    use_flydsl_gemm1: bool = False,
    use_flydsl_gemm2: bool = False,
    use_flydsl_scatter_reduce_q: bool = False,
) -> torch.Tensor:
    """Fixed-shape mxfp4 pipeline with per-kernel migration switches."""
    device = hidden_states.device
    w1_u8 = _as_u8_storage(w1)
    w2_u8 = _as_u8_storage(w2)
    sort = kimi_mxfp4_sort_16384(topk_ids, topk_weight)

    a_quant = torch.empty((TOKEN, MODEL_DIM // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((TOKEN, MODEL_DIM // 32), device=device, dtype=torch.uint8)
    aiter.mxfp4_moe_quant(
        a_input=hidden_states,
        a_quant=a_quant,
        a_scale=a_scale,
        bf16_zero_out=_empty_bf16(device),
        NE=EXPERTS,
        TOPK=TOPK,
        D_HIDDEN=MODEL_DIM,
        MB=sort.block_m,
    )

    padded_rows = ((sort.max_sorted + 31) // 32) * 32
    a_scale_sorted_shuffled = torch.empty(
        (padded_rows * (MODEL_DIM // 32) * 2,),
        device=device,
        dtype=torch.uint8,
    )
    aiter.mxfp4_moe_sort_scales(
        a_scale=a_scale,
        sorted_token_ids=sort.sorted_token_ids,
        cumsum_tensor=sort.cumsum_tensor,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        NE=EXPERTS,
        TOPK=TOPK,
        D_HIDDEN=MODEL_DIM,
        D_INTER=INTER_DIM,
        MB=sort.block_m,
        max_sorted=sort.max_sorted,
    )

    inter_sorted_quant = torch.empty(
        (sort.max_sorted, INTER_DIM // 2),
        device=device,
        dtype=torch.uint8,
    )
    inter_scale_cols = INTER_DIM // 32
    inter_scale_bytes = sort.max_sorted * (1024 // 64) * 4
    inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
    inter_scale_rows = ((inter_scale_rows + 31) // 32) * 32
    inter_sorted_shuffled_scale = torch.empty(
        (inter_scale_rows, inter_scale_cols),
        device=device,
        dtype=torch.uint8,
    )
    if use_flydsl_gemm1:
        kimi_mxfp4_gemm1_16384(
            a_quant=a_quant,
            a_scale_sorted_shuffled=a_scale_sorted_shuffled,
            w1=w1_u8,
            w1_scale=w1_scale,
            sorted_expert_ids=sort.sorted_expert_ids,
            m_indices=sort.m_indices,
            cumsum_tensor=sort.cumsum_tensor,
            inter_sorted_quant=inter_sorted_quant,
            inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        )
    else:
        aiter.mxfp4_moe_gemm1_a4w4(
            cumsum_tensor=sort.cumsum_tensor,
            a_quant=a_quant,
            a_scale_sorted_shuffled=a_scale_sorted_shuffled,
            w12_shuffled_quant=w1_u8,
            w12_shuffled_scale=w1_scale,
            sorted_expert_ids=sort.sorted_expert_ids,
            m_indices=sort.m_indices,
            inter_sorted_quant=inter_sorted_quant,
            inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
            hidden_states=hidden_states,
            kernelName=MXFP4_STAGE1_KERNEL,
        )

    flat_out_q = torch.empty(
        (sort.max_sorted, MODEL_DIM // 2),
        dtype=torch.uint8,
        device=device,
    )
    flat_out_scale = torch.empty(
        (sort.max_sorted, MODEL_DIM // 32),
        dtype=torch.uint8,
        device=device,
    )
    if use_flydsl_gemm2:
        kimi_mxfp4_gemm2_mxfp4out_16384(
            inter_sorted_quant=inter_sorted_quant,
            inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
            w2=w2_u8,
            w2_scale=w2_scale,
            sorted_expert_ids=sort.sorted_expert_ids,
            cumsum_tensor=sort.cumsum_tensor,
            flat_out_q=flat_out_q,
            flat_out_scale=flat_out_scale,
        )
    else:
        aiter.mxfp4_moe_gemm2_a4w4_mxfp4out(
            cumsum_tensor=sort.cumsum_tensor,
            inter_sorted_quant=inter_sorted_quant,
            inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
            w3_shuffled_quant=w2_u8,
            w3_shuffled_scale=w2_scale,
            sorted_expert_ids=sort.sorted_expert_ids,
            flat_out_q=flat_out_q,
            flat_out_scale=flat_out_scale,
            NE=EXPERTS,
            D_HIDDEN=MODEL_DIM,
            D_INTER=INTER_DIM,
            max_sorted=sort.max_sorted,
        )

    out = torch.empty((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device=device)
    if use_flydsl_scatter_reduce_q:
        kimi_mxfp4_scatter_reduce_q_16384(
            flat_out_q=flat_out_q,
            flat_out_scale=flat_out_scale,
            reverse_sorted=sort.reverse_sorted,
            sorted_weights=sort.sorted_weights,
            out=out,
        )
    else:
        aiter.mxfp4_moe_scatter_reduce_q(
            flat_out_q=flat_out_q,
            flat_out_scale=flat_out_scale,
            reverse_sorted=sort.reverse_sorted,
            sorted_weights=sort.sorted_weights,
            out=out,
            NE=EXPERTS,
            TOPK=TOPK,
            D_HIDDEN=MODEL_DIM,
            MB=sort.block_m,
        )
    return out


def run_kimi_fp4_mxfp4_moe_16384_aiter_ref(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Fixed-shape extraction of aiter's mxfp4 path, kept as the oracle."""
    return _run_mxfp4_pipeline_16384(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weight,
        w1_scale,
        w2_scale,
        use_flydsl_gemm1=False,
        use_flydsl_gemm2=False,
        use_flydsl_scatter_reduce_q=False,
    )


def run_kimi_fp4_mxfp4_moe_16384_opt(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Current migration candidate: aiter mxfp4 GEMMs plus FlyDSL scatter/reduce-q."""
    return _run_mxfp4_pipeline_16384(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weight,
        w1_scale,
        w2_scale,
        use_flydsl_gemm1=False,
        use_flydsl_gemm2=False,
        use_flydsl_scatter_reduce_q=True,
    )


def run_kimi_fp4_mxfp4_moe_16384_opt_gemm1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Experimental candidate with FlyDSL GEMM1 plus FlyDSL scatter/reduce-q."""
    return _run_mxfp4_pipeline_16384(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weight,
        w1_scale,
        w2_scale,
        use_flydsl_gemm1=True,
        use_flydsl_gemm2=False,
        use_flydsl_scatter_reduce_q=True,
    )


def run_kimi_fp4_mxfp4_moe_16384_opt_gemm2(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Experimental candidate with FlyDSL GEMM2 MXFP4-out plus FlyDSL scatter/reduce-q."""
    return _run_mxfp4_pipeline_16384(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weight,
        w1_scale,
        w2_scale,
        use_flydsl_gemm1=False,
        use_flydsl_gemm2=True,
        use_flydsl_scatter_reduce_q=True,
    )
