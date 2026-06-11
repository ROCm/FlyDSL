"""Fixed-shape Kimi mxfp4 MoE BM16 small-M path.

This file is the small-M counterpart of ``kimi_fp4_moe_16384_opt.py``.  The
target aiter route is:

    mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT
    mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT

The first migrated piece is the aiter ``prologue=0`` auxiliary kernel:
single-CTA sort plus full-grid zero-init of the bf16 atomic output buffer.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get
from kernels.mfma_preshuffle_pipeline import _buffer_load_vec, swizzle_xor16

from kimi_fp4_moe_16384 import (
    EXPERTS,
    INTER_DIM,
    MODEL_DIM,
    TOPK,
    _barrier,
    _ptr_view_safe,
    _run_compiled,
)
from kimi_fp4_moe_16384_opt import (
    _extract_global_ptr,
    _global_load_i32,
    _global_load_i32_vec,
    _global_load_f32,
    _ptr_buffer_resource,
    _global_store_f32,
    _global_store_i32,
    _global_store_vec4_i32,
    _lds_atomic_add_i32,
)


BM16 = 16
THREADS_SORT = 1024
INLINE_ZERO_CTAS = 128
SUPPORTED_SMALL_MS = (4, 8, 16, 32, 64, 128)
DPP_ROW_SHR_1 = 0x111
DPP_ROW_SHR_2 = 0x112
DPP_ROW_SHR_4 = 0x114
DPP_ROW_SHR_8 = 0x118
DPP_ROW_MASK = 0xF
DPP_BANK_MASK = 0xF


@dataclass
class SmallMxfp4SortBuffers:
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    cumsum_tensor: torch.Tensor
    reverse_sorted: torch.Tensor
    sorted_weights: torch.Tensor
    masked_m: torch.Tensor
    m_indices: torch.Tensor
    atomic_output: torch.Tensor
    max_sorted: int
    block_m: int = BM16


def mxfp4_small_max_sorted(m: int, block_m: int = BM16) -> int:
    active = min(EXPERTS, m * TOPK)
    cumsum_max = m * TOPK + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


def _unwrap_val(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _ptr_buffer_resource_dynamic(ptr, num_records_bytes):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64,
        num_records_bytes=num_records_bytes,
    )


def _record_current_stream(*items):
    stream = torch.cuda.current_stream()
    for item in items:
        if item is None:
            continue
        if isinstance(item, SmallMxfp4SortBuffers):
            _record_current_stream(
                item.sorted_token_ids,
                item.sorted_expert_ids,
                item.cumsum_tensor,
                item.reverse_sorted,
                item.sorted_weights,
                item.masked_m,
                item.m_indices,
                item.atomic_output,
            )
        elif isinstance(item, torch.Tensor):
            item.record_stream(stream)


def _dpp_allwave_prefix_sum_i32(val, lane, wave, scratch_mr, *, num_waves: int):
    """Inclusive i32 prefix sum across ``num_waves`` waves.

    This matches the shape of aiter's DPP scan: row_shr within each row, then
    bpermute for the 16/32-lane row carry, then a tiny LDS cross-wave scan.
    """

    zero = arith.constant(0, type=T.i32)
    val_raw = _unwrap_val(val)
    zero_raw = _unwrap_val(zero)
    for dpp_op, threshold in (
        (DPP_ROW_SHR_1, 1),
        (DPP_ROW_SHR_2, 2),
        (DPP_ROW_SHR_4, 4),
        (DPP_ROW_SHR_8, 8),
    ):
        remote = rocdl.update_dpp(
            T.i32,
            zero_raw,
            val_raw,
            dpp_op,
            DPP_ROW_MASK,
            DPP_BANK_MASK,
            True,
        )
        remote = ArithValue(remote)
        use_remote = arith.cmpi(CmpIPredicate.sge, lane, arith.constant(threshold, type=T.i32))
        val = arith.select(use_remote, val + remote, val)
        val_raw = _unwrap_val(val)

    src_lane_16 = (lane & arith.constant(0x30, type=T.i32)) - arith.constant(1, type=T.i32)
    remote16 = rocdl.ds_bpermute(T.i32, src_lane_16 * arith.constant(4, type=T.i32), val)
    use16 = arith.cmpi(CmpIPredicate.sge, lane, arith.constant(16, type=T.i32))
    val = arith.select(use16, val + ArithValue(remote16), val)

    src_lane_32 = (lane & arith.constant(0x30, type=T.i32)) - arith.constant(17, type=T.i32)
    remote32 = rocdl.ds_bpermute(T.i32, src_lane_32 * arith.constant(4, type=T.i32), val)
    use32 = arith.cmpi(CmpIPredicate.sge, lane, arith.constant(32, type=T.i32))
    val = arith.select(use32, val + ArithValue(remote32), val)

    lane_last = arith.cmpi(CmpIPredicate.eq, lane, arith.constant(63, type=T.i32))
    if_last = scf.IfOp(lane_last)
    with ir.InsertionPoint(if_last.then_block):
        memref.store(arith._to_raw(val), scratch_mr, [ArithValue(wave).index_cast(T.index)])
        scf.YieldOp([])
    gpu.barrier()

    cross = zero
    for w in range_constexpr(num_waves - 1):
        wt = ArithValue(memref.load(scratch_mr, [arith.index(w)]))
        add_w = arith.cmpi(CmpIPredicate.sgt, wave, arith.constant(w, type=T.i32))
        cross = arith.select(add_w, cross + wt, cross)
    return val + cross


@functools.lru_cache(maxsize=1)
def compile_kimi_mxfp4_sort_zero_init_bm16():
    """Compile the BM16 inline-quant prologue kernel.

    This mirrors aiter's
    ``launch_sort_only_with_zero_init<385,9,16,7168,128,1024>``.
    """

    threads = THREADS_SORT
    block_m = BM16
    max_count_slots = max(EXPERTS, threads)
    topk_nbytes = 128 * TOPK * 4
    topk_weight_nbytes = 128 * TOPK * 4
    max_sorted = mxfp4_small_max_sorted(128, block_m)
    sorted_nbytes = max_sorted * 4
    expert_nbytes = (max_sorted // block_m) * 4
    cumsum_nbytes = 4
    reverse_nbytes = 128 * TOPK * 4
    weights_nbytes = max_sorted * 4
    masked_nbytes = EXPERTS * 4
    output_nbytes = 128 * MODEL_DIM * 2

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_mxfp4_bm16_sort")
    lds_count_offset = allocator._align(allocator.ptr, 16)
    lds_cumsum_offset = lds_count_offset + max_count_slots * 4
    lds_counter_offset = lds_cumsum_offset + (EXPERTS + 1) * 4
    lds_scratch_offset = lds_counter_offset + EXPERTS * 4
    allocator.ptr = lds_scratch_offset + 16 * 4

    @flyc.kernel(
        name="flydsl_kimi_mxfp4_sort_zero_init_NE385_TOPK9_H7168_BM16_v9_cnt_select",
        known_block_size=[threads, 1, 1],
    )
    def sort_zero_init(
        arg_topk_ids: fx.Pointer,
        arg_topk_weight: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_cumsum_tensor: fx.Pointer,
        arg_reverse_sorted: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_masked_m: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_atomic_output: fx.Pointer,
        i32_m: fx.Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        vec4_i32 = T.vec(4, i32)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(i32, tx)
        bx_i32 = arith.index_cast(i32, bx)

        topk_ptr = _extract_global_ptr(arg_topk_ids)
        weight_ptr = _extract_global_ptr(arg_topk_weight)
        sorted_ptr = _extract_global_ptr(arg_sorted_token_ids)
        expert_ptr = _extract_global_ptr(arg_sorted_expert_ids)
        cumsum_ptr = _extract_global_ptr(arg_cumsum_tensor)
        reverse_ptr = _extract_global_ptr(arg_reverse_sorted)
        sorted_weight_ptr = _extract_global_ptr(arg_sorted_weights)
        masked_ptr = _extract_global_ptr(arg_masked_m)
        mindices_ptr = _extract_global_ptr(arg_m_indices)
        output_ptr = _extract_global_ptr(arg_atomic_output)

        base_ptr = allocator.get_base()
        count = SmemPtr(base_ptr, lds_count_offset, T.i32, shape=(max_count_slots,)).get()
        cumsum = SmemPtr(base_ptr, lds_cumsum_offset, T.i32, shape=(EXPERTS + 1,)).get()
        counter = SmemPtr(base_ptr, lds_counter_offset, T.i32, shape=(EXPERTS,)).get()
        scratch = SmemPtr(base_ptr, lds_scratch_offset, T.i32, shape=(16,)).get()

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c_topk = arith.constant(TOPK, type=i32)
        c_threads_idx = arith.constant(threads, index=True)
        c_experts_idx = arith.constant(EXPERTS, index=True)
        c_block_m = arith.constant(block_m, type=i32)
        c_bm_minus_1 = arith.constant(block_m - 1, type=i32)
        c_bm_mask = arith.constant(~(block_m - 1), type=i32)
        c_token_mask = arith.constant(0x00FFFFFF, type=i32)
        c_topk_shift = arith.constant(24, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        is_sort_cta = arith.cmpi(CmpIPredicate.eq, bx_i32, c0_i32)

        # CTA0 does the single-CTA sort. The remaining CTAs zero the atomic
        # output buffer so the sort CTA does not contend with the streaming
        # int4 stores.
        sort_if = scf.IfOp(is_sort_cta, has_else=True)
        with ir.InsertionPoint(sort_if.then_block):
            if tx < c_experts_idx:
                memref.store(c0_i32, count, [tx])
            gpu.barrier()

            m_idx = ArithValue(i32_m).index_cast(T.index)
            total_pairs_idx = m_idx * arith.constant(TOPK, index=True)

            # Supported BM16 M buckets are multiples of 4, so TOPK routes are
            # 4-aligned. Match aiter's int4 count path instead of issuing one
            # scalar global load per route.
            count_loop = scf.ForOp(
                tx * arith.constant(4, index=True),
                total_pairs_idx,
                c_threads_idx * arith.constant(4, index=True),
            )
            with ir.InsertionPoint(count_loop.body):
                idx = count_loop.induction_variable
                ids = _global_load_i32_vec(topk_ptr, idx, 4)
                for lane in range_constexpr(4):
                    eid = ArithValue(
                        vector.extract(ids, static_position=[lane], dynamic_position=[])
                    )
                    _lds_atomic_add_i32(count, eid, c1_i32)
                scf.YieldOp([])
            gpu.barrier()

            lane_i32 = tx_i32 % arith.constant(64, type=i32)
            wave_i32 = tx_i32 // arith.constant(64, type=i32)
            scan_valid = tx < c_experts_idx
            cnt = ArithValue(
                arith.select(scan_valid, memref.load(count, [tx]), arith._to_raw(c0_i32))
            )
            padded = (cnt + c_bm_minus_1) & c_bm_mask
            padded = ArithValue(arith.select(scan_valid, arith._to_raw(padded), arith._to_raw(c0_i32)))
            inclusive = _dpp_allwave_prefix_sum_i32(
                padded,
                lane_i32,
                wave_i32,
                scratch,
                num_waves=16,
            )
            start = inclusive - padded

            if_scan_store = scf.IfOp(scan_valid)
            with ir.InsertionPoint(if_scan_store.then_block):
                memref.store(arith._to_raw(start), cumsum, [tx])
                memref.store(c0_i32, counter, [tx])
                _global_store_i32(masked_ptr, tx_i32, padded)
                b0 = start // c_block_m
                b1 = inclusive // c_block_m
                fill_experts = scf.ForOp(
                    ArithValue(b0).index_cast(T.index),
                    ArithValue(b1).index_cast(T.index),
                    arith.index(1),
                )
                with ir.InsertionPoint(fill_experts.body):
                    b = fill_experts.induction_variable
                    _global_store_i32(expert_ptr, arith.index_cast(i32, b), tx_i32)
                    scf.YieldOp([])
                scf.YieldOp([])

            if_first = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tx_i32, c0_i32))
            with ir.InsertionPoint(if_first.then_block):
                memref.store(c0_i32, cumsum, [arith.index(0)])
                scf.YieldOp([])

            if_last_expert = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, tx_i32, arith.constant(EXPERTS - 1, type=i32))
            )
            with ir.InsertionPoint(if_last_expert.then_block):
                memref.store(arith._to_raw(inclusive), cumsum, [arith.index(EXPERTS)])
                _global_store_i32(cumsum_ptr, c0_i32, inclusive)
                scf.YieldOp([])
            gpu.barrier()

            token_id_init = tx_i32 // c_topk
            topk_id_init = tx_i32 % c_topk
            place_loop = scf.ForOp(
                tx,
                total_pairs_idx,
                c_threads_idx,
                [arith._to_raw(token_id_init), arith._to_raw(topk_id_init)],
            )
            with ir.InsertionPoint(place_loop.body):
                idx = place_loop.induction_variable
                token_id = ArithValue(place_loop.inner_iter_args[0])
                topk_id = ArithValue(place_loop.inner_iter_args[1])
                idx_i32 = arith.index_cast(i32, idx)
                eid = ArithValue(_global_load_i32(topk_ptr, idx_i32))
                eid_idx = eid.index_cast(T.index)
                pos = _lds_atomic_add_i32(counter, eid, c1_i32)
                start = ArithValue(memref.load(cumsum, [eid_idx]))
                sp = start + pos
                packed_id = (token_id & c_token_mask) | (topk_id << c_topk_shift)
                route_w = _global_load_f32(weight_ptr, idx_i32)
                _global_store_i32(sorted_ptr, sp, packed_id)
                _global_store_i32(mindices_ptr, sp, token_id & c_token_mask)
                _global_store_f32(sorted_weight_ptr, sp, route_w)
                _global_store_i32(reverse_ptr, idx_i32, sp)
                next_token_id = token_id + arith.constant(threads // TOPK, type=i32)
                next_topk_id = topk_id + arith.constant(threads % TOPK, type=i32)
                carry = arith.cmpi(CmpIPredicate.sge, next_topk_id, c_topk)
                next_topk_id = arith.select(carry, next_topk_id - c_topk, next_topk_id)
                next_token_id = arith.select(carry, next_token_id + c1_i32, next_token_id)
                scf.YieldOp([arith._to_raw(next_token_id), arith._to_raw(next_topk_id)])
            gpu.barrier()

            pad_valid = arith.cmpi(CmpIPredicate.ult, tx, c_experts_idx)
            if_pad = scf.IfOp(pad_valid)
            with ir.InsertionPoint(if_pad.then_block):
                e = tx
                e_i32 = tx_i32
                cnt = ArithValue(memref.load(count, [e]))
                start = ArithValue(memref.load(cumsum, [e]))
                end = ArithValue(memref.load(cumsum, [e + arith.index(1)]))
                real_end = start + cnt
                pad = i32_m & c_token_mask
                pad_loop = scf.ForOp(
                    ArithValue(real_end).index_cast(T.index),
                    ArithValue(end).index_cast(T.index),
                    arith.index(1),
                )
                with ir.InsertionPoint(pad_loop.body):
                    j = pad_loop.induction_variable
                    j_i32 = arith.index_cast(i32, j)
                    _global_store_i32(sorted_ptr, j_i32, pad)
                    _global_store_i32(mindices_ptr, j_i32, pad)
                    _global_store_f32(sorted_weight_ptr, j_i32, c0_f32)
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

        # Full-grid zero-init: int4 stores over M * H * sizeof(bf16).
        with ir.InsertionPoint(sort_if.else_block):
            zero_vec = vector.from_elements(vec4_i32, [c0_i32, c0_i32, c0_i32, c0_i32])
            m_idx = ArithValue(i32_m).index_cast(T.index)
            total_vecs = m_idx * arith.constant((MODEL_DIM * 2) // 16, index=True)
            zero_bx = bx - arith.index(1)
            gtid = zero_bx * arith.constant(threads, index=True) + tx
            step = arith.constant(INLINE_ZERO_CTAS * threads, index=True)
            zero_loop = scf.ForOp(gtid, total_vecs, step)
            with ir.InsertionPoint(zero_loop.body):
                vi = zero_loop.induction_variable
                elem_i32 = vi * arith.constant(4, index=True)
                _global_store_vec4_i32(output_ptr, elem_i32, zero_vec, nontemporal=True)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_sort_zero_init(
        topk_ids: fx.Pointer,
        topk_weight: fx.Pointer,
        sorted_token_ids: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        cumsum_tensor: fx.Pointer,
        reverse_sorted: fx.Pointer,
        sorted_weights: fx.Pointer,
        masked_m: fx.Pointer,
        m_indices: fx.Pointer,
        atomic_output: fx.Pointer,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        sort_zero_init(
            topk_ids,
            topk_weight,
            sorted_token_ids,
            sorted_expert_ids,
            cumsum_tensor,
            reverse_sorted,
            sorted_weights,
            masked_m,
            m_indices,
            atomic_output,
            m,
        ).launch(
            grid=(INLINE_ZERO_CTAS + 1, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch_sort_zero_init


def kimi_mxfp4_sort_zero_init_bm16(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    *,
    m: int | None = None,
    max_sorted: int | None = None,
    atomic_output: torch.Tensor | None = None,
) -> SmallMxfp4SortBuffers:
    """Run the fixed Kimi BM16 inline-quant prologue in FlyDSL."""

    if m is None:
        m = int(topk_ids.shape[0])
    if m not in SUPPORTED_SMALL_MS:
        raise ValueError(f"BM16 small path supports M in {SUPPORTED_SMALL_MS}, got {m}")
    if topk_ids.shape != (m, TOPK):
        raise ValueError(f"expected topk_ids shape {(m, TOPK)}, got {tuple(topk_ids.shape)}")
    if topk_weight.shape != (m, TOPK):
        raise ValueError(f"expected topk_weight shape {(m, TOPK)}, got {tuple(topk_weight.shape)}")

    device = topk_ids.device
    if max_sorted is None:
        max_sorted = mxfp4_small_max_sorted(m, BM16)
    sorted_token_ids = torch.empty((max_sorted,), device=device, dtype=torch.int32)
    sorted_expert_ids = torch.empty((max_sorted // BM16,), device=device, dtype=torch.int32)
    cumsum_tensor = torch.empty((1,), device=device, dtype=torch.int32)
    reverse_sorted = torch.empty((m * TOPK,), device=device, dtype=torch.int32)
    sorted_weights = torch.empty((max_sorted,), device=device, dtype=torch.float32)
    masked_m = torch.empty((EXPERTS,), device=device, dtype=torch.int32)
    m_indices = torch.empty((max_sorted,), device=device, dtype=torch.int32)
    if atomic_output is None:
        atomic_output = torch.empty((m, MODEL_DIM), device=device, dtype=torch.bfloat16)
    elif atomic_output.shape != (m, MODEL_DIM):
        raise ValueError(
            f"expected atomic_output shape {(m, MODEL_DIM)}, got {tuple(atomic_output.shape)}"
        )

    args = (
        _ptr_view_safe(topk_ids.contiguous().view(-1)),
        _ptr_view_safe(topk_weight.contiguous().view(-1)),
        _ptr_view_safe(sorted_token_ids.view(-1)),
        _ptr_view_safe(sorted_expert_ids.view(-1)),
        _ptr_view_safe(cumsum_tensor.view(-1)),
        _ptr_view_safe(reverse_sorted.view(-1)),
        _ptr_view_safe(sorted_weights.view(-1)),
        _ptr_view_safe(masked_m.view(-1)),
        _ptr_view_safe(m_indices.view(-1)),
        _ptr_view_safe(atomic_output.view(-1)),
        int(m),
        torch.cuda.current_stream(),
    )
    _run_compiled(compile_kimi_mxfp4_sort_zero_init_bm16(), args)

    buffers = SmallMxfp4SortBuffers(
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        masked_m=masked_m,
        m_indices=m_indices,
        atomic_output=atomic_output,
        max_sorted=max_sorted,
    )
    _record_current_stream(topk_ids, topk_weight, buffers)
    return buffers


@functools.lru_cache(maxsize=1)
def compile_kimi_mxfp4_gemm1_inline_bm16():
    """Compile FlyDSL BM16 inline-quant GEMM1.

    This is the fixed-shape port target for
    ``mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT``.
    """

    tile_m = BM16
    tile_n = 256
    tile_k = 256
    total_threads = 256
    num_n_blocks = (2 * INTER_DIM) // tile_n
    max_sorted = mxfp4_small_max_sorted(128, BM16)

    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    k_unroll = tile_k // 128
    num_k_tiles = MODEL_DIM // tile_k
    k_stages = 2
    a_stages = 3
    a_elem_vec_pack = 2
    eff_lds_stride = tile_k // a_elem_vec_pack
    single_x_bytes = tile_m * eff_lds_stride
    lds_x_bytes = a_stages * single_x_bytes
    lds_scale_bytes = num_k_tiles * 256
    lds_acc_bytes = tile_m * tile_n * 4

    hidden_max_nbytes = 128 * MODEL_DIM * 2
    w_nbytes = EXPERTS * (2 * INTER_DIM) * (MODEL_DIM // 2)
    w_scale_nbytes = EXPERTS * (2 * INTER_DIM) * (MODEL_DIM // 32)
    expert_nbytes = (max_sorted // BM16) * 4
    m_indices_nbytes = max_sorted * 4
    numids_nbytes = 4
    out_q_nbytes = max_sorted * (INTER_DIM // 2)
    out_scale_nbytes = (max_sorted // BM16) * 128 * 4

    b_kpack_bytes = 16
    b_kpack_elems = b_kpack_bytes
    b_c_k = MODEL_DIM // 2
    b_c_k0 = b_c_k // 64
    b_stride_nlane = b_kpack_elems
    b_stride_klane = 16 * b_stride_nlane
    b_stride_k0 = 4 * b_stride_klane
    b_stride_n0 = b_c_k0 * b_stride_k0
    expert_b_stride = ((2 * INTER_DIM) // 16) * b_stride_n0

    k_bs_stride_k0_dw = 64
    k_bs_stride_n0_dw = ((MODEL_DIM // 32) // 4 // 2) * 64
    k_bs_per_expert_dw = (((2 * INTER_DIM) // 16) // 2) * k_bs_stride_n0_dw
    k_out_as_per_chunk_dw = ((INTER_DIM // 32) // 4 // 2) * 64

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_mxfp4_bm16_gemm1")
    lds_offset = allocator._align(allocator.ptr, 16)
    lds_scale_offset = lds_offset + lds_x_bytes
    allocator.ptr = lds_offset + max(lds_x_bytes + lds_scale_bytes, lds_acc_bytes)

    module_name = "flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM16_INLINEQUANT_v13_knownblock"

    @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
    def gemm1_inline(
        arg_inter_sorted_quant: fx.Pointer,
        arg_inter_sorted_scale: fx.Pointer,
        arg_hidden: fx.Pointer,
        arg_w: fx.Pointer,
        arg_w_scale: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        i32_m: fx.Int32,
    ):
        i8 = T.i8
        i32 = T.i32
        i64 = T.i64
        f32 = T.f32
        vec1_i32 = T.vec(1, i32)
        vec4_i32 = T.vec(4, i32)
        vec2_bf16 = T.vec(2, T.bf16)
        vec2_i64 = T.vec(2, i64)
        vec4_i64 = T.vec(4, i64)
        vec8_i32 = T.vec(8, i32)
        vec4_f32 = T.vec(4, f32)
        vec16_x = T.vec(16, T.f8)

        hidden_bytes_i64 = (
            ArithValue(arith.ExtSIOp(i64, _unwrap_val(i32_m)).result)
            * arith.constant(MODEL_DIM * 2, type=i64)
        )
        hidden_rsrc = _ptr_buffer_resource_dynamic(arg_hidden, hidden_bytes_i64)
        out_q_rsrc = _ptr_buffer_resource(arg_inter_sorted_quant, out_q_nbytes)
        out_scale_rsrc = _ptr_buffer_resource(arg_inter_sorted_scale, out_scale_nbytes)
        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)
        sw_rsrc = _ptr_buffer_resource(arg_w_scale, w_scale_nbytes)
        expert_ptr = _extract_global_ptr(arg_expert_ids)
        m_indices_ptr = _extract_global_ptr(arg_m_indices)
        numids_ptr = _extract_global_ptr(arg_num_valid_ids)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")
        bx = gpu.block_id("y")
        tx_i32 = arith.index_cast(i32, tx)
        bx_m = bx * arith.constant(tile_m, index=True)
        bx_m_i32 = arith.index_cast(i32, bx_m)

        num_valid_i32 = _global_load_i32(numids_ptr, 0)
        num_valid_i32 = rocdl.readfirstlane(i32, num_valid_i32)
        blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
        expert_i32 = _global_load_i32(expert_ptr, bx)
        expert_i32 = rocdl.readfirstlane(i32, expert_i32)
        expert_idx = arith.index_cast(T.index, expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(EXPERTS, type=i32))
        do_gemm = arith.andi(blk_valid, exp_valid)

        base_ptr = allocator.get_base()
        lds_x_i8 = SmemPtr(base_ptr, lds_offset, T.f8, shape=(lds_x_bytes,)).get()
        lds_x_i32 = SmemPtr(base_ptr, lds_offset, T.i32, shape=(lds_x_bytes // 4,)).get()
        lds_scale_i32 = SmemPtr(
            base_ptr,
            lds_scale_offset,
            T.i32,
            shape=(lds_scale_bytes // 4,),
        ).get()
        lds_acc = SmemPtr(base_ptr, lds_offset, T.f32, shape=(tile_m * tile_n,)).get()

        shape_lds = fx.make_shape(tile_m, eff_lds_stride)
        layout_lds = fx.make_layout(shape_lds, fx.make_stride(eff_lds_stride, 1))
        k_blocks16 = arith.constant(eff_lds_stride // 16, index=True)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_wl = idx2crd(tx, layout_tx_wave_lane)
        wave_id = layout_get(coord_wl, 0)
        lane_id = layout_get(coord_wl, 1)
        wave_i32 = arith.index_cast(i32, wave_id)
        lane_i32 = arith.index_cast(i32, lane_id)
        coord_l16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * arith.constant(16, index=True)
        n_tile_base = wave_id * arith.constant(n_per_wave, index=True)
        by_n = by * arith.constant(tile_n, index=True)
        expert_b_base = expert_idx * arith.constant(expert_b_stride, index=True)

        r_in_chunk = wave_id * arith.constant(4, index=True) + lane_div_16
        r_in_chunk_i32 = arith.index_cast(i32, r_in_chunk)
        row_token = _global_load_i32(m_indices_ptr, bx_m_i32 + r_in_chunk_i32)
        row_token_idx = ArithValue(row_token).index_cast(T.index)

        n_blk_list = []
        n_intra_list = []
        for ni in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + arith.constant(ni * 16, index=True) + lane_mod_16
            n_blk_list.append(global_n // arith.constant(16, index=True))
            n_intra_list.append(global_n % arith.constant(16, index=True))

        def load_b_packs_k64(k_tile: int, ku: int, ni: int):
            k0 = arith.constant(k_tile * k_unroll + ku, index=True)
            idx_pack = (
                expert_b_base
                + n_blk_list[ni] * arith.constant(b_stride_n0, index=True)
                + k0 * arith.constant(b_stride_k0, index=True)
                + lane_div_16 * arith.constant(b_stride_klane, index=True)
                + n_intra_list[ni] * arith.constant(b_stride_nlane, index=True)
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
                cache_modifier=2,
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

        lane_word = lane_div_16 * arith.constant(16, index=True) + lane_mod_16

        def load_a_scale_tile(k_tile: int):
            scale_idx = arith.constant(k_tile * 64, index=True) + lane_word
            s = vector.extract(
                vector.load_op(T.vec(1, i32), lds_scale_i32, [scale_idx]),
                static_position=[0],
                dynamic_position=[],
            )
            return vector.from_elements(T.vec(1, i32), [s])

        def load_b_scale_tile(k_tile: int):
            b_scale_tile = []
            mni_base = by * arith.constant(tile_n // 16 // 2, index=True)
            mni_base = mni_base + wave_id * arith.constant(n_per_wave // 16 // 2, index=True)
            for ni in range_constexpr(num_acc_n // 2):
                scale_idx = (
                    expert_idx * arith.constant(k_bs_per_expert_dw, index=True)
                    + (mni_base + arith.constant(ni, index=True))
                    * arith.constant(k_bs_stride_n0_dw, index=True)
                    + arith.constant(k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = buffer_ops.buffer_load(sw_rsrc, scale_idx, vec_width=1, dtype=i32, cache_modifier=0)
                b_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return b_scale_tile

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c8_i32 = arith.constant(8, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        cffff_i32 = arith.constant(0xFFFF, type=i32)
        c7fff_i32 = arith.constant(0x7FFF, type=i32)
        c7fff7fff_i32 = arith.constant(0x7FFF7FFF, type=i32)
        chi_mask_i32 = arith.constant(-65536, type=i32)
        c0x200000_i32 = arith.constant(0x200000, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c1_f32 = arith.constant(1.0, type=f32)
        c025_f32 = arith.constant(0.25, type=f32)
        neg_log2e = arith.constant(-1.4426950408889634, type=f32)

        def _max_u32(a, b):
            return arith.select(arith.cmpi(CmpIPredicate.ugt, b, a), b, a)

        def _raw(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        def _pkmax_u16(a, b):
            return ArithValue(
                llvm.inline_asm(
                    i32,
                    [_raw(a), _raw(b)],
                    "v_pk_max_u16 $0, $1, $2",
                    "=v,v,v",
                    has_side_effects=False,
                )
            )

        def _max_u32_dpp_xor1(src):
            peer = ArithValue(
                llvm.call_intrinsic(
                    i32,
                    "llvm.amdgcn.update.dpp.i32",
                    [
                        _raw(src),
                        _raw(src),
                        arith.constant(0xB1, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(True, type=ir.IntegerType.get_signless(1)),
                    ],
                    [],
                    [],
                )
            )
            return _max_u32(src, peer)

        def _max_u32_dpp_xor2(src):
            peer = ArithValue(
                llvm.call_intrinsic(
                    i32,
                    "llvm.amdgcn.update.dpp.i32",
                    [
                        _raw(src),
                        _raw(src),
                        arith.constant(0x4E, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(True, type=ir.IntegerType.get_signless(1)),
                    ],
                    [],
                    [],
                )
            )
            return _max_u32(src, peer)

        def inline_quant_load_half(k_tile: int, b128: int):
            lane_div4 = (lane_id // arith.constant(4, index=True)) % arith.constant(4, index=True)
            lane_mod4 = lane_id % arith.constant(4, index=True)
            v_byte_off = (
                row_token_idx * arith.constant(MODEL_DIM * 2, index=True)
                + arith.constant(k_tile * tile_k * 2 + b128 * 256, index=True)
                + lane_div4 * arith.constant(64, index=True)
                + lane_mod4 * arith.constant(16, index=True)
            )
            h16 = _buffer_load_vec(
                buffer_ops,
                vector,
                hidden_rsrc,
                v_byte_off,
                elem_type=T.i8,
                vec_elems=16,
                elem_bytes=1,
                offset_in_bytes=True,
                cache_modifier=0,
            )
            return vector.bitcast(vec4_i32, h16)

        def inline_quant_finish_half(a_slot: int, k_tile: int, b128: int, h4):
            lane_div4 = (lane_id // arith.constant(4, index=True)) % arith.constant(4, index=True)
            lane_mod4 = lane_id % arith.constant(4, index=True)
            hm = []
            bf16_pairs = []
            for j in range_constexpr(4):
                w = ArithValue(vector.extract(h4, static_position=[j], dynamic_position=[]))
                hm.append(w & c7fff7fff_i32)
                bf16_pairs.append(
                    vector.bitcast(
                        vec2_bf16,
                        vector.from_elements(vec1_i32, [w]),
                    )
                )

            m01 = _pkmax_u16(hm[0], hm[1])
            m23 = _pkmax_u16(hm[2], hm[3])
            m0123 = _pkmax_u16(m01, m23)
            lo_abs = m0123 & cffff_i32
            hi_abs = m0123 >> c16_i32
            local_amax = _max_u32(lo_abs, hi_abs)
            local_amax = _max_u32_dpp_xor1(local_amax)
            local_amax = _max_u32_dpp_xor2(local_amax)

            f32bits = local_amax << c16_i32
            bexp = ((f32bits + c0x200000_i32) >> c23_i32) & arith.constant(255, type=i32)
            e8m0 = bexp - c2_i32
            e8m0 = arith.select(arith.cmpi(CmpIPredicate.slt, e8m0, c0_i32), c0_i32, e8m0)
            e8m0 = arith.select(arith.cmpi(CmpIPredicate.sgt, e8m0, c254_i32), c254_i32, e8m0)
            quant_scale = (e8m0 << c23_i32).bitcast(f32)

            packed = c0_i32
            for pair in range_constexpr(4):
                packed = ArithValue(
                    llvm.call_intrinsic(
                        i32,
                        "llvm.amdgcn.cvt.scalef32.pk.fp4.bf16",
                        [
                            packed,
                            bf16_pairs[pair],
                            quant_scale,
                            arith.constant(pair, type=i32),
                        ],
                        [],
                        [],
                    )
                )

            kb_in_kt = arith.constant(b128 * 4, index=True) + lane_div4
            col_sw = swizzle_xor16(r_in_chunk, kb_in_kt * arith.constant(16, index=True), k_blocks16)
            byte_idx = crd2idx(
                [r_in_chunk, col_sw + lane_mod4 * arith.constant(4, index=True)],
                layout_lds,
            )
            word_idx = byte_idx // arith.constant(4, index=True)
            word_idx = word_idx + arith.constant(a_slot * (single_x_bytes // 4), index=True)
            vector.store(
                vector.from_elements(T.vec(1, i32), [packed]),
                lds_x_i32,
                [word_idx],
                alignment=4,
            )
            return e8m0

        def inline_quant_write_scale(k_tile: int, e8m0_0, e8m0_1):
            lane_div4 = (lane_id // arith.constant(4, index=True)) % arith.constant(4, index=True)
            lane_tgt = lane_div4 * arith.constant(16, index=True) + r_in_chunk
            scale_idx = arith.constant(k_tile * 64, index=True) + lane_tgt
            scale_accum = e8m0_0 | (e8m0_1 << c16_i32)
            vector.store(
                vector.from_elements(T.vec(1, i32), [scale_accum]),
                lds_scale_i32,
                [scale_idx],
                alignment=4,
            )

        def inline_quant_tile(k_tile: int, a_slot: int):
            h4_0 = inline_quant_load_half(k_tile, 0)
            e8m0_0 = inline_quant_finish_half(a_slot, k_tile, 0, h4_0)
            h4_1 = inline_quant_load_half(k_tile, 1)
            e8m0_1 = inline_quant_finish_half(a_slot, k_tile, 1, h4_1)
            inline_quant_write_scale(k_tile, e8m0_0, e8m0_1)

        def lds_load_packs_k64(curr_row_a_lds, col_base, a_slot: int):
            col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
            idx_a16 = idx_a16 + arith.constant(a_slot * single_x_bytes, index=True)
            loaded_a16 = vector.load_op(vec16_x, lds_x_i8, [idx_a16])
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

        def load_a128_tile(a_slot: int):
            a128_vals = []
            for k_idx in range_constexpr(k_unroll):
                col_base = col_offset_base + arith.constant(k_idx * 64, index=True)
                a0, a1 = lds_load_packs_k64(row_a_lds, col_base, a_slot)
                a128_vals.append(pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64))
            return a128_vals

        def compute_tile(acc_in, b_tile, a_scale, b_scale, a_slot: int):
            acc_list = list(acc_in)
            a_scale_val = vector.extract(a_scale, static_position=[0], dynamic_position=[])
            for k_idx in range_constexpr(k_unroll):
                col_base = col_offset_base + arith.constant(k_idx * 64, index=True)
                a0, a1 = lds_load_packs_k64(row_a_lds, col_base, a_slot)
                a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                b_packs0, b_packs1 = b_tile[k_idx]
                for ni in range_constexpr(num_acc_n // 2):
                    b_scale_i32 = b_scale[ni]
                    b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                    for inxdl in range_constexpr(2):
                        ni_idx = ni * 2 + inxdl
                        b0 = b_packs0[ni_idx]
                        b1 = b_packs1[ni_idx]
                        b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                        acc_list[ni_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            vec4_f32,
                            [
                                a128,
                                b128,
                                acc_list[ni_idx],
                                cbsz,
                                blgp,
                                k_idx * 2,
                                a_scale_val,
                                k_idx * 2 + inxdl,
                                b_scale_val,
                            ],
                        )
            return acc_list

        def compute_tile_prefetch_b(acc_in, b_tile, a128_vals, a_scale, b_scale, next_k_tile: int):
            acc_list = list(acc_in)
            a_scale_val = vector.extract(a_scale, static_position=[0], dynamic_position=[])
            next_b_tile = []
            for k_idx in range_constexpr(k_unroll):
                next_b_tile.append(([], []))

            for ni in range_constexpr(num_acc_n // 2):
                b_scale_i32 = b_scale[ni]
                b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                for inxdl in range_constexpr(2):
                    ni_idx = ni * 2 + inxdl
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    for k_idx in range_constexpr(k_unroll):
                        b_packs0, b_packs1 = b_tile[k_idx]
                        b0 = b_packs0[ni_idx]
                        b1 = b_packs1[ni_idx]
                        b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                        acc_list[ni_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            vec4_f32,
                            [
                                a128_vals[k_idx],
                                b128,
                                acc_list[ni_idx],
                                cbsz,
                                blgp,
                                k_idx * 2,
                                a_scale_val,
                                k_idx * 2 + inxdl,
                                b_scale_val,
                            ],
                        )
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                    for k_idx in range_constexpr(k_unroll):
                        next_b0, next_b1 = load_b_packs_k64(next_k_tile, k_idx, ni_idx)
                        next_b_tile[k_idx][0].append(next_b0)
                        next_b_tile[k_idx][1].append(next_b1)
                    rocdl.sched_barrier(0)

            return acc_list, next_b_tile

        def _fmax_num(a, b):
            return ArithValue(arith.MaxNumFOp(arith._to_raw(a), arith._to_raw(b)).result)

        def _fmax_dpp_xor1(src):
            src_i32 = src.bitcast(i32)
            peer_i32 = ArithValue(
                llvm.call_intrinsic(
                    i32,
                    "llvm.amdgcn.update.dpp.i32",
                    [
                        _raw(src_i32),
                        _raw(src_i32),
                        arith.constant(0xB1, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(True, type=ir.IntegerType.get_signless(1)),
                    ],
                    [],
                    [],
                )
            )
            return _fmax_num(src, peer_i32.bitcast(f32))

        def _fmax_dpp_xor2(src):
            src_i32 = src.bitcast(i32)
            peer_i32 = ArithValue(
                llvm.call_intrinsic(
                    i32,
                    "llvm.amdgcn.update.dpp.i32",
                    [
                        _raw(src_i32),
                        _raw(src_i32),
                        arith.constant(0x4E, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(0xF, type=i32),
                        arith.constant(True, type=ir.IntegerType.get_signless(1)),
                    ],
                    [],
                    [],
                )
            )
            return _fmax_num(src, peer_i32.bitcast(f32))

        def _silu_mul(g, u):
            t = g * neg_log2e
            emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
            den = c1_f32 + emu
            sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
            return g * sig * u

        def _then_body():
            acc = [arith.constant_vector(0.0, vec4_f32)] * num_acc_n
            b_stages = []
            b_scale_stages = []
            for k_tile in range_constexpr(k_stages):
                inline_quant_tile(k_tile, k_tile)
                b_stages.append(load_b_tile(k_tile))
                b_scale_stages.append(load_b_scale_tile(k_tile))

            for offset in range_constexpr(0, num_k_tiles - k_stages):
                read_slot = offset % a_stages
                slot_b = offset % k_stages
                next_k_tile = offset + k_stages
                write_slot = next_k_tile % a_stages
                _barrier(lgkmcnt=0)
                a128_vals = load_a128_tile(read_slot)
                a_scale = load_a_scale_tile(offset)
                h4_0 = inline_quant_load_half(next_k_tile, 0)
                h4_1 = inline_quant_load_half(next_k_tile, 1)
                rocdl.sched_barrier(0)
                acc, next_b_tile = compute_tile_prefetch_b(
                    acc,
                    b_stages[slot_b],
                    a128_vals,
                    a_scale,
                    b_scale_stages[slot_b],
                    next_k_tile,
                )
                b_scale_stages[slot_b] = load_b_scale_tile(next_k_tile)
                e8m0_0 = inline_quant_finish_half(write_slot, next_k_tile, 0, h4_0)
                e8m0_1 = inline_quant_finish_half(write_slot, next_k_tile, 1, h4_1)
                inline_quant_write_scale(next_k_tile, e8m0_0, e8m0_1)
                b_stages[slot_b] = next_b_tile

            for tail in range_constexpr(0, k_stages):
                k_tile = num_k_tiles - k_stages + tail
                read_slot = k_tile % a_stages
                slot_b = k_tile % k_stages
                _barrier(lgkmcnt=0)
                a_scale = load_a_scale_tile(k_tile)
                acc = compute_tile(
                    acc,
                    b_stages[slot_b],
                    a_scale,
                    b_scale_stages[slot_b],
                    read_slot,
                )

            gpu.barrier()

            for j in range_constexpr(num_acc_n):
                j_local = j // 2
                col_local = (
                    wave_id * arith.constant(32, index=True)
                    + arith.constant(j_local * 16, index=True)
                    + lane_mod_16
                )
                if const_expr(j % 2 == 1):
                    col_local = col_local + arith.constant(128, index=True)
                row_base = lane_div_16 * arith.constant(4, index=True)
                for v_i in range_constexpr(4):
                    row = row_base + arith.constant(v_i, index=True)
                    val = vector.extract(acc[j], static_position=[v_i], dynamic_position=[])
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
            row_local = m_lane
            result_vals = []
            local_max = c0_f32
            for e in range_constexpr(8):
                col_in_grp = kk * arith.constant(8, index=True) + arith.constant(e, index=True)
                gate_col = wave_grp * arith.constant(32, index=True) + col_in_grp
                up_col = gate_col + arith.constant(128, index=True)
                gate_v = vector.extract(
                    vector.load_op(
                        T.vec(1, f32),
                        lds_acc,
                        [row_local * arith.constant(tile_n, index=True) + gate_col],
                    ),
                    static_position=[0],
                    dynamic_position=[],
                )
                up_v = vector.extract(
                    vector.load_op(
                        T.vec(1, f32),
                        lds_acc,
                        [row_local * arith.constant(tile_n, index=True) + up_col],
                    ),
                    static_position=[0],
                    dynamic_position=[],
                )
                res = _silu_mul(gate_v, up_v)
                result_vals.append(res)
                abs_v = llvm.call_intrinsic(f32, "llvm.fabs.f32", [res], [], [])
                if const_expr(e == 0):
                    local_max = abs_v
                else:
                    local_max = _fmax_num(local_max, abs_v)

            local_max = _fmax_dpp_xor1(local_max)
            local_max = _fmax_dpp_xor2(local_max)

            amax_i32 = local_max.bitcast(i32)
            quant_scale = (amax_i32 + c0x200000_i32).bitcast(f32) * c025_f32
            sb_raw = quant_scale.bitcast(i32) >> c23_i32
            e8m0 = arith.select(arith.cmpi(CmpIPredicate.ugt, sb_raw, c254_i32), c254_i32, sb_raw)

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

            if_scale = scf.IfOp(arith.cmpi(CmpIPredicate.eq, kk, arith.constant(0, index=True)))
            with ir.InsertionPoint(if_scale.then_block):
                ku = by >> arith.constant(1, index=True)
                ikxdl = by % arith.constant(2, index=True)
                dword_off = (
                    bx * arith.constant(k_out_as_per_chunk_dw, index=True)
                    + ku * arith.constant(64, index=True)
                    + wave_grp * arith.constant(16, index=True)
                    + m_lane
                )
                e8m0_i8 = arith.TruncIOp(i8, e8m0).result
                byte_off = arith.index_cast(
                    i32,
                    dword_off * arith.constant(4, index=True)
                    + ikxdl * arith.constant(2, index=True),
                )
                buffer_ops.buffer_store(
                    e8m0_i8,
                    out_scale_rsrc,
                    byte_off,
                    cache_modifier=4,
                    offset_is_bytes=True,
                )
                scf.YieldOp([])

        if_valid = scf.IfOp(do_gemm)
        with ir.InsertionPoint(if_valid.then_block):
            _then_body()
            scf.YieldOp([])

    @flyc.jit
    def launch_gemm1_inline(
        inter_sorted_quant: fx.Pointer,
        inter_sorted_scale: fx.Pointer,
        hidden: fx.Pointer,
        w: fx.Pointer,
        w_scale: fx.Pointer,
        expert_ids: fx.Pointer,
        m_indices: fx.Pointer,
        num_valid_ids: fx.Pointer,
        m: fx.Int32,
        m_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm1_inline(
            inter_sorted_quant,
            inter_sorted_scale,
            hidden,
            w,
            w_scale,
            expert_ids,
            m_indices,
            num_valid_ids,
            m,
        ).launch(
            grid=(num_n_blocks, m_blocks, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm1_inline


def kimi_mxfp4_gemm1_inline_bm16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    m_indices: torch.Tensor,
    cumsum_tensor: torch.Tensor,
    *,
    m: int | None = None,
    inter_sorted_quant: torch.Tensor | None = None,
    inter_sorted_shuffled_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the local FlyDSL BM16 inline-quant GEMM1."""

    if m is None:
        m = int(hidden_states.shape[0])
    max_sorted = int(sorted_expert_ids.shape[0]) * BM16
    device = hidden_states.device
    if inter_sorted_quant is None:
        inter_sorted_quant = torch.empty((max_sorted, INTER_DIM // 2), device=device, dtype=torch.uint8)
    if inter_sorted_shuffled_scale is None:
        inter_scale_cols = INTER_DIM // 32
        inter_scale_bytes = max_sorted * (1024 // 64) * 4
        inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
        inter_scale_rows = (inter_scale_rows + 31) // 32 * 32
        inter_sorted_shuffled_scale = torch.empty(
            (inter_scale_rows, inter_scale_cols),
            device=device,
            dtype=torch.uint8,
        )

    if w1.element_size() == 1 and w1.dtype != torch.uint8:
        w1 = w1.view(torch.uint8)
    args = (
        _ptr_view_safe(inter_sorted_quant.view(-1)),
        _ptr_view_safe(inter_sorted_shuffled_scale.view(-1)),
        _ptr_view_safe(hidden_states.view(-1)),
        _ptr_view_safe(w1.view(-1)),
        _ptr_view_safe(w1_scale.view(-1)),
        _ptr_view_safe(sorted_expert_ids.view(-1)),
        _ptr_view_safe(m_indices.view(-1)),
        _ptr_view_safe(cumsum_tensor.view(-1)),
        int(m),
        int(sorted_expert_ids.shape[0]),
        torch.cuda.current_stream(),
    )
    _run_compiled(compile_kimi_mxfp4_gemm1_inline_bm16(), args)
    _record_current_stream(
        hidden_states,
        w1,
        w1_scale,
        sorted_expert_ids,
        m_indices,
        cumsum_tensor,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    return inter_sorted_quant, inter_sorted_shuffled_scale


@functools.lru_cache(maxsize=1)
def compile_kimi_mxfp4_gemm2_atomic_bm16():
    """Compile FlyDSL BM16 atomic GEMM2.

    Fixed-shape port target:
    ``mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT``.
    """

    tile_m = BM16
    tile_n = 256
    tile_k = 256
    total_threads = 256
    num_n_blocks = MODEL_DIM // tile_n
    max_sorted = mxfp4_small_max_sorted(128, BM16)

    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    k_unroll = INTER_DIM // tile_k
    k_half = INTER_DIM // 2
    single_a_bytes = tile_m * (tile_k // 2)
    lds_a_bytes = k_unroll * single_a_bytes
    lds_acc_bytes = tile_m * tile_n * 4

    a_q_nbytes = max_sorted * k_half
    k_as_c_k1 = (INTER_DIM // 32) // 4 // 2
    k_as_per_chunk_dw = k_as_c_k1 * 64
    a_scale_nbytes = (max_sorted // BM16) * k_as_per_chunk_dw * 4
    b_q_nbytes = EXPERTS * MODEL_DIM * k_half
    k_bs_c_n1 = MODEL_DIM // 16 // 2
    k_bs_c_k1 = (INTER_DIM // 32) // 4 // 2
    k_bs_stride_k0_dw = 64
    k_bs_stride_n0_dw = k_bs_c_k1 * 64
    k_bs_per_expert_dw = k_bs_c_n1 * k_bs_stride_n0_dw
    b_scale_nbytes = EXPERTS * k_bs_per_expert_dw * 4
    sorted_nbytes = max_sorted * 4
    out_nbytes = 128 * MODEL_DIM * 2

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_mxfp4_bm16_gemm2")
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + max(lds_a_bytes, lds_acc_bytes)

    module_name = "flydsl_kimi_mxfp4_gemm2_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT_v11"

    @flyc.kernel(name=module_name)
    def gemm2_atomic(
        arg_inter_sorted_quant: fx.Pointer,
        arg_inter_sorted_scale: fx.Pointer,
        arg_w: fx.Pointer,
        arg_w_scale: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_cumsum_tensor: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_out: fx.Pointer,
        i32_m: fx.Int32,
    ):
        i32 = T.i32
        i64 = T.i64
        f32 = T.f32
        vec2_bf16 = T.vec(2, T.bf16)
        vec2_i64 = T.vec(2, i64)
        vec4_i64 = T.vec(4, i64)
        vec8_i32 = T.vec(8, i32)
        vec4_f32 = T.vec(4, f32)
        vec16_x = T.vec(16, T.f8)

        a_q_rsrc = _ptr_buffer_resource(arg_inter_sorted_quant, a_q_nbytes)
        a_scale_rsrc = _ptr_buffer_resource(arg_inter_sorted_scale, a_scale_nbytes)
        b_q_rsrc = _ptr_buffer_resource(arg_w, b_q_nbytes)
        b_scale_rsrc = _ptr_buffer_resource(arg_w_scale, b_scale_nbytes)
        out_ptr = _extract_global_ptr(arg_out)
        expert_ptr = _extract_global_ptr(arg_sorted_expert_ids)
        cumsum_ptr = _extract_global_ptr(arg_cumsum_tensor)
        sorted_token_ptr = _extract_global_ptr(arg_sorted_token_ids)
        sorted_weights_ptr = _extract_global_ptr(arg_sorted_weights)

        tx = gpu.thread_id("x")
        pid = gpu.block_id("x")
        m_block_idx = pid // arith.constant(num_n_blocks, index=True)
        n_block_idx = pid % arith.constant(num_n_blocks, index=True)
        m_row = m_block_idx * arith.constant(tile_m, index=True)

        expert_i32 = _global_load_i32(expert_ptr, m_block_idx)
        expert_i32 = rocdl.readfirstlane(i32, expert_i32)
        expert_idx = arith.index_cast(T.index, expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(EXPERTS, type=i32))
        num_valid_i32 = _global_load_i32(cumsum_ptr, 0)
        num_valid_i32 = rocdl.readfirstlane(i32, num_valid_i32)
        m_row_i32 = arith.index_cast(i32, m_row)
        blk_valid = arith.cmpi(CmpIPredicate.ult, m_row_i32, num_valid_i32)
        do_gemm = arith.andi(exp_valid, blk_valid)

        base_ptr = allocator.get_base()
        lds_a_i8 = SmemPtr(base_ptr, lds_offset, T.f8, shape=(lds_a_bytes,)).get()
        lds_acc = SmemPtr(base_ptr, lds_offset, T.f32, shape=(tile_m * tile_n,)).get()

        shape_lds = fx.make_shape(tile_m, tile_k // 2)
        layout_lds = fx.make_layout(shape_lds, fx.make_stride(tile_k // 2, 1))
        k_blocks16 = arith.constant((tile_k // 2) // 16, index=True)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_wl = idx2crd(tx, layout_tx_wave_lane)
        wave_id = layout_get(coord_wl, 0)
        lane_id = layout_get(coord_wl, 1)
        wave_i32 = arith.index_cast(i32, wave_id)
        lane_i32 = arith.index_cast(i32, lane_id)
        coord_l16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * arith.constant(16, index=True)
        lane_word = lane_div_16 * arith.constant(16, index=True) + lane_mod_16

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c8_i32 = arith.constant(8, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        c0_i64 = arith.constant(0, type=i64)
        zero_i32 = arith.constant(0)
        c0_f32 = arith.constant(0.0, type=f32)

        def load_a_to_lds(slot: int, kt: int):
            row_off = lane_id // arith.constant(8, index=True)
            lane_mod8 = lane_id % arith.constant(8, index=True)
            lds_row_base = wave_id * arith.constant(8, index=True)
            lds_row = lds_row_base + row_off
            actual_row = m_row + lds_row
            col_sw = swizzle_xor16(
                lds_row,
                lane_mod8 * arith.constant(16, index=True),
                k_blocks16,
            )
            global_off = (
                actual_row * arith.constant(k_half, index=True)
                + arith.constant(kt * (tile_k // 2), index=True)
                + col_sw
            )
            byte_idx = (
                arith.constant(slot * single_a_bytes, index=True)
                + crd2idx([lds_row_base, arith.index(0)], layout_lds)
            )
            lds_addr = memref.extract_aligned_pointer_as_index(lds_a_i8) + byte_idx
            lds_addr_i64 = arith.index_cast(i64, lds_addr)
            lds_addr_i64 = rocdl.readfirstlane(i64, lds_addr_i64)
            lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_addr_i64)
            rocdl.raw_ptr_buffer_load_lds(
                a_q_rsrc,
                lds_ptr,
                arith.constant(16, type=i32),
                arith.index_cast(i32, global_off),
                arith.constant(0, type=i32),
                arith.constant(0, type=i32),
                arith.constant(0, type=i32),
            )

        def guarded_load_a_to_lds(slot: int, kt: int):
            if_wave = scf.IfOp(arith.cmpi(CmpIPredicate.ult, wave_i32, c2_i32))
            with ir.InsertionPoint(if_wave.then_block):
                load_a_to_lds(slot, kt)
                scf.YieldOp([])

        def load_a_packs_from_lds(slot: int, k_idx: int):
            col_base = col_offset_base + arith.constant(k_idx * 64, index=True)
            col_sw = swizzle_xor16(row_a_lds, col_base, k_blocks16)
            idx_a16 = (
                arith.constant(slot * single_a_bytes, index=True)
                + crd2idx([row_a_lds, col_sw], layout_lds)
            )
            loaded = vector.load_op(vec16_x, lds_a_i8, [idx_a16])
            a_i64x2 = vector.bitcast(vec2_i64, loaded)
            return (
                vector.extract(a_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(a_i64x2, static_position=[1], dynamic_position=[]),
            )

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
            return vector.bitcast(vec8_i32, v4)

        n_base = n_block_idx * arith.constant(tile_n, index=True)
        n_wave_base = n_base + wave_id * arith.constant(n_per_wave, index=True)
        b_load_s_bases = []
        for j in range_constexpr(num_acc_n):
            col = n_wave_base + arith.constant(j * 16, index=True)
            b_load_s_bases.append(
                expert_idx * arith.constant(MODEL_DIM * k_half, index=True)
                + col * arith.constant(k_half, index=True)
            )

        def load_b_pack(kt: int, k_idx: int, j: int):
            v_off = (
                lane_div_16 * arith.constant(256, index=True)
                + lane_mod_16 * arith.constant(16, index=True)
                + arith.constant(kt * 2048 + k_idx * 1024, index=True)
            )
            b16 = _buffer_load_vec(
                buffer_ops,
                vector,
                b_q_rsrc,
                b_load_s_bases[j] + v_off,
                elem_type=T.i8,
                vec_elems=16,
                elem_bytes=1,
                offset_in_bytes=True,
                cache_modifier=2,
            )
            b_i64x2 = vector.bitcast(vec2_i64, b16)
            return (
                vector.extract(b_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(b_i64x2, static_position=[1], dynamic_position=[]),
            )

        def load_b_tile(kt: int):
            b_tile = []
            for k_idx in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for j in range_constexpr(num_acc_n):
                    b0, b1 = load_b_pack(kt, k_idx, j)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        def load_a_scale_kt(kt: int):
            scale_idx = (
                m_block_idx * arith.constant(k_as_per_chunk_dw, index=True)
                + arith.constant(kt * k_bs_stride_k0_dw, index=True)
                + lane_word
            )
            s = buffer_ops.buffer_load(a_scale_rsrc, scale_idx, vec_width=1, dtype=i32, cache_modifier=0)
            return vector.from_elements(T.vec(1, i32), [s])

        def load_b_scale_kt(kt: int):
            mni_base = (
                n_block_idx * arith.constant(tile_n // 16 // 2, index=True)
                + wave_id * arith.constant(n_per_wave // 16 // 2, index=True)
            )
            vals = []
            for mw in range_constexpr(num_acc_n // 2):
                scale_idx = (
                    expert_idx * arith.constant(k_bs_per_expert_dw, index=True)
                    + (mni_base + arith.constant(mw, index=True))
                    * arith.constant(k_bs_stride_n0_dw, index=True)
                    + arith.constant(kt * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = buffer_ops.buffer_load(b_scale_rsrc, scale_idx, vec_width=1, dtype=i32, cache_modifier=0)
                vals.append(vector.from_elements(T.vec(1, i32), [s]))
            return vals

        cbsz = 4
        blgp = 4

        def compute_kt(acc_in, slot: int, kt: int, b_tile, a_scale, b_scale):
            acc_list = list(acc_in)
            a_scale_val = vector.extract(a_scale, static_position=[0], dynamic_position=[])
            for k_idx in range_constexpr(k_unroll):
                a0, a1 = load_a_packs_from_lds(slot, k_idx)
                a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                b_packs0, b_packs1 = b_tile[k_idx]
                for mni in range_constexpr(num_acc_n // 2):
                    b_scale_val = vector.extract(b_scale[mni], static_position=[0], dynamic_position=[])
                    for inxdl in range_constexpr(2):
                        j = mni * 2 + inxdl
                        b128 = pack_i64x4_to_i32x8(b_packs0[j], b_packs1[j], c0_i64, c0_i64)
                        acc_list[j] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            vec4_f32,
                            [
                                a128,
                                b128,
                                acc_list[j],
                                cbsz,
                                blgp,
                                k_idx * 2,
                                a_scale_val,
                                k_idx * 2 + inxdl,
                                b_scale_val,
                            ],
                        )
            return acc_list

        def _then_body():
            acc = [arith.constant_vector(0.0, vec4_f32)] * num_acc_n

            guarded_load_a_to_lds(0, 0)
            guarded_load_a_to_lds(1, 1)
            b0 = load_b_tile(0)
            b1 = load_b_tile(1)
            a_scale0 = load_a_scale_kt(0)
            a_scale1 = load_a_scale_kt(1)
            b_scale0 = load_b_scale_kt(0)
            b_scale1 = load_b_scale_kt(1)

            rocdl.s_waitcnt(0)
            gpu.barrier()
            acc = compute_kt(acc, 0, 0, b0, a_scale0, b_scale0)
            acc = compute_kt(acc, 1, 1, b1, a_scale1, b_scale1)

            for j in range_constexpr(num_acc_n):
                col_local = wave_id * arith.constant(n_per_wave, index=True)
                col_local = col_local + arith.constant(j * 16, index=True) + lane_mod_16
                row_base = lane_div_16 * arith.constant(4, index=True)
                for v_i in range_constexpr(4):
                    row = row_base + arith.constant(v_i, index=True)
                    val = vector.extract(acc[j], static_position=[v_i], dynamic_position=[])
                    vector.store(
                        vector.from_elements(T.vec(1, f32), [val]),
                        lds_acc,
                        [row * arith.constant(tile_n, index=True) + col_local],
                        alignment=4,
                    )

            gpu.barrier()

            m_lane = tx // arith.constant(32, index=True)
            n_lane = tx % arith.constant(32, index=True)
            col_start = n_lane * arith.constant(2, index=True)
            col_start_i32 = arith.index_cast(i32, col_start)
            n_block_elem_i32 = arith.index_cast(
                i32,
                n_block_idx * arith.constant(tile_n, index=True),
            )

            for mr in range_constexpr(BM16 // 8):
                row_in_block = arith.constant(mr * 8, index=True) + m_lane
                sorted_pos = m_row + row_in_block
                sorted_pos_i32 = arith.index_cast(i32, sorted_pos)
                packed = _global_load_i32(sorted_token_ptr, sorted_pos)
                token_id = packed & arith.constant(0x00FFFFFF, type=i32)
                token_ok = arith.cmpi(CmpIPredicate.ult, token_id, i32_m)
                if_token = scf.IfOp(token_ok)
                with ir.InsertionPoint(if_token.then_block):
                    weight = _global_load_f32(sorted_weights_ptr, sorted_pos)
                    for s_idx in range_constexpr(4):
                        lds_col = (
                            col_start
                            + arith.constant(s_idx * 64, index=True)
                        )
                        v0 = vector.extract(
                            vector.load_op(
                                T.vec(1, f32),
                                lds_acc,
                                [row_in_block * arith.constant(tile_n, index=True) + lds_col],
                            ),
                            static_position=[0],
                            dynamic_position=[],
                        )
                        v1 = vector.extract(
                            vector.load_op(
                                T.vec(1, f32),
                                lds_acc,
                                [
                                    row_in_block * arith.constant(tile_n, index=True)
                                    + lds_col
                                    + arith.constant(1, index=True)
                                ],
                            ),
                            static_position=[0],
                            dynamic_position=[],
                        )
                        b0 = arith.trunc_f(T.bf16, v0 * weight)
                        b1 = arith.trunc_f(T.bf16, v1 * weight)
                        frag = vector.from_elements(vec2_bf16, [b0, b1])
                        elem_off = (
                            token_id * arith.constant(MODEL_DIM, type=i32)
                            + n_block_elem_i32
                            + col_start_i32
                            + arith.constant(s_idx * 64, type=i32)
                        )
                        byte_off = elem_off * c2_i32
                        byte_off_i64 = arith.ExtSIOp(i64, arith._to_raw(byte_off)).result
                        atomic_ptr = buffer_ops.get_element_ptr(
                            out_ptr,
                            byte_offset=byte_off_i64,
                            elem_type=T.i8,
                        )
                        frag_v = frag.ir_value() if hasattr(frag, "ir_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            atomic_ptr,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                    scf.YieldOp([])

        if_valid = scf.IfOp(do_gemm)
        with ir.InsertionPoint(if_valid.then_block):
            _then_body()
            scf.YieldOp([])

    @flyc.jit
    def launch_gemm2_atomic(
        inter_sorted_quant: fx.Pointer,
        inter_sorted_scale: fx.Pointer,
        w: fx.Pointer,
        w_scale: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        cumsum_tensor: fx.Pointer,
        sorted_token_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        out: fx.Pointer,
        m: fx.Int32,
        m_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm2_atomic(
            inter_sorted_quant,
            inter_sorted_scale,
            w,
            w_scale,
            sorted_expert_ids,
            cumsum_tensor,
            sorted_token_ids,
            sorted_weights,
            out,
            m,
        ).launch(
            grid=(m_blocks * num_n_blocks, 1, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm2_atomic


def kimi_mxfp4_gemm2_atomic_bm16(
    inter_sorted_quant: torch.Tensor,
    inter_sorted_shuffled_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    cumsum_tensor: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    atomic_output: torch.Tensor,
    *,
    m: int,
) -> torch.Tensor:
    """Run the local FlyDSL BM16 atomic GEMM2.

    ``atomic_output`` must already be zero-initialized.  The fused BM16 sort
    kernel in this file does that, matching the aiter small-M pipeline.
    """

    if w2.element_size() == 1 and w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)
    args = (
        _ptr_view_safe(inter_sorted_quant.view(-1)),
        _ptr_view_safe(inter_sorted_shuffled_scale.view(-1)),
        _ptr_view_safe(w2.view(-1)),
        _ptr_view_safe(w2_scale.view(-1)),
        _ptr_view_safe(sorted_expert_ids.view(-1)),
        _ptr_view_safe(cumsum_tensor.view(-1)),
        _ptr_view_safe(sorted_token_ids.view(-1)),
        _ptr_view_safe(sorted_weights.view(-1)),
        _ptr_view_safe(atomic_output.view(-1)),
        int(m),
        int(sorted_expert_ids.shape[0]),
        torch.cuda.current_stream(),
    )
    _run_compiled(compile_kimi_mxfp4_gemm2_atomic_bm16(), args)
    _record_current_stream(
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
        w2,
        w2_scale,
        sorted_expert_ids,
        cumsum_tensor,
        sorted_token_ids,
        sorted_weights,
        atomic_output,
    )
    return atomic_output


def run_kimi_fp4_mxfp4_moe_small_bm16_flydsl_gemm1_aiter_gemm2(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """FlyDSL sort + FlyDSL GEMM1 + aiter GEMM2 validation harness."""

    import aiter

    m = int(hidden_states.shape[0])
    if w2.element_size() == 1 and w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    sort = kimi_mxfp4_sort_zero_init_bm16(topk_ids, topk_weight, m=m)
    inter_sorted_quant, inter_sorted_shuffled_scale = kimi_mxfp4_gemm1_inline_bm16(
        hidden_states,
        w1,
        w1_scale,
        sort.sorted_expert_ids,
        sort.m_indices,
        sort.cumsum_tensor,
        m=m,
    )
    aiter.mxfp4_moe_gemm2_a4w4(
        cumsum_tensor=sort.cumsum_tensor,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        w3_shuffled_quant=w2,
        w3_shuffled_scale=w2_scale,
        sorted_token_ids=sort.sorted_token_ids,
        sorted_expert_ids=sort.sorted_expert_ids,
        sorted_weights=sort.sorted_weights,
        flat_out=sort.atomic_output,
        M_logical=m,
        max_sorted=sort.max_sorted,
        kernelName="mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT",
    )
    _record_current_stream(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weight,
        topk_ids,
        sort,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    sort.atomic_output._flydsl_keepalive = (
        sort,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    return sort.atomic_output


def run_kimi_fp4_mxfp4_moe_small_bm16_all_flydsl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """FlyDSL-only BM16 small-M mxfp4 MoE path."""

    m = int(hidden_states.shape[0])
    sort = kimi_mxfp4_sort_zero_init_bm16(topk_ids, topk_weight, m=m)
    inter_sorted_quant, inter_sorted_shuffled_scale = kimi_mxfp4_gemm1_inline_bm16(
        hidden_states,
        w1,
        w1_scale,
        sort.sorted_expert_ids,
        sort.m_indices,
        sort.cumsum_tensor,
        m=m,
    )
    kimi_mxfp4_gemm2_atomic_bm16(
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
        w2,
        w2_scale,
        sort.sorted_expert_ids,
        sort.cumsum_tensor,
        sort.sorted_token_ids,
        sort.sorted_weights,
        sort.atomic_output,
        m=m,
    )
    _record_current_stream(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weight,
        topk_ids,
        sort,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    sort.atomic_output._flydsl_keepalive = (
        sort,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    return sort.atomic_output


def run_kimi_fp4_mxfp4_moe_small_bm16_flydsl_sort_aiter_gemm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Small-M BM16 pipeline harness.

    The prologue is the local FlyDSL port above.  GEMM1/GEMM2 still call the
    aiter BM16 mxfp4 kernels and are kept here only to validate the migrated
    sort buffers before replacing those GEMMs with FlyDSL bodies.
    """

    import aiter

    m = int(hidden_states.shape[0])
    if hidden_states.shape != (m, MODEL_DIM):
        raise ValueError(
            f"expected hidden_states shape {(m, MODEL_DIM)}, got {tuple(hidden_states.shape)}"
        )
    if w1.element_size() == 1 and w1.dtype != torch.uint8:
        w1 = w1.view(torch.uint8)
    if w2.element_size() == 1 and w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    sort = kimi_mxfp4_sort_zero_init_bm16(topk_ids, topk_weight, m=m)
    device = hidden_states.device
    max_sorted = sort.max_sorted

    a_quant = torch.empty((m, MODEL_DIM // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((m, MODEL_DIM // 32), device=device, dtype=torch.uint8)
    a_scale_sorted_shuffled = torch.empty((0,), device=device, dtype=torch.uint8)

    inter_sorted_quant = torch.empty(
        (max_sorted, INTER_DIM // 2), device=device, dtype=torch.uint8
    )
    inter_scale_cols = INTER_DIM // 32
    inter_scale_bytes = max_sorted * (1024 // 64) * 4
    inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
    inter_scale_rows = (inter_scale_rows + 31) // 32 * 32
    inter_sorted_shuffled_scale = torch.empty(
        (inter_scale_rows, inter_scale_cols), device=device, dtype=torch.uint8
    )

    aiter.mxfp4_moe_gemm1_a4w4(
        cumsum_tensor=sort.cumsum_tensor,
        a_quant=a_quant,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        w12_shuffled_quant=w1,
        w12_shuffled_scale=w1_scale,
        sorted_expert_ids=sort.sorted_expert_ids,
        m_indices=sort.m_indices,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        hidden_states=hidden_states,
        kernelName="mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT",
    )

    aiter.mxfp4_moe_gemm2_a4w4(
        cumsum_tensor=sort.cumsum_tensor,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        w3_shuffled_quant=w2,
        w3_shuffled_scale=w2_scale,
        sorted_token_ids=sort.sorted_token_ids,
        sorted_expert_ids=sort.sorted_expert_ids,
        sorted_weights=sort.sorted_weights,
        flat_out=sort.atomic_output,
        M_logical=m,
        max_sorted=max_sorted,
        kernelName="mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT",
    )
    _record_current_stream(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weight,
        topk_ids,
        sort,
        a_quant,
        a_scale,
        a_scale_sorted_shuffled,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    sort.atomic_output._flydsl_keepalive = (
        sort,
        a_quant,
        a_scale,
        a_scale_sorted_shuffled,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
    )
    return sort.atomic_output

if __name__ == "__main__":
    compile_kimi_mxfp4_sort_zero_init_bm16()
