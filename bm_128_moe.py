
import torch
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
"""Fixed-shape FlyDSL mxfp4 MoE path for Kimi M=16384."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
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
from kernels.layout_utils import (
    _div_pow2,
    crd2idx,
    idx2crd,
    get as layout_get,
)
from kernels.mfma_epilogues import c_shuffle_epilog
from kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    swizzle_xor16,
    tile_chunk_coord_i32,
)

from kimi_fp4_moe_16384 import (
    EXPERTS,
    INTER_DIM,
    MODEL_DIM,
    TOKEN,
    TOPK,
    _ptr_view_safe,
    _run_compiled,
)

TOKEN = 16384
BLOCK_M = 128
TOPK = 9
EXPERTS = 385
def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _ptr_buffer_resource(ptr, num_records_bytes: int):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr) # TODO(是否要这个？)
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64,
        num_records_bytes=num_records_bytes)

def _elem_offset_to_i64(elem_offset):
    
    


def _lds_i32_ptr(lds_memref, elem_offset):
    elem_offset_i64 = _elem_offset_to_i64(elem_offset)
    byte_offset_idx = ArithValue(elem_offset_i64 * arith.constant(4, type=T.i64)).index_cast(T.index)
    base_idx = memref.extract_aligned_pointer_as_index(lds_memref)
    ptr_i64 = arith.index_cast(T.i64, base_idx + byte_offset_idx)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), ptr_i64)
def _lds_atomic_add_i32(lds_memref, elem_offset, value):
    ptr = _lds_i32_ptr(lds_memref, elem_offset)
    val = value.ir_value() if hasattr(value, "ir_value") else value
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, 
        ptr, 
        val, 
        llvm.AtomicOrdering.monotonic, 
        syncscope="workgroup",
        alignment=4
    ).result

def kimi_mxfp4_sort_16384(
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor):
    assert topk_ids.shape == (TOKEN, TOPK) == topk_weight.shape
    max_sorted = div_up(TOKEN * TOPK, BLOCK_M) + EXPERTS
    blocks = max_sorted // BLOCK_M
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_mxfp4_sort")
    lds_count_offset = allocator._align(allocator.ptr, 16)
    lds_padded_offset = lds_count_offset + EXPERTS * 4 # 前面的 int * experts个吧.. 这个不需要align么？
    
    total_pairs = TOKEN * TOPK 
    topk_nbytes = total_pairs * 4
    sort_ctas = 16    
    block_offsets_nbytes = EXPERTS * sort_ctas * 4 # TODO(是否开多了？为啥不和TOPK相关)?
    threads = 1024

    @flyc.kernel(
        name = "sort kernel",
        know_block_size = [BLOCK_M, 1, 1]
    )
    def sort_count(arg_topk_ids: fx.Pointer, 
                    arg_block_offsets: fx.Pointer):
        i32 = T.i32
        tx = gpu.thread_idx.x
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(i32, tx)
        bx_i32 = arith.index_cast(i32, bx)
        topk_rsrc = _ptr_buffer_resource(arg_topk_ids, topk_nbytes)
        offsets_rsrc = _ptr_buffer_resource(arg_block_offsets, block_offsets_nbytes)
        base_ptr = allocator.get_base()
        local_count = SmemPtr(base_ptr, lds_count_offset, T.i32, shape=(EXPERTS,)).get()
        if tx < arith.constant(EXPERTS, index=True):
            memref.store(arith.constant(0, type=i32), local_count, [tx])
            # local_count[tx] = 0
        gpu.barrier()
        per_cta = div_up(total_pairs, sort_ctas)
        c_start = bx * arith.constant(per_cta, index=True)
        c_end = c_start + arith.constant(per_cta, index=True)
        c_total = arith.constant(total_pairs, index=True)

        end = arith.select(arith.cmpi(CmpIPredicate.ult, c_end, c_total), c_end, c_total)
        for it in range_constexpr(div_up(per_cta, threads)):
            idx = c_start + tx + arith.constant(it * threads, index=True)
            if idx <= end:
                eid = buffer_ops.buffer_load(topk_rsrc, idx, vec_width=1, dtype=i32)
                _lds_atomic_add_i32(local_count, eid, 1)
                


    
