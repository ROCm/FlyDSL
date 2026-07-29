# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Eight-wave BF16/FP16 GEMM for AMD CDNA4.

The kernel uses a 2x4 wave arrangement and slice-MN ping-pong pipeline over a
256x256x64 workgroup tile.  A and B are row-major; B represents the transposed
right-hand operand, so the operation is C = A @ B.T.

Pipeline design adapted from the gfx950 Gluon inter-wave GEMM tutorial:
https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/inter_wave/a16w16
"""

import functools
from typing import Literal

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import GTensor, _run_compiled, get_dtype_in_kernel
from kernels.common.workgroup_mapping import remap_xcd_grouped_pid
from kernels.gemm.splitk_hgemm import swizzle_xor16


@functools.lru_cache(maxsize=128)
def compile_hgemm_8w(
    *,
    dtype: Literal["bf16", "fp16"],
    n: int,
    k: int,
    num_xcds: int = 8,
    group_size_m: int = 4,
):
    """Compile the gfx950 8-wave GEMM specialization."""
    assert get_rocm_arch() == "gfx950"
    assert dtype in ("bf16", "fp16")
    assert n >= 256 and n % 256 == 0
    assert k >= 128 and k % 128 == 0
    assert num_xcds > 0 and group_size_m > 0

    block_m = 256
    block_n = 256
    block_k = 64
    stages = 2
    block_threads = 512
    warp_size = 64
    dtype_bytes = 2
    dma_bytes = 16
    ldg_vec_size = 8

    mfma_m = 16
    mfma_n = 16
    mfma_k = 32
    mfma_a_frag_values = 8
    mfma_b_frag_values = 8
    mfma_c_frag_values = 4

    k_iters = k // block_k
    warp_k_steps = block_k // mfma_k
    half_m = block_m // 2
    half_n = block_n // 2
    waves_m = 2
    waves_n = 4
    quadrant_m_steps = half_m // waves_m // mfma_m
    quadrant_n_steps = half_n // waves_n // mfma_n
    quadrant_frags = quadrant_m_steps * quadrant_n_steps
    num_c_frags = 4 * quadrant_frags

    ldg_async_vec_size = dma_bytes // dtype_bytes
    ldg_a_x_threads = block_k // ldg_async_vec_size
    ldg_b_x_threads = block_k // ldg_async_vec_size
    ldg_a_half_count = half_m * block_k // ldg_async_vec_size // block_threads
    ldg_b_half_count = half_n * block_k // ldg_async_vec_size // block_threads
    ldg_c_x_threads = block_n // ldg_vec_size
    ldg_c_count = block_m * block_n // ldg_vec_size // block_threads
    n_blocks = n // block_n
    lds_elems = block_m * block_n
    elem_cls = fx.BFloat16 if dtype == "bf16" else fx.Float16
    kernel_dtype = "bf16" if dtype == "bf16" else "f16"
    kernel_name = f"hgemm_8wave_{dtype}_{block_m}x{block_n}x{block_k}_xcd{num_xcds}_gm{group_size_m}"

    @fx.struct
    class SharedStorage:
        a_top: fx.Array[elem_cls, stages * half_m * block_k, 16]
        a_bot: fx.Array[elem_cls, stages * half_m * block_k, 16]
        b_left: fx.Array[elem_cls, stages * half_n * block_k, 16]
        b_right: fx.Array[elem_cls, stages * half_n * block_k, 16]

    def mfma(a_frag, b_frag, c_frag):
        result_type = T.vec(mfma_c_frag_values, T.f32)
        operands = [a_frag, b_frag, c_frag, 0, 0, 0]
        if dtype == "bf16":
            return rocdl.mfma_f32_16x16x32_bf16(result_type, operands)
        return rocdl.mfma_f32_16x16x32_f16(result_type, operands)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        m: fx.Int32,
    ):
        dtype_ = get_dtype_in_kernel(kernel_dtype)
        acc_init = arith.constant_vector(0.0, T.vec(mfma_c_frag_values, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_ptr = lds.a_top.ptr
        smem_view = lds.a_top.view(fx.make_layout(lds_elems, 1))

        num_pid_m = (m + block_m - 1) // block_m
        block_m_idx, block_n_idx = remap_xcd_grouped_pid(
            fx.block_idx.x,
            num_pid_m,
            n_blocks,
            num_xcds=num_xcds,
            group_size_m=group_size_m,
        )

        tid = fx.thread_idx.x
        wave_id = tid // warp_size
        lane_id = tid % warp_size
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        m_offset = fx.Index(block_m_idx * block_m)
        n_offset = fx.Index(block_n_idx * block_n)
        k_blocks16 = fx.Int32(block_k * dtype_bytes // 16)
        ldmatrix_a_m_idx = lane_id % mfma_m
        ldmatrix_a_k_vec_idx = lane_id // mfma_m * mfma_a_frag_values
        ldmatrix_b_n_idx = lane_id % mfma_n
        ldmatrix_b_k_vec_idx = lane_id // mfma_n * mfma_b_frag_values
        c_frags = [acc_init] * num_c_frags
        ks_begin = fx.Int32(0)

        def cs_off(row, col):
            return row * block_n + col

        def wait_barrier(vmcnt=0):
            asm = f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier"
            llvm.InlineAsmOp(None, [], asm, "", has_side_effects=True)

        def get_dma_copy_warp_offset():
            return rocdl.readfirstlane(
                T.i64,
                arith.index_cast(
                    T.i64,
                    fx.Index(wave_id) * arith.constant(warp_size * dma_bytes, index=True),
                ),
            )

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset):
            asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds"
            llvm.InlineAsmOp(None, [lds_ptr, global_offset, rsrc], asm, "s,v,s", has_side_effects=True)

        def ldg_sts_a_half_async(k_offset, lds_stage, m_half):
            """Copy one 128x64 A half-tile, preserving the full-tile LDS layout."""
            a_half_lds = lds.a_top if const_expr(m_half == 0) else lds.a_bot
            for i in range_constexpr(ldg_a_half_count):
                global_tid = block_threads * i + tid
                m_in_half = global_tid // ldg_a_x_threads
                m_local_idx = fx.Index(m_half * half_m) + m_in_half
                k_local_idx = global_tid % ldg_a_x_threads * ldg_async_vec_size
                col_in_bytes = k_local_idx * dtype_bytes
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                row_idx = m_offset + m_local_idx
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset) + fx.Index(col_in_bytes // dtype_bytes)
                global_offset = A_.linear_offset((safe_row_idx, col_idx)) * dtype_bytes
                global_offset = arith.index_cast(T.i32, global_offset)
                if const_expr(i == 0):
                    lds_byte_offset = fx.Int64(lds_stage) * half_m * block_k * dtype_bytes
                    lds_base = fx.Int64(fx.ptrtoint(a_half_lds.ptr)) + lds_byte_offset
                    lds_ptr_base = buffer_ops.create_llvm_ptr(lds_base, address_space=3)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=block_threads * dma_bytes,
                    )
                buffer_load_lds_inline(A_.rsrc, lds_ptr, global_offset)

        def ldg_sts_b_half_async(k_offset, lds_stage, n_half):
            """Copy one 128x64 B half-tile, preserving the full-tile LDS layout."""
            b_half_lds = lds.b_left if const_expr(n_half == 0) else lds.b_right
            for i in range_constexpr(ldg_b_half_count):
                global_tid = block_threads * i + tid
                n_in_half = global_tid // ldg_b_x_threads
                n_local_idx = fx.Index(n_half * half_n) + n_in_half
                k_local_idx = global_tid % ldg_b_x_threads * ldg_async_vec_size
                col_in_bytes = k_local_idx * dtype_bytes
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                row_idx = n_offset + n_local_idx
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset) + fx.Index(col_in_bytes // dtype_bytes)
                global_offset = B_.linear_offset((safe_row_idx, col_idx)) * dtype_bytes
                global_offset = arith.index_cast(T.i32, global_offset)
                if const_expr(i == 0):
                    lds_byte_offset = fx.Int64(lds_stage) * half_n * block_k * dtype_bytes
                    lds_base = fx.Int64(fx.ptrtoint(b_half_lds.ptr)) + lds_byte_offset
                    lds_ptr_base = buffer_ops.create_llvm_ptr(lds_base, address_space=3)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=block_threads * dma_bytes,
                    )
                buffer_load_lds_inline(B_.rsrc, lds_ptr, global_offset)

        def ldmatrix_a_slice(lds_stage, m_half):
            s = fx.Index(lds_stage)
            a_half_lds = lds.a_top if const_expr(m_half == 0) else lds.a_bot
            a_frags = []
            for kk in range_constexpr(warp_k_steps):
                warp_atom_k_idx = kk * mfma_k
                for ii in range_constexpr(quadrant_m_steps):
                    row = wave_m * (half_m // waves_m) + ii * mfma_m + ldmatrix_a_m_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * dtype_bytes
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    a_frags.append(
                        fx.ptr_load(
                            a_half_lds.ptr + (fx.Int64(s) * half_m + row) * block_k + col_in_bytes // dtype_bytes,
                            result_type=fx.Vector.make_type(
                                mfma_a_frag_values,
                                elem_cls,
                            ),
                        ).ir_value()
                    )
            return a_frags

        def ldmatrix_b_slice(lds_stage, n_half):
            s = fx.Index(lds_stage)
            b_half_lds = lds.b_left if const_expr(n_half == 0) else lds.b_right
            b_frags = []
            for kk in range_constexpr(warp_k_steps):
                warp_atom_k_idx = kk * mfma_k
                for jj in range_constexpr(quadrant_n_steps):
                    row = wave_n * (half_n // waves_n) + jj * mfma_n + ldmatrix_b_n_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * dtype_bytes
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    b_frags.append(
                        fx.ptr_load(
                            b_half_lds.ptr + (fx.Int64(s) * half_n + row) * block_k + col_in_bytes // dtype_bytes,
                            result_type=fx.Vector.make_type(
                                mfma_b_frag_values,
                                elem_cls,
                            ),
                        ).ir_value()
                    )
            return b_frags

        def compute_slice_quadrant(a_frags, b_frags, c_quad):
            c_new = [cx for cx in c_quad]
            for kk in range_constexpr(warp_k_steps):
                for ii in range_constexpr(quadrant_m_steps):
                    a_frag = a_frags[kk * quadrant_m_steps + ii]
                    for jj in range_constexpr(quadrant_n_steps):
                        b_frag = b_frags[kk * quadrant_n_steps + jj]
                        c_idx = ii * quadrant_n_steps + jj
                        c_new[c_idx] = mfma(a_frag, b_frag, c_new[c_idx])
            return c_new

        warp_offset = get_dma_copy_warp_offset()

        def pipeline_barrier(vmcnt=None, lgkmcnt=None):
            rocdl.sched_barrier(0)
            # Use a real ROCDL wait op, rather than hiding s_waitcnt in
            # inline assembly.  LLVM's AMDGPU waitcnt insertion then sees
            # that LDS reads are complete and does not add a descending
            # lgkmcnt wait in front of nearly every MFMA.
            if const_expr(vmcnt is not None or lgkmcnt is not None):
                waitcnt = 0x70  # expcnt(7): do not wait for exports.
                waitcnt |= 0xF if const_expr(vmcnt is None) else vmcnt
                waitcnt |= (0xF if const_expr(lgkmcnt is None) else lgkmcnt) << 8
                rocdl.s_waitcnt(waitcnt)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # Match Gluon's two-stage prologue.  Each half-tile emits two
        # direct-to-LDS VMEM operations per wave, so the two stages leave
        # 16 outstanding operations.  vmcnt(12) makes B-left/A-top ready
        # while retaining the remaining prefetches in flight.
        for stage in range_constexpr(2):
            k_offset = ks_begin + stage * block_k
            ldg_sts_b_half_async(k_offset, stage, 0)
            ldg_sts_a_half_async(k_offset, stage, 0)
            ldg_sts_a_half_async(k_offset, stage, 1)
            ldg_sts_b_half_async(k_offset, stage, 1)
        pipeline_barrier(vmcnt=12)

        b_left = ldmatrix_b_slice(0, 0)
        a_top = ldmatrix_a_slice(0, 0)
        pipeline_barrier(lgkmcnt=0)

        # Open the one-stage phase shift used by Gluon's warp pipeline:
        # waves 4..7 wait here until waves 0..3 reach the first loop
        # barrier.  At each following boundary, one group executes MFMA
        # while the other performs LDS reads and direct-to-LDS copies.
        if wave_m != fx.Int32(0):
            pipeline_barrier()

        def end_compute_stage():
            rocdl.s_setprio(1)
            pipeline_barrier()

        def end_memory_stage():
            rocdl.s_setprio(0)
            # As in the Gluon ISA, retain ten younger direct-to-LDS
            # operations while waiting for this region's operands.
            pipeline_barrier(vmcnt=10, lgkmcnt=0)

        # The body is unrolled by two so stage indices are compile-time
        # constants.  B-left/A-top are loop-carried registers: the third
        # and fourth memory regions load them from the other LDS stage,
        # eliminating the extra all-wave wait at every K tile.
        init_state = [ks_begin] + c_frags + b_left + a_top
        for bki, state in range(0, k_iters - 2, 2, init=init_state):
            k_offset = state[0]
            c_tl = state[1 : 1 + quadrant_frags]
            c_bl = state[1 + quadrant_frags : 1 + 2 * quadrant_frags]
            c_tr = state[1 + 2 * quadrant_frags : 1 + 3 * quadrant_frags]
            c_br = state[1 + 3 * quadrant_frags : 1 + 4 * quadrant_frags]
            operand_base = 1 + 4 * quadrant_frags
            b_left_0 = state[operand_base : operand_base + warp_k_steps * quadrant_n_steps]
            a_top_0 = state[operand_base + warp_k_steps * quadrant_n_steps :]

            # K tile 2*bki, LDS stage 0.
            end_memory_stage()
            c_tl = compute_slice_quadrant(a_top_0, b_left_0, c_tl)
            end_compute_stage()

            a_bot_0 = ldmatrix_a_slice(0, 1)
            ldg_sts_b_half_async(k_offset + 2 * block_k, 0, 0)
            end_memory_stage()
            c_bl = compute_slice_quadrant(a_bot_0, b_left_0, c_bl)
            end_compute_stage()

            b_right_0 = ldmatrix_b_slice(0, 1)
            ldg_sts_a_half_async(k_offset + 2 * block_k, 0, 0)
            end_memory_stage()
            c_tr = compute_slice_quadrant(a_top_0, b_right_0, c_tr)
            end_compute_stage()

            b_left_1 = ldmatrix_b_slice(1, 0)
            ldg_sts_a_half_async(k_offset + 2 * block_k, 0, 1)
            end_memory_stage()
            c_br = compute_slice_quadrant(a_bot_0, b_right_0, c_br)
            end_compute_stage()

            a_top_1 = ldmatrix_a_slice(1, 0)
            ldg_sts_b_half_async(k_offset + 2 * block_k, 0, 1)

            # K tile 2*bki+1, LDS stage 1.
            end_memory_stage()
            c_tl = compute_slice_quadrant(a_top_1, b_left_1, c_tl)
            end_compute_stage()

            a_bot_1 = ldmatrix_a_slice(1, 1)
            ldg_sts_b_half_async(k_offset + 3 * block_k, 1, 0)
            end_memory_stage()
            c_bl = compute_slice_quadrant(a_bot_1, b_left_1, c_bl)
            end_compute_stage()

            b_right_1 = ldmatrix_b_slice(1, 1)
            ldg_sts_a_half_async(k_offset + 3 * block_k, 1, 0)
            end_memory_stage()
            c_tr = compute_slice_quadrant(a_top_1, b_right_1, c_tr)
            end_compute_stage()

            b_left_next = ldmatrix_b_slice(0, 0)
            ldg_sts_a_half_async(k_offset + 3 * block_k, 1, 1)
            end_memory_stage()
            c_br = compute_slice_quadrant(a_bot_1, b_right_1, c_br)
            end_compute_stage()

            a_top_next = ldmatrix_a_slice(0, 0)
            ldg_sts_b_half_async(k_offset + 3 * block_k, 1, 1)

            results = yield ([k_offset + fx.Int32(2 * block_k)] + c_tl + c_bl + c_tr + c_br + b_left_next + a_top_next)

        # Complementary extra barrier realigns both four-wave groups
        # before the non-pipelined two-tile drain and the output store.
        rocdl.s_setprio(0)
        if wave_m == fx.Int32(0):
            pipeline_barrier()

        c_tl = results[1 : 1 + quadrant_frags]
        c_bl = results[1 + quadrant_frags : 1 + 2 * quadrant_frags]
        c_tr = results[1 + 2 * quadrant_frags : 1 + 3 * quadrant_frags]
        c_br = results[1 + 3 * quadrant_frags : 1 + 4 * quadrant_frags]
        operand_base = 1 + 4 * quadrant_frags
        b_left = results[operand_base : operand_base + warp_k_steps * quadrant_n_steps]
        a_top = results[operand_base + warp_k_steps * quadrant_n_steps :]

        # Drain the two prefetched K tiles after all direct-to-LDS writes
        # have completed.  This is off the hot loop and intentionally uses
        # ordinary workgroup synchronization.
        wait_barrier(0)
        c_tl = compute_slice_quadrant(a_top, b_left, c_tl)
        a_bot = ldmatrix_a_slice(0, 1)
        c_bl = compute_slice_quadrant(a_bot, b_left, c_bl)
        b_right = ldmatrix_b_slice(0, 1)
        c_tr = compute_slice_quadrant(a_top, b_right, c_tr)
        c_br = compute_slice_quadrant(a_bot, b_right, c_br)

        b_left = ldmatrix_b_slice(1, 0)
        a_top = ldmatrix_a_slice(1, 0)
        rocdl.s_waitcnt(0)
        c_tl = compute_slice_quadrant(a_top, b_left, c_tl)
        a_bot = ldmatrix_a_slice(1, 1)
        c_bl = compute_slice_quadrant(a_bot, b_left, c_bl)
        b_right = ldmatrix_b_slice(1, 1)
        c_tr = compute_slice_quadrant(a_top, b_right, c_tr)
        c_br = compute_slice_quadrant(a_bot, b_right, c_br)
        c_frags = c_tl + c_bl + c_tr + c_br

        # Reuse the operand LDS allocation to stage the output tile.
        stmatrix_c_m_vec_idx = lane_id // mfma_n * mfma_c_frag_values
        stmatrix_c_n_idx = lane_id % mfma_n
        gpu.barrier()
        for m_half in range_constexpr(2):
            for n_half in range_constexpr(2):
                # c_frags is flattened TL, BL, TR, BR to match the compute
                # pipeline's operand-lifetime order.
                quad_base = (m_half + 2 * n_half) * quadrant_frags
                for ii in range_constexpr(quadrant_m_steps):
                    warp_atom_m_idx = m_half * half_m + wave_m * (half_m // waves_m) + ii * mfma_m
                    for jj in range_constexpr(quadrant_n_steps):
                        warp_atom_n_idx = n_half * half_n + wave_n * (half_n // waves_n) + jj * mfma_n
                        c_idx = quad_base + ii * quadrant_n_steps + jj
                        for kk in range_constexpr(mfma_c_frag_values):
                            lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                            lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                            val = vector.extract(c_frags[c_idx], static_position=[kk], dynamic_position=[])
                            fx.memref_store(
                                val.truncf(dtype_),
                                smem_view,
                                fx.Int32(cs_off(lds_m_idx, lds_n_idx)),
                            )

        gpu.barrier()
        for i in range_constexpr(ldg_c_count):
            global_tid = block_threads * i + tid
            m_local_idx = fx.Int64(global_tid // ldg_c_x_threads)
            n_local_idx = fx.Int64(global_tid % ldg_c_x_threads * ldg_vec_size)
            m_global_idx = m_offset + m_local_idx
            in_bounds = arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m))
            if_in_bounds = scf.IfOp(in_bounds, results_=[], has_else=False)
            with ir.InsertionPoint(if_in_bounds.then_block):
                vec = fx.ptr_load(
                    smem_ptr + cs_off(m_local_idx, n_local_idx),
                    result_type=fx.Vector.make_type(ldg_vec_size, elem_cls),
                ).ir_value()
                C_.vec_store((m_global_idx, n_offset + n_local_idx), vec, ldg_vec_size)
                scf.YieldOp([])

    @flyc.jit
    def launch_hgemm_8w(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        num_pid_m = (m + block_m - 1) // block_m
        hgemm_kernel._func.__name__ = kernel_name
        hgemm_kernel(C, A, B, m).launch(
            grid=(num_pid_m * n_blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_hgemm_8w


def hgemm_8w_(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_xcds: int = 8,
    group_size_m: int = 4,
    stream: torch.cuda.Stream | None = None,
):
    """Run C = A @ B.T using the cached compiled 8-wave kernel."""
    assert a.ndim == 2 and b.ndim == 2 and c.ndim == 2
    m, k = a.shape
    n, b_k = b.shape
    assert b_k == k and c.shape == (m, n)
    assert a.device == b.device == c.device
    assert a.dtype == b.dtype == c.dtype
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()

    if a.dtype == torch.bfloat16:
        dtype = "bf16"
    elif a.dtype == torch.float16:
        dtype = "fp16"
    else:
        raise NotImplementedError("hgemm_8w_ supports BF16 and FP16 tensors")

    exe = compile_hgemm_8w(
        dtype=dtype,
        n=n,
        k=k,
        num_xcds=num_xcds,
        group_size_m=group_size_m,
    )
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    _run_compiled(exe, c, a, b, m, stream)


__all__ = ["compile_hgemm_8w", "hgemm_8w_"]
