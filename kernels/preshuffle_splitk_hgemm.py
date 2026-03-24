import time
import torch
import argparse
import functools
import numpy as np
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir.dialects import llvm, fly, memref
from flydsl.compiler.protocol import fly_values

from kernels.tensor_shim import get_dtype_in_kernel, GTensor, STensor, _to_raw
fm_fast = arith.FastMathFlags.fast


C_OUT_TABLE = {}
class HGEMMOut:
    def __init__(self, m, n, dtype, device):
        global C_OUT_TABLE
        lookup_key = f"{m}x{n}:{dtype}:{device}"
        if C_OUT_TABLE.get(lookup_key, None) is None:
            C_OUT_TABLE[lookup_key] = [
                torch.zeros(m, n, dtype=dtype, device=device),
                torch.zeros(m, n, dtype=dtype, device=device),
                int(0)
            ]
        self.lookup_key = lookup_key
    
    def get(self):
        values = C_OUT_TABLE[self.lookup_key]
        data_idx = values[2]
        return values[data_idx]
    
    def get_next(self):
        values = C_OUT_TABLE[self.lookup_key]
        data_idx = values[2]
        next_data_idx = 1 - data_idx
        return values[next_data_idx]
    
    def switch(self):
        values = C_OUT_TABLE[self.lookup_key]
        values[2] = int(1 - values[2])


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


@functools.lru_cache(maxsize=1024)
def compile_hgemm_kernel(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_K: int = 64,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
    TILE_M: int = 128,
    TILE_N: int = 128,
    PACK_N: int = 2,
    STAGES : int = 2,
    ASYNC_COPY: bool = False,
    B_TO_LDS: bool = False,
    B_PRE_SHUFFLE: bool = True,
    SPLIT_K: int = 1,
):
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    assert BLOCK_M_WARPS * BLOCK_N_WARPS == 4
    assert STAGES in [2,]
    if B_PRE_SHUFFLE == True:
        assert B_TO_LDS == False
    if SPLIT_K > 1:
        assert PACK_N == 2

    # Fixed parameters:
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_FRAG_VALUES = 4
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8

    # Propagated parameters:
    MFMA_PER_WARP_K = LDG_VEC_SIZE // WMMA_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N * PACK_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert (BLOCK_K % WARP_ATOM_K == 0) and (WARP_K_STEPS >= 1)
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (m >= BLOCK_M) and (m % BLOCK_M == 0)
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // LDG_VEC_SIZE // BLOCK_THREADS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // LDG_VEC_SIZE // BLOCK_THREADS
    LDG_REG_C_COUNT = BLOCK_M * BLOCK_N // LDG_VEC_SIZE // BLOCK_THREADS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1)
    if SPLIT_K > 1:
        assert LDG_REG_C_COUNT >= 1
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS parameters:
    gpu_arch = get_rocm_arch()
    DMA_BYTES = 4 if gpu_arch == "gfx942" else 16
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    if B_TO_LDS:
        smem_b_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}TN"
    if B_PRE_SHUFFLE:
        KERNEL_NAME += "_BP"
    if SPLIT_K > 1:
        KERNEL_NAME += f"_SPK{SPLIT_K}"

    @flyc.kernel
    def hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        CLEAN: fx.Tensor,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        if dtype == 'bf16':
            mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
        else:
            mfma_fn = rocdl.mfma_f32_16x16x16f16
        c_zero_f = arith.constant(0.0, type=T.f32)
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        A_ = GTensor(A, dtype=dtype_, shape=(m, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(m, n))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        if B_TO_LDS:
            smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,))
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        if B_PRE_SHUFFLE:
            # origin: n // WARP_ATOM_N, WARP_ATOM_N, k // WARP_ATOM_K, WARP_ATOM_K // LDG_VEC_SIZE, LDG_VEC_SIZE
            SHUFFLED_B_ = GTensor(B, dtype=dtype_, shape=(
                n // WARP_ATOM_N, k // WARP_ATOM_K, WARP_ATOM_K // LDG_VEC_SIZE, WARP_ATOM_N, LDG_VEC_SIZE))
        
        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_idx = fx.block_idx.y
        ks_idx = fx.Index(fx.block_idx.z)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_pk_idx = w_tid % WMMA_N * PACK_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_FRAG_VALUES * MFMA_PER_WARP_K
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS * PACK_N
        c_frags = [acc_init] * C_FRAGS_LEN

        def zero_c():
            CLEAN_ = GTensor(CLEAN, dtype=dtype_, shape=(m, n))
            cond = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
            if cond:
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_C_X_THREADS
                    n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                    CLEAN_.vec_store((m_offset + m_local_idx, n_offset + n_local_idx), zero_vec, LDG_VEC_SIZE)
        
        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                vec = A_.vec_load((m_offset + m_local_idx, k_offset + k_local_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs
        
        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                as_.vec_store((fx.Index(lds_stage), m_local_idx, col_in_bytes // DTYPE_BYTES), vecs[i], LDG_VEC_SIZE)
        
        def ldg_sts_a_async(k_offset, lds_stage):
            LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
            LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
            LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                # get offset
                global_offset = A_.linear_offset((m_offset + m_local_idx, k_offset + col_in_bytes // DTYPE_BYTES)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = as_.linear_offset((fx.Index(lds_stage), m_local_idx, k_local_idx)) * DTYPE_BYTES
                # get lds ptr
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_addr = memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                lds_addr_ = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                # dma copy
                rocdl.raw_ptr_buffer_load_lds(
                    A_.rsrc,
                    lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )
                # vec = A_.vec_load((m_offset + m_local_idx, k_offset + col_in_bytes // DTYPE_BYTES), LDG_ASYNC_VEC_SIZE)
                # as_.vec_store((fx.Index(lds_stage), m_local_idx, k_local_idx), vec, LDG_ASYNC_VEC_SIZE)
        
        def ldg_b(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS
                k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                vec = B_.vec_load((n_offset + n_local_idx, k_offset + k_local_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs
        
        def sts_b(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS
                k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                bs_.vec_store((fx.Index(lds_stage), n_local_idx, col_in_bytes // DTYPE_BYTES), vecs[i], LDG_VEC_SIZE)
        
        def lds_matrix_a(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_FRAG_VALUES * MFMA_PER_WARP_K)
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            return a_frags
        
        def lds_matrix_b(lds_stage):
            s = fx.Index(lds_stage)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS * PACK_N)
            for ii in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row_pk = warp_atom_n_idx + ldmatrix_b_n_pk_idx
                    for pki in range_constexpr(PACK_N):
                        row = row_pk + pki
                        col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        vec = bs_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_FRAG_VALUES * MFMA_PER_WARP_K)
                        b_frags[(kk * WARP_N_STEPS + ii) * PACK_N + pki] = vec
            return b_frags
        
        def ldg_matrix_b(k_offset):
            vecs = []
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    n_pk_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_pk_idx
                    for pki in range_constexpr(PACK_N):
                        n_idx = n_pk_idx + pki
                        k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        if not B_PRE_SHUFFLE:
                            vec = B_.vec_load((n_idx, k_idx), WMMA_FRAG_VALUES * MFMA_PER_WARP_K)
                            vecs.append(vec)
                        else:
                            idx_0 = n_idx // WARP_ATOM_N
                            idx_1 = n_idx % WARP_ATOM_N
                            idx_2 = k_idx // WARP_ATOM_K
                            idx_2_temp = k_idx % WARP_ATOM_K
                            idx_3 = idx_2_temp // LDG_VEC_SIZE
                            vec = SHUFFLED_B_.vec_load((idx_0, idx_2, idx_3, idx_1, 0), LDG_VEC_SIZE)
                            vecs.append(vec)
            return vecs
        
        def block_mma_sync(a_frags, b_frags, c_frags):
            # wmma
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag_vec_pack = a_frags[kk * WARP_M_STEPS + ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        for pki in range_constexpr(PACK_N):
                            b_frag_vec_pack = b_frags[(kk * WARP_N_STEPS + jj) * PACK_N + pki]
                            # split a
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag_vec_pack)
                            a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                            a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                            a_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64]))
                            a_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64]))
                            # split b
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag_vec_pack)
                            b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                            b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                            b_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64]))
                            b_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64]))
                            # handle bf16
                            if dtype == 'bf16':
                                a_v0 = vector.bitcast(T.vec(4, T.i16), a_v0)
                                a_v1 = vector.bitcast(T.vec(4, T.i16), a_v1)
                                b_v0 = vector.bitcast(T.vec(4, T.i16), b_v0)
                                b_v1 = vector.bitcast(T.vec(4, T.i16), b_v1)
                            # wmma
                            c_idx = (ii * WARP_N_STEPS + jj) * PACK_N + pki
                            acc_in = c_frags[c_idx]
                            acc_mid = mfma_fn(T.f32x4, [a_v0, b_v0, acc_in, 0, 0, 0])
                            c_frags[c_idx] = mfma_fn(T.f32x4, [a_v1, b_v1, acc_mid, 0, 0, 0])
        
        if SPLIT_K > 1:
            zero_c()
        
        if B_TO_LDS:
            # SLOW PATH
            
            a_regs = ldg_a(ks_begin)
            sts_a(a_regs, 0)
            b_regs = ldg_b(ks_begin)
            sts_b(b_regs, 0)
            gpu.barrier()

            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 :]
                a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_regs_next = ldg_b(k_offset + BLOCK_K)
                a_frags = lds_matrix_a(current_stage)
                b_frags = lds_matrix_b(current_stage)
                block_mma_sync(a_frags, b_frags, c_frags)
                sts_a(a_regs_next, next_stage)
                sts_b(b_regs_next, next_stage)
                gpu.barrier()
                k_offset = k_offset + fx.Int32(BLOCK_K)
                results = yield [k_offset, next_stage] + c_frags
            
            k_offset = results[0]
            current_stage = results[1]
            c_frags = results[2 :]
            a_frags = lds_matrix_a(current_stage)
            b_frags = lds_matrix_b(current_stage)
            block_mma_sync(a_frags, b_frags, c_frags)
        
        else:

            a_regs = ldg_a(ks_begin)
            sts_a(a_regs, 0)
            b_frags = ldg_matrix_b(ks_begin)
            gpu.barrier()

            # ============ Main K-loop with scheduling ============
            # Initial scheduling barrier to reset hardware scheduler state
            rocdl.sched_barrier(0)
            def hot_loop_scheduler():
                assert MFMA_PER_WARP_K == 2
                mfma_group = WARP_M_STEPS * WARP_N_STEPS
                # mfma_total = WARP_K_STEPS * mfma_group * MFMA_PER_WARP_K
                LDG_REG_A_COUNT_PART = LDG_REG_A_COUNT // WARP_K_STEPS
                if LDG_REG_A_COUNT_PART == 0:
                    rocdl.sched_vmem(1)
                for sche_i in range_constexpr(WARP_K_STEPS):
                    rocdl.sched_vmem(LDG_REG_A_COUNT_PART) # ldg_a next
                    rocdl.sched_dsrd(WARP_M_STEPS) # lds_matrix_a
                    for pki in range_constexpr(PACK_N):
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_vmem(WARP_N_STEPS) # ldg_b next
                        rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dswr(LDG_REG_A_COUNT_PART) # sts a next
                rocdl.sched_barrier(0)

            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags + b_frags
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN :]
                if not ASYNC_COPY:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                else:
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                a_frags = lds_matrix_a(current_stage)
                block_mma_sync(a_frags, b_frags, c_frags)
                if not ASYNC_COPY:
                    sts_a(a_regs_next, next_stage)
                hot_loop_scheduler()
                gpu.barrier()
                k_offset = k_offset + fx.Int32(BLOCK_K)
                results = yield [k_offset, next_stage] + c_frags + b_frags_next
            
            k_offset = results[0]
            current_stage = results[1]
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN :]
            a_frags = lds_matrix_a(current_stage)
            block_mma_sync(a_frags, b_frags, c_frags)
        
        # store results
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_FRAG_VALUES
        stmatrix_c_n_pk_idx = w_tid % WMMA_N * PACK_N
        for ii in range_constexpr(WARP_M_STEPS):
            g_warp_atom_m_idx = m_offset + warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                g_warp_atom_n_idx = n_offset + warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_FRAG_VALUES):
                    out_m_idx = g_warp_atom_m_idx + stmatrix_c_m_vec_idx + kk
                    if PACK_N > 1:
                        pk_val = []
                        for pki in range_constexpr(PACK_N):
                            vec_ = c_frags[(ii * WARP_N_STEPS + jj) * PACK_N + pki]
                            val_ = vector.extract(vec_, static_position=[kk], dynamic_position=[])
                            pk_val.append(val_)
                        pk_val = vector.from_elements(T.vec(PACK_N, T.f32), pk_val)
                        pk_val = pk_val.truncf(T.vec(PACK_N, dtype_))
                        out_n_pk_idx = g_warp_atom_n_idx + stmatrix_c_n_pk_idx
                        if SPLIT_K > 1:
                            _ptr_type = ir.Type.parse("!llvm.ptr<1>")
                            _i64_type = T.i64
                            out_raw = fly_values(C)[0]
                            out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
                            out_base_int = llvm.PtrToIntOp(_i64_type, out_base_ptr).result
                            linear_bytes_offset = C_.linear_offset((out_m_idx, out_n_pk_idx)) * DTYPE_BYTES
                            byte_offset_i64 = arith.index_cast(T.i64, linear_bytes_offset)
                            addr_i64 = llvm.AddOp(out_base_int, byte_offset_i64, llvm.IntegerOverflowFlags(0)).result
                            out_ptr = llvm.IntToPtrOp(_ptr_type, addr_i64).result
                            out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                            pk_val_v = pk_val._value if hasattr(pk_val, "_value") else pk_val
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v,
                                pk_val_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=4,
                            )
                        else:
                            C_.vec_store((out_m_idx, out_n_pk_idx), pk_val, PACK_N)
                    else:
                        out_n_idx = g_warp_atom_n_idx + stmatrix_c_n_pk_idx
                        val = vector.extract(c_frags[ii * WARP_N_STEPS + jj], static_position=[kk], dynamic_position=[])
                        C_[out_m_idx, out_n_idx] = val.truncf(dtype_)
        return
    
    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        CLEAN: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = (n + BLOCK_N - 1) // BLOCK_N
        hgemm_kernel._func.__name__ = KERNEL_NAME
        hgemm_kernel(C, A, B, CLEAN).launch(grid=(bm, bn, SPLIT_K), block=(BLOCK_THREADS, 1, 1), stream=stream)
    
    return launch_hgemm_kernel


def hgemm_spk_shuffle_b(x, layout=(16, 16), pack_n=1, k_steps=2):
    x_shape = x.shape
    VEC_SIZE = 16 // x.element_size()
    BN = layout[0] * pack_n
    BK = layout[1] * k_steps
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"
    x = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // VEC_SIZE, VEC_SIZE)
    x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
    x = x.view(*x_shape)
    x.is_shuffled = True
    return x
