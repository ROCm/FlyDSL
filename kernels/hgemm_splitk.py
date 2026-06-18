# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
from abc import ABC, abstractmethod
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr
from kernels.tensor_shim import GTensor, STensor, _run_compiled, get_dtype_in_kernel

fm_fast = arith.FastMathFlags.fast


SPLIT_K_SEMAPHORE_MAX_LEN = 256


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class WmmaHalfBase(ABC):
    @abstractmethod
    def __init__(self, dtype: str):
        pass

    @abstractmethod
    def __call__(self, a_frag, b_frag, c_frag):
        pass

    def init(self, a_frag, b_frag):
        zero = arith.constant_vector(0.0, T.vec(self.WMMA_C_FRAG_VALUES, T.f32))
        return self(a_frag, b_frag, zero)


class WmmaHalf_m16n16k16(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_frag_vi16 = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_frag_vi16 = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            c_frag_new = rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_frag_vi16, b_frag_vi16, c_frag, 0, 0, 0])
            return c_frag_new
        else:
            c_frag_new = rocdl.mfma_f32_16x16x16f16(
                T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
            )
            return c_frag_new


class WmmaHalf_m16n16k32(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str, use_agpr_inline: bool = False):
        self.dtype = dtype
        self.use_agpr_inline = use_agpr_inline

    def __call__(self, a_frag, b_frag, c_frag):
        res_ty = T.vec(self.WMMA_C_FRAG_VALUES, T.f32)
        if self.dtype == "bf16":
            if self.use_agpr_inline:
                a_i32x4 = vector.bitcast(T.i32x4, a_frag)
                b_i32x4 = vector.bitcast(T.i32x4, b_frag)
                return llvm.inline_asm(
                    res_ty,
                    [
                        arith._to_raw(a_i32x4),
                        arith._to_raw(b_i32x4),
                        arith._to_raw(c_frag),
                    ],
                    "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0",
                    "=a,v,v,0",
                    has_side_effects=True,
                )
            operands = [a_frag, b_frag, c_frag, 0, 0, 0]
            return rocdl.mfma_f32_16x16x32_bf16(res_ty, operands)
        else:
            operands = [a_frag, b_frag, c_frag, 0, 0, 0]
            return rocdl.mfma_f32_16x16x32_f16(res_ty, operands)

    def init(self, a_frag, b_frag):
        res_ty = T.vec(self.WMMA_C_FRAG_VALUES, T.f32)
        if self.dtype == "bf16" and self.use_agpr_inline:
            a_i32x4 = vector.bitcast(T.i32x4, a_frag)
            b_i32x4 = vector.bitcast(T.i32x4, b_frag)
            return llvm.inline_asm(
                res_ty,
                [
                    arith._to_raw(a_i32x4),
                    arith._to_raw(b_i32x4),
                ],
                "v_mfma_f32_16x16x32_bf16 $0, $1, $2, 0",
                "=a,v,v",
                has_side_effects=True,
            )
        return super().init(a_frag, b_frag)


class OnlineScheduler:
    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


@functools.lru_cache(maxsize=1024)
def compile_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    STAGES: int = 2,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    BLOCK_K_WARPS: int = 1,
    B_TO_LDS: bool = False,
    A_LDS_K32_BLOCKING: bool = False,
    B_LDS_K32_BLOCKING: bool = False,
    K32_REGISTER_PIPELINE: bool = False,
    HAS_BIAS: bool = False,
):
    assert BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS <= 16
    assert TILE_M * TILE_N * TILE_K <= 256 * 256 * 64
    if (TILE_M == 256) and (TILE_N == 256):
        assert (TILE_K == 64) and (SPLIT_K == 1) and (STAGES == 2)
    assert STAGES >= 2
    N_BLOCKS = n // TILE_N
    assert (N_BLOCKS >= 1) and (n % TILE_N == 0)
    IS_SPLIT_K = SPLIT_K > 1
    IS_SLICE_K = BLOCK_K_WARPS > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    GPU_ARCH = get_rocm_arch()
    if GPU_ARCH == "gfx942":
        WMMA_IMPL = WmmaHalf_m16n16k16(dtype)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
        ASYNC_COPY = True
    else:
        WMMA_IMPL = WmmaHalf_m16n16k32(dtype, use_agpr_inline=(GPU_ARCH == "gfx950" and dtype == "bf16"))
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
        ASYNC_COPY = True

    # Fixed parameters:
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8

    # Propagated parameters:
    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    assert BLOCK_K_LOOPS >= STAGES
    WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K
    WARP_K_STEPS = BLOCK_K // WARP_GROUP_K
    assert (BLOCK_K % WARP_GROUP_K == 0) and (WARP_K_STEPS >= 1)
    K_SLICE = BLOCK_K // BLOCK_K_WARPS
    assert K_SLICE % WARP_ATOM_K == 0
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE
    BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    # LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1) and (LDG_REG_C_COUNT >= 1)
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS parameters:
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_a_offset + AS_BYTES
    SMEM_USE = AS_BYTES
    if B_TO_LDS:
        smem_b_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
        SMEM_USE += STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
        assert ASYNC_COPY
    SMEM_USE_ = max(SMEM_USE, BLOCK_K_WARPS * BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr += SMEM_USE_ - SMEM_USE
    assert SMEM_USE_ <= SMEM_CAPACITY_MAP[GPU_ARCH]
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_REG_B_COUNT_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_WAIT_COUNT = LDG_REG_B_COUNT_AS + LDG_REG_A_COUNT_AS
    assert ((STAGES - 2) * LDG_WAIT_COUNT) < 63
    # RP1: vmcnt(N) at the loop tail = "wait until <= N VMEM loads are in flight".
    # Smaller N is safer (forces more loads to land) but slower; larger N is faster
    # but risks reading not-yet-landed data. The tile read next (next_stage K0) was
    # prefetched 2 tiles earlier; between that bfld and this read only one k-group's
    # worth of newer bfld is issued (LDG_WAIT_COUNT // 2 loads), so that is the
    # largest N that still guarantees the target batch has landed. Perf is identical
    # to the looser LDG_WAIT_COUNT (the tail vmcnt is not the bottleneck), so use the
    # tighter, provably-safe bound.
    RP1_TAIL_VMCNT = LDG_WAIT_COUNT // 2
    # A is consumed one K32 slice at a time. Store A in a K32-blocked LDS view
    # when the shape supports it. For the gfx950 256x256x64 path this makes
    # each async A chunk 64x32 instead of 32x64.
    A_LDS_K_CHUNK = WARP_ATOM_K
    A_LDS_K_GROUPS = BLOCK_K // A_LDS_K_CHUNK
    A_LDS_X_THREADS_AS = A_LDS_K_CHUNK // LDG_ASYNC_VEC_SIZE
    A_LDS_ROWS_PER_CHUNK_AS = BLOCK_THREADS // A_LDS_X_THREADS_AS
    A_LDS_ROW_GROUPS = BLOCK_M // A_LDS_ROWS_PER_CHUNK_AS if A_LDS_ROWS_PER_CHUNK_AS <= BLOCK_M else 0
    A_LDS_K_BLOCKS16 = (A_LDS_K_CHUNK * DTYPE_BYTES) // 16
    A_LDS_K32_BLOCKED = (
        (A_LDS_K32_BLOCKING or K32_REGISTER_PIPELINE)
        and B_TO_LDS
        and ASYNC_COPY
        and (A_LDS_K_CHUNK == 32)
        and (A_LDS_X_THREADS_AS > 0)
        and (BLOCK_THREADS % A_LDS_X_THREADS_AS == 0)
        and (A_LDS_ROWS_PER_CHUNK_AS <= BLOCK_M)
        and (BLOCK_M % A_LDS_ROWS_PER_CHUNK_AS == 0)
        and (BLOCK_K % A_LDS_K_CHUNK == 0)
        and (LDG_REG_A_COUNT_AS == A_LDS_ROW_GROUPS * A_LDS_K_GROUPS)
    )
    B_LDS_K_CHUNK = WARP_ATOM_K
    B_LDS_K_GROUPS = BLOCK_K // B_LDS_K_CHUNK
    B_LDS_X_THREADS_AS = B_LDS_K_CHUNK // LDG_ASYNC_VEC_SIZE
    B_LDS_ROWS_PER_CHUNK_AS = BLOCK_THREADS // B_LDS_X_THREADS_AS
    B_LDS_ROW_GROUPS = BLOCK_N // B_LDS_ROWS_PER_CHUNK_AS if B_LDS_ROWS_PER_CHUNK_AS <= BLOCK_N else 0
    B_LDS_K_BLOCKS16 = (B_LDS_K_CHUNK * DTYPE_BYTES) // 16
    B_LDS_K32_BLOCKED = (
        (B_LDS_K32_BLOCKING or K32_REGISTER_PIPELINE)
        and B_TO_LDS
        and ASYNC_COPY
        and (B_LDS_K_CHUNK == 32)
        and (B_LDS_X_THREADS_AS > 0)
        and (BLOCK_THREADS % B_LDS_X_THREADS_AS == 0)
        and (B_LDS_ROWS_PER_CHUNK_AS <= BLOCK_N)
        and (BLOCK_N % B_LDS_ROWS_PER_CHUNK_AS == 0)
        and (BLOCK_K % B_LDS_K_CHUNK == 0)
        and (LDG_REG_B_COUNT_AS == B_LDS_ROW_GROUPS * B_LDS_K_GROUPS)
    )
    REGISTER_PREFETCH_K1 = (
        K32_REGISTER_PIPELINE
        and A_LDS_K32_BLOCKED
        and B_LDS_K32_BLOCKED
        and (STAGES == 2)
        and (WARP_K_STEPS == 2)
        and (BLOCK_K_WARPS == 1)
    )
    assert not K32_REGISTER_PIPELINE or REGISTER_PREFETCH_K1
    LDG_A_X_THREADS_AS = A_LDS_X_THREADS_AS if A_LDS_K32_BLOCKED else BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_B_X_THREADS_AS = B_LDS_X_THREADS_AS if B_LDS_K32_BLOCKED else BLOCK_K // LDG_ASYNC_VEC_SIZE

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}x{STAGES}_SPK{SPLIT_K}_W{BLOCK_M_WARPS}x{BLOCK_N_WARPS}x{BLOCK_K_WARPS}_BLDS{int(B_TO_LDS)}_TN"
    KERNEL_NAME += "_AS0" if not ASYNC_COPY else "_AS1"
    if A_LDS_K32_BLOCKED:
        KERNEL_NAME += "_AK32"
    if B_LDS_K32_BLOCKED:
        KERNEL_NAME += "_BK32"
    if REGISTER_PREFETCH_K1:
        KERNEL_NAME += "_RP1"
    if HAS_BIAS:
        KERNEL_NAME += "_BIAS"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        BIAS: fx.Tensor,
        m: fx.Int32,
        semaphore: fx.Tensor,
        signal: fx.Tensor,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        if const_expr(HAS_BIAS):
            BIAS_ = GTensor(BIAS, dtype=dtype_, shape=(n,))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        as_k32_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, A_LDS_K_GROUPS, BLOCK_M, A_LDS_K_CHUNK))
        if const_expr(B_TO_LDS):
            smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,))
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
            bs_k32_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, B_LDS_K_GROUPS, BLOCK_N, B_LDS_K_CHUNK))
        smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(BLOCK_K_WARPS * BLOCK_M * BLOCK_N,))
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_K_WARPS, BLOCK_M, BLOCK_N))
        if const_expr(IS_SPLIT_K):
            semaphore_ = GTensor(semaphore, dtype=T.i32, shape=(-1,))
            signal_ = GTensor(signal, dtype=T.i32, shape=(-1,))
            signal_idx = fx.Int32(fx.block_idx.x)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        wid_mn = wid % BLOCK_MN_WARPS
        wid_k = wid // BLOCK_MN_WARPS
        w_tid = tid % WARP_SIZE

        def swizzle_for_cache_reuse(pid):
            # Do nothing currently
            return pid // N_BLOCKS, pid % N_BLOCKS

        block_m_idx, block_n_idx = swizzle_for_cache_reuse(fx.block_idx.x)
        ks_idx = fx.Index(fx.block_idx.y)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid_mn % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        warp_k_slice_base = wid_k * K_SLICE
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        B_INITIAL_READS_CONST = 4 if WARP_N_STEPS > 4 else WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        def __barrier(vmcnt=0, use_s_barrier=True):
            if const_expr(use_s_barrier):
                asm = f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier"
            else:
                asm = f"s_waitcnt vmcnt({vmcnt})"
            llvm.InlineAsmOp(None, [], asm, "", has_side_effects=True)

        def __barrier_lgkmcnt(use_s_barrier=True, vmcnt=None):
            # vmcnt is None -> LDS-only sync; otherwise also bound in-flight global
            # loads. Both RP1 half barriers carry vmcnt + lgkmcnt(0): each half has a
            # ds_read of a globally-prefetched block (needs vmcnt to ensure the bfld
            # landed) and writes LDS that another wave will overwrite next iter
            # (needs lgkmcnt(0)+s_barrier for cross-wave WAR/visibility).
            wait = "s_waitcnt lgkmcnt(0)" if vmcnt is None else f"s_waitcnt vmcnt({vmcnt}) lgkmcnt(0)"
            if const_expr(use_s_barrier):
                asm = wait + "\n\ts_barrier"
            else:
                asm = wait
            llvm.InlineAsmOp(None, [], asm, "", has_side_effects=True)

        def get_llvm_ptr(ptr, offset, dtype_bytes, ptr_type=ir.Type.parse("!llvm.ptr<1>")):
            base_ptr = fly.extract_aligned_pointer_as_index(ptr_type, ptr)
            base_ptr = llvm.PtrToIntOp(T.i64, base_ptr).result
            byte_offset = arith.index_cast(T.i64, fx.Index(offset) * fx.Index(dtype_bytes))
            llvm_ptr = llvm.AddOp(base_ptr, byte_offset, llvm.IntegerOverflowFlags(0)).result
            llvm_ptr = llvm.IntToPtrOp(ptr_type, llvm_ptr).result
            ptr_v = llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr
            return ptr_v

        def zero_c():
            # zero c if current block is the first block
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_C_X_THREADS
                    n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                    row_idx = m_offset + fx.Index(m_local_idx)
                    init_vec = zero_vec
                    if const_expr(HAS_BIAS):
                        init_vec = BIAS_.vec_load((n_offset + n_local_idx,), LDG_VEC_SIZE)
                    cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m))
                    cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        bytes_offset = C_.linear_offset((row_idx, n_offset + n_local_idx))
                        bytes_offset_i32 = arith.index_cast(T.i32, bytes_offset)
                        c_ptr = get_llvm_ptr(C, bytes_offset_i32, DTYPE_BYTES)
                        llvm.InlineAsmOp(
                            None,
                            [c_ptr, init_vec],
                            "global_store_dwordx4 $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        scf.YieldOp([])
                gpu.barrier()
                # trigger signal when zeroc is done by the first arrived block
                is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_cond_if.then_block):
                    signal_ptr = get_llvm_ptr(signal, signal_idx, 4)
                    llvm.InlineAsmOp(
                        None,
                        [signal_ptr, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1",
                        "v,v",
                        has_side_effects=True,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([])

        def split_k_barrier():
            # spin-wait until signal triggered
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                init_cur = arith.constant(0, type=T.i32)
                w = scf.WhileOp([T.i32], [init_cur])
                before = ir.Block.create_at_start(w.before, [T.i32])
                after = ir.Block.create_at_start(w.after, [T.i32])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.CmpIOp(arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)).result
                    scf.ConditionOp(need_wait, [cur])
                with ir.InsertionPoint(after):
                    signal_ptr = get_llvm_ptr(signal, signal_idx, 4)
                    data = llvm.InlineAsmOp(
                        T.i32,
                        [signal_ptr],
                        "global_load_dword $0, $1, off sc1",
                        "=v,v",
                        has_side_effects=True,
                    ).result
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([data])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()
            # clean semaphore and signal if this is the last block within split-k group
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                semaphore_ptr = get_llvm_ptr(semaphore, signal_idx, 4)
                arrive_idx = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semaphore_ptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                cond_ksl = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(arrive_idx), fx.Index(SPLIT_K - 1))
                cond_ksl_if = scf.IfOp(cond_ksl, results_=[], has_else=False)
                with ir.InsertionPoint(cond_ksl_if.then_block):
                    semaphore_[signal_idx] = arith.constant(0, type=T.i32)
                    signal_[signal_idx] = arith.constant(0, type=T.i32)
                    scf.YieldOp([])
                scf.YieldOp([])
            gpu.barrier()

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)
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

        def get_dma_copy_warp_offset():
            warp_offset = rocdl.readfirstlane(
                T.i64,
                arith.index_cast(T.i64, fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True)),
            )
            return warp_offset

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset):
            if const_expr(DMA_BYTES == 16):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds"
            elif const_expr(DMA_BYTES == 8):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx2 $1, $2, 0 offen sc0 lds"
            elif const_expr(DMA_BYTES == 4):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, 0 offen sc0 lds"
            else:
                raise NotImplementedError(f"DMA_BYTES={DMA_BYTES} not supported")
            llvm.InlineAsmOp(None, [lds_ptr, global_offset, rsrc], asm, "s,v,s", has_side_effects=True)

        def get_async_lds_ptr(lds_tensor, lds_stage, i):
            lds_offset = lds_tensor.linear_offset((fx.Index(lds_stage), 0, 0)) * DTYPE_BYTES
            lds_base = memref.extract_aligned_pointer_as_index(lds_tensor.memptr) + lds_offset
            lds_ptr_base = buffer_ops.create_llvm_ptr(arith.index_cast(T.i64, lds_base), address_space=3)
            return buffer_ops.get_element_ptr(
                lds_ptr_base,
                warp_offset,
                static_byte_offset=i * BLOCK_THREADS * DMA_BYTES,
            )

        def get_async_lds_ptr_from_offset(lds_tensor, elem_offset):
            lds_offset = elem_offset * DTYPE_BYTES
            lds_base = memref.extract_aligned_pointer_as_index(lds_tensor.memptr) + lds_offset
            lds_ptr_base = buffer_ops.create_llvm_ptr(arith.index_cast(T.i64, lds_base), address_space=3)
            return buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)

        def ldg_sts_a_async_one(k_offset, lds_stage, i):
            if const_expr(A_LDS_K32_BLOCKED):
                row_group = i % A_LDS_ROW_GROUPS
                k_group = i // A_LDS_ROW_GROUPS
                row_in_group = tid // A_LDS_X_THREADS_AS
                k_local_idx = tid % A_LDS_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                m_local_idx = row_group * A_LDS_ROWS_PER_CHUNK_AS + row_in_group
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, fx.Int32(A_LDS_K_BLOCKS16))
                col_idx = fx.Index(k_offset + k_group * A_LDS_K_CHUNK + col_in_bytes // DTYPE_BYTES)
                lds_offset = as_k32_.linear_offset((fx.Index(lds_stage), k_group, row_group * A_LDS_ROWS_PER_CHUNK_AS, 0))
                lds_ptr = get_async_lds_ptr_from_offset(as_k32_, lds_offset)
            else:
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                lds_ptr = get_async_lds_ptr(as_, lds_stage, i)
            row_idx = m_offset + fx.Index(m_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                row_idx,
                fx.Index(0),
            )
            global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            buffer_load_lds_inline(A_.rsrc, lds_ptr, global_offset)

        def ldg_sts_a_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                ldg_sts_a_async_one(k_offset, lds_stage, i)

        def ldg_sts_b_async_one(k_offset, lds_stage, i):
            if const_expr(B_LDS_K32_BLOCKED):
                row_group = i % B_LDS_ROW_GROUPS
                k_group = i // B_LDS_ROW_GROUPS
                row_in_group = tid // B_LDS_X_THREADS_AS
                k_local_idx = tid % B_LDS_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                n_local_idx = row_group * B_LDS_ROWS_PER_CHUNK_AS + row_in_group
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, fx.Int32(B_LDS_K_BLOCKS16))
                col_idx = fx.Index(k_offset + k_group * B_LDS_K_CHUNK + col_in_bytes // DTYPE_BYTES)
                lds_offset = bs_k32_.linear_offset((fx.Index(lds_stage), k_group, row_group * B_LDS_ROWS_PER_CHUNK_AS, 0))
                lds_ptr = get_async_lds_ptr_from_offset(bs_k32_, lds_offset)
            else:
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS_AS
                k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                lds_ptr = get_async_lds_ptr(bs_, lds_stage, i)
            row_idx = n_offset + fx.Index(n_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                row_idx,
                fx.Index(0),
            )
            global_offset = B_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            buffer_load_lds_inline(B_.rsrc, lds_ptr, global_offset)

        def ldg_sts_b_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                ldg_sts_b_async_one(k_offset, lds_stage, i)

        def ldg_sts_async_one(k_offset, lds_stage, i):
            if const_expr(i < LDG_REG_B_COUNT_AS):
                ldg_sts_b_async_one(k_offset, lds_stage, i)
            else:
                ldg_sts_a_async_one(k_offset, lds_stage, i - LDG_REG_B_COUNT_AS)

        def ldg_sts_async_kgroup(k_offset, lds_stage, k_group):
            assert B_LDS_K32_BLOCKED and A_LDS_K32_BLOCKED
            for i in range_constexpr(B_LDS_ROW_GROUPS):
                ldg_sts_b_async_one(k_offset, lds_stage, k_group * B_LDS_ROW_GROUPS + i)
            for i in range_constexpr(A_LDS_ROW_GROUPS):
                ldg_sts_a_async_one(k_offset, lds_stage, k_group * A_LDS_ROW_GROUPS + i)

        def ldg_matrix_b(k_offset):
            vecs = []
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                    n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                    k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    vec = B_.vec_load((n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K)
                    vecs.append(vec)
            return vecs

        def spread_counts(numer: int, denom: int):
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + denom - 1) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def load_a_frag_from(lds_stage, warp_atom_k_idx, ii):
            s = lds_stage if isinstance(lds_stage, fx.Index) else fx.Index(lds_stage)
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            row = warp_atom_m_idx + ldmatrix_a_m_idx
            if const_expr(A_LDS_K32_BLOCKED):
                k_group = warp_atom_k_idx // A_LDS_K_CHUNK
                k_in_group = warp_atom_k_idx % A_LDS_K_CHUNK + ldmatrix_a_k_vec_idx
                col_in_bytes = k_in_group * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, fx.Int32(A_LDS_K_BLOCKS16))
                return as_k32_.vec_load(
                    (s, k_group, row, col_in_bytes // DTYPE_BYTES),
                    WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                )
            col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
            return as_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K)

        def load_b_frag_from(lds_stage, warp_atom_k_idx, jj):
            s = lds_stage if isinstance(lds_stage, fx.Index) else fx.Index(lds_stage)
            warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
            row = warp_atom_n_idx + ldmatrix_b_n_idx
            if const_expr(B_LDS_K32_BLOCKED):
                k_group = warp_atom_k_idx // B_LDS_K_CHUNK
                k_in_group = warp_atom_k_idx % B_LDS_K_CHUNK + ldmatrix_b_k_vec_idx
                col_in_bytes = k_in_group * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, fx.Int32(B_LDS_K_BLOCKS16))
                return bs_k32_.vec_load(
                    (s, k_group, row, col_in_bytes // DTYPE_BYTES),
                    WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                )
            col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
            return bs_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K)

        def load_initial_frags_from(lds_stage):
            b_frags = []
            for jj in range_constexpr(B_INITIAL_READS_CONST):
                b_frags.append(load_b_frag_from(lds_stage, warp_k_slice_base, jj))
            a_frag = load_a_frag_from(lds_stage, warp_k_slice_base, 0)
            return a_frag, b_frags

        def ldmatrix_compute_tile_streaming(
            lds_stage,
            c_frags,
            initial_b_frags=None,
            prefetched_initial_a_frag=None,
            prefetched_initial_b_frags=None,
            prefetch_k_offset=None,
            prefetch_lds_stage=0,
            init_zero=False,
        ):
            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            async_load_idx = 0
            row_group = 0
            ldg_total = LDG_REG_B_COUNT_AS + LDG_REG_A_COUNT_AS
            # Finish the next-tile VMEM earlier so the loop-top vmcnt wait has
            # more MFMA distance to hide global-to-LDS latency.
            prefetch_rows = (ldg_total + 1) // 2
            if const_expr(prefetch_rows > WARP_K_STEPS * WARP_M_STEPS):
                prefetch_rows = WARP_K_STEPS * WARP_M_STEPS
            prefetch_split_1 = WARP_N_STEPS // 2
            a_prefetch_jj = 1 if const_expr(WARP_N_STEPS > 1) else 0
            b_initial_reads = B_INITIAL_READS_CONST

            def load_a_frag(warp_atom_k_idx, ii):
                return load_a_frag_from(s, warp_atom_k_idx, ii)

            def load_b_frag(warp_atom_k_idx, jj):
                return load_b_frag_from(s, warp_atom_k_idx, jj)

            def do_mfma(ii, jj, a_frag, b_frag):
                if const_expr(MFMA_PER_WARP_K == 2):
                    # split a
                    a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                    a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    a_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64]))
                    a_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64]))
                    # split b
                    b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                    b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                    b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                    b_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64]))
                    b_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64]))
                    c_idx = ii * WARP_N_STEPS + jj
                    acc_in = c_frags_new[c_idx]
                    acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                    c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                elif const_expr(MFMA_PER_WARP_K == 1):
                    c_idx = ii * WARP_N_STEPS + jj
                    c_frags_new[c_idx] = WMMA_IMPL(a_frag, b_frag, c_frags_new[c_idx])
                else:
                    raise NotImplementedError(f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported")

            next_k_a_frag = None
            next_k_b_frags = [None] * WARP_N_STEPS
            next_k_prefetch_ii = WARP_M_STEPS - 2 if const_expr(WARP_M_STEPS > 1) else 0
            next_k_prefetch_a_jj = b_initial_reads if const_expr(b_initial_reads < WARP_N_STEPS) else WARP_N_STEPS - 1

            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                if const_expr(initial_b_frags is None and kk == 0 and prefetched_initial_b_frags is not None):
                    b_frags = [0] * WARP_N_STEPS
                    for jj in range_constexpr(b_initial_reads):
                        b_frags[jj] = prefetched_initial_b_frags[jj]
                elif const_expr(initial_b_frags is None and kk > 0):
                    b_frags = [0] * WARP_N_STEPS
                    for jj in range_constexpr(b_initial_reads):
                        b_frags[jj] = next_k_b_frags[jj]
                elif const_expr(initial_b_frags is None):
                    b_frags = [0] * WARP_N_STEPS
                    for jj in range_constexpr(b_initial_reads):
                        b_frags[jj] = load_b_frag(warp_atom_k_idx, jj)
                else:
                    b_frags = [initial_b_frags[i] for i in range_constexpr(kk * WARP_N_STEPS, (kk + 1) * WARP_N_STEPS)]
                if const_expr(initial_b_frags is None and kk == 0 and prefetched_initial_a_frag is not None):
                    a_frag = prefetched_initial_a_frag
                elif const_expr(initial_b_frags is None and kk > 0):
                    a_frag = next_k_a_frag
                else:
                    a_frag = load_a_frag(warp_atom_k_idx, 0)
                next_k_a_frag = None
                next_k_b_frags = [None] * WARP_N_STEPS
                rocdl.sched_barrier(0)
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag_next = a_frag
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[jj]
                        if const_expr(init_zero and kk == 0):
                            c_frags_new[ii * WARP_N_STEPS + jj] = WMMA_IMPL.init(a_frag, b_frag)
                        else:
                            do_mfma(ii, jj, a_frag, b_frag)
                        if const_expr(
                            initial_b_frags is None
                            and ii == 0
                            and jj + b_initial_reads < WARP_N_STEPS
                        ):
                            b_frags[jj + b_initial_reads] = load_b_frag(warp_atom_k_idx, jj + b_initial_reads)
                        if const_expr(ii + 1 < WARP_M_STEPS and jj == a_prefetch_jj):
                            a_frag_next = load_a_frag(warp_atom_k_idx, ii + 1)
                        if const_expr(
                            initial_b_frags is None
                            and kk + 1 < WARP_K_STEPS
                            and ii == next_k_prefetch_ii
                            and jj == 0
                        ):
                            next_k_b_frags = [None] * WARP_N_STEPS
                        if const_expr(
                            initial_b_frags is None
                            and kk + 1 < WARP_K_STEPS
                            and ii == next_k_prefetch_ii
                            and jj < b_initial_reads
                        ):
                            next_k_warp_atom_k_idx = warp_k_slice_base + (kk + 1) * WARP_ATOM_K
                            next_k_b_frags[jj] = load_b_frag(next_k_warp_atom_k_idx, jj)
                        if const_expr(
                            initial_b_frags is None
                            and kk + 1 < WARP_K_STEPS
                            and ii == next_k_prefetch_ii
                            and jj == next_k_prefetch_a_jj
                        ):
                            next_k_warp_atom_k_idx = warp_k_slice_base + (kk + 1) * WARP_ATOM_K
                            next_k_a_frag = load_a_frag(next_k_warp_atom_k_idx, 0)
                        if const_expr(
                            prefetch_k_offset is not None
                            and row_group < prefetch_rows
                            and async_load_idx < ldg_total
                            and (jj == 0 or jj == prefetch_split_1)
                        ):
                            ldg_sts_async_one(prefetch_k_offset, prefetch_lds_stage, async_load_idx)
                            async_load_idx += 1
                    a_frag = a_frag_next
                    row_group += 1
            return c_frags_new

        def ldmatrix_compute_tile_register_prefetch(
            lds_stage,
            c_frags,
            prefetched_initial_a_frag=None,
            prefetched_initial_b_frags=None,
            prefetch_k_offset=None,
            prefetch_lds_stage=0,
            init_zero=False,
            k0_prefetch_k_offset=None,
            k0_prefetch_lds_stage=None,
            next_k0_read_stage=None,
        ):
            # K32 sub-buffer ping-pong (see fp8-4wave-8buffer design notes):
            #   h1: bfld tile_{i+1} K1 -> prefetch_lds_stage (lead 1)
            #   h2: bfld tile_{i+2} K0 -> k0_prefetch_lds_stage = current_stage
            #       (lead 2; overwrites tile_i K0 which was consumed in h1).
            # If k0_prefetch_* is None the K0 group is not prefetched (epilogue).
            assert REGISTER_PREFETCH_K1
            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            k0_idx = warp_k_slice_base
            k1_idx = warp_k_slice_base + WARP_ATOM_K

            def load_a_frag(warp_atom_k_idx, ii):
                return load_a_frag_from(s, warp_atom_k_idx, ii)

            def load_b_frag(warp_atom_k_idx, jj):
                return load_b_frag_from(s, warp_atom_k_idx, jj)

            def load_a_slice(warp_atom_k_idx, initial_a=None):
                # initial_a may be a single frag (legacy) or a full list (full carry).
                if const_expr(isinstance(initial_a, list)):
                    return list(initial_a)
                out = [0] * WARP_M_STEPS
                for ii in range_constexpr(WARP_M_STEPS):
                    if const_expr(initial_a is not None and ii == 0):
                        out[ii] = initial_a
                    else:
                        out[ii] = load_a_frag(warp_atom_k_idx, ii)
                return out

            def load_b_slice(warp_atom_k_idx, initial_bs=None):
                # initial_bs may be a partial list (B_INITIAL_READS) or a full list.
                if const_expr(isinstance(initial_bs, list) and len(initial_bs) == WARP_N_STEPS):
                    return list(initial_bs)
                out = [0] * WARP_N_STEPS
                for jj in range_constexpr(WARP_N_STEPS):
                    if const_expr(initial_bs is not None and jj < B_INITIAL_READS_CONST):
                        out[jj] = initial_bs[jj]
                    else:
                        out[jj] = load_b_frag(warp_atom_k_idx, jj)
                return out

            def do_mfma(ii, jj, a_frag, b_frag):
                if const_expr(MFMA_PER_WARP_K == 2):
                    a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                    a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    a_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64]))
                    a_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64]))
                    b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                    b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                    b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                    b_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64]))
                    b_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64]))
                    c_idx = ii * WARP_N_STEPS + jj
                    acc_mid = WMMA_IMPL(a_v0, b_v0, c_frags_new[c_idx])
                    c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                elif const_expr(MFMA_PER_WARP_K == 1):
                    c_idx = ii * WARP_N_STEPS + jj
                    c_frags_new[c_idx] = WMMA_IMPL(a_frag, b_frag, c_frags_new[c_idx])
                else:
                    raise NotImplementedError(f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported")

            # Compute one K32 slice (64 MFMAs) while dripping the next slice's
            # operand ds_reads into the MFMA shadows. The operands consumed here
            # (a_frags/b_frags) are fully ready on entry (carried in registers from
            # the previous half); the reads issued here (read_k_idx into out_a/out_b)
            # are for the NEXT half and only consumed after this half's lgkmcnt(0),
            # so their ~64-cyc latency hides fully under the 64 MFMAs. ds_reads are
            # spread front-loaded-but-interleaved so all land before the half's
            # trailing lgkmcnt(0).
            def compute_slice_interleaved(
                a_frags, b_frags, read_k_idx, read_stage=None, zero_acc=False, bfld_tasks=None
            ):
                out_a = [0] * WARP_M_STEPS
                out_b = [0] * WARP_N_STEPS
                rstage = s if read_stage is None else fx.Index(read_stage)
                reads = []
                if const_expr(read_k_idx is not None):
                    for ii in range_constexpr(WARP_M_STEPS):
                        reads.append(("a", ii))
                    for jj in range_constexpr(WARP_N_STEPS):
                        reads.append(("b", jj))
                n_reads = len(reads)
                # Distribute the reads across the MFMA grid; keep them in the first
                # ~3/4 of the MFMAs so they all complete before the closing lgkmcnt.
                read_at = spread_counts(n_reads, WARP_M_STEPS * WARP_N_STEPS)
                # bfld (global->LDS) prefetch is also dripped one-per-MFMA into the
                # co-issue shadows instead of bursting at the region head.
                tasks = bfld_tasks if bfld_tasks is not None else []
                n_tasks = len(tasks)
                bfld_at = spread_counts(n_tasks, WARP_M_STEPS * WARP_N_STEPS)
                read_cursor = 0
                bfld_cursor = 0
                for ii in range_constexpr(WARP_M_STEPS):
                    for jj in range_constexpr(WARP_N_STEPS):
                        if const_expr(zero_acc):
                            c_frags_new[ii * WARP_N_STEPS + jj] = WMMA_IMPL.init(a_frags[ii], b_frags[jj])
                        else:
                            do_mfma(ii, jj, a_frags[ii], b_frags[jj])
                        step = ii * WARP_N_STEPS + jj
                        for _ in range_constexpr(bfld_at[step]):
                            tasks[bfld_cursor]()
                            bfld_cursor += 1
                        for _ in range_constexpr(read_at[step]):
                            kind, idx = reads[read_cursor]
                            if const_expr(kind == "a"):
                                out_a[idx] = load_a_frag_from(rstage, read_k_idx, idx)
                            else:
                                out_b[idx] = load_b_frag_from(rstage, read_k_idx, idx)
                            read_cursor += 1
                return out_a, out_b

            # K0 operands of this tile are fully carried in on entry (a0/b0 full
            # group); load_*_slice just unpacks the carried frags (no LDS read when
            # the full group is provided).
            a0_frags = load_a_slice(k0_idx, prefetched_initial_a_frag)
            b0_frags = load_b_slice(k0_idx, prefetched_initial_b_frags)

            def make_bfld_tasks(k_offset, lds_stage, k_group):
                # One callable per buffer_load in this k-group (B rows then A rows),
                # to be dripped into the MFMA loop instead of bursting up front.
                tasks = []
                for i in range_constexpr(B_LDS_ROW_GROUPS):
                    tasks.append(lambda i=i: ldg_sts_b_async_one(k_offset, lds_stage, k_group * B_LDS_ROW_GROUPS + i))
                for i in range_constexpr(A_LDS_ROW_GROUPS):
                    tasks.append(lambda i=i: ldg_sts_a_async_one(k_offset, lds_stage, k_group * A_LDS_ROW_GROUPS + i))
                return tasks

            # half-1: bfld next tile's K1 (lead 1) dripped into the MFMAs; compute K0
            # while reading this tile's K1 operands for half-2.
            h1_bfld = make_bfld_tasks(prefetch_k_offset, prefetch_lds_stage, 1) if prefetch_k_offset is not None else None
            a1_frags, b1_frags = compute_slice_interleaved(
                a0_frags, b0_frags, k1_idx, zero_acc=init_zero, bfld_tasks=h1_bfld
            )

            __barrier_lgkmcnt(vmcnt=RP1_TAIL_VMCNT)

            # half-2: bfld the tile-after-next's K0 (lead 2) dripped into the MFMAs;
            # compute K1 while reading the NEXT tile's K0 group (from
            # next_k0_read_stage) to carry into the next iteration.
            h2_bfld = (
                make_bfld_tasks(k0_prefetch_k_offset, k0_prefetch_lds_stage, 0)
                if k0_prefetch_k_offset is not None
                else None
            )
            next_a0, next_b0 = compute_slice_interleaved(
                a1_frags, b1_frags, k0_idx if next_k0_read_stage is not None else None,
                read_stage=next_k0_read_stage, bfld_tasks=h2_bfld,
            )
            return c_frags_new, next_a0, next_b0

        warp_offset = get_dma_copy_warp_offset()

        if const_expr(IS_SPLIT_K):
            zero_c()

        if const_expr(B_TO_LDS):

            for s in range_constexpr(STAGES - 1):
                ldg_sts_b_async(ks_begin + s * BLOCK_K, s)
                ldg_sts_a_async(ks_begin + s * BLOCK_K, s)
            if const_expr(REGISTER_PREFETCH_K1):
                # K0/K1 are prefetched on separate stage schedules: tile1's K0 is
                # never produced by an earlier compute (no tile -1), so seed it here
                # into stage 1. tile1's K1 is produced by tile0's compute (h1).
                ldg_sts_async_kgroup(ks_begin + BLOCK_K, arith.constant(1 % STAGES, index=True), 0)
            rocdl.sched_barrier(0)

            def hot_loop_scheduler(prefetched_initial=False, tail_prefetch_initial=False):
                LDG_TOTAL = LDG_REG_B_COUNT_AS + LDG_REG_A_COUNT_AS
                MFMA_PER_ROW = WARP_N_STEPS * MFMA_PER_WARP_K
                PREFETCH_ROWS = (LDG_TOTAL + 1) // 2
                if const_expr(PREFETCH_ROWS > WARP_K_STEPS * WARP_M_STEPS):
                    PREFETCH_ROWS = WARP_K_STEPS * WARP_M_STEPS
                PREFETCH_SPLIT_1 = WARP_N_STEPS // 2
                A_PREFETCH_JJ = 1 if const_expr(WARP_N_STEPS > 1) else 0
                B_INITIAL_READS = 4 if const_expr(WARP_N_STEPS > 4) else WARP_N_STEPS
                NEXT_K_PREFETCH_II = WARP_M_STEPS - 2 if const_expr(WARP_M_STEPS > 1) else 0
                NEXT_K_PREFETCH_A_JJ = B_INITIAL_READS if const_expr(B_INITIAL_READS < WARP_N_STEPS) else WARP_N_STEPS - 1
                row_group = 0
                for ki in range_constexpr(WARP_K_STEPS):
                    if const_expr(ki == 0 and not prefetched_initial):
                        for i in range_constexpr(B_INITIAL_READS):
                            rocdl.sched_dsrd(1)  # lds_matrix_b current
                        rocdl.sched_dsrd(1)  # first lds_matrix_a current
                    for i in range_constexpr(WARP_M_STEPS):
                        for j in range_constexpr(WARP_N_STEPS):
                            rocdl.sched_mfma(MFMA_PER_WARP_K)
                            if const_expr(i == 0 and j + B_INITIAL_READS < WARP_N_STEPS):
                                rocdl.sched_dsrd(1)  # next lds_matrix_b current
                            if const_expr(i + 1 < WARP_M_STEPS and j == A_PREFETCH_JJ):
                                rocdl.sched_dsrd(1)  # next lds_matrix_a current
                            if const_expr(ki + 1 < WARP_K_STEPS and i == NEXT_K_PREFETCH_II and j < B_INITIAL_READS):
                                rocdl.sched_dsrd(1)  # first lds_matrix_b next k-slice
                            if const_expr(ki + 1 < WARP_K_STEPS and i == NEXT_K_PREFETCH_II and j == NEXT_K_PREFETCH_A_JJ):
                                rocdl.sched_dsrd(1)  # first lds_matrix_a next k-slice
                            if const_expr(
                                row_group < PREFETCH_ROWS
                                and (j == 0 or j == PREFETCH_SPLIT_1)
                            ):
                                rocdl.sched_vmem(1)
                        row_group += 1
                rocdl.sched_barrier(0)
                if const_expr(tail_prefetch_initial):
                    for i in range_constexpr(B_INITIAL_READS):
                        rocdl.sched_dsrd(1)  # first lds_matrix_b next BLOCK_K tile
                    rocdl.sched_dsrd(1)  # first lds_matrix_a next BLOCK_K tile

            def hot_loop_scheduler_register_prefetch(prefetched_initial=False, tail_prefetch_initial=False):
                PHASE_VMEM = LDG_REG_B_COUNT_AS // 2 + LDG_REG_A_COUNT_AS // 2
                PHASE_MFMA = WARP_M_STEPS * WARP_N_STEPS
                PHASE0_DSRD = WARP_M_STEPS + WARP_N_STEPS
                if const_expr(prefetched_initial):
                    PHASE0_DSRD += (WARP_M_STEPS - 1) + (WARP_N_STEPS - B_INITIAL_READS_CONST)
                else:
                    PHASE0_DSRD += WARP_M_STEPS + WARP_N_STEPS

                # Match the manual interleave in the compute function: region-1
                # drips PHASE0_DSRD ds_reads across its MFMAs, region-2 only has vmem.
                dsrd_spread = spread_counts(PHASE0_DSRD, PHASE_MFMA)
                vmem1_spread = spread_counts(PHASE_VMEM, PHASE_MFMA)
                for i in range_constexpr(PHASE_MFMA):
                    rocdl.sched_mfma(MFMA_PER_WARP_K)
                    for _ in range_constexpr(dsrd_spread[i]):
                        rocdl.sched_dsrd(1)
                    for _ in range_constexpr(vmem1_spread[i]):
                        rocdl.sched_vmem(1)
                rocdl.sched_barrier(0)
                vmem2_spread = spread_counts(PHASE_VMEM, PHASE_MFMA)
                for i in range_constexpr(PHASE_MFMA):
                    rocdl.sched_mfma(MFMA_PER_WARP_K)
                    for _ in range_constexpr(vmem2_spread[i]):
                        rocdl.sched_vmem(1)
                rocdl.sched_barrier(0)
                if const_expr(tail_prefetch_initial):
                    for i in range_constexpr(B_INITIAL_READS_CONST):
                        rocdl.sched_dsrd(1)
                    rocdl.sched_dsrd(1)

            __barrier((STAGES - 2) * LDG_WAIT_COUNT)
            if const_expr(REGISTER_PREFETCH_K1):
                # tile0 compute: h1 bfld tile1.K1 -> stage1; h2 bfld tile2.K0 -> stage0.
                # half-2 reads tile1's K0 (stage1) into the carry for the next iter.
                c_frags, carry_a0, carry_b0 = ldmatrix_compute_tile_register_prefetch(
                    arith.constant(0, index=True),
                    c_frags,
                    prefetch_k_offset=ks_begin + BLOCK_K,
                    prefetch_lds_stage=arith.constant(1 % STAGES, index=True),
                    init_zero=True,
                    k0_prefetch_k_offset=ks_begin + 2 * BLOCK_K,
                    k0_prefetch_lds_stage=arith.constant(0, index=True),
                    next_k0_read_stage=arith.constant(1 % STAGES, index=True),
                )
                hot_loop_scheduler_register_prefetch(tail_prefetch_initial=True)
                next_stage_init = arith.constant(1 % STAGES, index=True)
                __barrier_lgkmcnt(vmcnt=RP1_TAIL_VMCNT)

                init_state = (
                    [ks_begin + fx.Int32(BLOCK_K), next_stage_init]
                    + c_frags
                    + carry_a0
                    + carry_b0
                )
                for bki, state in range(1, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state):
                    k_offset = state[0]
                    current_stage = fx.Index(state[1])
                    c_frags = state[2 : 2 + C_FRAGS_LEN]
                    carry_a0 = state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + WARP_M_STEPS]
                    carry_b0 = state[2 + C_FRAGS_LEN + WARP_M_STEPS : 2 + C_FRAGS_LEN + WARP_M_STEPS + WARP_N_STEPS]
                    next_stage = (current_stage + 1) % STAGES
                    prefetch_k_offset = k_offset + (STAGES - 1) * BLOCK_K
                    # lead-2 K0 offset runs past the last tile in the final loop
                    # iterations; clamp it so the (unused) prefetch stays in-bounds.
                    k0_pf_offset = k_offset + fx.Int32(STAGES * BLOCK_K)
                    k0_pf_max = ks_begin + fx.Int32((BLOCK_K_LOOPS - 1) * BLOCK_K)
                    k0_pf_offset = arith.select(
                        arith.cmpi(arith.CmpIPredicate.slt, k0_pf_offset, k0_pf_max),
                        k0_pf_offset,
                        k0_pf_max,
                    )
                    c_frags_new, carry_a0_next, carry_b0_next = ldmatrix_compute_tile_register_prefetch(
                        current_stage,
                        c_frags,
                        prefetched_initial_a_frag=carry_a0,
                        prefetched_initial_b_frags=carry_b0,
                        prefetch_k_offset=prefetch_k_offset,
                        prefetch_lds_stage=next_stage,
                        k0_prefetch_k_offset=k0_pf_offset,
                        k0_prefetch_lds_stage=current_stage,
                        next_k0_read_stage=next_stage,
                    )
                    k_offset_next = k_offset + fx.Int32(BLOCK_K)
                    hot_loop_scheduler_register_prefetch(prefetched_initial=True, tail_prefetch_initial=True)
                    __barrier_lgkmcnt(vmcnt=RP1_TAIL_VMCNT)
                    results = (
                        yield [k_offset_next, next_stage] + c_frags_new + carry_a0_next + carry_b0_next
                    )
                current_stage = fx.Index(results[1])
                c_frags = results[2 : 2 + C_FRAGS_LEN]
                carry_a0 = results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + WARP_M_STEPS]
                carry_b0 = results[2 + C_FRAGS_LEN + WARP_M_STEPS : 2 + C_FRAGS_LEN + WARP_M_STEPS + WARP_N_STEPS]
                for s in range_constexpr(0, STAGES - 1):
                    c_frags, carry_a0, carry_b0 = ldmatrix_compute_tile_register_prefetch(
                        current_stage,
                        c_frags,
                        prefetched_initial_a_frag=carry_a0,
                        prefetched_initial_b_frags=carry_b0,
                    )
                    rocdl.sched_barrier(0)
                    current_stage = (current_stage + 1) % STAGES

            if const_expr(not REGISTER_PREFETCH_K1):
                c_frags = ldmatrix_compute_tile_streaming(
                    arith.constant(0, index=True),
                    c_frags,
                    prefetch_k_offset=ks_begin + BLOCK_K,
                    prefetch_lds_stage=arith.constant((STAGES - 1) % STAGES, index=True),
                    init_zero=True,
                )
                hot_loop_scheduler(tail_prefetch_initial=True)
                next_stage_init = arith.constant(1 % STAGES, index=True)
                __barrier((STAGES - 2) * LDG_WAIT_COUNT)
                prefetched_tile_a_frag, prefetched_tile_b_frags = load_initial_frags_from(next_stage_init)

                init_state = (
                    [ks_begin + fx.Int32(BLOCK_K), next_stage_init]
                    + c_frags
                    + [prefetched_tile_a_frag]
                    + prefetched_tile_b_frags
                )
                for bki, state in range(1, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state):
                    k_offset = state[0]
                    current_stage = fx.Index(state[1])
                    c_frags = state[2 : 2 + C_FRAGS_LEN]
                    prefetched_tile_a_frag = state[2 + C_FRAGS_LEN]
                    prefetched_tile_b_frags = state[3 + C_FRAGS_LEN : 3 + C_FRAGS_LEN + B_INITIAL_READS_CONST]
                    next_stage = (current_stage + 1) % STAGES
                    write_stage = (current_stage + STAGES - 1) % STAGES
                    prefetch_k_offset = k_offset + (STAGES - 1) * BLOCK_K
                    c_frags_new = ldmatrix_compute_tile_streaming(
                        current_stage,
                        c_frags,
                        prefetched_initial_a_frag=prefetched_tile_a_frag,
                        prefetched_initial_b_frags=prefetched_tile_b_frags,
                        prefetch_k_offset=prefetch_k_offset,
                        prefetch_lds_stage=write_stage,
                    )
                    k_offset_next = k_offset + fx.Int32(BLOCK_K)
                    hot_loop_scheduler(prefetched_initial=True, tail_prefetch_initial=True)
                    __barrier((STAGES - 2) * LDG_WAIT_COUNT)
                    prefetched_tile_a_frag_next, prefetched_tile_b_frags_next = load_initial_frags_from(next_stage)
                    results = (
                        yield [k_offset_next, next_stage]
                        + c_frags_new
                        + [prefetched_tile_a_frag_next]
                        + prefetched_tile_b_frags_next
                    )
                current_stage = fx.Index(results[1])
                c_frags = results[2 : 2 + C_FRAGS_LEN]
                prefetched_tile_a_frag = results[2 + C_FRAGS_LEN]
                prefetched_tile_b_frags = results[3 + C_FRAGS_LEN : 3 + C_FRAGS_LEN + B_INITIAL_READS_CONST]
                for s in range_constexpr(0, STAGES - 1):
                    c_frags = ldmatrix_compute_tile_streaming(
                        current_stage,
                        c_frags,
                        prefetched_initial_a_frag=prefetched_tile_a_frag,
                        prefetched_initial_b_frags=prefetched_tile_b_frags,
                    )
                    current_stage = (current_stage + 1) % STAGES

        else:

            assert STAGES == 2
            sts_a(ldg_a(ks_begin), 0)
            b_frags_next = ldg_matrix_b(ks_begin)
            rocdl.sched_barrier(0)
            __barrier()

            def hot_loop_scheduler():
                LDG_REG_A_COUNT_ = LDG_REG_A_COUNT_AS if const_expr(ASYNC_COPY) else LDG_REG_A_COUNT
                LDG_TOTAL = LDG_REG_A_COUNT_ + WARP_K_STEPS * WARP_N_STEPS
                # ================ Ordered ================
                for i in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(1)
                for ki in range_constexpr(WARP_K_STEPS):
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_dsrd(1)
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags + b_frags_next
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN :]
                if const_expr(ASYNC_COPY):
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                c_frags_new = ldmatrix_compute_tile_streaming(current_stage, c_frags, b_frags)
                if const_expr(not ASYNC_COPY):
                    sts_a(a_regs_next, next_stage)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                hot_loop_scheduler()
                __barrier()
                results = yield [k_offset, next_stage] + c_frags_new + b_frags_next
            current_stage = fx.Index(results[1])
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN :]
            c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags, b_frags)

        # write to lds
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(c_frags[ii * WARP_N_STEPS + jj], static_position=[kk], dynamic_position=[])
                    val = val.truncf(dtype_)
                    if const_expr(IS_SLICE_K):
                        cs_[wid_k, lds_m_idx, lds_n_idx] = val
                    else:
                        cs_[0, lds_m_idx, lds_n_idx] = val

        # write back to global
        if const_expr(IS_SPLIT_K):
            split_k_barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                n_global_idx = n_offset + n_local_idx
                cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m))
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    pk_val = cs_.vec_load((0, m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        pk_val += cs_.vec_load((ksi, m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    linear_offset_c = C_.linear_offset((m_global_idx, n_global_idx))
                    # split to vec2s
                    vec2_ty = T.vec(2, dtype_)
                    for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                        e0 = vector.extract(pk_val, static_position=[vec_idx * 2], dynamic_position=[])
                        e1 = vector.extract(pk_val, static_position=[vec_idx * 2 + 1], dynamic_position=[])
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_v = pair._value if const_expr(hasattr(pair, "_value")) else pair
                        pair_ptr_v = get_llvm_ptr(C, fx.Int32(linear_offset_c + vec_idx * 2), DTYPE_BYTES)
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v,
                            pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                    scf.YieldOp([])
        else:
            gpu.barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m))
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    vec = cs_.vec_load((0, m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        vec += cs_.vec_load((ksi, m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    if const_expr(HAS_BIAS):
                        bias_vec = BIAS_.vec_load((n_offset + n_local_idx,), LDG_VEC_SIZE)
                        vec = vec + bias_vec
                    C_.vec_store((m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE)
                    scf.YieldOp([])
        return

    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        BIAS: fx.Tensor,
        m: fx.Int32,
        semaphore: fx.Tensor,
        signal: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bm = (m + BLOCK_M - 1) // BLOCK_M
        hgemm_kernel._func.__name__ = KERNEL_NAME
        hgemm_kernel(C, A, B, BIAS, m, semaphore, signal).launch(
            grid=(bm * N_BLOCKS, SPLIT_K, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch_hgemm_kernel


def get_default_kwargs(m, n, k):
    kwargs = {
        "TILE_M": 256,
        "TILE_N": 256,
        "TILE_K": 64,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_M_WARPS": 2,
        "BLOCK_N_WARPS": 4,
        "BLOCK_K_WARPS": 1,
        "B_TO_LDS": True,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs["TILE_M"] = 128
        kwargs["TILE_N"] = 128
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 4
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
    elif m <= 32 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 16
        kwargs["BLOCK_M_WARPS"] = 2
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
    elif m <= 32 and n == 7168 and k == 2048:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 1
        kwargs["BLOCK_K_WARPS"] = 2
    elif m <= 32 and n == 384 and k == 16384:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 256
        kwargs["STAGES"] = 3
        kwargs["SPLIT_K"] = 16
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
    elif m <= 16 and n == 5120 and k == 2880:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 3
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
    elif m <= 32 and n == 2880 and k == 2048:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 2
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
    return kwargs


selections = {
    "TILE_M": [16, 32, 48, 64, 96, 128, 256],
    "TILE_N": [64, 128, 256],
    "TILE_K": [64, 128, 256],
    "STAGES": [2, 3, 4, 5],
    "SPLIT_K": [i for i in range(1, 17)],
    "BLOCK_M_WARPS": [1, 2, 4],
    "BLOCK_N_WARPS": [1, 2, 4],
    "BLOCK_K_WARPS": [1, 2, 4],
}


@functools.lru_cache(maxsize=128)
def get_semaphore(stream, device):
    semaphore = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    signal = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    return semaphore, signal


def hgemm_splitk_(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hgemm_kwargs: dict = {},
    stream: torch.cuda.Stream = torch.cuda.current_stream(),
):
    global SPLIT_K_SEMAPHORE_MAX_LEN
    device = a.device
    semaphore, signal = get_semaphore(stream, device)
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m
    kwargs = get_default_kwargs(m, n, k)
    kwargs.update(hgemm_kwargs)
    kwargs["HAS_BIAS"] = False if bias is None else True
    if a.dtype == torch.half:
        exe = compile_hgemm_kernel("f16", n, k, **kwargs)
    elif a.dtype == torch.bfloat16:
        exe = compile_hgemm_kernel("bf16", n, k, **kwargs)
    else:
        raise NotImplementedError()
    if kwargs["SPLIT_K"] > 1:
        bm = (m + kwargs["TILE_M"] - 1) // kwargs["TILE_M"]
        bn = n // kwargs["TILE_N"]
        assert bm * bn <= SPLIT_K_SEMAPHORE_MAX_LEN
    bias_tensor = a if bias is None else bias
    _run_compiled(exe, c, a, b, bias_tensor, m, semaphore, signal, stream)
