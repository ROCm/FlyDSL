#!/usr/bin/env python3
"""WMMA GEMM kernel for RDNA4 (gfx12xx, wave32).

Computes C[M,N] = A[M,K] @ B[K,N] using WMMA 16x16x16 instructions.
Supports f16 inputs with f32 accumulation, output in f16 or f32.

Architecture:
  - Each workgroup handles a BLOCK_M x BLOCK_N output tile
  - Multiple WMMA 16x16x16 tiles per workgroup (WAVES_M x WAVES_N arrangement)
  - K dimension is tiled into BLOCK_K chunks loaded into LDS
  - Inner loop executes WMMA over K tiles

WMMA data layout for wave32 (verified on gfx1201):
  A operand: "row-of-cols" -- lane t loads A[t%16][(t/16)*8 + i], i=0..7
  B operand: "col-of-rows" -- lane t loads B[(t/16)*8 + i][t%16], i=0..7
  D result:  "col-of-rows" -- lane t holds D[(t/16)*8 + i][t%16], i=0..7
"""

import sys
import os

from flydsl.dialects.ext import flir, arith, memref, vector, rocdl, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as Textra


# =============================================================================
# Kernel configuration
# =============================================================================

# WMMA tile dimensions (fixed by hardware)
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

# Number of WMMA tiles per workgroup
WAVES_M = 2  # 2 tiles in M direction
WAVES_N = 2  # 2 tiles in N direction

# Derived block dimensions
BLOCK_M = WMMA_M * WAVES_M  # 32
BLOCK_N = WMMA_N * WAVES_N  # 32
BLOCK_K = WMMA_K  # 16 (one WMMA K step per LDS load)

# Threads per workgroup: WAVES_M * WAVES_N waves, 32 threads each
NUM_WAVES = WAVES_M * WAVES_N  # 4
THREADS_PER_BLOCK = NUM_WAVES * 32  # 128


def create_wmma_gemm_module(M: int, N: int, K: int, out_dtype="f32"):
    """Create a WMMA GEMM module.

    Args:
        M, N, K: matrix dimensions (must be multiples of BLOCK_M, BLOCK_N, BLOCK_K)
        out_dtype: "f32" or "f16" for output type

    Returns:
        FlyDSL MlirModule ready for compilation
    """
    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0, f"M={M} must be multiple of BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be multiple of BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be multiple of BLOCK_K={BLOCK_K}"

    num_k_tiles = K // BLOCK_K

    # Shared memory layout:
    # A tile: BLOCK_M x BLOCK_K = 32 x 16 f16 = 1024 bytes
    # B tile: BLOCK_K x BLOCK_N = 16 x 32 f16 = 1024 bytes
    # Total: 2048 bytes (tiny, fits easily in 64KB LDS)
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    def _out_elem_ty():
        return Textra.f32() if out_dtype == "f32" else Textra.f16()

    class _WmmaGemm(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            # Allocate LDS for A and B tiles
            _state["s_a"] = allocator.allocate_array(Textra.f16(), BLOCK_M * BLOCK_K)
            _state["s_b"] = allocator.allocate_array(Textra.f16(), BLOCK_K * BLOCK_N)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, Textra.f16()),  # [M, K]
            B: lambda: Textra.memref(S, S, Textra.f16()),  # [K, N]
            C: lambda: Textra.memref(S, S, _out_elem_ty()),  # [M, N]
        ):
            v8f16_ty = T.vec(8, T.f16)
            v8f32_ty = T.vec(8, T.f32)

            # Thread/block IDs
            tid = flir.thread_idx("x")
            bid_m = flir.block_idx("y")  # block row
            bid_n = flir.block_idx("x")  # block col

            # Which WMMA tile within the workgroup does this thread belong to?
            # wave_id = tid / 32 (0..3)
            # lane = tid % 32 (0..31)
            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32

            # Map wave_id to (wave_m, wave_n) within the WAVES_M x WAVES_N grid
            c_waves_n = arith.index(WAVES_N)
            wave_m = wave_id // c_waves_n  # 0..WAVES_M-1
            wave_n = wave_id % c_waves_n  # 0..WAVES_N-1

            # WMMA lane decomposition
            lane16 = lane % c16  # 0..15
            base8 = (lane // c16) * c8  # 0 or 8

            # Global tile origin
            tile_m_base = bid_m * arith.index(BLOCK_M)  # row offset
            tile_n_base = bid_n * arith.index(BLOCK_N)  # col offset

            # This wave's WMMA tile origin within the block
            wmma_m_base = wave_m * arith.index(WMMA_M)  # 0 or 16
            wmma_n_base = wave_n * arith.index(WMMA_N)  # 0 or 16

            # Get LDS references
            lds_base = allocator.get_base()
            As = _state["s_a"](lds_base)
            Bs = _state["s_b"](lds_base)

            # Initialize accumulator
            acc = arith.constant_vector(0.0, v8f32_ty)

            # --- K loop ---
            for kt in range_constexpr(num_k_tiles):
                k_base = arith.index(kt * BLOCK_K)

                # ===== Cooperative LDS load =====
                # 128 threads load BLOCK_M * BLOCK_K = 32 * 16 = 512 f16 elements for A
                # 128 threads load BLOCK_K * BLOCK_N = 16 * 32 = 512 f16 elements for B
                # Each thread loads 512/128 = 4 elements for A, 4 for B

                # Load A[BLOCK_M, BLOCK_K] into LDS
                # A is row-major [M, K]. LDS layout: row-major [BLOCK_M][BLOCK_K]
                # 512 elements, 128 threads, 4 elements each
                # Thread tid loads elements at linear positions tid*4, tid*4+1, tid*4+2, tid*4+3
                for elem in range_constexpr(4):
                    lin = tid * arith.index(4) + arith.index(elem)
                    a_row = lin // arith.index(BLOCK_K)  # 0..31
                    a_col = lin % arith.index(BLOCK_K)  # 0..15
                    g_row = tile_m_base + a_row
                    g_col = k_base + a_col
                    a_val = memref.load(A, [g_row, g_col])
                    lds_idx = a_row * arith.index(BLOCK_K) + a_col
                    As.store(a_val, [lds_idx])

                # Load B[BLOCK_K, BLOCK_N] into LDS
                # Same: 512 elements, 128 threads, 4 each
                for elem in range_constexpr(4):
                    lin = tid * arith.index(4) + arith.index(elem)
                    b_row = lin // arith.index(BLOCK_N)  # 0..15
                    b_col = lin % arith.index(BLOCK_N)  # 0..31
                    g_row = k_base + b_row
                    g_col = tile_n_base + b_col
                    b_val = memref.load(B, [g_row, g_col])
                    lds_idx = b_row * arith.index(BLOCK_N) + b_col
                    Bs.store(b_val, [lds_idx])

                # Barrier to ensure LDS is fully populated
                gpu.barrier()

                # ===== WMMA compute =====
                # Load A tile from LDS for this wave's WMMA
                # A in LDS: [BLOCK_M][BLOCK_K], row-major
                # WMMA A layout: row-of-cols => A[lane16][base8+i]
                # Map: LDS_A[wmma_m_base + lane16][base8 + i]
                # (since BLOCK_K = WMMA_K = 16, and base8 can be 0 or 8, this covers K=0..15)
                a_elems = []
                for i in range_constexpr(8):
                    ci = arith.index(i)
                    lds_row = wmma_m_base + lane16  # row in the A tile
                    lds_col = base8 + ci  # k index (0..7 or 8..15)
                    lds_idx = lds_row * arith.index(BLOCK_K) + lds_col
                    a_elems.append(As.load([lds_idx]))
                a_vec = vector.from_elements(v8f16_ty, a_elems)

                # Load B tile from LDS for this wave's WMMA
                # B in LDS: [BLOCK_K][BLOCK_N], row-major
                # WMMA B layout: col-of-rows => B[base8+i][lane16]
                # Map: LDS_B[base8 + i][wmma_n_base + lane16]
                b_elems = []
                for i in range_constexpr(8):
                    ci = arith.index(i)
                    lds_row = base8 + ci  # k index (0..7 or 8..15)
                    lds_col = wmma_n_base + lane16  # col in the B tile
                    lds_idx = lds_row * arith.index(BLOCK_N) + lds_col
                    b_elems.append(Bs.load([lds_idx]))
                b_vec = vector.from_elements(v8f16_ty, b_elems)

                # Execute WMMA: acc += A_tile * B_tile
                acc = rocdl.wmma_f32_16x16x16_f16(
                    v8f32_ty,
                    [arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)],
                )

                # Barrier before next K tile load
                gpu.barrier()

            # ===== Store results =====
            # D layout: col-of-rows => D[base8+i][lane16]
            # Global position: C[tile_m_base + wmma_m_base + base8 + i][tile_n_base + wmma_n_base + lane16]
            for i in range_constexpr(8):
                ci = arith.index(i)
                g_row = tile_m_base + wmma_m_base + base8 + ci
                g_col = tile_n_base + wmma_n_base + lane16
                val = vector.extract(acc, static_position=[i], dynamic_position=[])
                if out_dtype == "f16":
                    val = arith.truncf(val, type=T.f16)
                memref.store(val, C, [g_row, g_col])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, Textra.f16()),
            B: lambda: Textra.memref(S, S, Textra.f16()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            grid_m = arith.index(M // BLOCK_M)
            grid_n = arith.index(N // BLOCK_N)
            block_threads = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm", "wmma_gemm_kernel"],
                grid_size=(grid_n, grid_m, c1),  # (x=N-blocks, y=M-blocks)
                block_size=(block_threads, c1, c1),
                kernel_operands=[A, B, C],
            )

    return _WmmaGemm()
