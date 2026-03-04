#!/usr/bin/env python3
"""Basic WMMA test -- single 16x16x16 tile on gfx1201 (RDNA4, wave32).

Validates that the v_wmma_f32_16x16x16_f16 instruction works correctly
on Radeon 9700 (gfx1201) by running a single-wave 16x16 matmul.

WMMA data layout for wave32 (32 lanes, 8 elements per lane):
  For lane t (0..31), let lane16 = t % 16, base8 = (t / 16) * 8.

  A (16x16 f16): "row-of-cols" layout
    Lane t holds A[lane16][base8 + i] for i in 0..7
    (each lane loads 8 consecutive columns from one row)

  B (16x16 f16): "col-of-rows" layout
    Lane t holds B[base8 + i][lane16] for i in 0..7
    (each lane loads 8 consecutive rows from one column)

  D (16x16 f32): "col-of-rows" layout (same as B)
    Lane t holds D[base8 + i][lane16] for i in 0..7

  Verified empirically on gfx1201 (Radeon 9700, RDNA4).
"""

import sys
import os
import numpy as np
import pytest
import torch

import flydsl
from flydsl.dialects.ext import flir, arith, memref, vector, rocdl
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T
from _mlir import ir
import _mlir.extras.types as Textra

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


def create_wmma_kernel():
    """Create a minimal WMMA 16x16x16 f16 matmul kernel.

    Layout: C[16,16] = A[16,16] @ B[16,16]  (all f16, accumulate in f32)

    Uses row-major A and B in global memory.
    Each thread (lane) loads its 8 elements of A and B based on the WMMA lane
    mapping, executes one WMMA instruction, and stores the 8 result elements.
    """
    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    class _WmmaBasic(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_test"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def wmma_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, Textra.f16()),  # [M, K] row-major f16
            B: lambda: Textra.memref(S, S, Textra.f16()),  # [K, N] row-major f16
            C: lambda: Textra.memref(S, S, Textra.f32()),  # [M, N] row-major f32
        ):
            # WMMA vector types for wave32
            v8f16_ty = T.vec(8, T.f16)
            v8f32_ty = T.vec(8, T.f32)

            # Thread ID within the wave (0..31)
            tid = flir.thread_idx("x")

            # Decompose lane ID for WMMA mapping:
            # lane16 = tid % 16  (0..15)
            # base8  = (tid / 16) * 8  (0 or 8)
            c16 = arith.index(16)
            c8 = arith.index(8)
            lane16 = tid % c16  # 0..15
            base8 = (tid // c16) * c8  # 0 or 8

            # WMMA 16x16x16 wave32 data layout (empirically verified on gfx1201):
            #   A: "row-of-cols" -- lane t loads A[t%16][(t/16)*8 + i] for i in 0..7
            #   B: "col-of-rows" -- lane t loads B[(t/16)*8 + i][t%16] for i in 0..7
            #   D: "col-of-rows" -- lane t holds D[(t/16)*8 + i][t%16] for i in 0..7
            #
            # lane16 = tid % 16, base8 = (tid / 16) * 8

            # Load A: row-of-cols layout => A[lane16][base8 + i]
            a_elems = []
            for i in range_constexpr(8):
                ci = arith.index(i)
                a_val = memref.load(A, [lane16, base8 + ci])
                a_elems.append(a_val)
            a_vec = vector.from_elements(v8f16_ty, a_elems)

            # Load B: col-of-rows layout => B[base8 + i][lane16]
            b_elems = []
            for i in range_constexpr(8):
                ci = arith.index(i)
                b_val = memref.load(B, [base8 + ci, lane16])
                b_elems.append(b_val)
            b_vec = vector.from_elements(v8f16_ty, b_elems)

            # Initialize accumulator to zero
            zero_acc = arith.constant_vector(0.0, v8f32_ty)

            # Execute WMMA: D = A * B + C
            d_vec = rocdl.wmma_f32_16x16x16_f16(
                v8f32_ty,
                [arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(zero_acc)],
            )

            # Store D: col-of-rows layout => C[base8 + i][lane16]
            for i in range_constexpr(8):
                ci = arith.index(i)
                val = vector.extract(d_vec, static_position=[i], dynamic_position=[])
                memref.store(val, C, [base8 + ci, lane16])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, Textra.f16()),
            B: lambda: Textra.memref(S, S, Textra.f16()),
            C: lambda: Textra.memref(S, S, Textra.f32()),
        ):
            c1 = arith.index(1)
            c32 = arith.index(32)  # one wave = 32 threads
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_test", "wmma_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(c32, c1, c1),
                kernel_operands=[A, B, C],
            )

    return _WmmaBasic()


def test_wmma_basic():
    """Test single WMMA 16x16x16 f16 matmul on gfx1201."""
    gpu_arch = get_rocm_arch()
    print(f"\n{'=' * 60}")
    print(f"WMMA Basic Test - {gpu_arch}")
    print(f"{'=' * 60}")

    if not gpu_arch.startswith("gfx12"):
        pytest.skip(f"WMMA test requires RDNA4 (gfx12xx), got {gpu_arch}")

    # Create and compile kernel
    print("Creating WMMA kernel...")
    m = create_wmma_kernel()
    print("Compiling...")
    exe = flydsl.compile(m)

    # Prepare data - use small values to keep f16 precision reasonable
    np.random.seed(42)
    a_np = np.random.randn(16, 16).astype(np.float16) * 0.1
    b_np = np.random.randn(16, 16).astype(np.float16) * 0.1

    # Reference: f16 matmul accumulated in f32
    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    A = torch.tensor(a_np, device="cuda", dtype=torch.float16)
    B = torch.tensor(b_np, device="cuda", dtype=torch.float16)
    C = torch.zeros(16, 16, device="cuda", dtype=torch.float32)

    # Run kernel
    print("Launching kernel (1 workgroup, 32 threads)...")
    exe(A, B, C)
    torch.cuda.synchronize()

    c_host = C.cpu().numpy()

    # Verify
    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)

    print(f"Max absolute error: {error:.2e}")
    print(f"Max relative error: {rel_error:.2e}")
    print(f"Output sample (top-left 4x4):")
    print(c_host[:4, :4])
    print(f"Expected sample:")
    print(expected[:4, :4])

    # f16 matmul with 16 accumulation steps: expect ~1e-3 error
    assert rel_error < 1e-2, (
        f"WMMA result too far from reference: rel_error={rel_error:.2e}"
    )
    print(f"\nPASS - WMMA basic test succeeded!")


if __name__ == "__main__":
    test_wmma_basic()
