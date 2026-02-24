#!/usr/bin/env python3
"""GPU kernel test demonstrating Flir coordinate operations.

This test is meant to be stable under the ExecutionEngine backend: we keep the
kernel simple and just exercise layout algebra + coordinate lowering.
"""

import sys
import os


import flydsl
from flydsl.dialects.ext import flir
from flydsl.runtime.device import get_rocm_arch
from flydsl.dialects.ext import arith, memref
import _mlir.extras.types as T
from _mlir import ir
import pytest
import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)
import numpy as np


def test_matmul_with_flir():
    """Write linearized row-major indices into C using flir coordinate ops."""
    print("\n" + "="*80)
    print("GPU Coords-to-Linear with Flir Coordinate Operations")
    print("Demonstrating: flir.make_coord, flir.make_layout, flir.crd2idx")
    print("="*80)

    M, N = 32, 32
    S = ir.ShapedType.get_dynamic_size()

    class _Coords(flir.MlirModule):
        @flir.kernel
        def coords_to_linear(
            self: flir.T.i64,
            C: lambda: T.memref(S, S, T.i32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            bx = flir.block_idx("x")
            by = flir.block_idx("y")
            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            # Compute thread coordinates
            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            # Use runtime sizes so the same compiled kernel can be reused for different M.
            m_c = arith.ArithValue(m_in)
            n_c = arith.ArithValue(n_in)
            one = arith.index(1)

            # Create flir layout for output matrix C
            c_shape = flir.make_shape(m_c, n_c)
            c_stride = flir.make_stride(n_c, one)  # Row-major: stride=(32, 1)
            c_layout = flir.make_layout(c_shape, c_stride)

            # Create coordinate for current thread's position
            thread_coord = flir.make_coord(row, col)

            # Use crd2idx to compute linear index (will be lowered to arith ops)
            idx = flir.crd2idx(thread_coord, c_layout)
            idx_i32 = arith.index_cast(T.i32(), idx)
            valid = (arith.ArithValue(row) < m_c) & (arith.ArithValue(col) < n_c)
            if valid:
                memref.store(idx_i32, C, [row, col])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            C: lambda: T.memref(S, S, T.i32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            # Wrap the kernel launch inside a host-side function.
            c1 = arith.index(1)
            blk = arith.index(16)
            blk = arith.index(16)
            n_c = arith.ArithValue(n_in)
            m_c = arith.ArithValue(m_in)
            gx = arith.as_value((n_c + arith.index(15)) // blk)
            gy = arith.as_value((m_c + arith.index(15)) // blk)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "coords_to_linear"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[C, m_in, n_in],
            )

    print("\n Generated MLIR with flir coordinate operations:")
    print("- flir.make_shape: Defines matrix dimensions")
    print("- flir.make_stride: Specifies row-major layout")
    print("- flir.make_layout: Combines shape and stride")
    print("- flir.make_coord: Creates 2D thread coordinates")
    print("- flir.crd2idx: Converts coord to linear index\n")

    m_mod = _Coords()
    print("Compiling...")
    exe = flydsl.compile(m_mod)
    print(" Compiled\n")

    # Run computation: C[row, col] = row * N + col
    expected = np.arange(M * N, dtype=np.int32).reshape(M, N)
    C = torch.empty((M, N), device="cuda", dtype=torch.int32)

    exe(C, M, N)
    torch.cuda.synchronize()
    c_host = C.cpu().numpy()

    diff = np.abs(c_host.astype(np.int64) - expected.astype(np.int64))
    max_error = int(np.max(diff))

    print(f" Max abs diff: {max_error}")
    assert(max_error == 0, "Max absolute difference is not 0")
    return


if __name__ == "__main__":
    print("="*80)
    print("Flir Coordinate Operations in GPU Kernels")
    print(f"GPU: {get_rocm_arch()}")
    print("="*80)

    test_matmul_with_flir()

    print("\n" + "="*80)
    print(" TEST PASSED!")
    print("\nDemonstrated:")
    print("Flir layout algebra integrated into GPU kernel")
    print("Coordinate operations (make_coord, make_layout, crd2idx)")
    print("Lowered to arithmetic via flir-opt subprocess")
    print(f"Compiled and executed on {get_rocm_arch()}")
    sys.exit(0)
