#!/usr/bin/env python3
"""GPU kernel test demonstrating Rocir coordinate operations.

This test is meant to be stable under the ExecutionEngine backend: we keep the
kernel simple and just exercise layout algebra + coordinate lowering.
"""

import sys
import os


import rocdsl
from rocdsl.dialects.ext import rocir
from rocdsl.runtime.hip_util import get_hip_arch
from rocdsl.dialects.ext import arith
import _mlir.extras.types as T
import pytest
import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)
import numpy as np


def test_matmul_with_rocir():
    """Write linearized row-major indices into C using rocir coordinate ops."""
    print("\n" + "="*80)
    print("GPU Coords-to-Linear with Rocir Coordinate Operations")
    print("Demonstrating: rocir.make_coord, rocir.make_layout, rocir.crd2idx")
    print("="*80)

    M, N = 32, 32

    class _Coords(rocir.MlirModule):
        @rocir.kernel
        def coords_to_linear(
            self: rocir.T.i64,
            C: lambda: T.memref(M, N, T.i32()),
        ):
            bx = rocir.block_idx("x")
            by = rocir.block_idx("y")
            tx = rocir.thread_idx("x")
            ty = rocir.thread_idx("y")

            # Compute thread coordinates
            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            m_c = arith.index(M)
            n_c = arith.index(N)
            one = arith.index(1)

            # Create rocir layout for output matrix C
            c_shape = rocir.make_shape(m_c, n_c)
            c_stride = rocir.make_stride(n_c, one)  # Row-major: stride=(32, 1)
            c_layout = rocir.make_layout(c_shape, c_stride)

            # Create coordinate for current thread's position
            thread_coord = rocir.make_coord(row.value, col.value)

            # Use crd2idx to compute linear index (will be lowered to arith ops)
            idx = rocir.crd2idx(thread_coord, c_layout)
            idx_v = idx.value if hasattr(idx, "value") else idx
            idx_i32 = arith.IndexCastOp(T.i32(), idx_v).result
            idx_i32_v = idx_i32.value if hasattr(idx_i32, "value") else idx_i32
            rocir.memref.store(idx_i32_v, C, [row.value, col.value])

        @rocir.jit
        def __call__(
            self: rocir.T.i64,
            C: lambda: T.memref(M, N, T.i32()),
        ):
            # Wrap the kernel launch inside a host-side function.
            c1 = arith.index(1).value
            blk = arith.index(16).value
            gx = arith.index((N + 15) // 16).value
            gy = arith.index((M + 15) // 16).value
            rocir.gpu_ext.LaunchFuncOp(
                ["kernels", "coords_to_linear"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[C],
            )

    print("\n Generated MLIR with rocir coordinate operations:")
    print("- rocir.make_shape: Defines matrix dimensions")
    print("- rocir.make_stride: Specifies row-major layout")
    print("- rocir.make_layout: Combines shape and stride")
    print("- rocir.make_coord: Creates 2D thread coordinates")
    print("- rocir.crd2idx: Converts coord to linear index\n")

    m_mod = _Coords()
    print("Compiling...")
    exe = rocdsl.compile(m_mod)
    print(" Compiled\n")

    # Run computation: C[row, col] = row * N + col
    expected = np.arange(M * N, dtype=np.int32).reshape(M, N)
    C = torch.empty((M, N), device="cuda", dtype=torch.int32)

    exe(C)
    torch.cuda.synchronize()
    c_host = C.cpu().numpy()

    diff = np.abs(c_host.astype(np.int64) - expected.astype(np.int64))
    max_error = int(np.max(diff))

    print(f" Max abs diff: {max_error}")
    return max_error == 0


if __name__ == "__main__":
    print("="*80)
    print("Rocir Coordinate Operations in GPU Kernels")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)

    result = test_matmul_with_rocir()

    print("\n" + "="*80)
    if result:
        print(" TEST PASSED!")
        print("\nDemonstrated:")
        print("Rocir layout algebra integrated into GPU kernel")
        print("Coordinate operations (make_coord, make_layout, crd2idx)")
        print("Lowered to arithmetic via rocir-opt subprocess")
        print(f"Compiled and executed on {get_hip_arch()}")
        sys.exit(0)
    else:
        print("⚠️ TEST FAILED")
        sys.exit(1)
