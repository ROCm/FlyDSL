#!/usr/bin/env python3
"""GPU kernel test demonstrating Flir coordinate operations.

This test is meant to be stable under the ExecutionEngine backend: we keep the
kernel simple and just exercise layout algebra + coordinate lowering.
"""

import sys
import os


import pyflir
from pyflir.dialects.ext import flir
from pyflir.runtime.device import get_rocm_arch
from pyflir.dialects.ext import arith
import _mlir.extras.types as T
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

    class _Coords(flir.MlirModule):
        @flir.kernel
        def coords_to_linear(
            self: flir.T.i64,
            C: lambda: T.memref(M, N, T.i32()),
        ):
            bx = flir.block_idx("x")
            by = flir.block_idx("y")
            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            # Compute thread coordinates
            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            m_c = arith.index(M)
            n_c = arith.index(N)
            one = arith.index(1)

            # Create flir layout for output matrix C
            c_shape = flir.make_shape(m_c, n_c)
            c_stride = flir.make_stride(n_c, one)  # Row-major: stride=(32, 1)
            c_layout = flir.make_layout(c_shape, c_stride)

            # Create coordinate for current thread's position
            thread_coord = flir.make_coord(row.value, col.value)

            # Use crd2idx to compute linear index (will be lowered to arith ops)
            idx = flir.crd2idx(thread_coord, c_layout)
            idx_v = idx.value if hasattr(idx, "value") else idx
            idx_i32 = arith.IndexCastOp(T.i32(), idx_v).result
            idx_i32_v = idx_i32.value if hasattr(idx_i32, "value") else idx_i32
            flir.memref.store(idx_i32_v, C, [row.value, col.value])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            C: lambda: T.memref(M, N, T.i32()),
        ):
            # Wrap the kernel launch inside a host-side function.
            c1 = arith.index(1).value
            blk = arith.index(16).value
            gx = arith.index((N + 15) // 16).value
            gy = arith.index((M + 15) // 16).value
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "coords_to_linear"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[C],
            )

    print("\n Generated MLIR with flir coordinate operations:")
    print("- flir.make_shape: Defines matrix dimensions")
    print("- flir.make_stride: Specifies row-major layout")
    print("- flir.make_layout: Combines shape and stride")
    print("- flir.make_coord: Creates 2D thread coordinates")
    print("- flir.crd2idx: Converts coord to linear index\n")

    m_mod = _Coords()
    print("Compiling...")
    exe = pyflir.compile(m_mod)
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
    print("Flir Coordinate Operations in GPU Kernels")
    print(f"GPU: {get_rocm_arch()}")
    print("="*80)

    result = test_matmul_with_flir()

    print("\n" + "="*80)
    if result:
        print(" TEST PASSED!")
        print("\nDemonstrated:")
        print("Flir layout algebra integrated into GPU kernel")
        print("Coordinate operations (make_coord, make_layout, crd2idx)")
        print("Lowered to arithmetic via flir-opt subprocess")
        print(f"Compiled and executed on {get_rocm_arch()}")
        sys.exit(0)
    else:
        print("⚠️ TEST FAILED")
        sys.exit(1)
