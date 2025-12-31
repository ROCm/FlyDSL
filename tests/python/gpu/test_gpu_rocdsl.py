"""GPU kernel tests demonstrating Flir layout/coord concepts.

These are compilation-focused tests: they emit GPU kernels with flir ops and
run a light canonicalize/cse pipeline. They intentionally do not depend on HIP
runtime availability.
"""

import sys

from _mlir import ir
import _mlir.extras.types as T

from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.dialects.ext import arith, flir, memref


def test_vector_add_flir_crd2idx_emits():
    SIZE = 2048

    class _VecAdd(flir.MlirModule):
        @flir.kernel
        def vecAdd(
            self: flir.T.i64,
            A: lambda: T.memref(SIZE, T.f32()),
            B: lambda: T.memref(SIZE, T.f32()),
            C: lambda: T.memref(SIZE, T.f32()),
        ):
            tid = (flir.block_idx("x") * flir.block_dim("x") + flir.thread_idx("x"))
            vec_shape = flir.make_shape(SIZE)
            vec_stride = flir.make_stride(1)
            vec_layout = flir.make_layout(vec_shape, vec_stride)
            thread_coord = flir.make_coord(tid)
            linear_idx = flir.crd2idx(thread_coord, vec_layout)
            a = memref.load(A, [linear_idx])
            b = memref.load(B, [linear_idx])
            c = a + b
            memref.store(c, C, [linear_idx])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(SIZE, T.f32()),
            B: lambda: T.memref(SIZE, T.f32()),
            C: lambda: T.memref(SIZE, T.f32()),
        ):
            c1 = arith.index(1)
            blk = arith.index(256)
            gx = arith.index((SIZE + 255) // 256)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "vecAdd"],
                grid_size=(gx, c1, c1),
                block_size=(blk, c1, c1),
                kernel_operands=[A, B, C],
            )

    m = _VecAdd()
    s = str(m.module)
    assert "flir.make_coord" in s
    assert "flir.crd2idx" in s
    assert "gpu.func" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_matrix_transpose_flir_layout_emits():
    M, N = 32, 64

    class _Transpose(flir.MlirModule):
        @flir.kernel
        def matrixTranspose(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(N, M, T.f32()),
        ):
            bx = flir.block_idx("x")
            by = flir.block_idx("y")
            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            a_layout = flir.make_layout(flir.make_shape(M, N), flir.make_stride(N, 1))
            b_layout = flir.make_layout(flir.make_shape(N, M), flir.make_stride(M, 1))
            _ = a_layout
            _ = b_layout

            thread_coord = flir.make_coord(row, col)
            a_idx = flir.crd2idx(thread_coord, a_layout)
            transposed_coord = flir.make_coord(col, row)
            b_idx = flir.crd2idx(transposed_coord, b_layout)

            # Note: A/B are 2D memrefs, so use 2D indices for load/store.
            # We still emit flir.crd2idx above to exercise layout lowering.
            _ = a_idx
            _ = b_idx
            a_val = memref.load(A, [row, col])
            memref.store(a_val, B, [col, row])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(N, M, T.f32()),
        ):
            c1 = arith.index(1)
            blk = arith.index(16)
            gx = arith.index((N + 15) // 16)
            gy = arith.index((M + 15) // 16)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "matrixTranspose"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[A, B],
            )

    m = _Transpose()
    s = str(m.module)
    assert "flir.make_layout" in s
    assert "flir.crd2idx" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_matmul_uses_scf_for_and_flir_layout():
    M, N, K = 32, 32, 64

    class _Matmul(flir.MlirModule):
        @flir.kernel
        def matmul(
            self: flir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            bx = flir.block_idx("x")
            by = flir.block_idx("y")
            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            m_c = arith.index(M)
            n_c = arith.index(N)
            k_c = arith.index(K)
            one = arith.index(1)

            c_layout = flir.make_layout(flir.make_shape(m_c, n_c), flir.make_stride(n_c, one))
            thread_coord = flir.make_coord(row, col)
            _ = flir.crd2idx(thread_coord, c_layout)

            valid = (row < m_c) & (col < n_c)
            if valid:
                sum_val = arith.f32(0.0)
                # Python `for` + reassignment is auto-lowered into scf.for with iter_args/yield/results.
                for k in range(k_c):
                    a_val = memref.load(A, [row, k])
                    b_val = memref.load(B, [k, col])
                    sum_val = sum_val + (a_val * b_val)

                memref.store(sum_val, C, [row, col])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            c1 = arith.index(1)
            blk = arith.index(16)
            gx = arith.index((N + 15) // 16)
            gy = arith.index((M + 15) // 16)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "matmul"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[A, B, C],
            )

    m = _Matmul()
    s = str(m.module)
    assert "scf.for" in s
    assert "flir.make_layout" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


if __name__ == "__main__":
    # `run_tests.sh` executes these files directly (not via pytest),
    # so call the test functions explicitly.
    test_vector_add_flir_crd2idx_emits()
    test_matrix_transpose_flir_layout_emits()
    test_matmul_uses_scf_for_and_flir_layout()
    sys.exit(0)


