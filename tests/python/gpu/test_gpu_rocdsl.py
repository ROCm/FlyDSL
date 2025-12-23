"""GPU kernel tests demonstrating Rocir layout/coord concepts.

These are compilation-focused tests: they emit GPU kernels with rocir ops and
run a light canonicalize/cse pipeline. They intentionally do not depend on HIP
runtime availability.
"""

import sys

from _mlir import ir
import _mlir.extras.types as T

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import arith, rocir


def test_vector_add_rocir_crd2idx_emits():
    SIZE = 2048

    class _VecAdd(rocir.MlirModule):
        @rocir.kernel
        def vecAdd(
            self: rocir.T.i64,
            A: lambda: T.memref(SIZE, T.f32()),
            B: lambda: T.memref(SIZE, T.f32()),
            C: lambda: T.memref(SIZE, T.f32()),
        ):
            tid = (rocir.block_idx("x") * rocir.block_dim("x") + rocir.thread_idx("x"))
            vec_shape = rocir.make_shape(SIZE)
            vec_stride = rocir.make_stride(1)
            vec_layout = rocir.make_layout(vec_shape, vec_stride)
            thread_coord = rocir.make_coord(tid.value)
            linear_idx = rocir.crd2idx(thread_coord, vec_layout)
            idx = linear_idx.value if hasattr(linear_idx, "value") else linear_idx
            a = rocir.memref.load(A, [idx])
            b = rocir.memref.load(B, [idx])
            c = a + b
            rocir.memref.store(c.value, C, [idx])

        @rocir.jit
        def __call__(
            self: rocir.T.i64,
            A: lambda: T.memref(SIZE, T.f32()),
            B: lambda: T.memref(SIZE, T.f32()),
            C: lambda: T.memref(SIZE, T.f32()),
        ):
            c1 = arith.index(1).value
            blk = arith.index(256).value
            gx = arith.index((SIZE + 255) // 256).value
            rocir.gpu_ext.LaunchFuncOp(
                ["kernels", "vecAdd"],
                grid_size=(gx, c1, c1),
                block_size=(blk, c1, c1),
                kernel_operands=[A, B, C],
            )

    m = _VecAdd()
    s = str(m.module)
    assert "rocir.make_coord" in s
    assert "rocir.crd2idx" in s
    assert "gpu.func" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_matrix_transpose_rocir_layout_emits():
    M, N = 32, 64

    class _Transpose(rocir.MlirModule):
        @rocir.kernel
        def matrixTranspose(
            self: rocir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(N, M, T.f32()),
        ):
            bx = rocir.block_idx("x")
            by = rocir.block_idx("y")
            tx = rocir.thread_idx("x")
            ty = rocir.thread_idx("y")

            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            a_layout = rocir.make_layout(rocir.make_shape(M, N), rocir.make_stride(N, 1))
            b_layout = rocir.make_layout(rocir.make_shape(N, M), rocir.make_stride(M, 1))
            _ = a_layout
            _ = b_layout

            thread_coord = rocir.make_coord(row.value, col.value)
            a_idx = rocir.crd2idx(thread_coord, a_layout)
            transposed_coord = rocir.make_coord(col.value, row.value)
            b_idx = rocir.crd2idx(transposed_coord, b_layout)

            # Note: A/B are 2D memrefs, so use 2D indices for load/store.
            # We still emit rocir.crd2idx above to exercise layout lowering.
            _ = a_idx
            _ = b_idx
            a_val = rocir.memref.load(A, [row.value, col.value])
            rocir.memref.store(a_val.value, B, [col.value, row.value])

        @rocir.jit
        def __call__(
            self: rocir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(N, M, T.f32()),
        ):
            c1 = arith.index(1).value
            blk = arith.index(16).value
            gx = arith.index((N + 15) // 16).value
            gy = arith.index((M + 15) // 16).value
            rocir.gpu_ext.LaunchFuncOp(
                ["kernels", "matrixTranspose"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[A, B],
            )

    m = _Transpose()
    s = str(m.module)
    assert "rocir.make_layout" in s
    assert "rocir.crd2idx" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_matmul_uses_scf_for_and_rocir_layout():
    M, N, K = 32, 32, 64

    class _Matmul(rocir.MlirModule):
        @rocir.kernel
        def matmul(
            self: rocir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            bx = rocir.block_idx("x")
            by = rocir.block_idx("y")
            tx = rocir.thread_idx("x")
            ty = rocir.thread_idx("y")

            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            m_c = arith.index(M)
            n_c = arith.index(N)
            k_c = arith.index(K)
            one = arith.index(1)

            c_layout = rocir.make_layout(rocir.make_shape(m_c, n_c), rocir.make_stride(n_c, one))
            thread_coord = rocir.make_coord(row.value, col.value)
            _ = rocir.crd2idx(thread_coord, c_layout)

            valid = (row < m_c) & (col < n_c)
            if valid:
                sum_val = arith.f32(0.0)
                # Python `for` + reassignment is auto-lowered into scf.for with iter_args/yield/results.
                for k in range(k_c):
                    k_v = k.value if hasattr(k, "value") else k
                    a_val = rocir.memref.load(A, [row.value, k_v])
                    b_val = rocir.memref.load(B, [k_v, col.value])
                    sum_val = sum_val + (a_val * b_val)

                out_v = sum_val.value if hasattr(sum_val, "value") else sum_val
                rocir.memref.store(out_v, C, [row.value, col.value])

        @rocir.jit
        def __call__(
            self: rocir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            c1 = arith.index(1).value
            blk = arith.index(16).value
            gx = arith.index((N + 15) // 16).value
            gy = arith.index((M + 15) // 16).value
            rocir.gpu_ext.LaunchFuncOp(
                ["kernels", "matmul"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[A, B, C],
            )

    m = _Matmul()
    s = str(m.module)
    assert "scf.for" in s
    assert "rocir.make_layout" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


if __name__ == "__main__":
    # `run_tests.sh` executes these files directly (not via pytest),
    # so call the test functions explicitly.
    test_vector_add_rocir_crd2idx_emits()
    test_matrix_transpose_rocir_layout_emits()
    test_matmul_uses_scf_for_and_rocir_layout()
    sys.exit(0)


