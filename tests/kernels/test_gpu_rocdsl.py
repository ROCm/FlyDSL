"""GPU kernel tests demonstrating Flir layout/coord concepts.

These are compilation-focused tests: they emit GPU kernels with flir ops and
run a light canonicalize/cse pipeline. They intentionally do not depend on HIP
runtime availability.
"""

import sys

from _mlir import ir
import _mlir.extras.types as T

from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext import arith, flir, memref


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
    S = ir.ShapedType.get_dynamic_size()

    class _Transpose(flir.MlirModule):
        @flir.kernel
        def matrixTranspose(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, T.f32()),
            B: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            bx = flir.block_idx("x")
            by = flir.block_idx("y")
            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            m_c = arith.ArithValue(m_in)
            n_c = arith.ArithValue(n_in)
            one = arith.index(1)
            a_layout = flir.make_layout(flir.make_shape(arith.as_value(m_c), arith.as_value(n_c)), flir.make_stride(arith.as_value(n_c), one))
            b_layout = flir.make_layout(flir.make_shape(arith.as_value(n_c), arith.as_value(m_c)), flir.make_stride(arith.as_value(m_c), one))
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
            valid = (arith.ArithValue(row) < m_c) & (arith.ArithValue(col) < n_c)
            if valid:
                a_val = memref.load(A, [row, col])
                memref.store(a_val, B, [col, row])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, T.f32()),
            B: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            c1 = arith.index(1)
            blk = arith.index(16)
            n_c = arith.ArithValue(n_in)
            m_c = arith.ArithValue(m_in)
            gx = arith.as_value((n_c + arith.index(15)) // blk)
            gy = arith.as_value((m_c + arith.index(15)) // blk)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "matrixTranspose"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[A, B, m_in, n_in],
            )

    m = _Transpose()
    s = str(m.module)
    assert "flir.make_layout" in s
    assert "flir.crd2idx" in s
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_matmul_uses_scf_for_and_flir_layout():
    M, N, K = 32, 32, 64
    S = ir.ShapedType.get_dynamic_size()

    class _Matmul(flir.MlirModule):
        @flir.kernel
        def matmul(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, T.f32()),
            B: lambda: T.memref(S, S, T.f32()),
            C: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            bx = flir.block_idx("x")
            by = flir.block_idx("y")
            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            tile = arith.index(16)
            row = (by * tile + ty)
            col = (bx * tile + tx)

            m_c = arith.ArithValue(m_in)
            n_c = arith.ArithValue(n_in)
            k_c = arith.ArithValue(k_in)
            one = arith.index(1)

            c_layout = flir.make_layout(flir.make_shape(arith.as_value(m_c), arith.as_value(n_c)), flir.make_stride(arith.as_value(n_c), one))
            thread_coord = flir.make_coord(row, col)
            _ = flir.crd2idx(thread_coord, c_layout)

            valid = (arith.ArithValue(row) < m_c) & (arith.ArithValue(col) < n_c)
            if valid:
                sum_val = arith.f32(0.0)
                # Python `for` + reassignment is auto-lowered into scf.for with iter_args/yield/results.
                for k in range(arith.as_value(k_c)):
                    a_val = memref.load(A, [row, k])
                    b_val = memref.load(B, [k, col])
                    sum_val = sum_val + (a_val * b_val)

                memref.store(sum_val, C, [row, col])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, T.f32()),
            B: lambda: T.memref(S, S, T.f32()),
            C: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            c1 = arith.index(1)
            blk = arith.index(16)
            n_c = arith.ArithValue(n_in)
            m_c = arith.ArithValue(m_in)
            gx = arith.as_value((n_c + arith.index(15)) // blk)
            gy = arith.as_value((m_c + arith.index(15)) // blk)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "matmul"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[A, B, C, m_in, n_in, k_in],
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


