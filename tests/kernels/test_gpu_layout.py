"""GPU kernel tests using flir layout algebra API

These are compilation-only smoke tests: they build GPU kernels with layout-based
indexing and run a light canonicalize/cse pipeline.
"""

import sys

from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext import flir, memref, arith
import _mlir.extras.types as T
from _mlir import ir


def test_layout_based_transpose():
    """Matrix transpose using flir layout algebra (compilation smoke test)."""
    M, N = 32, 64
    S = ir.ShapedType.get_dynamic_size()

    class _Transpose(flir.MlirModule):
        @flir.kernel
        def transpose_layout(
            self: flir.T.i64,
            Input: lambda: T.memref(S, S, T.f32()),
            Output: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            bx, by = flir.block_idx("x"), flir.block_idx("y")
            tx, ty = flir.thread_idx("x"), flir.thread_idx("y")
            bdx, bdy = flir.block_dim("x"), flir.block_dim("y")

            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            M_c, N_c = arith.ArithValue(m_in), arith.ArithValue(n_in)
            one = arith.index(1)
            input_layout = flir.make_layout(
                flir.make_shape(arith.as_value(M_c), arith.as_value(N_c)),
                flir.make_stride(arith.as_value(N_c), one),
            )
            output_layout = flir.make_layout(
                flir.make_shape(arith.as_value(N_c), arith.as_value(M_c)),
                flir.make_stride(arith.as_value(M_c), one),
            )
            _ = input_layout
            _ = output_layout

            valid = (arith.ArithValue(row) < M_c) & (arith.ArithValue(col) < N_c)
            if valid:
                val = memref.load(Input, [row, col])
                memref.store(val, Output, [col, row])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(S, S, T.f32()),
            Output: lambda: T.memref(S, S, T.f32()),
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
                ["kernels", "transpose_layout"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[Input, Output, m_in, n_in],
            )

    m = _Transpose()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_strided_layout_access():
    """Strided layout with custom shape and stride using flir (compilation smoke test)."""
    M, N = 16, 32
    in_stride_val = N + 8
    out_stride_val = N + 4

    class _StridedCopy(flir.MlirModule):
        @flir.kernel
        def copy_with_layout(
            self: flir.T.i64,
            Input: lambda: T.memref(M * in_stride_val, T.f32()),
            Output: lambda: T.memref(M * out_stride_val, T.f32()),
        ):
            bx, by = flir.block_idx("x"), flir.block_idx("y")
            tx, ty = flir.thread_idx("x"), flir.thread_idx("y")
            bdx, bdy = flir.block_dim("x"), flir.block_dim("y")

            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            M_c, N_c = arith.index(M), arith.index(N)
            in_stride = arith.index(in_stride_val)
            out_stride = arith.index(out_stride_val)
            one = arith.index(1)

            in_layout = flir.make_layout(
                flir.make_shape(M_c, N_c),
                flir.make_stride(in_stride, one),
            )
            out_layout = flir.make_layout(
                flir.make_shape(M_c, N_c),
                flir.make_stride(out_stride, one),
            )
            _ = in_layout
            _ = out_layout

            valid = (row < M_c) & (col < N_c)
            if valid:
                in_idx = row * in_stride + col
                out_idx = row * out_stride + col
                v = memref.load(Input, [in_idx])
                memref.store(v, Output, [out_idx])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(M * in_stride_val, T.f32()),
            Output: lambda: T.memref(M * out_stride_val, T.f32()),
        ):
            c1 = arith.index(1)
            blk = arith.index(16)
            gx = arith.index((N + 15) // 16)
            gy = arith.index((M + 15) // 16)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "copy_with_layout"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[Input, Output],
            )

    m = _StridedCopy()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())


if __name__ == "__main__":
    # `run_tests.sh` executes these files directly (not via pytest),
    # so call the test functions explicitly.
    test_layout_based_transpose()
    test_strided_layout_access()
    sys.exit(0)


