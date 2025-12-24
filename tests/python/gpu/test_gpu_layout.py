"""GPU kernel tests using flir layout algebra API

These are compilation-only smoke tests: they build GPU kernels with layout-based
indexing and run a light canonicalize/cse pipeline.
"""

import sys

from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.arith import Index
import _mlir.extras.types as T


def test_layout_based_transpose():
    """Matrix transpose using flir layout algebra (compilation smoke test)."""
    M, N = 32, 64

    class _Transpose(flir.MlirModule):
        @flir.kernel
        def transpose_layout(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, T.f32()),
            Output: lambda: T.memref(N, M, T.f32()),
        ):
            bx, by = flir.block_idx("x"), flir.block_idx("y")
            tx, ty = flir.thread_idx("x"), flir.thread_idx("y")
            bdx, bdy = flir.block_dim("x"), flir.block_dim("y")

            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            M_c, N_c = Index(M), Index(N)
            one = Index(1)
            input_layout = flir.make_layout(
                flir.make_shape(M_c, N_c),
                flir.make_stride(N_c, one),
            )
            output_layout = flir.make_layout(
                flir.make_shape(N_c, M_c),
                flir.make_stride(M_c, one),
            )
            _ = input_layout
            _ = output_layout

            valid = (row < M_c) & (col < N_c)
            if valid:
                val = flir.memref.load(Input, [row.value, col.value])
                flir.memref.store(val.value, Output, [col.value, row.value])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, T.f32()),
            Output: lambda: T.memref(N, M, T.f32()),
        ):
            c1 = Index(1).value
            blk = Index(16).value
            gx = Index((N + 15) // 16).value
            gy = Index((M + 15) // 16).value
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "transpose_layout"],
                grid_size=(gx, gy, c1),
                block_size=(blk, blk, c1),
                kernel_operands=[Input, Output],
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

            M_c, N_c = Index(M), Index(N)
            in_stride = Index(in_stride_val)
            out_stride = Index(out_stride_val)
            one = Index(1)

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
                in_idx = (row * in_stride + col).value
                out_idx = (row * out_stride + col).value
                v = flir.memref.load(Input, [in_idx])
                flir.memref.store(v.value, Output, [out_idx])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(M * in_stride_val, T.f32()),
            Output: lambda: T.memref(M * out_stride_val, T.f32()),
        ):
            c1 = Index(1).value
            blk = Index(16).value
            gx = Index((N + 15) // 16).value
            gy = Index((M + 15) // 16).value
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


