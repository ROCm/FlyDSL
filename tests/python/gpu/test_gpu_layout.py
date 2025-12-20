"""GPU kernel tests using rocir layout algebra API

These are compilation-only smoke tests: they build GPU kernels with layout-based
indexing and run a light canonicalize/cse pipeline.
"""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index
import _mlir.extras.types as T


def test_layout_based_transpose():
    """Matrix transpose using rocir layout algebra (compilation smoke test)."""
    M, N = 32, 64

    class _Transpose(rocir.MlirModule):
        @rocir.kernel
        def transpose_layout(
            self: rocir.T.i64,
            Input: lambda: T.memref(M, N, T.f32()),
            Output: lambda: T.memref(N, M, T.f32()),
        ):
            bx, by = rocir.block_idx("x"), rocir.block_idx("y")
            tx, ty = rocir.thread_idx("x"), rocir.thread_idx("y")
            bdx, bdy = rocir.block_dim("x"), rocir.block_dim("y")

            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            M_c, N_c = Index(M), Index(N)
            one = Index(1)
            input_layout = rocir.make_layout(
                rocir.make_shape(M_c, N_c),
                rocir.make_stride(N_c, one),
            )
            output_layout = rocir.make_layout(
                rocir.make_shape(N_c, M_c),
                rocir.make_stride(M_c, one),
            )
            _ = input_layout
            _ = output_layout

            valid = (row < M_c) & (col < N_c)
            with rocir.scf_ext.IfOp(valid.value):
                val = rocir.memref.load(Input, [row.value, col.value])
                rocir.memref.store(val.value, Output, [col.value, row.value])
                rocir.scf_ext.yield_([])

    m = _Transpose()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())


def test_strided_layout_access():
    """Strided layout with custom shape and stride using rocir (compilation smoke test)."""
    M, N = 16, 32
    in_stride_val = N + 8
    out_stride_val = N + 4

    class _StridedCopy(rocir.MlirModule):
        @rocir.kernel
        def copy_with_layout(
            self: rocir.T.i64,
            Input: lambda: T.memref(M * in_stride_val, T.f32()),
            Output: lambda: T.memref(M * out_stride_val, T.f32()),
        ):
            bx, by = rocir.block_idx("x"), rocir.block_idx("y")
            tx, ty = rocir.thread_idx("x"), rocir.thread_idx("y")
            bdx, bdy = rocir.block_dim("x"), rocir.block_dim("y")

            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            M_c, N_c = Index(M), Index(N)
            in_stride = Index(in_stride_val)
            out_stride = Index(out_stride_val)
            one = Index(1)

            in_layout = rocir.make_layout(
                rocir.make_shape(M_c, N_c),
                rocir.make_stride(in_stride, one),
            )
            out_layout = rocir.make_layout(
                rocir.make_shape(M_c, N_c),
                rocir.make_stride(out_stride, one),
            )
            _ = in_layout
            _ = out_layout

            valid = (row < M_c) & (col < N_c)
            with rocir.scf_ext.IfOp(valid.value):
                in_idx = (row * in_stride + col).value
                out_idx = (row * out_stride + col).value
                v = rocir.memref.load(Input, [in_idx])
                rocir.memref.store(v.value, Output, [out_idx])
                rocir.scf_ext.yield_([])

    m = _StridedCopy()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())


