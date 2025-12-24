"""
Simple GPU kernel tests using flir Python API
Vector addition test with clean, readable syntax
"""

from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.dialects.ext import flir
import _mlir.extras.types as T


def test_vector_add():
    """Vector addition test: C = A + B"""
    M, N = 32, 64

    class _VecAdd(flir.MlirModule):
        @flir.kernel
        def vecAdd(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(M, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            # Get block/thread IDs and dimensions
            bx, by = flir.block_idx("x"), flir.block_idx("y")
            tx, ty = flir.thread_idx("x"), flir.thread_idx("y")
            bdx, bdy = flir.block_dim("x"), flir.block_dim("y")

            # Calculate global thread index
            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            # Vector addition: C[row,col] = A[row,col] + B[row,col]
            a = flir.memref.load(A, [row.value, col.value])
            b = flir.memref.load(B, [row.value, col.value])
            c = a + b
            flir.memref.store(c.value, C, [row.value, col.value])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(M, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            c1 = flir.arith_ext.index(1).value
            bx = flir.arith_ext.index(N // 16).value
            by = flir.arith_ext.index(M // 16).value
            b16 = flir.arith_ext.index(16).value
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "vecAdd"],
                grid_size=(bx, by, c1),
                block_size=(b16, b16, c1),
                kernel_operands=[A, B, C],
            )

    m = _VecAdd()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())
