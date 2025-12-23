"""
Simple GPU kernel tests using rocdsl Python API
Vector addition test with clean, readable syntax
"""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import rocir
import _mlir.extras.types as T


def test_vector_add():
    """Vector addition test: C = A + B"""
    M, N = 32, 64

    class _VecAdd(rocir.MlirModule):
        @rocir.kernel
        def vecAdd(
            self: rocir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(M, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            # Get block/thread IDs and dimensions
            bx, by = rocir.block_idx("x"), rocir.block_idx("y")
            tx, ty = rocir.thread_idx("x"), rocir.thread_idx("y")
            bdx, bdy = rocir.block_dim("x"), rocir.block_dim("y")

            # Calculate global thread index
            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            # Vector addition: C[row,col] = A[row,col] + B[row,col]
            a = rocir.memref.load(A, [row.value, col.value])
            b = rocir.memref.load(B, [row.value, col.value])
            c = a + b
            rocir.memref.store(c.value, C, [row.value, col.value])

        @rocir.jit
        def __call__(
            self: rocir.T.i64,
            A: lambda: T.memref(M, N, T.f32()),
            B: lambda: T.memref(M, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            c1 = rocir.arith_ext.index(1).value
            bx = rocir.arith_ext.index(N // 16).value
            by = rocir.arith_ext.index(M // 16).value
            b16 = rocir.arith_ext.index(16).value
            rocir.gpu_ext.LaunchFuncOp(
                ["kernels", "vecAdd"],
                grid_size=(bx, by, c1),
                block_size=(b16, b16, c1),
                kernel_operands=[A, B, C],
            )

    m = _VecAdd()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())
