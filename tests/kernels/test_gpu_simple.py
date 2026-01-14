"""
Simple GPU kernel tests using flir Python API
Vector addition test with clean, readable syntax
"""

from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext import flir, memref, arith
from _mlir import ir
import _mlir.extras.types as T


def test_vector_add():
    """Vector addition test: C = A + B"""
    M, N = 32, 64
    S = ir.ShapedType.get_dynamic_size()

    class _VecAdd(flir.MlirModule):
        @flir.kernel
        def vecAdd(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, T.f32()),
            B: lambda: T.memref(S, S, T.f32()),
            C: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            # Get block/thread IDs and dimensions
            bx, by = flir.block_idx("x"), flir.block_idx("y")
            tx, ty = flir.thread_idx("x"), flir.thread_idx("y")
            bdx, bdy = flir.block_dim("x"), flir.block_dim("y")

            # Calculate global thread index
            row = (by * bdy + ty)
            col = (bx * bdx + tx)

            # Vector addition with bounds guard so one compiled kernel can be reused for any M.
            m_v = arith.ArithValue(m_in)
            n_v = arith.ArithValue(n_in)
            valid = (arith.ArithValue(row) < m_v) & (arith.ArithValue(col) < n_v)
            if valid:
                a = memref.load(A, [row, col])
                b = memref.load(B, [row, col])
                c = a + b
                memref.store(c, C, [row, col])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, T.f32()),
            B: lambda: T.memref(S, S, T.f32()),
            C: lambda: T.memref(S, S, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
        ):
            c1 = arith.index(1)
            b16 = arith.index(16)
            n_v = arith.ArithValue(n_in)
            m_v = arith.ArithValue(m_in)
            bx = arith.as_value((n_v + arith.index(15)) // b16)
            by = arith.as_value((m_v + arith.index(15)) // b16)
            flir.gpu_ext.LaunchFuncOp(
                ["kernels", "vecAdd"],
                grid_size=(bx, by, c1),
                block_size=(b16, b16, c1),
                kernel_operands=[A, B, C, m_in, n_in],
            )

    m = _VecAdd()
    assert m.module.operation.verify()
    run_pipeline(m.module, Pipeline().canonicalize().cse())
