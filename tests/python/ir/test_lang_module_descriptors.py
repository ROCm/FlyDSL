"""Sanity test: MlirModule + @kernel/@jit re-exported under `rocir.*`."""

from rocdsl.dialects.ext import rocir


def test_mlirmodule_kernel_jit_emits_ops(ctx, insert_point):
    N = 8

    class VecAdd(rocir.MlirModule):
        @rocir.kernel
        def kernel(
            self: rocir.T.i64(),  # first argument is an MLIR scalar, not a Python instance
            A: rocir.T.memref(N, element_type=rocir.T.f32()),
            B: rocir.T.memref(N, element_type=rocir.T.f32()),
            C: rocir.T.memref(N, element_type=rocir.T.f32()),
        ):
            tid = rocir.thread_idx("x")
            a = rocir.memref.load(A, [tid.value])
            rocir.memref.store(a.value if hasattr(a, "value") else a, C, [tid.value])

        @rocir.jit
        def __call__(
            self: rocir.T.i64(),
            A: rocir.T.memref(N, element_type=rocir.T.f32()),
            B: rocir.T.memref(N, element_type=rocir.T.f32()),
            C: rocir.T.memref(N, element_type=rocir.T.f32()),
        ):
            # Just ensure this emits a func.func.
            _ = self
            return

    m = VecAdd()
    s = str(m.module)
    assert "gpu.module" in s
    assert "gpu.func" in s
    assert "func.func" in s


