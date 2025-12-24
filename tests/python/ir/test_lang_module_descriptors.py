"""Sanity test: MlirModule + @kernel/@jit re-exported under `flir.*`."""

from pyflir.dialects.ext import flir


def test_mlirmodule_kernel_jit_emits_ops(ctx, insert_point):
    N = 8

    class VecAdd(flir.MlirModule):
        @flir.kernel
        def kernel(
            self: flir.T.i64(),  # first argument is an MLIR scalar, not a Python instance
            A: flir.T.memref(N, element_type=flir.T.f32()),
            B: flir.T.memref(N, element_type=flir.T.f32()),
            C: flir.T.memref(N, element_type=flir.T.f32()),
        ):
            tid = flir.thread_idx("x")
            a = flir.memref.load(A, [tid.value])
            flir.memref.store(a.value if hasattr(a, "value") else a, C, [tid.value])

        @flir.jit
        def __call__(
            self: flir.T.i64(),
            A: flir.T.memref(N, element_type=flir.T.f32()),
            B: flir.T.memref(N, element_type=flir.T.f32()),
            C: flir.T.memref(N, element_type=flir.T.f32()),
        ):
            # Just ensure this emits a func.func.
            _ = self
            return

    m = VecAdd()
    s = str(m.module)
    assert "gpu.module" in s
    assert "gpu.func" in s
    assert "func.func" in s


