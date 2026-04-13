"""Regression tests for range kernel with dynamic upper bound."""

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def _range_kernel(loop_count: fx.Int32, out: fx.Tensor):
    out[0] = fx.Int32(0)
    for i in range(loop_count):
        out[0] = fx.arith.index_cast(fx.T.i32(), i + 1)


@flyc.jit
def _run_case(loop_count: fx.Int32, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _range_kernel(loop_count, out).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


class TestKernelRangeDynamicUpperBound:
    @pytest.mark.parametrize("loop_count", [1, 4, 8])
    def test_range_kernel_dynamic_upper_bound(self, loop_count):
        out = torch.full((1,), -1, device="cuda", dtype=torch.int32)
        _run_case(loop_count, out)
        torch.cuda.synchronize()
        assert out.item() == loop_count
