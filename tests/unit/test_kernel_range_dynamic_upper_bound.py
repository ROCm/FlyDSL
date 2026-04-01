"""Regression tests for range kernel with dynamic upper bound."""

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def _range_kernel(loop_count: fx.Int32):
    fx.printf("kernel loop_count={}", loop_count)
    for i in range(loop_count):
        fx.printf("helper i={}", i)


@flyc.jit
def _run_case(loop_count: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _range_kernel(loop_count).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


class TestKernelRangeDynamicUpperBound:
    @pytest.mark.parametrize("loop_count", [1, 4, 8])
    def test_range_kernel_dynamic_upper_bound(self, loop_count):
        _run_case(loop_count)
