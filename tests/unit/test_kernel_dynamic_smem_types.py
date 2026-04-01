"""Regression tests for dynamic shared memory argument typing."""

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
def _smem_probe_kernel():
    # Keep the kernel minimal: this test focuses on launch-time smem typing.
    fx.printf("[smem_probe] tid={}", fx.thread_idx.x)


@flyc.jit
def _run_with_fx_int32_smem(smem: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _smem_probe_kernel().launch(grid=(1, 1, 1), block=[1, 1, 1], smem=smem, stream=stream)


@flyc.jit
def _run_with_constexpr_smem(smem: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    _smem_probe_kernel().launch(grid=(1, 1, 1), block=[1, 1, 1], smem=smem, stream=stream)


@flyc.jit
def _run_with_python_int_smem(smem: int, stream: fx.Stream = fx.Stream(None)):
    _smem_probe_kernel().launch(grid=(1, 1, 1), block=[1, 1, 1], smem=smem, stream=stream)


class TestKernelDynamicSmemTypes:
    @pytest.mark.parametrize("smem_size", [0, 64, 128])
    def test_dynamic_smem_fx_int32(self, smem_size):
        _run_with_fx_int32_smem(smem_size)

    @pytest.mark.parametrize("smem_size", [0, 64, 128])
    def test_dynamic_smem_constexpr_int(self, smem_size):
        _run_with_constexpr_smem(smem_size)

    @pytest.mark.parametrize("smem_size", [0, 64, 128])
    def test_dynamic_smem_python_int(self, smem_size):
        _run_with_python_int_smem(smem_size)
