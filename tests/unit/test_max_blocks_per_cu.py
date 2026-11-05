# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def _kernel(x: fx.Tensor):
    pass


@flyc.jit
def _launch(x: fx.Tensor, smem: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    _kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1), smem=smem, stream=stream)


def test_max_blocks_per_cu():
    # CPU tensors are compile placeholders, so nothing is launched.
    free = flyc.compile(_launch, torch.zeros(64), 0).max_blocks_per_cu()
    lds_bound = flyc.compile(_launch, torch.zeros(64), 48 * 1024).max_blocks_per_cu()
    assert 1 <= lds_bound < free
