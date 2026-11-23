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
def _launch(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def test_binaries_returns_code_object():
    # CPU tensors are compile placeholders, so nothing is launched.
    binaries = flyc.compile(_launch, torch.zeros(64)).binaries()
    assert len(binaries) == 1
    (blob,) = binaries.values()
    assert isinstance(blob, bytes)
    assert blob.startswith(b"\x7fELF")
