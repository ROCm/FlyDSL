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
def _first(x: fx.Tensor):
    pass


@flyc.kernel
def _second(x: fx.Tensor):
    pass


@flyc.jit
def _launch(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _first(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    _second(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    _first(x).launch(grid=(2, 1, 1), block=(64, 1, 1), stream=stream)


def test_kernel_names():
    # CPU tensors are compile placeholders, so nothing is launched.
    compiled = flyc.compile(_launch, torch.zeros(64))

    # One symbol per launch site: the second launch of _first is its own kernel.
    names = compiled.kernel_names()
    assert [n.rsplit("_", 1)[0] for n in names] == ["_first", "_second", "_first"]
    assert len(set(names)) == 3

    (code,) = compiled.binaries().values()
    for name in names:
        assert name.encode() in code
