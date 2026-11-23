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

EM_AMDGPU = 224


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


def test_kernel_names_and_code_object():
    # CPU tensors are compile placeholders, so nothing is launched.
    compiled = flyc.compile(_launch, torch.zeros(64))

    names = compiled.kernel_names()
    assert [n.rsplit("_", 1)[0] for n in names] == ["_first", "_second"]

    code = compiled.code_object()
    assert code[:4] == b"\x7fELF"
    assert int.from_bytes(code[18:20], "little") == EM_AMDGPU
    for name in names:
        assert name.encode() in code
