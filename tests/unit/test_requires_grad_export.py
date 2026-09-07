#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""requires_grad tensors must wrap through from_dlpack / from_torch_tensor.

PyTorch refuses ``tensor.__dlpack__()`` when ``requires_grad`` is set
(``BufferError``). FlyDSL kernels are forward-only, so the wrap path detaches
(same storage, no copy) before export. aiter calls ``flyc.from_dlpack``
directly and never hits the ``@jit`` auto-adapt path.
"""

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TorchTensorJitArg

pytestmark = pytest.mark.l0_backend_agnostic


def _parameter():
    return torch.nn.Parameter(torch.empty(8, dtype=torch.float32), requires_grad=True)


def _fill_data_ptr(arg, live):
    ctype, fill = arg.__c_abi_spec__()[0]
    slot = ctype()
    fill(live, slot)
    return int(slot.value)


def test_from_dlpack_parameter_requires_grad_shares_storage():
    p = _parameter()
    arg = flyc.from_dlpack(p)
    assert not arg.dltensor.requires_grad
    assert arg.dltensor.data_ptr() == p.data_ptr()
    assert p.requires_grad


def test_from_dlpack_launch_fill_accepts_requires_grad_tensor():
    """Compiled kernels may receive a live Parameter, not the wrap-time object."""
    template = flyc.from_dlpack(torch.empty(8, dtype=torch.float32)).mark_layout_dynamic(leading_dim=0)
    live = _parameter()
    assert _fill_data_ptr(template, live) == live.data_ptr()
    assert live.requires_grad


def test_from_torch_tensor_parameter_requires_grad_shares_storage():
    p = _parameter()
    arg = flyc.from_torch_tensor(p)
    assert not arg.torch_tensor.requires_grad
    assert arg.torch_tensor.data_ptr() == p.data_ptr()
    assert p.requires_grad


def test_auto_adapt_parameter_requires_grad_shares_storage():
    p = _parameter()
    arg = TorchTensorJitArg(p)
    assert not arg.torch_tensor.requires_grad
    assert arg.torch_tensor.data_ptr() == p.data_ptr()
    assert p.requires_grad
