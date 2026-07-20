#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""System (device) tests: carry a Python list through dynamic if/for/while in a
real @flyc.kernel, launch it, and check the results against the expected values.

This is the end-to-end counterpart of test_dynamic_controlflow_list_carry.py.
The pattern (a list local reassigned inside a runtime-conditioned region) is the
one that previously failed to compile with
``TypeError: ... is list, not an MLIR Value``.
"""

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


def _out(n):
    out = torch.zeros(n, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    return out, t_out


# ── dynamic if carrying a list ──────────────────────────────────────────────


@flyc.kernel
def _kernel_if_list(Out: fx.Tensor, flag: fx.Int32):
    lst = [fx.Int32(1), fx.Int32(2)]
    if flag > fx.Int32(0):  # runtime condition -> dynamic scf.if
        lst = [lst[0] + fx.Int32(10), lst[1] + fx.Int32(20)]
    rsrc = fx.buffer_ops.create_buffer_resource(Out)
    fx.buffer_ops.buffer_store(lst[0], rsrc, fx.Int32(0))
    fx.buffer_ops.buffer_store(lst[1], rsrc, fx.Int32(1))


@flyc.jit
def _run_if_list(Out: fx.Tensor, flag: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_if_list(Out, flag).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── dynamic for carrying a list ─────────────────────────────────────────────


@flyc.kernel
def _kernel_for_list(Out: fx.Tensor, n: fx.Int32):
    lst = [fx.Int32(0), fx.Int32(100)]
    for i in range(n):
        lst = [lst[0] + fx.Int32(1), lst[1] - fx.Int32(1)]
    rsrc = fx.buffer_ops.create_buffer_resource(Out)
    fx.buffer_ops.buffer_store(lst[0], rsrc, fx.Int32(0))
    fx.buffer_ops.buffer_store(lst[1], rsrc, fx.Int32(1))


@flyc.jit
def _run_for_list(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_for_list(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── dynamic while carrying a list ───────────────────────────────────────────


@flyc.kernel
def _kernel_while_list(Out: fx.Tensor, n: fx.Int32):
    lst = [n, fx.Int32(0)]
    while lst[0] > fx.Int32(0):
        lst = [lst[0] - fx.Int32(1), lst[1] + fx.Int32(1)]
    rsrc = fx.buffer_ops.create_buffer_resource(Out)
    fx.buffer_ops.buffer_store(lst[0], rsrc, fx.Int32(0))
    fx.buffer_ops.buffer_store(lst[1], rsrc, fx.Int32(1))


@flyc.jit
def _run_while_list(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_while_list(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── dynamic ifexp (ternary) evaluating to a list ────────────────────────────


@flyc.kernel
def _kernel_ifexp_list(Out: fx.Tensor, flag: fx.Int32):
    lst = [fx.Int32(1), fx.Int32(2)] if flag > fx.Int32(0) else [fx.Int32(3), fx.Int32(4)]
    rsrc = fx.buffer_ops.create_buffer_resource(Out)
    fx.buffer_ops.buffer_store(lst[0], rsrc, fx.Int32(0))
    fx.buffer_ops.buffer_store(lst[1], rsrc, fx.Int32(1))


@flyc.jit
def _run_ifexp_list(Out: fx.Tensor, flag: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_ifexp_list(Out, flag).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestDynamicControlFlowListCarryDevice:
    def test_if_list_taken(self):
        out, t_out = _out(2)
        _run_if_list(t_out, fx.Int32(1))  # flag>0 -> then branch
        torch.cuda.synchronize()
        assert out[0].item() == 11 and out[1].item() == 22, out.tolist()

    def test_if_list_not_taken(self):
        out, t_out = _out(2)
        _run_if_list(t_out, fx.Int32(0))  # flag==0 -> else keeps original list
        torch.cuda.synchronize()
        assert out[0].item() == 1 and out[1].item() == 2, out.tolist()

    def test_for_list(self):
        out, t_out = _out(2)
        _run_for_list(t_out, fx.Int32(5))  # +1/-1 x5 -> [5, 95]
        torch.cuda.synchronize()
        assert out[0].item() == 5 and out[1].item() == 95, out.tolist()

    def test_while_list(self):
        out, t_out = _out(2)
        _run_while_list(t_out, fx.Int32(7))  # drain 7 -> [0, 7]
        torch.cuda.synchronize()
        assert out[0].item() == 0 and out[1].item() == 7, out.tolist()

    def test_ifexp_list_taken(self):
        out, t_out = _out(2)
        _run_ifexp_list(t_out, fx.Int32(1))  # flag>0 -> [1, 2]
        torch.cuda.synchronize()
        assert out[0].item() == 1 and out[1].item() == 2, out.tolist()

    def test_ifexp_list_not_taken(self):
        out, t_out = _out(2)
        _run_ifexp_list(t_out, fx.Int32(0))  # else -> [3, 4]
        torch.cuda.synchronize()
        assert out[0].item() == 3 and out[1].item() == 4, out.tolist()
