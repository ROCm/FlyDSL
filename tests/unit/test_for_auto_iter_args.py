#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for automatic iter_args inference in for loops."""

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

# ── Case 1: single iter_arg ──────────────────────────────────────────────────


@flyc.kernel
def _kernel_simple_acc(n: fx.Int32):
    acc = fx.Int32(0)
    for i in range(n):
        acc = acc + fx.Int32(1)
    fx.printf("acc={}", acc)


@flyc.jit
def _run_simple_acc(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_simple_acc(n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Case 2: multiple iter_args ───────────────────────────────────────────────


@flyc.kernel
def _kernel_multi_vars(n: fx.Int32):
    a = fx.Int32(0)
    b = fx.Int32(100)
    for i in range(n):
        a = a + fx.Int32(1)
        b = b - fx.Int32(1)
    fx.printf("a={} b={}", a, b)


@flyc.jit
def _run_multi_vars(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_multi_vars(n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Case 3: no iter_args (side-effect only loop) ─────────────────────────────


@flyc.kernel
def _kernel_no_iter_args(n: fx.Int32):
    for i in range(n):
        fx.printf("i={}", i)


@flyc.jit
def _run_no_iter_args(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_no_iter_args(n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Case 4: range with start, stop, step ─────────────────────────────────────


@flyc.kernel
def _kernel_range_3args(n: fx.Int32):
    acc = fx.Int32(0)
    for i in range(fx.Int32(0), n, fx.Int32(2)):
        acc = acc + fx.Int32(1)
    fx.printf("acc={}", acc)


@flyc.jit
def _run_range_3args(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_range_3args(n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Case 5: iv liveout (i initialized before loop, used after) ──────────────


@flyc.kernel
def _kernel_iv_liveout(Out: fx.Tensor, n: fx.Int32):
    i = fx.Int32(999)
    acc = fx.Int32(0)
    for i in range(n):
        acc = acc + fx.Int32(1)
    rsrc = fx.buffer_ops.create_buffer_resource(Out)
    fx.buffer_ops.buffer_store(i, rsrc, fx.Int32(0))
    fx.buffer_ops.buffer_store(acc, rsrc, fx.Int32(1))


@flyc.jit
def _run_iv_liveout(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_iv_liveout(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Case 6: iv assigned to iter_arg (count = iv) ───────────────────────────


@flyc.kernel
def _kernel_iv_assign(Out: fx.Tensor, start: fx.Int32, stop: fx.Int32):
    count = fx.Int32(0)
    for iv in range(start, stop):
        count = iv
    rsrc = fx.buffer_ops.create_buffer_resource(Out)
    fx.buffer_ops.buffer_store(count, rsrc, fx.Int32(0))


@flyc.jit
def _run_iv_assign(Out: fx.Tensor, start: fx.Int32, stop: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _kernel_iv_assign(Out, start, stop).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestForAutoIterArgs:
    def test_simple_acc(self):
        _run_simple_acc(fx.Int32(4))
        torch.cuda.synchronize()

    def test_multi_vars(self):
        _run_multi_vars(fx.Int32(3))
        torch.cuda.synchronize()

    def test_no_iter_args(self):
        _run_no_iter_args(fx.Int32(3))
        torch.cuda.synchronize()

    def test_range_3args(self):
        _run_range_3args(fx.Int32(8))
        torch.cuda.synchronize()

    def test_iv_liveout(self):
        out = torch.zeros(2, device="cuda", dtype=torch.int32)
        t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
        _run_iv_liveout(t_out, fx.Int32(5))
        torch.cuda.synchronize()
        assert out[0].item() == 4, f"iv liveout: expected 4, got {out[0].item()}"
        assert out[1].item() == 5, f"acc: expected 5, got {out[1].item()}"

    def test_iv_assign(self):
        out = torch.zeros(1, device="cuda", dtype=torch.int32)
        t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
        _run_iv_assign(t_out, fx.Int32(0), fx.Int32(10))
        torch.cuda.synchronize()
        assert out[0].item() == 9, f"iv assign: expected 9, got {out[0].item()}"
