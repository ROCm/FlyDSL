#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for automatic iter_args inference in for loops."""

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
