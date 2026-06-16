#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end tests for dynamic while loop dispatch.

Each test writes results to an output tensor, then verifies the value
matches the expected result.
"""

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import buffer_ops, const_expr  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_out_tensor(n=1, dtype=torch.int32):
    t = torch.zeros(n, device="cuda", dtype=dtype)
    return t, flyc.from_torch_tensor(t).mark_layout_dynamic(leading_dim=0, divisibility=1)


# ── Case 1: simple countdown with single yield var ──────────────────────────


@flyc.kernel
def _k_while_countdown(Out: fx.Tensor, n: fx.Int32):
    offset = n
    acc = fx.Int32(0)
    while offset > fx.Int32(0):
        acc = acc + offset
        offset = offset - fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_while_countdown(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_while_countdown(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_while_simple_countdown():
    """acc = sum(n, n-1, ..., 1) = n*(n+1)/2"""
    out, t_out = _make_out_tensor()
    _j_while_countdown(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out.item() == 15  # 5+4+3+2+1


# ── Case 2: while with store inside loop (side-effect only) ─────────────────


@flyc.kernel
def _k_while_store_in_loop(Out: fx.Tensor, n: fx.Int32):
    rsrc = buffer_ops.create_buffer_resource(Out)
    offset = n
    while offset > fx.Int32(0):
        buffer_ops.buffer_store(offset, rsrc, fx.Int32(0))
        offset = offset - fx.Int32(1)


@flyc.jit
def _j_while_store_in_loop(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_while_store_in_loop(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_while_store_in_loop():
    """Each iteration overwrites Out[0] with offset; last write is 1."""
    out, t_out = _make_out_tensor()
    _j_while_store_in_loop(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out.item() == 1


# ── Case 3: while with nested dynamic if ────────────────────────────────────


@flyc.kernel
def _k_while_with_if(Out: fx.Tensor, n: fx.Int32):
    offset = n
    acc = fx.Int32(0)
    while offset > fx.Int32(0):
        if offset > fx.Int32(3):
            acc = acc + fx.Int32(10)
        else:
            acc = acc + fx.Int32(1)
        offset = offset - fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_while_with_if(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_while_with_if(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_while_with_inner_if():
    """n=5: offsets 5,4 → +10 each; offsets 3,2,1 → +1 each → 20+3=23"""
    out, t_out = _make_out_tensor()
    _j_while_with_if(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out.item() == 23


# ── Case 4: while with nested dynamic for ───────────────────────────────────


@flyc.kernel
def _k_while_with_for(Out: fx.Tensor, n: fx.Int32):
    offset = n
    acc = fx.Int32(0)
    while offset > fx.Int32(0):
        for i in range(offset):
            acc = acc + fx.Int32(1)
        offset = offset - fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_while_with_for(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_while_with_for(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_while_with_inner_for():
    """n=3: iterations 3+2+1=6 → acc==6"""
    out, t_out = _make_out_tensor()
    _j_while_with_for(t_out, fx.Int32(3))
    torch.cuda.synchronize()
    assert out.item() == 6


# ── Case 5: for with nested dynamic while ───────────────────────────────────


@flyc.kernel
def _k_for_with_while(Out: fx.Tensor, n: fx.Int32):
    acc = fx.Int32(0)
    for i in range(n):
        x = fx.Int32(4)
        while x > fx.Int32(0):
            acc = acc + fx.Int32(1)
            x = x - fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_for_with_while(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_for_with_while(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_for_with_inner_while():
    """n=3 outer loops × 4 while iterations each = 12"""
    out, t_out = _make_out_tensor()
    _j_for_with_while(t_out, fx.Int32(3))
    torch.cuda.synchronize()
    assert out.item() == 12


# ── Case 6: constexpr while (compile-time unroll) ───────────────────────────


@flyc.kernel
def _k_while_constexpr(Out: fx.Tensor, iters: fx.Constexpr[int]):
    acc = fx.Int32(0)
    i = 0
    while const_expr(i < iters):
        acc = acc + fx.Int32(1)
        i += 1
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_while_constexpr(Out: fx.Tensor, iters: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    _k_while_constexpr(Out, iters).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_while_constexpr():
    """const_expr while should unroll at compile time → acc == iters"""
    out, t_out = _make_out_tensor()
    _j_while_constexpr(t_out, 5)
    torch.cuda.synchronize()
    assert out.item() == 5
