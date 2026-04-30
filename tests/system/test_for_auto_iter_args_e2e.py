#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end tests for automatic for-loop iter_args inference.

Each test modifies variables inside a for loop and writes the result
to an output tensor, then verifies the value matches the expected result.
"""

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_out_tensor(n=1, dtype=torch.int32):
    t = torch.zeros(n, device="cuda", dtype=dtype)
    return t, flyc.from_dlpack(t).mark_layout_dynamic(leading_dim=0, divisibility=1)


# ── Case 1: single accumulator ───────────────────────────────────────────────


@flyc.kernel
def _k_single_acc(Out: fx.Tensor, n: fx.Int32):
    acc = fx.Int32(0)
    for i in range(n):
        acc = acc + fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_single_acc(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_single_acc(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_single_acc_result():
    """acc = 0; for i in range(5): acc += 1  →  acc == 5"""
    out, t_out = _make_out_tensor()
    _j_single_acc(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out.item() == 5


# ── Case 2: multiple variables ───────────────────────────────────────────────


@flyc.kernel
def _k_multi_vars(OutA: fx.Tensor, OutB: fx.Tensor, n: fx.Int32):
    a = fx.Int32(0)
    b = fx.Int32(100)
    for i in range(n):
        a = a + fx.Int32(1)
        b = b - fx.Int32(1)
    rsrc_a = buffer_ops.create_buffer_resource(OutA)
    rsrc_b = buffer_ops.create_buffer_resource(OutB)
    buffer_ops.buffer_store(a, rsrc_a, fx.Int32(0))
    buffer_ops.buffer_store(b, rsrc_b, fx.Int32(0))


@flyc.jit
def _j_multi_vars(OutA: fx.Tensor, OutB: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_multi_vars(OutA, OutB, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_multi_vars_result():
    """a=0, b=100; for i in range(3): a+=1, b-=1  →  a==3, b==97"""
    out_a, t_a = _make_out_tensor()
    out_b, t_b = _make_out_tensor()
    _j_multi_vars(t_a, t_b, fx.Int32(3))
    torch.cuda.synchronize()
    assert out_a.item() == 3
    assert out_b.item() == 97


# ── Case 3: range with step ─────────────────────────────────────────────────


@flyc.kernel
def _k_range_step(Out: fx.Tensor, n: fx.Int32):
    acc = fx.Int32(0)
    for i in range(fx.Int32(0), n, fx.Int32(2)):
        acc = acc + fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_range_step(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_range_step(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_range_step_result():
    """range(0, 10, 2) → 5 iterations → acc == 5"""
    out, t_out = _make_out_tensor()
    _j_range_step(t_out, fx.Int32(10))
    torch.cuda.synchronize()
    assert out.item() == 5


# ── Case 4: accumulate expression (acc = acc * 2 + 1) ───────────────────────


@flyc.kernel
def _k_acc_expr(Out: fx.Tensor, n: fx.Int32):
    acc = fx.Int32(1)
    for i in range(n):
        acc = acc * fx.Int32(2)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))


@flyc.jit
def _j_acc_expr(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_acc_expr(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_acc_expr_result():
    """acc = 1; for i in range(4): acc *= 2  →  acc == 16"""
    out, t_out = _make_out_tensor()
    _j_acc_expr(t_out, fx.Int32(4))
    torch.cuda.synchronize()
    assert out.item() == 16


# ── Case 5: zero iterations ─────────────────────────────────────────────────


def test_zero_iterations_preserves_init():
    """range(0) → 0 iterations, acc stays at initial value."""
    out, t_out = _make_out_tensor()
    # Pre-fill with 42 to verify it gets overwritten by the kernel
    out.fill_(99)
    _j_single_acc(t_out, fx.Int32(0))
    torch.cuda.synchronize()
    assert out.item() == 0


# ── Case 6: per-thread accumulation (multi-thread) ──────────────────────────


@flyc.kernel
def _k_per_thread_acc(Out: fx.Tensor, n: fx.Int32, block_dim: fx.Constexpr[int]):
    tid = fx.thread_idx.x
    acc = fx.Int32(0)
    for i in range(n):
        acc = acc + fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(acc, rsrc, tid)


@flyc.jit
def _j_per_thread_acc(Out: fx.Tensor, n: fx.Int32, block_dim: fx.Constexpr[int],
                      stream: fx.Stream = fx.Stream(None)):
    _k_per_thread_acc(Out, n, block_dim).launch(
        grid=(1, 1, 1), block=(block_dim, 1, 1), stream=stream.value
    )


def test_per_thread_acc_result():
    """Each thread independently accumulates → all elements == n."""
    BLOCK = 64
    N = 7
    out, t_out = _make_out_tensor(n=BLOCK)
    _j_per_thread_acc(t_out, fx.Int32(N), BLOCK)
    torch.cuda.synchronize()
    expected = torch.full((BLOCK,), N, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# ── Case 7: loop variable liveout (i initialized before loop) ───────────────


@flyc.kernel
def _k_iv_liveout(Out: fx.Tensor, n: fx.Int32):
    i = fx.Int32(999)
    acc = fx.Int32(0)
    for i in range(n):
        acc = acc + fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(i, rsrc, fx.Int32(0))
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(1))


@flyc.jit
def _j_iv_liveout(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_iv_liveout(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_iv_liveout():
    """i=999 before loop; after for i in range(5): i should be 4 (last iteration)."""
    out, t_out = _make_out_tensor(n=2)
    _j_iv_liveout(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out[0].item() == 4, f"expected i=4, got {out[0].item()}"
    assert out[1].item() == 5, f"expected acc=5, got {out[1].item()}"


def test_iv_liveout_zero_iterations():
    """i=999 before loop; range(0) → 0 iterations, i should stay 999."""
    out, t_out = _make_out_tensor(n=2)
    _j_iv_liveout(t_out, fx.Int32(0))
    torch.cuda.synchronize()
    assert out[0].item() == 999, f"expected i=999, got {out[0].item()}"
    assert out[1].item() == 0, f"expected acc=0, got {out[1].item()}"
