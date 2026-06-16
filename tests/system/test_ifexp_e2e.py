#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
IfExp (ternary) end-to-end tests — verify that Python ternary expressions
compile through scf.if and produce correct GPU results for both static
and dynamic conditions.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def test_ifexp_static_cond_true(monkeypatch):
    """Static-true ternary: ``42 if Int32(1) > Int32(0) else 99`` → 42."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def ifexp_true_kernel(Out: fx.Tensor):
        a = fx.Int32(42) if fx.Int32(1) > fx.Int32(0) else fx.Int32(99)
        rsrc = fx.buffer_ops.create_buffer_resource(Out)
        fx.buffer_ops.buffer_store(a, rsrc, fx.Int32(0))

    @flyc.jit
    def ifexp_true_launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        ifexp_true_kernel(Out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    ifexp_true_launch(t_out)
    torch.cuda.synchronize()
    assert out[0].item() == 42, f"expected 42, got {out[0].item()}"
    print(f"[PASS] ifexp static true: out={out[0].item()}")


def test_ifexp_static_cond_false(monkeypatch):
    """Static-false ternary: ``42 if Int32(0) > Int32(1) else 99`` → 99."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def ifexp_false_kernel(Out: fx.Tensor):
        a = fx.Int32(42) if fx.Int32(0) > fx.Int32(1) else fx.Int32(99)
        rsrc = fx.buffer_ops.create_buffer_resource(Out)
        fx.buffer_ops.buffer_store(a, rsrc, fx.Int32(0))

    @flyc.jit
    def ifexp_false_launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        ifexp_false_kernel(Out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    ifexp_false_launch(t_out)
    torch.cuda.synchronize()
    assert out[0].item() == 99, f"expected 99, got {out[0].item()}"
    print(f"[PASS] ifexp static false: out={out[0].item()}")


def test_ifexp_dynamic_cond_true(monkeypatch):
    """Dynamic ternary with true condition: ``x+10 if x>0 else x-10``, x=5 → 15."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def ifexp_dyn_kernel(Out: fx.Tensor, x: fx.Int32):
        a = x + fx.Int32(10) if x > fx.Int32(0) else x - fx.Int32(10)
        rsrc = fx.buffer_ops.create_buffer_resource(Out)
        fx.buffer_ops.buffer_store(a, rsrc, fx.Int32(0))

    @flyc.jit
    def ifexp_dyn_launch(Out: fx.Tensor, x: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        ifexp_dyn_kernel(Out, x).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    ifexp_dyn_launch(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out[0].item() == 15, f"expected 15 (5+10), got {out[0].item()}"
    print(f"[PASS] ifexp dynamic true: x=5, out={out[0].item()}")


def test_ifexp_dynamic_cond_false(monkeypatch):
    """Dynamic ternary with false condition: ``x+10 if x>0 else x-10``, x=-3 → -13."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def ifexp_dyn_kernel(Out: fx.Tensor, x: fx.Int32):
        a = x + fx.Int32(10) if x > fx.Int32(0) else x - fx.Int32(10)
        rsrc = fx.buffer_ops.create_buffer_resource(Out)
        fx.buffer_ops.buffer_store(a, rsrc, fx.Int32(0))

    @flyc.jit
    def ifexp_dyn_launch(Out: fx.Tensor, x: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        ifexp_dyn_kernel(Out, x).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    ifexp_dyn_launch(t_out, fx.Int32(-3))
    torch.cuda.synchronize()
    assert out[0].item() == -13, f"expected -13 (-3-10), got {out[0].item()}"
    print(f"[PASS] ifexp dynamic false: x=-3, out={out[0].item()}")


def test_ifexp_nested(monkeypatch):
    """Nested IfExp: ``1 if (x > 0 if flag > 0 else x < 0) else 0``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def ifexp_nested_kernel(Out: fx.Tensor, x: fx.Int32, flag: fx.Int32):
        cond = (x > fx.Int32(0)) if (flag > fx.Int32(0)) else (x < fx.Int32(0))
        a = fx.Int32(1) if cond else fx.Int32(0)
        rsrc = fx.buffer_ops.create_buffer_resource(Out)
        fx.buffer_ops.buffer_store(a, rsrc, fx.Int32(0))

    @flyc.jit
    def ifexp_nested_launch(Out: fx.Tensor, x: fx.Int32, flag: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        ifexp_nested_kernel(Out, x, flag).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)

    ifexp_nested_launch(t_out, fx.Int32(5), fx.Int32(1))
    torch.cuda.synchronize()
    assert out[0].item() == 1, f"expected 1 (x=5, flag=1 → x>0 true), got {out[0].item()}"

    ifexp_nested_launch(t_out, fx.Int32(5), fx.Int32(-1))
    torch.cuda.synchronize()
    assert out[0].item() == 0, f"expected 0 (x=5, flag=-1 → x<0 false), got {out[0].item()}"
    print("[PASS] ifexp nested")


def test_ifexp_in_for_loop(monkeypatch):
    """IfExp inside a for-loop with loop-carried accumulator."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def ifexp_loop_kernel(Out: fx.Tensor, x: fx.Int32):
        acc = fx.Int32(0)
        for i in range(fx.Int32(4)):
            acc = acc + fx.Int32(1) if x > fx.Int32(0) else acc - fx.Int32(1)
        rsrc = fx.buffer_ops.create_buffer_resource(Out)
        fx.buffer_ops.buffer_store(acc, rsrc, fx.Int32(0))

    @flyc.jit
    def ifexp_loop_launch(Out: fx.Tensor, x: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        ifexp_loop_kernel(Out, x).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)

    ifexp_loop_launch(t_out, fx.Int32(5))
    torch.cuda.synchronize()
    assert out[0].item() == 4, f"expected 4 (4 iterations of +1), got {out[0].item()}"

    ifexp_loop_launch(t_out, fx.Int32(-3))
    torch.cuda.synchronize()
    assert out[0].item() == -4, f"expected -4 (4 iterations of -1), got {out[0].item()}"
    print("[PASS] ifexp in for loop")
