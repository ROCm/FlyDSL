#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Divisibility of integers: Int32/Int64 arithmetic follows the Core int_tuple leaf rules, a divisibility
declared with ``fx.Int32(x, divisibility=...)`` at a jit call site reaches the jit function and its kernels,
and the fact travels into and out of int_tuple leaves and across dynamic ifs."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func
from flydsl.expr.numeric import Int32, Int64


@pytest.mark.l0_backend_agnostic
def test_arithmetic_follows_core_leaf_rules():
    def build(x):
        assert x.divisibility == 1 and str(x) == "?"
        # A swizzled row offset, written in either order.
        assert (x * 256 + (x ^ 3) * 16).divisibility == 16
        assert (16 * (3 ^ x) + 256 * x).divisibility == 16
        x12 = x * 12
        assert str(x12) == "?{div=12}"
        assert (x12 // 4).divisibility == 3
        assert (x12 % 8).divisibility == 4
        assert (x12 & 0xFF0).divisibility == 16
        assert ((x * 8) << 2).divisibility == 32 and ((x * 16) >> 2).divisibility == 4
        assert (-x12).divisibility == 12 and Int64(x12).divisibility == 12
        assert (x > 0).select(x * 16, x * 24).divisibility == 8
        assert Int32(x12, divisibility=64).divisibility == 192 and x12.divisibility == 12

    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f = func.FuncOp("test", ir.FunctionType.get([ir.IntegerType.get_signless(32)], []))
            with ir.InsertionPoint(f.add_entry_block()):
                build(Int32(f.entry_block.arguments[0]))
                func.ReturnOp([])


@pytest.mark.l0_backend_agnostic
def test_declared_divisibility_on_host():
    n = Int32(48, divisibility=16)
    assert n.__cache_signature__() == (Int32, 16) and Int32(48).__cache_signature__() == (Int32,)
    with pytest.raises(ValueError):
        Int32(40, divisibility=16)
    with ir.Context(), pytest.raises(TypeError):
        fx.assume(n, 16)  # integers declare it on construction; assume takes int_tuple leaves


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_divisibility_through_calls_int_tuples_and_dynamic_ifs(monkeypatch):
    for name, value in {"ARCH": "gfx942", "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"}.items():
        monkeypatch.setenv(name, value)
    seen = {}

    @flyc.kernel
    def kern(x: fx.Tensor, n: fx.Int32, m: fx.Int32):
        seen["kernel arg"] = m.divisibility
        tid = fx.Int32(fx.thread_idx.x)
        t16 = fx.make_int_tuple(tid * 16)
        seen["leaves"] = (fx.get_scalar(t16).divisibility, t16.to_py_value().divisibility)
        seen["repacked"] = (str(fx.make_int_tuple((t16, 4)).type), str(fx.make_layout((4, 4), (t16, 1)).type))
        seen["assumed leaf"] = str(fx.assume(t16, 64).type)
        lds = fx.SharedAllocator().allocate(4096, alignment=16).peek().ptr
        seen["ptr"] = (lds + ((tid // 8) * 256 + (tid % 8 ^ tid // 8) * 16)).type.alignment
        off = tid * 16
        if n > 0:
            off = tid * 48
        seen["if"] = off.divisibility
        seen["ifexp"] = (tid * 32 if n > 1 else tid * 8).divisibility
        acc = tid * 16
        for _ in range(n):
            acc = acc + 16
        # A loop-carried value is fixed before the body is traced, so nothing is known about it.
        seen["carried"] = acc.divisibility

    @flyc.jit
    def launch(x: fx.Tensor, n: fx.Int32, m: fx.Int32):
        seen["jit arg"] = (n.divisibility, m.divisibility)
        kern(x, n, m).launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch(torch.empty(64, dtype=torch.int32), 3, fx.Int32(32, divisibility=16))
    assert seen == {
        "jit arg": (1, 16),
        "kernel arg": 16,
        "leaves": (16, 16),
        "repacked": ("!fly.int_tuple<(?{div=16},4)>", "!fly.layout<(4,4):(?{div=16},1)>"),
        "assumed leaf": "!fly.int_tuple<?{div=64}>",
        "ptr": 16,
        "if": 16,
        "ifexp": 8,
        "carried": 1,
    }
