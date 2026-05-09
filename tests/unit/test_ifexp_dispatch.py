#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for IfExp (ternary) → scf.if dispatch."""

import pytest

from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr.numeric import Int32


def test_ifexp_static_true_no_scf_if():
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_static_true", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    True,
                    lambda: Int32(42),
                    lambda: Int32(99),
                )
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" not in str(module)


def test_ifexp_static_false_no_scf_if():
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_static_false", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    False,
                    lambda: Int32(42),
                    lambda: Int32(99),
                )
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" not in str(module)


def test_ifexp_dynamic_builds_scf_if():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_dynamic", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    cond,
                    lambda: Int32(arith.ConstantOp(i32, 42).result),
                    lambda: Int32(arith.ConstantOp(i32, 99).result),
                )
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.if" in ir_text
        assert "-> (i32)" in ir_text


def test_ifexp_dynamic_type_mismatch_raises():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_type_mismatch", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                with pytest.raises(TypeError, match="type mismatch"):
                    ReplaceIfWithDispatch.scf_ifexp_dispatch(
                        cond,
                        lambda: Int32(arith.ConstantOp(i32, 1).result),
                        lambda: arith.ConstantOp(i64, 2).result,
                    )
