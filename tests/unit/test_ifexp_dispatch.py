#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for IfExp (ternary) → scf.if dispatch."""

import pytest

from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, F16Type, F32Type, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr.numeric import Float16, Float32, Int32


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


def test_ifexp_dynamic_builds_arith_select():
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
        assert "arith.select" in ir_text
        assert "scf.if" not in ir_text


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


def test_ifexp_nested_condition():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_nested", FunctionType.get([i1, i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond_outer = entry.arguments[0]
                cond_inner = entry.arguments[1]
                inner = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    cond_inner,
                    lambda: Int32(arith.ConstantOp(i32, 42).result),
                    lambda: Int32(arith.ConstantOp(i32, 99).result),
                )
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    cond_outer,
                    lambda: inner,
                    lambda: Int32(arith.ConstantOp(i32, 0).result),
                )
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert ir_text.count("arith.select") == 2
        assert "scf.if" not in ir_text


def test_ifexp_dynamic_float32():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            f = func.FuncOp("test_f32", FunctionType.get([i1], [f32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    cond,
                    lambda: Float32(arith.ConstantOp(f32, 1.0).result),
                    lambda: Float32(arith.ConstantOp(f32, 2.0).result),
                )
                assert isinstance(out, Float32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "arith.select" in ir_text
        assert "scf.if" not in ir_text


def test_ifexp_dynamic_float16():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        f16 = F16Type.get()
        with InsertionPoint(module.body):
            f = func.FuncOp("test_f16", FunctionType.get([i1], [f16]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    cond,
                    lambda: Float16(arith.ConstantOp(f16, 1.0).result),
                    lambda: Float16(arith.ConstantOp(f16, 2.0).result),
                )
                assert isinstance(out, Float16)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "arith.select" in ir_text
        assert "scf.if" not in ir_text
