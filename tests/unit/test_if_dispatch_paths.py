#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ast

import pytest

from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl._mlir.dialects import arith, func
from flydsl.compiler.ast_rewriter import ASTRewriter, ReplaceIfWithDispatch
from flydsl.expr.numeric import Int32


def test_collect_assigned_vars_supports_tuple_and_augassign():
    code = """
a, (b, c) = foo()
d += 1
"""
    stmts = ast.parse(code).body
    node = ast.If(test=ast.Constant(value=True), body=stmts, orelse=[])
    active_symbols = [{"a", "b", "c", "d"}]
    assigned = ReplaceIfWithDispatch._collect_assigned_vars(node, active_symbols)
    assert assigned == ["a", "b", "c", "d"]


def test_collect_assigned_vars_supports_annassign_walrus_with_except_for():
    code = """
x: int = 1
for i in range(4):
    y = i
with ctx() as w:
    z = w
try:
    pass
except Exception as e:
    err = e
if (n := foo()):
    out = n
"""
    stmts = ast.parse(code).body
    node = ast.If(test=ast.Constant(value=True), body=stmts, orelse=[])
    active_symbols = [{"x", "i", "y", "w", "z", "e", "err", "n", "out"}]
    assigned = ReplaceIfWithDispatch._collect_assigned_vars(node, active_symbols)
    assert assigned == ["x", "i", "y", "w", "z", "err", "n", "out"]


def test_scf_if_dispatch_static_with_states_no_ifop():
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_static_states", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                x = Int32(1)

                def then_fn(x):
                    return {"x": Int32(42)}

                def else_fn(x):
                    return {"x": Int32(99)}

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    True,
                    then_fn,
                    else_fn,
                    state_names=("x",),
                    state_values=(x,),
                )
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" not in str(module)


def test_scf_if_dispatch_dynamic_with_states_build_ifop():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_dynamic_states", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                x = Int32(arith.ConstantOp(i32, 1).result)

                def then_fn(x):
                    return {"x": Int32(arith.ConstantOp(i32, 42).result)}

                def else_fn(x):
                    return {"x": Int32(arith.ConstantOp(i32, 99).result)}

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    then_fn,
                    else_fn,
                    state_names=("x",),
                    state_values=(x,),
                )
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.if" in ir_text
        assert "-> (i32)" in ir_text


def test_scf_if_dispatch_dynamic_type_mismatch_has_clear_error():
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_dynamic_type_mismatch", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                x = Int32(arith.ConstantOp(i32, 1).result)

                def then_fn(x):
                    return {"x": arith.ConstantOp(i32, 2).result}

                def else_fn(x):
                    return {"x": arith.ConstantOp(i64, 3).result}

                with pytest.raises(TypeError, match="type mismatch|mismatched types"):
                    ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        then_fn,
                        else_fn,
                        state_names=("x",),
                        state_values=(x,),
                    )


def test_scf_if_dispatch_dynamic_non_mlir_value_is_promoted():
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_dynamic_non_mlir", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                x = Int32(arith.ConstantOp(i32, 1).result)

                def then_fn(x):
                    return {"x": 7}

                def else_fn(x):
                    return {"x": arith.ConstantOp(i32, 3).result}

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    then_fn,
                    else_fn,
                    state_names=("x",),
                    state_values=(x,),
                )
                assert isinstance(out, Int32)


def test_ast_rewrite_keeps_semantics_for_static_bool():
    called = {"n": 0}

    def sample(flag):
        x = 1
        if flag:
            x = 2
        else:
            x = 3
        return x

    ASTRewriter.transform(sample)
    original_dispatch = sample.__globals__["scf_if_dispatch"]

    def traced_dispatch(*args, **kwargs):
        called["n"] += 1
        return original_dispatch(*args, **kwargs)

    sample.__globals__["scf_if_dispatch"] = traced_dispatch
    assert sample(True) == 2
    assert sample(False) == 3
    assert called["n"] in (0, 2)


def test_ast_rewrite_does_not_rewrite_static_string_compare():
    called = {"n": 0}

    def sample(dtype_str):
        out = 0
        if dtype_str == "f32":
            out = 1
        else:
            out = 2
        return out

    ASTRewriter.transform(sample)
    original_dispatch = sample.__globals__["scf_if_dispatch"]

    def traced_dispatch(*args, **kwargs):
        called["n"] += 1
        return original_dispatch(*args, **kwargs)

    sample.__globals__["scf_if_dispatch"] = traced_dispatch
    assert sample("f32") == 1
    assert sample("bf16") == 2
    assert called["n"] == 2
