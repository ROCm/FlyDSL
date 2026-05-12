#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MLIR-level unit tests for scf_for_dispatch (no GPU required)."""

import pytest

from flydsl._mlir.ir import Context, FunctionType, IndexType, InsertionPoint, IntegerType, Location, Module
from flydsl._mlir.dialects import arith, func
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor, ReplaceIfWithDispatch
from flydsl.expr.numeric import Int32


def test_scf_for_dispatch_single_iter_arg():
    """for i in range(4): acc = acc + 1  →  acc should be scf.for result."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_single_iter_arg", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, acc):
                    one = Int32(arith.ConstantOp(i32, 1).result)
                    return {"acc": acc + one}

                result = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 4, 1,
                    body_fn,
                    result_names=("acc",),
                    result_values=(acc,),
                )
                assert isinstance(result, Int32)
                func.ReturnOp([result.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "-> (i32)" in ir_text


def test_scf_for_dispatch_multi_iter_args():
    """for i in range(3): a += 1; b -= 1  →  two iter_args."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_multi_iter_args", FunctionType.get([], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                a = Int32(arith.ConstantOp(i32, 0).result)
                b = Int32(arith.ConstantOp(i32, 100).result)

                def body_fn(iv, names, a, b):
                    one = Int32(arith.ConstantOp(i32, 1).result)
                    return {"a": a + one, "b": b - one}

                result = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 3, 1,
                    body_fn,
                    result_names=("a", "b"),
                    result_values=(a, b),
                )
                assert isinstance(result, tuple)
                assert len(result) == 2
                func.ReturnOp([result[0].ir_value(), result[1].ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "-> (i32, i32)" in ir_text


def test_scf_for_dispatch_no_iter_args():
    """Side-effect only loop: no iter_args, no yield values."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("test_no_iter_args", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):

                def body_fn(iv, names):
                    pass

                InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 4, 1,
                    body_fn,
                    result_names=(),
                    result_values=(),
                )
                func.ReturnOp([])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "-> (" not in ir_text


def test_scf_for_dispatch_none_value_raises_error():
    """result_values containing None should raise TypeError."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_none_error", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):

                def body_fn(iv, names, x):
                    return {"x": Int32(arith.ConstantOp(i32, 1).result)}

                with pytest.raises(TypeError, match="None"):
                    InsertEmptyYieldForSCFFor.scf_for_dispatch(
                        0, 4, 1,
                        body_fn,
                        result_names=("x",),
                        result_values=(None,),
                    )


def test_scf_for_dispatch_type_mismatch_raises_error():
    """Yielded type differs from init type → TypeError."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_type_mismatch", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                x = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, x):
                    return {"x": arith.ConstantOp(i64, 99).result}

                with pytest.raises(TypeError, match="type mismatch"):
                    InsertEmptyYieldForSCFFor.scf_for_dispatch(
                        0, 4, 1,
                        body_fn,
                        result_names=("x",),
                        result_values=(x,),
                    )


def test_scf_for_dispatch_range_with_step():
    """range(0, 8, 2) → 4 iterations, verify IR structure."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_range_step", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, acc):
                    return {"acc": acc + Int32(arith.ConstantOp(i32, 1).result)}

                result = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 8, 2,
                    body_fn,
                    result_names=("acc",),
                    result_values=(acc,),
                )
                func.ReturnOp([result.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "c8" in ir_text or "8" in ir_text


def test_ast_rewrite_for_generates_dispatch_call():
    """AST rewrite of for loop: verify scf_for_dispatch call is injected."""
    import ast, textwrap
    from flydsl.compiler.ast_rewriter import ASTRewriter

    def sample(n):
        acc = 0
        for i in range(n):
            acc = acc + 1
        return acc

    ASTRewriter.transform(sample)
    assert "scf_for_dispatch" in sample.__globals__, "scf_for_dispatch not injected into globals"
    assert "scf_for_collect_results" in sample.__globals__, "scf_for_collect_results not injected"


def test_ast_rewrite_for_multi_var_generates_dispatch():
    """AST rewrite of for loop with multiple vars: dispatch is injected."""
    from flydsl.compiler.ast_rewriter import ASTRewriter

    def sample(n):
        a = 0
        b = 100
        for i in range(n):
            a = a + 1
            b = b - 1
        return a, b

    ASTRewriter.transform(sample)
    assert "scf_for_dispatch" in sample.__globals__
