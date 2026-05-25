#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MLIR-level unit tests for scf_while_dispatch (no GPU required)."""

import pytest

from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import CanonicalizeWhile
from flydsl.expr.numeric import Int32


def test_scf_while_dispatch_single_result():
    """while offset > 0: offset = offset // 2  →  single loop-carried var."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_single", FunctionType.get([i32], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                offset = Int32(entry.arguments[0])

                def before_fn(names, offset):
                    return offset > Int32(arith.ConstantOp(i32, 0).result)

                def after_fn(names, offset):
                    return {"offset": offset // Int32(arith.ConstantOp(i32, 2).result)}

                result = CanonicalizeWhile.scf_while_dispatch(
                    before_fn,
                    after_fn,
                    result_names=("offset",),
                    result_values=(offset,),
                )
                assert isinstance(result, Int32)
                func.ReturnOp([result.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.while" in ir_text
        assert "scf.condition" in ir_text


def test_scf_while_dispatch_multi_results():
    """while offset > 0: acc += offset; offset //= 2  →  two loop-carried vars."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_multi", FunctionType.get([i32], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)
                offset = Int32(entry.arguments[0])

                def before_fn(names, acc, offset):
                    return offset > Int32(arith.ConstantOp(i32, 0).result)

                def after_fn(names, acc, offset):
                    return {"acc": acc + offset, "offset": offset // Int32(arith.ConstantOp(i32, 2).result)}

                result = CanonicalizeWhile.scf_while_dispatch(
                    before_fn,
                    after_fn,
                    result_names=("acc", "offset"),
                    result_values=(acc, offset),
                )
                assert isinstance(result, tuple)
                assert len(result) == 2
                func.ReturnOp([result[0].ir_value(), result[1].ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.while" in ir_text
        assert "-> (i32, i32)" in ir_text


def test_scf_while_dispatch_no_result():
    """Side-effect only while loop: no result_names, no yield values."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_no_result", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond_val = entry.arguments[0]

                def before_fn(names):
                    return cond_val

                def after_fn(names):
                    pass

                CanonicalizeWhile.scf_while_dispatch(
                    before_fn,
                    after_fn,
                    result_names=(),
                    result_values=(),
                )
                func.ReturnOp([])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.while" in ir_text


def test_scf_while_dispatch_simple_condition():
    """Simple condition with only a return in before_fn generates scf.while."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_simple_cond", FunctionType.get([i32], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                n = Int32(entry.arguments[0])
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def before_fn(names, acc, n):
                    return n > Int32(arith.ConstantOp(i32, 0).result)

                def after_fn(names, acc, n):
                    return {"acc": acc + Int32(arith.ConstantOp(i32, 1).result),
                            "n": n - Int32(arith.ConstantOp(i32, 1).result)}

                result = CanonicalizeWhile.scf_while_dispatch(
                    before_fn,
                    after_fn,
                    result_names=("acc", "n"),
                    result_values=(acc, n),
                )
                func.ReturnOp([result[0].ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.while" in ir_text
        assert "scf.condition" in ir_text


def test_scf_while_dispatch_none_var_raises_error():
    """None in result_values should raise TypeError for dynamic condition."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_none", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond_val = entry.arguments[0]

                def before_fn(names, x):
                    return cond_val

                def after_fn(names, x):
                    return {"x": x}

                with pytest.raises(TypeError, match="None"):
                    CanonicalizeWhile.scf_while_dispatch(
                        before_fn,
                        after_fn,
                        result_names=("x",),
                        result_values=(None,),
                    )
