#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import ASTRewriter, ReplaceIfWithDispatch
from flydsl.compiler.scf_control_flow import FlyScfBuilder
from flydsl.expr.numeric import Boolean, Int32


def test_scf_generator_dynamic_if_expression_scalar():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ifexp_scalar", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = FlyScfBuilder.execute_if_expression(cond, lambda: Int32(42), lambda: Int32(99))
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" in str(module)


def test_scf_generator_dynamic_if_expression_tuple():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ifexp_tuple", FunctionType.get([i1], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = FlyScfBuilder.execute_if_expression(
                    cond,
                    lambda: (Int32(1), Int32(2)),
                    lambda: (Int32(3), Int32(4)),
                )
                assert isinstance(out, tuple)
                func.ReturnOp([out[0].ir_value(), out[1].ir_value()])

        assert module.operation.verify()


def test_scf_if_dispatch_supports_structured_state():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_structured_if_state", FunctionType.get([i1], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                pair = (
                    Int32(arith.ConstantOp(i32, 0).result),
                    Int32(arith.ConstantOp(i32, 1).result),
                )

                def then_fn(pair):
                    return {"pair": (Int32(2), Int32(3))}

                def else_fn(pair):
                    return {"pair": (Int32(4), Int32(5))}

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    then_fn,
                    else_fn,
                    state_names=("pair",),
                    state_values=(pair,),
                )
                assert isinstance(out, tuple)
                func.ReturnOp([out[0].ir_value(), out[1].ir_value()])

        assert module.operation.verify()


def test_scf_generator_dynamic_if_expression_structure_mismatch():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ifexp_structure_mismatch", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                with pytest.raises(TypeError, match="changed structure|result"):
                    FlyScfBuilder.execute_if_expression(
                        cond,
                        lambda: (Int32(1), Int32(2)),
                        lambda: Int32(3),
                    )


def test_ast_rewrite_dynamic_if_expression():
    def sample(cond, x, y):
        return x if cond else y

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_ifexp", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                x = Int32(arith.ConstantOp(i32, 7).result)
                y = Int32(arith.ConstantOp(i32, 9).result)
                out = sample(cond, x, y)
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" in str(module)


def test_ast_rewrite_dynamic_if_expression_allows_calls():
    def make_value(x):
        return x

    def sample(cond, x, y):
        return make_value(x) if cond else y

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_ifexp_call", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                x = Int32(arith.ConstantOp(i32, 7).result)
                y = Int32(arith.ConstantOp(i32, 9).result)
                out = sample(cond, x, y)
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" in str(module)


def test_ast_rewrite_dynamic_boolop_keeps_bool_lowering_path():
    def sample(a, b):
        return (a < b) and (b < Int32(10))

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_boolop", FunctionType.get([i32, i32], [IntegerType.get_signless(1)]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                a, b = entry.arguments
                out = sample(Int32(a), Int32(b))
                assert isinstance(out, Boolean)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" not in str(module)


def test_ast_rewrite_dynamic_elif_nests_ifops():
    def sample(cond0, cond1, x):
        if cond0:
            x = 1
        elif cond1:
            x = 2
        else:
            x = 3
        return x

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_elif", FunctionType.get([i1, i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond0, cond1 = entry.arguments
                x = Int32(arith.ConstantOp(i32, 0).result)
                out = sample(cond0, cond1, x)
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert str(module).count("scf.if") >= 2


def test_ast_rewrite_nested_dynamic_if():
    def sample(cond0, cond1, x):
        if cond0:
            if cond1:
                x = 1
            else:
                x = 2
        else:
            x = 3
        return x

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_nested_if", FunctionType.get([i1, i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond0, cond1 = entry.arguments
                x = Int32(arith.ConstantOp(i32, 0).result)
                out = sample(cond0, cond1, x)
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert str(module).count("scf.if") >= 2


def test_ast_rewrite_dynamic_if_inside_nested_function():
    def sample(cond, x):
        def helper(flag, y):
            if flag:
                y = 11
            else:
                y = 13
            return y

        return helper(cond, x)

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_subfunction_if", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                x = Int32(arith.ConstantOp(i32, 0).result)
                out = sample(cond, x)
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" in str(module)


def test_ast_rewrite_dynamic_if_tracks_method_receiver_state():
    class Box:
        def __init__(self, value):
            self.value = value

        def __fly_values__(self):
            return [self.value.ir_value()]

        @classmethod
        def __fly_construct__(cls, values):
            return cls(Int32(values[0]))

        def set_value(self, value):
            self.value = value

    def sample(cond, box):
        if cond:
            box.set_value(Int32(17))
        else:
            box.set_value(Int32(19))
        return box.value

    ASTRewriter.transform(sample)

    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_ast_method_receiver_if", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                initial = Int32(arith.ConstantOp(i32, 0).result)
                box = Box(initial)
                out = sample(cond, box)
                assert isinstance(out, Int32)
                assert box.value is initial
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "scf.if" in str(module)


def test_ast_rewrite_dynamic_if_rejects_return():
    def sample(cond, x):
        if cond:
            return x
        return x

    with pytest.raises(SyntaxError, match="Early exit"):
        ASTRewriter.transform(sample)


def test_ast_rewrite_static_closure_if_allows_return():
    flag = True

    def sample(x):
        def helper(y):
            if flag:
                return y
            return y + Int32(1)

        return helper(x)

    ASTRewriter.transform(sample)
    assert sample(Int32(3)).value == 3
