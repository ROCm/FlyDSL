#!/usr/bin/env python3
"""Tests for if/else that produces values (scf_if_dispatch_v).

Tests both static (compile-time) and dynamic (MLIR scf.IfOp with results) paths.
"""

import ast
import pytest

from flydsl._mlir.ir import (
    Context, Location, Module, InsertionPoint,
    FunctionType, IntegerType, IndexType,
)
from flydsl._mlir.dialects import arith, func, scf
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch


# ---------------------------------------------------------------------------
# Unit tests for _collect_assigned_vars
# ---------------------------------------------------------------------------

class TestCollectAssignedVars:
    def test_simple_assigns(self):
        code = "a = 1\nb = 2\nc = 3"
        stmts = ast.parse(code).body
        result = ReplaceIfWithDispatch._collect_assigned_vars(stmts)
        assert result == ["a", "b", "c"]

    def test_aug_assign(self):
        code = "a += 1\nb = 2"
        stmts = ast.parse(code).body
        result = ReplaceIfWithDispatch._collect_assigned_vars(stmts)
        assert result == ["a", "b"]

    def test_no_duplicates(self):
        code = "a = 1\na = 2\nb = 3"
        stmts = ast.parse(code).body
        result = ReplaceIfWithDispatch._collect_assigned_vars(stmts)
        assert result == ["a", "b"]

    def test_no_assigns(self):
        code = "print(1)\nfoo()"
        stmts = ast.parse(code).body
        result = ReplaceIfWithDispatch._collect_assigned_vars(stmts)
        assert result == []

    def test_tuple_target_ignored(self):
        """Tuple unpacking targets are not Name nodes at top level."""
        code = "a, b = 1, 2"
        stmts = ast.parse(code).body
        result = ReplaceIfWithDispatch._collect_assigned_vars(stmts)
        # ast.Tuple target, not ast.Name — not collected
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests for scf_if_dispatch_v — static path
# ---------------------------------------------------------------------------

class TestStaticIfDispatchV:
    def test_single_var_true(self):
        def then_fn():
            return {"x": 42}
        def else_fn():
            return {"x": 99}
        result = ReplaceIfWithDispatch.scf_if_dispatch_v(True, then_fn, else_fn, ("x",))
        assert result == 42

    def test_single_var_false(self):
        def then_fn():
            return {"x": 42}
        def else_fn():
            return {"x": 99}
        result = ReplaceIfWithDispatch.scf_if_dispatch_v(False, then_fn, else_fn, ("x",))
        assert result == 99

    def test_multiple_vars_true(self):
        def then_fn():
            return {"a": 10, "b": 20}
        def else_fn():
            return {"a": 30, "b": 40}
        a, b = ReplaceIfWithDispatch.scf_if_dispatch_v(True, then_fn, else_fn, ("a", "b"))
        assert a == 10
        assert b == 20

    def test_multiple_vars_false(self):
        def then_fn():
            return {"a": 10, "b": 20}
        def else_fn():
            return {"a": 30, "b": 40}
        a, b = ReplaceIfWithDispatch.scf_if_dispatch_v(False, then_fn, else_fn, ("a", "b"))
        assert a == 30
        assert b == 40


# ---------------------------------------------------------------------------
# Dynamic path: scf.IfOp with results
# ---------------------------------------------------------------------------

class TestDynamicIfDispatchV:
    def test_single_result(self):
        """Dynamic if/else producing a single i32 value."""
        with Context() as ctx, Location.unknown():
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("test_single", FunctionType.get([i1], [i32]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]

                    def then_fn():
                        v = arith.ConstantOp(i32, 42).result
                        return {"x": v}

                    def else_fn():
                        v = arith.ConstantOp(i32, 99).result
                        return {"x": v}

                    x = ReplaceIfWithDispatch.scf_if_dispatch_v(
                        cond, then_fn, else_fn, ("x",)
                    )
                    func.ReturnOp([x])

            assert module.operation.verify()
            ir_text = str(module)
            assert "scf.if" in ir_text
            assert "scf.yield" in ir_text
            # Should have result type annotation
            assert "-> (i32)" in ir_text

    def test_multiple_results(self):
        """Dynamic if/else producing two i32 values."""
        with Context() as ctx, Location.unknown():
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("test_multi", FunctionType.get([i1], [i32, i32]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]

                    def then_fn():
                        a = arith.ConstantOp(i32, 10).result
                        b = arith.ConstantOp(i32, 20).result
                        return {"a": a, "b": b}

                    def else_fn():
                        a = arith.ConstantOp(i32, 30).result
                        b = arith.ConstantOp(i32, 40).result
                        return {"a": a, "b": b}

                    a, b = ReplaceIfWithDispatch.scf_if_dispatch_v(
                        cond, then_fn, else_fn, ("a", "b")
                    )
                    func.ReturnOp([a, b])

            assert module.operation.verify()
            ir_text = str(module)
            assert "scf.if" in ir_text
            assert "-> (i32, i32)" in ir_text

    def test_then_branch_ops_in_correct_region(self):
        """Verify that then-branch ops are inside the then-region, not the func body."""
        with Context() as ctx, Location.unknown():
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("test_regions", FunctionType.get([i1], [i32]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]

                    def then_fn():
                        c1 = arith.ConstantOp(i32, 1).result
                        c2 = arith.ConstantOp(i32, 2).result
                        v = arith.AddIOp(c1, c2).result
                        return {"x": v}

                    def else_fn():
                        v = arith.ConstantOp(i32, 0).result
                        return {"x": v}

                    x = ReplaceIfWithDispatch.scf_if_dispatch_v(
                        cond, then_fn, else_fn, ("x",)
                    )
                    func.ReturnOp([x])

            assert module.operation.verify()
            ir_text = str(module)
            # The arith.addi must be inside scf.if, not at function body level.
            # In valid MLIR, ops at function body level have 4-space indent,
            # while ops inside scf.if regions have 6+ space indent.
            assert "scf.if" in ir_text
            assert "arith.addi" in ir_text
            for line in ir_text.split("\n"):
                # arith.addi should never appear at function body level (4-space indent)
                if "arith.addi" in line:
                    # It should be nested inside scf.if (deeper indent)
                    leading = len(line) - len(line.lstrip())
                    assert leading > 4, \
                        f"arith.addi at function body level (indent={leading}): {line}"

    def test_mixed_types(self):
        """Dynamic if/else producing values of different types (i32 and i64)."""
        with Context() as ctx, Location.unknown():
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("test_mixed_types", FunctionType.get([i1], [i32, i64]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]

                    def then_fn():
                        a = arith.ConstantOp(i32, 1).result
                        b = arith.ConstantOp(i64, 100).result
                        return {"a": a, "b": b}

                    def else_fn():
                        a = arith.ConstantOp(i32, 2).result
                        b = arith.ConstantOp(i64, 200).result
                        return {"a": a, "b": b}

                    a, b = ReplaceIfWithDispatch.scf_if_dispatch_v(
                        cond, then_fn, else_fn, ("a", "b")
                    )
                    func.ReturnOp([a, b])

            assert module.operation.verify()
            ir_text = str(module)
            assert "-> (i32, i64)" in ir_text


# ---------------------------------------------------------------------------
# AST rewriter integration tests
# ---------------------------------------------------------------------------

class TestASTRewriterIntegration:
    def test_visit_if_detects_common_vars(self):
        """The AST rewriter should detect vars assigned in both branches."""
        code = """
if some_cond():
    a = expr1()
    b = expr2()
else:
    a = expr3()
    b = expr4()
"""
        tree = ast.parse(code.strip())
        if_node = tree.body[0]
        then_vars = ReplaceIfWithDispatch._collect_assigned_vars(if_node.body)
        else_vars = ReplaceIfWithDispatch._collect_assigned_vars(if_node.orelse)
        common = [v for v in then_vars if v in else_vars]
        assert common == ["a", "b"]

    def test_visit_if_no_common_vars(self):
        """No common vars => void dispatch."""
        code = """
if some_cond():
    a = expr1()
else:
    b = expr2()
"""
        tree = ast.parse(code.strip())
        if_node = tree.body[0]
        then_vars = ReplaceIfWithDispatch._collect_assigned_vars(if_node.body)
        else_vars = ReplaceIfWithDispatch._collect_assigned_vars(if_node.orelse)
        common = [v for v in then_vars if v in else_vars]
        assert common == []

    def test_visit_if_partial_overlap(self):
        """Only common vars are collected; non-common stay in branches."""
        code = """
if some_cond():
    a = expr1()
    c = extra()
else:
    a = expr3()
    d = other()
"""
        tree = ast.parse(code.strip())
        if_node = tree.body[0]
        then_vars = ReplaceIfWithDispatch._collect_assigned_vars(if_node.body)
        else_vars = ReplaceIfWithDispatch._collect_assigned_vars(if_node.orelse)
        common = [v for v in then_vars if v in else_vars]
        assert common == ["a"]


# ---------------------------------------------------------------------------
# Backward compatibility: void if/else still works
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_void_dispatch_static(self):
        """Void scf_if_dispatch still works for static conditions."""
        called = []
        def then_fn():
            called.append("then")
        def else_fn():
            called.append("else")

        ReplaceIfWithDispatch.scf_if_dispatch(True, then_fn, else_fn)
        assert called == ["then"]

        called.clear()
        ReplaceIfWithDispatch.scf_if_dispatch(False, then_fn, else_fn)
        assert called == ["else"]

    def test_void_dispatch_dynamic(self):
        """Void scf_if_dispatch still works for dynamic conditions."""
        with Context() as ctx, Location.unknown():
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("test_void", FunctionType.get([i1], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]

                    side_effects = []

                    def then_fn():
                        arith.ConstantOp(i32, 42)

                    def else_fn():
                        arith.ConstantOp(i32, 99)

                    ReplaceIfWithDispatch.scf_if_dispatch(cond, then_fn, else_fn)
                    func.ReturnOp([])

            assert module.operation.verify()
            ir_text = str(module)
            assert "scf.if" in ir_text
            # Void: no result type annotation
            assert "-> (" not in ir_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
