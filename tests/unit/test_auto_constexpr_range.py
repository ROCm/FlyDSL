#!/usr/bin/env python3
"""Tests for automatic constexpr range detection in AST rewriter."""

import ast
import pytest
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor


def _make_rewriter(func_code=None):
    """Create an InsertEmptyYieldForSCFFor with dynamic params from a function."""
    rewriter = InsertEmptyYieldForSCFFor(context=None, first_lineno=0)
    rewriter._dynamic_params = set()
    if func_code:
        tree = ast.parse(func_code.strip())
        func_node = tree.body[0]
        rewriter.visit_FunctionDef(func_node)
    return rewriter


class TestIsConstexprArg:
    """Unit tests for _is_constexpr_arg."""

    def test_integer_literal(self):
        r = _make_rewriter()
        node = ast.parse("4").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_string_literal(self):
        r = _make_rewriter()
        node = ast.parse("'hello'").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_name_variable(self):
        r = _make_rewriter()
        node = ast.parse("TLOOP").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_name_dynamic_param(self):
        """A function parameter without Constexpr annotation is dynamic."""
        r = _make_rewriter("def f(n: Int32): pass")
        node = ast.parse("n").body[0].value
        assert r._is_constexpr_arg(node) is False

    def test_name_constexpr_param(self):
        """A function parameter with Constexpr annotation is constexpr."""
        r = _make_rewriter("def f(n: Constexpr[int]): pass")
        node = ast.parse("n").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_binop_constants(self):
        r = _make_rewriter()
        node = ast.parse("TLOOP * 2").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_binop_with_dynamic_param(self):
        """BinOp involving a dynamic parameter is not constexpr."""
        r = _make_rewriter("def f(n: Int32): pass")
        node = ast.parse("n + 1").body[0].value
        assert r._is_constexpr_arg(node) is False

    def test_binop_nested(self):
        r = _make_rewriter()
        node = ast.parse("HEAD_SIZE // MFMA_N // NUM_WARPS").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_unaryop(self):
        r = _make_rewriter()
        node = ast.parse("-N").body[0].value
        assert r._is_constexpr_arg(node) is True

    def test_function_call_not_constexpr(self):
        r = _make_rewriter()
        node = ast.parse("arith.constant(4)").body[0].value
        assert r._is_constexpr_arg(node) is False

    def test_attribute_not_constexpr(self):
        r = _make_rewriter()
        node = ast.parse("obj.attr").body[0].value
        assert r._is_constexpr_arg(node) is False

    def test_subscript_not_constexpr(self):
        r = _make_rewriter()
        node = ast.parse("arr[0]").body[0].value
        assert r._is_constexpr_arg(node) is False

    def test_binop_with_call_not_constexpr(self):
        r = _make_rewriter()
        node = ast.parse("func() + 1").body[0].value
        assert r._is_constexpr_arg(node) is False


class TestIsAutoConstexprRange:
    """Unit tests for _is_auto_constexpr_range."""

    def test_literal_arg(self):
        """range(4) -> constexpr"""
        r = _make_rewriter()
        node = ast.parse("range(4)").body[0].value
        assert r._is_auto_constexpr_range(node) is True

    def test_name_arg(self):
        """range(TLOOP) -> constexpr (module-level constant)"""
        r = _make_rewriter()
        node = ast.parse("range(TLOOP)").body[0].value
        assert r._is_auto_constexpr_range(node) is True

    def test_dynamic_param_arg(self):
        """range(loop_count) -> NOT constexpr when loop_count is dynamic param"""
        r = _make_rewriter("def f(loop_count: Int32): pass")
        node = ast.parse("range(loop_count)").body[0].value
        assert r._is_auto_constexpr_range(node) is False

    def test_arithmetic_args(self):
        """range(TLOOP * 2) -> constexpr"""
        r = _make_rewriter()
        node = ast.parse("range(TLOOP * 2)").body[0].value
        assert r._is_auto_constexpr_range(node) is True

    def test_two_literal_args(self):
        """range(0, 4) -> constexpr"""
        r = _make_rewriter()
        node = ast.parse("range(0, 4)").body[0].value
        assert r._is_auto_constexpr_range(node) is True

    def test_three_args_all_constexpr(self):
        """range(0, N, 2) -> constexpr when N is not a dynamic param"""
        r = _make_rewriter()
        node = ast.parse("range(0, N, 2)").body[0].value
        assert r._is_auto_constexpr_range(node) is True

    def test_call_arg_not_constexpr(self):
        """range(arith.index_cast(x)) -> NOT constexpr"""
        r = _make_rewriter()
        node = ast.parse("range(arith.index_cast(x))").body[0].value
        assert r._is_auto_constexpr_range(node) is False

    def test_mixed_args_one_call(self):
        """range(0, func()) -> NOT constexpr"""
        r = _make_rewriter()
        node = ast.parse("range(0, func())").body[0].value
        assert r._is_auto_constexpr_range(node) is False

    def test_init_keyword_not_constexpr(self):
        """range(0, N, 1, init=[...]) -> NOT constexpr (loop-carried)"""
        r = _make_rewriter()
        node = ast.parse("range(0, N, 1, init=vals)").body[0].value
        assert r._is_auto_constexpr_range(node) is False

    def test_init_keyword_with_literal_args(self):
        """range(0, 4, init=[...]) -> NOT constexpr despite literal args"""
        r = _make_rewriter()
        node = ast.parse("range(0, 4, init=vals)").body[0].value
        assert r._is_auto_constexpr_range(node) is False


class TestVisitForAutoConstexpr:
    """Integration: verify visit_For correctly routes constexpr vs dynamic."""

    def _get_rewritten_iter_name(self, for_code, func_code=None):
        """Parse code, apply visit_For, return the iter function name."""
        r = _make_rewriter(func_code)
        tree = ast.parse(for_code.strip())
        for_node = tree.body[0]
        assert isinstance(for_node, ast.For)
        result = r.visit_For(for_node)
        if isinstance(result, ast.For):
            call = result.iter
            if isinstance(call, ast.Call):
                func = call.func
                return getattr(func, 'id', None) or getattr(func, 'attr', None)
        return None

    def test_literal_becomes_range(self):
        name = self._get_rewritten_iter_name("for i in range(4): pass")
        assert name == "range"

    def test_name_becomes_range(self):
        name = self._get_rewritten_iter_name("for i in range(TLOOP): pass")
        assert name == "range"

    def test_arithmetic_becomes_range(self):
        name = self._get_rewritten_iter_name("for i in range(N * 2): pass")
        assert name == "range"

    def test_dynamic_param_becomes_scf_range(self):
        """range(loop_count) where loop_count is a dynamic param -> scf_range"""
        name = self._get_rewritten_iter_name(
            "for i in range(loop_count): pass",
            "def f(loop_count: Int32): pass")
        assert name == "scf_range"

    def test_call_arg_becomes_scf_range(self):
        name = self._get_rewritten_iter_name("for i in range(func()): pass")
        assert name == "scf_range"

    def test_attribute_call_becomes_scf_range(self):
        name = self._get_rewritten_iter_name("for i in range(obj.method()): pass")
        assert name == "scf_range"

    def test_range_constexpr_still_works(self):
        name = self._get_rewritten_iter_name("for i in range_constexpr(4): pass")
        assert name == "range"

    def test_no_yield_added_for_constexpr(self):
        """Auto-constexpr range should NOT get yield appended."""
        code = "for i in range(4):\n    x = i"
        r = _make_rewriter()
        tree = ast.parse(code)
        result = r.visit_For(tree.body[0])
        body_unparsed = [ast.unparse(s) for s in result.body]
        assert not any("yield" in s for s in body_unparsed), \
            f"Constexpr range should not have yield, got: {body_unparsed}"

    def test_yield_added_for_dynamic(self):
        """Dynamic range should get yield appended."""
        code = "for i in range(func()):\n    x = i"
        r = _make_rewriter()
        tree = ast.parse(code)
        result = r.visit_For(tree.body[0])
        body_unparsed = [ast.unparse(s) for s in result.body]
        assert any("yield" in s for s in body_unparsed), \
            f"Dynamic range should have yield, got: {body_unparsed}"


class TestDynamicParamDetection:
    """Test that visit_FunctionDef correctly identifies dynamic parameters."""

    def test_int32_param_is_dynamic(self):
        r = _make_rewriter("def f(n: Int32): pass")
        assert "n" in r._dynamic_params

    def test_constexpr_param_is_not_dynamic(self):
        r = _make_rewriter("def f(n: Constexpr[int]): pass")
        assert "n" not in r._dynamic_params

    def test_fx_constexpr_param_is_not_dynamic(self):
        r = _make_rewriter("def f(n: fx.Constexpr[int]): pass")
        assert "n" not in r._dynamic_params

    def test_stream_param_is_not_dynamic(self):
        r = _make_rewriter("def f(s: Stream): pass")
        assert "s" not in r._dynamic_params

    def test_tensor_param_is_dynamic(self):
        r = _make_rewriter("def f(out: Tensor): pass")
        assert "out" in r._dynamic_params

    def test_no_annotation_is_dynamic(self):
        r = _make_rewriter("def f(x): pass")
        assert "x" in r._dynamic_params

    def test_mixed_params(self):
        r = _make_rewriter("def f(out: Tensor, n: Constexpr[int], scale: Int32): pass")
        assert "out" in r._dynamic_params
        assert "n" not in r._dynamic_params
        assert "scale" in r._dynamic_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
