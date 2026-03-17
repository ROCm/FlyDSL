#!/usr/bin/env python3
"""Tests for the AST rewriter transformations.

Focuses on the AST-level transformation output without requiring a full MLIR
context or GPU hardware.
"""

import ast
import types
from textwrap import dedent

import pytest


def _parse_func(src: str) -> ast.FunctionDef:
    """Parse a single function from source and return its AST node."""
    module = ast.parse(dedent(src))
    assert isinstance(module.body[0], ast.FunctionDef)
    return module.body[0]


def _apply_transformer(func_node, transformer_cls, first_lineno=0):
    """Run one NodeTransformer on a function AST and return the result."""
    ctx = types.SimpleNamespace()
    rewriter = transformer_cls(context=ctx, first_lineno=first_lineno)
    return rewriter.generic_visit(func_node)


class TestCanonicalizeWhileAST:
    """Verify the CanonicalizeWhile transformer produces the expected AST
    pattern without executing anything at the MLIR level."""

    @staticmethod
    def _get_transformer():
        try:
            from flydsl.compiler.ast_rewriter import CanonicalizeWhile
            return CanonicalizeWhile
        except ImportError:
            pytest.skip("flydsl not importable")

    def test_while_loop_generates_rebind_prologue(self):
        """Body should start with carry-variable rebinding from block args."""
        CW = self._get_transformer()

        src = """\
        def f():
            while x < 10:
                x = x + 1
        """
        func_node = _parse_func(src)
        result = _apply_transformer(func_node, CW)
        code = ast.unparse(result)

        assert "scf_while_gen" in code, "should generate scf_while_gen call"
        assert "__init__" in code, "should have walrus binding for block args"
        assert "scf_while_yield_" in code, "should yield updated carry vars"
        assert "scf_while_get_results_" in code, "should rebind from WhileOp results"

    def test_while_loop_generates_yield_epilogue(self):
        """Body should end with an explicit yield of updated carry vars."""
        CW = self._get_transformer()

        src = """\
        def f():
            while i < N:
                i = i + 1
        """
        func_node = _parse_func(src)
        result = _apply_transformer(func_node, CW)
        code = ast.unparse(result)

        assert "scf_while_yield_" in code
        # Both carry vars (i and N) should appear in post-loop rebinding
        assert "scf_while_get_results_" in code

    def test_constexpr_while_unchanged(self):
        """while const_expr(cond) should stay as a plain Python while."""
        CW = self._get_transformer()

        src = """\
        def f():
            while const_expr(x < 10):
                x = x + 1
        """
        func_node = _parse_func(src)
        result = _apply_transformer(func_node, CW)
        code = ast.unparse(result)

        assert "scf_while_gen" not in code, "const_expr while should NOT be lowered"


class TestNestedControlFlowAST:
    """Verify the _ASTREWRITE_MARKER fix allows nested control flow."""

    @staticmethod
    def _get_if_transformer():
        try:
            from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
            return ReplaceIfWithDispatch
        except ImportError:
            pytest.skip("flydsl not importable")

    @staticmethod
    def _get_while_transformer():
        try:
            from flydsl.compiler.ast_rewriter import CanonicalizeWhile
            return CanonicalizeWhile
        except ImportError:
            pytest.skip("flydsl not importable")

    def test_while_inside_if_branch_is_lowered(self):
        """A while loop nested inside a then-branch should be transformed
        by CanonicalizeWhile even though the branch is a generated function."""
        IfT = self._get_if_transformer()
        WhileT = self._get_while_transformer()

        src = """\
        def f():
            if some_cond():
                while x < 10:
                    x = x + 1
        """
        func_node = _parse_func(src)
        # First pass: if → scf_if_dispatch
        result = _apply_transformer(func_node, IfT)
        # Second pass: while → scf_while_gen
        result = _apply_transformer(result, WhileT)
        code = ast.unparse(result)

        assert "scf_while_gen" in code, \
            "while loop inside if-branch should be lowered after marker fix"


class TestFP32OutputValidation:
    """Test that compile_preshuffle_gemm_a8 accepts fp32 out_dtype."""

    @staticmethod
    def _get_compile_fn():
        try:
            from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
            return compile_preshuffle_gemm_a8
        except ImportError:
            pytest.skip("kernels.preshuffle_gemm not importable")

    def test_fp32_rejected_with_cshuffle(self):
        """fp32 output + cshuffle should raise ValueError."""
        fn = self._get_compile_fn()
        with pytest.raises(ValueError, match="fp32.*direct epilog"):
            fn(K=128, tile_m=64, tile_n=64, tile_k=64,
               out_dtype="fp32", use_cshuffle_epilog=True)

    def test_fp32_accepted_without_cshuffle(self):
        """fp32 output + direct epilog should NOT raise ValueError for dtype."""
        fn = self._get_compile_fn()
        # This will fail later (needs MLIR context) but should pass
        # the dtype validation stage.
        try:
            fn(K=128, tile_m=64, tile_n=64, tile_k=64,
               out_dtype="fp32", use_cshuffle_epilog=False)
        except ValueError as e:
            assert "out_dtype" not in str(e), \
                f"fp32 should be accepted as out_dtype, got: {e}"
        except Exception:
            pass  # other errors (MLIR context, GPU) are expected

    def test_invalid_out_dtype_rejected(self):
        """Unsupported out_dtype should raise ValueError."""
        fn = self._get_compile_fn()
        with pytest.raises(ValueError, match="out_dtype"):
            fn(K=128, tile_m=64, tile_n=64, tile_k=64, out_dtype="int8")
