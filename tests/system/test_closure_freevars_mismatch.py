#!/usr/bin/env python3
"""
Regression tests for the co_freevars mismatch fix in ast_rewriter.py.

When a kernel is a closure and AST rewriting eliminates a captured free
variable (e.g. const_expr(True) unpacked to True), the new code object
has fewer co_freevars than the original __closure__.  The fix in
ASTRewriter.transform detects this mismatch and rebuilds the function
via types.FunctionType with a matching closure.

These tests verify that the fix works for:
  - single const_expr elimination
  - multiple const_expr eliminations
  - non-closure kernels (no rebuild needed)
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1a_compile]


def test_closure_const_expr_freevars_mismatch():
    """Closure kernel with const_expr(True) unpacked: co_freevars shrinks from 2 to 1."""

    def make_kernel(batch_size):
        from flydsl.expr import const_expr

        @flyc.jit
        def kernel(x: fx.Int32) -> fx.Int32:
            if const_expr(True):
                return batch_size + x
            return x

        return kernel

    make_kernel(32)


def test_closure_multiple_const_expr_eliminated():
    """Multiple const_expr calls eliminated, co_freevars shrinks further."""

    def make_kernel(batch_size):
        from flydsl.expr import const_expr

        @flyc.jit
        def kernel(x: fx.Int32) -> fx.Int32:
            if const_expr(True):
                y = batch_size + x
            if const_expr(False):
                y = x
            return y

        return kernel

    make_kernel(64)


def test_non_closure_kernel_no_issue():
    """Top-level kernel (no closure) should not hit the enclosing_mod path."""

    @flyc.jit
    def kernel(x: fx.Int32) -> fx.Int32:
        return x + 1

    assert kernel.func.__closure__ is None
