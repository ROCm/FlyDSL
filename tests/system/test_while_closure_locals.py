#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests demonstrating the locals().get closure variable issue in while loops.

Problem
-------
The AST rewriter uses ``locals().get(name, None)`` (via ``_state_value_expr``)
to capture the initial value of each loop-carried variable before a dynamic
while loop.  ``locals()`` only returns the **local** variables of the current
function frame — closure (free) variables are invisible to it.

In normal Python, writing to a closure variable without ``nonlocal`` raises
``UnboundLocalError`` because the assignment makes the compiler treat it as
local.  However, the AST rewriter **moves** the assignment into a nested
function (``__while_after_N``), so Python no longer sees the variable as local
in the enclosing scope.  The result:

1. ``locals().get(name, None)`` silently returns ``None``.
2. The ``none_vars`` guard in ``scf_while_dispatch`` catches it and raises
   ``TypeError``, but the message says "initialized as None" — misleading,
   because the real cause is that ``locals()`` cannot see closure variables.

Harm
----
- **Without the none_vars guard**: ``None`` would be passed as an initial value
  to ``scf.WhileOp``, causing a cryptic MLIR crash or silently wrong IR.
- **With the none_vars guard** (current code): the error is caught, but the
  message points the user in the wrong direction (they might try initializing
  the variable to ``fx.Int32(0)`` when the real fix is to make it a local
  or a function parameter).
"""

import pytest


# ── Simulate the dispatch pattern used by the AST rewriter ────────────────────
#
# User writes:
#     while cond:
#         body
#
# AST rewriter produces (simplified):
#     def before_fn(result_names, *vars): return cond
#     def after_fn(result_names, *vars):  body; return {name: val, ...}
#     result = scf_while_dispatch(before_fn, after_fn,
#                                 result_names=(...),
#                                 result_values=(locals().get("var1"), ...))
#
# The problem: locals().get("x") returns None when x is a closure variable.


def _simulate_dispatch(before_fn, after_fn, init_values):
    """Minimal Python simulation of scf_while_dispatch loop semantics."""
    state = list(init_values)
    while before_fn(*state):
        result = after_fn(*state)
        state = list(result.values())
    return state


def test_local_var_correct_result():
    """Local variable: locals().get finds it, dispatch produces correct result.

    Equivalent user code:
        acc = 0
        n = 5
        while n > 0:
            acc = acc + n
            n = n - 1
        # acc == 15
    """

    def kernel_body():
        acc = 0
        n = 5

        def before_fn(acc, n):
            return n > 0

        def after_fn(acc, n):
            return {"acc": acc + n, "n": n - 1}

        init_acc = locals().get("acc", None)
        init_n = locals().get("n", None)
        return _simulate_dispatch(before_fn, after_fn, [init_acc, init_n])

    acc, n = kernel_body()
    assert acc == 15
    assert n == 0


def test_closure_var_wrong_result():
    """Closure variable: locals().get returns None, dispatch gets wrong initial value.

    Equivalent user code (inside a factory function):
        def make_kernel(start_n):
            def kernel():
                acc = 0
                while start_n > 0:       # start_n from closure
                    acc = acc + start_n
                    start_n = start_n - 1 # assignment → AST rewriter moves it
                return acc                # expected: 15

    Without AST rewriting, Python raises UnboundLocalError.
    With AST rewriting, the assignment moves into after_fn, so start_n
    is no longer local — locals().get returns None.
    """
    start_n = 5

    def kernel_body():
        acc = 0

        def before_fn(acc, start_n):
            return start_n > 0

        def after_fn(acc, start_n):
            return {"acc": acc + start_n, "n": start_n - 1}

        init_acc = locals().get("acc", None)
        init_start_n = locals().get("start_n", None)

        return init_acc, init_start_n

    init_acc, init_start_n = kernel_body()

    assert init_acc == 0       # acc is local → found
    assert init_start_n is None  # start_n is closure → NOT found, returns None

    # If we tried to dispatch with None:
    #   _simulate_dispatch(before_fn, after_fn, [0, None])
    # → TypeError: '>' not supported between instances of 'NoneType' and 'int'
    # In MLIR context, this would be passing None to scf.WhileOp → crash or wrong IR.


def test_normal_python_raises_unbound():
    """Normal Python (no AST rewriting): writing to closure var → UnboundLocalError.

    This is the error Python WOULD give. AST rewriting masks it by moving
    the assignment into a nested function, turning the error into a silent None.
    """
    start_n = 5

    def kernel_body():
        acc = 0
        while start_n > 0:
            acc = acc + start_n
            start_n = start_n - 1  # noqa: F841
        return acc

    with pytest.raises(UnboundLocalError):
        kernel_body()


def test_dispatch_with_none_crashes():
    """Dispatching with None initial value crashes — this is the real harm.

    If scf_while_dispatch's none_vars guard were missing, None would reach
    the MLIR WhileOp construction, causing either:
    - TypeError from _unwrap_value(None)
    - Cryptic MLIR verification failure
    - Silent generation of invalid IR
    """

    def before_fn(acc, n):
        return n > 0

    def after_fn(acc, n):
        return {"acc": acc + n, "n": n - 1}

    with pytest.raises(TypeError):
        _simulate_dispatch(before_fn, after_fn, [0, None])
