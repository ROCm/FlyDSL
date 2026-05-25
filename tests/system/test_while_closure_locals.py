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


# ── Part 1: Pure Python demonstration ──────────────────────────────────────────


def test_locals_does_not_see_closure_variables():
    """locals() returns only local variables, not closure (free) variables."""
    outer_var = 42

    def inner():
        return locals().get("outer_var", None)

    assert inner() is None


def test_locals_sees_local_copy_of_closure():
    """After assigning to a local, locals() finds it."""
    outer_var = 42

    def inner():
        local_copy = outer_var  # noqa: F841
        return locals().get("local_copy", None)

    assert inner() == 42


# ── Part 2: AST rewriter masking UnboundLocalError ────────────────────────────
#
# In normal Python, assigning to a closure variable without `nonlocal`
# triggers UnboundLocalError:
#
#     def outer():
#         n = 5
#         def inner():
#             while n > 0:     # UnboundLocalError: n is local (due to assignment below)
#                 n = n - 1
#
# But the AST rewriter moves `n = n - 1` into `__while_after_N`, so Python
# no longer treats `n` as local in `inner`.  Instead, `n` becomes a free
# variable, and `locals().get("n", None)` returns None.


def test_python_unbound_local_error():
    """Normal Python: assigning to closure variable → UnboundLocalError."""

    def outer():
        n = 5

        def inner():
            while n > 0:
                n = n - 1  # noqa: F841 — makes `n` local, but never assigned before read

        return inner

    with pytest.raises(UnboundLocalError):
        outer()()


def test_ast_rewrite_masks_unbound_error():
    """When assignment moves to a nested function, the error becomes None.

    This simulates what the AST rewriter does: the `n = n - 1` moves into
    __while_after_0, so the outer function no longer has a local `n`.
    ``locals().get("n", None)`` returns None instead of raising.
    """
    n_outer = 5

    def kernel_body():
        def __while_after_0(n):
            return n - 1

        initial_n = locals().get("n_outer", None)
        return initial_n

    assert kernel_body() is None
