# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""AST-level tests for the FallbackLocations rewriter (source-location floor)."""

import ast
import types

import pytest

from flydsl.compiler.ast_rewriter import _ASTREWRITE_MARKER, FallbackLocations

pytestmark = [pytest.mark.l0_backend_agnostic]


def _rewrite(src, first_lineno=100, filename="k.py"):
    fn = ast.parse(src).body[0]
    ctx = types.SimpleNamespace(filename=filename, python_globals={})
    out = FallbackLocations(context=ctx, first_lineno=first_lineno).visit(fn)
    return ast.unparse(ast.fix_missing_locations(out))


def _loc_calls(unparsed):
    """All ``_flydsl_loc(file, line, col)`` arg tuples in the rewritten source."""
    tree = ast.parse(unparsed)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_flydsl_loc":
            calls.append(tuple(a.value for a in node.args))
    return calls


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "0")
    src = "def k(a):\n    x = a + 1\n    return x\n"
    out = _rewrite(src)
    assert "_flydsl_loc" not in out


def test_wraps_top_level_and_compound_bodies(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    src = (
        "def k(a):\n"  # rel line 1
        "    x = a + 1\n"  # rel 2
        "    for i in range(4):\n"  # rel 3
        "        y = x + i\n"  # rel 4
        "    if x > 0:\n"  # rel 5
        "        z = y\n"  # rel 6
        "    return z\n"  # rel 7
    )
    out = _rewrite(src, first_lineno=100)
    calls = _loc_calls(out)
    # Absolute line = first_lineno + relative line.
    lines = {c[1] for c in calls}
    assert {102, 103, 104, 105, 107} <= lines  # x=, for, y=, if, return; z= nested
    # Every call carries (filename, line, col).
    for c in calls:
        assert c[0] == "k.py"
        assert len(c) == 3
    # Column for a top-level statement (4-space indent) is 4.
    top = [c for c in calls if c[1] == 102][0]
    assert top[2] == 4
    # Nested-body statement (8-space indent) is 8.
    nested = [c for c in calls if c[1] == 104][0]
    assert nested[2] == 8


def test_skips_nested_def_and_class(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    src = "def k(a):\n" "    def inner():\n" "        w = 1\n" "    class C:\n" "        pass\n" "    return a\n"
    out = _rewrite(src)
    tree = ast.parse(out)
    # The nested def/class statements themselves are not wrapped in a With.
    fn = tree.body[0]
    wrapped_kinds = []
    for stmt in fn.body:
        if isinstance(stmt, ast.With):
            wrapped_kinds.append(type(stmt.body[0]).__name__)
        else:
            wrapped_kinds.append(type(stmt).__name__)
    assert "FunctionDef" in wrapped_kinds  # inner def present, unwrapped
    assert "ClassDef" in wrapped_kinds  # class present, unwrapped
    # inner def's own body still gets wrapped.
    inner = next(s for s in fn.body if isinstance(s, ast.FunctionDef))
    assert isinstance(inner.body[0], ast.With)


def test_no_double_wrap(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    src = "def k(a):\n    x = a + 1\n    return x\n"
    once = _rewrite(src)
    # Feed the already-wrapped source back through the rewriter.
    twice = _rewrite(once)
    assert once.count("_flydsl_loc(") == twice.count("_flydsl_loc(")


def test_skips_marker_generated_functiondef(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    src = "def __gen():\n    x = 1\n    return x\n"
    fn = ast.parse(src).body[0]
    setattr(fn, _ASTREWRITE_MARKER, "ReplaceIfWithDispatch")
    ctx = types.SimpleNamespace(filename="k.py", python_globals={})
    out = FallbackLocations(context=ctx, first_lineno=100).visit(fn)
    assert "_flydsl_loc" not in ast.unparse(ast.fix_missing_locations(out))


def test_rewrite_globals_exposes_loc_helper():
    g = FallbackLocations.rewrite_globals()
    assert "_flydsl_loc" in g
    # The helper is a usable context manager factory.
    cm = g["_flydsl_loc"]
    assert hasattr(cm, "__call__")
