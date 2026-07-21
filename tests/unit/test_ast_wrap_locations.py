# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Behavioral tests for the FallbackLocations rewriter.

These exercise the real path: a function containing *bare* MLIR ops (ops not
built by an ``@dsl_loc_tracing`` primitive and with no explicit ``loc=``) is
run through ``ASTRewriter.transform`` and then executed inside an outer
``Location`` scope that stands in for the kernel ``def`` line (as
``kernel_function`` does during tracing). We then assert the location each bare
op actually received.

Without the rewriter every bare op inherits ``Location.current`` -- the def
line -- which is the Pattern-5 ATT-trace hotspot artifact. With it, each op
gets its own source line+column.
"""

import types

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith
from flydsl.compiler.ast_rewriter import ASTRewriter, FallbackLocations

pytestmark = [pytest.mark.l0_backend_agnostic]

_DEF_LINE = 10_000  # stand-in for the kernel def line; far from any real lineno


def _run_traced(func, *, transform):
    """Run *func* the way tracing does and return per-op location strings.

    ``func`` builds bare ``arith`` ops and returns them (single op or tuple).
    It reads its element type from a module-global ``I32`` we inject. When
    *transform* is True the function is first rewritten by ``ASTRewriter``
    (which includes ``FallbackLocations``); when False it runs as-is, modeling
    the pre-fix behavior.

    The outer ``Location`` scope uses ``_DEF_LINE`` to stand in for the kernel
    ``def`` line. ``FallbackLocations`` rewrites in terms of the *function's own*
    ``co_filename`` (this test file), so a corrected bare op keeps this file's
    name but gets its real source line; a degraded one keeps ``_DEF_LINE``.
    """
    fn = types.FunctionType(func.__code__, dict(func.__globals__), func.__name__)
    if transform:
        fn = ASTRewriter.transform(fn)

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        fn.__globals__["I32"] = ir.IntegerType.get_signless(32)
        module = ir.Module.create()
        def_loc = ir.Location.file("kernel_src.py", _DEF_LINE, 0, context=ctx)
        with ir.InsertionPoint(module.body), def_loc:
            result = fn()
        ops = result if isinstance(result, tuple) else (result,)
        return [str(op.operation.location) for op in ops]


# `_two_bare_ops` line offsets, measured from its own def line at import time.
def _two_bare_ops():
    a = arith.ConstantOp(I32, 1)  # noqa: F821
    b = arith.ConstantOp(I32, 2)  # noqa: F821
    return (a, b)


_TWO_BARE_DEF = _two_bare_ops.__code__.co_firstlineno
_LINE_A = _TWO_BARE_DEF + 1
_LINE_B = _TWO_BARE_DEF + 2


def _bare_op_in_branch():
    base = arith.ConstantOp(I32, 0)  # noqa: F821
    if True:
        base = arith.ConstantOp(I32, 7)  # noqa: F821
    return base


_BRANCH_DEF = _bare_op_in_branch.__code__.co_firstlineno
_BRANCH_IF_LINE = _BRANCH_DEF + 2  # the `if True:` statement line


def test_bare_ops_get_their_own_source_line(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    locs = _run_traced(_two_bare_ops, transform=True)

    # Each bare op carries its own source line (in this file), not the def line.
    assert f":{_LINE_A}:" in locs[0]
    assert f":{_LINE_B}:" in locs[1]
    assert f":{_DEF_LINE}:" not in locs[0]
    assert f":{_DEF_LINE}:" not in locs[1]
    # And the two ops are distinguished from each other.
    assert locs[0] != locs[1]


def test_without_rewriter_bare_ops_collapse_to_def_line(monkeypatch):
    """Control: this is the Pattern-5 degradation the rewriter fixes."""
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    locs = _run_traced(_two_bare_ops, transform=False)
    assert all(f":{_DEF_LINE}:" in loc for loc in locs)


def test_bare_op_inside_branch_gets_control_flow_line_not_def(monkeypatch):
    # `if True:` is turned into a dispatch helper by ReplaceIfWithDispatch, whose
    # generated FunctionDef FallbackLocations skips. The bare op inside the branch
    # therefore inherits the floor set on the wrapped *dispatch call* -- the `if`
    # statement line -- rather than collapsing onto the kernel def line. This is
    # the real interaction between the two rewriters, and still fixes Pattern-5.
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    locs = _run_traced(_bare_op_in_branch, transform=True)
    assert f":{_BRANCH_IF_LINE}:" in locs[0]
    assert f":{_DEF_LINE}:" not in locs[0]


def test_disabled_leaves_bare_ops_on_def_line(monkeypatch):
    """Gated off: rewriter is a no-op, so bare ops still inherit the def line."""
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "0")
    locs = _run_traced(_two_bare_ops, transform=True)
    assert all(f":{_DEF_LINE}:" in loc for loc in locs)


def test_explicit_loc_is_not_overridden(monkeypatch):
    """An op that passes its own loc= must win over the floor."""
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")

    def _explicit():
        loc = ir.Location.file("other.py", 42, 0)
        return arith.ConstantOp(I32, 1, loc=loc)  # noqa: F821

    locs = _run_traced(_explicit, transform=True)
    assert "other.py" in locs[0] and ":42:" in locs[0]


def test_rewrite_globals_exposes_loc_helper():
    g = FallbackLocations.rewrite_globals()
    assert "_flydsl_loc" in g
    # The helper is usable as a context manager and sets Location.current.
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        with g["_flydsl_loc"]("f.py", 7, 3):
            cur = str(ir.Location.current)
    assert "f.py" in cur and ":7:3" in cur
