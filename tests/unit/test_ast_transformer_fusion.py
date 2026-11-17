# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""One walk applies every structural rewrite, to the whole tree and to what it builds."""

import ast
import inspect
from textwrap import dedent

import pytest

from flydsl.compiler.ast_rewriter import (
    ASTRewriter,
    CanonicalizeWhile,
    FallbackLocations,
    InsertEmptyYieldForSCFFor,
    ReplaceIfWithDispatch,
    ReplaceYieldWithSCFYield,
    RewriteBoolOps,
    StructuralTransforms,
    Transformer,
)

pytestmark = [pytest.mark.l0_backend_agnostic]

_BASES = (
    RewriteBoolOps,
    ReplaceIfWithDispatch,
    InsertEmptyYieldForSCFFor,
    ReplaceYieldWithSCFYield,
    CanonicalizeWhile,
)


def _rewrite(func):
    """The rewritten source of *func*, without compiling or running it."""
    module = ast.parse(dedent(inspect.getsource(func)))
    context = type("Ctx", (), {"python_globals": func.__globals__, "filename": "<test>"})()
    for ctor in ASTRewriter.transformers:
        module.body[0] = ctor(context=context, first_lineno=0).visit(module.body[0])
    return ast.unparse(ast.fix_missing_locations(module))


def test_the_bases_handle_disjoint_node_types():
    """Two bases defining one visit_* would silently drop whichever the MRO ranks second."""
    seen = {}
    for base in _BASES:
        for name in vars(base):
            if name.startswith("visit_"):
                assert name not in seen, f"{base.__name__} and {seen.get(name)} both define {name}"
                seen[name] = base.__name__
    assert seen, "no handlers found"


def test_the_registered_passes_are_the_fused_one_and_the_locations_one():
    assert ASTRewriter.transformers == [StructuralTransforms, FallbackLocations]
    merged = StructuralTransforms.rewrite_globals()
    for base in _BASES:
        assert base.rewrite_globals().keys() <= merged.keys()


def _plain_python_loop(values, flag):
    total = 0
    for value in values:
        if flag and value:
            total = total + value
    return total


def test_a_plain_python_loop_still_has_its_body_rewritten():
    """Its own handler leaves the loop alone; the body underneath still needs every rewrite."""
    src = _rewrite(_plain_python_loop)
    assert "for value in values:" in src
    assert "scf_if_dispatch" in src
    assert "dsl_and_" in src


def _constexpr_test(traits):
    from flydsl.expr import const_expr

    if const_expr(not traits):
        return 1
    return 0


def test_a_constexpr_if_test_is_rewritten():
    src = _rewrite(_constexpr_test)
    assert "dsl_not_(traits)" in src
    assert "const_expr" not in src.split("\n")[2]


def _loop_with_trailing_yield(n, acc):
    from flydsl.expr import range as fx_range  # noqa: F401

    for _ in range(0, n, 1, init=[acc]):
        acc = acc + 1
        yield acc
    return acc


def test_a_trailing_yield_is_not_doubled():
    """The body is rewritten first, so the yield check has to recognise the rewritten form."""
    src = _rewrite(_loop_with_trailing_yield)
    assert src.count("scf_yield_") == 1, src


def _loop_body_without_an_explicit_end(n, acc):
    for _ in range(0, n, 1, init=[acc]):
        acc = acc + 1
    return acc


def test_an_appended_yield_is_rewritten_too():
    """The loop handler builds this yield after the walk passed the node."""
    src = _rewrite(_loop_body_without_an_explicit_end)
    assert "scf_yield_()" in src
    assert "yield" not in src.replace("scf_yield_", "")


def test_the_fused_pass_is_a_transformer():
    assert issubclass(StructuralTransforms, Transformer)
