# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``return`` inside a region the AST rewriter moves into a nested function: an ``if`` branch or a loop body.

A branch picked at trace time returns from the enclosing function, as in Python. A ``return`` that tracing reaches
in an ``scf.if`` branch or an ``scf.for`` / ``scf.while`` body raises, naming its source line.
"""

import inspect
import types

import pytest

from flydsl._mlir.dialects import func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.numeric import Int32
from flydsl.expr.primitive import const_expr, range_constexpr


def _rewrite(fn):
    """A rewritten copy of the module-level function *fn*."""
    return ASTRewriter.transform(types.FunctionType(fn.__code__, dict(fn.__globals__), fn.__name__))


def _where(fn, marker):
    """``file:line`` of the source line of *fn* that contains *marker*."""
    lines, first = inspect.getsourcelines(fn)
    (offset,) = [i for i, line in enumerate(lines) if marker in line]
    return f"{fn.__code__.co_filename}:{first + offset}"


def _trace_dynamic(fn, *args):
    """Call *fn* with a dynamic i32 operand followed by *args*, inside a function being built."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("trace_dynamic", FunctionType.get([i32], []))
            with InsertionPoint(f.add_entry_block()):
                fn(Int32(f.entry_block.arguments[0]), *args)
                func.ReturnOp([])
        assert module.operation.verify()
        return str(module)


def _pick(flag):
    if flag:
        return "A"
    return "B"


def _nested_returns(flag, inner, n, log):
    x = 1
    if flag:
        x = 2
        if inner:
            return x * 10
        for i in range_constexpr(n):
            if i == 2:
                return "loop"
    else:

        def helper():
            if flag is False:
                return "helper"
            return "unreachable"

        x = helper()
        if const_expr(inner):
            return None
    log.append(x)
    return x


def _handler_and_case_returns(flag, kind):
    if flag:
        try:
            {"ok": 1}[kind]
        except KeyError:
            return "except"
        match kind:
            case "never":
                pass
            case _:
                return "case"
    return "after"


def _counters_with_branch_return():
    hits, misses = 0, 0

    def record(flag):
        nonlocal hits, misses
        if flag:
            hits += 1
            misses += 10
            return hits, misses
        misses += 1
        return hits, misses

    return [record(True), record(False), record(True)]


def _return_under_dynamic_if(n):
    if n > 3:
        return n  # dynamic-if return
    return n + 1


def _return_in_dynamic_else(n):
    if n > 3:
        pass
    else:
        return  # dynamic-else return


def _return_under_static_if_in_dynamic_if(n, flag):
    if n > 3:
        if flag:
            return n  # nested return
        n = n + 1
    return n


def _return_in_dynamic_for(n):
    acc = Int32(0)
    for i in range(n):
        acc = acc + i
        return acc  # for return, carried


def _early_exit_in_dynamic_for(n):
    for i in range(n):
        return  # for return


def _return_in_dynamic_while(n):
    i = Int32(0)
    while i < n:
        i = i + 1
        return i  # while return


def test_branch_return_under_python_bool_condition_returns_from_function():
    pick = _rewrite(_pick)
    assert (pick(True), pick(False)) == ("A", "B")


def test_branch_return_through_nested_regions():
    fn = _rewrite(_nested_returns)
    log = []
    assert fn(True, True, 3, log) == 20
    assert fn(True, False, 3, log) == "loop"
    assert fn(True, False, 2, log) == 2
    assert fn(False, False, 3, log) == "helper"
    assert fn(False, True, 3, log) is None
    assert log == [2, "helper"]


def test_branch_return_from_except_handler_and_match_case():
    fn = _rewrite(_handler_and_case_returns)
    assert fn(True, "missing") == "except"
    assert fn(True, "ok") == "case"
    assert fn(False, "ok") == "after"


def test_branch_return_keeps_assignments_made_before_it():
    assert _rewrite(_counters_with_branch_return)() == [(1, 10), (1, 11), (2, 21)]


@pytest.mark.parametrize(
    "fn, marker, region",
    [
        (_return_under_dynamic_if, "# dynamic-if return", "dynamic value"),
        (_return_in_dynamic_else, "# dynamic-else return", "dynamic value"),
        (_return_in_dynamic_for, "# for return, carried", r"dynamic `for` loop body"),
        (_early_exit_in_dynamic_for, "# for return", r"dynamic `for` loop body"),
        (_return_in_dynamic_while, "# while return", r"dynamic `while` loop body"),
    ],
)
def test_return_in_dynamic_region_raises_with_location(fn, marker, region):
    with pytest.raises(NotImplementedError, match=region) as err:
        _trace_dynamic(_rewrite(fn))
    assert str(err.value).startswith(_where(fn, marker) + ": ")


def test_branch_return_not_reached_inside_dynamic_condition():
    fn = _rewrite(_return_under_static_if_in_dynamic_if)
    assert "scf.if" in _trace_dynamic(fn, False)
    with pytest.raises(NotImplementedError, match="dynamic value") as err:
        _trace_dynamic(fn, True)
    assert str(err.value).startswith(_where(_return_under_static_if_in_dynamic_if, "# nested return") + ": ")
