# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``and`` / ``or`` chains short-circuit like Python, evaluate each operand at most once, and emit one op per operand."""

import sys

import pytest

from flydsl._mlir import ir
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.meta import capture_user_location, tracing_context
from flydsl.expr.numeric import Boolean
from flydsl.expr.utils.arith import arith_const

pytestmark = [pytest.mark.l0_backend_agnostic]


def _chain(values, seen):
    def op(i):
        seen.append(i)
        return values[i]

    def all_of():
        return op(0) and op(1) and op(2) and op(3) and op(4) and op(5) and op(6) and op(7) and op(8)

    def any_of():
        return op(0) or op(1) or op(2) or op(3) or op(4) or op(5) or op(6) or op(7) or op(8)

    return ASTRewriter.transform(all_of), ASTRewriter.transform(any_of)


@pytest.mark.parametrize("set_bit", [None, 0, 3, 8])
def test_long_chains_match_python_and_evaluate_each_operand_once(set_bit):
    values = [i != set_bit for i in range(9)]
    seen = []
    all_of, any_of = _chain(values, seen)
    for rewritten, expected in ((all_of, _python_and(values)), (any_of, _python_or(values))):
        seen.clear()
        assert rewritten() == expected[0]
        assert seen == expected[1]


def _python_and(values):
    for i, value in enumerate(values):
        if not value:
            return value, list(range(i + 1))
    return values[-1], list(range(len(values)))


def _python_or(values):
    for i, value in enumerate(values):
        if value:
            return value, list(range(i + 1))
    return values[-1], list(range(len(values)))


def test_walrus_in_later_operand_binds_in_the_function():
    def f(x):
        return (x > 0 and (y := x + 1)), y

    assert ASTRewriter.transform(f)(2) == (3, 3)


def _walrus_in_a_dynamic_chain(a, b):
    return (a and (c := b)), c


def test_walrus_binds_in_the_function_even_when_the_chain_is_dynamic():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            a, b = (Boolean(arith_const(1, ir.IntegerType.get_signless(1))) for _ in range(2))
            chain, bound = ASTRewriter.transform(_walrus_in_a_dynamic_chain)(a, b)
        names = [op.operation.name for op in module.body.operations]
    assert bound is b
    assert isinstance(chain, Boolean)
    assert names.count("arith.andi") == 1


def _dynamic_chains(a, b, c, d):
    return (a and b and c and d), (a or b or c or d)


def test_dynamic_chains_emit_one_op_per_later_operand():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            operands = [Boolean(arith_const(1, ir.IntegerType.get_signless(1))) for _ in range(4)]
            ASTRewriter.transform(_dynamic_chains)(*operands)
        names = [op.operation.name for op in module.body.operations]
    assert names.count("arith.andi") == 3
    assert names.count("arith.ori") == 3


def _located_operand(a):
    return a and capture_user_location()  # operand


def test_later_operand_reports_its_own_position_without_a_thunk_frame():
    fn = ASTRewriter.transform(_located_operand)
    source = _located_operand.__code__.co_filename
    with open(source) as f:
        line, text = next((i, t) for i, t in enumerate(f, 1) if t.rstrip().endswith("# operand"))
    col = text.index("capture_user_location") if sys.version_info >= (3, 11) else 0
    with ir.Context(), ir.Location.unknown(), tracing_context(fn):
        loc = fn(True)
    # The thunk lends its position to the function frame instead of adding a call-site level.
    assert str(loc) == f'loc("{source}":{line}:{col})'
