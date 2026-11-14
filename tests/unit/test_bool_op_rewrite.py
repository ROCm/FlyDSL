# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``and`` / ``or`` chains of plain bools short-circuit like Python and evaluate each operand at most once."""

import pytest

from flydsl.compiler.ast_rewriter import ASTRewriter

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
