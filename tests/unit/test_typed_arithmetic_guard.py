#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

from scripts.check_typed_arithmetic_usage import scan_source


@pytest.mark.parametrize(
    "source",
    [
        "result = value.maximumf(peer)",
        "result = value.minimumf(peer)",
        "from flydsl.expr import arith\nresult = arith.maximumf(a, b)",
        "from flydsl._mlir.dialects import arith as raw\nresult = raw.ceildivui(a, b)",
        "from flydsl.expr.arith import maxsi as raw_max\nresult = raw_max(a, b)",
        "import flydsl.expr.arith as typed_arith\nresult = typed_arith.maxnumf(a, b)",
    ],
)
def test_guard_rejects_new_legacy_arithmetic(source):
    assert scan_source(source)


@pytest.mark.parametrize(
    "source",
    [
        "result = fx.max(a, b)",
        "result = fx.min(a, b)",
        "result = fx.maxnumf(a, b)",
        "result = fx.ceildiv(a, b)",
        "result = (a + b - 1) // b",
    ],
)
def test_guard_accepts_typed_arithmetic(source):
    assert scan_source(source) == []


def test_guard_checks_only_added_lines():
    source = "old = value.maximumf(peer)\nnew = fx.max(value, peer)\n"
    assert scan_source(source, {2}) == []
    assert scan_source(source, {1})
