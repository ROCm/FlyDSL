#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from pathlib import Path

import pytest

from scripts.check_typed_arithmetic_usage import _parse_added_kernel_lines, scan_source


@pytest.mark.parametrize(
    "source",
    [
        "result = value.maximumf(peer)",
        "result = value.minimumf(peer)",
        "from flydsl.expr import arith\nresult = arith.maximumf(a, b)",
        "from flydsl._mlir.dialects import arith as raw\nresult = raw.ceildivui(a, b)",
        "from flydsl.expr.arith import maxsi as raw_max\nresult = raw_max(a, b)",
        "import flydsl.expr.arith as typed_arith\nresult = typed_arith.maxnumf(a, b)",
        "import flydsl.expr as fx\ndef kernel(fx, peer):\n    return fx.maximumf(peer)",
        "import flydsl.expr\n\ndef kernel(flydsl, peer):\n    return flydsl.expr.minimumf(peer)",
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
        "import flydsl.expr as fx\nresult = fx.maximumf(a, b)",
        "from flydsl import expr as typed_fx\nresult = typed_fx.minimumf(a, b)",
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


@pytest.mark.parametrize(
    ("diff", "source"),
    [
        (
            """\
diff --git a/kernels/example.py b/kernels/example.py
--- a/kernels/example.py
+++ b/kernels/example.py
@@ -1,2 +1 @@
--- separator
-old = 0
+result = value.maximumf(peer)
""",
            "result = value.maximumf(peer)\n",
        ),
        (
            """\
diff --git a/kernels/example.py b/kernels/example.py
--- a/kernels/example.py
+++ b/kernels/example.py
@@ -1 +1 @@
-old = 0
+++value.maximumf(peer)
""",
            "++value.maximumf(peer)\n",
        ),
    ],
)
def test_diff_parser_does_not_confuse_source_with_file_headers(diff, source):
    path = Path("kernels/example.py")
    added = _parse_added_kernel_lines(diff)
    assert added == {path: {1}}
    assert scan_source(source, added[path])
