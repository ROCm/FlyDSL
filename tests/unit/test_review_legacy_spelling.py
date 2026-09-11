#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Offline CLI regressions; also runnable without pytest or a FlyDSL build."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.l0_backend_agnostic

SCRIPT = Path(__file__).resolve().parents[2] / ".claude/skills/flydsl-code-review/scripts/scan_legacy_spelling.py"
SPEC = importlib.util.spec_from_file_location("scan_legacy_spelling", SCRIPT)
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)


def added_diff(source, path="kernels/example.py", start=1):
    lines = source.splitlines()
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -{start - 1},0 +{start},{len(lines)} @@\n" + "".join(f"+{line}\n" for line in lines)
    )


class LegacySpellingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="legacy-spelling-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = self.root / "input.diff"
        self.patch.write_text("", encoding="utf-8")

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)

    def scan_cli(self, diff):
        self.patch.write_text(diff, encoding="utf-8")
        return self.cli("--diff", self.patch)

    def test_invalid_arguments_report_usage(self):
        for args in [
            (),
            ("--diff",),
            ("--diff", self.patch, "extra"),
        ]:
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_unreadable_and_invalid_text_diff_fail(self):
        result = self.cli("--diff", self.root / "missing.diff")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.patch.write_bytes(b"\xff")
        result = self.cli("--diff", self.patch)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)

    def test_empty_diff_and_clean_kernel_exit_zero(self):
        for diff in ["", added_diff("value = fx.Float32(0.0)\nvalue = fx.max(value, peer)")]:
            with self.subTest(diff=diff):
                result = self.scan_cli(diff)
                self.assertEqual(result.returncode, 0)
                self.assertIn("no legacy spelling candidates", result.stdout)

    def test_malformed_or_truncated_diff_fails(self):
        for diff in [
            "not a diff\n",
            added_diff("value = ir.Type()").replace("+1,1", "+1,2"),
            added_diff("value = ir.Type()").replace("@@ -0,0 +1,1 @@", "@@ broken @@"),
            added_diff("value = ir.Type()").replace("+++ b/kernels/example.py\n", ""),
            added_diff("value = ir.Type()").replace("+++ b/kernels/example.py", "+++ unknown.py"),
            added_diff("value = ir.Type()") + "+unaccounted = ir.Type()\n",
        ]:
            with self.subTest(diff=diff):
                result = self.scan_cli(diff)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)

    def test_only_kernel_consumers_are_candidates(self):
        for path in [
            "python/flydsl/expr/example.py",
            "tests/example.py",
            ".claude/skills/flydsl-code-review/scripts/scan_legacy_spelling.py",
            "kernels/common/buffer_ops.py",
            "kernels/example.md",
        ]:
            with self.subTest(path=path):
                self.assertEqual(SCANNER.scan(added_diff("value = buffer_ops.load()", path)), [])
        hits = SCANNER.scan(added_diff("value = buffer_ops.load()", "kernels/subdir/example.py"))
        self.assertEqual(len(hits), 1)

    def test_all_five_rules_have_candidate_output(self):
        source = "\n".join(
            [
                "i32_type = ir.IntegerType.get_signless(32)",
                "if_op = scf.IfOp(cond_i1, [], has_else=False)",
                "value = buffer_ops.buffer_load(rsrc, index, vec_width=1, dtype=fx.Int32)",
                'allocator = SmemAllocator(None, arch="gfx942")',
                "pointer = fx.make_ptr(pointer_type, [addr])",
            ]
        )
        result = self.scan_cli(added_diff(source, start=9))
        self.assertEqual(result.returncode, 1)
        for line in range(9, 14):
            self.assertIn(f"kernels/example.py:{line}:", result.stdout)
        self.assertIn("coderfeli #33 #433 #540 #582", result.stdout)
        self.assertIn("ordinary pointer construction is valid", result.stdout)

    def test_scf_operation_wrappers_and_grouped_output(self):
        source = "\n".join(
            [
                "if_op = scf.IfOp(cond_i1, [], has_else=False)",
                "loop = scf.ForOp(lower, upper, step)",
                "wait_loop = scf.WhileOp([T.i32], [init_cur])",
                "scf.YieldOp([cur])",
            ]
        )
        diff = added_diff(source, start=9)
        self.assertEqual([hit[1] for hit in SCANNER.scan(diff)], [9, 10, 11, 12])
        result = self.scan_cli(diff)
        self.assertEqual(result.returncode, 1)
        self.assertIn("kernels/example.py:9: scf.* control flow x4", result.stdout)
        self.assertEqual(result.stdout.count("maintainers:"), 1)

    def test_comments_strings_and_imports_are_ignored(self):
        source = '''# ir.Type() SmemAllocator()
value = fx.Float32(0)  # buffer_ops.load()
example = "ir.Type() and make_ptr(value)"
other = 'scf.For() and SmemAllocator()'
text = """ir.Type()
SmemAllocator()
buffer_ops.load()
"""
from flydsl._mlir import ir
from flydsl.utils.smem_allocator import (
    SmemAllocator,
)
from flydsl.expr import make_ptr
import flydsl._mlir.ir
'''
        self.assertEqual(SCANNER.scan(added_diff(source)), [])

    def test_code_after_string_or_import_is_still_checked(self):
        source = 'text = "ir.Type()"; value = ir.Type()\nimport flydsl._mlir.ir; scf.For()'
        hits = SCANNER.scan(added_diff(source))
        self.assertEqual([(hit[1], hit[2]) for hit in hits], [(1, "raw ir.* / ArithValue"), (2, "scf.* control flow")])

    def test_fstring_literal_text_is_ignored(self):
        source = 'message = f"prefer SmemAllocator and ir.Type() for {name}"'
        self.assertEqual(SCANNER.scan(added_diff(source)), [])

    @unittest.skipUnless(sys.version_info >= (3, 12), "f-string expression tokens require Python 3.12")
    def test_fstring_interpolated_code_is_still_checked(self):
        source = 'message = f"prefer SmemAllocator: {ir.Type()}"'
        hits = SCANNER.scan(added_diff(source))
        self.assertEqual([(hit[1], hit[2]) for hit in hits], [(1, "raw ir.* / ArithValue")])

    def test_context_preserves_multiline_string_and_import_state(self):
        diff = '''diff --git a/kernels/example.py b/kernels/example.py
--- a/kernels/example.py
+++ b/kernels/example.py
@@ -4,4 +4,6 @@
 text = """
+ir.Type() and SmemAllocator()
 """
 from flydsl.utils.smem_allocator import (
+    SmemAllocator,
 )
'''
        self.assertEqual(SCANNER.scan(diff), [])

    def test_incomplete_hunk_statements_and_strings_are_supported(self):
        self.assertEqual(SCANNER.scan(added_diff('text = """\nir.Type()\nSmemAllocator()')), [])
        hits = SCANNER.scan(added_diff("result = function(\n    ir.Type(),"))
        self.assertEqual([(hit[1], hit[2]) for hit in hits], [(2, "raw ir.* / ArithValue")])

    def test_only_added_lines_are_reported_with_new_line_numbers(self):
        diff = """diff --git a/kernels/example.py b/kernels/example.py
--- a/kernels/example.py
+++ b/kernels/example.py
@@ -7,4 +7,4 @@
 old = ir.Type()
-removed = buffer_ops.load()
+current = fx.Float32(0)
 unchanged = SmemAllocator()
+new = scf.For()
-removed = ir.Type()
"""
        hits = SCANNER.scan(diff)
        self.assertEqual([(hit[0], hit[1], hit[2]) for hit in hits], [("kernels/example.py", 10, "scf.* control flow")])

    def test_mixed_files_and_hunks_reset_paths_and_line_numbers(self):
        diff = (
            added_diff("ir.Type()", start=4)
            + "@@ -20,0 +21 @@\n+scf.For()\n"
            + added_diff("buffer_ops.load()", "docs/example.py", start=8)
            + "diff --git a/kernels/deleted.py b/kernels/deleted.py\n--- a/kernels/deleted.py\n+++ /dev/null\n"
            + "@@ -1 +0,0 @@\n-SmemAllocator()\n"
            + added_diff("SmemAllocator()", "kernels/second.py", start=12)
        )
        hits = SCANNER.scan(diff)
        self.assertEqual(
            [(hit[0], hit[1]) for hit in hits],
            [("kernels/example.py", 4), ("kernels/example.py", 21), ("kernels/second.py", 12)],
        )

    def test_source_lines_resembling_headers_remain_in_the_hunk(self):
        diff = """diff --git a/kernels/example.py b/kernels/example.py
--- a/kernels/example.py
+++ b/kernels/example.py
@@ -1,2 +1 @@
--- separator
-old = 0
+++scf.For()
\\ No newline at end of file
"""
        hits = SCANNER.scan(diff)
        self.assertEqual([(hit[0], hit[1], hit[2]) for hit in hits], [("kernels/example.py", 1, "scf.* control flow")])


if __name__ == "__main__":
    unittest.main()
