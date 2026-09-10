#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CLI regression tests; run directly without FlyDSL, native libraries or pytest."""

import difflib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.l0_backend_agnostic


_SCANNER = Path(__file__).resolve().parents[2] / ".claude/skills/review-flydsl-kernel/scan_unreachable_tests.py"


class TestReviewUnreachableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "sample.py"
        self.diff = self.root / "change.diff"

    def scan(self, after, before="", *, diff=None, head=None, root=None):
        after = textwrap.dedent(after).lstrip("\n")
        before = textwrap.dedent(before).lstrip("\n")
        self.source.write_text(after if head is None else head)
        self.diff.write_text(
            diff
            if diff is not None
            else "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile="a/sample.py",
                    tofile="b/sample.py",
                )
            )
        )
        return subprocess.run(
            [sys.executable, str(_SCANNER), "--diff", str(self.diff), str(root or self.root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_status(self, result, expected):
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_pytest_whole_file_covers_functions_and_methods(self):
        result = self.scan("""
            import pytest
            def test_added():
                pass
            class TestAdded:
                def test_method(self):
                    pass
            if __name__ == "__main__":
                pytest.main([__file__])
        """)
        self.assert_status(result, 0)
        self.assertIn("2 added test definition line(s) have a statically visible", result.stdout)

    def test_pytest_aliases_and_keyword_arguments(self):
        for imports, call in [
            ("import pytest as pt", "pt.main([__file__])"),
            ("import pytest as pt, sys", "pt.main([__file__])"),
            ("from pytest import main", "main([__file__])"),
            ("from pytest import main as run_tests", "run_tests(args=[__file__])"),
            ("import pytest", "pytest.main([__file__, '-q', '--tb=short'])"),
        ]:
            with self.subTest(call=call):
                result = self.scan(
                    f"{imports}\ndef test_added():\n    pass\n" f'if __name__ == "__main__":\n    {call}\n'
                )
                self.assert_status(result, 0)

    def test_pytest_alias_in_called_helper(self):
        result = self.scan("""
            def test_added():
                pass
            def run():
                import pytest as pt
                return pt.main([__file__])
            if __name__ == "__main__":
                raise SystemExit(run())
        """)
        self.assert_status(result, 0)

    def test_selectors_and_dynamic_pytest_arguments_are_unknown(self):
        for call in [
            "pytest.main([__file__, '-k', 'existing'])",
            "pytest.main([__file__, '-m', 'slow'])",
            "pytest.main([__file__ + '::test_existing'])",
            "pytest.main(['other.py'])",
            "pytest.main(sys.argv[1:])",
            "pytest.main()",
            "pytest.main([__file__], plugins=plugins)",
        ]:
            with self.subTest(call=call):
                result = self.scan(
                    "import pytest\ndef test_added():\n    pass\n" f'if __name__ == "__main__":\n    {call}\n'
                )
                self.assert_status(result, 1)
                self.assertIn("whole-file coverage is unknown", result.stdout)
                self.assertNotIn("have a statically visible", result.stdout)

    def test_class_method_is_not_confused_with_reached_global_function(self):
        before = """
            def test_same():
                pass
            if __name__ == "__main__":
                test_same()
        """
        result = self.scan(
            """
            class TestAdded:
                def test_same(self):
                    pass
            def test_same():
                pass
            if __name__ == "__main__":
                test_same()
        """,
            before,
        )
        self.assert_status(result, 1)
        self.assertIn("TestAdded.test_same:2", result.stdout)
        self.assertNotIn("have a statically visible", result.stdout)

    def test_same_method_name_in_two_classes_keeps_both_candidates(self):
        result = self.scan("""
            class TestFirst:
                def test_same(self):
                    pass
            class TestSecond:
                def test_same(self):
                    pass
            if __name__ == "__main__":
                pass
        """)
        self.assert_status(result, 1)
        self.assertIn("TestFirst.test_same:2", result.stdout)
        self.assertIn("TestSecond.test_same:5", result.stdout)

    def test_only_added_definition_lines_are_reported(self):
        before = """
            class TestExisting:
                def test_old(self):
                    pass
            def test_existing():
                pass
            if __name__ == "__main__":
                test_existing()
        """
        result = self.scan(
            """
            class TestExisting:
                def test_old(self):
                    pass
            def test_new():
                pass
            def test_existing():
                pass
            if __name__ == "__main__":
                test_existing()
        """,
            before,
        )
        self.assert_status(result, 1)
        self.assertIn("test_new:4", result.stdout)
        self.assertNotIn("TestExisting.test_old", result.stdout)
        self.assertIn("1 of 1 ADDED", result.stdout)

    def test_multiline_string_is_not_a_test_definition(self):
        result = self.scan('''
            example = """
            def test_fake():
                pass
            """
            if __name__ == "__main__":
                pass
        ''')
        self.assert_status(result, 0)
        self.assertIn("no test definitions added", result.stdout)

    def test_signature_edit_is_a_candidate_but_body_only_edit_is_not(self):
        before = 'def test_existing():\n    pass\nif __name__ == "__main__":\n    pass\n'
        result = self.scan(before.replace("test_existing()", "test_existing(value=1)"), before)
        self.assert_status(result, 1)
        self.assertIn("ADDED test definition line", result.stdout)
        result = self.scan(before.replace("    pass", "    print('changed')", 1), before)
        self.assert_status(result, 0)
        self.assertIn("no test definitions added", result.stdout)

    def test_empty_diff_and_deleted_files_have_no_candidates(self):
        for diff in [
            "",
            "diff --git a/deleted.py b/deleted.py\n--- a/deleted.py\n+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n-def test_old():\n-    pass\n",
        ]:
            with self.subTest(diff=diff):
                result = self.scan("", diff=diff)
                self.assert_status(result, 0)
                self.assertIn("no test definitions added", result.stdout)

    def test_transitive_calls_and_recursion(self):
        result = self.scan("""
            def test_added():
                pass
            def run_all():
                wrapper()
            def wrapper():
                run_all()
                test_added()
            if "__main__" == __name__:
                run_all()
        """)
        self.assert_status(result, 0)

    def test_attribute_call_does_not_reach_same_named_global(self):
        result = self.scan("""
            def test_added():
                pass
            if __name__ == "__main__":
                other.test_added()
        """)
        self.assert_status(result, 1)
        self.assertIn("test_added:1", result.stdout)

    def test_uncalled_nested_function_does_not_reach_tests(self):
        result = self.scan("""
            def test_added():
                pass
            def run():
                def unused():
                    test_added()
            if __name__ == "__main__":
                run()
        """)
        self.assert_status(result, 1)

    def test_main_guard_requires_name_equality(self):
        for guard in ['mode == "__main__"', '__name__ != "__main__"', '"__main__" in modes']:
            with self.subTest(guard=guard):
                result = self.scan(f"def test_added():\n    pass\nif {guard}:\n    test_added()\n")
                self.assert_status(result, 0)
                self.assertIn("no supported __main__ guard", result.stdout)
                self.assertNotIn("have a statically visible", result.stdout)

    def test_async_direct_call_needs_review(self):
        result = self.scan("""
            async def test_added():
                pass
            if __name__ == "__main__":
                test_added()
        """)
        self.assert_status(result, 1)

    def test_unconsumed_generator_does_not_reach_tests(self):
        result = self.scan("""
            def test_added():
                pass
            def run():
                yield 1
                test_added()
            if __name__ == "__main__":
                run()
        """)
        self.assert_status(result, 1)

    def test_shadowed_function_or_pytest_alias_cannot_prove_coverage(self):
        for source in [
            """
            def test_added():
                pass
            def run(test_added):
                test_added()
            if __name__ == "__main__":
                run(other)
        """,
            """
            import pytest as pt
            def test_added():
                pass
            def run(pt):
                pt.main([__file__])
            if __name__ == "__main__":
                run(other)
        """,
            """
            import pytest
            def test_added():
                pass
            __file__ = "other.py"
            if __name__ == "__main__":
                pytest.main([__file__])
        """,
        ]:
            with self.subTest(source=source):
                self.assert_status(self.scan(source), 1)

    def test_duplicate_global_definitions_do_not_collapse_line_identity(self):
        result = self.scan("""
            def test_same():
                pass
            def test_same():
                pass
            if __name__ == "__main__":
                test_same()
        """)
        self.assert_status(result, 1)
        self.assertIn("test_same:1", result.stdout)
        self.assertIn("test_same:3", result.stdout)

    def test_pytest_cannot_cover_overwritten_same_named_definitions(self):
        result = self.scan("""
            import pytest
            def test_same():
                pass
            def test_same():
                pass
            if __name__ == "__main__":
                pytest.main([__file__])
        """)
        self.assert_status(result, 1)
        self.assertIn("test_same:2", result.stdout)
        self.assertIn("test_same:4", result.stdout)

    def test_source_that_looks_like_diff_metadata_remains_source(self):
        result = self.scan('''
            notes = """
            ++ b/other.py
            -- separator
            """
            def test_added():
                pass
            if __name__ == "__main__":
                pass
        ''')
        self.assert_status(result, 1)
        self.assertIn("test_added:5", result.stdout)

    def test_invalid_input_is_exit_two(self):
        for kwargs in [
            {"head": "def test_different():\n    pass\n"},
            {"root": self.root / "missing"},
            {"root": self.source},
            {"diff": "not a diff"},
            {"diff": "--- a/sample.py\n+++ b/sample.py\n@@ -0,0 +1,2 @@\n+pass\n"},
            {"diff": "diff --git a/sample.py b/sample.py\n@@ -0,0 +1 @@\n+pass\n"},
            {"diff": "--- a/sample.py\n+++ b/sample.py\n@@ -0,0 +1 @@\n+pass\n+extra\n"},
        ]:
            with self.subTest(kwargs=kwargs):
                result = self.scan("def test_added():\n    pass\n", **kwargs)
                self.assert_status(result, 2)
                self.assertIn("input error:", result.stderr)
                self.assertNotIn("no test definitions", result.stdout)

    def test_missing_head_and_diff_files_are_exit_two(self):
        result = self.scan("def test_added():\n    pass\n")
        self.assert_status(result, 0)
        self.source.unlink()
        result = subprocess.run(
            [sys.executable, str(_SCANNER), "--diff", str(self.diff), str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assert_status(result, 2)
        self.diff.unlink()
        result = subprocess.run(
            [sys.executable, str(_SCANNER), "--diff", str(self.diff), str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assert_status(result, 2)

    def test_syntax_error_is_exit_two(self):
        result = self.scan("def test_added(:\n    pass\n")
        self.assert_status(result, 2)
        self.assertIn("input error:", result.stderr)

    def test_reviewed_source_is_never_executed(self):
        result = self.scan("""
            raise RuntimeError("reviewed code must not execute")
            def test_added():
                pass
            if __name__ == "__main__":
                test_added()
        """)
        self.assert_status(result, 0)


if __name__ == "__main__":
    unittest.main()
