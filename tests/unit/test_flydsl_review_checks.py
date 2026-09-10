#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Regression tests for the combined FlyDSL review-check entry point."""

import difflib
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


_CHECKER = Path(__file__).resolve().parents[2] / ".claude/skills/flydsl-review-checks/check_candidates.py"


class TestFlyDSLReviewChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.head = self.root / "head"
        (self.head / "kernels").mkdir(parents=True)
        self.diff = self.root / "change.diff"

    def run_checks(self, source):
        path = self.head / "kernels/sample.py"
        path.write_text(source, encoding="utf-8")
        self.diff.write_text(
            "".join(
                difflib.unified_diff(
                    [],
                    source.splitlines(keepends=True),
                    fromfile="a/kernels/sample.py",
                    tofile="b/kernels/sample.py",
                )
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable,
                str(_CHECKER),
                "--diff",
                str(self.diff),
                "--head",
                str(self.head),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_clean_inputs_run_both_checks(self):
        result = self.run_checks("value = fx.Float32(0.0)\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("== legacy API spelling candidates ==", result.stdout)
        self.assertIn("== added test entry-point candidates ==", result.stdout)
        self.assertIn("no legacy spelling candidates", result.stdout)
        self.assertIn("no test definitions added", result.stdout)

    def test_candidates_from_both_checks_share_exit_one(self):
        result = self.run_checks(
            "value = ir.Type()\n" "def test_added():\n" "    pass\n" 'if __name__ == "__main__":\n' "    pass\n"
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("raw ir.* / ArithValue", result.stdout)
        self.assertIn("test_added:2", result.stdout)

    def test_input_failure_is_not_clean_and_both_checks_run(self):
        result = subprocess.run(
            [
                sys.executable,
                str(_CHECKER),
                "--diff",
                str(self.root / "missing.diff"),
                "--head",
                str(self.root / "missing-head"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("== legacy API spelling candidates ==", result.stdout)
        self.assertIn("== added test entry-point candidates ==", result.stdout)
        self.assertIn("error:", result.stderr)
        self.assertIn("input error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
