#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Run the repository's non-style pre-checks behind one entry point.

Each of these checks used to be its own step in the ``python-style`` job, so
adding one meant editing ``.github/workflows/pre-checks.yaml`` and re-plumbing
``BASE_SHA`` / ``HEAD_SHA`` by hand, and a contributor had to know each script
by name to reproduce CI locally.

Registering a check here instead means:

* the workflow keeps a single step, whatever the check count;
* base/head resolution is written once and shared;
* ``python3 scripts/check_repo.py`` reproduces all of them locally;
* every check runs even when an earlier one fails, so one run shows everything.

To add a check, append to ``CHECKS``. Do not add a workflow step.

The black/ruff and clang-format gates are deliberately *not* here: they run
through ``.github/scripts/check_{python,cpp}_style.sh``, which are wired to
reviewdog for inline PR annotations. Reproduce those with their own wrappers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Check:
    name: str
    script: str
    summary: str
    # Checks that understand --base/--head get the shared revision range.
    takes_revisions: bool = True
    extra_args: list[str] = field(default_factory=list)


CHECKS: list[Check] = [
    Check(
        name="typed-arithmetic",
        script="scripts/check_typed_arithmetic_usage.py",
        summary="reject newly added legacy arithmetic spellings in kernel source",
    ),
    Check(
        name="agent-docs",
        script="scripts/check_docs_api.py",
        summary="CLAUDE.md and .claude/skills/ still match the Python surface",
    ),
]


def _resolve_base(head: str) -> str:
    env = os.environ.get("BASE_SHA", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "merge-base", head, "origin/main"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _run(check: Check, base: str, head: str) -> int:
    argv = [sys.executable, str(REPO / check.script), *check.extra_args]
    if check.takes_revisions:
        if base:
            argv += ["--base", base]
        argv += ["--head", head]
    print(f"\n=== {check.name}: {check.summary} ===", flush=True)
    return subprocess.run(argv, cwd=REPO).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="base commit; defaults to BASE_SHA or merge-base with origin/main")
    ap.add_argument("--head", default=os.environ.get("HEAD_SHA", "HEAD"), help="head commit")
    ap.add_argument("--only", action="append", metavar="NAME", help="run only the named check (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the registered checks and exit")
    args = ap.parse_args()

    if args.list:
        for c in CHECKS:
            print(f"{c.name:<20} {c.script:<42} {c.summary}")
        return 0

    selected = CHECKS
    if args.only:
        names = {n for n in args.only}
        unknown = names - {c.name for c in CHECKS}
        if unknown:
            print(f"unknown check(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"available: {', '.join(c.name for c in CHECKS)}", file=sys.stderr)
            return 2
        selected = [c for c in CHECKS if c.name in names]

    base = args.base if args.base is not None else _resolve_base(args.head)

    # Run every check even after a failure: one run should surface everything a
    # contributor has to fix, not just the first problem.
    failed = [c.name for c in selected if _run(c, base, args.head) != 0]

    print()
    if failed:
        print(f"check_repo: {len(failed)} of {len(selected)} check(s) FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"check_repo: all {len(selected)} check(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
