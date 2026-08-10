#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reject newly added legacy arithmetic spellings in kernel source."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_RAW_REPLACEMENTS = {
    "maximumf": "fx.max",
    "minimumf": "fx.min",
    "maxsi": "fx.max",
    "maxui": "fx.max",
    "minsi": "fx.min",
    "minui": "fx.min",
    "maxnumf": "fx.maxnumf",
    "ceildivsi": "fx.ceildiv",
    "ceildivui": "fx.ceildiv",
}
_METHOD_REPLACEMENTS = {
    "maximumf": "fx.max",
    "minimumf": "fx.min",
}
_ARITH_MODULES = {
    "flydsl.expr.arith",
    "flydsl._mlir.dialects.arith",
}
_ARITH_PARENTS = {
    "flydsl.expr",
    "flydsl._mlir.dialects",
}
_HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Violation:
    line: int
    column: int
    spelling: str
    replacement: str


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _arith_bindings(tree: ast.AST) -> tuple[set[str], dict[str, str]]:
    module_aliases: set[str] = set()
    direct_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in _ARITH_PARENTS:
                for alias in node.names:
                    if alias.name == "arith":
                        module_aliases.add(alias.asname or alias.name)
            elif node.module in _ARITH_MODULES:
                for alias in node.names:
                    if alias.name in _RAW_REPLACEMENTS:
                        direct_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _ARITH_MODULES and alias.asname:
                    module_aliases.add(alias.asname)
    return module_aliases, direct_aliases


def scan_source(source: str, added_lines: set[int] | None = None) -> list[Violation]:
    tree = ast.parse(source)
    module_aliases, direct_aliases = _arith_bindings(tree)
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        if added_lines is not None and not any(line in added_lines for line in range(start, end + 1)):
            continue

        spelling = None
        replacement = None
        if isinstance(node.func, ast.Attribute):
            dotted = _dotted_name(node.func)
            if node.func.attr in _METHOD_REPLACEMENTS:
                spelling = f".{node.func.attr}(...)"
                replacement = _METHOD_REPLACEMENTS[node.func.attr]
            elif node.func.attr in _RAW_REPLACEMENTS:
                owner = _dotted_name(node.func.value)
                if owner in module_aliases or any(dotted == f"{module}.{node.func.attr}" for module in _ARITH_MODULES):
                    spelling = f"{owner}.{node.func.attr}(...)"
                    replacement = _RAW_REPLACEMENTS[node.func.attr]
        elif isinstance(node.func, ast.Name) and node.func.id in direct_aliases:
            raw_name = direct_aliases[node.func.id]
            spelling = f"{node.func.id}(...)"
            replacement = _RAW_REPLACEMENTS[raw_name]

        if spelling is not None:
            violations.append(Violation(start, node.col_offset + 1, spelling, replacement))
    return violations


def _added_kernel_lines(base: str, head: str) -> dict[Path, set[int]]:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "--diff-filter=ACMR", base, head, "--", "kernels"],
        check=True,
        capture_output=True,
        text=True,
    )
    added: dict[Path, set[int]] = {}
    current_path = None
    next_line = None

    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_path = Path(line[6:])
            if current_path.suffix == ".py":
                added.setdefault(current_path, set())
            else:
                current_path = None
            continue
        match = _HUNK_RE.match(line)
        if match:
            next_line = int(match.group(1))
            continue
        if current_path is None or next_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added[current_path].add(next_line)
            next_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            next_line += 1
    return {path: lines for path, lines in added.items() if lines}


def _default_base(head: str) -> str:
    env_base = os.environ.get("BASE_SHA", "").strip()
    if env_base and set(env_base) != {"0"}:
        return env_base
    return subprocess.check_output(["git", "merge-base", head, "origin/main"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base commit; defaults to BASE_SHA or merge-base with origin/main")
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA", "HEAD"), help="head commit")
    args = parser.parse_args()

    base = args.base or _default_base(args.head)
    files = _added_kernel_lines(base, args.head)
    violations = []
    for path, lines in files.items():
        try:
            source = path.read_text()
            found = scan_source(source, lines)
        except (OSError, SyntaxError) as exc:
            print(f"{path}: unable to scan: {exc}")
            return 2
        violations.extend((path, violation) for violation in found)

    if violations:
        print("New legacy typed-arithmetic usage is not allowed:")
        for path, violation in violations:
            print(
                f"  {path}:{violation.line}:{violation.column}: " f"{violation.spelling} -> use {violation.replacement}"
            )
        return 1

    print(f"Typed arithmetic usage check passed ({base}..{args.head}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
