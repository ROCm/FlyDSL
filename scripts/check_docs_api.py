#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Check that agent-facing documentation matches the actual Python surface.

CLAUDE.md and the skills under ``.claude/skills/`` are loaded by coding agents and
are acted on directly, so a stale API name or a path that no longer resolves is a
defect with the same blast radius as broken code -- but nothing else in CI reads
them. ``check_typed_arithmetic_usage.py`` scans ``.py`` only.

Four checks, all static (this script never imports FlyDSL):

1. ``fx.<name>`` / ``rocdl.<name>`` symbols resolve.
2. Repo-relative paths in backticks exist.
3. Skill frontmatter parses as YAML, uses only recognized keys, and its ``name``
   matches the directory the command is derived from.
4. ``**<name>** skill`` cross-references point at a skill that exists.

The symbol universe is the union of what ``python/flydsl/expr/`` defines and what
``kernels/``, ``tests/`` and ``examples/`` actually call. The second half matters:
``fx.rocdl`` re-exports the upstream MLIR ROCDL dialect with ``import *``, so
symbols like ``s_wait_dscnt`` have no in-tree definition and can only be
recognized by the fact that real kernels use them.

Placeholders in prose are skipped: anything containing ``*``, ``<`` or ``>``, and
the names in ``_PLACEHOLDERS``. A line may opt out entirely with a trailing
``<!-- api-check: ignore -->``.

Scope is deliberately limited to the always-loaded and agent-invoked files. The
published ``docs/`` tree is not covered yet -- see ``--include-docs``.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files whose prose is executed by an agent rather than read by a human.
DEFAULT_TARGETS = ["CLAUDE.md", ".claude/skills/*/SKILL.md"]
DOCS_TARGETS = ["docs/**/*.md", "docs/**/*.rst"]

# Trees whose call sites vouch for a symbol that has no in-tree definition
# (upstream MLIR dialect re-exports, generated ODS builders).
USAGE_TREES = ["kernels", "tests", "examples"]

# Frontmatter keys Claude Code recognizes. An unrecognized key is silently
# ignored at load time, which is how `tools:` and `note:` went unnoticed.
SKILL_KEYS = {
    "name",
    "description",
    "when_to_use",
    "argument-hint",
    "arguments",
    "disable-model-invocation",
    "user-invocable",
    "allowed-tools",
    "disallowed-tools",
    "model",
    "effort",
    "context",
    "agent",
    "background",
    "hooks",
    "paths",
    "shell",
    "metadata",
    "license",
    "compatibility",
}

# Names that only ever appear as stand-ins in explanatory prose.
_PLACEHOLDERS = {"foo", "bar", "baz", "qux", "Tensor_", "my_kernel", "something"}

# Upstream MLIR ROCDL ops that `from ..._mlir.dialects.rocdl import *` re-exports
# but no in-tree caller uses yet, so _used_symbols() cannot vouch for them.
# Verified against mlir/dialects/_rocdl_ops_gen.py; drop an entry once a kernel
# starts calling it (it will then be recognized automatically).
_UPSTREAM_ROCDL = {"s_wait_loadcnt", "s_wait_storecnt"}

# Only these fences hold Python. MLIR assembly, TableGen and C++ legitimately
# spell ops as `rocdl.mfma.f32...`, which is not a Python attribute reference.
_PY_FENCES = {"python", "py", ""}

_PATH_PREFIXES = (
    "kernels/",
    "python/",
    "tests/",
    "lib/",
    "include/",
    "scripts/",
    "examples/",
    "docs/",
    ".github/",
    ".claude/",
)

# The lookbehind keeps fx./rocdl. from matching inside a URL, path or filename.
_SYM = re.compile(r"(?<![\w./-])(?:fx\.rocdl|fx|rocdl)\.([A-Za-z_][A-Za-z0-9_]*)")
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_SKILL_REF = re.compile(r"\*\*([a-z][a-z0-9-]+)\*\* skill")
_IGNORE = "<!-- api-check: ignore -->"


def _defined_symbols() -> set[str]:
    """Every name ``python/flydsl/expr/`` binds at module scope, plus module names."""
    out: set[str] = set()
    root = REPO / "python" / "flydsl" / "expr"
    for path in root.rglob("*.py"):
        out.add(path.stem)
        out.add(path.parent.name)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        out.add(tgt.id)
                    # __all__ = [...] enumerates the re-exported surface.
                    if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                        for elt in getattr(node.value, "elts", []):
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.add(elt.value)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    out.add(alias.asname or alias.name.split(".")[0])
    return out


def _used_symbols() -> set[str]:
    """Names real call sites reach through ``fx.`` / ``rocdl.``.

    This is what recognizes upstream ROCDL ops that `import *` pulls in.
    """
    out: set[str] = set()
    for tree_name in USAGE_TREES:
        tree = REPO / tree_name
        if not tree.is_dir():
            continue
        for path in tree.rglob("*.py"):
            try:
                out.update(_SYM.findall(path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                continue
    return out


def _iter_targets(patterns: list[str]) -> list[Path]:
    seen: list[Path] = []
    for pat in patterns:
        for p in sorted(REPO.glob(pat)):
            if p.is_file():
                seen.append(p)
    return seen


def _check_symbols(path: Path, text: str, known: set[str]) -> list[str]:
    bad = []
    fence: str | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            # Only the first word of the info string is the language; the rest
            # can be attributes. "python3" counts as Python too.
            info = stripped[3:].strip().lower().split()
            lang = info[0].rstrip("3") if info else ""
            fence = None if fence is not None else lang
            continue
        if fence is not None and fence not in _PY_FENCES:
            continue  # MLIR / TableGen / C++ / shell block
        if _IGNORE in line:
            continue
        for name in _SYM.findall(line):
            if name in known or name in _PLACEHOLDERS or name in _UPSTREAM_ROCDL:
                continue
            if name.endswith("_"):  # trailing half of a rocdl.mfma_* wildcard
                continue
            bad.append(f"{path.relative_to(REPO)}:{lineno}: unknown symbol `{name}`")
    return bad


def _check_paths(path: Path, text: str) -> list[str]:
    bad = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _IGNORE in line:
            continue
        for tok in _BACKTICKED.findall(line):
            tok = tok.strip()
            if tok.startswith("./"):
                tok = tok[2:]
            if not tok.startswith(_PATH_PREFIXES):
                continue
            if ":" in tok:  # a file:line citation, not a path
                continue
            if any(c in tok for c in "<>*$ "):  # templates and shell fragments
                continue
            if not (REPO / tok.rstrip("/")).exists():
                bad.append(f"{path.relative_to(REPO)}:{lineno}: path does not exist: {tok}")
    return bad


def _split_frontmatter(text: str) -> str | None:
    """Return the frontmatter block, delimited by ``---`` on its own LINE.

    A substring split would cut at the first ``---`` anywhere, so a value like
    ``description: pass --- to separate args`` truncates the block and makes a
    valid file look like invalid YAML.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i])
    return None


def _check_frontmatter(path: Path) -> list[str]:
    rel = path.relative_to(REPO)
    raw = _split_frontmatter(path.read_text(encoding="utf-8"))
    if raw is None:
        return [f"{rel}: no YAML frontmatter"]
    try:
        import yaml
    except ImportError:  # pragma: no cover - CI installs it
        return [f"{rel}: PyYAML unavailable, cannot validate frontmatter"]
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        # Invalid frontmatter is dropped wholesale, taking allowed-tools with it.
        return [f"{rel}: frontmatter is not valid YAML ({str(exc).splitlines()[0]})"]
    if not isinstance(data, dict):
        return [f"{rel}: frontmatter is not a mapping"]
    bad = []
    for key in data:
        if key not in SKILL_KEYS:
            bad.append(f"{rel}: unrecognized frontmatter key `{key}` (silently ignored at load)")
    name, dirname = data.get("name"), path.parent.name
    if name is not None and name != dirname:
        bad.append(f"{rel}: frontmatter name `{name}` != directory `{dirname}`")
    return bad


def _check_skill_refs(path: Path, text: str, skills: set[str]) -> list[str]:
    bad = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _IGNORE in line:
            continue
        for ref in _SKILL_REF.findall(line):
            if ref not in skills:
                bad.append(f"{path.relative_to(REPO)}:{lineno}: no such skill: {ref}")
    return bad


def _changed_files(base: str, head: str) -> list[str] | None:
    """Repo-relative paths changed between *base* and *head*, or None if git fails."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", base, head],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [line for line in out.stdout.splitlines() if line]


def _default_base(head: str) -> str:
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


def _scope_targets(patterns: list[str], base: str, head: str) -> tuple[list[Path], str]:
    """Narrow the scan to what this change can have invalidated.

    Diff-scoping a *cross-reference* check needs care: renaming a symbol under
    ``python/flydsl/`` can invalidate any document, not just the ones the commit
    touched. So a source change widens the scan back to everything, and only a
    docs-only change narrows it.
    """
    everything = _iter_targets(patterns)
    if not base:
        return everything, "whole tree (no base revision)"
    changed = _changed_files(base, head)
    if changed is None:
        return everything, "whole tree (git unavailable)"
    if any(c.startswith("python/flydsl/") for c in changed):
        return everything, "whole tree (python/flydsl changed)"
    changed_abs = {(REPO / c).resolve() for c in changed}
    scoped = [p for p in everything if p.resolve() in changed_abs]
    return scoped, f"{len(scoped)} changed file(s) vs {base[:12]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-docs", action="store_true", help="also scan docs/ (not yet clean; see PR #1027)")
    ap.add_argument("--base", help="base commit; defaults to BASE_SHA or merge-base with origin/main")
    ap.add_argument("--head", default=os.environ.get("HEAD_SHA", "HEAD"), help="head commit")
    ap.add_argument("--all", action="store_true", help="scan every target regardless of the diff")
    args = ap.parse_args()

    known = _defined_symbols() | _used_symbols()
    skills = (
        {p.name for p in (REPO / ".claude" / "skills").iterdir() if p.is_dir()}
        if (REPO / ".claude" / "skills").is_dir()
        else set()
    )

    patterns = list(DEFAULT_TARGETS) + (DOCS_TARGETS if args.include_docs else [])
    if args.all:
        targets, scope = _iter_targets(patterns), "whole tree (--all)"
    else:
        base = args.base if args.base is not None else _default_base(args.head)
        targets, scope = _scope_targets(patterns, base, args.head)
    problems: list[str] = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        problems += _check_symbols(path, text, known)
        problems += _check_paths(path, text)
        problems += _check_skill_refs(path, text, skills)
        if path.name == "SKILL.md":
            problems += _check_frontmatter(path)

    print(
        f"check_docs_api: {scope}; scanned {len(targets)} file(s), " f"{len(known)} known symbols, {len(skills)} skills"
    )
    if problems:
        print(f"\n{len(problems)} problem(s):\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nEach finding is either a real drift or a placeholder the checker cannot "
            "recognize.\nFor a genuine placeholder, append '<!-- api-check: ignore -->' "
            "to the line.",
            file=sys.stderr,
        )
        return 1
    print("check_docs_api: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
