#!/usr/bin/env python3
"""Add or update SPDX license headers on git-changed files.

Operates on the union of staged + unstaged + untracked files (or an explicit list).
Skips files that already have a correct, up-to-date header.

License templates are resolved per file type:
  - Own code  -> MIT + AMD copyright
  - LLVM-derived code -> skipped (detected by existing header content)

Supported: .py, .c, .cc, .cpp, .cxx, .h, .hpp, .hxx, .cu, .hip, .sh, .bash
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HEADERS = {
    "mit_py": (
        "# SPDX-License-Identifier: MIT\n"
        "# Copyright (C) 2024-{year}, Advanced Micro Devices, Inc. All rights reserved.\n"
    ),
    "mit_c": (
        "// SPDX-License-Identifier: MIT\n"
        "// Copyright (C) 2024-{year}, Advanced Micro Devices, Inc. All rights reserved.\n"
    ),
    "mit_sh": (
        "# SPDX-License-Identifier: MIT\n"
        "# Copyright (C) 2024-{year}, Advanced Micro Devices, Inc. All rights reserved.\n"
    ),
}

LLVM_MARKERS = [
    "Part of the LLVM Project",
    "Apache-2.0 WITH LLVM-exception",
    "LLVM Exceptions",
]

SPDX_RE = re.compile(r"^(#|//)\s*SPDX-License-Identifier:", re.MULTILINE)
COPYRIGHT_AMD_RE = re.compile(
    r"^(#|//)\s*Copyright\s+\(C\)\s+(\d{4})-(\d{4}),?\s+Advanced Micro Devices",
    re.MULTILINE,
)

EXT_TO_LANG = {}
for _ext in (".py",):
    EXT_TO_LANG[_ext] = "py"
for _ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".cu", ".hip"):
    EXT_TO_LANG[_ext] = "c"
for _ext in (".sh", ".bash"):
    EXT_TO_LANG[_ext] = "sh"


def git_root(path: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def changed_files(repo_root: str) -> list[str]:
    """Return absolute paths of staged + unstaged + untracked files (no deletions)."""
    staged = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "--cached", "--diff-filter=d"],
            cwd=repo_root,
        )
        .decode()
        .splitlines()
    )
    unstaged = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=d"],
            cwd=repo_root,
        )
        .decode()
        .splitlines()
    )
    untracked = (
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
        )
        .decode()
        .splitlines()
    )
    paths = sorted(set(staged + unstaged + untracked))
    return [os.path.join(repo_root, p) for p in paths if p]


def is_llvm_derived(content: str) -> bool:
    return any(marker in content for marker in LLVM_MARKERS)


def process_file(filepath: str, year: int, dry_run: bool = False) -> str | None:
    """Add or update license header. Returns a status string or None if skipped."""
    ext = Path(filepath).suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if lang is None:
        return None

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    if not content.strip():
        return None

    if is_llvm_derived(content):
        return "skipped (LLVM-derived)"

    # Existing SPDX header -> update year if needed
    if SPDX_RE.search(content):
        m = COPYRIGHT_AMD_RE.search(content)
        if m:
            old_end = int(m.group(3))
            if old_end < year:
                prefix = m.group(1)
                start = int(m.group(2))
                old_line = m.group(0)
                new_line = f"{prefix} Copyright (C) {start}-{year}, Advanced Micro Devices, Inc. All rights reserved."
                updated = content.replace(old_line, new_line, 1)
                if not dry_run:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(updated)
                return "updated year"
        return "already has header"

    # Add new header
    hdr = HEADERS[f"mit_{lang}"].format(year=year)
    lines = content.split("\n", 1)

    if lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        rest = lines[1] if len(lines) > 1 else ""
        new_content = shebang + hdr + "\n" + rest
    elif lines[0].startswith("# -*- coding"):
        coding = lines[0] + "\n"
        rest = lines[1] if len(lines) > 1 else ""
        new_content = coding + hdr + "\n" + rest
    else:
        new_content = hdr + "\n" + content

    if not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
    return "added header"


def main():
    parser = argparse.ArgumentParser(description="Add/update SPDX license headers")
    parser.add_argument(
        "files",
        nargs="*",
        help="Explicit files to process (default: git changed files)",
    )
    parser.add_argument(
        "--repo", default=".", help="Git repo root (default: current dir)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without writing"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help="Copyright end year (default: current year)",
    )
    args = parser.parse_args()

    repo = git_root(os.path.abspath(args.repo))
    if not repo:
        print("Error: not inside a git repository", file=sys.stderr)
        sys.exit(1)

    if args.files:
        files = [os.path.abspath(f) for f in args.files]
    else:
        files = changed_files(repo)

    if not files:
        print("No changed files to process.")
        return

    results: dict[str, list[str]] = {
        "added header": [],
        "updated year": [],
        "already has header": [],
        "skipped (LLVM-derived)": [],
    }
    for fpath in files:
        status = process_file(fpath, args.year, dry_run=args.dry_run)
        if status and status in results:
            results[status].append(os.path.relpath(fpath, repo))

    prefix = "[dry-run] " if args.dry_run else ""
    for status, flist in results.items():
        if flist:
            print(f"\n{prefix}{status} ({len(flist)} files):")
            for f in flist:
                print(f"  {f}")

    total_changed = len(results["added header"]) + len(results["updated year"])
    if total_changed == 0:
        print("\nAll files already have correct headers.")
    else:
        print(f"\n{prefix}Modified {total_changed} file(s).")


if __name__ == "__main__":
    main()
