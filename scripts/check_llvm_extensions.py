#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Every LLVM extension must be switchable at run time.

An extension must add an ``llvm::cl::opt`` or ``getenv`` guard and branch on it,
so a regression can be turned off without rebuilding LLVM. This checks the diff
syntactically; a switch wired to a constant is left to review. REQUIRED_PATCHES
are exempt: switching one off only breaks the build.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "thirdparty" / "llvm-extensions"

_BUILD_LLVM = REPO / "scripts" / "build_llvm.sh"


def required_patches():
    """REQUIRED_PATCHES, as build_llvm.sh itself reports them."""
    out = subprocess.run(
        ["bash", str(_BUILD_LLVM), "--list-patches"], check=True, capture_output=True, text=True
    ).stdout
    return {name for kind, name in (line.split(None, 1) for line in out.splitlines()) if kind == "required"}


_GETENV = re.compile(r"getenv\s*\(")
# `cl::opt<bool> Name(` on one line.
_OPT_SAME_LINE = re.compile(r"cl::opt<[^>]*>\s+([A-Za-z_]\w*)")
# `cl::opt<bool>` alone, with `Name(` on the next line.
_OPT_WRAPPED = re.compile(r"cl::opt<[^>]*>\s*$")
_WRAPPED_NAME = re.compile(r"^\+\s*([A-Za-z_]\w*)\s*\(")
# Lines that are part of a declaration rather than a use of the switch.
_DECL_NOISE = re.compile(r"cl::(opt|desc|init|Hidden|ReallyHidden|cat|value_desc)")

HELP = """\
       Every LLVM extension must be disablable without rebuilding LLVM.
       Add an llvm::cl::opt (default off) or a getenv guard, and branch on it:

         static cl::opt<bool> EnableFlyThing(
             "fly-thing", cl::Hidden, cl::init(false),
             cl::desc("FlyDSL: ..."));
         if (EnableFlyThing) { /* new behavior */ }

       See CONTRIBUTING.md, "Add an LLVM Extension"."""


def _added_lines(text: str) -> list[str]:
    """Lines the extension adds. A switch it did not introduce does not count."""
    return [ln for ln in text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def _switch_names(added: list[str]) -> list[str]:
    names = []
    for i, line in enumerate(added):
        names += _OPT_SAME_LINE.findall(line)
        # Only the line immediately after a bare `cl::opt<...>`: scanning every
        # `Name(` line would pick up the calls the extension adds and count one
        # of them as the switch.
        if _OPT_WRAPPED.search(line) and i + 1 < len(added):
            m = _WRAPPED_NAME.match(added[i + 1])
            if m:
                names.append(m.group(1))
    return names


def has_switch(text: str) -> bool:
    added = _added_lines(text)
    if any(_GETENV.search(ln) for ln in added):
        return True
    uses = [ln for ln in added if not _DECL_NOISE.search(ln)]
    return any(re.search(rf"\b{re.escape(name)}\b", ln) for name in _switch_names(added) for ln in uses)


def main() -> int:
    failed = False
    required = required_patches()
    for ext in sorted(EXT_DIR.glob("*.patch")):
        if ext.name in required:
            print(f"  {ext.name}: skipped (required patch)")
            continue
        if has_switch(ext.read_text(encoding="utf-8")):
            print(f"  {ext.name}: OK")
        else:
            print(f"Error: {ext.name} adds no runtime switch.", file=sys.stderr)
            print(HELP, file=sys.stderr)
            failed = True

    if failed:
        return 1
    print("All checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
