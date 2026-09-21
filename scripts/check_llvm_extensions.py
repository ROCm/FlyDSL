#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Every LLVM extension must be switchable at run time.

An extension changes a compiler FlyDSL does not own. When one turns out to
regress a kernel, finding that out costs a full LLVM rebuild unless the
behavior can be turned off in place. So each extension must introduce a switch
-- an ``llvm::cl::opt`` or a ``getenv`` guard -- and must branch on it, rather
than changing behavior unconditionally.

This checks syntax, not semantics: that the diff adds a switch and references
it outside its own declaration. It catches the common failure (nobody thought
about a switch); it cannot catch a switch wired to a constant. That is what
review is for.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "thirdparty" / "llvm-extensions"

_BUILD_LLVM = REPO / "scripts" / "build_llvm.sh"
_REQUIRED_ARRAY = re.compile(r"^REQUIRED_PATCHES=\(\n(.*?)^\)", re.M | re.S)


def required_patches():
    """Patches both profiles carry, read from build_llvm.sh.

    These make LLVM usable for FlyDSL rather than faster, so the run-time switch
    requirement does not apply -- a switch that turns off a required patch would
    just produce a broken build. Reading the array keeps this from becoming a
    second list that can disagree with the one the build actually uses.
    """
    m = _REQUIRED_ARRAY.search(_BUILD_LLVM.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"Could not find REQUIRED_PATCHES in {_BUILD_LLVM}")
    return {line.strip() for line in m.group(1).splitlines() if line.strip() and not line.strip().startswith("#")}


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
    ap = argparse.ArgumentParser(description=__doc__)
    # Accepted and ignored: the whole set is scanned, because an extension whose
    # switch was removed by an unrelated commit is out of any revision range.
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.parse_args()

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
