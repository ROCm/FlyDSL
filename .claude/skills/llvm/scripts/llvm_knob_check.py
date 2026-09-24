#!/usr/bin/env python3
"""Check whether an LLVM/AMDGPU knob exists in the LLVM that will compile kernels.

A flag appearing in a shipped kernel is NOT evidence that it exists: this repo
ships `amdgpu-schedule-regions` (absent from the pinned LLVM) and a test that
asserts `lsr-drop-solution=4` (an invalid value for a boolOrDefault option).
Run this before adding anything to an `llvm_options` dict.

The check distinguishes three outcomes, because they fail in different ways:

  cl::opt     - a real command-line option; usable via `llvm_options`.
  attribute   - an IR function attribute, NOT a command-line flag. Passing it as
                one fails outright; it must be set as a kernel attribute instead.
                `amdgpu-waves-per-eu` and `amdgpu-num-vgpr` are both of these.
  missing     - not present in this LLVM at all.

Usage:
    llvm_knob_check.py <flag> [<flag>...] [--llc PATH] [--verbose]

Exit codes: 0 all exist as cl::opt, 1 something is missing or is an attribute,
2 usage error.
"""

import sys

# Parsed before anything runs, so this guard only protects against *runtime* use
# of newer syntax; PEP 585/604 annotations below are never evaluated on an old
# interpreter. The host python is 3.6 -- run this inside the dev container.
if sys.version_info < (3, 10):  # pragma: no cover - exercised via subprocess
    sys.stderr.write(
        ".claude/skills/llvm/scripts/llvm_knob_check.py requires Python 3.10+ (running %s).\n"
        "This repository targets 3.10+ (CONTRIBUTING.md; ruff target-version=py310).\n"
        "The host interpreter is too old; run it in the dev container:\n"
        "  docker exec <dev-container> python3 .claude/skills/llvm/scripts/llvm_knob_check.py ...\n"
        % sys.version.split()[0]
    )
    raise SystemExit(2)

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

# Attributes that read like flags but are parsed via getFnAttribute(). Passing
# any of these on a command line fails with "Unknown command line argument".
# Note: amdgpu-sched-strategy and amdgpu-expert-scheduling-mode are BOTH a
# cl::opt and a function attribute, so they are deliberately absent here --
# the cl::opt branch reports them correctly.
KNOWN_ATTRIBUTES = {
    "amdgpu-waves-per-eu",
    "amdgpu-num-vgpr",
    "amdgpu-num-sgpr",
    "amdgpu-flat-work-group-size",
    "amdgpu-agpr-alloc",
    "amdgpu-max-num-workgroups",
    "amdgpu-dynamic-vgpr-block-size",
    "amdgpu-memory-bound",
    "amdgpu-wave-limiter",
    "amdgpu-cluster-dims",
    "amdgpu-color-export",
    "amdgpu-ieee",
    "amdgpu-dx10-clamp",
    "amdgpu-lds-size",
    "amdgpu-uniform-work-group-size",
}

# Checked after $FLYDSL_COMPILE_LLVM_DIR and before $PATH. Extend with
# $FLYDSL_LLC or --llc rather than editing this list.
CANDIDATE_LLC = tuple(p for p in (os.environ.get("FLYDSL_LLC"),) if p)


def find_llc(explicit: str | None) -> str:
    """Locate an llc. Explicit path wins, then $FLYDSL_COMPILE_LLVM_DIR, then known builds."""
    if explicit:
        if not os.access(explicit, os.X_OK):
            sys.exit(f"error: not executable: {explicit}")
        return explicit

    llvm_dir = os.environ.get("FLYDSL_COMPILE_LLVM_DIR", "").strip()
    if llvm_dir:
        cand = Path(llvm_dir) / "bin" / "llc"
        if os.access(cand, os.X_OK):
            return str(cand)

    for cand in CANDIDATE_LLC:
        if os.access(cand, os.X_OK):
            return cand

    found = shutil.which("llc")
    if found:
        return found

    sys.exit(
        "error: no llc found. Point at one with --llc PATH, $FLYDSL_LLC, or\n"
        "       $FLYDSL_COMPILE_LLVM_DIR. It should be built from the pin in\n"
        "       thirdparty/llvm-build-info.json, or flag existence will not match."
    )


def registered_options(llc: str) -> set[str]:
    """Every option name llc --help-hidden registers.

    A binary that fails to launch (missing GLIBC, wrong arch) writes only to
    stderr. Parsing that as help text yields an empty option set and every flag
    reads as MISSING -- a dead instrument reporting confident negatives. Treat
    anything that does not produce a real listing on stdout as a tool failure.
    """
    try:
        proc = subprocess.run([llc, "--help-hidden"], capture_output=True, text=True, timeout=120)
    except OSError as exc:
        sys.exit(f"error: cannot run {llc}: {exc}")

    # --help-hidden exits non-zero on some builds, so the listing itself is the
    # signal -- but it must come from stdout, not from a loader error on stderr.
    opts = set(re.findall(r"^\s+--?([A-Za-z0-9][\w.-]*)", proc.stdout, re.MULTILINE))
    if not opts:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        first = detail[0] if detail else f"exit code {proc.returncode}, no output"
        sys.exit(
            f"error: {llc} did not list any options -- it is not a usable llc.\n"
            f"       {first}\n"
            "       Nothing was checked. Point at a working llc before drawing\n"
            "       any conclusion about whether a flag exists."
        )
    return opts


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that LLVM knobs exist before you rely on them.")
    ap.add_argument("flags", nargs="+", help="flag names, with or without leading dashes")
    ap.add_argument("--llc", help="path to llc (default: auto-detect)")
    ap.add_argument("--verbose", action="store_true", help="also print the llc used")
    args = ap.parse_args()

    llc = find_llc(args.llc)
    opts = registered_options(llc)

    if args.verbose:
        ver = subprocess.run([llc, "--version"], capture_output=True, text=True)
        first = next((ln.strip() for ln in ver.stdout.splitlines() if "version" in ln.lower()), "?")
        print(f"# llc: {llc}\n# {first}\n# {len(opts)} options registered\n")

    worst = 0
    for raw in args.flags:
        name = raw.lstrip("-").split("=", 1)[0]
        if name in opts:
            verdict, note = "cl::opt", "usable via llvm_options"
        elif name in KNOWN_ATTRIBUTES:
            verdict, note = "attribute", "NOT a flag - set it as a kernel attribute"
            worst = max(worst, 1)
        else:
            verdict, note = "MISSING", "not in this LLVM - do not use"
            worst = max(worst, 1)
        print(f"{name:<45} {verdict:<10} {note}")

    return worst


if __name__ == "__main__":
    sys.exit(main())
