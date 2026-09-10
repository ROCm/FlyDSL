#!/usr/bin/env python3
"""Run the supplemental FlyDSL review candidate checks once."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_scanner(label, command):
    print(f"== {label} ==")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        print(f"error: could not run {command[1]}: {error}", file=sys.stderr)
        return 2

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode not in {0, 1, 2}:
        print(f"error: {command[1]} exited with unsupported status {result.returncode}", file=sys.stderr)
        return 2
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", required=True, type=Path, metavar="FILE")
    parser.add_argument("--head", required=True, type=Path, metavar="DIR")
    args = parser.parse_args(argv)

    skill_dir = Path(__file__).resolve().parent
    statuses = [
        run_scanner(
            "legacy API spelling candidates",
            [
                sys.executable,
                str(skill_dir / "scan_legacy_spelling.py"),
                "--diff",
                str(args.diff),
            ],
        ),
        run_scanner(
            "added test entry-point candidates",
            [
                sys.executable,
                str(skill_dir / "scan_unreachable_tests.py"),
                "--diff",
                str(args.diff),
                "--head",
                str(args.head),
            ],
        ),
    ]
    if 2 in statuses:
        return 2
    return 1 if 1 in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
