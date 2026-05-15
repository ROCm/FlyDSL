#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Convert run_benchmark.sh table output to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _is_metric(value: str) -> bool:
    if value in {"-", "skip"}:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def _status(tbps: str, tflops: str) -> str:
    if tbps == "skip" or tflops == "skip":
        return "skip"
    if tbps == "-" and tflops == "-":
        return "missing"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows: list[list[str]] = []
    for line in args.input.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        op, shape, dtype, tbps, tflops = parts
        if op == "op" or set(op) == {"-"}:
            continue
        if not (_is_metric(tbps) and _is_metric(tflops)):
            continue
        rows.append([op, shape, dtype, tbps, tflops, _status(tbps, tflops)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "shape", "dtype", "tbps", "tflops", "status"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
