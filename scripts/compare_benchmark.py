#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Report performance ratios between two run_benchmark.sh CSV outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRow:
    op: str
    shape: str
    dtype: str
    metric_name: str
    metric_value: float | None
    status: str


def _parse_float(value: str) -> float | None:
    if value in {"", "-", "skip"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv(path: Path) -> dict[tuple[str, str, str], BenchmarkRow]:
    rows: dict[tuple[str, str, str], BenchmarkRow] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"op", "shape", "dtype", "tbps", "tflops", "status"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")

        for raw in reader:
            op = raw["op"]
            shape = raw["shape"]
            dtype = raw["dtype"]
            tflops = _parse_float(raw["tflops"])
            tbps = _parse_float(raw["tbps"])
            if tflops is not None:
                metric_name = "TFLOPS"
                metric_value = tflops
            else:
                metric_name = "TB/s"
                metric_value = tbps
            rows[(op, shape, dtype)] = BenchmarkRow(
                op=op,
                shape=shape,
                dtype=dtype,
                metric_name=metric_name,
                metric_value=metric_value,
                status=raw["status"],
            )
    return rows


def _format_key(key: tuple[str, str, str]) -> str:
    op, shape, dtype = key
    return f"{op:>18s} {shape:>34s} {dtype:>8s}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("current_csv", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="current")
    args = parser.parse_args()

    baseline = _read_csv(args.baseline_csv)
    current = _read_csv(args.current_csv)

    print(f"=== Benchmark: {args.current_label} vs {args.baseline_label} ===")

    compared = 0

    for key in sorted(current.keys() & baseline.keys()):
        base = baseline[key]
        curr = current[key]
        if base.metric_value is None:
            continue

        if curr.metric_value is None:
            print(f"  {_format_key(key)}  {args.current_label}=missing  [SKIP]")
            continue
        if curr.metric_name != base.metric_name:
            print(
                f"  {_format_key(key)}  metric mismatch: "
                f"{args.baseline_label}={base.metric_name}, "
                f"{args.current_label}={curr.metric_name}  [SKIP]"
            )
            continue

        compared += 1
        delta = curr.metric_value - base.metric_value
        delta_pct = (delta / base.metric_value) * 100.0 if base.metric_value else 0.0
        ratio = curr.metric_value / base.metric_value if base.metric_value else 0.0

        print(
            f"  {_format_key(key)}  "
            f"{args.baseline_label}={base.metric_value:9.3f} {base.metric_name:<6s}  "
            f"{args.current_label}={curr.metric_value:9.3f} {curr.metric_name:<6s}  "
            f"ratio={ratio:6.3f}x  delta={delta:+9.3f} ({delta_pct:+6.1f}%)"
        )

    skipped_new = len(set(current) - set(baseline))
    if skipped_new:
        print(f"\nSkipped {skipped_new} new current-only benchmark row(s).")

    skipped_missing = 0
    for key in sorted(set(baseline) - set(current)):
        base = baseline[key]
        if base.metric_value is None:
            continue
        skipped_missing += 1
        print(f"  {_format_key(key)}  {args.current_label}=missing row  [SKIP]")
    if skipped_missing:
        print(f"\nSkipped {skipped_missing} baseline-only benchmark row(s).")

    if compared == 0:
        print("No comparable benchmark rows found.")

    print("\nBenchmark comparison report completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
