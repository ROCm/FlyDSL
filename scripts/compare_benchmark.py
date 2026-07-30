#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Report benchmark deltas and enforce calibrated raw-us regression gates."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_compare import ThresholdConfig, compare_values


@dataclass(frozen=True)
class BenchmarkRow:
    op: str
    shape: str
    dtype: str
    metric_name: str
    metric_value: float | None
    avg_us: float | None
    statistic: str
    instrument: str
    schedule: str
    cache_policy: str
    warmup: str
    iters: str
    arch: str
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
                avg_us=_parse_float(raw.get("avg_us", "")),
                statistic=raw.get("statistic", "") or "",
                instrument=raw.get("instrument", "") or "",
                schedule=raw.get("schedule", "") or "",
                cache_policy=raw.get("cache_policy", "") or "",
                warmup=raw.get("warmup", "") or "",
                iters=raw.get("iters", "") or "",
                arch=raw.get("arch", "") or "",
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
    parser.add_argument("--arch", default="")
    parser.add_argument("--threshold-config", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    baseline = _read_csv(args.baseline_csv)
    current = _read_csv(args.current_csv)
    thresholds = ThresholdConfig.from_path(args.threshold_config) if args.threshold_config else None
    baseline_arches = {row.arch for row in baseline.values() if row.arch}
    current_arches = {row.arch for row in current.values() if row.arch}
    current_arch = args.arch or (next(iter(current_arches)) if len(current_arches) == 1 else "")
    if thresholds and args.fail_on_regression and args.arch and baseline_arches != {args.arch}:
        print(
            f"Cannot enforce benchmark gate: baseline CSV architectures {sorted(baseline_arches)!r} "
            f"do not match trusted runner architecture {args.arch!r}.",
            file=sys.stderr,
        )
        return 2
    if thresholds and args.fail_on_regression and args.arch and current_arches != {args.arch}:
        print(
            f"Cannot enforce benchmark gate: CSV architectures {sorted(current_arches)!r} "
            f"do not match trusted runner architecture {args.arch!r}.",
            file=sys.stderr,
        )
        return 2
    if thresholds and args.fail_on_regression and not thresholds.supports_arch(current_arch):
        print(
            f"Cannot enforce benchmark gate: architecture {current_arch or '<missing>'!r} "
            "is not declared in the threshold config.",
            file=sys.stderr,
        )
        return 2

    print(f"=== Benchmark: {args.current_label} vs {args.baseline_label} ===")

    compared = 0
    failures = 0

    for key in sorted(current.keys() & baseline.keys()):
        base = baseline[key]
        curr = current[key]
        arch = current_arch or curr.arch or base.arch
        arch_mismatch = bool(base.arch and curr.arch and base.arch != curr.arch)
        threshold = None
        gate_line = None
        if arch_mismatch:
            gate_line = f"[GATE SKIP: arch mismatch {base.arch} vs {curr.arch}]"
        elif thresholds:
            threshold = thresholds.match(arch=arch, op=curr.op, shape=curr.shape, dtype=curr.dtype)

        if threshold is not None:
            if base.avg_us is None:
                gate_line = "[GATE SKIP: baseline has no raw us]"
            elif curr.avg_us is None:
                failures += 1
                gate_line = "[BROKEN: current raw us unavailable]"
            elif (
                base.statistic,
                base.instrument,
                base.schedule,
                base.cache_policy,
                base.warmup,
                base.iters,
            ) != (
                curr.statistic,
                curr.instrument,
                curr.schedule,
                curr.cache_policy,
                curr.warmup,
                curr.iters,
            ):
                failures += 1
                gate_line = "[BROKEN: measurement contract mismatch]"
            else:
                regression = compare_values(base.avg_us, curr.avg_us, threshold)
                tag = "REGRESSION" if regression.regressed else "OK"
                if regression.regressed:
                    failures += 1
                gate_line = (
                    f"latency {base.avg_us:.2f} -> {curr.avg_us:.2f} us  "
                    f"delta={regression.delta:+.2f} us ({regression.delta_pct:+.1f}%)  [{tag}]"
                )

        if base.metric_value is None:
            if gate_line:
                print(f"  {_format_key(key)}  {gate_line}")
            continue
        if curr.metric_value is None:
            print(f"  {_format_key(key)}  {args.current_label}=missing throughput  [SKIP]")
        elif curr.metric_name != base.metric_name:
            print(
                f"  {_format_key(key)}  metric mismatch: "
                f"{args.baseline_label}={base.metric_name}, "
                f"{args.current_label}={curr.metric_name}  [SKIP]"
            )
        else:
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
        if gate_line:
            print(" " * 57 + gate_line)

    skipped_new = len(set(current) - set(baseline))
    if skipped_new:
        print(f"\nSkipped {skipped_new} new current-only benchmark row(s).")

    skipped_missing = 0
    for key in sorted(set(baseline) - set(current)):
        base = baseline[key]
        if current_arch and base.arch and current_arch != base.arch:
            skipped_missing += 1
            print(
                f"  {_format_key(key)}  {args.current_label}=missing row  "
                f"[GATE SKIP: arch mismatch {base.arch} vs {current_arch}]"
            )
            continue
        threshold = (
            thresholds.match(
                arch=current_arch or base.arch,
                op=base.op,
                shape=base.shape,
                dtype=base.dtype,
            )
            if thresholds
            else None
        )
        if threshold is not None and base.avg_us is not None:
            failures += 1
            print(f"  {_format_key(key)}  {args.current_label}=missing row  [BROKEN]")
        else:
            skipped_missing += 1
            print(f"  {_format_key(key)}  {args.current_label}=missing row  [SKIP]")
    if skipped_missing:
        print(f"\nSkipped {skipped_missing} baseline-only benchmark row(s).")

    if compared == 0:
        print("No comparable benchmark rows found.")

    if failures:
        print(f"\nBenchmark comparison found {failures} gated regression(s).")
        return 1 if args.fail_on_regression else 0
    print("\nBenchmark comparison report completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
