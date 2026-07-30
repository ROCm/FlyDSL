#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Extract normalized benchmark metrics from FlyDSL benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedMetrics:
    tbps: float | None = None
    tflops: float | None = None
    avg_us: float | None = None
    statistic: str = "reported"
    warmup: str = "-"
    iters: str = "-"
    instrument: str = "reported"
    schedule: str = "unknown"
    cache_policy: str = "unknown"


def _last_match(pattern: str, text: str):
    matches = list(re.finditer(pattern, text))
    return matches[-1] if matches else None


def parse_metrics(text: str) -> ParsedMetrics:
    tbps = tflops = avg_us = None
    statistic = "reported"

    match = _last_match(
        r"Throughput:\s*([0-9.]+)\s*us.*?([0-9.]+)\s*TFLOPS.*?BW:\s*([0-9.]+)\s*TB/s",
        text,
    )
    if match:
        avg_us, tflops, tbps = map(float, match.groups())
        statistic = "mean"

    if tbps is None or tflops is None:
        match = _last_match(
            r"FlyDSL MoE .*?:\s*([0-9.]+)\s*us,\s*([0-9.]+)\s*TFLOPS.*?([0-9.]+)\s*TB/s",
            text,
        )
        if match:
            avg_us, tflops, tbps = map(float, match.groups())

    if tflops is None:
        match = _last_match(
            r"\|\s+(?:PASS|FAIL|--)\s+\|\s+[0-9.eE+-]+\s+[0-9.]+\s+\|\s+([0-9.]+)\s+([0-9.]+)",
            text,
        )
        if match:
            avg_us, tflops = map(float, match.groups())

    if tbps is None or tflops is None:
        match = _last_match(r"TFLOPS=([0-9.]+)\s+TB/s=([0-9.]+)", text)
        if match:
            tflops, tbps = map(float, match.groups())
            timing = _last_match(r"us_p50=([0-9.]+)", text)
            if timing:
                avg_us = float(timing.group(1))
                statistic = "median"

    if tbps is None:
        bandwidth = next(re.finditer(r"Bandwidth:\s*([0-9.]+)\s*GB/s", text), None)
        if bandwidth:
            tbps = float(bandwidth.group(1)) / 1000.0
            timing = next(re.finditer(r"Kernel avg time:\s*([0-9.]+)\s*ms", text), None)
            if timing:
                avg_us = float(timing.group(1)) * 1000.0
                statistic = "mean"

    contract = _last_match(
        r"Benchmark contract:\s+instrument=(\S+)\s+schedule=(\S+)\s+"
        r"cache=(\S+)\s+statistic=(\S+)\s+warmup=(\d+)\s+iters=(\d+)",
        text,
    )
    if contract:
        instrument, schedule, cache_policy, statistic, warmup, iters = contract.groups()
    else:
        warmup, iters, instrument, schedule, cache_policy = ("-", "-", "reported", "unknown", "unknown")

    return ParsedMetrics(
        tbps=tbps,
        tflops=tflops,
        avg_us=avg_us,
        statistic=statistic,
        warmup=warmup,
        iters=iters,
        instrument=instrument,
        schedule=schedule,
        cache_policy=cache_policy,
    )


def parse_moe_stage2(text: str):
    pattern = re.compile(
        r"FlyDSL MoE stage2 \[[^]]+\]\s+(\S+)\s+(atomic|reduce)\b.*?"
        r"([0-9.]+)\s*us,\s*([0-9.]+)\s*TFLOPS.*?([0-9.]+)\s*TB/s"
    )
    found = {}
    for match in pattern.finditer(text):
        dtype, mode = match.group(1), match.group(2)
        found[mode] = (
            dtype,
            ParsedMetrics(
                avg_us=float(match.group(3)),
                tflops=float(match.group(4)),
                tbps=float(match.group(5)),
                statistic="reported",
            ),
        )
    return found


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def measurement_contract(op: str):
    if op in {"softmax", "layernorm", "rmsnorm", "rmsnorm_mixed_weight"}:
        return ("10", "100", "torch_profiler", "per_iter_sync", "warm")
    if op in {"rmsnorm_mixed_w_bwd", "rmsnorm_add_mixed_bwd"}:
        return ("10", "100", "device_event", "pipelined", "warm")
    return ("-", "-", "reported", "unknown", "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    row = subparsers.add_parser("row")
    row.add_argument("op")
    row.add_argument("shape")
    row.add_argument("dtype")
    row.add_argument("log", type=Path)

    moe = subparsers.add_parser("moe-stage2")
    moe.add_argument("op_prefix")
    moe.add_argument("shape")
    moe.add_argument("log", type=Path)

    args = parser.parse_args()
    text = args.log.read_text(errors="ignore") if args.log.exists() else ""

    if args.command == "row":
        metrics = parse_metrics(text)
        if metrics.instrument == "reported":
            warmup, iters, instrument, schedule, cache_policy = measurement_contract(args.op)
        else:
            warmup = metrics.warmup
            iters = metrics.iters
            instrument = metrics.instrument
            schedule = metrics.schedule
            cache_policy = metrics.cache_policy
        print(
            f"{args.op}\t{args.shape}\t{args.dtype}\t{_fmt(metrics.tbps)}\t"
            f"{_fmt(metrics.tflops)}\t{_fmt(metrics.avg_us)}\t{metrics.statistic}\t"
            f"{warmup}\t{iters}\t{instrument}\t{schedule}\t{cache_policy}"
        )
        return 0

    found = parse_moe_stage2(text)
    emitted = False
    for mode in ("atomic", "reduce"):
        if mode not in found:
            continue
        dtype, metrics = found[mode]
        print(
            f"{args.op_prefix}_{mode}\t{args.shape}\t{dtype}\t{_fmt(metrics.tbps)}\t"
            f"{_fmt(metrics.tflops)}\t{_fmt(metrics.avg_us)}\t{metrics.statistic}\t"
            "-\t-\treported\tunknown\tunknown"
        )
        emitted = True
    if not emitted:
        print(f"{args.op_prefix}_atomic\t{args.shape}\t-\t-\t-\t-\treported\t" "-\t-\treported\tunknown\tunknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
