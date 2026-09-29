# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Analyze rank*.json soft counters without importing Torch or requiring a GPU.

The counters describe the last instrumented graph layer. They are diagnostics,
not replacement kernel latency measurements. In particular, wave durations
overlap, GPU clocks are not aligned across ranks, and mid polling is nested in
the down stage. Run this file directly on the downloaded output directory.
"""

import argparse
import json
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "min": values[0],
        "p50": statistics.median(values),
        "p95": values[int(0.95 * (len(values) - 1))],
        "max": values[-1],
    }


def analyze(path):
    data = json.loads(path.read_text())
    fields = data["fields"]
    ticks_per_us = data["ticks_per_us"]
    if data["counter"] != "s_memrealtime" or ticks_per_us <= 0 or data["mode"] not in (1, 2):
        raise ValueError(f"{path}: unsupported clock")
    records = []
    expected = []
    for cta, waves in enumerate(data["durations"]):
        for wave, values in enumerate(waves):
            selected = wave == 0 if data["mode"] == 1 else cta % 31 == 0
            if selected:
                expected.append((cta, wave))
            record = dict(zip(fields, values))
            if record["epoch_tag"]:
                if not selected or len(values) != len(fields):
                    raise ValueError(f"{path}: unexpected record {cta}/{wave}")
                records.append({"cta": cta, "wave": wave, **record})
    if len(records) != len(expected):
        raise ValueError(f"{path}: missing sampled CTA/wave records")
    tags = {record["epoch_tag"] for record in records}
    if len(tags) != 1:
        raise ValueError(f"{path}: mixed graph epochs {tags}")
    for record in records:
        if any(record[key] < 0 for key in fields if key.endswith("_ticks")):
            raise ValueError(f"{path}: negative duration")
        if record["mid_poll_ticks"] > record["down_ticks"]:
            raise ValueError(f"{path}: polling exceeds enclosing down duration")

    producers = [record for record in records if record["producer"]]
    consumers = [record for record in records if record["consumer"]]
    origin = min(record["entry_tick"] for record in records)
    last_up = max(record["ug_end_tick"] for record in producers)

    def durations(items, key):
        return distribution([record[key] / ticks_per_us for record in items])

    def offset(tick):
        return (tick - origin) / ticks_per_us

    stages = {"ug": durations(producers, "ug_ticks")}
    for key in ("down", "pack", "tp_push", "tp_wait_reduce"):
        # Only wave 0 performs final tile packing and TP reduction in S4/D16.
        items = consumers if key == "down" else [r for r in consumers if r["wave"] == 0]
        stages[key] = durations(items, f"{key}_ticks")
    if data["mode"] == 2:
        stages["mid_poll_nested_in_down"] = durations(consumers, "mid_poll_ticks")
        stages["down_excluding_poll"] = distribution(
            [(r["down_ticks"] - r["mid_poll_ticks"]) / ticks_per_us for r in consumers]
        )

    critical = max(consumers, key=lambda r: r["entry_tick"] + r["total_ticks"])
    wave_spread = []
    for cta in sorted({r["cta"] for r in consumers}):
        times = [r["down_end_tick"] for r in consumers if r["cta"] == cta]
        if len(times) > 1:
            wave_spread.append((max(times) - min(times)) / ticks_per_us)
    return {
        "file": str(path),
        "rank": data["rank"],
        "mode": data["mode"],
        "scope": data["scope"],
        "epoch_tag": next(iter(tags)),
        "pipeline_config": data["pipeline_config"],
        "sampled_waves": len(records),
        "producer_waves": len(producers),
        "consumer_waves": len(consumers),
        "sampled_span_us": offset(max(r["entry_tick"] + r["total_ticks"] for r in records)),
        "entry_spread_us": offset(max(r["entry_tick"] for r in records)),
        "ug_last_submit_offset_us": offset(last_up),
        "down_begin_offset_us": distribution([offset(r["down_begin_tick"]) for r in consumers]),
        "down_end_offset_us": distribution([offset(r["down_end_tick"]) for r in consumers]),
        "consumer_waves_starting_before_last_ug_submit": sum(r["down_begin_tick"] < last_up for r in consumers),
        "stages_us": stages,
        "poll_fraction_of_down": (
            distribution([r["mid_poll_ticks"] / r["down_ticks"] for r in consumers]) if data["mode"] == 2 else None
        ),
        "clock_pair_us": durations(records, "clock_pair_ticks"),
        "down_end_wave_spread_us": distribution(wave_spread),
        "critical_sampled_consumer": {
            "cta": critical["cta"],
            "wave": critical["wave"],
            "entry_us": offset(critical["entry_tick"]),
            "down_begin_us": offset(critical["down_begin_tick"]),
            "down_end_us": offset(critical["down_end_tick"]),
            "tp_end_us": offset(critical["tp_end_tick"]),
            **{
                key.removesuffix("_ticks") + "_us": critical[key] / ticks_per_us
                for key in fields
                if key.endswith("_ticks")
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("rank*.json"))
    if not paths:
        parser.error("no rank*.json files found")
    ranks = [analyze(path) for path in paths]
    for key in ("mode", "scope", "epoch_tag", "pipeline_config"):
        if any(rank[key] != ranks[0][key] for rank in ranks):
            raise ValueError(f"ranks have inconsistent {key}")
    print("Diagnostic microseconds; per-rank clocks, overlapping stages, last graph layer only.")
    print("rank   span   UG p50   down p50   poll p50   push p50   TP wait p50")
    for rank in ranks:
        stages = rank["stages_us"]
        poll = stages.get("mid_poll_nested_in_down", {"p50": float("nan")})["p50"]
        print(
            f"{rank['rank']:4d} {rank['sampled_span_us']:6.2f}"
            f" {stages['ug']['p50']:8.2f} {stages['down']['p50']:10.2f} {poll:10.2f}"
            f" {stages['tp_push']['p50']:10.2f} {stages['tp_wait_reduce']['p50']:13.2f}"
        )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "units": "microseconds for durations and offsets; ratios for fractions",
                    "limitations": [
                        "Single final graph-layer snapshot per rank; not a repeat distribution.",
                        "Sampled spans omit unsampled waves and kernel prologue/epilogue.",
                        "Do not sum overlapping wave or stage durations into kernel latency.",
                        "Clock-pair time excludes bookkeeping/stores and is not total observer overhead.",
                        "Down minus polling still includes weight/LDS stalls, unpack, MFMA and barriers.",
                        "UG and TP submission timestamps do not confirm global/peer store completion.",
                        "Rank clocks are not aligned; do not compare raw timestamps across GPUs.",
                    ],
                    "ranks": ranks,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
