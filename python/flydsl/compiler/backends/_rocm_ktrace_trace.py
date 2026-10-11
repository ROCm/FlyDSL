# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from ...expr.experimental import ktrace as _ktrace
from ...expr.rocdl import ktrace_format as _format

NS_PER_TICK = 10.0


def _json_payload(value):
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def _wave_key(rec: dict) -> tuple[int, ...]:
    return (
        rec.get("queue_id", 0),
        rec.get("dispatch_id", 0),
        rec["xcc_id"],
        rec["se"],
        rec["sh"],
        rec["cu"],
        rec["simd"],
        rec["wave"],
    )


def _wave_label(rec: dict) -> str:
    return f"XCD{rec['xcc_id']}/SE{rec['se']}/SH{rec['sh']}/CU{rec['cu']}/SIMD{rec['simd']}/wave{rec['wave']}"


def group_launches(records: list[dict]) -> list[tuple[tuple[int, int], list[dict]]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec.get("queue_id", 0), rec.get("dispatch_id", 0))].append(rec)
    return sorted(grouped.items(), key=lambda item: min(rec["ts"] for rec in item[1]))


def build_chrome_trace(records: list[dict], names: dict[str, int], *, kernel: str = "kernel") -> dict:
    by_id = {v: k for k, v in names.items()}
    if not records:
        return {"traceEvents": [], "displayTimeUnit": "ns"}

    t0 = min(r["ts"] for r in records)
    launches = group_launches(records)
    launch_pid = {identity: ordinal for ordinal, (identity, _) in enumerate(launches, 1)}
    waves: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for rec in records:
        waves[_wave_key(rec)].append(rec)

    events: list[dict] = [
        {
            "ph": "M",
            "pid": ordinal,
            "tid": 0,
            "name": "process_name",
            "args": {"name": f"{kernel} launch {ordinal} (queue {identity[0]:x}, dispatch {identity[1]})"},
        }
        for ordinal, (identity, _) in enumerate(launches, 1)
    ]
    for tid, (key, group) in enumerate(sorted(waves.items())):
        group.sort(key=lambda r: r["ts"])
        events.append(
            {
                "ph": "M",
                "pid": launch_pid[key[:2]],
                "tid": tid,
                "name": "thread_name",
                "args": {"name": _wave_label(group[0])},
            }
        )
        events.extend(_wave_events(group, by_id, tid, launch_pid[key[:2]], t0))

    return {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "otherData": {
            "kernel": kernel,
            "launches": [{"queue_id": identity[0], "dispatch_id": identity[1]} for identity, _ in launches],
        },
    }


def _wave_events(group: list[dict], by_id: dict[int, str], tid: int, pid: int, t0: int) -> list[dict]:
    out = []
    open_names: dict[int, str] = {}
    for rec in group:
        us = (rec["ts"] - t0) * NS_PER_TICK / 1000.0
        name = by_id.get(rec["event_id"], f"event{rec['event_id']}")
        base = {"pid": pid, "tid": tid, "ts": us}
        args = {"payload": _json_payload(rec["payload"])} if rec["payload"] is not None else {}
        kind = rec["kind"]

        if kind == _ktrace.KIND_MARK:
            out.append({**base, "ph": "i", "s": "t", "name": name, "args": args})
        elif kind == _ktrace.KIND_RANGE_PUSH:
            out.append({**base, "ph": "B", "name": name, "args": args})
        elif kind == _ktrace.KIND_RANGE_POP:
            out.append({**base, "ph": "E"})
        elif kind == _ktrace.KIND_RANGE_START:
            open_names[rec["slot"]] = name
            out.append({**base, "ph": "b", "id": rec["slot"], "cat": "range", "name": name, "args": args})
        elif kind == _ktrace.KIND_RANGE_END:
            if rec["start_slot"] != _format.UNPAIRED_SLOT:
                end_name = open_names.pop(rec["start_slot"], name)
                out.append({**base, "ph": "e", "id": rec["start_slot"], "cat": "range", "name": end_name, "args": args})
    return out


def summarize(records: list[dict], names: dict[str, int]) -> dict:
    by_id = {v: k for k, v in names.items()}
    waves: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for rec in records:
        waves[_wave_key(rec)].append(rec)

    totals: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_ns": 0.0})
    for group in waves.values():
        group.sort(key=lambda r: r["ts"])
        stack: list[dict] = []
        open_tokens: dict[int, dict] = {}
        for rec in group:
            kind = rec["kind"]
            if kind == _ktrace.KIND_RANGE_PUSH:
                stack.append(rec)
            elif kind == _ktrace.KIND_RANGE_POP and stack:
                start = stack.pop()
                _accumulate(totals, by_id.get(start["event_id"]), rec["ts"] - start["ts"])
            elif kind == _ktrace.KIND_RANGE_START:
                open_tokens[rec["slot"]] = rec
            elif kind == _ktrace.KIND_RANGE_END:
                if rec["start_slot"] == _format.UNPAIRED_SLOT:
                    continue
                start = open_tokens.pop(rec["start_slot"], None)
                if start is not None:
                    _accumulate(totals, by_id.get(start["event_id"]), rec["ts"] - start["ts"])
            elif kind == _ktrace.KIND_MARK:
                _accumulate(totals, by_id.get(rec["event_id"]), 0)

    return {
        "waves": len(waves),
        "records": len(records),
        "phases": {k: {"count": v["count"], "total_ns": v["total_ns"]} for k, v in sorted(totals.items())},
    }


def _accumulate(totals, name, ticks) -> None:
    if name is None:
        return
    entry = totals[name]
    entry["count"] += 1
    entry["total_ns"] += ticks * NS_PER_TICK


def write_trace(records: list[dict], names: dict[str, int], *, kernel: str = "kernel", out_dir=None) -> str:
    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    path = _unique(os.path.join(out_dir, f"{kernel}_ktrace.trace.json"))
    with open(path, "w") as fh:
        json.dump(build_chrome_trace(records, names, kernel=kernel), fh, allow_nan=False)

    raw = _unique(os.path.join(out_dir, f"{kernel}_ktrace.json"))
    with open(raw, "w") as fh:
        json.dump(
            {
                "kernel": kernel,
                "ns_per_tick": NS_PER_TICK,
                "names": names,
                "records": [{**record, "payload": _json_payload(record["payload"])} for record in records],
            },
            fh,
            indent=2,
            allow_nan=False,
        )
    return path


def _unique(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}.{n}{ext}"):
        n += 1
    return f"{stem}.{n}{ext}"
