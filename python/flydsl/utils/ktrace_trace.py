# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Turn ktrace records into a Chrome Trace JSON timeline.

Perfetto reads Chrome Trace JSON natively, so no protobuf dependency is needed and the
output also opens in any tool that understands the format.

Two things the reconstruction has to get right:

* Records arrive in *cursor* order, which is global atomic order, not per-wave order.
  Group by ``(xcc_id, hw_id)`` -- that tuple is the wave identity -- and sort each group
  by timestamp before interpreting anything.
* ``event_id`` names a *phase*, so N loop iterations of one range all share it.  Token
  ranges therefore pair by ``start_slot``, the buffer slot of their own START record,
  which is exact and order-independent.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from ..expr import ktrace as _ktrace

# s_memrealtime counts a constant-rate 100 MHz reference clock.
NS_PER_TICK = 10.0


def _wave_key(rec: dict) -> tuple[int, int]:
    return (rec["xcc_id"], rec["hw_id"])


def _wave_label(rec: dict) -> str:
    return f"XCD{rec['xcc_id']}/SE{rec['se']}/SH{rec['sh']}/CU{rec['cu']}" f"/SIMD{rec['simd']}/wave{rec['wave']}"


def build_chrome_trace(records: list[dict], names: dict[str, int], *, kernel: str = "kernel") -> dict:
    """Build a Chrome Trace document from decoded records."""
    by_id = {v: k for k, v in names.items()}
    if not records:
        return {"traceEvents": [], "displayTimeUnit": "ns"}

    t0 = min(r["ts"] for r in records)
    waves: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rec in records:
        waves[_wave_key(rec)].append(rec)

    events: list[dict] = []
    for tid, (key, group) in enumerate(sorted(waves.items())):
        group.sort(key=lambda r: r["ts"])
        events.append(
            {
                "ph": "M",
                "pid": key[0],
                "tid": tid,
                "name": "thread_name",
                "args": {"name": _wave_label(group[0])},
            }
        )
        events.append({"ph": "M", "pid": key[0], "tid": 0, "name": "process_name", "args": {"name": f"XCD{key[0]}"}})
        events.extend(_wave_events(group, by_id, tid, key[0], t0))

    return {"traceEvents": events, "displayTimeUnit": "ns", "otherData": {"kernel": kernel}}


def _wave_events(group: list[dict], by_id: dict[int, str], tid: int, pid: int, t0: int) -> list[dict]:
    out = []
    for rec in group:
        us = (rec["ts"] - t0) * NS_PER_TICK / 1000.0
        name = by_id.get(rec["event_id"], f"event{rec['event_id']}")
        base = {"pid": pid, "tid": tid, "ts": us}
        # `is not None`, not truthiness: decode_records always produces an int, and a
        # payload of 0 is the FIRST loop index -- the one the annotated GEMM records for
        # k_tile 0. A falsy test dropped it, so iteration 0 rendered with no payload while
        # every later iteration had one.
        args = {"payload": rec["payload"]} if rec["payload"] is not None else {}
        kind = rec["kind"]

        if kind == _ktrace.KIND_MARK:
            out.append({**base, "ph": "i", "s": "t", "name": name, "args": args})
        elif kind == _ktrace.KIND_RANGE_PUSH:
            out.append({**base, "ph": "B", "name": name, "args": args})
        elif kind == _ktrace.KIND_RANGE_POP:
            # Chrome Trace pairs B/E by nesting, so the name is not repeated here.
            out.append({**base, "ph": "E"})
        elif kind == _ktrace.KIND_RANGE_START:
            out.append({**base, "ph": "b", "id": rec["slot"], "cat": "range", "name": name, "args": args})
        elif kind == _ktrace.KIND_RANGE_END:
            # Pair by start_slot, not by name: N iterations of one range share a name.
            # A range whose START was suppressed has no counterpart to close; emitting it
            # would leave a dangling "e" that Perfetto renders as a zero-length artifact.
            if rec["start_slot"] != _ktrace.UNPAIRED_SLOT:
                out.append({**base, "ph": "e", "id": rec["start_slot"], "cat": "range", "name": name, "args": args})
    return out


def summarize(records: list[dict], names: dict[str, int]) -> dict:
    """Per-phase totals, for a quick answer without opening a viewer."""
    by_id = {v: k for k, v in names.items()}
    waves: dict[tuple[int, int], list[dict]] = defaultdict(list)
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
                if rec["start_slot"] == _ktrace.UNPAIRED_SLOT:
                    continue  # its START record was never written; nothing to pair with
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
    """Write ``<kernel>_ktrace.trace.json`` plus the raw records; returns the trace path.

    Filenames are not overwritten: an existing file gets a numeric suffix, since a trace
    that took a profiling run to collect should not be lost to a re-run.
    """
    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    path = _unique(os.path.join(out_dir, f"{kernel}_ktrace.trace.json"))
    with open(path, "w") as fh:
        json.dump(build_chrome_trace(records, names, kernel=kernel), fh)

    raw = _unique(os.path.join(out_dir, f"{kernel}_ktrace.json"))
    with open(raw, "w") as fh:
        json.dump(
            {"kernel": kernel, "ns_per_tick": NS_PER_TICK, "names": names, "records": records},
            fh,
            indent=2,
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
