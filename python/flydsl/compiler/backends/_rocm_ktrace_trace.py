# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import gzip
import math
import os
import struct
from collections import defaultdict

from ...expr import ktrace as _ktrace
from ...expr.rocdl import ktrace_format as _format

NS_PER_TICK = 10.0


def _wave_key(rec: dict) -> tuple:
    return (
        rec.get("queue_id", 0),
        rec.get("dispatch_id", 0),
        tuple(rec.get("block", (0, 0, 0))),
        rec.get("wave_id", 0),
        rec["xcc_id"],
        rec["se"],
        rec["sh"],
        rec["cu"],
        rec["simd"],
        rec["wave"],
    )


def _cu_label(rec: dict) -> str:
    return f"XCD{rec['xcc_id']}/SE{rec['se']}/SH{rec['sh']}/CU{rec['cu']}"


def group_launches(records: list[dict]) -> list[tuple[tuple[int, int], list[dict]]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec.get("queue_id", 0), rec.get("dispatch_id", 0))].append(rec)
    return sorted(grouped.items(), key=lambda item: min(rec["ts"] for rec in item[1]))


def build_timeline(records: list[dict], names: dict[str, int], *, kernel: str = "kernel") -> dict:
    """Nest records as kernel -> launch -> CU -> workgroup -> wave tracks holding slices and instants.

    Timestamps are nanoseconds from the earliest record."""
    by_id = {v: k for k, v in names.items()}
    timeline: dict[str, list[dict]] = {"tracks": [], "slices": [], "instants": []}
    if not records:
        return timeline

    def track(name, parent=None, order=0):
        timeline["tracks"].append({"uuid": len(timeline["tracks"]) + 1, "name": name, "parent": parent, "order": order})
        return len(timeline["tracks"])

    t0 = min(r["ts"] for r in records)
    root = track(kernel)
    for ordinal, (identity, group) in enumerate(group_launches(records), 1):
        launch = track(f"launch {ordinal} (queue {identity[0]:x}, dispatch {identity[1]})", root, ordinal)
        waves: dict[tuple, list[dict]] = defaultdict(list)
        for rec in group:
            waves[_wave_key(rec)].append(rec)
        cus: dict[str, int] = {}
        wgs: dict[tuple, int] = {}
        for key, wave_records in sorted(waves.items()):
            wave_records.sort(key=lambda r: (r["ts"], r["slot"]))
            cu = _cu_label(wave_records[0])
            if cu not in cus:
                cus[cu] = track(cu, launch, len(cus))
            block = key[2]
            if (cu, block) not in wgs:
                wgs[cu, block] = track("WG(%d, %d, %d)" % block, cus[cu], len(wgs))
            wave = track(f"Wave{key[3]}", wgs[cu, block], key[3])
            _wave_events(wave_records, by_id, wave, t0, timeline, track)
    return timeline


def _wave_events(group: list[dict], by_id: dict[int, str], wave: int, t0: int, timeline: dict, track) -> None:
    """Emit one wave's records. ``seq`` orders same-timestamp events on a track by record order."""

    def ns(rec):
        return round((rec["ts"] - t0) * NS_PER_TICK)

    def label(rec):
        return by_id.get(rec["event_id"], f"event{rec['event_id']}")

    def add_slice(track_uuid, start, end, seq, name, closed=True, annotated=True):
        timeline["slices"].append(
            {
                "track": track_uuid,
                "start": ns(start),
                "end": ns(end),
                "seq": seq,
                "name": name if closed else f"{name} (unclosed)",
                "payload": start["payload"] if annotated else None,
                "payload_type": start.get("payload_type"),
            }
        )

    n = len(group)
    first, last = group[0], group[-1]
    stack: list[tuple[int, dict]] = []
    open_tokens: dict[int, tuple[int, dict]] = {}
    token_ranges: list[tuple[dict, dict, tuple[int, int], bool]] = []
    for i, rec in enumerate(group):
        kind = rec["kind"]
        if kind == _ktrace.KIND_MARK:
            timeline["instants"].append(
                {
                    "track": wave,
                    "ts": ns(rec),
                    "seq": i,
                    "name": label(rec),
                    "payload": rec["payload"],
                    "payload_type": rec.get("payload_type"),
                }
            )
        elif kind == _ktrace.KIND_RANGE_PUSH:
            stack.append((i, rec))
        elif kind == _ktrace.KIND_RANGE_POP and stack:
            j, start = stack.pop()
            add_slice(wave, start, rec, (j, i), label(start))
        elif kind == _ktrace.KIND_RANGE_START:
            open_tokens[rec["slot"]] = (i, rec)
        elif kind == _ktrace.KIND_RANGE_STOP and rec["start_slot"] in open_tokens:
            j, start = open_tokens.pop(rec["start_slot"])
            token_ranges.append((start, rec, (j, i), True))
    while stack:
        j, start = stack.pop()
        add_slice(wave, start, last, (j, 2 * n - 1 - len(stack)), label(start), closed=False)
    token_ranges.extend((start, last, (j, n), False) for j, start in open_tokens.values())
    add_slice(wave, first, last, (-1, 2 * n), "Wave Span", annotated=False)

    # Token ranges may overlap without nesting: give each the first lane whose previous range
    # stopped at an earlier record. Comparing timestamps would let equal-tick overlaps share a lane.
    lanes: list[list[int]] = []
    for start, end, seq, closed in sorted(token_ranges, key=lambda r: r[2]):
        lane = next((lane for lane in lanes if lane[1] < seq[0]), None)
        if lane is None:
            lane = [track(f"ranges {len(lanes)}", wave, len(lanes)), -1]
            lanes.append(lane)
        lane[1] = seq[1]
        add_slice(lane[0], start, end, seq, label(start), closed)


# Perfetto TracePacket / TrackDescriptor / TrackEvent / DebugAnnotation field numbers.
_PACKET = 1
_PACKET_TIMESTAMP = 8
_PACKET_SEQUENCE_ID = 10
_PACKET_TRACK_EVENT = 11
_PACKET_TRACK_DESCRIPTOR = 60
_EVENT_ANNOTATION = 4
_EVENT_TYPE = 9
_EVENT_TRACK = 11
_EVENT_CATEGORY = 22
_EVENT_NAME = 23
_SLICE_BEGIN, _SLICE_END, _INSTANT = 1, 2, 3
_CHILD_ORDER_EXPLICIT = 3


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _uint(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _blob(field: int, value: bytes | str) -> bytes:
    value = value.encode() if isinstance(value, str) else value
    return _varint(field << 3 | 2) + _varint(len(value)) + value


def _payload_annotation(payload, payload_type=None) -> bytes:
    annotation = _blob(10, "payload")
    if isinstance(payload, bool):
        annotation += _uint(2, int(payload))
    elif isinstance(payload, float) and not math.isfinite(payload):
        annotation += _blob(6, str(payload))
    elif isinstance(payload, float):
        annotation += _varint(5 << 3 | 1) + struct.pack("<d", payload)
    elif payload_type == "uint64" or payload >= 1 << 63:
        annotation += _uint(3, payload)
    else:
        annotation += _uint(4, payload)
    return _blob(_EVENT_ANNOTATION, annotation)


def _track_event(ts: int, track: int, kind: int, name: str | None = None, payload=None, payload_type=None) -> bytes:
    event = _uint(_EVENT_TYPE, kind) + _uint(_EVENT_TRACK, track) + _blob(_EVENT_CATEGORY, "ktrace")
    if name is not None:
        event += _blob(_EVENT_NAME, name)
    if payload is not None:
        event += _payload_annotation(payload, payload_type)
    packet = _uint(_PACKET_TIMESTAMP, ts) + _uint(_PACKET_SEQUENCE_ID, 1) + _blob(_PACKET_TRACK_EVENT, event)
    return _blob(_PACKET, packet)


def encode_pftrace(timeline: dict) -> bytes:
    out = bytearray()
    for t in timeline["tracks"]:
        descriptor = _uint(1, t["uuid"]) + _blob(2, t["name"]) + _uint(11, _CHILD_ORDER_EXPLICIT)
        if t["parent"] is not None:
            descriptor += _uint(5, t["parent"]) + _uint(12, t["order"])
        out += _blob(_PACKET, _blob(_PACKET_TRACK_DESCRIPTOR, descriptor))

    events = []
    for sl in timeline["slices"]:
        events.append(
            (
                sl["start"],
                sl["seq"][0],
                _track_event(sl["start"], sl["track"], _SLICE_BEGIN, sl["name"], sl["payload"], sl.get("payload_type")),
            )
        )
        events.append((sl["end"], sl["seq"][1], _track_event(sl["end"], sl["track"], _SLICE_END)))
    for inst in timeline["instants"]:
        events.append(
            (
                inst["ts"],
                inst["seq"],
                _track_event(
                    inst["ts"], inst["track"], _INSTANT, inst["name"], inst["payload"], inst.get("payload_type")
                ),
            )
        )
    events.sort(key=lambda e: e[:2])
    for event in events:
        out += event[2]
    return bytes(out)


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
            elif kind == _ktrace.KIND_RANGE_STOP:
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

    path = _unique(os.path.join(out_dir, f"{kernel}_ktrace.pftrace.gz"))
    with open(path, "wb") as fh:
        fh.write(gzip.compress(encode_pftrace(build_timeline(records, names, kernel=kernel)), mtime=0))
    return path


def _unique(path: str) -> str:
    if not os.path.exists(path):
        return path
    ext = ".pftrace.gz" if path.endswith(".pftrace.gz") else os.path.splitext(path)[1]
    stem = path[: -len(ext)]
    n = 1
    while os.path.exists(f"{stem}.{n}{ext}"):
        n += 1
    return f"{stem}.{n}{ext}"
