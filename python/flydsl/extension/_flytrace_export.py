# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral trace validation and Perfetto JSON export."""

import json
import re
from collections import defaultdict
from pathlib import Path


def trace_stats(waves):
    return {
        "waves": len(waves),
        "records": sum(len(wave["events"]) for wave in waves),
        "dropped": sum(max(0, wave["attempted_events"] - len(wave["events"])) for wave in waves),
    }


def _event_args(event):
    args = {"site": event["site"]} if "site" in event else {}
    if event.get("payload") is not None:
        args["payload"] = event["payload"]
    return args


def perfetto_events(waves, clock_hz, *, origin=None):
    """Convert decoded wave records to the Perfetto Trace Event schema."""

    if not isinstance(clock_hz, int) or clock_hz <= 0:
        raise ValueError("flytrace clock frequency must be a positive integer")
    if origin is None:
        origin = min((wave["epoch"] for wave in waves), default=0)
    elif not isinstance(origin, int):
        raise TypeError("flytrace origin tick must be an integer or None")
    scale = 1_000_000 / clock_hz
    output = []
    for tid, wave in enumerate(waves, 1):
        output.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 1,
                "tid": tid,
                "args": {"name": f"{wave['kernel']} Block {wave['block']} / wave {wave['wave']}"},
            }
        )
        stack = []
        token_ranges = defaultdict(list)
        current = None

        if wave.get("overflow"):
            output.append(
                {
                    "ph": "i",
                    "s": "t",
                    "name": "flytrace overflow",
                    "cat": "flytrace.warning",
                    "pid": 1,
                    "tid": tid,
                    "ts": (wave["end_tick"] - origin) * scale,
                    "args": {
                        "attempted": wave["attempted_events"],
                        "captured": len(wave["events"]),
                    },
                }
            )

        def interval(start, finish):
            args = _event_args(start)
            if finish.get("payload") is not None:
                args["end_payload"] = finish["payload"]
            output.append(
                {
                    "ph": "X",
                    "name": start["name"],
                    "pid": 1,
                    "tid": tid,
                    "ts": (start["tick"] - origin) * scale,
                    "dur": (finish["tick"] - start["tick"]) * scale,
                    "args": args,
                }
            )

        for event in wave["events"]:
            kind = event["kind"]
            if kind == "mark":
                output.append(
                    {
                        "ph": "i",
                        "s": "t",
                        "name": event["name"],
                        "pid": 1,
                        "tid": tid,
                        "ts": (event["tick"] - origin) * scale,
                        "args": _event_args(event),
                    }
                )
            elif kind == "push":
                stack.append(event)
            elif kind == "pop":
                if not stack:
                    raise ValueError("unmatched flytrace.range_pop()")
                interval(stack.pop(), event)
            elif kind == "range_start":
                token_ranges[event.get("range_id", event["name"])].append(event)
            elif kind == "range_end":
                starts = token_ranges[event.get("range_id", event["name"])]
                if not starts:
                    raise ValueError(f"unmatched flytrace.range_end() for {event['name']!r}")
                interval(starts.pop(), event)
            elif kind in ("boundary", "end"):
                if current is not None:
                    interval(current, event)
                current = event if kind == "boundary" else None
            else:
                raise ValueError(f"unknown flytrace event kind: {kind}")
        unfinished_tokens = [identifier for identifier, starts in token_ranges.items() if starts]
        if (stack or current is not None or unfinished_tokens) and not wave.get("overflow"):
            raise ValueError("unfinished flytrace range")
    return output


def export_perfetto_json(waves, path, clock_hz, *, origin=None):
    events = perfetto_events(waves, clock_hz, origin=origin)
    Path(path).write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}) + "\n")
    return trace_stats(waves)


def _safe_filename(name):
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return filename or "kernel"


def export_perfetto_per_kernel(groups, directory, clock_hz):
    """Write one aligned Perfetto JSON file per compiled kernel.

    ``groups`` contains ``recording``, ``kernel``, and ``waves`` fields. Files
    share the capture-wide clock origin, so their timestamps remain comparable.
    A manifest identifies files unambiguously when different compiled modules
    use the same kernel symbol.
    """

    if not isinstance(clock_hz, int) or clock_hz <= 0:
        raise ValueError("flytrace clock frequency must be a positive integer")
    groups = list(groups)
    waves = [wave for group in groups for wave in group["waves"]]
    origin = min((wave["epoch"] for wave in waves), default=0)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for index, group in enumerate(groups):
        filename = f"{index:03d}-{_safe_filename(group['kernel'])}.json"
        stats = export_perfetto_json(group["waves"], target / filename, clock_hz, origin=origin)
        files.append(
            {
                "recording": group["recording"],
                "kernel": group["kernel"],
                "file": filename,
                **stats,
            }
        )
    manifest = {
        "format": "flytrace.per_kernel.v1",
        "clock_hz": clock_hz,
        "origin_tick": origin,
        "kernels": len(files),
        **trace_stats(waves),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def save_raw_trace(waves, path, metadata):
    Path(path).write_text(json.dumps({"format": "flytrace.raw.v1", **metadata, "waves": waves}) + "\n")
    return trace_stats(waves)
