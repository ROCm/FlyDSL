# SPDX-License-Identifier: Apache-2.0
"""Write GLM flytrace records as a hierarchical Perfetto protobuf trace."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

_STAGE_ORDER = (
    "qkv_a",
    "cache",
    "q_b",
    "uk",
    "split",
    "uv",
    "o",
    "router",
    "shared_prefetch",
    "gating",
    "expert_prepare",
    "shared_expert",
    "ug",
    "down",
)
_SLICE_PREFIX = "FlyDSL · "

# Stable Perfetto proto fields used below. Keeping this writer dependency-free
# avoids requiring the full Perfetto protobuf package in benchmark environments.
# Definitions: trace.proto, trace_packet.proto, track_descriptor.proto, and
# track_event.proto in the upstream Perfetto repository.
_TRACE_PACKET = 1
_PACKET_TIMESTAMP = 8
_PACKET_SEQUENCE_ID = 10
_PACKET_TRACK_EVENT = 11
_PACKET_TRACK_DESCRIPTOR = 60
_EVENT_DEBUG_ANNOTATION = 4
_EVENT_TYPE = 9
_EVENT_TRACK_UUID = 11
_EVENT_CATEGORY = 22
_EVENT_NAME = 23
_TYPE_SLICE_BEGIN = 1
_TYPE_SLICE_END = 2
_TYPE_INSTANT = 3
_CHILD_ORDER_LEXICOGRAPHIC = 1
_CHILD_ORDER_EXPLICIT = 3


def _slice_name(name: str) -> str:
    # Perfetto derives slice colors from display names. This namespace gives
    # FlyDSL traces a stable project-specific palette.
    return f"{_SLICE_PREFIX}{name}"


def _varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _uint(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _blob(field: int, value: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(value)) + value


def _text(field: int, value: str) -> bytes:
    return _blob(field, value.encode())


def _trace_packet(payload: bytes) -> bytes:
    return _blob(_TRACE_PACKET, payload)


def _track_descriptor(
    uuid: int,
    name: str,
    *,
    parent_uuid: int | None = None,
    child_ordering: int | None = None,
    sibling_order_rank: int | None = None,
) -> bytes:
    descriptor = _uint(1, uuid) + _text(2, name)
    if parent_uuid is not None:
        descriptor += _uint(5, parent_uuid)
    if child_ordering is not None:
        descriptor += _uint(11, child_ordering)
    if sibling_order_rank is not None:
        descriptor += _uint(12, sibling_order_rank)
    return _trace_packet(_blob(_PACKET_TRACK_DESCRIPTOR, descriptor))


def _debug_annotation(name: str, value: int | str) -> bytes:
    annotation = _text(10, name)
    if isinstance(value, int):
        annotation += _uint(4, value)
    else:
        annotation += _text(6, value)
    return _blob(_EVENT_DEBUG_ANNOTATION, annotation)


def _track_event(
    timestamp_ns: int,
    track_uuid: int,
    kind: int,
    name: str,
    annotations: dict[str, int | str] | None = None,
) -> bytes:
    event = (
        _uint(_EVENT_TYPE, kind)
        + _uint(_EVENT_TRACK_UUID, track_uuid)
        + _text(_EVENT_CATEGORY, "flydsl.glm5")
        + _text(_EVENT_NAME, name)
    )
    for key, value in (annotations or {}).items():
        event += _debug_annotation(key, value)
    packet = (
        _uint(_PACKET_TIMESTAMP, timestamp_ns)
        + _uint(_PACKET_SEQUENCE_ID, 1)
        + _blob(_PACKET_TRACK_EVENT, event)
    )
    return _trace_packet(packet)


def _timestamp_ns(event: dict) -> int:
    return round(event["ts"] * 1000)


def _range_ns(event: dict) -> tuple[int, int]:
    start_ns = _timestamp_ns(event)
    end_ns = round((event["ts"] + event["dur"]) * 1000)
    if end_ns < start_ns:
        raise ValueError(f"negative flytrace range: {event.get('name', '<unnamed>')}")
    return start_ns, end_ns


class _TraceBuilder:
    def __init__(self):
        self._next_uuid = 1
        self._descriptors: list[bytes] = []
        self._events: list[tuple[int, int, int, int, bytes]] = []
        self._event_order = 0
        self.slices = 0
        self.instants = 0

    @property
    def tracks(self) -> int:
        return self._next_uuid - 1

    def track(
        self,
        name: str,
        *,
        parent: int | None = None,
        child_ordering: int | None = None,
        sibling_order: int | None = None,
    ) -> int:
        uuid = self._next_uuid
        self._next_uuid += 1
        self._descriptors.append(
            _track_descriptor(
                uuid,
                name,
                parent_uuid=parent,
                child_ordering=child_ordering,
                sibling_order_rank=sibling_order,
            )
        )
        return uuid

    def _event(
        self,
        timestamp_ns: int,
        track: int,
        kind: int,
        name: str,
        *,
        depth: int,
        annotations: dict[str, int | str] | None = None,
    ) -> None:
        if kind == _TYPE_SLICE_END:
            type_order, depth_order = 0, -depth
        elif kind == _TYPE_INSTANT:
            type_order, depth_order = 1, 0
        else:
            type_order, depth_order = 2, depth
        packet = _track_event(timestamp_ns, track, kind, name, annotations)
        self._events.append((timestamp_ns, type_order, depth_order, self._event_order, packet))
        self._event_order += 1

    def slice(
        self,
        track: int,
        name: str,
        start_ns: int,
        end_ns: int,
        *,
        depth: int,
        annotations: dict[str, int | str] | None = None,
    ) -> None:
        end_ns = max(start_ns, end_ns)
        self._event(start_ns, track, _TYPE_SLICE_BEGIN, name, depth=depth, annotations=annotations)
        self._event(end_ns, track, _TYPE_SLICE_END, name, depth=depth)
        self.slices += 1

    def instant(
        self,
        track: int,
        name: str,
        timestamp_ns: int,
        *,
        depth: int,
        annotations: dict[str, int | str] | None = None,
    ) -> None:
        self._event(timestamp_ns, track, _TYPE_INSTANT, name, depth=depth, annotations=annotations)
        self.instants += 1

    def write(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=str(target), mode="wb", compresslevel=9, mtime=0) as output:
            for descriptor in self._descriptors:
                output.write(descriptor)
            for event in sorted(self._events):
                output.write(event[-1])


def _stage_groups(events: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_key = None
    for event in events:
        stage, separator, phase = event.get("name", "").partition("/")
        if not separator:
            if current:
                groups.append(current)
                current = []
            groups.append([event])
            current_key = None
            continue
        key = (stage, event.get("args", {}).get("payload"))
        if current and (key != current_key or phase == "start"):
            groups.append(current)
            current = []
        current.append(event)
        current_key = key
        if phase == "publish":
            groups.append(current)
            current = []
            current_key = None
    if current:
        groups.append(current)
    return groups


def _wave_identity(label: str, tid: int) -> tuple[str, str, str, int]:
    if not isinstance(label, str):
        return "Flytrace Grid", f"Block track {tid}", "Wave00", 0
    for separator in (" Block ", " CTA "):
        try:
            kernel, location = label.split(separator, 1)
            block, wave = location.rsplit(" / wave ", 1)
            wave_number = int(wave)
            return kernel, f"Block {block}", f"Wave{wave_number:02d}", wave_number
        except (ValueError, TypeError):
            pass
    return "Flytrace Grid", f"Block track {tid}", "Wave00", 0


def export_hierarchical_pftrace(source: Path, target: Path, rank: int) -> dict[str, int]:
    """Convert full-wave Chrome JSON into an AITER-style nested .pftrace.gz."""
    if not str(target).endswith(".pftrace.gz"):
        raise ValueError("hierarchical Perfetto output must end in .pftrace.gz")
    try:
        source_trace = json.loads(source.read_text())
        source_events = source_trace["traceEvents"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid flytrace JSON: {source}") from error
    if not isinstance(source_events, list):
        raise ValueError("flytrace traceEvents must be a list")
    thread_names: dict[int, str] = {}
    wave_events: dict[int, list[dict]] = defaultdict(list)
    wave_ranges: dict[int, list[dict]] = defaultdict(list)
    outer_ranges: dict[int, dict] = {}
    stage_events: dict[str, list[dict]] = defaultdict(list)
    stage_ranges: dict[str, list[dict]] = defaultdict(list)
    stage_phase_ranges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in source_events:
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            thread_names[event["tid"]] = event["args"]["name"]
        elif event.get("ph") == "i":
            wave_events[event["tid"]].append(event)
            stage, separator, _ = event.get("name", "").partition("/")
            if separator:
                stage_events[stage].append(event)
        elif event.get("ph") == "X":
            if event.get("name") == "glm5_mla_moe":
                outer_ranges[event["tid"]] = event
            else:
                wave_ranges[event["tid"]].append(event)
                stage, separator, phase = event.get("name", "range").partition("/")
                stage_ranges[stage].append(event)
                if separator:
                    stage_phase_ranges[stage, phase].append(event)

    trace_tids = set(wave_events) | set(wave_ranges)
    if not trace_tids:
        raise ValueError("flytrace contains no wave events")
    missing_outer = sorted(trace_tids - set(outer_ranges))
    if missing_outer:
        raise ValueError(f"flytrace waves are missing glm5_mla_moe ranges: {missing_outer[:8]}")

    builder = _TraceBuilder()
    root = builder.track("Root", child_ordering=_CHILD_ORDER_EXPLICIT)
    rank_track = builder.track(
        f"Rank {rank}", parent=root, child_ordering=_CHILD_ORDER_EXPLICIT, sibling_order=rank
    )
    layer_number = next(
        (event.get("args", {}).get("payload", 0) for event in outer_ranges.values()),
        0,
    )
    layer = builder.track(
        f"GLM5 Layer {layer_number}",
        parent=rank_track,
        child_ordering=_CHILD_ORDER_EXPLICIT,
        sibling_order=layer_number,
    )
    stage_parent = builder.track(
        "Stage Breakdown", parent=layer, child_ordering=_CHILD_ORDER_EXPLICIT, sibling_order=0
    )

    present_stages = set(stage_events) | set(stage_ranges)
    ordered_stages = [stage for stage in _STAGE_ORDER if stage in present_stages]
    ordered_stages.extend(sorted(present_stages - set(ordered_stages)))
    for stage_index, stage in enumerate(ordered_stages):
        events = stage_events[stage]
        ranges = stage_ranges[stage]
        starts = [event for event in events if event["name"].endswith("/start")] or events
        ends = [event for event in events if event["name"].endswith("/publish")] or events
        range_bounds = [_range_ns(event) for event in ranges]
        start_ns = min([_timestamp_ns(event) for event in starts] + [start for start, _ in range_bounds])
        end_ns = max([_timestamp_ns(event) for event in ends] + [end for _, end in range_bounds])
        payloads = [event.get("args", {}).get("payload") for event in events + ranges]
        payloads = [payload for payload in payloads if payload is not None]
        annotations: dict[str, int | str] = {"events": len(events) + len(ranges)}
        if payloads:
            annotations.update(payload_min=min(payloads), payload_max=max(payloads))
        phases = [phase for parent, phase in stage_phase_ranges if parent == stage]
        track = builder.track(
            stage,
            parent=stage_parent,
            child_ordering=_CHILD_ORDER_EXPLICIT if phases else None,
            sibling_order=stage_index,
        )
        builder.slice(track, _slice_name(stage), start_ns, end_ns, depth=1, annotations=annotations)
        for phase_index, phase in enumerate(phases):
            phase_ranges = stage_phase_ranges[stage, phase]
            phase_bounds = [_range_ns(event) for event in phase_ranges]
            phase_track = builder.track(phase, parent=track, sibling_order=phase_index)
            builder.slice(
                phase_track,
                _slice_name(f"{stage}/{phase}"),
                min(start for start, _ in phase_bounds),
                max(end for _, end in phase_bounds),
                depth=1,
                annotations={"ranges": len(phase_ranges)},
            )

    grid_tracks: dict[str, int] = {}
    block_tracks: dict[tuple[str, str], int] = {}
    block_order: dict[str, int] = defaultdict(int)
    grid_outer: dict[str, tuple[int, int, int]] = {}
    for tid in sorted(trace_tids):
        kernel, block_name, wave_name, wave_number = _wave_identity(thread_names.get(tid, ""), tid)
        if kernel not in grid_tracks:
            grid_tracks[kernel] = builder.track(
                kernel,
                parent=layer,
                child_ordering=_CHILD_ORDER_EXPLICIT,
                sibling_order=100 + len(grid_tracks),
            )
        grid = grid_tracks[kernel]
        block_key = (kernel, block_name)
        if block_key not in block_tracks:
            block_tracks[block_key] = builder.track(
                block_name,
                parent=grid,
                child_ordering=_CHILD_ORDER_EXPLICIT,
                sibling_order=block_order[kernel],
            )
            block_order[kernel] += 1
        wave = builder.track(
            wave_name,
            parent=block_tracks[block_key],
            child_ordering=_CHILD_ORDER_EXPLICIT,
            sibling_order=wave_number,
        )
        stacked = builder.track(
            "StackedRanges", parent=wave, child_ordering=_CHILD_ORDER_LEXICOGRAPHIC, sibling_order=100
        )
        outer = outer_ranges[tid]
        outer_start, outer_end = _range_ns(outer)
        builder.slice(
            stacked,
            _slice_name("timeline envelope"),
            outer_start,
            outer_end,
            depth=1,
            annotations={"layer": outer.get("args", {}).get("payload", layer_number)},
        )
        if kernel in grid_outer:
            grid_start, grid_end, grid_track = grid_outer[kernel]
            grid_outer[kernel] = min(grid_start, outer_start), max(grid_end, outer_end), grid_track
        else:
            grid_outer[kernel] = outer_start, outer_end, grid

        for group in _stage_groups(wave_events[tid]):
            first, last = group[0], group[-1]
            first_ns, last_ns = _timestamp_ns(first), _timestamp_ns(last)
            if last_ns < first_ns:
                raise ValueError(f"non-monotonic flytrace stage range on wave {tid}")
            stage, separator, first_phase = first.get("name", "").partition("/")
            payload = first.get("args", {}).get("payload")
            annotations = {} if payload is None else {"task": payload}
            if not separator or len(group) == 1:
                builder.instant(
                    stacked,
                    _slice_name(first.get("name", "mark")),
                    first_ns,
                    depth=2,
                    annotations=annotations,
                )
                continue
            task_name = f"{stage}[{payload}]" if payload is not None else stage
            builder.slice(stacked, _slice_name(task_name), first_ns, last_ns, depth=2, annotations=annotations)
            previous_phase = first_phase
            for left, right in zip(group, group[1:]):
                _, _, right_phase = right["name"].partition("/")
                builder.slice(
                    stacked,
                    _slice_name(f"{previous_phase} → {right_phase}"),
                    _timestamp_ns(left),
                    _timestamp_ns(right),
                    depth=3,
                    annotations=annotations,
                )
                previous_phase = right_phase

        for event in sorted(wave_ranges[tid], key=_timestamp_ns):
            start_ns, end_ns = _range_ns(event)
            name = event.get("name", "range")
            payload = event.get("args", {}).get("payload")
            display_name = f"{name}[sample={payload}]" if payload is not None else name
            annotations = {} if payload is None else {"sample": payload}
            builder.slice(
                stacked,
                _slice_name(display_name),
                start_ns,
                end_ns,
                depth=4,
                annotations=annotations,
            )

    for kernel, (start_ns, end_ns, track) in grid_outer.items():
        builder.slice(
            track,
            _slice_name("layer envelope"),
            start_ns,
            end_ns,
            depth=1,
            annotations={"rank": rank, "layer": layer_number, "waves": len(trace_tids), "kernel": kernel},
        )

    builder.write(target)
    return dict(pftrace_tracks=builder.tracks, pftrace_slices=builder.slices, pftrace_instants=builder.instants)
