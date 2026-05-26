#!/usr/bin/env python3
"""Summarize cycle cost for a contiguous instruction segment in rocprof UI traces.

The rocprofv3 UI trace format stores a global instruction table in code.json and
per-wave execution records in se*_sm*_sl*_wv*.json.  Each wave instruction record
has the shape:

    [timestamp, type, stall, latency, code_idx]

This script selects a contiguous segment from code.json, finds every exact
occurrence of that segment in the wave traces, and reports the segment cycle
cost plus the slowest instructions inside the segment.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CodeRow:
    idx: int
    isa: str
    line: int
    vaddr: int
    hit: int
    latency: int
    stall: int
    idle: int


@dataclass(frozen=True)
class SegmentEvent:
    code_idx: int
    start_ts: int
    latency: int
    stall: int


@dataclass(frozen=True)
class Occurrence:
    wave_file: str
    se: int | None
    simd: int | None
    slot: int | None
    wave_id: int | None
    start_pos: int
    start_ts: int
    cycles_sum: int
    cycles_span: int
    events: tuple[SegmentEvent, ...]


@dataclass(frozen=True)
class IntervalResult:
    wave_file: str
    se: int | None
    simd: int | None
    slot: int | None
    wave_id: int | None
    start_pos: int
    end_pos: int
    start_ts: int
    end_ts: int
    cycles: int
    inst_count: int
    start_code_indices: tuple[int, ...]
    end_code_indices: tuple[int, ...]
    events: tuple[SegmentEvent, ...]


@dataclass(frozen=True)
class IntervalInfo:
    description: str | None = None
    inc_inst_pattern: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SampleIntervalAnalysis:
    idx: int
    description: str | None
    inc_pattern: tuple[str, ...] | None
    start_code_sequences: list[set[int]]
    end_code_sequences: list[set[int]]
    intervals: list[IntervalResult]
    primary_intervals: list[IntervalResult]
    inc_matched_intervals: list[IntervalResult]
    inc_unmatched_intervals: list[IntervalResult]


@dataclass(frozen=True)
class SummaryRecord:
    idx_label: str
    description: str | None
    intervals: list[IntervalResult]
    inc_pattern: tuple[str, ...] | None


@dataclass(frozen=True)
class KernelAnalysis:
    name: str
    baseline: bool
    rows: list[CodeRow]
    trace_dir: Path
    sample_intervals: list[SampleIntervalAnalysis]
    summary_records: list[SummaryRecord]
    all_intervals: list[IntervalResult]


def parse_int(value: str | int | None) -> int:
    if value is None:
        raise ValueError("missing integer value")
    if isinstance(value, int):
        return value
    return int(value, 0)


def normalize_inst(text: str) -> str:
    return " ".join(text.strip().split())


def split_instruction_text(text: str, separator: str | None) -> list[str]:
    text = text.replace("\\n", "\n")
    if separator is None:
        return text.splitlines()
    sep = separator.replace("\\n", "\n").replace("\\t", "\t")
    return text.split(sep)


def normalized_instruction_lines(instruction_lines: list[str]) -> list[str]:
    return [normalize_inst(line) for line in instruction_lines if line.strip() and not line.lstrip().startswith("#")]


def extract_top_level_list_literals(text: str, path: Path) -> list[object]:
    literals: list[object] = []
    start: int | None = None
    depth = 0
    quote: str | None = None
    escaped = False

    for idx, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"{path} has unmatched closing bracket")
            if depth == 0 and start is not None:
                literals.append(ast.literal_eval(text[start : idx + 1]))
                start = None

    if depth != 0:
        raise SystemExit(f"{path} has unmatched opening bracket")
    if not literals:
        raise SystemExit(f"{path} does not contain a top-level list")
    return literals


def interval_info_from_description(description: str | None) -> IntervalInfo:
    return IntervalInfo(description=description.strip() if description else None)


def interval_info_from_object(obj: object, path: Path, idx: int) -> IntervalInfo:
    if isinstance(obj, str):
        return interval_info_from_description(obj)
    if not isinstance(obj, dict):
        raise SystemExit(f"{path} interval information {idx} must be a string or object")

    description: str | None = None
    key = obj.get("key")
    if key == "desc":
        value = obj.get("value")
        if value is not None and not isinstance(value, str):
            raise SystemExit(f"{path} interval information {idx} desc value must be a string")
        description = value
    elif isinstance(obj.get("desc"), str):
        description = obj["desc"]
    elif isinstance(obj.get("description"), str):
        description = obj["description"]
    elif isinstance(obj.get("value"), str):
        description = obj["value"]

    pattern = obj.get("inc_inst_pattern")
    inc_inst_pattern: tuple[str, ...] | None = None
    if pattern is not None:
        if not isinstance(pattern, list) or not all(isinstance(item, str) for item in pattern):
            raise SystemExit(f"{path} interval information {idx} inc_inst_pattern must be a list of strings")
        inc_inst_pattern = tuple(normalize_inst(item) for item in pattern if item.strip())
        if not inc_inst_pattern:
            raise SystemExit(f"{path} interval information {idx} inc_inst_pattern is empty")

    return IntervalInfo(description=description.strip() if description else None, inc_inst_pattern=inc_inst_pattern)


def load_interval_points_file(path: Path) -> tuple[list[list[str]], list[IntervalInfo] | None]:
    text = path.read_text()
    literals = extract_top_level_list_literals(text, path)
    if len(literals) > 2:
        raise SystemExit(f"{path} may contain at most two top-level lists: points and optional descriptions")

    data = literals[0]
    if not isinstance(data, list) or len(data) < 2:
        raise SystemExit(f"{path} must contain a 2D list with at least two sample points")

    points: list[list[str]] = []
    for point_idx, point in enumerate(data):
        if not isinstance(point, list):
            raise SystemExit(f"{path} sample point {point_idx} is not a list")
        if not all(isinstance(item, str) for item in point):
            raise SystemExit(f"{path} sample point {point_idx} must contain only strings")
        normalized = normalized_instruction_lines(point)
        if not normalized:
            raise SystemExit(f"{path} sample point {point_idx} is empty after filtering")
        points.append(normalized)

    interval_infos: list[IntervalInfo] | None = None
    if len(literals) == 2:
        raw_infos = literals[1]
        if not isinstance(raw_infos, list):
            raise SystemExit(f"{path} second top-level list must contain interval descriptions")
        if len(raw_infos) != len(points) - 1:
            raise SystemExit(
                f"{path} description count must equal sample interval count "
                f"({len(points) - 1}), got {len(raw_infos)}"
            )
        interval_infos = [interval_info_from_object(item, path, idx) for idx, item in enumerate(raw_infos)]
    return points, interval_infos


def load_json_config(path: Path) -> dict[str, object]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def parse_config_sample_points(config: dict[str, object], path: Path) -> list[list[str]]:
    raw_points = config.get("sample points") or config.get("sample_points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise SystemExit(f"{path} must contain 'sample points' with at least two entries")

    points: list[list[str]] = []
    for point_idx, raw_point in enumerate(raw_points):
        instructions: object
        if isinstance(raw_point, list):
            instructions = raw_point
        elif isinstance(raw_point, dict):
            if "instruction" in raw_point:
                instructions = raw_point["instruction"]
            elif len(raw_point) == 1:
                point_value = next(iter(raw_point.values()))
                if not isinstance(point_value, dict) or "instruction" not in point_value:
                    raise SystemExit(f"{path} sample point {point_idx} must contain an instruction list")
                instructions = point_value["instruction"]
            else:
                raise SystemExit(f"{path} sample point {point_idx} must contain an instruction list")
        else:
            raise SystemExit(f"{path} sample point {point_idx} must be an object or list")

        if not isinstance(instructions, list) or not all(isinstance(item, str) for item in instructions):
            raise SystemExit(f"{path} sample point {point_idx} instruction must be a list of strings")
        normalized = normalized_instruction_lines(instructions)
        if not normalized:
            raise SystemExit(f"{path} sample point {point_idx} is empty after filtering")
        points.append(normalized)
    return points


def parse_config_interval_infos(config: dict[str, object], points: list[list[str]], path: Path) -> list[IntervalInfo] | None:
    raw_infos = config.get("interval information") or config.get("interval_information")
    if raw_infos is None:
        return None
    if not isinstance(raw_infos, list):
        raise SystemExit(f"{path} 'interval information' must be a list")
    if len(raw_infos) != len(points) - 1:
        raise SystemExit(
            f"{path} interval information count must equal sample interval count "
            f"({len(points) - 1}), got {len(raw_infos)}"
        )
    return [interval_info_from_object(item, path, idx) for idx, item in enumerate(raw_infos)]


def config_value(config: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in config:
            return config[key]
    return None


def optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return str(value)
    raise SystemExit(f"expected string or integer config value, got {type(value).__name__}")


def load_code(trace_dir: Path) -> list[CodeRow]:
    with (trace_dir / "code.json").open() as f:
        raw = json.load(f)["code"]

    rows: list[CodeRow] = []
    for idx, row in enumerate(raw):
        rows.append(
            CodeRow(
                idx=idx,
                isa=str(row[0]),
                line=int(row[2]),
                vaddr=int(row[5]),
                hit=int(row[6]),
                latency=int(row[7]),
                stall=int(row[8]),
                idle=int(row[9]),
            )
        )
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def range_segment(rows: list[CodeRow], start: int, end: int, attr: str) -> list[int]:
    if start > end:
        raise SystemExit(f"{attr} start must be <= end")
    selected = [row.idx for row in rows if start <= getattr(row, attr) <= end]
    if not selected:
        raise SystemExit(f"no code rows found for {attr} range {start}..{end}")
    return selected


def regex_segment(rows: list[CodeRow], start_pat: str, end_pat: str, occurrence: int) -> list[int]:
    if occurrence < 1:
        raise SystemExit("--code-occurrence is 1-based and must be >= 1")

    start_re = re.compile(start_pat)
    end_re = re.compile(end_pat)
    candidates: list[tuple[int, int]] = []
    for start_row in rows:
        if not start_re.search(start_row.isa):
            continue
        for end_row in rows[start_row.idx :]:
            if end_re.search(end_row.isa):
                candidates.append((start_row.idx, end_row.idx))
                break

    if not candidates:
        raise SystemExit("no segment matched --start-regex/--end-regex")
    if occurrence > len(candidates):
        raise SystemExit(
            f"--code-occurrence {occurrence} exceeds {len(candidates)} matched code segments"
        )

    start, end = candidates[occurrence - 1]
    return list(range(start, end + 1))


def exact_instruction_segment(
    rows: list[CodeRow],
    instruction_lines: list[str],
    occurrence: int,
    source_name: str,
) -> list[int]:
    targets = normalized_instruction_lines(instruction_lines)
    if not targets:
        raise SystemExit(f"{source_name} does not contain any instruction lines")
    if occurrence < 1:
        raise SystemExit("--code-occurrence is 1-based and must be >= 1")

    normalized_code = [normalize_inst(row.isa) for row in rows]
    matches: list[int] = []
    for start in range(0, len(rows) - len(targets) + 1):
        if normalized_code[start : start + len(targets)] == targets:
            matches.append(start)

    if not matches:
        raise SystemExit(f"{source_name} did not match any contiguous code.json instruction range")
    if occurrence > len(matches):
        raise SystemExit(
            f"--code-occurrence {occurrence} exceeds {len(matches)} exact instruction matches"
        )

    start = matches[occurrence - 1]
    return list(range(start, start + len(targets)))


def segment_file_segment(rows: list[CodeRow], segment_file: Path, occurrence: int) -> list[int]:
    return exact_instruction_segment(
        rows,
        segment_file.read_text().splitlines(),
        occurrence,
        str(segment_file),
    )


def segment_text_segment(
    rows: list[CodeRow],
    segment_text: str,
    separator: str | None,
    occurrence: int,
) -> list[int]:
    instruction_lines = split_instruction_text(segment_text, separator)
    return exact_instruction_segment(rows, instruction_lines, occurrence, "--segment-text")


def select_segment(args: argparse.Namespace, rows: list[CodeRow]) -> list[int]:
    selectors = [
        args.start_idx is not None or args.end_idx is not None,
        args.start_line is not None or args.end_line is not None,
        args.start_vaddr is not None or args.end_vaddr is not None,
        args.start_regex is not None or args.end_regex is not None,
        args.segment_file is not None,
        args.segment_text is not None,
    ]
    if sum(bool(s) for s in selectors) != 1:
        raise SystemExit(
            "select exactly one segment method: idx range, line range, vaddr range, "
            "regex pair, --segment-file, or --segment-text"
        )

    if args.start_idx is not None or args.end_idx is not None:
        if args.start_idx is None or args.end_idx is None:
            raise SystemExit("--start-idx and --end-idx must be provided together")
        segment = list(range(args.start_idx, args.end_idx + 1))
    elif args.start_line is not None or args.end_line is not None:
        if args.start_line is None or args.end_line is None:
            raise SystemExit("--start-line and --end-line must be provided together")
        segment = range_segment(rows, args.start_line, args.end_line, "line")
    elif args.start_vaddr is not None or args.end_vaddr is not None:
        if args.start_vaddr is None or args.end_vaddr is None:
            raise SystemExit("--start-vaddr and --end-vaddr must be provided together")
        segment = range_segment(
            rows,
            parse_int(args.start_vaddr),
            parse_int(args.end_vaddr),
            "vaddr",
        )
    elif args.start_regex is not None or args.end_regex is not None:
        if args.start_regex is None or args.end_regex is None:
            raise SystemExit("--start-regex and --end-regex must be provided together")
        segment = regex_segment(rows, args.start_regex, args.end_regex, args.code_occurrence)
    elif args.segment_file is not None:
        segment = segment_file_segment(rows, Path(args.segment_file), args.code_occurrence)
    else:
        segment = segment_text_segment(rows, args.segment_text, args.segment_separator, args.code_occurrence)

    if min(segment) < 0 or max(segment) >= len(rows):
        raise SystemExit(f"selected code_idx range is outside code.json: {min(segment)}..{max(segment)}")

    if args.include_zero_hit:
        return segment

    executable = [idx for idx in segment if rows[idx].hit > 0]
    skipped = len(segment) - len(executable)
    if skipped:
        print(f"note: skipped {skipped} selected code rows with global hit=0")
    if not executable:
        raise SystemExit("selected segment has no executable rows with global hit > 0")
    return executable


def find_matches(instructions: list[list[int]], target: list[int]) -> Iterable[int]:
    if not target:
        return
    first = target[0]
    n = len(target)
    limit = len(instructions) - n + 1
    for pos in range(max(0, limit)):
        if instructions[pos][4] != first:
            continue
        for offset, code_idx in enumerate(target[1:], start=1):
            if instructions[pos + offset][4] != code_idx:
                break
        else:
            yield pos


def parse_wave_name(path: Path) -> tuple[int | None, int | None, int | None, int | None]:
    match = re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json$", path.name)
    if not match:
        return None, None, None, None
    return tuple(int(group) for group in match.groups())


def parse_index_filter(value: str | None, name: str) -> set[int] | None:
    if value is None:
        return None
    if value.strip() == "*":
        return None

    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text, 0)
            end = int(end_text, 0)
            if start > end:
                raise SystemExit(f"--{name} range start must be <= end: {item}")
            result.update(range(start, end + 1))
        else:
            result.add(int(item, 0))

    if not result:
        raise SystemExit(f"--{name} did not contain any valid index")
    return result


def matches_index_filter(value: int | None, allowed: set[int] | None) -> bool:
    return allowed is None or value in allowed


def filtered_wave_file_entries(
    trace_dir: Path,
    wave_glob: str,
    se_filter: set[int] | None,
    sm_filter: set[int] | None,
    sl_filter: set[int] | None,
    wv_filter: set[int] | None,
) -> list[tuple[Path, int | None, int | None, int | None, int | None]]:
    wave_files = sorted(trace_dir.glob(wave_glob))
    if not wave_files:
        raise SystemExit(f"no wave files matched {wave_glob!r} under {trace_dir}")

    filtered: list[tuple[Path, int | None, int | None, int | None, int | None]] = []
    for wave_path in wave_files:
        se, simd, slot, wave_id = parse_wave_name(wave_path)
        if not (
            matches_index_filter(se, se_filter)
            and matches_index_filter(simd, sm_filter)
            and matches_index_filter(slot, sl_filter)
            and matches_index_filter(wave_id, wv_filter)
        ):
            continue
        filtered.append((wave_path, se, simd, slot, wave_id))

    if not filtered:
        raise SystemExit("no wave files remained after applying --se/--sm/--sl/--wv filters")
    return filtered


def default_instruction_stats() -> dict[str, object]:
    return {
        "count": 0,
        "latency": 0,
        "stall": 0,
        "max_latency": 0,
        "max_stall": 0,
        "latency_samples": [],
        "stall_samples": [],
    }


def scan_waves(
    trace_dir: Path,
    wave_glob: str,
    se_filter: set[int] | None,
    sm_filter: set[int] | None,
    sl_filter: set[int] | None,
    wv_filter: set[int] | None,
    target: list[int],
    rows: list[CodeRow],
) -> tuple[list[Occurrence], dict[int, dict[str, object]], list[tuple[int, int, str, int, int, int]]]:
    occurrences: list[Occurrence] = []
    aggregate: dict[int, dict[str, object]] = defaultdict(default_instruction_stats)
    top_events: list[tuple[int, int, str, int, int, int]] = []

    for wave_path, se, simd, slot, wave_id in filtered_wave_file_entries(
        trace_dir, wave_glob, se_filter, sm_filter, sl_filter, wv_filter
    ):
        with wave_path.open() as f:
            data = json.load(f)
        wave = data["wave"]
        instructions = wave["instructions"]

        for pos in find_matches(instructions, target):
            events = instructions[pos : pos + len(target)]
            segment_events = tuple(
                SegmentEvent(
                    code_idx=int(event[4]),
                    start_ts=int(event[0]),
                    latency=int(event[3]),
                    stall=int(event[2]),
                )
                for event in events
            )
            cycles_sum = sum(int(event[3]) for event in events)
            cycles_span = int(events[-1][0]) + int(events[-1][3]) - int(events[0][0])
            occurrences.append(
                Occurrence(
                    wave_file=wave_path.name,
                    se=se,
                    simd=simd,
                    slot=slot,
                    wave_id=wave_id,
                    start_pos=pos,
                    start_ts=int(events[0][0]),
                    cycles_sum=cycles_sum,
                    cycles_span=cycles_span,
                    events=segment_events,
                )
            )

            for event_offset, event in enumerate(segment_events):
                entry = aggregate[event.code_idx]
                entry["count"] += 1
                entry["latency"] += event.latency
                entry["stall"] += event.stall
                entry["max_latency"] = max(entry["max_latency"], event.latency)
                entry["max_stall"] = max(entry["max_stall"], event.stall)
                entry["latency_samples"].append(event.latency)
                entry["stall_samples"].append(event.stall)
                top_events.append(
                    (
                        event.latency,
                        event.stall,
                        wave_path.name,
                        pos + event_offset,
                        event.code_idx,
                        event.start_ts,
                    )
                )

    return occurrences, aggregate, top_events


def exact_instruction_sequences(
    rows: list[CodeRow],
    instruction_text: str,
    separator: str | None,
    label: str,
) -> list[set[int]]:
    targets = normalized_instruction_lines(split_instruction_text(instruction_text, separator))
    return exact_instruction_sequences_from_lines(rows, targets, label)


def exact_instruction_sequences_from_lines(
    rows: list[CodeRow],
    targets: list[str],
    label: str,
) -> list[set[int]]:
    if not targets:
        raise SystemExit(f"{label} instruction sequence is empty")

    normalized_code = [normalize_inst(row.isa) for row in rows]
    candidates: list[set[int]] = []
    for inst in targets:
        matches = {idx for idx, code_inst in enumerate(normalized_code) if code_inst == inst}
        if not matches:
            raise SystemExit(f"{label} instruction did not match any code.json row: {inst!r}")
        candidates.append(matches)
    return candidates


def match_any_sequence(
    instructions: list[list[int]],
    pos: int,
    sequence_candidates: list[set[int]],
) -> tuple[int, ...] | None:
    if pos + len(sequence_candidates) > len(instructions):
        return None

    matched: list[int] = []
    for offset, candidates in enumerate(sequence_candidates):
        code_idx = int(instructions[pos + offset][4])
        if code_idx not in candidates:
            return None
        matched.append(code_idx)
    return tuple(matched)


def scan_intervals(
    trace_dir: Path,
    wave_glob: str,
    se_filter: set[int] | None,
    sm_filter: set[int] | None,
    sl_filter: set[int] | None,
    wv_filter: set[int] | None,
    start_code_sequences: list[set[int]],
    end_code_sequences: list[set[int]],
) -> list[IntervalResult]:
    intervals: list[IntervalResult] = []

    for wave_path, se, simd, slot, wave_id in filtered_wave_file_entries(
        trace_dir, wave_glob, se_filter, sm_filter, sl_filter, wv_filter
    ):
        with wave_path.open() as f:
            data = json.load(f)
        instructions = data["wave"]["instructions"]

        open_start: tuple[int, list[int], tuple[int, ...]] | None = None
        pos = 0
        while pos < len(instructions):
            event = instructions[pos]
            if open_start is None:
                start_seq = match_any_sequence(instructions, pos, start_code_sequences)
                if start_seq is not None:
                    open_start = (pos, event, start_seq)
                    pos += len(start_seq)
                    continue
                pos += 1
                continue

            end_seq = match_any_sequence(instructions, pos, end_code_sequences)
            if end_seq is not None:
                start_pos, start_event, start_seq = open_start
                start_ts = int(start_event[0])
                end_ts = int(event[0])
                interval_events = tuple(
                    SegmentEvent(
                        code_idx=int(interval_event[4]),
                        start_ts=int(interval_event[0]),
                        latency=int(interval_event[3]),
                        stall=int(interval_event[2]),
                    )
                    for interval_event in instructions[start_pos : pos + len(end_seq)]
                )
                intervals.append(
                    IntervalResult(
                        wave_file=wave_path.name,
                        se=se,
                        simd=simd,
                        slot=slot,
                        wave_id=wave_id,
                        start_pos=start_pos,
                        end_pos=pos,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        cycles=end_ts - start_ts,
                        inst_count=pos + len(end_seq) - start_pos,
                        start_code_indices=start_seq,
                        end_code_indices=end_seq,
                        events=interval_events,
                    )
                )
                open_start = None
                pos += len(end_seq)
                continue

            pos += 1

    return intervals


def format_optional(value: int | None) -> str:
    return "-" if value is None else str(value)


def format_code_sequence(seq: tuple[int, ...]) -> str:
    if not seq:
        return "-"
    if len(seq) == 1:
        return str(seq[0])
    return f"{seq[0]}..{seq[-1]}"


def format_code_candidates(candidates: set[int]) -> str:
    ordered = sorted(candidates)
    if len(ordered) == 1:
        return str(ordered[0])
    return "[" + ",".join(str(item) for item in ordered) + "]"


def print_code_sequence(rows: list[CodeRow], seq: tuple[int, ...]) -> None:
    print(f"  code_idx={format_code_sequence(seq)}")
    for code_idx in seq:
        row = rows[code_idx]
        print(f"    line={row.line} vaddr=0x{row.vaddr:x} {row.isa}")


def print_matched_code_sequences(
    rows: list[CodeRow],
    start_sequences: list[tuple[int, ...]],
    end_sequences: list[tuple[int, ...]],
) -> None:
    print("Matched start code sequences:")
    for seq in sorted(set(start_sequences)):
        print_code_sequence(rows, seq)
    print("Matched end code sequences:")
    for seq in sorted(set(end_sequences)):
        print_code_sequence(rows, seq)


def print_segment(rows: list[CodeRow], target: list[int], limit: int) -> None:
    print("\nSelected executable segment:")
    shown = target[:limit]
    for idx in shown:
        row = rows[idx]
        print(f"  idx={row.idx:5d} line={row.line:5d} vaddr=0x{row.vaddr:x} hit={row.hit:6d}  {row.isa}")
    if len(target) > len(shown):
        print(f"  ... {len(target) - len(shown)} more instructions")


def print_occurrences(occurrences: list[Occurrence], limit: int) -> None:
    print("\nMatched occurrences:")
    for occ in occurrences[:limit]:
        print(
            "  "
            f"{occ.wave_file} se={format_optional(occ.se)} simd={format_optional(occ.simd)} "
            f"slot={format_optional(occ.slot)} wave={format_optional(occ.wave_id)} "
            f"pos={occ.start_pos} ts={occ.start_ts} "
            f"cycles={occ.cycles_sum} span={occ.cycles_span}"
        )
    if len(occurrences) > limit:
        print(f"  ... {len(occurrences) - limit} more occurrences")


def print_interval_results(rows: list[CodeRow], intervals: list[IntervalResult], limit: int) -> None:
    sorted_intervals = sorted(intervals, key=lambda interval: interval.cycles, reverse=True)
    shown_limit = len(sorted_intervals) if limit == 0 else min(limit, len(sorted_intervals))
    print(f"\n[start, end) intervals ({len(sorted_intervals)} total):")
    wave_width = max(len("wave_file"), *(len(interval.wave_file) for interval in sorted_intervals[:shown_limit]))
    header = (
        f"{'idx':>4s} {'wave_file':<{wave_width}s} {'se':>2s} {'sm':>2s} {'sl':>2s} {'wv':>2s} "
        f"{'start_pos':>9s} {'end_pos':>7s} {'start_ts':>8s} {'end_ts':>8s} "
        f"{'cycles':>6s} {'inst_count':>10s} {'start_code_range':>16s} {'end_code_range':>14s}"
    )
    print(header)
    for idx, interval in enumerate(sorted_intervals[:shown_limit], start=1):
        start_range = format_code_sequence(interval.start_code_indices)
        end_range = format_code_sequence(interval.end_code_indices)
        print(
            f"{idx:4d} {interval.wave_file:<{wave_width}s} "
            f"{format_optional(interval.se):>2s} {format_optional(interval.simd):>2s} "
            f"{format_optional(interval.slot):>2s} {format_optional(interval.wave_id):>2s} "
            f"{interval.start_pos:9d} {interval.end_pos:7d} "
            f"{interval.start_ts:8d} {interval.end_ts:8d} {interval.cycles:6d} "
            f"{interval.inst_count:10d} {start_range:>16s} {end_range:>14s}"
        )
    if len(sorted_intervals) > shown_limit:
        print(f"... skipped {len(sorted_intervals) - shown_limit} intervals")

    cycles = [float(interval.cycles) for interval in sorted_intervals]
    inst_counts = [float(interval.inst_count) for interval in sorted_intervals]
    print(
        "\nInterval cycles summary: "
        f"count={len(cycles)} "
        f"avg_cycles={statistics.mean(cycles):.1f} "
        f"p50_cycles={percentile(cycles, 0.50):.1f} "
        f"p90_cycles={percentile(cycles, 0.90):.1f} "
        f"min_cycles={min(cycles):.0f} max_cycles={max(cycles):.0f} "
        f"avg_inst_count={statistics.mean(inst_counts):.1f} "
        f"min_inst_count={min(inst_counts):.0f} max_inst_count={max(inst_counts):.0f}"
    )

def print_interval_cycles_summary(label: str, intervals: list[IntervalResult]) -> None:
    if not intervals:
        print(f"\n{label}: count=0")
        return
    cycles = [float(interval.cycles) for interval in intervals]
    inst_counts = [float(interval.inst_count) for interval in intervals]
    print(
        f"\n{label}: "
        f"count={len(cycles)} "
        f"avg_cycles={statistics.mean(cycles):.1f} "
        f"avg_inst_count={statistics.mean(inst_counts):.1f} "
        f"p50_cycles={percentile(cycles, 0.50):.1f} "
        f"p90_cycles={percentile(cycles, 0.90):.1f} "
        f"min_cycles={min(cycles):.0f} max_cycles={max(cycles):.0f}"
    )


def opcode_for_isa(isa: str) -> str:
    return isa.split(None, 1)[0] if isa.strip() else ""


def interval_matches_inc_pattern(rows: list[CodeRow], interval: IntervalResult, pattern: tuple[str, ...]) -> bool:
    if not pattern or len(pattern) > len(interval.events):
        return False
    opcodes = [opcode_for_isa(rows[event.code_idx].isa) for event in interval.events]
    for start in range(0, len(opcodes) - len(pattern) + 1):
        if opcodes[start : start + len(pattern)] == list(pattern):
            return True
    return False


def split_intervals_by_pattern(
    rows: list[CodeRow],
    intervals: list[IntervalResult],
    pattern: tuple[str, ...] | None,
) -> tuple[list[IntervalResult], list[IntervalResult]]:
    if pattern is None:
        return [], []
    matched = [interval for interval in intervals if interval_matches_inc_pattern(rows, interval, pattern)]
    unmatched = [interval for interval in intervals if interval not in matched]
    return matched, unmatched


def format_pattern(pattern: tuple[str, ...] | None) -> str:
    return "[" + ", ".join(pattern) + "]" if pattern else "-"


def print_instruction_sequence(label: str, lines: list[str]) -> None:
    print(label)
    for line in lines:
        print(f"  {line}")


def interval_summary_row(
    idx_label: str,
    description: str | None,
    intervals: list[IntervalResult],
    rows: list[CodeRow],
    inc_pattern: tuple[str, ...] | None,
) -> str:
    pattern_text = format_pattern(inc_pattern)
    matched, unmatched = split_intervals_by_pattern(rows, intervals, inc_pattern)
    inc_avg = f"{statistics.mean([float(interval.cycles) for interval in matched]):.1f}" if matched else "-"
    no_inc_avg = f"{statistics.mean([float(interval.cycles) for interval in unmatched]):.1f}" if unmatched else "-"

    if not intervals:
        return (
            f"{idx_label:>4s} {description or '-':<28.28s} {0:5d} {'-':>8s} {'-':>8s} "
            f"{'-':>8s} {'-':>8s} {'-':>8s} {0:5d} {0:6d} {'-':>8s} {'-':>10s} {pattern_text:<32.32s}"
        )

    cycles = [float(interval.cycles) for interval in intervals]
    return (
        f"{idx_label:>4s} {description or '-':<28.28s} {len(cycles):5d} "
        f"{statistics.mean(cycles):8.1f} {percentile(cycles, 0.50):8.1f} "
        f"{percentile(cycles, 0.90):8.1f} {min(cycles):8.0f} {max(cycles):8.0f} "
        f"{len(matched):5d} {len(unmatched):6d} {inc_avg:>8s} {no_inc_avg:>10s} {pattern_text:<32.32s}"
    )


def interval_summary_sort_key(intervals: list[IntervalResult]) -> float:
    if not intervals:
        return float("-inf")
    return statistics.mean([float(interval.cycles) for interval in intervals])


def interval_summary_record(
    idx_label: str,
    description: str | None,
    intervals: list[IntervalResult],
    rows: list[CodeRow],
    inc_pattern: tuple[str, ...] | None,
) -> tuple[float, str]:
    pattern_text = format_pattern(inc_pattern)
    matched, unmatched = split_intervals_by_pattern(rows, intervals, inc_pattern)

    if not intervals:
        line = (
            f"{idx_label:>4s} {description or '-':<28.28s} {0:5d} {'-':>10s} {'-':>8s} "
            f"{'-':>8s} {'-':>8s} {'-':>8s} {'-':>14s} {'-':>14s} {'-':>14s} "
            f"{0:5d} {0:6d} {pattern_text:<32.32s}"
        )
        return float("-inf"), line

    cycles = [float(interval.cycles) for interval in intervals]
    inst_counts = [float(interval.inst_count) for interval in intervals]
    avg_cycles = statistics.mean(cycles)
    line = (
        f"{idx_label:>4s} {description or '-':<28.28s} {len(cycles):5d} "
        f"{avg_cycles:10.1f} {percentile(cycles, 0.50):8.1f} "
        f"{percentile(cycles, 0.90):8.1f} {min(cycles):8.0f} {max(cycles):8.0f} "
        f"{statistics.mean(inst_counts):14.1f} {min(inst_counts):14.0f} {max(inst_counts):14.0f} "
        f"{len(matched):5d} {len(unmatched):6d} {pattern_text:<32.32s}"
    )
    return avg_cycles, line


def sorted_intervals_for_output(intervals: list[IntervalResult]) -> list[IntervalResult]:
    return sorted(intervals, key=lambda interval: interval.cycles, reverse=True)


def interval_metric_summary(intervals: list[IntervalResult]) -> dict[str, float]:
    if not intervals:
        return {"count": 0.0, "avg_cycles": math.nan, "avg_inst_count": math.nan}
    return {
        "count": float(len(intervals)),
        "avg_cycles": statistics.mean([float(interval.cycles) for interval in intervals]),
        "avg_inst_count": statistics.mean([float(interval.inst_count) for interval in intervals]),
    }


def pct_delta(delta: float, baseline_value: float) -> float:
    if baseline_value == 0 or math.isnan(baseline_value):
        return math.nan
    return delta / baseline_value * 100.0


def format_pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.1f}%"


def build_summary_records(
    rows: list[CodeRow],
    interval_summaries: list[tuple[int, IntervalInfo | None, list[IntervalResult]]],
) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for idx, interval_info, intervals in interval_summaries:
        description = interval_info.description if interval_info is not None else None
        inc_pattern = interval_info.inc_inst_pattern if interval_info is not None else None
        matched, unmatched = split_intervals_by_pattern(rows, intervals, inc_pattern)
        primary_intervals = unmatched if inc_pattern and matched else intervals
        primary_pattern = None if inc_pattern and matched else inc_pattern
        records.append(SummaryRecord(str(idx + 1), description, primary_intervals, primary_pattern))
        if inc_pattern and matched:
            records.append(
                SummaryRecord(
                    f"{idx + 1}i",
                    f"{description or '-'} [inc_inst_pattern matched]",
                    matched,
                    inc_pattern,
                )
            )
    return records


def sorted_summary_records(rows: list[CodeRow], records: list[SummaryRecord]) -> list[SummaryRecord]:
    return sorted(
        records,
        key=lambda record: interval_summary_sort_key(record.intervals),
        reverse=True,
    )


def analyze_kernel_config(
    name: str,
    baseline: bool,
    config: dict[str, object],
    config_path: Path,
) -> KernelAnalysis:
    raw_trace_dir = config_value(config, "trace_dir", "trace-dir", "trace dir", "trace")
    if raw_trace_dir is None:
        raise SystemExit(f"{config_path} kernel {name} missing trace_dir")
    trace_dir = Path(str(raw_trace_dir)).resolve()
    if not (trace_dir / "code.json").is_file():
        raise SystemExit(f"missing code.json under {trace_dir}")

    rows = load_code(trace_dir)
    points = parse_config_sample_points(config, config_path)
    interval_infos = parse_config_interval_infos(config, points, config_path)
    if interval_infos is None:
        interval_infos = [IntervalInfo() for _ in range(len(points) - 1)]

    se_filter = parse_index_filter(optional_str(config_value(config, "se")), "se")
    sm_filter = parse_index_filter(optional_str(config_value(config, "sm")), "sm")
    sl_filter = parse_index_filter(optional_str(config_value(config, "sl")), "sl")
    wv_filter = parse_index_filter(optional_str(config_value(config, "wv")), "wv")
    wave_glob = str(config_value(config, "wave_glob", "wave-glob") or "se*_sm*_sl*_wv*.json")

    sample_intervals: list[SampleIntervalAnalysis] = []
    interval_summaries: list[tuple[int, IntervalInfo | None, list[IntervalResult]]] = []
    all_intervals: list[IntervalResult] = []
    for idx in range(len(points) - 1):
        interval_info = interval_infos[idx]
        inc_pattern = interval_info.inc_inst_pattern
        start_code_sequences = exact_instruction_sequences_from_lines(rows, points[idx], f"{name} sample point {idx}")
        end_code_sequences = exact_instruction_sequences_from_lines(rows, points[idx + 1], f"{name} sample point {idx + 1}")
        intervals = scan_intervals(
            trace_dir,
            wave_glob,
            se_filter,
            sm_filter,
            sl_filter,
            wv_filter,
            start_code_sequences,
            end_code_sequences,
        )
        matched, unmatched = split_intervals_by_pattern(rows, intervals, inc_pattern)
        primary_intervals = unmatched if inc_pattern and matched else intervals
        all_intervals.extend(primary_intervals)
        all_intervals.extend(matched)
        sample_intervals.append(
            SampleIntervalAnalysis(
                idx=idx,
                description=interval_info.description,
                inc_pattern=inc_pattern,
                start_code_sequences=start_code_sequences,
                end_code_sequences=end_code_sequences,
                intervals=intervals,
                primary_intervals=primary_intervals,
                inc_matched_intervals=matched,
                inc_unmatched_intervals=unmatched,
            )
        )
        interval_summaries.append((idx, interval_info, intervals))

    return KernelAnalysis(
        name=name,
        baseline=baseline,
        rows=rows,
        trace_dir=trace_dir,
        sample_intervals=sample_intervals,
        summary_records=build_summary_records(rows, interval_summaries),
        all_intervals=all_intervals,
    )


def parse_kernel_versions(config: dict[str, object], config_path: Path) -> list[KernelAnalysis]:
    raw_versions = config.get("kernel versions") or config.get("kernel_versions")
    if not isinstance(raw_versions, list) or len(raw_versions) < 2:
        raise SystemExit(f"{config_path} compare config must contain at least two kernel versions")

    analyses: list[KernelAnalysis] = []
    for version_idx, raw_version in enumerate(raw_versions):
        if not isinstance(raw_version, dict):
            raise SystemExit(f"{config_path} kernel version {version_idx} must be an object")
        raw_config = raw_version.get("config")
        if not isinstance(raw_config, dict):
            raise SystemExit(f"{config_path} kernel version {version_idx} missing config object")
        name = str(raw_version.get("value") or raw_version.get("name") or f"version_{version_idx}")
        baseline = bool(raw_version.get("baseline", False))
        analyses.append(analyze_kernel_config(name, baseline, raw_config, config_path))

    baselines = [analysis for analysis in analyses if analysis.baseline]
    if len(baselines) != 1:
        raise SystemExit(f"{config_path} compare config must mark exactly one kernel version as baseline")
    return analyses


def print_compare_metric_row(
    rank: int,
    baseline_interval: IntervalResult,
    other_interval: IntervalResult,
    baseline_name: str,
    other_name: str,
) -> None:
    delta_cycles = other_interval.cycles - baseline_interval.cycles
    delta_inst = other_interval.inst_count - baseline_interval.inst_count
    delta_cycles_pct = pct_delta(float(delta_cycles), float(baseline_interval.cycles))
    delta_inst_pct = pct_delta(float(delta_inst), float(baseline_interval.inst_count))
    print(
        f"{rank:4d} "
        f"{baseline_interval.wave_file:<24.24s} {other_interval.wave_file:<24.24s} "
        f"{baseline_interval.cycles:10d} {other_interval.cycles:10d} {delta_cycles:12d} {format_pct(delta_cycles_pct):>12s} "
        f"{baseline_interval.inst_count:14d} {other_interval.inst_count:14d} {delta_inst:15d} {format_pct(delta_inst_pct):>12s}"
    )


def print_interval_comparison_table(
    title: str,
    baseline: KernelAnalysis,
    other: KernelAnalysis,
    baseline_intervals: list[IntervalResult],
    other_intervals: list[IntervalResult],
    interval_print_limit: int,
) -> None:
    if not baseline_intervals or not other_intervals:
        print(f"\n{title}: skipped, one side has no matched intervals")
        return
    baseline_sorted = sorted_intervals_for_output(baseline_intervals)
    other_sorted = sorted_intervals_for_output(other_intervals)
    count_all = min(len(baseline_sorted), len(other_sorted))
    count = count_all if interval_print_limit == 0 else min(interval_print_limit, count_all)
    print(f"\n{title}: comparing {count_all} rows by baseline order")
    if len(baseline_sorted) != len(other_sorted):
        print(
            f"note: row count differs, baseline={len(baseline_sorted)} "
            f"{other.name}={len(other_sorted)}; comparing first {count_all}"
        )
    if count < count_all:
        print(f"note: printing first {count} rows due to interval-print-limit")
    print(
        f"{'rank':>4s} "
        f"{baseline.name + '_wave':<24.24s} {other.name + '_wave':<24.24s} "
        f"{baseline.name + '_cycles':>10.10s} {other.name + '_cycles':>10.10s} {'delta_cycles':>12s} {'delta_cycles_pct':>12s} "
        f"{baseline.name + '_inst':>14.14s} {other.name + '_inst':>14.14s} {'delta_inst':>15s} {'delta_inst_pct':>12s}"
    )
    for rank in range(count):
        print_compare_metric_row(
            rank + 1,
            baseline_sorted[rank],
            other_sorted[rank],
            baseline.name,
            other.name,
        )


def summary_compare_key(record: SummaryRecord) -> tuple[str, str | None]:
    return record.idx_label, record.description


def print_summary_compare_row(
    baseline_record: SummaryRecord,
    other_record: SummaryRecord,
    baseline_name: str,
    other_name: str,
) -> None:
    base_metrics = interval_metric_summary(baseline_record.intervals)
    other_metrics = interval_metric_summary(other_record.intervals)
    delta_avg_cycles = other_metrics["avg_cycles"] - base_metrics["avg_cycles"]
    delta_avg_inst = other_metrics["avg_inst_count"] - base_metrics["avg_inst_count"]
    print(
        f"{baseline_record.idx_label:>4s} {baseline_record.description or '-':<36.36s} "
        f"{int(base_metrics['count']):8d} {int(other_metrics['count']):8d} "
        f"{int(other_metrics['count'] - base_metrics['count']):8d} "
        f"{base_metrics['avg_cycles']:12.1f} {other_metrics['avg_cycles']:12.1f} "
        f"{delta_avg_cycles:13.1f} {format_pct(pct_delta(delta_avg_cycles, base_metrics['avg_cycles'])):>17s} "
        f"{base_metrics['avg_inst_count']:16.1f} {other_metrics['avg_inst_count']:16.1f} "
        f"{delta_avg_inst:17.1f} {format_pct(pct_delta(delta_avg_inst, base_metrics['avg_inst_count'])):>16s}"
    )


def print_all_summary_compare(
    baseline: KernelAnalysis,
    other: KernelAnalysis,
    summary_pairs: list[tuple[SummaryRecord, SummaryRecord]],
) -> None:
    base_sum_cycles = 0.0
    other_sum_cycles = 0.0
    base_sum_inst = 0.0
    other_sum_inst = 0.0
    for base_record, other_record in summary_pairs:
        base_metrics = interval_metric_summary(base_record.intervals)
        other_metrics = interval_metric_summary(other_record.intervals)
        base_count = base_metrics["count"]
        base_sum_cycles += base_count * base_metrics["avg_cycles"]
        other_sum_cycles += base_count * other_metrics["avg_cycles"]
        base_sum_inst += base_count * base_metrics["avg_inst_count"]
        other_sum_inst += base_count * other_metrics["avg_inst_count"]

    delta_sum_cycles = other_sum_cycles - base_sum_cycles
    delta_sum_inst = other_sum_inst - base_sum_inst
    print("\n=== All Sample Interval Cycles Summary Compare ===")
    print(
        f"{'baseline':<16s} {'other':<16s} "
        f"{'base_sum_cycles':>16s} {'other_sum_cycles':>17s} {'delta_sum_cycles':>17s} {'delta_sum_cycles_pct':>21s} "
        f"{'base_sum_inst':>14s} {'other_sum_inst':>15s} {'delta_sum_inst':>15s} {'delta_sum_inst_pct':>19s}"
    )
    print(
        f"{baseline.name:<16.16s} {other.name:<16.16s} "
        f"{base_sum_cycles:16.1f} {other_sum_cycles:17.1f} "
        f"{delta_sum_cycles:17.1f} {format_pct(pct_delta(delta_sum_cycles, base_sum_cycles)):>21s} "
        f"{base_sum_inst:14.1f} {other_sum_inst:15.1f} "
        f"{delta_sum_inst:15.1f} {format_pct(pct_delta(delta_sum_inst, base_sum_inst)):>19s}"
    )


def run_compare_config(config: dict[str, object], config_path: Path, interval_print_limit: int) -> None:
    analyses = parse_kernel_versions(config, config_path)
    baseline = next(analysis for analysis in analyses if analysis.baseline)
    others = [analysis for analysis in analyses if not analysis.baseline]

    print(f"Compare config: {config_path}")
    print(f"Baseline kernel: {baseline.name}")
    print(f"Interval print limit: {interval_print_limit}")
    print("Kernel versions:")
    for analysis in analyses:
        print(f"  {analysis.name}: trace_dir={analysis.trace_dir} baseline={analysis.baseline}")

    for other in others:
        print(f"\n================ Compare {baseline.name} (baseline) vs {other.name} ================")
        sample_count = min(len(baseline.sample_intervals), len(other.sample_intervals))
        for idx in range(sample_count):
            base_sample = baseline.sample_intervals[idx]
            other_sample = other.sample_intervals[idx]
            print(f"\n=== Compare Sample interval {idx + 1}/{sample_count}: point {idx} -> point {idx + 1} ===")
            print(f"Baseline description: {base_sample.description or '-'}")
            print(f"Other description:    {other_sample.description or '-'}")

            print(f"\n--- {baseline.name} matched code information ---")
            print(f"Start code_idx candidates by instruction: {[format_code_candidates(c) for c in base_sample.start_code_sequences]}")
            print(f"End code_idx candidates by instruction:   {[format_code_candidates(c) for c in base_sample.end_code_sequences]}")
            if base_sample.intervals:
                print_matched_code_sequences(
                    baseline.rows,
                    [interval.start_code_indices for interval in base_sample.intervals],
                    [interval.end_code_indices for interval in base_sample.intervals],
                )
            print(f"Matched intervals: {len(base_sample.primary_intervals)}")

            print(f"\n--- {other.name} matched code information ---")
            print(f"Start code_idx candidates by instruction: {[format_code_candidates(c) for c in other_sample.start_code_sequences]}")
            print(f"End code_idx candidates by instruction:   {[format_code_candidates(c) for c in other_sample.end_code_sequences]}")
            if other_sample.intervals:
                print_matched_code_sequences(
                    other.rows,
                    [interval.start_code_indices for interval in other_sample.intervals],
                    [interval.end_code_indices for interval in other_sample.intervals],
                )
            print(f"Matched intervals: {len(other_sample.primary_intervals)}")

            print_interval_comparison_table(
                "Primary interval comparison",
                baseline,
                other,
                base_sample.primary_intervals,
                other_sample.primary_intervals,
                interval_print_limit,
            )

            if base_sample.inc_matched_intervals and other_sample.inc_matched_intervals:
                print_interval_comparison_table(
                    "inc_inst_pattern matched interval comparison",
                    baseline,
                    other,
                    base_sample.inc_matched_intervals,
                    other_sample.inc_matched_intervals,
                    interval_print_limit,
                )

        print("\n=== Sample Interval Summary Compare ===")
        print(
            f"{'idx':>4s} {'description':<36s} "
            f"{baseline.name + '_count':>8.8s} {other.name + '_count':>8.8s} {'d_count':>8s} "
            f"{baseline.name + '_avg_cycles':>12.12s} {other.name + '_avg_cycles':>12.12s} {'d_avg_cycles':>13s} {'d_avg_cycles_pct':>17s} "
            f"{baseline.name + '_avg_inst':>16.16s} {other.name + '_avg_inst':>16.16s} {'d_avg_inst':>17s} {'d_avg_inst_pct':>16s}"
        )
        other_records = {summary_compare_key(record): record for record in other.summary_records}
        summary_pairs: list[tuple[SummaryRecord, SummaryRecord]] = []
        for base_record in sorted_summary_records(baseline.rows, baseline.summary_records):
            other_record = other_records.get(summary_compare_key(base_record))
            if other_record is None:
                continue
            summary_pairs.append((base_record, other_record))
            print_summary_compare_row(base_record, other_record, baseline.name, other.name)

        print_all_summary_compare(baseline, other, summary_pairs)


def run_interval_points_file(
    rows: list[CodeRow],
    trace_dir: Path,
    wave_glob: str,
    se_filter: set[int] | None,
    sm_filter: set[int] | None,
    sl_filter: set[int] | None,
    wv_filter: set[int] | None,
    points_file: Path,
    interval_print_limit: int,
) -> None:
    points, interval_infos = load_interval_points_file(points_file)
    print(f"Interval points file: {points_file}")
    run_interval_points_data(
        rows,
        trace_dir,
        wave_glob,
        se_filter,
        sm_filter,
        sl_filter,
        wv_filter,
        points,
        interval_infos,
        interval_print_limit,
    )


def run_interval_points_data(
    rows: list[CodeRow],
    trace_dir: Path,
    wave_glob: str,
    se_filter: set[int] | None,
    sm_filter: set[int] | None,
    sl_filter: set[int] | None,
    wv_filter: set[int] | None,
    points: list[list[str]],
    interval_infos: list[IntervalInfo] | None,
    interval_print_limit: int,
) -> None:
    print(f"Sample points: {len(points)}")
    if interval_infos is not None:
        print(f"Interval descriptions: {len(interval_infos)}")

    all_intervals: list[IntervalResult] = []
    interval_summaries: list[tuple[int, IntervalInfo | None, list[IntervalResult]]] = []
    for idx in range(len(points) - 1):
        start_lines = points[idx]
        end_lines = points[idx + 1]
        interval_info = interval_infos[idx] if interval_infos is not None else None
        description = interval_info.description if interval_info is not None else None
        inc_pattern = interval_info.inc_inst_pattern if interval_info is not None else None
        start_code_sequences = exact_instruction_sequences_from_lines(
            rows, start_lines, f"sample point {idx}"
        )
        end_code_sequences = exact_instruction_sequences_from_lines(
            rows, end_lines, f"sample point {idx + 1}"
        )
        intervals = scan_intervals(
            trace_dir,
            wave_glob,
            se_filter,
            sm_filter,
            sl_filter,
            wv_filter,
            start_code_sequences,
            end_code_sequences,
        )
        matched_for_pattern, unmatched_for_pattern = split_intervals_by_pattern(rows, intervals, inc_pattern)
        primary_intervals = unmatched_for_pattern if inc_pattern and matched_for_pattern else intervals
        all_intervals.extend(primary_intervals)
        if matched_for_pattern:
            all_intervals.extend(matched_for_pattern)
        interval_summaries.append((idx, interval_info, intervals))

        print(f"\n=== Sample interval {idx + 1}/{len(points) - 1}: point {idx} -> point {idx + 1} ===")
        if description or inc_pattern:
            detail = f"Description: {description or '-'}"
            if inc_pattern:
                detail += (
                    f" | inc_inst_pattern={format_pattern(inc_pattern)} "
                    f"| matched={len(matched_for_pattern)} unmatched={len(unmatched_for_pattern)}"
                )
            print(detail)
        print(f"Start code_idx candidates by instruction: {[format_code_candidates(c) for c in start_code_sequences]}")
        print(f"End code_idx candidates by instruction:   {[format_code_candidates(c) for c in end_code_sequences]}")
        if intervals:
            print_matched_code_sequences(
                rows,
                [interval.start_code_indices for interval in intervals],
                [interval.end_code_indices for interval in intervals],
            )
        else:
            print_instruction_sequence("Interval start instruction sequence:", start_lines)
            print_instruction_sequence("Interval end instruction sequence:", end_lines)
        print(f"Matched intervals: {len(primary_intervals)}" + (f" (excluding {len(matched_for_pattern)} inc_inst_pattern matched)" if matched_for_pattern else ""))
        if primary_intervals:
            print_interval_results(rows, primary_intervals, interval_print_limit)
            if matched_for_pattern:
                print_interval_cycles_summary("inc_inst_pattern unmatched cycles summary", primary_intervals)
        elif intervals:
            print("All matched intervals were moved to inc_inst_pattern matched sample interval.")
        if matched_for_pattern:
            print("\n--- inc_inst_pattern matched intervals (separate sample interval) ---")
            print_interval_results(rows, matched_for_pattern, interval_print_limit)
            print_interval_cycles_summary("inc_inst_pattern matched cycles summary", matched_for_pattern)
        if not intervals:
            print("No [start, end) intervals matched for this sample interval.")

    if interval_infos is not None:
        print("\n=== Sample Interval Summary With Descriptions ===")
        header = (
            f"{'idx':>4s} {'description':<28s} {'count':>5s} {'avg_cycles':>10s} "
            f"{'p50_cycles':>8s} {'p90_cycles':>8s} {'min_cycles':>8s} {'max_cycles':>8s} "
            f"{'avg_inst_count':>14s} {'min_inst_count':>14s} {'max_inst_count':>14s} "
            f"{'inc':>5s} {'no_inc':>6s} {'inc_pattern':<32s}"
        )
        print(header)
        summary_records: list[tuple[float, int, str, str]] = []
        for idx, interval_info, intervals in interval_summaries:
            description = interval_info.description if interval_info is not None else None
            inc_pattern = interval_info.inc_inst_pattern if interval_info is not None else None
            matched, unmatched = split_intervals_by_pattern(rows, intervals, inc_pattern)
            primary_intervals = unmatched if inc_pattern and matched else intervals
            primary_pattern = None if inc_pattern and matched else inc_pattern
            avg_cycles, line = interval_summary_record(
                str(idx + 1), description, primary_intervals, rows, primary_pattern
            )
            summary_records.append((avg_cycles, idx * 2, str(idx + 1), line))
            if inc_pattern and matched:
                matched_description = f"{description or '-'} [inc_inst_pattern matched]"
                avg_cycles, line = interval_summary_record(
                    f"{idx + 1}i", matched_description, matched, rows, inc_pattern
                )
                summary_records.append((avg_cycles, idx * 2 + 1, f"{idx + 1}i", line))
        for _avg_cycles, _tie_breaker, _idx_label, line in sorted(
            summary_records, key=lambda item: (item[0], -item[1]), reverse=True
        ):
            print(line)

def ranked_aggregate_items(
    aggregate: dict[int, dict[str, object]],
    rank_by: str,
) -> list[tuple[int, dict[str, object]]]:
    def sort_key(item: tuple[int, dict[str, object]]) -> float:
        _idx, stats = item
        count = max(1.0, stats["count"])
        if rank_by == "avg":
            return stats["latency"] / count
        if rank_by == "max":
            return stats["max_latency"]
        if rank_by == "sum":
            return stats["latency"]
        raise AssertionError(rank_by)

    return sorted(aggregate.items(), key=sort_key, reverse=True)


def print_aggregate_topk(rows: list[CodeRow], aggregate: dict[int, dict[str, object]], k: int, rank_by: str) -> None:
    ranked = ranked_aggregate_items(aggregate, rank_by)
    print(f"\nMean top {min(k, len(ranked))} instructions by {rank_by} latency:")
    print("rank code_idx line vaddr avg_lat max_lat avg_stall max_stall instruction")
    for rank, (code_idx, stats) in enumerate(ranked[:k], start=1):
        row = rows[code_idx]
        count = max(1.0, stats["count"])
        print(
            f"{rank:4d} {code_idx:8d} {row.line:4d} 0x{row.vaddr:x} "
            f"{stats['latency'] / count:7.1f} "
            f"{int(stats['max_latency']):7d} {stats['stall'] / count:9.1f} "
            f"{int(stats['max_stall']):9d} {row.isa}"
        )


def print_occurrence_detail_tables(
    rows: list[CodeRow],
    occurrences: list[Occurrence],
    k: int,
    occurrence_limit: int,
) -> None:
    limit = len(occurrences) if occurrence_limit == 0 else min(occurrence_limit, len(occurrences))
    print(f"\nPer-occurrence top {k} instructions by latency:")
    for idx, occ in enumerate(occurrences[:limit], start=1):
        ranked = sorted(occ.events, key=lambda event: event.latency, reverse=True)
        print(
            f"\nOccurrence {idx}/{len(occurrences)}: "
            f"{occ.wave_file} se={format_optional(occ.se)} simd={format_optional(occ.simd)} "
            f"slot={format_optional(occ.slot)} wave={format_optional(occ.wave_id)} "
            f"pos={occ.start_pos} start_ts={occ.start_ts} cycles={occ.cycles_sum} span={occ.cycles_span}"
        )
        print("rank code_idx line vaddr start_ts latency stall instruction")
        for rank, event in enumerate(ranked[:k], start=1):
            row = rows[event.code_idx]
            print(
                f"{rank:4d} {event.code_idx:8d} {row.line:4d} 0x{row.vaddr:x} "
                f"{event.start_ts:8d} {event.latency:7d} {event.stall:5d} {row.isa}"
            )
    if len(occurrences) > limit:
        print(f"\n... skipped {len(occurrences) - limit} occurrence detail tables")


def print_cycle_samples(
    rows: list[CodeRow],
    aggregate: dict[int, dict[str, object]],
    k: int,
    rank_by: str,
    sample_limit: int,
) -> None:
    ranked = ranked_aggregate_items(aggregate, rank_by)
    print(f"\nPer-instruction latency samples for top {min(k, len(ranked))} instructions:")
    for rank, (code_idx, stats) in enumerate(ranked[:k], start=1):
        row = rows[code_idx]
        samples = list(stats["latency_samples"])
        shown_samples = samples if sample_limit == 0 else samples[:sample_limit]
        suffix = "" if sample_limit == 0 or len(samples) <= sample_limit else f" ... ({len(samples) - sample_limit} more)"
        mean_latency = statistics.mean(samples) if samples else 0.0
        print(
            f"{rank:4d} code_idx={code_idx} line={row.line} count={len(samples)} "
            f"cycles={shown_samples}{suffix} mean={mean_latency:.1f}  {row.isa}"
        )


def print_event_topk(rows: list[CodeRow], top_events: list[tuple[int, int, str, int, int, int]], k: int) -> None:
    ranked = sorted(top_events, reverse=True)
    print(f"\nTop {min(k, len(ranked))} individual instruction events by latency:")
    print("rank latency stall wave_file pos ts code_idx line vaddr instruction")
    for rank, (latency, stall, wave_file, pos, code_idx, ts) in enumerate(ranked[:k], start=1):
        row = rows[code_idx]
        print(
            f"{rank:4d} {latency:7d} {stall:5d} {wave_file} {pos:5d} {ts:8d} "
            f"{code_idx:8d} {row.line:4d} 0x{row.vaddr:x} {row.isa}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report cycle cost for a contiguous instruction segment in rocprof UI traces."
    )
    parser.add_argument("trace_dir", nargs="?", type=Path, help="Directory containing code.json and se*_sm*_sl*_wv*.json")
    parser.add_argument(
        "--config-json",
        "--config",
        type=Path,
        help="JSON config file containing trace_dir/sample points/filters/print options.",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of slow instructions to print")
    parser.add_argument("--wave-glob", default="se*_sm*_sl*_wv*.json", help="Wave file glob to scan")
    parser.add_argument("--se", help="Filter SE indices, e.g. '0', '0,2', or '0-3'")
    parser.add_argument("--sm", help="Filter SIMD indices, e.g. '0', '0,2', or '0-3'")
    parser.add_argument("--sl", help="Filter wave slot indices, e.g. '0', '0,1', or '0-1'")
    parser.add_argument("--wv", help="Filter wave instance indices, e.g. '0', '0,2', or '0-3'")
    parser.add_argument(
        "--rank-by",
        choices=("avg", "max", "sum"),
        default="avg",
        help="How to rank aggregate top-k instructions",
    )
    parser.add_argument(
        "--top-events",
        action="store_true",
        help="Also print top-k individual instruction events instead of only aggregate rows",
    )
    parser.add_argument(
        "--hide-occurrence-details",
        "--hide-cycle-samples",
        dest="hide_occurrence_details",
        action="store_true",
        help="Do not print one top-k latency table for each matched occurrence",
    )
    parser.add_argument(
        "--occurrence-detail-limit",
        type=int,
        default=200,
        help="Max occurrence detail tables to print; use 0 for unlimited",
    )
    parser.add_argument("--show-segment", action="store_true", help="Print selected segment instructions")
    parser.add_argument("--segment-print-limit", type=int, default=80, help="Max selected instructions to print")
    parser.add_argument("--show-occurrences", action="store_true", help="Print matched occurrences")
    parser.add_argument("--occurrence-print-limit", type=int, default=80, help="Max occurrences to print")
    parser.add_argument(
        "--interval-print-limit",
        type=int,
        default=0,
        help="Max [start, end) intervals to print in interval mode; use 0 for unlimited",
    )
    parser.add_argument(
        "--include-zero-hit",
        action="store_true",
        help="Keep selected code rows whose global code.json hit count is zero",
    )

    parser.add_argument("--start-idx", type=int, help="Start code.json array index, inclusive")
    parser.add_argument("--end-idx", type=int, help="End code.json array index, inclusive")
    parser.add_argument("--start-line", type=int, help="Start code.json LineNumber, inclusive")
    parser.add_argument("--end-line", type=int, help="End code.json LineNumber, inclusive")
    parser.add_argument("--start-vaddr", help="Start instruction virtual address, inclusive")
    parser.add_argument("--end-vaddr", help="End instruction virtual address, inclusive")
    parser.add_argument("--start-regex", help="Regex matching the first instruction in code.json")
    parser.add_argument("--end-regex", help="Regex matching the last instruction in code.json")
    parser.add_argument(
        "--segment-file",
        help="File containing exact normalized instruction lines to match in code.json",
    )
    parser.add_argument(
        "--segment-text",
        help=(
            "Exact instruction lines as one string. Use real newlines or escaped \\n; "
            "blank lines and # comments are ignored."
        ),
    )
    parser.add_argument(
        "--segment-separator",
        help="Optional separator for --segment-text, for example '||' or ';'. Defaults to newline splitting.",
    )
    parser.add_argument(
        "--code-occurrence",
        type=int,
        default=1,
        help="1-based occurrence to use for regex, segment-file, or segment-text code selection",
    )
    parser.add_argument(
        "--interval-start",
        "--start-inst",
        dest="interval_start",
        help="Exact start instruction sequence for interval mode. Use real newlines or escaped \\n.",
    )
    parser.add_argument(
        "--interval-end",
        "--end-inst",
        dest="interval_end",
        help="Exact end instruction sequence for interval mode. Measures to the first end instruction timestamp.",
    )
    parser.add_argument(
        "--interval-separator",
        help="Optional separator for --interval-start/--interval-end, for example '||'. Defaults to newline splitting.",
    )
    parser.add_argument(
        "--interval-points-file",
        type=Path,
        help=(
            "2D list file of sample-point instruction sequences. Adjacent points are measured as "
            "[point_i, point_i+1) intervals using the same trace-order matching logic."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = args.config_json
    if config_path is None and args.trace_dir is not None and args.trace_dir.suffix == ".json":
        config_path = args.trace_dir
        args.trace_dir = None
    config: dict[str, object] | None = load_json_config(config_path) if config_path is not None else None
    if config is not None and ("kernel versions" in config or "kernel_versions" in config):
        interval_print_limit_raw = (
            args.interval_print_limit
            if args.interval_print_limit != 0
            else config_value(config, "interval-print-limit", "interval_print_limit")
        )
        interval_print_limit = int(interval_print_limit_raw) if interval_print_limit_raw is not None else 0
        run_compare_config(config, config_path or Path("<config>"), interval_print_limit)
        return

    trace_dir_value = args.trace_dir
    if trace_dir_value is None and config is not None:
        raw_trace_dir = config_value(config, "trace_dir", "trace-dir", "trace dir", "trace")
        if raw_trace_dir is not None:
            trace_dir_value = Path(str(raw_trace_dir))
    if trace_dir_value is None:
        raise SystemExit("trace_dir is required unless the JSON config contains trace_dir")

    trace_dir = trace_dir_value.resolve()
    if not (trace_dir / "code.json").is_file():
        raise SystemExit(f"missing code.json under {trace_dir}")

    rows = load_code(trace_dir)
    se_arg = args.se if args.se is not None else optional_str(config_value(config, "se")) if config is not None else None
    sm_arg = args.sm if args.sm is not None else optional_str(config_value(config, "sm")) if config is not None else None
    sl_arg = args.sl if args.sl is not None else optional_str(config_value(config, "sl")) if config is not None else None
    wv_arg = args.wv if args.wv is not None else optional_str(config_value(config, "wv")) if config is not None else None
    se_filter = parse_index_filter(se_arg, "se")
    sm_filter = parse_index_filter(sm_arg, "sm")
    sl_filter = parse_index_filter(sl_arg, "sl")
    wv_filter = parse_index_filter(wv_arg, "wv")
    wave_glob = str(config_value(config, "wave_glob", "wave-glob") or args.wave_glob) if config is not None else args.wave_glob
    interval_print_limit_raw = (
        args.interval_print_limit
        if args.interval_print_limit != 0 or config is None
        else config_value(config, "interval-print-limit", "interval_print_limit")
    )
    interval_print_limit = int(interval_print_limit_raw) if interval_print_limit_raw is not None else 0

    segment_selectors = [
        args.start_idx is not None or args.end_idx is not None,
        args.start_line is not None or args.end_line is not None,
        args.start_vaddr is not None or args.end_vaddr is not None,
        args.start_regex is not None or args.end_regex is not None,
        args.segment_file is not None,
        args.segment_text is not None,
    ]
    config_points: list[list[str]] | None = None
    config_interval_infos: list[IntervalInfo] | None = None
    if config is not None and ("sample points" in config or "sample_points" in config):
        config_points = parse_config_sample_points(config, config_path or Path("<config>"))
        config_interval_infos = parse_config_interval_infos(config, config_points, config_path or Path("<config>"))

    if args.interval_points_file is not None or config_points is not None:
        if args.interval_start is not None or args.interval_end is not None:
            raise SystemExit("--interval-points-file cannot be combined with --interval-start/--interval-end")
        if any(segment_selectors):
            raise SystemExit("--interval-points-file cannot be combined with segment selection options")

        print(f"Trace dir: {trace_dir}")
        print(f"Wave glob: {wave_glob}")
        if any(value is not None for value in (se_arg, sm_arg, sl_arg, wv_arg)):
            print(f"Wave filters: se={se_arg or '*'} sm={sm_arg or '*'} sl={sl_arg or '*'} wv={wv_arg or '*'}")
        if config_points is not None:
            print(f"Config JSON: {config_path}")
            run_interval_points_data(
                rows,
                trace_dir,
                wave_glob,
                se_filter,
                sm_filter,
                sl_filter,
                wv_filter,
                config_points,
                config_interval_infos,
                interval_print_limit,
            )
        else:
            run_interval_points_file(
                rows,
                trace_dir,
                wave_glob,
                se_filter,
                sm_filter,
                sl_filter,
                wv_filter,
                args.interval_points_file,
                interval_print_limit,
            )
        return

    interval_mode = args.interval_start is not None or args.interval_end is not None
    if interval_mode:
        if args.interval_start is None or args.interval_end is None:
            raise SystemExit("--interval-start and --interval-end must be provided together")
        if any(segment_selectors):
            raise SystemExit("interval mode cannot be combined with segment selection options")

        start_code_sequences = exact_instruction_sequences(
            rows, args.interval_start, args.interval_separator, "--interval-start"
        )
        end_code_sequences = exact_instruction_sequences(
            rows, args.interval_end, args.interval_separator, "--interval-end"
        )
        intervals = scan_intervals(
            trace_dir,
            wave_glob,
            se_filter,
            sm_filter,
            sl_filter,
            wv_filter,
            start_code_sequences,
            end_code_sequences,
        )

        print(f"Trace dir: {trace_dir}")
        print("Interval start instruction sequence:")
        for line in normalized_instruction_lines(split_instruction_text(args.interval_start, args.interval_separator)):
            print(f"  {line}")
        print("Interval end instruction sequence:")
        for line in normalized_instruction_lines(split_instruction_text(args.interval_end, args.interval_separator)):
            print(f"  {line}")
        print(f"Start code_idx candidates by instruction: {[format_code_candidates(c) for c in start_code_sequences]}")
        print(f"End code_idx candidates by instruction:   {[format_code_candidates(c) for c in end_code_sequences]}")
        print(f"Wave glob: {wave_glob}")
        if any(value is not None for value in (se_arg, sm_arg, sl_arg, wv_arg)):
            print(f"Wave filters: se={se_arg or '*'} sm={sm_arg or '*'} sl={sl_arg or '*'} wv={wv_arg or '*'}")
        print(f"Matched intervals: {len(intervals)}")
        if not intervals:
            raise SystemExit("no [start, end) intervals matched in the scanned wave traces")
        print_interval_results(rows, intervals, interval_print_limit)
        return

    target = select_segment(args, rows)
    occurrences, aggregate, top_events = scan_waves(
        trace_dir,
        wave_glob,
        se_filter,
        sm_filter,
        sl_filter,
        wv_filter,
        target,
        rows,
    )

    print(f"Trace dir: {trace_dir}")
    print(
        f"Segment code_idx: {target[0]}..{target[-1]} "
        f"({len(target)} executable instructions after filtering)"
    )
    print(f"Wave glob: {wave_glob}")
    if any(value is not None for value in (se_arg, sm_arg, sl_arg, wv_arg)):
        print(f"Wave filters: se={se_arg or '*'} sm={sm_arg or '*'} sl={sl_arg or '*'} wv={wv_arg or '*'}")
    print(f"Matched occurrences: {len(occurrences)}")
    if not occurrences:
        raise SystemExit("selected instruction segment did not occur in the scanned wave traces")

    cycles = [float(occ.cycles_sum) for occ in occurrences]
    spans = [float(occ.cycles_span) for occ in occurrences]
    print(f"Total cycles across all occurrences/waves: {int(sum(cycles))}")
    print(
        "Per-occurrence cycles: "
        f"avg={statistics.mean(cycles):.1f} "
        f"p50={percentile(cycles, 0.50):.1f} "
        f"p90={percentile(cycles, 0.90):.1f} "
        f"min={min(cycles):.0f} max={max(cycles):.0f}"
    )
    print(
        "Per-occurrence timestamp span: "
        f"avg={statistics.mean(spans):.1f} "
        f"p50={percentile(spans, 0.50):.1f} "
        f"p90={percentile(spans, 0.90):.1f} "
        f"min={min(spans):.0f} max={max(spans):.0f}"
    )

    if args.show_segment:
        print_segment(rows, target, args.segment_print_limit)
    if args.show_occurrences:
        print_occurrences(occurrences, args.occurrence_print_limit)

    if not args.hide_occurrence_details:
        print_occurrence_detail_tables(rows, occurrences, args.top_k, args.occurrence_detail_limit)
    print_aggregate_topk(rows, aggregate, args.top_k, args.rank_by)
    if args.top_events:
        print_event_topk(rows, top_events, args.top_k)


if __name__ == "__main__":
    main()
