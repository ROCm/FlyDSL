# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Offline gfx942 ATT / flytrace correlation and Perfetto Trace Event export.

ATT wave IDs are hardware slots, not CTA-local wave numbers. Correlation uses
HW_ID, an absolute realtime window, and the complete s_memrealtime fingerprint.
The exported ISA clock is interpolated between recorded timer reads per wave.
Raw shader cycles and cross-validation errors remain available in the output.
"""

import json
import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


def _fit(xs, ys):
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    if not denominator:
        raise ValueError("Clock samples do not span time")
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denominator
    offset = ym - slope * xm
    errors = [y - (slope * x + offset) for x, y in zip(xs, ys)]
    return slope, offset, math.sqrt(statistics.mean(e * e for e in errors))


def _interpolate(xs, ys, x):
    # Extrapolation is only used for the small wave prologue/epilogue outside
    # the first/last timer read. Integer subtraction precedes float conversion.
    i = min(max(bisect_right(xs, x) - 1, 0), len(xs) - 2)
    return ys[i] + (x - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])


def _heldout_errors(xs, ys):
    # Predict each interior read from its two neighbours. Local calibration
    # tolerates changing GPU frequency without fitting away a wrong wave's
    # instruction-by-instruction timing fingerprint.
    return [
        abs(ys[i - 1] + (xs[i] - xs[i - 1]) * (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1]) - ys[i])
        for i in range(1, len(xs) - 1)
    ]


def _match(raw, directory):
    if raw.get("format") != "flytrace.raw.v1" or raw.get("arch", "").split(":")[0] != "gfx942":
        raise ValueError("ATT merging requires a gfx942 flytrace.raw.v1 capture")
    frequency = raw["clock_hz"]
    realtime = json.loads((directory / "realtime.json").read_text())
    if realtime["metadata"]["frequency"] != frequency:
        raise ValueError("ATT and flytrace realtime clock frequencies differ")
    code = json.loads((directory / "code.json").read_text())["code"]
    if not code:
        raise ValueError("ATT contains no decoded ISA")
    origin = min(w["epoch"] for w in raw["waves"])
    index = defaultdict(list)
    for wave in raw["waves"]:
        if "hardware" not in wave:
            raise ValueError("Capture with flytrace.capture(hardware=True) before merging ATT")
        hw = wave["hardware"]
        if hw["sh"] != 0:
            raise ValueError("Only gfx942 shader-array 0 is supported")
        index[hw["se"], hw["cu"], hw["simd"], hw["slot"]].append(wave)
    matches, used, engine_xcc = [], set(), {}
    for path in sorted(directory.glob("se*_sm*_sl*_wv*.json")):
        decoded = json.loads(path.read_text())
        att = decoded["wave"]
        instructions = att["instructions"]
        if decoded["num_insts"] != decoded["num_stitched"] or len(instructions) != decoded["num_insts"]:
            raise ValueError(f"Incomplete ATT instruction history: {path.name}")
        se = int(decoded["name"].removeprefix("SE"))
        clock_points = realtime[decoded["name"]]
        if len(clock_points) < 2:
            raise ValueError("At least two ATT realtime calibration points are required")
        shader = [p[0] for p in clock_points]
        real = [p[1] - origin for p in clock_points]
        if any(b <= a for a, b in zip(shader, shader[1:])) or any(b <= a for a, b in zip(real, real[1:])):
            raise ValueError("ATT clock calibration points must increase")
        coarse_rate = (real[-1] - real[0]) / (shader[-1] - shader[0])
        if any(len(i) != 5 or not 0 <= i[4] < len(code) or not 0 <= i[2] <= i[3] for i in instructions):
            raise ValueError(f"Unsupported ATT instruction schema: {path.name}")
        clocks = [i for i in instructions if code[i[4]][0].startswith("s_memrealtime ")]
        xs = [i[0] + i[2] for i in clocks]  # successful issue, not first issue attempt
        if len(xs) < 8 or any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("Need at least six flytrace events and monotonic timer instructions for reliable matching")
        candidates = []
        for wave in index[se, att["cu"], att["simd"], att["slot"]]:
            ticks = [wave["epoch"], *[e["tick"] for e in wave["events"]], wave["end_tick"]]
            if len(ticks) != len(xs):
                continue
            ys = [t - origin for t in ticks]
            # Coarse absolute-time gate rejects unrelated launches. The two
            # ATT calibration packets alone can drift by microseconds.
            if abs(ys[0] - _interpolate(shader, real, xs[0])) > frequency * 30e-6:
                continue
            slope, _, affine_rms = _fit(xs, ys)
            if slope <= 0 or abs(slope / coarse_rate - 1) > 0.10:
                continue
            heldout = _heldout_errors(xs, ys)
            rms = math.sqrt(statistics.mean(e * e for e in heldout))
            candidates.append((rms, wave, ys, affine_rms, heldout))
        candidates.sort(key=lambda c: c[0])
        if not candidates:
            raise ValueError(f"No matching flytrace wave for {path.name}; collect both during the same launch")
        rms, wave, ys, affine_rms, heldout = candidates[0]
        rms_ns = rms * 1e9 / frequency
        if rms_ns > 200:
            raise ValueError(f"Clock fingerprint mismatch for {path.name}: RMS {rms_ns:.1f} ns")
        if len(candidates) > 1 and candidates[1][0] < max(rms * 2, rms + frequency * 20e-9):
            raise ValueError(f"Ambiguous block/wave identity for {path.name}; refusing to guess")
        identity = (wave["kernel"], tuple(wave["block"]), wave["wave"])
        if identity in used:
            raise ValueError("Two ATT waves matched the same flytrace wave")
        used.add(identity)
        xcc = wave["hardware"]["xcc"]
        if engine_xcc.setdefault(se, xcc) != xcc:
            raise ValueError("ATT shader engine matched inconsistent XCC identities")
        heldout = [e * 1e9 / frequency for e in heldout]
        if max(heldout) > 500:
            raise ValueError(f"Clock interpolation error exceeds 500 ns: {path.name}")
        if any(b < a for a, b in zip(ys, ys[1:])):
            raise ValueError("Timer samples must not go backwards")
        matches.append(
            dict(
                fly=wave,
                att=att,
                xs=xs,
                ys=ys,
                filename=path.name,
                alignment=dict(
                    method="per-wave timer-anchor interpolation",
                    anchors=len(xs),
                    fingerprint_rms_ns=rms_ns,
                    affine_fit_rms_ns=affine_rms * 1e9 / frequency,
                    heldout_max_error_ns=max(heldout),
                    runner_up_rms_ns=candidates[1][0] * 1e9 / frequency if len(candidates) > 1 else None,
                ),
            )
        )
    if not matches:
        raise ValueError("No ATT waves found")
    return matches, code, origin, frequency


def _phase_events(wave, pid, tid, origin, frequency):
    events, stack, current = [], [], None

    def emit(a, b=None):
        name = a["name"] if a["payload"] is None else f"{a['name']}[{a['payload']}]"
        event = dict(
            ph="i" if b is None else "X",
            cat="flytrace.phase",
            name=name,
            pid=pid,
            tid=tid,
            ts=(a["tick"] - origin) * 1e6 / frequency,
            args=dict(ordinal=a["ordinal"]),
        )
        if a["payload"] is not None:
            event["args"]["payload"] = a["payload"]
        if b is None:
            event["s"] = "t"
        else:
            event["dur"] = (b["tick"] - a["tick"]) * 1e6 / frequency
        events.append(event)

    for event in wave["events"]:
        kind = event["kind"]
        if kind == "mark":
            emit(event)
        elif kind == "push":
            stack.append(event)
        elif kind == "pop":
            if not stack:
                raise ValueError("Unmatched flytrace.pop()")
            emit(stack.pop(), event)
        elif kind in ("boundary", "end"):
            if current is not None:
                emit(current, event)
            current = event if kind == "boundary" else None
        else:
            raise ValueError(f"Unknown flytrace event kind: {kind}")
    if stack or current:
        raise ValueError("Unfinished flytrace phase")
    return events


def merge_att(flytrace_path, att_directory, output_path, *, max_blocks=0):
    """Merge one simultaneous flytrace raw capture and decoded ATT dispatch.

    Requires gfx942, capture(hardware=True), a complete instruction history,
    at least six trace events per wave, and no additional s_memrealtime reads.
    Ambiguous identities or inconsistent clocks raise ValueError. max_blocks=0
    exports all matched blocks; a positive limit selects earliest blocks.
    Timing between timer samples is interpolated, not cycle-exact wall time.
    """
    if not isinstance(max_blocks, int) or max_blocks < 0:
        raise ValueError("max_blocks must be a nonnegative integer")
    directory = Path(att_directory)
    raw = json.loads(Path(flytrace_path).read_text())
    matches, code, origin, frequency = _match(raw, directory)
    groups = defaultdict(list)
    for match in matches:
        w = match["fly"]
        groups[w["kernel"], tuple(w["block"])].append(match)
    ordered = sorted(groups, key=lambda key: min(m["fly"]["epoch"] for m in groups[key]))
    selected = ordered[:max_blocks] if max_blocks else ordered
    # Use a small relative origin for Perfetto numeric precision. Every wave
    # keeps the same origin, so inter-wave phase timing is preserved.
    shift = min(_interpolate(m["xs"], m["ys"], m["att"]["begin"]) for k in selected for m in groups[k])
    shift = math.floor(shift)
    events, mapping, instruction_count = [], [], 0
    for pid, key in enumerate(selected, 1):
        kernel, block = key
        events.append(dict(ph="M", name="process_name", pid=pid, args=dict(name=f"{kernel} · CTA {block}")))
        events.append(dict(ph="M", name="process_sort_index", pid=pid, args=dict(sort_index=pid)))
        for rank, match in enumerate(sorted(groups[key], key=lambda m: m["fly"]["wave"])):
            w, att = match["fly"], match["att"]
            phase_tid, isa_tid = rank * 2 + 1, rank * 2 + 2
            for tid, label in ((phase_tid, "phases"), (isa_tid, "ISA")):
                events.append(
                    dict(ph="M", name="thread_name", pid=pid, tid=tid, args=dict(name=f"wave {w['wave']} · {label}"))
                )
                events.append(dict(ph="M", name="thread_sort_index", pid=pid, tid=tid, args=dict(sort_index=tid)))
            events.extend(_phase_events(w, pid, phase_tid, origin + shift, frequency))

            def time(cycle):
                # Quantize endpoints before subtracting. Independently rounding
                # ts and dur creates artificial 1 ns overlaps in Perfetto.
                return round((_interpolate(match["xs"], match["ys"], cycle) - shift) * 1e9 / frequency)

            for ordinal, (start, category, stall, duration, line) in enumerate(att["instructions"]):
                row = code[line]
                asm = row[0]
                cat = "att.timer" if asm.startswith("s_memrealtime ") else "att.isa"
                events.append(
                    dict(
                        ph="X",
                        cat=cat,
                        name=asm,
                        pid=pid,
                        tid=isa_tid,
                        ts=time(start) / 1000,
                        dur=(time(start + duration) - time(start)) / 1000,
                        args=dict(
                            source=row[3],
                            code_object=row[4],
                            pc=hex(row[5]),
                            instruction_index=ordinal,
                            shader_cycle=start,
                            duration_cycles=duration,
                            stall_cycles=stall,
                            stall_ns=time(start + stall) - time(start),
                            issue_cycles=duration - stall,
                            category=category,
                        ),
                    )
                )
                instruction_count += 1
            mapping.append(
                dict(
                    block=list(block),
                    wave=w["wave"],
                    hardware=w["hardware"],
                    att_file=match["filename"],
                    instructions=len(att["instructions"]),
                    **match["alignment"],
                )
            )
    # Longer complete events first at equal timestamps gives stable nesting
    # of the outer gemm range and the phase ranges in the JSON importer.
    events.sort(key=lambda e: (e.get("ts", -1), -e.get("dur", 0)))
    summary = dict(
        format="flytrace.att.perfetto.v1",
        att_dispatch=directory.name,
        matched_waves=len(matches),
        matched_blocks=len(groups),
        exported_blocks=len(selected),
        exported_waves=len(mapping),
        instructions=instruction_count,
        clock_hz=frequency,
        origin_tick=origin + shift,
        wave_mapping=mapping,
        timing="Original flytrace ticks; ISA interpolated between corresponding s_memrealtime issue points. "
        "gfx9 ISA duration is stall + issue, not instruction completion latency. "
        "Held-out errors describe interpolation consistency, not a hardware accuracy guarantee.",
    )
    Path(output_path).write_text(
        json.dumps(dict(traceEvents=events, displayTimeUnit="ns", otherData=summary), separators=(",", ":")) + "\n"
    )
    Path(output_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
