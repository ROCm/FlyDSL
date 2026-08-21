#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Summarize and compare per-kernel resources in FlyDSL ISA dumps.

The tool is self-contained and only consumes ``*final_isa.s`` files produced by
``FLYDSL_DUMP_IR``:

    python3 scripts/isa_resource_table.py diff /tmp/before /tmp/after
    python3 scripts/isa_resource_table.py summarize <dump-dir> --json snapshot.json
    python3 scripts/isa_resource_table.py diff before.json after.json

``diff`` accepts either two dump directories or two JSON snapshots, and the two
sides may be mixed.  Every metric is scoped to one kernel, even when an ISA file
contains multiple kernels.

The tool is architecture-general.  Register counts come from the per-kernel
``.set <kernel>.num_vgpr`` / ``.num_agpr`` / ``.numbered_sgpr`` assembler symbols,
which LLVM emits on every AMDGPU target, rather than from the ``.agpr_count``
metadata field, which only exists on MFMA-capable (CDNA) targets.  LDS traffic is
counted under either mnemonic spelling (``ds_read``/``ds_write`` on gfx9 and gfx10,
``ds_load``/``ds_store`` on gfx11 and later).

Each metric cell is one of three states: a parsed value, ``n/a`` when the quantity
does not exist on the target, or ``?`` when the tool could not read something it
claims to report.  Only the third is a failure: the tool exits 1 for a resource
regression and 2 whenever it cannot produce a trustworthy answer.
"""

import sys

# The whole module is parsed before anything runs, so this guard only protects
# against *runtime* use of newer syntax -- which is why the file deliberately
# avoids `from __future__ import annotations`, the walrus operator and `match`.
# PEP 585/604 annotations below are never evaluated on an old interpreter.
if sys.version_info < (3, 10):  # pragma: no cover - exercised via subprocess
    sys.stderr.write(
        "scripts/isa_resource_table.py requires Python 3.10+ (running %s).\n"
        "This repository targets 3.10+ (CONTRIBUTING.md; ruff target-version=py310).\n"
        "Try: python3.10 scripts/isa_resource_table.py ...\n" % sys.version.split()[0]
    )
    raise SystemExit(2)

import argparse  # noqa: E402  - imports must follow the interpreter guard
import json  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402
from bisect import bisect_left  # noqa: E402
from dataclasses import dataclass, field, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TextIO  # noqa: E402

SCHEMA_VERSION = 2
TOOL_NAME = "isa_resource_table"


class SnapshotError(ValueError):
    """An ISA snapshot cannot be parsed or compared safely."""


# --------------------------------------------------------------------------------------
# Metric cells: a value, "not applicable here", or "could not read it"
# --------------------------------------------------------------------------------------

VALUE = "value"
NA = "n/a"
UNPARSED = "unparsed"


@dataclass(frozen=True)
class Cell:
    """One metric for one kernel, in one of three explicitly distinct states.

    Keeping ``NA`` separate from ``UNPARSED`` is the point of the design: a target
    with no accumulator register file legitimately has no AGPR count, and treating
    that as a parse failure made the tool unusable on every RDNA target.
    """

    state: str
    value: int | None = None
    reason: str = ""

    @staticmethod
    def of(value: int) -> "Cell":
        return Cell(VALUE, value)

    @staticmethod
    def na(reason: str) -> "Cell":
        return Cell(NA, None, reason)

    @staticmethod
    def unparsed(reason: str) -> "Cell":
        return Cell(UNPARSED, None, reason)

    def to_json(self) -> dict:
        if self.state == VALUE:
            return {"state": VALUE, "value": self.value}
        return {"state": self.state, "reason": self.reason}

    @staticmethod
    def from_json(raw: object, where: str) -> "Cell":
        if not isinstance(raw, dict):
            raise SnapshotError(f"{where} must be an object")
        state = raw.get("state")
        if state == VALUE:
            value = raw.get("value")
            if not isinstance(value, int) or isinstance(value, bool):
                raise SnapshotError(f"{where}.value must be an integer")
            return Cell.of(value)
        if state in (NA, UNPARSED):
            reason = raw.get("reason", "")
            if not isinstance(reason, str):
                raise SnapshotError(f"{where}.reason must be a string")
            return Cell(state, None, reason)
        raise SnapshotError(f"{where}.state must be one of {VALUE!r}, {NA!r}, {UNPARSED!r}")


# --------------------------------------------------------------------------------------
# The metric table
# --------------------------------------------------------------------------------------

TRIGGER = "trigger"
INFO = "info"

METADATA = "metadata"
SYMBOL = "symbol"
INSTRUCTION = "instruction"


@dataclass(frozen=True)
class Metric:
    key: str
    kind: str
    source: str
    field: str


# Exactly one metric per physical quantity is a TRIGGER.  In particular `vgpr` is the
# total (arch + accumulator) that LLVM itself uses for occupancy; `arch_vgpr` and `agpr`
# are its informational split.  Counting all three would report a single AGPR change as
# two regressions, and would flag moving accumulators into AGPRs -- which the kernel
# tuning guide recommends -- as a regression.
METRICS = (
    Metric("vgpr", TRIGGER, METADATA, "vgpr_count"),
    Metric("arch_vgpr", INFO, SYMBOL, "num_vgpr"),
    Metric("agpr", INFO, SYMBOL, "num_agpr"),
    Metric("sgpr", TRIGGER, METADATA, "sgpr_count"),
    Metric("numbered_sgpr", INFO, SYMBOL, "numbered_sgpr"),
    Metric("vgpr_spill", TRIGGER, METADATA, "vgpr_spill_count"),
    Metric("sgpr_spill", TRIGGER, METADATA, "sgpr_spill_count"),
    Metric("scratch_bytes", TRIGGER, METADATA, "private_segment_fixed_size"),
    Metric("lds_static_bytes", TRIGGER, METADATA, "group_segment_fixed_size"),
    Metric("lds_read", INFO, INSTRUCTION, "lds_read"),
    Metric("lds_write", INFO, INSTRUCTION, "lds_write"),
    Metric("scratch_store", INFO, INSTRUCTION, "scratch_store"),
    Metric("scratch_load", INFO, INSTRUCTION, "scratch_load"),
    Metric("matrix_ops", INFO, INSTRUCTION, "matrix_ops"),
)
KEYS = tuple(metric.key for metric in METRICS)
BY_KEY = {metric.key: metric for metric in METRICS}
TRIGGERS = frozenset(metric.key for metric in METRICS if metric.kind == TRIGGER)

KNOWN_SET_SUFFIXES = frozenset({"num_vgpr", "num_agpr", "numbered_sgpr", "private_seg_size"})

LEGEND = (
    "* = regression trigger; other columns are informational.\n"
    "  vgpr = total (arch+acc, LLVM's occupancy number); arch_vgpr/agpr are its split "
    "-- do not add them."
)


# --------------------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------------------

RE_PROC = re.compile(r"^gfx([0-9a-f]{3,})$")
# FeatureArchitectedFlatScratch on gfx9: these spill through scratch_*, gfx90a does not.
GFX9_FLAT_SCRATCH = frozenset({"gfx940", "gfx941", "gfx942", "gfx950"})
HIGHEST_KNOWN_GEN = 13


@dataclass(frozen=True)
class Arch:
    target_id: str = ""
    processor: str | None = None
    gen: int | None = None

    @property
    def known(self) -> bool:
        return self.processor is not None

    @property
    def extrapolated(self) -> bool:
        return self.gen is not None and self.gen > HIGHEST_KNOWN_GEN

    def spills_via_scratch(self) -> bool | None:
        """True/False when known, None when the target could not be identified."""
        if self.processor is None:
            return None
        if self.gen is not None and self.gen >= 11:
            return True
        return self.processor in GFX9_FLAT_SCRATCH

    def to_json(self) -> dict:
        return {"target_id": self.target_id, "processor": self.processor, "gen": self.gen}

    @staticmethod
    def from_json(raw: object) -> "Arch":
        if raw is None:
            return Arch()
        if not isinstance(raw, dict):
            raise SnapshotError("arch must be an object")
        processor = raw.get("processor")
        gen = raw.get("gen")
        if processor is not None and not isinstance(processor, str):
            raise SnapshotError("arch.processor must be a string or null")
        if gen is not None and (not isinstance(gen, int) or isinstance(gen, bool)):
            raise SnapshotError("arch.gen must be an integer or null")
        return Arch(str(raw.get("target_id") or ""), processor, gen)


def parse_target_id(target_id: str) -> Arch:
    """Split ``amdgcn-amd-amdhsa--gfx950:sramecc+:xnack+`` into its parts."""
    if not target_id:
        return Arch()
    tail = target_id.rsplit("--", 1)[-1]
    processor = tail.split(":")[0]
    m = RE_PROC.match(processor)
    if not m:
        return Arch(target_id, None, None)
    return Arch(target_id, processor, int(m.group(1)[:-2]))


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

# A label is at column 0 and may carry a trailing comment: `my_kernel:  ; @my_kernel`.
RE_LABEL = re.compile(r'^(?P<name>"[^"]*"|[A-Za-z_.$][\w.$]*):[ \t]*(?:;.*)?$')
# `.size <kernel>, .Lfunc_endN-<kernel>` names the kernel's terminator explicitly.
RE_SIZE = re.compile(
    r'^[ \t]*\.size[ \t]+(?P<name>"[^"]*"|[A-Za-z_.$][\w.$]*)' r"[ \t]*,[ \t]*(?P<end>\.Lfunc_end\d+)-(?P=name)[ \t]*$"
)
RE_SET = re.compile(r"^[ \t]*\.set[ \t]+(?P<sym>[A-Za-z_.$][\w.$]*)[ \t]*,[ \t]*(?P<val>-?\d+)[ \t]*$")
RE_SET_ANY = re.compile(r"^[ \t]*\.set[ \t]+(?P<sym>[A-Za-z_.$][\w.$]*)[ \t]*,")
# Kernel-level metadata keys sit at column 4, or on the `  - ` sequence-dash line.
# Argument keys sit at column 6 or 8, so both branches must be literal.
RE_MD_KEY = re.compile(r"^(?:  - |    )\.(?P<key>[A-Za-z_]\w*):[ \t]*(?P<val>.*?)[ \t]*$")
RE_MNEMONIC = re.compile(r"^[ \t]+(?P<m>[a-z][a-z0-9_]*)\b")
RE_AMDGCN_TARGET = re.compile(r'^[ \t]*\.amdgcn_target[ \t]+"([^"]*)"[ \t]*$')
RE_MD_TARGET = re.compile(r"^amdhsa\.target:[ \t]*(?:\"([^\"]*)\"|'([^']*)'|(\S+))[ \t]*$")

# `(?:_|2)` keeps ds_read2_b32 / ds_write2_b32 while rejecting gfx11+'s
# ds_storexchg_rtn_b32 -- the renamed gfx9 atomic ds_wrxchg_rtn_b32, which a bare
# `^ds_store` would miscount as an LDS write.
_CATEGORIES = (
    ("ds_read", re.compile(r"^ds_read(?:_|2)")),
    ("ds_load", re.compile(r"^ds_load(?:_|2)")),
    ("ds_write", re.compile(r"^ds_write(?:_|2)")),
    ("ds_store", re.compile(r"^ds_store(?:_|2)")),
    ("scratch_store", re.compile(r"^scratch_store")),
    ("scratch_load", re.compile(r"^scratch_load")),
    ("matrix_ops", re.compile(r"^v_(?:mfma|wmma)")),
)

METADATA_BEGIN = ".amdgpu_metadata"
METADATA_END = ".end_amdgpu_metadata"
DESCRIPTOR_BEGIN = ".amdhsa_kernel"
DESCRIPTOR_END = ".end_amdhsa_kernel"


def _unquote(name: str) -> str:
    if len(name) >= 2 and name[0] == name[-1] == '"':
        return name[1:-1]
    return name


def clean_md_value(value: str) -> str:
    """Strip a YAML tag and surrounding quotes: ``!str n`` -> ``n``."""
    v = value.strip()
    if v.startswith("!"):
        v = v.split(" ", 1)[1].strip() if " " in v else ""
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v


def _categorize(mnemonic: str) -> str | None:
    for name, pattern in _CATEGORIES:
        if pattern.match(mnemonic):
            return name
    return None


@dataclass
class _Scan:
    labels: dict = field(default_factory=dict)
    sizes: dict = field(default_factory=dict)
    symbols: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)
    instr_lines: list = field(default_factory=list)
    instr_cats: list = field(default_factory=list)
    target_id: str | None = None
    md_target_id: str | None = None
    saw_metadata: bool = False


def _record_set(line: str, symbols: dict) -> None:
    m = RE_SET.match(line)
    if m:
        sym, value = m.group("sym"), int(m.group("val"))
    else:
        loose = RE_SET_ANY.match(line)
        if not loose:
            return
        # e.g. `.set k.num_vgpr, max(32, .Lhelper.num_vgpr)` when the kernel calls a
        # function: record it as present-but-unreadable rather than dropping it.
        sym, value = loose.group("sym"), None
    base, _, suffix = sym.rpartition(".")
    if not base or suffix not in KNOWN_SET_SUFFIXES:
        return
    if base.startswith(".L"):
        # 90 of 233 real dumps emit `.set .L<kernel>.num_vgpr` even for .globl kernels,
        # because the local-linkage test differs from the one behind metadata `.name`.
        base = base[2:]
    symbols.setdefault(base, {})[suffix] = value


def _scan_text(text: str) -> _Scan:
    """One pass over the file; no regex ever runs over the whole text."""
    scan = _Scan()
    in_metadata = False
    in_descriptor = False

    for index, line in enumerate(text.splitlines()):
        if not line:
            continue
        if line[0] not in " \t":
            if in_metadata:
                if line.startswith("amdhsa.target:"):
                    m = RE_MD_TARGET.match(line)
                    if m:
                        scan.md_target_id = next(g for g in m.groups() if g is not None)
                continue
            m = RE_LABEL.match(line)
            if m:
                scan.labels.setdefault(_unquote(m.group("name")), index)
            continue

        stripped = line.lstrip()
        # `.end_*` must be tested first: it is a prefix collision with `.amdgpu_metadata`.
        if stripped.startswith(METADATA_END):
            in_metadata = False
            continue
        if stripped.startswith(METADATA_BEGIN):
            in_metadata = True
            scan.saw_metadata = True
            continue
        if stripped.startswith(DESCRIPTOR_END):
            in_descriptor = False
            continue
        if stripped.startswith(DESCRIPTOR_BEGIN):
            in_descriptor = True
            continue
        if in_descriptor:
            continue
        if in_metadata:
            m = RE_MD_KEY.match(line)
            if m:
                if line.startswith("  - "):
                    scan.entries.append({})
                if scan.entries:
                    scan.entries[-1][m.group("key")] = clean_md_value(m.group("val"))
            continue
        if stripped.startswith(".size"):
            m = RE_SIZE.match(line)
            if m:
                scan.sizes[_unquote(m.group("name"))] = m.group("end")
            continue
        if stripped.startswith(".set"):
            _record_set(line, scan.symbols)
            continue
        if stripped.startswith(".amdgcn_target"):
            m = RE_AMDGCN_TARGET.match(line)
            if m:
                scan.target_id = m.group(1)
            continue
        if stripped[0] in ".;":
            continue
        m = RE_MNEMONIC.match(line)
        if m:
            category = _categorize(m.group("m"))
            if category:
                scan.instr_lines.append(index)
                scan.instr_cats.append(category)
    return scan


@dataclass(frozen=True)
class KernelRecord:
    name: str
    arch: Arch
    metrics: dict
    source: str = ""
    notes: tuple = ()
    problems: tuple = ()

    @property
    def unparsed_keys(self) -> tuple:
        return tuple(key for key in KEYS if self.metrics[key].state == UNPARSED)


def _kernel_name(entry: dict) -> tuple:
    """(name, problems) -- metadata `.name`, cross-checked against `.symbol`."""
    name = entry.get("name") or ""
    symbol = entry.get("symbol") or ""
    from_symbol = symbol[:-3] if symbol.endswith(".kd") else symbol
    if name and from_symbol and name != from_symbol:
        return name, (f"metadata .name {name!r} disagrees with .symbol {symbol!r}",)
    if name:
        return name, ()
    if from_symbol:
        return from_symbol, ()
    return "", ("metadata entry has neither .name nor .symbol",)


def _metadata_cell(entry: dict, metric: Metric) -> Cell:
    raw = entry.get(metric.field)
    if raw is None:
        return Cell.unparsed(f"metadata field .{metric.field} is absent")
    try:
        return Cell.of(int(raw))
    except ValueError:
        return Cell.unparsed(f"metadata field .{metric.field} is not an integer: {raw!r}")


def _symbol_cell(symbols: dict, name: str, metric: Metric) -> Cell:
    if name not in symbols or metric.field not in symbols[name]:
        return Cell.unparsed(f"assembler symbol .set {name}.{metric.field} is absent")
    value = symbols[name][metric.field]
    if value is None:
        return Cell.unparsed(
            f".set {name}.{metric.field} is an unresolved expression "
            "(the kernel calls a function); use the metadata counts"
        )
    return Cell.of(value)


def _instruction_cells(counts: dict, arch: Arch, body_error: str | None) -> tuple:
    """Returns (cells, notes, problems) for the five instruction-derived metrics."""
    if body_error is not None:
        cells = {metric.key: Cell.unparsed(body_error) for metric in METRICS if metric.source == INSTRUCTION}
        return cells, (), ()

    problems = []
    if counts.get("ds_read", 0) and counts.get("ds_load", 0):
        problems.append("mixed ds_read/ds_load spellings in one body; the dump is inconsistent")
    if counts.get("ds_write", 0) and counts.get("ds_store", 0):
        problems.append("mixed ds_write/ds_store spellings in one body; the dump is inconsistent")

    cells = {
        "lds_read": Cell.of(counts.get("ds_read", 0) + counts.get("ds_load", 0)),
        "lds_write": Cell.of(counts.get("ds_write", 0) + counts.get("ds_store", 0)),
        "matrix_ops": Cell.of(counts.get("matrix_ops", 0)),
    }

    scratch = arch.spills_via_scratch()
    if scratch is False:
        reason = (
            f"{arch.processor} spills through buffer_* instructions; "
            "a scratch instruction count is not a spill signal on this target"
        )
        cells["scratch_store"] = Cell.na(reason)
        cells["scratch_load"] = Cell.na(reason)
    else:
        cells["scratch_store"] = Cell.of(counts.get("scratch_store", 0))
        cells["scratch_load"] = Cell.of(counts.get("scratch_load", 0))
    return cells, (), tuple(problems)


def parse_isa(path: str | Path) -> dict:
    """Parse one ``*final_isa.s`` file into ``{kernel name: KernelRecord}``."""
    path = Path(path)
    # Assembly is ASCII in practice.  Replacing a stray byte degrades one instruction
    # count instead of aborting the run with a traceback and a misleading exit code.
    text = path.read_text(encoding="utf-8", errors="replace")
    scan = _scan_text(text)

    if not scan.saw_metadata:
        return {}

    arch = parse_target_id(scan.target_id or scan.md_target_id or "")
    arch_problems = []
    if scan.target_id and scan.md_target_id and scan.target_id != scan.md_target_id:
        arch_problems.append(f".amdgcn_target {scan.target_id!r} disagrees with amdhsa.target {scan.md_target_id!r}")

    records = {}
    for entry in scan.entries:
        name, name_problems = _kernel_name(entry)
        if not name:
            continue

        metrics = {}
        for metric in METRICS:
            if metric.source == METADATA:
                metrics[metric.key] = _metadata_cell(entry, metric)
            elif metric.source == SYMBOL:
                metrics[metric.key] = _symbol_cell(scan.symbols, name, metric)

        counts, body_error = _body_counts(scan, name)
        instruction_cells, notes, instruction_problems = _instruction_cells(counts, arch, body_error)
        metrics.update(instruction_cells)

        notes = list(notes)
        lds_static = metrics["lds_static_bytes"]
        lds_traffic = metrics["lds_read"].value or 0
        lds_traffic += metrics["lds_write"].value or 0
        if lds_static.state == VALUE and lds_static.value == 0 and lds_traffic > 0:
            notes.append("dynamic LDS in use; a static size of 0 does not bound this kernel's LDS")

        cross = _cross_check_scratch(metrics, scan.symbols.get(name, {}))
        records[name] = KernelRecord(
            name=name,
            arch=arch,
            metrics=metrics,
            source=str(path),
            notes=tuple(notes),
            problems=tuple(name_problems) + tuple(arch_problems) + tuple(instruction_problems) + cross,
        )
    return records


def _cross_check_scratch(metrics: dict, symbols: dict) -> tuple:
    """`.private_seg_size` and `.private_segment_fixed_size` measure the same quantity."""
    cell = metrics["scratch_bytes"]
    other = symbols.get("private_seg_size")
    if cell.state != VALUE or other is None or other == cell.value:
        return ()
    return (f"scratch bytes disagree: metadata {cell.value} vs .set private_seg_size {other}",)


def _body_counts(scan: _Scan, name: str) -> tuple:
    """Instruction counts for one kernel body, or (empty, reason) when unbounded."""
    if name not in scan.labels:
        return {}, f"no code label {name!r} in the file"
    terminator = scan.sizes.get(name)
    if terminator is None:
        return {}, f"no .size line for {name!r}, so the end of its body is unknown"
    if terminator not in scan.labels:
        return {}, f"terminator {terminator!r} for {name!r} is never defined"

    lo = scan.labels[name] + 1
    hi = scan.labels[terminator]
    if hi <= lo:
        return {}, f"body of {name!r} is empty or inverted ({lo}..{hi})"

    start = bisect_left(scan.instr_lines, lo)
    stop = bisect_left(scan.instr_lines, hi)
    counts: dict = {}
    for category in scan.instr_cats[start:stop]:
        counts[category] = counts.get(category, 0) + 1
    return counts, None


# --------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------

RE_STAGE_PREFIX = re.compile(r"^(\d+)_")


@dataclass(frozen=True)
class Snapshot:
    kernels: dict = field(default_factory=dict)
    warnings: tuple = ()
    problems: tuple = ()

    def __len__(self) -> int:
        return len(self.kernels)

    def __bool__(self) -> bool:
        return bool(self.kernels)

    @property
    def trustworthy(self) -> bool:
        return not self.problems and not any(record.unparsed_keys for record in self.kernels.values())

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "trustworthy": self.trustworthy,
            "warnings": list(self.warnings),
            "kernels": {
                key: {
                    "source": record.source,
                    "arch": record.arch.to_json(),
                    "notes": list(record.notes),
                    "problems": list(record.problems),
                    "metrics": {k: record.metrics[k].to_json() for k in KEYS},
                }
                for key, record in sorted(self.kernels.items())
            },
        }


def _pick_stage_file(paths: list) -> tuple:
    """Newest final-ISA file in one directory, plus a warning naming the ignored ones."""
    if len(paths) == 1:
        return paths[0], None

    def sort_key(path):
        m = RE_STAGE_PREFIX.match(path.name)
        stage = int(m.group(1)) if m else -1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (stage, mtime, path.name)

    ordered = sorted(paths, key=sort_key)
    chosen = ordered[-1]
    ignored = ", ".join(p.name for p in ordered[:-1])
    warning = f"{chosen.parent}: {len(paths)} final-ISA files, using {chosen.name} (ignoring stale {ignored})"
    return chosen, warning


def collect(root: str | Path) -> Snapshot:
    """Summarize every ``*final_isa.s`` under ``root``."""
    root = Path(root)
    if not root.exists():
        raise SnapshotError(f"dump directory not found: {root}")
    if not root.is_dir():
        raise SnapshotError(f"not a directory: {root}")

    by_directory: dict = {}
    for path in sorted(root.rglob("*final_isa.s")):
        by_directory.setdefault(path.parent, []).append(path)
    if not by_directory:
        raise SnapshotError(f"no *final_isa.s under {root} (is FLYDSL_DUMP_IR set?)")

    kernels: dict = {}
    warnings: list = []
    problems: list = []
    files_read = 0

    for directory in sorted(by_directory):
        chosen, warning = _pick_stage_file(by_directory[directory])
        if warning:
            warnings.append(warning)
        files_read += 1

        records = parse_isa(chosen)
        if not records:
            warnings.append(f"{chosen}: not an LLVM AMDHSA dump (no {METADATA_BEGIN}); skipped")
            continue

        relative_dir = chosen.parent.relative_to(root).as_posix()
        relative_dir = "" if relative_dir == "." else relative_dir
        if relative_dir and Path(relative_dir).name not in records and len(records) == 1:
            warnings.append(
                f"{chosen.parent}: directory name does not match kernel "
                f"{next(iter(records))!r}; keys are path-derived"
            )
        for name, record in records.items():
            # The key is a pure function of one file's location and one kernel's own
            # name.  It must never depend on what else happens to be in the tree, or
            # adding one kernel to the "after" run renames every pre-existing key.
            key = f"{relative_dir}::{name}"
            if key in kernels:
                problems.append(f"duplicate kernel key {key!r}")
                continue
            kernels[key] = replace(record, source=str(chosen.relative_to(root)))
            problems.extend(f"{key}: {p}" for p in record.problems)

        if records:
            arch = next(iter(records.values())).arch
            if not arch.known:
                warnings.append(
                    f"{chosen}: architecture unknown; scratch instruction counts are raw " "traffic, not proven spills"
                )
            elif arch.extrapolated:
                warnings.append(
                    f"{chosen}: arch {arch.processor} is newer than this tool knows "
                    f"(gen {arch.gen}); its bucket is extrapolated"
                )

    if not kernels:
        raise SnapshotError(f"no kernels found in {files_read} file(s) under {root}")

    warnings.append(
        "FLYDSL_DUMP_IR writes one directory per kernel *name* with no specialization key, "
        "so different JIT specializations overwrite each other and cache hits emit no dump. "
        "Compare only trees from a single run with FLYDSL_RUNTIME_ENABLE_CACHE=0."
    )
    return Snapshot(kernels=kernels, warnings=tuple(warnings), problems=tuple(problems))


_V1_ALIASES = {
    "vgpr": "vgpr",
    "sgpr": "sgpr",
    "agpr": "agpr",
    "vgpr_spill": "vgpr_spill",
    "sgpr_spill": "sgpr_spill",
    "scratch_bytes": "scratch_bytes",
    "lds_bytes": "lds_static_bytes",
    "scratch_store": "scratch_store",
    "scratch_load": "scratch_load",
    "ds_read": "lds_read",
}


def _load_v1(raw: dict, path: Path) -> Snapshot:
    kernels = {}
    for kernel, metrics in raw.items():
        if not isinstance(kernel, str) or not isinstance(metrics, dict):
            raise SnapshotError(f"{path} contains an invalid kernel entry")
        cells = {key: Cell.unparsed("absent from this v1 snapshot") for key in KEYS}
        for old_key, new_key in _V1_ALIASES.items():
            if old_key not in metrics:
                continue
            value = metrics[old_key]
            if value is None:
                # v1 could not tell "not applicable" from "failed to parse".
                cells[new_key] = Cell.unparsed("null in a v1 snapshot; regenerate it")
            elif isinstance(value, int) and not isinstance(value, bool):
                cells[new_key] = Cell.of(value)
            else:
                raise SnapshotError(f"{path}: {kernel}.{old_key} must be an integer or null")
        kernels[kernel] = KernelRecord(name=kernel, arch=Arch(), metrics=cells, source=str(path))
    return Snapshot(
        kernels=kernels,
        warnings=(f"{path} is a v1 snapshot; regenerate it with this version of the tool",),
    )


def load_snapshot(path: str | Path) -> Snapshot:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SnapshotError(f"{path} must contain a JSON object")
    if "schema" not in raw:
        return _load_v1(raw, path)

    if raw.get("schema") != SCHEMA_VERSION:
        raise SnapshotError(f"{path}: unsupported schema {raw.get('schema')!r}")
    entries = raw.get("kernels")
    if not isinstance(entries, dict):
        raise SnapshotError(f"{path}: 'kernels' must be an object")

    kernels = {}
    for kernel, body in entries.items():
        if not isinstance(kernel, str) or not isinstance(body, dict):
            raise SnapshotError(f"{path} contains an invalid kernel entry")
        raw_metrics = body.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise SnapshotError(f"{path}: {kernel}.metrics must be an object")
        cells = {}
        for key in KEYS:
            if key not in raw_metrics:
                cells[key] = Cell.unparsed(f"absent from {path.name}")
            else:
                cells[key] = Cell.from_json(raw_metrics[key], f"{path}: {kernel}.{key}")
        kernels[kernel] = KernelRecord(
            name=kernel,
            arch=Arch.from_json(body.get("arch")),
            metrics=cells,
            source=str(body.get("source") or ""),
            notes=tuple(body.get("notes") or ()),
            problems=tuple(body.get("problems") or ()),
        )

    problems = []
    if raw.get("trustworthy") is False:
        problems.append(f"{path} was written from data the tool could not fully parse")
    return Snapshot(
        kernels=kernels,
        warnings=tuple(raw.get("warnings") or ()),
        problems=tuple(problems),
    )


def load_input(path: str | Path) -> Snapshot:
    """Load a JSON snapshot or summarize a FLYDSL_DUMP_IR directory."""
    path = Path(path)
    if path.is_dir():
        return collect(path)
    if not path.exists():
        raise SnapshotError(f"input not found: {path}")
    if path.suffix != ".json":
        raise SnapshotError(f"expected a dump directory or a .json snapshot: {path}")
    return load_snapshot(path)


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------

ONLY_IN_BEFORE = "ONLY IN BEFORE"
ONLY_IN_AFTER = "ONLY IN AFTER"


@dataclass(frozen=True)
class DiffRow:
    name: str
    before: KernelRecord | None
    after: KernelRecord | None
    blocked: tuple = ()

    @property
    def side(self) -> str | None:
        """The one label for a one-sided row -- defined once, used by every caller."""
        if self.before is None:
            return ONLY_IN_AFTER
        if self.after is None:
            return ONLY_IN_BEFORE
        return None

    @property
    def comparable(self) -> bool:
        return self.side is None and not self.blocked

    def delta(self, key: str) -> int | None:
        if not self.comparable:
            return None
        before_cell = self.before.metrics[key]
        after_cell = self.after.metrics[key]
        if before_cell.state != VALUE or after_cell.state != VALUE:
            return None
        return after_cell.value - before_cell.value

    @property
    def changed(self) -> bool:
        """The single definition of 'this kernel changed', used by every caller."""
        return any(self.delta(key) for key in KEYS)


@dataclass(frozen=True)
class Comparison:
    rows: tuple = ()
    problems: tuple = ()
    warnings: tuple = ()
    notes: tuple = ()
    total_kernels: int = 0

    @property
    def compared_kernels(self) -> int:
        return sum(1 for row in self.rows if row.comparable)

    @property
    def unchanged_kernels(self) -> int:
        return sum(1 for row in self.rows if row.comparable and not row.changed)

    @property
    def changed_kernels(self) -> int:
        return sum(1 for row in self.rows if row.comparable and row.changed)

    @property
    def worsened_metrics(self) -> int:
        return self._count_triggers(lambda d: d > 0)

    @property
    def improved_metrics(self) -> int:
        return self._count_triggers(lambda d: d < 0)

    def _count_triggers(self, predicate) -> int:
        total = 0
        for row in self.rows:
            if not row.comparable:
                continue
            for key in TRIGGERS:
                delta = row.delta(key)
                if delta is not None and predicate(delta):
                    total += 1
        return total

    @property
    def exit_code(self) -> int:
        if self.problems:
            return 2
        return 1 if self.worsened_metrics else 0

    @property
    def verdict(self) -> str:
        return {0: "RESULT: OK", 1: "RESULT: REGRESSION", 2: "RESULT: NOT TRUSTWORTHY"}[self.exit_code]


def _match_keys(before: Snapshot, after: Snapshot) -> tuple:
    """Exact keys first, then pair leftovers by bare kernel name across layout drift."""
    common = before.kernels.keys() & after.kernels.keys()
    pairs = {key: key for key in common}
    notes: list = []

    left = sorted(before.kernels.keys() - common)
    right = sorted(after.kernels.keys() - common)
    if left and right:

        def by_kernel(keys):
            out: dict = {}
            for key in keys:
                out.setdefault(key.rpartition("::")[2], []).append(key)
            return out

        left_by, right_by = by_kernel(left), by_kernel(right)
        for kernel, left_keys in left_by.items():
            right_keys = right_by.get(kernel, [])
            if len(left_keys) == 1 and len(right_keys) == 1:
                pairs[left_keys[0]] = right_keys[0]
                notes.append(
                    "matched by kernel name across differing layouts:\n"
                    f"      before  {left_keys[0]}\n"
                    f"      after   {right_keys[0]}"
                )
    return pairs, tuple(notes)


def compare_snapshots(before: Snapshot, after: Snapshot) -> Comparison:
    problems = list(before.problems) + list(after.problems)
    if not before or not after:
        problems.append(f"one or both inputs contain no kernels (before={len(before)}, after={len(after)})")

    pairs, notes = _match_keys(before, after)
    matched_after = set(pairs.values())

    rows = []
    for key in sorted(before.kernels.keys() | after.kernels.keys()):
        if key in pairs:
            before_record = before.kernels[key]
            after_record = after.kernels[pairs[key]]
        elif key in after.kernels and key not in matched_after:
            rows.append(DiffRow(key, None, after.kernels[key]))
            problems.append(f"{key}: {ONLY_IN_AFTER}")
            continue
        elif key in after.kernels:
            continue  # already consumed as the right-hand side of a drift match
        else:
            rows.append(DiffRow(key, before.kernels[key], None))
            problems.append(f"{key}: {ONLY_IN_BEFORE}")
            continue

        blocked = []
        if before_record.arch.processor != after_record.arch.processor:
            if before_record.arch.known and after_record.arch.known:
                blocked.append(
                    f"architecture differs ({before_record.arch.processor} vs "
                    f"{after_record.arch.processor}); resource counts are not comparable"
                )
        elif before_record.arch.target_id != after_record.arch.target_id:
            notes = notes + (
                f"{key}: target features differ ({before_record.arch.target_id} vs "
                f"{after_record.arch.target_id}); comparing anyway",
            )

        for metric_key in KEYS:
            before_state = before_record.metrics[metric_key].state
            after_state = after_record.metrics[metric_key].state
            if UNPARSED in (before_state, after_state):
                reason = (
                    before_record.metrics[metric_key].reason
                    if before_state == UNPARSED
                    else after_record.metrics[metric_key].reason
                )
                blocked.append(f"unparsed metric {metric_key}: {reason}")
            elif (before_state == NA) != (after_state == NA):
                blocked.append(
                    f"metric {metric_key} is {before_state} before and {after_state} after; " "applicability changed"
                )

        rows.append(DiffRow(key, before_record, after_record, tuple(blocked)))
        problems.extend(f"{key}: {reason}" for reason in blocked)

    total = len(before.kernels.keys() | after.kernels.keys()) - (
        len(pairs) - len(before.kernels.keys() & after.kernels.keys())
    )
    return Comparison(
        rows=tuple(rows),
        problems=tuple(problems),
        warnings=tuple(before.warnings) + tuple(after.warnings),
        notes=tuple(notes),
        total_kernels=total,
    )


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

MIN_NAME_WIDTH = 20
MAX_NAME_WIDTH = 60
NAME_TAIL = 12
MIN_CELL_WIDTH = 5


def _abbreviate(name: str, width: int) -> str:
    """Keep the tail: real kernel names differ in their trailing config suffix."""
    if len(name) <= width:
        return name
    if width <= NAME_TAIL + 1:
        return name[-width:]
    head = width - NAME_TAIL - 1
    return name[:head] + "~" + name[-NAME_TAIL:]


def _name_column(names: list) -> dict:
    if not names:
        return {}
    longest = max(len(name) for name in names)
    width = min(max(MIN_NAME_WIDTH, longest), MAX_NAME_WIDTH)
    shown = {name: _abbreviate(name, width) for name in names}
    if len(set(shown.values())) != len(shown):
        # Rows must never be indistinguishable; drop the cap rather than the identity.
        return {name: name for name in names}
    return shown


def _format_table(headers: list, rows: list) -> list:
    """Fixed-width table whose columns are sized from the data. Nothing is truncated."""
    widths = []
    for index, header in enumerate(headers):
        longest = max([len(header)] + [len(row[index]) for row in rows]) if rows else len(header)
        widths.append(max(longest, MIN_CELL_WIDTH if index else len(header)))

    def line(cells):
        parts = [cells[0].ljust(widths[0])]
        parts.extend(cell.rjust(widths[i]) for i, cell in enumerate(cells[1:], start=1))
        return " ".join(parts).rstrip()

    header_line = line(headers)
    out = [header_line, "-" * max(len(header_line), *(len(line(row)) for row in rows), 1)]
    out.extend(line(row) for row in rows)
    return out


def _header_cells() -> list:
    return ["kernel"] + [("*" + key if key in TRIGGERS else key) for key in KEYS]


def _cell_text(cell: Cell) -> str:
    if cell.state == VALUE:
        return str(cell.value)
    return "n/a" if cell.state == NA else "?"


def render_snapshot(snapshot: Snapshot, output: TextIO, error: TextIO | None = None) -> None:
    error = error if error is not None else sys.stderr
    names = _name_column(sorted(snapshot.kernels))
    rows = []
    for key in sorted(snapshot.kernels):
        record = snapshot.kernels[key]
        rows.append([names[key]] + [_cell_text(record.metrics[k]) for k in KEYS])

    print(LEGEND, file=output)
    for line in _format_table(_header_cells(), rows):
        print(line, file=output)

    for key in sorted(snapshot.kernels):
        for note in snapshot.kernels[key].notes:
            print(f"  note: {names[key]}: {note}", file=output)

    print(f"\n{len(snapshot)} kernels", file=output)
    _render_problems(snapshot.warnings, snapshot.problems, output, error)


def _render_problems(warnings, problems, output: TextIO, error: TextIO) -> None:
    for warning in warnings:
        print(f"warning: {warning}", file=error)
    if problems:
        print(f"{len(problems)} kernel(s)/metric(s) not trustworthy (see stderr)", file=output)
        print(f"\nNOT TRUSTWORTHY ({len(problems)} problem(s)):", file=error)
        for problem in problems[:20]:
            print(f"  {problem}", file=error)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=error)


def render_comparison(comparison: Comparison, output: TextIO, error: TextIO) -> None:
    printable = [row for row in comparison.rows if not row.comparable or row.changed]
    names = _name_column([row.name for row in printable])

    rows = []
    for row in printable:
        if row.side is not None:
            rows.append([names[row.name], row.side] + [""] * (len(KEYS) - 1))
            continue
        if row.blocked:
            rows.append([names[row.name], "BLOCKED: " + row.blocked[0]] + [""] * (len(KEYS) - 1))
            continue
        cells = [names[row.name]]
        for key in KEYS:
            delta = row.delta(key)
            if delta:
                before_value = row.before.metrics[key].value
                after_value = row.after.metrics[key].value
                cells.append(f"{before_value}->{after_value}({delta:+d})")
            else:
                cells.append(_cell_text(row.after.metrics[key]))
        rows.append(cells)

    print(LEGEND, file=output)
    for line in _format_table(_header_cells(), rows):
        print(line, file=output)

    for note in comparison.notes:
        print(f"\nnote: {note}", file=output)

    print(
        f"\ncompared {comparison.compared_kernels} of {comparison.total_kernels} kernels; "
        f"{comparison.unchanged_kernels} unchanged; {comparison.changed_kernels} changed; "
        f"worsened: {comparison.worsened_metrics}; improved: {comparison.improved_metrics}",
        file=output,
    )
    _render_problems(comparison.warnings, comparison.problems, output, error)
    print(comparison.verdict, file=output)


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def do_diff(
    before_path: str | Path,
    after_path: str | Path,
    *,
    output: TextIO | None = None,
    error: TextIO | None = None,
    json_path: str | Path | None = None,
    quiet: bool = False,
) -> int:
    output = output if output is not None else sys.stdout
    error = error if error is not None else sys.stderr
    comparison = compare_snapshots(load_input(before_path), load_input(after_path))
    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(_comparison_json(comparison), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if quiet:
        _render_problems(comparison.warnings, comparison.problems, output, error)
        print(comparison.verdict, file=output)
    else:
        render_comparison(comparison, output, error)
    return comparison.exit_code


def _comparison_json(comparison: Comparison) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "verdict": comparison.verdict,
        "exit_code": comparison.exit_code,
        "compared_kernels": comparison.compared_kernels,
        "total_kernels": comparison.total_kernels,
        "worsened_metrics": comparison.worsened_metrics,
        "improved_metrics": comparison.improved_metrics,
        "problems": list(comparison.problems),
        "warnings": list(comparison.warnings),
        "rows": [
            {
                "kernel": row.name,
                "side": row.side,
                "blocked": list(row.blocked),
                "deltas": {key: row.delta(key) for key in KEYS if row.delta(key)},
            }
            for row in comparison.rows
            if not row.comparable or row.changed
        ],
    }


def do_summarize(
    dump_dir: str | Path,
    *,
    json_path: str | Path | None = None,
    output: TextIO | None = None,
    error: TextIO | None = None,
    quiet: bool = False,
) -> int:
    output = output if output is not None else sys.stdout
    error = error if error is not None else sys.stderr
    snapshot = collect(dump_dir)
    if json_path is not None:
        Path(json_path).write_text(json.dumps(snapshot.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if quiet:
        _render_problems(snapshot.warnings, snapshot.problems, output, error)
    else:
        render_snapshot(snapshot, output, error)

    # Fail closed on unreadable metrics, not merely on an empty directory: a snapshot
    # whose every metric is unparsed must not be reported -- or persisted -- as success.
    if not snapshot.trustworthy:
        unparsed = sorted({key for record in snapshot.kernels.values() for key in record.unparsed_keys})
        if unparsed:
            print(
                f"\nNOT TRUSTWORTHY: unparsed metric(s): {', '.join(unparsed)}",
                file=error,
            )
        print("RESULT: NOT TRUSTWORTHY", file=output)
        return 2
    print("RESULT: OK", file=output)
    return 0


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    summarize = commands.add_parser("summarize", help="summarize a FLYDSL_DUMP_IR directory")
    summarize.add_argument("dump_dir", type=Path)
    summarize.add_argument("--json", type=Path, metavar="PATH", help="write a reusable JSON snapshot")
    summarize.add_argument("-q", "--quiet", action="store_true", help="suppress the table")

    diff = commands.add_parser("diff", help="compare two dump directories or JSON snapshots")
    diff.add_argument("before", type=Path, help="before dump directory or JSON snapshot")
    diff.add_argument("after", type=Path, help="after dump directory or JSON snapshot")
    diff.add_argument("--json", type=Path, metavar="PATH", help="write the comparison as JSON")
    diff.add_argument("-q", "--quiet", action="store_true", help="suppress the table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    try:
        if args.command == "diff":
            return do_diff(args.before, args.after, json_path=args.json, quiet=args.quiet)
        return do_summarize(args.dump_dir, json_path=args.json, quiet=args.quiet)
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("RESULT: NOT TRUSTWORTHY")
        return 2
    except Exception:  # noqa: BLE001 - a crash must never be reported as a regression
        traceback.print_exc()
        print("error: internal error; the result is not trustworthy", file=sys.stderr)
        print("RESULT: NOT TRUSTWORTHY")
        return 2


if __name__ == "__main__":
    sys.exit(main())
