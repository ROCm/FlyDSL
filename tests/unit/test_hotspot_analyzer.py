# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the kernel-trace-analysis hotspot_analyzer CSV row selection.

The analyzer reads authoritative VGPR/SGPR/LDS/occupancy data from
``*_kernel_trace.csv`` and must pick the right row for the dispatch under
analysis.  Row selection is plain string/CSV matching and is the part most
prone to silent mis-selection, so it is covered here:

  - legacy dir-name heuristic (timestamped dirs) still matches
  - ``ui_output_agent_*_dispatch_*`` dirs return {} without ``--kernel``
  - ``--kernel`` + ``Dispatch_Id`` selects the correct row
  - ``--kernel`` without a ``Dispatch_Id`` column falls back to name match
  - argparse wires ``--kernel`` through to ``read_kernel_metadata``
"""

import csv
import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".claude" / "skills" / "kernel-trace-analysis" / "scripts" / "hotspot_analyzer.py"

_SPEC = importlib.util.spec_from_file_location("hotspot_analyzer", _SCRIPT)
hotspot_analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hotspot_analyzer)


# Minimal column set: the header must contain "Accum_VGPR_Count" for the CSV to
# be recognized as a kernel-trace file, plus the fields read_kernel_metadata returns.
_BASE_ROW = {
    "VGPR_Count": "100",
    "Accum_VGPR_Count": "0",
    "SGPR_Count": "50",
    "LDS_Block_Size": "4096",
    "Workgroup_Size_X": "256",
    "Workgroup_Size_Y": "1",
    "Workgroup_Size_Z": "1",
}


def _write_csv(dispatch_dir, rows):
    """Write an out_kernel_trace.csv into dispatch_dir with the given rows."""
    os.makedirs(dispatch_dir, exist_ok=True)
    path = os.path.join(dispatch_dir, "out_kernel_trace.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def test_legacy_timestamp_heuristic_still_matches(tmp_path):
    # Timestamped dir name vs trailing-index Kernel_Name -> bidirectional substring match.
    d = str(tmp_path / "20240101_120000_pa_decode_kernel")
    _write_csv(d, [{**_BASE_ROW, "Kernel_Name": "pa_decode_kernel_0", "VGPR_Count": "111"}])

    meta = hotspot_analyzer.read_kernel_metadata(d)

    assert meta and meta["csv_vgpr"] == 111


def test_ui_output_dir_without_kernel_filter_returns_empty(tmp_path):
    # ui_output_agent_*_dispatch_* dir carries no kernel name, so the legacy
    # heuristic cannot match -> {} (the bug this PR addresses).
    d = str(tmp_path / "ui_output_agent_15249_dispatch_223")
    _write_csv(d, [{**_BASE_ROW, "Kernel_Name": "pa_mqa_logits_fp4_kernel_0"}])

    assert hotspot_analyzer.read_kernel_metadata(d) == {}


def test_kernel_filter_with_dispatch_id_selects_correct_row(tmp_path):
    # Two rows share the name substring; Dispatch_Id from the dir name disambiguates.
    d = str(tmp_path / "ui_output_agent_15249_dispatch_223")
    _write_csv(
        d,
        [
            {**_BASE_ROW, "Kernel_Name": "pa_mqa_logits_fp4_kernel_0", "Dispatch_Id": "999", "VGPR_Count": "11"},
            {**_BASE_ROW, "Kernel_Name": "pa_mqa_logits_fp4_kernel_0", "Dispatch_Id": "223", "VGPR_Count": "22"},
        ],
    )

    meta = hotspot_analyzer.read_kernel_metadata(d, kernel_filter="pa_mqa_logits_fp4_kernel")

    assert meta["csv_vgpr"] == 22


def test_kernel_filter_without_dispatch_column_falls_back_to_name(tmp_path):
    # No Dispatch_Id column -> name-only substring match.
    d = str(tmp_path / "ui_output_agent_15249_dispatch_223")
    _write_csv(d, [{**_BASE_ROW, "Kernel_Name": "pa_mqa_logits_fp4_kernel_0", "VGPR_Count": "77"}])

    meta = hotspot_analyzer.read_kernel_metadata(d, kernel_filter="pa_mqa_logits_fp4")

    assert meta["csv_vgpr"] == 77


def test_ambiguous_match_without_dispatch_id_warns_and_picks_first(tmp_path, capsys):
    # Dir has no dispatch_<id> suffix, so even with a Dispatch_Id column there is
    # nothing to disambiguate -> first match wins, with a warning.
    d = str(tmp_path / "plain_dir")
    _write_csv(
        d,
        [
            {**_BASE_ROW, "Kernel_Name": "some_kernel_0", "Dispatch_Id": "1", "VGPR_Count": "11"},
            {**_BASE_ROW, "Kernel_Name": "some_kernel_1", "Dispatch_Id": "2", "VGPR_Count": "22"},
        ],
    )

    meta = hotspot_analyzer.read_kernel_metadata(d, kernel_filter="some_kernel")
    out = capsys.readouterr().out

    assert meta["csv_vgpr"] == 11
    assert "matched 2 rows" in out and "warning" in out


def test_argparse_wires_kernel_through_to_read_kernel_metadata(tmp_path, monkeypatch):
    # End-to-end: --kernel on the command line reaches read_kernel_metadata.
    d = tmp_path / "ui_output_agent_1_dispatch_5"
    d.mkdir()

    captured = {}

    def fake_read(dispatch_dir, kernel_filter=""):
        captured["kernel_filter"] = kernel_filter
        return {}

    class _FakeInst:
        stall_cycles = 1
        total_cycles = 2

    monkeypatch.setattr(hotspot_analyzer, "read_kernel_metadata", fake_read)
    monkeypatch.setattr(hotspot_analyzer, "load_instructions", lambda _d: [_FakeInst()])
    monkeypatch.setattr(hotspot_analyzer, "aggregate_by_source", lambda _i: [])
    monkeypatch.setattr(hotspot_analyzer, "load_source_map", lambda _d: {})
    monkeypatch.setattr(hotspot_analyzer, "detect_arch_and_reg_pressure", lambda _i, _m: {})
    monkeypatch.setattr(hotspot_analyzer, "print_reg_pressure", lambda _r: None)
    monkeypatch.setattr(hotspot_analyzer, "print_stall_type_summary", lambda _i, _t: None)
    monkeypatch.setattr(hotspot_analyzer, "print_source_hotspots", lambda *a, **k: None)
    monkeypatch.setattr(hotspot_analyzer, "print_asm_hotspots", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["hotspot_analyzer.py", str(d), "--kernel", "my_kernel_substr"])

    assert hotspot_analyzer.main() == 0
    assert captured["kernel_filter"] == "my_kernel_substr"
