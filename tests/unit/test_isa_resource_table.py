#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for scripts/isa_resource_table.py.

The parser reads LLVM's AMDGPU assembly text, which is not a stable interface.
These tests pin the exact shape it depends on -- the `amdhsa.kernels` metadata
list, the two-space `  - ` entry indent, and the `<name>:` / `.Lfunc_endN:`
body delimiters -- so a future LLVM format change fails here rather than in a
resource comparison.

The parser must also never guess: when the format does not match it reports
None, and --diff turns that into a non-zero exit. Silently returning a wrong
number would be worse than reporting nothing, because the whole point of the
tool is to catch regressions a passing test suite hides.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "isa_resource_table.py"
_spec = importlib.util.spec_from_file_location("isa_resource_table", _SCRIPT)
irt = importlib.util.module_from_spec(_spec)
sys.modules["isa_resource_table"] = irt
_spec.loader.exec_module(irt)


TWO_KERNEL_ISA = """\t.text
first_kernel_0:
\tds_read_b64 v[0:1], v2
\tds_read_b64 v[2:3], v4
\tscratch_store_dword off, v0, s0
\ts_endpgm
.Lfunc_end0:
\t.size\tfirst_kernel_0, .Lfunc_end0-first_kernel_0
second_kernel_1:
\tds_read_b64 v[6:7], v8
\ts_endpgm
.Lfunc_end1:
\t.size\tsecond_kernel_1, .Lfunc_end1-second_kernel_1
\t.amdgpu_metadata
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .name:           first_kernel_0
    .sgpr_count:     47
    .vgpr_count:     16
    .vgpr_spill_count: 4
  - .agpr_count:     0
    .name:           second_kernel_1
    .sgpr_count:     18
    .vgpr_count:     32
    .vgpr_spill_count: 0
.end_amdgpu_metadata
"""


def _write(tmp_path, text, name="21_final_isa.s"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_each_kernel_parsed_separately(tmp_path):
    """Registers come from the kernel's own metadata entry, not the first match."""
    got = irt.parse_isa(_write(tmp_path, TWO_KERNEL_ISA))
    assert set(got) == {"first_kernel_0", "second_kernel_1"}
    assert (got["first_kernel_0"]["vgpr"], got["first_kernel_0"]["sgpr"]) == (16, 47)
    assert (got["second_kernel_1"]["vgpr"], got["second_kernel_1"]["sgpr"]) == (32, 18)
    assert got["first_kernel_0"]["spill"] == 4
    assert got["second_kernel_1"]["spill"] == 0


def test_instruction_counts_scoped_to_one_body(tmp_path):
    """Counting file-wide would give the first kernel 3 ds_read instead of 2."""
    got = irt.parse_isa(_write(tmp_path, TWO_KERNEL_ISA))
    assert got["first_kernel_0"]["ds_read"] == 2
    assert got["second_kernel_1"]["ds_read"] == 1
    assert got["first_kernel_0"]["scratch_store"] == 1
    assert got["second_kernel_1"]["scratch_store"] == 0


def test_body_bounded_by_next_label_when_end_marker_renamed(tmp_path):
    """The next kernel's label also terminates a body, so counts stay correct
    even if .Lfunc_end numbering stops lining up with declaration order."""
    text = TWO_KERNEL_ISA.replace(".Lfunc_end0:", ".Lsomething_else0:")
    got = irt.parse_isa(_write(tmp_path, text))
    assert got["first_kernel_0"]["ds_read"] == 2  # not 3: second kernel excluded
    assert got["first_kernel_0"]["vgpr"] == 16


def test_unterminated_body_reports_none(tmp_path):
    """With no terminator at all, refuse to guess rather than run to EOF."""
    text = TWO_KERNEL_ISA.replace(".Lfunc_end0:", ".Lx0:").replace(".Lfunc_end1:", ".Lx1:")
    got = irt.parse_isa(_write(tmp_path, text))
    # second_kernel_1 is last and now has neither a .Lfunc_end nor a following label
    assert got["second_kernel_1"]["ds_read"] is None
    assert got["second_kernel_1"]["vgpr"] == 32  # metadata is still readable


def test_unknown_metadata_layout_reports_none(tmp_path):
    """A renamed register field must not silently read as zero."""
    text = TWO_KERNEL_ISA.replace(".vgpr_count:", ".vector_gpr_count:")
    got = irt.parse_isa(_write(tmp_path, text))
    assert got["first_kernel_0"]["vgpr"] is None


def test_collect_qualifies_only_ambiguous_names(tmp_path):
    d = tmp_path / "first_kernel_0"
    d.mkdir()
    _write(d, TWO_KERNEL_ISA)
    got = irt.collect(str(tmp_path))
    assert "first_kernel_0" in got
    assert "first_kernel_0::second_kernel_1" in got


def _diff(tmp_path, before, after):
    a, b = tmp_path / "b.json", tmp_path / "a.json"
    a.write_text(json.dumps(before))
    b.write_text(json.dumps(after))
    return irt.do_diff(str(a), str(b))


_OK = {"vgpr": 10, "sgpr": 10, "spill": 0, "scratch_store": 0, "scratch_load": 0, "ds_read": 0}


def test_diff_clean_run_succeeds(tmp_path):
    assert _diff(tmp_path, {"k": dict(_OK)}, {"k": dict(_OK)}) == 0


def test_diff_reports_regression(tmp_path):
    worse = dict(_OK, vgpr=12)
    assert _diff(tmp_path, {"k": dict(_OK)}, {"k": worse}) == 1


def test_diff_ignores_improvement(tmp_path):
    better = dict(_OK, vgpr=8)
    assert _diff(tmp_path, {"k": dict(_OK)}, {"k": better}) == 0


@pytest.mark.parametrize(
    "before,after,why",
    [
        ({}, {}, "no kernels on either side"),
        ({"k": dict(_OK)}, {}, "after side empty"),
        ({"k": dict(_OK)}, {"other": dict(_OK)}, "kernel only on one side"),
        ({"k": dict(_OK, vgpr=None)}, {"k": dict(_OK, vgpr=None)}, "unparsed metric"),
    ],
)
def test_diff_fails_when_data_is_untrustworthy(tmp_path, before, after, why):
    """Cannot distinguish 'no regressions' from 'no data' -- must not return 0."""
    assert _diff(tmp_path, before, after) == 2, why
