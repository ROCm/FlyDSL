# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Tests for triage.py. Run: python3 -m pytest .claude/skills/review-pr/test_triage.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import triage  # noqa: E402

RULES_MD = (HERE / "rules.md").read_text(encoding="utf-8")


def diff_for(path: str, added: list[str], deleted: list[str] = (), new: bool = False, start: int = 10) -> str:
    """A one-hunk unified diff that adds `added` (and deletes `deleted`) in `path`."""
    head = [f"diff --git a/{path} b/{path}"]
    if new:
        head += ["new file mode 100644", "--- /dev/null"]
    else:
        head += [f"--- a/{path}"]
    head += [f"+++ b/{path}", f"@@ -{start},{len(deleted) + 1} +{start},{len(added) + 1} @@ def f():", " context"]
    return "\n".join(head + [f"-{x}" for x in deleted] + [f"+{x}" for x in added]) + "\n"


# --------------------------------------------------------------------------- diff parsing


def test_added_lines_track_new_line_numbers_across_hunks():
    diff = (
        "diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n"
        "@@ -1,3 +1,4 @@\n a\n-b\n+B1\n+B2\n c\n"
        "@@ -20,2 +21,3 @@\n x\n+Y\n y\n"
    )
    assert list(triage.added_lines(diff)) == [("k.py", 2, "B1"), ("k.py", 3, "B2"), ("k.py", 22, "Y")]


def test_parse_diff_records_new_deleted_and_renamed_files():
    diff = (
        "diff --git a/a.py b/a.py\nnew file mode 100644\n--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n"
        "diff --git a/b.py b/b.py\ndeleted file mode 100644\n--- a/b.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-y\n"
        "diff --git a/c.py b/d.py\nsimilarity index 90%\nrename from c.py\nrename to d.py\n"
    )
    files = triage.parse_diff(diff)
    assert files["a.py"]["new"] and files["b.py"]["deleted"] and files["d.py"]["renamed_from"] == "c.py"


# --------------------------------------------------------------------------- derivation


def families(diff: str, title: str = "x") -> dict:
    return dict(triage.derive(diff, title, REPO))


def test_docs_and_agent_metadata_only_derive_nothing():
    diff = diff_for("docs/kernel_authoring_guide.md", ["text"]) + diff_for(".claude/skills/x/SKILL.md", ["y"])
    assert triage.derive(diff, "x", REPO) == [("docs-only", "NONE")]


def test_tier1_compiler_file_requires_step4_and_cache_rules():
    fams = families(diff_for("python/flydsl/compiler/jit_function.py", ["    key = (a, b)"]))
    assert "STEP4" in fams["compiler-core"] and "A3" in fams["cache-surface"]


def test_kernel_file_derives_kernel_family_but_not_tests():
    fams = families(diff_for("kernels/norm/rmsnorm_kernel.py", ["    y = x + 1"]))
    assert "kernel" in fams and "tests" not in fams


def test_uncovered_arch_fires_on_code_not_on_prose():
    code = diff_for("kernels/gemm/foo.py", ['    if arch == "gfx1250":'])
    prose = diff_for("kernels/gemm/foo.py", ["    # gfx1250 is wave32"]) + diff_for("docs/a.md", ["gfx1250"])
    assert "uncovered-arch" in families(code) and "uncovered-arch" not in families(prose)


def test_new_source_file_without_spdx_header_derives_hk9():
    bare = diff_for("kernels/new_op.py", ["import torch"], new=True)
    licensed = diff_for("kernels/new_op.py", ["# SPDX-License-Identifier: Apache-2.0", "import torch"], new=True)
    assert "new-source" in families(bare) and "new-source" not in families(licensed)


def test_perf_claim_comes_from_the_title():
    diff = diff_for("kernels/norm/rmsnorm_kernel.py", ["    y = x + 1"])
    assert "perf-claim" in families(diff, "[Perf] Optimize rmsnorm") and "perf-claim" not in families(diff, "[Fix] x")


# --------------------------------------------------------------------------- scan


def hits(diff: str, key: str) -> list:
    return triage.scan(diff)[key].hits


def test_truncating_division_in_a_count_is_a_candidate():
    assert hits(diff_for("kernels/g.py", ["    num_a_loads = bytes_per_thread_a // a_load_bytes"]), "trunc_div")
    assert not hits(diff_for("kernels/g.py", ["    half = n // 2"]), "trunc_div")


def test_index_width_ignores_widened_products():
    assert hits(diff_for("kernels/a.py", ["    off = row * row_stride"]), "index_width")
    assert not hits(diff_for("kernels/a.py", ["    off = fx.Int64(row) * row_stride"]), "index_width")


def test_deleted_guard_counts_only_when_not_re_added():
    gone = diff_for("kernels/a.py", ["    pass_through()"], deleted=["    assert x.is_contiguous()"])
    moved = diff_for("kernels/a.py", ["  assert x.is_contiguous()"], deleted=["    assert x.is_contiguous()"])
    assert hits(gone, "guard_removed") and not hits(moved, "guard_removed")


def test_stream_context_manager_is_a_stream_candidate():
    assert hits(diff_for("tests/kernels/test_a.py", ["    with torch.cuda.stream(s):"]), "stream")


def test_legacy_buffer_and_raw_ir_skip_imports_and_the_sanctioned_module():
    assert hits(diff_for("kernels/g.py", ["    r = buffer_ops.create_buffer_resource(x)"]), "legacy_buffer")
    assert not hits(diff_for("kernels/common/buffer_ops.py", ["    r = buffer_ops.helper(x)"]), "legacy_buffer")
    assert not hits(diff_for("kernels/g.py", ["from flydsl._mlir import ir"]), "raw_mlir")
    assert hits(diff_for("kernels/g.py", ["    t = ir.IntegerType.get_signless(32)"]), "raw_mlir")


def test_deleted_checks_barriers_and_tests_are_removed_guards():
    mlir = diff_for("tests/mlir/Conversion/x.mlir", ["// CHECK: arith.addi"], deleted=["// CHECK: arith.muli"])
    test = diff_for("tests/kernels/test_a.py", ["    pass"], deleted=["def test_tail_tile():"])
    barrier = diff_for("kernels/a.py", ["    y = x"], deleted=["    gpu.barrier()"])
    for diff in (mlir, test, barrier):
        assert hits(diff, "guard_removed")


def test_skill_files_are_not_scanned():
    diff = diff_for(".claude/skills/review-pr/triage.py", ["    off = row * row_stride", "    k = num_x // tile_k"])
    assert all(not c.hits for c in triage.scan(diff).values())


# --------------------------------------------------------------------------- rule text and mapping


def test_every_family_rule_has_a_body_and_every_body_is_reachable():
    referenced = {
        r
        for rules, _ in list(triage.PATH_FAMILIES.values()) + list(triage.SCAN_FAMILIES.values())
        for r in rules.split()
    } - {"NONE", "STEP4"}
    ids = set(triage.all_rule_ids(RULES_MD))
    assert referenced - ids == set(), "a family emits a rule with no body in rules.md"
    assert ids - referenced == set(), "rules.md holds a rule no family can emit"


def test_expand_emits_exactly_the_derived_rules():
    text, unresolved = triage.expand_rules("[kernel] E1 HK3\n", RULES_MD)
    assert unresolved == [] and "**E1 — " in text and "| HK3 |" in text and "**E2 — " not in text


def test_mapping_md_is_regenerated_from_the_deriver():
    assert (HERE / "MAPPING.md").read_text(encoding="utf-8") == triage.render_mapping(RULES_MD)


def test_tier_table_paths_exist_in_the_tree():
    missing = [p for p in triage.TIER1 + triage.TIER1C + triage.TIER2_PREFIXES if not (REPO / p).exists()]
    assert missing == []


def test_shared_kernel_helper_is_tier2_by_fanin():
    assert triage.tier_of("kernels/common/buffer_ops.py", REPO) == "TIER2"
    assert triage.tier_of("kernels/conv/conv3d_implicit.py", REPO) is None


# --------------------------------------------------------------------------- evidence


def test_run_tests_dirs_match_the_ci_suite():
    dirs = triage.run_tests_dirs(REPO)
    assert "tests/kernels/" in dirs and "tests/unit/" in dirs


def test_ci_coverage_distinguishes_collected_gated_and_uncollected_tests(tmp_path):
    files = {
        "tests/kernels/test_new.py": "def test_a():\n    assert 1\n",
        "tests/perf/test_side.py": "def test_b():\n    assert 1\n",
        "tests/kernels/test_mgpu.py": "import pytest\n@pytest.mark.multi_gpu\ndef test_c():\n    assert 1\n",
        "tests/kernels/test_script.py": "def main():\n    print(1)\n",
    }
    diff = ""
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
        diff += diff_for(path, text.splitlines(), new=True)
    rows = {p: (status, notes) for p, status, notes in triage.ci_coverage(diff, REPO, tmp_path)}
    assert rows["tests/kernels/test_new.py"][0] == "RUNS in PR CI"
    assert rows["tests/perf/test_side.py"][0].startswith("NOT")
    assert rows["tests/kernels/test_mgpu.py"][0].startswith("NOT")
    assert any("collects nothing" in n for n in rows["tests/kernels/test_script.py"][1])


def test_bench_coverage_follows_imports_from_benchmarked_tests():
    rows = dict(triage.bench_coverage(diff_for("kernels/norm/rmsnorm_kernel.py", ["x"]), REPO))
    assert "tests/kernels/test_rmsnorm.py" in rows["kernels/norm/rmsnorm_kernel.py"]
    rows = dict(triage.bench_coverage(diff_for("kernels/conv/conv3d_implicit.py", ["x"]), REPO))
    assert rows["kernels/conv/conv3d_implicit.py"] == []


def test_buffer_bounds_treats_an_omitted_max_size_as_unbounded(tmp_path):
    src = [
        "def k(A, B, C, n):",
        "    a = fx.rocdl.make_buffer_tensor(A)",
        "    b = fx.rocdl.make_buffer_tensor(B, max_size=False)",
        "    c = buffer_ops.create_buffer_resource(C, num_records_bytes=n)",
        "    d = buffer_ops.create_buffer_resource(C, 0, True)",
        "    e = fx.rocdl.make_buffer_tensor(A)",
    ]
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels/k.py").write_text("\n".join(src) + "\n")
    diff = (
        "diff --git a/kernels/k.py b/kernels/k.py\n--- a/kernels/k.py\n+++ b/kernels/k.py\n"
        "@@ -1,2 +1,5 @@\n def k(A, B, C, n):\n" + "".join(f"+{x}\n" for x in src[1:5]) + "     e = 1\n"
    )
    rows = triage.buffer_bounds(diff, tmp_path)
    assert [(n, call) for _, n, call, _ in rows] == [(2, "make_buffer_tensor(A)"), (5, "create_buffer_resource(C)")]


# --------------------------------------------------------------------------- gates

DIFF = diff_for("kernels/norm/rmsnorm_kernel.py", ["    y = x * scale"]) + diff_for(
    "python/flydsl/compiler/jit_function.py", ["    key = key + (dtype,)"]
)
GOOD = {
    "pr.diff": DIFF,
    "rules.txt": "[kernel] E1 D1\n",
    "answers.txt": "\n".join(
        f"{q}: the change multiplies rmsnorm output by scale and adds dtype to the jit key here"
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "BLIND")
    ),
    "core_files.txt": "python/flydsl/compiler/jit_function.py TIER1 GAP -- the new dtype key entry has no test"
    " that compiles twice\n",
    "verdicts.txt": "E1 CLEAR -- no floor division in rmsnorm_kernel.py hunk\n"
    "D1 FIRE -- kernels/norm/rmsnorm_kernel.py:11 multiplies x by scale in int32\n",
    "ai_diagnostic.txt": "\n".join(
        f"{i}: looked at rmsnorm_kernel.py and jit_function.py, nothing new" for i in range(1, 7)
    ),
    "card.md": "## FlyDSL PR #1 — t\n\n**One sentence.**\n\nReview (advisory): ⚠️ NEEDS WORK\n"
    "CI (head abc1234): test (linux-flydsl-mi325-1) SUCCESS\nPerf (CI benchmark, advisory): NOT MEASURED — x\n\n"
    "🔴 [verified] rmsnorm_kernel.py:11 multiplies x by scale in int32; at M=65536, N=65536 it wraps. "
    "**Author must** widen it.\n",
    "refutations.txt": "RED SURVIVED -- opened kernels/norm/rmsnorm_kernel.py:11, no widening above it\n",
    "independent.txt": "SURVIVED -- opened head/kernels/norm/rmsnorm_kernel.py:11 and saw the int32 product\n",
}


def work(tmp_path, **override) -> Path:
    for name, text in {**GOOD, **override}.items():
        if text is not None:
            (tmp_path / name).write_text(text)
    return tmp_path


def problems(gate: str, tmp_path, **override) -> str:
    w = work(tmp_path, **override)
    return " | ".join(dict(triage.GATES)[gate](w, REPO, (w / "pr.diff").read_text()))


def test_a_complete_work_dir_passes_every_gate(tmp_path):
    assert triage.run_gates(work(tmp_path), REPO) == 0
    assert "7 green / 0 red" in (tmp_path / "GATES.txt").read_text()


def test_answers_gate_rejects_missing_and_formulaic_answers(tmp_path):
    out = problems("answers", tmp_path, **{"answers.txt": "Q1: no\nQ2: " + "x" * 50})
    assert "UNANSWERED BLIND" in out and "NO-SUBSTANCE Q1" in out


def test_corefiles_gate_rejects_missing_line_and_wrong_tier(tmp_path):
    assert "UNASSESSED" in problems("corefiles", tmp_path, **{"core_files.txt": "NONE -- " + "x" * 40})
    wrong = "python/flydsl/compiler/jit_function.py TIER2 COVERED -- " + "x" * 40
    assert "TIER-MISMATCH" in problems("corefiles", tmp_path, **{"core_files.txt": wrong})


def test_ledger_gate_rejects_unadjudicated_rules_and_untouched_citations(tmp_path):
    out = problems("ledger", tmp_path, **{"verdicts.txt": "D1 FIRE -- kernels/other/thing.py:3 int32 product\n"})
    assert "UNADJUDICATED E1" in out and "UNTOUCHED-CITATION D1" in out


def test_card_gate_rejects_rule_codes_untagged_and_unreported_fires(tmp_path):
    card = GOOD["card.md"].replace("🔴 [verified] rmsnorm_kernel.py:11", "⚠️ D1: the product") + "📝 a note\n"
    out = problems("card", tmp_path, **{"card.md": card})
    for cls in ("RULE-CODE", "UNTAGGED", "NO-ACTION", "UNREPORTED-FIRE D1"):
        assert cls in out


def test_card_gate_rejects_an_unanchored_or_unproven_red(tmp_path):
    card = GOOD["card.md"].split("🔴")[0] + "🔴 [verified] other.py looks racy. **Author must** fix it.\n"
    out = problems("card", tmp_path, **{"card.md": card})
    assert "UNANCHORED" in out and "UNPROVEN-RED" in out


def test_card_gate_requires_admitting_a_missing_independent_reader(tmp_path):
    assert "UNADMITTED" in problems("card", tmp_path, **{"independent.txt": "NONE AVAILABLE -- no second agent\n"})


def test_refutations_and_independent_gates_count_findings(tmp_path):
    assert "UNATTACKED" in problems(
        "refutations", tmp_path, **{"refutations.txt": "RED KILLED -- opened a.py, " + "x" * 20}
    )
    extra = GOOD["independent.txt"] * 2
    assert "COUNT-MISMATCH" in problems("independent", tmp_path, **{"independent.txt": extra})


@pytest.mark.parametrize("missing", ["answers.txt", "core_files.txt", "ai_diagnostic.txt", "card.md"])
def test_a_missing_artifact_fails_its_gate(tmp_path, missing):
    gate = {
        "answers.txt": "answers",
        "core_files.txt": "corefiles",
        "ai_diagnostic.txt": "diagnostic",
        "card.md": "card",
    }[missing]
    assert "MISSING" in problems(gate, tmp_path, **{missing: None})


def test_apply_refutation_drops_killed_findings_in_card_order(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("Review (advisory): x\n🔴 [verified] first a.py:1\n⚠️ [inferred] second a.py:2\n")
    indep = tmp_path / "independent.txt"
    indep.write_text("KILLED -- opened a.py:1, guarded above\nSURVIVED -- opened a.py:2, still unguarded\n")
    assert "removed 1" in triage.apply_refutation(card, indep)
    assert "first" not in card.read_text() and "second" in card.read_text()
    indep.write_text("SURVIVED -- only one line for two findings here\n")
    card.write_text("🔴 [verified] a a.py:1\n⚠️ [verified] b a.py:2\n")
    assert triage.apply_refutation(card, indep).startswith("no-op")


def test_apply_refutation_lists_corrections_by_surviving_order_once(tmp_path):
    card = tmp_path / "card.md"
    card.write_text(
        "Review (advisory): x\n🔴 [verified] first a.py:1\n⚠️ [inferred] second a.py:2\n📝 [verified] third a.py:3\n"
    )
    indep = tmp_path / "independent.txt"
    indep.write_text(
        "KILLED -- opened a.py:1, guarded above\n"
        "SURVIVED -- opened a.py:2 and it holds. Correction: the guard it cites is at a.py:20\n"
        "SURVIVED -- opened a.py:3 and it holds\n"
    )
    assert "listed 1 correction" in triage.apply_refutation(card, indep)
    text = card.read_text()
    assert "first" not in text and "- Finding 1: the guard it cites is at a.py:20" in text
    assert len(triage.card_findings(text)) == 2
    card.write_text("🔴 [verified] only a.py:1\n")
    indep.write_text("SURVIVED -- opened a.py:1. Correction: it is line 10\n")
    triage.apply_refutation(card, indep)
    triage.apply_refutation(card, indep)
    assert card.read_text().count(triage.CORRECTIONS_HEAD) == 1
