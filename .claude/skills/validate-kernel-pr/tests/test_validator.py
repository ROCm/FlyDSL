# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Regression assets for the FlyDSL PR validator.

Every stage here has been observed failing on a seeded defect and passing on a matched
control. A stage that has only ever been observed passing is decoration, not a check, so
each seeded-defect test below is paired with a negative control that must stay green.

    python3 -m pytest .claude/skills/validate-kernel-pr/tests/test_validator.py -q
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validate_pr", SKILL_DIR / "validate_pr.py")
vp = importlib.util.module_from_spec(_spec)
sys.modules["validate_pr"] = vp
_spec.loader.exec_module(vp)


# --------------------------------------------------------------------------------------
# Synthetic benchmark generator
# --------------------------------------------------------------------------------------


def make_rounds(n, base_value=10.0, head_factor=1.0, noise=0.01, seed=0, label="softmax|32768,8192|bf16", drift=0.0):
    """A/B/A rounds with multiplicative noise and optional monotonic drift.

    `drift` models the machine getting slower or faster over the run (clocks, a neighbour
    ramping up). The A/B/A design is supposed to cancel it; the tests below check that it
    actually does.
    """
    rng = random.Random(seed)
    rounds = []
    for i in range(n):
        d = 1.0 + drift * i

        def sample(factor):
            return base_value * factor * d * (1.0 + rng.uniform(-noise, noise))

        rounds.append(
            {
                "base_a": {label: sample(1.0)},
                "head": {label: sample(head_factor)},
                "base_b": {label: sample(1.0)},
            }
        )
    return rounds


def status_of(analysis, label="softmax|32768,8192|bf16"):
    return next(r["status"] for r in analysis["rows"] if r["label"] == label)


# --------------------------------------------------------------------------------------
# Seeded defect: a real performance regression must block
# --------------------------------------------------------------------------------------


def test_seeded_regression_is_detected():
    # head is 20% slower on a quiet machine: unambiguous.
    a = vp.analyze_perf(make_rounds(7, head_factor=0.80, noise=0.01, seed=1))
    assert status_of(a) == "regression"
    assert a["regressions"], "a 20% throughput loss must be reported"
    row = a["regressions"][0]
    assert row["change_pct"] < -15


def test_small_but_real_regression_on_a_quiet_machine_is_detected():
    # 6% loss with 1% noise is well outside the noise floor.
    a = vp.analyze_perf(make_rounds(9, head_factor=0.94, noise=0.01, seed=2))
    assert status_of(a) == "regression"


def test_regression_survives_monotonic_drift():
    # The machine slows 3% per round for unrelated reasons; the A/B/A sandwich must not
    # turn that into a fake regression, and must still see the real one.
    clean = vp.analyze_perf(make_rounds(7, head_factor=1.0, noise=0.01, seed=3, drift=0.03))
    assert status_of(clean) != "regression", "drift alone must not be reported as a regression"

    real = vp.analyze_perf(make_rounds(7, head_factor=0.80, noise=0.01, seed=3, drift=0.03))
    assert status_of(real) == "regression"


# --------------------------------------------------------------------------------------
# Negative controls: noise must not be reported as a regression
# --------------------------------------------------------------------------------------


def test_pure_noise_is_not_a_regression():
    for seed in range(12):
        a = vp.analyze_perf(make_rounds(7, head_factor=1.0, noise=0.02, seed=seed))
        assert status_of(a) != "regression", f"seed {seed}: identical code reported as a regression"


def test_high_noise_low_occupancy_row_does_not_flap():
    # run_benchmark.sh documents a softmax-backward tier that "swings 24% run to run".
    # A fixed threshold would flag it constantly; the measured noise floor must absorb it.
    for seed in range(12):
        a = vp.analyze_perf(make_rounds(7, head_factor=1.0, noise=0.24, seed=seed))
        assert status_of(a) != "regression", f"seed {seed}: 24% noise reported as a regression"


def test_small_regression_hidden_by_large_noise_is_not_claimed():
    # A 5% loss under 24% noise is genuinely unresolvable. Reporting "unchanged" is the
    # honest answer; claiming a regression here would be a coin flip.
    a = vp.analyze_perf(make_rounds(7, head_factor=0.95, noise=0.24, seed=7))
    assert status_of(a) in {"unchanged", "improvement"}
    assert a["rows"][0]["noise_floor"] > 0.05


# --------------------------------------------------------------------------------------
# Direction: PR #848 shipped inverted speedup columns. Prove the sign is right.
# --------------------------------------------------------------------------------------


def test_throughput_direction():
    faster = vp.analyze_perf(make_rounds(7, head_factor=1.30, noise=0.01, seed=4))
    assert status_of(faster) == "improvement"
    assert faster["rows"][0]["change_pct"] > 0


def test_latency_direction_is_inverted():
    # With lower_is_better, a LARGER head value is worse.
    slower = vp.analyze_perf(make_rounds(7, head_factor=1.30, noise=0.01, seed=5), lower_is_better=True)
    assert status_of(slower) == "regression"

    quicker = vp.analyze_perf(make_rounds(7, head_factor=0.70, noise=0.01, seed=6), lower_is_better=True)
    assert status_of(quicker) == "improvement"


# --------------------------------------------------------------------------------------
# A row present on only one side is incomparable, never silently "fine"
# --------------------------------------------------------------------------------------


def test_head_only_row_is_incomparable():
    rounds = make_rounds(4, seed=8)
    for r in rounds:
        r["head"]["gemm|new_shape|bf16"] = 5.0
    a = vp.analyze_perf(rounds)
    labels = {r["label"]: r["status"] for r in a["rows"]}
    assert labels["gemm|new_shape|bf16"] == "incomparable"


# --------------------------------------------------------------------------------------
# Metric parsing: PR #654 kept the LAST regex match and mislabelled layernorm for months
# --------------------------------------------------------------------------------------


TABLE = """
[run_benchmark] GPU arch: gfx950 (CDNA=true)
op                 shape          dtype     tbps   tflops
layernorm          32768,8192     bf16      5.601  -
fused_add_sq       32768,8192     bf16      1.690  -
gemm               4096,4096,4096 bf16      -      1204.5
skipped_row        1,1            bf16      skip   skip
"""


def test_table_parser_keeps_every_row_separate():
    m = vp.parse_metrics(TABLE, "flydsl-table")
    assert m["layernorm|32768,8192|bf16"] == 5.601
    assert m["fused_add_sq|32768,8192|bf16"] == 1.690, "the last row must not overwrite the first"
    assert m["gemm|4096,4096,4096|bf16"] == 1204.5
    assert not any(k.startswith("skipped_row") for k in m)


def test_regex_parser_labels_matches():
    text = "kernel=softmax bw=5.6 GB/s\nkernel=layernorm bw=1.69 GB/s\n"
    m = vp.parse_metrics(text, "regex", r"kernel=(?P<label>\w+) bw=(?P<value>[\d.]+)")
    assert m == {"softmax": 5.6, "layernorm": 1.69}


# --------------------------------------------------------------------------------------
# Cold-cache detection: a warm cache would serve the previous kernel
# --------------------------------------------------------------------------------------


def test_cold_cache_required_for_cpp_pass_change():
    assert vp.needs_cold_cache(["lib/Conversion/FlyToROCDL/FlyToROCDL.cpp"])
    assert vp.needs_cold_cache(["python/flydsl/expr/arith.py"])
    assert vp.needs_cold_cache(["kernels/common/kernels_common.py"])


def test_cold_cache_not_required_for_a_leaf_kernel():
    assert not vp.needs_cold_cache(["kernels/attention/pa_decode_tile.py"])
    assert not vp.needs_cold_cache(["tests/kernels/test_pa.py"])


def test_side_env_isolates_caches_and_forces_determinism(tmp_path):
    base = vp.side_env({}, tmp_path / "base", tmp_path / "cache-base", "3", cold_cache=True)
    head = vp.side_env({}, tmp_path / "head", tmp_path / "cache-head", "3", cold_cache=True)
    assert base["FLYDSL_RUNTIME_CACHE_DIR"] != head["FLYDSL_RUNTIME_CACHE_DIR"]
    assert base["FLYDSL_AUTOTUNE"] == "0"
    assert base["FLYDSL_RUNTIME_ENABLE_CACHE"] == "0"
    assert base["HIP_VISIBLE_DEVICES"] == "3"


# --------------------------------------------------------------------------------------
# Test policy
# --------------------------------------------------------------------------------------


TEST_ONLY_WIDENING = """--- a/tests/kernels/test_softmax.py
+++ b/tests/kernels/test_softmax.py
@@ -10,3 +10,3 @@
-    assert_close(got, ref, rtol=2e-3, atol=2e-3)
+    assert_close(got, ref, rtol=5e-2, atol=5e-2)
"""

WIDENING_WITH_KERNEL_CHANGE = TEST_ONLY_WIDENING + """--- a/kernels/norm/softmax_kernel.py
+++ b/kernels/norm/softmax_kernel.py
@@ -5,2 +5,2 @@
-    vec_width = 8
+    vec_width = 128 // elem_bits
"""


def _policy(patch_text, paths):
    rep = vp.Report("t")
    rep.repo["patch_paths"] = paths
    vp.stage_test_policy(rep, patch_text)
    return rep


def test_test_only_tolerance_widening_blocks():
    rep = _policy(TEST_ONLY_WIDENING, ["tests/kernels/test_softmax.py"])
    assert rep.stages["test_policy"].status == "fail"
    assert rep.verdict() == "BLOCK"


def test_widening_with_a_kernel_change_is_needs_work_not_block():
    rep = _policy(WIDENING_WITH_KERNEL_CHANGE, ["tests/kernels/test_softmax.py", "kernels/norm/softmax_kernel.py"])
    assert rep.stages["test_policy"].status == "pass"
    assert rep.verdict() == "NEEDS_WORK"


def test_clean_patch_has_no_policy_finding():
    clean = """--- a/kernels/norm/softmax_kernel.py
+++ b/kernels/norm/softmax_kernel.py
@@ -5,2 +5,2 @@
-    vec_width = 8
+    vec_width = 128 // elem_bits
"""
    rep = _policy(clean, ["kernels/norm/softmax_kernel.py"])
    assert rep.stages["test_policy"].status == "pass"
    assert not rep.findings


# --------------------------------------------------------------------------------------
# End-to-end perf stage: subprocess, interleaving, parsing and verdict together.
# The pure-function tests above prove the statistics; these prove the wiring.
# --------------------------------------------------------------------------------------


def _fake_side(tmp_path, name, tbps):
    """A directory whose 'benchmark' prints the repo's own 5-column table."""
    d = tmp_path / name
    d.mkdir()
    (d / "bench.py").write_text(
        "import random, sys\n"
        f"v = {tbps} * (1.0 + random.uniform(-0.01, 0.01))\n"
        "print('op                 shape          dtype     tbps   tflops')\n"
        f"print(f'layernorm          32768,8192     bf16      {{v:.4f}}  -')\n"
    )
    return d


def _run_perf(tmp_path, base_tbps, head_tbps, rounds=5):
    rep = vp.Report("e2e")
    base = _fake_side(tmp_path, "base", base_tbps)
    head = _fake_side(tmp_path, "head", head_tbps)
    env = {"PATH": "/usr/bin:/bin"}
    vp.stage_perf(
        rep,
        base,
        head,
        f"{sys.executable} bench.py",
        env,
        env,
        rounds=rounds,
        timeout=60,
        metric_format="flydsl-table",
        metric_regex=None,
        min_effect=0.03,
        lower_is_better=False,
    )
    return rep


def test_end_to_end_perf_stage_blocks_a_seeded_regression(tmp_path):
    rep = _run_perf(tmp_path, base_tbps=5.60, head_tbps=4.48)  # -20%
    assert rep.stages["perf"].status == "fail"
    assert rep.verdict() == "BLOCK"
    assert any("layernorm" in f.detail for f in rep.findings)


def test_end_to_end_perf_stage_passes_identical_code(tmp_path):
    rep = _run_perf(tmp_path, base_tbps=5.60, head_tbps=5.60)
    assert rep.stages["perf"].status == "pass", rep.stages["perf"].reason
    assert not [f for f in rep.findings if f.severity == "blocker"]


def test_end_to_end_perf_stage_skips_when_nothing_is_measurable(tmp_path):
    rep = vp.Report("e2e")
    d = tmp_path / "empty"
    d.mkdir()
    (d / "bench.py").write_text("print('no table here')\n")
    env = {"PATH": "/usr/bin:/bin"}
    vp.stage_perf(rep, d, d, f"{sys.executable} bench.py", env, env, 2, 60, "flydsl-table", None, 0.03, False)
    assert rep.stages["perf"].status == "skip"
    assert rep.verdict() == "INCONCLUSIVE"


# --------------------------------------------------------------------------------------
# Report contract
# --------------------------------------------------------------------------------------


def test_every_declared_stage_survives_into_the_report():
    d = vp.Report("t").as_dict()
    for name in ("merge_sim", "gpu_claim", "runtime_compat", "test_policy", "correctness", "perf", "diff_scan"):
        assert isinstance(d["stages"][name], dict)
        assert d["stages"][name]["status"] in {"pass", "fail", "skip", "info"}


def test_incomplete_run_cannot_report_pass():
    rep = vp.Report("t")
    for k in ("merge_sim", "gpu_claim", "runtime_compat", "test_policy", "correctness"):
        rep.stages[k].status = "pass"
    rep.stages["diff_scan"].status = "info"
    rep.stages["perf"].status = "skip"  # perf did not run
    assert rep.verdict() == "INCONCLUSIVE", "a skipped perf stage must not be reported as PASS"

    rep.stages["perf"].status = "pass"
    assert rep.verdict() == "PASS"


def test_a_perf_regression_produces_block():
    rep = vp.Report("t")
    for k in ("merge_sim", "gpu_claim", "runtime_compat", "test_policy", "correctness"):
        rep.stages[k].status = "pass"
    rep.stages["diff_scan"].status = "info"
    rep.stages["perf"].status = "fail"
    rep.finding("blocker", "perf", "layernorm regressed 18%")
    assert rep.verdict() == "BLOCK"
