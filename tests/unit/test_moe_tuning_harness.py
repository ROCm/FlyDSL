# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend-agnostic tests for the MXFP4 MoE tuning harness, spec, and ledger.

These exercise the pure host-side logic (decision predicates, stage-us parsing,
metric computation, provenance gating, attempt-ledger validation, and per-point
Pareto comparison) with no GPU and no compile.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
for p in (_REPO_ROOT, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import moe_tuning_harness as harness  # noqa: E402
import moe_tuning_ledger as ledger  # noqa: E402

from kernels import moe_tuning_spec as spec  # noqa: E402

pytestmark = pytest.mark.l0_backend_agnostic


# --- spec: locked values + predicates --------------------------------------


def test_locked_constants():
    assert spec.FP4_PEAK_TFLOPS == 4523.0
    assert spec.WIN_MARGIN == 0.10
    assert spec.REGRESSION_REL == 0.02
    assert spec.ABS_US_BAND == 2.0
    assert spec.WARMUP_ITERS == 10
    assert spec.BENCH_ITERS == 100
    assert spec.MFU_TARGET_BUCKETS == (16384, 32768)
    assert spec.LARGE_TOKEN_MIN == 4096
    assert spec.SMALL_TOKEN_MAX == 64
    assert spec.TARGET_ARCH == "gfx950"


def test_token_grids():
    assert spec.TOKEN_GRID_FULL[0] == 1 and spec.TOKEN_GRID_FULL[-1] == 32768
    assert len(spec.TOKEN_GRID_FULL) == 16
    assert spec.TOKEN_GRID_GPTOSS[0] == 256 and spec.TOKEN_GRID_GPTOSS[-1] == 32768


def test_models_in_scope_dtypes():
    by_name = {m.name: m for m in spec.MODELS}
    assert set(by_name) == {"deepseek_v3", "deepseek_v4", "kimi_k2", "gpt_oss"}
    # DeepSeek V4 is a8w4-only; i4 excluded everywhere.
    assert by_name["deepseek_v4"].dtypes == ("a8w4",)
    assert by_name["kimi_k2"].dtypes == ("a4w4", "a8w4")
    assert all("i4" not in m.dtypes for m in spec.MODELS)
    assert by_name["gpt_oss"].act == "swiglu"
    assert by_name["deepseek_v4"].model_dim == 7168 and by_name["deepseek_v4"].inter_dim == 512


def test_regression_predicate_requires_both_bands():
    # 1.5% over but only +1.5us: relative under 2%? 1.5% < 2% -> not a regression.
    assert not spec.is_regression(100.0, 101.5)
    # 3% over but only +0.3us absolute (small base): abs band not exceeded -> not a regression.
    assert not spec.is_regression(10.0, 10.3)
    # 5% over AND +5us: both bands exceeded -> regression.
    assert spec.is_regression(100.0, 105.0)
    # exactly at boundaries (strict >): 102.0 and +2.0 -> not a regression.
    assert not spec.is_regression(100.0, 102.0)


def test_large_shape_win_predicate():
    assert spec.is_large_shape_win(0.50, 0.55)  # exactly +10%
    assert not spec.is_large_shape_win(0.50, 0.549)


def test_small_token_win_predicate():
    # 12% faster AND >= 2us absolute -> win.
    assert spec.is_small_token_win(100.0, 88.0)
    # 12% faster but only 0.6us absolute (tiny base) -> rejected (abs floor).
    assert not spec.is_small_token_win(5.0, 4.4)
    # 8% faster -> rejected (under 10%).
    assert not spec.is_small_token_win(100.0, 92.0)


def test_effective_tflops_and_mfu_formula():
    # token*model_dim*inter_dim*3*topk*2 / us / 1e6
    tflops = spec.effective_tflops(4096, 7168, 256, 9, combined_us=1000.0)
    expected = 4096 * 7168 * 256 * 3 * 9 * 2 / 1000.0 / 1e6
    assert abs(tflops - expected) < 1e-9
    assert abs(spec.mfu(tflops) - tflops / 4523.0) < 1e-12


# --- harness: parsing / metrics / provenance -------------------------------


def test_parse_flydsl_stage_us():
    stdout = (
        "noise\n"
        "FlyDSL MoE stage1[fp4]: 1163.2 us, 1654.24 TFLOPS(logical, M=4608), 0.377 TB/s (doweight_stage1=False)\n"
        "FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | 7168x2048, E=32, K=8, M_eff=4608 | 845.5 us, 1200.00 TFLOPS, 0.300 TB/s\n"
        "FlyDSL MoE stage2 [moe_gemm2] fp4 reduce | 7168x2048, E=32, K=8, M_eff=4608 | 900.1 us, 1100.00 TFLOPS, 0.280 TB/s\n"
    )
    got = harness.parse_flydsl_stage_us(stdout)
    assert got["stage1_us"] == 1163.2
    # last matching stage2 line wins
    assert got["stage2_us"] == 900.1


def test_parse_flydsl_stage_us_missing():
    got = harness.parse_flydsl_stage_us("nothing here")
    assert got["stage1_us"] is None and got["stage2_us"] is None


def test_parse_aiter_output_pass_warning_line():
    out = (
        "calling test_fmoe(...)\n"
        "ck_moe_2stages:  234.56 us,   654.00 tflops......(quant:fp4x2)[checkAllclose passed~]\n"
        "logits_diff: 0.0008\n"
    )
    res = harness.parse_aiter_output(out)
    assert res["e2e_us"] == 234.56
    assert res["logits_diff"] == 0.0008
    assert res["correctness_pass"] is True


def test_parse_aiter_output_pass_markdown_row():
    # logits_diff below 1e-3 prints no warning line; it only appears in the
    # summary markdown row.  The loose "checkAllclose ... failed!" line is the
    # EXPECTED fp4 elementwise warning and must NOT fail correctness.
    out = (
        "ck_moe_2stages:   84.32 us,  18.80 tflops......(quant:fp4x2)[checkAllclose atol=0.01 rtol=0.01 failed!]\n"
        "moe_2stage summary (markdown):\n"
        "| dtype | token | ... |      us |   logits_diff | model   |\n"
        "|:------|------:| ... |--------:|--------------:|:--------|\n"
        "| torch.bfloat16 | 16 | ... | 87.195 |    9.6236e-06 | legacy  |\n"
    )
    res = harness.parse_aiter_output(out)
    assert res["e2e_us"] == 84.32
    assert res["logits_diff"] == 9.6236e-06
    assert res["correctness_pass"] is True


def test_parse_aiter_output_fail_cases():
    # logits over 0.01 (markdown row) -> fail.
    out_logits = "ck_moe_2stages:  100.00 us, 100.00 tflops\n" "| torch.bfloat16 | 16 | ... | 100.0 | 0.05 | legacy |\n"
    assert harness.parse_aiter_output(out_logits)["correctness_pass"] is False
    # hard assertion text -> fail even if a number was produced.
    out_assert = "ck_moe_2stages:  100.00 us\naccuracy check failed: err=1, logits_diff=0.2\n"
    assert harness.parse_aiter_output(out_assert)["correctness_pass"] is False
    # no logits at all -> fail (cannot confirm correctness).
    out_no_logits = "ck_moe_2stages:  100.00 us, 100.00 tflops\n"
    assert harness.parse_aiter_output(out_no_logits)["correctness_pass"] is False
    # no e2e number at all -> fail.
    assert harness.parse_aiter_output("nothing")["correctness_pass"] is False


def test_aiter_cmd_is_strict_aot_model_correct():
    # The aiter guardrail must use the strict/AOT/model-correct runner
    # (scripts/aiter_strict_point.py), NOT the non-strict legacy CLI, and must
    # carry the model's true act/gate, locked warmup/iters, and AOT enabled.
    rp = harness.RunPoint("kimi_k2", 7168, 256, 384, 8, "silu", "a4w4", 16)
    cmd = harness._aiter_cmd(rp)
    joined = " ".join(cmd)
    assert "aiter_strict_point.py" in joined
    # Must NOT be the legacy CLI path.
    assert "test_moe_2stage.py" not in joined
    assert "--no-flydsl-csv" not in cmd
    assert cmd[cmd.index("--aq") + 1] == "fp4"  # a4w4 -> fp4 activation
    assert cmd[cmd.index("--act") + 1] == "silu"
    assert cmd[cmd.index("--gate") + 1] == "separated"
    assert cmd[cmd.index("--warmup") + 1] == "10"
    assert cmd[cmd.index("--iters") + 1] == "100"
    assert "--no-aot" not in cmd  # AOT cache check ON by default
    assert cmd[cmd.index("-t") + 1] == "16"
    # a8w4 -> fp8 activation; swiglu model carries swiglu act.
    rpg = harness.RunPoint("gpt_oss", 3072, 3072, 128, 4, "swiglu", "a8w4", 512)
    cmdg = harness._aiter_cmd(rpg)
    assert cmdg[cmdg.index("--aq") + 1] == "fp8"
    assert cmdg[cmdg.index("--act") + 1] == "swiglu"
    # --no-aot toggle is honored.
    assert "--no-aot" in harness._aiter_cmd(rp, check_aot=False)


def test_parse_strict_aiter_output():
    ok = (
        'noise\nSTRICT_RESULT {"e2e_us": 80.7, "e2e_us_p95": 84.0, "logits_diff": 1.0e-05, '
        '"correctness_pass": true, "check_aot_cache": true, "error_category": ""}\n'
    )
    r = harness.parse_strict_aiter_output(ok)
    assert r["e2e_us"] == 80.7 and r["e2e_us_p95"] == 84.0 and r["correctness_pass"] is True
    assert r["aot_status"] == "checked"
    fail = (
        'STRICT_RESULT {"error": "AssertionError: accuracy check failed", '
        '"error_category": "correctness", "correctness_pass": false, "check_aot_cache": false}\n'
    )
    rf = harness.parse_strict_aiter_output(fail)
    assert rf["correctness_pass"] is False and "AssertionError" in rf["error"]
    assert rf["error_category"] == "correctness" and rf["aot_status"] == "no_aot"
    miss = harness.parse_strict_aiter_output("no result here")
    assert miss["correctness_pass"] is False and miss["error"] == "no_strict_result"


def test_parse_flydsl_stage_p95():
    stdout = (
        "FlyDSL MoE stage1[fp4]: 100.0 us, p95=105.0 us 1654.24 TFLOPS(logical, M=144), 4.0 TB/s (x)\n"
        "FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | 7168x256, ... | 50.0 us, p95=55.0 us 1200.0 TFLOPS, 3.0 TB/s\n"
    )
    g = harness.parse_flydsl_stage_us(stdout)
    assert g["stage1_us"] == 100.0 and g["stage1_p95"] == 105.0
    assert g["stage2_us"] == 50.0 and g["stage2_p95"] == 55.0
    # Without the p95 suffix, the p95 fields are None but median us still parses.
    g2 = harness.parse_flydsl_stage_us("FlyDSL MoE stage1[fp4]: 100.0 us, 1.0 TFLOPS(logical, M=1), 4.0 TB/s (x)\n")
    assert g2["stage1_us"] == 100.0 and g2["stage1_p95"] is None


# --- run-list coverage (full token grid from spec) -------------------------


def test_run_list_covers_full_dec6_grid():
    rl = harness.build_run_list()
    # DS V3 (16 tok x 2 dtype) + DS V4 (16 x 1) + Kimi (16 x 2) + GPT-OSS (8 x 2)
    assert len(rl) == 16 * 2 + 16 * 1 + 16 * 2 + 8 * 2 == 96
    keys = harness.expected_point_keys()
    # DeepSeek V4 is a8w4-only.
    assert ("deepseek_v4", "a8w4", "silu", "1") in keys
    assert ("deepseek_v4", "a4w4", "silu", "1") not in keys
    # GPT-OSS has no tiny-token regime; starts at 256.
    assert ("gpt_oss", "a4w4", "swiglu", "256") in keys
    assert ("gpt_oss", "a4w4", "swiglu", "1") not in keys
    # full small + large coverage for a skinny model.
    for tok in (1, 16, 64, 4096, 16384, 32768):
        assert ("kimi_k2", "a4w4", "silu", str(tok)) in keys


# --- baseline validation gate (negative tests) ------------------------


def _good_baseline_row(**over):
    row = {
        "gpu_id": "0",
        "gpu_model": "MI350X",
        "branch": "rlcr/mxfp4-moe",
        "commit": "523ca1c7deadbeef",
        "command": "python3 test_moe_gemm.py ... ; python3 test_moe_2stage.py ...",
        "warmup": "10",
        "iters": "100",
        "idle_gpu_verified": "True",
        "graph_capture": "False",
        "l2_flush_per_iter": "True",
        "clocks_pinned": "True",
        "model": "kimi_k2",
        "dtype": "a4w4",
        "act": "silu",
        "token": "16",
        # All required metric fields present and numeric.
        "stage1_us": "55.3",
        "stage2_us": "21.8",
        "sorting_us": "0.0",
        "kernel_path_us": "77.1",
        "kernel_path_us_p95": "79.0",
        "effective_tflops": "12.3",
        "mfu": "0.0027",
        "e2e_us": "150.0",
        "e2e_us_p95": "155.0",
        "logits_diff": "0.0008",
        "correctness_pass": "True",
    }
    row.update(over)
    return row


def test_validate_baseline_row_accepts_good_row():
    assert harness.validate_baseline_row(_good_baseline_row()) == []


@pytest.mark.parametrize(
    "over,expect",
    [
        ({"commit": "abc123"}, "commit_not_523ca1c7"),
        ({"commit": ""}, "missing_commit"),
        ({"idle_gpu_verified": "False"}, "idle_gpu_not_verified"),
        ({"command": ""}, "missing_command"),
        ({"dtype": ""}, "missing_dtype"),
        ({"act": ""}, "missing_act"),
        ({"e2e_us": ""}, "missing_e2e_us"),
        ({"logits_diff": ""}, "missing_logits_diff"),
        # Hardened metric-field requirements.
        ({"stage1_us": ""}, "missing_stage1_us"),
        ({"stage2_us": ""}, "missing_stage2_us"),
        ({"sorting_us": ""}, "missing_sorting_us"),
        ({"kernel_path_us": ""}, "missing_kernel_path_us"),
        ({"kernel_path_us_p95": ""}, "missing_kernel_path_us_p95"),
        ({"effective_tflops": ""}, "missing_effective_tflops"),
        ({"mfu": ""}, "missing_mfu"),
        ({"e2e_us_p95": ""}, "missing_e2e_us_p95"),
        ({"kernel_path_us": "not-a-number"}, "missing_kernel_path_us"),
        ({"correctness_pass": "False"}, "correctness_not_passed"),
        ({"correctness_pass": ""}, "correctness_not_passed"),
        ({"warmup": "2"}, "warmup_mismatch"),
        ({"iters": "5"}, "iters_mismatch"),
        ({"graph_capture": "True"}, "graph_capture_must_be_off"),
        ({"l2_flush_per_iter": "False"}, "l2_flush_must_be_on"),
        ({"clocks_pinned": "False"}, "clocks_must_be_pinned"),
    ],
)
def test_validate_baseline_row_rejections(over, expect):
    reasons = harness.validate_baseline_row(_good_baseline_row(**over))
    assert expect in reasons


def test_validate_baseline_csv_missing_coverage(tmp_path):
    # A single fully-valid row is not enough; the full workload must be covered.
    out = tmp_path / "baseline.csv"
    p = harness.Provenance(
        gpu_id="0", gpu_model="MI350X", branch="b", commit="523ca1c7", idle_gpu_verified=True, clocks_pinned=True
    )
    row = harness.PointRow(
        provenance=p,
        command="cmd",
        model="kimi_k2",
        model_dim=7168,
        inter_dim=256,
        experts=384,
        topk=8,
        dtype="a4w4",
        act="silu",
        token=16,
        stage1_us=55.3,
        stage2_us=21.8,
        sorting_us=0.0,
        kernel_path_us=77.1,
        kernel_path_us_p95=79.0,
        effective_tflops=12.3,
        mfu=0.0027,
        e2e_us=150.0,
        e2e_us_p95=155.0,
        logits_diff=0.0008,
        correctness_pass=True,
    )
    harness.write_csv([row], str(out))
    res = harness.validate_baseline_csv(str(out))
    assert res["valid"] is False
    assert res["missing_points"]  # almost all points missing
    assert res["row_errors"] == {}  # the one present row is itself fully valid


def test_validate_baseline_csv_rejects_missing_kernel_metrics(tmp_path):
    # Regression: a full-coverage CSV with e2e/logits present
    # but kernel metrics empty must NOT validate.
    out = tmp_path / "baseline.csv"
    p = harness.Provenance(
        gpu_id="0", gpu_model="MI350X", branch="b", commit="523ca1c7", idle_gpu_verified=True, clocks_pinned=True
    )
    rows = []
    for rp in harness.build_run_list():
        rows.append(
            harness.PointRow(
                provenance=p,
                command="cmd",
                model=rp.model,
                model_dim=rp.model_dim,
                inter_dim=rp.inter_dim,
                experts=rp.experts,
                topk=rp.topk,
                dtype=rp.dtype,
                act=rp.act,
                token=rp.token,
                # kernel metrics deliberately omitted
                e2e_us=150.0,
                e2e_us_p95=155.0,
                logits_diff=0.0008,
                correctness_pass=True,
            )
        )
    harness.write_csv(rows, str(out))
    res = harness.validate_baseline_csv(str(out))
    assert res["valid"] is False
    assert not res["missing_points"]  # coverage is complete...
    assert res["row_errors"]  # ...but rows fail on missing kernel metrics
    some = next(iter(res["row_errors"].values()))
    assert "missing_kernel_path_us" in some and "missing_mfu" in some


def test_combined_and_metrics():
    combined = harness.combined_kernel_path_us(1000.0, 800.0, 50.0)
    assert combined == 1850.0
    m = harness.compute_metrics(token=4096, model_dim=7168, inter_dim=256, topk=9, combined_us=combined)
    assert m["effective_tflops"] > 0 and 0 < m["mfu"] < 10


def test_summarize_median_p95():
    s = harness.summarize([10, 11, 12, 13, 100])
    assert s["median"] == 12
    assert s["p95"] == 100


def test_provenance_missing_fields_gate():
    p = harness.Provenance()  # gpu_id/gpu_model/branch/commit unset
    missing = p.missing_fields()
    assert "gpu_id" in missing and "commit" in missing
    assert not p.is_complete()
    p2 = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="rlcr/mxfp4-moe", commit="deadbeef")
    assert p2.is_complete()


def test_pointrow_csv_dict_has_all_columns():
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="c")
    row = harness.PointRow(
        provenance=p,
        command="cmd",
        model="kimi_k2",
        model_dim=7168,
        inter_dim=256,
        experts=384,
        topk=8,
        dtype="a4w4",
        act="silu",
        token=4096,
    )
    d = row.to_csv_dict()
    assert set(d.keys()) == set(harness.CSV_COLUMNS)
    assert d["metric_formula"] == harness.METRIC_FORMULA


def test_write_csv_roundtrip(tmp_path):
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="c")
    rows = [
        harness.PointRow(
            provenance=p,
            command="cmd",
            model="kimi_k2",
            model_dim=7168,
            inter_dim=256,
            experts=384,
            topk=8,
            dtype="a4w4",
            act="silu",
            token=4096,
            kernel_path_us=1850.0,
            e2e_us=2000.0,
            mfu=0.5,
        )
    ]
    out = tmp_path / "baseline.csv"
    harness.write_csv(rows, str(out))
    text = out.read_text()
    assert "kernel_path_us" in text.splitlines()[0]
    assert "kimi_k2" in text


# --- ledger: attempt validation + comparison -------------------------------


def _complete_attempt(**over):
    base = dict(
        config={"tile_m": 64},
        stage=1,
        model="kimi_k2",
        dtype="a4w4",
        act="silu",
        gpu_id="0",
        gpu_model="MI350X",
        branch="b",
        commit="c",
        command="cmd",
        warmup=10,
        iters=100,
        result="loss",
    )
    base.update(over)
    return ledger.Attempt(**base)


def test_attempt_missing_provenance_rejected(tmp_path):
    bad = _complete_attempt(commit="")  # missing required field
    assert "commit" in bad.missing_fields()
    with pytest.raises(ValueError):
        ledger.append_attempt(bad, path=str(tmp_path / "attempts.jsonl"))


def test_attempt_append_roundtrip(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    rec = ledger.append_attempt(_complete_attempt(result="win"), path=path, now=123.0)
    assert rec["timestamp"] == 123.0
    lines = open(path).read().strip().splitlines()
    assert len(lines) == 1 and '"result": "win"' in lines[0]


def _complete_rejected(**over):
    base = dict(
        model="kimi_k2",
        dtype="a4w4",
        act="silu",
        token=64,
        stage=0,
        config={"tile_m1": 16},
        reason="illegal candidate tiles: s1=fp4 tile_m<32",
        selection={"model": "kimi_k2", "dtype": "a4w4", "tokens": [64]},
        gpu_id="0",
        gpu_model="MI350X",
        branch="b",
        commit="c",
        command="python3 scripts/moe_tuning_harness.py candidate --tile-m1 16",
        warmup=10,
        iters=100,
        csv_path="",  # present-but-empty: no measured artifact pre-compile
        profile_path="",
    )
    base.update(over)
    return base


def test_rejected_candidate_full_provenance_roundtrip(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    rec = ledger.append_rejected_candidate(_complete_rejected(), path=path, now=7.0)
    assert rec["result"] == "rejected_candidate" and rec["timestamp"] == 7.0
    # csv_path/profile_path are present (empty allowed); selection is a non-empty dict.
    assert rec["csv_path"] == "" and rec["profile_path"] == "" and rec["selection"]
    # stage 0 is a valid value (candidate-tile rejection spanning both stages).
    rec0 = ledger.append_rejected_candidate(_complete_rejected(stage=0), path=path, now=8.0)
    assert rec0["stage"] == 0
    lines = open(path).read().strip().splitlines()
    assert len(lines) == 2


def test_rejected_candidate_missing_provenance_rejected(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    # Each required (non-empty) provenance field, when blanked, must be refused.
    for field in ("act", "gpu_id", "gpu_model", "branch", "commit", "command", "warmup", "iters"):
        bad = _complete_rejected(**{field: ""})
        with pytest.raises(ValueError, match="missing fields"):
            ledger.append_rejected_candidate(bad, path=path)
    # csv_path/profile_path keys must EXIST even though empty is allowed: drop them.
    for field in ("csv_path", "profile_path"):
        bad = _complete_rejected()
        del bad[field]
        with pytest.raises(ValueError, match="missing fields"):
            ledger.append_rejected_candidate(bad, path=path)
    # selection None/"" trips the missing-fields gate; {} / non-dict trips the
    # dedicated selection gate.
    for sel in (None, ""):
        with pytest.raises(ValueError, match="missing fields"):
            ledger.append_rejected_candidate(_complete_rejected(selection=sel), path=path)
    for sel in ({}, "a4w4"):
        with pytest.raises(ValueError, match="selection"):
            ledger.append_rejected_candidate(_complete_rejected(selection=sel), path=path)
    # The minimal-only record (the old contract) is now rejected.
    with pytest.raises(ValueError, match="missing fields"):
        ledger.append_rejected_candidate(
            {"model": "kimi_k2", "dtype": "a4w4", "token": 64, "config": {}, "reason": "x"}, path=path
        )
    # No partial file should have been written.
    assert not os.path.exists(path)


def test_committed_rejected_records_are_contract_complete():
    """Every committed rejected_candidate record must carry full provenance, unless
    it is an explicitly superseded pre-contract artifact (marked superseded_by)."""
    import json as _json

    attempts = os.path.join(_REPO_ROOT, "docs", "attempts.jsonl")
    if not os.path.exists(attempts):
        pytest.skip("no committed attempts ledger")
    required = set(ledger.REQUIRED_REJECTED_FIELDS)
    present_keys = set(ledger.REQUIRED_REJECTED_PRESENT_KEYS)
    offenders = []
    for ln in open(attempts):
        ln = ln.strip()
        if not ln:
            continue
        rec = _json.loads(ln)
        if rec.get("result") != "rejected_candidate":
            continue
        if "superseded_by" in rec:  # incomplete historical record, explicitly invalidated
            continue
        missing = [k for k in required if rec.get(k) in (None, "")]
        missing += [k for k in present_keys if k not in rec]
        sel = rec.get("selection")
        if not isinstance(sel, dict) or not sel:
            missing.append("selection")
        if missing:
            offenders.append((rec.get("timestamp"), missing))
    assert not offenders, f"incomplete committed rejected records: {offenders}"


def _csv(path, rows):
    import csv as _c

    with open(path, "w", newline="") as f:
        w = _c.DictWriter(f, fieldnames=["model", "dtype", "act", "token", "kernel_path_us", "e2e_us", "mfu"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _gate_csv(path, rows):
    import csv as _c

    cols = [
        "model",
        "dtype",
        "act",
        "token",
        "kernel_path_us",
        "e2e_us",
        "aot_status",
        "correctness_pass",
        "logits_diff",
    ]
    with open(path, "w", newline="") as f:
        w = _c.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _gate_row(**over):
    base = dict(
        model="kimi_k2",
        dtype="a4w4",
        act="silu",
        token=16,
        kernel_path_us=150.0,
        e2e_us=80.0,
        aot_status="checked",
        correctness_pass=True,
        logits_diff=0.001,
    )
    base.update(over)
    return base


def test_selected_candidate_gate_accepts_checked_correct(tmp_path):
    path = str(tmp_path / "cand.csv")
    _gate_csv(path, [_gate_row(token=16), _gate_row(token=16384, kernel_path_us=1700, e2e_us=1500)])
    res = ledger.selected_candidate_gate(path)
    assert res["passed"] is True and res["n_rows"] == 2 and res["violations"] == []


def test_selected_candidate_gate_rejects_no_aot_and_bad_correctness(tmp_path):
    # no_aot row (repeatability/diagnostic bypass) can never be promoted to a win.
    p1 = str(tmp_path / "no_aot.csv")
    _gate_csv(p1, [_gate_row(aot_status="no_aot")])
    r1 = ledger.selected_candidate_gate(p1)
    assert r1["passed"] is False and any("aot_status" in v[1] for v in r1["violations"])

    # failed correctness rejected.
    p2 = str(tmp_path / "bad_correct.csv")
    _gate_csv(p2, [_gate_row(correctness_pass=False)])
    r2 = ledger.selected_candidate_gate(p2)
    assert r2["passed"] is False and any("correctness_pass" in v[1] for v in r2["violations"])

    # logits over threshold rejected.
    p3 = str(tmp_path / "bad_logits.csv")
    _gate_csv(p3, [_gate_row(logits_diff=0.05)])
    r3 = ledger.selected_candidate_gate(p3)
    assert r3["passed"] is False and any("logits_diff" in v[1] for v in r3["violations"])

    # empty CSV: nothing to promote -> not passed.
    p4 = str(tmp_path / "empty.csv")
    _gate_csv(p4, [])
    assert ledger.selected_candidate_gate(p4)["passed"] is False


def test_scan_replay_consistency(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    import json as _json

    def _write(recs):
        with open(path, "w") as f:
            for r in recs:
                f.write(_json.dumps(r) + "\n")

    # multi-file attempt whose command replays BOTH files -> clean.
    good = {
        "result": "neutral",
        "csv_path": "docs/a.csv;docs/b.csv",
        "command": "h candidate --out docs/a.csv ; h candidate --out docs/b.csv ; repeatability_check",
        "timestamp": 1.0,
    }
    _write([good])
    assert ledger.scan_replay_consistency(path) == []

    # command misses b.csv -> offender.
    bad = dict(good, command="h candidate --out docs/a.csv", timestamp=2.0)
    _write([bad])
    off = ledger.scan_replay_consistency(path)
    assert off and off[0][0] == 2.0 and "docs/b.csv" in off[0][1]

    # brace shorthand does not literally contain either file -> offender.
    brace = dict(good, command="h candidate --out docs/{a,b}.csv", timestamp=3.0)
    _write([brace])
    assert ledger.scan_replay_consistency(path)

    # required file hidden behind a '#' comment -> offender.
    commented = dict(good, command="h candidate --out docs/a.csv  # then docs/b.csv", timestamp=4.0)
    _write([commented])
    assert ledger.scan_replay_consistency(path)

    # superseded records are skipped.
    superseded = dict(bad, superseded_by=9.0, timestamp=5.0)
    _write([superseded])
    assert ledger.scan_replay_consistency(path) == []


def test_committed_repeatability_attempts_replayable():
    """Committed multi-file repeatability attempts must replay all their CSVs."""
    attempts = os.path.join(_REPO_ROOT, "docs", "attempts.jsonl")
    if not os.path.exists(attempts):
        pytest.skip("no committed attempts ledger")
    off = ledger.scan_replay_consistency(attempts)
    assert off == [], f"non-replayable committed repeatability attempts: {off}"


def test_scan_duplicate_rejected_candidates(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    import json as _json

    def _probe(ts, sup=None):
        r = {
            "result": "rejected_candidate",
            "model": "deepseek_v3",
            "dtype": "a4w4",
            "act": "silu",
            "token": 32,
            "config": {"tile_m1": 256, "tile_n1": 32},
            "reason": "x",
            "timestamp": ts,
        }
        if sup is not None:
            r["superseded_by"] = sup
        return r

    # Two ACTIVE records for the same probe -> duplicate.
    open(path, "w").write(_json.dumps(_probe(1.0)) + "\n" + _json.dumps(_probe(2.0)) + "\n")
    dups = ledger.scan_duplicate_rejected_candidates(path)
    assert dups and sorted(dups[0][1]) == [1.0, 2.0]

    # Superseding the older one leaves exactly one active -> clean.
    open(path, "w").write(_json.dumps(_probe(1.0, sup=2.0)) + "\n" + _json.dumps(_probe(2.0)) + "\n")
    assert ledger.scan_duplicate_rejected_candidates(path) == []


def test_committed_rejected_candidates_unique():
    """Committed ledger must have exactly one active rejected record per probe."""
    attempts = os.path.join(_REPO_ROOT, "docs", "attempts.jsonl")
    if not os.path.exists(attempts):
        pytest.skip("no committed attempts ledger")
    dups = ledger.scan_duplicate_rejected_candidates(attempts)
    assert dups == [], f"duplicate active rejected-candidate records: {dups}"


def test_scan_superseded_rejected_candidates(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    import json as _json

    def _probe(ts, n, sup=None):
        r = {
            "result": "rejected_candidate",
            "model": "deepseek_v3",
            "dtype": "a4w4",
            "act": "silu",
            "token": 32,
            "config": {"tile_m1": 256, "tile_n1": n},
            "reason": "x",
            "timestamp": ts,
        }
        if sup is not None:
            r["superseded_by"] = sup
        return r

    # superseded record links to the matching active record of the SAME key -> clean.
    open(path, "w").write(_json.dumps(_probe(1.0, 32, sup=2.0)) + "\n" + _json.dumps(_probe(2.0, 32)) + "\n")
    assert ledger.scan_superseded_rejected_candidates(path) == []

    # superseded record links to a DIFFERENT probe's active record -> offender.
    open(path, "w").write(
        _json.dumps(_probe(1.0, 32, sup=3.0))  # links to the n=64 record, wrong key
        + "\n"
        + _json.dumps(_probe(2.0, 32))
        + "\n"
        + _json.dumps(_probe(3.0, 64))
        + "\n"
    )
    off = ledger.scan_superseded_rejected_candidates(path)
    assert off and off[0][0] == 1.0


def test_committed_superseded_links_valid():
    """Every committed superseded rejected record must link to an active record of the same key."""
    attempts = os.path.join(_REPO_ROOT, "docs", "attempts.jsonl")
    if not os.path.exists(attempts):
        pytest.skip("no committed attempts ledger")
    off = ledger.scan_superseded_rejected_candidates(attempts)
    assert off == [], f"superseded records linking to the wrong/no successor: {off}"


def test_row_missing_kernel_path():
    rp = harness.RunPoint("deepseek_v3", 7168, 256, 257, 9, "silu", "a4w4", 32)
    prov = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="c")
    # A row with no parsed stage times is "missing" (the tile_n=512 / tile_k!=256 case).
    blank = harness.PointRow(
        provenance=prov,
        command="x",
        model=rp.model,
        model_dim=rp.model_dim,
        inter_dim=rp.inter_dim,
        experts=rp.experts,
        topk=rp.topk,
        dtype=rp.dtype,
        act=rp.act,
        token=rp.token,
    )
    assert harness.row_missing_kernel_path(blank) is True
    # A row with kernel-path populated is not missing.
    blank.stage1_us = 90.0
    blank.stage2_us = 70.0
    blank.kernel_path_us = 160.0
    assert harness.row_missing_kernel_path(blank) is False


def test_compare_csvs_detects_regression_and_wins(tmp_path):
    base = str(tmp_path / "base.csv")
    cand = str(tmp_path / "cand.csv")
    _csv(
        base,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1000,
                "e2e_us": 1200,
                "mfu": 0.50,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 100,
                "e2e_us": 150,
                "mfu": 0.05,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 500,
                "e2e_us": 600,
                "mfu": 0.30,
            },
        ],
    )
    _csv(
        cand,
        [
            # large bucket: +10% MFU win, no kernel-path regression
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 950,
                "e2e_us": 1180,
                "mfu": 0.56,
            },
            # small token: 20% faster and >=2us -> win
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 80,
                "e2e_us": 150,
                "mfu": 0.05,
            },
            # mid token: regression on kernel-path (+10% and +50us)
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 550,
                "e2e_us": 600,
                "mfu": 0.30,
            },
        ],
    )
    cv = ledger.compare_csvs(base, cand)
    assert cv.any_regression is True  # the 128-token point regressed
    assert cv.coverage_complete  # candidate covers all 3 baseline points
    assert not cv.pareto_clean
    assert ("kimi_k2", "a4w4", "silu", "16384") in cv.large_wins
    assert ("kimi_k2", "a4w4", "silu", "16") in cv.small_wins


def _gated_compare_csv(path, rows):
    """Write a candidate/baseline CSV that ALSO carries the gate columns."""
    import csv as _c

    cols = [
        "model",
        "dtype",
        "act",
        "token",
        "kernel_path_us",
        "e2e_us",
        "mfu",
        "aot_status",
        "correctness_pass",
        "logits_diff",
    ]
    with open(path, "w", newline="") as f:
        w = _c.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _two_point_baseline_and_candidate(tmp_path, aot_status):
    """A fully-covered, non-regressing, otherwise-WINNING 2-point candidate whose
    gate columns are parameterized by ``aot_status``."""
    base = str(tmp_path / "base.csv")
    cand = str(tmp_path / "cand.csv")
    bl = [
        dict(
            model="kimi_k2",
            dtype="a4w4",
            act="silu",
            token=16384,
            kernel_path_us=1000,
            e2e_us=1200,
            mfu=0.50,
            aot_status="checked",
            correctness_pass=True,
            logits_diff=0.001,
        ),
        dict(
            model="kimi_k2",
            dtype="a4w4",
            act="silu",
            token=16,
            kernel_path_us=100,
            e2e_us=150,
            mfu=0.05,
            aot_status="checked",
            correctness_pass=True,
            logits_diff=0.001,
        ),
    ]
    # candidate: +12% MFU at 16384 (large win), 20% faster at 16 (small win), no regressions
    cd = [
        dict(
            model="kimi_k2",
            dtype="a4w4",
            act="silu",
            token=16384,
            kernel_path_us=950,
            e2e_us=1180,
            mfu=0.56,
            aot_status=aot_status,
            correctness_pass=True,
            logits_diff=0.001,
        ),
        dict(
            model="kimi_k2",
            dtype="a4w4",
            act="silu",
            token=16,
            kernel_path_us=80,
            e2e_us=150,
            mfu=0.05,
            aot_status=aot_status,
            correctness_pass=True,
            logits_diff=0.001,
        ),
    ]
    _gated_compare_csv(base, bl)
    _gated_compare_csv(cand, cd)
    return base, cand


def test_claimable_win_blocks_no_aot_winning_candidate(tmp_path):
    # The leak Codex flagged: an otherwise-winning, fully-covered, non-regressing
    # candidate measured with --no-aot-check must NOT be promotable.
    base, cand = _two_point_baseline_and_candidate(tmp_path, aot_status="no_aot")
    cv = ledger.compare_csvs(base, cand)
    # metrics still look winning...
    assert cv.pareto_clean is True
    assert cv.large_wins and cv.small_wins
    # ...but the hard gate fails, so the candidate is NOT claimable.
    assert cv.gate["passed"] is False
    assert cv.claimable_win is False
    # and the standalone gate agrees.
    assert ledger.selected_candidate_gate(cand)["passed"] is False


def test_claimable_win_allows_checked_correct_candidate(tmp_path):
    base, cand = _two_point_baseline_and_candidate(tmp_path, aot_status="checked")
    cv = ledger.compare_csvs(base, cand)
    assert cv.pareto_clean is True
    assert cv.large_wins and cv.small_wins
    assert cv.gate["passed"] is True
    assert cv.claimable_win is True


def test_compare_csvs_rejects_cherry_picked_candidate(tmp_path):
    # Baseline has 3 points; candidate reports only the single winning large
    # point and omits the others.  Coverage must be incomplete and the verdict
    # must NOT be pareto_clean -- a cherry-picked win cannot pass.
    base = str(tmp_path / "base.csv")
    cand = str(tmp_path / "cand.csv")
    _csv(
        base,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1000,
                "e2e_us": 1200,
                "mfu": 0.50,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 100,
                "e2e_us": 150,
                "mfu": 0.05,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 500,
                "e2e_us": 600,
                "mfu": 0.30,
            },
        ],
    )
    _csv(
        cand,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 900,
                "e2e_us": 1100,
                "mfu": 0.56,
            },
        ],
    )
    cv = ledger.compare_csvs(base, cand)
    assert not cv.coverage_complete
    assert ("kimi_k2", "a4w4", "silu", "16") in cv.missing_candidate_points
    assert ("kimi_k2", "a4w4", "silu", "128") in cv.missing_candidate_points
    assert not cv.pareto_clean  # forced False by incomplete coverage


def test_compare_csvs_rejects_missing_regime_fields(tmp_path):
    # Candidate covers every point but the large target bucket lacks mfu, and a
    # point lacks e2e.  Those points are incomplete -> not pareto_clean.
    base = str(tmp_path / "base.csv")
    cand = str(tmp_path / "cand.csv")
    _csv(
        base,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1000,
                "e2e_us": 1200,
                "mfu": 0.50,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 500,
                "e2e_us": 600,
                "mfu": 0.30,
            },
        ],
    )
    _csv(
        cand,
        [
            # large bucket missing mfu
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 900,
                "e2e_us": 1100,
                "mfu": "",
            },
            # mid point missing e2e
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 480,
                "e2e_us": "",
                "mfu": 0.30,
            },
        ],
    )
    cv = ledger.compare_csvs(base, cand)
    assert not cv.coverage_complete
    assert ("kimi_k2", "a4w4", "silu", "16384") in cv.incomplete_points
    assert ("kimi_k2", "a4w4", "silu", "128") in cv.incomplete_points
    assert not cv.pareto_clean


def test_repeatability_check(tmp_path):
    a = str(tmp_path / "a.csv")
    b = str(tmp_path / "b.csv")
    _csv(
        a,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1000,
                "e2e_us": 1200,
                "mfu": 0.5,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 100,
                "e2e_us": 150,
                "mfu": 0.05,
            },
        ],
    )
    # b: first point within band (1.5% < 2% and +15us... wait 15us>2us, so need <=max(2%*1000=20us,2us)=20us -> 1015 ok),
    # second point unstable (+10us on a 100us base -> band=max(2us,2us)=2us, 10>2 -> unstable).
    _csv(
        b,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1015,
                "e2e_us": 1210,
                "mfu": 0.5,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 110,
                "e2e_us": 150,
                "mfu": 0.05,
            },
        ],
    )
    res = ledger.repeatability_check(a, b)
    assert res["n_shared"] == 2
    assert not res["stable"]  # the 16-token kernel_path drifted > band
    assert any(u[0] == ("kimi_k2", "a4w4", "silu", "16") for u in res["unstable"]["kernel_path_us"])
    # 16384 kernel_path within band, e2e within band -> not flagged.
    assert all(u[0] != ("kimi_k2", "a4w4", "silu", "16384") for u in res["unstable"]["kernel_path_us"])


def test_quarantine_and_validated_keys():
    from kernels import moe_tuning_spec as spec

    # ALL a8w4 shapes are correctness-quarantined (the non-fp4-activation
    # e2e path fails the aiter correctness gate for fp8 AND bf16 activation; only
    # fp4 activation passes).  DS V3 a8w4 is included (its earlier legacy-path "pass" was the
    # legacy-Swiglu artifact, not a real Silu a8w4 pass).
    assert spec.is_quarantined("deepseek_v3", "a8w4")
    assert spec.is_quarantined("deepseek_v4", "a8w4")
    assert spec.is_quarantined("kimi_k2", "a8w4")
    assert spec.is_quarantined("gpt_oss", "a8w4")
    # a4w4 is NOT quarantined for any model.
    assert not spec.is_quarantined("deepseek_v3", "a4w4")
    assert not spec.is_quarantined("kimi_k2", "a4w4")

    vkeys = spec.validated_point_keys()
    # Validated = all a4w4: DS V3 (16) + Kimi (16) + GPT-OSS (8) = 40.
    assert len(vkeys) == 40
    assert ("deepseek_v3", "a4w4", "silu", "1") in vkeys
    assert ("deepseek_v3", "a8w4", "silu", "1") not in vkeys  # quarantined
    assert ("kimi_k2", "a8w4", "silu", "1") not in vkeys  # quarantined
    assert ("gpt_oss", "a8w4", "swiglu", "256") not in vkeys  # quarantined
    # validated subset is a strict subset of the full workload.
    assert vkeys < harness.expected_point_keys()


def test_validate_baseline_csv_subset_keys(tmp_path):
    # A CSV covering only the validated subset validates against validated keys,
    # but fails against the full workload (missing the quarantined points).
    from kernels import moe_tuning_spec as spec

    out = tmp_path / "sub.csv"
    p = harness.Provenance(
        gpu_id="0", gpu_model="MI350X", branch="b", commit="523ca1c7", idle_gpu_verified=True, clocks_pinned=True
    )
    rows = []
    for key in spec.validated_point_keys():
        model, dtype, act, token = key
        rows.append(
            harness.PointRow(
                provenance=p,
                command="cmd",
                model=model,
                model_dim=7168,
                inter_dim=256,
                experts=257,
                topk=9,
                dtype=dtype,
                act=act,
                token=int(token),
                stage1_us=10.0,
                stage2_us=5.0,
                sorting_us=0.0,
                kernel_path_us=15.0,
                kernel_path_us_p95=15.5,
                effective_tflops=1.0,
                mfu=0.01,
                e2e_us=12.0,
                e2e_us_p95=12.5,
                logits_diff=0.0001,
                correctness_pass=True,
            )
        )
    harness.write_csv(rows, str(out))
    assert harness.validate_baseline_csv(str(out), expected_keys=spec.validated_point_keys())["valid"] is True
    assert harness.validate_baseline_csv(str(out))["valid"] is False  # full workload not covered


def test_perf_dist_percentile():
    import importlib

    tc = importlib.import_module("tests.test_common")
    # nearest-rank p95 over 1..100: idx=round(0.95*99)=94 -> value 95 (0-based).
    assert tc._percentile(list(range(1, 101)), 0.95) == 95
    assert tc._percentile([], 0.95) is None
    assert "n_rotate" in tc.LAST_PERF_DIST


def test_timed_distribution_rotates_distinct_args():
    # Branch-level regression for the FLYDSL_PERF_DIST timed loop: it must cycle
    # the cache-sized rotated arg copies (iteration i -> rotate_args[i % n]) so
    # DISTINCT working sets reach func (the L2-flush behavior), and compute
    # median/p95 from the injected per-call timings.
    import importlib

    tc = importlib.import_module("tests.test_common")

    # 3 distinct arg copies; record which args each call received.
    rotate_args = [((tag,), {}) for tag in ("A", "B", "C")]
    seen = []

    def func(tag):
        seen.append(tag)
        return f"out-{tag}"

    # Injected timer returns a deterministic latency per call so we can check
    # median/p95 without a GPU.
    timings = iter([10.0, 30.0, 20.0, 50.0, 40.0, 60.0, 70.0])

    def time_call(fn, a_i, kw_i):
        out = fn(*a_i, **kw_i)
        return next(timings), out

    data, median, p95, n_rot = tc._timed_distribution(func, rotate_args, num_iters=7, time_call=time_call)
    # 7 iters over 3 copies -> A,B,C,A,B,C,A (distinct args actually reach func).
    assert seen == ["A", "B", "C", "A", "B", "C", "A"]
    assert n_rot == 3
    assert data == "out-A"  # last call's output
    # median of [10,30,20,50,40,60,70] sorted=[10,20,30,40,50,60,70] -> 40.
    assert median == 40.0
    # nearest-rank p95: idx=round(0.95*6)=6 -> 70.
    assert p95 == 70.0


def test_clock_pinning_helpers(monkeypatch):
    # pin_clocks parses the rocm-smi determinism-success message; clocks_pinned_state
    # treats determinism/manual/high as pinned and auto as DVFS (not pinned).
    outs = {}

    def fake_run(cmd):
        if "--setperfdeterminism" in cmd:
            return outs.get("set", "")
        if "--showperflevel" in cmd:
            return outs.get("level", "")
        return ""

    monkeypatch.setattr(harness, "_run", fake_run)
    outs["set"] = "GPU[0]: Successfully enabled performance determinism and set GFX clock frequency: 2200"
    assert harness.pin_clocks("0") is True
    outs["set"] = "GPU[0]: set_perf_level, Not supported on the given system"
    assert harness.pin_clocks("0") is False
    outs["level"] = "GPU[0]: Performance Level: determinism"
    assert harness.clocks_pinned_state("0") is True
    outs["level"] = "GPU[0]: Performance Level: auto"
    assert harness.clocks_pinned_state("0") is False


def test_setup_run_provenance_reflects_verified_clock_state(monkeypatch):
    # The live setup path must record the VERIFIED clock-pinned state, never the
    # static spec intent default. Provenance.clocks_pinned defaults to False.
    assert harness.Provenance().clocks_pinned is False

    calls = {"pin": 0}

    def fake_pin(gpu_id, *a, **k):
        calls["pin"] += 1
        return True

    monkeypatch.setattr(harness, "check_idle_gpu", lambda g, **k: True)
    monkeypatch.setattr(harness, "pin_clocks", fake_pin)
    monkeypatch.setattr(harness, "git_provenance", lambda *a, **k: {"branch": "b", "commit": "523ca1c7"})
    monkeypatch.setattr(harness, "gpu_provenance", lambda g: {"gpu_id": str(g), "gpu_model": "MI350X"})

    # Verified pinned -> clocks_pinned True.
    monkeypatch.setattr(harness, "clocks_pinned_state", lambda g: True)
    prov = harness.setup_run_provenance("0")
    assert calls["pin"] == 1  # the driver actually attempted to pin
    assert prov.clocks_pinned is True
    assert prov.idle_gpu_verified is True
    assert prov.commit == "523ca1c7" and prov.gpu_model == "MI350X"

    # Verification fails -> clocks_pinned MUST be False (not the intent default).
    monkeypatch.setattr(harness, "clocks_pinned_state", lambda g: False)
    prov2 = harness.setup_run_provenance("0")
    assert prov2.clocks_pinned is False
    # A row built from unverified provenance is rejected by the baseline validator.
    row = {
        "commit": "523ca1c7",
        "idle_gpu_verified": "True",
        "gpu_id": "0",
        "gpu_model": "MI350X",
        "branch": "b",
        "command": "c",
        "dtype": "a4w4",
        "act": "silu",
        "model": "kimi_k2",
        "token": "16",
        "stage1_us": "1",
        "stage2_us": "1",
        "sorting_us": "0",
        "kernel_path_us": "2",
        "kernel_path_us_p95": "2",
        "effective_tflops": "1",
        "mfu": "0.1",
        "e2e_us": "1",
        "e2e_us_p95": "1",
        "logits_diff": "0.0001",
        "correctness_pass": "True",
        "warmup": "10",
        "iters": "100",
        "graph_capture": "False",
        "l2_flush_per_iter": "True",
        "clocks_pinned": str(prov2.clocks_pinned),
    }
    assert "clocks_must_be_pinned" in harness.validate_baseline_row(row)


def test_main_clock_provenance_fail_closed(monkeypatch, tmp_path):
    # Direct regression around the live _main() path: it must pin+verify clocks,
    # write rows with the verified clocks_pinned, fail-closed (rc=2, no CSV) when
    # pinning cannot be verified, and proceed under --allow-unpinned.
    rp = harness.RunPoint("kimi_k2", 7168, 256, 384, 8, "silu", "a4w4", 16)
    monkeypatch.setattr(harness, "build_run_list", lambda: [rp])
    monkeypatch.setattr(harness, "check_idle_gpu", lambda g, **k: True)
    monkeypatch.setattr(harness, "git_provenance", lambda *a, **k: {"branch": "b", "commit": "523ca1c7"})
    monkeypatch.setattr(harness, "gpu_provenance", lambda g: {"gpu_id": str(g), "gpu_model": "MI350X"})

    written = {}

    def fake_write_csv(rows, path):
        written["rows"] = rows
        written["path"] = path

    def fake_run_point(rp_, tile, gpu, prov, **k):
        return harness.PointRow(
            provenance=prov,
            command="cmd",
            model=rp_.model,
            model_dim=rp_.model_dim,
            inter_dim=rp_.inter_dim,
            experts=rp_.experts,
            topk=rp_.topk,
            dtype=rp_.dtype,
            act=rp_.act,
            token=rp_.token,
        )

    monkeypatch.setattr(harness, "write_csv", fake_write_csv)
    monkeypatch.setattr(harness, "run_point", fake_run_point)
    monkeypatch.setattr(harness, "pin_clocks", lambda g, *a, **k: True)

    out = str(tmp_path / "b.csv")

    # (a) verified pinned -> rc 0, rows written with clocks_pinned True.
    written.clear()
    monkeypatch.setattr(harness, "clocks_pinned_state", lambda g: True)
    rc = harness._main(["baseline", "--gpu", "0", "--assume-idle", "--no-e2e", "--out", out])
    assert rc == 0
    assert written["rows"][0].provenance.clocks_pinned is True

    # (b) verification fails -> fail-closed: rc 2 and NO csv written.
    written.clear()
    monkeypatch.setattr(harness, "clocks_pinned_state", lambda g: False)
    rc = harness._main(["baseline", "--gpu", "0", "--assume-idle", "--no-e2e", "--out", out])
    assert rc == 2
    assert "rows" not in written  # fail-closed: did not write a false-pinned CSV

    # (c) --allow-unpinned proceeds, recording clocks_pinned False.
    written.clear()
    rc = harness._main(["baseline", "--gpu", "0", "--assume-idle", "--no-e2e", "--allow-unpinned", "--out", out])
    assert rc == 0
    assert written["rows"][0].provenance.clocks_pinned is False


def test_regime_aware_abs_floor():
    # Regime-aware floor: 8us for tokens<=64, 2us for tokens>=128.
    assert spec.abs_floor_us(1) == 8.0
    assert spec.abs_floor_us(64) == 8.0
    assert spec.abs_floor_us(128) == 2.0
    assert spec.abs_floor_us(32768) == 2.0


def test_is_regression_regime_aware():
    # Small token (16): a 5us drift on a 130us base is within the 8us floor -> NOT a regression.
    assert spec.is_regression(130.0, 135.0, token=16) is False
    # Small token: 9us drift on 130us base -> regression (exceeds 8us AND 2%).
    assert spec.is_regression(130.0, 139.0, token=16) is True
    # Large token (128): 5us drift on 130us base -> regression under the 2us floor.
    assert spec.is_regression(130.0, 135.0, token=128) is True
    # Back-compat: token=None keeps the strict 2us floor.
    assert spec.is_regression(130.0, 135.0) is True


def test_repeatability_check_regime_aware(tmp_path):
    a = str(tmp_path / "a.csv")
    b = str(tmp_path / "b.csv")
    _csv(
        a,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 130,
                "e2e_us": 40,
                "mfu": 0.05,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 290,
                "e2e_us": 250,
                "mfu": 0.3,
            },
        ],
    )
    _csv(
        b,
        [
            # token 16: +5us kernel-path -> within 8us small-token floor -> stable.
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 135,
                "e2e_us": 40,
                "mfu": 0.05,
            },
            # token 128: +7us -> exceeds 2us floor (and 2%) -> unstable.
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 297,
                "e2e_us": 250,
                "mfu": 0.3,
            },
        ],
    )
    res = ledger.repeatability_check(a, b)
    kp = res["unstable"]["kernel_path_us"]
    assert any(u[0] == ("kimi_k2", "a4w4", "silu", "128") for u in kp)  # 128 unstable
    assert all(u[0] != ("kimi_k2", "a4w4", "silu", "16") for u in kp)  # 16 stable under 8us


def test_select_run_points_filters():
    # Candidate selection filters the full grid by model/dtype/token.
    pts = harness.select_run_points(model="deepseek_v3", dtype="a4w4", tokens=[16, 16384])
    keys = {(p.model, p.dtype, p.token) for p in pts}
    assert keys == {("deepseek_v3", "a4w4", 16), ("deepseek_v3", "a4w4", 16384)}
    # dtype filter excludes a8w4.
    assert all(p.dtype == "a4w4" for p in harness.select_run_points(model="kimi_k2", dtype="a4w4"))
    # whole-grid when unfiltered equals build_run_list.
    assert len(harness.select_run_points()) == len(harness.build_run_list())


def test_candidate_tile_for_overrides_and_legality():
    rp = harness.RunPoint("deepseek_v3", 7168, 256, 257, 9, "silu", "a4w4", 16)
    # Legal override: stage1 tile_n -> 128 (the DS V3 lead).
    t = harness.candidate_tile_for(rp, {"tile_n1": 128})
    assert t["tile_n1"] == 128 and t["tile_m1"] == 64 and t["tile_k1"] == 256
    # No overrides -> the shape's default tiles.
    assert harness.candidate_tile_for(rp, {}) == harness.default_tile_for(rp)
    # Illegal override is rejected before any compile (e.g. fp4 tile_m < 32).
    import pytest as _pytest

    with _pytest.raises(ValueError):
        harness.candidate_tile_for(rp, {"tile_m1": 16})


def test_prepare_candidate_run_fail_closed(tmp_path, monkeypatch):
    # candidate run is fail-closed: requires explicit tiles, all-legal, non-empty.
    import moe_tuning_ledger as _ledger
    import pytest as _pytest

    # Capture rejected-candidate records instead of writing to the real ledger.
    captured = []
    monkeypatch.setattr(_ledger, "append_rejected_candidate", lambda rec, **k: captured.append(rec) or rec)

    no_override = {k: None for k in ("tile_m1", "tile_n1", "tile_k1", "tile_n2", "tile_k2")}
    # (1) no explicit tile -> reject (no silent default-tile fallback).
    with _pytest.raises(ValueError, match="at least one explicit"):
        harness.prepare_candidate_run(no_override, model="deepseek_v3", dtype="a4w4", tokens=[16])

    # (2) legal explicit tile -> returns (run_list, tiles) of equal length.
    ov = dict(no_override, tile_n1=128)
    rl, tiles = harness.prepare_candidate_run(ov, model="deepseek_v3", dtype="a4w4", tokens=[16, 64])
    assert len(rl) == len(tiles) == 2 and all(t["tile_n1"] == 128 for t in tiles)

    # (3) illegal explicit tile -> raise AND record a machine-readable rejection
    #     carrying the full provenance class (act/stage/branch/commit/command/...).
    bad = dict(no_override, tile_m1=16)  # fp4 tile_m<32 illegal
    prov = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="c")
    with _pytest.raises(ValueError, match="illegal candidate"):
        harness.prepare_candidate_run(
            bad, model="deepseek_v3", dtype="a4w4", tokens=[16], prov=prov, command="python3 harness candidate ..."
        )
    rec = captured[-1]
    assert rec and rec["reason"] and rec["model"] == "deepseek_v3"
    # Every full-provenance field is present and non-empty (stage 0 is valid).
    for k in ("act", "gpu_id", "gpu_model", "branch", "commit", "command", "warmup", "iters", "selection"):
        assert rec.get(k) not in (None, ""), k
    assert rec["stage"] == 0 and rec["act"] == "silu"
    # The record satisfies the ledger's own rejected-candidate contract.
    assert not [f for f in _ledger.REQUIRED_REJECTED_FIELDS if rec.get(f) in (None, "")]

    # (4) empty selection -> reject.
    with _pytest.raises(ValueError, match="matched no points"):
        harness.prepare_candidate_run(ov, model="nonesuch", dtype="a4w4", tokens=[16])
