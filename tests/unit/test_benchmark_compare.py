# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GPU-free tests for benchmark regression policy and CSV compatibility."""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_compare import Threshold, ThresholdConfig, compare_values  # noqa: E402

compare_benchmark = importlib.import_module("compare_benchmark")


@pytest.mark.parametrize(
    "current,relative,absolute,expected",
    [
        (121.0, 20.0, 10.0, True),
        (119.0, 20.0, 10.0, False),
        (121.0, 25.0, 10.0, False),
        (109.0, 5.0, 10.0, False),
    ],
)
def test_dual_threshold_requires_relative_and_absolute(current, relative, absolute, expected):
    result = compare_values(
        100.0,
        current,
        Threshold(relative_pct=relative, absolute=absolute),
    )
    assert result.regressed is expected


def test_higher_better_direction_flags_decrease():
    result = compare_values(
        100.0,
        75.0,
        Threshold(relative_pct=20.0, absolute=10.0, direction="higher_better"),
    )
    assert result.regressed
    assert result.delta == 25.0


def test_threshold_config_is_an_arch_allowlist():
    config = ThresholdConfig(
        {
            "version": 1,
            "architectures": {
                "gfx950": [
                    {
                        "op": "rmsnorm",
                        "shape": "*",
                        "dtype": "bf16",
                        "relative_pct": 20,
                        "absolute_us": 10,
                    }
                ]
            },
        }
    )

    assert config.match(arch="gfx950", op="rmsnorm", shape="1x2", dtype="bf16") is not None
    assert config.match(arch="gfx942", op="rmsnorm", shape="1x2", dtype="bf16") is None


def _write_csv(
    path,
    *,
    avg_us,
    status="ok",
    arch="gfx950",
    tbps="4.0",
    tflops="-",
    statistic="mean",
):
    fields = [
        "op",
        "shape",
        "dtype",
        "tbps",
        "tflops",
        "status",
        "avg_us",
        "statistic",
        "instrument",
        "schedule",
        "cache_policy",
        "warmup",
        "iters",
        "arch",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "op": "softmax",
                "shape": "32768x8192",
                "dtype": "bf16",
                "tbps": tbps,
                "tflops": tflops,
                "status": status,
                "avg_us": avg_us,
                "statistic": statistic,
                "instrument": "reported",
                "schedule": "unknown",
                "cache_policy": "unknown",
                "warmup": "10",
                "iters": "100",
                "arch": arch,
            }
        )


def test_compare_benchmark_hard_fails_allowlisted_latency(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100")
    _write_csv(current, avg_us="125")
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx950": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )

    assert compare_benchmark.main() == 1


def test_compare_benchmark_accepts_legacy_csv_without_raw_us(tmp_path, monkeypatch):
    header = "op,shape,dtype,tbps,tflops,status\n"
    row = "softmax,32768x8192,bf16,4.0,-,ok\n"
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    baseline.write_text(header + row)
    current.write_text(header + row)
    monkeypatch.setattr(sys, "argv", ["compare_benchmark.py", str(baseline), str(current)])

    assert compare_benchmark.main() == 0


def test_compare_benchmark_fails_when_allowlisted_row_disappears(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100")
    current.write_text(
        "op,shape,dtype,tbps,tflops,status,avg_us,arch\n"
        "gemm,1x1x1,bf16,-,1.0,ok,10,gfx950\n"
    )
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx950": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )

    assert compare_benchmark.main() == 1


def test_raw_us_gate_is_independent_of_throughput_metric(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", tbps="4.0", tflops="-")
    _write_csv(current, avg_us="125", tbps="-", tflops="10.0")
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx950": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 1


def test_arch_mismatch_never_hard_gates(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", arch="gfx942")
    _write_csv(current, avg_us="150", arch="gfx950")
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx950": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 0


def test_allowlisted_current_row_without_raw_us_is_broken(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100")
    _write_csv(current, avg_us="-")
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx950": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 1


def test_baseline_only_row_on_another_arch_is_not_gated(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", arch="gfx942")
    current.write_text(
        "op,shape,dtype,tbps,tflops,status,avg_us,arch\n"
        "gemm,1x1x1,bf16,-,1.0,ok,10,gfx950\n"
    )
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx942": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ],
                    "gfx950": [],
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 0


def test_measurement_contract_mismatch_fails_closed(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", statistic="mean")
    _write_csv(current, avg_us="100", statistic="median")
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "architectures": {
                    "gfx950": [
                        {
                            "op": "softmax",
                            "shape": "*",
                            "dtype": "*",
                            "relative_pct": 20,
                            "absolute_us": 10,
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 1


def test_unknown_arch_cannot_disable_hard_gate(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", arch="unknown")
    _write_csv(current, avg_us="100", arch="unknown")
    thresholds.write_text(json.dumps({"version": 1, "architectures": {"gfx950": []}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 2


def test_csv_arch_cannot_override_trusted_runner_arch(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", arch="gfx950")
    _write_csv(current, avg_us="100", arch="gfx1201")
    thresholds.write_text(
        json.dumps({"version": 1, "architectures": {"gfx950": [], "gfx1201": []}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--arch",
            "gfx950",
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 2


def test_baseline_arch_must_match_trusted_runner_arch(tmp_path, monkeypatch):
    baseline = tmp_path / "base.csv"
    current = tmp_path / "current.csv"
    thresholds = tmp_path / "thresholds.json"
    _write_csv(baseline, avg_us="100", arch="gfx1201")
    _write_csv(current, avg_us="100", arch="gfx950")
    thresholds.write_text(
        json.dumps({"version": 1, "architectures": {"gfx950": [], "gfx1201": []}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark.py",
            str(baseline),
            str(current),
            "--arch",
            "gfx950",
            "--threshold-config",
            str(thresholds),
            "--fail-on-regression",
        ],
    )
    assert compare_benchmark.main() == 2
