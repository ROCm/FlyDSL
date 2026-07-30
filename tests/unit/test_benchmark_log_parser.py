# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Golden log tests for raw-us benchmark extraction."""

import csv
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_log_parser import parse_metrics, parse_moe_stage2  # noqa: E402

benchmark_output_to_csv = importlib.import_module("benchmark_output_to_csv")


@pytest.mark.parametrize(
    "text,expected_us,expected_tbps,expected_tflops,expected_statistic",
    [
        (
            "[flyc] Throughput: 123.4 us, 55.6 TFLOPS, BW: 1.234 TB/s",
            123.4,
            1.234,
            55.6,
            "mean",
        ),
        ("| PASS | 1.0e-3 0.999 | 42.5 100.2", 42.5, None, 100.2, "reported"),
        (
            "cos_diff=1e-3  TFLOPS=88.0  TB/s=2.5  err=0  us_p50=91.5  us_range=[90, 93]",
            91.5,
            2.5,
            88.0,
            "median",
        ),
        (
            "Kernel avg time: 0.1250 ms\nBandwidth: 3500.0 GB/s",
            125.0,
            3.5,
            None,
            "mean",
        ),
    ],
)
def test_parse_metrics_normalizes_raw_microseconds(
    text,
    expected_us,
    expected_tbps,
    expected_tflops,
    expected_statistic,
):
    metrics = parse_metrics(text)
    assert metrics.avg_us == expected_us
    assert metrics.tbps == expected_tbps
    assert metrics.tflops == expected_tflops
    assert metrics.statistic == expected_statistic


def test_softmax_style_keeps_first_base_measurement():
    metrics = parse_metrics(
        "Kernel avg time: 0.1000 ms\nBandwidth: 4000 GB/s\n"
        "Kernel avg time: 0.5000 ms\nBandwidth: 1000 GB/s\n"
    )
    assert metrics.avg_us == 100.0
    assert metrics.tbps == 4.0


def test_machine_readable_contract_overrides_fallback_metadata():
    metrics = parse_metrics(
        "Kernel avg time: 0.1000 ms\n"
        "Benchmark contract: instrument=device_event schedule=pipelined "
        "cache=cold statistic=median warmup=7 iters=50\n"
        "Bandwidth: 4000 GB/s\n"
    )
    assert metrics.instrument == "device_event"
    assert metrics.schedule == "pipelined"
    assert metrics.cache_policy == "cold"
    assert metrics.statistic == "median"
    assert metrics.warmup == "7"
    assert metrics.iters == "50"


def test_parse_moe_stage2_preserves_modes_and_latency():
    found = parse_moe_stage2(
        "FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | x | "
        "1163.2 us, 1654.24 TFLOPS, 0.377 TB/s\n"
        "FlyDSL MoE stage2 [moe_gemm2] fp4 reduce | x | "
        "1200.0 us, 1600.00 TFLOPS, 0.350 TB/s"
    )
    assert found["atomic"][1].avg_us == 1163.2
    assert found["reduce"][1].tflops == 1600.0


def test_legacy_five_column_output_converts_to_enriched_csv(tmp_path, monkeypatch):
    source = tmp_path / "benchmark.out"
    destination = tmp_path / "benchmark.csv"
    long_shape = "5120x5120x8320_tile128x256x128_2tg"
    source.write_text(
        "op shape dtype TB/s TFLOPS\n"
        f"gemm {long_shape} bf16 4.000 -\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_output_to_csv.py", str(source), str(destination)],
    )

    assert benchmark_output_to_csv.main() == 0
    with destination.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["tbps"] == "4.000"
    assert rows[0]["shape"] == long_shape
    assert rows[0]["avg_us"] == "-"
    assert rows[0]["statistic"] == "reported"
    assert rows[0]["instrument"] == "reported"
