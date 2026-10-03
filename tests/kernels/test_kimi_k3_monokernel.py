# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end GPU coverage for the Kimi-K3 TP8 MonoKernel."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"Kimi-K3 MonoKernel requires gfx950, got {_ARCH}", allow_module_level=True)


def _run_tp8_tool(tool: str, *args: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(ROOT), env.get("PYTHONPATH", "")) if path)
    result = subprocess.run(
        [sys.executable, str(ROOT / tool), "--npes", "8", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, f"expected one JSON result, got stdout:\n{result.stdout}"
    return records[0]


def _run_agentic_tool(*args: str) -> list[dict]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(ROOT), env.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "kernels/monokernel/k3/tools/agentic_kda.py"),
            *args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    payloads = [
        json.loads(line.removeprefix("BENCHMARK_JSON="))
        for line in result.stdout.splitlines()
        if line.startswith("BENCHMARK_JSON=")
    ]
    assert len(payloads) == 1, (
        f"expected one BENCHMARK_JSON payload, got stdout:\n{result.stdout}"
    )
    return payloads[0]


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_kimi_k3_monokernel_mla_baseline_tp8() -> None:
    result = _run_tp8_tool(
        "kernels/monokernel/k3/tools/mla.py",
        "--samples",
        "1",
        "--layer-idx",
        "3",
        "--check",
    )
    assert result["rank_equal"] is True
    assert result["finite"] is True
    assert result["pre_attn_rel_l2"] < 1e-3
    assert result["output_rel_l2"] < 1e-2
    assert result["selection_equal"] is True


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
@pytest.mark.parametrize(
    ("samples", "negative_slot"),
    ((1, False), (4, False), (8, False), (4, True)),
)
def test_kimi_k3_kda_attention_tp8(samples: int, negative_slot: bool) -> None:
    extra_args = ("--negative-slot",) if negative_slot else ()
    result = _run_tp8_tool(
        "kernels/monokernel/k3/tools/monokernel.py",
        "--samples",
        str(samples),
        "--layer-idx",
        "1",
        "--attention-only",
        "--check",
        "--bench",
        "--layers",
        "2",
        "--repeats",
        "2",
        *extra_args,
    )
    assert result["rank_equal"] is True
    assert result["finite"] is True
    assert result["attention_rel_l2"] < 2e-3
    assert result["conv_state_rel_l2"] < 5e-4
    assert result["recurrent_state_rel_l2"] < 5e-4
    assert result["median_us"] > 0
    if negative_slot:
        assert result["negative_output_zero"] is True


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
@pytest.mark.parametrize(("samples", "layer_idx"), ((1, 1), (1, 12), (8, 1)))
def test_kimi_k3_monokernel_kda_tp8(samples: int, layer_idx: int) -> None:
    result = _run_tp8_tool(
        "kernels/monokernel/k3/tools/monokernel.py",
        "--samples",
        str(samples),
        "--layer-idx",
        str(layer_idx),
        "--check",
    )
    assert result["rank_equal"] is True
    assert result["finite"] is True
    assert result["attention_rel_l2"] < 2e-3
    assert result["conv_state_rel_l2"] < 5e-4
    assert result["recurrent_state_rel_l2"] < 5e-4
    assert result["selection_equal"] is True
    assert result["output_rel_l2"] < 1e-2


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_kimi_k3_monokernel_staged_baseline_tp8() -> None:
    result = _run_tp8_tool(
        "kernels/monokernel/k3/tools/monokernel.py",
        "--staged",
        "--samples",
        "8",
        "--layer-idx",
        "1",
        "--check",
    )
    assert result["launch_mode"] == "staged"
    assert result["rank_equal"] is True
    assert result["finite"] is True
    assert result["selection_equal"] is True
    assert result["output_rel_l2"] < 1e-2


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
@pytest.mark.parametrize("samples", (4, 8))
def test_kimi_k3_monokernel_mtp_tp8(samples: int) -> None:
    result = _run_tp8_tool(
        "kernels/monokernel/k3/tools/monokernel.py",
        "--samples",
        str(samples),
        "--layer-idx",
        "1",
        "--mtp",
        "--check",
        "--bench",
        "--layers",
        "32",
        "--repeats",
        "2",
        "--rank-skew-ms",
        "25",
    )
    assert result["mtp"] is True
    assert result["rank_equal"] is True
    assert result["finite"] is True
    assert result["attention_rel_l2"] < 2e-3
    assert result["conv_state_rel_l2"] < 5e-4
    assert result["recurrent_state_rel_l2"] < 5e-4
    assert result["selection_equal"] is True
    assert result["routed_norm_rel_l2"] < 1e-2
    assert result["output_rel_l2"] < 8e-2
    assert result["graph_rank_equal"] is True


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_kimi_k3_agentic_skewed_graph_tp8() -> None:
    results = _run_agentic_tool(
        "--batches",
        "1",
        "2",
        "4",
        "--rank-skew-ms",
        "25",
    )
    assert [result["batch"] for result in results] == [1, 2, 4]
    for result in results:
        assert result["rank_skew_ms"] == 25
        assert result["parity"]["eager"]["rank_equal"] is True
        assert result["parity"]["graph"]["rank_equal"] is True
        assert result["parity"]["graph"]["conv_exact"] is True
