# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end GPU coverage for the complete Kimi-K3 TP8 layer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.common.mx_formats import dequantize_mxfp8, quantize_mxfp8
from kernels.mla_moe_layer.mxfp8_linear import Mxfp8Linear

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"Kimi-K3 full layer requires gfx950, got {_ARCH}", allow_module_level=True)


def test_mxfp8_linear_matches_quantized_reference() -> None:
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(77)
    source = torch.randn(4, 256, generator=generator, device=device, dtype=torch.bfloat16)
    dense_weight = (torch.randn(64, 256, generator=generator, device=device) / (256**0.5)).to(torch.bfloat16)
    weight, weight_scale = quantize_mxfp8(dense_weight)
    source_quantized, source_scale = quantize_mxfp8(source)
    expected = torch.mm(
        dequantize_mxfp8(source_quantized, source_scale).to(torch.bfloat16),
        dequantize_mxfp8(weight, weight_scale).to(torch.bfloat16).T,
    )

    output = torch.empty(4, 64, device=device, dtype=torch.bfloat16)
    Mxfp8Linear(weight, weight_scale, 4)(source, output)

    torch.testing.assert_close(output, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_kimi_k3_full_layer_tp8() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(ROOT), env.get("PYTHONPATH", "")) if path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "kernels/mla_moe_layer/tools/kimi_k3_full.py"),
            "--npes",
            "8",
            "--samples",
            "1",
            "--layer-idx",
            "0",
            "--check",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
