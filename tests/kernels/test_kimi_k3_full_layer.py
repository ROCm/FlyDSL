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

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"Kimi-K3 full layer requires gfx950, got {_ARCH}", allow_module_level=True)


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
