# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.unit.test_register_allocator_codegen import SCALAR_TYPES, run_allocator

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize(
    "mode,bank",
    [(mode, "VGPR") for mode in ["scalar", "vector", "packed", "sgpr", "union", "loop", "branch", "fma", *SCALAR_TYPES]]
    + [(mode, "AGPR") for mode in ["scalar", "vector"]],
)
@pytest.mark.parametrize("automatic", [False, True])
def test_register_allocator_values(mode, bank, automatic):
    run_allocator(mode, run=True, bank=bank, automatic=automatic, register_alignment=4 if automatic else 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("automatic", [False, True])
def test_native_agpr_gemm_values(tmp_path, automatic):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in {"gfx942", "gfx950"}:
        pytest.skip("native AGPR GEMM is covered on gfx942/gfx950")
    example_path = Path(__file__).resolve().parents[2] / "examples/04-preshuffle_gemm.py"
    script = f"""
import importlib.util
import resource
import sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
spec = importlib.util.spec_from_file_location("agpr_gemm", {str(example_path)!r})
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)
example.M, example.N, example.K = 256, 256, 512
example.EXPLICIT_REGISTERS = True
example.MMA_REG_C = (example.fx.rocdl.AGPR, 128)
if {automatic!r}:
    example.MMA_REG_A = (example.fx.rocdl.AGPR, None)
    example.MMA_REG_B = (example.fx.rocdl.AGPR, None)
    example.MMA_REG_C = (example.fx.rocdl.AGPR, None)
    example.MMA_REG_ALIGNMENT = 4
example.main()
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        ARCH=arch,
        COMPILE_ONLY="0",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Result correct: True" in result.stdout, result.stdout + result.stderr
