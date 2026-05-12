# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""fly-opt tool usage guardrails."""

import os
from pathlib import Path
import subprocess

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fly_opt_path() -> Path:
    configured = os.environ.get("FLYDSL_FLY_OPT") or os.environ.get("FLY_OPT")
    candidates = [
        Path(configured) if configured else None,
        _REPO_ROOT / "build-fly" / "bin" / "fly-opt",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    pytest.skip("fly-opt binary not available; set FLYDSL_FLY_OPT to run this regression")


def test_fly_opt_gpu_module_to_binary_has_llvm_translation(tmp_path):
    """Regression for gpu-module-to-binary failing before target serialization.

    Without GPU-to-LLVMIR translations registered in fly-opt, this real
    gpu-module-to-binary invocation fails with:
    "missing LLVMTranslationDialectInterface registration for dialect for op:
    gpu.module".
    """

    input_mlir = tmp_path / "kernel.mlir"
    input_mlir.write_text(
        """module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    llvm.func @kernel() attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
  }
}
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(_fly_opt_path()),
            str(input_mlir),
            "--pass-pipeline=builtin.module(gpu-module-to-binary{format=llvm})",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert "missing LLVMTranslationDialectInterface registration" not in result.stderr
    if result.returncode != 0 and "the `AMDGPU` target was not built" in result.stderr:
        return

    assert result.returncode == 0, result.stderr
    assert "gpu.binary @kernels" in result.stdout
