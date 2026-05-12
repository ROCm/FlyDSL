# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Import-time guardrails for backend-specific expression modules.

These checks run in a **subprocess** so MLIR value-caster registration (process
global) is not executed twice in the pytest interpreter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SUBPROCESS_CHECK = r"""
import importlib
import importlib.abc
import sys


class _BlockRocdlDialect(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "flydsl._mlir.dialects.rocdl":
            raise ModuleNotFoundError("simulated missing ROCDL dialect", name=fullname)
        return None


sys.meta_path.insert(0, _BlockRocdlDialect())
expr = importlib.import_module("flydsl.expr")
assert expr.Tensor is not None
assert expr.Int32 is not None
assert expr.thread_idx is not None
assert expr.block_idx is not None
assert expr.gpu.barrier is not None
assert expr.math is not None
try:
    expr.rocdl
except ModuleNotFoundError as exc:
    assert exc.name == "flydsl._mlir.dialects.rocdl"
else:
    raise AssertionError("expected explicit ROCDL access to require the ROCDL dialect")
"""


def test_expr_import_does_not_require_rocdl_dialect():
    """The generic expression namespace should import without ROCDL bindings."""

    pkg = _REPO_ROOT / "build-fly" / "python_packages" / "flydsl"
    if not pkg.is_dir():
        pytest.skip("build-fly python_packages not found (run scripts/build.sh)")

    env = os.environ.copy()
    bpkg = str(_REPO_ROOT / "build-fly" / "python_packages")
    prev = env.get("PYTHONPATH", "")
    # Load `flydsl` only from the build tree (matches CI). Prepending `repo/python`
    # can shadow `_mlir` when the build tree is incomplete.
    env["PYTHONPATH"] = os.pathsep.join([bpkg] + ([prev] if prev else []))

    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CHECK],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "subprocess import check failed\n" f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from None
