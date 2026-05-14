# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Import-time guardrails for backend-specific expression modules.

These checks run in subprocesses so MLIR value-caster registration (process
global) is not executed twice in the pytest interpreter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCKED_ROCDL_MODULES = (
    "flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL",
    "flydsl._mlir.dialects.fly_rocdl",
    "flydsl._mlir.dialects.rocdl",
)

_BOOTSTRAP_SOURCE_WITH_BUILD_MLIR = """
import os
import flydsl

build_pkg = os.environ["FLYDSL_TEST_BUILD_FLYDSL_PKG"]
if build_pkg not in flydsl.__path__:
    flydsl.__path__.append(build_pkg)
"""

_MISSING_ROCDL_CHECK = rf"""
import importlib
import importlib.abc
import sys

BLOCKED = {set(_BLOCKED_ROCDL_MODULES)!r}


class _BlockRocdlBindings(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in BLOCKED:
            raise ModuleNotFoundError("simulated missing ROCDL binding", name=fullname)
        return None


sys.meta_path.insert(0, _BlockRocdlBindings())
expr = importlib.import_module("flydsl.expr")
assert expr.Tensor is not None
assert expr.Int32 is not None
assert expr.thread_idx is not None
assert expr.block_idx is not None
assert expr.gpu.barrier is not None
assert expr.math is not None
assert "buffer_ops" not in expr.__dict__
assert "rocdl" not in expr.__dict__
assert "tdm_ops" not in expr.__dict__


def _assert_rocdl_import_error(alias, exc):
    details = f"{{getattr(exc, 'name', '')}} {{exc}}".lower()
    assert "rocdl" in details, (alias, type(exc).__name__, details)


for name in ("buffer_ops", "rocdl", "tdm_ops"):
    try:
        getattr(expr, name)
    except ImportError as exc:
        _assert_rocdl_import_error(name, exc)
    else:
        raise AssertionError(f"expected explicit {{name}} access to require ROCDL bindings")
"""

_LAZY_ALIAS_CHECK = """
import importlib

expr = importlib.import_module("flydsl.expr")
assert "buffer_ops" not in expr.__dict__
assert "rocdl" not in expr.__dict__
assert "tdm_ops" not in expr.__dict__

from flydsl.expr import buffer_ops, rocdl, tdm_ops

assert buffer_ops.__name__ == "flydsl.expr.buffer_ops"
assert rocdl.__name__ == "flydsl.expr.rocdl"
assert tdm_ops.__name__ == "flydsl.expr.rocdl.tdm_ops"
assert expr.buffer_ops is buffer_ops
assert expr.rocdl is rocdl
assert expr.tdm_ops is tdm_ops
"""


def _build_env():
    pkg = _REPO_ROOT / "build-fly" / "python_packages" / "flydsl"
    if not pkg.is_dir():
        pytest.skip("build-fly python_packages not found (run scripts/build.sh)")

    env = os.environ.copy()
    bpkg = str(_REPO_ROOT / "build-fly" / "python_packages")
    spkg = str(_REPO_ROOT / "python")
    prev = env.get("PYTHONPATH", "")
    # Load Python sources under test, then resolve generated `_mlir` from the
    # build tree by extending `flydsl.__path__` inside the subprocess.
    env["PYTHONPATH"] = os.pathsep.join([spkg, bpkg] + ([prev] if prev else []))
    env["FLYDSL_TEST_BUILD_FLYDSL_PKG"] = str(pkg)
    return env


def _run_subprocess(code: str):
    proc = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP_SOURCE_WITH_BUILD_MLIR + code],
        cwd=str(_REPO_ROOT),
        env=_build_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "subprocess import check failed\n" f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from None


def test_expr_import_does_not_require_rocdl_bindings():
    """The generic expression namespace should import without ROCDL bindings."""

    _run_subprocess(_MISSING_ROCDL_CHECK)


def test_expr_rocdl_aliases_are_lazy_and_compatible():
    """Existing convenience imports resolve only when explicitly requested."""

    _run_subprocess(_LAZY_ALIAS_CHECK)
