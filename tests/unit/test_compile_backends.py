# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""compile backend default behavior and registry guardrails."""

import importlib
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPILER_DIR = _REPO_ROOT / "python" / "flydsl" / "compiler"


def _load_backends(monkeypatch):
    """Import flydsl.compiler.backends without importing JIT-only compiler exports."""
    for name in list(sys.modules):
        if name == "flydsl.compiler" or name.startswith("flydsl.compiler.backends"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    compiler_pkg = types.ModuleType("flydsl.compiler")
    compiler_pkg.__path__ = [str(_COMPILER_DIR)]
    monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler_pkg)
    return importlib.import_module("flydsl.compiler.backends")


def test_default_compile_backend_stays_rocm(monkeypatch):
    backends = _load_backends(monkeypatch)
    monkeypatch.delenv("FLYDSL_COMPILE_BACKEND", raising=False)

    backend = backends.get_backend(arch="gfx942")

    assert backends.compile_backend_name() == "rocm"
    assert backend.target.backend == "rocm"
    assert backend.target.arch == "gfx942"


def test_registering_extra_backend_does_not_change_default(monkeypatch):
    backends = _load_backends(monkeypatch)
    monkeypatch.delenv("FLYDSL_COMPILE_BACKEND", raising=False)

    class _DummyBackend(backends.BaseBackend):
        @staticmethod
        def supports_target(target):
            return target.backend == "dummy"

        @staticmethod
        def detect_target():
            return backends.GPUTarget(backend="dummy", arch="dummy0", warp_size=1)

        @classmethod
        def make_target(cls, arch):
            return backends.GPUTarget(backend="dummy", arch=arch or "dummy0", warp_size=1)

        def pipeline_fragments(self, *, compile_hints):
            return []

        def gpu_module_targets(self):
            return []

        def native_lib_patterns(self):
            return []

        def jit_runtime_lib_basenames(self):
            return []

    backends.register_backend("dummy", _DummyBackend)

    assert backends.compile_backend_name() == "rocm"
    assert backends.get_backend(arch="gfx942").target.backend == "rocm"
    assert backends.get_backend("dummy", arch="dummy0").target.backend == "dummy"
