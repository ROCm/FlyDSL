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


@pytest.mark.parametrize("wave_size", [32, 64])
def test_rocm_wave_hint_matches_both_module_targets(monkeypatch, wave_size):
    backends = _load_backends(monkeypatch)
    from flydsl.compiler.kernel_function import CompilationContext

    backend = backends.get_backend(arch="gfx1201")
    assert backend.supports_wave_size_hint
    hints = {"wave_size": wave_size}
    with CompilationContext.compile_hints(hints):
        targets = backend.gpu_module_targets()
        pipeline = backend.pipeline_fragments(compile_hints=hints)
    features = f"+wavefrontsize{wave_size},-wavefrontsize{96 - wave_size}"
    assert features in targets[0]
    attach = next(part for part in pipeline if part.startswith("rocdl-attach-target"))
    assert f"features={features}" in attach
    assert f"wave64={'true' if wave_size == 64 else 'false'}" in attach
    assert backend.gpu_module_targets() == ['#rocdl.target<chip = "gfx1201">']


@pytest.mark.parametrize("wave_size", [True, False, "64", 64.0, 0, 16, 128])
def test_rocm_rejects_invalid_wave_hint(monkeypatch, wave_size):
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend(arch="gfx1201")
    with pytest.raises((TypeError, ValueError), match="wave_size"):
        backend.pipeline_fragments(compile_hints={"wave_size": wave_size})


def test_cdna_rejects_wave32(monkeypatch):
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend(arch="gfx942")
    with pytest.raises(ValueError, match="does not support wave_size=32"):
        backend.pipeline_fragments(compile_hints={"wave_size": 32})


def test_rocm_wave_hints_do_not_leak_between_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    backends = _load_backends(monkeypatch)
    from flydsl.compiler.kernel_function import CompilationContext

    backend = backends.get_backend(arch="gfx1201")
    ready = Barrier(2)

    def targets(wave_size):
        with CompilationContext.compile_hints({"wave_size": wave_size}):
            ready.wait(timeout=10)
            return backend.gpu_module_targets()

    with ThreadPoolExecutor(max_workers=2) as pool:
        wave32, wave64 = list(pool.map(targets, [32, 64]))
    assert "+wavefrontsize32,-wavefrontsize64" in wave32[0]
    assert "+wavefrontsize64,-wavefrontsize32" in wave64[0]
    assert backend.gpu_module_targets() == ['#rocdl.target<chip = "gfx1201">']
