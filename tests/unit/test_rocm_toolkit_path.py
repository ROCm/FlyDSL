# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm toolkit resolution for AMDGCN device bitcode.

``fly-emit-gpu-binary`` links in process, so the only thing the toolkit path
still supplies is ``<root>/amdgcn/bitcode``.  FlyDSL bundles that bitcode and
points ``toolkit=`` at it, so kernels calling ``__ocml_*`` compile regardless of
where a container installs ROCm.
"""

from pathlib import Path

import pytest

from flydsl.compiler.backends.rocm import BINARY_PASS_NAME, RocmBackend, rocm_toolkit_path

pytestmark = [pytest.mark.l0_backend_agnostic]

_ROCM_ENV_VARS = ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME")


@pytest.fixture(autouse=True)
def _clear_resolution_cache():
    rocm_toolkit_path.cache_clear()
    yield
    rocm_toolkit_path.cache_clear()


def test_toolkit_resolves_without_rocm_environment(monkeypatch):
    for var in _ROCM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FLYDSL_COMPILE_ROCM_PATH", raising=False)

    toolkit = rocm_toolkit_path()
    if not toolkit:
        pytest.skip("build did not bundle AMDGCN device bitcode (FLYDSL_ROCM_BITCODE_DIR unset at configure time)")

    assert (Path(toolkit) / "amdgcn" / "bitcode" / "ocml.bc").is_file()


def test_explicit_override_wins_over_bundled(monkeypatch, tmp_path):
    bitcode_dir = tmp_path / "amdgcn" / "bitcode"
    bitcode_dir.mkdir(parents=True)
    (bitcode_dir / "ocml.bc").write_bytes(b"")
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(tmp_path))

    assert rocm_toolkit_path() == str(tmp_path)


def test_override_without_bitcode_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(tmp_path / "missing"))

    assert rocm_toolkit_path() != str(tmp_path / "missing")


def test_path_with_pipeline_metacharacters_is_rejected(monkeypatch, tmp_path):
    root = tmp_path / "rocm dir"
    (root / "amdgcn" / "bitcode").mkdir(parents=True)
    (root / "amdgcn" / "bitcode" / "ocml.bc").write_bytes(b"")
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(root))

    with pytest.raises(ValueError, match="unsupported character"):
        rocm_toolkit_path()


def test_binary_fragment_carries_the_resolved_toolkit(monkeypatch, tmp_path):
    bitcode_dir = tmp_path / "amdgcn" / "bitcode"
    bitcode_dir.mkdir(parents=True)
    (bitcode_dir / "ocml.bc").write_bytes(b"")
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(tmp_path))

    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    binary_fragment = backend.pipeline_fragments(compile_hints={})[-1]

    assert binary_fragment.startswith(BINARY_PASS_NAME)
    assert f"toolkit={tmp_path}" in binary_fragment
