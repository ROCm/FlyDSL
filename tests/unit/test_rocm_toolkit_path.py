# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm toolkit resolution for AMDGCN device bitcode.

``fly-emit-gpu-binary`` links in process, so the only thing the toolkit path
still supplies is ``<root>/amdgcn/bitcode``.  FlyDSL bundles that bitcode and
points ``toolkit=`` at it, so kernels calling ``__ocml_*`` compile regardless of
where a container installs ROCm.  External codegen with in-process LLD stops at
``format=isa`` and finishes the link in process, so it can use the same
bitcode-only toolkit.
"""

import os
from pathlib import Path

import pytest

from flydsl.compiler.backends import rocm as rocm_backend
from flydsl.compiler.backends.rocm import (
    BUNDLED_ROCM_PATH,
    FLY_GPU_BINARY_PASS,
    HAS_INPROCESS_LLD,
    RocmBackend,
    rocm_toolkit_path,
)

pytestmark = [pytest.mark.l0_backend_agnostic]

_ROCM_ENV_VARS = ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME")


@pytest.fixture(autouse=True)
def _clear_resolution_cache():
    rocm_toolkit_path.cache_clear()
    yield
    rocm_toolkit_path.cache_clear()


def _make_bitcode_root(tmp_path: Path, *, with_lld: bool = False) -> Path:
    bitcode_dir = tmp_path / "amdgcn" / "bitcode"
    bitcode_dir.mkdir(parents=True)
    (bitcode_dir / "ocml.bc").write_bytes(b"")
    if with_lld:
        lld_dir = tmp_path / "llvm" / "bin"
        lld_dir.mkdir(parents=True)
        lld_name = "ld.lld.exe" if os.name == "nt" else "ld.lld"
        (lld_dir / lld_name).write_bytes(b"")
    return tmp_path


def _embedded_override_root(tmp_path: Path) -> Path:
    """A toolkit that ``rocm_toolkit_path()`` accepts for the embedded pipeline."""
    # Without in-process LLD the embedded path falls back to upstream fatbin and
    # therefore also needs ld.lld.
    return _make_bitcode_root(tmp_path, with_lld=not HAS_INPROCESS_LLD)


def test_toolkit_resolves_without_rocm_environment(monkeypatch):
    for var in _ROCM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FLYDSL_COMPILE_ROCM_PATH", raising=False)

    toolkit = rocm_toolkit_path()
    if not toolkit:
        pytest.skip("build did not bundle AMDGCN device bitcode (FLYDSL_ROCM_BITCODE_DIR unset at configure time)")

    assert (Path(toolkit) / "amdgcn" / "bitcode" / "ocml.bc").is_file()
    if HAS_INPROCESS_LLD:
        assert Path(toolkit) == BUNDLED_ROCM_PATH


def test_explicit_override_wins_over_bundled(monkeypatch, tmp_path):
    root = _embedded_override_root(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(root))

    assert rocm_toolkit_path() == str(root)


def test_override_without_bitcode_warns_and_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(tmp_path / "missing"))

    with pytest.warns(UserWarning, match="FLYDSL_COMPILE_ROCM_PATH"):
        assert rocm_toolkit_path() != str(tmp_path / "missing")


def test_path_with_pipeline_metacharacters_is_rejected(monkeypatch, tmp_path):
    root = _embedded_override_root(tmp_path / "rocm,dir")
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(root))

    with pytest.raises(ValueError, match="unsupported character"):
        rocm_toolkit_path()


def test_path_with_whitespace_is_accepted_when_quoted(monkeypatch, tmp_path):
    root = _embedded_override_root(tmp_path / "rocm dir")
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(root))

    assert rocm_toolkit_path() == str(root)
    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    binary_fragment = backend.pipeline_fragments(compile_hints={})[-1]
    assert f'toolkit="{root}"' in binary_fragment


def test_binary_fragment_carries_the_resolved_toolkit(monkeypatch, tmp_path):
    root = _embedded_override_root(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(root))

    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    binary_fragment = backend.pipeline_fragments(compile_hints={})[-1]

    assert binary_fragment.startswith(FLY_GPU_BINARY_PASS)
    assert f'toolkit="{root}"' in binary_fragment


def test_external_toolkit_matches_embedded_when_inprocess_lld(monkeypatch, tmp_path):
    for var in _ROCM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FLYDSL_COMPILE_ROCM_PATH", raising=False)

    bitcode_only = _make_bitcode_root(tmp_path / "bitcode_only")
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(bitcode_only))

    if HAS_INPROCESS_LLD:
        # External stops at format=isa; bitcode-only toolkit is enough.
        assert rocm_toolkit_path(for_external=True) == str(bitcode_only)
        backend = RocmBackend(RocmBackend.make_target("gfx942"))
        _, binary = backend.external_binary_pipeline_fragments(compile_hints={})
        assert binary.startswith("gpu-module-to-binary")
        assert "format=isa" in binary
        assert f'toolkit="{bitcode_only}"' in binary
        return

    # Without in-process LLD, external still needs ld.lld for format=fatbin.
    with pytest.warns(UserWarning, match="ld.lld"):
        external = rocm_toolkit_path(for_external=True)
    assert external != str(bitcode_only)

    full = _make_bitcode_root(tmp_path / "full", with_lld=True)
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(full))
    rocm_toolkit_path.cache_clear()
    assert rocm_toolkit_path(for_external=True) == str(full)

    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    _, binary = backend.external_binary_pipeline_fragments(compile_hints={})
    assert "format=fatbin" in binary
    assert f'toolkit="{full}"' in binary


def test_toolkit_fingerprint_matches_hash_resolver(monkeypatch, tmp_path):
    root = _embedded_override_root(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(root))

    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    assert str(root) in backend.hash()
    assert backend.toolkit_fingerprint().startswith("rocm-bitcode:")
    # Both paths must hash the same resolved toolkit (not MLIR's bake-in ROCM_PATH).
    assert backend.hash().rsplit(":", 1)[-1] == backend.toolkit_fingerprint().split(":", 1)[-1]


def test_therock_lib_llvm_layout_is_accepted(monkeypatch, tmp_path):
    """TheRock keeps bitcode under lib/llvm rather than the ROCDL top-level layout."""
    for var in _ROCM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FLYDSL_COMPILE_ROCM_PATH", raising=False)

    prefix = tmp_path / "rocm_sdk"
    llvm = prefix / "lib" / "llvm"
    bitcode = llvm / "amdgcn" / "bitcode"
    bitcode.mkdir(parents=True)
    (bitcode / "ocml.bc").write_bytes(b"therock")
    if not HAS_INPROCESS_LLD:
        lld_dir = llvm / "bin"
        lld_dir.mkdir(parents=True)
        lld_name = "ld.lld.exe" if os.name == "nt" else "ld.lld"
        (lld_dir / lld_name).write_bytes(b"")

    monkeypatch.setenv("ROCM_PATH", str(prefix))
    # Prefer the env candidate over bundled so this layout is what we resolve.
    monkeypatch.setenv("FLYDSL_COMPILE_ROCM_PATH", str(prefix))

    toolkit = Path(rocm_toolkit_path())
    assert (toolkit / "amdgcn" / "bitcode" / "ocml.bc").is_file()
    if HAS_INPROCESS_LLD:
        assert toolkit == llvm
    else:
        # Upstream needs llvm/bin/ld.lld, so TheRock trees are staged.
        assert (toolkit / "llvm" / "bin" / ("ld.lld.exe" if os.name == "nt" else "ld.lld")).is_file()


def test_mlir_or_opt_rocm_fallback_avoids_none_fingerprint(monkeypatch, tmp_path):
    """Dev builds without bundled bitcode still fingerprint a real ROCm tree."""
    for var in _ROCM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FLYDSL_COMPILE_ROCM_PATH", raising=False)
    monkeypatch.setattr(rocm_backend, "BUNDLED_ROCM_PATH", tmp_path / "no_bundle")
    fallback = _embedded_override_root(tmp_path / "mlir_rocm")
    monkeypatch.setattr(rocm_backend, "_mlir_baked_rocm_path", lambda: fallback)

    assert Path(rocm_toolkit_path()) == fallback
    fingerprint = RocmBackend(RocmBackend.make_target("gfx942")).toolkit_fingerprint()
    assert fingerprint != "rocm-bitcode:none"
    assert fingerprint.startswith("rocm-bitcode:")


def test_readonly_home_skips_therock_staging(monkeypatch, tmp_path):
    """A read-only HOME must not crash; staging failure skips that candidate."""
    for var in _ROCM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FLYDSL_COMPILE_ROCM_PATH", raising=False)
    monkeypatch.setattr(rocm_backend, "HAS_INPROCESS_LLD", False)
    monkeypatch.setattr(rocm_backend, "BUNDLED_ROCM_PATH", tmp_path / "no_bundle")
    monkeypatch.setattr(rocm_backend, "_mlir_baked_rocm_path", lambda: None)

    prefix = tmp_path / "rocm_sdk"
    llvm = prefix / "lib" / "llvm"
    (llvm / "amdgcn" / "bitcode").mkdir(parents=True)
    (llvm / "amdgcn" / "bitcode" / "ocml.bc").write_bytes(b"")
    (llvm / "bin").mkdir(parents=True)
    (llvm / "bin" / ("ld.lld.exe" if os.name == "nt" else "ld.lld")).write_bytes(b"")
    monkeypatch.setenv("ROCM_PATH", str(prefix))

    readonly_home = tmp_path / "ro_home"
    readonly_home.mkdir()
    readonly_home.chmod(0o555)
    monkeypatch.setenv("HOME", str(readonly_home))
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(readonly_home))

    # Staging cannot write under HOME; fall through to /opt/rocm or empty.
    # The important part is that resolution does not raise PermissionError.
    toolkit = rocm_toolkit_path()
    assert toolkit == "" or Path(toolkit).exists()


def test_capi_library_resolves_the_versioned_name(monkeypatch, tmp_path):
    """Wheels ship only ``libFlyPythonCAPI.so.<ver>``; the in-process LLD probe must still find it."""
    libs = tmp_path / "_mlir_libs"
    libs.mkdir()
    versioned = libs / f"{rocm_backend._mlir_capi_lib_name()}.24.0git"
    versioned.write_bytes(b"")
    monkeypatch.setattr(rocm_backend, "BUNDLED_ROCM_PATH", tmp_path / "_rocm")
    loaded = []
    monkeypatch.setattr("ctypes.CDLL", lambda path: loaded.append(path) or object())
    rocm_backend.load_capi_library.cache_clear()
    try:
        rocm_backend.load_capi_library()
    finally:
        rocm_backend.load_capi_library.cache_clear()
    assert loaded == [str(versioned)]
