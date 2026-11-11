# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Native-library lookups in a split symlink-tree install.

A build system can install a package as a tree of per-file symlinks where the
pure-Python files and the native libraries link into different directories
(e.g. when the files shared by the per-Python-version wheels are stored once).
Python imports ``flydsl._mlir._mlir_libs`` from the symlink tree, so FlyDSL
must look for its native files there too, not next to the resolved
``__file__``.
"""

import hashlib
import sys
from pathlib import Path

import pytest

import flydsl
from flydsl.compiler import jit_executor, jit_function
from flydsl.compiler.backends import _get_backend_class, get_backend
from flydsl.runtime import _libraries

pytestmark = [
    pytest.mark.l0_backend_agnostic,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX symlinks"),
]

LIBS = "flydsl/_mlir/_mlir_libs"


def _link(tree: Path, real_root: Path, rel: str, data: bytes = b"") -> Path:
    """Write ``real_root/rel`` and symlink ``tree/rel`` to it."""
    real = real_root / rel
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes(data)
    link = tree / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    return real


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Symlink tree whose .py files link into ``py/``; tests link native files into ``native/``."""
    tree = tmp_path / "tree"
    _link(tree, tmp_path / "py", "flydsl/__init__.py")
    _link(tree, tmp_path / "py", "flydsl/compiler/jit_executor.py")
    monkeypatch.setattr(flydsl, "__file__", str(tree / "flydsl/__init__.py"))
    monkeypatch.setattr(jit_executor, "__file__", str(tree / "flydsl/compiler/jit_executor.py"))
    monkeypatch.setenv("ARCH", "gfx942")  # skip GPU detection
    jit_executor._resolve_runtime_libs.cache_clear()
    yield tree
    jit_executor._resolve_runtime_libs.cache_clear()


def test_aot_runtime_archive(tree, tmp_path):
    basename = _get_backend_class().aot_runtime_config().archive_basename
    real = _link(tree, tmp_path / "native", f"{LIBS}/{basename}")
    assert _libraries._find_aot_runtime_archive() == real.resolve()


def test_jit_runtime_libs(tree, tmp_path):
    names = get_backend().jit_runtime_lib_basenames()
    for name in names:
        _link(tree, tmp_path / "native", f"{LIBS}/{name}")
    assert [Path(p).name for p in jit_executor._resolve_runtime_libs()] == names


def test_cache_key_hashes_native_libs(tree, tmp_path):
    data = b"native library"
    _link(tree, tmp_path / "native", f"{LIBS}/libfly_jit_runtime.so", data)
    key = jit_function._flydsl_key_cached.__wrapped__(False, "")
    assert f"libfly_jit_runtime.so:{hashlib.sha256(data).hexdigest()}" in key


def test_native_digests_skip_soname_aliases(tmp_path):
    (tmp_path / "libfoo.so.1").write_bytes(b"foo")
    (tmp_path / "libfoo.so").symlink_to("libfoo.so.1")
    assert jit_function._native_library_digests(tmp_path) == [f"libfoo.so.1:{hashlib.sha256(b'foo').hexdigest()}"]
