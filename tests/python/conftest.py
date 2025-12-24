"""Pytest configuration for FLIR tests.

This test suite now uses FLIR's embedded MLIR Python bindings (the `_mlir`
package under `build/python_packages/pyflir`) and no longer relies on an
external MLIR Python installation.
"""

import os
import sys
from pathlib import Path

# Ensure embedded `_mlir` is importable for the test suite.
# This must run before importing `pyflir` (which imports `_mlir` at module import time).
_repo_root = Path(__file__).resolve().parents[2]

# Preferred build layout (new): `.flir/build` (see build.sh)
_embedded_pkg_dir = _repo_root / ".flir" / "build" / "python_packages" / "pyflir"
# Legacy fallback: `build/python_packages/pyflir`
if not _embedded_pkg_dir.exists():
    _embedded_pkg_dir = _repo_root / "build" / "python_packages" / "pyflir"

# Prefer in-tree Python sources (so tests exercise source `flir/`), while still
# making the embedded `_mlir` runtime available for native extensions.
_src_py_dir = _repo_root / "pyflir" / "src"
if _src_py_dir.exists():
    _p2 = str(_src_py_dir)
    if _p2 in sys.path:
        sys.path.remove(_p2)
    sys.path.insert(0, _p2)

if _embedded_pkg_dir.exists():
    os.environ.setdefault("FLIR_USE_EMBEDDED_MLIR", "1")
    # Help flir locate the embedded python_packages root correctly.
    os.environ.setdefault("FLIR_BUILD_DIR", str(_embedded_pkg_dir.parents[2]))
    _p = str(_embedded_pkg_dir)
    if _p in sys.path:
        sys.path.remove(_p)
    # Keep embedded after source so imports resolve to in-tree sources.
    sys.path.insert(1, _p)

import pytest

from pyflir.compiler.context import ensure_flir_python_extensions
from _mlir.ir import Context, Location, Module, InsertionPoint


@pytest.fixture
def ctx():
    """Provide a fresh MLIR context for each test."""
    with Context() as context:
        # Ensure Flir + upstream dialects/passes/translations are registered.
        ensure_flir_python_extensions(context)
        
        # Set default location
        with Location.unknown(context):
            # Create module and set up insertion point
            module = Module.create()
            
            # Provide context, module, and insertion point
            yield type("MLIRContext", (), {
                "context": context,
                "module": module,
                "location": Location.unknown(context),
            })()


@pytest.fixture
def module(ctx):
    """Provide module from context."""
    return ctx.module


@pytest.fixture
def insert_point(ctx):
    """Provide insertion point for the module body."""
    with InsertionPoint(ctx.module.body):
        yield InsertionPoint.current


def pytest_sessionfinish(session, exitstatus):
    """Prevent pytest from erroring on empty test files."""
    if exitstatus == 5:
        session.exitstatus = 0
