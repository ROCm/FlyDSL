"""Pytest configuration for RocDSL tests.

This test suite now uses RocDSL's embedded MLIR Python bindings (the `_mlir`
package under `build/python_packages/rocdsl`) and no longer relies on an
external MLIR Python installation.
"""

import os
import sys
from pathlib import Path

# Ensure embedded `_mlir` is importable for the test suite.
# This must run before importing `rocdsl` (which imports `_mlir` at module import time).
_repo_root = Path(__file__).resolve().parents[2]

# Preferred build layout (new): `.rocdsl/build` (see build.sh)
_embedded_pkg_dir = _repo_root / ".rocdsl" / "build" / "python_packages" / "rocdsl"
# Legacy fallback: `build/python_packages/rocdsl`
if not _embedded_pkg_dir.exists():
    _embedded_pkg_dir = _repo_root / "build" / "python_packages" / "rocdsl"

# Prefer in-tree Python sources (so tests exercise source `rocdsl/`), while still
# making the embedded `_mlir` runtime available for native extensions.
_src_py_dir = _repo_root / "python"
if _src_py_dir.exists():
    _p2 = str(_src_py_dir)
    if _p2 in sys.path:
        sys.path.remove(_p2)
    sys.path.insert(0, _p2)

if _embedded_pkg_dir.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
    # Help rocdsl locate the embedded python_packages root correctly.
    os.environ.setdefault("ROCDSL_BUILD_DIR", str(_embedded_pkg_dir.parents[2]))
    _p = str(_embedded_pkg_dir)
    if _p in sys.path:
        sys.path.remove(_p)
    # Keep embedded after source so `import rocdsl` resolves to in-tree package.
    sys.path.insert(1, _p)

import pytest

from rocdsl.compiler.context import ensure_rocir_python_extensions
from _mlir.ir import Context, Location, Module, InsertionPoint


@pytest.fixture
def ctx():
    """Provide a fresh MLIR context for each test."""
    with Context() as context:
        # Ensure Rocir + upstream dialects/passes/translations are registered.
        ensure_rocir_python_extensions(context)
        
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
