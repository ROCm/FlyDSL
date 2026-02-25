"""Pytest configuration for Fly DSL tests.

Supports both the new Fly dialect (build-fly/) and the legacy FLIR dialect.
"""

import os
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[1]

# New Fly dialect build
_fly_pkg_dir = _repo_root / "build-fly" / "python_packages"
if _fly_pkg_dir.exists():
    _p = str(_fly_pkg_dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Legacy: .flir/build or build/ (old flir dialect)
for _legacy in [
    _repo_root / ".flir" / "build" / "python_packages" / "flydsl",
    _repo_root / "build" / "python_packages" / "flydsl",
    _repo_root / "build" / "lib.linux-x86_64-cpython-312",
]:
    if _legacy.exists():
        _p = str(_legacy)
        if _p not in sys.path:
            sys.path.append(_p)
        break

# Legacy: in-tree flydsl source (for old API tests)
_src_py_dir = _repo_root / "flydsl" / "src"
if _src_py_dir.exists() and (_src_py_dir / "flydsl").exists():
    _p = str(_src_py_dir)
    if _p not in sys.path:
        sys.path.append(_p)

# Try importing new or old context setup
_ensure_extensions = None
try:
    from flydsl.compiler.context import ensure_flir_python_extensions
    _ensure_extensions = ensure_flir_python_extensions
except ImportError:
    pass

try:
    from flydsl._mlir.ir import Context, Location, Module, InsertionPoint
except ImportError:
    try:
        from _mlir.ir import Context, Location, Module, InsertionPoint
    except ImportError:
        Context = Location = Module = InsertionPoint = None


@pytest.fixture
def ctx():
    """Provide a fresh MLIR context for each test."""
    if Context is None:
        pytest.skip("MLIR Python bindings not available")
    with Context() as context:
        if _ensure_extensions:
            _ensure_extensions(context)
        with Location.unknown(context):
            module = Module.create()
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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "large_shape: marks tests with large shapes that are slow to run (deselect with '-m \"not large_shape\"')",
    )


def pytest_sessionfinish(session, exitstatus):
    """Prevent pytest from erroring on empty test files."""
    if exitstatus == 5:
        session.exitstatus = 0
