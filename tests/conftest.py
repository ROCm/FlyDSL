# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Pytest configuration for Fly DSL tests.

Supports both the new Fly dialect (build-fly/) and legacy build paths.
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
    _already = _p in sys.path or any(os.path.isdir(ep) and os.path.samefile(ep, _p) for ep in sys.path if ep)
    if not _already:
        sys.path.insert(0, _p)

# Legacy: .flydsl/build or build/
for _legacy in [
    _repo_root / ".flydsl" / "build" / "python_packages" / "flydsl",
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
    from flydsl.compiler.context import ensure_flydsl_python_extensions

    _ensure_extensions = ensure_flydsl_python_extensions
except ImportError:
    pass

try:
    from flydsl._mlir.ir import Context, InsertionPoint, Location, Module
except ImportError:
    try:
        from _mlir.ir import Context, InsertionPoint, Location, Module
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
            yield type(
                "MLIRContext",
                (),
                {
                    "context": context,
                    "module": module,
                    "location": Location.unknown(context),
                },
            )()


@pytest.fixture
def module(ctx):
    """Provide module from context."""
    return ctx.module


@pytest.fixture
def insert_point(ctx):
    """Provide insertion point for the module body."""
    with InsertionPoint(ctx.module.body):
        yield InsertionPoint.current


def pytest_addoption(parser):
    """Add FlyDSL test-session options that map to env variables."""
    group = parser.getgroup("flydsl")
    group.addoption(
        "--flydsl-compile-backend",
        action="store",
        default=None,
        help="Set FLYDSL_COMPILE_BACKEND for this pytest session.",
    )
    group.addoption(
        "--flydsl-compile-arch",
        action="store",
        default=None,
        help="Set ARCH for this pytest session.",
    )


_AITER_ERROR = None
try:
    import aiter as _aiter
except Exception as _exc:  # a version-mismatched aiter raises ImportError from inside itself
    _aiter = None
    _AITER_ERROR = _exc


def pytest_report_header(config):
    """Say whether aiter imported.

    Nineteen suites skip themselves at module level when it does not, and
    pytest_sessionfinish below rewrites the resulting exit code 5 to 0, so a run
    that collected nothing is indistinguishable from a run that passed.
    """
    if _aiter is not None:
        return f"aiter: {getattr(_aiter, '__version__', 'unknown version')}"
    return f"aiter: NOT IMPORTABLE ({_AITER_ERROR}) -- the suites that need it will collect nothing"


def pytest_configure(config):
    """Apply FlyDSL env overrides from CLI options.

    Note: marker registration lives in pytest.ini (single source of truth).
    """
    backend = config.getoption("--flydsl-compile-backend")
    arch = config.getoption("--flydsl-compile-arch")
    # Intentionally set process-level env vars so downstream code (env.py)
    # picks them up. The pytest process exits after the session, so no cleanup needed.
    if backend:
        os.environ["FLYDSL_COMPILE_BACKEND"] = backend
    if arch:
        os.environ["ARCH"] = arch

    # Opt-in for CI, where a silently empty run is worse than a red one.
    if _aiter is None and os.environ.get("FLYDSL_REQUIRE_AITER", "") not in ("", "0", "false", "False"):
        pytest.exit(f"FLYDSL_REQUIRE_AITER is set but aiter is not importable: {_AITER_ERROR}", returncode=1)


def pytest_sessionfinish(session, exitstatus):
    """Prevent pytest from erroring on empty test files."""
    if exitstatus == 5:
        session.exitstatus = 0
