"""Pytest configuration for RocDSL tests.

This test suite now uses RocDSL's embedded MLIR Python bindings (the `_mlir`
package under `build/python_packages/rocdsl`) and no longer relies on an
external MLIR Python installation.
"""

import pytest

from rocdsl.compiler.context import ensure_rocir_python_extensions
from mlir.ir import Context, Location, Module, InsertionPoint


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
