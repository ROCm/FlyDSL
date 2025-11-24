"""Pytest configuration for RocDSL tests."""

import pytest
from mlir.ir import Context, Location, Module, InsertionPoint


@pytest.fixture
def ctx():
    """Provide a fresh MLIR context for each test."""
    with Context() as context:
        # Load required dialects
        context.load_all_available_dialects()
        
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


def pytest_sessionfinish(session, exitstatus):
    """Prevent pytest from erroring on empty test files."""
    if exitstatus == 5:
        session.exitstatus = 0
