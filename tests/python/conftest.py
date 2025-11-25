"""Pytest configuration for RocDSL tests."""

import pytest
from mlir.ir import Context, Location, Module, InsertionPoint
import ctypes
import os

# Register CuTe passes by loading the shared library
def register_cute_passes():
    """Register CuTe passes with MLIR."""
    try:
        # Try to find and load the CuTe dialect library
        lib_paths = [
            '/mnt/raid0/felix/rocDSL/build/lib/Dialect/Cute/libCuteDialect.so',
            '/mnt/raid0/felix/rocDSL/build/lib/libCuteDialect.so',
        ]
        
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                break
                
        # Also try loading transforms library
        transform_paths = [
            '/mnt/raid0/felix/rocDSL/build/lib/Transforms/libCuteTransforms.so',
            '/mnt/raid0/felix/rocDSL/build/lib/libCuteTransforms.so',
        ]
        
        for lib_path in transform_paths:
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                break
                
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load CuTe libraries: {e}")

# Register passes at module import time
register_cute_passes()


@pytest.fixture
def ctx():
    """Provide a fresh MLIR context for each test."""
    with Context() as context:
        # Allow unregistered dialects (for external dialects like cute)
        context.allow_unregistered_dialects = True
        
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


@pytest.fixture
def insert_point(ctx):
    """Provide insertion point for the module body."""
    with InsertionPoint(ctx.module.body):
        yield InsertionPoint.current


def pytest_sessionfinish(session, exitstatus):
    """Prevent pytest from erroring on empty test files."""
    if exitstatus == 5:
        session.exitstatus = 0
