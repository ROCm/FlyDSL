"""
FlyDSL Dialect Bindings

This module provides Python bindings for MLIR dialects used by FlyDSL.
Following the NV Cutlass pattern, all dialects are self-contained under flydsl._mlir.dialects.

Available dialects:
    - fly: FlyDSL's custom dialect for GPU kernels
    - fly_rocdl: FlyDSL's ROCDL extensions

Usage:
    from flydsl._mlir.dialects import fly
    from flydsl._mlir.dialects import fly_rocdl
"""

# Note: Actual dialect implementations will be imported from _fly and _fly_rocdl extensions
# This __init__.py just provides the package structure

__all__ = []
