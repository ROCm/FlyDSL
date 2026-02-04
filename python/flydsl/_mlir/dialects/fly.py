"""
FlyDSL Fly Dialect Python Bindings

This module provides Python bindings for the Fly dialect.
All types and operations from the Fly dialect are exposed here.

Usage:
    from flydsl._mlir.dialects import fly
    from flydsl._mlir import ir
    
    ctx = ir.Context()
    # Register Fly dialect
    from flydsl._mlir._mlir_libs._fly import _register_dialect
    _register_dialect(ctx)
    
    # Use Fly types
    int_tuple = fly.IntTupleType.get(42, ctx)
"""

# Import all Fly dialect types from _fly extension
from ..._mlir._mlir_libs._fly import (
    # Fly Types
    IntTupleType,
    LayoutType,
    SwizzleType,
    PointerType,
    CopyAtomUniversalCopyType,
    MmaAtomUniversalFMAType,
    
    # Utilities
    DLTensorAdaptor,
    
    # Helper functions
    infer_int_tuple_type,
    rank,
    depth,
)

# Import all generated operations from _fly_ops_gen
from ._fly_ops_gen import *

# Import enums
from ._fly_enum_gen import *

__all__ = [
    # Types
    'IntTupleType',
    'LayoutType',
    'SwizzleType',
    'PointerType',
    'CopyAtomUniversalCopyType',
    'MmaAtomUniversalFMAType',
    'DLTensorAdaptor',
    # Helper functions
    'infer_int_tuple_type',
    'rank',
    'depth',
]
