"""
FlyDSL IR Module

This module provides all IR types and utilities from flydsl's _fly extension.
It mirrors the mlir.ir module structure for compatibility.
"""

# Re-export everything from _fly extension
from ._mlir_libs._fly import *

# Explicitly import commonly used items for IDE support
from ._mlir_libs._fly import (
    Context,
    Module,
    Location,
    InsertionPoint,
    Attribute,
    Type,
    Value,
    Block,
    Region,
    Operation,
    OpView,
    IntegerType,
    IndexType,
    F16Type,
    F32Type,
    F64Type,
    BF16Type,
    MemRefType,
    VectorType,
    FunctionType,
    IntegerAttr,
    FloatAttr,
    StringAttr,
    ArrayAttr,
    DictAttr,
    TypeAttr,
    UnitAttr,
    SymbolRefAttr,
    FlatSymbolRefAttr,
    TypeID,
    register_dialect,
    register_operation,
    register_type_caster,
    register_value_caster,
)

# register_op_adaptor is only available in newer MLIR versions
# For older MLIR, provide a no-op stub since the generated ops don't use OpAdaptor
try:
    from ._mlir_libs._fly import register_op_adaptor
except ImportError:
    def register_op_adaptor(kind, replace=False):
        """No-op stub for older MLIR versions without OpAdaptor support."""
        def decorator(adaptor_class):
            return adaptor_class
        return decorator

# Import register_attribute_builder from upstream if not in _fly
try:
    from ._mlir_libs._fly import register_attribute_builder
except ImportError:
    from mlir.ir import register_attribute_builder
