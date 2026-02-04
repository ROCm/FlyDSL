"""
FlyDSL MLIR Python Bindings

This module provides self-contained MLIR Python bindings for FlyDSL,
following the NV Cutlass pattern. All MLIR functionality is under flydsl._mlir.

Structure:
    flydsl._mlir.ir       - Core IR types (Context, Type, Attribute, etc.)
    flydsl._mlir.dialects - Dialect bindings (fly, arith, func, etc.)

Usage:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import fly
    
    ctx = ir.Context()
    # Use FlyDSL types
"""

# Re-export core IR types from _fly extension
# The _fly extension has populateIRCore/Types/Attributes registered
from .._mlir._mlir_libs._fly import (
    # Core classes
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
    # Specific types (from populateIRTypes)
    IntegerType,
    IndexType,
    F16Type,
    F32Type,
    F64Type,
    BF16Type,
    MemRefType,
    VectorType,
    FunctionType,
    # Specific attributes (from populateIRAttributes)
    IntegerAttr,
    FloatAttr,
    StringAttr,
    ArrayAttr,
    DictAttr,
    TypeAttr,
    UnitAttr,
    SymbolRefAttr,
    FlatSymbolRefAttr,
    # Utilities
    TypeID,
    # Registration functions
    register_dialect,
    register_operation,
    register_type_caster,
    register_value_caster,
)

# register_op_adaptor is only available in newer MLIR versions
# For older MLIR, provide a no-op stub since the generated ops don't use OpAdaptor
try:
    from .._mlir._mlir_libs._fly import register_op_adaptor
except ImportError:
    def register_op_adaptor(kind, replace=False):
        """No-op stub for older MLIR versions without OpAdaptor support."""
        def decorator(adaptor_class):
            return adaptor_class
        return decorator

# Create an 'ir' namespace for compatibility with mlir.ir import style
class _IRNamespace:
    """Namespace that mimics mlir.ir for compatibility"""
    Context = Context
    Module = Module
    Location = Location
    InsertionPoint = InsertionPoint
    Attribute = Attribute
    Type = Type
    Value = Value
    Block = Block
    Region = Region
    Operation = Operation
    OpView = OpView
    IntegerType = IntegerType
    IndexType = IndexType
    F16Type = F16Type
    F32Type = F32Type
    F64Type = F64Type
    BF16Type = BF16Type
    MemRefType = MemRefType
    VectorType = VectorType
    FunctionType = FunctionType
    IntegerAttr = IntegerAttr
    FloatAttr = FloatAttr
    StringAttr = StringAttr
    ArrayAttr = ArrayAttr
    DictAttr = DictAttr
    TypeAttr = TypeAttr
    UnitAttr = UnitAttr
    SymbolRefAttr = SymbolRefAttr
    FlatSymbolRefAttr = FlatSymbolRefAttr
    TypeID = TypeID
    # Registration functions
    register_dialect = register_dialect
    register_operation = register_operation
    register_op_adaptor = register_op_adaptor
    register_type_caster = register_type_caster
    register_value_caster = register_value_caster

ir = _IRNamespace()

__all__ = [
    'ir',
    'Context', 'Module', 'Location', 'InsertionPoint',
    'Attribute', 'Type', 'Value', 'Block', 'Region', 'Operation', 'OpView',
    'IntegerType', 'IndexType', 'F16Type', 'F32Type', 'F64Type', 'BF16Type',
    'MemRefType', 'VectorType', 'FunctionType',
    'IntegerAttr', 'FloatAttr', 'StringAttr', 'ArrayAttr', 'DictAttr',
    'TypeAttr', 'UnitAttr', 'TypeID',
    'register_dialect', 'register_operation', 'register_op_adaptor',
    'register_type_caster', 'register_value_caster',
]
