"""
FlyDSL Extras - Type Utilities

This module provides type utilities and aliases, similar to mlir.extras.types,
but using flydsl's own type system.
"""

from ..._mlir import ir

# Type classes
F16Type = ir.F16Type
F32Type = ir.F32Type
F64Type = ir.F64Type
BF16Type = ir.BF16Type
IndexType = ir.IndexType
IntegerType = ir.IntegerType
MemRefType = ir.MemRefType
VectorType = ir.VectorType
FunctionType = ir.FunctionType

# Type factory functions (matching mlir.extras.types API)
def f16():
    """Get F16 type"""
    return F16Type.get()

def f32():
    """Get F32 type"""
    return F32Type.get()

def f64():
    """Get F64 type"""
    return F64Type.get()

def bf16():
    """Get BF16 type"""
    return BF16Type.get()

def index():
    """Get Index type"""
    return IndexType.get()

# Integer type factory functions
def i1():
    """Get i1 (boolean) type"""
    return IntegerType.get_signless(1)

def i8():
    """Get i8 type"""
    return IntegerType.get_signless(8)

def i16():
    """Get i16 type"""
    return IntegerType.get_signless(16)

def i32():
    """Get i32 type"""
    return IntegerType.get_signless(32)

def i64():
    """Get i64 type"""
    return IntegerType.get_signless(64)

__all__ = [
    # Type classes
    'F16Type', 'F32Type', 'F64Type', 'BF16Type',
    'IndexType', 'IntegerType', 'MemRefType', 'VectorType', 'FunctionType',
    # Factory functions
    'f16', 'f32', 'f64', 'bf16', 'index',
    'i1', 'i8', 'i16', 'i32', 'i64',
]
