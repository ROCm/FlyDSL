"""Test operator overloading for elegant Pythonic syntax."""

import pytest
from _mlir.ir import Context, Location, Module, InsertionPoint

try:
    from pyflir.dialects.ext import arith, flir
except ImportError:
    pytest.skip("FLIR dialect not available", allow_module_level=True)


def test_arithmetic_operators(ctx):
    """Test +, -, *, / operators."""
    with InsertionPoint(ctx.module.body):
        @flir.jit
        def test_arithmetic():
            # Create constants
            a = arith.index(10)
            b = arith.index(3)
            
            # Use Python operators instead of explicit ops
            c = a + b      # AddIOp
            d = a - b      # SubIOp
            e = a * b      # MulIOp
            f = a // b     # DivSIOp
            
            return c
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_mixed_operators(ctx):
    """Test operators with Python literals."""
    with InsertionPoint(ctx.module.body):
        @flir.jit
        def test_mixed():
            a = arith.index(100)
            
            # Mix with Python int
            b = a + 50     # Should create constant 50 and add
            c = a * 2      # Should create constant 2 and multiply
            d = 10 + a     # Reverse operator
            
            return b
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_comparison_operators(ctx):
    """Test <, >, <=, >=, ==, != operators."""
    with InsertionPoint(ctx.module.body):
        @flir.jit
        def test_comparison():
            a = arith.index(10)
            b = arith.index(5)
            
            # Comparison operators
            lt = a < b     # CmpIOp(slt)
            le = a <= b    # CmpIOp(sle)
            gt = a > b     # CmpIOp(sgt)
            ge = a >= b    # CmpIOp(sge)
            eq = a == b    # CmpIOp(eq)
            ne = a != b    # CmpIOp(ne)
            
            return lt
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_float_operators(ctx):
    """Test operators with floating point values."""
    with InsertionPoint(ctx.module.body):
        @flir.jit
        def test_float_ops():
            a = arith.f32(3.14)
            b = arith.f32(2.0)
            
            # Float arithmetic
            c = a + b      # AddFOp
            d = a * b      # MulFOp
            e = a / b      # DivFOp
            
            # Float comparison
            gt = a > b     # CmpFOp(ogt)
            
            return c
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_chained_operations(ctx):
    """Test chaining multiple operators."""
    with InsertionPoint(ctx.module.body):
        @flir.jit
        def test_chained():
            a = arith.index(10)
            b = arith.index(5)
            c = arith.index(2)
            
            # Complex expression: (a + b) * c - 3
            result = (a + b) * c - 3
            
            return result
    
    ctx.module.operation.verify()
    # Apply lowering

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
