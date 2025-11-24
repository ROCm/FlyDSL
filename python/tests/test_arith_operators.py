"""Test operator overloading for elegant Pythonic syntax."""

import pytest
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func

try:
    from rocdsl.dialects.ext import arith
except ImportError:
    pytest.skip("RocDSL dialect not available", allow_module_level=True)


def test_arithmetic_operators(ctx):
    """Test +, -, *, / operators."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_arithmetic")
        def test_ops():
            # Create constants
            a = arith.index(10)
            b = arith.index(3)
            
            # Use Python operators instead of explicit ops
            c = a + b      # AddIOp
            d = a - b      # SubIOp
            e = a * b      # MulIOp
            f = a // b     # DivSIOp
            
            return c.value
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "arith.addi" in ir
    assert "arith.subi" in ir
    assert "arith.muli" in ir
    assert "arith.divsi" in ir


def test_mixed_operators(ctx):
    """Test operators with Python literals."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_mixed")
        def test_mixed():
            a = arith.index(100)
            
            # Mix with Python int
            b = a + 50     # Should create constant 50 and add
            c = a * 2      # Should create constant 2 and multiply
            d = 10 + a     # Reverse operator
            
            return b.value
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "arith.addi" in ir
    assert "arith.muli" in ir


def test_comparison_operators(ctx):
    """Test <, >, <=, >=, ==, != operators."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_comparison")
        def test_cmp():
            a = arith.index(10)
            b = arith.index(5)
            
            # Comparison operators
            lt = a < b     # CmpIOp(slt)
            le = a <= b    # CmpIOp(sle)
            gt = a > b     # CmpIOp(sgt)
            ge = a >= b    # CmpIOp(sge)
            eq = a == b    # CmpIOp(eq)
            ne = a != b    # CmpIOp(ne)
            
            return lt.value
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "arith.cmpi" in ir


def test_float_operators(ctx):
    """Test operators with floating point values."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_float_ops")
        def test_float():
            a = arith.f32(3.14)
            b = arith.f32(2.0)
            
            # Float arithmetic
            c = a + b      # AddFOp
            d = a * b      # MulFOp
            e = a / b      # DivFOp
            
            # Float comparison
            gt = a > b     # CmpFOp(ogt)
            
            return c.value
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "arith.addf" in ir
    assert "arith.mulf" in ir
    assert "arith.divf" in ir
    assert "arith.cmpf" in ir


def test_chained_operations(ctx):
    """Test chaining multiple operators."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_chained")
        def test_chain():
            a = arith.index(10)
            b = arith.index(5)
            c = arith.index(2)
            
            # Complex expression: (a + b) * c - 3
            result = (a + b) * c - 3
            
            return result.value
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    # Should have add, mul, and sub
    assert ir.count("arith.addi") >= 1
    assert ir.count("arith.muli") >= 1
    assert ir.count("arith.subi") >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
