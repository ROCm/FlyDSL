"""Test basic CuTe operations: make_shape, make_stride, make_layout, size, rank, etc."""

import pytest
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func

try:
    from rocdsl.dialects.ext import arith, cute
except ImportError:
    pytest.skip("RocDSL dialect not available", allow_module_level=True)


def test_make_shape(ctx):
    """Test creating shapes with different ranks."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_shape_rank2")
        def test_shape():
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            
            # Create 2D shape
            shape = cute.make_shape(c8, c16)
            
            # Verify rank
            rank = cute.rank(shape)
            return rank
    
    # Verify the module
    ctx.module.operation.verify()
    
    # Check IR contains make_shape
    ir = str(ctx.module)
    assert "cute.make_shape" in ir
    assert "!cute.shape<2>" in ir


def test_make_layout(ctx):
    """Test creating layouts from shape and stride."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_layout_creation")
        def test_layout():
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            # Create shape and stride
            shape = cute.make_shape(c8, c16)
            stride = cute.make_stride(c1, c8)
            
            # Create layout (column-major 8x16)
            layout = cute.make_layout(shape, stride)
            
            return layout
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.make_layout" in ir
    assert "!cute.layout<2>" in ir


def test_size_operation(ctx):
    """Test size computation for shapes and layouts."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_size")
        def test_size():
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            
            # Create shape
            shape = cute.make_shape(c4, c8)
            
            # Compute size (should be 32) - Pythonic way!
            total_size = cute.size(shape)
            
            # Could also verify: 4 * 8 = 32
            expected = c4 * c8  # Using * operator instead of MulIOp!
            
            return total_size
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.size" in ir
    assert "arith.muli" in ir  # From c4 * c8


def test_get_shape_stride(ctx):
    """Test extracting shape and stride from layout."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_extract")
        def test_extract():
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            shape = cute.make_shape(c8, c16)
            stride = cute.make_stride(c1, c8)
            layout = cute.make_layout(shape, stride)
            
            # Extract shape and stride
            extracted_shape = cute.get_shape(layout)
            extracted_stride = cute.get_stride(layout)
            
            # Compute size from extracted shape
            size_val = cute.size(extracted_shape)
            
            return size_val
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.get_shape" in ir
    assert "cute.get_stride" in ir


def test_rank_operation(ctx):
    """Test rank operation on shapes and layouts."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_rank")
        def test_rank_func():
            c2 = arith.constant(2, index=True)
            c3 = arith.constant(3, index=True)
            c4 = arith.constant(4, index=True)
            
            # Create 3D shape
            shape = cute.make_shape(c2, c3, c4)
            
            # Get rank (should be 3)
            rank_val = cute.rank(shape)
            
            return rank_val
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.rank" in ir
    assert "!cute.shape<3>" in ir


def test_cosize_operation(ctx):
    """Test cosize (stride extent) computation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_cosize")
        def test_cosize_func():
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            shape = cute.make_shape(c8, c16)
            stride = cute.make_stride(c1, c8)
            layout = cute.make_layout(shape, stride)
            
            # Compute cosize
            cosize_val = cute.cosize(layout)
            
            return cosize_val
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.cosize" in ir


def test_composition(ctx):
    """Test layout composition with Pythonic operators."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_compose")
        def test_compose():
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c4 = arith.constant(4, index=True)
            c2 = arith.constant(2, index=True)
            c1 = arith.constant(1, index=True)
            
            # First layout
            shape_a = cute.make_shape(c8, c16)
            stride_a = cute.make_stride(c1, c8)
            layout_a = cute.make_layout(shape_a, stride_a)
            
            # Second layout
            shape_b = cute.make_shape(c4, c2)
            stride_b = cute.make_stride(c2, c1)
            layout_b = cute.make_layout(shape_b, stride_b)
            
            # Compose
            composed = cute.composition(layout_a, layout_b)
            
            # Use Pythonic operator for verification
            total_elements = c8 * c16  # Instead of MulIOp(c8, c16)
            
            return composed
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.composition" in ir
    assert "arith.muli" in ir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
