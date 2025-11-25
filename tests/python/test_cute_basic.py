"""Tests for basic CuTe operations: make_shape, make_stride, make_layout, size, rank."""

import pytest
from mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from mlir.dialects import func, arith

# Import our CuTe wrappers
import rocdsl.dialects.ext.cute as cute


def test_make_shape(ctx, insert_point):
    """Test shape creation."""
    
    @func.FuncOp.from_py_func(IndexType.get(), IndexType.get())
    def create_shape(dim0, dim1):
        shape = cute.make_shape(dim0, dim1)
        size = cute.size(shape)
        return size
    
    # Verify the module
    ctx.module.operation.verify()
    # Apply lowering

    


def test_make_layout(ctx, insert_point):
    """Test layout creation from shape and stride."""
    
    @func.FuncOp.from_py_func(IndexType.get(), IndexType.get(), IndexType.get())
    def create_layout(dim0, dim1, stride_val):
        shape = cute.make_shape(dim0, dim1)
        stride = cute.make_stride(stride_val, dim0)
        layout = cute.make_layout(shape, stride)
        size = cute.size(layout)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_constant_shape(ctx, insert_point):
    """Test shape with constant dimensions."""
    
    @func.FuncOp.from_py_func()
    def constant_shape():
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        
        shape = cute.make_shape(c8, c16)
        size = cute.size(shape)
        
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_rank_operation(ctx, insert_point):
    """Test rank operation."""
    
    @func.FuncOp.from_py_func(IndexType.get(), IndexType.get(), IndexType.get())
    def get_rank(dim0, dim1, dim2):
        shape = cute.make_shape(dim0, dim1, dim2)
        rank_val = cute.rank(shape)
        return rank_val
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_get_shape_stride(ctx, insert_point):
    """Test extracting shape and stride from layout."""
    
    @func.FuncOp.from_py_func(IndexType.get(), IndexType.get())
    def extract_components(dim0, dim1):
        c1 = arith.constant(IndexType.get(), 1)
        
        shape = cute.make_shape(dim0, dim1)
        stride = cute.make_stride(c1, dim0)
        layout = cute.make_layout(shape, stride)
        
        extracted_shape = cute.get_shape(layout)
        extracted_stride = cute.get_stride(layout)
        
        size1 = cute.size(extracted_shape)
        size2 = cute.cosize(layout)
        
        result = arith.addi(size1, size2)
        return result
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_2d_layout(ctx, insert_point):
    """Test 2D column-major layout."""
    
    @func.FuncOp.from_py_func()
    def layout_2d():
        # Create 8x16 column-major layout
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c1 = arith.constant(IndexType.get(), 1)
        
        shape = cute.make_shape(c8, c16)
        stride = cute.make_stride(c1, c8)  # Column-major: stride (1, 8)
        layout = cute.make_layout(shape, stride)
        
        size = cute.size(layout)  # Should be 128
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    
