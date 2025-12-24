"""Tests for core Flir operations: make_shape, make_stride, make_layout, size, rank."""

import pytest
from _mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from _mlir.dialects import arith

# Import Flir wrappers
import pyflir.dialects.ext.flir as flir
from pyflir.dialects.ext.arith import Index


def _unwrap(val):
    """Unwrap ArithValue to get underlying MLIR Value."""
    if hasattr(val, "value"):
        return val.value
    return val


def test_make_shape(ctx, insert_point):
    """Test shape creation."""
    
    @flir.jit(IndexType.get(), IndexType.get())
    def create_shape(dim0, dim1):
        shape = flir.make_shape(dim0, dim1)
        size = flir.size(shape)
        return (_unwrap(size),)  # Unwrap and return tuple
    
    # Verify the module
    ctx.module.operation.verify()
    # Apply lowering

    


def test_make_layout(ctx, insert_point):
    """Test layout creation from shape and stride."""
    
    @flir.jit(IndexType.get(), IndexType.get(), IndexType.get())
    def create_layout(dim0, dim1, stride_val):
        shape = flir.make_shape(dim0, dim1)
        stride = flir.make_stride(stride_val, dim0)
        layout = flir.make_layout(shape, stride)
        size = flir.size(layout)
        return (_unwrap(size),)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_constant_shape(ctx, insert_point):
    """Test shape with constant dimensions."""
    
    @flir.jit
    def constant_shape():
        c8 = Index(8)
        c16 = Index(16)
        
        shape = flir.make_shape(c8, c16)
        size = flir.size(shape)
        
        return (_unwrap(size),)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_rank_operation(ctx, insert_point):
    """Test rank operation."""
    
    @flir.jit(IndexType.get(), IndexType.get(), IndexType.get())
    def get_rank(dim0, dim1, dim2):
        shape = flir.make_shape(dim0, dim1, dim2)
        rank_val = flir.rank(shape)
        return (_unwrap(rank_val),)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_get_shape_stride(ctx, insert_point):
    """Test extracting shape and stride from layout."""
    
    @flir.jit(IndexType.get(), IndexType.get())
    def extract_components(dim0, dim1):
        c1 = Index(1)
        
        shape = flir.make_shape(dim0, dim1)
        stride = flir.make_stride(c1, dim0)
        layout = flir.make_layout(shape, stride)
        
        extracted_shape = flir.get_shape(layout)
        extracted_stride = flir.get_stride(layout)
        
        size1 = flir.size(extracted_shape)
        size2 = flir.cosize(layout)
        
        result = arith.addi(_unwrap(size1), _unwrap(size2))
        return (_unwrap(result),)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_2d_layout(ctx, insert_point):
    """Test 2D column-major layout."""
    
    @flir.jit
    def layout_2d():
        # Create 8x16 column-major layout
        c8 = Index(8)
        c16 = Index(16)
        c1 = Index(1)
        
        shape = flir.make_shape(c8, c16)
        stride = flir.make_stride(c1, c8)  # Column-major: stride (1, 8)
        layout = flir.make_layout(shape, stride)
        
        size = flir.size(layout)  # Should be 128
        return (_unwrap(size),)
    
    ctx.module.operation.verify()
    # Apply lowering

    
