"""Tests for CuTe product operations (tiling)."""

import pytest
from mlir.ir import IndexType
from mlir.dialects import func, arith

import rocdsl.dialects.ext.rocir as rocir


def test_logical_product(ctx, insert_point):
    """Test logical product for basic tiling."""
    
    @func.FuncOp.from_py_func()
    def tile_layout():
        # Base layout: 16x32
        c16 = arith.constant(IndexType.get(), 16)
        c32 = arith.constant(IndexType.get(), 32)
        c1 = arith.constant(IndexType.get(), 1)
        
        base_shape = rocir.make_shape(c16, c32)
        base_stride = rocir.make_stride(c1, c16)
        base = rocir.make_layout(base_shape, base_stride)
        
        # Tiler: 4x8
        c4 = arith.constant(IndexType.get(), 4)
        c8 = arith.constant(IndexType.get(), 8)
        
        tile_shape = rocir.make_shape(c4, c8)
        tile_stride = rocir.make_stride(c1, c4)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        
        # Product creates 4D tiled layout
        tiled = rocir.logical_product(base, tiler)
        
        size = rocir.size(tiled)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_zipped_product(ctx, insert_point):
    """Test zipped product for interleaved tiling."""
    
    @func.FuncOp.from_py_func()
    def zipped_tile():
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c2 = arith.constant(IndexType.get(), 2)
        c4 = arith.constant(IndexType.get(), 4)
        c1 = arith.constant(IndexType.get(), 1)
        
        base_shape = rocir.make_shape(c8, c16)
        base_stride = rocir.make_stride(c1, c8)
        base = rocir.make_layout(base_shape, base_stride)
        
        tile_shape = rocir.make_shape(c2, c4)
        tile_stride = rocir.make_stride(c1, c2)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        
        zipped = rocir.zipped_product(base, tiler)
        
        size = rocir.size(zipped)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_tiled_product(ctx, insert_point):
    """Test tiled product."""
    
    @func.FuncOp.from_py_func()
    def tiled_layout():
        c32 = arith.constant(IndexType.get(), 32)
        c64 = arith.constant(IndexType.get(), 64)
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c1 = arith.constant(IndexType.get(), 1)
        
        base_shape = rocir.make_shape(c32, c64)
        base_stride = rocir.make_stride(c1, c32)
        base = rocir.make_layout(base_shape, base_stride)
        
        tile_shape = rocir.make_shape(c8, c16)
        tile_stride = rocir.make_stride(c1, c8)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        
        tiled = rocir.tiled_product(base, tiler)
        
        size = rocir.size(tiled)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_flat_product(ctx, insert_point):
    """Test flat product."""
    
    @func.FuncOp.from_py_func()
    def flat_layout():
        c16 = arith.constant(IndexType.get(), 16)
        c8 = arith.constant(IndexType.get(), 8)
        c4 = arith.constant(IndexType.get(), 4)
        c2 = arith.constant(IndexType.get(), 2)
        c1 = arith.constant(IndexType.get(), 1)
        
        base_shape = rocir.make_shape(c16, c8)
        base_stride = rocir.make_stride(c1, c16)
        base = rocir.make_layout(base_shape, base_stride)
        
        tile_shape = rocir.make_shape(c4, c2)
        tile_stride = rocir.make_stride(c1, c4)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        
        flat = rocir.flat_product(base, tiler)
        
        size = rocir.size(flat)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_raked_product(ctx, insert_point):
    """Test raked product."""
    
    @func.FuncOp.from_py_func()
    def raked_layout():
        c32 = arith.constant(IndexType.get(), 32)
        c8 = arith.constant(IndexType.get(), 8)
        c4 = arith.constant(IndexType.get(), 4)
        c1 = arith.constant(IndexType.get(), 1)
        
        base_shape = rocir.make_shape(c32, c8)
        base_stride = rocir.make_stride(c1, c32)
        base = rocir.make_layout(base_shape, base_stride)
        
        tile_shape = rocir.make_shape(c8, c4)
        tile_stride = rocir.make_stride(c1, c8)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        
        raked = rocir.raked_product(base, tiler)
        
        size = rocir.size(raked)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_blocked_product(ctx, insert_point):
    """Test blocked product."""
    
    @func.FuncOp.from_py_func()
    def blocked_layout():
        c64 = arith.constant(IndexType.get(), 64)
        c16 = arith.constant(IndexType.get(), 16)
        c8 = arith.constant(IndexType.get(), 8)
        c4 = arith.constant(IndexType.get(), 4)
        c1 = arith.constant(IndexType.get(), 1)
        
        base_shape = rocir.make_shape(c64, c16)
        base_stride = rocir.make_stride(c1, c64)
        base = rocir.make_layout(base_shape, base_stride)
        
        tile_shape = rocir.make_shape(c8, c4)
        tile_stride = rocir.make_stride(c1, c8)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        
        blocked = rocir.blocked_product(base, tiler)
        
        size = rocir.size(blocked)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    
