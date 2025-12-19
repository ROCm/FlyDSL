"""Tests for Rocir divide operations (partitioning)."""

import pytest
from _mlir.ir import IndexType
from _mlir.dialects import arith

import rocdsl.dialects.ext.rocir as rocir
from rocdsl.dialects.ext.arith import Index

def _unwrap(val):
    """Unwrap ArithValue to get underlying MLIR Value."""
    if hasattr(val, "value"):
        return val.value
    return val
from test_utils import unwrap_values


def test_logical_divide(ctx, insert_point):
    """Test logical divide for basic partitioning."""
    
    @rocir.jit
    def partition_layout():
        # Global layout: 128x256
        c128 = Index(128)
        c256 = Index(256)
        c16 = Index(16)
        c32 = Index(32)
        c1 = Index(1)
        
        global_shape = rocir.make_shape(c128, c256)
        global_stride = rocir.make_stride(c1, c128)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        # Tile: 16x32
        tile_shape = rocir.make_shape(c16, c32)
        tile_stride = rocir.make_stride(c1, c16)
        tile = rocir.make_layout(tile_shape, tile_stride)
        
        # Divide creates partitioned layout
        partitioned = rocir.logical_divide(global_layout, tile)
        
        size = rocir.size(partitioned)
        return unwrap_values(size)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_zipped_divide(ctx, insert_point):
    """Test zipped divide."""
    
    @rocir.jit
    def zipped_partition():
        c64 = Index(64)
        c128 = Index(128)
        c8 = Index(8)
        c16 = Index(16)
        c1 = Index(1)
        
        global_shape = rocir.make_shape(c64, c128)
        global_stride = rocir.make_stride(c1, c64)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        tile_shape = rocir.make_shape(c8, c16)
        tile_stride = rocir.make_stride(c1, c8)
        tile = rocir.make_layout(tile_shape, tile_stride)
        
        zipped = rocir.zipped_divide(global_layout, tile)
        
        size = rocir.size(zipped)
        return unwrap_values(size)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_tiled_divide(ctx, insert_point):
    """Test tiled divide."""
    
    @rocir.jit
    def tiled_partition():
        c32 = Index(32)
        c64 = Index(64)
        c4 = Index(4)
        c8 = Index(8)
        c1 = Index(1)
        
        global_shape = rocir.make_shape(c32, c64)
        global_stride = rocir.make_stride(c1, c32)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        tile_shape = rocir.make_shape(c4, c8)
        tile_stride = rocir.make_stride(c1, c4)
        tile = rocir.make_layout(tile_shape, tile_stride)
        
        tiled = rocir.tiled_divide(global_layout, tile)
        
        size = rocir.size(tiled)
        return unwrap_values(size)
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_flat_divide(ctx, insert_point):
    """Test flat divide."""
    
    @rocir.jit
    def flat_partition():
        c16 = Index(16)
        c32 = Index(32)
        c4 = Index(4)
        c8 = Index(8)
        c1 = Index(1)
        
        global_shape = rocir.make_shape(c16, c32)
        global_stride = rocir.make_stride(c1, c16)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        tile_shape = rocir.make_shape(c4, c8)
        tile_stride = rocir.make_stride(c1, c4)
        tile = rocir.make_layout(tile_shape, tile_stride)
        
        flat = rocir.flat_divide(global_layout, tile)
        
        size = rocir.size(flat)
        return unwrap_values(size)
    
    ctx.module.operation.verify()
    # Apply lowering

    
