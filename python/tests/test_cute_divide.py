"""Tests for CuTe divide operations (partitioning)."""

import pytest
from mlir.ir import IndexType
from mlir.dialects import func, arith

import rocdsl.dialects.ext.cute as cute


def test_logical_divide(ctx, insert_point):
    """Test logical divide for basic partitioning."""
    
    @func.FuncOp.from_py_func()
    def partition_layout():
        # Global layout: 128x256
        c128 = arith.constant(IndexType.get(), 128)
        c256 = arith.constant(IndexType.get(), 256)
        c16 = arith.constant(IndexType.get(), 16)
        c32 = arith.constant(IndexType.get(), 32)
        c1 = arith.constant(IndexType.get(), 1)
        
        global_shape = cute.make_shape(c128, c256)
        global_stride = cute.make_stride(c1, c128)
        global_layout = cute.make_layout(global_shape, global_stride)
        
        # Tile: 16x32
        tile_shape = cute.make_shape(c16, c32)
        tile_stride = cute.make_stride(c1, c16)
        tile = cute.make_layout(tile_shape, tile_stride)
        
        # Divide creates partitioned layout
        partitioned = cute.logical_divide(global_layout, tile)
        
        size = cute.size(partitioned)
        return size
    
    ctx.module.operation.verify()
    
    ir_text = str(ctx.module)
    assert "cute.logical_divide" in ir_text
    assert "!cute.layout<4>" in ir_text


def test_zipped_divide(ctx, insert_point):
    """Test zipped divide."""
    
    @func.FuncOp.from_py_func()
    def zipped_partition():
        c64 = arith.constant(IndexType.get(), 64)
        c128 = arith.constant(IndexType.get(), 128)
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c1 = arith.constant(IndexType.get(), 1)
        
        global_shape = cute.make_shape(c64, c128)
        global_stride = cute.make_stride(c1, c64)
        global_layout = cute.make_layout(global_shape, global_stride)
        
        tile_shape = cute.make_shape(c8, c16)
        tile_stride = cute.make_stride(c1, c8)
        tile = cute.make_layout(tile_shape, tile_stride)
        
        zipped = cute.zipped_divide(global_layout, tile)
        
        size = cute.size(zipped)
        return size
    
    ctx.module.operation.verify()
    
    ir_text = str(ctx.module)
    assert "cute.zipped_divide" in ir_text


def test_tiled_divide(ctx, insert_point):
    """Test tiled divide."""
    
    @func.FuncOp.from_py_func()
    def tiled_partition():
        c32 = arith.constant(IndexType.get(), 32)
        c64 = arith.constant(IndexType.get(), 64)
        c4 = arith.constant(IndexType.get(), 4)
        c8 = arith.constant(IndexType.get(), 8)
        c1 = arith.constant(IndexType.get(), 1)
        
        global_shape = cute.make_shape(c32, c64)
        global_stride = cute.make_stride(c1, c32)
        global_layout = cute.make_layout(global_shape, global_stride)
        
        tile_shape = cute.make_shape(c4, c8)
        tile_stride = cute.make_stride(c1, c4)
        tile = cute.make_layout(tile_shape, tile_stride)
        
        tiled = cute.tiled_divide(global_layout, tile)
        
        size = cute.size(tiled)
        return size
    
    ctx.module.operation.verify()
    
    ir_text = str(ctx.module)
    assert "cute.tiled_divide" in ir_text


def test_flat_divide(ctx, insert_point):
    """Test flat divide."""
    
    @func.FuncOp.from_py_func()
    def flat_partition():
        c16 = arith.constant(IndexType.get(), 16)
        c32 = arith.constant(IndexType.get(), 32)
        c4 = arith.constant(IndexType.get(), 4)
        c8 = arith.constant(IndexType.get(), 8)
        c1 = arith.constant(IndexType.get(), 1)
        
        global_shape = cute.make_shape(c16, c32)
        global_stride = cute.make_stride(c1, c16)
        global_layout = cute.make_layout(global_shape, global_stride)
        
        tile_shape = cute.make_shape(c4, c8)
        tile_stride = cute.make_stride(c1, c4)
        tile = cute.make_layout(tile_shape, tile_stride)
        
        flat = cute.flat_divide(global_layout, tile)
        
        size = cute.size(flat)
        return size
    
    ctx.module.operation.verify()
    
    ir_text = str(ctx.module)
    assert "cute.flat_divide" in ir_text
