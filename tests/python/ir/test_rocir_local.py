"""Tests for CuTe local operations (thread/block partitioning)."""

import pytest
from mlir.ir import IndexType
from mlir.dialects import func, arith

import rocdsl.dialects.ext.rocir as rocir


def test_local_partition(ctx, insert_point):
    """Test local_partition for thread-level data partitioning."""
    
    @func.FuncOp.from_py_func()
    def thread_partition():
        # Global tensor: 128x256
        c128 = arith.constant(IndexType.get(), 128)
        c256 = arith.constant(IndexType.get(), 256)
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c0 = arith.constant(IndexType.get(), 0)
        c1 = arith.constant(IndexType.get(), 1)
        
        global_shape = rocir.make_shape(c128, c256)
        global_stride = rocir.make_stride(c1, c128)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        # Thread tile: 8x16
        tile_shape = rocir.make_shape(c8, c16)
        tile_stride = rocir.make_stride(c1, c8)
        tile = rocir.make_layout(tile_shape, tile_stride)
        
        # Partition for thread 0
        thread_data = rocir.local_partition(global_layout, tile, c0)
        
        size = rocir.size(thread_data)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_tile(ctx, insert_point):
    """Test local_tile for block-level tile extraction."""
    
    @func.FuncOp.from_py_func()
    def block_tile():
        # Global tensor: 128x256
        c128 = arith.constant(IndexType.get(), 128)
        c256 = arith.constant(IndexType.get(), 256)
        c32 = arith.constant(IndexType.get(), 32)
        c64 = arith.constant(IndexType.get(), 64)
        c0 = arith.constant(IndexType.get(), 0)
        c1 = arith.constant(IndexType.get(), 1)
        
        global_shape = rocir.make_shape(c128, c256)
        global_stride = rocir.make_stride(c1, c128)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        # CTA tile shape: 32x64
        cta_shape = rocir.make_shape(c32, c64)
        
        # CTA coordinates: (0, 0)
        cta_coord = rocir.make_shape(c0, c0)
        
        # Extract CTA tile
        cta_tile = rocir.local_tile(global_layout, cta_shape, cta_coord)
        
        size = rocir.size(cta_tile)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_composition(ctx, insert_point):
    """Test composition of two layouts."""
    
    @func.FuncOp.from_py_func()
    def compose_layouts():
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c4 = arith.constant(IndexType.get(), 4)
        c2 = arith.constant(IndexType.get(), 2)
        c1 = arith.constant(IndexType.get(), 1)
        
        shape_a = rocir.make_shape(c8, c16)
        stride_a = rocir.make_stride(c1, c8)
        layout_a = rocir.make_layout(shape_a, stride_a)
        
        shape_b = rocir.make_shape(c4, c2)
        stride_b = rocir.make_stride(c2, c1)
        layout_b = rocir.make_layout(shape_b, stride_b)
        
        # Compose layouts
        composed = rocir.composition(layout_a, layout_b)
        
        size = rocir.size(composed)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_thread_block_hierarchy(ctx, insert_point):
    """Test hierarchical thread/block partitioning."""
    
    @func.FuncOp.from_py_func()
    def hierarchical_partition():
        # Global: 256x512
        c256 = arith.constant(IndexType.get(), 256)
        c512 = arith.constant(IndexType.get(), 512)
        c64 = arith.constant(IndexType.get(), 64)
        c128 = arith.constant(IndexType.get(), 128)
        c8 = arith.constant(IndexType.get(), 8)
        c16 = arith.constant(IndexType.get(), 16)
        c0 = arith.constant(IndexType.get(), 0)
        c1 = arith.constant(IndexType.get(), 1)
        
        # Global layout
        global_shape = rocir.make_shape(c256, c512)
        global_stride = rocir.make_stride(c1, c256)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        # Block tile: 64x128
        block_shape = rocir.make_shape(c64, c128)
        block_coord = rocir.make_shape(c0, c0)
        block_tile = rocir.local_tile(global_layout, block_shape, block_coord)
        
        # Thread tile within block: 8x16
        thread_shape = rocir.make_shape(c8, c16)
        thread_stride = rocir.make_stride(c1, c8)
        thread_layout = rocir.make_layout(thread_shape, thread_stride)
        
        thread_data = rocir.local_partition(block_tile, thread_layout, c0)
        
        size = rocir.size(thread_data)
        return size
    
    ctx.module.operation.verify()
    # Apply lowering

    
