"""Test local operations: local_partition and local_tile with Pythonic operators."""

import pytest
from mlir.ir import InsertionPoint
from mlir.dialects import func

from rocdsl.dialects.ext import arith, rocir


def test_local_partition(ctx):
    """Test local_partition for thread-level partitioning."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_local_partition")
        def test_partition():
            c128 = 128
            c256 = 256
            c8 = 8
            c16 = 16
            c0 = 0
            c1 = 1
            
            # Global tensor: 128x256
            global_shape = rocir.make_shape(c128, c256)
            global_stride = rocir.make_stride(c1, c128)
            global_layout = rocir.make_layout(global_shape, global_stride)
            
            # Thread layout: 8x16 (thread block)
            thread_shape = rocir.make_shape(c8, c16)
            thread_stride = rocir.make_stride(c1, c8)
            thread_layout = rocir.make_layout(thread_shape, thread_stride)
            
            # Partition for thread 0
            partitioned = rocir.local_partition(global_layout, thread_layout, c0)
            
            # Compute elements per thread using operators
            threads_m = c128 // c8  # Pythonic division
            threads_n = c256 // c16
            total_threads = threads_m * threads_n  # Pythonic multiplication
            
            return [partitioned]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_tile(ctx):
    """Test local_tile for thread-level tiling."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_local_tile")
        def test_tile():
            c64 = 64
            c128 = 128
            c4 = 4
            c8 = 8
            c0 = 0
            c1 = 1
            
            # Base layout: 64x128
            base_shape = rocir.make_shape(c64, c128)
            base_stride = rocir.make_stride(c1, c64)
            base_layout = rocir.make_layout(base_shape, base_stride)
            
            # Tile size: 4x8
            tile_shape = rocir.make_shape(c4, c8)
            tile_stride = rocir.make_stride(c1, c4)
            tile_layout = rocir.make_layout(tile_shape, tile_stride)
            
            # Tile for thread 0
            tiled = rocir.local_tile(base_layout, tile_layout, c0)
            
            # Calculate tile dimensions
            tiles_m = c64 // c4
            tiles_n = c128 // c8
            elements_per_tile = c4 * c8
            
            return [tiled]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_partition_rank2(ctx):
    """Test 2D local partition with stride calculation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_partition_2d")
        def test_2d_partition():
            c1024 = 1024
            c32 = 32
            c16 = 16
            c5 = 5
            c1 = 1
            
            # Large tensor: 1024x1024
            tensor_shape = rocir.make_shape(c1024, c1024)
            tensor_stride = rocir.make_stride(c1, c1024)
            tensor_layout = rocir.make_layout(tensor_shape, tensor_stride)
            
            # Thread grid: 32x32
            grid_shape = rocir.make_shape(c32, c32)
            grid_stride = rocir.make_stride(c1, c32)
            grid_layout = rocir.make_layout(grid_shape, grid_stride)
            
            # Partition for thread (5, 5)
            thread_id = c5 * c32 + c5  # row * cols + col
            partitioned = rocir.local_partition(tensor_layout, grid_layout, thread_id)
            
            # Each thread gets: (1024/32) x (1024/32) = 32x32 elements
            elems_per_thread_m = c1024 // c32
            elems_per_thread_n = c1024 // c32
            total_per_thread = elems_per_thread_m * elems_per_thread_n
            
            return [partitioned]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_tile_rank2(ctx):
    """Test 2D local tile with complex indexing."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_tile_2d")
        def test_2d_tile():
            c512 = 512
            c16 = 16
            c8 = 8
            c3 = 3
            c1 = 1
            
            # Medium tensor: 512x512
            tensor_shape = rocir.make_shape(c512, c512)
            tensor_stride = rocir.make_stride(c1, c512)
            tensor_layout = rocir.make_layout(tensor_shape, tensor_stride)
            
            # Tile: 16x8
            tile_shape = rocir.make_shape(c16, c8)
            tile_stride = rocir.make_stride(c1, c16)
            tile_layout = rocir.make_layout(tile_shape, tile_stride)
            
            # Tile for thread 3
            tiled = rocir.local_tile(tensor_layout, tile_layout, c3)
            
            # Calculate tiling parameters
            num_tiles_m = c512 // c16
            num_tiles_n = c512 // c8
            tile_area = c16 * c8
            
            return [tiled]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_combined_local_ops(ctx):
    """Test combining partition and tile operations."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_combined")
        def test_combined():
            c256 = 256
            c512 = 512
            c16 = 16
            c32 = 32
            c4 = 4
            c8 = 8
            c0 = 0
            c1 = 1
            
            # Global: 256x512
            global_shape = rocir.make_shape(c256, c512)
            global_stride = rocir.make_stride(c1, c256)
            global_layout = rocir.make_layout(global_shape, global_stride)
            
            # First partition: 16x32 blocks
            block_shape = rocir.make_shape(c16, c32)
            block_stride = rocir.make_stride(c1, c16)
            block_layout = rocir.make_layout(block_shape, block_stride)
            
            partitioned = rocir.local_partition(global_layout, block_layout, c0)
            
            # Then tile: 4x8 tiles within partition
            tile_shape = rocir.make_shape(c4, c8)
            tile_stride = rocir.make_stride(c1, c4)
            tile_layout = rocir.make_layout(tile_shape, tile_stride)
            
            tiled = rocir.local_tile(partitioned, tile_layout, c0)
            
            # Hierarchical computation
            # Blocks: (256/16) x (512/32)
            blocks_m = c256 // c16
            blocks_n = c512 // c32
            
            # Tiles per block: (16/4) x (32/8)
            tiles_per_block_m = c16 // c4
            tiles_per_block_n = c32 // c8
            tiles_per_block = tiles_per_block_m * tiles_per_block_n
            
            # Elements per tile
            elems_per_tile = c4 * c8
            
            return [tiled]
    
    ctx.module.operation.verify()
    # Apply lowering

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
