"""Test local operations: local_partition and local_tile with Pythonic operators."""

import pytest
from mlir.ir import InsertionPoint
from mlir.dialects import func

try:
    from rocdsl.dialects.ext import arith, cute
except ImportError:
    pytest.skip("RocDSL dialect not available", allow_module_level=True)


def test_local_partition(ctx):
    """Test local_partition for thread-level partitioning."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_local_partition")
        def test_partition():
            c128 = arith.constant(128, index=True)
            c256 = arith.constant(256, index=True)
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            
            # Global tensor: 128x256
            global_shape = cute.make_shape(c128, c256)
            global_stride = cute.make_stride(c1, c128)
            global_layout = cute.make_layout(global_shape, global_stride)
            
            # Thread layout: 8x16 (thread block)
            thread_shape = cute.make_shape(c8, c16)
            thread_stride = cute.make_stride(c1, c8)
            thread_layout = cute.make_layout(thread_shape, thread_stride)
            
            # Partition for thread 0
            partitioned = cute.local_partition(global_layout, thread_layout, c0)
            
            # Compute elements per thread using operators
            threads_m = c128 // c8  # Pythonic division
            threads_n = c256 // c16
            total_threads = threads_m * threads_n  # Pythonic multiplication
            
            return partitioned
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_tile(ctx):
    """Test local_tile for thread-level tiling."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_local_tile")
        def test_tile():
            c64 = arith.constant(64, index=True)
            c128 = arith.constant(128, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            
            # Base layout: 64x128
            base_shape = cute.make_shape(c64, c128)
            base_stride = cute.make_stride(c1, c64)
            base_layout = cute.make_layout(base_shape, base_stride)
            
            # Tile size: 4x8
            tile_shape = cute.make_shape(c4, c8)
            tile_stride = cute.make_stride(c1, c4)
            tile_layout = cute.make_layout(tile_shape, tile_stride)
            
            # Tile for thread 0
            tiled = cute.local_tile(base_layout, tile_layout, c0)
            
            # Calculate tile dimensions
            tiles_m = c64 // c4
            tiles_n = c128 // c8
            elements_per_tile = c4 * c8
            
            return tiled
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_partition_rank2(ctx):
    """Test 2D local partition with stride calculation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_partition_2d")
        def test_2d_partition():
            c1024 = arith.constant(1024, index=True)
            c32 = arith.constant(32, index=True)
            c16 = arith.constant(16, index=True)
            c5 = arith.constant(5, index=True)
            c1 = arith.constant(1, index=True)
            
            # Large tensor: 1024x1024
            tensor_shape = cute.make_shape(c1024, c1024)
            tensor_stride = cute.make_stride(c1, c1024)
            tensor_layout = cute.make_layout(tensor_shape, tensor_stride)
            
            # Thread grid: 32x32
            grid_shape = cute.make_shape(c32, c32)
            grid_stride = cute.make_stride(c1, c32)
            grid_layout = cute.make_layout(grid_shape, grid_stride)
            
            # Partition for thread (5, 5)
            thread_id = c5 * c32 + c5  # row * cols + col
            partitioned = cute.local_partition(tensor_layout, grid_layout, thread_id)
            
            # Each thread gets: (1024/32) x (1024/32) = 32x32 elements
            elems_per_thread_m = c1024 // c32
            elems_per_thread_n = c1024 // c32
            total_per_thread = elems_per_thread_m * elems_per_thread_n
            
            return partitioned
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_local_tile_rank2(ctx):
    """Test 2D local tile with complex indexing."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_tile_2d")
        def test_2d_tile():
            c512 = arith.constant(512, index=True)
            c16 = arith.constant(16, index=True)
            c8 = arith.constant(8, index=True)
            c3 = arith.constant(3, index=True)
            c1 = arith.constant(1, index=True)
            
            # Medium tensor: 512x512
            tensor_shape = cute.make_shape(c512, c512)
            tensor_stride = cute.make_stride(c1, c512)
            tensor_layout = cute.make_layout(tensor_shape, tensor_stride)
            
            # Tile: 16x8
            tile_shape = cute.make_shape(c16, c8)
            tile_stride = cute.make_stride(c1, c16)
            tile_layout = cute.make_layout(tile_shape, tile_stride)
            
            # Tile for thread 3
            tiled = cute.local_tile(tensor_layout, tile_layout, c3)
            
            # Calculate tiling parameters
            num_tiles_m = c512 // c16
            num_tiles_n = c512 // c8
            tile_area = c16 * c8
            
            return tiled
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_combined_local_ops(ctx):
    """Test combining partition and tile operations."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_combined")
        def test_combined():
            c256 = arith.constant(256, index=True)
            c512 = arith.constant(512, index=True)
            c16 = arith.constant(16, index=True)
            c32 = arith.constant(32, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            
            # Global: 256x512
            global_shape = cute.make_shape(c256, c512)
            global_stride = cute.make_stride(c1, c256)
            global_layout = cute.make_layout(global_shape, global_stride)
            
            # First partition: 16x32 blocks
            block_shape = cute.make_shape(c16, c32)
            block_stride = cute.make_stride(c1, c16)
            block_layout = cute.make_layout(block_shape, block_stride)
            
            partitioned = cute.local_partition(global_layout, block_layout, c0)
            
            # Then tile: 4x8 tiles within partition
            tile_shape = cute.make_shape(c4, c8)
            tile_stride = cute.make_stride(c1, c4)
            tile_layout = cute.make_layout(tile_shape, tile_stride)
            
            tiled = cute.local_tile(partitioned, tile_layout, c0)
            
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
            
            return tiled
    
    ctx.module.operation.verify()
    # Apply lowering

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
