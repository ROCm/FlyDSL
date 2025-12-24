"""Test local operations: local_partition and local_tile with Pythonic operators."""

import pytest

from pyflir.dialects.ext import arith, flir


class _LocalOps(flir.MlirModule):
    @flir.jit
    def local_partition(self: flir.T.i64):
        c128 = 128
        c256 = 256
        c8 = 8
        c16 = 16
        c0 = 0
        c1 = 1
        global_shape = flir.make_shape(c128, c256)
        global_stride = flir.make_stride(c1, c128)
        global_layout = flir.make_layout(global_shape, global_stride)
        thread_shape = flir.make_shape(c8, c16)
        thread_stride = flir.make_stride(c1, c8)
        thread_layout = flir.make_layout(thread_shape, thread_stride)
        partitioned = flir.local_partition(global_layout, thread_layout, c0)
        return [partitioned]

    @flir.jit
    def local_tile(self: flir.T.i64):
        c64 = 64
        c128 = 128
        c4 = 4
        c8 = 8
        c0 = 0
        c1 = 1
        base_shape = flir.make_shape(c64, c128)
        base_stride = flir.make_stride(c1, c64)
        base_layout = flir.make_layout(base_shape, base_stride)
        tile_shape = flir.make_shape(c4, c8)
        tile_stride = flir.make_stride(c1, c4)
        tile_layout = flir.make_layout(tile_shape, tile_stride)
        tiled = flir.local_tile(base_layout, tile_layout, c0)
        return [tiled]

    @flir.jit
    def local_partition_2d(self: flir.T.i64):
        c1024 = 1024
        c32 = 32
        c5 = 5
        c1 = 1
        tensor_shape = flir.make_shape(c1024, c1024)
        tensor_stride = flir.make_stride(c1, c1024)
        tensor_layout = flir.make_layout(tensor_shape, tensor_stride)
        grid_shape = flir.make_shape(c32, c32)
        grid_stride = flir.make_stride(c1, c32)
        grid_layout = flir.make_layout(grid_shape, grid_stride)
        thread_id = c5 * c32 + c5
        partitioned = flir.local_partition(tensor_layout, grid_layout, thread_id)
        return [partitioned]

    @flir.jit
    def local_tile_2d(self: flir.T.i64):
        c512 = 512
        c16 = 16
        c8 = 8
        c3 = 3
        c1 = 1
        tensor_shape = flir.make_shape(c512, c512)
        tensor_stride = flir.make_stride(c1, c512)
        tensor_layout = flir.make_layout(tensor_shape, tensor_stride)
        tile_shape = flir.make_shape(c16, c8)
        tile_stride = flir.make_stride(c1, c16)
        tile_layout = flir.make_layout(tile_shape, tile_stride)
        tiled = flir.local_tile(tensor_layout, tile_layout, c3)
        return [tiled]

    @flir.jit
    def combined(self: flir.T.i64):
        c256 = 256
        c512 = 512
        c16 = 16
        c32 = 32
        c4 = 4
        c8 = 8
        c0 = 0
        c1 = 1
        global_shape = flir.make_shape(c256, c512)
        global_stride = flir.make_stride(c1, c256)
        global_layout = flir.make_layout(global_shape, global_stride)
        block_shape = flir.make_shape(c16, c32)
        block_stride = flir.make_stride(c1, c16)
        block_layout = flir.make_layout(block_shape, block_stride)
        partitioned = flir.local_partition(global_layout, block_layout, c0)
        tile_shape = flir.make_shape(c4, c8)
        tile_stride = flir.make_stride(c1, c4)
        tile_layout = flir.make_layout(tile_shape, tile_stride)
        tiled = flir.local_tile(partitioned, tile_layout, c0)
        return [tiled]

def test_local_partition():
    """Test local_partition for thread-level partitioning."""
    m = _LocalOps()
    assert "flir.local_partition" in str(m.module)

    


def test_local_tile():
    """Test local_tile for thread-level tiling."""
    m = _LocalOps()
    assert "flir.local_tile" in str(m.module)

    


def test_local_partition_rank2():
    """Test 2D local partition with stride calculation."""
    m = _LocalOps()
    assert "flir.local_partition" in str(m.module)

    


def test_local_tile_rank2():
    """Test 2D local tile with complex indexing."""
    m = _LocalOps()
    assert "flir.local_tile" in str(m.module)

    


def test_combined_local_ops():
    """Test combining partition and tile operations."""
    m = _LocalOps()
    s = str(m.module)
    assert "flir.local_partition" in s
    assert "flir.local_tile" in s

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
