"""Test product (tiling) and divide (partitioning) operations with Pythonic operators."""

import pytest

from pyflir.dialects.ext import arith, flir

class _ProductDivide(flir.MlirModule):
    @flir.jit
    def logical_product(self: flir.T.i64):
        c16 = 16
        c32 = 32
        c4 = 4
        c8 = 8
        c1 = 1
        base_shape = flir.make_shape(c16, c32)
        base_stride = flir.make_stride(c1, c16)
        base = flir.make_layout(base_shape, base_stride)
        tile_shape = flir.make_shape(c4, c8)
        tile_stride = flir.make_stride(c1, c4)
        tiler = flir.make_layout(tile_shape, tile_stride)
        tiled = flir.logical_product(base, tiler)
        return [tiled]

    @flir.jit
    def zipped_product(self: flir.T.i64):
        c8 = 8
        c16 = 16
        c2 = 2
        c4 = 4
        c1 = 1
        base_shape = flir.make_shape(c8, c16)
        base_stride = flir.make_stride(c1, c8)
        base = flir.make_layout(base_shape, base_stride)
        tile_shape = flir.make_shape(c2, c4)
        tile_stride = flir.make_stride(c1, c2)
        tiler = flir.make_layout(tile_shape, tile_stride)
        zipped = flir.zipped_product(base, tiler)
        return [zipped]

    @flir.jit
    def flat_product(self: flir.T.i64):
        c12 = 12
        c24 = 24
        c3 = 3
        c6 = 6
        c1 = 1
        base_shape = flir.make_shape(c12, c24)
        base_stride = flir.make_stride(c1, c12)
        base = flir.make_layout(base_shape, base_stride)
        tile_shape = flir.make_shape(c3, c6)
        tile_stride = flir.make_stride(c1, c3)
        tiler = flir.make_layout(tile_shape, tile_stride)
        flat = flir.flat_product(base, tiler)
        return [flat]

    @flir.jit
    def blocked_product(self: flir.T.i64):
        c64 = 64
        c128 = 128
        c16 = 16
        c1 = 1
        base_shape = flir.make_shape(c64, c128)
        base_stride = flir.make_stride(c1, c64)
        base = flir.make_layout(base_shape, base_stride)
        block_shape = flir.make_shape(c16, c16)
        block_stride = flir.make_stride(c1, c16)
        blocker = flir.make_layout(block_shape, block_stride)
        blocked = flir.blocked_product(base, blocker)
        return [blocked]

    @flir.jit
    def raked_product(self: flir.T.i64):
        c32 = 32
        c8 = 8
        c4 = 4
        c1 = 1
        base_shape = flir.make_shape(c32, c32)
        base_stride = flir.make_stride(c1, c32)
        base = flir.make_layout(base_shape, base_stride)
        rake_shape = flir.make_shape(c4, c8)
        rake_stride = flir.make_stride(c1, c4)
        raker = flir.make_layout(rake_shape, rake_stride)
        raked = flir.raked_product(base, raker)
        return [raked]

    @flir.jit
    def logical_divide(self: flir.T.i64):
        c128 = 128
        c256 = 256
        c32 = 32
        c64 = 64
        c1 = 1
        tensor_shape = flir.make_shape(c128, c256)
        tensor_stride = flir.make_stride(c1, c128)
        tensor = flir.make_layout(tensor_shape, tensor_stride)
        tile_shape = flir.make_shape(c32, c64)
        tile_stride = flir.make_stride(c1, c32)
        tile = flir.make_layout(tile_shape, tile_stride)
        divided = flir.logical_divide(tensor, tile)
        return [divided]

    @flir.jit
    def zipped_divide(self: flir.T.i64):
        c64 = 64
        c16 = 16
        c1 = 1
        tensor_shape = flir.make_shape(c64, c64)
        tensor_stride = flir.make_stride(c1, c64)
        tensor = flir.make_layout(tensor_shape, tensor_stride)
        part_shape = flir.make_shape(c16, c16)
        part_stride = flir.make_stride(c1, c16)
        part = flir.make_layout(part_shape, part_stride)
        zipped = flir.zipped_divide(tensor, part)
        return [zipped]

    @flir.jit
    def flat_divide(self: flir.T.i64):
        c96 = 96
        c12 = 12
        c1 = 1
        tensor_shape = flir.make_shape(c96, c96)
        tensor_stride = flir.make_stride(c1, c96)
        tensor = flir.make_layout(tensor_shape, tensor_stride)
        part_shape = flir.make_shape(c12, c12)
        part_stride = flir.make_stride(c1, c12)
        part = flir.make_layout(part_shape, part_stride)
        flat = flir.flat_divide(tensor, part)
        return [flat]

    @flir.jit
    def tiled_divide(self: flir.T.i64):
        c256 = 256
        c128 = 128
        c32 = 32
        c16 = 16
        c1 = 1
        tensor_shape = flir.make_shape(c256, c128)
        tensor_stride = flir.make_stride(c1, c256)
        tensor = flir.make_layout(tensor_shape, tensor_stride)
        tile_shape = flir.make_shape(c32, c16)
        tile_stride = flir.make_stride(c1, c32)
        tile = flir.make_layout(tile_shape, tile_stride)
        tiled = flir.tiled_divide(tensor, tile)
        return [tiled]


def test_logical_product():
    """Test logical product (basic tiling) with operator usage."""
    ir = str(_ProductDivide().module)
    assert "flir.logical_product" in ir


def test_zipped_product():
    """Test zipped product (interleaved tiling)."""
    ir = str(_ProductDivide().module)
    assert "flir.zipped_product" in ir


def test_flat_product():
    """Test flat product."""
    assert "flir.flat_product" in str(_ProductDivide().module)

@pytest.mark.skip(reason="outer_product operation not implemented yet")

def test_outer_product(ctx):
    """Test outer product."""
    assert "flir.outer_product" in str(_ProductDivide().module)


def test_blocked_product():
    """Test blocked product with stride computation."""
    ir = str(_ProductDivide().module)
    assert "flir.blocked_product" in ir


def test_raked_product():
    """Test raked product."""
    assert "flir.raked_product" in str(_ProductDivide().module)


def test_logical_divide():
    """Test logical divide with offset calculation."""
    ir = str(_ProductDivide().module)
    assert "flir.logical_divide" in ir


def test_zipped_divide():
    """Test zipped divide."""
    assert "flir.zipped_divide" in str(_ProductDivide().module)


def test_flat_divide():
    """Test flat divide."""
    assert "flir.flat_divide" in str(_ProductDivide().module)


def test_tiled_divide():
    """Test tiled divide with complex stride calculation."""
    ir = str(_ProductDivide().module)
    assert "flir.tiled_divide" in ir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
