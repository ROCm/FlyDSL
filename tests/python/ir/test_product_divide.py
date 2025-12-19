"""Test product (tiling) and divide (partitioning) operations with Pythonic operators."""

import pytest

from rocdsl.dialects.ext import arith, rocir

class _ProductDivide(rocir.MlirModule):
    @rocir.jit
    def logical_product(self: rocir.T.i64):
        c16 = 16
        c32 = 32
        c4 = 4
        c8 = 8
        c1 = 1
        base_shape = rocir.make_shape(c16, c32)
        base_stride = rocir.make_stride(c1, c16)
        base = rocir.make_layout(base_shape, base_stride)
        tile_shape = rocir.make_shape(c4, c8)
        tile_stride = rocir.make_stride(c1, c4)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        tiled = rocir.logical_product(base, tiler)
        return [tiled]

    @rocir.jit
    def zipped_product(self: rocir.T.i64):
        c8 = 8
        c16 = 16
        c2 = 2
        c4 = 4
        c1 = 1
        base_shape = rocir.make_shape(c8, c16)
        base_stride = rocir.make_stride(c1, c8)
        base = rocir.make_layout(base_shape, base_stride)
        tile_shape = rocir.make_shape(c2, c4)
        tile_stride = rocir.make_stride(c1, c2)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        zipped = rocir.zipped_product(base, tiler)
        return [zipped]

    @rocir.jit
    def flat_product(self: rocir.T.i64):
        c12 = 12
        c24 = 24
        c3 = 3
        c6 = 6
        c1 = 1
        base_shape = rocir.make_shape(c12, c24)
        base_stride = rocir.make_stride(c1, c12)
        base = rocir.make_layout(base_shape, base_stride)
        tile_shape = rocir.make_shape(c3, c6)
        tile_stride = rocir.make_stride(c1, c3)
        tiler = rocir.make_layout(tile_shape, tile_stride)
        flat = rocir.flat_product(base, tiler)
        return [flat]

    @rocir.jit
    def blocked_product(self: rocir.T.i64):
        c64 = 64
        c128 = 128
        c16 = 16
        c1 = 1
        base_shape = rocir.make_shape(c64, c128)
        base_stride = rocir.make_stride(c1, c64)
        base = rocir.make_layout(base_shape, base_stride)
        block_shape = rocir.make_shape(c16, c16)
        block_stride = rocir.make_stride(c1, c16)
        blocker = rocir.make_layout(block_shape, block_stride)
        blocked = rocir.blocked_product(base, blocker)
        return [blocked]

    @rocir.jit
    def raked_product(self: rocir.T.i64):
        c32 = 32
        c8 = 8
        c4 = 4
        c1 = 1
        base_shape = rocir.make_shape(c32, c32)
        base_stride = rocir.make_stride(c1, c32)
        base = rocir.make_layout(base_shape, base_stride)
        rake_shape = rocir.make_shape(c4, c8)
        rake_stride = rocir.make_stride(c1, c4)
        raker = rocir.make_layout(rake_shape, rake_stride)
        raked = rocir.raked_product(base, raker)
        return [raked]

    @rocir.jit
    def logical_divide(self: rocir.T.i64):
        c128 = 128
        c256 = 256
        c32 = 32
        c64 = 64
        c1 = 1
        tensor_shape = rocir.make_shape(c128, c256)
        tensor_stride = rocir.make_stride(c1, c128)
        tensor = rocir.make_layout(tensor_shape, tensor_stride)
        tile_shape = rocir.make_shape(c32, c64)
        tile_stride = rocir.make_stride(c1, c32)
        tile = rocir.make_layout(tile_shape, tile_stride)
        divided = rocir.logical_divide(tensor, tile)
        return [divided]

    @rocir.jit
    def zipped_divide(self: rocir.T.i64):
        c64 = 64
        c16 = 16
        c1 = 1
        tensor_shape = rocir.make_shape(c64, c64)
        tensor_stride = rocir.make_stride(c1, c64)
        tensor = rocir.make_layout(tensor_shape, tensor_stride)
        part_shape = rocir.make_shape(c16, c16)
        part_stride = rocir.make_stride(c1, c16)
        part = rocir.make_layout(part_shape, part_stride)
        zipped = rocir.zipped_divide(tensor, part)
        return [zipped]

    @rocir.jit
    def flat_divide(self: rocir.T.i64):
        c96 = 96
        c12 = 12
        c1 = 1
        tensor_shape = rocir.make_shape(c96, c96)
        tensor_stride = rocir.make_stride(c1, c96)
        tensor = rocir.make_layout(tensor_shape, tensor_stride)
        part_shape = rocir.make_shape(c12, c12)
        part_stride = rocir.make_stride(c1, c12)
        part = rocir.make_layout(part_shape, part_stride)
        flat = rocir.flat_divide(tensor, part)
        return [flat]

    @rocir.jit
    def tiled_divide(self: rocir.T.i64):
        c256 = 256
        c128 = 128
        c32 = 32
        c16 = 16
        c1 = 1
        tensor_shape = rocir.make_shape(c256, c128)
        tensor_stride = rocir.make_stride(c1, c256)
        tensor = rocir.make_layout(tensor_shape, tensor_stride)
        tile_shape = rocir.make_shape(c32, c16)
        tile_stride = rocir.make_stride(c1, c32)
        tile = rocir.make_layout(tile_shape, tile_stride)
        tiled = rocir.tiled_divide(tensor, tile)
        return [tiled]


def test_logical_product():
    """Test logical product (basic tiling) with operator usage."""
    ir = str(_ProductDivide().module)
    assert "rocir.logical_product" in ir


def test_zipped_product():
    """Test zipped product (interleaved tiling)."""
    ir = str(_ProductDivide().module)
    assert "rocir.zipped_product" in ir


def test_flat_product():
    """Test flat product."""
    assert "rocir.flat_product" in str(_ProductDivide().module)

@pytest.mark.skip(reason="outer_product operation not implemented yet")

def test_outer_product(ctx):
    """Test outer product."""
    assert "rocir.outer_product" in str(_ProductDivide().module)


def test_blocked_product():
    """Test blocked product with stride computation."""
    ir = str(_ProductDivide().module)
    assert "rocir.blocked_product" in ir


def test_raked_product():
    """Test raked product."""
    assert "rocir.raked_product" in str(_ProductDivide().module)


def test_logical_divide():
    """Test logical divide with offset calculation."""
    ir = str(_ProductDivide().module)
    assert "rocir.logical_divide" in ir


def test_zipped_divide():
    """Test zipped divide."""
    assert "rocir.zipped_divide" in str(_ProductDivide().module)


def test_flat_divide():
    """Test flat divide."""
    assert "rocir.flat_divide" in str(_ProductDivide().module)


def test_tiled_divide():
    """Test tiled divide with complex stride calculation."""
    ir = str(_ProductDivide().module)
    assert "rocir.tiled_divide" in ir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
