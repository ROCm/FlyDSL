"""Test product (tiling) and divide (partitioning) operations with Pythonic operators."""

import pytest
from mlir.ir import InsertionPoint
from mlir.dialects import func

try:
    from rocdsl.dialects.ext import arith, cute
except ImportError:
    pytest.skip("RocDSL dialect not available", allow_module_level=True)


def test_logical_product(ctx):
    """Test logical product (basic tiling) with operator usage."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_logical_product")
        def test_product():
            c16 = arith.constant(16, index=True)
            c32 = arith.constant(32, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c1 = arith.constant(1, index=True)
            
            # Base layout: 16x32
            base_shape = cute.make_shape(c16, c32)
            base_stride = cute.make_stride(c1, c16)
            base = cute.make_layout(base_shape, base_stride)
            
            # Tiler: 4x8
            tile_shape = cute.make_shape(c4, c8)
            tile_stride = cute.make_stride(c1, c4)
            tiler = cute.make_layout(tile_shape, tile_stride)
            
            # Tile the layout
            tiled = cute.logical_product(base, tiler)
            
            # Verify tile count using operators: (16/4) * (32/8) = 4 * 4 = 16
            tiles_m = c16 // c4  # Pythonic division!
            tiles_n = c32 // c8
            total_tiles = tiles_m * tiles_n  # Pythonic multiplication!
            
            return tiled
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.logical_product" in ir
    assert "arith.divsi" in ir  # From //
    assert "arith.muli" in ir   # From *


def test_zipped_product(ctx):
    """Test zipped product (interleaved tiling)."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_zipped_product")
        def test_zipped():
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c2 = arith.constant(2, index=True)
            c4 = arith.constant(4, index=True)
            c1 = arith.constant(1, index=True)
            
            base_shape = cute.make_shape(c8, c16)
            base_stride = cute.make_stride(c1, c8)
            base = cute.make_layout(base_shape, base_stride)
            
            tile_shape = cute.make_shape(c2, c4)
            tile_stride = cute.make_stride(c1, c2)
            tiler = cute.make_layout(tile_shape, tile_stride)
            
            # Zipped product
            zipped = cute.zipped_product(base, tiler)
            
            # Compute expected size: 8 * 16 = 128
            expected_size = c8 * c16
            
            return zipped
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.zipped_product" in ir
    assert "arith.muli" in ir


def test_flat_product(ctx):
    """Test flat product."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_flat_product")
        def test_flat():
            c12 = arith.constant(12, index=True)
            c24 = arith.constant(24, index=True)
            c3 = arith.constant(3, index=True)
            c6 = arith.constant(6, index=True)
            c1 = arith.constant(1, index=True)
            
            base_shape = cute.make_shape(c12, c24)
            base_stride = cute.make_stride(c1, c12)
            base = cute.make_layout(base_shape, base_stride)
            
            tile_shape = cute.make_shape(c3, c6)
            tile_stride = cute.make_stride(c1, c3)
            tiler = cute.make_layout(tile_shape, tile_stride)
            
            flat = cute.flat_product(base, tiler)
            
            return flat
    
    ctx.module.operation.verify()
    assert "cute.flat_product" in str(ctx.module)


def test_outer_product(ctx):
    """Test outer product."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_outer_product")
        def test_outer():
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c2 = arith.constant(2, index=True)
            c1 = arith.constant(1, index=True)
            
            shape_a = cute.make_shape(c4, c8)
            stride_a = cute.make_stride(c1, c4)
            layout_a = cute.make_layout(shape_a, stride_a)
            
            shape_b = cute.make_shape(c2, c2)
            stride_b = cute.make_stride(c1, c2)
            layout_b = cute.make_layout(shape_b, stride_b)
            
            outer = cute.outer_product(layout_a, layout_b)
            
            return outer
    
    ctx.module.operation.verify()
    assert "cute.outer_product" in str(ctx.module)


def test_blocked_product(ctx):
    """Test blocked product with stride computation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_blocked_product")
        def test_blocked():
            c64 = arith.constant(64, index=True)
            c128 = arith.constant(128, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            base_shape = cute.make_shape(c64, c128)
            base_stride = cute.make_stride(c1, c64)
            base = cute.make_layout(base_shape, base_stride)
            
            block_shape = cute.make_shape(c16, c16)
            block_stride = cute.make_stride(c1, c16)
            blocker = cute.make_layout(block_shape, block_stride)
            
            blocked = cute.blocked_product(base, blocker)
            
            # Compute block stride using operators
            block_size = c16 * c16  # Elements per block
            
            return blocked
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.blocked_product" in ir
    assert "arith.muli" in ir


def test_raked_product(ctx):
    """Test raked product."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_raked_product")
        def test_raked():
            c32 = arith.constant(32, index=True)
            c8 = arith.constant(8, index=True)
            c4 = arith.constant(4, index=True)
            c1 = arith.constant(1, index=True)
            
            base_shape = cute.make_shape(c32, c32)
            base_stride = cute.make_stride(c1, c32)
            base = cute.make_layout(base_shape, base_stride)
            
            rake_shape = cute.make_shape(c4, c8)
            rake_stride = cute.make_stride(c1, c4)
            raker = cute.make_layout(rake_shape, rake_stride)
            
            raked = cute.raked_product(base, raker)
            
            return raked
    
    ctx.module.operation.verify()
    assert "cute.raked_product" in str(ctx.module)


def test_logical_divide(ctx):
    """Test logical divide with offset calculation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_logical_divide")
        def test_divide():
            c128 = arith.constant(128, index=True)
            c256 = arith.constant(256, index=True)
            c32 = arith.constant(32, index=True)
            c64 = arith.constant(64, index=True)
            c1 = arith.constant(1, index=True)
            
            tensor_shape = cute.make_shape(c128, c256)
            tensor_stride = cute.make_stride(c1, c128)
            tensor = cute.make_layout(tensor_shape, tensor_stride)
            
            tile_shape = cute.make_shape(c32, c64)
            tile_stride = cute.make_stride(c1, c32)
            tile = cute.make_layout(tile_shape, tile_stride)
            
            divided = cute.logical_divide(tensor, tile)
            
            # Calculate partition count: (128/32) * (256/64)
            parts_m = c128 // c32
            parts_n = c256 // c64
            total_parts = parts_m * parts_n
            
            return divided
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.logical_divide" in ir
    assert "arith.divsi" in ir
    assert "arith.muli" in ir


def test_zipped_divide(ctx):
    """Test zipped divide."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_zipped_divide")
        def test_zipped_div():
            c64 = arith.constant(64, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            tensor_shape = cute.make_shape(c64, c64)
            tensor_stride = cute.make_stride(c1, c64)
            tensor = cute.make_layout(tensor_shape, tensor_stride)
            
            part_shape = cute.make_shape(c16, c16)
            part_stride = cute.make_stride(c1, c16)
            part = cute.make_layout(part_shape, part_stride)
            
            zipped = cute.zipped_divide(tensor, part)
            
            return zipped
    
    ctx.module.operation.verify()
    assert "cute.zipped_divide" in str(ctx.module)


def test_flat_divide(ctx):
    """Test flat divide."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_flat_divide")
        def test_flat_div():
            c96 = arith.constant(96, index=True)
            c12 = arith.constant(12, index=True)
            c1 = arith.constant(1, index=True)
            
            tensor_shape = cute.make_shape(c96, c96)
            tensor_stride = cute.make_stride(c1, c96)
            tensor = cute.make_layout(tensor_shape, tensor_stride)
            
            part_shape = cute.make_shape(c12, c12)
            part_stride = cute.make_stride(c1, c12)
            part = cute.make_layout(part_shape, part_stride)
            
            flat = cute.flat_divide(tensor, part)
            
            return flat
    
    ctx.module.operation.verify()
    assert "cute.flat_divide" in str(ctx.module)


def test_tiled_divide(ctx):
    """Test tiled divide with complex stride calculation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_tiled_divide")
        def test_tiled_div():
            c256 = arith.constant(256, index=True)
            c128 = arith.constant(128, index=True)
            c32 = arith.constant(32, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            tensor_shape = cute.make_shape(c256, c128)
            tensor_stride = cute.make_stride(c1, c256)
            tensor = cute.make_layout(tensor_shape, tensor_stride)
            
            tile_shape = cute.make_shape(c32, c16)
            tile_stride = cute.make_stride(c1, c32)
            tile = cute.make_layout(tile_shape, tile_stride)
            
            tiled = cute.tiled_divide(tensor, tile)
            
            # Pythonic stride computation
            col_stride = c256 // c32  # Tiles per row
            row_stride = c128 // c16  # Tiles per column
            
            return tiled
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.tiled_divide" in ir
    assert "arith.divsi" in ir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
