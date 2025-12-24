"""Basic example of using FLIR Python bindings for layout algebra."""

from pyflir.dialects.ext import arith, flir


def create_basic_layout_example():
    """Create a simple layout and compute its size."""
    class _M(flir.MlirModule):
        @flir.jit
        def test_layout(self: flir.T.i64):
            c8 = arith.index(8)
            c16 = arith.index(16)
            c1 = arith.index(1)
            shape = flir.make_shape(c8.value, c16.value)
            stride = flir.make_stride(c1.value, c8.value)
            layout = flir.make_layout(shape, stride)
            return flir.size(layout).value

    m = _M()
    print(m.module)
    return m


def create_tiled_layout_example():
    """Create a tiled layout using product operations."""
    class _M(flir.MlirModule):
        @flir.jit
        def test_tiled_layout(self: flir.T.i64):
            c32 = arith.index(32)
            c64 = arith.index(64)
            c1 = arith.index(1)
            base_shape = flir.make_shape(c32.value, c64.value)
            base_stride = flir.make_stride(c1.value, c32.value)
            base_layout = flir.make_layout(base_shape, base_stride)

            c4 = arith.index(4)
            tile_shape = flir.make_shape(c4.value, c4.value)
            tile_stride = flir.make_stride(c1.value, c4.value)
            tile_layout = flir.make_layout(tile_shape, tile_stride)
            tiled = flir.logical_product(base_layout, tile_layout)
            return flir.size(tiled).value

    m = _M()
    print(m.module)
    return m


def create_partition_example():
    """Create a partitioned layout for multi-threading."""
    class _M(flir.MlirModule):
        @flir.jit
        def test_partition(self: flir.T.i64):
            c128 = arith.index(128)
            c1 = arith.index(1)
            c0 = arith.index(0)
            global_shape = flir.make_shape(c128.value, c128.value)
            global_stride = flir.make_stride(c1.value, c128.value)
            global_layout = flir.make_layout(global_shape, global_stride)

            c8 = arith.index(8)
            tile_shape = flir.make_shape(c8.value, c8.value)
            tile_stride = flir.make_stride(c1.value, c8.value)
            tile_layout = flir.make_layout(tile_shape, tile_stride)
            thread_data = flir.local_partition(global_layout, tile_layout, c0.value)
            return flir.size(thread_data).value

    m = _M()
    print(m.module)
    return m


if __name__ == "__main__":
    print("=== Basic Layout Example ===")
    create_basic_layout_example()
    
    print("\n=== Tiled Layout Example ===")
    create_tiled_layout_example()
    
    print("\n=== Partition Example ===")
    create_partition_example()
