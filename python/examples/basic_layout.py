"""Basic example of using RocDSL Python bindings for layout algebra (Flyx style)."""

from rocdsl.dialects.ext import arith, rocir


def create_basic_layout_example():
    """Create a simple layout and compute its size."""
    class _M(rocir.MlirModule):
        @rocir.jit
        def test_layout(self: rocir.T.i64):
            c8 = arith.index(8)
            c16 = arith.index(16)
            c1 = arith.index(1)
            shape = rocir.make_shape(c8.value, c16.value)
            stride = rocir.make_stride(c1.value, c8.value)
            layout = rocir.make_layout(shape, stride)
            return rocir.size(layout).value

    m = _M()
    print(m.module)
    return m


def create_tiled_layout_example():
    """Create a tiled layout using product operations."""
    class _M(rocir.MlirModule):
        @rocir.jit
        def test_tiled_layout(self: rocir.T.i64):
            c32 = arith.index(32)
            c64 = arith.index(64)
            c1 = arith.index(1)
            base_shape = rocir.make_shape(c32.value, c64.value)
            base_stride = rocir.make_stride(c1.value, c32.value)
            base_layout = rocir.make_layout(base_shape, base_stride)

            c4 = arith.index(4)
            tile_shape = rocir.make_shape(c4.value, c4.value)
            tile_stride = rocir.make_stride(c1.value, c4.value)
            tile_layout = rocir.make_layout(tile_shape, tile_stride)
            tiled = rocir.logical_product(base_layout, tile_layout)
            return rocir.size(tiled).value

    m = _M()
    print(m.module)
    return m


def create_partition_example():
    """Create a partitioned layout for multi-threading."""
    class _M(rocir.MlirModule):
        @rocir.jit
        def test_partition(self: rocir.T.i64):
            c128 = arith.index(128)
            c1 = arith.index(1)
            c0 = arith.index(0)
            global_shape = rocir.make_shape(c128.value, c128.value)
            global_stride = rocir.make_stride(c1.value, c128.value)
            global_layout = rocir.make_layout(global_shape, global_stride)

            c8 = arith.index(8)
            tile_shape = rocir.make_shape(c8.value, c8.value)
            tile_stride = rocir.make_stride(c1.value, c8.value)
            tile_layout = rocir.make_layout(tile_shape, tile_stride)
            thread_data = rocir.local_partition(global_layout, tile_layout, c0.value)
            return rocir.size(thread_data).value

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
