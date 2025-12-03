"""Basic example of using RocDSL Python bindings for layout algebra."""

from mlir.ir import Context, Module, InsertionPoint, Location
from mlir.dialects import func, arith
import rocdsl.dialects.ext.rocir as rocir


def create_basic_layout_example():
    """Create a simple layout and compute its size."""
    with Context() as ctx, Location.unknown():
        module = Module.create()
        
        with InsertionPoint(module.body):
            @func.FuncOp.from_py_func(name="test_layout")
            def test_layout():
                c8 = arith.constant(8, index=True)
                c16 = arith.constant(16, index=True)
                c1 = arith.constant(1, index=True)
                
                shape = rocir.make_shape(c8, c16)
                stride = rocir.make_stride(c1, c8)
                layout = rocir.make_layout(shape, stride)
                total_size = rocir.size(layout)
                
                return total_size
        
        print(module)


def create_tiled_layout_example():
    """Create a tiled layout using product operations."""
    with Context() as ctx, Location.unknown():
        module = Module.create()
        
        with InsertionPoint(module.body):
            @func.FuncOp.from_py_func(name="test_tiled_layout")
            def test_tiled_layout():
                c32 = arith.constant(32, index=True)
                c64 = arith.constant(64, index=True)
                c1 = arith.constant(1, index=True)
                
                base_shape = rocir.make_shape(c32, c64)
                base_stride = rocir.make_stride(c1, c32)
                base_layout = rocir.make_layout(base_shape, base_stride)
                
                c4 = arith.constant(4, index=True)
                tile_shape = rocir.make_shape(c4, c4)
                tile_stride = rocir.make_stride(c1, c4)
                tile_layout = rocir.make_layout(tile_shape, tile_stride)
                
                tiled = rocir.logical_product(base_layout, tile_layout)
                result_size = rocir.size(tiled)
                
                return result_size
        
        print(module)


def create_partition_example():
    """Create a partitioned layout for multi-threading."""
    with Context() as ctx, Location.unknown():
        module = Module.create()
        
        with InsertionPoint(module.body):
            @func.FuncOp.from_py_func(name="test_partition")
            def test_partition():
                c128 = arith.constant(128, index=True)
                c1 = arith.constant(1, index=True)
                c0 = arith.constant(0, index=True)
                
                global_shape = rocir.make_shape(c128, c128)
                global_stride = rocir.make_stride(c1, c128)
                global_layout = rocir.make_layout(global_shape, global_stride)
                
                c8 = arith.constant(8, index=True)
                tile_shape = rocir.make_shape(c8, c8)
                tile_stride = rocir.make_stride(c1, c8)
                tile_layout = rocir.make_layout(tile_shape, tile_stride)
                
                thread_data = rocir.local_partition(global_layout, tile_layout, c0)
                thread_size = rocir.size(thread_data)
                
                return thread_size
        
        print(module)


if __name__ == "__main__":
    print("=== Basic Layout Example ===")
    create_basic_layout_example()
    
    print("\n=== Tiled Layout Example ===")
    create_tiled_layout_example()
    
    print("\n=== Partition Example ===")
    create_partition_example()
