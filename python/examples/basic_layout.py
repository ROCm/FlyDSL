"""Basic example of using RocDSL Python bindings for CuTe layout algebra."""

from mlir.ir import Context, Module, InsertionPoint, Location
from mlir.dialects import func, arith
import rocdsl.dialects.ext.cute as cute


def create_basic_layout_example():
    """Create a simple layout and compute its size."""
    with Context() as ctx, Location.unknown():
        module = Module.create()
        
        with InsertionPoint(module.body):
            # Define a function that creates a 2D layout
            @func.FuncOp.from_py_func(name="test_layout")
            def test_layout():
                # Create dimension constants
                c8 = arith.constant(8, index=True)
                c16 = arith.constant(16, index=True)
                c1 = arith.constant(1, index=True)
                
                # Create shape (8, 16)
                shape = cute.make_shape(c8, c16)
                
                # Create stride (1, 8) - column-major
                stride = cute.make_stride(c1, c8)
                
                # Create layout
                layout = cute.make_layout(shape, stride)
                
                # Compute size
                total_size = cute.size(layout)
                
                return total_size
        
        print(module)


def create_tiled_layout_example():
    """Create a tiled layout using product operations."""
    with Context() as ctx, Location.unknown():
        module = Module.create()
        
        with InsertionPoint(module.body):
            @func.FuncOp.from_py_func(name="test_tiled_layout")
            def test_tiled_layout():
                # Create a 32x64 base layout
                c32 = arith.constant(32, index=True)
                c64 = arith.constant(64, index=True)
                c1 = arith.constant(1, index=True)
                
                base_shape = cute.make_shape(c32, c64)
                base_stride = cute.make_stride(c1, c32)
                base_layout = cute.make_layout(base_shape, base_stride)
                
                # Create a 4x4 tiler
                c4 = arith.constant(4, index=True)
                tile_shape = cute.make_shape(c4, c4)
                tile_stride = cute.make_stride(c1, c4)
                tile_layout = cute.make_layout(tile_shape, tile_stride)
                
                # Compute product
                tiled = cute.logical_product(base_layout, tile_layout)
                
                # Get size
                result_size = cute.size(tiled)
                
                return result_size
        
        print(module)


def create_partition_example():
    """Create a partitioned layout for multi-threading."""
    with Context() as ctx, Location.unknown():
        module = Module.create()
        
        with InsertionPoint(module.body):
            @func.FuncOp.from_py_func(name="test_partition")
            def test_partition():
                # Global 128x128 tensor
                c128 = arith.constant(128, index=True)
                c1 = arith.constant(1, index=True)
                c0 = arith.constant(0, index=True)
                
                global_shape = cute.make_shape(c128, c128)
                global_stride = cute.make_stride(c1, c128)
                global_layout = cute.make_layout(global_shape, global_stride)
                
                # Thread tile 8x8
                c8 = arith.constant(8, index=True)
                tile_shape = cute.make_shape(c8, c8)
                tile_stride = cute.make_stride(c1, c8)
                tile_layout = cute.make_layout(tile_shape, tile_stride)
                
                # Partition for thread 0
                thread_data = cute.local_partition(global_layout, tile_layout, c0)
                
                # Get size of thread data
                thread_size = cute.size(thread_data)
                
                return thread_size
        
        print(module)


if __name__ == "__main__":
    print("=== Basic Layout Example ===")
    create_basic_layout_example()
    
    print("\n=== Tiled Layout Example ===")
    create_tiled_layout_example()
    
    print("\n=== Partition Example ===")
    create_partition_example()
