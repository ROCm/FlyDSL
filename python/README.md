# RocDSL Python Bindings

Python bindings for RocDSL - ROCm Domain Specific Language for CuTe Layout Algebra.

## Overview

RocDSL provides Python wrappers for CuTe dialect operations in MLIR, making it easy to:
- Construct layouts from Python
- Perform layout transformations (product, divide, partition, tile)
- Generate MLIR IR programmatically for GPU kernels

## Installation

### From source

```bash
cd python
pip install -e .
```

### Requirements

- Python 3.8+
- MLIR Python bindings (from LLVM/MLIR build)
- RocDSL C++ libraries (from main project build)

## Quick Start

### Basic Layout Creation

```python
from mlir.ir import Context, Module, InsertionPoint
from mlir.dialects import func, arith
import rocdsl.dialects.ext.cute as cute

with Context() as ctx:
    module = Module.create()
    
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(name="create_layout")
        def create_layout():
            # Create constants
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c1 = arith.constant(1, index=True)
            
            # Create shape (8, 16)
            shape = cute.make_shape(c8, c16)
            
            # Create stride (1, 8) - column-major
            stride = cute.make_stride(c1, c8)
            
            # Create layout
            layout = cute.make_layout(shape, stride)
            
            # Compute total size
            total = cute.size(layout)  # Returns 128
            
            return total
    
    print(module)
```

### Layout Products (Tiling)

```python
# Create a 32x64 base layout
base_shape = cute.make_shape(c32, c64)
base_stride = cute.make_stride(c1, c32)
base_layout = cute.make_layout(base_shape, base_stride)

# Create a 4x4 tiler
tile_shape = cute.make_shape(c4, c4)
tile_stride = cute.make_stride(c1, c4)
tile_layout = cute.make_layout(tile_shape, tile_stride)

# Tile the layout
tiled = cute.logical_product(base_layout, tile_layout)
```

### Thread-Level Partitioning

```python
# Global 128x128 tensor
global_shape = cute.make_shape(c128, c128)
global_stride = cute.make_stride(c1, c128)
global_layout = cute.make_layout(global_shape, global_stride)

# Thread tile 8x8
tile_shape = cute.make_shape(c8, c8)
tile_stride = cute.make_stride(c1, c8)
tile_layout = cute.make_layout(tile_shape, tile_stride)

# Partition for thread 0
thread_data = cute.local_partition(global_layout, tile_layout, c0)
```

### Block-Level Tiling

```python
# Extract a CTA tile from global tensor
cta_shape = cute.make_shape(c32, c64)
cta_coord = cute.make_shape(c0, c0)

cta_tile = cute.local_tile(global_layout, cta_shape, cta_coord)
```

## API Reference

### Types

- `ShapeType.get(rank)` - Create a shape type
- `StrideType.get(rank)` - Create a stride type
- `LayoutType.get(rank)` - Create a layout type

### Basic Operations

- `make_shape(*dims)` - Create a shape from dimension values
- `make_stride(*strides)` - Create a stride from stride values
- `make_layout(shape, stride)` - Create a layout
- `size(shape_or_layout)` - Compute total size
- `cosize(layout)` - Compute stride extent
- `rank(shape_or_layout)` - Get number of dimensions
- `get_shape(layout)` - Extract shape from layout
- `get_stride(layout)` - Extract stride from layout
- `composition(layout_a, layout_b)` - Compose two layouts

### Product Operations (Tiling)

- `logical_product(block, tiler)` - Basic layout tiling
- `zipped_product(block, tiler)` - Zipped tiling
- `tiled_product(block, tiler)` - Shaped tiling
- `flat_product(block, tiler)` - Flattened product
- `raked_product(block, tiler)` - Raked/interleaved product
- `blocked_product(block, tiler)` - Blocked product

### Divide Operations (Partitioning)

- `logical_divide(layout, tiler)` - Basic partitioning
- `zipped_divide(layout, tiler)` - Zipped partitioning
- `tiled_divide(layout, tiler)` - Tiled partitioning
- `flat_divide(layout, tiler)` - Flattened divide

### Thread/Block Operations

- `local_partition(layout, tile, index)` - Partition for thread/block index
- `local_tile(layout, tiler, coord)` - Extract tile at coordinates

## Examples

See the `examples/` directory for more examples:
- `basic_layout.py` - Basic layout creation and manipulation
- More examples coming soon!

## Architecture

The Python bindings wrap the CuTe MLIR dialect operations:

```
Python API (cute.make_layout)
    ↓
MLIR Dialect Ops (MakeLayoutOp)
    ↓
C++ Lowering Passes
    ↓
GPU IR (ROCm/CUDA)
```

## Development

### Running Examples

```bash
cd python/examples
python basic_layout.py
```

### Testing

```bash
pytest tests/
```

## Notes

- These bindings require the CuTe dialect to be registered in your MLIR installation
- The operations are designed to lower through MLIR passes to GPU code
- For actual execution, you need the full RocDSL compilation pipeline

## License

Apache License 2.0

## Contributing

Contributions welcome! Please see the main RocDSL repository for guidelines.
