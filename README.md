# ROCDSL - MLIR Compiler Infrastructure for high performance rocm kernels

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![MLIR](https://img.shields.io/badge/MLIR-amd--staging-orange)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

A modern MLIR-based compiler infrastructure for high performance ROCm kernels, providing a high-level IR for layout algebra and tensor operations with hardware-specific optimizations.

## 🎯 Features

- **Rocir Dialect**: Layout algebra IR inspired by CuTe/CUTLASS
  - Core abstractions: `!rocir.shape`, `!rocir.stride`, `!rocir.layout`
  - Powerful algebra: Composition, Product, Divide, Partition operations
  - Compile-time + Runtime hybrid evaluation
- **Python Bindings**: Fluent Python API (`rocdsl`) for kernel construction
  - `Pipeline` API for easy pass management
  - `ArithValue` wrapper for intuitive arithmetic expressions
- **Hardware Support**:
  - AMD MI300 (gfx940/gfx942) optimization support (MFMA)
  - Generic ROCm/HIP support
- **Transformation Passes**:
  - `rocir-coord-lowering`: Lowers layout abstractions to efficient arithmetic
  - `gpu-to-rocdl`: Full lowering pipeline to HSACO binary

## 🚀 Quick Start

### Build

```bash
# Inside Docker container or environment

# 1. Build the llvm-project (if needed)
cd rocdsl
./build_llvm.sh

# 2. Install Python requirements
pip install -r python/requirements.txt

# 3. Build the rocdsl project (C++ and Python bindings)
./build.sh

# 4. Install python bindings in editable mode
cd python
python setup.py develop
```

### Run Tests

```bash
# Run the full test suite (C++ and Python tests)
./run_tests.sh

# Run specific Python benchmark
python tests/benchmark/vecAdd.py
```

## 📐 Layout System

ROCDSL introduces a powerful layout system to manage complex data mapping patterns on GPUs (tiling, swizzling, vectorization).

### Core Abstractions

1.  **Shape**: The extent of dimensions (e.g., `(M, N)`).
2.  **Stride**: The distance between elements in memory (e.g., `(1, M)` for column-major).
3.  **Layout**: A pair of `(Shape, Stride)` that maps a logical **Coordinate** to a physical linear **Index**.

Formula: `Index = dot(Coord, Stride) = sum(c_i * s_i)`

### Operations

*   **Construction**: `make_shape`, `make_stride`, `make_layout`, `make_coord`
*   **Mapping**:
    *   `crd2idx(coord, layout) -> index`: Convert logical coordinate to physical index.
    *   `idx2crd(index, layout) -> coord`: Convert physical index to logical coordinate.
*   **Inspection**: `size`, `cosize`, `rank`
*   **Algebra**:
    *   `composition(A, B)`: Compose layouts (A ∘ B).
    *   `product(A, B)`: Combine layouts (Logical, Tiled, Blocked, etc.).
    *   `divide(A, B)`: Partition layout A by B (Logical, Tiled, etc.).
    *   `local_partition(layout, tile, index)`: Slice layout for a specific thread/block.

### Example (MLIR)

```mlir
func.func @layout_example(%i: !rocir.int, %j: !rocir.int) -> !rocir.int {
  // Create 2D layout (8, 16) with column-major stride (1, 8)
  %shape = rocir.make_shape %c8, %c16 : (!rocir.int, !rocir.int) -> !rocir.shape<2>
  %stride = rocir.make_stride %c1, %c8 : (!rocir.int, !rocir.int) -> !rocir.stride<2>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>

  // Convert coordinate (i, j) to linear index
  %coord = rocir.make_coord %i, %j : (!rocir.int, !rocir.int) -> !rocir.coord<2>
  %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<2>, !rocir.layout<2>) -> !rocir.int

  return %idx : !rocir.int
}
```

## 🐍 Python API (`rocdsl`)

ROCDSL provides a high-level Python API for generating kernels.

### Layout Construction

```python
from rocdsl.dialects.ext import rocir, arith

# Create constants
c8 = arith.constant(8, index=True)
c16 = arith.constant(16, index=True)

# Create Layout
shape = rocir.make_shape(c8, c16)
stride = rocir.make_stride(arith.constant(1, index=True), c8)
layout = rocir.make_layout(shape, stride)

# Coordinate to Index
coord = rocir.make_coord(i, j)
idx = rocir.crd2idx(coord, layout)
```

### Pipeline API

Easy-to-use compilation pipeline:

```python
from rocdsl.compiler.pipeline import Pipeline

# Build and run optimization pipeline
pipeline = Pipeline() \
    .rocir_coord_lowering() \
    .canonicalize() \
    .cse() \
    .rocdl_attach_target(chip="gfx942") \
    .Gpu(Pipeline().convert_gpu_to_rocdl(runtime="HIP")) \
    .gpu_to_llvm() \
    .lower_to_llvm() \
    .gpu_module_to_binary(format="bin")

binary_module = pipeline.run(module)
```

## ✅ Testing Status

| Category | Status | Description |
|----------|--------|-------------|
| **MLIR Core** | ✅ Passing | Type parsing, Op verification, Basic transforms |
| **Rocir Ops** | ✅ Passing | Layout algebra, Coordinate lowering |
| **GPU Backend**| ✅ Passing | GPU kernel compilation, Shared memory, Vectorization |
| **Hardware** | ✅ Passing | MFMA (Matrix Fused Multiply-Add) execution on MI300 |

**Verified Platforms**:
*   AMD MI300X (gfx942)
*   Linux / ROCm 6.x

## 🗂️ Project Structure

```
rocdsl/
├── include/rocir/          # C++ Dialect definitions
├── lib/                    # C++ Implementation (Dialect, Transforms)
├── python/                 # Python bindings package (rocdsl)
│   ├── rocdsl/
│   │   ├── dialects/       # MLIR Dialect wrappers
│   │   └── compiler/       # Pipeline and Context utilities
│   ├── requirements.txt    # Python dependencies
│   └── setup.py            # Build script
├── tools/                  # CLI tools (rocir-opt)
└── tests/                  # Test suite
    ├── mlir/               # Lit tests for C++ components
    └── python/             # Pytest suite for Python API
        ├── ir/             # IR generation tests
        ├── gpu/            # GPU execution tests
        └── benchmark/      # Performance benchmarks
```

## 📄 License

Apache License 2.0
