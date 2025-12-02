# ROCDSL - MLIR Compiler Infrastructure for high performance rocm kernels

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![MLIR](https://img.shields.io/badge/MLIR-amd--staging-orange)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

A modern MLIR-based compiler infrastructure for high performance rocm kernels, providing a high-level IR for layout algebra and tensor operations with hardware-specific optimizations.

## 🎯 Features

- **Rocir Dialect**: Layout algebra IR with custom types and operations
  - Type system: `!rocir.int`, `!rocir.shape<N>`, `!rocir.stride<N>`, `!rocir.layout<N>`, `!rocir.coord<N>`
  - Operations: `make_shape`, `make_stride`, `make_layout`, `make_coord`, `size`, `crd2idx`
- **Transformation Passes**: Lowering ROCDSL to standard MLIR dialects
- **rocir-opt Tool**: MLIR optimization and transformation tool
- **Modern MLIR API**: Built with latest MLIR (amd-staging branch)

## 🚀 Quick Start

### Build

```bash
# Inside Docker container or environment

# 1. Build the llvm-project (If you don't have this llvm-project)
cd rocdsl
./build_llvm.sh

# 2. Build the rocdsl project
./build.sh

# 3. Install python bindings
cd python
python setup.py develop
```

### Test

```bash
# Test type parsing
./build/tools/rocir-opt/rocir-opt tests/mlir/test_basic.mlir

# Test all operations
./build/tools/rocir-opt/rocir-opt tests/mlir/test_ops.mlir

# Test layout operations
./build/tools/rocir-opt/rocir-opt tests/mlir/test_layout.mlir

# Run transformation pass
./build/tools/rocir-opt/rocir-opt tests/mlir/test_pass.mlir --rocir-to-standard

# run python test
pytest -sv tests/python/test_rocir_basic.py
```

## 📝 Example Usage

### Basic Type Usage

```mlir
module {
  func.func @test_types(%i1: !rocir.int, %i2: !rocir.int) -> !rocir.layout<2> {
    %shape = rocir.make_shape %i1, %i2 : (!rocir.int, !rocir.int) -> !rocir.shape<2>
    %stride = rocir.make_stride %i1, %i2 : (!rocir.int, !rocir.int) -> !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
    return %layout : !rocir.layout<2>
  }
  
  func.func @test_all_types(%s: !rocir.shape<3>, %st: !rocir.stride<3>, 
                            %l: !rocir.layout<2>, %c: !rocir.coord<2>) {
    return
  }
}
```

### Layout Operations

```mlir
module {
  func.func @test_make_layout(%i8: !rocir.int, %i16: !rocir.int, %i1: !rocir.int) -> !rocir.layout<2> {
    // Create shape and stride
    %shape = rocir.make_shape %i8, %i16 : (!rocir.int, !rocir.int) -> !rocir.shape<2>
    %stride = rocir.make_stride %i1, %i8 : (!rocir.int, !rocir.int) -> !rocir.stride<2>
    
    // Create layout from shape and stride
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
    
    return %layout : !rocir.layout<2>
  }
  
  func.func @test_coord_to_index(%layout: !rocir.layout<2>, %i3: !rocir.int, %i5: !rocir.int) -> !rocir.int {
    %coord = rocir.make_coord %i3, %i5 : (!rocir.int, !rocir.int) -> !rocir.coord<2>
    
    // Convert coordinate to linear index
    %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<2>, !rocir.layout<2>) -> !rocir.int
    
    return %idx : !rocir.int
  }
}
```

## 🗂️ Project Structure

```
rocdsl/
├── include/rocir/          # Dialect definitions (TableGen)
│   ├── RocirDialect.h      # Dialect and type declarations (5 types)
│   ├── RocirDialect.td     # Dialect definition with custom type parsing
│   ├── RocirOps.td         # 6 operations with modern MLIR API
│   ├── RocirPasses.td      # Pass definitions
│   └── RocirPasses.h       # Pass interface
├── lib/
│   ├── Dialect/Rocir/
│   │   └── RocirDialect.cpp    # Dialect implementation (90 lines)
│   └── Transforms/
│       ├── RocirToStandard.cpp # Lowering pass (partial implementation)
│       ├── RocirToRocm.cpp
│       └── RocirNvgpuToNvgpu.cpp
├── tools/rocir-opt/        # Optimization tool
│   └── rocir-opt.cpp       # Tool entry point
├── tests/mlir/            # MLIR test files
│   ├── test_basic.mlir    # Type parsing test
│   ├── test_ops.mlir      # All operations test
│   ├── test_layout.mlir   # Layout operations test
│   └── test_pass.mlir     # Pass transformation test
└── CMakeLists.txt
```

## 🧩 Type System

| Type | Syntax | Description |
|------|--------|-------------|
| `IntType` | `!rocir.int` | Compile-time integer value |
| `ShapeType` | `!rocir.shape<N>` | N-dimensional shape (N elements) |
| `StrideType` | `!rocir.stride<N>` | N-dimensional stride (N elements) |
| `LayoutType` | `!rocir.layout<N>` | Combined shape+stride layout |
| `CoordType` | `!rocir.coord<N>` | N-dimensional coordinate |

## 🔧 Operations

| Operation | Description | Signature |
|-----------|-------------|-----------|
| `make_shape` | Create shape from integers | `(!rocir.int, ...) -> !rocir.shape<N>` |
| `make_stride` | Create stride from integers | `(!rocir.int, ...) -> !rocir.stride<N>` |
| `make_layout` | Create layout from shape+stride | `(!rocir.shape<N>, !rocir.stride<N>) -> !rocir.layout<N>` |
| `make_coord` | Create coordinate from integers | `(!rocir.int, ...) -> !rocir.coord<N>` |
| `size` | Get total size of shape | `!rocir.shape<N> -> !rocir.int` |
| `crd2idx` | Convert coord to linear index | `(!rocir.coord<N>, !rocir.layout<N>) -> !rocir.int` |

## 🎨 Passes

| Pass | Flag | Status | Description |
|------|------|--------|-------------|
| `RocirToStandardPass` | `--rocir-to-standard` | ✅ Partial | Lower ROCDSL to standard dialects (only `crd2idx` implemented) |
| `RocirToRocmPass` | `--rocir-to-rocm` | ⚠️ Skeleton | Lower to ROCm-specific operations |
| `RocirNvgpuToNvgpuPass` | `--rocir-nvgpu-to-nvgpu` | ⚠️ Skeleton | Lower to NVGPU dialect |

## ✅ Testing Status

- ✅ **Type parsing**: All 5 types parse and print correctly
- ✅ **Operations**: All 6 operations parse successfully
- ✅ **Pass registration**: `--rocir-to-standard` registered in rocir-opt
- ⚠️ **Pass execution**: Only `crd2idx` lowering implemented (type conversion warnings are expected)

## 🛠️ Prerequisites

- **MLIR/LLVM**: amd-staging branch (commit 04f968b02917)
  - Build path: `llvm-project/buildmlir`
- **CMake 3.18+**
- **C++17 compiler**
- **Docker**: felixatt container recommended

## 🔍 Implementation Notes

- **Type System**: Manual implementation using `TypeBase` and `TypeStorage` (not TableGen `TypeDef`)
- **Type Parsing**: Custom `parseType()` and `printType()` in `RocirDialect.cpp`
- **Pass System**: Modern MLIR using `GEN_PASS_DEF` macros and `impl::PassBase` inheritance
- **Dependencies**: Requires `MLIRSCFDialect` for pass infrastructure
- **API Compatibility**: Uses modern `llvm::isa<>()` instead of legacy `.isa<>()`

## 📄 License

Apache License 2.0

## 🙏 Acknowledgments

Built on:
- [MLIR](https://mlir.llvm.org/) - Multi-Level IR framework
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra

---

**Version**: 0.2.0-alpha | **MLIR**: amd-staging (04f968b02917)

## Running Tests

All test files have been organized in the `tests/` directory:

```bash
# Run all tests with the test suite
./run_tests.sh

# Or run individual tests
./build/tools/rocir-opt/rocir-opt --rocir-to-standard tests/mlir/test_crd2idx.mlir
./build/tools/rocir-opt/rocir-opt --rocir-to-standard tests/mlir/test_size.mlir
./build/tools/rocir-opt/rocir-opt --rocir-to-standard tests/mlir/test_rank.mlir
./build/tools/rocir-opt/rocir-opt --rocir-to-standard tests/mlir/test_cosize.mlir
./build/tools/rocir-opt/rocir-opt --rocir-to-standard tests/mlir/comprehensive_test.mlir
```

### Test Coverage

- ✅ **test_crd2idx.mlir** - Coordinate to linear index conversion
- ✅ **test_size.mlir** - Shape size computation (product of dimensions)
- ✅ **test_rank.mlir** - Rank extraction (compile-time constant)
- ✅ **test_cosize.mlir** - Codomain size (max span computation)
- ✅ **comprehensive_test.mlir** - All operations together

All tests pass with optimal lowering to standard MLIR arithmetic operations.
