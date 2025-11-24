# CuTe IR - MLIR Compiler Infrastructure for CUDA Template Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![MLIR](https://img.shields.io/badge/MLIR-amd--staging-orange)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

A modern MLIR-based compiler infrastructure for CuTe (CUDA Template Library), providing a high-level IR for layout algebra and tensor operations with hardware-specific optimizations.

## 🎯 Features

- **CuTe Dialect**: Layout algebra IR with custom types and operations
  - Type system: `!cute.int`, `!cute.shape<N>`, `!cute.stride<N>`, `!cute.layout<N>`, `!cute.coord<N>`
  - Operations: `make_shape`, `make_stride`, `make_layout`, `make_coord`, `size`, `crd2idx`
- **Transformation Passes**: Lowering CuTe IR to standard MLIR dialects
- **cute-opt Tool**: MLIR optimization and transformation tool
- **Modern MLIR API**: Built with latest MLIR (amd-staging branch)

## 🚀 Quick Start

### Build

```bash
# Inside Docker container
cd cute_ir_tablegen
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/mnt/raid0/felix/llvm-project/buildmlir/lib/cmake/mlir
make -j8
```

### Test

```bash
# Test type parsing
./build/tools/cute-opt/cute-opt tests/test_basic.mlir

# Test all operations
./build/tools/cute-opt/cute-opt tests/test_ops.mlir

# Test layout operations
./build/tools/cute-opt/cute-opt tests/test_layout.mlir

# Run transformation pass
./build/tools/cute-opt/cute-opt tests/test_pass.mlir --cute-to-standard
```

## 📝 Example Usage

### Basic Type Usage

```mlir
module {
  func.func @test_types(%i1: !cute.int, %i2: !cute.int) -> !cute.layout<2> {
    %shape = cute.make_shape %i1, %i2 : (!cute.int, !cute.int) -> !cute.shape<2>
    %stride = cute.make_stride %i1, %i2 : (!cute.int, !cute.int) -> !cute.stride<2>
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    return %layout : !cute.layout<2>
  }
  
  func.func @test_all_types(%s: !cute.shape<3>, %st: !cute.stride<3>, 
                            %l: !cute.layout<2>, %c: !cute.coord<2>) {
    return
  }
}
```

### Layout Operations

```mlir
module {
  func.func @test_make_layout(%i8: !cute.int, %i16: !cute.int, %i1: !cute.int) -> !cute.layout<2> {
    // Create shape and stride
    %shape = cute.make_shape %i8, %i16 : (!cute.int, !cute.int) -> !cute.shape<2>
    %stride = cute.make_stride %i1, %i8 : (!cute.int, !cute.int) -> !cute.stride<2>
    
    // Create layout from shape and stride
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    return %layout : !cute.layout<2>
  }
  
  func.func @test_coord_to_index(%layout: !cute.layout<2>, %i3: !cute.int, %i5: !cute.int) -> !cute.int {
    %coord = cute.make_coord %i3, %i5 : (!cute.int, !cute.int) -> !cute.coord<2>
    
    // Convert coordinate to linear index
    %idx = cute.crd2idx %coord, %layout : (!cute.coord<2>, !cute.layout<2>) -> !cute.int
    
    return %idx : !cute.int
  }
}
```

## 🗂️ Project Structure

```
cute_ir_tablegen/
├── include/cute/          # Dialect definitions (TableGen)
│   ├── CuteDialect.h      # Dialect and type declarations (5 types)
│   ├── CuteDialect.td     # Dialect definition with custom type parsing
│   ├── CuteOps.td         # 6 operations with modern MLIR API
│   ├── CutePasses.td      # Pass definitions
│   └── CutePasses.h       # Pass interface
├── lib/
│   ├── Dialect/Cute/
│   │   └── CuteDialect.cpp    # Dialect implementation (90 lines)
│   └── Transforms/
│       ├── CuteToStandard.cpp # Lowering pass (partial implementation)
│       ├── CuteToRocm.cpp
│       └── CuteNvgpuToNvgpu.cpp
├── tools/cute-opt/        # Optimization tool
│   └── cute-opt.cpp       # Tool entry point
├── tests/                 # MLIR test files
│   ├── test_basic.mlir    # Type parsing test
│   ├── test_ops.mlir      # All operations test
│   ├── test_layout.mlir   # Layout operations test
│   └── test_pass.mlir     # Pass transformation test
└── CMakeLists.txt
```

## 🧩 Type System

| Type | Syntax | Description |
|------|--------|-------------|
| `IntType` | `!cute.int` | Compile-time integer value |
| `ShapeType` | `!cute.shape<N>` | N-dimensional shape (N elements) |
| `StrideType` | `!cute.stride<N>` | N-dimensional stride (N elements) |
| `LayoutType` | `!cute.layout<N>` | Combined shape+stride layout |
| `CoordType` | `!cute.coord<N>` | N-dimensional coordinate |

## 🔧 Operations

| Operation | Description | Signature |
|-----------|-------------|-----------|
| `make_shape` | Create shape from integers | `(!cute.int, ...) -> !cute.shape<N>` |
| `make_stride` | Create stride from integers | `(!cute.int, ...) -> !cute.stride<N>` |
| `make_layout` | Create layout from shape+stride | `(!cute.shape<N>, !cute.stride<N>) -> !cute.layout<N>` |
| `make_coord` | Create coordinate from integers | `(!cute.int, ...) -> !cute.coord<N>` |
| `size` | Get total size of shape | `!cute.shape<N> -> !cute.int` |
| `crd2idx` | Convert coord to linear index | `(!cute.coord<N>, !cute.layout<N>) -> !cute.int` |

## 🎨 Passes

| Pass | Flag | Status | Description |
|------|------|--------|-------------|
| `CuteToStandardPass` | `--cute-to-standard` | ✅ Partial | Lower CuTe IR to standard dialects (only `crd2idx` implemented) |
| `CuteToRocmPass` | `--cute-to-rocm` | ⚠️ Skeleton | Lower to ROCm-specific operations |
| `CuteNvgpuToNvgpuPass` | `--cute-nvgpu-to-nvgpu` | ⚠️ Skeleton | Lower to NVGPU dialect |

## ✅ Testing Status

- ✅ **Type parsing**: All 5 types parse and print correctly
- ✅ **Operations**: All 6 operations parse successfully
- ✅ **Pass registration**: `--cute-to-standard` registered in cute-opt
- ⚠️ **Pass execution**: Only `crd2idx` lowering implemented (type conversion warnings are expected)

## 🛠️ Prerequisites

- **MLIR/LLVM**: amd-staging branch (commit 04f968b02917)
  - Build path: `llvm-project/buildmlir`
- **CMake 3.18+**
- **C++17 compiler**
- **Docker**: felixatt container recommended

## 🔍 Implementation Notes

- **Type System**: Manual implementation using `TypeBase` and `TypeStorage` (not TableGen `TypeDef`)
- **Type Parsing**: Custom `parseType()` and `printType()` in `CuteDialect.cpp`
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
