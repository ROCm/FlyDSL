# CuTe IR - MLIR Compiler Infrastructure for CUDA Template Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

A complete MLIR-based compiler infrastructure for CuTe, enabling high-performance GPU kernel generation through Layout algebra and hardware-specific optimizations.

## 🎯 Features

- **Complete MLIR Dialects**: `cute_ir` (Layout algebra) + `cute_nvgpu_ir` (GPU hardware)
- **C++ Runtime**: Kernel compilation, execution, and memory management
- **Python API**: High-level interface with NumPy integration

## 📦 Quick Start

```bash
# Install
pip install numpy pybind11
cd cute_ir_tablegen/
pip install .

# Test
python -c "import cute_runtime; print(cute_runtime.get_device_info())"

# Run example
python python/examples/test_gemm.py
```

## 🚀 Usage Example

```python
import numpy as np
import cute_runtime as cute

# Create GEMM executor
M, N, K = 1024, 1024, 1024
gemm = cute.Gemm(M, N, K, arch='sm90', use_tma=True)

# Compile from MLIR
mlir_code = open('kernel.mlir').read()
gemm.compile(mlir_code)

# Execute
A = np.random.randn(M, K).astype(np.float16)
B = np.random.randn(K, N).astype(np.float16)
C = gemm(A, B)  # Returns (M, N) float32 array
```

## 🗂️ Project Structure

```
cute_ir_tablegen/
├── include/cute/          # TableGen dialect definitions
│   ├── CuteDialect.td
│   ├── CuteOps.td
│   ├── CuteNvgpuDialect.td
│   ├── CuteNvgpuOps.td
│   └── CutePasses.td
├── lib/Transforms/        # Pass implementations
│   ├── CuteToStandard.cpp
│   └── CuteNvgpuToNvgpu.cpp
├── runtime/               # C++ runtime library
│   ├── include/cute_runtime.h
│   └── src/cute_runtime.cpp
├── python/                # Python bindings
│   ├── cute_runtime/
│   └── examples/
├── docs/                  # Documentation
├── examples/              # MLIR examples
└── setup.py              # Python package installer
```

## 🧩 Architecture

```
Python/C++ API
    ↓
CuTe IR (Layout Algebra)
    ↓ cute-to-standard
Standard Dialects (arith, scf, memref)
    ↓
CuTe NVGPU IR (Hardware-aware)
    ↓ cute-nvgpu-to-nvgpu
NVGPU Dialect
    ↓ convert-nvgpu-to-nvvm
NVVM (LLVM IR)
    ↓ mlir-translate
PTX Assembly
    ↓ ptxas
CUBIN Binary (Executable)
```

## 🛠️ Prerequisites

- **CMake 3.18+**
- **Python 3.8+**
- **C++17 compiler**
- **MLIR/LLVM** (optional, for full compilation)


## 📄 License

Apache License 2.0

## 🙏 Acknowledgments

Built on:
- [MLIR](https://mlir.llvm.org/) - Multi-Level IR framework
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates
- [pybind11](https://github.com/pybind/pybind11) - Python bindings

---

**Version**: 0.1.0  
