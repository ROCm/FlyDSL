# FlyDSL (<span style="color:#2f81f7"><strong>F</strong></span>lexible <span style="color:#2f81f7"><strong>l</strong></span>ayout p<span style="color:#2f81f7"><strong>y</strong></span>thon DSL)
> A Python DSL and a MLIR stack for authoring high‑performance GPU kernels with explicit layouts and tiling. 

FlyDSL is the **Python front‑end** of the project: a *Flexible Layout Python DSL* for expressing
tiling, partitioning, data movement, and kernel structure at a high level.

**FlyDSL**: FlyDSL is powered by FLIR (**F**lexible **L**ayout **I**ntermediate **R**epresentation):
an end‑to‑end, MLIR‑native compiler stack for GPU kernels. Its core is the `fly` dialect—a first‑class
layout IR with explicit algebra and coordinate mapping, plus a composable lowering pipeline to GPU/ROCDL.

## Overview

- **FlyDSL (Python DSL)**: author kernels in Python and compile them through FLIR
  - Primary package: `python/flydsl/`
  - Kernel examples: `kernels/` (importable as `kernels.*`)
- **Fly dialect**: the layout IR and compiler foundation
  - Core abstractions: `!fly.int_tuple`, `!fly.layout`, `!fly.coord_tensor`, `!fly.memref`
  - Algebra ops: composition/product/divide/partition + coordinate mapping ops
- **Embedded MLIR Python runtime** (`_mlir`)
  - No external `mlir` python wheel is required: MLIR python bindings are built and staged into `build-fly/python_packages/flydsl/_mlir`

### Repository layout

```
FlyDSL/
├── scripts/                   # build & test scripts
│   ├── build_llvm.sh          # build LLVM/MLIR from source
│   ├── build.sh               # build FlyDSL (C++ + Python bindings)
│   ├── run_tests.sh           # run pytest GEMM tests
│   └── run_benchmark.sh       # run performance benchmarks
├── include/flydsl/            # C++ dialect headers
├── lib/                       # C++ dialect implementation
├── python/flydsl/             # Python DSL sources
│   ├── expr/                  # DSL expression API (primitive, arith, etc.)
│   ├── compiler/              # JIT compilation pipeline
│   └── _mlir/                 # (symlink to embedded MLIR bindings)
├── kernels/                   # Python GPU kernels (importable as `kernels.*`)
├── tests/                     # pytest-based tests
├── CMakeLists.txt             # top-level CMake
└── setup.py                   # Python packaging
```

## Getting started

### Prerequisites

- **ROCm**: required for GPU execution (tested on ROCm 6.x, 7.x)
- **Build tools**: `cmake` (≥3.20), C++17 compiler, optionally `ninja`
- **Python**: Python 3.10+ with `pip`
- **Python deps**: `nanobind`, `numpy`, `pybind11` (installed by `build_llvm.sh`)

### Step 1: Build LLVM/MLIR

If you already have an MLIR build with Python bindings enabled, skip this step.

```bash
# Clone ROCm LLVM and build MLIR (takes ~30min with -j64)
bash scripts/build_llvm.sh -j64
```

Or point to an existing build:
```bash
export MLIR_PATH=/path/to/llvm-project/build-flydsl/mlir_install
```

### Step 2: Build FlyDSL

```bash
bash scripts/build.sh -j64
```

`build.sh` auto-detects `MLIR_PATH` from common locations. Override with:
```bash
MLIR_PATH=/path/to/mlir_install bash scripts/build.sh -j64
```

> **Note**: If `MLIR_PATH` is set in your environment pointing to a wrong LLVM build, `unset MLIR_PATH` first.

After a successful build, you will have:
- `build-fly/python_packages/flydsl/` — the complete Python package with embedded MLIR bindings

### Step 3: Install (development mode)

```bash
pip install -e .
# or equivalently:
python setup.py develop
```

This creates an editable install — changes to `python/flydsl/` are immediately reflected.

**Without installing**, you can also set paths manually:
```bash
export PYTHONPATH=$(pwd)/build-fly/python_packages:$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH
```

### Step 4: Run tests

```bash
# Run GEMM correctness tests (fast, ~15s)
bash scripts/run_tests.sh

# Run performance benchmarks
bash scripts/run_benchmark.sh
```

### Quick reference

```bash
# Full build from scratch:
bash scripts/build_llvm.sh -j64   # one-time: build LLVM/MLIR
bash scripts/build.sh -j64        # build FlyDSL
pip install -e .                   # install in dev mode
bash scripts/run_tests.sh          # verify

# Rebuild after code changes (C++ only):
bash scripts/build.sh -j64

# Rebuild after Python-only changes:
# No rebuild needed — editable install picks up changes automatically.
```

### Troubleshooting

- **Wrong LLVM picked up** (`std::gcd not found`, `redeclaration` errors)
  - `unset MLIR_PATH` and let `build.sh` auto-detect, or set it to the correct path.

- **`No module named flydsl`**
  - Run `pip install -e .` or set `PYTHONPATH` as shown above.

- **MLIR `.so` load errors**
  - Set `LD_LIBRARY_PATH` to include the MLIR libs directory:
    ```bash
    export LD_LIBRARY_PATH=$(pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH
    ```

- **Kernel cache issues** (stale results after code changes)
  - Clear: `rm -rf ~/.flydsl/cache`
  - Or disable: `export FLYDSL_RUNTIME_ENABLE_CACHE=0`

## 📐 Layout System

FlyDSL introduces a layout system to express complex data mapping patterns on GPUs (tiling, swizzling, vectorization).

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

## 🐍 Python API (`flydsl`)

### `@flyc.kernel` / `@flyc.jit` API

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

@flyc.kernel
def my_kernel(arg_a: fx.Tensor, arg_b: fx.Tensor, n: fx.Int32):
    tx = gpu.thread_id("x")
    bx = gpu.block_id("x")
    # ... kernel body using layout ops ...

@flyc.jit
def launch(arg_a: fx.Tensor, arg_b: fx.Tensor, n: fx.Int32):
    my_kernel(arg_a, arg_b, n).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
    )
```

## ✅ Testing Status

| Category | Status | Description |
|----------|--------|-------------|
| **Preshuffle GEMM** | ✅ Passing | FP8, INT8, INT4, BF16 (16 tests) |
| **FP4 GEMM** | ⏭ Skipped | Requires gfx950 |
| **GPU Backend**| ✅ Passing | GPU kernel compilation, shared memory, vectorization |
| **CUDA Graph** | ✅ Passing | Graph capture and replay |

**Verified Platforms**:
*   AMD MI300X/MI308X (gfx942), AMD MI350 (gfx950)
*   Linux / ROCm 6.x, 7.x

## 📄 License

Apache License 2.0
