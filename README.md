# FlyDSL (<span style="color:#2f81f7"><strong>F</strong></span>lexible <span style="color:#2f81f7"><strong>l</strong></span>ayout p<span style="color:#2f81f7"><strong>y</strong></span>thon DSL)
> A Python DSL and a MLIR stack for authoring high‑performance GPU kernels with explicit layouts and tiling. 

FlyDSL is the **Python front‑end** of the project: a *Flexible Layout Python DSL* for expressing
tiling, partitioning, data movement, and kernel structure at a high level.

**FlyDSL**: FlyDSL is powered by FLIR (**F**lexible **L**ayout **I**ntermediate **R**epresentation):
an end‑to‑end, MLIR‑native compiler stack for GPU kernels. Its core is the `flir` dialect—a first‑class
layout IR with explicit algebra and coordinate mapping, plus a composable lowering pipeline to GPU/ROCDL.

## Overview

- **FlyDSL (Python DSL)**: author kernels in Python and compile them through FLIR
  - Primary package: `flydsl/` (`flydsl/src/flydsl/`)
  - Kernel examples: `kernels/` (importable as `kernels.*`)
- **FLIR (`flir` dialect)**: the layout IR and compiler foundation
  - Core abstractions: `!flir.shape`, `!flir.stride`, `!flir.layout`, `!flir.coord`
  - Algebra ops: composition/product/divide/partition + coordinate mapping ops
  - Tooling: `flir-opt` for pass testing and IR experimentation
- **Embedded MLIR Python runtime** (`_mlir`)
  - No external `mlir` python wheel is required: MLIR python bindings are built and staged into `.flir/build/python_packages/flydsl/_mlir` (default; legacy `build/` also works)
  - Python package root: `.flir/build/python_packages/flydsl/`

### Repository layout

```
FlyDSL/
├── scripts/                   # helper scripts (build llvm, tests, packaging)
├── flir/                      # C++ sources + build scripts (CMake, embedded python bindings)
│   ├── CMakeLists.txt
│   ├── build.sh               # build FLIR + python bindings (recommended)
│   ├── include/flir/          # dialect headers + TableGen definitions
│   ├── lib/                   # dialect implementation (Dialect/, Transforms/)
│   ├── python_bindings/       # MLIR python bindings + runtime wrappers
│   └── tools/flir-opt/        # flir-opt CLI tool
├── flydsl/                    # Python sources (src/flydsl) + python-only docs/reqs
├── tests/                     # mlir + python tests/benchmarks
│   ├── mlir/                  # MLIR file tests
│   ├── pyir/                  # Python IR tests (no GPU required)
│   └── kernels/               # GPU execution tests
└── kernels/                   # Python kernels (importable as `kernels.*`)
```

## Getting started

- **ROCm**: required for GPU execution tests/benchmarks (IR-only tests do not need a GPU).
- **Build tools**: `cmake`, C++ compiler, and optionally `ninja` (faster).
- **Python**: Python 3 + `pip`.
  - `scripts/build_llvm.sh` installs `nanobind`, `numpy`, `pybind11`.
  - `flydsl/requirements.txt` exists for auxiliary deps (`numpy`, ) for runtime data initialize and result check.

### Build

### A) Build / use an existing llvm-project (MLIR)

If you already have an MLIR build, set:

```bash
export MLIR_PATH=/path/to/llvm-project/build
```

Or use the helper script (clones ROCm llvm-project and builds MLIR):

```bash
bash scripts/build_llvm.sh
```

### B) Build FLIR (C++ + embedded python package)

```bash
./flir/build.sh
```

After a successful build, you will have:

- `.flir/build/bin/flir-opt` (default; legacy `build/bin/flir-opt` also works)
- Python package root at:
  - `.flir/build/python_packages/flydsl/`
  - This contains:
    - `flydsl/` (your Python API)
    - `_mlir/` (embedded MLIR python bindings)

### Python install


```bash
python3 -m pip install -e .
#for development, you can also use:  
python setup.py develop
```

Build a wheel (default output under `dist/`):

```bash
python3 setup.py bdist_wheel
ls dist/
```


### Run tests

```bash
bash scripts/run_tests.sh
```

What `run_tests.sh` does (high level):

- **MLIR file tests**: runs `tests/mlir/*.mlir` through `flir-opt --flir-to-standard`
- **Python IR tests**: runs `tests/pyir/test_*.py` (no GPU required)
- **Kernel/GPU execution tests** (only if ROCm is detected): runs `tests/kernels/test_*.py`

For the test folder organization, see `tests/` (`mlir/`, `pyir/`, `kernels/`).

### Troubleshooting

- **`flir-opt not found`**
  - Run `./flir/build.sh`, or build it explicitly:
    - `cmake --build build --target flir-opt -j$(nproc)`

- **Python import issues (`No module named flydsl` / `No module named mlir`)**
  - Ensure you are using the embedded package:
    - `export PYTHONPATH=$(pwd)/build/python_packages/flydsl:$PYTHONPATH`
  - Or prefer in-tree sources:
    - `export PYTHONPATH=$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$PYTHONPATH`

- **MLIR `.so` load errors**
  - Add MLIR build lib dir to the loader path:
    - `export LD_LIBRARY_PATH=$MLIR_PATH/lib:$LD_LIBRARY_PATH`

## 📐 FLIR Layout System

> FLIR = **F**lexible **L**ayout **I**ntermediate **R**epresentation.

FLIR introduces a layout system to express complex data mapping patterns on GPUs (tiling, swizzling, vectorization).

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
func.func @layout_example(%i: index, %j: index) -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index

  // Create 2D layout (8, 16) with column-major stride (1, 8)
  %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(8,16)>
  %stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(1,8)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(8,16)>, !flir.stride<(1,8)>) -> !flir.layout<(8,16):(1,8)>

  // Convert coordinate (i, j) to linear index
  %coord = flir.make_coord %i, %j : (index, index) -> !flir.coord<(?,?)>
  %idx = flir.crd2idx %coord, %layout : (!flir.coord<(?,?)>, !flir.layout<(8,16):(1,8)>) -> index

  return %idx : index
}
```

## 🐍 Python API (`flydsl`)

> Python package: `flydsl` (C++/dialect namespace: `flir`).

FLIR provides a high-level Python API for generating kernels.

### Layout Construction

```python
from flydsl.dialects.ext import flir

class _LayoutExample(flir.MlirModule):
    @flir.jit
    def layout_ops(self: flir.T.i64):
        # Create Layout (8x16, column-major)
        shape = flir.make_shape(8, 16)
        stride = flir.make_stride(1, 8)
        layout = flir.make_layout(shape, stride)

        # Query layout properties
        total_size = flir.size(shape)
        layout_rank = flir.rank(layout)
        return total_size
```

### Pipeline API

Easy-to-use compilation pipeline:

```python
from flydsl.compiler.pipeline import Pipeline

# Build and run optimization pipeline
pipeline = (
    Pipeline()
    .flir_to_standard()
    .canonicalize()
    .cse()
    .rocdl_attach_target(chip="gfx942")
    # convert-gpu-to-rocdl must run under gpu.module
    .Gpu(Pipeline().convert_gpu_to_rocdl(runtime="HIP"))
    .gpu_to_llvm()
    .lower_to_llvm()
    .gpu_module_to_binary(format="bin")
)

binary_module = pipeline.run(module)
```

## ⚙️ Hierarchical Kernel Control

FLIR keeps the tiling hierarchy explicit across block, warp, thread, and instruction scopes:

```python
# Define thread and value layouts
thr_layout = flir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
val_layout = flir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))

# Create tiled copy with vectorized atoms
copy_atom = flir.make_copy_atom(T.f32(), vector_size=8)
tiled = flir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                                thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))

# Partition tensor across blocks and threads
tensor_A = flir.make_tensor(A, shape=(M, N), strides=(N, 1))
tiles = flir.zipped_divide(tensor_A, (THR_M * VAL_M, THR_N * VAL_N))
blk_tile = tiles[(flir.block_idx("y"), flir.block_idx("x"))]
thr_tile = tiled.get_slice(tid_linear).partition_S(blk_tile)
```

With per-level partitions, you can allocate register fragments, emit predicate masks, and schedule MFMA/vector instructions while retaining full knowledge of the execution hierarchy.

## 🧮 Minimal VecAdd Example

This condensed snippet mirrors `tests/kernels/test_vec_add.py`, showing how to define GPU kernels with tiled copies and fragments:

```python
import flydsl
from flydsl.dialects.ext import flir
import _mlir.extras.types as T

THREADS, TILE, VEC = 256, 8, 4

class VecAddKernel(flir.MlirModule):
    GPU_MODULE_NAME = "vec_kernels"
    GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942">']

    @flir.kernel
    def vec_add(self: flir.T.i64,
                A: lambda: T.memref(T.dynamic(), T.f32()),
                B: lambda: T.memref(T.dynamic(), T.f32()),
                C: lambda: T.memref(T.dynamic(), T.f32()),
                n: lambda: T.index()):
        tid = flir.thread_idx("x")
        bid = flir.block_idx("x")

        # Define thread/value layouts for tiled copy
        thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
        val_layout = flir.make_ordered_layout((TILE,), order=(0,))
        copy_atom = flir.make_copy_atom(T.f32(), vector_size=VEC)
        tiled = flir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                                        thr_shape=(THREADS,), val_shape=(TILE,))

        # Partition tensors across blocks and threads
        tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
        tiles_A = flir.zipped_divide(tensor_A, (THREADS * TILE,))
        blkA = tiles_A[(bid,)]
        thrA = tiled.get_slice(tid).partition_S(blkA)

        # Load to registers, compute, store
        frgA = flir.make_fragment_like(thrA, T.f32())
        flir.copy(tiled, thrA, frgA)
        # ... repeat for B/C, add, store results

# Compile and run
module = VecAddKernel().module
exe = flydsl.compile(module)
exe(a_dev, b_dev, c_dev, size)
```

See `tests/kernels/test_vec_add.py` for the complete implementation with benchmarking.

## ✅ Testing Status

| Category | Status | Description |
|----------|--------|-------------|
| **MLIR Core** | ✅ Passing | Type parsing, Op verification, Basic transforms |
| **Flir Ops** | ✅ Passing | Layout algebra, Coordinate lowering |
| **GPU Backend**| ✅ Passing | GPU kernel compilation, Shared memory, Vectorization |
| **Hardware** | ✅ Passing | MFMA (Matrix Fused Multiply-Add) execution on MI300-family GPUs |

**Verified Platforms**:
*   AMD MI300X/MI308X (gfx942), AMD MI350 (gfx950)
*   Linux / ROCm 6.x, 7.x

## 🙏 Acknowledgements

FLIR's design is inspired by ideas from several projects:

- [Categorical Foundations for CuTe Layouts](https://arxiv.org/abs/2601.05972) – mathematical framework for layout algebra ([companion code](https://github.com/ColfaxResearch/layout-categories))
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) – CuTe layout algebra concepts (BSD-3-Clause parts only; no EULA-licensed code was referenced)
- [ROCm Composable Kernel](https://github.com/ROCm/composable_kernel) – tile-based kernel design patterns for AMD GPUs
- [Triton](https://github.com/triton-lang/triton) – Python DSL for GPU kernel authoring

## 📄 License

Apache License 2.0
