# ROCDSL - MLIR Compiler Infrastructure for high performance ROCm kernels

ROCDSL is an MLIR-based compiler infrastructure for high performance ROCm kernels.
It provides a custom layout-algebra IR (Rocir dialect), a lowering pipeline to GPU/ROCDL,
and a Python API (`rocdsl`) for constructing and running kernels.

## Features

- **Rocir Dialect** (layout algebra inspired by CuTe/CUTLASS)
  - Core abstractions: `!rocir.shape`, `!rocir.stride`, `!rocir.layout`, `!rocir.coord`
  - Algebra ops: composition/product/divide/partition + coordinate mapping ops
- **Python bindings** (`python/rocdsl/`) with an embedded MLIR python package
  - No external `mlir` python wheel is required: MLIR python bindings are built and staged into `build/python_packages/rocdsl/_mlir`
- **GPU lowering** to HSACO via MLIR GPU → ROCDL pipeline
- **Tools**: `rocir-opt` for pass testing and IR experimentation

## Repository layout (current)

```
rocDSL/
├── CMakeLists.txt
├── build_llvm.sh              # build/prepare llvm-project (optional helper)
├── build.sh                   # build rocDSL + python bindings (recommended)
├── run_tests.sh               # run MLIR + Python tests
├── include/                   # C++ headers (dialect/pass declarations)
├── lib/                       # C++ dialect + transforms + CAPI
├── tools/                     # rocir-opt
├── python/                    # Python package sources (rocdsl + helpers)
├── python_bindings/           # CMake targets for python extensions/bindings
└── tests/                     # mlir + python tests/benchmarks
```

## Prerequisites

- **ROCm**: required for GPU execution tests/benchmarks (IR-only tests do not need a GPU).
- **Build tools**: `cmake`, C++ compiler, and optionally `ninja` (faster).
- **Python**: Python 3 + `pip`.
  - `build_llvm.sh` installs `nanobind`, `numpy`, `pybind11`.
  - `python/requirements.txt` exists for auxiliary deps (`pybind11`, `hip-python`).

## Build

### A) Build / use an existing llvm-project (MLIR)

If you already have an MLIR build, set:

```bash
export MLIR_PATH=/path/to/llvm-project/build
```

Or use the helper script (clones ROCm llvm-project and builds MLIR):

```bash
./build_llvm.sh
```

### B) Build rocDSL (C++ + embedded python package)

```bash
./build.sh
```

After a successful build, you will have:

- `build/bin/rocir-opt`
- Python package root at:
  - `build/python_packages/rocdsl/`
  - This contains:
    - `rocdsl/` (your Python API)
    - `_mlir/` (embedded MLIR python bindings)
    - optional `mlir/` shim (if present)

## Using the Python bindings

Always point `PYTHONPATH` at the build-staged package:

```bash
export PYTHONPATH="$(pwd)/build/python_packages/rocdsl:${PYTHONPATH}"
```

If you see dynamic loader errors for MLIR shared libraries, also set:

```bash
export LD_LIBRARY_PATH="${MLIR_PATH:-$(pwd)/../llvm-project/buildmlir}/lib:${LD_LIBRARY_PATH}"
```

## Run tests

```bash
./run_tests.sh
```

What `run_tests.sh` does (high level):

- **MLIR file tests**: runs `tests/mlir/*.mlir` through `rocir-opt --rocir-coord-lowering`
- **Python IR tests**: runs `tests/python/ir/test_*.py` (no GPU required)
- **Python examples**: runs `tests/python/examples/test_*.py`
- **GPU execution tests** (only if ROCm is detected): runs `tests/python/gpu/test_*.py`
- **Benchmarks** (only if ROCm is detected): runs `tests/benchmark/*.py` via `pytest`

For the Python test folder organization, see `tests/python/README.md`.

## Troubleshooting

- **`rocir-opt not found`**
  - Run `./build.sh`, or build it explicitly:
    - `cmake --build build --target rocir-opt -j$(nproc)`

- **Python import issues (`No module named rocdsl` / `No module named mlir`)**
  - Ensure you are using the embedded package:
    - `export PYTHONPATH=$(pwd)/build/python_packages/rocdsl:$PYTHONPATH`

- **MLIR `.so` load errors**
  - Add MLIR build lib dir to the loader path:
    - `export LD_LIBRARY_PATH=$MLIR_PATH/lib:$LD_LIBRARY_PATH`

## 📐 Layout System

ROCDSL introduces a layout system to express complex data mapping patterns on GPUs (tiling, swizzling, vectorization).

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

## ⚙️ Hierarchical Kernel Control

RocDSL keeps the tiling hierarchy explicit across cluster, block, warp, thread, and instruction scopes. Declare tile shapes at each level, derive layouts, and partition tensors deterministically:

```python
THR_M, THR_N = 4, 32
VAL_M, VAL_N = 4, 4
CLUSTER_M, CLUSTER_N = 2, 2

thr_layout = rocir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
val_layout = rocir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))

copy_atom = rocir.make_copy_atom(T.f32(), vector_size=8)
tiled = rocir.make_tiled_copy_tv(
    copy_atom, thr_layout, val_layout,
    thr_shape=(THR_M, THR_N),
    val_shape=(VAL_M, VAL_N),
)

tensor_A = rocir.make_tensor(A, shape=(M, N), strides=(N, 1))
cluster_tiles = rocir.zipped_divide(
    tensor_A,
    (CLUSTER_M * THR_M * VAL_M, CLUSTER_N * THR_N * VAL_N),
)

blk_coord = (rocir.block_idx("y"), rocir.block_idx("x"))
blkA = cluster_tiles[blk_coord]
tid_linear = (rocir.thread_idx("y") * rocir.block_dim("x") + rocir.thread_idx("x")).value
thr_tiles = tiled.get_slice(tid_linear).partition_S(blkA)
```

With the per-level partitions in hand, you can allocate register fragments, emit predicate masks, and schedule MFMA/vector instructions while the compiler retains full knowledge of the execution hierarchy.

## 🧮 Minimal VecAdd Example

This condensed snippet mirrors `tests/benchmark/vecAdd.py`, highlighting how tiled copies, fragments, and benchmarking fit together:

```python
from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, rocir
import mlir.extras.types as T

THREADS = 256
TILE = 8
VEC = 4

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

@gpu.module("vec_kernels", ["#rocdl.target<abi = \"500\">"])
def mod():
    pass

@gpu.func(emit=True)
def vecAdd(A: T.memref(20480000, T.f32()),
           B: T.memref(20480000, T.f32()),
           C: T.memref(20480000, T.f32())):
    tid_linear = (rocir.thread_idx("y") * rocir.block_dim("x") +
                  rocir.thread_idx("x")).value
    thr_layout = rocir.make_ordered_layout((THREADS,), order=(0,))
    val_layout = rocir.make_ordered_layout((TILE,), order=(0,))
    copy_atom = rocir.make_copy_atom(T.f32(), vector_size=VEC)
    tiled = rocir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                                     thr_shape=(THREADS,), val_shape=(TILE,))
    tensor_A = rocir.make_tensor(A, shape=(20480000,), strides=(1,))
    tiles_A = rocir.zipped_divide(tensor_A, (THREADS * TILE,))
    blkA = tiles_A[(rocir.block_idx("x"),)]
    thrA = tiled.get_slice(tid_linear).partition_S(blkA)
    frgA = rocir.make_fragment_like(thrA, T.f32())
    rocir.copy(tiled, thrA, frgA)
    # repeat for B/C fragments, add, then store results
```

Compile the module with the pipeline, set up HIP device buffers, and invoke the helper utilities
in the tests/benchmarks for timing—just like the full benchmark.

## ✅ Testing Status

| Category | Status | Description |
|----------|--------|-------------|
| **MLIR Core** | ✅ Passing | Type parsing, Op verification, Basic transforms |
| **Rocir Ops** | ✅ Passing | Layout algebra, Coordinate lowering |
| **GPU Backend**| ✅ Passing | GPU kernel compilation, Shared memory, Vectorization |
| **Hardware** | ✅ Passing | MFMA (Matrix Fused Multiply-Add) execution on MI300 |

**Verified Platforms**:
*   AMD MI300X (gfx942), AMD MI350 (gfx950)
*   Linux / ROCm 6.x, 7.x

## License

Apache License 2.0
