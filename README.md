# FLIR - MLIR Compiler Infrastructure for high performance ROCm kernels

FLIR is an MLIR-based compiler infrastructure for high performance ROCm kernels.
It provides a custom layout-algebra IR (Flir dialect), a lowering pipeline to GPU/ROCDL,
and a Python API (`pyflir`) for constructing and running kernels.

## Features

- **Flir Dialect** (layout algebra inspired by CuTe/CUTLASS)
  - Core abstractions: `!flir.shape`, `!flir.stride`, `!flir.layout`, `!flir.coord`
  - Algebra ops: composition/product/divide/partition + coordinate mapping ops
- **Python bindings** (`python/pyflir/`) with an embedded MLIR python package
  - No external `mlir` python wheel is required: MLIR python bindings are built and staged into `.flir/build/python_packages/pyflir/_mlir` (default; legacy `build/` also works)
- **GPU lowering** to HSACO via MLIR GPU → ROCDL pipeline
- **Tools**: `flir-opt` for pass testing and IR experimentation

## Repository layout (current)

```
FLIR/
├── CMakeLists.txt
├── build_llvm.sh              # build/prepare llvm-project (optional helper)
├── build.sh                   # build FLIR + python bindings (recommended)
├── run_tests.sh               # run MLIR + Python tests
├── include/                   # C++ headers (dialect/pass declarations)
├── lib/                       # C++ dialect + transforms + CAPI
├── tools/                     # flir-opt
├── python/                    # Python package sources (pyflir + helpers)
├── python_bindings/           # CMake targets for python extensions/bindings
└── tests/                     # mlir + python tests/benchmarks
```

## Prerequisites

- **ROCm**: required for GPU execution tests/benchmarks (IR-only tests do not need a GPU).
- **Build tools**: `cmake`, C++ compiler, and optionally `ninja` (faster).
- **Python**: Python 3 + `pip`.
  - `build_llvm.sh` installs `nanobind`, `numpy`, `pybind11`.
  - `python/requirements.txt` exists for auxiliary deps (`numpy`, `torch`) for runtime data initialize and result check.

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

### B) Build FLIR (C++ + embedded python package)

```bash
./build.sh
```

After a successful build, you will have:

- `.flir/build/bin/flir-opt` (default; legacy `build/bin/flir-opt` also works)
- Python package root at:
  - `.flir/build/python_packages/pyflir/`
  - This contains:
    - `pyflir/` (your Python API)
    - `_mlir/` (embedded MLIR python bindings)
    - optional `mlir/` shim (if present)

## Using the Python bindings


```bash
python3 -m pip install -e .
```

Build a wheel (default output under `.flir/dist/`):

```bash
python3 setup.py bdist_wheel
ls dist/
```

## Run tests

```bash
./run_tests.sh
```

What `run_tests.sh` does (high level):

- **MLIR file tests**: runs `tests/mlir/*.mlir` through `flir-opt --flir-to-standard`
- **Python IR tests**: runs `tests/python/ir/test_*.py` (no GPU required)
- **Python examples**: runs `tests/python/examples/test_*.py`
- **GPU execution tests** (only if ROCm is detected): runs `tests/python/gpu/test_*.py`
- **Benchmarks** (only if ROCm is detected): runs `tests/benchmark/*.py` via `pytest`

For the Python test folder organization, see `tests/python/README.md`.

## Troubleshooting

- **`flir-opt not found`**
  - Run `./build.sh`, or build it explicitly:
    - `cmake --build build --target flir-opt -j$(nproc)`

- **Python import issues (`No module named pyflir` / `No module named mlir`)**
  - Ensure you are using the embedded package:
    - `export PYTHONPATH=$(pwd)/build/python_packages/pyflir:$PYTHONPATH`

- **MLIR `.so` load errors**
  - Add MLIR build lib dir to the loader path:
    - `export LD_LIBRARY_PATH=$MLIR_PATH/lib:$LD_LIBRARY_PATH`

## 📐 Layout System

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
func.func @layout_example(%i: !flir.int, %j: !flir.int) -> !flir.int {
  // Create 2D layout (8, 16) with column-major stride (1, 8)
  %shape = flir.make_shape %c8, %c16 : (!flir.int, !flir.int) -> !flir.shape<2>
  %stride = flir.make_stride %c1, %c8 : (!flir.int, !flir.int) -> !flir.stride<2>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<2>, !flir.stride<2>) -> !flir.layout<2>

  // Convert coordinate (i, j) to linear index
  %coord = flir.make_coord %i, %j : (!flir.int, !flir.int) -> !flir.coord<2>
  %idx = flir.crd2idx %coord, %layout : (!flir.coord<2>, !flir.layout<2>) -> !flir.int

  return %idx : !flir.int
}
```

## 🐍 Python API (`pyflir`)

FLIR provides a high-level Python API for generating kernels.

### Layout Construction

```python
from pyflir.dialects.ext import flir, arith

# Create constants
c8 = arith.constant(8, index=True)
c16 = arith.constant(16, index=True)

# Create Layout
shape = flir.make_shape(c8, c16)
stride = flir.make_stride(arith.constant(1, index=True), c8)
layout = flir.make_layout(shape, stride)

# Coordinate to Index
coord = flir.make_coord(i, j)
idx = flir.crd2idx(coord, layout)
```

### Pipeline API

Easy-to-use compilation pipeline:

```python
from pyflir.compiler.pipeline import Pipeline

# Build and run optimization pipeline
pipeline = Pipeline() \
    .flir_to_standard() \
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

FLIR keeps the tiling hierarchy explicit across cluster, block, warp, thread, and instruction scopes. Declare tile shapes at each level, derive layouts, and partition tensors deterministically:

```python
THR_M, THR_N = 4, 32
VAL_M, VAL_N = 4, 4
CLUSTER_M, CLUSTER_N = 2, 2

thr_layout = flir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
val_layout = flir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))

copy_atom = flir.make_copy_atom(T.f32(), vector_size=8)
tiled = flir.make_tiled_copy_tv(
    copy_atom, thr_layout, val_layout,
    thr_shape=(THR_M, THR_N),
    val_shape=(VAL_M, VAL_N),
)

tensor_A = flir.make_tensor(A, shape=(M, N), strides=(N, 1))
cluster_tiles = flir.zipped_divide(
    tensor_A,
    (CLUSTER_M * THR_M * VAL_M, CLUSTER_N * THR_N * VAL_N),
)

blk_coord = (flir.block_idx("y"), flir.block_idx("x"))
blkA = cluster_tiles[blk_coord]
tid_linear = (flir.thread_idx("y") * flir.block_dim("x") + flir.thread_idx("x")).value
thr_tiles = tiled.get_slice(tid_linear).partition_S(blkA)
```

With the per-level partitions in hand, you can allocate register fragments, emit predicate masks, and schedule MFMA/vector instructions while the compiler retains full knowledge of the execution hierarchy.

## 🧮 Minimal VecAdd Example

This condensed snippet mirrors `tests/benchmark/vecAdd.py`, highlighting how tiled copies, fragments, and benchmarking fit together:

```python
from pyflir.compiler.context import RAIIMLIRContextModule
from pyflir.dialects.ext import gpu, flir
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
    tid_linear = (flir.thread_idx("y") * flir.block_dim("x") +
                  flir.thread_idx("x")).value
    thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
    val_layout = flir.make_ordered_layout((TILE,), order=(0,))
    copy_atom = flir.make_copy_atom(T.f32(), vector_size=VEC)
    tiled = flir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                                     thr_shape=(THREADS,), val_shape=(TILE,))
    tensor_A = flir.make_tensor(A, shape=(20480000,), strides=(1,))
    tiles_A = flir.zipped_divide(tensor_A, (THREADS * TILE,))
    blkA = tiles_A[(flir.block_idx("x"),)]
    thrA = tiled.get_slice(tid_linear).partition_S(blkA)
    frgA = flir.make_fragment_like(thrA, T.f32())
    flir.copy(tiled, thrA, frgA)
    # repeat for B/C fragments, add, then store results
```

Compile the module with the pipeline, set up HIP device buffers, and invoke the helper utilities
in the tests/benchmarks for timing—just like the full benchmark.

## ✅ Testing Status

| Category | Status | Description |
|----------|--------|-------------|
| **MLIR Core** | ✅ Passing | Type parsing, Op verification, Basic transforms |
| **Flir Ops** | ✅ Passing | Layout algebra, Coordinate lowering |
| **GPU Backend**| ✅ Passing | GPU kernel compilation, Shared memory, Vectorization |
| **Hardware** | ✅ Passing | MFMA (Matrix Fused Multiply-Add) execution on MI300 |

**Verified Platforms**:
*   AMD MI300X (gfx942), AMD MI350 (gfx950)
*   Linux / ROCm 6.x, 7.x

## License

Apache License 2.0
