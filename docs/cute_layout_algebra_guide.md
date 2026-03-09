# CuTe Layout Algebra Reference for FlyDSL

> FlyDSL implements the CuTe layout algebra for AMD GPUs. This guide covers the mathematical foundations of the layout algebra and how FlyDSL exposes them through its Python API.

The CuTe layout algebra was introduced in the [CUTLASS](https://github.com/NVIDIA/cutlass) C++ library under BSD-3-Clause license (`include/cute/`). FlyDSL adopts the same algebraic framework — shapes, strides, coordinate mappings, products, and divides — and provides a Python API targeting AMD ROCm/HIP GPUs via MLIR.

---

## 1. Overview

### 1.1 What is the CuTe Layout Algebra?

The CuTe layout algebra is a mathematical framework for describing multidimensional data layouts as compositions of shapes, strides, and coordinate transformations. It provides:

- **Layouts** as first-class objects: a pair `(Shape, Stride)` that maps logical coordinates to physical offsets
- **Algebraic operations**: composition, complement, products, and divides that transform layouts while preserving correctness
- **Tiling and partitioning**: systematic decomposition of data across threads, warps/wavefronts, and blocks

The algebra is defined in the C++ headers of CUTLASS (BSD-3-Clause):
- `include/cute/layout.hpp` — Layout type, shape/stride types, core operations
- `include/cute/tensor.hpp` — Tensor type (pointer + layout)
- `include/cute/algorithm/` — Copy, GEMM, and other algorithmic building blocks
- `include/cute/numeric/integral_constant.hpp` — Compile-time integer constants

A pure-Python reference implementation also exists in PyTorch:
- `torch/distributed/_pycute/layout.py` — Layout class with all algebra operations

### 1.2 FlyDSL as an AMD Implementation

FlyDSL implements the CuTe layout algebra for AMD GPUs through the Fly MLIR dialect:

| Aspect | CuTe C++ (CUTLASS) | FlyDSL |
|---|---|---|
| **Language** | C++ templates | Python + MLIR emission |
| **Hardware** | NVIDIA CUDA GPUs | AMD ROCm/HIP GPUs |
| **IR backend** | C++ templates → CUDA/PTX | Fly MLIR dialect → ROCDL → HSACO |
| **Kernel model** | C++ kernel functions | `@flyc.kernel` + `@flyc.jit` |
| **Memory model** | GMEM → SMEM → RMEM | GMEM → LDS → VGPR |
| **Compilation** | nvcc / CUTLASS build | Python → MLIR → ROCDL → HSACO binary |
| **Wave/Warp size** | 32 threads (warp) | 64 threads (wavefront) |

---

## 2. Layout Algebra Fundamentals

### 2.1 Core Types

A **Layout** is defined by a pair `(Shape, Stride)`:

| Concept | Mathematical Definition | FlyDSL API |
|---|---|---|
| **Shape** | Tuple of positive integers describing dimensions | `fx.make_shape(M, N)` |
| **Stride** | Tuple of integers describing step sizes per dimension | `fx.make_stride(s0, s1)` |
| **Layout** | Pair `(Shape, Stride)` defining a coordinate → index mapping | `fx.make_layout(shape, stride)` |
| **Coord** | Tuple of integers identifying a position in logical space | `fx.make_coord(i, j)` |

> **Reference:** `include/cute/layout.hpp` — `Layout<Shape, Stride>` template class.

**FlyDSL example:**
```python
import flydsl.expr as fx

shape = fx.make_shape(128, 64)
stride = fx.make_stride(1, 128)    # Column-major
layout = fx.make_layout(shape, stride)
coord = fx.make_coord(3, 5)
```

### 2.2 Query Operations

| Operation | Formula | FlyDSL API |
|---|---|---|
| **size** | `product(shape)` — total number of elements | `fx.size(layout)` |
| **cosize** | `max(index) + 1` — size of the codomain | `fx.cosize(layout)` |
| **rank** | Number of modes (top-level dimensions) | `fx.rank(layout)` |
| **size of mode i** | `shape[i]` | `fx.get(fx.get_shape(layout), i)` |

> **Reference:** `include/cute/layout.hpp` — `size()`, `cosize()`, `rank()` functions.

### 2.3 Coordinate Mapping

The fundamental operation of a layout is mapping a logical coordinate to a physical index:

```
index = crd2idx(coord, shape, stride) = dot(coord, stride)
```

For a layout `L = ((S0, S1), (d0, d1))` and coordinate `(c0, c1)`:

```
index = c0 * d0 + c1 * d1
```

The inverse operation recovers a coordinate from a linear index:

```
coord = idx2crd(index, shape, stride)
```

| Operation | Definition | FlyDSL API |
|---|---|---|
| **crd2idx** | `coord → index = sum(c_i * d_i)` | `fx.crd2idx(coord, layout)` |
| **idx2crd** | `index → coord` (successive div/mod by shape elements) | `fx.idx2crd(idx, layout)` |

> **Reference:** `include/cute/layout.hpp` — `crd2idx()`, `idx2crd()`.

### 2.4 Layout Algebra Operations

All operations below are defined mathematically in the CuTe algebra and implemented in FlyDSL with identical semantics.

#### Composition

Given layouts `A = (S_A, d_A)` and `B = (S_B, d_B)`, the composition `A ∘ B` creates a new layout where B's indices are fed through A:

```
(A ∘ B)(c) = A(B(c))
```

FlyDSL: `fx.composition(A, B)`

> **Reference:** `include/cute/layout.hpp` — `composition()`.

#### Complement

The complement of layout `A` with respect to a codomain size `M` produces a layout `B` such that `(A, B)` together cover `[0, M)`:

FlyDSL: `fx.complement(layout, cotarget)`

> **Reference:** `include/cute/layout.hpp` — `complement()`.

#### Coalesce

Merges adjacent modes with compatible strides into a single mode, producing a simplified but functionally equivalent layout:

FlyDSL: `fx.coalesce(layout)`

> **Reference:** `include/cute/layout.hpp` — `coalesce()`.

#### Products

Products combine two layouts to create higher-rank layouts. They differ in how the result modes are organized:

| Product | Description | FlyDSL API |
|---|---|---|
| **Logical Product** | Append B's modes as new outer modes of A | `fx.logical_product(A, B)` |
| **Zipped Product** | Like logical, but zip inner modes together | `fx.zipped_product(A, B)` |
| **Tiled Product** | Like logical, but group by tile | `fx.tiled_product(A, B)` |
| **Flat Product** | Flatten all result modes | `fx.flat_product(A, B)` |
| **Raked Product** | Interleave A and B elements (raked distribution) | `fx.raked_product(A, B)` |
| **Blocked Product** | Block A elements together, then B (blocked distribution) | `fx.blocked_product(A, B)` |

> **Reference:** `include/cute/layout.hpp` — `logical_product()`, `zipped_product()`, `tiled_product()`, `flat_product()`, `raked_product()`, `blocked_product()`.

#### Divides

Divides decompose a layout by a tiler, creating a hierarchical layout with "tile" and "remainder" modes:

| Divide | Description | FlyDSL API |
|---|---|---|
| **Logical Divide** | Split A by tiler, keep full mode hierarchy | `fx.logical_divide(A, tiler)` |
| **Zipped Divide** | Like logical, but zip tile modes | `fx.zipped_divide(A, tiler)` |
| **Tiled Divide** | Like logical, but group by tile | `fx.tiled_divide(A, tiler)` |
| **Flat Divide** | Flatten tile and remainder modes | `fx.flat_divide(A, tiler)` |

> **Reference:** `include/cute/layout.hpp` — `logical_divide()`, `zipped_divide()`, `tiled_divide()`, `flat_divide()`.

#### Partitioning Utilities

| Operation | Description | FlyDSL API |
|---|---|---|
| **local_partition** | Partition a layout among threads/tiles | `fx.local_partition(layout, ...)` |
| **local_tile** | Extract a tile from a layout | `fx.local_tile(layout, ...)` |

> **Reference:** `include/cute/algorithm/` — `local_partition.hpp`, `local_tile.hpp`.

---

## 3. FlyDSL Kernel Development

FlyDSL kernels are defined using `@flyc.kernel` for GPU device functions and `@flyc.jit` for host-side launch wrappers:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    # Kernel body — use layout algebra here
    ...

@flyc.jit
def launch(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    my_kernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream,
    )
```

**Key elements:**
- `@flyc.kernel` decorator compiles the function body into GPU IR via AST rewriting
- `@flyc.jit` decorator wraps a host-side function that constructs and launches kernels
- `fx.Tensor` denotes a GPU tensor argument
- `fx.Constexpr[int]` denotes a compile-time constant (affects cache key)
- `fx.Int32` denotes a dynamic int32 argument
- `fx.Stream` denotes a GPU stream argument

---

## 4. Thread and Block Hierarchy

GPU kernels organize threads into a hierarchy of blocks and grids. FlyDSL provides direct access to thread/block indices:

| Concept | FlyDSL API | Description |
|---|---|---|
| Thread index | `fx.thread_idx.x` | Thread index within block |
| Block index | `fx.block_idx.x` | Block index within grid |
| Block dimension | `fx.block_dim.x` | Number of threads per block |

Supported dimensions: `.x`, `.y`, `.z`.

**Hardware mapping (NVIDIA → AMD):**

| NVIDIA Concept | AMD Concept | Notes |
|---|---|---|
| Warp (32 threads) | Wavefront (64 threads) | Fundamental SIMD unit |
| Thread Block | Workgroup | Cooperative thread group |
| SM (Streaming Multiprocessor) | CU (Compute Unit) | Processing unit |
| Tensor Core (HMMA/GMMA) | MFMA (Matrix Fused Multiply-Add) | Matrix math unit |
| CUDA Core | Shader Processor | Scalar ALU |

---

## 5. Tensor Creation and Memory

### 5.1 Tensor Construction

FlyDSL provides tensor operations with layout-aware partitioning:

```python
import flydsl.expr as fx

# Create a buffer tensor from a tensor argument (AMD buffer descriptor)
A = fx.rocdl.make_buffer_tensor(A)

# Partition using layout algebra
tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
tA = fx.slice(tA, (None, bid))

# Register fragment
frag = fx.make_fragment_like(partition_src)
```

### 5.2 Memory Hierarchy

| Level | NVIDIA | AMD | Typical Size |
|---|---|---|---|
| Global Memory (GMEM) | Global Memory | Global Memory (HBM) | GBs |
| Shared/Local Memory | SMEM (48–228 KB) | LDS (64–160 KB) | Per-CU |
| Register File | RMEM (256 KB/SM) | VGPR (512 KB/CU) | Per-thread |
| L2 Cache | L2 Cache | L2 Cache | MBs |

**LDS allocation in FlyDSL:**
```python
from flydsl.utils.smem_allocator import SmemAllocator

allocator = SmemAllocator(ctx, arch="gfx942")
lds_gen = allocator.allocate_array(T.f16(), num_elems=128*64)
allocator.finalize()

base = allocator.get_base()
lds_ptr = lds_gen(base)
```

### 5.3 Swizzling (Bank Conflict Avoidance)

Swizzling remaps addresses to avoid bank conflicts in shared/local memory. FlyDSL provides XOR-based swizzling at 16-byte granularity:

```python
col_swizzled = fx.swizzle_xor16(row, col_bytes, k_blocks16)
```

The swizzle function XORs the row index into the column address at 16-byte boundaries, distributing accesses across LDS banks.

---

## 6. Data Movement

### 6.1 Copy Atoms and Tiled Copies

FlyDSL uses the CuTe copy abstraction: a **copy atom** defines a single thread's copy capability, and a **tiled copy** distributes the atom across all threads:

```python
import flydsl.expr as fx

# Create copy atom (e.g., 128-bit buffer copy)
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# Create tiled copy via raked product
thr_layout = fx.make_layout((4, 1), (1, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
layout_thr_val = fx.raked_product(thr_layout, val_layout)
tile_mn = fx.make_tile(4, 8)
tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

# Get thread slice
thr_copy = tiled_copy.get_slice(tid)
src_partition = thr_copy.partition_S(src_tensor)
dst_partition = thr_copy.partition_D(dst_tensor)

# Execute copy
fx.copy(copy_atom, src_partition, dst_partition)
```

### 6.2 Buffer Loads (AMD-specific)

AMD GPUs provide buffer load instructions for efficient global memory access. FlyDSL exposes these via the ``rocdl`` submodule:

```python
import flydsl.expr as fx

# Create a buffer tensor from a tensor argument (uses AMD buffer descriptor)
A_buf = fx.rocdl.make_buffer_tensor(A)

# Use buffer copy atoms for efficient memory access
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
```

---

## 7. Compute Operations (MFMA)

AMD GPUs use MFMA (Matrix Fused Multiply-Add) instructions for matrix math. FlyDSL provides direct access to MFMA intrinsics:

```python
import flydsl.expr as fx

# Create MFMA atom (16x16x4 FP32)
mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))

# Partition and execute GEMM
thr_mma = tiled_mma.thr_slice(tid)
frag_A = thr_mma.make_fragment_A(partition_A)
frag_B = thr_mma.make_fragment_B(partition_B)
frag_C = thr_mma.make_fragment_C(partition_C)
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

**MFMA instruction reference (AMD CDNA):**

| Instruction | Data Type | M×N×K | Architecture |
|---|---|---|---|
| `mfma_f32_16x16x16f16` | FP16 | 16×16×16 | GFX942+ |
| `mfma_f32_16x16x32_fp8_fp8` | FP8 | 16×16×32 | GFX942+ |
| `mfma_i32_16x16x32_i8` | INT8 | 16×16×32 | GFX942+ |
| `mfma_f32_32x32x8f16` | FP16 | 32×32×8 | GFX942+ |
| `mfma_scale_x128` | MXFP4 | 16×16×128 | GFX950 |

**K64-byte micro-step pattern (2× K32 per step):**
```python
for ku in range(tile_k_bytes // 64):
    a_val = lds_load_pack_k32(...)   # Load A from LDS
    b_val = load_b_pack_k32(...)     # Load B from GMEM
    c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_val, b_val, c_acc)
    # second half
    a_val2 = lds_load_pack_k32(...)
    b_val2 = load_b_pack_k32(...)
    c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_val2, b_val2, c_acc)
```

---

## 8. Synchronization

| FlyDSL API | Description |
|---|---|
| `gpu.barrier()` | Workgroup-level barrier (equivalent to `__syncthreads`) |

```python
import flydsl.expr as fx
fx.gpu.barrier()
```

---

## 9. Compilation and Execution

### 9.1 Compilation Pipeline

FlyDSL compiles Python → MLIR IR → ROCDL dialect → HSACO binary:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

# Define kernel and launch wrapper
@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, ...):
    ...

@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, C, ...,
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, C, ...).launch(
        grid=(...), block=(...), stream=stream,
    )

# Call the jit function — compilation happens automatically on first call
launch(A_torch, B_torch, C_torch, ..., stream=torch.cuda.Stream())
```

### 9.2 Environment Variables

| Variable | Description |
|---|---|
| `ARCH` | Target architecture (e.g., `gfx942`, `gfx950`) |
| `FLYDSL_DUMP_IR=1` | Dump intermediate MLIR IR |
| `FLYDSL_DUMP_DIR=/path` | IR dump location |
| `FLYDSL_COMPILE_ONLY=1` | Skip execution, compile only |
| `FLYDSL_NO_CACHE=1` | Disable compilation cache |
| `FLYDSL_RUNTIME_CACHE_DIR=/path` | Cache directory (default: `~/.flydsl/cache/`) |

---

## 10. Complete Example: GEMM with Layout Algebra

This example shows how layout algebra concepts come together in a FlyDSL GEMM kernel. The layout algebra handles data distribution across threads and memory hierarchy; MFMA instructions handle the compute.

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.slice(fx.zipped_divide(A, tileA), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, bid))

    # MFMA atom setup
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    # Tiled copies for A, B, C
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)

    # Partition and copy to register fragments
    frag_A = thr_mma.make_fragment_A(thr_mma.partition_A(bA))
    frag_B = thr_mma.make_fragment_B(thr_mma.partition_B(bB))
    frag_C = thr_mma.make_fragment_C(thr_mma.partition_C(bC))

    # ... copy data to fragments, then GEMM ...
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # Store result back to C
    # ...

@flyc.jit
def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
             stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
```

See `examples/03-tiledMma.py` for a complete working GEMM example, and
`kernels/preshuffle_gemm.py` for a production-quality GEMM implementation.

---

## 11. References

### CuTe Layout Algebra (BSD-3-Clause)
- **C++ headers:** [CUTLASS `include/cute/`](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
  - `layout.hpp` — Layout type, all algebra operations
  - `tensor.hpp` — Tensor type (pointer + layout)
  - `algorithm/` — Copy, GEMM, partitioning algorithms
- **GTC presentations:** "CuTe: A Layout Algebra for CUTLASS" — mathematical foundations and design rationale
- **PyCute reference:** `torch/distributed/_pycute/layout.py` — pure-Python layout algebra (open source, PyTorch)

### FlyDSL Source Files
- `python/flydsl/expr/` — Layout algebra and expression API (`primitive.py`, `derived.py`, etc.)
- `python/flydsl/expr/rocdl/` — ROCDL-specific operations
- `python/flydsl/compiler/` — JIT compilation pipeline (`kernel_function.py`, `jit_function.py`)
- `python/flydsl/utils/smem_allocator.py` — SmemAllocator
- `examples/01-vectorAdd.py` — VecAdd example with layout algebra
- `examples/02-tiledCopy.py` — Tiled copy example
- `examples/03-tiledMma.py` — Tiled MFMA GEMM example
- `kernels/preshuffle_gemm.py` — Production GEMM implementation
- `kernels/preshuffle_gemm_flyc.py` — GEMM using `@flyc.kernel` API
