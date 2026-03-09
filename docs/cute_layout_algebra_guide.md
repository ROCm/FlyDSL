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

FlyDSL implements the CuTe layout algebra for AMD GPUs through the FLIR MLIR dialect:

| Aspect | CuTe C++ (CUTLASS) | FlyDSL |
|---|---|---|
| **Language** | C++ templates | Python + MLIR emission |
| **Hardware** | NVIDIA CUDA GPUs | AMD ROCm/HIP GPUs |
| **IR backend** | C++ templates → CUDA/PTX | FLIR MLIR dialect → ROCDL → HSACO |
| **Kernel model** | C++ kernel functions | `MlirModule` class + `@kernel`/`@jit` |
| **Memory model** | GMEM → SMEM → RMEM | GMEM → LDS → VGPR |
| **Compilation** | nvcc / CUTLASS build | Python → MLIR → ROCDL → HSACO binary |
| **Wave/Warp size** | 32 threads (warp) | 64 threads (wavefront) |

---

## 2. Layout Algebra Fundamentals

### 2.1 Core Types

A **Layout** is defined by a pair `(Shape, Stride)`:

| Concept | Mathematical Definition | FlyDSL API |
|---|---|---|
| **Shape** | Tuple of positive integers describing dimensions | `flir.make_shape(M, N)` |
| **Stride** | Tuple of integers describing step sizes per dimension | `flir.make_stride(s0, s1)` |
| **Layout** | Pair `(Shape, Stride)` defining a coordinate → index mapping | `flir.make_layout(shape, stride)` |
| **Coord** | Tuple of integers identifying a position in logical space | `flir.make_coord(i, j)` |

> **Reference:** `include/cute/layout.hpp` — `Layout<Shape, Stride>` template class.

**FlyDSL example:**
```python
from flydsl.dialects.ext import flir

shape = flir.make_shape(128, 64)
stride = flir.make_stride(1, 128)    # Column-major
layout = flir.make_layout(shape, stride)
coord = flir.make_coord(3, 5)
```

### 2.2 Query Operations

| Operation | Formula | FlyDSL API |
|---|---|---|
| **size** | `product(shape)` — total number of elements | `flir.size(layout)` |
| **cosize** | `max(index) + 1` — size of the codomain | `flir.cosize(layout)` |
| **rank** | Number of modes (top-level dimensions) | `flir.rank(layout)` |
| **size of mode i** | `shape[i]` | `flir.get(flir.get_shape(layout), i)` |

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
| **crd2idx** | `coord → index = sum(c_i * d_i)` | `flir.crd2idx(coord, layout)` |
| **idx2crd** | `index → coord` (successive div/mod by shape elements) | `flir.idx2crd(idx, layout)` |

> **Reference:** `include/cute/layout.hpp` — `crd2idx()`, `idx2crd()`.

### 2.4 Layout Algebra Operations

All operations below are defined mathematically in the CuTe algebra and implemented in FlyDSL with identical semantics.

#### Composition

Given layouts `A = (S_A, d_A)` and `B = (S_B, d_B)`, the composition `A ∘ B` creates a new layout where B's indices are fed through A:

```
(A ∘ B)(c) = A(B(c))
```

FlyDSL: `flir.composition(A, B)`

> **Reference:** `include/cute/layout.hpp` — `composition()`.

#### Complement

The complement of layout `A` with respect to a codomain size `M` produces a layout `B` such that `(A, B)` together cover `[0, M)`:

FlyDSL: `flir.complement(layout, cotarget)`

> **Reference:** `include/cute/layout.hpp` — `complement()`.

#### Coalesce

Merges adjacent modes with compatible strides into a single mode, producing a simplified but functionally equivalent layout:

FlyDSL: `flir.coalesce(layout)`

> **Reference:** `include/cute/layout.hpp` — `coalesce()`.

#### Products

Products combine two layouts to create higher-rank layouts. They differ in how the result modes are organized:

| Product | Description | FlyDSL API |
|---|---|---|
| **Logical Product** | Append B's modes as new outer modes of A | `flir.logical_product(A, B)` |
| **Zipped Product** | Like logical, but zip inner modes together | `flir.zipped_product(A, B)` |
| **Tiled Product** | Like logical, but group by tile | `flir.tiled_product(A, B)` |
| **Flat Product** | Flatten all result modes | `flir.flat_product(A, B)` |
| **Raked Product** | Interleave A and B elements (raked distribution) | `flir.raked_product(A, B)` |
| **Blocked Product** | Block A elements together, then B (blocked distribution) | `flir.blocked_product(A, B)` |

> **Reference:** `include/cute/layout.hpp` — `logical_product()`, `zipped_product()`, `tiled_product()`, `flat_product()`, `raked_product()`, `blocked_product()`.

#### Divides

Divides decompose a layout by a tiler, creating a hierarchical layout with "tile" and "remainder" modes:

| Divide | Description | FlyDSL API |
|---|---|---|
| **Logical Divide** | Split A by tiler, keep full mode hierarchy | `flir.logical_divide(A, tiler)` |
| **Zipped Divide** | Like logical, but zip tile modes | `flir.zipped_divide(A, tiler)` |
| **Tiled Divide** | Like logical, but group by tile | `flir.tiled_divide(A, tiler)` |
| **Flat Divide** | Flatten tile and remainder modes | `flir.flat_divide(A, tiler)` |

> **Reference:** `include/cute/layout.hpp` — `logical_divide()`, `zipped_divide()`, `tiled_divide()`, `flat_divide()`.

#### Partitioning Utilities

| Operation | Description | FlyDSL API |
|---|---|---|
| **local_partition** | Partition a layout among threads/tiles | `flir.local_partition(layout, ...)` |
| **local_tile** | Extract a tile from a layout | `flir.local_tile(layout, ...)` |

> **Reference:** `include/cute/algorithm/` — `local_partition.hpp`, `local_tile.hpp`.

---

## 3. FlyDSL Kernel Development

FlyDSL kernels are defined as Python classes extending `MlirModule`, using `@kernel` for GPU device functions and `@jit` for host-side launch wrappers:

```python
from flydsl.lang.ir.module import MlirModule, kernel, jit
from flydsl.dialects.ext import flir, arith, gpu
import _mlir.extras.types as T

class MyKernel(MlirModule):
    GPU_MODULE_NAME = "my_kernel"
    GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942">']

    @kernel
    def my_kernel(self,
                  A: T.memref(T.dynamic(), T.f16()),
                  B: T.memref(T.dynamic(), T.f16()),
                  C: T.memref(T.dynamic(), T.f16()),
                  M: T.index()):
        # Kernel body — use layout algebra here
        ...

    @jit
    def __call__(self, A, B, C, M, stream_ptr: T.i64()):
        # Host-side launch wrapper
        ...
```

**Key elements:**
- `MlirModule` base class provides MLIR module construction
- `@kernel` decorator marks GPU device functions
- `@jit` decorator marks host-side launch wrappers
- Parameters use MLIR types: `T.memref(...)` for tensors, `T.index()` for integer scalars
- `GPU_MODULE_NAME` sets the kernel name in the compiled binary
- `GPU_MODULE_TARGETS` specifies target architecture(s)

---

## 4. Thread and Block Hierarchy

GPU kernels organize threads into a hierarchy of blocks and grids. FlyDSL provides direct access to thread/block indices:

| Concept | FlyDSL API | Description |
|---|---|---|
| Thread index | `flir.thread_idx("x")` | Thread index within block |
| Block index | `flir.block_idx("x")` | Block index within grid |
| Block dimension | `flir.block_dim("x")` | Number of threads per block |

Supported dimensions: `"x"`, `"y"`, `"z"`.

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

FlyDSL uses `TensorView` as its tensor abstraction, wrapping an MLIR memref with layout information:

```python
# Global memory tensor view
gmem_A = flir.TensorView(
    arg_a,                  # memref argument
    (tile_m, tile_k),       # shape
    strides=(1, M),         # strides
    base_indices=(base_m,), # base offset
    element_type=T.f16(),
)

# Register fragment
rmem_frag = flir.make_fragment_like(template_tensor)
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
from flydsl.utils import SmemAllocator

allocator = SmemAllocator(ctx, arch="gfx942")
lds_gen = allocator.allocate_array(T.f16(), num_elems=128*64)
allocator.finalize()  # Inside gpu.module body

# Inside kernel:
base = allocator.get_base()
lds_ptr = lds_gen(base)  # SmemPtr for typed access
```

### 5.3 Swizzling (Bank Conflict Avoidance)

Swizzling remaps addresses to avoid bank conflicts in shared/local memory. FlyDSL provides XOR-based swizzling at 16-byte granularity:

```python
col_swizzled = flir.swizzle_xor16(row, col_bytes, k_blocks16)
```

The swizzle function XORs the row index into the column address at 16-byte boundaries, distributing accesses across LDS banks.

---

## 6. Data Movement

### 6.1 Copy Atoms and Tiled Copies

FlyDSL uses the CuTe copy abstraction: a **copy atom** defines a single thread's copy capability, and a **tiled copy** distributes the atom across all threads:

```python
# Create copy atom
atom = flir.make_copy_atom(T.f16(), vector_size=8)

# Create tiled copy (distributed across threads)
thr_layout = flir.make_ordered_layout(256, 8)  # 256 threads, 8 elements each
val_layout = flir.make_layout(flir.make_shape(1, 8), flir.make_stride(0, 1))
tiled_copy = flir.make_tiled_copy_tv(atom, thr_layout, val_layout)

# Get thread slice
thr_copy = tiled_copy.get_slice(tid)
src_partition = thr_copy.partition_S(src_tensor)
dst_partition = thr_copy.partition_D(dst_tensor)

# Execute copy
flir.copy(tiled_copy, src_tensor, dst_tensor)
```

### 6.2 Buffer Loads (AMD-specific)

AMD GPUs provide buffer load instructions for efficient global memory access. FlyDSL exposes these as intrinsics:

```python
from flydsl.dialects.ext import buffer_ops

# Buffer resource for global memory (AMD buffer load dwordx4)
rsrc = buffer_ops.create_buffer_resource(arg_a, num_records)

# Buffer load 16 bytes
from kernels.mfma_preshuffle_pipeline import buffer_copy_gmem16_dwordx4
vec = buffer_copy_gmem16_dwordx4(flir, arg=arg_a, elem_type=T.i8(),
                                  idx_i32=offset, atom_g2r16=atom, rsrc=rsrc)
```

---

## 7. Compute Operations (MFMA)

AMD GPUs use MFMA (Matrix Fused Multiply-Add) instructions for matrix math. FlyDSL provides direct access to MFMA intrinsics:

```python
from flydsl.dialects.ext import rocdl

# FP16: mfma_f32_16x16x16f16 (K=16)
c_acc = rocdl.mfma_f32_16x16x16f16(a_pack, b_pack, c_acc)

# FP8: mfma_f32_16x16x32_fp8_fp8 (K=32)
c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_pack, b_pack, c_acc)

# INT8: mfma_i32_16x16x32_i8 (K=32)
c_acc = rocdl.mfma_i32_16x16x32_i8(a_pack, b_pack, c_acc)

# MXFP4 scaled MMA (GFX950 only)
c_acc = rocdl.mfma_scale_x128(a_pack, b_pack, c_acc, scale_a, scale_b)
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
from flydsl.dialects.ext import gpu
gpu.barrier()
```

---

## 9. Compilation and Execution

### 9.1 Compilation Pipeline

FlyDSL compiles Python → MLIR IR → ROCDL dialect → HSACO binary:

```python
import flydsl

# Option 1: Using compile() (preferred for production)
mod = MyKernel()
executor = flydsl.compile(mod)
executor(A_torch, B_torch, C_torch, M)

# Option 2: Using compile_to_hsaco() (for tests)
from tests.utils import compile_to_hsaco
hsaco_bytes = compile_to_hsaco(mod)
```

### 9.2 Environment Variables

| Variable | Description |
|---|---|
| `FLIR_CHIP` / `FLIR_GPU_ARCH` | Target architecture (e.g., `gfx942`, `gfx950`) |
| `FLIR_DUMP_IR=1` | Dump intermediate MLIR IR |
| `FLIR_DUMP_DIR=/path` | IR dump location |
| `COMPILE_ONLY=1` | Skip execution, compile only |
| `FLIR_NO_CACHE=1` | Disable compilation cache |
| `FLIR_TIME_COMPILE=1` | Print compilation timing |

---

## 10. Complete Example: GEMM with Layout Algebra

This example shows how layout algebra concepts come together in a FlyDSL GEMM kernel. The layout algebra handles data distribution across threads and memory hierarchy; MFMA instructions handle the compute.

```python
from flydsl.lang.ir.module import MlirModule, kernel, jit
from flydsl.dialects.ext import flir, arith, gpu, rocdl
from flydsl.utils import SmemAllocator
import _mlir.extras.types as T

class GemmKernel(MlirModule):
    GPU_MODULE_NAME = "gemm"
    GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942">']

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    @kernel
    def gemm(self, A, B, C, M, N, K):
        tid = flir.thread_idx("x")
        bid = flir.block_idx("x")

        # 1. Layout algebra: define tile shapes
        tile_shape = flir.make_shape(self.BLOCK_M, self.BLOCK_N)

        # 2. Layout algebra: partition work across blocks
        block_coord = flir.idx2crd(bid, flir.make_shape(M // self.BLOCK_M,
                                                         N // self.BLOCK_N))

        # 3. Layout algebra: create tiled copies for GMEM → LDS
        copy_atom = flir.make_copy_atom(T.f16(), vector_size=8)
        thr_layout = flir.make_ordered_layout(256, 8)
        tiled_copy = flir.make_tiled_copy_tv(copy_atom, thr_layout, ...)

        # 4. LDS allocation
        base = self.allocator.get_base()
        lds_a = self.lds_a_gen(base)
        lds_b = self.lds_b_gen(base)

        # 5. K-loop: load tiles, compute MFMA, accumulate
        for k_tile in range(K // self.BLOCK_K):
            # Load A, B tiles from GMEM to LDS
            flir.copy(tiled_copy, gmem_a_tile, lds_a)
            flir.copy(tiled_copy, gmem_b_tile, lds_b)
            gpu.barrier()

            # MFMA compute
            for ki in range(self.BLOCK_K // 16):
                a_frag = ...  # Load from LDS
                b_frag = ...  # Load from LDS
                c_acc = rocdl.mfma_f32_16x16x16f16(a_frag, b_frag, c_acc)
            gpu.barrier()

        # 6. Store C tile back to GMEM
        ...
```

See `kernels/preshuffle_gemm.py` for a complete, production-quality GEMM implementation.

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
- `flydsl/src/flydsl/dialects/ext/flir.py` — Layout algebra Python API
- `flydsl/src/flydsl/lang/ir/module.py` — MlirModule, @kernel, @jit
- `flydsl/src/flydsl/compiler/compiler.py` — `flir.compile()` pipeline
- `flydsl/src/flydsl/utils/smem_allocator.py` — SmemAllocator
- `flydsl/src/flydsl/dialects/ext/rocm.py` — ROCm dialect helpers
- `kernels/preshuffle_gemm.py` — GEMM implementation example
- `tests/kernels/test_vec_add.py` — VecAdd example with full layout algebra
