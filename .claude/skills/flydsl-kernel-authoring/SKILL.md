---
name: flydsl-kernel-authoring
description: >
  Comprehensive reference for authoring FlyDSL GPU kernels on AMD GPUs.
  Covers the layout algebra, tiled copy/MMA, buffer ops, scf.for loops,
  SmemAllocator, autotuning, and common patterns. Use when writing,
  reviewing, or understanding FlyDSL kernel code.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# FlyDSL Kernel Authoring Skill

## Overview

FlyDSL is a Python DSL and MLIR-based compiler for writing high-performance GPU kernels on AMD GPUs (MI300X/MI350). It provides explicit layout algebra for controlling data movement, tiling, and memory access patterns. The layout system is the core abstraction that distinguishes FlyDSL from Triton/Gluon.

**Repository**: `/FlyDSL/` (installed in editable mode)
**Target GPU**: gfx942 (MI300X, CDNA3), gfx950 (MI350, CDNA4)
**Python**: 3.12, ROCm 7.2

---

## 1. Architecture and Compilation

### Pipeline
```
Python (@flyc.kernel/@flyc.jit)
  -> AST Rewriting (for/if -> scf.for/scf.if)
  -> MLIR Tracing (generates Fly dialect + gpu/arith/scf/memref ops)
  -> MlirCompiler.compile() (Fly -> ROCDL -> LLVM -> HSACO binary)
  -> JITCFunction (ExecutionEngine wrapper)
```

### Key Passes
1. `gpu-kernel-outlining` - Move kernel bodies to `gpu.func`
2. `fly-layout-lowering` - Lower layout algebra to arithmetic
3. `convert-fly-to-rocdl` - Fly ops -> ROCDL intrinsics
4. `gpu-module-to-binary` - Emit HSACO binary

### Key Source Paths
- `python/flydsl/compiler/` - JIT compilation (jit_function.py, kernel_function.py)
- `python/flydsl/expr/` - DSL expression API (primitive.py, derived.py, typing.py)
- `python/flydsl/expr/primitive.py` - All layout algebra functions
- `python/flydsl/expr/derived.py` - CopyAtom, MmaAtom, TiledCopy, TiledMma wrappers
- `python/flydsl/expr/gpu.py` - GPU operations (thread_idx, block_idx, barrier)
- `python/flydsl/expr/buffer_ops.py` - AMD buffer load/store intrinsics
- `python/flydsl/expr/rocdl.py` - MFMA and other ROCm intrinsics
- `python/flydsl/utils/smem_allocator.py` - LDS (shared memory) management
- `kernels/` - Pre-built kernels (preshuffle_gemm.py, layernorm, softmax, rmsnorm)

---

## 2. Layout System (Core Abstraction)

### Core Types
| Type | Description | Example |
|------|-------------|---------|
| `!fly.int_tuple` | Integer tuple (can be nested) | `(8, 16)`, `(8, (4, 2))` |
| `!fly.layout` | (Shape, Stride) pair | `(8, 16):(1, 8)` (col-major) |
| `!fly.memref` | Memory reference with layout | Typed pointer + layout info |

### Construction
```python
import flydsl.expr as fx

shape = fx.make_shape(8, 16)              # IntTuple (8, 16)
stride = fx.make_stride(1, 8)             # IntTuple (1, 8)
layout = fx.make_layout(shape, stride)    # Layout (8,16):(1,8)

# Shorthand with Python tuples
layout = fx.make_layout((8, 16), (1, 8))

# Coordinates
coord = fx.make_coord(i, j)

# Nested shapes for hierarchical tiling
shape_nested = fx.make_shape(9, (4, 8))   # (9, (4, 8))

# Identity layout
identity = fx.make_identity_layout((M, N))
```

### Coordinate Mapping
The fundamental operation maps logical coordinates to physical memory indices.

**Formula**: `Index = sum(coord_i * stride_i)`

```python
idx = fx.crd2idx(coord, layout)    # Coordinate -> linear index
coord = fx.idx2crd(idx, layout)    # Linear index -> coordinate
s = fx.size(layout)                # Total element count (product of shape)
```

**Example**: For layout `(8, 16):(1, 8)` (8x16, column-major):
- `crd2idx((3, 5), layout)` = `3*1 + 5*8` = 43
- `idx2crd(43, layout)` = `(43 % 8, 43 / 8)` = `(3, 5)`

### Query Operations
```python
fx.size(layout)           # Total element count
fx.get_shape(layout)      # Extract shape IntTuple
fx.get_stride(layout)     # Extract stride IntTuple
fx.get(int_tuple, i)      # Get i-th element
fx.rank(int_tuple)        # Number of top-level modes
```

### Layout Algebra Operations

#### Composition: `fx.composition(A, B)`
Compose two layouts: `result(x) = A(B(x))`. Used to apply permutations or tile coordinate mappings.

#### Complement: `fx.complement(tiler, target_size)`
Compute remaining modes not covered by tiler, up to target_size. Internal building block for divides.

#### Coalesce: `fx.coalesce(layout)`
Simplify layout by merging adjacent modes. Preserves mapping but flattens structure.

#### Right Inverse: `fx.right_inverse(layout)`
Compute right inverse of layout mapping.

#### Recast: `fx.recast_layout(layout, old_bits, new_bits)`
Adjust layout for type width change (e.g., FP16->FP8).

### Product Operations (Combine Layouts)
Products combine two layouts to create a larger layout:

```python
fx.logical_product(layout, tiler)   # Basic mode-wise concatenation
fx.raked_product(thr, val)          # Interleaved access pattern (common for TiledCopy)
fx.block_product(layout, tiler)     # Blocked access pattern
fx.zipped_product(layout, tiler)    # Zipped modes
fx.tiled_product(layout, tiler)     # Hierarchical tiled structure
fx.flat_product(layout, tiler)      # Flattened result
```

### Divide Operations (Partition Layouts)
Divides split a layout by a divisor, creating tile + rest dimensions:

```python
fx.logical_divide(layout, divisor)  # Basic partitioning (uses complement internally)
fx.zipped_divide(layout, divisor)   # Zipped division
fx.tiled_divide(layout, divisor)    # Hierarchical tiled division
fx.flat_divide(layout, divisor)     # Flattened division
```

### Structural Operations
```python
fx.select(int_tuple, indices=[0, 2])      # Pick specific modes
fx.group(int_tuple, begin=1, end=3)        # Group modes into nested tuple
fx.append(base, elem)                      # Append mode
fx.prepend(base, elem)                     # Prepend mode
fx.zip(lhs, rhs)                           # Zip two IntTuples
fx.slice(src, coord)                       # Slice at coordinate (None = keep mode)
```

---

## 3. Writing Kernels

### Basic Pattern
```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, buffer_ops, range_constexpr
from flydsl.expr.typing import T

@flyc.kernel
def my_kernel(
    A: fx.Tensor,         # GPU tensor (memref via DLPack)
    B: fx.Tensor,
    N: fx.Constexpr[int], # Compile-time constant
):
    tid = gpu.thread_idx.x    # Returns Int32
    bid = gpu.block_idx.x
    # ... kernel body ...

@flyc.jit
def launch(
    A: fx.Tensor,
    B: fx.Tensor,
    N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    my_kernel(A, B, N).launch(
        grid=(N // 256,), block=(256,), stream=stream
    )

# Usage:
import torch
A = torch.randn(1024, device="cuda", dtype=torch.float32)
B = torch.empty(1024, device="cuda", dtype=torch.float32)
launch(A, B, 1024)
```

### Parameter Types
| Type | Description | At host boundary |
|------|-------------|-----------------|
| `fx.Tensor` | GPU tensor (memref) | Auto-converted from torch.Tensor via DLPack |
| `fx.Constexpr[int]` | Compile-time constant | Different values -> different compiled kernels |
| `fx.Int32` | Runtime i32 | Auto-converted from Python int |
| `fx.Stream` | CUDA/HIP stream | `fx.Stream(None)` for default stream |

### Thread/Block Hierarchy
```python
from flydsl.expr import gpu

tid_x = gpu.thread_idx.x    # Thread index (Int32)
bid_x = gpu.block_idx.x     # Block index (Int32)
bdim_x = gpu.block_dim.x    # Block dimension
gdim_x = gpu.grid_dim.x     # Grid dimension
gpu.barrier()                # Workgroup synchronization
```

### Control Flow
```python
from flydsl.expr import range_constexpr

# Compile-time unrolled loop (emitted inline in IR)
for i in range_constexpr(N):
    ...

# Runtime loop (lowered to scf.for via AST rewriting)
for i in range(runtime_value):
    ...
```

### scf.for with Loop-Carried Values (Software Pipelining)

Use `init=` on `range()` to create an `scf.for` with explicit SSA phi nodes for loop-carried state. This is required for software pipelining (prefetch patterns) where data must flow across iterations.

**Pattern** (from `preshuffle_gemm.py`):
```python
# Prologue: load first tile
tile_0 = prefetch(0)
init_state = [acc_init, tile_0_flat_val1, tile_0_flat_val2, ...]

# scf.for with loop-carried state
# CRITICAL: bounds MUST be arith.index() values, NOT Python ints!
_start = arith.index(0)
_stop = arith.index(N - 1)
_step = arith.index(1)
for iv, state in range(_start, _stop, _step, init=init_state):
    acc_in = state[0]
    tile_in = state[1:]

    next_tile = prefetch(iv + 1)      # load NEXT data
    acc_in = compute(acc_in, tile_in)  # compute CURRENT

    results = yield [acc_in] + next_tile  # carry to next iter

# Epilogue: process last tile from results
acc_final = results[0]
tile_final = results[1:]
compute(acc_final, tile_final)
```

**How it works in MLIR:**
| Element | Meaning |
|---|---|
| `init=init_state` | List of SSA values that seed the `scf.for` block arguments for iteration 0 |
| `state` | The loop-carried block arguments (phi nodes) for THIS iteration |
| `yield [...]` | `scf.yield` feeds values back as next iteration's `state` |
| `results` | After loop exits, holds the last `yield`'s values (the `scf.for` op results) |

**Three critical pitfalls (all verified by debugging):**

1. **Loop bounds must be `arith.index()`, NOT Python ints.** If you write `range(0, 15, 1, init=...)`, the AST rewriter treats constant bounds as a Python `range` and unrolls the loop — silently ignoring `init=`. Use `arith.index(0)`, `arith.index(15)`, `arith.index(1)` instead.

2. **All `init` values must be raw MLIR `ir.Value`s.** FlyDSL wrappers like `Int32` / `Float32` don't have `.type` (only `.dtype`), and `scf.ForOp.__init__` calls `arg.type`. Unwrap via:
   ```python
   def _unwrap(v):
       return v.ir_value() if hasattr(v, 'ir_value') else v
   init_state = [_unwrap(v) for v in raw_list]
   ```

3. **Clear `SmemPtr._view_cache` before epilogue.** `SmemPtr.get()` caches the `memref.view` it creates. If called inside the `scf.for` body, the cached view is defined in the loop scope. Using it in the epilogue (outside the loop) causes an SSA dominance error. Fix:
   ```python
   # After the scf.for loop, before epilogue compute:
   my_smem_ptr._view_cache = None
   ```

### Arithmetic Operations
```python
from flydsl.expr import arith

c42 = fx.Index(42)                              # index type constant (preferred)
c3_14 = fx.Float32(3.14)                        # f32 constant (preferred)
mask = fx.Int32(0xFF)                            # i32 constant (preferred)
c42 = arith.constant(42, index=True)            # legacy index constant
c3_14 = arith.constant(3.14, type=T.f32())     # legacy f32 constant

# NOTE: arith.constant takes `type` as keyword arg, NOT positional
result = arith.addf(a, b)    # float add
result = arith.mulf(a, b)    # float multiply
result = arith.negf(a)       # float negate
result = arith.maximumf(a, b)  # float max (works on scalars AND vectors)
result = arith.select(cond, true_val, false_val)

# Compare floats (returns i1/vector<Nxi1>)
is_less = arith.cmpf(a, b, predicate="olt")    # ordered less-than
```

### Vector Arithmetic (IMPORTANT)
All arith ops (`addf`, `mulf`, `negf`, `maximumf`, `cmpf`, `select`) work on **both scalars and vectors**.
To broadcast a scalar to a vector, use `arith.constant_vector`:

```python
from flydsl._mlir.ir import VectorType

# Create a splat constant vector (e.g., all 2.0)
vec_type = VectorType.get([vec_width], fx.T.f32())
scale_vec = arith.constant_vector(2.0, vec_type)

# Now use it with vector ops
vA = fx.memref_load_vec(rA)        # load vec from register
vC = arith.mulf(vA, scale_vec)    # element-wise scale
```

### Arith Ops Availability Table
| Operation | Function | Works on Vectors | Notes |
|-----------|----------|-----------------|-------|
| Add | `arith.addf(a, b)` | Yes | |
| Multiply | `arith.mulf(a, b)` | Yes | |
| Negate | `arith.negf(a)` | Yes | |
| Max | `arith.maximumf(a, b)` | Yes | Good for ReLU |
| Compare | `arith.cmpf(a, b, pred)` | Yes | Returns i1/vec<i1> |
| Select | `arith.select(cond, t, f)` | Yes | |
| Abs | `arith.absf(a)` | **NO - does not exist** | Use `negf+cmpf+select` |
| FMA | `arith.fma(a, b, c)` | Not verified | Use `mulf+addf` instead |
| Splat const | `arith.constant_vector(val, vty)` | Creates vector | For scalar broadcast |

### Printf Debugging
```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```

---

## 4. Data Movement Patterns

### Layout-Based Copy (Preferred for Element-wise Kernels)

The standard pattern: divide tensor by tile size, slice by block/thread, copy via atoms.

```python
@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, BLOCK_DIM: fx.Constexpr[int]):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # 1. Divide tensor into blocks
    tA = fx.logical_divide(A, fx.make_layout(BLOCK_DIM, 1))
    tB = fx.logical_divide(B, fx.make_layout(BLOCK_DIM, 1))

    # 2. Select this block's tile
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))

    # 3. Further divide for per-thread access
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))  # 1 element per thread
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))

    # 4. Allocate registers
    RABTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA = fx.memref_alloca(RABTy, fx.make_layout(1, 1))

    # 5. Copy: global -> register -> compute -> global
    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    # ... compute on register values ...
    fx.copy_atom_call(copyAtom, rA, fx.slice(tB, (None, tid)))
```

### Vectorized Loads (Wide Copies)
```python
VEC_WIDTH = 4
copy_bits = VEC_WIDTH * 32   # 128 bits
MemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register)
copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

rA = fx.memref_alloca(MemRefTy, fx.make_layout(VEC_WIDTH, 1))

# Divide for VEC_WIDTH elements per thread
tA = fx.logical_divide(tA, fx.make_layout(VEC_WIDTH, 1))
fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)

# Load/store as vectors
vec = fx.memref_load_vec(rA)     # Load vector from register memref
fx.memref_store_vec(vec, rA)     # Store vector to register memref
```

### TiledCopy Abstraction (for 2D Copies)
```python
# Define thread and value layouts
thr_layout = fx.make_layout((4, 1), (1, 1))    # 4 threads
val_layout = fx.make_layout((1, 8), (1, 1))    # 8 values per thread

# Create copy atom
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# Build tiled copy with raked product layout
layout_thr_val = fx.raked_product(thr_layout, val_layout)
tile_mn = fx.make_tile(4, 8)
tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

# Get this thread's slice and partition
thr_copy = tiled_copy.get_slice(tid)
partition_src = thr_copy.partition_S(src_tensor)
partition_dst = thr_copy.partition_D(dst_tensor)
frag = fx.make_fragment_like(partition_src)

# Execute copy: src -> fragment -> dst
fx.copy(copy_atom, partition_src, frag)
fx.copy(copy_atom, frag, partition_dst)
```

### Buffer Load/Store (AMD Intrinsics)
```python
from flydsl.expr import buffer_ops

rsrc = buffer_ops.create_buffer_resource(tensor)
# offset is in ELEMENTS (not bytes)
data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
buffer_ops.buffer_store(data, rsrc, offset)
```

### Copy Atom Types
| Type | Bits | Usage |
|------|------|-------|
| `fx.UniversalCopy32b()` | 32 | 1x f32 element copy |
| `fx.UniversalCopy(64)` | 64 | 2x f32 elements |
| `fx.UniversalCopy(128)` | 128 | 4x f32 elements |
| `fx.rocdl.BufferCopy128b()` | 128 | AMD buffer load 4xf32 |

---

## 5. Shared Memory (LDS)

### SmemAllocator Pattern
```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.expr.typing import T
from flydsl.compiler.kernel_function import CompilationContext

allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")
lds_a = allocator.allocate_array(T.f16, 8192)  # Allocate typed arrays
lds_b = allocator.allocate_array(T.f16, 8192)

@flyc.kernel
def my_kernel(A: fx.Tensor, ...):
    lds_base = allocator.get_base()       # Get base ptr inside kernel
    lds_a_ptr = lds_a(lds_base)           # SmemPtr for typed access
    val = lds_a_ptr.load([idx])
    lds_a_ptr.store(val, [idx])

    # Finalize in GPU module body (before launch)
    comp_ctx = CompilationContext.get_current()
    with ir.InsertionPoint(comp_ctx.gpu_module_body):
        allocator.finalize()
```

### LDS Capacity
| Architecture | GPU | LDS per CU |
|---|---|---|
| gfx942 | MI300X | 64 KB |
| gfx950 | MI350 | 160 KB |

---

## 6. MFMA Integration (Matrix Math)

### Available MFMA Instructions
```python
from flydsl.expr import rocdl

# FP16/BF16 MFMA
result = rocdl.mfma_f32_16x16x16_f16(a, b, acc)

# FP8 MFMA
result = rocdl.mfma_f32_16x16x32_fp8(a, b, acc)

# INT8 MFMA
result = rocdl.mfma_i32_16x16x32i8(a, b, acc)
```

### GEMM Pattern (Preshuffle)
The preshuffle GEMM pattern in `kernels/preshuffle_gemm.py`:
1. B matrix is pre-shuffled to layout: (N/16, K/64, 4, 16, kpack_bytes)
2. A tiles loaded from global to LDS with XOR16 swizzle for bank-conflict avoidance
3. K64-byte micro-steps: each step issues 2x K32 MFMA operations
4. Ping-pong LDS (lds_stage=2) for overlapping loads with compute
5. Epilogue: either direct row-major store or CShuffle via LDS for packing

---

## 7. Reduction Patterns

### Warp Reduction (AMD wave64)
XOR-shuffle-based intra-wave reduction:
```python
width_i32 = arith.constant(64, type=T.i32())
for sh in [32, 16, 8, 4, 2, 1]:
    off = arith.constant(sh, type=T.i32())
    peer = gpu.ShuffleOp(val, off, width_i32, mode="xor").shuffleResult
    val = arith.AddFOp(val, peer).result  # or MaximumFOp for max
```

### Block Reduction
1. Intra-wave XOR shuffle (shifts: 32, 16, 8, 4, 2, 1)
2. Lane 0 writes per-wave partial to LDS
3. `gpu.barrier()`
4. Wave 0 reads and reduces NUM_WAVES partials from LDS

See `kernels/reduce.py` for reusable implementations.

---

## 8. Common Patterns and Recipes

### Element-wise Kernel Template
```python
@flyc.kernel
def elementwise_kernel(In: fx.Tensor, Out: fx.Tensor, BLOCK: fx.Constexpr[int], VEC: fx.Constexpr[int]):
    bid, tid = fx.block_idx.x, fx.thread_idx.x
    tile = BLOCK * VEC
    tIn = fx.logical_divide(In, fx.make_layout(tile, 1))
    tOut = fx.logical_divide(Out, fx.make_layout(tile, 1))
    tIn = fx.slice(tIn, (None, bid))
    tOut = fx.slice(tOut, (None, bid))
    tIn = fx.logical_divide(tIn, fx.make_layout(VEC, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(VEC, 1))
    MemTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(VEC, 1), fx.AddressSpace.Register)
    copy = fx.make_copy_atom(fx.UniversalCopy(VEC * 32), fx.Float32)
    rIn = fx.memref_alloca(MemTy, fx.make_layout(VEC, 1))
    rOut = fx.memref_alloca(MemTy, fx.make_layout(VEC, 1))
    fx.copy_atom_call(copy, fx.slice(tIn, (None, tid)), rIn)
    # Transform
    v = fx.memref_load_vec(rIn)
    v = fx.arith.mulf(v, v)  # example: square
    fx.memref_store_vec(v, rOut)
    fx.copy_atom_call(copy, rOut, fx.slice(tOut, (None, tid)))
```

### Element-wise Kernel Cookbook (GPU-Verified)
All recipes below follow the same vectorized copy_atom pattern (256 threads, vec_width=4, 128-bit loads).
Only the compute section between `memref_load_vec` and `memref_store_vec` differs.

```python
from flydsl._mlir.ir import VectorType

# --- Scale: C = A * scalar ---
vA = fx.memref_load_vec(rA)
vec_ty = VectorType.get([vec_width], fx.T.f32())
scale = arith.constant_vector(2.0, vec_ty)
vC = arith.mulf(vA, scale)

# --- Multiply: C = A * B ---
vC = arith.mulf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))

# --- FMA: D = A * B + C ---
vAB = arith.mulf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
vD = arith.addf(vAB, fx.memref_load_vec(rC))

# --- ReLU: C = max(A, 0) ---
vA = fx.memref_load_vec(rA)
vec_ty = VectorType.get([vec_width], fx.T.f32())
zero_vec = arith.constant_vector(0.0, vec_ty)
vC = arith.maximumf(vA, zero_vec)

# --- Abs: C = |A| (arith.absf does NOT exist) ---
vA = fx.memref_load_vec(rA)
vec_ty = VectorType.get([vec_width], fx.T.f32())
zero_vec = arith.constant_vector(0.0, vec_ty)
neg_vA = arith.negf(vA)
is_neg = arith.cmpf(vA, zero_vec, predicate="olt")
vC = arith.select(is_neg, neg_vA, vA)
```

### Naive GEMM Template (for understanding, not performance)
```python
@flyc.kernel
def naive_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               M: fx.Constexpr[int], N: fx.Constexpr[int], K: fx.Constexpr[int],
               BM: fx.Constexpr[int], BN: fx.Constexpr[int]):
    tid, bid = gpu.thread_idx.x, gpu.block_idx.x
    bm, bn = bid // (N // BN), bid % (N // BN)
    tm, tn = tid // BN, tid % BN
    row, col = bm * BM + tm, bn * BN + tn
    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)
    rsrc_c = buffer_ops.create_buffer_resource(C)
    acc = arith.constant(0.0, type=fx.T.f32())
    for k in range_constexpr(K):
        a = buffer_ops.buffer_load(rsrc_a, row * K + k, vec_width=1)
        b = buffer_ops.buffer_load(rsrc_b, k * N + col, vec_width=1)
        acc = arith.addf(acc, arith.mulf(a, b))
    buffer_ops.buffer_store(acc, rsrc_c, row * N + col)
```

---

## 9. Environment and Debugging

### IR Dump
```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_kernel.py
```
Produces numbered `.mlir` files per pipeline stage plus `final_isa.s`.

### Key Environment Variables
| Variable | Default | Description |
|---|---|---|
| `FLYDSL_DUMP_IR` | false | Dump IR at each stage |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | false | Emit DWARF debug info (source-to-asm mapping) |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | true | Enable kernel disk caching (in-memory cache is always active) |
| `FLYDSL_RUNTIME_CACHE_DIR` | ~/.flydsl/cache | Cache directory |
| `FLYDSL_COMPILE_OPT_LEVEL` | 2 | Optimization level (0-3) |
| `ARCH` | auto-detect | Override GPU architecture |

### Disk Cache Invalidation
The JIT disk cache auto-invalidates when kernel source or closure values change. Set `FLYDSL_RUNTIME_ENABLE_CACHE=0` only when modifying C++ passes or non-closure helper functions:
```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_kernel.py  # or: rm -rf ~/.flydsl/cache
```

### Source-to-Assembly Debug Info

FlyDSL supports source-to-assembly mapping for rocprofv3 ATT traces via the MLIR
`ensure-debug-info-scope-on-llvm-func` pass (equivalent to Triton's `add_di_scope`).

**How it works**:
1. FlyDSL's `FuncLocationTracker` generates MLIR `loc()` metadata pointing to Python source lines
2. The `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` pass converts MLIR locations into LLVM `DISubprogramAttr` / `DICompileUnitAttr` metadata
3. The `-g` flag in `gpu-module-to-binary` preserves this metadata as `.debug_line` in the HSACO binary
4. rocprofv3 ATT reads `.debug_line` to produce `code.json` with `"source_file:line"` entries

**Pipeline position**: After `reconcile-unrealized-casts`, before `gpu-module-to-binary`:
```
... -> reconcile-unrealized-casts
    -> ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}  (conditional on enable_debug_info)
    -> gpu-module-to-binary{format=fatbin opts=-g}
```

**Verification**: With `FLYDSL_DUMP_IR=1`, check `final_isa.s` for `.file` and `.loc` directives.
The PA decode kernel achieves 99.9% coverage (1109/1110 ISA instructions mapped to source).

**Key insight**: Without this pass, MLIR `loc()` metadata is silently dropped during MLIR-to-LLVM-IR
translation. The `-g` flag alone is useless — it preserves debug info, but there's none to preserve
without the DI scope pass.

### Autotune Module

FlyDSL includes a Triton-style autotune module at `/FlyDSL/python/flydsl/autotune.py`:

```python
from flydsl.autotune import autotune, Config, do_bench

@autotune(
    configs=[
        Config(block_dim=64, vec_width=4),
        Config(block_dim=128, vec_width=4),
        Config(block_dim=256, vec_width=4),
    ],
    key=['const_n'],     # re-tune when these arg values change
    warmup=5, rep=25,    # benchmark timing params
)
@flyc.jit
def myKernel(A, C, n: fx.Int32, const_n: fx.Constexpr[int],
             block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
             stream: fx.Stream = fx.Stream(None)):
    ...
```

- `Config` kwargs become `Constexpr` args injected into `@jit` call
- `Config.num_warps`, `waves_per_eu`, `maxnreg` are special compiler-level options
- First call benchmarks all configs; subsequent calls use cached best
- Disk cache at `~/.flydsl/autotune/{func_name}.json`
- `do_bench(fn, warmup=5, rep=25)` benchmarks using CUDA/HIP events, returns median ms

**IMPORTANT**: `waves_per_eu` does NOT work via `gpu-module-to-binary opts=`. It needs to be
set as an LLVM function attribute or through `rocdl-attach-target`. This is a known limitation.

**DLTensorAdaptor bug**: Do NOT use `flyc.from_dlpack()` with pre-wrapped tensors when calling
a `@jit` function with varying `Constexpr` values. The `DLTensorAdaptor` caches MLIR types from
the first `ir.Context`, which become invalid when a new context is created (causes segfault).
Pass raw `torch.Tensor` objects instead.

---

## 10. Troubleshooting

### Common Issues

1. **`arith.constant` signature**: Use `arith.constant(value, type=T.f32())` -- `type` is a keyword argument, NOT positional.

2. **`buffer_ops.buffer_load` offset**: The `offset` parameter is in ELEMENTS, not bytes.

3. **Cache stale after code changes**: The disk cache auto-invalidates on source/closure changes. Only set `FLYDSL_RUNTIME_ENABLE_CACHE=0` or clear `~/.flydsl/cache/` if you changed C++ passes or non-closure helpers.

4. **LDS overflow**: Check capacity (64KB on gfx942, 160KB on gfx950). Use `SmemAllocator` which tracks allocations.

5. **Dynamic vs Constexpr**: `Constexpr[int]` values are baked into IR -- different values produce different compiled kernels. Use `Int32` for truly dynamic values.

6. **Tensor layout marking**: For dynamic shapes or alignment, use `flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`.

7. **SmemAllocator finalize**: Must call `allocator.finalize()` inside the GPU module body (use `CompilationContext.get_current().gpu_module_body`).

8. **AMD wavefront size**: Always 64 on gfx9xx. Use shifts [32, 16, 8, 4, 2, 1] for full-wave reduction.

9. **tile_k alignment for GEMM**: `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step).

10. **INT4 (W4A8)**: A matrix is int8, B matrix is packed int4 (2 values/byte), unpacked to int8 in-kernel.

11. **`arith.absf` does not exist**: FlyDSL does not expose `arith.absf`. Use `negf + cmpf("olt") + select` pattern instead. See Element-wise Kernel Cookbook.

12. **Scalar broadcast to vector**: Use `arith.constant_vector(value, VectorType.get([width], fx.T.f32()))` to create a splat constant vector. Do NOT try to use a scalar directly with vector `mulf`/`addf` — types must match.

---

## 11. Comparison with Triton/Gluon

| Aspect | FlyDSL | Triton | Gluon |
|--------|--------|--------|-------|
| Layout control | Explicit layout algebra (Shape, Stride, Layout) | Implicit via block pointers | Implicit |
| Tiling | Manual via divide/product operations | Auto-tiling with `tl.program_id` | Auto-tiling |
| Memory access | Copy atoms, buffer load/store, TiledCopy | `tl.load`/`tl.store` | `gluon.load`/`gluon.store` |
| MFMA | Direct `rocdl.mfma_*` intrinsics | `tl.dot` | `gluon.dot` |
| Shared memory | SmemAllocator with explicit management | Implicit scratchpad | Implicit |
| Abstraction level | Low (near hardware) | Medium | Medium-High |
| Compilation | MLIR (Fly dialect -> LLVM -> HSACO) | MLIR (Triton dialect -> LLVM) | MLIR |
| Control | Maximum control over data layout and movement | Less control, more automation | Least control |

FlyDSL gives maximum control at the cost of verbosity. The layout algebra is the key differentiator -- it enables precise control over how data is arranged in registers, shared memory, and global memory, and how threads map to data.

---

## 12. Running Kernels

### SSH to Remote Host
```bash
# Run a kernel
ssh -o LogLevel=ERROR hjbog-srdc-39.amd.com 'docker exec hungry_dijkstra bash -c "cd /FlyDSL && python3 my_kernel.py"'

# Run existing tests
ssh -o LogLevel=ERROR hjbog-srdc-39.amd.com 'docker exec hungry_dijkstra bash -c "cd /FlyDSL && python3 tests/kernels/test_vec_add.py"'

# Run benchmarks
ssh -o LogLevel=ERROR hjbog-srdc-39.amd.com 'docker exec hungry_dijkstra bash -c "cd /FlyDSL && bash scripts/run_benchmark.sh"'
```
