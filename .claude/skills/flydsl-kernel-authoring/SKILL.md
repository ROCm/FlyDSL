---
name: flydsl-kernel-authoring
description: >
  Comprehensive reference for authoring FlyDSL GPU kernels on AMD GPUs.
  Covers the layout algebra, tiled copy/MMA, buffer ops, loop-carried range loops,
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
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl

@flyc.kernel
def my_kernel(
    A: fx.Tensor,         # GPU tensor (memref via DLPack)
    B: fx.Tensor,
    N: fx.Constexpr[int], # Compile-time constant
):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")
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

### Current Syntax Quick Reference

Use the current public FlyDSL surface from `kernels/preshuffle_gemm.py` when writing new kernels:

```python
Vec = fx.Vector

tx = gpu.thread_id("x")
bx = gpu.block_id("x")
by = gpu.block_id("y")

i32_m: fx.Int32
c_m = fx.Index(i32_m)
c4 = fx.Index(4)
zero_f = fx.Float32(0.0)

layout = fx.make_layout((4, 64), (64, 1))
coord = fx.idx2crd(tx, layout)
wave_id = fx.get(coord, 0)
lane_id = fx.get(coord, 1)

acc = Vec.filled(4, 0.0, fx.Float32)
v_i64 = Vec(raw_vec).bitcast(fx.Int64)
elem0 = v_i64[0]

rsrc = buffer_ops.create_buffer_resource(tensor, max_size=True)
word = buffer_ops.buffer_load(rsrc, fx.Int32(offset), vec_width=4, dtype=fx.Int32)
```

Older code may use `gpu.thread_idx.x`, `gpu.block_idx.x`, `arith.constant(...)`, `T.i32`, and raw `vector.*` helpers. Keep those when editing existing code that already uses them heavily, but prefer `gpu.thread_id/block_id`, `fx.Index`/`fx.Int32`/`fx.Float32`, and `fx.Vector` for new code.

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

tid_x = gpu.thread_id("x")   # Preferred current spelling
bid_x = gpu.block_id("x")
bid_y = gpu.block_id("y")

# Legacy spelling still appears in older kernels:
tid_x = gpu.thread_idx.x
bid_x = gpu.block_idx.x

gpu.barrier()                # Workgroup synchronization
```

### Control Flow
```python
from flydsl.expr import range_constexpr

# Compile-time unrolled loop (emitted inline in IR)
for i in range_constexpr(N):
    ...

# Runtime loop (lowered by AST rewriting)
for i in range(runtime_value):
    ...
```

### Runtime vs Compile-Time Conditions (Current Style)

Use Python/DSL operators for runtime SSA comparisons. The AST rewriter lowers dynamic `if` conditions to `scf.IfOp`, and comparison operators like `==`, `<`, `>=` generate the needed MLIR predicates.

```python
tid = gpu.thread_id("x")
lane = tid % fx.Index(64)
c_zero = fx.Index(0)
c_limit = fx.Index(8)

# Preferred: readable DSL comparisons
if lane == c_zero:
    ...

in_range = lane < c_limit
val = fx.arith.select(in_range, good_val, zero_val)

# Avoid for simple integer comparisons
in_range = arith.cmpi(arith.CmpIPredicate.slt, lane, c_limit)
```

Use `const_expr(...)` only for values known at trace/compile time, such as Python booleans, constexpr arguments, loop-unroll choices, or type/layout branches:

```python
if const_expr(trans_v):
    ...

if const_expr(max_context_partition_num <= WARP_SIZE):
    ...
```

Do **not** wrap GPU runtime values in `const_expr`. Even with `@flyc.kernel(known_block_size=(256, 1, 1))`, `gpu.thread_id("x")`, `lane`, and `warp_id` are runtime SSA values; the compiler knows their range, not the current lane instance.

```python
# Wrong: lane depends on gpu.thread_id("x")
if const_expr(lane == c_zero):
    ...

# Correct
if lane == c_zero:
    ...
```

Keep explicit `arith.cmpi(...)` / `arith.unwrap(...)` for low-level manual MLIR construction, such as passing a raw condition to `scf.IfOp` directly:

```python
cond = arith.unwrap(partition_idx >= visible_tile_count)
if_op = scf.IfOp(cond, has_else=False)
```

### Frontend Semantic Restrictions
When writing or reviewing `@flyc.kernel` / `@flyc.jit` code, proactively avoid these patterns because they can conflict with MLIR construction even if they look valid in plain Python.

1. **Do not define values inside `if/else` and use them later outside the branch.** Keep a single explicit definition path.
   ```python
   if cond:
       dst = a
   else:
       dst = b
   use(dst)  # avoid this pattern
   ```

2. **Do not mutate captured outer variables inside nested helper functions.** Read-only closure capture is acceptable, but writes should go through explicit parameters and return values.
   ```python
   def kernel():
       acc = fx.Float32(0.0)

       def helper(acc):
           acc = acc + fx.Float32(1.0)
           return acc

       acc = helper(acc)
   ```

3. **Avoid early `return`, and do not place `return` / `yield` inside `if/else` branches.** Prefer a single explicit exit so the frontend can determine result types.
   ```python
   if cond:
       out = v0
   else:
       out = v1
   return out
   ```

4. **Compile-time conditions must use `const_expr(...)`.** Use `if const_expr(flag): ...` for constexpr flags and other static decisions. A plain Python `if` is only safe when the condition is already a Python `bool`.

5. **Runtime branches inside helper functions should be dispatched via local `@flyc.jit`.** When a branch body has side effects, loop-carried values, or branch-local definitions, split the branch bodies into local helpers and wrap the `if` in a local JIT helper:
   ```python
   def then_path():
       ...

   def else_path():
       ...

   @flyc.jit
   def dispatch():
       if runtime_cond:
           then_path()
       else:
           else_path()

   dispatch()
   ```

### Runtime Loops with Loop-Carried Values (Software Pipelining)

Use `init=` on `range()` to create a runtime loop with explicit SSA phi nodes for loop-carried state. This is required for software pipelining (prefetch patterns) where data must flow across iterations.

**Pattern** (from `preshuffle_gemm.py`):
```python
# Prologue: load first tile
tile_0 = prefetch(0)
init_state = [acc_init, tile_0_flat_val1, tile_0_flat_val2, ...]

# Runtime loop with loop-carried state
# Use fx.Index(...) bounds so the AST rewriter does not treat this as a Python unrolled range.
_start = fx.Index(0)
_stop = fx.Index(N - 1)
_step = fx.Index(1)
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
| `init=init_state` | List of SSA values that seed the runtime loop block arguments for iteration 0 |
| `state` | The loop-carried block arguments (phi nodes) for THIS iteration |
| `yield [...]` | Feeds values back as next iteration's `state` |
| `results` | After loop exits, holds the last yielded values |

**Three critical pitfalls (all verified by debugging):**

1. **Loop bounds must be DSL index values, NOT Python ints.** If you write `range(0, 15, 1, init=...)`, the AST rewriter treats constant bounds as a Python `range` and unrolls the loop — silently ignoring `init=`. Use `fx.Index(0)`, `fx.Index(15)`, `fx.Index(1)` instead.

2. **Prefer internal types, but unwrap at hard boundaries.** Most `range(..., init=...)` uses accept DSL numeric/vector values. If a lower-level helper explicitly expects raw `ir.Value`, unwrap with `v.ir_value()` / `_raw(v)` at that boundary only.

3. **Clear `SmemPtr._view_cache` before epilogue.** `SmemPtr.get()` caches the view it creates. If called inside the runtime loop body, the cached view is defined in the loop scope. Using it in the epilogue (outside the loop) causes an SSA dominance error. Fix:
   ```python
   # After the runtime loop, before epilogue compute:
   my_smem_ptr._view_cache = None
   ```

### Arithmetic Operations
```python
c42 = fx.Index(42)                              # index type constant (preferred)
c3_14 = fx.Float32(3.14)                        # f32 constant (preferred)
mask = fx.Int32(0xFF)                            # i32 constant (preferred)

# Prefer operators / Numeric methods
result = a + b
result = a * scale
result = cond.select(true_val, false_val)

# Keep direct arith.*FOp only when explicit fastmath flags are required.
```

### Internal Types: Vector and Numeric (PREFERRED)

Use FlyDSL's internal typed system instead of raw MLIR ops. The `Vector` class wraps `vector<NxTy>` with operator overloading and type-safe methods.

```python
Vec = fx.Vector

# Wrap raw vector values
acc = Vec(frag_C.load())      # vector<Nxf32> → Vector with * / + operators

# Indexing (replaces vector.extract)
val = acc[idx]                 # returns Float32 scalar

# Bitcast (replaces vector.bitcast)
v_f32 = Vec(raw_vec).bitcast(fx.Float32)  # vector<Nxi32> → vector<Nxf32>

# Type conversion (replaces arith.trunc_f / arith.ext_f)
bf16_val = f32_val.to(fx.BFloat16)     # f32 → bf16

# Arithmetic — use Python operators, not arith.mulf/addf
result = (val * scale_a) * scale_b

# Splat constant vector
zeros = Vec.filled(N, 0.0, fx.Float32)

# Index cast — use fx.Int32 instead of arith.index_cast
idx = fx.Int32(gpu.block_id("x") * tile_m)
```

**Prefer internal types over raw ops:**
| Raw MLIR op | Internal type equivalent |
|-------------|------------------------|
| `vector.extract(v, static_position=[i], ...)` | `Vec(v)[i]` |
| `vector.bitcast(target_ty, v)` | `Vec(v).bitcast(Float32)` |
| `arith.trunc_f(ty, v)` | `v.to(BFloat16)` |
| `arith.mulf(a, b)` | `a * b` |
| `arith.addf(a, b)` | `a + b` |
| `arith.index_cast(T.i32, v)` | `fx.Int32(v)` |

Use `Vec.filled(...)` for splats and `Vec.from_elements(...)` for vectors from scalars.

### Arith Ops Availability Table
| Operation | Function | Works on Vectors | Notes |
|-----------|----------|-----------------|-------|
| Add | `a + b` | Yes | Use direct FOp only for explicit fastmath |
| Multiply | `a * b` | Yes | Use direct FOp only for explicit fastmath |
| Negate | `-a` | Yes | |
| Max | `a.maximumf(b)` | Yes | Good for ReLU |
| Compare | `arith.cmpf(a, b, pred)` | Yes | Returns i1/vec<i1> |
| Select | `cond.select(t, f)` | Yes | |
| Abs | no direct helper | Use `-v`, comparison, and `cond.select(...)` |
| FMA | `a * b + c` | Yes | Use direct FOp only when explicit fastmath is needed |
| Splat const | `Vec.filled(width, val, dtype)` | Creates vector | For scalar broadcast |

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
width_i32 = fx.Int32(64)
for sh in [32, 16, 8, 4, 2, 1]:
    off = fx.Int32(sh)
    peer = gpu.ShuffleOp(val, off, width_i32, mode="xor").shuffleResult
    val = ArithValue(val) + peer  # use explicit FOp only if fastmath flags are needed
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
v = Vec(fx.memref_load_vec(rIn))
v = v * v  # example: square
    fx.memref_store_vec(v, rOut)
    fx.copy_atom_call(copy, rOut, fx.slice(tOut, (None, tid)))
```

### Element-wise Kernel Cookbook (GPU-Verified)
All recipes below follow the same vectorized copy_atom pattern (256 threads, vec_width=4, 128-bit loads).
Only the compute section between `memref_load_vec` and `memref_store_vec` differs.

```python
# --- Scale: C = A * scalar ---
vA = Vec(fx.memref_load_vec(rA))
scale = Vec.filled(vec_width, 2.0, fx.Float32)
vC = vA * scale

# --- Multiply: C = A * B ---
vC = Vec(fx.memref_load_vec(rA)) * Vec(fx.memref_load_vec(rB))

# --- FMA: D = A * B + C ---
vAB = Vec(fx.memref_load_vec(rA)) * Vec(fx.memref_load_vec(rB))
vD = vAB + Vec(fx.memref_load_vec(rC))

# --- ReLU: C = max(A, 0) ---
vA = Vec(fx.memref_load_vec(rA))
zero_vec = Vec.filled(vec_width, 0.0, fx.Float32)
vC = vA.maximumf(zero_vec)

# --- Abs: C = |A| (arith.absf does NOT exist) ---
vA = fx.memref_load_vec(rA)
zero_vec = Vec.filled(vec_width, 0.0, fx.Float32)
neg_vA = -vA
is_neg = vA < zero_vec
vC = is_neg.select(neg_vA, vA)
```

### Naive GEMM Template (for understanding, not performance)
```python
@flyc.kernel
def naive_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               M: fx.Constexpr[int], N: fx.Constexpr[int], K: fx.Constexpr[int],
               BM: fx.Constexpr[int], BN: fx.Constexpr[int]):
    tid, bid = gpu.thread_id("x"), gpu.block_id("x")
    bm, bn = bid // (N // BN), bid % (N // BN)
    tm, tn = tid // BN, tid % BN
    row, col = bm * BM + tm, bn * BN + tn
    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)
    rsrc_c = buffer_ops.create_buffer_resource(C)
    acc = fx.Float32(0.0)
    for k in range_constexpr(K):
        a = buffer_ops.buffer_load(rsrc_a, row * K + k, vec_width=1)
        b = buffer_ops.buffer_load(rsrc_b, k * N + col, vec_width=1)
        acc = acc + a * b
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

1. **Constants/casts**: Prefer `fx.Int32(...)`, `fx.Int64(...)`, `fx.Index(...)`, and `fx.Float32(...)`. Use `arith.constant(...)` only at low-level boundaries.

2. **`buffer_ops.buffer_load` offset**: The `offset` parameter is in ELEMENTS, not bytes.

3. **Cache stale after code changes**: The disk cache auto-invalidates on source/closure changes. Only set `FLYDSL_RUNTIME_ENABLE_CACHE=0` or clear `~/.flydsl/cache/` if you changed C++ passes or non-closure helpers.

4. **LDS overflow**: Check capacity (64KB on gfx942, 160KB on gfx950). Use `SmemAllocator` which tracks allocations.

5. **Dynamic vs Constexpr**: `Constexpr[int]` values are baked into IR -- different values produce different compiled kernels. Use `Int32` for truly dynamic values.

6. **Tensor layout marking**: For dynamic shapes or alignment, use `flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`.

7. **SmemAllocator finalize**: Must call `allocator.finalize()` inside the GPU module body (use `CompilationContext.get_current().gpu_module_body`).

8. **AMD wavefront size**: Always 64 on gfx9xx. Use shifts [32, 16, 8, 4, 2, 1] for full-wave reduction.

9. **tile_k alignment for GEMM**: `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step).

10. **INT4 (W4A8)**: A matrix is int8, B matrix is packed int4 (2 values/byte), unpacked to int8 in-kernel.

11. **`arith.absf` does not exist**: Prefer `Vector`/`ArithValue` operators: `neg = -v`, `is_neg = v < zero`, `out = is_neg.select(neg, v)`.

12. **Scalar broadcast to vector**: Use `Vec.filled(width, value, fx.Float32)` to create a splat constant vector. Do NOT use raw vector ops for ordinary arithmetic.

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
