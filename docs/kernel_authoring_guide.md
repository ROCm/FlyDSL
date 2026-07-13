# Kernel Authoring Guide

> Writing GPU kernels with FlyDSL: `@flyc.jit`, `@flyc.kernel`, expression API, launch configuration, shared memory, and synchronization.

> **API**: This guide documents the `@flyc.kernel`/`@flyc.jit` API from `flydsl.compiler` and `flydsl.expr` (`python/flydsl/`).

## Quick Reference

| Concept | API | Description |
|---|---|---|
| **JIT host func** | `@flyc.jit` | Emit host-side launcher with JIT compilation |
| **GPU kernel** | `@flyc.kernel` | Define GPU kernel function |
| **Launch** | `kernel(...).launch(grid=, block=)` | Configure and emit GPU launch |
| **Thread ID** | `fx.gpu.thread_idx.x` | Get thread index in workgroup |
| **Block ID** | `fx.gpu.block_idx.x` | Get block/workgroup index |
| **Block dim** | `fx.gpu.block_dim.x` | Get block dimension size |
| **Compile-time** | `fx.Constexpr[int]` | Compile-time constant parameter |
| **Tensor arg** | `fx.Tensor` | GPU tensor argument (via DLPack) |
| **Stream arg** | `fx.Stream` | CUDA/HIP stream argument |
| **Barrier** | `fx.gpu.barrier()` | Workgroup synchronization |
| **Constants** | `fx.Int32` / `fx.Index` / `fx.Float32` | Create typed DSL constants |
| **Range loop** | `range_constexpr(n)` | Compile-time unrolled loop |
| **Buffer load** | `buffer_ops.buffer_load(rsrc, off)` | AMD buffer load intrinsic |

---

## 1. Basic Kernel Pattern

### 1.1 `@flyc.kernel` + `@flyc.jit`

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

@flyc.kernel
def vec_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    idx = bid * 256 + tid
    # ... kernel body using fx.*, ArithValue, Vector, and buffer ops ...

@flyc.jit
def vec_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    vec_add_kernel(A, B, C, N).launch(
        grid=(N // 256,),
        block=(256,),
        stream=stream,
    )

# Usage:
import torch
A = torch.randn(1024, device="cuda", dtype=torch.float32)
B = torch.randn(1024, device="cuda", dtype=torch.float32)
C = torch.empty(1024, device="cuda", dtype=torch.float32)

vec_add(A, B, C, 1024)
```

### 1.2 How It Works

1. `@flyc.kernel` wraps the function as a `KernelFunction`
2. `@flyc.jit` wraps the function as a `JitFunction`
3. On first call, `JitFunction.__call__` triggers:
   - AST rewriting (Python loops/ifs → MLIR scf ops)
   - MLIR module creation with `gpu.container_module`
   - Tracing the jit function body to generate MLIR ops
   - Calling `vec_add_kernel(...)` emits a `gpu.func` in `gpu.module`
   - `.launch()` emits `gpu.launch_func`
   - `MlirCompiler.compile()` runs the full pass pipeline
   - `JITCFunction` wraps the resulting ExecutionEngine
4. Subsequent calls with the same type signature use the cached binary

---

## 2. Parameter Types

### 2.1 `fx.Tensor`

Maps a PyTorch tensor to an MLIR memref descriptor via DLPack:

```python
@flyc.kernel
def my_kernel(input: fx.Tensor, output: fx.Tensor):
    # input and output are Tensor wrappers around ir.Value (memref)
    ...
```

At the host boundary, `torch.Tensor` is automatically converted via `TensorAdaptor`.

### 2.2 `fx.Constexpr[T]`

Compile-time constant. Value is embedded directly in the generated IR:

```python
@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int], dtype: fx.Constexpr[str]):
    for i in range_constexpr(N // 64):  # unrolled at compile time
        ...
```

Different `Constexpr` values produce different compiled kernels (separate cache entries).

### 2.3 `fx.Int32`

Runtime integer parameter (passed as `i32`):

```python
@flyc.jit
def launch(data: fx.Tensor, size: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    ...
```

Python `int` values are automatically converted to `Int32` via the `JitArgumentRegistry`.

### 2.4 `fx.Stream`

CUDA/HIP stream for asynchronous kernel launch:

```python
@flyc.jit
def launch(data: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    my_kernel(data).launch(grid=(1,), block=(256,), stream=stream)

# Launch on specific stream:
stream = torch.cuda.Stream()
launch(data, stream=fx.Stream(stream))
```

### 2.5 Custom Argument Types

Register new Python types for the JIT boundary:

```python
from flydsl.compiler import JitArgumentRegistry

@JitArgumentRegistry.register(MyCustomType, dsl_type=MyDslType)
class MyCustomAdaptor:
    def __init__(self, value: MyCustomType):
        self.value = value

    def __get_ir_types__(self):
        return [...]  # MLIR types for this argument

    def __get_c_pointers__(self):
        return [...]  # ctypes pointers for invocation
```

---

## 3. Thread / Block Hierarchy

```python
from flydsl.expr import gpu

# Thread index within workgroup (returns Int32)
tid_x = gpu.thread_idx.x
tid_y = gpu.thread_idx.y
tid_z = gpu.thread_idx.z

# Block (workgroup) index within grid
bid_x = gpu.block_idx.x
bid_y = gpu.block_idx.y

# Block dimensions
bdim_x = gpu.block_dim.x

# Grid dimensions
gdim_x = gpu.grid_dim.x

# Low-level (returns raw ir.Value)
raw_tid = gpu.thread_id("x")
raw_bid = gpu.block_id("x")
```

---

## 4. Expression API (`flydsl.expr`)

### 4.1 Arithmetic and Numeric Types

```python
import flydsl.expr as fx

# Constants (prefer DSL numeric types)
c42 = fx.Index(42)          # index type constant
c3_14 = fx.Float32(3.14)    # f32 constant
mask = fx.Int32(0xFF)       # i32 constant

# Arithmetic (operator overloading via ArithValue / Numeric)
result = a + b
result = a * 2
result = a // 4
result = a % 16

# Cast (prefer DSL numeric constructors)
idx = fx.Index(int_val)     # cast to index type
i32_val = fx.Int32(idx)     # cast to i32

# Select
result = cond.select(true_val, false_val)  # when cond is an ArithValue

# Bitwise
result = a & b
result = a ^ b
result = a << 4
```

Use direct `arith.*FOp(..., fastmath=...)` only where explicit fastmath flags are performance-critical.

### 4.2 Vector Values (`Vector`)

```python
from flydsl.expr.typing import Vector as Vec

# Build vector from elements
vec = Vec.from_elements([a, b, c, d], fx.Float32)

# Vector store to memref
vec.store(memref, [idx])

# Extract, bitcast, and convert
elem = vec[idx]
as_i32 = vec.bitcast(fx.Int32)
as_bf16 = vec.to(fx.BFloat16)
```

### 4.3 Buffer Operations (`fx.buffer_ops`)

AMD buffer load/store intrinsics for efficient global memory access:

```python
from flydsl.expr import buffer_ops

# Create buffer resource descriptor from memref
rsrc = buffer_ops.create_buffer_resource(memref_value)

# Buffer load (vectorized)
data = buffer_ops.buffer_load(rsrc, byte_offset, vec_width=4)

# Buffer store
buffer_ops.buffer_store(data, rsrc, byte_offset)
```

### 4.4 ROCm Intrinsics (`fx.rocdl`)

#### High-Level Helpers

```python
from flydsl.expr import rocdl

# Buffer tensor — wraps a Tensor with AMD buffer resource descriptor
A_buf = rocdl.make_buffer_tensor(A)

# MFMA MMA atom constructor — returns MmaAtomCDNA3_MFMAType
atom_type = rocdl.MFMA(m=16, n=16, k=32, elem_ty_ab=fx.Float8E4M3FNUZ)

# Buffer copy atom types
copy_op = rocdl.BufferCopy128b()   # 128-bit buffer copy
copy_op = rocdl.BufferCopy64b()    # 64-bit buffer copy
copy_op = rocdl.BufferCopy32b()    # 32-bit buffer copy
```

#### MFMA Instructions

Signature: `(result_type, [a, b, c, cbsz, abid, blgp])` — trailing ints default to 0.

```python
result = rocdl.mfma_f32_16x16x16f16(result_type, [a, b, acc])
result = rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, [a, b, acc])
result = rocdl.mfma_i32_16x16x32_i8(result_type, [a, b, acc])
result = rocdl.mfma_f32_16x16x16bf16_1k(result_type, [a, b, acc])   # BF16 1K variant

# GFX950 scaled MFMA (MXFP4/FP6/FP8)
result = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
    result_type, [a, b, acc, cbsz, blgp, opselA, scaleA, opselB, scaleB]
)
```

#### Instruction Scheduling Barriers

Control instruction scheduling for performance tuning:

```python
rocdl.sched_mfma(cnt)    # wait for cnt MFMA instructions to complete
rocdl.sched_vmem(cnt)    # wait for cnt VMEM reads to complete
rocdl.sched_dsrd(cnt)    # wait for cnt DS (LDS) reads to complete
rocdl.sched_dswr(cnt)    # wait for cnt DS (LDS) writes to complete
```

#### Math Intrinsics

Single-instruction hardware math (guaranteed 1 VALU cycle, lower precision than `math.*`):

```python
# Base-2 exponential (v_exp_f32)
result = rocdl.exp2(T.f32, x)

# Reciprocal (v_rcp_f32)
result = rocdl.rcp(T.f32, x)
```

#### Low-Level Ops

```python
# Warp shuffle
val = rocdl.ds_bpermute(idx, src)

# Buffer load/store (raw)
data = rocdl.raw_ptr_buffer_load(rsrc, offset, soffset, aux)
rocdl.raw_ptr_buffer_store(data, rsrc, offset, soffset, aux)
```

### 4.5 GPU Operations (`fx.gpu`)

```python
from flydsl.expr import gpu

# Barrier (workgroup synchronization)
gpu.barrier()

# Shared memory address space attribute
addrspace = gpu.smem_space()
addrspace_int = gpu.smem_space(int=True)
```

---

## 5. Control Flow

### 5.1 Python Loops

The `ASTRewriter` automatically transforms Python `for` loops:

```python
@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int]):
    # Compile-time unrolled loop
    for i in range_constexpr(N):
        # This loop is fully unrolled in the generated IR
        ...

    # Runtime loop (lowered by the AST rewriter)
    for i in range(runtime_value):
        ...
```

### 5.2 `const_expr()`

Mark a value as compile-time constant:

```python
from flydsl.expr import const_expr

@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int]):
    tile_size = const_expr(N // 4)
    for i in range_constexpr(tile_size):
        ...
```

---

## 6. Shared Memory (LDS)

### 6.1 `fx.SharedAllocator` + `@fx.struct`

New kernels declare their LDS layout as a `@fx.struct` storage type and
allocate it with `fx.SharedAllocator` (from `flydsl.expr.gpu`, reached as
`fx.SharedAllocator`) inside the kernel body. Each field is an `fx.Array[elem,
count, align]`; `.allocate(StorageStruct).peek()` returns a handle whose fields
expose typed views:

```python
import flydsl.expr as fx

# Declare the LDS layout. Fields may be conditional on compile-time config.
@fx.struct
class SharedStorage:
    s_red: fx.Array[fx.Float32, red_slots, 16]   # red_slots elems, 16B aligned
    s_red2: fx.Array[fx.Float32, red_slots, 16]

@flyc.kernel
def my_kernel(...):
    # Allocate the storage struct in LDS (inside the @kernel body).
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()

    # Get a logical view over each field and use it with the layout API.
    s_red = lds.s_red.view(fx.make_layout(red_slots, 1))
    s_red2 = lds.s_red2.view(fx.make_layout(red_slots, 1))
```

By default `SharedAllocator` is `static=True`: each leaf emits a per-leaf static
LDS global that the compiler sizes, so `launch(smem=...)` is left unset. Use
`static=False` (dynamic) mode to have the launch wrapper auto-infer `smem` from
`SharedAllocator.allocated_bytes` when `smem=None` (an explicit `smem` must be
`>=` that size). See `kernels/gemm/preshuffle_gemm.py` and
`kernels/norm/rmsnorm_kernel.py` for real usage.

### 6.2 Legacy `SmemAllocator`

The older `SmemAllocator` / `SmemPtr` path
(`python/flydsl/utils/smem_allocator.py`) remains for un-migrated kernels: it
tracks byte offsets manually (`_align` / `finalize` / `get_base`) and its
`finalize()` must be called inside the `gpu.module` body. Prefer
`fx.SharedAllocator` for new kernels.

### 6.3 LDS Capacity

| Architecture | LDS per CU |
|---|---|
| `gfx942` (MI300X) | 64 KB |
| `gfx950` (MI350/MI355X) | 160 KB |
| `gfx1201` (Radeon AI PRO R9700) | 64 KB |
| `gfx1250` | 320 KB |

---

## 7. Launch Configuration

### 7.1 `KernelLauncher.launch()`

```python
@flyc.jit
def launch(data: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    my_kernel(data).launch(
        grid=(num_blocks_x, num_blocks_y, num_blocks_z),
        block=(threads_x, threads_y, threads_z),
        smem=shared_mem_bytes,     # dynamic shared memory
        stream=stream,             # CUDA/HIP stream
    )
```

Grid and block dimensions accept:
- `int` — static value
- `ir.Value` — dynamic MLIR value
- Tuple of 1–3 values — missing dimensions default to 1

### 7.2 Dynamic Grid/Block Dimensions

```python
@flyc.jit
def launch(data: fx.Tensor, M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    grid_x = M // 256
    my_kernel(data, M).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )
```

---

## 8. Synchronization

```python
from flydsl.expr import gpu

# Workgroup barrier (s_barrier)
gpu.barrier()
```

---

## 9. Compilation & Caching

### 9.1 Automatic Caching

JIT-compiled functions are cached automatically:

- **In-memory cache** — keyed by argument type signature
- **Disk cache** — stored in `~/.flydsl/cache/` (configurable via `FLYDSL_RUNTIME_CACHE_DIR`)
- **Cache key** includes: source code hash, dependency sources, closure values, FlyDSL version, LLVM version

### 9.2 Cache Invalidation

Cache is invalidated when:
- Source code of the function or its dependencies changes
- Argument types change (different tensor shapes/dtypes)
- `Constexpr` values change
- FlyDSL or LLVM version changes

### 9.3 Disk Cache Invalidation

The JIT disk cache auto-invalidates when kernel source code or closure values change. Set `FLYDSL_RUNTIME_ENABLE_CACHE=0` only when modifying C++ passes or non-closure helper functions:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_script.py  # or: rm -rf ~/.flydsl/cache
```

### 9.4 Compile-Only Mode

```bash
COMPILE_ONLY=1 python my_script.py
```

---

## 10. Debugging

### 10.1 Dumping IR

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./my_dumps python my_script.py
```

### 10.2 Printing IR

```python
# After compilation, access IR from the compiled function:
result = launch(A, B, C, 1024)

# Or use JITCFunction directly:
compiled_func.print_ir()              # compiled MLIR IR
compiled_func.print_ir(compiled=False) # original IR before passes
```

### 10.3 AST Diff

```bash
FLYDSL_DEBUG_AST_DIFF=1 python my_script.py
```

Shows the diff between original and rewritten AST for debugging control flow transformations.

---

## 11. Complete Example: Preshuffle GEMM

From `kernels/gemm/preshuffle_gemm.py`:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

def compile_preshuffle_gemm(*, N, K, tile_m, tile_n, tile_k,
                             in_dtype="fp8", out_dtype="bf16",
                             epilogue="none", lds_stage=2, ...):
    a_lds_elems = tile_m * tile_k

    # Declare the LDS layout as a storage struct.
    @fx.struct
    class SharedStorage:
        a0: fx.Array[layout_elem, a_lds_elems, 16]
        if lds_stage == 2:
            a1: fx.Array[layout_elem, a_lds_elems, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor, arg_scale_b: fx.Tensor, arg_bias: fx.Tensor,
        i32_m: fx.Int32, i32_n: fx.Int32,
        tiled_mma_arg: fx.TiledMma, tiled_copy_g2s: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        gA = fx.rocdl.make_buffer_tensor(arg_a, ...)
        gB = fx.rocdl.make_buffer_tensor(arg_b)
        gC = fx.rocdl.make_buffer_tensor(arg_c, ...)

        # Allocate LDS and take typed views over the storage fields.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # ... GEMM implementation using MFMA, LDS, tiling ...

    @flyc.jit
    def launch_fn(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor, arg_scale_b: fx.Tensor, arg_bias: fx.Tensor,
        M_val: fx.Int32, N_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias,
                    M_val, N_val, tiled_mma, tiled_copy_g2s).launch(
            grid=(grid_x, grid_y), block=(256,), stream=stream,
        )

    return launch_fn
```

`M` is a runtime argument (`i32_m` / `M_val`), not a compile-time parameter, so
one compiled kernel serves any `M`. Output post-processing is selected with
`epilogue=` (`"none"`, `"bias"`, `"bias_relu"`, `"bias_silu"`, `"bias_gelu"`).

---

## 12. Decision Tree

```
Writing a new kernel?
│
├── Simple element-wise?
│   ├── Use @flyc.kernel + @flyc.jit
│   ├── fx.gpu.thread_idx.x for thread indexing
│   └── See tests/kernels/test_vec_add.py
│
├── Reduction (norm, softmax)?
│   ├── Reductions are inline (wave/block reduce helpers within the kernel)
│   └── See kernels/norm/rmsnorm_kernel.py, kernels/norm/softmax_kernel.py
│
├── Matrix multiply (GEMM)?
│   ├── Use @flyc.kernel + fx.SharedAllocator + MFMA
│   ├── B-preshuffle layout from kernels/mma/mfma_preshuffle_pipeline.py
│   └── See kernels/gemm/preshuffle_gemm.py
│
├── Need shared memory?
│   ├── Declare a @fx.struct storage layout
│   ├── Allocate with fx.SharedAllocator().allocate(Storage).peek()
│   └── Take .view(...) over each field inside @kernel
│
└── Need compile-time specialization?
    ├── Use Constexpr[T] parameters
    └── Use range_constexpr() for unrolled loops
```

---

## 13. Source Files

| File | Description |
|---|---|
| `python/flydsl/compiler/__init__.py` | Public API: `jit`, `kernel`, `from_dlpack` |
| `python/flydsl/compiler/jit_function.py` | `@jit` decorator, `MlirCompiler`, `JitCacheManager` |
| `python/flydsl/compiler/kernel_function.py` | `@kernel` decorator, `KernelFunction`, `KernelLauncher` |
| `python/flydsl/compiler/jit_executor.py` | `JITCFunction` (ExecutionEngine wrapper) |
| `python/flydsl/compiler/jit_argument.py` | `JitArgumentRegistry`, `TensorAdaptor` |
| `python/flydsl/compiler/ast_rewriter.py` | `ASTRewriter` — Python AST → MLIR control flow |
| `python/flydsl/expr/typing.py` | `Types` (`T`), `Tensor`, `Stream`, `Constexpr` |
| `python/flydsl/expr/arith.py` | Arithmetic operations |
| `python/flydsl/expr/vector.py` | Vector dialect operations |
| `python/flydsl/expr/gpu.py` | GPU operations (thread_id, barrier, ...) |
| `python/flydsl/expr/buffer_ops.py` | AMD buffer load/store operations |
| `python/flydsl/expr/rocdl/` | ROCm dialect intrinsics (MFMA/WMMA, buffer, TDM, cluster) |
| `python/flydsl/expr/primitive.py` | Layout algebra primitives (make_shape, crd2idx, etc.) |
| `python/flydsl/expr/gpu.py` | `SharedAllocator`, GPU ops (thread_id, barrier, ...) |
| `python/flydsl/utils/smem_allocator.py` | Legacy `SmemAllocator` / `SmemPtr` LDS management |
| `kernels/gemm/preshuffle_gemm.py` | Preshuffle GEMM kernel example |
| `tests/kernels/test_vec_add.py` | Vector add kernel test |
| `tests/kernels/test_preshuffle_gemm.py` | Preshuffle GEMM test |
