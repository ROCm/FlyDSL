# Kernel Authoring Guide

> Writing GPU kernels with FlyDSL: `@flyc.jit`, `@flyc.kernel`, expression API, launch configuration, shared memory, and synchronization.

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
| **Constants** | `arith.constant(val)` | Create MLIR constant value |
| **Range loop** | `range_constexpr(n)` | Compile-time unrolled loop |
| **Buffer load** | `buffer_ops.buffer_load(rsrc, off)` | AMD buffer load intrinsic |

---

## 1. Basic Kernel Pattern

### 1.1 `@flyc.kernel` + `@flyc.jit`

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

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
    # ... kernel body using arith/vector/buffer ops ...

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
   - `JitCompiledFunction` wraps the resulting ExecutionEngine
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

    def __ir_types__(self):
        return [...]  # MLIR types for this argument

    def __c_pointers__(self):
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

### 4.1 Arithmetic (`fx.arith`)

```python
from flydsl.expr import arith
from flydsl.expr.typing import T

# Constants
c42 = arith.constant(42, index=True)     # index type
c3_14 = arith.constant(3.14, T.f32)      # f32 type

# Arithmetic (operator overloading via ArithValue)
result = a + b
result = a * 2
result = a // 4
result = a % 16

# Cast
idx = arith.index_cast(T.index, int_val)

# Select
result = arith.select(cond, true_val, false_val)

# Bitwise
result = arith.andi(a, b)
result = arith.xori(a, b)
result = arith.shli(a, b)
```

### 4.2 Vector Operations (`fx.vector`)

```python
from flydsl.expr import vector

# Build vector from elements
vec = vector.from_elements(vec_type, [a, b, c, d])

# Vector store to memref
vector.store(vec, memref, [idx])

# Extract/insert
elem = vector.extractelement(vec, idx)
vec2 = vector.insertelement(vec, elem, idx)
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

```python
from flydsl.expr import rocdl

# MFMA instructions
result = rocdl.mfma_f32_16x16x16_f16(a, b, acc)
result = rocdl.mfma_f32_16x16x32_fp8(a, b, acc)
result = rocdl.mfma_i32_16x16x32i8(a, b, acc)

# Warp shuffle
val = rocdl.ds_bpermute(idx, src)

# LDS operations
rocdl.ds_write_b128(lds_ptr, offset, data)
data = rocdl.ds_read_b128(lds_ptr, offset)
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

### 5.1 Python Loops → MLIR SCF

The `ASTRewriter` automatically transforms Python `for` loops:

```python
@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int]):
    # Compile-time unrolled loop
    for i in range_constexpr(N):
        # This loop is fully unrolled in the generated IR
        ...

    # Runtime loop (lowered to scf.for)
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

### 6.1 `SmemAllocator`

```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.expr.typing import T

# Create allocator for target architecture
allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")

# Allocate typed arrays
lds_a = allocator.allocate_array(T.f16, 8192)
lds_b = allocator.allocate_array(T.f16, 8192)

# Inside kernel: get base pointer and typed views
lds_base = allocator.get_base()
lds_a_ptr = lds_a(lds_base)  # SmemPtr
lds_b_ptr = lds_b(lds_base)  # SmemPtr

# Load/store through SmemPtr
val = lds_a_ptr.load([idx])
lds_b_ptr.store(val, [idx])
```

### 6.2 Finalizing LDS Allocation

For `@flyc.kernel` style kernels, emit `memref.global` in the GPU module:

```python
comp_ctx = CompilationContext.get_current()
with ir.InsertionPoint(comp_ctx.gpu_module_body):
    allocator.finalize()
```

### 6.3 LDS Capacity

| Architecture | LDS per CU |
|---|---|
| `gfx942` (MI300X) | 64 KB |
| `gfx950` (MI350) | 160 KB |

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

### 9.3 Disabling Cache

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_script.py
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

# Or use JitCompiledFunction directly:
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

From `kernels/preshuffle_gemm_flyc.py`:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

def compile_preshuffle_gemm_a8(*, M, N, K, tile_m, tile_n, tile_k,
                                 in_dtype="fp8", lds_stage=2, ...):
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    lds_a = allocator.allocate_array(T.i8, tile_m * tile_k)
    # ... more allocations ...

    @flyc.kernel
    def gemm_kernel(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor, arg_scale_b: fx.Tensor,
        m_in: fx.Int32, n_in: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        # ... complex GEMM implementation using MFMA, LDS, tiling ...

    @flyc.jit
    def launch_fn(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor, arg_scale_b: fx.Tensor,
        M_val: fx.Int32, N_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b,
                    M_val, N_val).launch(
            grid=(grid_x, grid_y), block=(256,),
            smem=smem_bytes, stream=stream,
        )

    return launch_fn
```

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
│   ├── Use warp_reduce / block_reduce from kernels/reduce.py
│   └── See kernels/layernorm_kernel.py, kernels/softmax_kernel.py
│
├── Matrix multiply (GEMM)?
│   ├── Use @flyc.kernel + SmemAllocator + MFMA
│   ├── B-preshuffle layout from mfma_preshuffle_pipeline.py
│   └── See kernels/preshuffle_gemm_flyc.py
│
├── Need shared memory?
│   ├── Use SmemAllocator with target arch
│   ├── Call finalize() in GPU module body
│   └── Call get_base() inside @kernel
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
| `python/flydsl/compiler/jit_executor.py` | `JitCompiledFunction` (ExecutionEngine wrapper) |
| `python/flydsl/compiler/jit_argument.py` | `JitArgumentRegistry`, `TensorAdaptor` |
| `python/flydsl/compiler/ast_rewriter.py` | `ASTRewriter` — Python AST → MLIR control flow |
| `python/flydsl/expr/typing.py` | `Types` (`T`), `Tensor`, `Stream`, `Constexpr` |
| `python/flydsl/expr/arith.py` | Arithmetic operations |
| `python/flydsl/expr/vector.py` | Vector dialect operations |
| `python/flydsl/expr/gpu.py` | GPU operations (thread_id, barrier, ...) |
| `python/flydsl/expr/buffer_ops.py` | AMD buffer load/store operations |
| `python/flydsl/expr/rocdl.py` | ROCm dialect intrinsics |
| `python/flydsl/expr/primitive.py` | Layout algebra primitives (make_shape, crd2idx, etc.) |
| `python/flydsl/utils/smem_allocator.py` | `SmemAllocator`, `SmemPtr`, LDS management |
| `kernels/preshuffle_gemm_flyc.py` | Preshuffle GEMM kernel example |
| `kernels/reduce.py` | Warp/block reduction primitives |
| `tests/kernels/test_vec_add.py` | Vector add kernel test |
| `tests/kernels/test_preshuffle_gemm.py` | Preshuffle GEMM test |
