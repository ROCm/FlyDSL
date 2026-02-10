# Kernel Authoring Guide

> Writing GPU kernels with FlyDSL: MlirModule, thread hierarchy, TensorView, tiled copies, MFMA, shared memory, and synchronization.

## Quick Reference

| Concept | API | Description |
|---|---|---|
| **Module** | `class MyKernel(MlirModule)` | Base class for kernel modules |
| **Kernel** | `@kernel` decorator | Emit `gpu.func` with `gpu.kernel` attribute |
| **Host func** | `@jit` decorator | Emit host-side `func.func` |
| **Thread ID** | `flir.thread_idx("x")` | Get thread index in workgroup |
| **Block ID** | `flir.block_idx("x")` | Get block/workgroup index |
| **Block dim** | `flir.block_dim("x")` | Get block dimension size |
| **Tensor** | `flir.make_tensor(ptr, shape, strides)` | Create a TensorView |
| **Fragment** | `flir.make_fragment_like(template)` | Register memory buffer |
| **Copy atom** | `flir.make_copy_atom(dtype, vec_size)` | Copy descriptor |
| **Tiled copy** | `flir.make_tiled_copy_tv(atom, thr, val)` | Distributed copy |
| **Copy** | `flir.copy(tiled, src, dst)` | Execute data movement |
| **LDS** | `SmemAllocator` | Shared memory management |
| **Barrier** | `gpu.barrier()` | Workgroup synchronization |
| **Compile** | `compile(module)` | Full pipeline → executor |

---

## 1. Module Structure

### 1.1 Using `MlirModule` (Structured Pattern)

```python
from flydsl.lang.ir.module import MlirModule, kernel, jit
from flydsl.dialects.ext import flir, arith, gpu
import _mlir.extras.types as T

class VecAddKernel(MlirModule):
    GPU_MODULE_NAME = "vec_add"

    @kernel
    def vecAdd(self, A: T.memref(1024, T.f32()),
                     B: T.memref(1024, T.f32()),
                     C: T.memref(1024, T.f32())):
        tid = flir.thread_idx("x")
        bid = flir.block_idx("x")
        # ... kernel body ...

# Instantiate to emit MLIR
mod = VecAddKernel()
print(mod.module)  # view MLIR IR
```

### 1.2 Using `RAIIMLIRContextModule` (Standalone Pattern)

```python
from flydsl.compiler.context import RAIIMLIRContextModule
from flydsl.dialects.ext import gpu, flir
import _mlir.extras.types as T

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

@gpu.module("my_kernels", ["#rocdl.target<abi = \"500\">"])
def mod():
    pass

@flir.kernel
def my_kernel(A: T.memref(1024, T.f32())):
    tid = flir.thread_idx("x")
    # ... kernel body ...
```

### 1.3 `MlirModule` Class Attributes

| Attribute | Default | Description |
|---|---|---|
| `GPU_MODULE_NAME` | `"kernels"` | Name of the `gpu.module` container |
| `GPU_MODULE_TARGETS` | `None` | Target list (overridden by `flir.compile()`) |
| `ALLOW_UNREGISTERED_DIALECTS` | `True` | Allow unknown dialects in context |

### 1.4 `init_gpu_module()` Hook

Override this method in your `MlirModule` subclass to insert ops at the beginning of the `gpu.module` body (e.g., `memref.global` for shared memory):

```python
class MyKernel(MlirModule):
    GPU_MODULE_NAME = "my_kernel"

    def init_gpu_module(self):
        # Called before any @kernel methods
        self.smem.finalize()  # emit memref.global for LDS
```

---

## 2. Thread / Block Hierarchy

```python
# Thread index within workgroup (0 to blockDim-1)
tid_x = flir.thread_idx("x")  # returns index-typed Value
tid_y = flir.thread_idx("y")
tid_z = flir.thread_idx("z")

# Block (workgroup) index within grid
bid_x = flir.block_idx("x")
bid_y = flir.block_idx("y")

# Block dimensions
bdim_x = flir.block_dim("x")

# Linear thread index (common pattern)
tid_linear = (flir.thread_idx("y") * flir.block_dim("x")
              + flir.thread_idx("x")).value
```

---

## 3. TensorView

`TensorView` is the Python-side representation of a multi-dimensional memory region:

```python
# Create from memref pointer with shape and strides
tensor_A = flir.make_tensor(A, shape=(M, N), strides=(N, 1))

# Create 1D view
tensor_flat = flir.make_tensor(ptr, shape=(N,), strides=(1,))
```

Key attributes:
- `shape` -- tuple of dimension extents
- `strides` -- memory strides per dimension
- `base_indices` -- base offsets into the backing memref
- `element_type` -- MLIR element type
- `memref` -- backing MLIR memref value

You can also construct a `TensorView` directly:

```python
view = flir.TensorView(
    memref_value,
    shape=(M, N),
    strides=(N, 1),
    base_indices=(offset_m, offset_n),
    element_type=T.f16(),
)
```

---

## 4. Tiled Copies

Tiled copies distribute data movement across threads in a workgroup.

### 4.1 Copy Atom

A `CopyAtom` describes a single thread's copy operation:

```python
# Copy atom: element type, vector width, coalesced flag
copy_atom = flir.make_copy_atom(T.f32(), vector_size=8)
copy_atom = flir.make_copy_atom(T.f16(), vector_size=16, is_coalesced=True)
```

### 4.2 Tiled Copy

A `TiledCopy` distributes a copy atom across threads using thread and value layouts:

```python
THREADS = 256
TILE = 8
VEC = 4

# Thread layout: how threads are arranged
thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))

# Value layout: how many elements each thread handles
val_layout = flir.make_ordered_layout((TILE,), order=(0,))

# Create tiled copy
tiled = flir.make_tiled_copy_tv(
    copy_atom, thr_layout, val_layout,
    thr_shape=(THREADS,), val_shape=(TILE,),
)
```

### 4.3 Thread Slicing and Partitioning

```python
# Get this thread's slice of the copy
thr_copy = tiled.get_slice(tid_linear)

# Partition source and destination tensors for this thread
thr_src = thr_copy.partition_S(src_tensor)
thr_dst = thr_copy.partition_D(dst_tensor)
```

### 4.4 Copy Execution

```python
# Register fragment (destination buffer)
frag = flir.make_fragment_like(thr_src, T.f32())

# Execute copy: src → frag
flir.copy(tiled, thr_src, frag)

# Copy with predication (bounds checking)
flir.copy(tiled, thr_src, frag, pred=mask)

# Copy with LDS swizzle
flir.copy(atom, src, dst,
    dst_swizzle_xor16_kblocks=k_blocks,
    dst_swizzle_xor16_dims=(0, 1))

# Copy returning vector (for buffer loads)
vec = flir.copy(atom, src, None,
    return_vector=True,
    src_buffer_resource=rsrc,
    alignment=16)
```

---

## 5. Register Fragments

Fragments represent per-thread register storage:

```python
# Create fragment matching a TensorView's shape
frag = flir.make_fragment_like(tensor_view, T.f32())

# Create fragment matching a TiledCopy
frag = flir.make_fragment_like(tiled_copy)

# Create explicit register tensor
rmem = flir.make_rmem_tensor((16, 16), T.f32())

# Create identity/coordinate tensor
identity = flir.make_identity_tensor((M, N))
```

---

## 6. MFMA Operations

Matrix Fused Multiply-Add instructions for AMD GPUs:

### MLIR ROCm Dialect

```mlir
// Single MFMA instruction
%d = flir.rocm.mfma %a, %b, %c {
    shape = [32, 32, 8], arch = "gfx942"
} : (!flir.tensor<f16, ...>, !flir.tensor<f16, ...>,
     !flir.tensor<f32, ...>) -> !flir.tensor<f32, ...>

// Tiled MFMA (multi-wavefront)
%d = flir.rocm.tiled_mfma %a, %b, %c {
    atom = #flir_rocm.mfma_atom<[32, 32, 8], f16, f16, f32, gfx942>,
    tile_shape = [128, 128, 32]
} : (...)
```

### Supported MFMA Shapes (GFX942)

| Instruction | M | N | K | A Type | D Type |
|---|---|---|---|---|---|
| `mfma_f32_32x32x8_f16` | 32 | 32 | 8 | FP16 | FP32 |
| `mfma_f32_16x16x16_f16` | 16 | 16 | 16 | FP16 | FP32 |
| `mfma_f32_32x32x16_bf16` | 32 | 32 | 16 | BF16 | FP32 |
| `mfma_f64_16x16x4_f64` | 16 | 16 | 4 | FP64 | FP64 |
| `mfma_f32_16x16x32_f8` | 16 | 16 | 32 | FP8 | FP32 |
| `mfma_i32_16x16x32_i8` | 16 | 16 | 32 | INT8 | INT32 |

### In Python Kernel Code

MFMA is typically used through the `rocdl` intrinsics after lowering, but the FLIR ROCm ops provide a structured interface during IR construction. See the GEMM kernels in `kernels/preshuffle_gemm.py` for complete examples.

---

## 7. Shared Memory (LDS)

### 7.1 `SmemAllocator`

```python
from flydsl.utils.smem_allocator import SmemAllocator
import _mlir.extras.types as T

class MyGemmKernel(MlirModule):
    GPU_MODULE_NAME = "gemm"

    def __init__(self):
        self.smem = SmemAllocator(None, arch="gfx942")
        # Reserve LDS space before emitting kernels
        self.lds_a = self.smem.allocate_array(T.f16(), 8192)
        self.lds_b = self.smem.allocate_array(T.f16(), 8192)
        super().__init__()

    def init_gpu_module(self):
        # Emit memref.global for the LDS buffer
        self.smem.finalize()

    @kernel
    def gemm_kernel(self, ...):
        # Get LDS base pointer inside kernel
        lds_base = self.smem.get_base()
        # Get typed views
        lds_a_ptr = self.lds_a(lds_base)  # SmemPtr
        lds_b_ptr = self.lds_b(lds_base)  # SmemPtr
        # Load/store through SmemPtr
        val = lds_a_ptr.load([idx])
        lds_b_ptr.store(val, [idx])
```

### 7.2 `SmemAllocator` API

| Method | Description |
|---|---|
| `allocate(size_bytes)` | Allocate raw bytes |
| `allocate(mlir_type)` | Allocate one scalar of given type |
| `allocate_array(dtype, num_elems)` | Allocate contiguous array |
| `allocate_tensor(layout, element_type, swizzle=None)` | Allocate tensor with layout |
| `finalize()` | Emit `memref.global` (call in `init_gpu_module`) |
| `get_base()` | Get base pointer (call inside kernel) |

### 7.3 LDS Capacity

| Architecture | LDS per CU |
|---|---|
| `gfx942` (MI300X) | 64 KB |
| `gfx950` (MI350) | 160 KB |

The allocator checks capacity and raises `RuntimeError` on overflow.

### 7.4 `SmemPtr`

Returned by allocator generators, provides typed access to LDS regions:

```python
ptr = allocator_gen(lds_base)  # SmemPtr
memref_view = ptr.get()        # ir.Value (memref view)
val = ptr.load([idx])          # Load from LDS
ptr.store(val, [idx])          # Store to LDS
```

---

## 8. Synchronization

```python
from flydsl.dialects.ext import gpu

# Workgroup barrier (s_barrier)
gpu.barrier()
```

```mlir
// MLIR
gpu.barrier

// ROCm dialect
flir.rocm.barrier              // workgroup barrier
flir.rocm.wavefront_barrier    // wavefront-level barrier
```

---

## 9. Complete Example: VecAdd

From `tests/kernels/test_vec_add.py`:

```python
from flydsl.compiler.context import RAIIMLIRContextModule
from flydsl.compiler.compiler import compile
from flydsl.dialects.ext import gpu, flir
import _mlir.extras.types as T

N = 20480000
THREADS = 256
TILE = 8
VEC = 4

# Set up MLIR context
ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

@gpu.module("vec_kernels", ["#rocdl.target<abi = \"500\">"])
def mod():
    pass

@flir.kernel
def vecAdd(A: T.memref(N, T.f32()),
           B: T.memref(N, T.f32()),
           C: T.memref(N, T.f32())):
    # Linear thread index
    tid_linear = (flir.thread_idx("y") * flir.block_dim("x")
                  + flir.thread_idx("x")).value

    # Thread and value layouts
    thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
    val_layout = flir.make_ordered_layout((TILE,), order=(0,))

    # Copy atom: f32, 4-wide vector loads
    copy_atom = flir.make_copy_atom(T.f32(), vector_size=VEC)

    # Tiled copy across all threads
    tiled = flir.make_tiled_copy_tv(
        copy_atom, thr_layout, val_layout,
        thr_shape=(THREADS,), val_shape=(TILE,),
    )

    # Create tensors and partition into tiles
    tensor_A = flir.make_tensor(A, shape=(N,), strides=(1,))
    tiles_A = flir.zipped_divide(tensor_A, (THREADS * TILE,))
    blkA = tiles_A[(flir.block_idx("x"),)]

    # Get this thread's portion
    thrA = tiled.get_slice(tid_linear).partition_S(blkA)
    frgA = flir.make_fragment_like(thrA, T.f32())
    flir.copy(tiled, thrA, frgA)

    # ... repeat for B, add, store to C ...

# Compile
executor = compile(ctx.module)
```

---

## 10. Kernel Launching

After compilation, launch kernels through the executor:

```python
import torch

# Create device tensors
A = torch.randn(N, device="cuda", dtype=torch.float32)
B = torch.randn(N, device="cuda", dtype=torch.float32)
C = torch.empty(N, device="cuda", dtype=torch.float32)

# Launch
grid = (N // (THREADS * TILE), 1, 1)
block = (THREADS, 1, 1)

# For MlirModule-based kernels:
executor = compile(mod)
executor.vecAdd(A, B, C)

# For GPUFunc-based kernels:
vecAdd(A, B, C, grid_size=grid, block_size=block)
```

---

## 11. Decision Tree

```
Writing a new kernel?

├── Simple element-wise?
│   ├── Use @flir.kernel with TensorView + copy
│   └── See tests/kernels/test_vec_add.py
│
├── Reduction (norm, softmax)?
│   ├── Use warp_reduce / block_reduce from kernels/reduce.py
│   └── See kernels/layernorm_kernel.py, kernels/softmax_kernel.py
│
├── Matrix multiply (GEMM)?
│   ├── Use MlirModule + SmemAllocator + MFMA
│   ├── B-preshuffle layout from mfma_preshuffle_pipeline.py
│   └── See kernels/preshuffle_gemm.py
│
├── Need shared memory?
│   ├── Use SmemAllocator in MlirModule.__init__
│   ├── Call finalize() in init_gpu_module()
│   └── Call get_base() inside @kernel
│
└── Need custom pipeline?
    ├── Use Pipeline fluent API
    └── Or use flir.compile() with defaults
```

---

## 12. Source Files

| File | Description |
|---|---|
| `flydsl/src/flydsl/lang/ir/module.py` | `MlirModule`, `@kernel`, `@jit` decorators |
| `flydsl/src/flydsl/dialects/ext/flir.py` | Layout API, TensorView, CopyAtom, TiledCopy, `copy()` |
| `flydsl/src/flydsl/dialects/ext/gpu.py` | GPU dialect: `barrier()`, `LaunchFuncOp`, `GPUFuncOp` |
| `flydsl/src/flydsl/dialects/ext/rocm.py` | ROCm helpers: MfmaOp, Copy ops |
| `flydsl/src/flydsl/utils/smem_allocator.py` | `SmemAllocator`, `SmemPtr`, LDS capacity checks |
| `flydsl/src/flydsl/compiler/compiler.py` | `compile()` entry point |
| `flydsl/src/flydsl/compiler/context.py` | `RAIIMLIRContextModule` |
| `flir/include/flir/FlirRocmOps.td` | ROCm dialect ops (MFMA, LDS, copy, barriers) |
| `kernels/reduce.py` | Warp/block reduction primitives |
| `tests/kernels/test_vec_add.py` | VecAdd example kernel |

## 13. Test Files

| File | Description |
|---|---|
| `tests/kernels/test_vec_add.py` | Vector add kernel test |
| `tests/kernels/test_softmax.py` | Softmax kernel test |
| `tests/kernels/test_layernorm.py` | LayerNorm kernel test |
| `tests/kernels/test_preshuffle_gemm.py` | Preshuffle GEMM test |
| `tests/kernels/test_eltwise_add.py` | Element-wise add test |
| `tests/kernels/test_gpu_simple.py` | Simple GPU kernel tests |
| `tests/kernels/test_shared_working.py` | Shared memory tests |
