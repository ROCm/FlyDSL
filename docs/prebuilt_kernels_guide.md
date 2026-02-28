# Pre-built Kernel Library Guide

> Available FlyDSL kernels: Normalization, Softmax, GEMM — configuration, data types, pipelines, and shared utilities.

## Quick Reference

| Kernel | Builder Function | API Style | Dtypes | Key Feature |
|---|---|---|---|---|
| **LayerNorm** | `build_layernorm_module(M, N, dtype)` | Legacy (`MlirModule`) | f32, f16, bf16 | Two-pass vectorized normalization |
| **RMSNorm** | `build_rmsnorm_module(M, N, dtype)` | Legacy (`MlirModule`) | f32, f16, bf16 | LDS-cached 3-pass pipeline |
| **Softmax** | `build_softmax_module(M, N, dtype)` | Legacy (`MlirModule`) | f32, f16, bf16 | Online softmax, adaptive block size |
| **GEMM** | `compile_preshuffle_gemm_a8(...)` | New (`@flyc.kernel`) | fp8, int8, int4, fp16, bf16, fp4 | Preshuffle B, ping-pong LDS, MFMA 16x16 |

> **Note on API styles**: The normalization and softmax kernels use the legacy `MlirModule`-based API from `flydsl_` (imported via `from flydsl.dialects.ext import flir`). The GEMM kernel uses the new `@flyc.kernel`/`@flyc.jit` API from `python/flydsl/`. Both styles are fully functional.

---

## 1. Normalization Kernels

### 1.1 LayerNorm (`kernels/layernorm_kernel.py`)

Computes `LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta` for each row.

**Builder:**
```python
from kernels.layernorm_kernel import build_layernorm_module

executor = build_layernorm_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration Constants:**
| Constant | Value | Description |
|---|---|---|
| `BLOCK_THREADS` | 256 | Threads per block |
| `WARP_SIZE` | 64 | AMD wavefront size |
| `VEC_WIDTH` | 8 | Vector load/store width |
| `VEC_ALIGN` | 16 | Alignment for vector ops (bytes) |
| `EPS` | 1e-5 | Numerical stability epsilon |
| `USE_NONTEMPORAL` | True | Non-temporal stores for output |

**Algorithm:**
- **Two-pass normalization**: Pass 1 computes mean and variance, Pass 2 applies affine transform
- **Fast path**: When `N == BLOCK_THREADS * VEC_WIDTH * 4` (e.g., N=8192), uses fully register-resident computation with no scalar tail
- **Generic path**: Handles arbitrary N with vector body + scalar tail
- **bf16 handling**: Software round-to-nearest-even (RNE) pack on gfx942; hardware `cvt_pk_bf16_f32` on gfx950+
- **Warp reduction**: XOR-shuffle-based intra-wave reduction (shifts: 32, 16, 8, 4, 2, 1), then LDS-based cross-wave synchronization

**Kernel signature** (inside the `_LayerNorm` MlirModule):
```
GPU_MODULE_NAME = "layernorm_module"

@kernel
layernorm_kernel(self, Input, Gamma, Beta, Output, m_in)

@jit
__call__(self, Input, Gamma, Beta, Output, m_in)
```

### 1.2 RMSNorm (`kernels/rmsnorm_kernel.py`)

Computes `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma`.

**Builder:**
```python
from kernels.rmsnorm_kernel import build_rmsnorm_module

executor = build_rmsnorm_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration Constants:** Same as LayerNorm (BLOCK_THREADS=256, VEC_WIDTH=8, etc.)

**Algorithm (3-pass with LDS caching):**
1. **Pass 0**: Global → LDS row cache (one-pass global read, vectorized)
2. **Pass 1**: Sum-of-squares computation from LDS row cache
3. **Pass 2**: Normalize + gamma multiply + store with software pipeline for Gamma prefetch

**Kernel signature:**
```
GPU_MODULE_NAME = "rmsnorm_module"

@kernel
rmsnorm_kernel(self, Input, Gamma, Output, m_in)
```

---

## 2. Softmax Kernel

### 2.1 Softmax (`kernels/softmax_kernel.py`)

Computes row-wise softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.

**Builder:**
```python
from kernels.softmax_kernel import build_softmax_module

executor = build_softmax_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration:**
| Parameter | Value | Description |
|---|---|---|
| `BLOCK_SIZE` | `min(256, next_power_of_2(N))`, min 32 | Adaptive block size |
| `VEC_WIDTH` | 8 | Vector load/store width |
| `WARP_SIZE` | 64 | AMD wavefront size |

**Algorithm (6 stages):**
1. **Load Data**: Vectorized global loads into register buffer with validity masks
2. **Local Max**: Per-thread vector reduction (`maxnumf`)
3. **Global Max**: Block-wide shuffle reduction (intra-wave XOR → wave0 finalize via LDS)
4. **Local Exp + Sum**: `exp2(x * log2(e))` approximation, accumulate partial sums
5. **Global Sum**: Block-wide reduction for sum
6. **Normalize + Store**: Divide by sum, convert to output dtype, vectorized store

**Kernel signature:**
```
GPU_MODULE_NAME = f"softmax_{dtype_str}"

@kernel
softmax_kernel(self, A, C, m_in)
```

---

## 3. GEMM Kernel

### 3.1 Preshuffle GEMM (`kernels/preshuffle_gemm_flyc.py`)

MFMA 16x16-based GEMM with B-matrix preshuffle layout: `C[M,N] = A[M,K] @ B[N,K]^T`.

Uses the new `@flyc.kernel` / `@flyc.jit` API.

**Builder:**
```python
from kernels.preshuffle_gemm_flyc import compile_preshuffle_gemm_a8

launch_fn = compile_preshuffle_gemm_a8(
    M=16, N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",
    lds_stage=2,
    use_cshuffle_epilog=False,
)
```

Returns a `@flyc.jit`-decorated function that auto-compiles on first call.

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `M, N, K` | int | GEMM dimensions: A[M,K], B[N,K], C[M,N]. M and N can be 0 (dynamic). |
| `tile_m, tile_n, tile_k` | int | Block tile sizes |
| `in_dtype` | str | `"fp8"`, `"int8"`, `"int4"`, `"fp16"`, `"bf16"`, `"fp4"` |
| `lds_stage` | int | `2` = ping-pong LDS (tuned), `1` = single LDS buffer |
| `use_cshuffle_epilog` | bool | CK-style LDS CShuffle epilogue |
| `waves_per_eu` | int | Occupancy hint (None = default, 1-4 = limit occupancy) |
| `use_async_copy` | bool | Use async DMA for A tile global-to-LDS transfer |

**Key constraints:**
- `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step)
- INT4 is W4A8: A is int8, B is packed int4 (2 values/byte), unpacked to int8 in-kernel

**Pipeline details:**
- **lds_stage=2 (ping-pong)**: Two LDS buffers for A tiles. Cross-tile A0 prefetch overlaps VMEM with LDS reads
- **lds_stage=1 (single)**: CK-style intrawave schedule with single LDS buffer
- **K64-byte micro-step**: Each step issues 2x K32 MFMA operations
- **XOR16 swizzle**: Byte-level swizzle on LDS to avoid bank conflicts
- **B-preshuffle**: Shape (N0, K0, KLane, NLane, KPackBytes) = (N/16, K/64, 4, 16, kpack_bytes)
- **CShuffle epilogue**: Write C tile to LDS in row-major, remap threads for half2 packing via `ds_bpermute`

**Launch function signature:**
```python
launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M_val, N_val, stream)
```

Where:
- `arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b`: PyTorch tensors (auto-converted to memref)
- `M_val, N_val`: Python int (auto-converted to Int32)
- `stream`: `fx.Stream` (default stream if omitted)

---

## 4. Shared Utilities

### 4.1 Reduction Helpers (`kernels/reduce.py`)

Reusable warp and block reduction functions (used by normalization and softmax kernels).

| Function | Description |
|---|---|
| `reduce_vec_max(vec, VEC_WIDTH, ...)` | Vector reduction to max via `maxnumf` |
| `reduce_vec_sum(vec, VEC_WIDTH, ...)` | Vector reduction to sum via `add` |
| `make_block_reduce(tid, BLOCK_SIZE, ...)` | Block-wide reduction: intra-wave XOR shuffle → LDS cross-wave sync |
| `make_block_reduce_add(tid, ...)` | Block reduction for addition (single-wave fast path) |
| `make_block_reduce_add2(tid, ...)` | Dual independent scalar reduction |

**Reduction pattern:**
1. Intra-wave: XOR shuffle with shifts 32, 16, 8, 4, 2, 1 (wave64)
2. Lane 0 writes per-wave partial to LDS
3. Barrier
4. Wave 0 reduces `NUM_WAVES` partials from LDS

### 4.2 MFMA Epilogues (`kernels/mfma_epilogues.py`)

Configurable epilogue strategies for MFMA 16x16 kernels.

| Function | Description |
|---|---|
| `default_epilog(...)` | Standard row-iterator: `row = bx_m + mi*16 + lane_div_16*4 + ii` |
| `c_shuffle_epilog(...)` | CK-style LDS CShuffle: write to LDS → barrier → remap threads → half2 store |
| `mfma_epilog(use_cshuffle, ...)` | Dispatcher: calls default or CShuffle based on flag |

### 4.3 Preshuffle Pipeline (`kernels/mfma_preshuffle_pipeline.py`)

Shared data movement and layout utilities for preshuffle GEMM kernels.

| Function | Description |
|---|---|
| `make_preshuffle_b_layout(...)` | Build B-preshuffle layout: (N/16, K/64, 4, 16, kpack_bytes) |
| `load_b_pack_k32(...)` | Load B pack for K32 MFMA micro-step (returns i64) |
| `tile_chunk_coord_i32(...)` | Map (thread, chunk) → (row, col) for tile loads |
| `buffer_copy_gmem16_dwordx4(...)` | 16-byte global load via buffer-load dwordx4 |
| `lds_store_16b_xor16(...)` | Store 16B to LDS with XOR16 swizzle |
| `lds_load_pack_k32(...)` | Load A-pack from LDS for K32 micro-step |
| `swizzle_xor16(...)` | XOR-based swizzle for LDS bank-conflict avoidance |

### 4.4 Layout Utilities (`kernels/layout_utils.py`)

Pure-arith layout helpers for static-stride layouts:

| Function | Description |
|---|---|
| `crd2idx(crd, layout)` | Coordinate → flat index using pure arith ops |
| `idx2crd(idx, layout)` | Flat index → coordinate list using pure arith ops |
| `get(int_tuple, mode)` | Extract element at index from Python list/tuple |

---

## 5. Kernel API Comparison

### Legacy API (Normalization / Softmax)

Used by `layernorm_kernel.py`, `rmsnorm_kernel.py`, `softmax_kernel.py`:

```python
from flydsl.dialects.ext import flir, arith
from flydsl.utils import SmemAllocator

class _MyKernel(flir.MlirModule):
    GPU_MODULE_NAME = "my_kernel"

    def init_gpu_module(self):
        allocator.finalize()

    @flir.kernel
    def kernel_func(self, Input, Output, m_in):
        tid = flir.thread_idx("x")
        # ... uses flir.*, arith.* directly ...

    @flir.jit
    def __call__(self, Input, Output, m_in):
        # Host launcher
        ...
```

### New API (GEMM)

Used by `preshuffle_gemm_flyc.py`:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, buffer_ops, rocdl

@flyc.kernel
def gemm_kernel(arg_c: fx.Tensor, arg_a: fx.Tensor, ...):
    tid = gpu.thread_idx.x
    # ... uses fx.*, arith.*, buffer_ops.*, rocdl.* ...

@flyc.jit
def launch_fn(arg_c: fx.Tensor, ..., stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(arg_c, ...).launch(grid=..., block=..., stream=stream)
```

---

## 6. Kernel Decision Tree

```
What operation do you need?
│
├── Normalization
│   ├── Need bias (beta) term? → LayerNorm (layernorm_kernel.py)
│   └── No bias term?         → RMSNorm (rmsnorm_kernel.py)
│
├── Softmax
│   └── Row-wise softmax      → Softmax (softmax_kernel.py)
│
├── Matrix Multiply (GEMM)
│   ├── Standard GEMM (uniform precision)
│   │   ├── FP8 / INT8 / INT4(W4A8) / FP16 / BF16 / FP4
│   │   └── → compile_preshuffle_gemm_a8()
│   │
│   └── Uses new @flyc.kernel API
│       └── See kernels/preshuffle_gemm_flyc.py
│
└── Building blocks
    ├── Warp/block reduction     → reduce.py
    ├── MFMA epilogue selection  → mfma_epilogues.py
    ├── Preshuffle data movement → mfma_preshuffle_pipeline.py
    └── Static layout helpers    → layout_utils.py
```

---

## 7. Source Files

| File | Description |
|---|---|
| `kernels/__init__.py` | Package marker |
| `kernels/layernorm_kernel.py` | LayerNorm builder (`_LayerNorm` MlirModule, legacy API) |
| `kernels/rmsnorm_kernel.py` | RMSNorm builder (`_RMSNorm` MlirModule, legacy API) |
| `kernels/softmax_kernel.py` | Softmax builder (`_Softmax` MlirModule, legacy API) |
| `kernels/preshuffle_gemm_flyc.py` | Preshuffle GEMM builder (new `@flyc.kernel` API) |
| `kernels/reduce.py` | Shared warp/block reduction helpers |
| `kernels/mfma_epilogues.py` | MFMA epilogue strategies (default, CShuffle) |
| `kernels/mfma_preshuffle_pipeline.py` | Preshuffle data movement and layout utilities |
| `kernels/layout_utils.py` | Pure-arith layout helpers (`crd2idx`, `idx2crd`) |

## 8. Test Files

| File | Tests |
|---|---|
| `tests/kernels/test_layernorm.py` | LayerNorm correctness + perf |
| `tests/kernels/test_rmsnorm.py` | RMSNorm correctness + perf |
| `tests/kernels/test_softmax.py` | Softmax correctness + perf |
| `tests/kernels/test_preshuffle_gemm.py` | GEMM fp8/int8/int4/bf16/fp4 correctness + perf |
| `tests/kernels/benchmark_common.py` | Shared benchmark infrastructure |
