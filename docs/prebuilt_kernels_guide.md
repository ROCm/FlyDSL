# Pre-built Kernel Library Guide

> Available FlyDSL kernels: Normalization, Softmax, GEMM, Mixed-precision GEMM, MoE GEMM -- configuration, data types, pipelines, and shared utilities.

## Quick Reference

| Kernel | Builder Function | Dtypes | Key Feature |
|---|---|---|---|
| **LayerNorm** | `build_layernorm_module(M, N, dtype_str)` | f32, f16, bf16 | Two-pass vectorized normalization |
| **RMSNorm** | `build_rmsnorm_module(M, N, dtype_str)` | f32, f16, bf16 | LDS-cached 3-pass pipeline |
| **Softmax** | `build_softmax_module(M, N, dtype_str)` | f32, f16, bf16 | Online softmax, adaptive block size |
| **GEMM** | `compile_preshuffle_gemm_a8(...)` | fp8, int8, int4, fp16, bf16 | Preshuffle B, ping-pong LDS, MFMA 16x16 |
| **Mixed GEMM** | `compile_mxfp4_preshuffle_gemm(...)` | A:fp8/fp4, B:fp4 | MXFP4 mixed-precision with per-block scales |
| **MoE GEMM** | `compile_moe_gemm1(...)` | fp8, fp16, int8, int4 | Dual accumulator, SiLU gating, expert routing |
| **Mixed MoE GEMM** | `compile_mixed_moe_gemm1(...)` | A:fp8/fp4, B:fp4 | MXFP4 MoE with separate A/B scales |

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
GPU_MODULE_NAME = f"softmax_{dtype_str}"  # e.g., "softmax_bf16"

@kernel
softmax_kernel(self, A, C, m_in)
```

---

## 3. GEMM Kernels

### 3.1 Preshuffle GEMM (`kernels/preshuffle_gemm.py`)

MFMA 16x16-based GEMM with B-matrix preshuffle layout: `C[M,N] = A[M,K] @ B[N,K]^T`.

**Builder:**
```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

executor = compile_preshuffle_gemm_a8(
    M=16, N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",
    lds_stage=2,
    use_cshuffle_epilog=False,
)
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `M, N, K` | int | GEMM dimensions: A[M,K], B[N,K], C[M,N] |
| `tile_m, tile_n, tile_k` | int | Block tile sizes |
| `in_dtype` | str | `"fp8"`, `"int8"`, `"int4"`, `"fp16"`, `"bf16"` |
| `lds_stage` | int | `2` = ping-pong LDS (tuned), `1` = single LDS buffer |
| `use_cshuffle_epilog` | bool | CK-style LDS CShuffle epilogue |

**Key constraints:**
- `tile_m * tile_k * elem_bytes` must be divisible by `total_threads` (256)
- `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step)
- INT4 is W4A8: A is int8, B is packed int4 (2 values/byte), unpacked to int8 in-kernel

**Pipeline details:**
- **lds_stage=2 (ping-pong)**: Two LDS buffers for A tiles. Cross-tile A0 prefetch overlaps VMEM with LDS reads
- **lds_stage=1 (single)**: CK-style intrawave schedule with single LDS buffer
- **K64-byte micro-step**: Each step issues 2x K32 MFMA operations (fp8/int8: 64 elements, fp16/bf16: 32 elements)
- **XOR16 swizzle**: Byte-level swizzle on LDS to avoid bank conflicts
- **B-preshuffle**: Shape (N0, K0, KLane, NLane, KPackBytes) = (N/16, K/64, 4, 16, kpack_bytes)
- **CShuffle epilogue**: Write C tile to LDS in row-major, remap threads for half2 packing via `ds_bpermute`

### 3.2 Mixed-Precision GEMM (`kernels/mixed_preshuffle_gemm.py`)

MXFP4 mixed-precision GEMM with separate A/B quantization scales.

**Builder:**
```python
from kernels.mixed_preshuffle_gemm import compile_mxfp4_preshuffle_gemm

executor = compile_mxfp4_preshuffle_gemm(
    M=16, N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    a_dtype="fp8", b_dtype="fp4",
    lds_stage=2,
    use_cshuffle_epilog=True,
)
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `a_dtype` | str | `"fp8"` or `"fp4"` for A matrix |
| `b_dtype` | str | `"fp4"` for B matrix |
| `use_cshuffle_epilog` | bool | Default `True` for mixed GEMM |

**Key features:**
- K128 stepping with per-pack scale factors
- Packed FP4 element handling (2 elements per byte)
- Separate scale loading for A and B paths with `quant_block_size=32`
- Uses `mfma_scale_x128` on gfx950+ (9-operand scaling MFMA)

---

## 4. MoE GEMM Kernels

### 4.1 MoE GEMM 2-Stage (`kernels/moe_gemm_2stage.py`)

Mixture-of-Experts GEMM with dual accumulators for gate/up projections and SiLU gating.

**Stage 1 Builder:**
```python
from kernels.moe_gemm_2stage import compile_moe_gemm1

executor = compile_moe_gemm1(
    model_dim=8192, inter_dim=8192,
    experts=16, topk=4,
    tile_m=64, tile_n=128, tile_k=128,
    doweight_stage1=True,
    in_dtype="fp8", out_dtype="f16",
)
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `model_dim` | int | Model hidden dimension |
| `inter_dim` | int | Intermediate dimension |
| `experts` | int | Number of experts |
| `topk` | int | Top-K experts per token |
| `tile_m, tile_n, tile_k` | int | Block tile sizes |
| `doweight_stage1` | bool | Apply sorted weights in stage1 epilogue |
| `in_dtype` | str | `"fp8"`, `"fp16"`, `"int8"`, `"int4"` |
| `out_dtype` | str | `"f16"` or `"bf16"` |

**Key features:**
- **Dual accumulator**: `acc_gate` and `acc_up` for gate and up projections
- **SiLU activation**: `silu(x) = x * sigmoid(x)` using `exp2` + `rcp` for efficiency
- **Dynamic expert routing**: Expert ID lookup per M-tile
- **Sorted token masking**: Sentinel value for out-of-bounds token safety
- **Block validity gating**: Skip invalid M tiles early
- **CShuffle epilogue**: Optional, with sorted weight and scale blending

**Kernel signature:**
```
GPU_MODULE_NAME = f"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}_t{tile_m}x{tile_n}x{tile_k}_abi3"

@kernel
moe_gemm1(self, arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w,
          arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
          arg_max_token_ids, tokens_in, inter_in, k_in, size_expert_ids_in)
```

### 4.2 Mixed MoE GEMM (`kernels/mixed_moe_gemm_2stage.py`)

MXFP4 variant of MoE GEMM combining dual-accumulator pattern with mixed-precision scales.

**Builder:**
```python
from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

executor = compile_mixed_moe_gemm1(
    model_dim=8192, inter_dim=8192,
    experts=16, topk=4,
    tile_m=64, tile_n=128, tile_k=256,
    doweight_stage1=True,
    a_dtype="fp8", b_dtype="fp4",
    out_dtype="f16",
)
```

---

## 5. Shared Utilities

### 5.1 Reduction Helpers (`kernels/reduce.py`)

Reusable warp and block reduction functions.

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

### 5.2 MFMA Epilogues (`kernels/mfma_epilogues.py`)

Configurable epilogue strategies for MFMA 16x16 kernels.

| Function | Description |
|---|---|
| `default_epilog(...)` | Standard row-iterator: `row = bx_m + mi*16 + lane_div_16*4 + ii` |
| `c_shuffle_epilog(...)` | CK-style LDS CShuffle: write to LDS → barrier → remap threads (8,32) → half2 store |
| `mfma_epilog(use_cshuffle, ...)` | Dispatcher: calls default or CShuffle based on flag |

### 5.3 Preshuffle Pipeline (`kernels/mfma_preshuffle_pipeline.py`)

Shared data movement and layout utilities for preshuffle GEMM/MoE kernels.

| Function | Description |
|---|---|
| `make_preshuffle_b_layout(flir, arith, N, K, ...)` | Build B-preshuffle layout: (N/16, K/64, 4, 16, kpack_bytes) |
| `make_preshuffle_scale_layout(...)` | Build scale layout for MXFP4 quantization |
| `load_b_pack_k32(...)` | Load B pack for K32 MFMA micro-step (returns i64) |
| `tile_chunk_coord_i32(...)` | Map (thread, chunk) → (row, col) for tile loads |
| `buffer_copy_gmem16_dwordx4(...)` | 16-byte global load via buffer-load dwordx4 |
| `lds_store_16b_xor16(...)` | Store 16B to LDS with XOR16 swizzle |
| `lds_store_8b_xor16(...)` | Store 8B to LDS with XOR16 swizzle |
| `lds_store_4b_xor16(...)` | Store 4B to LDS with XOR16 swizzle |
| `lds_load_pack_k32(...)` | Load A-pack from LDS for K32 micro-step |

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
│   │   ├── FP8 / INT8 / INT4(W4A8) / FP16 / BF16
│   │   └── → compile_preshuffle_gemm_a8()
│   │
│   ├── Mixed-precision GEMM (FP4 weights)
│   │   ├── A: fp8 or fp4,  B: fp4
│   │   └── → compile_mxfp4_preshuffle_gemm()
│   │
│   ├── MoE GEMM (expert routing)
│   │   ├── Uniform precision (fp8/fp16/int8/int4)
│   │   └── → compile_moe_gemm1()
│   │
│   └── Mixed MoE GEMM (FP4 expert weights)
│       ├── A: fp8 or fp4,  B: fp4
│       └── → compile_mixed_moe_gemm1()
│
└── Building blocks
    ├── Warp/block reduction     → reduce.py
    ├── MFMA epilogue selection  → mfma_epilogues.py
    └── Preshuffle data movement → mfma_preshuffle_pipeline.py
```

---

## 7. Source Files

| File | Description |
|---|---|
| `kernels/__init__.py` | Package marker |
| `kernels/kernels_common.py` | `stream_ptr_to_async_token()` helper |
| `kernels/layernorm_kernel.py` | LayerNorm builder (`_LayerNorm` MlirModule) |
| `kernels/rmsnorm_kernel.py` | RMSNorm builder (`_RMSNorm` MlirModule) |
| `kernels/softmax_kernel.py` | Softmax builder (`_Softmax` MlirModule) |
| `kernels/preshuffle_gemm.py` | Preshuffle GEMM builder (`_GEMM` MlirModule) |
| `kernels/mixed_preshuffle_gemm.py` | Mixed-precision GEMM builder (`_GEMM` MlirModule) |
| `kernels/moe_gemm_2stage.py` | MoE GEMM 2-stage builder (`_MOE1` MlirModule) |
| `kernels/mixed_moe_gemm_2stage.py` | Mixed MoE GEMM builder (`_MOE1` MlirModule) |
| `kernels/reduce.py` | Shared warp/block reduction helpers |
| `kernels/mfma_epilogues.py` | MFMA epilogue strategies (default, CShuffle) |
| `kernels/mfma_preshuffle_pipeline.py` | Preshuffle data movement and layout utilities |

## 8. Test Files

| File | Tests |
|---|---|
| `tests/kernels/test_layernorm.py` | LayerNorm correctness + perf vs PyTorch/AIter |
| `tests/kernels/test_rmsnorm.py` | RMSNorm correctness + perf |
| `tests/kernels/test_softmax.py` | Softmax correctness + perf vs PyTorch/AIter |
| `tests/kernels/test_preshuffle_gemm.py` | GEMM fp8/int8/int4/bf16 correctness + perf |
| `tests/kernels/test_moe_gemm.py` | MoE GEMM stage1/stage2 correctness + perf |
| `tests/kernels/benchmark_common.py` | Shared benchmark infrastructure (FLIR vs AIter) |
