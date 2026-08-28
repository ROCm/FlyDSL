# Pre-built kernel library guide

This guide covers the available FlyDSL kernels — normalization, softmax, GEMM, and attention — along with their configuration options, supported data types, pipeline designs, and shared utilities.

## Quick reference

| Kernel | Builder function | API style | Dtypes | Key feature |
|---|---|---|---|---|
| **LayerNorm** | `build_layernorm_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | Two-pass vectorized normalization |
| **RMSNorm** | `build_rmsnorm_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16; optional fp32 weight | LDS-cached 3-pass pipeline |
| **Softmax** | `build_softmax_module(M, N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | Register-buffered softmax, opt-in autotuning |
| **Softmax backward** | `build_softmax_bwd_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | fp32 dot reduction, native-dtype register buffering |
| **GEMM** | `compile_preshuffle_gemm(...)` | `@flyc.kernel` | fp8, int8, fp16, bf16 | Preshuffle B, ping-pong LDS, MFMA 16x16 |
| **FlashAttention** | `build_flash_attn_func_module(...)` | `@flyc.kernel` | bf16, f16 (any arch); fp8 e4m3fn (gfx950, D=128, dense) | Dual-wave SWP fwd, GQA/MQA, causal, descale ABI |

All kernels use the `@flyc.kernel`/`@flyc.jit` API from `flydsl.compiler` and `flydsl.expr` (`python/flydsl/`).

---

## 1. Normalization kernels

### 1.1 LayerNorm (`kernels/norm/layernorm_kernel.py`)

Computes `LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta` for each row.

**Builder:**
```python
from kernels.norm.layernorm_kernel import build_layernorm_module

executor = build_layernorm_module(N=8192, dtype_str="bf16")
```

**Configuration constants:**
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
- **Fast path**: When `N == BLOCK_THREADS * VEC_WIDTH * 4` (for example, N=8192), uses fully register-resident computation with no scalar tail
- **Generic path**: Handles arbitrary N with vector body + scalar tail
- **bf16 handling**: Software round-to-nearest-even (RNE) pack on gfx942; hardware `cvt_pk_bf16_f32` on gfx950+
- **Warp reduction**: XOR-shuffle-based intra-wave reduction (shifts: 32, 16, 8, 4, 2, 1), then LDS-based cross-wave synchronization

**Kernel signature** (using `@flyc.kernel` API):
```
GPU_MODULE_NAME = "layernorm_module"

@kernel
layernorm_kernel(self, Input, Gamma, Beta, Output, m_in)

@jit
__call__(self, Input, Gamma, Beta, Output, m_in)
```

### 1.2 RMSNorm (`kernels/norm/rmsnorm_kernel.py`)

Computes `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma`.

**Builder:**
```python
from kernels.norm.rmsnorm_kernel import build_rmsnorm_module

executor = build_rmsnorm_module(N=8192, dtype_str="bf16", store_rstd=False)
```

`build_rmsnorm_module(N, dtype_str, store_rstd=False, eps=EPS,
BLOCK_THREADS=BLOCK_THREADS, weight_dtype_str=None)` optionally writes the
per-row reciprocal std (`rstd`) for use by the backward pass.
`weight_dtype_str` defaults to `dtype_str`; FP16/BF16 activations additionally
support FP32 weights.

**Backward:** `build_rmsnorm_bwd_module(N, dtype_str,
weight_dtype_str=None)` builds the fused RMSNorm backward kernel (grid `(M,)`,
one block per row). Kernel signature
`rmsnorm_bwd_kernel(Input, Gamma, DY, Rstd, DX, DWeight)`: reads the forward
`Rstd`, writes `DX` (input grad), and atomic-adds into `DWeight` (fp32 weight
grad). The forward bakes `eps` into `Rstd`, so the backward does not need it.
The public plain and fused-add training wrappers return `dweight` in the
original weight dtype.

**Configuration constants:** Same as LayerNorm (BLOCK_THREADS=256, VEC_WIDTH=8, etc.)

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

## 2. Softmax kernel

### 2.1 Softmax (`kernels/norm/softmax_kernel.py`)

Computes row-wise softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.

**Builder:**
```python
from kernels.norm.softmax_kernel import build_softmax_module

executor = build_softmax_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration:**
| Parameter | Value | Description |
|---|---|---|
| `BLOCK_THREADS` | 256 by default; 64/128/256/512 for full-row candidates | Total threads per block |
| `THREADS_PER_ROW` | Defaults to `BLOCK_THREADS`; 8/16/32/64 for short-row candidates | Reduction subgroup assigned to one row |
| `ROWS_PER_BLOCK` | 1 by default; derived from `BLOCK_THREADS / THREADS_PER_ROW` | Independent rows packed into one block |
| `vec_width` | `128 // elem_bits` (8 for f16/bf16, 4 for f32) | Derived from the 128-bit transaction contract |
| `WARP_SIZE` | 64 on CDNA, 32 on RDNA | Wavefront size, resolved from the target arch |

`THREADS_PER_ROW` also selects the data-movement path: with
`tile_cols = THREADS_PER_ROW * vec_width`, a row takes the vectorized fast path when
`N % tile_cols == 0` and the scalar generic path otherwise.

**Opt-in autotuning** (`kernels/norm/softmax_autotune.py`):
```python
from kernels.norm.softmax_autotune import softmax_autotuned

softmax_autotuned(x, y)              # serves the tuned or default config, never searches
```
Ordinary calls follow the searched-winner cache → offline artifact → compatibility default
(`BLOCK_THREADS=256`) ordering and never benchmark. `FLYDSL_AUTOTUNE=1` forces a search over
a bounded, shape-aware space:

- full-row `BLOCK_THREADS ∈ {64,128,256,512}` ×
  `waves_per_eu ∈ {none,1,2,4}`;
- Quack-style short-row packing that decouples `THREADS_PER_ROW` from total
  block threads and processes several rows per block.

The search-space rationale was checked against AITER
`536118aaf94047b0b559e0730749352659419b34`, SGLang
`955704544c60e920672aa434cefa2ce78c0ceb4c`, and Tri Dao's Quack
`60d88082272a256fa9b3b2ab631c82cfa78337c6`. Quack's portable ideas are the
row-width-dependent reduction subgroup, a separate 128/256-thread CTA size,
multiple rows per CTA, and an online/non-online algorithm choice; its
multi-CTA cluster reduction is NVIDIA-specific. AITER's standalone Triton
kernel is a fixed two-pass online/reload implementation (`.cg`, eight warps,
two stages, `waves_per_eu=2`), not an autotuned space. The pinned SGLang tree
has attention-local and top-k softmax implementations but no directly comparable
standalone row-wise kernel; attention tile/stage choices are therefore not
imported here.

Every candidate is numerically validated before ranking and uses the shared
GPU-backlog and batched-event timer. Input cache policy is deliberately not a
search axis: on gfx950, non-temporal loads changed rank between repeated use of
one address and rotation across fresh addresses. Cache residency is absent from
the shape-only artifact identity, so persisting either result would encode an
unstated workload assumption. The tested three-pass reload algorithm is also
excluded because it lost to the register-buffered compatibility path; a future
algorithm axis should implement a true online pair reduction before entering
the default search.

Within a 2% timing tie, the selector favors the compatibility default, then no
explicit `waves_per_eu`, then more rows per block. Larger measured improvements
still win; the tie rule only avoids persisting noise-level differences between
6–10 µs candidates. Softmax uses 10 warmup and 100 measured launches, divided
into five GPU-backlogged event windows, so bandwidth-scale candidates are also
ranked from a stable sample.

Artifacts use the name `softmax_fwd` and cover forward only. Softmax backward
has no autotune adopter in this change. See [`autotune_guide.md`](autotune_guide.md).

**Algorithm (6 stages):**
1. **Load data**: Vectorized global loads into register buffer with validity masks
2. **Local max**: Per-thread vector reduction (`maxnumf`)
3. **Global max**: Block-wide shuffle reduction (intra-wave XOR → wave0 finalize via LDS)
4. **Local exp + sum**: `exp2(x * log2(e))` approximation, accumulate partial sums
5. **Global sum**: Block-wide reduction for sum
6. **Normalize + store**: Divide by sum, convert to output dtype, vectorized store

**Kernel signature:**
```python
build_softmax_module(M, N, dtype_str="f32", BLOCK_THREADS=256)  # M is vestigial
launch_softmax(A, C, m_in, stream=...)                          # returned launcher

# Direct-JIT entry point used by the autotuner
softmax_direct(A, C, m_in, N, dtype_str, BLOCK_THREADS, tuning_schema, stream=...)
```

### 2.2 Softmax backward (`kernels/norm/softmax_bwd_kernel.py`)

Computes the row-wise Softmax gradient: `dx = y * (dy - sum(dy * y))`, with the
dot reduction accumulated in fp32.

**Builder:**
```python
from kernels.norm.softmax_bwd_kernel import build_softmax_bwd_module

launch = build_softmax_bwd_module(N=8192, dtype_str="bf16")
launch(dy, y, dx, M, stream=torch.cuda.current_stream())
```

The builder takes `N` only; the row count is the runtime `m_in` launch argument.
Inputs must be **contiguous 2-D** tensors — reshape a 4-D attention gradient to
`(B*H*S, S)` before calling, since the buffer-tensor path assumes row-major rows.

**Paths:**
| Condition | Behaviour |
|---|---|
| `N >= tile_cols and N % tile_cols == 0` | 128-bit vectorized load/store (`tile_cols` = 1024 for f32, 2048 for 16-bit) |
| otherwise | masked scalar path for arbitrary `N` |
| `N <= 16384` | both operands register-resident across the reduction — ideal 3-unit traffic |
| `16384 < N <= 32768` | `Y` resident, `DY` re-read — 4 units |
| `N > 32768` | neither resident — 5 units |

Ideal traffic is 3 units (read `Y`, read `DY`, write `DX`); each operand dropped
from registers adds one more. The residency cap is on elements held per thread
(`N / BLOCK_THREADS`), so the tier boundaries fall at the same `N` for every
dtype. Use `softmax_bwd_buffered_operands(N, dtype_str)` to query the tier.

Both bounds are measured on an idle gfx950, not assumed. Pushing the middle tier
out to `N = 65536` spills and costs 29% (337.4 µs vs 261.8 µs at 2048x65536
bf16); dropping the middle tier costs 30-38% on the shapes it covers (4096x32768
bf16: 169.3 µs with `Y` resident vs 220.4 µs without).

Benchmark these on an **idle** GPU. A neighbouring tenant on the same device
distorts results by 20-35%, and single-sample idleness checks miss bursty
neighbours — sample repeatedly and reject a device that is busy in any sample.

**Notes:**
- One block per row. Small `M`/`N` are launch-bound rather than bandwidth-bound;
  effective bandwidth reads as a few percent of peak there and that is expected.
- The generic path unrolls `2 * ceil(N / 256)` scalar bodies, so compile time
  grows with `N` for large non-aligned rows.

---

## 3. GEMM kernel

### 3.1 Preshuffle GEMM (`kernels/gemm/preshuffle_gemm.py`)

MFMA 16x16-based GEMM with B-matrix preshuffle layout: `C[M,N] = A[M,K] @ B[N,K]^T`.

Uses the `@flyc.kernel` / `@flyc.jit` API.

**Builder:**
```python
from kernels.gemm.preshuffle_gemm import compile_preshuffle_gemm

launch_fn = compile_preshuffle_gemm(
    N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",
    out_dtype="bf16",
    epilogue="none",
    lds_stage=2,
)
```

Returns a `@flyc.jit`-decorated function that auto-compiles on first call.

**Parameters** (keyword-only):
| Parameter | Type | Description |
|---|---|---|
| `N, K` | int | GEMM dimensions: A[M,K], B[N,K], C[M,N]. M is a runtime arg, not a compile-time parameter. |
| `tile_m, tile_n, tile_k` | int | Block tile sizes |
| `in_dtype` | str | `"fp8"`, `"int8"`, `"fp16"`, `"bf16"` (default `"fp8"`) |
| `out_dtype` | str | Output dtype (default `"bf16"`) |
| `epilogue` | str | Fused epilogue: `"none"`, `"bias"`, `"bias_relu"`, `"bias_silu"`, `"bias_gelu"` (default `"none"`) |
| `lds_stage` | int | `2` = ping-pong LDS (tuned), `1` = single LDS buffer |
| `waves_per_eu` | int | Occupancy hint (None = default, 1-4 = limit occupancy) |
| `enable_scheduler` | bool | Enable the MLIR instruction scheduler (default `True`) |
| `use_async_copy` | bool | Use async DMA for A tile global-to-LDS transfer |
| `xcd_swizzle` | int | XCD remap factor for grid launch (0 = disabled) |

**Key constraints:**
- `tile_k` must be a positive divisor of `K`
- MX (block-scaled) GEMM is a separate kernel (`kernels/gemm/mxfp4_preshuffle.py`, `kernels/gemm/fp4_gemm_4wave.py`); INT4 is not supported by this kernel.

**MX A x MXFP4 B GEMM (`kernels/gemm/mxfp4_preshuffle.py`, gfx950):** the
`launch_gemm` `@flyc.jit` launcher runs `A x preshuffled MXFP4 B` with per-32
E8M0 scales, selecting the A element type via `a_dtype` (`"fp4"`, `"fp6"`, or
`"fp8"`; B is always MXFP4). This unified `launch_gemm` is the current gfx950
entry point (it replaced the earlier standalone `compile_mxfp6_gemm` from #780);
the separate `compile_mxfp4_gemm` in `kernels/gemm/gemm_fp8fp4_gfx1250.py` is the
distinct gfx1250 kernel. `batch>1` runs a strided-batched GEMM over `grid.z`.
Covered by `tests/kernels/test_preshuffle_gemm.py`.

**Pipeline details:**
- **lds_stage=2 (ping-pong)**: Two LDS buffers for A tiles. Cross-tile A0 prefetch overlaps VMEM with LDS reads
- **lds_stage=1 (single)**: CK-style intrawave schedule with single LDS buffer
- **K64-byte micro-step**: Each step issues 2x K32 MFMA operations
- **XOR16 swizzle**: Byte-level swizzle on LDS to avoid bank conflicts
- **B-preshuffle**: Shape (N0, K0, KLane, NLane, KPackBytes) = (N/16, K/64, 4, 16, kpack_bytes)
- **Fused epilogue**: selected via `epilogue=` (bias add + optional relu/silu/gelu activation)

**Launch function signature:**
```python
launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, M_val, N_val, stream)
```

- `arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias`: PyTorch tensors (auto-converted to memref). `arg_bias` is the fused epilogue bias (per-N, `out_dtype`); unused when `epilogue == "none"`.
- `M_val, N_val`: Python int (auto-converted to Int32)
- `stream`: `fx.Stream` (default stream if omitted)

---

## 3b. FlashAttention forward (`kernels/attention/flash_attn_generic.py`, `kernels/attention/flash_attn_gfx950.py`, `kernels/attention/flash_attn_fp8_gfx950.py`)

Dense FlashAttention forward. `build_flash_attn_func_module(num_heads, head_dim,
causal=..., dtype_str=..., num_kv_heads=...)` is the public builder; on
gfx950 + `head_dim == 128` it routes to the dual-wave software-pipelined fast path
(`build_flash_attn_dualwave_swp_module`), otherwise to the generic fallback.
Supports MHA and GQA/MQA (`num_kv_heads <= num_heads`), causal and non-causal,
arbitrary sequence length, and (bf16/f16) packed varlen + split-K.

The paged BF16/F16 interface accepts linear or AITER-style vectorized-5D KV
caches. Short-query attention supports physical page sizes 16 and 64, including
causal sliding windows through `sliding_window`; the dual-wave long-query path
remains page-size 64 only.

### fp8 (e4m3fn) forward

| Property | Value |
|---|---|
| Arch / shape | gfx950 (CDNA4) only; `head_dim == 128`; dense only |
| Inputs | **pre-quantized** Q/K/V in `torch.float8_e4m3fn` (OCP e4m3fn, not fnuz); no in-kernel quantization |
| Descales | per-tensor shape-`[1]` fp32 `q_descale`, `k_descale`, `v_descale` (launch kwargs) |
| Math | QK on native `mfma_f32_32x32x16_fp8_fp8`, with `q_descale*k_descale*sm_scale` on fp32 logits; fp32 online softmax; PV applies `v_descale`; **fp32 accumulation** throughout |
| Output | `bf16` only |
| Unsupported (rejected with a clear error) | fp8 split-K (`num_kv_splits > 1`) and fp8 packed varlen (`cu_seqlens`) |

The PV path dequantizes fp8 V to bf16 in-kernel and accumulates P*V in bf16, keeping
the softmax probabilities at high precision. Build/launch example:

```python
from kernels.attention.flash_attn_generic import build_flash_attn_func_module

exe = build_flash_attn_func_module(num_heads=H, head_dim=128, causal=False,
                                   dtype_str="fp8", num_kv_heads=H_kv)
# Q/K/V are e4m3fn [B,S,H,D]; O is bf16; descales are shape-[1] fp32.
exe(q_fp8.view(-1), k_fp8.view(-1), v_fp8.view(-1), o_bf16.view(-1), B, S,
    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
```

Reproduce the fp8 correctness sweep and the FlyDSL-fp8 vs aiter-ASM-fp8 comparison:

```bash
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --warmup 3 --iters 3
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --warmup 10 --iters 50
```

---

## 4. Shared utilities

### 4.1 Common kernel helpers (`kernels/common/kernels_common.py`)

Shared kernel utilities used across GEMM/MoE/norm kernels.

| Function | Description |
|---|---|
| `get_warp_size(arch=None)` | Wave size for the arch: `32` on gfx10/11/12, else `64` |
| `dtype_to_elem_type(dtype_str)` | Map a dtype string to the Fly element type |
| `validate_moe_dtypes(a_dtype, b_dtype)` | Validate an allowed MoE A/B dtype pairing |
| `get_llvm_ptr(ptr, offset, dtype_bytes, ...)` | Compute a byte-offset LLVM pointer |
| `atomic_add(...)` | Emit an atomic add |
| `_if_then(if_op, scf=None)` / `_if_else(if_op, scf=None)` | SCF `if`/`else` region context managers |

### 4.2 MFMA epilogues (`kernels/mma/mfma_epilogues.py`)

Configurable epilogue strategies for MFMA 16x16 kernels.

| Function | Description |
|---|---|
| `default_epilog(...)` | Standard row-iterator: `row = bx_m + mi*16 + lane_div_16*4 + ii` |
| `c_shuffle_epilog(...)` | CK-style LDS CShuffle: write to LDS → barrier → remap threads → half2 store |
| `mfma_epilog(use_cshuffle, ...)` | Dispatcher: calls default or CShuffle based on flag |

### 4.3 Preshuffle pipeline (`kernels/mma/mfma_preshuffle_pipeline.py`)

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

### 4.4 Layout coordinate helpers

Native Fly dialect coordinate mapping (in `flydsl.expr` and `kernels/mma/mfma_preshuffle_pipeline.py`):

| Function | Description |
|---|---|
| `fx.crd2idx(crd, layout)` | Coordinate → flat index (Fly dialect op) |
| `fx.idx2crd(idx, layout)` | Flat index → coordinate tuple (Fly dialect op) |
| `fx.get(int_tuple, mode)` | Extract element at index from `!fly.int_tuple` |
| `crd2idx(crd, layout)` | Wrapper in `kernels/mma/mfma_preshuffle_pipeline.py` (auto index cast) |

---

## 5. Kernel API comparison

### New API (GEMM)

Used by `kernels/gemm/preshuffle_gemm.py`:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl

@flyc.kernel
def gemm_kernel(arg_c: fx.Tensor, arg_a: fx.Tensor, ...):
    tid = gpu.thread_idx.x
    # ... uses fx.*, Numeric/Vector, gpu.*, rocdl.* ...

@flyc.jit
def launch_fn(arg_c: fx.Tensor, ..., stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(arg_c, ...).launch(grid=..., block=..., stream=stream)
```

---

## 6. Kernel decision tree

```
What operation do you need?
│
├── Normalization
│   ├── Need bias (beta) term? → LayerNorm (kernels/norm/layernorm_kernel.py)
│   └── No bias term?         → RMSNorm (kernels/norm/rmsnorm_kernel.py)
│
├── Softmax
│   ├── Row-wise softmax      → Softmax (kernels/norm/softmax_kernel.py)
│   └── Softmax gradient      → Softmax backward (kernels/norm/softmax_bwd_kernel.py)
│
├── Matrix Multiply (GEMM)
│   ├── Standard GEMM (uniform precision)
│   │   ├── FP8 / INT8 / FP16 / BF16
│   │   └── → compile_preshuffle_gemm()
│   │
│   └── Uses new @flyc.kernel API
│       └── See kernels/gemm/preshuffle_gemm.py
│
├── MoE (Mixture of Experts)
│   ├── Blockscale MoE (gate+up+reduce)
│   └── Standard MoE (fp8/f16/bf16/int8/int4)
│       └── → kernels/moe/moe_gemm_2stage.py
│
└── Building blocks
    ├── Common kernel helpers    → kernels/common/kernels_common.py
    ├── MFMA epilogue selection  → kernels/mma/mfma_epilogues.py
    └── Preshuffle data movement → kernels/mma/mfma_preshuffle_pipeline.py
```

---

## 7. Source files

| File | Description |
|---|---|
| `kernels/gemm/preshuffle_gemm.py` | GEMM (preshuffle layout) |
| `kernels/moe/moe_gemm_2stage.py` | MoE GEMM 2-stage (gate/up + reduce) |
| `kernels/moe/mxfp_moe/` | Fused a4w4/a8w4 MoE 2-stage GEMM (device fp4 re-quant) |
| `kernels/attention/pa_decode_fp8.py` | Paged attention decode (FP8) |
| `kernels/attention/flash_attn_generic.py` | FlashAttention generic fallback |
| `kernels/attention/flash_attn_gfx950.py` | FlashAttention gfx950 bf16/f16 fast path |
| `kernels/attention/flash_attn_fp8_gfx950.py` | FlashAttention gfx950 fp8 dense fast path |
| `kernels/norm/layernorm_kernel.py` | LayerNorm (layout API) |
| `kernels/norm/rmsnorm_kernel.py` | RMSNorm (layout API) |
| `kernels/norm/softmax_kernel.py` | Softmax (layout API) |
| `kernels/norm/softmax_bwd_kernel.py` | Softmax backward (layout API) |
| `kernels/norm/softmax_autotune.py` | Softmax opt-in autotune adopter |
| `kernels/attention/fused_rope_cache_kernel.py` | Fused RoPE + KV cache |
| `kernels/comm/custom_all_reduce.py` | Multi-GPU all-reduce |
| `kernels/gemm/rdna_f16_gemm.py` | RDNA FP16 GEMM |
| `kernels/gemm/rdna_fp8_preshuffle_gemm.py` | RDNA FP8 GEMM |
| `kernels/gemm/gemm_common_gfx1250.py` | GFX1250 GEMM common |
| `kernels/gemm/gemm_fp8fp4_gfx1250.py` | GFX1250 FP8/FP4 GEMM |
| `kernels/gemm/wmma_gemm_gfx1250.py` | GFX1250 WMMA GEMM |
| `kernels/mma/mfma_epilogues.py` | MFMA epilogue helpers |
| `kernels/mma/mfma_preshuffle_pipeline.py` | Preshuffle data movement and layout utilities |
| `kernels/mma/pipeline_utils.py` | Pipeline utility helpers |
| `kernels/common/kernels_common.py` | Common kernel utilities |
| `kernels/common/tensor_shim.py` | GTensor/STensor abstraction |

## 8. Test files

| File | Tests |
|---|---|
| `tests/kernels/test_preshuffle_gemm.py` | GEMM fp8/int8/fp16/bf16 |
| `tests/kernels/test_moe_gemm.py` | MoE GEMM |
| `tests/kernels/test_moe_reduce.py` | MoE reduce kernel |
| `tests/kernels/test_pa.py` | Paged attention decode |
| `tests/kernels/test_flash_attn_fwd.py` | FlashAttention |
| `tests/kernels/test_layernorm.py` | LayerNorm |
| `tests/kernels/test_rmsnorm.py` | RMSNorm |
| `tests/kernels/test_softmax.py` | Softmax |
| `tests/kernels/test_softmax_bwd.py` | Softmax backward |
| `tests/kernels/test_softmax_autotune.py` | Softmax autotune selection and candidate correctness |
| `tests/kernels/test_fused_rope_cache.py` | Fused RoPE + KV cache |
| `tests/kernels/test_allreduce.py` | Multi-GPU all-reduce |
| `tests/kernels/test_rdna_gemm.py` | RDNA GEMM |
| `tests/kernels/test_gemm_fp8fp4_gfx1250.py` | GFX1250 FP8/FP4 GEMM |
| `tests/kernels/test_wmma_gemm_gfx1250.py` | GFX1250 WMMA GEMM |
| `tests/kernels/test_vec_add.py` | Vector addition |
| `tests/kernels/test_quant.py` | Quantization utilities |
| `tests/kernels/benchmark_common.py` | Shared benchmark infrastructure |
