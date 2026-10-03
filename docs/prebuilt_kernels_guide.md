# Pre-built kernel library guide

This guide covers the available FlyDSL kernels — normalization, softmax, GEMM, attention, and gfx120x/RDNA4 suite kernels — along with their configuration options, supported data types, pipeline designs, and shared utilities.

## Quick reference

| Kernel | Builder function | API style | Dtypes | Key feature |
|---|---|---|---|---|
| **LayerNorm** | `build_layernorm_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | Two-pass vectorized normalization |
| **RMSNorm** | `build_rmsnorm_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16; optional fp32 weight | LDS-cached 3-pass pipeline |
| **Softmax** | `build_softmax_module(M, N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | Register-buffered softmax, opt-in autotuning |
| **Softmax backward** | `build_softmax_bwd_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | fp32 dot reduction, native-dtype register buffering |
| **GEMM** | `compile_preshuffle_gemm(...)` | `@flyc.kernel` | fp8, int8, fp16, bf16 | Preshuffle B, ping-pong LDS, MFMA 16x16 |
| **FlashAttention** | `build_flash_attn_func_module(...)` | `@flyc.kernel` | bf16, f16 (any arch); fp8 e4m3fn (gfx950, D=128, dense) | Dual-wave SWP fwd, GQA/MQA, causal, descale ABI |
| **FlashAttention (gfx120x)** | `flydsl_flash_attn_func` / gfx120x host | `@flyc.kernel` | bf16, f16, fp8 e4m3fn, fp8 e5m2, int8, iu4 | Dense self/cross, causal, KV-pad, additive mask, D up to 384. bf16/fp16 in-kernel: dense, return_lse, sink, ALiBi (uniform and per-head), packed varlen, paged linear / linear3d / vectorized, varlen+paged, split-K partial plus wave32 combine, GQA when Hq % Hkv == 0. Still raise: varlen per-head bias, varlen+paged per-head bias, ragged paged KV with bias, non-divisible GQA, an explicit KV head count that does not match the KV tensor, batch-varying ALiBi, other paged layouts. fp8, int8, and iu4 dense: bias, ALiBi, sink, LSE, GQA; no varlen, paged, or split-K. Not silent gfx950. |
| **gfx120x scaled_mm FP8** | `scaled_mm_fp8_auto` (default) / `build_scaled_mm_fp8_module(...)` | `@flyc.kernel` | fp8 e4m3fn, fp8 e5m2 → bf16 | Prefer `_auto` tile picker; plain module for fixed tiles |
| **gfx120x scaled_mm FP8 fused** | `build_scaled_mm_fp8_fused_module(...)` | `@flyc.kernel` | fp8 e4m3fn, fp8 e5m2, bf16 LoRA residual | Act-quant+mm; multi-LoRA host residual |
| **gfx120x int8 linear fused** | `build_int8_linear_fused_module(...)` / `int8_linear_fused` | `@flyc.kernel` | int8, bf16 LoRA residual | Act-quant+iu8; multi-LoRA host residual |
| **gfx120x int8 linear (iu8)** | `build_int8_linear_module(...)` | `@flyc.kernel` | int8 → bf16, fp16, or fp32 (iu8 WMMA) | Default out follows `x.dtype` |
| **gfx120x W8A16 linear** | `build_w8a16_linear_module(...)` | `@flyc.kernel` | bf16 or fp16 acts; int8, fp8 e4m3fn, fp8 e5m2 weights | Default when K < 256; K ≥ 256 goes to iu8 |
| **gfx120x int8 linear size dispatcher** | `int8_linear_auto` / `int8_linear_dispatched` | host | W8A16 or iu8 | **DEFAULT**: prefer iu8 when K ≥ 256, else W8A16 (measured R9700; see `gfx120x_autotune_tables.py`) |
| **gfx120x iu4 GEMM** | `iu4_gemm` / `build_iu4_gemm_module` | `@flyc.kernel` | packed int4 → bf16, fp16, fp32, or i32 | Native iu4 WMMA (default when K%16==0); unpack→iu8 fallback |
| **gfx120x zero-LDS N-major / fused_gemm_TN** | `build_fused_gemm_tn_module` / `fused_gemm_tn` / `fused_swiglu_mlp_inreg` / `fused_swiglu_mlp_lds` / `fused_swiglu_mlp_nmajor` | `@flyc.kernel` | bf16, fp16 | A/B-swap N-major D0. In-reg SiLU×mul when K and FFN are 16. Otherwise, if M/K/FFN/N_out are multiples of 16, one kernel loops K and FFN and keeps that fragment in LDS. Other shapes use two GEMMs and a global mid. |
| **gfx120x RoPE / RMS+RoPE / AdaLN** | `build_rope_*` / `build_rms_rope_*` / `build_adaln_module` | `@flyc.kernel` | f32, f16, bf16 | Incl. Q/K fused builders |
| **gfx120x quant / SwiGLU / ConvRot** | same builders; ConvRot linear default `linear_dtype="int4"` | `@flyc.kernel` | fp8, int8, bf16, W4 | ConvRot default = native iu4; AWQ/SVD are separate families |


## How to call gfx120x defaults

There is no environment flag. Call the default entry for each family.
Copy/paste snippets for ConvRot int4 and bare iu4 GEMM are below.

| Goal | Default call (concrete) | Force older / other path |
|---|---|---|
| Int8-weight linear | `int8_linear_auto(x, w_int8, …)` / `int8_linear_dispatched(...)` | `force_kernel="w8a16"` or `"iu8"` |
| FP8 scaled_mm | `scaled_mm_fp8_auto(a_fp8, b_fp8, scale_a, scale_b, …)` | `build_scaled_mm_fp8_module(...)` with a fixed tile. There is no `scaled_mm_fp8()` host. |
| ConvRot W4A4 linear | `q, s = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)` then `convrot_w4a4_linear(x, q, s, convrot_groupsize=64)` — default `linear_dtype="int4"` | `linear_dtype="int8"` for unpack→iu8 |
| Bare packed int4 GEMM | `a_p = pack_int4_row_major(a_i4); b_p = pack_int4_row_major(b_i4); iu4_gemm(a_p, b_p, scale_a, scale_b, out_dtype=torch.bfloat16)` | `prefer_native=False` |

### Minimal ConvRot int4 call (copy/paste)

```python
from kernels.quant.rdna4_convrot_w4a4 import (
    quantize_convrot_w4a4_weight,
    convrot_w4a4_linear,
)

qweight, wscales = quantize_convrot_w4a4_weight(w_bf16, convrot_groupsize=64)
y = convrot_w4a4_linear(x_bf16, qweight, wscales, convrot_groupsize=64)
# y is bf16 [M, N]; acts were ConvRot-quantized to int4 inside the call.
```

### Minimal bare iu4 GEMM call (copy/paste)

```python
from kernels.quant.rdna4_int4_codec import pack_int4_row_major
from kernels.gemm.rdna4_iu4_gemm import iu4_gemm

a_packed = pack_int4_row_major(a_logical_i4)  # [M, K//2] int8
b_packed = pack_int4_row_major(b_logical_i4)  # [N, K//2] int8
y = iu4_gemm(a_packed, b_packed, scale_a, scale_b, out_dtype=torch.bfloat16)
```

**Measured defaults:** int8 and FP8 size/tile rules come from Speed-versus-HIP
crossovers on one R9700. Wired defaults live in
`kernels/common/gfx120x_autotune_tables.py` (int8: prefer iu8 when `K >= 256`,
else W8A16). Remeasure on other RDNA4 boards or ROCm stacks before treating
the numbers as universal.

**Native int4 vs AWQ:** native iu4 / ConvRot int4 ties or beats HIP on
measured shapes. AWQ W4A16 is a different family — do not read AWQ numbers as
“int4 is slow.”

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
the separate `launch_gemm_a8w4_mxscale` entry point in
`kernels/gemm/gemm_a8w4_mxscale_gfx1250.py` is the distinct gfx1250 kernel.
`batch>1` runs a strided-batched GEMM over `grid.z`.
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

## 3b. FlashAttention forward (`kernels/attention/flash_attn_generic.py`, `kernels/attention/flash_attn_gfx950.py`, `kernels/attention/flash_attn_fp8_gfx950.py`, `kernels/attention/flash_attn_*_gfx120x.py`)

Dense FlashAttention forward. `build_flash_attn_func_module` does not pick the
architecture. `flydsl_flash_attn_func` does. On gfx950 with `head_dim == 128`
that function uses `build_flash_attn_dualwave_swp_module`. Otherwise the gfx950
and gfx942 paths use the generic builder. On gfx120x, `flydsl_flash_attn_func`
returns before either of those paths.
Supports MHA and GQA/MQA (`num_kv_heads <= num_heads`), causal and non-causal,
arbitrary sequence length, and (bf16, f16) packed varlen + split-K where noted.

| Arch | Modules | Dtypes | Coverage |
|---|---|---|---|
| **gfx120x (RDNA4)** | `flash_attn_gfx120x.py`, `flash_attn_fp8_gfx120x.py`, `flash_attn_int8_gfx120x.py`, `flash_attn_iu4_gfx120x.py`, `flash_attn_gfx120x_host.py`, `flash_attn_gfx120x_splitk.py`, `flash_attn_gfx120x_ext.py`, `_flash_attn_iu4_native_body_gfx120x.py` | bf16, f16, fp8 e4m3fn, fp8 e5m2, int8, iu4 | Dense self/cross; causal (self+cross); descales for FP8, int8, iu4; KV tile pad + `seq_len_kv_valid`; additive attn mask/bias; D up to 384. bf16/fp16 in-kernel: dense, return_lse, sink, ALiBi (uniform and per-head), packed varlen, paged linear / linear3d / vectorized, varlen+paged, split-K partial plus wave32 combine, GQA when Hq % Hkv == 0. Still raise: varlen per-head bias, varlen+paged per-head bias, ragged paged KV with bias, non-divisible GQA, an explicit KV head count that does not match the KV tensor, batch-varying ALiBi, other paged layouts. fp8, int8, and iu4 dense: bias, ALiBi, sink, LSE, GQA; no varlen, paged, or split-K on those kernels. Not gfx950 dualwave. |
| **gfx950** | `flash_attn_gfx950.py`, `flash_attn_fp8_gfx950.py`, paged | bf16, f16; fp8 e4m3fn (D=128, dense) | Dual-wave SWP, GQA/MQA, varlen, split-K, paged KV, bias/ALiBi/sink |
| **generic / gfx942** | `flash_attn_generic.py` | bf16, f16 | Dense fallback |

Native iu4 WMMA FlashAttention uses the gfx120x iu4 atom path for dense GQA when
bias, ALiBi, sink, and LSE are unset. The native body has no knobs for those;
if any is set, the host unpacks to int8 FA instead of dropping them. Unpack→iu8
also remains the fallback when native packing does not apply.

### fp8 (e4m3fn) forward, gfx950

This subsection is the gfx950 MFMA path. It is not the gfx120x path. gfx120x fp8
uses `build_flash_attn_func_fp8_module` in `flash_attn_fp8_gfx120x.py`. The
builder below raises on `dtype_str="fp8"`.

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


## 3c. gfx120x / RDNA4 kernels (R9700)

Additive suite for **wave32 WMMA** on the GFX120X family (device id often
`gfx1201`). Prefer these over CDNA MFMA / gfx950 paths when targeting RDNA4.
Architecture notes (LDS/CU, wave size): `docs/architecture_guide.md`.
Atom / builder notes: `docs/kernel_authoring_guide.md` (RDNA4 WMMA).
Tuning context (LDS ping-pong, bandwidth vs math): `docs/kernel_tuning_guide.md`.

**Measured defaults:** size/tile/block-threads tables live in
`kernels/common/gfx120x_autotune_tables.py` (R9700, HIP 7.17.26374, 2026-09-30).
Timing follows FlyDSL `do_bench` (`docs/autotune_guide.md`).

**Defaults:** call `int8_linear_auto`, `scaled_mm_fp8_auto`, and
ConvRot with default `linear_dtype="int4"` (nibble-packed native iu4). These
cutoffs come from measured crossovers on one R9700 — not a full FlyDSL
autotune search. Force alternate paths explicitly when you already know the shape.

**Int4 packing vocabulary:** *nibble-packed* means two signed 4-bit values per
`int8` byte (`K/2` or `D/2`, low nibble = even index), loaded as i32 fragments
into the gfx120x iu4 WMMA atom. This is **not** AWQ’s uint4 layout.

### Int8 linear — two pipelines + size dispatcher

| Surface | Module | Pipeline |
|---|---|---|
| W8A16 linear | `kernels/gemm/rdna4_w8a16_linear.py` | int8, fp8 e4m3fn, fp8 e5m2 weights cast to the activation dtype (bf16 or fp16); output defaults to that same dtype (fp32 optional). Float WMMA, no iu8 atom |
| iu8 int8 linear | `kernels/gemm/rdna4_int8_linear.py` | int8 activations × int8 weights. Output is `out_dtype`, defaulting to `x.dtype` (bf16, fp16, or fp32) |
| Fused act-quant + iu8 (+ LoRA residual) | `kernels/gemm/rdna4_int8_linear_fused.py` | Multi-LoRA as host bf16 residuals (same policy as FP8 fused) |
| Size dispatcher | `kernels/gemm/rdna4_int8_linear_dispatch.py` | Host rule: `K < 256` → W8A16, else iu8. A large M with a small K stays W8A16 |
| Auto entry | `kernels/gemm/rdna4_int8_linear_auto.py` | Re-export of the default dispatcher in `rdna4_int8_linear_dispatch.py` |

Without the dispatcher, forcing W8A16 on large `[1024,4096,4096]` is slower than HIP
(~×0.57). The auto entry picks iu8 and is faster (~×1.57) on that shape. See the
`gfx120x_autotune_tables` / size-dispatch modules for the wired rule.

#### Call examples (int8)

```python
from kernels.gemm.rdna4_int8_linear_auto import int8_linear_auto
y = int8_linear_auto(x_bf16, w_int8, w_scale, out_dtype=torch.bfloat16)
y = int8_linear_auto(x_bf16, w_int8, w_scale, force_kernel="iu8")
```

Fused LoRA: host bf16 residual after one quantized base GEMM; see fused module docstring.


### FP8 GEMM

| Surface | Module | Notes |
|---|---|---|
| Scaled FP8 GEMM | `kernels/gemm/rdna4_scaled_mm_fp8.py` | e4m3fn and e5m2 |
| Fused act-quant + scaled_mm (+ LoRA residual) | `kernels/gemm/rdna4_scaled_mm_fp8_fused.py` | Multi-LoRA as host bf16 residuals |
| Tile auto-picker | `kernels/gemm/rdna4_scaled_mm_fp8_auto.py` | Same kernel; measured tile pick on [1024,5120,5120] (~×1.00 → ~×3.97) |

#### Call examples (FP8)

```python
from kernels.gemm.rdna4_scaled_mm_fp8_auto import scaled_mm_fp8_auto
y = scaled_mm_fp8_auto(a_fp8, b_nk, scale_a, scale_b, e5m2=False)
y = scaled_mm_fp8_auto(a_fp8, b_nk, scale_a, scale_b, e5m2=True)  # e5m2
```

| Native iu4 GEMM | `kernels/gemm/rdna4_iu4_gemm.py` | Packed int4 → iu4 atom; ConvRot default `linear_dtype='int4'` |
| Zero-LDS N-major / fused_gemm_TN (+ in-reg SwiGLU) | `kernels/gemm/rdna4_fused_mlp_nmajor.py` | A/B swap on GEMM0; in-reg SiLU×mul; 16×16 panels; production layouts untouched |

### Norm / RoPE

| Surface | Module |
|---|---|
| RoPE (+ split-half, Q/K fused) | `kernels/norm/rope_gfx120x.py` |
| RMS+RoPE (+ split, Q/K fused) | `kernels/norm/rms_rope_gfx120x.py` |
| AdaLN | `kernels/norm/adaln_gfx120x.py` |
| Shared helpers | `kernels/common/gfx120x_buf_helpers.py` |
| **FlashAttention bf16, fp16** | `kernels/attention/flash_attn_gfx120x.py` + `_host` |
| **FlashAttention FP8 e4m3fn, e5m2** | `kernels/attention/flash_attn_fp8_gfx120x.py` + `_host` |
| **FlashAttention int8 / iu4** | `flash_attn_int8_gfx120x` / `flash_attn_iu4_gfx120x` + `_host` |

#### Call examples (FlashAttention)

```python
from kernels.attention.flash_attn_gfx120x_host import (
    flydsl_flash_attn_func,
    flydsl_flash_attn_fp8_func,
    flydsl_flash_attn_int8_func,
    flydsl_flash_attn_iu4_func,
)
out = flydsl_flash_attn_func(q, k, v, causal=False)           # bf16/fp16 BSHD
out = flydsl_flash_attn_fp8_func(q8, k8, v8, q_descale=d, k_descale=d, v_descale=d)
out = flydsl_flash_attn_int8_func(qi, ki, vi, q_descale=s, k_descale=s, v_descale=s)
out = flydsl_flash_attn_iu4_func(qp, kp, vp, q_descale=s, k_descale=s, v_descale=s)
```

D ∈ [64, 384], D % 32 == 0. See FlashAttention gfx120x section above.


### Quant / elementwise

| Surface | Module |
|---|---|
| FP8 quant/dequant | `kernels/quant/rdna4_fp8_quant.py` |
| Stochastic FP8 | `kernels/quant/rdna4_stoch_fp8.py` |
| SwiGLU (SiLU×mul / chunk) | `kernels/common/gfx120x_swiglu.py` (`kernels/quant/rdna4_swiglu.py` re-exports it) |
| Int8 rowwise quant | `kernels/quant/rdna4_quantize_int8_rowwise.py` |
| Int8 tensorwise quant | `kernels/quant/rdna4_quantize_int8_tensorwise.py` |
| Int8 ConvRot weight quant + linear host | `kernels/quant/rdna4_int8_convrot.py` |
| ConvRot W4A4 weight quant + linear (**default** native `int4`; `int8` = unpack→iu8) | `kernels/quant/rdna4_convrot_w4a4.py` |
| Asym W4A8 dequant (+ host quant) + int8-linear | `kernels/quant/rdna4_asym_w4a8.py` |
| AWQ W4A16 dequant + fused GEMV | `kernels/quant/rdna4_awq_w4a16.py` |
| SVDQuant W4A4 dequant + fused scaled_mm (host LoRA) | `kernels/quant/rdna4_svdquant_w4a4.py` |

#### Call examples (AWQ / SVDQuant — separate families)

```python
from kernels.quant.rdna4_awq_w4a16 import gemv_awq_w4a16
y = gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=64)  # often slower than HIP

from kernels.quant.rdna4_svdquant_w4a4 import scaled_mm_svdquant_w4a4
y = scaled_mm_svdquant_w4a4(act, wgt, ascales, wscales, lora_act_in, lora_up, group_size=64)
```

| Shared int4 pack/unpack + groupwise dequant | `kernels/quant/rdna4_int4_codec.py` |
| Shared helpers | `kernels/common/gfx120x_buf_helpers.py` |

### Compiler atoms (iu8 / iu4)

GFX120X integer WMMA in `include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td`,
`lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp`, Python bind in
`lib/Bindings/Python/FlyROCDLExtension.cpp`:

| Atom | Intrinsic | A/B packing |
|---|---|---|
| iu8 | `wmma_i32_16x16x16_iu8` | gfx12 integer WMMA (see atom tests) |
| iu4 | `wmma_i32_16x16x16_iu4` | **scalar i32** A/B (gfx12 ABI; not gfx11 `v2i32`) |

Device pin: `tests/kernels/test_rdna4_integer_wmma_atom.py` (alongside fp
`test_rdna4_wmma_atom.py`). FileCheck: `tests/mlir/Conversion/wmma_gfx120x.mlir`.

Native GEMM on iu4: `kernels/gemm/rdna4_iu4_gemm.py`. Unpack→iu8 remains a
valid W4 fallback when native packing does not apply.


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

### 4.2 Preshuffle layout (`kernels/common/mma/mfma_preshuffle_pipeline.py`)

Shared layout and block-remapping utilities for preshuffle GEMM and MoE kernels.

| Function | Description |
|---|---|
| `make_preshuffle_b_layout(...)` | Build B-preshuffle layout: (N/16, K/64, 4, 16, kpack_bytes) |
| `xcd_remap_bx_by(...)` | Remap blocks across XCDs and group tiles along M |

### 4.3 Layout coordinate helpers

Coordinate mapping in `flydsl.expr`:

| Function | Description |
|---|---|
| `fx.crd2idx(crd, layout)` | Coordinate → flat index (Fly dialect op) |
| `fx.idx2crd(idx, layout)` | Flat index → coordinate tuple (Fly dialect op) |
| `fx.get_(int_tuple, mode).unpack()` | Extract a scalar element at index from `!fly.int_tuple` |

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
│   ├── gfx120x / RDNA4 (Wave32 WMMA)
│   │   ├── FP8 scaled_mm (+ fused / tile auto) → kernels/gemm/rdna4_scaled_mm_fp8*.py
│   │   ├── Int8 fused act-quant + iu8 (+ host LoRA) → kernels/gemm/rdna4_int8_linear_fused.py
│   │   ├── iu8 int8 linear / W8A16 linear / size dispatcher → rdna4_int8_linear / rdna4_w8a16_linear / rdna4_int8_linear_dispatch*
│   │   ├── Native iu4 GEMM → rdna4_iu4_gemm (ConvRot default int4)
│   │   └── See kernels/common/gfx120x_autotune_tables.py
│   │
│   └── Uses new @flyc.kernel API
│       └── See kernels/gemm/preshuffle_gemm.py
│
├── MoE (Mixture of Experts)
│   ├── Blockscale MoE (gate+up+reduce)
│   └── Standard MoE (fp8/f16/bf16/int8/int4)
│       └── → kernels/moe/moe_gemm_2stage/
│
└── Building blocks
    ├── Common kernel helpers → kernels/common/kernels_common.py
    └── Preshuffle layout     → kernels/common/mma/mfma_preshuffle_pipeline.py
```

---

## 7. Source files

| File | Description |
|---|---|
| `kernels/gemm/preshuffle_gemm.py` | GEMM (preshuffle layout) |
| `kernels/moe/moe_gemm_2stage/` | MoE GEMM 2-stage (gate/up + reduce) |
| `kernels/moe/mxfp_moe/` | Fused a4w4/a8w4 MoE 2-stage GEMM (device fp4 re-quant) |
| `kernels/attention/pa_decode_fp8.py` | Paged attention decode (FP8) |
| `kernels/attention/flash_attn_generic.py` | FlashAttention generic fallback |
| `kernels/attention/flash_attn_gfx950.py` | FlashAttention gfx950 bf16/f16 fast path |
| `kernels/attention/flash_attn_fp8_gfx950.py` | FlashAttention gfx950 fp8 dense fast path |
| `kernels/attention/flash_attn_gfx120x.py` | FlashAttention gfx120x bf16, f16 |
| `kernels/attention/flash_attn_fp8_gfx120x.py` | FlashAttention gfx120x fp8 e4m3fn, e5m2 |
| `kernels/attention/flash_attn_int8_gfx120x.py` | FlashAttention gfx120x int8 QKV |
| `kernels/attention/flash_attn_iu4_gfx120x.py` | FlashAttention gfx120x native iu4 QKV |
| `kernels/attention/flash_attn_gfx120x_host.py` | FlashAttention gfx120x host routing |
| `kernels/attention/flash_attn_gfx120x_splitk.py` | gfx120x split-K combine (`build_splitk_combine_module`) |
| `kernels/attention/flash_attn_gfx120x_ext.py` | Host guards, bias slice, and sink LSE fold. Does not run attention. |
| `kernels/attention/_flash_attn_iu4_native_body_gfx120x.py` | Native iu4 dense body when bias, ALiBi, sink, and return_lse are unset |
| `kernels/norm/layernorm_kernel.py` | LayerNorm (layout API) |
| `kernels/norm/rmsnorm_kernel.py` | RMSNorm (layout API) |
| `kernels/norm/softmax_kernel.py` | Softmax (layout API) |
| `kernels/norm/softmax_bwd_kernel.py` | Softmax backward (layout API) |
| `kernels/norm/softmax_autotune.py` | Softmax opt-in autotune adopter |
| `kernels/attention/fused_rope_cache_kernel.py` | Fused RoPE + KV cache |
| `kernels/comm/custom_all_reduce.py` | Multi-GPU all-reduce |
| `kernels/gemm/rdna_f16_gemm.py` | RDNA FP16 GEMM |
| `kernels/gemm/rdna_fp8_preshuffle_gemm.py` | RDNA FP8 GEMM |
| `kernels/gemm/rdna4_scaled_mm_fp8.py` | gfx120x FP8 scaled_mm |
| `kernels/gemm/rdna4_scaled_mm_fp8_fused.py` | gfx120x FP8 scaled_mm fused (+ LoRA residual) |
| `kernels/gemm/rdna4_int8_linear_fused.py` | gfx120x int8 linear fused (+ LoRA host residual) |
| `kernels/gemm/rdna4_scaled_mm_fp8_auto.py` | gfx120x FP8 scaled_mm tile auto-picker |
| `kernels/gemm/rdna4_int8_linear.py` | gfx120x iu8 int8 linear |
| `kernels/gemm/rdna4_w8a16_linear.py` | gfx120x W8A16 linear (int8 / FP8 e4m3fn / e5m2 weights) |
| `kernels/gemm/rdna4_int8_linear_dispatch.py` | gfx120x W8A16 / iu8 size dispatcher (gate) |
| `kernels/gemm/rdna4_int8_linear_auto.py` | gfx120x int8 linear size-dispatcher (gate) entry |
| `kernels/gemm/rdna4_fused_mlp_nmajor.py` | gfx120x zero-LDS N-major / fused_gemm_TN + in-reg SwiGLU fuse |
| `kernels/gemm/rdna4_iu4_gemm.py` | gfx120x native iu4 WMMA GEMM |
| `kernels/norm/rope_gfx120x.py` | gfx120x RoPE (+ Q/K fused) |
| `kernels/norm/rms_rope_gfx120x.py` | gfx120x RMS+RoPE (+ Q/K fused) |
| `kernels/norm/adaln_gfx120x.py` | gfx120x AdaLN |
| `kernels/common/gfx120x_buf_helpers.py` | gfx120x buffer helpers (norm/quant/RoPE) |
| `kernels/quant/rdna4_fp8_quant.py` | gfx120x FP8 quant/dequant |
| `kernels/quant/rdna4_stoch_fp8.py` | gfx120x stochastic FP8 |
| `kernels/quant/rdna4_swiglu.py` | gfx120x SwiGLU |
| `kernels/quant/rdna4_quantize_int8_rowwise.py` | gfx120x int8 rowwise quant |
| `kernels/quant/rdna4_quantize_int8_tensorwise.py` | gfx120x int8 tensorwise quant |
| `kernels/quant/rdna4_int8_convrot.py` | gfx120x int8 ConvRot weight quant |
| `kernels/quant/rdna4_convrot_w4a4.py` | gfx120x ConvRot W4A4 weight quant |
| `kernels/quant/rdna4_asym_w4a8.py` | gfx120x Asym W4A8 dequant / host quant |
| `kernels/quant/rdna4_awq_w4a16.py` | gfx120x AWQ W4A16 dequant + fused GEMV |
| `kernels/quant/rdna4_svdquant_w4a4.py` | gfx120x SVDQuant W4A4 fused scaled_mm (packed, host LoRA) |
| `kernels/quant/rdna4_int4_codec.py` | gfx120x shared int4/uint4 pack + groupwise dequant |
| `kernels/common/gfx120x_autotune_tables.py` | gfx120x measured size/tile/BT defaults |
| `kernels/gemm/gemm_common_gfx1250.py` | GFX1250 GEMM common |
| `kernels/gemm/gemm_bf16_gfx1250.py` | GFX1250 BF16/FP16 GEMM |
| `kernels/gemm/gemm_a8w8_gfx1250.py` | GFX1250 FP8 GEMM (per-token/per-channel and 128x128 blockscale) |
| `kernels/gemm/gemm_a8w4_mxscale_gfx1250.py` | GFX1250 FP8 x MXFP4 GEMM |
| `kernels/common/mma/mfma_preshuffle_pipeline.py` | Preshuffle layout and block remapping |
| `kernels/gemm/fp8_gemm_utils.py` | FP8 GEMM helper utilities |
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
| `tests/kernels/test_flash_attn_gfx120x.py` | FlashAttention gfx120x (bf16, f16, fp8, int8, iu4) |
| `tests/kernels/test_layernorm.py` | LayerNorm |
| `tests/kernels/test_rmsnorm.py` | RMSNorm |
| `tests/kernels/test_softmax.py` | Softmax |
| `tests/kernels/test_softmax_bwd.py` | Softmax backward |
| `tests/kernels/test_softmax_autotune.py` | Softmax autotune selection and candidate correctness |
| `tests/kernels/test_fused_rope_cache.py` | Fused RoPE + KV cache |
| `tests/kernels/test_allreduce.py` | Multi-GPU all-reduce |
| `tests/kernels/test_rdna_gemm.py` | RDNA GEMM |
| `tests/kernels/test_rdna4_scaled_mm_fp8.py` | gfx120x FP8 scaled_mm |
| `tests/kernels/test_rdna4_scaled_mm_fp8_fused.py` | gfx120x FP8 scaled_mm fused |
| `tests/kernels/test_rdna4_int8_linear_fused.py` | gfx120x int8 linear fused |
| `tests/kernels/test_rdna4_scaled_mm_fp8_auto.py` | gfx120x FP8 scaled_mm size dispatcher (gate) |
| `tests/kernels/test_rdna4_int8_linear.py` | gfx120x int8 linear |
| `tests/kernels/test_rdna4_w8a16_linear.py` | gfx120x W8A16 linear |
| `tests/kernels/test_rdna4_int8_linear_dispatch.py` | gfx120x int8 linear size dispatcher |
| `tests/kernels/test_rdna4_int8_linear_auto.py` | gfx120x int8 linear size dispatcher (gate) |
| `tests/kernels/test_rdna4_fused_mlp_nmajor.py` | gfx120x zero-LDS N-major / fused_gemm_TN / in-reg SwiGLU |
| `tests/kernels/test_rdna4_integer_wmma_atom.py` | gfx120x iu8 / iu4 WMMA atom |
| `tests/kernels/test_gfx120x_norm_rope.py` | gfx120x RoPE / RMS+RoPE / AdaLN |
| `tests/kernels/test_rdna4_fp8_quant.py` | gfx120x FP8 quant |
| `tests/kernels/test_rdna4_stoch_fp8.py` | gfx120x stochastic FP8 |
| `tests/kernels/test_rdna4_swiglu.py` | gfx120x SwiGLU |
| `tests/kernels/test_rdna4_quantize_int8_rowwise.py` | gfx120x int8 rowwise |
| `tests/kernels/test_rdna4_quantize_int8_tensorwise.py` | gfx120x int8 tensorwise |
| `tests/kernels/test_rdna4_int8_convrot.py` | gfx120x int8 ConvRot |
| `tests/kernels/test_rdna4_convrot_w4a4.py` | gfx120x ConvRot W4A4 |
| `tests/kernels/test_rdna4_asym_w4a8.py` | gfx120x Asym W4A8 |
| `tests/kernels/test_rdna4_awq_w4a16.py` | gfx120x AWQ W4A16 dequant / fused GEMV |
| `tests/kernels/test_rdna4_svdquant_w4a4.py` | gfx120x SVDQuant W4A4 |
| `tests/kernels/test_rdna4_iu4_gemm.py` | gfx120x iu4 GEMM |
| `tests/kernels/test_rdna4_iu4_wmma_probe.py` | gfx120x iu4 WMMA probe |
| `tests/kernels/test_gfx120x_fa_routing_guards.py` | gfx120x FA host guards |
| `tests/kernels/test_gfx120x_dispatch_hosts.py` | gfx120x default-host launches |
| `tests/kernels/test_gfx120x_public_entries.py` | gfx120x CPU import check |
| `tests/mlir/Conversion/wmma_gfx120x.mlir` | gfx120x WMMA FileCheck |
| `tests/kernels/test_gemm_fp8fp4_gfx1250.py` | GFX1250 FP8/FP4 GEMM |
| `tests/kernels/test_gemm_bf16_gfx1250.py` | GFX1250 BF16/FP16 GEMM |
| `tests/kernels/test_vec_add.py` | Vector addition |
| `tests/kernels/test_quant.py` | Quantization utilities |
| `tests/kernels/benchmark_common.py` | Shared benchmark infrastructure |

### GFX120X iu4 WMMA (atom + GEMM)

See **Compiler atoms** above and the ConvRot / iu4 call examples in this guide
for packing and ConvRot default `linear_dtype='int4'`. AWQ / SVDQuant stay on
fused / unpack paths; ConvRot default remains native `linear_dtype='int4'`
(pass `linear_dtype='int8'` for unpack→iu8).
