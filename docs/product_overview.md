# Product overview

This page is a durable map of **what FlyDSL is, who it is for, and how it
reaches production**. Use it for onboarding, planning, and agent context.
Per-pass compiler details live in the [architecture guide](architecture_guide.md);
per-kernel parameters live in the [prebuilt kernels guide](prebuilt_kernels_guide.md).

FlyDSL (**F**lexible **l**ayout P**y**thon **DSL**) is a Python DSL and MLIR
compiler stack for authoring high-performance AMD GPU kernels with explicit
layout algebra, tiling, copy atoms, and MMA atoms. It targets ROCm/HIP through
the Fly and FlyROCDL dialects and lowers to ROCDL / HSACO.

The published package is `flydsl` on PyPI. Source of truth:
[github.com/ROCm/FlyDSL](https://github.com/ROCm/FlyDSL). HTML docs:
[rocm.github.io/FlyDSL](https://rocm.github.io/FlyDSL). Performance dashboard:
[rocm.github.io/FlyDSL/ci-dashboard](https://rocm.github.io/FlyDSL/ci-dashboard/).

---

## 1. Problem FlyDSL solves

NVIDIA has two mature kernel-authoring stacks:

| Stack | Strength | Cost |
|---|---|---|
| **CUTLASS / CuTe** | Explicit layout algebra: tiling, swizzle, and thread/value partitions are composable math | C++ templates |
| **Triton** | Python kernels with a compiler that infers much of the schedule | Less explicit control of layout and instruction placement |

On ROCm the historical alternatives are Composable Kernel (C++ templates),
hand-written ASM in AITER, and Triton-on-ROCm. FlyDSL takes CuTe-style explicit
layout algebra, exposes it as a Python JIT, and lowers through MLIR to
ROCDL / MFMA / WMMA / TDM.

**One-line product claim:** write AMD GPU kernels in Python with near
hand-written control of layout and data movement, then compile, cache, and
ship them into the ROCm inference stack.

FlyDSL wins when a framework needs a fused, quantized, paged, or MoE kernel
that a BLAS library cannot express and ASM is too slow to iterate. Triton is
usually a better fit for a short, portable kernel where the author does not
want to name layouts.

---

## 2. Product layers

| Layer | What it is | Who uses it |
|---|---|---|
| **FlyDSL** (`python/flydsl/`) | Python front-end: `@flyc.kernel` / `@flyc.jit`, expression API `fx.*` | Kernel authors, AITER |
| **Fly dialect** (`include/`, `lib/`) | MLIR layout IR: `!fly.layout`, composition / product / divide, coordinate mapping | Compiler pipeline |
| **`kernels/`** | Importable production kernels (GEMM, attention, MoE, norm, conv, comm) | Framework integration, in-tree benchmarks |
| **Embedded `_mlir`** | MLIR Python runtime shipped in the wheel | `pip install flydsl` users (no separate MLIR wheel) |

`python/flydsl/expr/` is **target-neutral**. ROCm-specific code (MFMA, WMMA,
buffer copy, TDM, cluster) lives in `python/flydsl/expr/rocdl/` (`cdna3`,
`cdna4`, `cdna5`, `rdna3`, `rdna4`, …) and is lazy-loaded. The only fully
landed backend today is ROCm.

Public API stability for `python/flydsl/` is defined in
[API stability](api_stability.md). The project is still in the **0.x** series
(version in `python/flydsl/__init__.py`); a patch release must not break a
stable API, and a minor release must keep previously valid stable call paths.

---

## 3. Compilation path

Authors write Python. The first call traces and compiles; later calls with the
same type signature reuse the JIT cache (default `~/.flydsl/cache`).
`Constexpr` values are part of the cache key, so tile size and dtype variants
compile to distinct binaries.

```text
Python (@flyc.kernel / @flyc.jit)
  → AST rewrite + tracing
  → Fly / gpu / arith / scf MLIR
  → Stage A: Fly → ROCDL (layout lowering, atom→SSA, MFMA/WMMA/TDM)
  → Stage B: → LLVM
  → Stage C: gpu-module-to-binary → HSACO fatbin
  → JIT cache + ExecutionEngine
```

The pass list is built by `RocmBackend` in
`python/flydsl/compiler/backends/rocm.py`. See the
[architecture guide](architecture_guide.md) for the per-pass table, JIT
host vs device split (`@flyc.jit` / `@flyc.kernel`), and environment
variables.

Preferred kernel-authoring surface (not the legacy byte-offset path):

- `fx.rocdl.make_buffer_tensor()` plus layout ops
- `fx.copy` / `fx.gemm` (prefer these over `copy_atom_call` / `mma_atom_call`)
- `SharedAllocator` (`fx.SharedAllocator`) for LDS

---

## 4. Differentiator: layout is a first-class object

The product is not “HIP with decorators”. The CuTe layout algebra is the
core abstraction:

- **Shape / Stride / Layout / Coord**, with `Index = dot(Coord, Stride)`
- **composition / product / divide** for tiling, partition, and swizzle
- **Copy atom / MMA atom / TiledCopy** so one `ds_read`, one `mfma`, or one
  buffer load is a composable atom, then spread with a thread-value layout

Unlike Triton, FlyDSL expects the author to make **block / warp / thread /
instruction** partitions explicit. The cost is a steeper learning curve. The
payoff on CDNA is precise MFMA scheduling, LDS swizzle, and prefetch — close
to ASM, in Python.

Layout math: [layout system guide](layout_system_guide.md) and
[CuTe layout algebra](cute_layout_algebra_guide.md).
Authoring constraints (single exit path, loop-carried state, LDS view
cache): [kernel authoring guide](kernel_authoring_guide.md).

---

## 5. Hardware matrix

| Arch | Hardware | Wave | MMA | Role |
|---|---|---|---|---|
| **gfx942** | MI300X / MI308X | 64 | MFMA | CDNA3 baseline; preshuffle GEMM, paged-attention decode |
| **gfx950** | MI350 / MI355X | 64 | MFMA | CDNA4; FP8 / FP4, scale MMA, 160 KB LDS; current serving flagship |
| **gfx11\*** | RDNA3 / Strix Halo (e.g. gfx1151) | 32 | WMMA | No MFMA; no native FP8 |
| **gfx120\*** | RDNA4 (e.g. gfx1201) | 32 | WMMA | Native FP8; v8-operand WMMA ABI |
| **gfx1250** | — | 32 | WMMA / TDM | FP8 / FP4, 320 KB LDS, async TDM copy |

For inference serving, CDNA3/4 is the throughput path. RDNA covers client /
workstation. gfx1250 kernels (GEMM, MoE, TDM) are forward-looking, not the
default vLLM / SGLang path today.

`tests/arch_compat.py` is the source of truth for which examples and tests
are RDNA-compatible versus CDNA-only.

---

## 6. Production kernel catalog

`kernels/` is the in-tree delivery surface (import as `kernels.*`). Search
that tree before editing; this table is routing, not an inventory. Public
builder docs: [prebuilt kernels guide](prebuilt_kernels_guide.md).

### Attention

| Area | Entry | Notes |
|---|---|---|
| Paged decode | `kernels/attention/pa_decode_fp8.py` | Sliding-window is a backend mode, not a separate public entry |
| FlashAttention fwd | `kernels/attention/flash_attn_generic.py`, `flash_attn_gfx950.py` | gfx950 + `head_dim==128` dual-wave SWP fast path |
| FlashAttention FP8 | `kernels/attention/flash_attn_fp8_gfx950.py` | gfx950, `D=128`, dense, pre-quantized e4m3fn + descale ABI |
| MLA decode | `kernels/attention/mla_fwd_decode.py` | |
| RoPE / KV | `kernels/attention/fused_rope_cache_kernel.py`, `qk_norm_rope_quant.py` | |

Start paged-attention changes in `pa_decode_fp8.py` and
`tests/kernels/test_pa.py`.

### GEMM

Choose by architecture and dtype first (CDNA MFMA, RDNA wave32 WMMA, gfx1250
TDM / WMMA / MX-scale):

- CDNA preshuffle: `kernels/gemm/preshuffle_gemm.py` (fp8 / int8 / fp16 / bf16;
  B preshuffle, ping-pong LDS, XOR swizzle)
- MX / FP4: `mxfp4_preshuffle.py`, `fp4_gemm_4wave.py`
- FP8 4/8-wave, gfx950 A16W16, gfx1250 FP8/FP4/WMMA
- RDNA: `rdna_f16_gemm.py`, `rdna_fp8_preshuffle_gemm.py`, `rdna3_int8_gemm.py`

### MoE

- Two-stage MoE GEMM: `kernels/moe/moe_gemm_2stage/`
- MXFP fused a4w4 / a8w4: `kernels/moe/mxfp_moe/`
- Sorting and top-k gating softmax
- `kernels/mega_moe/` — newer fused path
- gfx1250 a8w4 mxscale

### Other

- LayerNorm / RMSNorm (fwd + bwd) / Softmax (fwd + bwd, opt-in autotune)
- Implicit-GEMM 3D conv: `kernels/conv/`
- Multi-GPU all-reduce: `kernels/comm/custom_all_reduce.py` (8-GPU CI job;
  not in the default `scripts/run_tests.sh` flow)

In-tree benchmarks compare against **AITER ASM / Triton**, not abstract peak.
See [testing and benchmarking](testing_benchmarking_guide.md).

---

## 7. How kernels reach frameworks

Frameworks do not typically import `kernels.*` directly. The production chain
is:

```text
FlyDSL (this repo, pip package flydsl)
    → AITER (ROCm/aiter) depends on flydsl
    → vLLM / SGLang ROCm paths call AITER
```

Nightly and wheel machinery that encodes this contract:

| Workflow | Role |
|---|---|
| `.github/workflows/flydsl.yaml` | In-tree correctness + AITER performance comparison |
| `.github/workflows/flydsl-vllm-integration.yaml` | Nightly: build AITER against `rocm/vllm-dev:nightly`, run vLLM |
| `.github/workflows/flydsl-sglang-integration.yaml` | Nightly: SGLang on MI355×8 (`linux-flydsl-mi355-8`, `GPU_ARCH=gfx950`) |
| `.github/workflows/publish-pypi.yaml`, `promote.yaml`, `test-whl.yaml` | Wheel publish and promotion |

**Product success, in this order:**

1. AITER can pin a new `flydsl` wheel without breaking its kernel surface.
2. vLLM / SGLang paged attention, GEMM, MoE, and norm do not regress.
3. Arch dispatch is correct: gfx942 (MI300X) must not regress while gfx950
   (MI355X) takes CDNA4 instructions.

---

## 8. Adjacent ROCm components

| Component | Role | Relation to FlyDSL |
|---|---|---|
| **AITER** | ROCm inference kernel aggregation (ASM + Triton + FlyDSL) | Direct downstream; CI clones it as baseline |
| **Triton** | Higher-level Python kernels | Counterpart: less explicit layout/atom control |
| **Composable Kernel (CK)** | C++ template kernel library | Parallel hand-written path; GEMM docs compare CK-style schedules |
| **hipBLASLt** | Library GEMM | Frameworks can call the library; FlyDSL is for custom fused kernels |

---

## 9. Maturity signals

Already product-like:

- Documented stable API (`docs/api_stability.md`)
- Test tiers L0 → L1a → L1b → L2 device; `large_shape` is the CI full suite
- Autotune (`@autotune` / `Config`); softmax and RMSNorm adopt it
- External LLVM bitcode link-in (`ffi` + `link_extern`)
- Pre-checks: black + ruff, clang-format, `scripts/check_repo.py` (including
  docs/API path checks for agent-facing files)

Still early:

- **0.x** versioning; not a 1.0 compatibility promise beyond the stability doc
- Kernel trees in some older docs lag the repo (for example `mega_moe` and
  extra PA / FA variants) — treat `kernels/` as source of truth
- Authoring rules are strict (merged values after branches, explicit
  `scf.for` state, no mutation of captured variables in nested helpers)
- `expr/rocdl` and `kernels/` fork quickly per arch; **arch coverage vs a
  single public API** remains a standing tension

---

## 10. How to read this product later

When planning work or reviewing a change, look at these five axes:

1. **Platform** — layout algebra, atoms, JIT cache, arch dispatch, stable API
2. **Kernel shelf** — attention decode/prefill, FP8/FP4 GEMM, MoE, norm:
   does it cover vLLM / SGLang hot paths?
3. **Silicon** — gfx942 must not regress; gfx950 should use CDNA4; RDNA /
   gfx1250 only if the change is in the external commitment
4. **Integration health** — PyPI / nightly wheel → AITER pin → vLLM / SGLang
   nightly
5. **Perf narrative** — ratio vs AITER ASM / Triton, not absolute TFLOPS

Deeper how-tos from here:

| Need | Doc |
|---|---|
| Pass pipeline, JIT, env vars | [Architecture](architecture_guide.md) |
| Shape / stride / layout ops | [Layout system](layout_system_guide.md) |
| Writing a kernel | [Kernel authoring](kernel_authoring_guide.md) |
| Tiling, LDS, prefetch, MFMA | [Kernel tuning](kernel_tuning_guide.md) |
| Builder APIs and dtypes | [Prebuilt kernels](prebuilt_kernels_guide.md) |
| Pytest markers, FileCheck | [Testing](testing_benchmarking_guide.md), `tests/README.md` |
