# Shared/reuse MLA + MoE kernel

[中文](PERFORMANCE_zh.md)

This directory contains a FlyDSL implementation of one fixed GLM-5 MLA + MoE
shard. The production path is exposed as `SharedReuseMlaMoeLayer` and generated
by `build_shared_reuse_kernel`. It contains no imported TileRT kernel body or
embedded assembly. The former assembly capture, rewriting, and launch bridge
has been deleted.

TileRT remains an optional benchmark dependency in `native_baseline.py`. That
adapter converts the same generated tensors to the released TileRT wrapper so
the two implementations can be compared directly. It is not imported by the
FlyDSL execution path.

## Fixed shard and arithmetic modes

Every rank uses eight attention heads, expert intermediate size 256, hidden
size 6144, 256 routed experts, one shared expert, and top-8 routing. The 2- and
4-GPU measurements therefore exercise the same fixed shards with smaller peer
groups; they are not full-model TP2 or TP4 shapes.

The public `MoeMode` values are:

| Mode | Expert activation | Expert weight | Up/gate to down handoff |
|---|---|---|---|
| `w8a8` | dynamic FP8 E4M3, per 128 | block-scaled FP8 E4M3 | FP8 |
| `w8a16` | BF16 | block-scaled FP8 E4M3 | BF16 |
| `a16w4` | BF16 | MXFP4, per-1x32 E8M0 | BF16 |
| `a8w4` | MXFP8 E4M3, per-1x32 E8M0 | MXFP4, per-1x32 E8M0 | MXFP8 |

Attention weights stay block-scaled FP8 in all modes. Supported sample counts
are 1, 2, 4, and 8; supported peer counts are 1, 2, 4, and 8. The host wrapper
validates the complete fixed-shard contract before allocating GPU buffers.

## ATOM storage-layout audit

The MXFP4 modes now default to the storage layouts that can be consumed directly
from ATOM without repacking the expert tensors:

| Tensor | Logical order | Physical storage in the default MXFP4 path | ATOM compatibility |
|---|---|---|---|
| `w_ug` | 256 routed experts followed by shared expert 256; gate rows then up rows | AITER `shuffle_weight(..., layout=(16, 16), is_guinterleave=False)` | Exact |
| `w_dn` | Same expert numbering; output rows by input-intermediate columns | Same AITER 16x16 shuffle | Exact |
| `s_ug`, `s_dn` | One E8M0 byte per 1x32 block | AITER non-interleaved `shuffle_scale` | Exact |
| `w_r` | 256 BF16 router rows | Mono-kernel MFMA packing by default; optional ATOM row-major | Logical match; default is not zero-copy |
| MLA projection weights | `qkv_a=[q_a; kv_a]`; each `q_b` head is `[nope; rope]` | Existing block-128 FP8 MFMA packing | Not compatible with the BF16 attention weights in `amd/GLM-5.1-MXFP4` |
| KV cache | Per-token `[k_c(512), k_pe(64)]` | One contiguous BF16 `[tokens, 576]` tensor | Exact shape, stride, and dtype in the matched benchmark; ATOM serving can select FP8 by configuration |

The expert value and scale packers were compared byte-for-byte against the
current AITER `shuffle_weight` and `shuffle_scale` implementations for both
GLM-5 up/gate and down shapes. Norm vectors and routing bias are already plain
contiguous tensors. ATOM's default `ATOM_MOE_GU_ITLV=0` also matches this
kernel's gate-then-up row order.

The matched ATOM layer exposes logical expert tensors with shapes
`w13=[257,512,3072]`, `w13_scale=[257,512,192]`, `w2=[257,6144,128]`, and
`w2_scale=[257,6144,8]`. After AITER postprocessing, the value shapes are
unchanged and the scale buffers become `[131584,192]` and `[1579008,8]`.
These shapes and the bytes in each buffer match the FlyDSL default MXFP4 path.

`--router-weight-layout atom` is retained for zero-copy experiments. It is not
the default: with otherwise ATOM-compatible A16W4 storage, TP1 S=1 improved
slightly from 31.19 to 31.08 us/layer, but S=8 regressed from 70.80 to 72.47
us/layer (2.4%). The standalone ATOM GEMM can tile a row-major router internally;
the mono-kernel's direct MFMA loads instead cross the 6144-element row stride.

The attention-format difference, and any configured serving cache-dtype
difference, are not permutation-only issues. This change therefore does not
claim complete zero-copy loading of an `amd/GLM-5.1-MXFP4` checkpoint. It
aligns every MXFP4 expert value/scale and the 576-wide cache allocation while
preserving the faster existing attention and router paths.

## Code layout

| File | Responsibility |
|---|---|
| `config.py` | Model profiles, public arithmetic/storage modes, and host validation. |
| `../common/mx_formats.py` | Reusable Torch MXFP4/MXFP8 quantization and dequantization. |
| `packing.py` | Model-aware attention packing plus native and ATOM/AITER-compatible expert packing. |
| `kernel_common.py` | Low-level helpers shared by model-configured kernels. |
| `kernel_layout.py` | Model-configured scratch, peer-buffer, and stage-layout calculations. |
| `shared_reuse_moe_kernel.py` | Performance-specialized GLM-5 scheduling, communication, MLA, routing, and experts. |
| `layer.py` | Stable host wrapper for the tuned GLM-5 mono-kernel. |
| `indexed_mla_moe_kernel.py` | Extensible indexed MLA + MoE kernel parameterized by `LayerConfig`. |
| `indexed_layer.py` | Generic indexed wrapper plus the GLM-5 compatibility and Kimi-K3 MLA adapters. |
| `kimi_k3.py` | Kimi-K3 full-layer adapter composing indexed MLA with latent MoE and reductions. |
| `kda.py` | Production-shape TP8 KDA attention and its explicit state-pool API. |
| `kda_conv.py`, `kda_recurrence.py` | Slot-indexed KDA convolution and fused recurrence/gated-RMSNorm kernels. |
| `kimi_k3_attn_res.py` | Fused Kimi-K3 AttnRes mixing, RMSNorm, and MXFP8 input quantization. |
| `router_projection.py`, `mxfp8_linear.py` | Fused router/top-k plus reusable gfx950 MXFP8 projection support. |
| `kimi_k3_tail.py` | Overlapped routed reduction/RMSNorm, MXFP8 shared and latent projections, and final TP reduction. |
| `router.py`, `router_projection.py` | Reusable native routing and fused projection/top-k kernels. |
| `runtime.py`, `../common/hip_ipc.py` | Shared symmetric HIP IPC lifecycle and deterministic remote-handle cleanup. |
| `symmetric_allreduce.py`, `torch_fusions.py` | Graph-safe reductions and compiled Torch output helpers. |
| `reference.py` | Independent Torch stage and end-to-end calculations. |
| `native_baseline.py` | Optional same-weight TileRT comparison adapter. |
| `tools/benchmark_atom.py` | Native ATOM GLM-5.1 decoder-layer benchmark with preselected sparse indices. |
| `tools/kimi_k3_full.py` | Kimi-K3 correctness, profiling, and full-layer benchmark driver. |
| `tools/kimi_k3_kda_full.py` | KDA attention and KDA + latent-MoE TP8 correctness/performance driver. |

The kernel uses FlyDSL operations for wave reductions, hardware math,
mailbox polling, buffer access, and MFMA issue. Peer payloads are rounded to
BF16 and accumulated in rank order so every rank produces exactly the same
hidden state and routing decisions.

`SharedReuseMlaMoeLayer` owns its remote HIP IPC mappings. Call `close()` after
the last rank barrier, or use it as a context manager.

## Correctness status

Both `w8a8` and `w8a16` passed the full 2/4/8-GPU by S=1/2/4 matrix. Each of
the nine configurations ran five changing inputs and checked:

- stage outputs against the independent Torch calculations;
- exact final-output agreement across ranks;
- the final down projection and BF16 peer reduction;
- finite outputs and stable HIP graph replay.

The independent FP32 end-to-end check passed 41 inputs at 1.49-2.85% relative
L2. Four inputs used the existing near-tied-routing skip rule because a one-BF16-
ulp attention difference changed the selected expert set; those inputs still
passed the stage checks and exact rank agreement. Existing tolerances were not
relaxed. One NP2/S4 `w8a16` intermediate used the existing one-BF16-ulp bound,
while its final down/output check matched exactly.

S=8 also passed the complete stage suite for `w8a8` and `w8a16` on 1, 2, 4,
and 8 GPUs with one fresh input per configuration. The larger peer payload uses
two 64-lane send batches; exact output agreement across ranks was retained.
The NP4 `w8a16` normalized expert input differed from the independent reduction
by one BF16 ulp on one element, within the existing BF16 handoff bound.

The new `a16w4` and `a8w4` modes passed the complete stage suite at S=1 and
S=8 on one GPU, and at S=8 on two and eight GPUs. The checks covered the
packed MXFP4 weight path, per-1x32 E8M0 scales, A8W4 activation quantization,
the final BF16 peer reduction, and exact final-output agreement across ranks.
At TP8/S8, independent end-to-end relative L2 was 0.430% for `a16w4` and
2.87% for `a8w4`. Existing tolerances were retained.

The ATOM-layout path additionally passed default-layout TP1/S1, TP1/S8, and
TP8/S8 checks for both MXFP4 modes. A8W4 S>1 initially exposed
a shared-expert-only scale-association bug: all eight routed slots were correct,
while shared slot 0 was not. The final implementation keeps the direct
lane-group ATOM load for routed experts and uses canonical K32 gathers only for
the shared expert whose MFMA columns represent different samples.

A direct TP1/S1 output comparison against the same-weight TileRT wrapper gave:

| Mode | Maximum absolute error | Relative L2 |
|---|---:|---:|
| `w8a8` | 0.1171875 | 2.263% |
| `w8a16` | 0.03125 | 0.357% |

## Performance status

### MXFP4: FlyDSL mono-kernel versus native ATOM layer

The primary MXFP4 comparison runs on 8 x MI355X (gfx950), TP8, at position
3000 with sparse top-2048 and a BF16 `[tokens,576]` KV cache. Each HIP graph
contains one decoder-layer invocation. After two warmup replays, the benchmark
records 30 replays, takes the slowest rank for every replay, removes the five
fastest and five slowest samples, and reports the mean of the remaining 20.

ATOM registers GLM-5.1 as `GlmMoeDsaForCausalLM`. That production model reuses
`atom.models.deepseek_v2` and constructs `DeepseekV2DecoderLayer`, so the
baseline directly instantiates that exact GLM-selected layer class with the
GLM-5.1 config; the inherited class name does not mean a DeepSeek model config
is used. Its native multi-operator chain includes RMSNorm, BF16 MLA projections
and sparse attention, router, MXFP4 FusedMoE, and TP collectives. Only the
indexer that produces sparse top-2048 indices is bypassed, because the FlyDSL
API also receives those indices as input. The revisions are ATOM
`b104cf915aeced8c0c319fe1e0fcf6cf70b323ba` and AITER
`d4e9afc85857e30e03b417606e6f25071ffaaa9a`.

| Batch | FlyDSL mono-kernel | ATOM native layer | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 74.77 us | 136.84 us | 1.83x | 45.4% |
| 2 | 73.69 us | 145.43 us | 1.97x | 49.3% |
| 4 | 94.77 us | 144.12 us | 1.52x | 34.2% |
| 8 | 124.66 us | 158.99 us | 1.28x | 21.6% |

There is no measured regression against ATOM at any tested batch, so no
attention/router-MoE/collective regression split was required. This is an
equivalent-layer latency comparison, not a pure fusion-only comparison:
experts and KV storage use the matched MXFP4/BF16 formats, but the released
`amd/GLM-5.1-MXFP4` ATOM attention weights are BF16 while the FlyDSL
mono-kernel keeps its block-128 FP8 attention path.

### Historical and layout A/B measurements

The established W8A8 graph measurements below use 128 layer launches per HIP
graph, two warmups, nine measured replays, and the median critical-rank time.
Hardware was 8 x MI355X (gfx950), with position 3000 and sparse top-2048.

| GPUs | Backend | S=1 | S=2 | S=4 |
|---:|---|---:|---:|---:|
| 2 | FlyDSL | 33.98 us | 39.80 us | 53.55 us |
| 4 | FlyDSL | 34.30 us | 41.51 us | 54.42 us |
| 8 | FlyDSL | 35.55 us | 42.72 us | 56.32 us |
| 8 | TileRT | 35.85 us | 42.90 us | 55.93 us |

FlyDSL is faster in the measured eight-GPU S=1 and S=2 cases and 0.7% slower
at S=4. It has therefore not beaten TileRT in every configuration.

A short TP1/S1 smoke measurement using eight layers and one measured replay
gave 33.840 us versus 33.520 us for `w8a8`, and 34.735 us versus 32.895 us for
`w8a16`. These short runs verify the benchmark path and are not publication-
quality latency results.

A separate 16-layer, three-replay TP1 run measured W8A8 at 52.03 us for S=4
and 84.20 us for S=8. TileRT has no S=8 whole-layer baseline.

After adding the MXFP4 paths, a same-process 16-layer, three-replay TP1 run
measured the following medians. These are short development measurements, not
publication-quality results:

| Mode | S=1 | S=8 |
|---|---:|---:|
| `w8a8` | 33.38 us | 83.46 us |
| `w8a16` | 34.63 us | 90.13 us |
| `a16w4` | 33.22 us | 78.80 us |
| `a8w4` | 35.21 us | 87.00 us |

The `a16w4` path was fastest in this short TP1 comparison. `a8w4` still pays
for per-1x32 activation quantization and BF16 MFMA staging, so it did not beat
`w8a8` in both sample counts. The released TileRT comparison adapter accepts
only `w8a8` and `w8a16`; there is no valid same-weight TileRT baseline for the
two MXFP4 modes in this harness.

As secondary evidence for the storage-layout change, TP1 used 128 layer
launches per graph, two warmups, 15 measured replays, and median latency.
Native measurements bracketed the ATOM expert/cache run to expose drift; the
percentage uses the mean of the two native medians.

| Mode | Samples | Native expert/scale + split cache | ATOM expert/scale + fused cache | Change |
|---|---:|---:|---:|---:|
| `a16w4` | 1 | 31.80-32.33 us | 31.14 us | 2.9% faster |
| `a16w4` | 8 | 74.65-75.04 us | 71.37 us | 4.6% faster |
| `a8w4` | 1 | 32.98-33.02 us | 32.08 us | 2.8% faster |
| `a8w4` | 8 | 81.64-81.80 us | 77.21 us | 5.5% faster |

The lightweight TP8 graph benchmark uses 16 launches per graph and reports the
slowest rank. It also showed no regression:

| Mode | Samples | Native | ATOM expert/cache default |
|---|---:|---:|---:|
| `a16w4` | 1 | 35.2 us | 34.8 us |
| `a16w4` | 8 | 87.2 us | 82.1 us |
| `a8w4` | 1 | 36.9 us | 35.4 us |
| `a8w4` | 8 | 92.9 us | 87.5 us |

The first direct implementation of the ATOM expert layout used scattered
dword loads and register-side transposition; A16W4 S=8 rose to roughly 115 us.
Mapping the four lane groups directly to ATOM's four K32 tiles removed that
regression. ATOM scale storage and the fused cache were then kept with the
optimized value path. For A8W4 multi-sample shared-expert computation, the
small canonical-load fallback described above preserves correctness while the
final end-to-end result remains 5.5% faster than the native layout at TP1/S=8.

Segment traces guided two retained scheduling changes: BF16-packed peer
exchange and one sample per router CTA. For the earlier S=4 schedule, the last
instrumented CTA reached attention publication, router publication, routed
up/gate publication, and down completion at 30.23, 35.12, 50.12, and 57.52 us.
After the changes, those marks were 27.66, 30.65, 45.73, and 52.97 us. The
comparison plot is `/root/glm5-perf-results/s4-segment-milestones.png`.

## Reproduce

Use the existing FlyDSL compiler build and this worktree:

```bash
cd /root/FlyDSL-glm5-mxfp4-atom-layout
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-mxfp4-atom-layout
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode a16w4 --npes 8 --samples 1 2 4 8 \
  --layers 1 --repeats 30 --trim 5

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark_atom.py \
  --npes 8 --samples 1 2 4 8 --layers 1 --repeats 30 --trim 5
```

Repeat the correctness command for the other modes and peer counts.

The ATOM benchmark defaults to `/root/ATOM` and `/root/aiter`; override those
paths with `--atom-root` and `--aiter-root` when needed. Run GPU jobs
sequentially.

Add `--trace --layers 16 --trace-dir <directory>` to a FlyDSL benchmark for
stage timestamps, then inspect a rank with:

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  <directory>/w8a8-s4/rank0/trace.pt
```

Trace instrumentation drains memory operations and changes scheduling. Use
uninstrumented graph measurements for latency comparisons.

## Kimi-K3 full MLA + latent-MoE layer

`KimiK3MlaLayer` remains the reusable full-attention component. The new
`KimiK3MlaMoeLayer` implements the complete production-TP8 decoder-layer data
path used by Kimi-K3:

- hidden size 7168, 1536 Q-LoRA rank, and 512 KV-LoRA rank;
- 12 local heads with 128 non-positional, 64 RoPE, and 128 value dimensions;
- BF16 attention weights and the sigmoid attention-output gate;
- 12-layer AttnRes source mixing before attention and before MoE;
- BF16 router projection with FP32 sigmoid/correction-bias selection, 896 experts,
  and normalized top-16;
- replicated MXFP8 7168-to-3584 latent projection;
- FlyDSL device-side sorting and two-stage A16W4/MXFP4 routed experts with SiTU;
- TP8-local MXFP8 shared experts, latent RMSNorm, and rank-local MXFP8
  3584-to-896 tail;
- one TP reduction in latent space and one final TP reduction before the
  residual update.

The A16W4 launcher and tuned Kimi-K3 configuration now live under
`kernels/moe/moe_2stage_a16wmix/host.py`; tests import that production module
instead of owning the host implementation.

Correctness was checked on 8 x MI355X (gfx950) for S=1/4/8 at layer 0, plus
S=4 at layer 1 and layer 12 to cover both non-write and new-block AttnRes
branches. All outputs were finite and bit-identical across the eight ranks.
In the final optimized S=4/S=8 runs, using the implementation's own
post-attention state, top-16 selection had zero mismatches, routed-MoE relative
L2 was 0.549%/0.538-0.551%, full-output relative L2 was approximately
0.469%/0.466-0.474%, and KV-cache relative L2 was approximately 1e-8. Five
additional fresh-process S=8 checks retained exact rank agreement and did not
reproduce the one earlier transient high-error sample. Layer 0/1/12 HIP-graph
capture and replay also completed.

As in the existing GLM-5 tests, the independent end-to-end comparison records
but does not fail on a near-tied synthetic route changed by a legal upstream
MLA rounding difference. The isolated MoE check feeds the implementation's
post-attention tensor to the independent MoE reference, so it distinguishes a
real router/expert regression from this synthetic boundary effect.

The MLA component retains its two scheduling optimizations: S >= 2 reuses one
staged 64-key KV tile and one 16-column score MFMA for both local head groups,
and the output gate is fused into each W_UV producer. The attention-only TP8
improvements remain 0.6%, 2.3%, and 12.0% for S=1, S=4, and S=8 respectively.

### Synthetic MLA tuning performance

The original synthetic `layer_idx=0` MLA harness was measured with TP8,
position 3000, 16 layer launches per HIP graph, two eager warmup forwards, two graph warmup replays,
100 measured replays, and the median critical-rank time. All eight GPUs were
checked for competing processes immediately before the run. The final path
uses fused BF16 router/top-16 plus MXFP8 latent/shared projections, `bm16`
MXFP4 routed-MoE tiles, fused communication/RMSNorm/tail work, and graph-safe
symmetric peer mailboxes.

| Version | S=4 | S=8 |
|---|---:|---:|
| Initial correct sequential full layer | 383.85 us | 404.81 us |
| Common-module source before the deep tuning pass | 258.7604 us | 282.4305 us |
| Before fused router projection | 204.4796 us | 228.6948 us |
| Fused-router source before MXFP8/tail integration | 194.9370 us | 221.5046 us |
| Final MXFP8/overlapped source | **109.4035 us** | **124.0237 us** |
| Speedup from the common-module source | **2.37x** | **2.28x** |

The earlier controlled NCCL-versus-symmetric-reduce comparison changed S=4
from 258.9820 us to 256.3328 us and S=8 from 281.1871 us to 279.5571 us, only
about 0.6-1.0%. Communication replacement alone was therefore insufficient.
The final pass addressed launch overhead, under-occupancy, redundant memory
traffic, and overlap together:

- The routed experts use the existing tuned `bm16` MXFP4 GEMM path, matching
  ATOM's low-token choice. This reduced S=4/S=8 from 235.29/255.39 us to
  226.59/247.25 us at that point in the tuning pass.
- `mxfp8_linear.py` follows the repository's gfx950 scaled-MFMA preshuffle
  layout but specializes scheduling for M=1/2/4/8. Post-AttnRes produces the
  quantized activation once; latent-down and shared up/gate consume that same
  packed input.
- `router_projection.py` co-locates BF16 router projection/top-16 with both
  MXFP8 projections. Four active projection waves per 512-thread CTA increase
  the number of independent projection CTAs; the isolated latent/shared pass
  improved from about 24.9 us to 16.2 us. Sample-specific CTA sizing reduces
  the complete fused kernel to 22.44 us at S=4 and 23.36 us at S=8.
- K3's `S <= 8`, top-16 contract means every active expert needs exactly one
  padded 16-row tile. An LDS atomic histogram and per-route atomic tickets
  replace repeated route scans, removing roughly 8 us from the S=8 full layer.
- `kimi_k3_tail.py` overlaps routed TP reduction/RMSNorm with independent
  shared-down compute. Shared partials are sent to their output owner early;
  each owner then adds its rank-local latent-up shard and broadcasts the final
  value. This collapses two projections, two communication phases, RMSNorm,
  accumulation, and the residual update into a 14.97-us/16.83-us kernel. The
  shared-down and latent-up MXFP8 weights and scales now use the same
  ATOM/AITER preshuffle as the other MXFP8 projections, including 256-row scale
  padding. Against the immediately preceding build, this layout change reduced
  full-layer S=4/S=8 latency from 111.3647/126.0886 us to
  109.4035/124.0237 us.
- Dedicated AttnRes kernels keep the mixed state in one launch and quantize the
  post-attention output directly for MXFP8 consumers. The post-AttnRes kernel
  measures 8.22 us at S=4 and 8.51 us at S=8.

The indexed attention path also accepts an ATOM-compatible fused BF16
`[tokens,576]` KV/PE cache through `--kv-cache-layout atom`. It is numerically
identical to the split-cache path; its measured difference was about 0.1 us,
so the split layout remains the default.

For the isolated 7168-to-3584 projection, a 16-launch HIP graph measured the
MXFP8 kernel at 7.05 us for M=4 and 7.75 us for M=8. The corresponding BF16
`torch.mm` calls were 11.86 us and 12.09 us, so the specialized MXFP8 GEMM is
1.68x and 1.56x faster while staying near `2e-4` relative L2 to dequantized
MXFP8 matmul.

The router projection and correction bias now also use BF16, matching ATOM's
production gate contract; FP32 remains confined to sigmoid, comparison, and
route normalization. This dtype correction did not materially change latency,
but removes an implementation mismatch from the comparison.

The requested comparison against the original ATOM implementation uses ATOM
commit `3cea04f45` and directly instantiates its production
`atom.models.kimi_k3.KimiDecoderLayer`, including AttnRes, MLA, routing,
routed/shared MoE, latent transforms, TP reductions, dual streams, and HIP
graph replay.

| Batch | FlyDSL full layer | ATOM full layer | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 4 | **109.4035 us** | 225.6496 us | **2.063x** | **51.52%** |
| 8 | **124.0237 us** | 256.5222 us | **2.068x** | **51.65%** |

These are observed end-to-end timings for a synthetic MLA geometry, not a
production layer-0 result. The attention work is also not identical. ATOM's
`KimiFullAttention` scans a dense 3001-token KV context,
whereas FlyDSL consumes caller-supplied top-2048 KV indices at position 3000.
The MoE and hidden/model shapes, TP8 topology, graph length, warmups, repeats,
and critical-rank timing rule match. Therefore the table is useful as a direct
implementation-level full-layer baseline, but its delta is not a normalized
same-attention-work kernel comparison.

### Production MLA and KDA layer families

Production Kimi-K3 uses MLA at zero-based layer 3 and every fourth layer after
that, while most remaining layers use KDA. Layer 0 is KDA plus a dense FFN and
is intentionally outside the current scope. The KDA path now implements the
BF16 fused input projection, low-rank `f_a -> f_b` gate, slot-indexed causal
convolution, FP32 recurrent delta-rule update, gated per-head RMSNorm, BF16
output projection, TP8 reduction, both AttnRes stages, and the existing latent
MXFP4 MoE.

The `f_b` projection is fused into the convolution/recurrence kernel. Each
per-head CTA computes its 128 gate channels from the shared 128-wide `f_a`
input immediately before consuming them, removing a separate low-token GEMM
without changing the explicit graph-safe state-pool API.

The following results pair the same production layer index, TP8 topology, HIP
graph depth (16), warmups, repeats (30), and critical-rank median. The ATOM
harness initializes the production AttnRes block state rather than timing
uninitialized storage.

| Family / representative layer | Batch | FlyDSL | ATOM | Speedup | Latency reduction |
|---|---:|---:|---:|---:|---:|
| MLA + latent MoE, layer 3 | 4 | **116.1023 us** | 222.5597 us | **1.917x** | **47.83%** |
| MLA + latent MoE, layer 3 | 8 | **127.9274 us** | 253.9299 us | **1.985x** | **49.62%** |
| KDA + latent MoE, layer 1 | 4 | **126.7786 us** | 197.2968 us | **1.556x** | **35.74%** |
| KDA + latent MoE, layer 1 | 8 | **138.2513 us** | 227.3771 us | **1.645x** | **39.20%** |

The production MLA S=8 case is approximately 2x, while KDA remains below the
2x target. The retained fused KDA convolution/recurrence/RMSNorm core itself is
about 1.95-1.98x faster than ATOM's three-kernel sequence at S=4/8. The
remaining full-layer gap is dominated by the two BF16 attention projections,
TP reduction, and AttnRes boundaries. A faster split-K input projection was
rejected because it enlarged recurrent-state error, and direct mailbox
consumption in post-AttnRes was rejected because it regressed latency.

### Why K3 is still much slower than the GLM kernel in absolute time

The GLM-5 W8A8 S=4 result above is 56.217 us, versus 109.4035 us for the K3
A16W4 full layer. These numbers should not be treated as the same-workload
optimization target. K3 has hidden size 7168 instead of 6144, 896 routed
experts/top-16/intermediate 384 instead of 256/top-8/intermediate 256, and 12
local attention heads instead of 8. Its router projection alone is about 4.08
times larger: `(7168 * 896) / (6144 * 256)`.

K3 also performs AttnRes mixing, replicated 7168-to-3584 latent projection,
shared experts, latent RMSNorm, a rank-local 3584-to-896 tail, and two MoE TP
reductions. The final K3 path still uses six application kernels, while the GLM
fast path places most work inside one persistent monokernel. The retained
changes nevertheless adopt the useful GLM mechanisms: persistent attention,
owner-reduce/broadcast communication, low-token MXFP4 tiles, direct routing
metadata, and explicit compute/communication overlap.

The final one-layer kernel breakdown is:

| Stage | S=4 | S=8 |
|---|---:|---:|
| Persistent MLA | 34.07 us | 43.00 us |
| Router + latent/shared projection | 22.44 us | 23.36 us |
| MXFP4 routed GEMM1 | 21.07 us | 22.41 us |
| Fused shared/latent tail + TP communication | 14.97 us | 16.83 us |
| MXFP4 routed GEMM2 | 9.03 us | 10.15 us |
| Post-attention AttnRes | 8.22 us | 8.51 us |

The stage sum is close to the uninstrumented latency, so the remaining gap is
now compute-dominated rather than a hidden framework or collective bubble.
Separate HIP-stream overlap was tested and rejected because the small kernels
contended for compute resources and added cross-stream synchronization. The
retained overlap is inside the fused tail: shared memory traffic and remote
sends progress while the routed reduction/RMSNorm and latent-up dependency are
resolved. Moving the selector CTA ahead of the projection CTAs was also tested;
after correcting its range guards it remained correct but regressed S=8 to
127.12 us, so selector-last scheduling was retained.

Reproduce the complete MLA + MoE checks and measurements with:

```bash
cd /root/FlyDSL-kimi-k3
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-kimi-k3:/root/tilert_pkg

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_full.py \
  --npes 8 --samples 4 --layer-idx 3 --check \
  --bench --kernel-profile --layers 16 --repeats 100 \
  --output /root/kimi-k3-perf-results/full-moe/final-s4.json

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_full.py \
  --npes 8 --samples 8 --layer-idx 3 --check \
  --bench --kernel-profile --layers 16 --repeats 100 \
  --output /root/kimi-k3-perf-results/full-moe/final-s8.json

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_kda_full.py \
  --npes 8 --samples 4 --layer-idx 1 --check --bench \
  --layers 16 --repeats 30 \
  --output /root/kimi-k3-perf-results/full-moe/flydsl-kda-production-layer1-s4.json
```

Use `--eager-attn-res`, `--eager-router`, or `--eager-shared-experts` for
controlled optimization A/B runs. Use `--profile` for median eager GPU event
timings. The default `--reduce-backend symmetric` can be changed to `nccl` for
a communication A/B. Add `--kernel-profile` to either full-layer harness to
record one rank-0 graph replay broken down by GPU kernel. Uninstrumented
HIP-graph replay remains the latency source of truth.

Reproduce the ATOM baseline from the separate checkout with its original layer
implementation. The local AITER JIT build requires its composable-kernel
submodule and `pybind11==3.0.1`, matching the AITER core ABI used here:

```bash
git -C /root/ATOM-k3-baseline checkout 3cea04f45
git -C /root/aiter submodule update --init --recursive -- 3rdparty/composable_kernel
/opt/venv/bin/python -m pip install --upgrade --target /tmp/atom-k3-deps pybind11==3.0.1

cd /root/FlyDSL-glm5-mxfp4-atom-layout
/opt/venv/bin/python kernels/mla_moe_layer/tools/atom_kimi_k3_full.py \
  --samples 4 --output /root/kimi-k3-perf-results/full-moe/atom-s4.json
/opt/venv/bin/python kernels/mla_moe_layer/tools/atom_kimi_k3_full.py \
  --samples 8 --output /root/kimi-k3-perf-results/full-moe/atom-s8.json
```
