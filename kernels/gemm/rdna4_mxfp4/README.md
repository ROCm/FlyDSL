# RDNA4 MXFP4 × FP8 GEMM

Dense OCP MXFP4 weights and prequantized FP8 E4M3 activations on ROCm gfx1200/gfx1201.
The kernels use FlyDSL layouts, tiled copies and WMMA atoms. Weights stay
compressed; conversion to exact FP8 operands happens inside GEMM, followed by
native FP8 WMMA and FP32 accumulation. Hardware FP4 instructions are unnecessary.
Performance validation currently covers gfx1201 (Radeon AI PRO R9700).

```python
import torch
from flydsl.kernels.gemm.rdna4_mxfp4 import prepare_weights, gemm, choose_config

# packed: uint8 [N, K/2], even K in low nibble, odd K in high nibble.
# scales: uint8 [N, K/32], OCP E8M0 exponents.
weights = prepare_weights(packed, scales)
y = gemm(x_fp8_e4m3, weights, x_scale_fp32, cache_policy="rotating")
print(choose_config(x_fp8_e4m3.shape[0], weights.n, weights.k, weights.mode,
                    cache_policy="rotating"))
```

M and N can have tails. K must be a positive multiple of 32. Inputs and output
are contiguous on the same GPU; output is BF16 by default or FP16. Activation
scales are FP32 [M] or [M, 1]. Each addressed buffer must be smaller than 4 GiB.
Preparation is a load-time operation; GEMM supports current-stream execution
and graph replay with changed inputs. Prepared storage owns its device.

Preparation retains 4 bits per weight, one E8M0 byte per 32 weights, padding to
16 output columns, and one FP32 reference exponent per padded output column for
folded plans. It does not retain dense decoded weights. Auto preparation uses
lossless E4M3 folding, then E5M2 folding for wider exponent spans, then exact
grouped K32 rescaling. Forced folding raises when it would lose information.

## Two strategies

| Strategy | Dataflow | Configuration parameters |
|---|---|---|
| Register | Copy A and compressed B into register fragments; decode B before WMMA | M/N wave distribution and repeats, K32 unroll, masked buffer copies, byte-conversion method, epilogue |
| Hybrid | Keep A in registers; cooperatively decode and stage B in LDS with operand lookahead | M/N/K tile, wave distribution, one/two B stages, operand lookahead, M cache grouping, B padding/swizzle, barriers/scheduling, epilogue |

These are algorithm strategies. Tile sizes, unroll, stages and LDS layouts are
configurations of a strategy. Exact grouped scaling uses register GEMM.
Cooperative A+B LDS GEMM and historical HIP-style ports are not included.

The CPU selector first uses measured configuration profiles for matching N/K
families. It interpolates M only within matching static/aligned/ragged compiler
classes and a bounded launch-grid ratio. Direct interpolation requires an M
ratio no greater than 1.25; farther profiles and compatible nearby families
become proposals for the configuration-cost model. Replacing a proposal requires
a predicted 10% gain. Narrow aligned queries reject odd-M profile sources.
Register transfer checks both the chosen K loop and the available unroll factors.
All proposals pass tile, accumulator, LDS and masked-copy legality checks.

`auto` balances hot and rotating weights, `hot` targets reuse of one weight, and
`rotating` targets cycling through model layers. Selection imports only the
Python standard library and caches its results on the CPU. It does not benchmark
GPU candidates. `rank_configurations` describes the selected priorities; these
are not latency measurements or a guarantee that every unseen shape is optimal.

The profile/classifier data spans 15,429 development shapes, including Qwen3.8-27B
and Gemma4-12B/31B linear dimensions at TP1/2/4. The cost model adds explicitly
consumed N/K development cases and their complete bounded proposal banks, for
15,936 fitting shapes and 1,259,942 hot/cold observations. Each N/K family has
equal total fitting weight. JSON metadata records the provenance. Model
holdouts, actual-checkpoint confirmation and the final fresh N/K/aligned tests
are excluded from this fit.

The source-frozen selector completes the full model and checkpoint validation
matrix, including Qwen3.8-27B and Gemma4-12B/31B TP1/2/4. The table reports
balanced (`auto`) latency against the best configuration in each frozen,
freshly measured reference bank. Ratios above one are slower; the final column
compares against the preceding native selector.

| Validation cohort | Cases / configuration trials | Geomean latency / reference | Worst | Geomean speedup over preceding selector |
|---|---:|---:|---:|---:|
| Model families, unseen M | 2,194 / 90,739 | 1.00524× | 1.2180× | 1.1190× |
| Checkpoint rank/dtype confirmation | 1,678 / 8,460 | 1.00290× | 1.0611× | 1.0586× |
| Fresh N/K families | 195 / 10,703 | 1.04820× | 1.5544× | 1.3067× |
| Fresh aligned shapes | 91 / 3,980 | 1.02982× | 1.5768× | 1.5264× |

All trials pass sampled numerical checks against independent FP32 weight
decoding/GEMM on fixed FP8 inputs, changed-input graph replay and output guards.
Checkpoint confirmation uses representative pinned tensors for every declared
linear family and all 168 rank shards, BF16/FP16 output and synthetic FP8
activations. Two oversized Gemma12 TP1 output cases are excluded explicitly.
The reference is a finite configuration bank. In particular, fresh-shape tails
still expose proposal and cost-ranking misses; some cases regress against the
preceding selector despite the geometric mean gains. These isolated GEMM
results do not establish end-to-end serving gains.

[Configurations, per-case results, checkpoint/source hashes and ISA metadata](../../../tests/perf/data/rdna4_mxfp4_comprehensive.json)
also include two separately labeled consumed aligned development cohorts.
The canonical package passes 75 CPU tests and preserves 170,199 public choices
over 18,911 shapes against the measured source. The GPU suite passes 100 tests
with 10 explicit skips. All 2,633 model-family warmup keys compile without GPU
initialization using the installed SDK and existing compilation cache.

Every audited selected executable uses native FP8/BF8 WMMA. Four selected
hybrid objects in the model holdout use 8–108 bytes of scratch and VGPR spills.
Their 16 policy selections are all within 3.57% of the measured bank best and
faster than the preceding selector; 11 are the measured winners. They remain
selected on timing evidence. The checkpoint and fresh-shape cohorts have zero
selected scratch/VGPR-spill objects. Scalar spills also occur and are recorded
per executable. These resource checks are not hardware-counter roofline data.

The paired Qwen3.8-27B TP1 simple-agent comparison against the preceding native
selector has effectively flat output TPS (within 0.32%). Mean TTFT changes are
+0.29%, +0.90%, -3.81% and -0.55% at streams 1/2/4/8. Both retained passes have
complete requests, zero inference JIT warnings and no executable events at or
above 50ms. These are point estimates from one workload on separate GPUs at
default clocks, with small generated-history differences. They do not establish
an all-concurrency TTFT win.

Paired full GSM8K scores are 1,271/1,319 (96.36%) for this selector versus
1,269/1,319 (96.21%) for the preceding native selector, with zero invalid
answers. Both use five shots, greedy generation, thinking disabled and a
1024-token output cap. The +0.1516 percentage-point difference has a paired
95% interval of [-0.2274, +0.6065] percentage points, supporting no observed
accuracy regression rather than a proven accuracy gain.

Performance fitting covers normal E4M3-foldable weights on gfx1201. FP16 output,
E5M2 folding and exact grouped rescaling have separate correctness tests. TP4
coverage means per-rank kernel shapes on the available two GPUs; it does not
establish four-device serving performance. The public 12B MLX packing is adapted
for operand tests and does not establish vLLM checkpoint-loading compatibility.

For a deployment's new shapes, calibrate before graph capture:

```python
from flydsl.kernels.gemm.rdna4_mxfp4 import tune_gemm

config = tune_gemm(x_fp8_e4m3, weights, x_scale_fp32,
                   cache_policy="rotating")
y = gemm(x_fp8_e4m3, weights, x_scale_fp32, cache_policy="rotating")
```

Calibration uses FlyDSL's native `Autotuner` with at most 64 legal register/hybrid
configurations. It checks sampled outputs against independent FP64 MXFP4
decoding before timing each candidate. Hot and rotating-weight graph timings
match the requested cache objective; `auto` minimizes their geometric mean.
Calibration compiles and validates its candidate configurations, makes temporary
weight copies and measures device timings. A normal configuration-cache miss
uses the CPU fallback without candidate search, temporary weight pools or
device timing. It can still JIT-compile the selected executable on first use.

Prefill executables share a runtime M dimension through FlyDSL dynamic tensor
layouts. Aligned and partial M tiles have separate epilogues; batches
1/2/4/8/16 keep their specialized decode paths. Executable identity includes
N/K, configuration, folding mode, dtype and device. Reusing a configuration
across different row counts therefore reuses its compiled executable.

Compile the required configuration space before serving:

```python
from flydsl.kernels.gemm.rdna4_mxfp4 import get_warmup_keys, compile_gemm

keys = get_warmup_keys(
    weights.n, weights.k, max_m=2048,
    mode=weights.mode, normal_only=weights.normal_only,
    nibble_layout=weights.nibble_layout, dtype=torch.bfloat16,
    device_id=weights.packed.device.index, cache_policy="rotating",
)
for key in keys:
    compile_gemm(key)
```

This enumerates distinct heuristic and persisted measured configuration keys.
Compilation uses meta tensors and FlyDSL's compile-only mode, with no real
tensor allocation or dummy GPU launches. Run it during single-threaded startup
before inference and graph capture. First use can load the compiled artifact.
Different N/K, dtypes, scale plans or configurations require corresponding
warmup keys. Configuration calibration remains a separate optional operation.

Measured choices persist in `FLYDSL_AUTOTUNE_CACHE_DIR` (default
`~/.flydsl/autotune`) and are keyed by source, shape, dtype, layout, scale mode,
cache objective, device and FlyDSL's compiler/environment fingerprints.
Different processes serialize writes with a file lock. Ordinary GEMM reads this
cache without timing. `FLYDSL_AUTOTUNE=1` requests calibration on first use in
the process and must be used before graph capture; exact grouped plans retain
the conservative register fallback. Recalibrate after changing compiler or
kernel sources. Calibration finds the best measured member of its bounded
shortlist, not a mathematical optimum over all possible configurations.

Explicit configuration is available for offline experiments:

```python
y = gemm(x_fp8_e4m3, weights, x_scale_fp32,
         config={"engine": "register", "waves_m": 1, "waves_n": 4,
                 "repeat_m": 1, "repeat_n": 1, "k_unroll": 4})
```

Run focused tests from a built FlyDSL checkout:

```bash
python -m pytest tests/unit/test_rdna4_mxfp4_dispatch.py
python -m pytest tests/unit/test_rdna4_mxfp4_tuning.py
python -m pytest tests/kernels/test_rdna4_mxfp4.py
```

## Runtime-M and prefill dispatch validation

The rotating-weight prefill policy was frozen before testing 50 new cases:
30 actual model projections at M=293/613/997/1549/1973 and 20 varied skinny
projections at M=173/701/1477/1933. Compared with the preceding implementation,
geometric mean GEMM time ratios are 0.8275 and 0.4189. Weighting the real
projections by their frequency in the model gives forward-GEMM ratios of
0.9960/0.8657/0.9985/0.9517/1.0032. Every measured candidate has zero relative RMS against
the preceding selected output. One skinny crossover remains 27.2% above its
measured shortlist best; the heuristic does not guarantee the fastest tile.

Eighteen decode cases (M=1/8/16 across six projection families) retain identical
instruction fingerprints, with geometric mean time ratio 1.0014 and worst
ratio 1.0069. Six M=8192 cases have mean ratio 0.9499 and worst 1.0025. All 24
final ISA samples use native FP8 WMMA with zero scratch or register spills.
Seven randomized paired rounds use GPU events and graph replay, rotating
weight copies and a cache flush outside the timed interval. These checks use
normal E4M3-foldable weights on gfx1201 with the installed 0.3.4.1 compiler.

[Raw measurements, configurations, source hashes and ISA metadata](../../../tests/perf/data/rdna4_mxfp4_ttft.json)
also record the limitations and the exact prototype versus final source hashes.
Public API tests separately confirm compile-only warmup allocates no GPU
memory, arbitrary runtime row counts execute without compilation, and changed
inputs and guarded output tails remain correct under graph replay.

## Historical port performance validation

The source port was compared with the preceding native implementation on two Radeon AI PRO R9700 GPUs at default clocks, using identical explicit configurations. Eight shapes, both GPUs, hot/rotating weights and 15 randomized paired rounds give 32 latency comparisons. Port/reference latency is 0.9991× geometrically; the worst is 1.0064×. Every pair is bitwise equal, reads changed inputs on graph replay, and has the same decoded instruction fingerprint. All executed kernels use native FP8/BF8 WMMA with zero scratch or register spills. This isolates the source port from the quality of default configuration selection.

The table contains geometric means across GPUs. Ratios above one are slower. Raw timing rounds, configurations, ELF/ISA hashes and resource metadata are in [the validation data](../../../tests/perf/data/rdna4_mxfp4_port.json). Validation uses this branch’s exact kernel sources with the installed FlyDSL 0.3.4.1 compiler; the compiler SDK was not rebuilt.

| M | N | K | Hot port/reference | Rotating port/reference |
|---:|---:|---:|---:|---:|
| 1 | 5120 | 17408 | 0.9974× | 0.9996× |
| 8 | 34816 | 5120 | 1.0022× | 0.9995× |
| 64 | 8192 | 4096 | 1.0021× | 1.0010× |
| 73 | 4099 | 1568 | 1.0021× | 1.0025× |
| 128 | 17408 | 5120 | 0.9941× | 0.9865× |
| 512 | 5120 | 8704 | 0.9986× | 1.0003× |
| 8191 | 6144 | 8192 | 0.9991× | 0.9998× |
| 8192 | 5120 | 17408 | 1.0000× | 1.0004× |


## Unseen-shape calibration validation

A separate source/shortlist-frozen test covers 18 shapes across six N/K families
absent from every prior training and validation cohort. The independent
reference uses broad screening, refinement and finalist selection, followed by
15 paired rounds on both GPUs. Each shape is evaluated with hot and rotating
weights, giving 36 cache-regime comparisons.

| Selection method | Geomean latency/reference | Worst | Within 5% |
|---|---:|---:|---:|
| CPU auto fallback | 1.1034× | 1.4100× | 17/36 |
| CPU matching cache hint | 1.1010× | 1.5720× | 19/36 |
| Calibrated auto | 1.0159× | 1.1533× | 31/36 |
| Calibrated matching hint | 1.0158× | 1.1600× | 31/36 |

Calibration substantially reduces configuration misses while retaining a
bounded search. The worst gap is 16%; it does not always find the reference's
fastest configuration. All 164 confirmed method/GPU pairs pass full eager and
changed-input graph checks; maximum relative RMS is 0.001669. The focused
suites pass 661 standalone tests, 157 FlyDSL tests and 10 vLLM backend tests.
Unsupported hybrid/grouped combinations are skipped explicitly.

ISA inspection of 130 selected method/GPU instances (63 unique ELFs) confirms
native FP8/BF8 WMMA with zero scratch usage and no VGPR spills. Eight instances
have 23–24 scalar spills saved in VGPR lanes, visible as lane write/read
instructions. Those are included in the measured configurations. This audit
is broader than the completely spill-free paired port subset above.

[Per-shape results, configurations and resource metadata](../../../tests/perf/data/rdna4_mxfp4_selection.json)
include the source-model/shortlist digests and separate prediction/calibration
results. Performance testing covers normal E4M3-foldable weights on gfx1201.

## Benchmark and deployment warmup

From a checkout with the kernel package installed, benchmark reused and rotating weights:

```bash
python tests/perf/bench_rdna4_mxfp4.py --m 1 8 128 8192 --nk 5120x17408 --tune
```

For normal E4M3-foldable deployment weights, pre-calibrate specified shapes without a serving-time search:

```bash
FLYDSL_AUTOTUNE_CACHE_DIR=/path/to/cache \
python tests/perf/bench_rdna4_mxfp4.py --gpu 0 --m 1 2 4 8 16 \
  --nk 17408x5120 5120x8704 --cache-policy rotating --tune --calibrate-only \
  --output warmup.json
```

The helper uses synthetic operands with varied E8M0 exponents within the normal E4M3 range, so its cache entries cover normal E4M3-foldable plans. For other scale modes, call `tune_gemm` with actual prepared weights. Repeat for each device and share the same cache directory with the serving process. Compilation and calibration costs are reported separately. Ordinary GEMM timing excludes those costs. Reported effective GB/s uses minimum logical traffic rather than hardware DRAM counters.
