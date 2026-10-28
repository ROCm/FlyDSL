# PMC cases: measured work and readiness

Historical GLM measurements illustrate cache parallelization, expert grouping, quantization reuse, and ordered down projection. Each pair used a fresh collection; compare within a pair. PMC, stage clocks, and uninstrumented timings came from separate experiments and are not an additive time decomposition. Collection methods are in [profiling](profiling.md).

<a id="pmc-scope"></a>

## Shared measurement scope

The four PMC pairs used MI355X (`gfx950`), TP4, C2/S6 (two requests × six queries), indexed native A8W4, and GPU0 device counting. Each interval contained 32 layer calls plus two step updates; main kernels used 256 workgroups × eight wavefronts, and each step used four wavefronts. Counts were not summed across ranks; observed wave totals matched the 65,544-wave launch count.

```text
expected_waves = 32 × 256 × 8 + 2 × 4 = 65,544
waves_per_layer_with_step = 65,544 / 32 = 2,048.25
counter_per_layer = sum(valid instances in one arm/agent/interval) / 32
DRAM_request_MB = TCC_EA0_RDREQ_DRAM_32B × 32 / 1,000,000
TCC_hit_pct = 100 × sum(TCC_HIT) / (sum(TCC_HIT) + sum(TCC_MISS))
```

`SQ_INSTS_MFMA` counts issued MFMA, not FLOPs; `SQ_INSTS_LDS` counts LDS instructions, including FLAT in this provider, not bank conflicts. DRAM counts use 32-byte requests, with 64/128-byte requests counted as 2/4; they do not distinguish weights, payload, or scratch. TCC misses include UC reads. SYS_PERFMON permission/lock warnings limit these counters to diagnostic cross-checks, not validated absolute HBM bandwidth.

**Timing for cases 1–3:** `5 trials × 256 replays × 16 layer calls`, aggregated as rank median, then trial median. Improvements apply to this statistic; they do not establish slowest-rank or request-completion gains. A current completion-time target needs its own [paired timing](../../test/references/validation.md#timing).

<a id="pmc-cache"></a>

## 1. Cache parallelization: similar work, earlier completion

**Context and change.** One workgroup serialized cache norm/RoPE/write across queries, delaying the indexer. Distribute cache tasks by query while preserving math, write ownership, readiness, and downstream workgroup offsets.

| Per-layer metric | Before | After |
|---|---:|---:|
| MFMA instructions | 824,384 | 824,384 |
| LDS instructions | 3,114,615 | 3,110,823 |
| DRAM request MB | 592.292 | 591.338 |
| TCC hit rate | 63.998% | 64.087% |

Matched `s_memrealtime` markers used the environment's verified 100 MHz timebase. Completion points below use rank-local origins, aggregated as rank median then snapshot median over 15 snapshots/rank; they are not stage durations or cross-rank latest completions.

| Completion point, µs | Before | After |
|---|---:|---:|
| QKV | 23.960 | 23.830 |
| Cache | 39.190 | 25.420 |
| Index selection | 70.300 | 56.455 |
| Output projection | 128.040 | 114.540 |
| Router | 134.430 | 120.730 |

**Result and limits.** Uninstrumented latency fell **256.291→242.410 µs (5.42%)**. Similar instruction/traffic counts and earlier dependent completions support reduced serialization. Markers changed private-segment allocation from 428/432 to 452/468 B/thread and increased the composite down interval from 49.14 to 52.54 µs, so instrumented timing was not the acceptance metric.

The scoped change passed the original eight-point checks; a broader policy regressed C1/S6 with indexer off, so dispatch was narrowed. Remaining repeated MoE work led to expert grouping. In another workload, earlier cache completion without downstream movement would question its criticality.

<a id="pmc-grouped-ug"></a>

## 2. Grouped UG: predict the instruction reduction

**Context and change.** The fixture had 96 routed slots, 35 distinct experts, a shard-local intermediate width of 512, eight outputs/tile, and 48 MFMA/tile. Group tokens sharing an expert into MFMA columns, retaining original-slot mapping, FP32 intermediates, and down math/quantization/reduction order. Shared-expert reuse was already present.

```text
predicted_saved_MFMA = (96 - 35) × (512 / 8) × 48 = 187,392
live_tasks_including_shared = (35 + 1) × 64 = 2,304
```

The initial task mapping assumed 16 outputs while the routine wrote eight, leaving consumers waiting on missing payload. Fix [coverage](cases.md#coverage) before measuring reuse; capacity was not the live task count.

| Per-layer count | Before | Grouped | Change |
|---|---:|---:|---:|
| MFMA | 824,384 | 636,992 | −187,392; −22.73% |
| LDS instructions | 3,110,823 | 2,578,217 | −17.12% |
| DRAM 32B requests | 18,421,314.625 | 12,729,284.375 | −30.90% |

**Result and limits.** Uninstrumented latency fell **237.437→197.272 µs (16.92%)**. Exact MFMA agreement supports skipped repeated execution, not merely masked columns. UG staging padding also changed; whole-kernel private-segment allocation fell 432→120 B/thread and VGPR spill metadata 107→29. Traffic changes cannot all be assigned to weight reuse, nor spills solely to UG.

Next inspect repeated consumer quantization. If a new grouping misses its predicted MFMA reduction, check actual task skipping, masked instructions, selected binary, and normalization.

<a id="pmc-quant-once"></a>

## 3. Quant-once: logical bytes and DRAM traffic differ

**Context and change.** After grouping, every down workgroup still read tagged FP32 activations and repeated quantization. One UG8 tile could not form a 32-element scale group. Assigning four tiles to one producer reduced parallelism; the selected design kept UG scheduling and used one owner to gather, quantize, and publish packed values/scales. It preserved the FP32→MXFP8 rule and slot/K/wavefront/TP accumulation order, adding no launch or global completion flag.

At T12, logical payload per down workgroup fell **442,368→124,416 bytes (71.875%)**, plus the owner's per-rank FP32 read and packed publication cost.

| Per-layer count | Repeated quantization | Quant-once | Change |
|---|---:|---:|---:|
| MFMA | 636,992 | 636,992 | 0 |
| LDS instructions | 2,578,217 | 2,439,977 | −5.36% |
| DRAM 32B requests | 12,855,439.75 | 12,780,522.4375 | −0.58% |

**Result and limits.** Uninstrumented latency fell **199.860→193.182 µs (3.34%)**. Unchanged MFMA and lower LDS execution support less consumer preparation. Only 0.58% fewer DRAM requests rejects proportional HBM traffic savings as the explanation; repeated logical reads can hit cache. These measurements do not isolate quantization time or bank conflicts.

Quantization was now shared. The next opportunity was down-projection reuse within the original summation order.

<a id="pmc-down"></a>

## 4. Ordered down: preserve the contribution sequence

**Context and change.** Group tokens with the same expert within each original wavefront and slot, preserving each token's slot/K sequence and final wavefront reduction. Reuse LDS tables only after all readers finish; an earlier candidate failed this lifetime rule. The PMC fixture had 432→380 units/output group, two MFMA/unit, and 192 groups.

```text
predicted_saved_MFMA = (432 - 380) × 2 × 192 = 19,968
observed_saved_MFMA = 636,992 - 617,024 = 19,968
```

| Per-layer count | Before | Ordered grouping | Change |
|---|---:|---:|---:|
| MFMA | 636,992 | 617,024 | −3.13% |
| LDS instructions | 2,439,977 | 2,222,633 | −8.91% |
| DRAM 32B requests | 12,634,031.3125 | 11,472,098.1875 | −9.20% |

**Result and limits.** Compatibility checks and uninstrumented real-weight C2/S6 indexed timing supported adoption: **218.342→203.897 µs**, versus unfused 204.365 µs. The roughly 0.23% lead over unfused was not a stable-win claim. The PMC prediction uses its own fixture's route counts; a host order audit does not replace native numerical checks.

A cross-slot queue offered more reuse but regressed S6/on by 2.50%. After removing repeated work and reported private-segment allocation/VGPR spills in the affected shapes, reassess the critical path instead of reusing the old spill diagnosis.

<a id="pmc-missing"></a>

## 5. Batching and prefetch: leave an unmeasured cause unresolved

**Context and result.** A later kernel already included norm reuse, UG double tiles, and earlier down preparation. Four-round UG batching regressed **0.53%–1.38%**; two-tile prefetch regressed **0.58%–1.23%**. LDS decreased, private-segment allocation and reported VGPR spills were zero, and resident capacity was unchanged. Neither candidate had matching PMC/ATT, so resources alone did not explain the regressions.

UG/down markers overlapped, but down entry included polling. The useful follow-up was to mark full-32-element-group publication/consumption and first down computation, then collect ATT covering those stages. Observer SGPR spill metadata rose 110→752, requiring less intrusive measurement. Retain uninstrumented timing for acceptance; old traces cannot establish the current cause.
