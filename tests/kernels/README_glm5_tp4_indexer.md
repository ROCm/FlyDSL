# GLM-5.2 TP4 fused indexer: C1/C2 MTP4

Measured on four MI355X GPUs on 2026-10-10. This branch starts from FlyDSL
`main` at `3ad47c18` and compares against ATOM PR #2435 at `a3e2b1a25`.
The layer graph has 16 GLM layers per replay. Layer times below are the median
rank's microseconds per layer; indexer probes run on one GPU. The GLM layer
uses native MXFP4 MFMA, FP8 PTPC attention,
FP8 ATOM-layout attention KV, a 3,000-token starting context, top-2048, and
five MTP4 rows per request. C1 has one request (S=5); C2 has two (S=10).
The synthetic timing weights leave attention and experts zero-filled. Indexer
weights, inputs and its cache are randomized. A separate correctness run uses
nonzero attention weights and KV values.

| Path | C1, S=5 | C2, S=10 | Included work |
| --- | ---: | ---: | --- |
| ATOM PR #2435 MonoKernel | 91.18 | 184.35 | Layer, external indexer excluded |
| FlyDSL TP4 MonoKernel | 91.71 | 187.77 | Layer, external indexer excluded |
| External AITER indexer core | 23.16 | 23.13 | Q/K RoPE, FP8 quant/cache, paged score, stable top-k |
| ATOM layer + external core | 114.34 | 207.48 | Lower bound: excludes indexer projections and index conversion |
| External projection + core proxy | 53.38 | 56.47 | Adds RMSNorm, duplicate QKV, index Q/K/W projections; excludes index conversion |
| ATOM layer + projection proxy | 144.56 | 240.82 | Controlled estimate, not deployed serving |
| FlyDSL fused TP4 | **118.42** | **224.49** | Index K/Q/W projections, BF16 index cache, score, top-k, attention and MoE |

Against the controlled projection proxy, fusion lowers C1 time by **18.1%**
(1.22×) and C2 time by **6.8%** (1.07×). The proxy uses AITER CK for FP8
projections and PyTorch BF16 linear for the K/weight projection because this
installed AITER's tuned FlyDSL GEMMs are incompatible with the local FlyDSL
runtime. It omits selected-index conversion and runs on one GPU, whereas the
layer was measured on four ranks. The fused kernel is 3.6% slower than the
C1 core-only lower bound and 8.2% slower than the C2 lower bound. This
machine has no GLM checkpoint or serving setup to measure the deployed full
path or TPOT. FlyDSL's fused index cache is BF16, whereas ATOM's external
indexer uses a quantized FP8 cache, so their precision and storage differ.

The TP8 roughly 2× indexed kernel result does not transfer directly to TP4.
TP4 has 16 local attention heads, filling all 256 resident CTAs for the Q-B
projection. The index Q projection therefore runs after Q-B on those CTAs;
TP8 uses eight local heads and has spare CTAs for overlap. In the instrumented
C2 run, the final expert-down stage alone took about 55 µs. A 2× result for
this full TP4 layer would require an external indexer cost of about 146 µs
for C1 or 265 µs for C2, well beyond the 53/56 µs projection proxy.

## Correctness

The fused top-2048 set matched the independent PyTorch reference for every
C1 and C2 row (overlap 1.00000), including a test with shuffled physical
pages. With nonzero attention weights and KV values, mapping the fused
selected positions through the page table and supplying them to the unfused
TP4 kernel gave exactly the same output (`max diff = 0`), with nonzero
attention signals of 0.02478 for C1 and 0.04639 for C2. The paired ATOM and
FlyDSL kernels without indexer also gave identical outputs.

## Run

The benchmark scripts are `bench_glm5_tp4_native.py`,
`bench_glm5_tp4_compare.py`, and `bench_glm5_tp4_indexer_core.py` in this
directory. Put the FlyDSL runtime, ATOM PR checkout, and AITER on
`PYTHONPATH` before running. The fused benchmark command is:

```bash
python tests/kernels/bench_glm5_tp4_native.py \
  --samples 5 10 --native 1 --fused-indexer --replays 32
```

For correctness, add `--check-golden --check-attention --shuffled-pages`
and use a small replay count. The external core benchmark needs
`AITER_DISABLE_FLYDSL_TOPK_DECODE=1` with this installed AITER runtime. Add
`--projections` for the controlled projection-inclusive estimate.

The fused TP4 API takes a BF16 index cache with 128 values per physical row,
an `int32` block table for each request, five rows per request, and the ATOM
`positions`, `slot_mapping`, and `sparse_kv_indptr` metadata. It maps selected
logical positions to physical attention KV slots inside the persistent
kernel. The current fused path is limited to an index window of 4,096 tokens
and does not implement ATOM's FP8 index cache format.


## Shared TP4/TP8 implementation

The follow-up refactor uses `glm/kernel.py`, `glm/layout.py`, and `glm/op.py`
for both TP geometries. The `tp4_*` modules preserve the PR's public entry
points and BF16 RoPE defaults. TP8's original flat-cache entry point retains
FP32 RoPE. The paged benchmark accepts `--npes 4` or `--npes 8`.

| Compile-time dimension | TP4 | TP8 |
| --- | ---: | ---: |
| Local attention heads | 16 | 8 |
| Local expert intermediate width | 512 | 256 |
| Indexer heads × head dimension | 32 × 128 | 32 × 128 |
| Eight-row expert task rounds | 2 | 1 |

Attention formats, cache layouts, and expert storage select load paths at
compile time. Native MXFP4 keeps the eight-row gate/up interleave and its
matching tile-major E8M0 scales. ATOM storage keeps its original layout and
native FP4 MFMA configuration. Sparse-attention tile sizes and down-projection
prefetch depths retain their geometry and storage-specific tuning.

The common UG schedule handles complete 256-CTA rounds. This also fixes the
original TP4 S=1 schedule, which omitted half of its 512 expert tasks. The down stage
retains the original routing schedule. Native weight prefetch and activation
staging are ordered by sample count and format.
Unreachable legacy UG schedules have been removed. This refactor does not
add indexer context parallelism.

### Validation protocol

Measurements use node46's MI355X GPUs, the exact PR #1256 head `bb298cf2` as
baseline, and alternating baseline/refactor runs. Each replay contains 16
layer launches at one weight address and advances the step once. Reported
latencies are in microseconds per layer and use median rank times. A latency
increase of at most 0.5% is the regression acceptance threshold.

The paged regression uses the original PR protocol: context 3000, top-2048,
MTP4 request width 5, native MXFP4 MFMA, FP8 PTPC attention and ATOM-layout
FP8 attention KV, BF16 index cache; 20 warmup graphs and three trials of 32
graphs, with three outer alternating runs. Its attention/expert timing weights
are zero-filled. Nonzero attention correctness is checked separately.

The native Conc 1 matrix uses S=1/2/4/8, context 3000, top-2048, FP8 block-scaled
attention, flat BF16 caches, and FP8 or MXFP4 expert weights. It uses seven alternating baseline/refactor rounds, each with 50 warmup
graphs and 4096 timed layer calls. Both implementations run in one process
per rank and consume the same input, cache and packed-weight tensors; weight
addresses are asserted equal. Scratch, peer buffers, step storage and output
also share addresses in the native comparison. Outputs and ordered indices
are checked for
exact equality before timing. This controls allocation-dependent differences
seen when the implementations ran in separate processes. Real TP4 geometry
can be selected in the existing native script with `--model-tp 4 --npes 4`; without
`--model-tp`, historical smaller peer-count tests keep their TP8 shard geometry.

Correctness retains the original numerical tolerances and independent
PyTorch goldens. Nonuniform MXFP4 scales exercise the packed scale addressing.
The original independent end-to-end check remains conditional on matching
near-tied routing decisions; intermediate and stage checks still run. Paged
top-2048 validation retains the PR's 99% selected-set overlap threshold.


### Shared implementation results

All 20 paired regression cases pass the 0.5% latency threshold; the largest
increase is 0.461%. The native tests pass 27 cases: eight real TP4/TP8 MXFP4
cases with nonuniform scales, eight real TP4/TP8 FP8 cases, and 11 original
cases. The paged S=5/10 checks pass on both TP4 and TP8. The smallest paged
selected-set overlap is 0.99951, above the unchanged 0.99 threshold; attention
output matches the selected-index control exactly. Paired source comparisons
also match output and ordered indices exactly.

Native TP8, Conc 1, context 3000, microseconds per layer:

| Expert weights | S | Fused indexer | PR baseline | Shared implementation | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| FP8 | 1 | No | 37.869 | 37.900 | 0.999× |
| FP8 | 1 | Yes | 54.587 | 54.443 | 1.003× |
| FP8 | 2 | No | 44.189 | 44.020 | 1.004× |
| FP8 | 2 | Yes | 62.202 | 61.481 | 1.012× |
| FP8 | 4 | No | 59.269 | 59.172 | 1.002× |
| FP8 | 4 | Yes | 82.261 | 81.622 | 1.008× |
| FP8 | 8 | No | 98.121 | 97.967 | 1.002× |
| FP8 | 8 | Yes | 132.137 | 130.838 | 1.010× |
| MXFP4 | 1 | No | 37.385 | 37.372 | 1.000× |
| MXFP4 | 1 | Yes | 54.293 | 53.683 | 1.011× |
| MXFP4 | 2 | No | 43.363 | 43.223 | 1.003× |
| MXFP4 | 2 | Yes | 61.446 | 60.837 | 1.010× |
| MXFP4 | 4 | No | 55.223 | 55.046 | 1.003× |
| MXFP4 | 4 | Yes | 77.369 | 77.085 | 1.004× |
| MXFP4 | 8 | No | 83.338 | 83.391 | 0.999× |
| MXFP4 | 8 | Yes | 115.279 | 114.185 | 1.010× |

TP4 paged MTP4, the PR's original fixture and event timing protocol. Seven
paired outer rounds reuse the same inputs, cache, output, and packed weights
between source implementations; their runtime mailboxes remain separate.
The published PR measurements are included for reference.

| Conc / S | Fused indexer | Published PR | Paired PR baseline | Shared implementation | Latency change |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 / 5 | No | 91.71 | 91.340 | 91.396 | +0.062% |
| 1 / 5 | Yes | 118.42 | 117.507 | 118.039 | +0.453% |
| 2 / 10 | No | 187.77 | 185.386 | 186.240 | +0.461% |
| 2 / 10 | Yes | 224.49 | 224.525 | 225.009 | +0.216% |

The additional TP4 native flat-cache matrix is functional at S=1/2/4/8.
Current latencies are below; these are coverage measurements rather than a
claimed regression comparison against the PR's paged fixture. In particular,
the original TP4 S=1 expert schedule did not supply a working baseline.

| S | FP8, no indexer | FP8, fused indexer | MXFP4, no indexer | MXFP4, fused indexer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 52.905 | 68.198 | 50.901 | 66.571 |
| 2 | 61.556 | 81.276 | 56.387 | 74.003 |
| 4 | 88.873 | 111.211 | 73.871 | 94.608 |
| 8 | 149.437 | 185.984 | 121.086 | 155.236 |
