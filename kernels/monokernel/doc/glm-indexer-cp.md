# GLM MonoKernel: shared TP4/TP8 indexer context parallelism

This change combines PR #1256's TP4 paged decode path with the indexer CP
implementation from `5f30d2b0`. TP4 and TP8 compile the same kernel, schedule,
scratch layout, and host operator. Each rank still launches one persistent
kernel per layer, covering index projections, cache update, score, exact
selection, sparse MLA, routing, MoE, and both TP reductions.

## Shared implementation

| Component | TP4 | TP8 |
| --- | --- | --- |
| Local attention heads | 16 | 8 |
| Local expert intermediate width | 512 | 256 |
| Indexer heads / head dimension | 32 / 128 | 32 / 128 |
| CP participants per group | 4 | 4 or 8 |
| Native flat-cache test | Single request, S=1/2/4/8 | Single request, S=1/2/4/8 |
| Paged MTP4 test | C1: S=5; C2: S=10 | C1: S=5; C2: S=10 |

`glm/kernel.py`, `glm/layout.py`, and `glm/op.py` contain the implementation.
The `tp4_*` modules retain compatibility entry points rather than separate
kernel bodies. Compile-time geometry controls attention head groups, expert
work rounds, task counts, and storage sizes. TP4's S=1 expert work now covers
all 512 tasks in two rounds; the former 256-task schedule could stall waiting
for missing intermediate values. TP8 retains its single round.

The implementation preserves the existing native MXFP4 gate/up row interleave,
TP8's S=8 sparse-attention tile size, and the storage-specific down-projection
prefetch depths. ATOM expert storage and native FP4 MFMA retain the PR's layout
and tuning. Native and ATOM weights share the compute schedule with distinct
compile-time load formats.

## CP algorithm and input contract

Enable CP with `with_indexer=True, indexer_cp=True`; it defaults to off.
Each rank owns a contiguous 64-token-aligned shard of each query's live causal
context. Every score still sums all 32 index heads. Index weights, inputs,
positions, and cache contents must be identical across ranks. Index caches
remain replicated; CP distributes computation rather than cache storage.

For TP8 with S=1 and a compiled capacity above 16K and up to 128K, two
independent groups of four consecutive TP ranks each produce a complete
selection. This reduces coordination overhead for a single query. Other TP8
shapes use all eight ranks, including the 1M-context path. TP4 uses four ranks.
The group size is chosen in the shared layout helper; score partitions,
mailboxes, and selection use that same size. Attention and MoE reductions
continue to span the full TP group.

For live contexts up to 16,384 tokens, ranks exchange score shards once and
run the exact local radix selector. If the compiled capacity also fits this
limit, index-Q projection is sharded across ranks. Longer contexts use bounded
4K-token tiles and four distributed radix rounds, followed by global count,
offset, and emit stages. The short/long dispatch is per query, so a launch can
cross the 16K boundary. A short request in a larger allocated cache still
partitions work by its live length.

Selection preserves the existing ordered top-2048 contract: ascending token
IDs above the score threshold, followed by the earliest IDs tied at the
threshold, with zero padding below 2048 tokens. The short selector includes
the previously validated barrier separating radix-threshold reads from scan
scratch reuse. Peer messages have epoch tags and two launch slots. CP adds no
host collective and no separate GPU communication kernel.

The common operator accepts either flat BF16 KV/PE and index caches for one
consecutive request, or ATOM attention KV plus a paged BF16 index cache.
Paged attention and index caches share the request block tables. Selected
logical token IDs are translated to physical attention slots in the kernel.
`index_request_width` rows belong to one request; their positions must be
consecutive. Different requests may have different starting positions.

The original TP8 entry point defaults to FP32 RoPE tables. The TP4 compatibility
entry point retains BF16 RoPE and five rows per request for S=5/10. The common
operator selects these explicitly via `rope_dtype` and `index_request_width`.
`index_max_seq` must be a positive multiple of 64 and accommodate every query.
Use distinct layer ordinals for shared scratch and call `advance_step()` once
after the graph's 16 layer launches. Fused indexer CP currently requires
`dcp_size=1`; it is independent of the external-indexer DCP path.

## Reproduction and validation

Use four or eight MI355X GPUs with the FlyDSL runtime on `PYTHONPATH`.
The existing benchmark keeps its TP4 defaults; `--npes 8` exercises the same
paged layout on TP8.

```bash
# PR #1256 performance protocol: 16 layers/graph, 20 warmup graphs,
# three event-timed trials of 32 graphs, median across trials then ranks.
python tests/kernels/bench_glm5_tp4_native.py \
  --npes 4 --samples 5 10 --native 1 --fused-indexer --indexer-cp --replays 32

# Paged correctness, different request lengths, and prolonged replay.
python tests/kernels/bench_glm5_tp4_native.py \
  --npes 4 --samples 5 10 --native 1 --fused-indexer --indexer-cp \
  --check-cp --check-golden --check-attention --shuffled-pages --request-pos-stride 127

# Nonzero attention and expert weights, real model shard geometry.
python tests/kernels/test_glm5_monokernel.py \
  --npes 4 --model-tp 4 -S 8 --pos 3000 --indexer --indexer-cp --mxfp4 --iters 2

python -m pytest tests/kernels/test_glm5_monokernel.py -q
```

Repeat without `--indexer-cp` for the same-source control, and without
`--fused-indexer` for the PR's external-indexer layer timing. `--model-tp`
selects real model geometry; the historical `--npes 1/2/4` native tests without
that option retain their TP8-shard peer-protocol checks. `--native 0` exercises
the paged expert path using software dequantization.

Native numerical tolerances and independent goldens are unchanged. CP checks
add equality of ordered indices, outputs, and updated cache bytes, plus 306
graph replays of 16 layers. The PR's FP8 cache fixture has an FNUZ tensor type
while the kernel consumes FN bytes, so cache equality compares storage bytes:
FN negative zero (`0x80`) appears as NaN through an FNUZ numerical view.
The paged golden retains the PR's 99% selected-set overlap criterion; exact
CP-on/off ordering is a separate, stricter requirement. Its expert timing
weights are zero-filled; full nonzero MoE golden validation comes from the
native tests.

## Performance and test results

The regression baseline is the preceding shared-implementation commit,
`eb043133`. Its independent comparison against PR #1256 is recorded in
`tests/kernels/README_glm5_tp4_indexer.md`. CP is disabled for the regression
gate; a latency increase of at most 0.5%, computed from unrounded medians,
passes. CP-on speedups compare with CP off in this same implementation.

Native measurements use real TP4/TP8 shard geometry, Conc 1, S=1/2/4/8,
a 3,000-token starting position, top-2048, FP8 block-scaled attention, and
flat BF16 attention/index caches. Expert weights are FP8 or MXFP4. Five
alternating rounds each use 50 warmup graphs and 4096 timed layer calls.
Each graph launches 16 layers at one weight address and advances the step
once. Timing uses the median across ranks, then across paired rounds.

Paged measurements preserve the PR's native MXFP4 MFMA, FP8 PTPC attention,
FP8 ATOM attention cache, BF16 index cache, and MTP4 request width of five.
C1/S5 and C2/S10 use 20 warmup graphs, three trials of 32 graphs, and seven
outer alternating rounds. Timing takes the median across trials, ranks,
and outer rounds, in that order.

Both comparisons share inputs, caches, outputs, and packed-weight addresses.
The baseline and CP-off variants additionally share scratch, peer mailboxes,
and step storage. CP-on uses the extra peer storage required by CP. Outputs
and ordered indices are checked before timing.

### Regression and correctness

All 40 default-path regression points pass the 0.5% gate. The largest
increase versus the preceding shared implementation is 0.443%. The first
commit independently passed all 20 comparisons against PR #1256, with a
maximum increase of 0.461%.

One native TP4 MXFP4 S=8 comparison initially measured +1.118% with
independently loaded executables. Their complete compiled modules, including
GPU binaries, were byte-identical; another process measured -0.742%. The
repeated baseline-graph control differed by only 0.017%. After verifying
complete module equality, the final control reused the same loaded executable
and measured +0.047% over nine rounds. The remaining 39 points use the
original paired executables. This controls code-loading variation without
changing the kernel or discarding the initial measurements.

The original CP validation passed 35 native cases and 18 paged cases.
The final CP-group tuning passed seven focused checks, including four new
cases, for 39 distinct native CP cases in total. These cover short contexts,
16K dispatch boundaries, 32K contexts, short requests in a larger allocation,
128K single-query groups, stable ties, shuffled pages, and both paged expert
compute paths. Output/cache bytes and ordered indices match CP-off controls;
306 graph replays of 16 layers also match. Of 75 configurations audited after
the final group tuning, 73 complete compiled modules remain byte-identical.
The two changed binaries, TP8/S4 at 128K and 1M, passed fresh native goldens
and paired performance comparisons against their preceding CP implementation;
both remain within the 0.5% regression gate. Their final CP-on/off measurements
are included in the long-context table below.

| CP binary revalidation | Previous CP (µs) | Final CP (µs) | Latency change |
| --- | ---: | ---: | ---: |
| TP8, S=4, 128K | 173.764 | 173.780 | +0.009% |
| TP8, S=4, 1M | 263.672 | 263.696 | +0.009% |

All numbers below are synthetic **per-layer latency**, not serving TPOT.

| Regression matrix | Cases | Maximum latency increase |
| --- | ---: | ---: |
| Native TP4, MXFP4 | 8 | 0.326% |
| Native TP4, FP8 | 8 | 0.055% |
| Paged TP4 | 4 | 0.443% |
| Native TP8, MXFP4 | 8 | 0.117% |
| Native TP8, FP8 | 8 | 0.153% |
| Paged TP8 | 4 | 0.157% |

### Native, Conc 1, position 3000

| TP | Expert weights | S | CP off (µs) | CP on (µs) | Speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | MXFP4 | 1 | 66.024 | 64.206 | 1.028× |
| 4 | MXFP4 | 2 | 74.772 | 71.626 | 1.044× |
| 4 | MXFP4 | 4 | 95.719 | 91.933 | 1.041× |
| 4 | MXFP4 | 8 | 156.739 | 143.495 | 1.092× |
| 4 | FP8 | 1 | 68.529 | 66.290 | 1.034× |
| 4 | FP8 | 2 | 79.411 | 76.061 | 1.044× |
| 4 | FP8 | 4 | 110.608 | 107.114 | 1.033× |
| 4 | FP8 | 8 | 185.462 | 173.957 | 1.066× |
| 8 | MXFP4 | 1 | 53.919 | 51.982 | 1.037× |
| 8 | MXFP4 | 2 | 60.600 | 58.679 | 1.033× |
| 8 | MXFP4 | 4 | 76.913 | 74.932 | 1.026× |
| 8 | MXFP4 | 8 | 114.265 | 104.370 | 1.095× |
| 8 | FP8 | 1 | 54.136 | 52.760 | 1.026× |
| 8 | FP8 | 2 | 61.107 | 59.543 | 1.026× |
| 8 | FP8 | 4 | 82.498 | 81.645 | 1.010× |
| 8 | FP8 | 8 | 131.179 | 122.661 | 1.069× |

### Paged MTP4, position 3000

| TP | Conc / S | CP off (µs) | CP on (µs) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 4 | 1 / 5 | 119.735 | 115.011 | 1.041× |
| 4 | 2 / 10 | 225.072 | 221.320 | 1.017× |
| 8 | 1 / 5 | 100.294 | 99.254 | 1.010× |
| 8 | 2 / 10 | 155.407 | 145.620 | 1.067× |

### Actual 128K and 1M contexts

These Conc 1 measurements use MXFP4 expert weights (A8W4), FP8
block-scaled attention, and flat BF16 caches. 128K means 131,072 tokens;
1M means 1,048,576 tokens. The starting position is `context - S`, so the
last query reaches the stated live context. These are full-context runs,
rather than short requests in oversized cache allocations.

The paired timing uses the same weights, inputs and caches, 16 layers
per graph, one step update, 20 warmup graphs and three alternating rounds
of 32 graphs. Medians are taken across ranks, then rounds. Each case also
runs the existing native golden and CP byte/order/replay checks.
The two TP8/S4 rows use the final binary revalidation protocol: five
alternating rounds, 50 warmup graphs and 4096 timed layer calls per round;
the 16-layer graph and median aggregation are unchanged.

| Context | TP | S | CP off (µs) | CP on (µs) | Speedup | Native golden |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K | 4 | 1 | 139.800 | 127.033 | 1.101× | Pass |
| 128K | 4 | 2 | 172.886 | 144.106 | 1.200× | 1-index difference* |
| 128K | 4 | 4 | 236.994 | 176.389 | 1.344× | Pass |
| 128K | 4 | 8 | 379.593 | 252.398 | 1.504× | Pass |
| 128K | 8 | 1 | 127.924 | 110.957 | 1.153× | Pass |
| 128K | 8 | 2 | 156.724 | 153.224 | 1.023× | 1-index difference* |
| 128K | 8 | 4 | 213.845 | 173.780 | 1.231× | Pass |
| 128K | 8 | 8 | 333.970 | 221.686 | 1.507× | Pass |
| 1M | 4 | 1 | 557.896 | 213.445 | 2.614× | Pass |
| 1M | 4 | 2 | 784.776 | 266.682 | 2.943× | Pass |
| 1M | 4 | 4 | 1207.871 | 371.830 | 3.248× | Pass |
| 1M | 4 | 8 | 2113.567 | 608.315 | 3.474× | Pass |
| 1M | 8 | 1 | 545.150 | 174.538 | 3.123× | Pass |
| 1M | 8 | 2 | 773.381 | 210.087 | 3.681× | Pass |
| 1M | 8 | 4 | 1200.967 | 263.696 | 4.554× | Pass |
| 1M | 8 | 8 | 2079.874 | 378.920 | 5.489× | Pass |

*The strict independent golden passes 14/16 cases. At 128K/S2 on both TP4
and TP8, one query selects a different boundary token: 2047/2048 indices
match, and the two reference scores differ by 0.000275. The index-Q difference
is 0.001953125 from projection rounding. Recomputing top-k independently
from the kernel's actual Q/W values gives an exact selected-set match.
CP-on and CP-off outputs, cache bytes, ordered indices, and replay results
are identical in all 16 cases. The original strict golden remains unchanged;
these two cases are recorded as reference mismatches, not golden passes.
All eight 1M cases pass the native golden.*

At 128K/TP8/S1, the initial full-TP CP group measured 131.205 µs against
127.913 µs with CP off. Two independent CP4 groups remove this regression:
110.957 µs against the paired 127.924 µs control. The 1M path retains CP8.

Functional reproduction using the existing native test:

```bash
python tests/kernels/test_glm5_monokernel.py \
  --npes 8 --model-tp 8 -S 1 --pos 131071 --max-seq 131072 \
  --indexer --indexer-cp --mxfp4 --iters 1

python tests/kernels/test_glm5_monokernel.py \
  --npes 8 --model-tp 8 -S 8 --pos 1048568 --max-seq 1048576 \
  --indexer --indexer-cp --mxfp4 --iters 1
```
