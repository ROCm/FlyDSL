# GLM mono-kernel indexer context parallelism

`Glm5MonoKernel(..., with_indexer=True, indexer_cp=True)` distributes indexer
work across the tensor-parallel ranks while keeping index projection, scoring,
exact global top-2048 selection, sparse MLA, MoE, and TP reductions in one
persistent kernel launch per rank and layer. The option defaults to `False`.

## Interface and scope

```python
op = Glm5MonoKernel(
    weights,
    samples=4,
    rank=rank,
    npes=8,
    group=group,
    with_indexer=True,
    indexer_cp=True,
    index_max_seq=131072,
    launches_per_step=16,
)
```

Forward arguments are unchanged. `samples` means consecutive query tokens
from one request, at positions `cur_pos + sample`; it does not add independent
request batching. `index_max_seq` must be a positive multiple of 64 and all
positions must fit the index, KV, PE, and RoPE capacities. Supported peer counts
remain 1, 2, 4, and 8.

All ranks must use identical indexer weights, inputs, positions, cache contents,
and CP settings. Attention and expert weights retain their existing TP shards.
The BF16 index cache remains physically replicated as `[index_max_seq, 128]`;
CP partitions computation, not cache storage. The change does not add ATOM's
FP4 paged index-cache layout or change MLA/MoE quantization.

Layers sharing scratch and peer storage must use compatible layouts and
distinct layer ordinals. Advance the device step once after the captured group
of layers, following the existing epoch contract.

## Short- and long-context algorithms

Each rank owns a contiguous, 64-token-aligned shard of the live causal context.
Sharding uses the live length rather than the allocated capacity, so a short
request in a large cache still distributes scoring. Every token receives the
complete score over all 32 index heads:

```text
score(token) = sum_head weight(head) * relu(dot(query(head), key(token)))
```

For live contexts up to 16,384 tokens, ranks publish their disjoint score shards
to every peer once, and each rank performs the original exact local radix
selection. One wave handles each destination peer, and consumers batch mailbox
loads. This replaces four global histogram/threshold exchanges on the initial
CP short path. When the compiled capacity is also at most 16,384, index-Q
projection rows are sharded and the BF16 results are exchanged inside the
kernel. Every scoring task still consumes the full 32-head query. Batched Q
loads apply to both CP settings.

For longer live contexts, 4K-token tasks build partial 256-bin histograms for
four radix rounds. Local coordinators reduce the partials; rank 0 combines
rank histograms and publishes the next threshold prefix. Distributed
count/scan/emit stages publish the final indices to every rank. This path
avoids exchanging the full score vector. LDS capacity remains bounded; scores
and partial histograms use global scratch.

Both paths retain exact top-2048 and the original ordering: ascending token IDs
strictly above the threshold, then the earliest token IDs tied at the threshold.
Contexts shorter than 2048 retain zero padding. Tagged peer payloads use separate
radix-round regions and two launch slots; CP adds no separate collective kernel.

When CP is disabled, capacities above 16,384 use the tiled local selector to
provide a same-capacity long-context control. The CP-on short/long dispatch is
per query and can handle queries on both sides of the boundary in one launch.
For a large allocated cache with a short live context, the measured gain also
includes the different selector path; it is not purely the gain from sharding.

The design borrows in-kernel peer exchange and separate short/long paths from
MiniMax M3. Its selection semantics differ: GLM needs exact global top-2048
tokens after aggregating all heads, rather than M3's per-head block selection.
The inspected ATOM M3 reference gates its CP path at more than 512 index blocks
(65,536 tokens); that reference does not establish that CP itself accounts for
M3's short-context improvements.

## Replay correctness fix

Extended TP8/S8 replay exposed an LDS lifetime race in the original short
selector. The final radix threshold occupied the same LDS words later reused
for scan wave totals. A fast wave could overwrite those words before a late
wave read the threshold, causing compaction to exceed top-2048 and overwrite
readiness tags. A workgroup barrier now completes all threshold reads before
that storage is reused. The long-context selector already used this barrier.

The diagnosis used independent SDMA snapshots after a stall, without inserting
GPU tracing instructions. Scores and Q were ready on every rank, while one
rank's final sample had corrupted ordered indices and token IDs in the following
readiness mailbox. Full task tracing had masked the race. A trial memory-clobber
change did not resolve it and is not retained.

## Validation

| Check | Result |
| --- | --- |
| Complete native pytest suite | 24 passed in 295.49 s; 24 bitwise CP-control and 24 long-replay comparisons |
| TP8, S8, position 1,048,568, capacity 1,048,576 | PASS; independent full golden relative-L2 0.439%; bitwise control and replay |
| Single-peer CP, S1, position 3000 | PASS; independent full golden relative-L2 0.208%; bitwise control and replay |
| TP8/S8 long-replay stress | Eight alternating CP-off/on trials passed; 32,768 timed CP-on calls per rank, plus warmup |
| Final performance matrix | All 20 configurations completed five paired trials |
| Style | Black, Ruff, and `git diff --check` passed |

The pytest log contains **5 individual full-golden comparisons skipped** under
the original rule for a near-tied expert-routing decision changed by a 1-ulp
attention difference. These are not skipped pytest cases and are not counted as
independent full-golden passes. Stage checks and CP-control/replay comparisons
still ran. No tolerance was relaxed.

The native checks retain every original numerical tolerance, the independent
golden, and its routing-mismatch reporting. Additional CP checks require equal
ordered indices across ranks and bitwise CP-on/off equality of layer outputs,
ordered indices, and KV/PE/index-cache updates. Correctness replay now covers
50+256 graphs of 16 layer calls and delays the last rank; it previously covered
only three graphs.

The test matrix covers S=1/2/4/8, empty context shards, top-k boundaries, equal
scores, capacity 64, the 16K boundary (including mixed short/long queries), large
caches with short requests, long contexts, and multiple peer counts.

```bash
# Native golden, bitwise CP control, and long graph replay.
python tests/kernels/test_glm5_monokernel.py \
  --npes 8 -S 8 --pos 3000 --indexer --indexer-cp --mxfp4 --iters 2

# Long-context functionality.
python tests/kernels/test_glm5_monokernel.py \
  --npes 8 -S 8 --pos 1048568 --max-seq 1048576 \
  --indexer --indexer-cp --mxfp4 --iters 1

# Repeat with and without --indexer-cp for a timing comparison.
python tests/kernels/test_glm5_monokernel.py \
  --npes 8 -S 4 --pos 3000 --indexer --indexer-cp --mxfp4 \
  --bench --bench-iters 4096

python -m pytest tests/kernels/test_glm5_monokernel.py -q
```

`--bench` remains timing-only. A functionality pass is a separate requirement.

## Performance

Measured on 2026-10-10 on node46, 8×MI355X/gfx950, TP8, **Conc 1**, synthetic
weights, with the fused indexer, MLA, and MoE included. The table compares CP
disabled and enabled in the same final source, not different engine deployments.
Both include the batched-Q improvement and the selector correctness fix.

Expert GEMM1 uses FP8 activations and MXFP4 weights; GEMM2 uses BF16 activations
and MXFP4 weights, both using BF16 MFMA. Attention uses block-FP8 weights;
index/KV caches are BF16. This is the A8W4 expert path, not A4W4.

Each graph repeats the same layer 16 times and advances the step once. Each arm
warms up 50 graphs and times 256 graph replays (4096 layer calls). Results are
medians across eight ranks, followed by medians across five alternating paired
trials. Packing and compilation are excluded. No competing GPU clients were
observed. Environment: Torch 2.10.0/ROCm 7.2.4, FlyDSL runtime 0.3.4.1.

### Short and medium contexts

| Cache capacity | Position | S | CP off (μs/layer) | CP on (μs/layer) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 3,000 | 1 | 54.478 | 53.286 | 1.022× |
| 4,096 | 3,000 | 2 | 61.186 | 59.738 | 1.024× |
| 4,096 | 3,000 | 4 | 77.247 | 76.925 | 1.004× |
| 4,096 | 3,000 | 8 | 115.465 | 104.925 | 1.100× |
| 8,192 | 8,184 | 1 | 61.059 | 57.037 | 1.071× |
| 8,192 | 8,184 | 2 | 68.340 | 64.222 | 1.064× |
| 8,192 | 8,184 | 4 | 85.301 | 79.980 | 1.067× |
| 8,192 | 8,184 | 8 | 129.030 | 112.003 | 1.152× |
| 16,384 | 16,376 | 1 | 76.731 | 71.870 | 1.068× |
| 16,384 | 16,376 | 2 | 85.736 | 77.555 | 1.105× |
| 16,384 | 16,376 | 4 | 108.513 | 93.441 | 1.161× |
| 16,384 | 16,376 | 8 | 153.976 | 126.423 | 1.218× |

### Long contexts

| Cache capacity | Position | S | CP off (μs/layer) | CP on (μs/layer) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 131,072 | 131,068 | 1 | 125.409 | 134.678 | 0.931× |
| 131,072 | 131,068 | 4 | 202.490 | 178.394 | 1.135× |
| 1,048,576 | 1,048,572 | 1 | 519.820 | 181.311 | 2.867× |
| 1,048,576 | 1,048,572 | 4 | 1120.516 | 264.000 | 4.244× |

### Short requests with larger allocated caches

| Cache capacity | Position | S | CP off (μs/layer) | CP on (μs/layer) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 131,072 | 3,000 | 1 | 104.174 | 62.022 | 1.680× |
| 131,072 | 3,000 | 4 | 137.143 | 94.346 | 1.454× |
| 1,048,576 | 3,000 | 1 | 341.257 | 61.010 | 5.593× |
| 1,048,576 | 3,000 | 4 | 426.724 | 95.686 | 4.460× |

Short-context CP now ranges from nearly neutral at 3K/S4 (1.004×) to 1.218×
at 16K/S8. At 3K, S1/S2 improve by about 2%, while S8 improves by 10%. The
five paired-trial speedups were 1.020–1.023× for 3K/S1 and 1.099–1.101× for
3K/S8. These small S1/S2 gains should not be interpreted as a uniform speedup
for every workload.

At 1M, S1/S4 retain large gains (2.867× / 4.244×). At 128K, S4 improves
(1.135×), but S1 regresses (0.931×). CP should therefore remain workload
dependent; these measurements do not justify enabling it unconditionally.

### Graph stage observations

Separate task tracing at position 3000, capacity 4096, uses a 16-call graph
and 50 graph warmups. The table reports the stage completion timestamp from
the first task start, taking the median across eight ranks. It includes waiting
on dependencies; the values are not isolated operation durations.

| S | Stage | CP off completion (μs) | CP on completion (μs) |
| ---: | --- | ---: | ---: |
| 1 | `index_q` | 12.27 | 7.38 |
| 1 | `index_score` | 15.41 | 12.71 |
| 1 | `index_select` | 22.15 | 19.92 |
| 4 | `index_q` | 17.15 | 9.54 |
| 4 | `index_score` | 20.20 | 17.14 |
| 4 | `index_select` | 28.26 | 26.59 |

Index-Q and selection complete earlier with CP. Later MLA/MoE stages and their
dependencies limit how much of that improvement reaches total layer latency.
Tracing changes instruction scheduling, so the performance tables above use
separate runs without tracing.

The final compiler-cache audit examined 179 artifacts, including
historical experiments and all newly compiled final variants. One discarded
v7 polling-instrumentation variant reported 208 bytes of private scratch; its
compilation timestamp falls within that diagnostic run. The other
178 artifacts, including final variants, reported private-segment
size 0. Maximum LDS across the audited cache was
95,824 bytes. Final variants
have no private scratch allocation; this does not imply that
scalar-register spills into vector registers are absent.

These are single-layer steady-state latencies, not full-model TPOT. The results
do not establish full-model accuracy, independent-request batching, or an
end-to-end engine speedup. CP remains an explicit option, disabled by default.
