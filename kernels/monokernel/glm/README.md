# GLM-5 indexed decode MonoKernel

`Glm5MonoKernel` uses one persistent launch per tensor-parallel rank for
attention, routing, experts, and both TP reductions. With `with_indexer=True`,
the same launch also projects index Q/K/weights, updates the index cache,
scores tokens, and selects the top 2048 logical positions. TP4 and TP8 share
`kernel.py`, `layout.py`, and `op.py`.

| Shard geometry | TP4 | TP8 |
| --- | ---: | ---: |
| Local attention heads | 16 | 8 |
| Local expert intermediate width | 512 | 256 |
| Indexer heads × dimension | 32 × 128 | 32 × 128 |

## Operator contract

Import `Glm5MonoKernel` from `kernels.monokernel.glm` and select the weight
geometry with `glm5_tp_config(4)` or `glm5_tp_config(8)`. The host API accepts
1–16 total query tokens, including chunked calls. Native ATOM MXFP4 weights
support every row count in that range; the legacy path uses
`GLM5_KERNEL_SAMPLES`. Native batches above twelve rows require a full
256-CU MI355X and execute bounded row tiles inside the persistent kernel.

Attention weights, KV layout, expert storage, and activation precision select
compile-time load and compute paths. `native_fp4_mfma=True` requires ATOM
MXFP4 expert storage. `moe_activation_dtype="fp4"` enables A4W4 on that path;
the default activation format is FP8.

The fused indexer accepts either flat BF16 KV/PE and index caches for one
consecutive request, or ATOM attention KV with a physically paged BF16 index
cache. It does not implement ATOM's FP8 index-cache format. For paged calls:

- `index_request_width` consecutive query rows belong to one request; set it
  to five for MTP4 C1/S5 or C2/S10. Different requests may start at different
  positions.
- `block_tables` is contiguous int32 with one row per request. Attention and
  index caches share its physical page IDs. The kernel converts selected
  logical positions to physical attention slots.
- Supply `positions`, `slot_mapping`, and `sparse_kv_indptr` for the ATOM cache
  layout. Each physical index-cache row contains 128 BF16 values.
- `index_max_seq` must be positive, divisible by 64 and `index_page_size`, and
  large enough for every query. The block-table stride must cover that window.

The common operator defaults to FP32 RoPE tables. Set `rope_dtype="bf16"`
explicitly for BF16 tables, as the paged benchmark does. Fused indexer calls
require `dcp_size=1` and do not support host chunking.

Graph buckets can reuse a layer's packed weights, and layers may share runtime
buffers. Assign each layer a distinct ordinal below `launches_per_step`;
after a graph's layer launches, call `advance_step()` once. Mailbox epochs
and the step update are device operations compatible with HIP graph capture.

## Indexer context parallelism

Enable `with_indexer=True, indexer_cp=True` to distribute index scoring across
TP ranks. Each rank computes a contiguous, 64-token-aligned context shard,
summing all 32 index heads. Indexer weights, inputs, positions, and cache
contents must be identical across ranks; the caches remain replicated.

TP4 uses four participants. TP8 single-row tiles with compiled capacity above
16K and at most 128K use two independent groups of four ranks; other TP8 cases
use all eight. Attention and MoE reductions still span the full TP group.

Up to 16,384 live tokens, ranks exchange score shards and run the local radix
selector. If the compiled capacity also fits that limit, index-Q projection
is sharded. Longer contexts use bounded 4K scoring tiles and four distributed
radix rounds, followed by count, offset, and emit stages. A short request in
a larger allocation partitions scoring by its live length.

Selection emits ascending token IDs above the threshold, followed by the
earliest IDs tied at the threshold, with zero padding below 2048 tokens.
Tagged peer mailboxes with two launch slots keep consecutive selections
separate. CP adds no host collective or separate communication kernel.

## Validation and benchmarks

Run from the repository root with a compatible FlyDSL runtime. GPU checks
require MI355X (`gfx950`); select four or eight GPUs to match `--npes`.

```bash
# Host token limits; no GPU allocation required.
python -m pytest tests/unit/test_glm_token_limit.py -q

# Independent PyTorch attention/MoE/indexer goldens with nonzero weights.
python tests/kernels/test_glm5_monokernel.py \
  --npes 4 --model-tp 4 -S 8 --pos 3000 --indexer --indexer-cp --mxfp4 --iters 2

# Paged MTP4: golden selection, nonzero attention, shuffled physical pages,
# CP-on/off output/cache/index equality, and repeated graph replay.
python tests/kernels/bench_glm5_monokernel.py \
  --npes 4 --samples 5 10 --fused-indexer --indexer-cp \
  --check-golden --check-attention --shuffled-pages --check-cp --replays 4

# Synthetic layer timing: 16 launches per graph, 20 warmup graphs,
# three trials, median across trials and then ranks.
python tests/kernels/bench_glm5_monokernel.py \
  --npes 4 --samples 5 10 --fused-indexer --replays 32

# Long-context CP with the independent native golden.
python tests/kernels/test_glm5_monokernel.py \
  --npes 8 --model-tp 8 -S 1 --pos 131071 --max-seq 131072 \
  --indexer --indexer-cp --mxfp4 --iters 1
```

Repeat the paged commands with `--npes 8` for TP8, without `--indexer-cp` for
local selection, or without `--fused-indexer` for externally supplied indices.
`--native 0` selects software expert dequantization. The paged fused fixture
supports S=1/5/10. The golden test's `--model-tp` selects actual model shard
geometry; omitting it retains the historical TP8 shard used for peer-protocol
checks with fewer GPUs.

The paged benchmark randomizes indexer weights, inputs, and index cache.
Attention and expert timing weights are otherwise zero-filled;
`--check-attention` adds nonzero attention weights and KV. Its selected-set
golden requires at least 99% overlap. CP control checks separately require
exact ordered indices, output/cache bytes, and 306 graph replays of 16 layers.
The nonzero expert golden in `test_glm5_monokernel.py` covers FP8 and
native-storage MXFP4 with software dequantization. It does not enable
`native_fp4_mfma=True`; native MFMA needs a separate numerical comparison.

These timings measure synthetic per-layer latency. They exclude checkpoint
prefill, scheduling, and serving overhead. Historical ATOM proxy comparisons
are retained in [PR #1256](https://github.com/ROCm/FlyDSL/pull/1256)'s history;
the benchmark here requires no ATOM checkout.
