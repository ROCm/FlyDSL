# GLM-5.2 TP4 fused indexer: C1/C2 MTP4

Measured on four MI355X GPUs on 2026-10-10. This branch starts from FlyDSL
`main` at `3ad47c18` and compares against ATOM PR #2435 at `a3e2b1a25`.
The graph has 16 GLM layers per replay. Times below are the median rank's
microseconds per layer; all runs use native MXFP4 MFMA, FP8 PTPC attention,
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
| FlyDSL fused TP4 | **118.42** | **224.49** | Index K/Q/W projections, BF16 index cache, score, top-k, attention and MoE |

The fused kernel is 3.6% slower than the C1 lower bound and 8.2% slower than
the C2 lower bound. ATOM's deployed path also runs RMSNorm, a duplicate QKV
projection, index Q and K/weight projections, and selected-index conversion
before its MonoKernel. These missing costs determine the actual comparison;
this machine has no GLM checkpoint or serving setup to measure them. The
FlyDSL fused index cache is BF16, whereas ATOM's external indexer uses a
quantized FP8 cache, so the two indexer implementations also differ in
precision and storage.

The TP8 roughly 2× indexed kernel result does not transfer directly to TP4.
TP4 has 16 local attention heads, filling all 256 resident CTAs for the Q-B
projection. The index Q projection therefore runs after Q-B on those CTAs;
TP8 uses eight local heads and has spare CTAs for overlap. In the instrumented
C2 run, the final expert-down stage alone took about 55 µs. A 2× result for
this full TP4 layer would require the ATOM projection and conversion work
omitted above to add another 123 µs for C1 or 241 µs for C2. No such cost has
been measured.

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
`AITER_DISABLE_FLYDSL_TOPK_DECODE=1` with this installed AITER runtime.

The fused TP4 API takes a BF16 index cache with 128 values per physical row,
an `int32` block table for each request, five rows per request, and the ATOM
`positions`, `slot_mapping`, and `sparse_kv_indptr` metadata. It maps selected
logical positions to physical attention KV slots inside the persistent
kernel. The current fused path is limited to an index window of 4,096 tokens
and does not implement ATOM's FP8 index cache format.
