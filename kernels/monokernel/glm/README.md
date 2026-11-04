# GLM-5 indexed decode MonoKernel

`Glm5MonoKernel` uses one persistent launch per tensor-parallel rank for
attention, routing, experts, and both TP reductions. With `with_indexer=True`,
the same launch also projects index Q/K/weights, updates the index cache,
scores tokens, and selects the top 2048 logical positions. TP4 and TP8 share the same stage emitters and compile-time plan.

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
consecutive request, or ATOM attention KV with a physically paged index
cache. The index cache defaults to BF16; `indexer_quant=True` selects the
mono-kernel's logical MXFP4 format and quantizes both index queries and keys.
ATOM's shuffled FP4 and FP8 index-cache formats require conversion before use.
For paged calls:

- `index_request_width` consecutive query rows belong to one request; set it
  to five for MTP4 C1/S5 or C2/S10. Different requests may start at different
  positions.
- `block_tables` is contiguous int32 with one row per request. Attention and
  index caches share its physical page IDs. The kernel converts selected
  logical positions to physical attention slots.
- Supply `positions`, `slot_mapping`, and `sparse_kv_indptr` for the ATOM cache
  layout. Each physical index-cache row contains 128 BF16 values by default,
  or 68 bytes with `indexer_quant=True`: 64 bytes of low-nibble-first E2M1
  values and four E8M0 scales, one scale per 32 values.
- `index_max_seq` must be positive, divisible by 64 and `index_page_size`, and
  large enough for every query. The block-table stride must cover that window.

The common operator defaults to FP32 RoPE tables. Set `rope_dtype="bf16"`
explicitly for BF16 tables, as the paged benchmark does. Fused indexer calls
require `dcp_size=1` and do not support host chunking.

Graph buckets can reuse a layer's packed weights, and layers may share runtime
buffers. Assign each layer a distinct ordinal below `launches_per_step`;
after a graph's layer launches, call `advance_step()` once. Mailbox epochs
and the step update are device operations compatible with HIP graph capture.

## Optional FP4 indexer

Enable `with_indexer=True, indexer_quant=True` and supply a contiguous uint8
index cache with shape `[rows,68]` or `[pages,page_size,68]`. The helper
`pack_index_fp4` in `index_quant.py` converts BF16 keys to this format;
`unpack_index_fp4` decodes them for inspection and independent references.
Both helpers run outside HIP graph capture. The kernel quantizes new keys,
updates their physical cache rows, and scores against quantized queries.

FP4 changes the numerical contract relative to BF16 indexing. Its reference
must include query and key quantization; agreement with that reference does
not establish full-model quality. The TP4 C2/S8 real-checkpoint layer fixture
has been validated with CP off at 128K and 256K. It passes independent
quantization-aware goldens, exact v163 equivalence, cache/state checks and
HIP graph replay at both contexts. Paired measurements reproduce v163
latency; 128K has an independent timing confirmation, while additional 256K
timing runs were excluded by the shared-node contention guard. These checks
do not establish full-model quality or serving performance. The CP-on
implementation is retained but has not been GPU-tested in this port. The
commands below exercise the existing BF16 path unless extended with a
quantization-aware fixture.

### Indexer launch organization

The kernel composes pre-attention, indexer, attention and MoE IR emitters.
Compile-time partition selection controls which stages contribute to each
kernel; calling an emitter does not add a device launch.

CP-off uses one fused kernel per TP rank. BF16 indexing also retains one fused
kernel with CP enabled. Enabling both `indexer_quant=True` and `indexer_cp=True`
builds separate front, index, and back kernels and launches them in order on the
same stream. The index partition uses its own grid and LDS layout.

## Source organization

| Module | Responsibility |
| --- | --- |
| `kernel.py` | Public factory, per-tile setup, stage order, tile barriers and launches |
| `plan.py` | Compile-time format selection, task placement, cohorts and overlaid LDS geometry |
| `context.py` | Compile-time plan, execution metadata and grouped shared operations |
| `interfaces.py` | Semantic inputs, outputs, weights and lazy tensor views for stage boundaries |
| `preattn.py` | Input/QKV normalization, projections, RoPE, index Q/K preparation and cache publication |
| `indexer.py` | BF16/FP4 index scores and exact local/distributed sparse selection |
| `attention.py` | Sparse MLA, split merge, W_UV/W_o and attention TP residual |
| `moe.py` | Router/activation publication, shared/routed gate/up and SiLU, down/route/TP reduction |
| `cache.py` | Deferred KV/PE commit after all cache readers retire |
| `tools/` | Shared memory, reductions, normalization, quantization, GEMM, routing and communication primitives |

`kernel.py` shows the complete per-tile phase sequence: input/QKV, query norm,
cache publication, query projection/RoPE, index-query quantization, absorbed
query, index scoring/selection, sparse attention, split merge/W_UV, W_o/TP
residual, router, expert gate/up, expert down and deferred cache commit. It also
owns the front retirement and serial-tile barriers, partition selection and
single versus three-kernel launches. Pre-attention and attention factories
return named inline stages so their implementations can stay in their modules
without hiding the execution order.

Stage interfaces take `inputs`, `outputs` and `ctx`. Tensor contracts live in
`interfaces.py`; `ctx` contains only the compile-time plan, thread/tile state,
workspace, token metadata, communication and shared operations. Model weights
and activation edges are explicit op inputs, not fields in a global state bag.
`LinearWeight(weight, scale)` keeps each projection's parameters together.
Mailboxes are lazy views of existing scratch: resolving them at each original
load/store site preserves the readiness protocol and IR emission order.

Names follow AITER's op contracts: `hidden_states`, `q`, `kv_buffer`,
`kv_indices`, `gating_output`, `topk_ids`, `topk_weights`, `w1` and `w2`.
The main sequence calls `rmsnorm_qkv`, `rmsnorm_q`, `rope_cache`, `q_proj`,
`index_quant`, `q_absorb`, `mla_decode`, `mla_reduce`, `output_projection`,
`router`, `moe_stage1` and `moe_stage2`. This is semantic alignment, not a
call-through to AITER or a torch-tensor ABI: sparse MLA retains separate latent
and RoPE buffers and online-softmax max/sum scratch; `mla_reduce` includes W_UV.
The router publishes sigmoid scores and quantized activations; the expert
stages retain GLM's bias-corrected top-k, shared expert and TP residual rules.
Some down schedules recompute routes from scores, so those scores and the
correction bias remain explicit stage-2 inputs as well as top-k outputs.

The stage-local aliases preserve the DSL's local and loop-carried scopes.
Context objects exist during Python IR construction, with no device allocation
or additional launches. `KernelPlan` is a named tuple so specialization values
participate in FlyDSL's closure cache key; stage and tool factories use
`@flyc.jit` so their transitive source dependencies are tracked.
For older runtimes whose dependency walk stops at directory boundaries, the
factory also captures `stage_source_key()` in the kernel closure. This digest
covers every GLM Python module and tool; it affects cache identity and emits no
device instructions.

All expert formats share `run_units → mma_units → reduce_rows` and the same
routing, activation and communication stages. GEMM unit constructors select
weight layout, operand decoding and hardware instructions. A8W8, A8W4 and A4W4
retain their different scale groups and rounding boundaries: per-128 FP8 versus
per-32 E8M0 scaling, the A4 BF16 input boundary, and native scaled versus decoded
BF16 MFMA. `quant_moe` is the common activation entry. These format differences
must remain explicit when extending a shared tool.

The cleanup removed `_ld_pair`, its private `_qptr`, `emit_ptpc_weight`,
`quant_block`, and the selector's `score_key`: none has a live caller, including
inside compile-time branches. No environment variable enables them. Public
BF16/FP8/MXFP4, local/CP/DCP, cohort and serial-tile specializations
remain supported. A historical version label or a default-off option is not
proof that a branch is dead.

Production GLM code has no timeline option, clock writes, stamp/mark helpers or
timeline buffer allocation/reporting. The indexer pointer-table argument is
named `index_params` and contains only its seven functional weight pointers.
The single-row MoE path no longer publishes diagnostic `sel`/`prob` records;
its consumers recompute routing. Tests check the selected experts through
nonzero gate/up intermediates and weighted output. Grouped execution retains
route mailboxes required by global gate/up and cohort-pair down consumers.
Read-only mailbox observations live in `tests/kernels/glm_observer.py`; test
fixtures attach those observers without adding device instrumentation.

The launcher accepts device pointers with ATOM GLM/model-op names. Token
metadata uses `positions`, `slot_mapping`, `sparse_kv_indptr` and `block_tables`;
`decode_position` is the scalar decode cursor. External sparse keys (`kv_indices`)
and the fused indexer's persistent `index_cache` are separate inputs. Weights
use normalization/projection names; absorbed MLA W_K/W_V use `k_proj`/`v_proj`,
and AITER MoE's `w1`/`w2` represent gate/up and down. Execution state uses
`workspace`, `peer_buffer`, `peer_ptrs`, `step_counter`, `tp_rank` and `layer_id`.
The host adapter binds keywords and flattens them at the device boundary shared
by full and split launches. The public `forward` signature remains compatible.

Run CPU quantization and token-limit checks with:

```bash
COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx950 HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
  python -m pytest -q tests/unit/test_glm_index_quant.py \
  tests/unit/test_glm_token_limit.py
```

GPU correctness and performance must be reported separately from these CPU
checks. Structural validation against the previous implementation compares
location-free kernel IR with identical options; full IR and compiled artifacts
belong on the remote test server.

The preceding source snapshot was checked against numerical-fix control `1bae7130`
with 54 identical location-free kernel IR specializations, six successful full
ROCm code-generation pairs, and 56 CPU unit checks. Coverage includes the
C1/C2 × S4/S8 matrix at 8K/128K/1M capacity, BF16/FP4 indexing, external
indexing, A8W8/A8W4/A16W4/A4W4, CP, TP8, serial/cohort tiles and the then-present timeline.
That snapshot's MI355X GPU validation against `1bae7130` passed all 24 cases (96 rank
results): CP-off TP4, C1/C2 × S4/S8, contexts 8K/128K/1M, indexer-off layer 3
and FP4-indexer layer 6. Independent golden, cache/intermediate, BF16 fallback,
poison, delayed-rank, two-step and 256-replay checks retained their tolerances;
FP4 indices, scores and output were bitwise equal to the control. All 72 CPU
binary plans compiled, and executable `.text` matched for all 24 paired
configurations. Five paired GPU-event timing trials per case used max-rank
latency and serial TP4 jobs. The initial 128K indexer-off C1S8 +1.13% changed
to -0.36% in a fresh four-shape confirmation; no repeatable performance
regression was observed. Other initial changes were -0.72% to +0.82%.
All seven GPU jobs completed uncontended and released their GPUs. Evidence:
`glm_refactor_gpu_20261029/GPU_REPORT.json` under the remote test workspace.
This GPU coverage does not extend to CP/DCP or other shapes and weight formats.
A DCP8 frontend probe fails in both control and refactor
on the tested legacy runtime (`Cannot extract IR values from fp8x2` in UV
owner staging); the DCP implementation is retained without claiming new coverage.


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
