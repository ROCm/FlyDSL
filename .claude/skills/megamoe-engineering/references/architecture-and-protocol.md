# MegaMoE architecture and protocol

This reference captures durable reasoning and a dated architecture snapshot.
Always inspect the current source before relying on a launch boundary or tune
default.

## Mathematical contract

For source token `t`, router slot `s`, selected global expert `e(t,s)` and
routing weight `p(t,s)`, MegaMoE must compute the standard routed-MoE result:

```text
u(t,s) = W2[e] (SwiGLU(W1[e] x[t]))
y[t]   = sum_s p(t,s) * u(t,s)
```

The fused implementation may change layouts, quantize intermediate values,
overlap communication with GEMMs and scatter results directly to the source
rank. It may not change which activation is quantized, which expert owns a
route, where the routing weight is applied or which slots are combined.

At the operator boundary each rank owns some tokens with the complete hidden
dimension. Router IDs are global expert IDs. An expert owner is normally:

```text
owner_rank  = global_expert // experts_per_rank
local_expert = global_expert % experts_per_rank
```

The result must return in the original source-rank/token order. A source map
therefore needs source rank, source token and top-k slot even when payload rows
are deduplicated.

## Production AITER snapshot

As of 2026-09-09, AITER main contains the architecture restored by AITER #5346
(the corrected successor of #5001). Verify the current tree under
`aiter/ops/flydsl/kernels/mega_moe/` before changing it.

The logical pipeline is:

```text
BF16 input
  -> fixed-slot: standalone 1x32 MXFP8 quant
     compact: prepare kernel fuses 1x32 MXFP8 quant with count/group/plan
  -> Stage1: P2P dispatch + expert-major FP8xFP4 GEMM1
             + SwiGLU + 1x32 MXFP8 requant
  -> Stage2: FP8xFP4 GEMM2 + routing weight + P2P scatter
  -> terminal combine: sum top-k slots, dequantizing FP8 P2P payload if used
```

The host wrapper may call Stage2 and combine as one logical operation while
they remain distinct GPU kernels. Confirm launch count and overlap from the
trace rather than from the Python function name.

Current production concepts include:

- paired Stage1/Stage2 bundle selection and AOT preload;
- compact prepare outside the high-LDS GEMM1 kernel;
- BF16-to-MXFP8 quant work fused into compact prepare;
- deterministic runtime fanout/payload dedup metadata;
- producer CTAs followed by oversubscribed consumers;
- fixed-slot direct dispatch for small MTPR;
- compact group-major dispatch for larger capacities;
- ready-tile order for sufficiently skewed compact workloads;
- source-indexed payload storage at the largest capacity;
- zero-token-rank participation and generic top-k support currently bounded by
  the implementation.

These are related protocol pieces, not independent flags. Moving only one part
between branches can create a build that compiles but has mismatched buffer or
ready semantics.

## Customer FlyDSL-universe paths

The ByteDance branch `ghu/mega_moe_v1_copy` is a distinct product fork. Its
delivered code must not be assumed identical to AITER main.

The important paths are:

| Path | Activation flow | Notes |
|---|---|---|
| Native MX A8W4 | BF16 -> 1x32 MXFP8/E8M0 -> dispatch/GEMM1 | Decode was originally restricted and tuned through TPR 512; prefill enablement/tuning is separate work. |
| SmoothQuant A8W4 | BF16 -> SmoothQuant to INT8 -> dispatch/GEMM1 | Customer-specific. The delivered design keeps SmoothQuant before Stage1; do not silently move quantization after BF16 dispatch. |
| A8W8 smooth prefill | customer-specific quant/communication path | Treat separately; do not infer its transport from native MX A8W4. |

At the 2026-09-09 delivered Byte head `5846a600`, native M13 A8W4 explicitly
accepted only `tokens=MTPR` through 512. Larger MTPR support therefore needs
protocol/capacity validation before tuning. This commit is historical evidence,
not a promise about the latest branch.

The abandoned experiment that embedded SmoothQuant inside Stage1 was not the
final customer baseline. It increased register pressure and degraded many
shapes. Preserve the delivered pre-Stage1 semantics unless a new trace-backed,
fully equivalent design wins the complete matrix.

## Fixed-slot and compact

The exact threshold belongs to the current selector. In the #5346 AITER
snapshot, `FIXED_SLOT_MAX_MTPR=255`; the Byte fork can use different boundaries.

Fixed-slot:

- derives destination slots from fixed capacity;
- avoids the complete compact prefix/packing plan;
- targets small CUDA Graph decode capacities;
- has direct-dispatch constraints, including the supported experts/rank range;
- still needs all ranks to participate, including zero-token ranks.

Compact:

- counts routes, groups them and builds expert-major row ranges;
- returns destination-owned bases to sources;
- allocates rows for actual traffic plus alignment rather than every fixed slot;
- is the normal large-MTPR/prefill path;
- can use chunk/tile readiness and fanout metadata.

`MTPR=tokens` and `MTPR=MAX` exercise different capacity, protocol and buffer
costs for the same actual token count. Always scan both when the deployment can
use both.

## Prepare, Stage1 and role scheduling

Separating compact prepare from GEMM1 is deliberate:

- count/group/plan needs modest LDS but serialized cross-rank decisions;
- GEMM1 needs substantial LDS/VGPR and benefits from a compute-focused grid;
- quant can run as independent specialist CTAs in the prepare launch;
- prepare publishes a complete layout before Stage1 consumes it.

Current Stage1 assigns producer/consumer work according to the current source
implementation. Do not rely on remembered claims such as "no atomics" or
"static work". At the #5346 snapshot, compact/fixed consumer paths still use
sharded atomic work heads, while arrival tickets and oversubscription ensure
required roles can become resident. A future static assignment is valid only
if it proves load balance, producer residency and every supported shape.

The forward-progress question is more important than the exact role count:

1. Can every required producer become resident while consumers wait?
2. Do producer/owner CTAs retire or join useful compute after finite role work?
3. Can queued consumers backfill released CUs?
4. Does any persistent consumer wait on work that only a non-resident producer
   can publish?

## Fanout and payload dedup

"Fanout" means one transmitted activation payload can feed more than one
selected expert at the same destination. The current pair specialization:

1. selects a pair of local expert IDs for each destination at runtime;
2. detects tokens whose routed top-k contains both experts;
3. emits one canonical shared segment instead of two activation payload rows;
4. preserves the two routing weights, source slots and expert identities;
5. computes/scatters both expert results with aligned-pair or residual Stage2
   work.

The runtime pair table is double-buffered by parity and kept out of the compile
key so one AOT artifact can serve different routing distributions. Expert IDs,
not only a 64-bit bitmap, are required when a destination owns more than 64
experts. Top-k grouping width and metadata encoding remain explicit limits.

Dedup saves the large activation+scale payload; it does not permit dropping
per-route weights or return mappings. Audit the non-deduplicated fallback and
mixed pair/non-pair routes.

## Stage1 payload readiness

A compact plan creates expert/tile metadata and producer tasks. A producer
typically writes:

- quantized activation row;
- 1x32 activation scales;
- routing weight;
- source map or route metadata.

Correct P2P publication is:

```text
remote payload stores
  -> wait for stores as required
  -> system release fence
  -> system-scope ready publication

consumer wait
  -> observe expected epoch/count
  -> system acquire fence
  -> all consuming waves converge
  -> load payload / execute GEMM
```

Plan-ready, pair-order-ready, payload-ready, tile-ready, ready-queue epoch and
launch-ready have different meanings. Reusing one as another needs a proof.

Payload chunks can release tiles before a source-to-expert transfer is fully
complete. The planner must compute each tile's exact expected publisher count;
the final publisher may enqueue that tile. Any layout transition must reset
ready state in the same coherence domain used by publishers.

## Stage1 GEMM1 and epilogue

The destination consumes expert-major rows:

```text
FP8 activation [M,H] x MXFP4 W1 [expert,H,2I]
  -> FP32 accumulators
  -> optional bounded SwiGLU semantics
  -> [M,I]
  -> 1x32 MXFP8/E8M0 requant for Stage2
```

Compile invariants include the MFMA fragment geometry, tile-N per wave,
Stage1 SBM, tile-K, LDS layout and output-buffer addressing. Validate these in
the selector before launching. A selector that returns `NUM_ACC_N=1` to a
kernel requiring an even accumulator count is a functional failure, not a tune
miss.

## Stage2 and combine

Stage2 maps `(M block, N block, expert)` work, performs GEMM2, applies the
routing weight and scatters to the source rank/token/top-k slot. Persistent,
strided/skew and aligned-pair variants are scheduling choices; the source map
and scatter wire format are common contracts.

Large-MTPR paths can quantize P2P Stage2 output to FP8 blockwise 1x32 plus E8M0
scales. Smaller paths can scatter BF16. This choice must be identical across
ranks for the operator instance and included in Stage2/bundle identity.

The final result is the sum of top-k slots. When timing, distinguish:

- Stage2 GEMM/scatter kernel;
- combine kernel;
- the host helper that may include both;
- the full first-to-last graph span.

## Capacity and addressing invariants

Check these from current code for every new profile:

- `experts % world_size == 0`;
- MTPR is supported, normally a positive power of two;
- every rank's actual tokens are `<= MTPR`;
- source rank/token/top-k encoding has enough bits;
- route/fanout segment encoding admits the global expert count and extra group
  segments;
- Stage2 block-M divides Stage1 sort-block-M;
- quant group dimensions are divisible by 32 where required;
- local and symmetric buffers cover worst-case routing and padding;
- a buffer-resource descriptor never addresses beyond its byte limit;
- offsets crossing 2/4 GiB are computed and rebased with 64-bit arithmetic;
- bundle identity contains every dtype/layout/protocol/geometry value that can
  change generated code or ABI.

At MTPR 32768, activation receive storage can exceed 4 GiB. The safe pattern is
to compute row/tile bases in 64 bits and create a local descriptor at the
rebased address, rather than issue one descriptor with a huge element offset.

## AOT and preload

Disk AOT and module preload solve different costs:

1. AOT creates compiled artifacts at image/build time.
2. Preload materializes the artifact and HIP module before the first request.

A build that contains artifacts can still show `hipModuleLoadData` on the first
request if it was not preloaded. Conversely, calling the real communication
kernel on dummy pointers is not a safe preload strategy.

Production AOT should reuse the exact runtime wrapper, selector and compile
kwargs. Bundle/job identity must include rank, EPR, world size, model shape,
top-k, MTPR, quant dtype/layout, wire format and all code-generating options.
Validate by building a fresh cache and starting a second process with strict
run-only mode. Missing artifacts must fail rather than silently JIT.

## Concurrency contract

Historically, one `MegaMoEV2` instance supports one ordered in-flight launch.
The instance owns mutable parity, counters, buffers and ready state. Do not
enable concurrent multi-stream/TBO re-entry unless the current implementation
has independent per-ubatch state, deterministic cross-rank launch order and a
dedicated multi-stream Graph test.
