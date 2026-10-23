# Models, geometry and shape onboarding

Model names are convenient labels, not compile contracts. Store explicit
geometry and verify it against the exact model revision and serving stack.

## Dated geometry anchors

The following values are working anchors collected during 2026 MegaMoE work.
Re-check official config/model code before adding production support.

| Model/path | Hidden | Per-expert intermediate | Routed experts | Top-k | EP | Experts/rank | Important semantics |
|---|---:|---:|---:|---:|---:|---:|---|
| DeepSeek V4-Pro | 7168 | 3072 | 384 | 6 | 8 | 48 | SwiGLU limit historically 10; production AITER/ATOM anchor. |
| ByteDance M13 | 3584 | 1280 | 384 | 8 | 8 | 48 | Native MX A8W4 and customer SmoothQuant A8W4 are distinct input/scale paths. |
| GLM-5.2 routed MoE | 6144 | 2048 | 256 | 8 | 8 | 32 | Shared expert is outside the routed MegaMoE result and must be integrated separately. |
| Kimi-K3 routing anchor | 3584 | 512 | 896 | 16 | 8 | 112 | Top-k and EPR exceed older bitmap/fixed assumptions; activation/integration must be verified. |

Do not call a shape supported because the selector compiles. Routing semantics,
activation/clamp, shared experts, scale layout, weight format and framework
combine behavior are part of support.

## Canonical shape tuple

Record at least:

```text
(hidden, intermediate, num_routed_experts, topk, ep_size,
 experts_per_rank, activation, activation_limit,
 input_dtype, weight_dtype, input_scale_layout, weight_scale_layout,
 route_weight_dtype, output_dtype, tokens_per_rank, MTPR)
```

Also record whether dimensions are logical or already account for gated
activations. Never infer `experts_per_rank` with integer division unless the
model/framework guarantees even placement.

## Onboarding a model

### 1. Establish authoritative semantics

Use the exact Hugging Face/config revision or framework model implementation.
Verify:

- routed expert count and top-k;
- hidden and routed intermediate dimensions;
- activation, gate/up packing and any clamp/limit;
- shared-expert count, dimensions and fusion location;
- router score normalization and routing-weight application;
- expert placement and EP group size;
- input/weight/output quantization and scale shapes;
- padding, noaux/top-k grouping or expert-bias behavior.

Save the source URL/revision in the task report. Model repositories can update;
do not turn an unpinned web lookup into an undated table entry.

### 2. Derive distributed geometry

Compute and validate:

```text
experts_per_rank
maximum routed assignments = MTPR * EP * topk
per-destination payload capacity
source-token/slot encoding range
fanout/dedup ID range
dispatch and combine buffer bytes
Stage1 and Stage2 tile divisibility/tails
```

Top-k greater than six and EPR greater than 64 are deliberate stress axes.
Audit every packed bitfield, bitmap and fixed-slot assertion. Do not add a
model-name exception around an insufficient encoding.

### 3. Add every config layer

Update together:

1. network/test registry;
2. Stage1 geometry selector;
3. Stage2 geometry selector;
4. terminal combine table or CSV;
5. AOT profile/job enumeration and cache key;
6. framework model/backend enablement;
7. focused and exhaustive selector tests.

Print resolved configs in the benchmark. A missing combine CSV entry may
silently choose a generic kernel and make a MegaMoE-versus-MORI comparison
invalid.

### 4. Validate semantic composition

When the framework computes a shared expert separately, compare:

```text
framework_output = routed_megamoe_output + shared_expert_output
```

against the framework reference at the same point. A test that feeds only the
routed experts should not claim full-layer model correctness.

For SmoothQuant, compare against the delivered customer implementation and
record the exact order of smoothing, per-token quantization, dispatch and
GEMM. For native MX A8W4, record the MX scale format/granularity used by the
active AITER implementation; do not label it generic INT8.

### 5. Tune by deployment regime

At minimum scan:

```text
tokens/rank: 1,4,8,16,32,64,128,256,512,
             1024,2048,4096,8192,16384,32768
MTPR:        tokens and each fixed deployment capacity
routes:      uniform, hot, mixed skew, unequal rank tokens, zero-token rank
```

The exact upper bound is deployment-specific. Decode selectors should not
silently activate a prefill load-balancing path; prefill support must not
regress small-token fixed-slot behavior.

## Historical model-specific cautions

### V4-Pro

This is the primary whole-model ATOM regression anchor. A common isolated shape
is `7168x3072`, EPR48, top-k6. Verify both fixed and compact modes and the
serving concurrency matrix; microbenchmark improvements can lose overlap in a
61-layer graph.

### ByteDance M13

Keep three labels distinct in reports:

```text
smoothquant a8w4
smoothquant a8w4 optimized
mx a8w4
```

Native MX A8W4 and SmoothQuant A8W4 may share dispatch/GEMM infrastructure but
do not share the same preprocessing or scale semantics. The delivered
`ghu/mega_moe_v1_copy` history covered decode through token 512; later prefill
tuning must be reported with the exact new commit and cannot be back-attributed
to the delivered baseline.

### GLM-5.2

Issue-driven support must include EPR32/top-k8 selectors, combine geometry and
AOT jobs. Verify the framework's shared-expert path and SwiGLU behavior. Do not
reuse a V4-Pro activation limit merely because hidden/intermediate tiles
compile.

### Kimi-K3

Top-k16 and EPR112 invalidate assumptions tailored to top-k6/EPR48. Runtime
pair IDs/fanout metadata should be range-checked end to end. A routing-only
microbenchmark is useful for protocol coverage but does not establish complete
model activation or quantization support.

## Model-support acceptance record

For each added model, retain:

```text
authoritative model revision and extracted fields
derived capacity/address calculations
selector boundary-test result
AOT expected/actual job count and run-only result
8-rank eager/reference/Graph/stress matrix
full token/MTPR performance table
framework integration result including shared experts
known unsupported modes
```
