# Indexed sparse MLA + MoE block

[中文](PERFORMANCE_zh.md)

This directory contains a shared, model-configured FlyDSL implementation of the
fixed GLM-5 MLA + MoE shard and a production-TP8 Kimi-K3 MLA + latent-MoE
layer. The persistent path is exposed through `IndexedMlaMoeBlock`, with thin
`Glm5IndexedMlaMoeBlock` and `KimiK3MlaLayer` model wrappers, and is generated
by `build_indexed_mla_moe_kernel` with FlyDSL and ROCDL APIs.

The block consumes caller-supplied sparse-attention indices. In the 78-layer
GLM-5 production schedule, 57 MoE layers use the symmetric eight-head-per-rank
reuse topology implemented here. The other 18 MoE layers refresh the selection
with an asymmetric topology: ranks 1-7 use a padded ten-head whole-layer path,
while rank 0 joins the attention reduction and then launches a standalone MoE.
That refresh topology and standalone entry point are outside this block. The
independent selector/indexer/top-2048/broadcast chain is also outside this
block and is not part of the MoE computation.

Within the supported topology, the block contains the complete MoE router,
expert top-8, shared and routed experts, up/gate activation, down projection,
expert weighting, and TP peer reduction. Its MoE math is shared with the
refresh path, but the production refresh loading and rank layout still need a
separate integration.

`KimiK3MlaMoeLayer` composes the K3 MLA shard with AttnRes, routing, A16W4
routed experts, shared experts, latent transforms, and TP reductions. These
paths contain no imported TileRT kernel body or embedded assembly.

TileRT remains an optional benchmark dependency in `native_baseline.py`. That
adapter converts the same generated tensors to the released TileRT wrapper so
the two implementations can be compared directly. It is not imported by the
FlyDSL execution path.

The external integration and measurement baseline follows
[InferenceX commit 8ac98344](https://github.com/SemiAnalysisAI/InferenceX/commit/8ac98344b038a3f2da20a565fe9b974772a67ef9),
which pins the GLM-5.3 MI355X TileRT environment used for the comparisons.

## Fixed shard and arithmetic modes

Every rank uses eight attention heads, expert intermediate size 256, hidden
size 6144, 256 routed experts, one shared expert, and top-8 routing. The 2- and
4-GPU measurements therefore exercise the same fixed shards with smaller peer
groups; they are not full-model TP2 or TP4 shapes.

The public `MoeMode` values are:

| Mode | Expert activation | Expert weight | Up/gate to down handoff |
|---|---|---|---|
| `w8a8` | dynamic FP8 E4M3, per 128 | block-scaled FP8 E4M3 | FP8 |
| `w8a16` | BF16 | block-scaled FP8 E4M3 | BF16 |
| `a16w4` | BF16 | MXFP4, per-1x32 E8M0 | BF16 |
| `a8w4` | MXFP8 E4M3, per-1x32 E8M0 | MXFP4, per-1x32 E8M0 | MXFP8 |

Attention weights stay block-scaled FP8 in all modes. Supported sample counts
are 1, 2, 4, and 8; supported peer counts are 1, 2, 4, and 8. The host wrapper
validates the complete fixed-shard contract before allocating GPU buffers.

## Code layout

| File | Responsibility |
|---|---|
| `config.py` | Fixed dimensions, public arithmetic modes, and host validation. |
| `../common/hip_ipc.py` | Shared host wrappers for HIP IPC handles and allocations. |
| `../common/mx_formats.py` | Reusable Torch MXFP4/MXFP8 quantization and dequantization. |
| `packing.py` | MFMA weight packing for FP8, BF16, and MXFP4 matrices. |
| `runtime.py` | Owned symmetric HIP IPC buffers and deterministic remote-handle cleanup. |
| `kernel_layout.py` | Shared production scratch/symmetric layouts, double-epoch slots, launch constants, and CTA stage schedules. |
| `kernel_common.py` | Shared production AMD wave/DPP, hardware-math, FP8, and MXFP4 kernel primitives. |
| `torch_fusions.py` | Shared graph-capturable RMSNorm, SiTU, AttnRes, and shared-expert production fusions. |
| `router.py` | Reusable native low-token BF16-logit sigmoid/correction-bias top-k router. |
| `symmetric_allreduce.py` | Reusable graph-safe BF16 TP all-reduce over tagged symmetric peer mailboxes. |
| `indexed_mla_moe_kernel.py` | FlyDSL kernel scheduling, communication, MLA, routing, and expert computation using the shared kernel modules. |
| `layer.py` | Public host wrapper, scratch allocation, launch arguments, tracing, and lifecycle. |
| `kimi_k3.py` | Complete Kimi-K3 TP8 AttnRes + MLA + latent-MoE layer using the shared production fusions. |
| `reference.py` | Independent Torch stage and end-to-end calculations used only as a validation oracle. |
| `native_baseline.py` | Optional same-weight TileRT comparison adapter. |
| `tools/kimi_k3_full.py` | TP8 correctness, stage-profile, and HIP-graph benchmark harness. |
| `tools/atom_kimi_k3_full.py` | Original ATOM `KimiDecoderLayer` TP8 HIP-graph benchmark harness. |

The reusable code above is extracted from the actual operator execution path:
`indexed_mla_moe_kernel.py` imports the common kernel layout and AMD
primitives, while `kimi_k3.py` imports the graph-capturable Torch fusions.
`reference.py` remains independent and is not the source of the production
abstractions.

The kernel uses FlyDSL operations for wave reductions, hardware math,
mailbox polling, buffer access, and MFMA issue. Peer payloads are rounded to
BF16 and accumulated in rank order so every rank produces exactly the same
hidden state and routing decisions.

Attention and FFN reductions each use two symmetric-buffer slots selected by
epoch parity. This prevents a faster rank from overwriting epoch `k` while a
slower peer is still consuming it when many layers are captured in one graph.

`Glm5IndexedMlaMoeBlock` owns its remote HIP IPC mappings. Call `close()` after
the last rank barrier, or use it as a context manager.

## Correctness status

The fixed eight-head reuse arithmetic path, which was unchanged by the final
peer-slot fix, passed the full 2/4/8-GPU by S=1/2/4 matrix for both `w8a8` and
`w8a16`. Each of the nine configurations ran five changing inputs and checked:

- stage outputs against the independent Torch calculations;
- exact final-output agreement across ranks;
- the final down projection and BF16 peer reduction;
- finite outputs and stable HIP graph replay.

The final source then passed exact eager-versus-replay checks on TP8 graphs
with one, 75, and 128 launches per step. This validates slot rotation for a
normal per-layer instance and a 75-launch stress sequence; it does not stand in
for the missing asymmetric refresh topology.

The independent FP32 end-to-end check passed 41 inputs at 1.49-2.85% relative
L2. Four inputs used the existing near-tied-routing skip rule because a one-BF16-
ulp attention difference changed the selected expert set; those inputs still
passed the stage checks and exact rank agreement. Existing tolerances were not
relaxed. One NP2/S4 `w8a16` intermediate used the existing one-BF16-ulp bound,
while its final down/output check matched exactly.

S=8 also passed the complete stage suite for `w8a8` and `w8a16` on 1, 2, 4,
and 8 GPUs with one fresh input per configuration. The larger peer payload uses
two 64-lane send batches; exact output agreement across ranks was retained.
The NP4 `w8a16` normalized expert input differed from the independent reduction
by one BF16 ulp on one element, within the existing BF16 handoff bound.

The new `a16w4` and `a8w4` modes passed the complete stage suite at S=1 and
S=8 on one GPU, and at S=8 on two and eight GPUs. The checks covered the
packed MXFP4 weight path, per-1x32 E8M0 scales, A8W4 activation quantization,
the final BF16 peer reduction, and exact final-output agreement across ranks.
At TP8/S8, independent end-to-end relative L2 was 0.430% for `a16w4` and
2.87% for `a8w4`. Existing tolerances were retained.

A direct TP1/S1 output comparison against the same-weight TileRT wrapper gave:

| Mode | Maximum absolute error | Relative L2 |
|---|---:|---:|
| `w8a8` | 0.1171875 | 2.263% |
| `w8a16` | 0.03125 | 0.357% |

## Performance status

The paired measurements below compare the symmetric reuse-selection topology.
They use 128 layer launches per HIP graph, five eager warmup steps, two
discarded graph timings, nine measured graph replays, and the median
critical-rank time. Hardware was 8 x MI355X (gfx950), with position 3000, seed
1234, and sparse top-2048. Lower deltas are better for FlyDSL.

| Mode | S | FlyDSL | TileRT | Delta |
|---|---:|---:|---:|---:|
| `w8a8` | 1 | 35.179 us | 35.995 us | -2.27% |
| `w8a8` | 2 | 42.461 us | 42.712 us | -0.59% |
| `w8a8` | 4 | 56.217 us | 55.361 us | +1.55% |
| `w8a16` | 1 | 35.747 us | 36.305 us | -1.54% |
| `w8a16` | 2 | 43.414 us | 43.448 us | -0.08% |
| `w8a16` | 4 | 57.246 us | 55.563 us | +3.03% |

FlyDSL is faster in four of the six measured cases. Both S=4 cases remain
within 3.03% of TileRT, so the measured W8A8 and W8A16 paths are near parity
but are not universally faster. The same source also passed repeated TP8 graphs
with one and 75 launches per step, directly covering both per-layer use and a
75-launch stress sequence without a peer-slot overwrite.

The released TileRT comparison adapter accepts only `w8a8` and `w8a16`, one
or eight peers, and S=1/2/4. It has no valid same-weight whole-layer baseline
for `a16w4`, `a8w4`, or S=8, so those supported FlyDSL extensions have
correctness coverage but no TileRT performance claim here. The TP2 and TP4
results exercise fixed per-rank shards and are scaling checks rather than
full-model TP2/TP4 comparisons.

Segment traces guided two retained scheduling changes: BF16-packed peer
exchange and one sample per router CTA. For the earlier S=4 schedule, the last
instrumented CTA reached attention publication, router publication, routed
up/gate publication, and down completion at 30.23, 35.12, 50.12, and 57.52 us.
After the changes, those marks were 27.66, 30.65, 45.73, and 52.97 us. The
comparison plot is `/root/glm5-perf-results/s4-segment-milestones.png`.

## Reproduce

Use the existing FlyDSL compiler build and this worktree:

```bash
cd /root/FlyDSL-glm5-perf
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-perf:/root/tilert_pkg
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

/opt/venv/bin/python tests/kernels/test_glm5_indexed_mla_moe.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode w8a8 --npes 8 --samples 1 2 4 \
  --layers 128 --repeats 9 --seed 1234 --pos 3000
```

Repeat the correctness command for the other modes and peer counts. Repeat the
benchmark with `--moe-mode w8a16`, then use `--backend tilert` for the paired
baseline. Run GPU jobs sequentially.

For a direct released-implementation comparison, keep `/root/tilert_pkg` on
`PYTHONPATH` and replace `--backend flydsl` with `--backend tilert`. The native
wrapper supports only `w8a8`/`w8a16`, one or eight peers, and sample counts
1/2/4. S=8 and the MXFP4 modes are FlyDSL-only extensions in this harness.

Add `--trace --layers 16 --trace-dir <directory>` to a FlyDSL benchmark for
stage timestamps, then inspect a rank with:

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  <directory>/w8a8-s4/rank0/trace.pt
```

Trace instrumentation drains memory operations and changes scheduling. Use
uninstrumented graph measurements for latency comparisons.

## Kimi-K3 full MLA + latent-MoE layer

`KimiK3MlaLayer` remains the reusable full-attention component. The new
`KimiK3MlaMoeLayer` implements the complete production-TP8 decoder-layer data
path used by Kimi-K3:

- hidden size 7168, 1536 Q-LoRA rank, and 512 KV-LoRA rank;
- 12 local heads with 128 non-positional, 64 RoPE, and 128 value dimensions;
- BF16 attention weights and the sigmoid attention-output gate;
- 12-layer AttnRes source mixing before attention and before MoE;
- BF16 router projection with FP32 sigmoid/correction-bias selection, 896 experts,
  and normalized top-16;
- replicated BF16 7168-to-3584 latent projection;
- FlyDSL device-side sorting and two-stage A16W4/MXFP4 routed experts with SiTU;
- TP8-local BF16 shared experts, latent RMSNorm, and rank-local 3584-to-896 tail;
- one TP reduction in latent space and one final TP reduction before the
  residual update.

The A16W4 launcher and tuned Kimi-K3 configuration now live under
`kernels/moe/moe_2stage_a16wmix/host.py`; tests import that production module
instead of owning the host implementation.

Correctness was checked on 8 x MI355X (gfx950) for S=1/4/8 at layer 0, plus
S=4 at layer 1 and layer 12 to cover both non-write and new-block AttnRes
branches. All outputs were finite and bit-identical across the eight ranks.
In the final optimized S=4/S=8 runs, using the implementation's own
post-attention state, top-16 selection had zero mismatches, routed-MoE relative
L2 was 0.524%/0.526%, full-output relative L2 was 0.424%/0.425%, and KV-cache
relative L2 was approximately 1e-8. Layer 0/1/12 HIP-graph capture and replay
also completed.

As in the existing GLM-5 tests, the independent end-to-end comparison records
but does not fail on a near-tied synthetic route changed by a legal upstream
MLA rounding difference. The isolated MoE check feeds the implementation's
post-attention tensor to the independent MoE reference, so it distinguishes a
real router/expert regression from this synthetic boundary effect.

The MLA component retains its two scheduling optimizations: S >= 2 reuses one
staged 64-key KV tile and one 16-column score MFMA for both local head groups,
and the output gate is fused into each W_UV producer. The attention-only TP8
improvements remain 0.6%, 2.3%, and 12.0% for S=1, S=4, and S=8 respectively.

### Complete-layer performance

The complete Kimi-K3 layer was measured with TP8, position 3000, 16 layer
launches per HIP graph, two eager warmup forwards, two graph warmup replays,
seven measured replays, and the median critical-rank time. The final results
use the native router, `bm16` routed-MoE tiles, compiled latent RMSNorm output,
and the graph-safe symmetric TP reduce backend.

| Version | S=4 | S=8 |
|---|---:|---:|
| Initial correct sequential full layer | 383.85 us | 404.81 us |
| Common-module source before the deep tuning pass | 258.7604 us | 282.4305 us |
| Final optimized source | 204.4796 us | 228.6948 us |
| Improvement from the common-module source | 20.98% | 19.03% |

The slowdown in the pre-tuning K3 path was not primarily a TP communication
problem. A controlled NCCL-versus-symmetric-reduce comparison changed S=4
from 258.9820 us to 256.3328 us and S=8 from 281.1871 us to 279.5571 us, only
about 0.6-1.0%. Kernel profiling instead identified three local scheduling
problems:

- The Torch top-k route used a roughly 27.96-us `gatherTopK` kernel plus a
  roughly 4.40-us sort. `router.py` now runs one wave per sample, evaluates 14
  experts per lane, and performs normalized top-16 selection in about 15.2 us.
- The routed expert GEMM used `bm32`, although ATOM selects `bm16` for these
  low-token batches. Switching to `bm16` reduced the S=4/S=8 layer latency
  from 235.29/255.39 us to 226.59/247.25 us at that point in the tuning pass.
- Latent RMSNorm was expressed as `copy_(rmsnorm(...))`, which graph capture
  expanded into approximately 25-30 us of elementwise/reduction work.
  `compiled_rmsnorm_out` now writes directly to the graph-stable destination,
  bringing the final layer to about 204/229 us.

The router projection and correction bias now also use BF16, matching ATOM's
production gate contract; FP32 remains confined to sigmoid, comparison, and
route normalization. This dtype correction did not materially change latency,
but removes an implementation mismatch from the comparison.

The requested comparison against the original ATOM implementation uses ATOM
commit `3cea04f45` and directly instantiates its production
`atom.models.kimi_k3.KimiDecoderLayer`, including AttnRes, MLA, routing,
routed/shared MoE, latent transforms, TP reductions, dual streams, and HIP
graph replay.

| Batch | FlyDSL full layer | ATOM full layer | FlyDSL delta vs ATOM |
|---:|---:|---:|---:|
| 4 | 204.4796 us | 225.6496 us | -9.38% |
| 8 | 228.6948 us | 256.5222 us | -10.85% |

These are observed end-to-end decoder-layer timings, but the attention work is
not identical. ATOM's `KimiFullAttention` scans a dense 3001-token KV context,
whereas FlyDSL consumes caller-supplied top-2048 KV indices at position 3000.
The MoE and hidden/model shapes, TP8 topology, graph length, warmups, repeats,
and critical-rank timing rule match. Therefore the table is useful as a direct
implementation-level full-layer baseline, but its delta is not a normalized
same-attention-work kernel comparison.

### Why K3 is still much slower than the GLM kernel in absolute time

The GLM-5 W8A8 S=4 result above is 56.217 us, versus 204.4796 us for the K3
A16W4 full layer. These numbers should not be treated as the same-workload
optimization target. K3 has hidden size 7168 instead of 6144, 896 routed
experts/top-16/intermediate 384 instead of 256/top-8/intermediate 256, and 12
local attention heads instead of 8. Its router projection alone is about 4.08
times larger: `(7168 * 896) / (6144 * 256)`.

K3 also performs AttnRes mixing, a replicated 7168-to-3584 latent projection,
shared experts, latent RMSNorm, a rank-local 3584-to-896 tail, and two MoE TP
reductions. The GLM fast path places most of its work inside one persistent
monokernel; K3 still composes the persistent MLA kernel with dense GEMMs, the
native router, sorting, two-stage routed experts, shared experts, and
collectives. The retained changes therefore align K3 with GLM's optimization
principles--native wave-level routing, low-token tiles, graph-stable output
fusion, and symmetric peer communication--but cannot make the unnormalized
model workloads have the same absolute latency.

On a one-layer S=4 profile, the largest remaining K3 kernels were persistent
MLA (34.20 us), routed GEMM1 (23.32 us), the two symmetric reductions together
(18.60 us), latent projection (15.52 us), native router selection (15.24 us),
and the shared-expert GEMMs. The next material step is deeper persistent
integration: fuse the router projection with selection, emit sorter-ready
metadata directly from routing, and combine latent normalization/tail work
where the ownership and mailbox protocol can be proven safe. A trial that
combined shared/tail accumulation with the final peer reduce was not retained
because it produced an illegal-address failure under TP8.

Overlapping shared and routed branches on separate HIP streams was also
tested and rejected: at S=1 it regressed from 336.68 us to 384.66 us because
the small GEMMs contend for compute resources and add cross-stream
synchronization. The production path remains single-stream.

Reproduce the complete MLA + MoE checks and measurements with:

```bash
cd /root/FlyDSL-kimi-k3
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:.

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_full.py \
  --npes 8 --samples 4 --layer-idx 0 --check \
  --bench --layers 16 --repeats 7 \
  --output /root/kimi-k3-perf-results/full-moe/final-optimized-s4.json

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_full.py \
  --npes 8 --samples 8 --layer-idx 0 --check \
  --bench --layers 16 --repeats 7 \
  --output /root/kimi-k3-perf-results/full-moe/final-optimized-s8.json
```

Use `--eager-attn-res`, `--eager-router`, or `--eager-shared-experts` for
controlled optimization A/B runs. Use `--profile` for median eager GPU event
timings. The default `--reduce-backend symmetric` can be changed to `nccl` for
a communication A/B. Add `--kernel-profile` to either full-layer harness to
record one rank-0 graph replay broken down by GPU kernel. Uninstrumented
HIP-graph replay remains the latency source of truth.

Reproduce the ATOM baseline from the separate checkout with its original layer
implementation. The local AITER JIT build requires its composable-kernel
submodule and `pybind11==3.0.1`, matching the AITER core ABI used here:

```bash
git -C /root/ATOM-k3-baseline checkout 3cea04f45
git -C /root/aiter submodule update --init --recursive -- 3rdparty/composable_kernel
/opt/venv/bin/python -m pip install --upgrade --target /tmp/atom-k3-deps pybind11==3.0.1

cd /root/FlyDSL-kimi-k3
/opt/venv/bin/python kernels/mla_moe_layer/tools/atom_kimi_k3_full.py \
  --samples 4 --output /root/kimi-k3-perf-results/full-moe/atom-s4.json
/opt/venv/bin/python kernels/mla_moe_layer/tools/atom_kimi_k3_full.py \
  --samples 8 --output /root/kimi-k3-perf-results/full-moe/atom-s8.json
```
