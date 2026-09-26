# Indexed sparse MLA + MoE block

[中文](PERFORMANCE_zh.md)

This directory contains a FlyDSL implementation of one fixed GLM-5 MLA + MoE
shard. The production path is exposed as `Glm5IndexedMlaMoeBlock` and generated
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
| `indexed_mla_moe_kernel.py` | FlyDSL kernel scheduling, communication, MLA, routing, and expert computation. |
| `layer.py` | Public host wrapper, scratch allocation, launch arguments, tracing, and lifecycle. |
| `reference.py` | Independent Torch stage and end-to-end calculations. |
| `native_baseline.py` | Optional same-weight TileRT comparison adapter. |

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
