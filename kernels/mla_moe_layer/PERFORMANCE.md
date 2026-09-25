# Shared/reuse MLA + MoE kernel

[中文](PERFORMANCE_zh.md)

This directory contains a FlyDSL implementation of one fixed GLM-5 MLA + MoE
shard. The production path is exposed as `SharedReuseMlaMoeLayer` and generated
by `build_shared_reuse_kernel`. It contains no imported TileRT kernel body or
embedded assembly. The former assembly capture, rewriting, and launch bridge
has been deleted.

TileRT remains an optional benchmark dependency in `native_baseline.py`. That
adapter converts the same generated tensors to the released TileRT wrapper so
the two implementations can be compared directly. It is not imported by the
FlyDSL execution path.

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
| `../common/mx_formats.py` | Reusable Torch MXFP4/MXFP8 quantization and dequantization. |
| `packing.py` | MFMA weight packing for FP8, BF16, and MXFP4 matrices. |
| `runtime.py` | Owned symmetric HIP IPC buffers and deterministic remote-handle cleanup. |
| `shared_reuse_moe_kernel.py` | FlyDSL kernel scheduling, communication, MLA, routing, and expert computation. |
| `layer.py` | Public host wrapper, scratch allocation, launch arguments, tracing, and lifecycle. |
| `reference.py` | Independent Torch stage and end-to-end calculations. |
| `native_baseline.py` | Optional same-weight TileRT comparison adapter. |

The kernel uses FlyDSL operations for wave reductions, hardware math,
mailbox polling, buffer access, and MFMA issue. Peer payloads are rounded to
BF16 and accumulated in rank order so every rank produces exactly the same
hidden state and routing decisions.

`SharedReuseMlaMoeLayer` owns its remote HIP IPC mappings. Call `close()` after
the last rank barrier, or use it as a context manager.

## Correctness status

Both `w8a8` and `w8a16` passed the full 2/4/8-GPU by S=1/2/4 matrix. Each of
the nine configurations ran five changing inputs and checked:

- stage outputs against the independent Torch calculations;
- exact final-output agreement across ranks;
- the final down projection and BF16 peer reduction;
- finite outputs and stable HIP graph replay.

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

The established W8A8 graph measurements below use 128 layer launches per HIP
graph, two warmups, nine measured replays, and the median critical-rank time.
Hardware was 8 x MI355X (gfx950), with position 3000 and sparse top-2048.

| GPUs | Backend | S=1 | S=2 | S=4 |
|---:|---|---:|---:|---:|
| 2 | FlyDSL | 33.98 us | 39.80 us | 53.55 us |
| 4 | FlyDSL | 34.30 us | 41.51 us | 54.42 us |
| 8 | FlyDSL | 35.55 us | 42.72 us | 56.32 us |
| 8 | TileRT | 35.85 us | 42.90 us | 55.93 us |

FlyDSL is faster in the measured eight-GPU S=1 and S=2 cases and 0.7% slower
at S=4. It has therefore not beaten TileRT in every configuration.

A short TP1/S1 smoke measurement using eight layers and one measured replay
gave 33.840 us versus 33.520 us for `w8a8`, and 34.735 us versus 32.895 us for
`w8a16`. These short runs verify the benchmark path and are not publication-
quality latency results.

A separate 16-layer, three-replay TP1 run measured W8A8 at 52.03 us for S=4
and 84.20 us for S=8. TileRT has no S=8 whole-layer baseline.

After adding the MXFP4 paths, a same-process 16-layer, three-replay TP1 run
measured the following medians. These are short development measurements, not
publication-quality results:

| Mode | S=1 | S=8 |
|---|---:|---:|
| `w8a8` | 33.38 us | 83.46 us |
| `w8a16` | 34.63 us | 90.13 us |
| `a16w4` | 33.22 us | 78.80 us |
| `a8w4` | 35.21 us | 87.00 us |

The `a16w4` path was fastest in this short TP1 comparison. `a8w4` still pays
for per-1x32 activation quantization and BF16 MFMA staging, so it did not beat
`w8a8` in both sample counts. The released TileRT comparison adapter accepts
only `w8a8` and `w8a16`; there is no valid same-weight TileRT baseline for the
two MXFP4 modes in this harness.

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

/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode a16w4 --npes 1 --samples 1 8 \
  --layers 16 --repeats 3
```

Repeat the correctness command for the other modes and peer counts. Run GPU
jobs sequentially.

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
