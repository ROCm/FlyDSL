# FlyDSL-only hotpath repro

This directory contains a minimal repro for the DeepSeek-V4-Pro non-DPA
`1k/1k c=256` regression.  It intentionally does not import AITER, ATOM, model
weights, or a serving stack.

The benchmark builds small FlyDSL kernels with AITER-like launcher signatures
and replays the rank-local call mix observed in the original trace:

| launcher kind | calls per prefill window |
| --- | ---: |
| `qk_norm_rope_quant_like` | 81 |
| `fused_compress_attn_like` | 80 |
| `hca_compress_forward_like` | 41 |
| `hca_norm_rope_scatter_like` | 41 |

The GPU work inside each synthetic kernel is intentionally small.  The signal
is host-side launch overhead after JIT compilation: signature binding,
cache-key construction, TensorAdaptor / PointerAdaptor handling, and CallState
reuse.  This is the part that is amplified by the high call count in the real
serving trace.

## Quick run

Run one FlyDSL tree directly:

```bash
PYTHONPATH=/path/to/FlyDSL/build-fly/python_packages:/path/to/FlyDSL/python:/path/to/FlyDSL \
python benchmarks/dsv4_hotpath_repro/bench_flydsl_hotpath.py \
  --label original \
  --case dsv4-c256 \
  --tokens 991 \
  --windows 16 \
  --output results/original.json
```

Compare original FlyDSL and the PR/fixed FlyDSL:

```bash
benchmarks/dsv4_hotpath_repro/run_matrix.sh \
  --original /path/to/flydsl-original \
  --fixed /path/to/flydsl-fixed
```

`--fixed` defaults to the current checkout, so from the fixed PR branch this is
usually enough:

```bash
benchmarks/dsv4_hotpath_repro/run_matrix.sh \
  --original /path/to/flydsl-original
```

For a shorter smoke run:

```bash
benchmarks/dsv4_hotpath_repro/run_matrix.sh \
  --original /path/to/flydsl-original \
  -- --windows 1 --gpu-event-calls 20
```

## What to look at

Each JSON result records the imported FlyDSL path, version, git head, GPU, ROCm
arch, call counts, per-kernel metrics, and mixed replay metrics.

The most important fields are:

- `per_kernel.qk.paths.jit_keyword_stream.host_wall_us_per_call`
- `per_kernel.qk.paths.compiled_positional.host_wall_us_per_call`
- `mixed_replay.jit.host_wall_us_per_call`
- `mixed_replay.compiled.host_wall_us_per_call`
- `gpu_event_us_per_call`

Expected interpretation:

- If `gpu_event_us_per_call` is similar but `jit_keyword_stream` is much slower
  than `compiled_positional`, the problem is host launch overhead, not the GPU
  kernel body.
- If the fixed FlyDSL lowers `jit_keyword_stream` and mixed replay time versus
  original FlyDSL, the PR is addressing the right hotpath.
- `qk_norm_rope_quant_like` is the primary case.  MoE can be enabled with
  `--include-moe`, but the regression is not MoE-only.

## Local smoke baseline

A 1-window smoke run in the `yhl_dev` ROCm container on gfx950 produced:

| stack | mixed jit host us/call | mixed compiled host us/call | qk jit host us/call | qk compiled host us/call |
| --- | ---: | ---: | ---: | ---: |
| original FlyDSL `0.2.0-pristine` | 104.23 | 10.09 | 68.68 | 8.63 |
| fixed `directraw-kw-patch` | 91.36 | 10.01 | 64.89 | 8.66 |

Use these only as a direction baseline.  Absolute values vary by host CPU,
Python build, ROCm stack, and GPU queue state.

## Relation to the full E2E baseline

Known local full-stack c=256 results:

| stack | total tok/s | output tok/s | TPOT | TTFT |
| --- | ---: | ---: | ---: | ---: |
| old good `68a2d29` old stack | 8496.85 | 4250.48 | 58.04 ms | 719.77 ms |
| bad original `2984891` new stack | 7810.44 | 3907.11 | 63.32 ms | 798.97 ms |
| full hotpath fix | 8450.04 | 4227.06 | 58.30 ms | 768.43 ms |
| reviewer-safe nocachesig fix | 8225.67 | 4114.83 | 59.95 ms | 734.25 ms |

This FlyDSL-only repro does not try to reproduce model math or full serving
throughput.  It isolates the necessary FlyDSL-side condition: warm repeated
`@flyc.jit` launches with many tensor/pointer arguments must not rebuild the
same expensive host-side state on every call.

The cache-signature semantics are not part of this repro requirement.  The
script compares original behavior with the fixed hotpath behavior while keeping
the benchmark independent of AITER-side changes.
