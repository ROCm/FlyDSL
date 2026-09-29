# DSV4-Pro A8W4 layer forward and MoE mono-kernel

This experimental implementation fuses the router, routed experts, native FP8
shared expert, output combine, and local TP reduction into one resident GPU
launch on gfx950. `Dsv4MonoKernel` exposes a complete per-layer forward using
native ATOM indexer/attention/cache/mHC operations and this MoE component.
The initial layer implementation issues multiple launches, as allowed by the
layer-forward contract; only the MoE component is one resident launch.

See [PROGRESS.md](PROGRESS.md) for the validated scope, current results,
reproduction commands, acceptance criteria, and next optimization targets.

## Layer interface

Following K3 PR #1204's `KimiK3MonoKernel` host-wrapper form, prepare once and
call `forward` once per layer. For DSV4, bind an already loaded ATOM `Block`
with `ATOM_DSV4_MONOKERNEL=1`; its prepared FFN weights and scratch are reused:

```python
from kernels.monokernel.dsv4 import Dsv4MonoKernel

op = Dsv4MonoKernel(block, samples=4, layer_idx=block.layer_id,
                    rank=rank, npes=tp, group=tp_cpu_group)
hc_state = op.forward(hc_state, positions)
# Alias: op.mono_kernel_forward(hc_state, positions)
```

The caller establishes ATOM's forward context with native attention metadata
and input IDs, and binds the paged cache through the native metadata builder.
`HCState` retains residual, post/combination mixing, previous sublayer output,
and residual-layout state. The pipeline is attention mHC → native indexer and
attention → FFN mHC → MoE. CSA's indexer already runs inside attention; HCA
has no indexer. One call updates each cache/state once. The next layer consumes
the delayed mHC post state. There is no extra host step/epoch increment.

ATOM's `Block.forward` selects this interface for supported decode shapes.
`block.mono_kernel_forward(hc_state, positions)` is also available explicitly;
it rejects unsupported calls. Normal `Block.forward` retains native fallback.
`unfused=True` selects five MoE launches within the same layer pipeline.
Both paths use a stable Torch BF16 compressor projection for supported decode
shapes. The binding belongs to the prepared layer, borrows existing weights,
and disables itself when its MoE bucket closes. No global GEMM CSV is needed.
`close()` releases borrowed bucket IPC resources collectively; close wrappers
only after every graph using that layer has finished.

The complete-layer comparison follows the PR's tool layout:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m kernels.monokernel.dsv4.tools.monokernel \
  --tp 4 --checkpoint /shared/data/amd_int/models/DeepSeek-V4-Pro \
  --layer-idx 3 --batch-size 1 --seq-lens 1 2 3 4 \
  --check --bench --output results/pro-layer-forward-tp4.json
```

This uses one real checkpoint layer, native decode metadata and a private
initially zero cache at position 128. It compares all four HCState tensors and logical cache values after restoring
the same initial state for each path. Both paths run captured CUDA Graphs;
`--baseline-repeats` (default 2) also records the staged path's own same-input
drift, without relaxing any mono/staged acceptance check. FP32 compressor rings and the newly
completed FP8/FP4 compressed rows use the same NRMSE limit as outputs; the
quantized rows are decoded with their actual native scales, including gfx950's
transposed FP4 scale row indices. All bytes outside
these explicitly addressed regions must match exactly, including other slots
and uncommitted rows, and packed KV padding. Raw byte differences are retained for diagnosis.
It is a layer fixture, not model generation or speculative accept/reject
acceptance. Cache value or out-of-region byte mismatches fail even when output NRMSE is
small. Upstream ATOM split-K nondeterminism can fail these checks; failures are
recorded and are not timed. The MoE-only script below isolates
the resident schedule from native attention and dense-projection drift.

`--profile-launches` exports one replay per path as a Chrome trace and records
the actual device kernel count and names in `launch_profiles`. It excludes
cache reset and TP setup, and counts device kernels rather than the single
host Graph launch. Profiles can be collected for a failed accuracy case to
diagnose the pipeline; this does not enable performance acceptance for it.

`tools/diagnose_accuracy.py` records the native ATOM MoE's router, shared input
quantization, GEMM1, activation, intermediate quantization, GEMM2, and routed
output errors separately. Its fixed-input/fixed-routing comparisons isolate
where errors first appear. These are diagnostics, not a replacement for the
normal acceptance checks:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m kernels.monokernel.dsv4.tools.diagnose_accuracy \
  --tp 4 --checkpoint /shared/data/amd_int/models/DeepSeek-V4-Pro \
  --layer 3 --seq-lens 1 2 3 4 --repeats 3 \
  --output results/pro-moe-accuracy-tp4.json
```

`--fused-epilogue-reference` additionally compares routed output against
FP32 activation quantized directly to FP8, as used by native FlyDSL `*_fp8`
GEMM1. It records both upward scales and the native fused scale rule. These
fixed-routing diagnostics supplement the normal FP32/native-scale oracle;
the acceptance limit remains 1.5%. Shared GEMM1 reports errors against both
round-to-nearest and truncated BF16 references to identify backend rounding
differences. Any BF16/shared GEMM CSV overrides are recorded in the output.

The initial public shape contract is one request with query length 1, 2, 3, or
4 (decode through MTP3). TP4 and TP8 are deployment targets; TP1 and TP2 are
also accepted for diagnostics. Multiple requests, prefill, and query lengths
outside 1..4 are rejected by the standalone entry point. The ATOM adapter
falls back to the existing path for unsupported host metadata.

## Configuration and quantization

`checkpoint.load_moe` reads configuration through ATOM's
`DeepseekV4Args.from_hf_config`. `Dsv4Config.validate_pro` rejects a Flash or
otherwise mismatched configuration. The native Pro MoE geometry is:

| Property | Value |
| --- | --- |
| Hidden size | 7168 |
| Global expert intermediate | 3072 |
| Routed experts / selected experts | 384 / 6 |
| Shared experts | 1 |
| Router scoring / route scale | sqrtsoftplus / 2.5 |
| Hash-routing layers | 0, 1, 2 |
| SwiGLU limit | 10 |

Routed expert weights use native MXFP4 with per-32 E8M0 scales. Input
activations use E4M3 with per-32 E8M0 scales rounded upward. Routed
SwiGLU stays FP32 until FP8 quantization, with the native fused exponent rule:
`max((((amax_bits + 0x400000) & -0x800000) >> 23) - 8, 0)`. The
shared expert retains its native E4M3 weights and 128-by-128 E8M0 scales, with
per-128 activation quantization. The router uses BF16 weights and materialized
BF16 logits. The shared and routed outputs are each rounded to BF16 before
their BF16 sum and TP reduction.

This is ATOM's A8W4 recipe with `AITER_BF16_FP8_MOE_BOUND=0` and
`ATOM_MOE_GU_ITLV=1`. The default `MoEActivationQuant.BF16` enum alone does not
identify the actual activation dispatch; the adapter checks
`resolve_activation_dtype`.

## Pipeline

The starting point is K3 PR #1204 head `21a3d1ee` (the former
`codex/kimi-k3-mxfp4-fused` branch, subsequently merged into main). The design
uses the GLM/K3 resident grid, CTA role offsets, tagged payloads, and double
buffered peer exchange:

1. Router CTAs split the hidden reduction across waves and publish BF16 logits.
2. A selector consumes the logits, applies bias top-k or an in-kernel token-ID
   table lookup, and publishes IDs and probabilities. Bias changes selection,
   not the probability values.
3. Input quantizers publish separate per-32 routed and per-128 shared formats.
4. Up-projection CTAs prefetch weight fragments before waiting for input tiles.
   Each produces complete quantization groups; the shared branch preserves its
   BF16 GEMM/activation rounding boundaries.
5. Down-projection CTAs combine the selected experts and shared output and
   exchange tagged BF16 pairs with local TP peers.

There are 256 CTAs of 512 threads. No grid-wide barrier or host epoch update
is used. For fused seq2–4, each phase distributes tasks from all query tokens
across the resident grid using CTA-stride loops and the original per-token
tagged scratch. Seq1 and the five-stage reference retain the serial schedule.
This implementation does not yet overlap down-projection weight prefetch with its
input wait, and does not claim the final K3/GLM performance profile.

Every query bucket owns independent scratch, epochs, and IPC buffers. Packed
weights are shared across the four buckets. A runtime object must be used on
one stream at a time. Call `close()` collectively before destroying its TP
process group. Input/output aliasing is rejected.

ATOM integration borrows its existing GU-interleaved routed weight Parameters;
the kernel maps their 16-row gate/up groups directly. This avoids retaining a
second full expert-weight copy. Compact unshuffled scales, router/shared
packing, and runtime buffers still require additional memory. Replacing or
updating model weights after preparation is not supported.

## Validation and performance script

Run from the FlyDSL root with FlyDSL 0.3.2, a compatible ATOM checkout, and
AITER on `PYTHONPATH`:

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export PYTHONPATH=/path/to/ATOM:/path/to/aiter:/path/to/flydsl032:$PWD
torchrun --standalone --nproc-per-node=4 \
  -m kernels.monokernel.dsv4.tools.compare \
  --tp 4 --batch-size 1 --seq-lens 1 2 3 4 \
  --checkpoint /shared/data/amd_int/models/DeepSeek-V4-Pro \
  --layer 3 --replays 3 --atom-baseline --atom-samples 3 \
  --bench --warmup 5 --repeats 30 --graph-iters 10 \
  --output results/pro-layer3-tp4.json
```

Use `--tp 8` with eight torchrun workers for TP8; use `--layer 0` to exercise
native hash routing. Checkpoint routing mode is inferred and conflicting
overrides are rejected. Omitting `--checkpoint` generates synthetic weights;
`--shared-format fp4` is only a synthetic fixture and does not represent the
native Pro checkpoint.

Use `--atom-module` in place of `--atom-baseline` to instantiate the real ATOM
`MoE` module, run its normal post-load hooks, and compare mono enabled/disabled.
This additionally checks the borrowed weight ownership and requires bitwise
agreement of its captured mono output with the standalone kernel on changed
inputs. It loads one MoE layer, without attention or a server.

The default ATOM profile remains the installed native configuration. To test
the separately labelled stable/RNE baseline, generate a local profile from
the installed AITER merged CSVs (paths may differ by installation):

```bash
python -m kernels.monokernel.dsv4.tools.accuracy_config \
  --bf16-source /tmp/aiter_configs/bf16_tuned_gemm.csv \
  --shared-source /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
  --tp 4 --output-dir results/stable-rne-tp4
```

Add `--atom-profile stable-rne --atom-config-dir results/stable-rne-tp4` to
the comparison command. Generate a separate TP8 profile for TP8. The profile
uses Torch BF16 for the router and CK RNE for shared GEMM1, only at M=1..4;
it records the source/output hashes and never edits upstream CSVs. Default
and stable/RNE results must be reported separately. Threshold failures are
saved in per-rank JSON, return a nonzero exit status, and disable timing for
that case across all TP ranks.

The script checks:

- Changed hidden values and token IDs at fixed addresses across CUDA Graph
  replays, including query lengths 1..4.
- Router IDs exactly and probability values with `rtol=2e-5, atol=2e-6`.
- Mono versus five separate GPU stages bitwise. These use the same mathematics
  and layouts; this isolates fusion, synchronization, and communication.
- Each implementation versus an independent PyTorch reference which explicitly
  reproduces quantization and BF16 boundaries. The default NRMSE limit is 0.015,
  configurable with `--max-nrmse` and recorded in the JSON.
- With `--atom-baseline`, the standard ATOM/AITER operation path versus the same
  independent oracle at the same limit. Direct mono/ATOM differences and ATOM
  repeat drift are also recorded. ATOM's BF16 split-K reductions can vary with
  execution order, so a single ATOM sample is not used as an exact oracle.

Timing uses CUDA Graph replay and GPU events, excludes loading and packing,
and only runs after each case passes correctness. Each rank writes its own
JSON; rank 0 also reports the slowest rank median for each timed path. Keep
all-rank logs, repeat distinct seeds, and verify suspicious timing jumps before
reporting performance. These are MoE timings, not full-layer or server results.

`tools/checkpoint.py` provides a separate CPU-only checkpoint manifest check.
It verifies shard headers, tensor extents, and index consistency; it does not
compute a checksum of every tensor payload.

## ATOM entry point

The companion ATOM branch `codex/dsv4-flydsl-monokernel` adds
`ATOM_DSV4_MONOKERNEL=1` for the per-layer interface, or
`ATOM_DSV4_MOE_MONOKERNEL=1` for only the MoE dispatch. Both default off.
The adapter prepares before child
weight shuffling and dispatches through ATOM's existing opaque MoE custom op.
Native ATOM, local TP, the A8W4 recipe above, native FP8 shared weights, and
`ATOM_V4_USE_TRITON_FUSION=0` are required. DP, EP, PP, PCP, TBO, online
requantization, and competing communication-fused backends use the existing
path. Full server accuracy and speculative acceptance require additional
validation beyond these component and layer fixtures.
