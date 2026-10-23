# Validation strategy

MegaMoE needs layered validation because a single synchronized call misses
cross-rank drift, CUDA Graph reuse, AOT cache gaps and serving integration.
Use the cheapest gate that can falsify the candidate, then advance.

## Gate 0: provenance and diff scope

Before execution:

```bash
git status --short --branch
git rev-parse HEAD
git diff --check <base>...HEAD
git diff --name-status <base>...HEAD
```

Print imported module paths and runtime versions. Confirm the candidate changes
only expected files. For a refactor, save the base command and output before
editing.

## Gate 1: static and CPU invariants

Run project style, compile and focused unit checks. For AITER this normally
includes:

```bash
python -m compileall -q aiter/ops/flydsl/kernels/mega_moe
python -m pytest -q op_tests/flydsl_tests/test_mega_moe_config.py
```

Add an exhaustive selector test when changing configuration. Sweep every model,
EPR, supported MTPR and token bucket, plus boundary values between buckets.
Validate at least:

- Stage1 MFMA geometry (`tile_n`, waves and accumulator fragments);
- Stage2 block-M divides Stage1 sort-block-M;
- dispatch CU/group divisibility and non-empty consumer capacity;
- fixed-slot expert limits;
- source/top-k/fanout bit encodings;
- cross-rank wire tuple is invariant over different local token buckets;
- buffer capacity and 32-/64-bit address ranges;
- every code-generating value is present in cache/bundle identity;
- zero-token rank resolves to a participating variant.

Compile/lower representative Stage1, Stage2, prepare, quant and combine variants
without launching P2P kernels where a CPU dummy path is supported.

## Gate 2: focused eight-rank correctness

The production AITER runner is
`op_tests/multigpu_tests/test_mega_moe_v2.py`. Inspect `--help` at the tested
commit; do not rely on a copied flag list.

A typical launch is:

```bash
MORI_SHMEM_HEAP_SIZE=40G \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/test_mega_moe_v2.py \
  --network v4_pro \
  --bs-list 1,4,8,16,32,64,128,256,512 \
  --max-tok-per-rank 512 \
  --iters 10
```

Use the current exact parser. A passing gate should include:

- all eight ranks complete;
- distributed FP32/reference error within the declared threshold;
- eager result is finite;
- CUDA Graph capture and repeated replay complete;
- replay output is exactly stable when inputs/state are unchanged;
- no rank reports overflow, invalid address or a different protocol/config;
- task-owned processes exit.

Do not use `--skip-acc` for a functional gate. It is allowed only after the same
candidate/shape has passed accuracy.

## Required shape axes

For a change to shared production code, cover the intersection of supported
values rather than one headline point.

### Token buckets

```text
1, 4, 8, 16, 32, 64, 128, 256, 512,
1024, 2048, 4096, 8192, 16384, 32768
```

Add boundary tokens around selector transitions, such as one below/above the
midpoint used by nearest-bucket logic.

### Capacity modes

- `MTPR=tokens`: per-shape instance, exposing fixed/bounded/large transitions.
- fixed deployment MTPR (commonly 256, 8192 or 32768): actual tokens vary while
  buffer/wire protocol remains fixed.

### Routing and rank distribution

- uniform/random;
- hot expert/hot destination;
- rank-balanced hotspot;
- mixed skew;
- different local token counts on all eight ranks;
- one or more zero-token ranks;
- invalid/padding sentinel IDs where supported;
- fanout pair present, absent and changing between replays.

Route distribution and rank-token distribution are different axes. A
`rank-mixed-skew` route generator does not prove support for unequal tensor
lengths.

### Model geometry

At minimum retain anchors for V4-Pro and every newly supported model. Shared
Stage1/Stage2 edits should also exercise top-k 8/EPR32 and, when supported,
top-k 16/EPR above 64.

## Variable local-token reference

An accuracy reference based on `all_gather` of equal-shape tensors does not
support unequal rank token counts. Pad to the maximum, gather token counts and
slice by per-rank prefix offsets. A burst test that only checks completion is a
hang gate, not an accuracy proof.

Every rank must agree whether to enter reference collectives. Reduce a local
"can run reference" flag before branching.

## Stage-specific checks

Run isolated Stage1/Stage2 only after a full operator pass establishes their
inputs and metadata are valid.

- Stage1 reference: validate compact/fixed row layout, expert IDs, source maps,
  quantized output/scales and padding.
- Stage2 reference: pass the active Stage1 config/wire contract explicitly;
  validate routing weights, destination slot and P2P dequantization.
- Combine: validate all top-k slots and zero/padding behavior.

An isolated Stage2 call that omits the active config can benchmark a different
SBM/P2P protocol than the preceding Stage1.

## CUDA Graph and queued-chain stress

Capture/replay is mandatory for decode. In addition to one replay, test:

- 20 or more identical replays for ordinary gates;
- alternating supported token buckets if the framework creates separate graph
  buckets;
- a chain matching the production MoE layer count (historically 61 for
  V4-Pro);
- multiple consecutive forward steps without host synchronization;
- asymmetric rank tokens and zero-token ranks;
- runtime fanout metadata changes across epochs;
- same-stream ordered reuse of the same operator instance.

If concurrent multi-stream/TBO use is intended, it needs its own test with
independent state. A same-stream pass does not authorize multi-stream re-entry.

Use a watchdog with a finite timeout. Preserve per-rank logs and the active
kernel when it fires; do not just kill and rerun with added synchronizations.

## ByteDance customer-path gates

Known runner families include:

```text
/home/ghu/opus_verify_47/run_universe_a8w4m13_8.sh
/home/ghu/opus_verify_46/run8.sh
```

For native M13 MX A8W4, use `network=m13`, `quant=a8w4` and explicitly record
`tokens`, `MTPR` and whether input is BF16 or prequantized. For the delivered
SmoothQuant path, use `tests/kernels/test_mega_moe_int8.py` with the exact
`mode` and dispatch-quant choice from the branch runner.

The historical environment used `/opt/venv/bin/python`, eight one-rank
processes, a 64-GiB MORI heap and cache disabled for source experiments. Verify
the current runner because host46 and host47 have used different FlyDSL builds
and Torch/ROCm versions.

Customer-path acceptance requires comparison with the delivered baseline's
accuracy, not merely a broad `rtol`. If a purportedly algebraic refactor should
be bitwise equivalent, require zero eager difference and exact Graph replay.

## AOT gates

AOT changes require all of:

1. enumerate the deployment profiles and expected jobs;
2. compile into a new empty cache using the production AOT entry point;
3. require zero failed/no-kernel jobs;
4. save an artifact manifest;
5. start a fresh process with `FLYDSL_RUNTIME_RUN_ONLY=1`;
6. run eight-rank accuracy/Graph across every bundled token bucket;
7. prove no request-path `hipModuleLoadData` in a serving trace;
8. test a deliberate missing artifact and confirm it fails.

Do not test only the PR head if CI will merge it with a newer base. A prior AOT
failure escaped because head CI did not exercise the synthetic merge state and
a later API required `ep_size`. Reproduce both head and merge-result semantics
when interfaces changed on main.

## CI reachability

Reading a new case arm is not proof CI executes it. Trace the driver from file
discovery through skip lists, mode guards and shell `continue` statements.
Run the driver locally with an explicit target and verify the command appears in
the log. A script that prints `Skipping test` and exits zero provides no
coverage.

When adding a GPU case, verify the workflow actually has the requested GPU
count/label and sufficient heap/time. Keep a cheap CPU selector/invariant gate
so obvious geometry failures do not depend on scarce eight-GPU CI.

## Performance validation after correctness

Use the same graph/event timing path as the baseline. Ten timed iterations are
appropriate for a full shape scan in the known customer runner; increase paired
repetitions for changes near noise. Report rank mean/max, not rank0 alone.

After per-shape tuning, run the complete matrix again from clean processes and
without ad-hoc overrides. Regressions at untuned buckets must be addressed or
explicitly excluded by a production selector rule.

## Whole-model gates

Kernel tests do not prove framework behavior. For changes to protocol, MTPR,
AOT, launch order or performance-sensitive defaults, run the production
framework with pinned versions.

ATOM's established V4-Pro matrix uses:

| Total concurrency | Random range ratio |
|---:|---:|
| 512 | 1.0 |
| 512 | 0.8 |
| 4096 | 1.0 |
| 4096 | 0.8 |

Record completed requests, errors, throughput, input/output length range, EPLB
events, server health and whether TBO/speculation are enabled. Compare against
the standard baseline, not only a favorable historical patch.

For SGLang, keep fixed-length and AgentX/variable-length as separate workloads.
A fixed 8192-token input can hide full-MTPR padding that dominates a broad
variable-ISL corpus.

## Final review gate

After tests pass:

```text
review full diff and generated config identities
remove temporary env gates and debug counters
run cleanup/style/static checks
repeat key accuracy and performance anchors from clean processes
confirm no task-owned GPU/server process remains
archive raw logs and write a provenance-rich Markdown result
```
