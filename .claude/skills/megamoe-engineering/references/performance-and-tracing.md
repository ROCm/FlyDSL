# Performance and trace workflow

MegaMoE performance is a distributed pipeline result. A headline kernel time
without rank, launch path, route and capacity provenance is not actionable.

## Establish a paired baseline

Freeze these before changing code:

```text
host/GPU and ownership window
container image and immutable digest
AITER, FlyDSL, MORI and framework commits
actual imported Python modules and loaded shared objects
JIT/AOT mode and cache directory
model geometry, dtype path and activation semantics
tokens per rank and the complete rank-token vector
MTPR, route distribution and seed
Graph/eager timing method, warmup and iteration count
heap size, environment switches and resolved tune config
```

Run base and candidate adjacent in time. Alternate their order when contention
or thermal drift is plausible. Use at least three paired samples for a change
near noise; report median plus range. A ten-iteration inner loop is normally
enough for a broad shape scan after the candidate has passed correctness.

Do not compare an eager baseline with a CUDA Graph candidate, prequantized
input with BF16 input, or `MTPR=tokens` with a fixed deployment MTPR.

## Timing semantics

Collect both rank-local detail and the distributed critical path:

| Metric | Meaning |
|---|---|
| prepare/quant | Serial work before Stage1, including route plan and input quantization when present. |
| Stage1 | Dispatch producer plus GEMM1 consumers for the current architecture. |
| Stage2 | GEMM2, return communication and any work included by the measured wrapper. |
| combine | Terminal top-k reduction when it is not included in Stage2. |
| operator E2E | First operator launch to completion on one rank. |
| distributed E2E | Earliest relevant launch to latest completion across all ranks. |

Stage kernels, streams and ranks overlap. Therefore:

```text
Stage1_us + Stage2_us != E2E_us in general
```

Record rank mean, rank maximum and the identity of the critical rank. A rank0
number alone can hide load imbalance. Synchronize only where required by the
declared timing method; adding synchronization can destroy the overlap being
measured.

Use the framework's CUDA Graph/event timing path for decode whenever possible.
An ATT/profile trace is for attribution, not automatically a replacement for
the acceptance timing path.

## Decompose bytes before changing CU counts

For each stage, estimate bytes that must cross each interface:

```text
input activation and input scale reads
routed payload and metadata P2P writes/reads
W1/W2 and weight-scale reads
intermediate activation and scale writes/reads
returned output and route-weight traffic
ready/counter/control traffic
```

Then calculate an effective rate with the measured critical-span time:

```text
effective_bandwidth = required_bytes / critical_time
```

State whether the byte count is logical or physical. Fanout/dedup, padding,
cache reuse, coalescing and protocol retries can make the two differ. Compare
against current GPU HBM/XGMI counters or authoritative specifications; do not
hard-code a remembered peak bandwidth as the achieved denominator.

A low dispatch CU count can be optimal: enough producers can saturate XGMI
while leaving CUs for GEMM consumers. Conversely, reducing producers below the
latency/bandwidth knee starves consumers. Measure producer bandwidth and
consumer idle gaps before assigning more CUs.

## Trace ladder

Escalate only as far as needed to falsify the current hypothesis.

### 1. Graph/profile trace

Capture every rank and identify:

- exact kernel names and launch order;
- first launch, last completion and idle gaps;
- prepare/quant serialization;
- Stage1 producer/consumer overlap;
- Stage2 and terminal combine relationship;
- unexpected module loads, compilation or host gaps;
- rank that defines the distributed critical path.

Match a trace kernel to the current source and resolved config before tuning.
Generated names can encode tile shapes, but verify the mapping rather than
decoding by memory.

### 2. ATT instruction trace

Capture the dominant kernel on the critical rank, then inspect:

- wave issue timeline and CU role distribution;
- VMEM/LDS/MFMA issue gaps;
- waits on ready flags or work queues;
- producer retirement and consumer replacement;
- atomic contention and cache-line serialization;
- occupancy, VGPR/SGPR/LDS limits;
- tail tiles and expert/rank imbalance;
- whether communication and compute actually overlap.

Use a narrow kernel/rank/CU filter first. A full eight-rank ATT capture can be
too large and perturb the workload.

### 3. PMC passes

When ATT cannot distinguish causes, collect a small number of counter passes
for HBM/XGMI/L2/LDS, instruction mix, MFMA utilization, stalls and occupancy.
Keep the workload and seed identical across passes. Counters from different
runs are not cycle-aligned; use them as aggregate evidence.

### 4. ISA/resource comparison

For a source/API refactor, compare generated ISA, code-object metadata,
VGPR/SGPR/LDS use and launch geometry. Identical high-level Python does not
prove identical ISA; identical ISA does not prove identical distributed launch
behavior.

Historical local helpers that may accelerate this workflow include:

```text
/data/ghu/parse_megamoe_trace.py
/data/ghu/analyze_megamoe_kernel_trace.py
/data/ghu/run_universe_a8w4m13_8_att_flexible_20260828.sh
```

Inspect and copy them into a task-owned directory before use. They are evidence
artifacts, not repository APIs.

## Bottleneck taxonomy

Classify the dominant loss before editing:

| Class | Typical evidence | Candidate levers |
|---|---|---|
| Host/JIT/AOT | launch gap, module load, cache miss | preload/bundle coverage, cache identity, remove request-path compile |
| Prepare/quant | long serial kernel before Stage1 | fuse only semantically compatible work, vectorize, reduce scans, tune prepare CUs |
| Dispatch producer | XGMI below attainable rate or queue starvation | producer CUs, packet width, dedup/fanout, coalescing, ready granularity |
| Stage1 GEMM | low MFMA issue, poor occupancy, consumer idle | tile geometry, wave count, preshuffle, producer retirement/consumer assignment |
| Dynamic scheduling | atomic hot spot, duplicate/skipped work risk | static assignment where valid, sharded heads, monotonic ownership protocol |
| Stage2/GEMM2 | weight traffic or small-M inefficiency | block-M/N/K, persistent CU count, skew/tail strategy, aligned-pair path |
| Combine | top-k reduction dominates tiny tokens | vector width, cooperative reduction, layout, separate/fused boundary |
| Load imbalance | one rank/expert extends tail | route-aware work distribution, bounded over-subscription, tail handling |
| Protocol wait | long ready polling despite available work | publish granularity, epoch reuse, visibility/fence audit |

Do not introduce a permanent environment switch for each experiment. Use a
temporary override to establish causality, encode the winning bounded selector
rule, then delete the switch and dead branches.

## Shape tuning discipline

Tune all requested token buckets, not only token 128. Treat small decode,
medium decode and prefill as different regimes. For each regime:

1. print the selected prepare, Stage1, Stage2 and combine configs;
2. test a bounded grid around the current geometry;
3. reject any candidate that fails accuracy, Graph replay or a neighboring
   bucket;
4. encode a deterministic selector with exhaustive boundary tests;
5. rerun without command-line overrides or experimental environment flags.

Always scan both `MTPR=tokens` and the fixed deployment MTPR requested by the
serving stack. A selector that keys on runtime tokens while changing a wire
field is invalid even when every equal-token microbenchmark passes.

## Interpreting a claimed win

A credible performance conclusion includes:

```text
base and candidate full commits
same-stack paired commands
raw per-rank samples
median/range and rank max
exact resolved config
accuracy and Graph gate for the measured binary
trace evidence explaining the delta
neighboring bucket/full-scan result
whole-model result when launch/protocol/AOT behavior changed
```

If a microbenchmark improves but the serving workload regresses, preserve both
facts and trace the framework case. MegaMoE exists to overlap communication and
compute across the real graph; the smallest isolated kernel time is not the
sole objective.
