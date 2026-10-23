# Correctness, hang and race debugging

MegaMoE bugs often present as a hang on one rank and a harmless-looking wait
on seven peers. Diagnose the first violated invariant, not the last printed
line.

## Failure classification

Classify the observation before changing code:

| Symptom | First questions |
|---|---|
| All ranks idle on host | rendezvous, Python exception, compiler/cache lock, framework barrier? |
| One rank exited, peers busy | what failed first on the missing rank? |
| GPUs busy indefinitely | which kernel is resident, and which ready/counter value cannot advance? |
| Finite but wrong output | route/source map, quant scale, routing weight, epoch reuse, uninitialized padding? |
| Graph-only failure | captured pointer/state mutation, epoch reset, queued reuse, hidden host operation? |
| Large-MTPR-only failure | buffer capacity, address width, epoch wrap, source encoding, selector boundary? |
| Mixed-rank-only failure | rank-dependent wire/config choice, zero-token early return, unequal collective path? |
| AOT-only failure | missing job, incomplete cache key, changed constructor/API, synthetic merge mismatch? |

Preserve the first failing logs, process list, active kernel and source commit.
A clean rerun after killing everything is not a diagnosis.

## Minimal reproducible ladder

Keep the failing semantic axis while reducing cost:

1. reproduce the exact eight-rank case with a watchdog;
2. reduce queued iterations/layers, not rank asymmetry or route skew;
3. identify prepare, Stage1, Stage2 or combine as the resident/failing phase;
4. reduce tokens while retaining the same fixed/compact mode and MTPR;
5. isolate the phase with inputs captured from a passing full operator;
6. add progress counters or invariant traps outside the performance build;
7. prove one missing write, invalid state transition or memory overlap.

If the failure disappears when synchronizing, treat that as evidence of a
race/visibility/lifetime defect. Do not ship the synchronization unless it is
part of the intended protocol and its performance is accepted.

## Distributed protocol audit

For every shared field, write down:

```text
owner rank/wave
allocation and reset point
epoch or generation
writer(s)
release/fence operation
published ready/counter value
reader(s)
wait predicate
acquire/invalidate operation
reuse condition
capacity and address type
```

Then verify these invariants:

- one semantic object has one owner or a proven atomic multi-writer protocol;
- payload and metadata become visible before readiness is published;
- readers acquire visibility after observing readiness;
- epoch/generation prevents a later launch from satisfying an earlier wait;
- a slot is not reused until all readers that require it are finished;
- all ranks choose the same wire layout, P2P dtype and ready protocol;
- local compute geometry does not change the cross-rank ABI;
- padding/sentinel entries cannot be interpreted as real routes;
- zero-token ranks still publish/participate as required;
- address arithmetic cannot wrap at supported maximum capacity.

Equality waits such as `counter == expected` deserve special scrutiny. If a
producer can advance past the expected value before a delayed consumer loads
it, equality can wait forever. Use a proven monotonic/generation protocol;
changing `==` to `>=` without analyzing reuse and wraparound can accept stale
work.

## Known high-value failure patterns

These are diagnostic patterns, not assumptions that the current source still
contains the bug.

### Rank-local selector changes a wire field

Local token count selects a different SBM, P2P dtype, packet shape or ready
layout on different ranks. Equal-token tests pass; variable local tokens hang
or corrupt. Separate rank-invariant wire configuration from rank-local compute
tuning and add an exhaustive invariant test.

### Zero-token rank returns early

The empty rank skips dispatch/ready publication while peers send to or wait on
it. Keep collective/protocol participation; only skip local arithmetic that is
provably irrelevant.

### Cross-wave LDS slot overwrite

A dynamic Stage1 queue reuses one LDS work descriptor while another wave still
consumes it. This can duplicate/skip GEMM tiles and is a correctness race even
if it first appears as variable performance. Give each live consumer a stable
slot or prove handoff with release/acquire and non-overlapping lifetime.

### Counter/epoch backpressure race

A persistent producer or consumer observes a value from the next iteration,
or overwrites a slot before the previous generation retires. Reproduce with
queued launches and many graph replays; a single synchronized eager call is
insufficient.

### Buffer-resource or address-width limit

Legacy raw buffer resources can impose a 4-GiB addressing window or truncate
byte offsets. Large MTPR/top-k/hidden sizes expose this. Prefer current tensor
buffer APIs and prove 64-bit capacity arithmetic; do not retain a private raw
pointer compatibility branch indefinitely.

### Quantization-order drift

Moving SmoothQuant into a different stage can quantize after dispatch instead
of before it, change scale granularity or increase Stage1 register pressure.
Draw the exact BF16 -> smooth scale -> INT8/MX payload -> GEMM sequence and
compare against the delivered reference. Algebraic similarity is not enough
when rounding points move.

### Stage1/Stage2 contract mismatch

An isolated Stage2 wrapper chooses a default SBM/config instead of consuming
the active Stage1 contract. This can pass a narrow shape and fail AOT or a new
model. Make cross-stage fields explicit and include them in bundle/cache
identity.

### AOT API drift

An AOT generator calls a selector with an obsolete signature or omits a new
keyword such as EP size. JIT tests pass because runtime supplies it. Test the
real AOT enumeration in a fresh cache and the synthetic merge result used by
CI; require the expected job count and zero `produced no kernel` results.

### Stale binary or import-path split

Python comes from one checkout while FlyDSL bindings or an AITER extension
come from another Torch/ROCm ABI. Print `module.__file__`, inspect loaded
libraries and rebuild against the active venv. Do not modify kernel protocol to
compensate for an ABI failure.

### One rank fails before a collective

Peers appear wedged in MegaMoE or a barrier, but the root error is an OOM,
assert or Python exception on another rank. Inspect every rank log and preserve
the earliest timestamped error.

## Progress instrumentation

Temporary debug state should be monotonic and low-risk:

- per-rank phase ID and epoch;
- per-producer sent/retired count;
- per-consumer acquired/completed tile count;
- per-destination published/observed ready count;
- last expert/source/tile identifier;
- overflow or invalid-route flag.

Prefer host-visible pinned or independently readable state. A normal D2H copy
on the same stream may sit behind the hung persistent kernel and reveal
nothing. Remove instrumentation and its buffers from the production diff after
the root cause is proven.

## CUDA Graph and re-entry audit

Check all captured state:

- device pointers are stable for the graph's lifetime;
- counters/epochs reset in device-visible order;
- host scalars are not read dynamically during replay;
- route/fanout metadata can change only where the graph contract permits;
- one operator instance is not entered concurrently on unsupported streams;
- preload/AOT creates no request-path compilation or allocation;
- back-to-back layer launches cannot reuse scratch before retirement.

If multi-stream/TBO is required, either allocate independent state per in-flight
instance or implement an explicit ownership protocol. Same-stream ordering is
not proof of multi-stream safety.

## Proving a fix

A fix report must name:

1. the violated invariant;
2. the precise producer/consumer interleaving or invalid value;
3. why the patch prevents that interleaving for all supported generations;
4. a focused regression that failed before and passes after;
5. eight-rank eager, Graph replay and queued-stress results;
6. performance impact on fixed and compact anchors;
7. whether the defect was introduced by the PR or already present on its base.

To decide new versus pre-existing, run the same reproducer and environment on
the merge base. Do not infer ancestry from where the suspicious line first
caught attention.
