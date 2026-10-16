# Case studies: implementation failures and performance regressions

These self-contained GLM/K3/DSV4 extracts preserve observations and their limits. They need no original checkout. Historical sizes are not new-model defaults. Arithmetic/routing diagnosis is in [numerics](../../dev/references/numerics.md); quantitative optimization sequences are in [PMC](pmc-cases.md) and [ATT](att-cases.md).

<a id="shape-cliff"></a>

## GLM: a slower standalone stage improves the fused kernel

**Context.** Native TP4 showed a latency jump at total rows `T = conc × seq = 10`. Its fully unrolled 90-unit down-projection body incurred private memory and VGPR spills in the fused kernel.

**Change.** Replace full unrolling with nine runtime groups of ten, preserving per-wave contribution order, task ownership, rounding, LDS layout, and epochs. This grouping was specific to the existing mapping.

**Measurements.** Conc2/seq5 hot-layer HIP graph timing: median of rank medians, then median over both allocation orders. Durations are µs/layer; private-segment sizes are bytes/thread.

| Indexer | Before → after, µs | Latency reduction | Private segment |
|---|---:|---:|---:|
| off | 169.348 → 127.148 | 24.92% | 464 → 0 |
| on | 226.550 → 184.364 | 18.62% | 476 → 0 |

VGPR spill metadata fell from 115/118 to zero; SGPR spill metadata remained. Affected outputs/state and 256-replay trials passed for conc2/seq5 and conc1/seq10.

**Interpretation.** Standalone down became 2–5 µs slower while the fused layer improved: evaluate the generated code and performance at the target fusion boundary. Dynamic MFMA work was unchanged. Dynamic scratch traffic was not measured, so the gain cannot be assigned entirely to spill removal. These rank-median measurements do not establish request-completion or full-model latency.

<a id="resident"></a>

## K3: persistent fusion reduces standalone efficiency

**Construction.** General-purpose A16W4 gate/up and down routines using 256-thread workgroups were placed in a persistent loop, retaining SiTU, BF16 intermediate activations, BF16 atomic scatter and separate routing and output stages. This removed a launch without redesigning output ownership.

**Controls.** Ordinary stages N, resident-configured separate stages P, and fused F held the target math fixed. Unprofiled single-GPU HIP graph timing used 16 expert sections with common clear/step work, 100 replays and alternating order. Representative seed 1234 results in µs/section:

| Execution | S4 | S8 |
|---|---:|---:|
| N | About 32.5 | About 51.3 |
| P | 50.7 | 79.3 |
| F, tile-ready, 256 workgroups | 102.0 | 162.0 |
| F, grid-wide barrier, 256 workgroups | 86.5 | 122.6 |
| F, grid-wide barrier, 512 workgroups | 107.1 | 150.2 |

Each configuration had its own paired baseline; cross-row subtraction is not fence time. Standalone UG used VGPR 152/158 and 3 workgroups/CU; down VGPR72 and 7 workgroups/CU; pair VGPR190/232 and 2 workgroups/CU. A grid of 512 workgroups satisfied that experiment's co-residency check but increased latency. Uniform indexing, first-iteration unrolling and changes to the `waves_per_eu` compiler setting did not fix it; one candidate spilled.

**Interpretation.** P regresses before fusion, so the persistent implementation and its resource usage already impose a cost. F adds the effects of combined code generation and scheduling. This does not prove tile readiness always loses: flags around ordinary BF16 payloads differ from a tagged loader that directly consumes ready fragments. There was no matching PMC/ATT for this case.

**Next experiment.** Build output-owned down independently, gather required routes, and compare at the same materialized-output boundary. Validate changed summation/rounding, then test tagged UG handoff and tile-ready reduction. If the standalone implementation is slow, revisit tile mapping and resource usage. If only F is slow, inspect combined resources and synchronization. If the local improvement does not reduce target latency, inspect the remaining latent, shared-expert, and front-end dependencies. These are proposed mechanisms, not measured wins.

Checks of intermediate results with fixed routes and replay with changing inputs passed within the tested scope. Independent full-layer reference validation remained incomplete. Retain staged B and record the validated scope of C; additional fusion alone does not establish a performance improvement or correctness for arbitrary checkpoints.

<a id="coverage"></a>

## GLM: a tile change creates permanent payload waits

The grouped gate/up task mapping specified 16 outputs per task, but the device routine wrote eight. Half the intermediate rows were never published, so down waited forever. Capacity was sufficient but live production was incomplete.

Derive task count, tail mask and consumer addresses from the device routine's actual output width; enumerate missing/duplicate contributions. Recheck worst-case union and debug buffers. Enlarging queues or adding fences cannot create missing payload. Validate coverage before measuring grouping savings.

<a id="epoch"></a>

## DSV4: staged partitions must share an invocation generation

One builder emitted standalone router/route/quant/up/down or the full partition. Earlier stages used `epochs[cta]+1`; only the final down stage or full kernel committed it:

```text
g = epochs[cta] + 1
execute selected device stages using g
if this is the final partition:
    finish required protocol obligations
    epochs[cta] = g
```

Incrementing after every stage makes producer and consumer tags disagree. This sketch requires compatible workgroup/epoch-slot mapping, matching mailbox tags, ordered launches, all required commits including empty tasks, no overlapping use of the workspace, and completed old readers. The counter update alone does not prove completion.

Changed grids/concurrent streams need an explicit invocation generation and stable mailbox owners, isolated workspaces, or another justified mapping. Retain staged/partial/fused controls, changing inputs, poisoned workspace and repeated invocations; fixed input can hide stale payload.

<a id="replay-race"></a>

## GLM: short checks pass, uninstrumented long replay hangs

A TP8/S8 fused indexer passed three functional replays, then hung in long HIP graph replay. Q/score epochs matched, but `indices_ready` contained token IDs. The corruption source preceded the waiter.

The final radix pass stored threshold/count in LDS `keys[256:258]`; compaction reused `keys[256:264]` for wavefront counts before every wavefront read the old values. Late wavefronts read overwritten thresholds, emitted excess indices and wrote out of bounds into ready state.

Load threshold/count into registers in every participating wavefront, then place a workgroup barrier before reuse. Extra mailbox clobbers did not fix it. Instrumentation masked the failure, so passing a traced run was insufficient.

Preserve the failing binary/input/geometry, inspect neighboring bounds and last-reader lifetimes, then replay the original uninstrumented failure plus affected selection, tail and delayed-peer cases. A wait does not establish a missing fence; corrupted indices can overwrite correct flags.
