# Design persistent workgroups and synchronization

## 1. Choose the execution lifetime

| Mode | Meaning | Additional obligations |
|---|---|---|
| Finite invocation | A bounded worker grid processes multiple tasks/stages, then returns | Coverage, dependencies, progress, generation, completion and reuse |
| Cross-invocation service | Workers remain active and accept later submissions | Descriptor publication, backpressure, isolation, completion acknowledgement, drain/stop |
| HIP graph replay | Kernels exit and the HIP graph launches subsequent calls | Valid pointers, ordering, and replay generations; this is not a service loop |

Start with workgroups that process a finite invocation and return, unless the task requires a longer lifetime. A worker in this recipe is a workgroup that repeatedly claims or is assigned tasks. Residency describes which workgroups can execute concurrently; persistence describes worker reuse and lifetime. A single launch does not imply persistence across decode steps or retention of all weights in registers, LDS, or cache.

## 2. Start with a static workgroup schedule

Define tasks by generation, stage, request/token, expert/context shard, and output tile. Routing-dependent live counts must be published before consumption; capacity is not live work. Derive each stage's worker set, role-local ID, stride, and maximum number of task iterations per workgroup.

```text
prepare generation g, inputs and control state in legal stream/event order
launch finite workers:
    for stage in this partition's dependency order:
        for task = role_local_id; task < live_count; task += role_workers:
            execute the shared device routine
            finish all payload writers; publish data for g
        perform the dependency handoff required by this edge
    finish this worker's protocol obligations; return
```

This is a control-flow sketch, not an API or a synchronization proof. A stage or grid-wide barrier requires established placement/co-residency and progress for every participant, including empty-task workgroups. `grid <= aggregate occupancy capacity` alone is insufficient. Use actual compiled resources, device partitions and role placement; otherwise retain the launch boundary or choose roles with a valid alternative forward-progress argument. Ensure external communication kernels have the resources and scheduling opportunity to execute if the protocol needs them.

Initialize queues/counters before any consumer can use them. One workgroup must not reset shared state already in use. Dynamic task selection must be uniform across all wavefronts participating in a device routine: claim with a leader and broadcast, or an equivalent legal scheme.

## 3. Select a scheduling policy for the observed bottleneck

| Policy | Useful condition | Cost and progress question |
|---|---|---|
| Static stage-major loops | First implementation; regular work | Tail rounds and whole-stage waits |
| Fixed producer/consumer roles | A tile-ready consumer can overlap production | Can producers finish while consumers wait? What resources remain idle? |
| Dynamic/chunked stage claims | Uneven routes or task costs | Atomic and indexing overhead, locality, and unique workgroup-uniform claims |
| Cross-stage ready queue | Starting consumer computation earlier reduces completion time | Readiness aggregation, compatible roles/resources, unique enqueue/claim, termination |

Do not let every worker block while required producers remain unclaimed. With two workers and P0/P1/P2, both workers can finish P0/P1 and wait on a consumer requiring P2, stranding it. All workgroups can be co-resident and still deadlock. Finish the producer loop or keep workers available for ready production. Include worker program order and resource waits in the dependency argument, not only the mathematical DAG.

A ready queue needs generation-local WAITING→READY→CLAIMED→DONE behavior. Establish all predecessor payload visibility before publishing READY; a counter reaching zero alone does not prove visibility. Ensure each task is claimed, executed, and completed exactly once. An empty queue may coexist with running producers or messages. Preserve slot/K/reduction order where the numerical contract requires it.

## 4. Specify data publication and buffer reuse

```text
all payload writers complete → required-scope visibility → publish ready(g)
consumer acquires matching g → reads payload → last reader permits reuse
```

For multi-wavefront writes, each writer completes its relevant stores before workgroup aggregation and publication. A wait instruction such as `s_waitcnt` executed by the publishing wavefront does not complete stores issued by other wavefronts. A workgroup barrier does not establish cross-rank visibility. Read current helper/compiler contracts for release/acquire, atomicity, cache scope and reordering; choose synchronization primitives from those contracts.

A payload with an embedded generation tag and a payload with a separate ready flag require different synchronization protocols. Specify whether the value and tag are transferred atomically, how the consumer retries an inconsistent read, and when the producer can overwrite the payload. LDS reuse likewise waits for every old reader, including wavefronts delayed before loading thresholds; see the [GLM replay race](../../tune/references/cases.md#replay-race).

Use one generation per logical invocation across staged/fused partitions. Define who allocates and commits the generation, how it is initialized and wraps around, and how concurrent invocations are isolated. The [DSV4 scheme](../../tune/references/cases.md#epoch) commits only at the last partition under compatible mappings; changed grids or concurrent streams need a new mapping or isolated workspaces. Repeated calls in one HIP graph need distinct invocation identity even if a step counter changes only once.

## 5. Establish invocation completion

First define when the task set is closed. Fixed live counts may suffice; EP/routing-generated work needs count/dispatch closure. Account for a child before retiring the parent task's final outstanding-work count, avoiding a false zero-work interval.

Completion establishes: no new work can enter, all required tasks executed/published, transfers and state effects completed, final output committed, and last readers of reusable storage finished. Use a coherent closure/count/coordinator protocol; unrelated counter snapshots are insufficient. “Queue empty,” “all claimed,” or one local done flag does not establish this.

TP must allow missing partials to be produced; EP covers dispatch→expert→return→combine; CP covers scan→candidate/partial merge. Empty ranks still owe protocol completion. Specify transport progress and delayed-peer behavior. Kernel return waits for its workgroups, not automatically every asynchronous cross-rank effect.

## 6. If a service loop is required

Wrap validated finite work in `WAIT_SUBMISSION → EXECUTE(g) → PUBLISH_COMPLETION(g) → WAIT_SUBMISSION`, initially one generation at a time. A coordinator publishes the active descriptor/generation to all participants; workgroups must not independently dequeue different requests and then enter one shared barrier.

Use bounded slots with sequence numbers, valid CPU/GPU or GPU/GPU publication/acquire, backpressure, and acknowledgement before pointer or workspace reuse. A producer queued behind the waiting kernel on the same stream cannot run; another stream still needs resource progress. Distinguish output-ready from permission to recycle storage.

Stop accepting work, drain accepted work/transfers, publish stopped, confirm exit, then release resources. An immediate per-worker stop can strand peers. Define failure recovery or declare it unsupported. Events behind a never-returning kernel cannot time its requests; use a validated independent completion signal/timebase. HIP graph replay of finite workers does not validate HIP graph capture of a service loop.

Validate tasks below/equal/above workers, tails/zero work, delayed producers/peers, changing inputs, poisoned workspace and generations. Service mode adds idle submission, full queue and stop/drain. Tests check the implementation against the forward-progress argument; successful runs alone do not prove liveness.
