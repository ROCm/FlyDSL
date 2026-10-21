---
name: mi355-mono-kernel-dev
description: Develop low-concurrency inference mono-kernels for AMD Instinct MI355X GPUs (gfx950). Define model and numerical contracts, implement standalone stages, and fuse them incrementally with persistent workgroups. Covers kernel development and unit testing.
---

# Develop a mono-kernel for AMD Instinct MI355X

Build the requested fused computation from independently testable stages. Define the outputs, cache and state updates, and required kernel launch boundary before implementation. Keep the independent reference **R**, fastest validated implementation **B**, and current candidate **C** separate.

## Development recipe

| Step | Implementation guidance | Ready to proceed when |
|---|---|---|
| 1. Define the operator contract | Resolve the [model configuration, workload, tensor interfaces, and fusion scope](references/boundary.md). Consult [numerics](references/numerics.md) for quantization, routing, or state updates, and [parallel execution](references/parallel.md) for multiple ranks | Every operation and state update maps to actual kernel arguments; unsupported inputs are explicit |
| 2. Establish a reproducible reference | Use the [development walkthrough](references/build.md) to build a test fixture, independent reference, and complete staged implementation | Runs restored from the same state expose intermediate tensors and final results; reference nondeterminism is characterized |
| 3. Implement each device routine | Replace one stage at a time; derive its [tile mapping, output ownership, and storage layout](references/design.md) within the [MI355X hardware budget](references/hardware.md) | The standalone stage and complete target subgraph pass their checks |
| 4. Validate persistent execution | Design the [persistent workgroup schedule](references/persistent.md) and compare standalone, separately launched persistent, and fused implementations | Task coverage, resource usage, generation matching, forward progress, and completion are established |
| 5. Extend fusion one dependency at a time | Fuse the smallest useful subgraph, retain staged controls, and extend to the requested launch boundary | Intermediate tensors, final outputs, state updates, replay, and paired latency measurements validate the new partition |
| 6. Optimize the remaining bottleneck | Apply the [tuning recipe](../tune/SKILL.md) and [unit tests](../test/SKILL.md) | The next experiment is actionable, or the requested scope is complete |

Load references for the current step. Start a new model with the contract and walkthrough; investigate an existing replay hang with the persistent design and corresponding case study.

## RSI loop: inspect, change, validate, and iterate

**Reconcile code and results → select a hypothesis → implement or profile → validate → retain or revert → select the next experiment.** Use existing project notes to record the observation, predicted change in work or readiness, validation method, result, and next action.

Resolve the earliest failure in the dependency chain: reference, standalone stage, persistent protocol, fused subgraph, then the complete requested operation. A correct candidate may cover more of the requested fusion scope while remaining slower than B. Continue that candidate only when a specific subsequent change could recover the cost, with a bounded experiment that can reject the hypothesis. Retain B when C fails, and track optimizations already implemented before selecting another.

Keep runnable kernel, correctness-check, and benchmark entry points. Verify actual device launch counts. Partial fusion is an intermediate milestone for a full-layer mono-kernel; validate the complete layer with a unit-test harness. Framework integration and serving validation are outside this recipe.
