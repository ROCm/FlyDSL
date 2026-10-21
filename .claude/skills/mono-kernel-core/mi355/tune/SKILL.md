---
name: mi355-mono-kernel-tune
description: Tune MI355X inference mono-kernels using controlled benchmarks, hardware performance counters (PMC), and Advanced Thread Trace (ATT). Optimize data reuse, persistent workgroup scheduling, memory access, and TP/EP/CP communication, or diagnose fusion regressions.
---

# Tune one bottleneck at a time

Start from the current implementation, test fixture, workload, parallelization strategy, and timing boundary. Keep the independent reference **R**, fastest validated implementation **B**, and candidate **C** separate. Resolve correctness and forward-progress failures with [unit tests](../test/SKILL.md) before comparing performance.

## RSI tuning recipe

1. **Inspect the current result.** Review the code, completed measurements, and rejected experiments. Identify eliminated work and the remaining latency. For a sharp jump between neighboring shapes, compare dispatch, generated loops/unrolling, registers, and spills using the [shape comparison](references/directions.md#shape-cliff).
2. **Bound and choose a hypothesis.** Use the [MI355X hardware budget](../dev/references/hardware.md) to identify the binding residency, instruction, traffic, or dependency limit, then select an [optimization mechanism](references/directions.md). Specify the predicted change, added overhead, alternative explanation, and the result that would reject it.
3. **Implement or profile.** Make the smallest relevant change, or collect the [PMC/ATT evidence](references/profiling.md) needed to distinguish the explanations. Profile only when it can resolve an open question.
4. **Validate and measure.** Check affected intermediate results, state updates, and forward progress, then run paired benchmarks without instrumentation. Update B only when C is correct and consistently faster at the stated boundary. A candidate retained to enable further fusion needs a bounded follow-up experiment.
5. **Record the decision.** Keep the measured result, supported or unresolved explanation, conditions for retrying a rejected approach, and next experiment. Reassess the bottleneck after each accepted change.

Use the [implementation cases](references/cases.md) for shape cliffs and execution failures; read the relevant [PMC](references/pmc-cases.md) or [ATT](references/att-cases.md) case for detailed measurement examples.

## Separate persistent-execution overhead from fusion overhead

Hold arithmetic, executed work, parallelization strategy, and measurement scope fixed:

- **N:** efficient standalone stage kernels.
- **P:** the proposed persistent workgroup configuration, with stages launched separately.
- **F:** the fused kernel using that configuration and its synchronization protocol.

A regression in P points to the device implementation, task assignment, or resource usage. A regression only in F points to combined resource usage, synchronization, or execution order. These differences do not isolate fence latency. Standalone timing is diagnostic: a local slowdown can accompany a faster fused kernel ([GLM case](references/cases.md#shape-cliff)). Judge the candidate at the target boundary; if a subgraph improves but the complete operation does not, inspect the remaining critical path and communication.

Treat [TP, EP, and CP](../dev/references/parallel.md) as tuning choices where the task permits. Hold the parallelization strategy fixed to measure fusion, and hold execution organization fixed to compare strategies. Compare the final result with the best known unfused implementation under equal resource and output requirements; missing controls limit attribution to the combined change.

Report latency and causal evidence separately. A repeatable speedup can have an unresolved explanation; a lower counter value alone is not a performance win. Missing or invalid observations leave the hypothesis unresolved. Results within measurement noise remain inconclusive and warrant a bounded follow-up, not a wider parameter sweep.
