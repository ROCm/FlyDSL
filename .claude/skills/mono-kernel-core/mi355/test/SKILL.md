---
name: mi355-mono-kernel-test
description: Validate MI355X mono-kernel numerics, task coverage, cache and state updates, replay, forward progress, and performance. Use after changing a device routine, layout, fused partition, or parallelization strategy. Covers unit tests, not serving integration.
---

# Validate the modified stage and complete target subgraph

Use the project's test harness and [operator contract](../dev/references/boundary.md). Restore identical inputs, KV cache, and state into separate reference and candidate buffers. Check the modified stage first, then the complete requested subgraph and its state updates. A full-layer target requires a layer-level unit test.

| Change | Required checks for the affected path |
|---|---|
| Weight packing, tile mapping, or loads | Address mapping, valid operands and padding, partial tiles, odd dimensions, write coverage, guard regions, and index bounds |
| GEMM, activation, or quantization | Independent checks after accumulation, activation, rounding, and quantization; zeros, signs, extrema, saturation, and scale boundaries |
| Routing, expert grouping, or reduction | Fixed logits and bias; slot ordering, exact ties, and nearly equal scores; identical, disjoint, and overlapping routes; inactive columns and shared-expert contributions |
| Fusion, queues, epochs, or LDS reuse | Staged/fused equivalence with shared device routines; changing inputs and generations; poisoned workspace; task counts below, equal to, and above worker count; empty work, delayed producers or peers, and replay without instrumentation |
| KV cache or recurrent state | Written values and regions, byte-for-byte preservation of untouched regions, page and slot boundaries, independent request histories, consecutive steps, and kernel-owned commit behavior |
| TP, EP, or CP | Global output reconstruction, ownership, empty or imbalanced partitions, required reduction/return/merge semantics, and forward progress across ranks |

Use [validation and timing](references/validation.md) for fixture isolation, first-divergence diagnosis, and paired benchmarks. Consult [numerical contracts](../dev/references/numerics.md) when formats, routing behavior, or state semantics are unclear.

Distinguish equivalence to an earlier implementation, staged/fused equivalence, and correctness against an independent reference. Preserve the original error metrics, tolerances, and observations of reference nondeterminism. Diagnostic alignment or a deterministic control does not resolve a contract violation. Reject unsupported inputs before any state update, and rerun the checks invalidated by the change.

On a mismatch, capture the inputs to the first divergent stage and reproduce it independently. On a hang, inspect task coverage, generations, nearby out-of-bounds writes, and buffer lifetimes before adding fences; the [case studies](../tune/references/cases.md) show concrete failure chains. After a fix, rerun the original reproducer without instrumentation.

Measure performance only after correctness. Match work, resources, output requirements, HIP graph organization, and weight working set; confirm a screening winner with fresh paired measurements. Measure the stated completion boundary: the median across ranks is not slowest-rank latency. Mark unexecuted checks as `not_run`.

Keep runnable kernel, correctness-check, and benchmark commands with code, fixture, and parallel configuration identifiers, validated shapes, results, and remaining gaps. A concise project note is sufficient.
