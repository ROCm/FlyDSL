# Validate correctness and measure kernel latency

## Establish independent intermediate checks

Run the reference and candidate from separate copies of the same state snapshot. Repeat the reference first to characterize nondeterminism. Use real weights and nonzero state; synthetic fixtures help isolate a mechanism but do not establish checkpoint compatibility. Record the shape, execution path, precision, parallel configuration, and source and binary identifiers with each result.

Capture intermediate results needed to locate the first mismatch: normalized inputs, logits, selected experts and probabilities, quantized values and scales, gate/up outputs and activations, down-projection partials, reconstructed outputs, and state updates. The reference must not call the implementation under test. Preserve the original error definition, reference scale or denominator, and tolerance. Use bitwise comparisons where required by the contract; they do not compensate for an incompatible reference.

For stateful operations, compare updated logical values and verify untouched regions byte for byte. Include padding, guards, invalid slots, page boundaries, and uncommitted rows. Test independent requests and consecutive steps when affected. After changing the parallelization strategy, reconstruct global outputs and required state updates; local arithmetic agreement does not validate EP return mapping or CP merge semantics.

## Test the affected boundaries

Select cases within the supported range: just below, at, and above a tile or worker-count boundary; partial final groups; minimum and maximum expert overlap; short live contexts with large cache capacity; dispatch thresholds; and occupancy transitions. Empty work can still require readiness publication. Boundary tests do not establish correctness for every interior configuration or require unrelated full-model tests.

Changes to persistent execution need a forward-progress argument and tests with task counts below, equal to, and above the worker count. Include workload imbalance, delayed producers or peers, changing inputs and generations, poisoned workspace, and legal buffer reuse. Test supported generation wraparound with a controlled counter setup. Test concurrent calls only when the interface supports them. A cross-invocation service loop also needs idle-then-submit, backpressure, and stop-and-drain tests using its independent completion channel.

## Isolate the first divergent stage

1. Verify actual inputs, weights, scales, layouts, initial state, and selected execution paths.
2. Locate the first differing intermediate result and capture its inputs for an independent standalone test.
3. Distinguish expert permutations, exact ties, nearly equal scores, and upstream numerical differences. Align experts for diagnosis while retaining the original slot results.
4. For a hang, inspect task coverage, published task counts, generation mapping, buffer bounds, and outstanding readers. Preserve the original reproducer if instrumentation changes the failure.
5. Fix the affected stage, rerun the original failing case and related boundary cases, then validate the complete requested subgraph.

Preserve the original tolerances, routes, and reference while diagnosing a candidate. Report reference nondeterminism and results from a deterministic diagnostic control separately.

<a id="timing"></a>

## Run paired benchmarks without instrumentation

Use a correct control with identical global inputs and work, numerical precision, GPU resources and topology, required outputs and state updates, and initial state. Include per-invocation preparation, communication, combine, and output materialization. Report one-time compilation, weight packing, and setup separately from steady-state latency. Include redistribution when it is performed on every invocation.

Match warmup, HIP graph construction and replay, buffers, and the weight working set. Repeatedly accessing one weight allocation can hide memory latency exposed by rotating through independent allocations. Distinguish one or eight weight allocations from one or eight requests. For stateful repetitions, restore or advance valid state consistently in both implementations; repeated invalid cache or state writes do not represent the target workload.

Run competing GPU jobs separately. Use repeated paired trials with alternating or reversed order, and retain individual trial results and variability. A comparison of identical binaries can estimate measurement noise. Screen candidates, then confirm the winner with fresh pairs and affected workload buckets. Treat results within the noise range as inconclusive. Repeat broader testing only when a change, failure, or unresolved question warrants it.

For multiple ranks, define the completion protocol and aggregation. Comparing rank-local event durations or their maximum requires comparable start and completion boundaries; do not subtract unsynchronized absolute timestamps across ranks. A median across ranks can hide an overloaded rank and is not request or slowest-rank latency. Include transport completion required by the output. A service loop needs a validated submission-to-result timing protocol; events queued behind a kernel that never returns cannot measure individual requests.

Report `speedup=t_control/t_candidate` separately from `latency_reduction=1-t_candidate/t_control`. Distinguish latency of the fused subgraph from latency of the complete target operation. Use profiling to explain behavior and uninstrumented pairs to confirm performance. Record numerics, state correctness, forward progress, and performance separately, including `not_run`, and retain the fastest validated configuration for each supported workload and parallel configuration.
