# Diagnose a bottleneck with PMC and ATT

Use hardware performance counters (PMC), Advanced Thread Trace (ATT), and timing markers to test a specific explanation for kernel behavior. Follow the sequence **observation → predicted work or dependency change → matched measurement → interpretation → minimal change → uninstrumented validation → next experiment**.

## Select the measurement that distinguishes the hypotheses

| Hypothesis | Expected observation | Useful evidence |
|---|---|---|
| Repeated expert computation | Fewer issued tasks and MFMA instructions, as predicted from actual routes | Task counts, device instruction counts, and PMC |
| Repeated quantization or operand preparation | Unchanged MFMA count with fewer preparation and LDS operations | Quantizer ISA, LDS instruction counts, and consumer timing |
| Serialized producer | Similar work with earlier readiness of dependent stages | Task mapping and completion markers |
| Resource-limited occupancy | More resident workgroups/waves after the binding allocation tier changes | [Hardware budget](../../dev/references/hardware.md), compiled allocation, and runtime occupancy |
| Spill overhead | Less dynamic scratch traffic or shorter dependent waits, possibly at unchanged occupancy | Spill metadata and ISA, then dynamic traffic and matched wait behavior |
| Exposed memory latency | Higher stall cycles per execution, or additional scattered loads | ATT call sites, ISA, and applicable cache and traffic counters |
| Rank or communication imbalance | The latest partial result, return, or merge determines completion | Work distribution and completion times across ranks |

Match the binary, fixture, parallel configuration, computation, outputs, units, and collection interval. Distinguish unsupported metrics, missing measurements, and valid zero values. Collect additional data only when it can change the decision.

## PMC: verify collection scope and normalization

Read the current metric definitions, dimensions, agent scope, collection-pass behavior, and warnings. Reusing historical event selectors without checking the current provider can produce invalid measurements. Identify raw records by run, interval, pass, agent, metric, and instance; aggregate each compatible instance once. Avoid double-counting cumulative snapshots or replayed collection passes.

Derive primary and auxiliary kernel launches and wavefront counts from the actual HIP graph or trace. Normalize by completed work and retain amortized auxiliary work. The historical interval's `32*256*8 + 2*4 = 65,544` wavefronts checks launch accounting; it does not validate permissions or every metric. The [PMC case studies](pmc-cases.md) include definitions, paired measurements, and collection limits.

- **MFMA instructions:** agreement with a task-level prediction supports a reduction in issued matrix operations. Instruction count is not the same as useful FLOPs.
- **LDS instructions:** a lower count indicates less LDS instruction execution. Bank conflicts require an appropriate metric or analysis of lane addresses and DS instructions.
- **DRAM requests:** use the metric's transaction size rather than unique tensor bytes. Compute aggregate cache hit rate as total hits divided by total hits plus misses, rather than averaging slice percentages.
- **Private memory and spills:** static private-segment size and spill metadata do not measure dynamic scratch traffic or identify the responsible stage.
- **Device-wide intervals:** account for polling and auxiliary kernels. After an EP or CP change, lower counts on one rank may reflect redistributed work; reconstruct compatible global work accounting.

If the expected reduction is absent, verify the selected binary, dispatch path, active task count, masked instruction execution, and denominator. Unchanged counts with lower latency can indicate improved scheduling rather than reduced work.

## ATT: verify stage coverage before interpreting hotspots

Map traced wavefront PCs to stages, tasks, and workgroup roles. Selecting a compute unit (CU) or shader engine (SE) does not guarantee coverage of the gate/up or down projection. Check decoder column definitions, incomplete or lost records, instruction reconstruction, retained wavefronts and launches, and a consistent warmup exclusion rule. One historical decoder placed Idle, not Stall, in its final column.

The decoder's Hit or Hitcount field counts traced instruction executions; it is not a cache-hit counter. Match the captured code object's `.text` section to debug information and the complete inline call chain. PCs can move between builds, so align semantic call sites and compare `sum(stalls)/sum(hits)`. Retain total stalls, execution counts, coverage, CUs, seeds, and exclusions. Normalization does not eliminate differences in executed paths; see the [working-set study](att-cases.md#att-workset).

For a load hotspot, determine when its address becomes available and what independent work can execute before the value is consumed. For `vmcnt` waits, identify outstanding payload, weight, and scratch accesses. Keep waits that cover both VMEM and LDS in a mixed category. At a barrier, investigate the last arriving wavefront and its preceding work: summed barrier stalls are not directly recoverable kernel latency. VMEM-load stalls alone do not establish HBM misses, bandwidth saturation, or a full request queue.

The [shared-prefetch case](att-cases.md#att-prefetch) shows stalls moving between call sites, while the [layout case](att-cases.md#att-layout) shows why changed execution counts require normalization. Samples from another stage or binary cannot resolve the current bottleneck.

## Relate measurements to completion latency

Define each marker precisely: instruction issue, store completion, visible publication, first useful computation, or final completion. Do not add overlapping intervals as if they were serial. A down-projection marker recorded on entry to a polling loop does not demonstrate compute overlap. Compare the first useful down computation with completion of the required gate/up production.

Use rank-local time origins unless cross-rank clock synchronization is established. Verify the timer frequency for the current environment. Compare instrumented and uninstrumented code and resource usage; identical instrumentation can perturb two candidates differently.

State what the evidence supports: less executed work, earlier completion of similar work, redistributed waiting, a performance change with an unresolved cause, or insufficient data. Retain an alternative explanation and a discriminating follow-up where useful. [Paired benchmarks](../../test/references/validation.md#timing) establish performance; profiling supports the explanation. Report an unexplained speedup honestly, and do not promote a counter reduction without a latency improvement at the target boundary.
