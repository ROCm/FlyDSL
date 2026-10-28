# Select a kernel optimization

Start with the [MI355X hardware budget](../../dev/references/hardware.md): allocated resources and resident workers, task rounds, exact instruction throughput, traffic by memory level, and the serial dependency path. Use the binding limit to select among repeated-work removal, earlier readiness, persistent execution, memory access, and work balancing. Reuse valid bounds; refine uncertain terms only when they can change the choice.

<a id="shape-cliff"></a>

## Investigate an abrupt latency change between shapes

Confirm the jump with matched measurements. Check dispatch changes, actual task counts including padding and tail rounds, precision, and the parallel plan. If these do not explain it, compare the abnormal shape with nearby controls using the [combined code-object inspection](../../dev/references/design.md). Examine generated loops and unrolling in both standalone and fused variants.

If longer unrolling or live ranges coincide with new private memory or spills, test a smaller unroll group or bounded loop. Preserve contribution order, ownership, rounding, and synchronization obligations. Predict the code/resource change; fewer static MFMA sites inside a loop do not imply less dynamic work. Validate affected numerics, uninstrumented latency at the target fusion boundary, and neighboring specializations. The [GLM example](cases.md#shape-cliff) shows why standalone speed and static spill metadata alone cannot decide the result.

## Reduce repeated work while preserving parallelism

<a id="reuse"></a>

### Reuse weights across tokens and expert routes

Dense and shared-expert projections can reuse a weight tile W across g query columns. Splitting T queries into `ceil(T/g)` groups increases logical weight reads; the cache hierarchy determines the resulting HBM traffic. Historical K3 S4 groups 4/2/1 required 68.25/136.5/273 MiB of logical dense/router/latent/shared-up weight reads per layer/rank. The runtime loop for one-token groups also increased resource usage. Earlier completion of individual tokens did not offset those costs.

Within-stage `token × stage_task` distribution is different. For n tasks/token and W role workers, stride through `[0,T*n)`, reconstructing `token=id//n`, `local_task=id%n`. This can distribute the final partial task iteration more evenly while preserving stage dependencies and weight reuse across token columns. Irregular domains need prefix ranges/task tables. Check maximum rounds, not just total work.

For T*k routed slots and U distinct experts, grouped UG can reuse weights across tokens sharing an expert. Use:

```text
unique_experts[u], token_mask[u], original_slot[u,token]
mid[u,token] or original (token,slot) output mapping
route_weight[token,slot]  # never index by union position accidentally
```

Assign `(expert, Ntile)` tasks and active tokens to MFMA columns. Initially retain original-slot FP32 intermediate activations and unchanged down/quantization for a clean control. Account for shared experts separately. Allocate for the worst-case `min(E,T*k)` distinct routed experts plus shared experts, provide masks wide enough for T tokens, and define values for inactive columns. For EP, derive slot and distinct-expert counts on each owner after dispatch.

Predict the reduction in issued MFMA instructions from the tasks actually skipped and the generated instructions per task. Masking inactive columns alone may leave the instruction count unchanged. When U approaches T*k, union construction/metadata can outweigh reuse. [Grouped UG](pmc-cases.md#pmc-grouped-ug) matched a 187,392-instruction prediction; its first version hung because the task width differed from the device routine's output width.

### Assign one owner to quantize each complete group

If down workgroups repeatedly quantize the same FP32 intermediate activations, one owner can prepare packed values/scales. A gate/up tile that produces eight elements cannot compute a scale for a 32-element group from those values alone. Compare:

- One producer gathers four adjacent tiles: simpler local group ownership, fewer tasks and longer live ranges.
- Keep UG8 tasks, then an owner gathers the complete published group: preserves UG parallelism but adds gather/publication work.

The second was used by historical Pack3. It added a device stage without another kernel launch. Preserve the quantizer and down-projection accumulation order. Account for the new owner's work as well as the repeated preparation and payload reads eliminated. [PMC](pmc-cases.md#pmc-quant-once) showed unchanged MFMA, less LDS execution and only a small DRAM change; logical-byte savings alone do not establish reduced HBM traffic as the source of the speedup.

### Reuse down-projection work while preserving accumulation order

Remapping gate/up tasks can preserve each row's accumulation order along K. The down projection combines contributions across routing slots, K tiles, and wavefronts, so changes in execution order require closer numerical validation. First group identical experts within an original wavefront/slot, or reuse only currently legal queue heads. Preserve each token's contribution sequence and shared/TP order unless a changed numerical contract is explicitly validated.

[Ordered down](pmc-cases.md#pmc-down) predicted and observed 19,968 fewer MFMA. More cross-slot reuse regressed: table/queue/synchronization costs and loss of locality can outweigh theoretical work removal. Reuse LDS metadata only after every reader has finished.

## Tune persistent execution and the critical path

| Observation | Minimal experiment | Tradeoff and interpretation |
|---|---|---|
| P already loses to N | Tune the standalone persistent implementation or change output ownership | [K3](cases.md#resident): additional workgroups and unrolling did not recover down-projection occupancy. Measure the standalone stage before fusion |
| Work counts stay similar but producer serializes dependents | Distribute only independent query/tile production | Added roles/tails may offset it; [cache case](pmc-cases.md#pmc-cache) follows readiness into dependent completion |
| Consumer waits for the whole intermediate | Publish/consume useful tiles earlier | Added polls, buffers/live ranges, and readiness cost; measure first useful computation, not entry into a wait |
| Atomic scatter/combine or reduction sets the tail | Test output-owned down standalone, then tile-ready communication | Changed summation and larger per-owner work need independent checks; a candidate design that requires validation |
| Uneven tasks leave a final long round | Within-stage remap or chunked claims | Atomics/indexing/locality versus reduced tail; derive actual role-local rounds |
| Long context or a rank hotspot dominates | Bounded selector, staged CP/EP/TP alternative | Use [parallel controls](../../dev/references/parallel.md); include merge/return and global completion |

Larger tiles can improve operand reuse but reduce task count and increase live ranges; more workers can shrink rounds but reduce per-worker resources or prevent progress. Prefetch/buffering trades load overlap for registers/LDS/code size. Compare actual compiled resources and occupancy tiers rather than optimizing a single resource number.

<a id="layout"></a>

## Co-design the data layout and prefetch schedule

Identify the weights and scales consumed together by the current MFMA instructions and lane mapping. Write logical-index→packed-offset mappings for initializer and loader together, including scale groups and gate/up pairing. Validate address mappings with real data, check the selected kernel arguments and generated load and address instructions, then measure target latency without instrumentation. One-time packing is distinct from per-invocation conversions.

Historical A4W4 packed scales by 16 rows×128 K with 8 gate+8 up rows. This improved locality for scale loads. The [ATT layout case](att-cases.md#att-layout) won with **more** VGPRs; copying its constants to native scaled MFMA would require a new mapping of operands to consuming lanes. For LDS padding, calculate real DS bank/lane addresses and the overlapping footprint; fewer LDS instructions do not prove fewer conflicts.

Prefetch requires addresses known early, independent work, legal waits, and affordable live ranges. Shared expert addresses may be known before routing; routed addresses usually are not. Move a complete useful load set and check ISA plus downstream wait operands. K3 moved an early load that a poll's `vmcnt(0)` immediately waited for; earlier instruction placement did not produce useful overlap. [Shared early](att-cases.md#att-prefetch) improved kernel latency even as routed-load stalls increased. Scale-only prefetch and blind barrier deletion did not help.

Do not copy cache hints, deepen buffering, or rank candidates by VGPR alone. Retry a failed direction only when layout, working set, precision, compiler, or wait organization changes its premise.

## Run a bounded parameter sweep

Select a small set of tile dimensions, wavefront counts, K batches, vector widths, and buffer counts. State how each is expected to improve reuse, task balance, or memory access. Prune mapping, numerical, capacity and progress violations before timing. Distinguish statically feasible, correctness-tested and performance-confirmed regions; static feasibility does not establish correctness, and correctness does not establish performance.

Screen correct candidates with short matched pairs, then confirm the winner with fresh pairs and affected shape/route/switch boundaries. Keep compilation/code-size/GPU-time limits proportional to the task. Combined optimizations form a new candidate. Stop pursuing an optimization when valid evidence contradicts its expected benefit; expand the search only for a changed premise or an unresolved discriminating question.
