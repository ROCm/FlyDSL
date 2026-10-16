# Build standalone stages, then fuse a persistent subgraph

Follow this recipe when bringing up a new model. The mixture-of-experts (MoE) example illustrates how dependencies determine the fusion sequence. Derive the actual graph from the model contract; the interface sketches are pseudocode.

## 1. Establish a runnable staged chain

Use the existing project structure for the resolved model specification, test fixture with snapshot and restore, independent reference, packing and layout definitions, device routines, standalone and fused launch wrappers, and validation commands. Create modules only where they serve the implementation.

A representative MoE subgraph is shown below. UG denotes the gate/up projection stage; down denotes the down projection. Preserve the model-specific activation and gating formulas.

```text
x → norm ─┬→ router logits → selection → ids/probabilities ───────┐
          ├→ input quantization → routed UG → activation/quant ──┼→ down/combine → y
          └→ shared branch ─────────────────────────────────────┘
```

Locate route weighting, shared/residual addition, latent branches, and cross-rank operations from the real code. With EP, insert dispatch before owner-local expert math and return before original-token combine. Attention/indexer/recurrence may be outside this subgraph or a better initial fusion target.

Start with existing operators connected by tensors in global memory. For each replacement:

1. Capture the stage's actual input/weights/scales/metadata from R.
2. Implement a standalone kernel for the stage and validate its output independently.
3. Insert it into the staged implementation and validate the complete target subgraph, including state updates.
4. Retain that runnable version, then select the next stage.

For MoE, first validate the gate/up and down projections with fixed inputs and routes, then enable routing. Diagnose gate/up errors at that stage while preserving the original tolerances. Existing operators can remain separate launches. To fuse a stage, express it as a device routine or compile-time emitter; a host-side PyTorch or library call cannot execute inside a GPU kernel.

## 2. Share device routines between staged and fused kernels

Compile each stage implementation into both standalone and fused kernels, while keeping the correctness reference independent. For example:

```python
# Interface sketch; emitters expand into a compiled kernel.
emit_stage(stage, task, tensor_views, workspace, generation)
run_partitions([['A'], ['B'], ['C']])
run_partitions([['A', 'B'], ['C']])
run_partitions([['A', 'B', 'C']])
```

The partition list must execute each logical operation once, preserve DAG topology, and expose cross-partition tensors. Branches cannot always be represented by arbitrary contiguous cuts. Ordered launches satisfy only their actual dependencies. Keep compile/capture identity bound to the chosen partition list; a Python wrapper or HIP graph invocation is not a device launch count.

## 3. Select a fusion boundary by dependency and expected savings

| Edge | First implementation | Before joining it |
|---|---|---|
| Element/tile-local epilogue | Apply consumer math to producer registers | Preserve intermediate rounding, valid lanes, and vector width |
| Workgroup-local exchange/reduction | LDS plus workgroup synchronization | Compatible participating wavefronts and last-reader lifetime |
| Cross-workgroup tile | Global payload and explicit ready protocol | Coverage, visibility, generation, progress, and safe reuse |
| Full-vector statistics/selection | Separate producer/statistics stage | Complete reduction and unique output/task-count publication |
| Cross-rank partial/return/merge | Existing external communication | Ownership, transport progress, numerical order, final completion |
| Cross-token recurrence | Explicit predecessor state | Independent request chains and ordered state commit |

Estimate the launch overhead, intermediate-memory traffic, or repeated computation that fusion can remove. Balance those savings against longer live ranges, LDS usage, queue and polling overhead, layout conversions, and execution tails. Select the smallest subgraph with a testable benefit. Cross-workgroup dependencies may still require data in global memory after fusion.

## 4. First fused subgraph: gate/up → quantization → down

Work backward from one down output tile: which route/K partials does its owner need, in what order, and who publishes them? Derive the UG tile/task domain and full quantization-group ownership before writing the executor. A tile narrower than scale group q cannot quantize independently; see [reuse alternatives](../../tune/references/directions.md#reuse).

Implement [finite workers](persistent.md) with a simple stage-major schedule. Keep router, input quantization, and communication external initially. Hold arithmetic, executed work, parallel configuration, and output requirements fixed when comparing:

- **N:** efficient standalone stage kernels.
- **P:** the proposed persistent workgroup configuration, task loops, and storage, with stages still launched separately.
- **F:** a fused kernel with the proposed producer-consumer synchronization.

P consumes real preceding outputs or valid initialized protocol state; it must not wait for absent producers. Inspect compiled resources and actual occupancy, then validate intermediate results and changing generations before measuring latency without instrumentation.

| Result | Next action |
|---|---|
| P regresses before fusion | Tune tile dimensions, K blocking, wavefront count, task iterations, layout, or shared resource usage; [K3 case](../../tune/references/cases.md#resident) |
| P is competitive; F regresses | Isolate joint compilation/resources, waiting for all intermediate activations, payload polling, or consumer granularity |
| F reduces local latency but not latency of the complete subgraph | Trace remaining dependencies, communication and final materialization |
| F hangs or differs numerically | Check task coverage, generation matching, buffer lifetime, or the first differing intermediate result; resolve the failure before extending fusion |

A first correct fused kernel can synchronize between stages when the grid has a valid forward-progress guarantee. Introduce tile-level readiness when starting useful consumer computation earlier can reduce completion time. Add a general task queue only if the workload benefits from it.

## 5. Extend the partition and scale the workload

One possible ladder is `[router][select][quant][UG][down][comm] → [router][select][quant][UG+down][comm] → [router][select][quant][UG+down+comm] → [whole MoE]`. Include activation/quantization in their real positions and retain any other branches. EP/CP require their own communication chains. Extend to attention/front-end/state nodes only if the requested scope includes them.

For each fused dependency, retain a standalone or staged control, use a consistent invocation generation, validate intermediate results and state updates, and compare the complete old and new partitions. The [DSV4 commit case](../../tune/references/cases.md#epoch) shows why advancing epochs after every stage breaks shared mailboxes.

Scale one query to multiple queries by examining reuse and task tails, not by blindly looping the entire chain per token. Within-stage `token × task` distribution, dense token-column reuse, and expert grouping are distinct transformations. Multiple tokens of one request do not validate independent-request ownership.

A correct but slower candidate can be retained to enable further fusion when a specific boundary or ownership change is expected to recover the cost. Keep B available and bound the follow-up experiment. Stop the approach if the evidence rejects its premise. Complete the requested launch scope and unit checks, then use the [tuning loop](../../tune/SKILL.md) to select the remaining bottleneck.
