# Define the operator contract and fusion scope

Start with a reproducible specification of inputs, outputs, and required state updates. Resolve each configuration field before implementing a stage that depends on it. If a field is unavailable, document the gap and continue work that does not depend on it.

## 1. Trace what the current model actually executes

Search the current project, then follow the selected symbols:

```bash
rg --files -g '*config*' -g '*args*' -g '*model*' -g '*loader*' -g '*runner*'
rg -n 'model_type|architectures|load_config|load_weights|weight_loader|quantization_config' .
rg -n 'top_k|num_experts|intermediate_size|page_table|slot_mapping|state_indices' .
```

Trace `checkpoint config → parsed defaults/overrides → layer constructor → loader/sharding/packing → forward/dispatch → actual kernel arguments`. Record the relevant source symbol and revision, resolved value, local tensor shape, strides, dtype, and selected execution path. Cover the layer variants present in the model: dense or MoE, attention or recurrence, hash or bias routing, index refresh or reuse, and first- or last-layer exceptions.

Check three common ambiguities:

- **Logical versus stored:** packed FP4 bytes, logical K, padding, and scale groups differ. Derive views from loader/packer address formulas, not `numel()` alone.
- **Global versus local:** experts, intermediate dimensions, shared/latent branches, and heads can shard differently. Divide an axis by TP only when the loader does so; EP may retain a full per-expert matrix.
- **Configured versus executed:** a CP option or multi-request wrapper does not prove the selected kernel supports it. Follow guards and builder arguments; traces establish actual launches.

Historical model names identify cases, never constants for a new model.

## 2. Separate required semantics from implementation choices

| Boundary | Questions that must have concrete answers |
|---|---|
| Computation and launch scope | What exact GPU DAG/formulas are requested? Which nodes remain external? Where is the final output materialized? Is the target a MoE subgraph or a complete layer, and how many launches are required? |
| Workload | Global requests B, per-request query counts S_i, total T; current-token convention; live context versus allocated capacity; decode/prefill distinction; legal/rejected shapes, routes, pages, masks and tails |
| Tensor ABI | Logical/packed shapes, valid extents, strides, A/W/scale formats and layouts, alignment, aliasing, workspace capacity and pointer lifetime |
| Numerical path | Per-stage MFMA/accumulator, quantizer formula/groups, activation and rounding sequence, route weighting, slot/K/wavefront/rank reduction order, original oracle/error criteria |
| Cache and state updates | Cache/state written and untouched regions; request/position/page/slot owner; recurrence predecessors; snapshots and commit behavior touched by this kernel |
| Parallel ownership | Actual groups/members, splits/replicas, global→local expert IDs, KV ownership, partial/final output owners, required transfers and merges |
| Execution lifetime | Finite invocation or explicitly required service loop; generation owner, concurrent-call/stream assumptions, initialization/completion/reuse, resource and timing scope |

Expand host calls into the GPU operations they launch. Distinguish tensor views and aliases from data copies. Include cache writes, residual additions, shared-expert branches, reductions, and layout conversions in the directed acyclic graph (DAG). For each edge, identify the tensor or partial result, producer, consumer, and readiness requirement. Account for latent normalization and apply residual and shared-expert contributions exactly once.

Preserve the mathematical definition, global inputs and state updates, numerical criteria, and user constraints. Treat the unfused parallelization strategy as a starting point. Changing it requires revisiting local shapes, ownership, metadata, and communication; use [parallel execution](parallel.md). Distinguish the required support range from currently implemented specializations.

## 3. Turn the contract into the first runnable fixture

Borrow real prepared layer weights where available; include inputs, routing metadata, positions/page tables, and nontrivial legal initial state. Synthetic tensors help isolate a device routine, but do not establish checkpoint compatibility. Select a small legal shape and a target shape exposing scaling constraints. If multiple requests are required, represent independent page tables/predecessors from the start even when first executing one request.

Snapshot and restore separate, identical initial buffers for R and C. Repeat R from that snapshot to expose reference nondeterminism. Capture the relevant numerical checkpoints: normalized inputs, logits, selected experts and probabilities, quantized values and scales, intermediate activations, local partial results, final outputs, and state updates. The reference must be independent of the device routine or epilogue under test. Staged/fused equivalence does not replace validation against an independent reference.

Numerical details belong in [the numerical contract](numerics.md); implement only those present in this model. Keep a small coverage record distinguishing configured support, compile feasibility, correctness-tested shapes, and performance-confirmed shapes. Reject unsupported inputs before writing outputs or updating state.

Proceed when every target operation and state update maps to actual arguments, reference execution from identical state is reproducible or its nondeterminism is characterized, and the first device routine has an independent output check. Unresolved reference arithmetic remains a validation gap.
