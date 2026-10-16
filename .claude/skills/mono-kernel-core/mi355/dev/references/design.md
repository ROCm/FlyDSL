# Derive task ownership, tiles, and storage together

Use this recipe to implement a device routine, change workgroup geometry, or connect producer and consumer layouts. Derive the complete mapping and validate it with a small address and coverage check.

Use **workgroup** for a HIP thread block, **wavefront** for the hardware execution group, and **compute unit (CU)** for the execution resource. Some historical code uses CTA for a workgroup; preserve existing identifiers. **LDS** is the local data share. Use **workspace** for application-managed temporary buffers and **scratch/private memory** for compiler-managed per-thread storage. Static private-segment allocation and dynamic spill traffic are separate measurements.

## 1. Start from output ownership

Map one required contribution in both directions:

```text
(request, token, expert/route, output tile, K/context range)
↔ task ID and valid extent
↔ rank / workgroup role / wavefront / lane / MFMA fragment
↔ register/LDS element and packed byte offset / scale group
↔ consumer tile and final output owner
```

Distinguish partials from final outputs. Each contribution executes once; each final element has a unique owner or an explicit reduction protocol. Output-owned down can combine routes locally and feed communication directly; scatter-based down can expose more independent tasks but adds atomic/combine work and rounding. Neither is universally faster. Preserve or independently validate the changed summation organization.

Derive task counts from the actual output coverage of the device routine, not an unrelated tile constant. For a role with W workers and n tasks, calculate `ceil(n/W)` and which workers own the final round. An aggregate task count hides tail imbalance. Released frontend workgroups do not automatically become backend workers in a static mapping.

Enumerate a small real mapping on the host: tile boundaries/tails, odd token counts, minimum/maximum expert union, role offsets, and producer/consumer addresses. Check missing/duplicate contributions and final-output coverage. This catches structural errors before device debugging; it does not validate device numerics or progress.

## 2. Define masked operands and quantization-group ownership

Define operand values for invalid lanes before MFMA/reduction/selection, not only masked final stores. Use the operation's identity and empty/all-masked behavior. Stale padding can corrupt an amax, top-k threshold, or reduction even if the final write is masked.

A scale owner must see the complete logical quantization group, including contract-defined padding. Align production with q, or gather multiple producer tiles at a separate quantization owner. Preserve rounding when an intermediate leaves global memory: removing storage does not remove a required conversion.

Empty tasks may still owe barrier arrival, readiness, or completion. Early exits cannot bypass workgroup synchronization needed by other wavefronts. All wavefronts participating in a device routine must select the same workgroup task and follow the routine's fragment, LDS, and synchronization requirements.

Calculate maximum byte offsets for weights, scales, pages, routes, payloads and tags, including vector width and padding. Check intermediate integer width as well as final pointer width; long contexts and large expert workspaces can overflow an intermediate. Keep alignment and capacity assertions tied to actual mappings.

## 3. Choose the storage level and prove reuse

For each data/control region record producer, readers, physical layout/bytes, first write, last read, and the event permitting reuse.

| Storage | Appropriate use | Main cost or constraint |
|---|---|---|
| Registers | Same-thread/wavefront fragments and epilogues | Longer live ranges may cross a VGPR/occupancy tier or spill |
| LDS | workgroup exchange, reductions, operand staging | All participating wavefronts and banks/strides; shared footprint and barriers |
| Global or peer payload | Cross-workgroup/rank dependencies | Publication/acquire, tags, consumer access, and last-reader acknowledgement |

Overlap, prefetch, double buffering, and asynchronous sends extend live intervals. Budget simultaneous buffers, metadata, alignment and padding. Two slots or alternating parity is insufficient without proving old readers have finished. Publishing “output ready” does not necessarily release workspace still being read by a transfer.

Derive packer and reader offsets together from co-consumed fragments; check the selected path receives the packed tensors. For LDS, inspect actual lane/DS addresses before adding padding. A historical 16-byte pad is not a universal bank-conflict fix. The [layout case](../../tune/references/att-cases.md#att-layout) shows how changed loads and resources must be measured together.

## 4. Inspect the combined code object

Inspect workgroup geometry, vector and scalar register (VGPR/SGPR) usage, LDS allocation, private-segment size, spill metadata, code size, generated load/wait/MFMA sequences, and runtime occupancy for both standalone and fused code objects. The fused resource footprint is not necessarily the sum or maximum of the standalone footprints. Shorter source-level lifetimes may not reduce register allocation, and LDS declared in one branch can affect the entire kernel.

Convert these measurements into the [MI355X residency and throughput budget](hardware.md). Identify the binding resource and task rounds before choosing a tile, register target, or buffer count; a lower non-limiting resource does not by itself increase residency.

Resolve coverage errors in the mapping before changing the scheduler. If persistent execution regresses, use N/P/F to isolate the cost before extending fusion. If longer live ranges offset the benefit of overlap, compare a smaller fused subgraph, fewer buffers, or a different tile. Follow [persistent design](persistent.md) for synchronization and forward progress, and [optimization mechanisms](../../tune/references/directions.md) for tuning choices.
