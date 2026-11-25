---
name: prefetch-data-load
description: >
  Apply prefetch optimization to FlyDSL kernel loops: pre-load the first
  iteration's data before the loop, issue async loads for the next iteration
  inside the loop body, and swap buffers at the loop tail via runtime
  loop-carried values. This overlaps data load latency with compute
  instructions. Use when a kernel has a loop where buffer_load feeds into
  MFMA/compute and load latency is exposed.
  Usage: /prefetch-data-load
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Prefetch Data Load Optimization

Apply software prefetch (double-buffering) to overlap async data loads with
compute in FlyDSL GPU kernel loops.

**API note.** The examples are *schematic*, transcribed from the PA decode
kernel. Loads are spelled as the legacy `buffer_ops.buffer_load`
(`kernels/common/buffer_ops.py`); new code uses `fx.rocdl.make_buffer_tensor`
+ `fx.copy`, and MMA is shown as `fx.gemm(mma, d, a, b, c)`. See the
**kernel-code-cleanup** skill for the mapping.

## Core Principle

Global loads (`buffer_load_dwordx4`) are asynchronous: the instruction returns
at once and only the first *use* of the data waits. Issuing the next
iteration's loads before the current iteration's compute hides their latency:

```
for i in range(N):                      # without prefetch
    data = load(i); compute(data)       # |load|stall|compute|load|stall|compute|

next_data = load(0)                     # with prefetch
for i in range(N):
    data = next_data
    if i + 1 < N:
        next_data = load(i + 1)         # in flight during compute(data)
    compute(data)                       # |load0|compute0+load1|compute1+load2|...
```

Total time drops from `N * (load + compute)` to roughly
`load + N * max(load, compute)`.

## FlyDSL Implementation: `range(..., init=...)` with Loop-Carried Prefetch

Write the loop as a runtime `range(..., init=...)` so the prefetched values are
real loop-carried block arguments (phi nodes). In an unrolled `range_constexpr`
loop the `data = next_data` swap is only Python name aliasing: it creates no
loop-carried value, and nothing keeps a load away from its use. The
`flydsl-kernel-authoring` skill, "Runtime Loops with Loop-Carried Values", has
the loop semantics and pitfalls (bounds, DSL values in `init`, LDS views).

### Transformation Steps

Given a loop like:

```python
for i in range(START, END):
    offsets = compute_offsets(i)
    data_A = buffer_ops.buffer_load(rsrc_A, offsets, vec_width=4)
    data_B = buffer_ops.buffer_load(rsrc_B, offsets, vec_width=4)
    fx.gemm(mma, acc, transform(data_A), transform(data_B), acc)   # d, a, b, c
```

1. **Prologue**: load iteration `START` before the loop.
2. **Loop** over the remaining iterations but one, carrying the prefetched data
   and the accumulator: unpack `state` (the swap, with no computation), issue
   the next iteration's loads, then compute on the current data.
3. **Epilogue**: compute the last iteration from `results`.

```python
offsets_0 = compute_offsets(START)
init_state = [buffer_ops.buffer_load(rsrc_A, offsets_0, vec_width=4),
              buffer_ops.buffer_load(rsrc_B, offsets_0, vec_width=4), acc]

for iv, state in range(START, END - 1, 1, init=init_state):
    data_A, data_B, acc = state                                    # swap: prefetched -> current
    offsets_next = compute_offsets(iv + 1)
    next_A = buffer_ops.buffer_load(rsrc_A, offsets_next, vec_width=4)
    next_B = buffer_ops.buffer_load(rsrc_B, offsets_next, vec_width=4)
    fx.gemm(mma, acc, transform(data_A), transform(data_B), acc)   # overlaps the loads above
    results = yield [next_A, next_B, acc]

data_A, data_B, acc = results
fx.gemm(mma, acc, transform(data_A), transform(data_B), acc)
```

Carry everything the next iteration's loads need the same way: block-table
entries, offsets, and scale factors (e.g. `next_block_id = load_block_table(iv + 1)`
before the loads that use it). Loads that depend on another load (block table
-> cache) stay in that order inside the prefetch block, and conditional loads
(`KV_QUANT_MODE` scales) keep the same conditions.

### PA decode example

The verified PA decode loop carries 15 values: 8 `vector<4xi32>` K loads
(4 tiles x 2), the partition start, 2 `i32` block-table values, the online
softmax `running_max`/`running_sum`, and 2 `vector<4xf32>` PV accumulators.
`init` and `yield` take a flat list, so small `_pack`/`_unpack` helpers
flatten and rebuild this state around the loop body.

**ISA result**: the 8 K-prefetch `buffer_load_dwordx4` sit at the END of the
loop body (after the PV MFMA), overlapping the MFMA pipeline drain; the
prologue has 8 K loads before the loop, and the epilogue only 8 V loads.

## When to Apply

| Signal | Description |
|--------|-------------|
| `for ... in range(N)` loop with `buffer_load` followed by MFMA | Load-then-compute in a loop body |
| Block table lookup inside loop | `buffer_load(block_table_rsrc, idx)` followed by `buffer_load(cache_rsrc, page_id * stride)` |
| KV cache iteration | Paged attention, flash attention, any tiled GEMM with paged memory |
| Scale factor loads | FP8 per-token quantization scales loaded per KV block |

Not worth it for single-iteration loops, for loops with no compute to overlap,
when MFMA utilization is already >90% (compute-bound), or when the kernel
already spills (see Register Budget).

## Compiler Constraints

The compiler, not the kernel, places `s_waitcnt`. Prefetch only moves the
loads earlier in the source so that the wait lands after enough compute. It
cannot set `vmcnt(N)` values, force `vmcnt(N>0)` over `vmcnt(0)`, remove
`s_barrier`s (those come from `gpu.barrier()` or cross-wave reductions), or
guarantee ISA issue order (see "Source Order Is Not Issue Order").

### Register Budget

Prefetch buffers live in arch VGPRs: each `buffer_load_dwordx4` holds 4, so
one extra set of 8 K loads costs about 32. On CDNA3 (gfx942) the arch and
accumulator (AGPR) files share **one 512-entry budget per SIMD**, so prefetch
buffers compete with MFMA accumulators, and occupancy follows the sum:

| arch_vgpr + accum_vgpr | Max waves/SIMD |
|---|---|
| <= 128 | 4 |
| <= 170 | 3 |
| <= 256 | 2 |
| <= 512 | 1 |
| > 512 | spill: severe regression |

Each file saturates at 256, so a report never shows a sum above 512: demand
beyond it appears as `arch=256, accum=256` plus a non-zero spill count. Treat
any spill as the regression. Crossing 256 drops a kernel from 2 waves to 1,
which may or may not pay off; profile both. Read the counts with the
**isa-resource-diff** skill (compile-only). `maxnreg` has been removed (it
never reached LLVM), and forcing `accum_vgpr=0` routes MFMA results through
`v_accvgpr_read` copies, measured at a 4.5x kernel regression; target
occupancy with `waves_per_eu`.

### Hoisting Loads into Barrier-Wait Regions

When one phase stalls in `s_barrier` (e.g. a softmax cross-wave reduce) and the
next phase needs global data (e.g. V values), issue those loads before the
barrier phase. The barrier waits anyway, so the load latency is free:

```python
v_data = buffer_ops.buffer_load(rsrc_v, offsets, vec_width=4)  # issued before the reduce
softmax_reduce(qk_scores)  # ~96K barrier stall cycles hide the ~17K-cycle V load
use(v_data)
```

### Source Order Is Not Issue Order

Writing a load early in the Python source does not mean it is issued early.
LLVM's machine scheduler reorders instructions within each basic block, and it
may move a load down next to the instruction that consumes it, for example to
lower register pressure. Basic-block boundaries limit it: a runtime `if` in the
loop body (lowered to `scf.if`) is a boundary, and loads above it stay above it.
A kernel can depend on such a branch without anyone noticing. Removing or
peeling the branch then moves the loads, even though no load code changed.

Observed on aiter's FlyDSL `pa_decode` (gfx950): peeling a per-tile validity
`if` out of the KV loop let the scheduler sink the V loads down to the PV MFMAs,
and bf16 cases with page size 64/128 got 17% slower. Putting an explicit
fence where the branch used to be restored the early issue, and the change then
cut geomean kernel time by 3.4%:

```python
v_next = load_v_tile(...)   # meant to be in flight during QK + softmax
rocdl.sched_barrier(0)      # nothing is scheduled across this point
qk = compute_qk(...)
```

Whenever you add or remove a runtime branch in a hot loop, or move loads, diff
the final ISA (`FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>`, then `*_final_isa.s`).
Check where the `buffer_load`/`global_load` instructions sit relative to the
MFMAs and to the `s_waitcnt vmcnt` that guards their use. Add
`rocdl.sched_barrier(0)` only where the dump shows that a load moved.

### Branch Variants That Start with the Same Loads

A wave-uniform runtime `if` can pick between unrolled variants that begin with
the same loads, for example a 16- and a 32-element rung of a reduction chosen
by a live count. LLVM hoists the shared loads above the branch, together with
their first uses (bf16 unpacks, masked selects), and waits for them there, so
the larger variant issues its remaining loads one memory round trip late. Two
obvious fixes do not work:

- Issue only the extra loads under the `if` and yield a constant in the other
  branch: InstCombine folds the conversion into the branch (the wait moves
  inside it), and MachineSink sinks the unconditional loads below the `if` when
  nothing before it uses them.
- Yield the already-loaded values from both branches: the yield forces register
  copies, and with them the waits, before the branch.

Give the variants different first instructions instead: issue one variant's
shared loads in reverse order and its other loads after them. Then check in the
final ISA that each variant issues all of its loads before its first
`s_waitcnt vmcnt`. Observed on gfx942 (FlyDSL 0.5.0): a split-K reduce kernel's
16/32 rung pair waited `vmcnt(0)` on 16 hoisted loads before branching; with
the 32 rung's loads reversed, each rung issued all its loads in one batch, and
the results were bit-identical.

## Verification

After applying prefetch:

1. **Correctness**: Run the existing test suite. Output must match bit-for-bit (fp32 accumulation) or within tolerance (fp8/bf16).
2. **Issue order**: Check in the final ISA dump that the prefetch loads are still issued where you put them (see "Source Order Is Not Issue Order").
3. **Performance**: Time the kernel with HIP events or graph replay. Do not use `rocprofv3 --kernel-trace` for this: profiler overhead dominates kernels of a few microseconds. For stalls, capture ATT (`/kernel-trace-analysis`) and look for fewer `VMEM` stall cycles in the loop body and higher MFMA utilization.
4. **Register pressure**: Check that occupancy did not drop and nothing spills. If it did, prefetch fewer buffers (e.g. only keys, not values).
