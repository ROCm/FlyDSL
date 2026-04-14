---
name: prefetch-data-load
description: >
  Apply prefetch optimization to FlyDSL kernel loops: pre-load the first
  iteration's data before the loop, issue async loads for the next iteration
  inside the loop body, and swap buffers at the loop tail via scf.for
  loop-carried values. This overlaps data load latency with compute
  instructions. Use when a kernel has a loop where buffer_load feeds into
  MFMA/compute and load latency is exposed.
  Usage: /prefetch-data-load
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Prefetch Data Load Optimization

Apply software prefetch (double-buffering) to overlap async data loads with
compute in FlyDSL GPU kernel loops.

## Core Principle

GPU global memory loads (`buffer_ops.buffer_load`, `buffer_load_dwordx4`)
are **asynchronous** -- the load instruction returns immediately and the
hardware fetches data in the background. The data is only needed when a
subsequent instruction actually **consumes** it. If we issue the load early
enough, the data arrives by the time we need it, effectively hiding the load
latency behind compute work.

**Without prefetch** (load latency fully exposed):

```
for i in range(N):
    data = load(ptr + i)     # <-- stall: wait for data
    result = compute(data)   # <-- cannot start until load completes
```

Timeline:
```
|--load--|--stall--|--compute--|--load--|--stall--|--compute--|
```

**With prefetch** (load overlapped with compute):

```
# Pre-load first iteration BEFORE the loop
next_data = load(ptr + 0)

for i in range(N):
    # Swap: the prefetched data becomes current
    data = next_data

    # Issue load for NEXT iteration (async, non-blocking)
    if i + 1 < N:
        next_data = load(ptr + i + 1)

    # Compute using CURRENT data -- overlaps with next load
    result = compute(data)
```

Timeline:
```
|--load₀--|--compute₀ + load₁--|--compute₁ + load₂--|--compute₂--|
```

The total time drops from `N * (load + compute)` to roughly
`load + N * max(load, compute)`.

## FlyDSL Implementation: scf.for with Loop-Carried Prefetch

In FlyDSL kernels, Python-level `for _pi in range(N)` gets traced into N flat
copies that LLVM re-rolls. This makes the `data = next_data` swap **invisible**
to MLIR — both variables alias the same SSA value, so LLVM hoists loads as
loop-invariant.

**Solution**: Use FlyDSL's `scf.for` with `init=` (loop-carried values) to
create genuine SSA phi nodes. See the `flydsl-kernel-authoring` skill, section
"scf.for with Loop-Carried Values", for the full pattern and three critical
pitfalls.

### Transformation Steps

Given a loop like:

```python
for i in range(START, END):
    # === LOAD PHASE ===
    offsets = compute_offsets(i)
    data_A = buffer_ops.buffer_load(rsrc_A, offsets, vec_width=4)
    data_B = buffer_ops.buffer_load(rsrc_B, offsets, vec_width=4)

    # === COMPUTE PHASE ===
    result = rocdl.mfma_f32_16x16x16_f16(transform(data_A), transform(data_B), acc)
```

Apply the following transformation using `scf.for` with `init=`:

#### Step 1: Prologue — load first iteration before loop

```python
offsets_0 = compute_offsets(START)
next_data_A = buffer_ops.buffer_load(rsrc_A, offsets_0, vec_width=4)
next_data_B = buffer_ops.buffer_load(rsrc_B, offsets_0, vec_width=4)

init_state = [_unwrap(v) for v in [next_data_A, next_data_B, acc]]
```

#### Step 2: scf.for with loop-carried state

```python
_start = arith.index(0)
_stop = arith.index(N - 1)  # N-1 iterations; last handled in epilogue
_step = arith.index(1)

for iv, state in range(_start, _stop, _step, init=init_state):
    # Swap: prefetched -> current
    data_A = state[0]
    data_B = state[1]
    acc = state[2]

    # Prefetch next iteration (async, non-blocking)
    offsets_next = compute_offsets(iv + 1)
    next_data_A = buffer_ops.buffer_load(rsrc_A, offsets_next, vec_width=4)
    next_data_B = buffer_ops.buffer_load(rsrc_B, offsets_next, vec_width=4)

    # Compute using current data (overlaps with next load)
    acc = rocdl.mfma_f32_16x16x16_f16(transform(data_A), transform(data_B), acc)

    results = yield [_unwrap(v) for v in [next_data_A, next_data_B, acc]]
```

#### Step 3: Epilogue — process last iteration

```python
data_A = results[0]
data_B = results[1]
acc = results[2]
acc = rocdl.mfma_f32_16x16x16_f16(transform(data_A), transform(data_B), acc)
```

### Handling auxiliary data (block tables, scales)

Any offset calculations, block table lookups, or scale factor loads needed
for the *next* iteration's data should also be carried as loop state:

```python
init_state = [_unwrap(v) for v in [
    next_data_A, next_data_B, next_block_id, next_scale, acc
]]

for iv, state in range(_start, _stop, _step, init=init_state):
    data_A, data_B, block_id, scale, acc = state

    # Prefetch next iteration
    next_block_id = load_block_table(iv + 1)
    offsets_next = compute_offsets(iv + 1, next_block_id)
    next_data_A = buffer_ops.buffer_load(rsrc_A, offsets_next, vec_width=4)
    next_data_B = buffer_ops.buffer_load(rsrc_B, offsets_next, vec_width=4)
    next_scale = buffer_ops.buffer_load(rsrc_scale, next_block_id, vec_width=1)

    # Compute with current data
    acc = rocdl.mfma_f32_16x16x16_f16(
        transform(data_A) * scale, transform(data_B), acc
    )

    results = yield [_unwrap(v) for v in [
        next_data_A, next_data_B, next_block_id, next_scale, acc
    ]]
```

### PA Decode Kernel Example (verified, 112us, 0.75x vs Gluon)

State inventory (15 values carried across iterations):
- 8 x `vector<4xi32>` — K data (4 tiles x 2 loads)
- 1 x `i32` — partition_start
- 2 x `i32` — block table values (phys_block/page_off or phys_0/phys_1)
- 2 x `f32` — running_max, running_sum (online softmax)
- 2 x `vector<4xf32>` — PV accumulators

```python
# Pack/unpack helpers
def _pack(kv_flat, part_start, bt_vals, rmax, rsum, acc_pv):
    raw = kv_flat + [part_start] + bt_vals + [rmax, rsum] + acc_pv
    return [v.ir_value() if hasattr(v, 'ir_value') else v for v in raw]

def _unpack(state):
    kv_flat = list(state[0:8])
    kv = [[kv_flat[t*2], kv_flat[t*2+1]] for t in range(4)]
    return kv, state[8], list(state[9:11]), state[11], state[12], [state[13], state[14]]

# Prologue
pf_0 = issue_bt_k_loads(partition_0)
init_state = _pack(flatten(pf_0['kv']), pf_0['part_start'], ...)

# scf.for (bounds MUST be arith.index, not Python ints!)
for iv, state in range(arith.index(0), arith.index(N-1), arith.index(1), init=init_state):
    kv, part_start, bt, rmax, rsum, acc = _unpack(state)
    rmax, rsum, acc = compute_qk_softmax_pv(kv, part_start, bt, rmax, rsum, acc)
    pf_next = issue_bt_k_loads(next_partition(iv + 1))
    results = yield _pack(flatten(pf_next['kv']), pf_next['part_start'], ...)

# Epilogue: clear SmemPtr caches, compute last partition, write output
smem_ptr._view_cache = None
kv, part_start, bt, rmax, rsum, acc = _unpack(results)
compute_qk_softmax_pv(kv, part_start, bt, rmax, rsum, acc)
write_output(rmax, rsum, acc)
```

**ISA result**: 8 K-prefetch `buffer_load_dwordx4` appear at the END of the
loop body (after PV MFMA), overlapping with the MFMA pipeline drain. The
prologue has 8 K loads before the loop. The epilogue has 8 V loads only (no
K loads needed).

### Three Critical Pitfalls

1. **Loop bounds must be `arith.index()`, NOT Python ints.** If you write
   `range(0, 15, 1, init=...)`, the AST rewriter treats constant bounds as a
   Python `range` and unrolls the loop — silently ignoring `init=`. Use
   `arith.index(0)`, `arith.index(15)`, `arith.index(1)` instead.

2. **All `init` values must be raw MLIR `ir.Value`s.** FlyDSL wrappers like
   `Int32` / `Float32` don't have `.type` (only `.dtype`), and
   `scf.ForOp.__init__` calls `arg.type`. Unwrap via:
   ```python
   def _unwrap(v):
       return v.ir_value() if hasattr(v, 'ir_value') else v
   init_state = [_unwrap(v) for v in raw_list]
   ```

3. **Clear `SmemPtr._view_cache` before epilogue.** `SmemPtr.get()` caches the
   `memref.view` it creates. If called inside the `scf.for` body, the cached
   view is defined in the loop scope. Using it in the epilogue (outside the loop)
   causes an SSA dominance error. Fix:
   ```python
   my_smem_ptr._view_cache = None
   ```

## Applicable Patterns

This optimization applies whenever you see this pattern in a kernel:

| Signal | Description |
|--------|-------------|
| `for ... in range(N)` loop with `buffer_load` followed by MFMA | Load-then-compute in a loop body |
| Block table lookup inside loop | `buffer_load(block_table_rsrc, idx)` followed by `buffer_load(cache_rsrc, page_id * stride)` |
| KV cache iteration | Paged attention, flash attention, any tiled GEMM with paged memory |
| Scale factor loads | FP8 per-token quantization scales loaded per KV block |

## Compiler Constraints

FlyDSL kernels compile to GCN ISA where `s_waitcnt` insertion is controlled by
the **compiler**, not by the programmer. You cannot directly eliminate `s_waitcnt`
instructions. Instead, prefetch restructures the code so the compiler places
`s_waitcnt` after enough compute work to hide the latency.

### Register Budget

**Always check register headroom before adding prefetch buffers:**

On CDNA3 (gfx942 MI300X/MI308), VGPRs are split into **two separate register files**:
- **arch_vgpr** (256 per SIMD): used by VALU, VMEM loads, LDS ops, and prefetch buffers
- **accum_vgpr / AGPR** (256 per SIMD): used exclusively by MFMA result writeback

Prefetch buffers consume **arch_vgpr only** (they hold global load results). MFMA accumulators use **accum_vgpr only**. These do not compete.

```python
# Estimate arch_vgpr cost of prefetch buffers:
#   - Each buffer_load_dwordx4 = 4 arch_vgpr per load
#   - 8 K-cache loads = 8 x 4 = 32 arch_vgpr for one buffer set
#   - Double-buffering = 2 x 32 = 64 arch_vgpr (but one set is reused)
#   - Net additional arch_vgpr ~ 32 (the "next" buffer)
#
# On MI300X (gfx942): 256 arch_vgpr + 256 accum_vgpr per SIMD
# Occupancy = 256 / max(arch_vgpr, accum_vgpr) waves per SIMD
#
# Example: arch=148, accum=148 -> occupancy bottleneck = 148 -> 1 wave
# Adding 32 arch_vgpr -> arch=180, accum=148 -> bottleneck = 180 -> still 1 wave (safe)
# Adding 120 arch_vgpr -> arch=268 -> SPILL (critical, exceeds 256 per SIMD)
```

**Critical thresholds (gfx942, per register file):**
| Register File | Count | Max Waves/SIMD | Impact |
|--------------|-------|---------------|--------|
| arch_vgpr <= 128 | or accum <= 128 | 2 | Good occupancy |
| arch_vgpr <= 256 | or accum <= 256 | 1 | Minimum occupancy |
| arch_vgpr > 256 | or accum > 256 | **SPILL** | Register overflow -> severe perf regression |

**How to check current VGPR allocation** (from rocprofv3 database):
```sql
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
WHERE ks.KernelName LIKE '%target_kernel%'
LIMIT 5;
```

**WARNING**: Do NOT use `maxnreg` to force `accum_vgpr=0` in hopes of freeing
register space for prefetch. This forces MFMA results through arch_vgpr via
`v_accvgpr_read` spills, causing massive slowdown (measured 4.5x GPU kernel
regression).

### What Prefetch Can and Cannot Do

**CAN do:**
- Restructure the loop so `buffer_load` is issued earlier via `scf.for` loop-carried values
- The compiler will then schedule the corresponding `s_waitcnt` further from the load
- Overlap next iteration's loads with current iteration's MFMA compute

**CANNOT do:**
- Directly control `s_waitcnt vmcnt(N)` counter values
- Force the compiler to use `vmcnt(N>0)` instead of `vmcnt(0)`
- Eliminate barriers (`s_barrier`) — these come from explicit `gpu.barrier()` or cross-wave reduce primitives

### Hoisting Loads into Barrier-Wait Regions

A powerful technique specific to multi-phase kernels (like paged attention with
softmax reduce):

If a kernel has a phase that spends time in `s_barrier` waits (e.g., softmax
cross-wave reduce), and the **next** phase needs data from global memory (e.g.,
V-value loads), hoist those loads into the barrier-stalling region. The barrier
must wait regardless — issuing loads during that wait is essentially free.

```python
# BEFORE: V-value loads happen AFTER softmax reduce completes
softmax_reduce(qk_scores)  # <-- 96K stall cycles in barriers
v_data = buffer_ops.buffer_load(rsrc_v, offsets, vec_width=4)  # <-- additional load latency

# AFTER: V-value loads issued BEFORE/DURING softmax reduce
v_data_prefetch = buffer_ops.buffer_load(rsrc_v, offsets, vec_width=4)  # <-- async, non-blocking
softmax_reduce(qk_scores)  # <-- barrier stalls now overlap with v_data fetch
v_data = v_data_prefetch  # <-- data likely already arrived
```

This works because:
- `buffer_load` returns immediately (async)
- The barrier stalls are **dead time** where no useful work happens
- By the time barriers complete (~96K cycles), the V-value load (~17K cycles)
  has long since arrived

## Rules and Pitfalls

### Do
- **Prefetch ALL data** needed for the next iteration: keys, values, scales, block table entries
- **Place prefetch loads** immediately after the swap, BEFORE any compute that consumes current data
- **Use scf.for with init=** to carry prefetched data across iterations (Python variable swap is invisible to MLIR)
- **Minimize work between load and consume**: the more compute between prefetch issue and data use, the better the overlap
- **Keep the swap simple**: just unpack from `state`, no computation
- **Check VGPR budget**: calculate `current_arch_vgpr + prefetch_vgprs <= 256` to avoid spills
- **Hoist cross-phase loads into barrier regions**: if a kernel has barrier-heavy phases (reduce/sync), issue the next phase's loads before/during those barriers
- **Unwrap all init values to raw ir.Value**: use `v.ir_value() if hasattr(v, 'ir_value') else v`

### Don't
- **Don't prefetch if loop body is already memory-bound**: prefetching helps when compute (MFMA) duration >= load latency. If the loop is purely loads with no compute, prefetching won't help.
- **Don't prefetch too many buffers**: each prefetched variable occupies registers. If register pressure is already high (causing spills), prefetching more data makes it worse. Check `waves_per_eu` / occupancy.
- **Don't assume occupancy can increase**: on MI308 with 512 max VGPRs, adding prefetch buffers that push total VGPRs above 256 will drop occupancy from 2 to 1 wave/SIMD. This may or may not be acceptable — profile both configurations.
- **Don't reorder loads that have data dependencies**: if `load_B` depends on the result of `load_A` (e.g., block table lookup -> cache load), they must stay sequential within the prefetch block.
- **Don't forget to handle conditional branches**: if scale loads are conditional (`KV_QUANT_MODE`), the prefetch must replicate the same conditions.
- **Don't break the prologue/epilogue semantics**: the prologue covers iteration 0; `scf.for` runs N-1 iterations carrying prefetched data; epilogue processes the last iteration from `results`.
- **Don't use Python ints as scf.for bounds**: use `arith.index()` or the loop will be unrolled, silently ignoring `init=`.

## Verification

After applying prefetch:

1. **Correctness**: Run the existing test suite. Output must match bit-for-bit (fp32 accumulation) or within tolerance (fp8/bf16).
2. **Performance**: Profile with `rocprofv3 --kernel-trace`. Look for:
   - Reduced `VMEM` stall cycles in the loop body
   - Higher MFMA utilization percentage
   - Overall kernel duration reduction
3. **Register pressure**: Check that `waves_per_eu` (occupancy) didn't drop. If it did, consider prefetching fewer buffers (e.g., only keys, not values).

## When NOT To Use

- **Single-iteration loops** (`range(1)`): no next iteration to prefetch
- **Compute-bound kernels**: if MFMA utilization is already >90%, the bottleneck is compute, not memory — prefetching won't help
- **Very high register pressure**: if occupancy is already 1 wave/EU and the kernel spills, adding prefetch buffers will make it worse