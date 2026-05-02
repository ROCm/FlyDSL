---
name: oob-detection
description: >
  Detect out-of-bounds memory accesses in CPU or GPU code using static interval
  analysis and runtime assertions/printfs. Use when investigating OOB, buffer
  overrun, invalid memory access, HIP/ROCm illegal address, CUDA illegal memory
  access, silent tensor corruption, or suspicious buffer_load/store address
  arithmetic.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# OOB Detection

Use this skill when a kernel, runtime, or host path may read or write outside
its intended buffer or logical tile. Prefer proving the address range first
instead of relying only on runtime failures.

## 1. Classify the OOB

Decide which boundary may be violated:

| Boundary | Meaning | Typical detector |
|---|---|---|
| Physical allocation OOB | Address leaves the allocated tensor/buffer | HIP illegal address, runtime failure |
| Logical object OOB | Address stays in allocation but crosses a row/head/tile | Static interval analysis, explicit runtime check |
| Lane/thread ownership OOB | Thread reads another lane's slot | Static interval analysis, debug printf/assert |
| LDS/shared-memory OOB | Address exceeds allocated shared-memory region | Static interval analysis, LDS index guard |

Physical tools usually miss logical OOB because the access can still be inside
the same allocation.

## 2. Static Interval Analysis

For each memory access, write the exact element range:

```text
start = base + offset_expression
end   = start + vec_width - 1
legal = [object_base, object_base + object_extent - 1]
```

Then substitute known ranges:

- Thread/lane ids: `threadIdx.x`, `lane`, `lane16id`, `warp_id`, `rowid`
- Compile-time loops: `range_constexpr(N)` gives `i in [0, N-1]`
- Vector widths: `buffer_load(..., vec_width=W, dtype=T)` reads `W` elements of `T`
- Strides and shapes: tensor `.shape`, `.stride`, layout shape/stride, tile extents
- Masks/clamps: `select(valid, value, safe_value)` changes the range only if it dominates the load/store

If `max(end) > object_base + object_extent - 1`, the OOB is statically proven.
If `min(start) < object_base`, the lower-bound OOB is statically proven.

### FlyDSL Example

For a Q head with `HEAD_SIZE = 128`:

```text
q_elem = q_base + lane16id * 8
load_start = q_elem + qwi * 4
load_end = load_start + 3
lane16id in [0, 15], qwi in [0, 3]

max(load_end) = q_base + 15*8 + 3*4 + 3 = q_base + 135
legal head end = q_base + 127
```

This proves logical OOB for the head. If each lane owns only 8 elements, then
`qwi in [2, 3]` also proves lane-slot OOB for every lane.

## 3. Add Runtime Logical Checks

When static proof is not enough or the formula depends on runtime values, add a
temporary guard immediately before the load/store. In FlyDSL kernels, prefer a
small `printf` with the failing coordinates:

```python
load_start = q_elem + arith.constant(qwi * 4, type=T.i32)
load_end = load_start + arith.constant(vec_width - 1, type=T.i32)
legal_end = q_base + arith.constant(HEAD_SIZE - 1, type=T.i32)

if load_end > legal_end:
    fx.printf(
        "OOB q load: lane=%d qwi=%d q_base=%d load=[%d,%d] legal_end=%d\n",
        lane16id,
        arith.constant(qwi, type=T.i32),
        q_base,
        load_start,
        load_end,
        legal_end,
    )
```

For stores, include the output tensor/tile coordinates and the flattened offset.
Keep debug prints narrow; too many lanes printing can hide the useful signal.

## 4. Fix Strategy

Prefer fixing the invariant, not masking the fault:

- If the loop trip count is wrong, reduce the loop or vector width.
- If the per-lane ownership changed, update the layout, LDS store, and reader together.
- If a boundary tile is partial, clamp or predicate before the load/store.
- If a descriptor/resource range is too large or offset overflows i32, chunk the buffer resource or widen arithmetic before truncation.
- After the fix, rerun both the focused failing test and one neighboring shape that exercises the boundary.

## 5. Report Format

When reporting an OOB investigation, include:

- Access expression and element units
- Proven or observed failing range
- Legal range and which boundary was violated
- Minimal fix and validation command
