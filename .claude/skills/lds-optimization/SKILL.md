---
name: lds-optimization
description: >
  Optimize LDS (Local Data Share / shared memory) access patterns in FlyDSL
  GPU kernels. Diagnose bank conflicts and high lgkmcnt stalls from ATT trace
  data, then apply swizzle or padding layouts to eliminate conflicts. Also
  increase the distance between LDS write and subsequent LDS read to hide LDS
  latency. LDS read preceded by write always requires a sync (s_waitcnt
  lgkmcnt or s_barrier). Use when trace analysis shows ds_read/ds_write/lgkmcnt
  as a bottleneck.
  Usage: /lds-optimization
tools: Read,Edit,Bash,Grep,Glob,Agent
---

# LDS Optimization

Diagnose and fix LDS (shared memory) performance issues in FlyDSL kernels
on AMD CDNA GPUs (MI300X/MI308/MI350).

## When To Use

Run `/kernel-trace-analysis` first. Apply this skill when the trace shows:

| Signal | Threshold | Example |
|--------|-----------|---------|
| `s_waitcnt lgkmcnt(0)` with high stall | > 3000 cycles per instance | `L605: stall=4080 s_waitcnt lgkmcnt(0)` |
| `ds_write` / `ds_read` with high latency | > 500 cycles per instance | `L761: stall=960 ds_write2_b32` |
| Multiple `s_barrier` between `ds_write` and `ds_read` | Barrier stall > 5000 | `L606: stall=17024 s_barrier` |
| Total LDS-related stall > 15% of kernel stall | Sum all lgkmcnt + ds stalls | Softmax reduce phase in PA decode |

## LDS Architecture on CDNA3 (gfx942)

### Hardware Facts

- LDS size: **64 KB per CU** (workgroup-shared)
- LDS is organized into **32 banks**, each **4 bytes wide**
- Bank index = `(byte_address / 4) % 32`
- **Bank conflict**: when 2+ threads in the same wavefront access **different addresses** in the **same bank** in the same cycle, accesses are serialized
- **Broadcast**: when 2+ threads access the **same address** in the same bank, hardware broadcasts (no conflict)
- LDS throughput: **128 bytes/cycle** (peak, no conflicts)
- LDS latency: **~20-40 cycles** (async, hidden if enough work between write and read)
- **VGPR context**: LDS ops use **arch_vgpr** (not accum_vgpr). On CDNA3, arch_vgpr and accum_vgpr are separate 256-entry register files. LDS optimization does not interact with MFMA accumulator register pressure. See `/kernel-trace-analysis` Section 5.5 for VGPR architecture details.

## LDS Architecture on CDNA4 (gfx950)

### Hardware Facts

- LDS size: **160 KB per CU** (2.5x larger than gfx942's 64 KB)
- LDS is organized into **64 banks**, each **4 bytes wide** (640 DWords per bank)
- Bank index = `(byte_address / 4) % 64`
- **Bank conflict**: same rule as gfx942 — when 2+ threads in the same wavefront access **different addresses** in the **same bank** in the same cycle, accesses are serialized
- **Broadcast**: same as gfx942 — when 2+ threads access the **same address** in the same bank, hardware broadcasts (no conflict)
- LDS throughput: **256 bytes/cycle** (peak, no conflicts; 2x gfx942 due to 64 banks)
- LDS latency: **~2-64 cycles** per operation depending on bank conflicts (2 cycles best case, 64 cycles worst case with all threads conflicting on one bank)
- LDS allocation granularity: **1280 bytes** on **1280-byte alignment**; LDS allocations do not wrap around the LDS storage
- **Wavefront dispatch**: reads across a 64-thread wavefront are dispatched over **4 cycles** in waterfall fashion
- **32 concurrent LDS operations**: hardware can concurrently execute 32 read or write instructions (each 32-bit); extended instructions (read2/write2) can be 64-bit each
- **32 integer atomic units** for unordered atomic operations
- **VGPR context**: same as gfx942 — LDS ops use **arch_vgpr** (not accum_vgpr). On CDNA4, arch_vgpr and accum_vgpr are separate 256-entry register files.

### Key Differences from gfx942

| Aspect | gfx942 (CDNA3) | gfx950 (CDNA4) |
|--------|----------------|-----------------|
| LDS size per CU | 64 KB | 160 KB |
| Number of banks | 32 | 64 |
| Bank index formula | `(addr/4) % 32` | `(addr/4) % 64` |
| Peak throughput | 128 bytes/cycle | 256 bytes/cycle |
| LDS allocation granularity | 256 bytes | 1280 bytes |
| Max LDS per workgroup | 64 KB | 160 KB |
| MFMA Transpose Load | Not available | `DS_READ_B64_TR_B16/B8/B4`, `DS_READ_B96_TR_B6` |

### Impact on Bank Conflict Analysis

Because gfx950 has **64 banks** instead of 32, the bank conflict patterns change:

- **Stride that causes conflicts**: multiples of 64 banks (256 bytes) instead of 32 banks (128 bytes)
- A stride of 128 bytes that caused **full conflict on gfx942** (all threads hit same bank) will only cause **partial conflict on gfx950** (threads alternate between 2 banks)
- To cause full 64-way conflict on gfx950, the stride must be a multiple of `64 * 4 = 256` bytes
- **XOR swizzle masks may need adjustment** — masks designed for 32-bank gfx942 may be suboptimal on 64-bank gfx950

### MFMA Transpose Load from LDS (gfx950 only)

CDNA4 introduces dedicated instructions for transposing data while loading from LDS to VGPRs, eliminating the need for explicit transpose via `ds_write` + `ds_read` with permuted addresses:

| Instruction | Element Size | VGPRs Written | Description |
|-------------|-------------|---------------|-------------|
| `DS_READ_B64_TR_B16` | 16-bit (fp16/bf16) | 2 VGPRs | Load column-major A or row-major B matrix; two instructions load a complete matrix. Each lane holds 4 consecutive M or N values. |
| `DS_READ_B64_TR_B8` | 8-bit (fp8/bf8) | 2 VGPRs | Same as B16 but for 8-bit elements. First loads K=0..7,16..23,32..39,48..55; second loads remaining K values. |
| `DS_READ_B64_TR_B4` | 4-bit (int4) | 2 VGPRs | Same pattern for 4-bit elements. First loads K=0..15,32..47; second loads remaining K values. |
| `DS_READ_B96_TR_B6` | 6-bit | 3 VGPRs | 6-bit element transpose load into 3 VGPRs. Does NOT require even-VGPR alignment. |

Requirements:
- EXEC mask must be set to all 1's before executing
- LDS address must be aligned to the data size
- DS ops reading/writing 64-bit or larger data must use even-aligned VGPRs (except `DS_READ_B96_TR_B6`)

These instructions are useful for MFMA operand preparation — loading A/B matrices from LDS in the transposed layout needed by MFMA instructions without explicit LDS-based transpose.

### LDS Instruction Model

LDS operations (`ds_read_*`, `ds_write_*`, `ds_bpermute_*`, `ds_swizzle_*`) are **asynchronous**:

```
ds_write_b32 v_addr, v_data    ; issues async write, returns immediately
; ... other instructions ...    ; LDS write completes in background
s_waitcnt lgkmcnt(0)            ; stall until all LDS/SMEM ops complete
ds_read_b32 v_result, v_addr   ; now safe to read
```

Key rules:
1. **Write-before-read requires sync**: any `ds_read` that depends on a prior `ds_write` must have `s_waitcnt lgkmcnt(0)` or `s_barrier` in between
2. **`s_barrier` implies cross-wave sync**: if wave A writes and wave B reads, `s_barrier` is required (not just `lgkmcnt`)
3. **Longer write-read distance = better latency hiding**: more instructions between `ds_write` and the subsequent `s_waitcnt lgkmcnt(0)` allow the write to complete in the background

## Diagnosing LDS Bottlenecks from Trace

### Step 1: Identify LDS-heavy regions

```python
import json

with open('ui_output_agent_XXX_dispatch_YYY/code.json') as f:
    data = json.load(f)
instructions = data['code']
# Columns: [ISA, _, LineNum, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle]

# Find all LDS-related instructions
lds_insts = [i for i in instructions if i[0].startswith('ds_') or
             ('lgkmcnt' in i[0] and i[8] > 0)]

total_lds_stall = sum(i[8] for i in lds_insts)
total_stall = sum(i[8] for i in instructions)
print(f"LDS stall: {total_lds_stall} / {total_stall} = {100*total_lds_stall/total_stall:.1f}%")

# Show hottest LDS instructions
for i in sorted(lds_insts, key=lambda x: x[8], reverse=True)[:15]:
    print(f"  L{i[2]:>4d}  stall={i[8]:>6d}  idle={i[9]:>6d}  {i[0][:55]}  | :{i[3].split(':')[-1]}")
```

### Step 2: Classify the bottleneck type

**Type A: Bank Conflicts** (high stall on `ds_read`/`ds_write` themselves)

```
L 766  stall=  160  ds_read2_b64 v[44:47], v28 offset1:8        ; <-- bank conflict
L 767  stall=  320  ds_read2_b64 v[36:39], v28 offset0:16 offset1:24  ; <-- bank conflict
```

Signs:
- `ds_read_*` / `ds_write_*` instructions with stall > 100 cycles per hit
- Multiple reads/writes with similar base address but different offsets that map to same banks
- `ds_read2_b64` / `ds_write2_b32` with offsets that are multiples of the bank count:
  - **gfx942**: multiples of 32 (= same bank, 32-bank LDS)
  - **gfx950**: multiples of 64 (= same bank, 64-bank LDS)

**Type B: Write-Read Latency Exposed** (high stall on `s_waitcnt lgkmcnt(0)` after `ds_write`)

```
L 761  stall=  960  ds_write2_b32 v28, v41, v43 offset0:32 offset1:48
L 764  stall= 4560  s_waitcnt lgkmcnt(0)    ; <-- write latency fully exposed
L 765  stall= 1468  s_barrier
L 766  stall=  160  ds_read2_b64 v[44:47], v28 offset1:8
```

Signs:
- `s_waitcnt lgkmcnt(0)` with > 2000 stall cycles immediately after `ds_write`
- Very few instructions between `ds_write` and `s_waitcnt`
- This means the write hasn't completed by the time we need to wait

**Type C: Cross-Wave Reduce Serialization** (high stall on `s_barrier` in reduce chains)

```
L 605  stall= 4080  s_waitcnt lgkmcnt(0)     ; wait for ds_bpermute
L 606  stall=17024  s_barrier                 ; cross-wave sync
L 607  stall=27220  s_waitcnt vmcnt(0)        ; also waiting for global loads
```

Signs:
- `ds_bpermute` -> `lgkmcnt(0)` -> `s_barrier` -> `ds_write LDS` -> `lgkmcnt(0)` -> `s_barrier` -> `ds_read LDS` pattern
- Multiple barriers (> 4) in a reduce region

## Optimization Method 1: Swizzle Layout

### The Problem

When multiple threads access LDS with a stride that is a multiple of the bank count, every access hits the same bank:

- **gfx942 (32 banks)**: stride multiple of 128 bytes causes full conflict
- **gfx950 (64 banks)**: stride multiple of 256 bytes causes full conflict; stride of 128 bytes causes 2-way conflict (threads alternate between 2 banks)

```
# gfx942 (32 banks): stride=128 -> full conflict
Thread 0: addr = base + 0*128  -> bank (0*128/4)%32 = 0
Thread 1: addr = base + 1*128  -> bank (1*128/4)%32 = 0  <- CONFLICT
Thread 2: addr = base + 2*128  -> bank (2*128/4)%32 = 0  <- CONFLICT

# gfx950 (64 banks): stride=128 -> only 2-way conflict (NOT full conflict)
Thread 0: addr = base + 0*128  -> bank (0*128/4)%64 = 0
Thread 1: addr = base + 1*128  -> bank (1*128/4)%64 = 32  <- different bank!
Thread 2: addr = base + 2*128  -> bank (2*128/4)%64 = 0   <- conflict with thread 0

# gfx950 (64 banks): stride=256 -> full conflict
Thread 0: addr = base + 0*256  -> bank (0*256/4)%64 = 0
Thread 1: addr = base + 1*256  -> bank (1*256/4)%64 = 0  <- CONFLICT
...
```

### The Solution: XOR-Based Swizzle

Swizzle XORs bits of the row index into the column index of the LDS address, distributing accesses across different banks:

```
swizzled_col = original_col XOR (row >> shift)
```

This ensures threads accessing the same column in different rows hit different banks.

### FlyDSL XOR Swizzle with SmemAllocator

In FlyDSL, LDS is managed through `SmemAllocator`. To apply swizzle, XOR the
row index into the LDS address when computing store/load offsets:

```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.expr import arith

allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")
lds_key = allocator.allocate_array(T.f16, KV_BLOCK_SIZE * HEAD_SIZE)

@flyc.kernel
def my_kernel(...):
    lds_base = allocator.get_base()
    lds_key_ptr = lds_key(lds_base)

    # XOR-swizzle address: distribute bank accesses
    # row_idx and col_idx are the logical 2D coordinates
    # XOR_BITS controls swizzle width (typically 4 for fp16 vec=8)
    swizzled_col = arith.xori(col_idx, arith.andi(row_idx, XOR_MASK))
    lds_offset = row_idx * PADDED_STRIDE + swizzled_col
    lds_key_ptr.store(data, [lds_offset])
```

### Choosing Swizzle Parameters

The goal is to make vectorized access span enough banks:

| Data Type | Element Size | Recommended Vec Width | Banks Covered per Vec |
|-----------|-------------|----------------------|----------------------|
| fp32      | 4 bytes     | 4                    | 4 banks (16 bytes)   |
| fp16/bf16 | 2 bytes     | 8                    | 4 banks (16 bytes)   |
| fp8       | 1 byte      | 16                   | 4 banks (16 bytes)   |

For XOR mask:
- **gfx942 (32 banks)**: use `32 / (vec * element_size / 4) - 1` to ensure full bank coverage
- **gfx950 (64 banks)**: use `64 / (vec * element_size / 4) - 1` — the doubled bank count means wider XOR masks may be needed to fully distribute accesses

### Example: Fix Bank Conflicts in KV Cache Load to LDS

Before (conflict-prone, linear layout):

```python
# Linear shared memory layout — threads in same warp hit same banks
lds_key = allocator.allocate_array(T.f16, KV_BLOCK_SIZE * HEAD_SIZE)
# Store key tile: all threads write to column 0,1,2... -> bank conflicts
lds_offset = row * HEAD_SIZE + col
lds_key_ptr.store(data, [lds_offset])
```

After (swizzled, conflict-free):

```python
# XOR-swizzle distributes accesses across banks
XOR_BITS = 4  # for fp16 vec=8: covers 4 banks per vec
lds_key = allocator.allocate_array(T.f16, KV_BLOCK_SIZE * HEAD_SIZE)
swizzled_col = arith.xori(col, arith.shli(arith.andi(row, 0x7), XOR_BITS))
lds_offset = row * HEAD_SIZE + swizzled_col
lds_key_ptr.store(data, [lds_offset])  # now conflict-free
```

## Optimization Method 2: Padding

### The Problem

Same as swizzle — stride-aligned accesses cause bank conflicts. Padding adds extra unused elements to change the effective stride.

### The Solution

Add 1 element of padding per row to break the alignment:

```python
# gfx942 (32 banks):
# Without padding: row stride = HEAD_SIZE (e.g., 128)
# Bank stride = 128 * 2 / 4 = 64 -> 64 % 32 = 0 -> ALL rows hit same bank column
# With padding: row stride = HEAD_SIZE + 1 (e.g., 129)
# Bank stride = 129 * 2 / 4 = 64.5 -> fractional -> conflicts eliminated

# gfx950 (64 banks):
# Without padding: row stride = HEAD_SIZE (e.g., 128)
# Bank stride = 128 * 2 / 4 = 64 -> 64 % 64 = 0 -> ALL rows hit same bank column (still conflicts!)
# With padding: row stride = HEAD_SIZE + 1 (e.g., 129)
# Bank stride = 129 * 2 / 4 = 64.5 -> fractional -> conflicts eliminated
```

### FlyDSL Padding Implementation

```python
PADDING = 1  # or a small number
PADDED_HEAD_SIZE = HEAD_SIZE + PADDING

# Allocate with extra column for padding
lds_key = allocator.allocate_array(T.f16, KV_BLOCK_SIZE * PADDED_HEAD_SIZE)

@flyc.kernel
def my_kernel(...):
    lds_base = allocator.get_base()
    lds_key_ptr = lds_key(lds_base)

    # Write key data using padded stride (ignore padding column)
    lds_offset = row * PADDED_HEAD_SIZE + col
    lds_key_ptr.store(data, [lds_offset])

    # Read back using same padded stride
    data = lds_key_ptr.load([row * PADDED_HEAD_SIZE + col])
```

### Padding Amount

The minimum padding to eliminate all bank conflicts:

```
# gfx942 (32 banks):
padding_elements = 32 / (element_size_bytes)  # worst case

# gfx950 (64 banks):
padding_elements = 64 / (element_size_bytes)  # worst case
```

But usually 1-4 elements suffice. The cost is extra LDS usage:
- 1 element padding per row: `KV_BLOCK_SIZE * element_size` extra bytes
- Must ensure total LDS usage stays within **64 KB** per CU (gfx942) or **160 KB** per CU (gfx950)

### Swizzle vs Padding Trade-offs

| Aspect | Swizzle | Padding |
|--------|---------|---------|
| LDS overhead | None (zero extra bytes) | Extra bytes per row |
| Complexity | Need correct XOR mask params (arch-dependent) | Simple: just add 1 to stride |
| Address computation | XOR adds ~1 SALU instruction | Simple offset, no extra compute |
| Risk | Wrong params = silent bank conflicts | Exceeding LDS limit = kernel fail |
| LDS limit | N/A | gfx942: 64 KB, gfx950: 160 KB |
| Preferred when | LDS near capacity, need zero overhead | Simple cases, LDS has headroom |

**Recommendation**: Prefer swizzle (zero overhead). Use padding only when swizzle layout is hard to integrate with the kernel's access pattern. On gfx950, the 160 KB LDS gives much more headroom for padding.

## Optimization Method 3: Increase Write-Read Distance

### The Problem

When `ds_write` is immediately followed by `s_waitcnt lgkmcnt(0)` and then `ds_read`, the ~20-40 cycle LDS write latency is fully exposed as stall:

```
ds_write_b32 ...          ; async write issued
s_waitcnt lgkmcnt(0)      ; STALL: write hasn't completed yet (3000+ cycles)
ds_read_b32 ...           ; read must wait for write
```

### The Solution

Insert useful compute work between the write and the wait:

```
ds_write_b32 ...          ; async write issued
; --- insert independent compute here ---
v_mfma_f32_16x16x32 ...  ; MFMA takes ~64 cycles, overlaps with LDS write
v_add_f32 ...             ; more independent ALU work
v_mul_f32 ...
; --- write has completed by now ---
s_waitcnt lgkmcnt(0)      ; no stall (or minimal stall)
ds_read_b32 ...           ; data ready immediately
```

### FlyDSL-Level Implementation

At the Python/FlyDSL level, you control write-read distance by reordering operations:

```python
# BEFORE: write and read are close together
lds_ptr.store(data, [offset])               # ds_write
gpu.barrier()                                # s_barrier (includes lgkmcnt wait)
result = lds_ptr.load([offset])             # ds_read

# AFTER: insert independent work between write and barrier
lds_ptr.store(data, [offset])               # ds_write (async)

# Do independent compute that doesn't need the LDS data
next_offsets = compute_next_offsets()        # SALU/VALU work
next_data = buffer_ops.buffer_load(rsrc, next_offsets, vec_width=4)  # global load (also async)
scale_factor = buffer_ops.buffer_load(rsrc_scale, scale_off, vec_width=1)

gpu.barrier()                                # by now, LDS write has completed
result = lds_ptr.load([offset])             # ds_read (no stall)
```

### What to Insert Between Write and Read

Prioritize by latency-hiding value:

1. **Global loads for next phase** (`buffer_ops.buffer_load`) — these are also async, ~300+ cycle latency
2. **Address computation** (`compute_offsets`) — SALU/VALU, ~4-8 cycles each
3. **Independent MFMA chains** — if available, ~64 cycles per MFMA
4. **Scalar loads** (`s_load_dword*`) — kernel arguments, ~20 cycles

Avoid inserting:
- Operations that depend on the LDS write result (data dependency)
- More LDS operations (would compete for LDS bandwidth)
- Operations that increase register pressure beyond budget

## Verification Checklist

After applying LDS optimizations:

1. **Correctness**: Run tests. Swizzle changes must be applied consistently to both write and read paths — if the write uses swizzled addresses, the read must use the same swizzle.

2. **Re-profile**: Run `/kernel-trace-analysis` and check:
   - `ds_read_*` / `ds_write_*` stall should decrease
   - `s_waitcnt lgkmcnt(0)` stall after `ds_write` should decrease
   - No new bank conflicts introduced

3. **LDS usage**: Check total LDS consumption:
   ```python
   # Estimate: sum of all allocator.allocate_array() sizes * element_size
   # gfx942: Must be <= 65536 bytes (64 KB) per workgroup
   # gfx950: Must be <= 163840 bytes (160 KB) per workgroup
   # Note: gfx950 allocates LDS in 1280-byte granularity (1280-byte aligned blocks)
   ```

4. **Register pressure**: Swizzle adds ~1-2 SALU instructions for address XOR. Padding doesn't add register pressure but uses more LDS. Neither should significantly impact VGPR count.

## Quick Reference: Common LDS Patterns in Paged Attention

| Pattern | Location | Typical Issue | Fix |
|---------|----------|---------------|-----|
| K/V cache tile in LDS | QK/PV MFMA loop | Bank conflicts from stride=HEAD_SIZE | Swizzle with XOR on row index |
| Softmax reduce via LDS | `ds_write -> barrier -> ds_read` | Write-read latency exposed + too many barriers | Increase write-read distance; replace with `ds_bpermute` chain |
| Cross-wave max/sum broadcast | `ds_write -> barrier -> ds_read` from different wave | Cross-wave sync overhead | Merge max+sum into single reduce pass |
| MFMA accumulator shuffle | `ds_write accum -> barrier -> ds_read permuted` | Bank conflicts if accumulator layout misaligns | Swizzle or use `ds_bpermute` for permutation |

## Output

After optimization, report:
- Which LDS bottleneck type was identified (bank conflict / write-read latency / reduce serialization)
- Which optimization was applied (swizzle / padding / distance increase)
- Before/after `lgkmcnt` stall cycles and `ds_*` instruction stalls
- LDS usage before/after (bytes)
- Any impact on VGPR count or occupancy