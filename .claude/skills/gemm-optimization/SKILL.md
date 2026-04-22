---
name: gemm-optimization
description: >
  Comprehensive guide to optimizing GEMM (General Matrix Multiply) kernels in
  FlyDSL on AMD CDNA GPUs. Covers tiling strategy, LDS ping-pong double-buffer,
  XOR bank-conflict swizzle, A/B data prefetch pipeline, 2-stage software
  pipelining, MFMA instruction scheduling (hot_loop_scheduler), epilogue
  strategies (direct store vs CShuffle), TFLOPS/bandwidth calculation, main-loop
  instruction count analysis, and bottleneck identification from ATT traces.
  Based on the production preshuffle_gemm kernel.
  Usage: /gemm-optimization
allowed-tools: Read Edit Bash Grep Glob Agent
---

# GEMM Optimization Guide

Comprehensive guide to writing and optimizing high-performance GEMM kernels in
FlyDSL on AMD CDNA GPUs (MI300X gfx942, MI350 gfx950).

Based on the production `kernels/preshuffle_gemm.py` implementation.

---

## 1. Tiling Strategy

### 1.1 Three-Level Tiling

GEMM tiles the output C[M, N] and the reduction K into blocks:

```
C[M, N] = A[M, K] × B[K, N]^T

Grid mapping:
  block_x → M tiles (tile_m rows each)
  block_y → N tiles (tile_n cols each)

Thread mapping (256 threads = 4 waves × 64 lanes):
  wave_id = tid // 64  ∈ [0, 3]     → N dimension partitioning
  lane_id = tid % 64   ∈ [0, 63]    → M + N dimension within wave
  lane_div_16 = lane_id // 16       → M dimension (4 groups of 16)
  lane_mod_16 = lane_id % 16        → N dimension within MFMA
```

### 1.2 Derived Tile Parameters

```python
m_repeat   = tile_m // 16            # M-direction 16x16 MFMA repeat count
n_per_wave = tile_n // 4             # N range per wave (4 waves split tile_n)
num_acc_n  = n_per_wave // 16        # N-direction 16x16 accumulators per wave
k_unroll   = tile_k_bytes // a_elem_vec_pack // 64  # K-steps per tile (K64 micro-steps)
```

### 1.3 Recommended Tile Configurations

| Scenario | tile_m | tile_n | tile_k | Data Type | Notes |
|----------|--------|--------|--------|-----------|-------|
| Small batch (M ≤ 32) | 16 | 64-128 | 256-512 | FP8/INT8 | Memory-bound, large tile_k for reuse |
| Medium batch | 64 | 256 | 128 | FP8/INT8/BF16 | Balanced compute/memory |
| Large batch (M ≥ 4096) | 128 | 256 | 128 | FP8/INT8 | Compute-dense, needs async copy |
| FP4 (gfx950) | 32-64 | 128-256 | 256 | FP4 | MFMA_SCALE instructions |

### 1.4 Tile Size Constraints

- `tile_m` must be multiple of 16 (MFMA M dimension)
- `tile_n` must be multiple of 64 (4 waves × 16 N per MFMA)
- `tile_k * elem_bytes` must be multiple of 64 (K64-byte micro-step)
- `tile_m * tile_k * elem_bytes` should fit comfortably in LDS (64KB on gfx942, 160KB on gfx950)
- B matrix is pre-shuffled to `(N/16, K/64, 4, 16, kpack_bytes)` layout — tile_k must divide K evenly

### 1.5 MFMA Count Per Tile

Total MFMA instructions per tile:

```
MFMA_per_tile = k_unroll × m_repeat × num_acc_n × 2
                                                  ↑ 2x K32 per K64 micro-step

Example (tile 64×256×128, FP8):
  k_unroll = 128 / 64 = 2
  m_repeat = 64 / 16 = 4
  num_acc_n = 256 / 4 / 16 = 4
  MFMA_per_tile = 2 × 4 × 4 × 2 = 64 MFMAs

Example (tile 64×256×512, FP8):
  k_unroll = 512 / 64 = 8
  MFMA_per_tile = 8 × 4 × 4 × 2 = 256 MFMAs
```

---

## 2. LDS Ping-Pong Double Buffer (2-Stage Pipeline)

### 2.1 Concept

With `lds_stage=2`, the kernel allocates **two separate LDS buffers** for the A
tile. While one buffer is used for MFMA computation, the next K-tile's A data
is loaded into the other buffer. This hides the global-to-LDS load latency.

```
Time →
Buffer PONG: [Compute tile_k=0] [   Load tile_k=2  ] [Compute tile_k=2] ...
Buffer PING: [   Load tile_k=1  ] [Compute tile_k=1] [   Load tile_k=3  ] ...
```

### 2.2 FlyDSL Implementation

```python
# Two independent SmemAllocators (separate LDS regions)
allocator_pong = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")
allocator_ping = SmemAllocator(None, arch="gfx942", global_sym_name="smem1")

lds_a_pong = allocator_pong.allocate_array(T.i8, buffer_size_bytes)
lds_a_ping = allocator_ping.allocate_array(T.i8, buffer_size_bytes)
```

### 2.3 Main Loop Structure (2-Stage)

Each iteration processes **2 K-tiles** (one pong, one ping):

```python
def _build_pingpong_body(k_iv, inner_state):
    accs_in, bt_flat_in, a0pf_in = _unpack_state(inner_state)
    b_tile_pong_in = _unflatten_b_tile(bt_flat_in)

    # Phase 1: compute on PONG, prefetch to PING
    next_k1 = k_iv + tile_k
    store_a_tile_to_lds(prefetch_a_tile(next_k1), lds_a_ping)  # A → PING LDS
    b_tile_ping = prefetch_b_tile(next_k1)                      # B → VGPR
    accs_in, _ = compute_tile(accs_in, b_tile_pong_in, lds_a_pong,
                              a0_prefetch=a0pf_in)
    hot_loop_scheduler()                                         # instruction hints
    rocdl.s_waitcnt(num_b_loads)
    gpu.barrier()
    a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

    # Phase 2: compute on PING, prefetch to PONG
    next_k2 = k_iv + (tile_k * 2)
    store_a_tile_to_lds(prefetch_a_tile(next_k2), lds_a_pong)  # A → PONG LDS
    b_tile_pong_new = prefetch_b_tile(next_k2)                   # B → VGPR
    accs_in, _ = compute_tile(accs_in, b_tile_ping, lds_a_ping,
                              a0_prefetch=a0_prefetch_ping)
    hot_loop_scheduler()
    rocdl.s_waitcnt(num_b_loads)
    gpu.barrier()
    a0_prefetch_pong_new = prefetch_a0_pack(lds_a_pong)

    return _pack_state(accs_in, _flatten_b_tile(b_tile_pong_new),
                       a0_prefetch_pong_new)
```

### 2.4 LDS Size Budget

```
lds_tile_bytes = tile_m × tile_k × elem_bytes
2-stage total = 2 × lds_tile_bytes
+ CShuffle epilogue (optional): tile_m × tile_n × 2 bytes

Example (64×128, FP8): 2 × 64 × 128 = 16 KB total
Example (128×128, FP8): 2 × 128 × 128 = 32 KB total
```

Limits: 64 KB on gfx942, 160 KB on gfx950.

---

## 3. LDS XOR Bank-Conflict Swizzle

### 3.1 The Problem

A tile stored row-major in LDS with stride = tile_k creates bank conflicts when
multiple rows are read simultaneously (threads in the same wave access the same
bank for different addresses).

### 3.2 XOR Swizzle Formula

```python
def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle at 16-byte granularity."""
    rem = row % k_blocks16
    return col ^ (rem * 16)
```

- `k_blocks16 = tile_k_bytes // a_elem_vec_pack // 16` — number of 16-byte blocks in K
- Applied to both **write** (global → LDS) and **read** (LDS → VGPR) paths
- Zero LDS overhead (no extra bytes), ~1 SALU instruction per address

### 3.3 Write Path

```python
# In store_a_tile_to_lds():
col_swz_bytes = swizzle_xor16(row_a_local, col_local_bytes, k_blocks16)
lds_offset = row_a_local * lds_stride_bytes + col_swz_bytes
lds_ptr.store(data, [lds_offset])
```

### 3.4 Read Path

```python
# In lds_load_packs_k64():
col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
lds_offset = curr_row_a_lds * lds_stride_bytes + col_base_swz_bytes
a_pack = lds_ptr.load([lds_offset])
```

**Critical**: swizzle must be consistent between write and read. If one path
uses swizzle but the other doesn't, data will be read from wrong positions.

---

## 4. Data Prefetch Pipeline

### 4.1 A Matrix: Global → LDS

Two paths for loading A into LDS:

**Synchronous** (default): Global → VGPR → LDS
```python
a_regs = prefetch_a_tile(base_k)          # buffer_load_dwordx4 → VGPR
store_a_tile_to_lds(a_regs, lds_buffer)   # ds_write from VGPR → LDS
```

**Asynchronous** (use_async_copy=True): Global → LDS directly
```python
prefetch_a_to_lds(base_k, lds_buffer)  # raw_ptr_buffer_load_lds (DMA)
```
Async copy bypasses VGPR, reducing register pressure. Available on gfx942/gfx950.

### 4.2 B Matrix: Global → VGPR (Preshuffle)

B is pre-shuffled to match MFMA register layout, loaded directly to VGPR:

```python
b_tile = prefetch_b_tile(base_k)  # buffer_load_dwordx4 → VGPR
# b_tile structure: k_unroll × [(packs0[num_acc_n], packs1[num_acc_n])]
```

Each K64 micro-step needs `2 × num_acc_n` i64 values for B (K32 × 2).

### 4.3 A0 Prefetch (Cross-Tile LDS Prefetch)

After `gpu.barrier()` completes (LDS is valid), immediately load the first A
pack from LDS into VGPR registers, overlapping with upcoming VMEM loads:

```python
a0_prefetch = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)
```

This hides the first `ds_read` latency (~20-40 cycles) behind the global loads
that follow.

### 4.4 Pipeline Timeline

```
Iter i:
  1. [VMEM] Load A(i+1) → PING LDS, Load B(i+1) → VGPR
  2. [MFMA] Compute tile(i) using PONG LDS + B(i) VGPR
  3. [SCHED] hot_loop_scheduler() — interleave MFMA with pending loads
  4. [SYNC] s_waitcnt + barrier — wait for PING LDS to be valid
  5. [LDS] A0 prefetch from PING — ds_read first pack

  Swap PING ↔ PONG, repeat for i+1
```

---

## 5. Instruction Scheduling (hot_loop_scheduler)

### 5.1 Purpose

The `hot_loop_scheduler()` inserts `rocdl.sched_*` hints between the MFMA
compute phase and the next iteration's loads. These hints tell the compiler
how to interleave different instruction types to maximize pipeline utilization.

### 5.2 Scheduling Primitives

| Hint | Meaning | Maps to |
|------|---------|---------|
| `rocdl.sched_barrier(0)` | Full scheduling barrier — no reordering across | Compiler fence |
| `rocdl.sched_mfma(N)` | Allow N MFMA instructions | `v_mfma_*` |
| `rocdl.sched_dsrd(N)` | Allow N LDS read instructions | `ds_read_*` |
| `rocdl.sched_dswr(N)` | Allow N LDS write instructions | `ds_write_*` |
| `rocdl.sched_vmem(N)` | Allow N global memory instructions | `buffer_load_*` |

### 5.3 Standard Schedule Pattern (gfx942, sync copy)

```python
def hot_loop_scheduler():
    mfma_group = num_acc_n
    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
    mfma_per_iter = 2 * mfma_group
    sche_iters = mfma_total // mfma_per_iter

    # Prologue: pre-load first 2 LDS packs, interleave with first few MFMAs
    rocdl.sched_dsrd(2)                    # 2 ds_read for a0_prefetch
    rocdl.sched_mfma(1)
    rocdl.sched_mfma(1)

    # Main schedule: each iteration = 1 VMEM + mfma_group MFMAs + 1 ds_read + mfma_group MFMAs
    dswr_tail = num_a_loads
    dswr_start = max(sche_iters - dswr_tail - 2, 0)
    for sche_i in range_constexpr(sche_iters):
        rocdl.sched_vmem(1)                # 1 global load (B tile or A tile)
        rocdl.sched_mfma(mfma_group)       # N MFMA instructions
        rocdl.sched_dsrd(1)                # 1 LDS read (A data)
        rocdl.sched_mfma(mfma_group)       # N more MFMAs
        if sche_i >= dswr_start - 1:
            rocdl.sched_dswr(1)            # LDS write (next A tile, tail end)

    rocdl.sched_barrier(0)                 # fence
```

### 5.4 Key Scheduling Insights

1. **MFMA instructions dominate**: they form the backbone of the schedule
2. **LDS reads (ds_read) interleave with MFMAs**: one ds_read per 2×mfma_group MFMAs
3. **Global loads (VMEM) interleave**: one buffer_load per scheduler iteration
4. **LDS writes (ds_write) go at the tail**: they overlap with the last MFMAs
   of the current tile, landing before the `gpu.barrier()` at iteration boundary
5. **dswr_start** ensures LDS writes are scheduled early enough to complete
   before the barrier, but late enough to not interfere with compute

### 5.5 Async Copy Schedule (gfx950)

For async copy, the scheduler uses `_build_scheduler()` to evenly distribute
ds_read and VMEM loads across all MFMAs:

```python
dsrd_schedule = _build_scheduler(num_ds_load - dsrd_preload, mfma_total)
vmem_schedule = _build_scheduler(num_gmem_loads, mfma_total)
```

This produces a per-MFMA schedule: after each `sched_mfma(1)`, emit the
appropriate number of `sched_dsrd` and `sched_vmem` hints.

---

## 6. MFMA Inner Loop Structure

### 6.1 K64 Micro-Step (FP8/INT8)

Each K64 micro-step performs 2× K32 MFMA calls:

```python
for ku in range_constexpr(k_unroll):           # K dimension (K64 steps)
    b_packs0, b_packs1 = b_tile_in[ku]        # B data for this K64 step
    col_base = col_offset_base_bytes + ku * 64 # LDS column offset

    for mi in range_constexpr(m_repeat):       # M dimension (16-row blocks)
        curr_row_a_lds = row_a_lds + (mi * 16)
        a0, a1 = lds_load_packs_k64(...)      # Load A from LDS (2× i64)

        for ni in range_constexpr(num_acc_n):  # N dimension (16-col accumulators)
            acc[mi * num_acc_n + ni] = mfma_k64_bytes(
                acc[mi * num_acc_n + ni],
                a0, a1,
                b_packs0[ni], b_packs1[ni]
            )
```

### 6.2 MFMA Instruction Selection

| Data Type | K per MFMA | Instruction | Accumulator |
|-----------|-----------|-------------|-------------|
| FP8 | K=32 | `mfma_f32_16x16x32_fp8_fp8` | f32×4 |
| INT8 | K=32 | `mfma_i32_16x16x32i8` | i32×4 |
| BF16 | K=16 | `mfma_f32_16x16x16bf16_1k` | f32×4 |
| FP16 | K=16 | `mfma_f32_16x16x16f16` | f32×4 |
| FP4 (gfx950) | K=128 | `mfma_scale_f32_16x16x128_f8f6f4` | f32×4 |

---

## 7. Epilogue Strategies

### 7.1 Direct Store (Default)

Each thread writes its MFMA accumulator elements directly to global memory:

```python
# Row mapping: MFMA output layout → global C matrix
for mi in range_constexpr(m_repeat):
    for ii in range(4):  # 4 rows per lane_div_16 group
        row = bx_m + mi * 16 + lane_div_16 * 4 + ii
        for ni in range_constexpr(num_acc_n):
            col = by_n + wave_id * n_per_wave + ni * 16 + lane_mod_16
            # scale + truncate + store
            val = acc[mi * num_acc_n + ni][ii] * scale_a * scale_b
            buffer_store(truncate(val, out_dtype), c_rsrc, row * N + col)
```

**Pros**: no extra LDS, simple
**Cons**: non-coalesced stores for some tile sizes

### 7.2 CShuffle Epilogue

Rearranges thread-to-element mapping via LDS for coalesced global writes:

1. **Write to LDS**: accumulator values written row-major to `lds_out`
2. **Barrier**: synchronize all threads
3. **Shuffle read**: threads re-map to `(MLane=8, NLane=32)` for contiguous output
4. **Store**: `buffer_store_dwordx2` for 4-element vectorized writes

```python
# CShuffle parameters
e_vec = 4 if (tile_n % 128 == 0) else 2
m_reps_shuffle = tile_m // 8
n_reps_shuffle = tile_n // (32 * e_vec)
```

**Pros**: coalesced stores, higher memory throughput
**Cons**: extra LDS allocation + barrier

**When to use**: for large tile_n (≥ 128) where output coalescing matters.

---

## 8. Performance Metrics and Bottleneck Analysis

### 8.1 TFLOPS Calculation

```python
flops = 2 * M * N * K                      # each multiply-add = 2 FLOPs
tflops = flops / (us / 1e6) / 1e12         # TFLOPS

# Peak references (gfx942 MI300X, single GCD):
#   FP8:  ~653 TFLOPS peak (mfma_f32_16x16x32_fp8)
#   BF16: ~326 TFLOPS peak
#   INT8: ~653 TOPS peak
```

### 8.2 Bandwidth Calculation

```python
# FP8/INT8:
bytes_moved = (M * K * elem_bytes)     # A matrix
            + (N * K * elem_bytes)     # B matrix (pre-shuffled)
            + (M * N * 2)              # C output (bf16/fp16)
            + (M + N) * 4             # per-token scales (f32)

# INT4:
bytes_moved = (M * K) + (N * K) // 2 + (M * N * 2) + (M + N) * 4

# FP4 (MXFP4):
bytes_moved = (M * K) // 2 + (N * K) // 2 + (M * N * 2) + (M + N) * (K // 32)

tbps = bytes_moved / 1e12 / (us / 1e6)  # TB/s
```

### 8.3 Memory-Bound vs Compute-Bound

```
Arithmetic Intensity = flops / bytes_moved

AI < roofline_crossover → memory-bound
AI > roofline_crossover → compute-bound

Practical rule: M ≤ 512 → memory-bound (focus on bandwidth)
               M > 512 → compute-bound (focus on MFMA utilization)
```

### 8.4 Bottleneck Identification from ATT Traces

Run `/kernel-trace-analysis` on the GEMM kernel, then check:

| Symptom | Bottleneck | Action |
|---------|-----------|--------|
| High `s_waitcnt vmcnt(0)` stall before MFMA | Global load latency exposed | Improve prefetch overlap, increase tile_k |
| High `s_waitcnt lgkmcnt(0)` stall | LDS latency exposed | Increase write-read distance, check bank conflicts |
| High `s_barrier` stall | Workgroup sync overhead | Check LDS stage, reduce barrier count |
| Low MFMA utilization (< 50%) | Memory-bound | Increase tile size, prefetch more aggressively |
| Many `s_nop` between MFMAs | Pipeline bubbles | Interleave loads between MFMAs, tune scheduler |
| High-cycle `buffer_load` | TA-blocked | Reduce concurrent loads, check access coalescing |

---

## 9. Main-Loop Instruction Count Analysis

### 9.1 Counting Method

Dump ISA and count instructions in the main MFMA loop:

```bash
FLYDSL_DUMP_IR=1 python my_gemm.py
# Check final_isa.s for the hot loop between two s_barrier instructions
```

Or use rocprofv3 ATT trace `code.json` to identify the loop body by examining
instructions between repeated `s_barrier` patterns.

### 9.2 Expected Instruction Counts Per Tile (FP8, sync copy)

For tile (64, 256, 128), FP8, lds_stage=2:

| Category | Count | Formula |
|----------|-------|---------|
| **MFMA** | 64 | k_unroll × m_repeat × num_acc_n × 2 = 2×4×4×2 |
| **ds_read** (A from LDS) | ~16 | k_unroll × m_repeat × 2 (a0, a1 per mi) |
| **buffer_load** (B from global) | ~16 | k_unroll × 2 × num_acc_n |
| **buffer_load** (A to VGPR) | ~8 | num_a_loads (A tile for next iter) |
| **ds_write** (A VGPR → LDS) | ~8 | num_a_loads (store to LDS) |
| **s_barrier** | 1 | synchronization |
| **SALU** (address, swizzle) | ~20-30 | offset computation, XOR swizzle |
| **Total** | ~130-150 | depends on tile config |

### 9.3 Ideal Ratios

```
MFMA ratio = MFMA_count / total_instructions
  > 40%: good (compute-dominant loop)
  30-40%: acceptable (some overhead)
  < 30%: too much non-MFMA overhead, review scheduling

Memory instructions = ds_read + buffer_load + ds_write
Memory ratio = memory_count / total_instructions
  < 40%: good overlap
  > 50%: memory-dominant, try larger tile_k or fewer loads
```

### 9.4 Comparing with Reference Kernels

When aligning FlyDSL GEMM with reference implementations (e.g., aiter):

```bash
# Count key instructions in ISA
grep -c "v_mfma"        final_isa.s       # MFMA count
grep -c "s_barrier"     final_isa.s       # barrier count
grep -c "buffer_load"   final_isa.s       # global loads
grep -c "ds_read"       final_isa.s       # LDS reads
grep -c "ds_write"      final_isa.s       # LDS writes
```

Target: FlyDSL MFMA count should match reference; barrier count ≤ reference.

---

## 10. Register Budget

### 10.1 VGPR Estimation

```
Accumulators: m_repeat × num_acc_n × 4 VGPRs (f32×4 per accumulator)
B tile:       k_unroll × 2 × num_acc_n × 2 VGPRs (i64 per B pack)
A prefetch:   2 × 2 VGPRs (a0 prefetch, 2× i64)
A tile regs:  num_a_loads × 4 VGPRs (if sync copy, dwordx4 per load)
Address:      ~10-20 VGPRs (offsets, indices)
```

Example (tile 64×256×128, FP8):
```
Accumulators: 4 × 4 × 4 = 64 VGPRs
B tile:       2 × 2 × 4 × 2 = 32 VGPRs
A prefetch:   4 VGPRs
A tile regs:  8 × 4 = 32 VGPRs
Address:      ~16 VGPRs
Total:        ~148 arch_vgpr
```

### 10.2 Occupancy Impact

On gfx942 (256 arch_vgpr + 256 accum_vgpr per SIMD):

| arch_vgpr | accum_vgpr | Waves/SIMD | Assessment |
|-----------|-----------|------------|------------|
| ≤ 128 | ≤ 128 | 2 | Good |
| 129-256 | ≤ 256 | 1 | Acceptable for compute-bound |
| > 256 | any | SPILL | Critical regression |

MFMA accumulators use **accum_vgpr** (separate file). Prefetch buffers, B tile,
and A tile use **arch_vgpr**. These do not compete.

---

## 11. Async Copy (gfx942/gfx950)

### 11.1 When to Use

- `tile_m ≥ 128` (enough compute to hide async DMA latency)
- Saves arch_vgpr (A data bypasses VGPR, goes directly Global → LDS)
- Requires `use_async_copy=True`

### 11.2 Implementation

```python
# Direct global → LDS DMA
rocdl.raw_ptr_buffer_load_lds(
    a_rsrc, lds_ptr, size_i32, global_offset,
    soffset, offset_imm, aux,
)
# gfx942: 4 bytes per DMA op
# gfx950: 16 bytes per DMA op
```

### 11.3 Trade-offs

| Aspect | Sync Copy | Async Copy |
|--------|----------|------------|
| Path | Global → VGPR → LDS | Global → LDS (DMA) |
| arch_vgpr usage | +32 for A tile regs | 0 (A bypasses VGPR) |
| Scheduling | Explicit ds_write interleaving | DMA engine handles transfer |
| Best for | Small tile_m, low register pressure | Large tile_m (≥ 128) |
| gfx942 granularity | 16B (dwordx4) | 4B (1 dword per DMA) |
| gfx950 granularity | 16B (dwordx4) | 16B (4 dwords per DMA) |

---

## 12. B Matrix Preshuffle Layout

### 12.1 Preshuffle Format

B is pre-transposed and reshuffled on CPU before kernel launch:

```
Original B: [N, K]  (row-major)
Preshuffle: [N/16, K/kpack, 4, 16, kpack_bytes]
```

Where:
- `kpack = 64 // elem_bytes` for FP8/INT8 (kpack=64), `4` for BF16/FP16 (kpack=4)
- The `4` dimension maps to 4 dwords per lane (buffer_load_dwordx4)
- The `16` dimension maps to 16 lanes within MFMA

### 12.2 Benefits

- Global loads map directly to MFMA register layout — no VALU shuffle needed
- Coalesced global access (consecutive threads load consecutive addresses)
- One-time CPU cost, amortized over many kernel invocations

---

## 13. Quick Reference: Optimization Checklist

| Stage | Check | Action if Failing |
|-------|-------|-------------------|
| **Tiling** | tile_m × tile_n fills GPU (enough blocks) | Reduce tile size |
| **Tiling** | tile_k × elem_bytes ≤ LDS budget / 2 | Reduce tile_k |
| **LDS** | Bank conflict count (trace ds_read stalls) | Apply XOR swizzle |
| **Prefetch** | VMEM stalls before MFMA in trace | Improve prefetch pipeline |
| **2-Stage** | Using lds_stage=2 | Enable double-buffer |
| **Scheduler** | s_nop / idle between MFMAs | Tune hot_loop_scheduler |
| **Epilogue** | Output store bandwidth | Use CShuffle for large tile_n |
| **Registers** | arch_vgpr ≤ 256 | Reduce buffers, use async copy |
| **ISA Count** | MFMA ratio ≥ 40% | Reduce non-MFMA overhead |
| **Performance** | TFLOPS vs peak | Identify bottleneck category |

---

## 14. Worked Example: Optimizing a 5120×5120×8320 FP8 GEMM

### Step 1: Choose tile size

```
tile_m=64, tile_n=256, tile_k=128
Grid: (5120/64) × (5120/256) = 80 × 20 = 1600 blocks
```

### Step 2: Estimate MFMA count

```
k_unroll = 128/64 = 2, m_repeat = 64/16 = 4, num_acc_n = 256/4/16 = 4
MFMA_per_tile = 2 × 4 × 4 × 2 = 64
Total MFMAs per block = 64 × (8320/128) = 64 × 65 = 4160
```

### Step 3: Estimate LDS usage

```
lds_tile = 64 × 128 = 8 KB
2-stage = 16 KB (well within 64 KB limit)
```

### Step 4: Estimate VGPR

```
Accumulators: 4 × 4 × 4 = 64 (→ accum_vgpr)
B tile: 2 × 2 × 4 × 2 = 32
A tile: 8 × 4 = 32
Total arch_vgpr ≈ 80 + overhead → ~120 (occupancy = 2 waves)
```

### Step 5: Calculate theoretical performance

```
flops = 2 × 5120 × 5120 × 8320 = 436 GFLOP
Target: ~500 TFLOPS → ~0.87 ms
bytes = 5120×8320 + 5120×8320 + 5120×5120×2 + (5120+5120)×4
     = 85.2M + 52.4M + 0.04M = 137.6 MB
Bandwidth: 137.6 MB / 0.87 ms = 158 GB/s (well below HBM peak)
→ Compute-bound, focus on MFMA utilization
```

### Step 6: Profile and iterate

```bash
rocprofv3 --kernel-trace --stats -f csv -- python test_preshuffle_gemm.py \
    --in_dtype fp8 -M 5120 -N 5120 -K 8320 --tile_m 64 --tile_n 256 --tile_k 128
```

Compare GPU kernel time with theoretical minimum. If >1.5× theoretical,
run ATT trace analysis (`/kernel-trace-analysis`) to identify bottleneck.