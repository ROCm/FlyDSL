# Re-diff at HEAD f019ad12: gemm1 a16w4 vs aiter (POST wave-split + pipeline)

Both FlyDSL-compiled to gfx950, disassembled `llvm-objdump -d --mcpu=gfx950`
(aiter) / `21_final_isa.s` external-llc dump (ours). Shape: K=3584, inter=512,
E=896, topk=16, BM=32, **tile_n=64**, tile_k=256, act=situv2, gate SEPARATED.

- OURS: `gemm1_a16w4_port_a16w4_h3584_i512_ne896_bm32_tn64_bcm0`
  (dump `/tmp/g1_dump_f019/.../21_final_isa.s`).
- AITER: `moe_gemm1_0` from `compile_mixed_moe_gemm1_a16w4` (tile_m32/tn64/tk256,
  situv2, SEPARATED), HSACO extracted from JIT cache pkl (`/tmp/aiter_g1.hsaco`,
  24480 B gfx950 ELF).

## The OLD diff (323fc5b0) is fully STALE

The wave-split (f019ad12) + pipeline (9eb81ed5) commits already fixed every
finding in `isa-diff-gemm1.md`:

| metric               | OLD ours | HEAD ours | AITER | verdict |
|----------------------|---------:|----------:|------:|---------|
| v_mfma / kernel      | 1792     | **448**   | 448   | FIXED (== aiter) |
| cvt / kernel         | 3584     | **896**   | 896   | FIXED (== aiter) |
| VGPR                 | 256      | **94**    | 134   | ours now LOWER |
| s_barrier / kernel   | 28 (2/it)| **14 (1/it)** | 15 | FIXED (1/iter) |
| vmcnt(N>0) partial   | 4        | **34**    | 29    | FIXED (now pipelined) |
| buffer_store_short   | 32       | **8**     | 0     | shrank (wave-split) |

So the residual 1.5x is a NEW, different signature. Instruction MIX is now
near-identical (buffer_load 140 vs 149, ds_read 224 vs 226, MFMA/cvt exact).

## Occupancy is NOT the cause (measured, rocprofv3 PMC, dev7)

| tokens | MeanOccupancyPerActiveCU | MeanOccupancyPerCU |
|-------:|-------------------------:|-------------------:|
| 128    | 3.82                     | 3.59               |
| 16384  | 3.96                     | 3.89               |

Ours runs at ~3.8-4.0 resident waves/CU (VGPR 94 -> not VGPR-capped; LDS 32 KB
-> not LDS-capped; well past the "stuck at 2" hypothesis). AITER at VGPR 134 /
LDS 41 KB would sit LOWER on occupancy. Occupancy is ruled out.

## THE residual: A-LDS read/DMA ordering -> extra vmcnt(0) drains

Full-kernel `s_waitcnt vmcnt(0)`: **OURS 36 vs AITER 19**. Per steady-state
iteration (between barriers):

| per K-iter        | OURS               | AITER              |
|-------------------|--------------------|--------------------|
| instr / iter      | ~172               | **~157**           |
| vmcnt(0) / iter   | **2 avg** (1 on even, 3 on odd/pong) | **1** (only at iter end) |
| vmcnt(N>0) / iter | 1-4                | 1-2                |
| max vmcnt(N) depth| shallow            | **vmcnt(8)**       |
| barrier / iter    | 1                  | 1                  |

### AITER iteration (verbatim structure, iter 621-778)
```
s_barrier
ds_read_b128 ... x16          ; read ALL of CURRENT tile's A-LDS first
                              ;   (buffer was DMA'd LAST iter, fully resident)
s_waitcnt vmcnt(5)            ; partial, only for last few
ds_read_b128 ...
buffer_load_dwordx4 ... lds   ; NOW issue NEXT tile's 4 A-DMA (after reads done)
buffer_load_dwordx4 ... lds  x4
s_waitcnt vmcnt(8)            ; partial -- 8 loads stay in flight
... (MFMA cluster overlaps the in-flight DMA) ...
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)   ; ONE drain, at END, protects buffer swap
s_barrier
```

### OUR iteration (verbatim, odd/pong iter 526-703, 3 vmcnt(0))
```
buffer_load_dwordx4 ... lds   ; next-tile A DMA, interleaved into the cluster
s_add3_u32 ...
s_waitcnt vmcnt(0)            ; <-- FULL DRAIN before ds_read (682)
ds_read_b128 v[66:69], v28 offset:304   ; A-LDS read for CURRENT tile
...
s_waitcnt vmcnt(0)            ; (639) another drain before a ds_read cluster
ds_read_b128 ...
s_waitcnt vmcnt(0)            ; (666) and another
ds_read_b128 ... ; buffer_load ... lds interleaved right after
```

**Root cause.** `buffer_load ... offen lds` (direct-to-LDS A DMA) is a VMEM op:
its LDS-write completion is tracked by **vmcnt**, not lgkmcnt. Our body issues the
next-tile A-DMA *interleaved among the same iteration's ds_read cluster*, so the
compiler must insert `s_waitcnt vmcnt(0)` before each ds_read that could alias the
in-flight LDS write — and because the A-DMA and B (mxfp4 W) dwordx4 loads share
the vmcnt counter, each such drain also flushes the B pipeline. Result: 2 vmcnt(0)
per iter (3 on the pong iter) and a shallow in-flight depth.

AITER avoids this by **phase-separating** within the iteration: consume the whole
current A-LDS buffer (ds_read x16) FIRST, THEN issue the next tile's A-DMA, and
keep a **single** end-of-iter `vmcnt(0)` to protect the double-buffer swap. Its
in-flight depth reaches vmcnt(8) and it is ~15 instr/iter shorter.

This ordering costs ~10% on instr/iter and, more importantly, the extra
per-iter VMEM drains serialize the B-weight loads. On the 14-tile K loop this is
the ~1.5x mid-band residual.

## ds_read / cvt scheduling is fine
cvt/MFMA = 2.0 both. ds_read/MFMA 0.5 both. The cvt cluster is densely
interleaved with MFMA in ours (see iter 703-875) -- no cvt/ds starvation of the
MFMA issue. Not a cause.

## Epilogue (secondary, small-token weighted)
OURS 8x `buffer_store_short` (uncoalesced 16-bit) vs AITER 8x `ds_write_b16` +
2x `buffer_store_dwordx2` (LDS cshuffle -> coalesced 64-bit). Only 8 stores now
(wave-split), so this is a small fraction even at small tokens; secondary.

## Large-M tok16384 = 2.53x: it's TILE_N, not split-K
Aiter's tuned CSV (kimik3_fp4_tuned_fmoe.csv, abf16_wfp4) uses **t32x128x256**
(tile_n=128) for tok>=1024 and t32x64x256 only for tok<=256; NO `_sk` suffix on
any a16w4 row -> **k_batch=1, no split-K**. We are pinned at tile_n=64. tile_n=128
halves the grid's N-block count and doubles per-CTA N work: fewer, fatter tiles ->
better B-load reuse and amortized epilogue at large M. This is the dominant lever
for tok>=1024 (the 1024 1.49x AND the 16384 2.53x), NOT split-K.

## Verdict / lever order
1. **tile_n=128 for M>=~256** (per-token dispatch) -- biggest, drives tok>=1024.
2. **A-LDS read/DMA phase-separation** (read-all-then-DMA, single end-drain) --
   the mid-band pipelining residual; harder (scheduler-sensitive).
3. waves_per_eu / xcd_swizzle constexpr knobs to match aiter's per-token CSV.
4. epilogue cshuffle -- secondary.

---

## LANDED (measured, dev7, median-of-3 cold)

Lever swept tile_n 64/128/256: for OUR kernel **tile_n=256 dominates** (wins or
ties 4/5 tokens; only a ~3% wash at tok128 that is within gfx950 clock noise),
and BEATS aiter at tok16. Per-token 128 was NOT worth a dispatch branch. No
split-K (aiter uses k_batch=1 for a16w4, confirmed).

### Lever 1 (9e9fb9c5): default TILE_N 64 -> 256
### Lever 2 (ec5ce0ea): phase-separate A-LDS read from next-tile A-DMA

s1 median-of-3 (us):

| tok   | f019 (tn64) | +Lever1 (tn256) | +Lever2 | aiter s1 | final gap |
|------:|------------:|----------------:|--------:|---------:|----------:|
| 16    | 117.1       | 89.1            | 88.2    | 92.6     | **0.95x (BEAT)** |
| 128   | 345.7       | 289.3           | 280.9   | 228.6    | 1.23x     |
| 1024  | 406.8       | 341.0           | 335.3   | 276.6    | 1.21x     |
| 4096  | 1005.4      | 770.9           | 774.3   | -        | -         |
| 16384 | 3334.5      | 1953.5          | 1915.2  | 1313.3   | **1.46x** (was 2.53x) |

### Why the residual stops here (root-cause of the remaining 1.2-1.46x)
- **tok16384 is now MFMA-BOUND** (PMC: SQ_VALU_MFMA_BUSY / SQ_WAVE_CYCLES = 1.05,
  LDS_wait/WAVE 0.065). Occupancy = 2 waves/CU (VGPR 256-capped at tn256). Aiter
  at tile_n=128 (VGPR 170) sits at occupancy ~2 as well but its 1.46x edge is a
  denser per-wave MFMA schedule, not something the pipelining/vmcnt levers reach.
- **tok128 is latency-bound** (MFMA/WAVE 0.64, SQ_WAIT_ANY/WAVE 0.345). The
  phase-sep (Lever 2) recovered part of the vmcnt(0) drains but at the fat tn256
  tile the drains were already ~1/iter, so the win is small (~1%).
- waves_per_eu can't raise occupancy (would force VGPR spill at num_acc_n=4).
- xcd_swizzle is not wired on the a16w4 gemm1 grid; at tok16384 the kernel is
  MFMA-bound (not HBM-channel-bound), so a grid swizzle would not help there
  (unlike gemm2, which WAS HBM-channel-imbalanced).
