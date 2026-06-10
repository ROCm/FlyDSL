---
name: a8w4-pf-race-analysis
description: Root cause and fix for A8W4 LDS prefetch race in gemm_fp8fp4_gfx1250.py
metadata:
  type: project
---

## Problem

Current kernel has precision regression vs baseline for A8W4 GEMM.
Non-deterministic results confirm TDM-vs-ds_load LDS race conditions.

## Two bugs found and fixed

### 1. pf seed: missing inter-wave barrier

`s_wait_tensorcnt(0)` is per-wave. Under `wave_specialized_tdm` each wave only
issues its own TDM (wave0→A, wave1→B, wave2→As, wave3→Bs). After
`s_wait_tensorcnt(0)`, this wave's TDM is done but other waves may still be
writing their LDS regions. `_issue_pf_all_ks()` then ds_loads from ALL regions.

**Fix**: Replace bare `s_wait_tensorcnt(0)` with `_pipeline_fence(outstanding=0)`
which adds a workgroup barrier after the tensorcnt wait. Two sites:
- Prologue pf seed (line ~2589)
- Tail pf seed for loop_iters==0 (line ~2797)

### 2. tail: skipped tensorcnt wait before next-tile prefetch

When `_tail_fence_out == 0 and _load_stage is None`, the tail skips
`s_wait_tensorcnt` and only does `s_barrier_signal`. But earlier tail steps'
TDMs may still be in-flight. When the interleaved prefetch then ds_loads from
a recently-loaded stage, it reads stale LDS data.

**Fix**: Don't skip tensorcnt wait when `_use_tail_pf and _tail_next_cs is not None`.

## Verified

Both M=1 and M=64 configs produce bit-exact results vs baseline.
Determinism confirmed (3 runs identical).
No `s_wait_tensorcnt(0)` added inside compute_tile — the fix is purely
at fence/barrier call sites outside compute.
