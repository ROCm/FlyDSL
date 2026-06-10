---
name: a8w4-pf-race-analysis
description: Root cause analysis of A8W4 LDS prefetch race condition in gemm_fp8fp4_gfx1250.py
metadata:
  type: project
---

## Problem

Current kernel has precision regression vs baseline for A8W4 GEMM configs with
`tile_k=512, K=3072, num_buffers=4` (loop_iters=0, all work in tail).
Non-deterministic results (two identical runs differ), confirming a TDM-vs-ds_load
LDS race condition.

## Root cause identified

In `_use_lds_pf` path, the pf seed issues `s_wait_tensorcnt(0)` then immediately
does `_issue_pf_all_ks()` which ds_loads from LDS (A, B, As, Bs regions).

Under `wave_specialized_tdm`, each wave only issues its own TDM (wave0→A, wave1→B,
wave2→As, wave3→Bs). `s_wait_tensorcnt(0)` is **per-wave** — it only guarantees
the current wave's TDM has completed. Other waves may still be writing their LDS
regions when this wave starts ds_loading from them.

The baseline avoids this by NOT doing early ds_load prefetch — it uses the normal
`pipeline_fence_signal + pipeline_fence_wait` (which includes barrier) before
every compute_tile's ds_loads.

## Two race sites

1. **Prologue pf seed** (line ~2592): `s_wait_tensorcnt(0)` + `_issue_pf_all_ks(stage 0)`
   - Missing inter-wave barrier after tensorcnt wait
2. **Tail pf seed** (line ~2800): same pattern for `loop_iters == 0` case

## Fix approach

Need a barrier between `s_wait_tensorcnt(0)` and `_issue_pf_all_ks()`.

**CANNOT use `gpu.barrier()` / `workgroup_barrier()`** — caused GPU hang/deadlock.
Likely because this code point is inside a wave-specialized branch or the barrier
type is incompatible with the WGP barrier protocol used elsewhere.

Must use the same barrier mechanism as the pipeline fence:
`rocdl.s_barrier_signal(WGP_BARRIER_ID)` + `rocdl.s_barrier_wait(WGP_BARRIER_ID)`
(i.e. `pipeline_fence_wait` pattern).

## Tail next_lds race (secondary)

In the tail, interleaved `next_lds` prefetch can also race when the target stage's
TDM was issued too recently for the fence outstanding to guarantee completion.
Specifically at tail step i=2 (compute=2, next_cs=3): stage 3 TDM from step 0
should be guaranteed by fence(outstanding=2) draining 3→2, making step 0 complete.
This might actually be safe — need to recheck after fixing the primary barrier issue.

## Files

- `kernels/gemm_fp8fp4_gfx1250.py` — current kernel (has _use_lds_pf)
- `kernels/gemm_fp8fp4_gfx1250_baseline.py` — baseline (no _use_lds_pf)
- `scripts/check_a8w4_precision.py` — comparison script
- `kernels/gemm_common_gfx1250.py` — pipeline_fence*, workgroup_barrier, WGP_BARRIER_ID
