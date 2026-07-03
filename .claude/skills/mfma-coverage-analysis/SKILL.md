---
name: mfma-coverage-analysis
description: From an ATT trace, find which hot-loop instructions are NOT hidden behind MFMA execution -- i.e. what is keeping cyc/mfma above the MFMA execute floor (16 for fp4, 32 for fp8 16x16x128). Models the matrix unit as a pipeline (next_free) for the EXPOSED %, then attributes exposed cycles by per-instruction STALL field intersected with the idle gaps (stall==0 ops and MFMA-hidden stall never count); the uncovered remainder is pure-idle scheduling slack. Use when a GEMM/attention kernel is MFMA-bound and you want to push cyc/mfma toward the floor, or the user asks "what isn't hidden behind MFMA", "哪些指令没被 mfma 掩盖", "暴露的指令", or which non-MFMA op is the biggest exposed stall.
---

# MFMA Coverage Analysis

## The model (user-validated, next_free pipeline)

The matrix unit is a PIPELINE: each MFMA occupies ONE EXEC-cycle execute slot, but
consecutive MFMAs can **issue < EXEC apart** and still both be hidden (dense fp4
MFMAs issue ~8 cyc apart yet each execute-slot is 16). Track `next_free` = the cycle
the unit becomes free:
- MFMA issues at `t <= next_free` → hidden (co-issued in the shadow); `next_free += EXEC`
- MFMA issues at `t >  next_free` → the unit was IDLE for `t - next_free` → **EXPOSED**

EXPOSED = sum of those idle gaps; shrink it toward 0 so cyc/mfma → EXEC (see
[[feedback-mfma-bound-reduce-scalar-toward-cyc16]]).

`next_free` gives the total EXPOSED % (matrix-unit idle fraction). It REPLACES the
older "union of `[issue, issue+EXEC)` windows" model, which capped overlapping
windows and over-reported exposure.

## Attribution: STALL field ∩ idle gap (both filters matter)

Each ATT wave row is `[cycle, issue_dur, STALL, total_dur, code_id]`. **`r[2]` is
the authoritative per-instruction stall.** Attribute EXPOSED cycles with TWO filters:

1. **Only stall>0 ops block.** Ops with stall==0 (`ds_read_b128`, `s_add`) are pure
   fill between MFMAs and never block — do NOT attribute by "which op sits in the
   gap" (first/longest/occupancy geometry all mislabeled fill and gave contradictory
   answers on the same trace: ds_read 33%, then s_add 34%, then buffer_load 30% — all wrong).
2. **Only the part of the stall overlapping a next_free GAP counts as exposed.** A
   stall that overlaps MFMA execute is HIDDEN (the pipe is busy). Because MFMAs issue
   ~8 cyc apart but each execute-slot is 16, several overlapping executes keep the
   unit busy well past any single `[issue,issue+16)`; intersecting with the next_free
   idle gaps handles this. (Skipping this over-reported lgkmcnt: raw stall 976 →
   true-exposed 504.)

Whatever exposed cycles no stalling instruction covers = **PURE IDLE** (scheduling /
dependency slack, not one op's fault) — usually the TOP bucket.

Example (fp4_gemm_4wave, cyc/mfma 18.73, EXPOSED 14% = 6344 cyc): **pure-idle 72% +
s_barrier 15% + s_waitcnt(lgkmcnt) 7% + buffer_load 3% + s_waitcnt(vmcnt) 0%.** The
biggest lever is packing MFMAs tighter (pure-idle), not cutting any single op; the
per-op stalls are small once MFMA-hidden cycles are removed.

So: an op that issues between MFMAs is only a problem if its `r[2]` stall > 0.
s_waitcnt is split into vmcnt / lgkmcnt so you can see whether VMEM or LDS waits
dominate (here: LDS/lgkmcnt, i.e. waiting on ds_read; VMEM already hidden).

EXEC is the MFMA **execute** latency, NOT issue latency:
- fp4 `mfma_scale_f32_16x16x128_f8f6f4` → **16**
- fp8 `mfma..16x16x128` → **32** (its issue latency)
Pass the right `--exec`.

## How to run

```bash
# 1. capture ATT (see att-hotloop-benchmark skill) on an idle GPU, long-K shape
# 2. run, first without --range to print the cycle span:
python3 .claude/skills/mfma-coverage-analysis/scripts/mfma_coverage.py <dispatch_dir>
# 3. pick a mid steady slice spanning ~10 outer-loop iters, re-run:
python3 .claude/skills/mfma-coverage-analysis/scripts/mfma_coverage.py \
    <dispatch_dir> --range 219000,245000 --exec 16
```

Output: MFMA-covered %, EXPOSED %, cyc/mfma vs floor, then the exposed cycles split
into pure-idle + per-op (stall ∩ idle-gap) buckets.

## Reading it

- **pure-idle top** → the matrix unit is idle with NO stalling instruction there;
  it's scheduling/dependency slack. Attack by packing MFMAs tighter (spread thunks
  differently, interleave_stride, deepen prefetch so operands are ready earlier),
  NOT by cutting one op. This is usually the biggest bucket once stalls are filtered.
- **Rank named ops by the stall column (`r[2]`) ∩ idle gap**, not by count or by which
  op sits in a gap. An op with stall==0 is free; a stall overlapping MFMA execute is
  hidden. Only stall>0 ∩ idle-gap cycles are real exposure.
- **s_barrier high** → the wave is waiting at the per-iter cross-wave sync (the
  4 waves finish the iter at slightly different times). Structural for a fixed wave
  count; not cuttable without changing sync granularity / wave layout.
- **s_waitcnt(lgkmcnt) high** → waiting on LDS reads (ds_read). Move the producing
  ds_read earlier / deepen the register prefetch so the value is ready.
- **s_waitcnt(vmcnt) high** → waiting on VMEM (global/g2s/scale gather). Deepen the
  VMEM prefetch or relax the vmcnt bound.
- **buffer_load / readfirstlane with stall** → the load's own issue stalled, or the
  v→s readfirstlane serialized; precompute wave-uniform values once (SGPR).

## Caveats

- Uses one wave file (se0_sm0_sl0_wv0.json by default). Different waves are
  equivalent repeats; pass `--wave` to check another.
- Row schema: `[cycle, issue_dur, STALL, total_dur, code_id]`. `r[2]` (stall) is the
  authoritative blocking metric; `code.json["code"][cid][0]` is the asm text. The
  `next_free` pass gives EXPOSED %; the stall column gives the per-op ranking. (See
  att-hotloop-benchmark for the complementary barrier-window cyc/mfma summary.)
