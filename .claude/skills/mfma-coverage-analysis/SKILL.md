---
name: mfma-coverage-analysis
description: From an ATT trace, find which hot-loop instructions are NOT hidden behind MFMA execution -- i.e. what is keeping cyc/mfma above the MFMA execute floor (16 for fp4, 32 for fp8 16x16x128). Models the matrix unit as a pipeline (next_free) for the EXPOSED %, and ranks blockers by the authoritative per-instruction STALL field (ops with stall==0 never block). Use when a GEMM/attention kernel is MFMA-bound and you want to push cyc/mfma toward the floor, or the user asks "what isn't hidden behind MFMA", "哪些指令没被 mfma 掩盖", "暴露的指令", or which non-MFMA op is the biggest exposed stall.
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

## Attribution: use the per-instruction STALL field, NOT gap geometry

Each ATT wave row is `[cycle, issue_dur, STALL, total_dur, code_id]`. **`r[2]` is
the authoritative stall** — the cycles that instruction actually blocked. Rank
blockers by summed stall per op (excluding the MFMAs' own execute stall, which is
the floor). Do NOT attribute by "which op sits in the idle gap" — every geometric
heuristic tried (first-in-gap / longest-in-gap / occupancy-overlap) mislabeled
non-blocking fill ops and gave contradictory answers on the SAME trace:

- gap-geometry variously blamed `ds_read_b128` 33%, then `s_add` 34%, then
  `buffer_load` 30% — all WRONG.
- the stall field showed `ds_read_b128` and `s_add` have **stall == 0** (pure fill,
  never block), and the real blockers were **s_barrier 39% + s_waitcnt(lgkmcnt) 35%
  + buffer_load 11% + readfirstlane 7% + s_waitcnt(vmcnt) 5%**.

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

Output: MFMA-covered %, EXPOSED %, cyc/mfma vs floor, and the exposed-gap cycles
bucketed by blocking instruction.

## Reading it

- **Rank by the stall column (`r[2]`), not by count or by which op sits in a gap.**
  An op with stall==0 is free no matter where it sits; only stall>0 ops block.
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
