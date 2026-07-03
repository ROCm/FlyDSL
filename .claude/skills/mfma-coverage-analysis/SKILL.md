---
name: mfma-coverage-analysis
description: From an ATT trace, find which hot-loop instructions are NOT hidden behind MFMA execution -- i.e. what is keeping cyc/mfma above the MFMA execute floor (16 for fp4, 32 for fp8 16x16x128). Models the matrix unit as a pipeline (next_free) for the EXPOSED cycles, then tiles those idle cycles by OCCUPYING instruction (issue_dur + stall, intersected with the idle gaps) so the ranking says which non-MFMA ops steal issue bandwidth between MFMAs. Reports %exp and %all. Use when a GEMM/attention kernel is MFMA-bound and you want to push cyc/mfma toward the floor, or the user asks "what isn't hidden behind MFMA", "哪些指令没被 mfma 掩盖", "暴露的指令", or which non-MFMA op is the biggest exposed stall.
---

# MFMA Coverage Analysis

## The model

**Step 1 — EXPOSED via next_free (matrix-unit pipeline).** Each MFMA occupies ONE
EXEC-cycle execute slot, but consecutive MFMAs can **issue < EXEC apart** and still
both be hidden (dense fp4 MFMAs issue ~8 cyc apart yet each execute-slot is 16).
Track `next_free` = the cycle the unit becomes free:
- MFMA issues at `t <= next_free` → hidden; `next_free += EXEC`
- MFMA issues at `t >  next_free` → unit IDLE for `t - next_free` → **EXPOSED**

EXPOSED = sum of idle gaps; shrink toward 0 so cyc/mfma → EXEC (see
[[feedback-mfma-bound-reduce-scalar-toward-cyc16]]). (This replaces an older
union-of-`[issue,issue+EXEC)` model that over-reported exposure.)

**Step 2 — attribute EXPOSED by OCCUPANCY.** For every idle cycle, credit it to
whichever instruction was on the issue port then. Each non-MFMA instruction occupies
`[its issue, next instruction's issue)` = its issue_dur PLUS any stall (r[2]) — we do
NOT split issue vs stall, and do NOT ask "who blocks". The question is simply: while
the matrix unit sat idle, which non-MFMA ops were stealing issue bandwidth? Intersect
each op's occupancy with the idle gaps and sum. This tiles the whole EXPOSED window
(every idle cycle has an owner), so the ranking directly says which ops must be
REMOVED / SHRUNK from between the MFMAs to push cyc/mfma toward the floor.

Two percent columns: **%exp** (of the EXPOSED cycles) and **%all** (of the whole
window incl. MFMA — the real wall-clock lever, since MFMA execute is ~85%+).

Example (fp4_gemm_4wave, cyc/mfma 18.73, EXPOSED 6344 = 14.4% of全局):
`buffer_load_dwordx4 30%exp/4.3%all · v_readfirstlane 17%/2.5% · s_barrier 16%/2.3% ·
s_waitcnt(lgkmcnt) 8%/1.2% · s_add_i32 6%/0.8% · ds_read_b128 6%/0.8% · …`.
Read it as: the exposed time is scale/g2s **gather + readfirstlane + address
arithmetic** stealing issue slots between MFMAs; cut their count/occupancy.
Row schema: `[cycle, issue_dur, stall, total_dur, code_id]`; s_waitcnt is split
vmcnt / lgkmcnt so you see whether VMEM or LDS waits dominate.

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

Output: MFMA-covered %, EXPOSED % (and % of全局), cyc/mfma vs floor, then the exposed
cycles tiled by OCCUPYING instruction with two columns (%exp, %all).

## Reading it

- **Attack the top occupancy ops** = the non-MFMA instructions eating the most idle
  cycles between MFMAs. Reduce their COUNT or per-op occupancy so more MFMAs pack in:
  fewer/cheaper address ops, precompute wave-uniform values once (kill readfirstlane),
  merge/spread loads, deepen prefetch so a value is on the port earlier.
- **buffer_load / ds_read high** → too many gathers/reads issue between MFMAs; merge
  (wider load), move to a different MFMA's shadow, or cut the count.
- **v_readfirstlane high** → v→s serialization; precompute the wave-uniform SGPR once
  outside the loop instead of per-use.
- **s_add / s_and / v_cmp / s_lshl** (address & loop arithmetic) → recompute less;
  hoist loop-invariants, use s_add increment chains, fold offsets into instruction
  immediates.
- **s_barrier high** → per-iter cross-wave sync (4 waves finish the iter at slightly
  different times). Structural for a fixed wave count; not cuttable without changing
  sync granularity / wave layout.
- **s_waitcnt(lgkmcnt/vmcnt)** → LDS/VMEM waits; deepen the corresponding prefetch or
  relax the count. Usually small once loads are hidden.

## Caveats

- Uses one wave file (se0_sm0_sl0_wv0.json by default). Different waves are
  equivalent repeats; pass `--wave` to check another.
- Occupancy tiles the EXPOSED window with no gaps (every idle cycle has an owner), so
  %exp sums to ~100. It counts issue_dur + stall together on purpose — the goal is
  "what steals issue bandwidth between MFMAs", not "who stalls".
- Row schema: `[cycle, issue_dur, stall, total_dur, code_id]`;
  `code.json["code"][cid][0]` is the asm text. (See att-hotloop-benchmark for the
  complementary barrier-window cyc/mfma summary.)
