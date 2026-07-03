---
name: mfma-coverage-analysis
description: From an ATT trace, find which hot-loop instructions are NOT hidden behind MFMA execution -- i.e. what is keeping cyc/mfma above the MFMA execute floor (16 for fp4, 32 for fp8 16x16x128). Models the matrix unit as a pipeline (next_free): an MFMA issuing after the unit is free exposes the idle gap, attributed to the first non-MFMA op in it. Use when a GEMM/attention kernel is MFMA-bound and you want to push cyc/mfma toward the floor, or the user asks "what isn't hidden behind MFMA", "哪些指令没被 mfma 掩盖", "暴露的指令", or which non-MFMA op is the biggest exposed stall.
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

IMPORTANT — this REPLACES the older "union of `[issue, issue+EXEC)` windows" model,
which **capped overlapping windows and so mislabeled shadow-hidden loads as exposed**
(e.g. it blamed `ds_read_b128` for 33% when dense 8-cyc-apart MFMAs actually hid all
of them; the next_free model correctly showed ds_read ~0% and the real blockers were
`s_add` address arithmetic 34% + idle/s_waitcnt). If a load appears as a top exposed
blocker, sanity-check it isn't just filling the natural MFMA issue cadence.

Attribute each exposed gap to the **first non-MFMA instruction in it** = what
blocked the next MFMA from issuing on time. That ranks the real targets by cycles,
which is very different from ranking by instruction *count* or by the generic ATT
stall-type breakdown (a scalar op that issues alone for 1 cycle and an op that
gates a 50-cycle VMEM wait look identical by count but not by exposed cycles).

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

- **A named instruction with high exposed cycles is the real target**, even if it
  is a tiny fraction of the instruction count and not in the generic stall top-N.
  Example (fp4_gemm_4wave, m0-incr commit, cyc/mfma 20.25, exposed 23%): the
  biggest blocker was `buffer_load_dword` (scale) at 30% of exposed cycles — only
  3.3% of instructions and absent from the stall-type top-25, yet the #1 thing
  keeping cyc/mfma above 16. This reversed an earlier "scale load isn't worth it"
  call that was based on the wrong (count / stall-type) metric.
- **`idle` gaps** = pure latency stalls (waitcnt drain / dependency) with no issuing
  instruction — usually not directly cuttable, attack the named buckets first.
- `s_waitcnt` high → the wait itself gates; consider relaxing vmcnt/lgkmcnt or
  moving the producing load earlier (deeper prefetch).

## Caveats

- Uses one wave file (se0_sm0_sl0_wv0.json by default). Different waves are
  equivalent repeats; pass `--wave` to check another.
- The "first non-MFMA in gap" heuristic attributes the whole gap to one op; a gap
  containing a producer + its consumer is charged to the producer. Good enough to
  rank targets, not exact per-op liveness.
- Data structures: instruction record = `[cycle, dur, _, _, code_id]`; code_id ->
  `code.json["code"][cid][0]` is the asm text. (See att-hotloop-benchmark for the
  complementary barrier-window cyc/mfma summary.)
