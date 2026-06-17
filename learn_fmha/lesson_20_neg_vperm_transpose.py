# SPDX-License-Identifier: Apache-2.0
"""Lesson 20 (NEGATIVE) — v_perm wide V-transpose: crush one counter, lose overall.

A/B:  fmha_prefill_fp8_8wave (baseline, 16 ds_write_b8 scatter per V tile)
      vs  fmha_prefill_fp8_v14 (v_perm register transpose -> 4 WIDE stores).

THE IDEA: Lesson 11 builds the V transpose with 48 narrow ds_write_b8 (the dominant LDS
op, per Lesson 12's PMC). Replace them with an in-register 4x4 byte transpose using
v_perm_b32, then a few WIDE ds_write_b32. Fewer, wider LDS writes -> less LDS pressure.

WHAT THE PROFILE SAID: it WORKED on the targeted counter and STILL regressed.
  ds_write_b8: 48 -> 0     (goal achieved!)
  LDS-wait/busy: ~102% -> ~27%   (LDS pressure crushed!)
  ...but busy CYCLES went UP, and TFLOPS dropped ~46 -> ~44.
WHY: the v_perm transpose added ~48 VALU ops/tile and pushed VGPR 164 -> 179 (occupancy
down). The transpose WORK is irreducible — you pay it either on the DS unit (ds_write_b8)
OR on the VALU (v_perm). Trading DS for VALU just moved the bottleneck onto a unit that,
here, costs more. PyISA does the v_perm path and still wins ONLY via hand-scheduling +
pinned registers (same ceiling as Lesson 19).

THE PUNCHLINE: when a cost is irreducible in the current data layout, no amount of
shuffling between hardware units removes it. The real fix is Lesson 17 — change the V
LAYOUT (column-major) so the transpose never exists. Lessons 11, 12, 20 are three failed
ways to move the same cost; 17 deletes it.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_20_neg_vperm_transpose.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [16384, 32768]
    compare(["fmha_prefill_fp8_8wave", "fmha_prefill_fp8_v14"], seqs=seqs)
    print("\nbaseline -> v14(v_perm transpose): ds_write_b8 48->0, LDS-wait 102%->27%, yet ~46->~44")
    print("TF (REGRESSED): +48 v_perm VALU/tile, VGPR 164->179, busy cycles UP. The transpose WORK")
    print("is irreducible (pay on DS OR VALU). Real fix = change the LAYOUT (column-V, Lesson 17).")
