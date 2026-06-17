# SPDX-License-Identifier: Apache-2.0
"""Lesson 19 (NEGATIVE) — 2-rep software pipeline: VGPR blowup kills occupancy.

A/B:  fmha_prefill_fp8_8wave (baseline)  vs  fmha_prefill_fp8_v6 (2 kv-tiles in flight).

THE IDEA: process two kv-tiles per loop iteration with the second tile's loads issued
during the first tile's MFMAs — a deeper software pipeline (this is what hand-written
PyISA assembly does to reach ~240 TF). More work in flight -> hide more latency.

WHY IT FAILED (regressed ~46 -> ~44 TF): two reasons, both visible in the profile.
  1. VGPR 164 -> 215. Holding two tiles' K/V packs + two next-tile load sets live at once
     blew the register budget. Occupancy = floor(512 / vgpr) dropped 3 -> 2 waves/SIMD.
     The occupancy loss outweighed the extra overlap.
  2. The two reps SERIALIZE on the online-softmax dependency (rep1's running max/o depends
     on rep0's output). FlyDSL's compiler will NOT reorder rep1's independent K-reads ahead
     across that dependency, so the intended overlap never materialized.

WHY PYISA GETS AWAY WITH IT: hand assembly PINS registers and HAND-SCHEDULES the
independent loads early — two things a DSL abstracts away from you. So this is a
DSL-vs-hand-scheduling ceiling, not an algorithm bug.

Contrast Lesson 16 (diagonal-pair): that also "does two tiles," but SEQUENTIALLY reusing
registers -> VGPR 164->168, no occupancy loss -> it WINS. Sequential reuse vs concurrent
in-flight is the whole difference.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_19_neg_2rep_pipeline.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [16384, 32768]
    compare(["fmha_prefill_fp8_8wave", "fmha_prefill_fp8_v6"], seqs=seqs)
    print("\nbaseline -> v6(2-rep pipeline): ~46/53 -> ~44/49 TF (REGRESSED). VGPR 164->215 ->")
    print("occupancy 3->2 waves/SIMD; compiler won't interleave across the softmax dependency.")
    print("Diagonal-pair (L16) does 2 tiles SEQUENTIALLY (VGPR 164->168) and WINS. Sequential")
    print("reuse beats concurrent-in-flight when you don't control register allocation.")
