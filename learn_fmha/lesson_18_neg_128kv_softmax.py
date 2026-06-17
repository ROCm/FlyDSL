# SPDX-License-Identifier: Apache-2.0
"""Lesson 18 (NEGATIVE) — 128-kv-per-softmax: optimizing the wrong bottleneck.

A/B:  fmha_prefill_fp8_8wave (baseline)  vs  fmha_prefill_fp8_v5 (128-kv/softmax).

THE IDEA (sounded good): the online-softmax bookkeeping (running max/sum, the corr rescale
of the o accumulator, the m/l updates) runs once per kv-TILE. If we make the tile bigger —
one softmax over 128 kv instead of four softmaxes over 32 — we do that bookkeeping 4x less.
Less VALU, right?

WHY IT FAILED (neutral / slightly slower): VALU bookkeeping was NOT the binding bottleneck.
The PMC (Lesson 12) says the kernel is LDS-bound (LDS-wait ~54%); the softmax VALU we
amortized wasn't on the critical path. Worse, the coarser 128-kv grouping loads all 4
subtiles before one softmax, which REDUCED load/compute overlap vs the per-32 pipeline.
Net: ~44 vs 46 TF @16384 — a small regression.

LESSON: amortizing an op only helps if that op is the bottleneck. Always classify FIRST
(Lesson 12's two ratios), then optimize the binding cost — not the one that's easy to cut.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_18_neg_128kv_softmax.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [4096, 16384]
    compare(["fmha_prefill_fp8_8wave", "fmha_prefill_fp8_v5"], seqs=seqs)
    print("\nbaseline -> v5(128-kv/softmax): ~46 -> ~44 TF @16384 (NEUTRAL/REGRESSED).")
    print("VALU wasn't the wall (it's LDS-bound). Amortizing a non-bottleneck op does nothing,")
    print("and the coarse grouping hurt overlap. Classify the bottleneck before optimizing it.")
