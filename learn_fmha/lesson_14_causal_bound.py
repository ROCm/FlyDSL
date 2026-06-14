# SPDX-License-Identifier: Apache-2.0
"""Lesson 14 — Skip fully-masked kv tiles + fold the mask into one bound.

Explainer. Under a causal mask, a q-tile at rows [q0, q0+BM) can only attend to kv up to
q0+BM-1+(sk-sq). Every kv-tile BEYOND that is 100% masked -> computing it is pure waste.
The trick (in fmha_prefill_fp8_8wave.py): cap the kv loop count at runtime,
    n_kv = min(full_tiles, ceil((q_max + (sk-sq) + 1) / BN))
so early q-tiles iterate far fewer kv-tiles. Also fold the two per-element mask compares
(kv < sk AND kv <= causal_bound) into ONE compare against a loop-invariant `eff_bound`,
hoisted out of the kv loop (halves the mask VALU).

### The subtle lesson: cutting WORK != cutting TIME on an under-occupied machine
PMC showed this cut total work ~45% (busy cycles 2805M -> 1547M) yet wall-clock barely
moved at the benchmark's small grid — because at grid=64 on 80 CUs the GPU is UNDER-
utilized, so doing less work per workgroup just leaves CUs idle sooner, it doesn't shorten
the critical path. The bound helps most when the grid is large (many workgroups competing
for CUs) — i.e. it compounds with diagonal-pair (Lesson 16), which fills the machine.

No new kernel: read the causal-bound + eff_bound code in fmha_prefill_fp8_8wave.py. This
script just prints the algebra so you can see how many kv-tiles each q-tile skips.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_14_causal_bound.py
"""

import math


def kv_tiles(qtile, BM, BN, sk, sq, full):
    q_max = qtile * BM + BM - 1
    kv_max = q_max + (sk - sq)
    bound = min(full, (kv_max + BN) // BN)
    return bound


if __name__ == "__main__":
    BM, BN, sk, sq = 128, 32, 4096, 4096
    full = math.ceil(sk / BN)
    print(f"causal, sq=sk={sq}, BM={BM}, BN={BN}, full_kv_tiles={full}")
    print("q-tile :  kv-tiles computed  (vs full = wasted work avoided)")
    total_capped = 0
    for t in range(0, sq // BM, max(1, sq // BM // 8)):
        n = kv_tiles(t, BM, BN, sk, sq, full)
        total_capped += n
        print(f"  {t:4d}  :  {n:4d}        (skipped {full - n})")
    print(f"\nEarly q-tiles skip most kv-tiles. Summed over all q-tiles the causal bound roughly")
    print("HALVES total kv work. BUT (PMC): at small grid that ~45% work cut barely moved wall")
    print("time — an under-occupied machine just idles sooner. It compounds with diagonal-pair")
    print("(L16), which fills the grid. Lesson: 'less work' only = 'less time' when CUs are busy.")
