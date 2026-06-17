# Lesson 19 (NEGATIVE) — 2-rep software pipeline: VGPR blowup kills occupancy

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_19_neg_2rep_pipeline.py`
Live (sq16384/32768): `8wave 46/53 → v6 44/49 TF` (regressed).

## The idea
Process two kv-tiles per loop iteration, issuing the second tile's loads during the first tile's
MFMAs — a deeper software pipeline. This is what hand-written PyISA assembly does to reach ~240 TF.

## Why it failed (read the profile)
1. **VGPR 164 → 215.** Two tiles' K/V packs + two next-tile load sets live at once blew the register
   budget. Occupancy = `⌊512/vgpr⌋` dropped **3 → 2 waves/SIMD**; the occupancy loss outweighed the
   extra overlap.
2. **The two reps serialize** on the online-softmax dependency (rep1's running max/o depends on
   rep0's output). FlyDSL's compiler won't hoist rep1's *independent* K-reads across that dependency,
   so the intended overlap never materialized.

## Why PyISA gets away with it (the ceiling)
Hand assembly **pins registers** and **hand-schedules** the independent loads early — two things a
DSL abstracts away. So this is a **DSL-vs-hand-scheduling ceiling**, not an algorithm bug.

## The sharp contrast with Lesson 16
Diagonal-pair *also* "does two tiles," but **sequentially**, reusing the same registers → VGPR
164→168, occupancy unchanged → it **wins**. v6 runs two tiles **concurrently in flight** → VGPR
blowup → it **loses**. **Sequential reuse beats concurrent-in-flight when you don't control register
allocation.** That single distinction is the most important practical takeaway in Part E.

A/B kernels: `kernels/fmha_prefill_fp8_8wave.py` vs `kernels/fmha_prefill_fp8_v6.py`.
