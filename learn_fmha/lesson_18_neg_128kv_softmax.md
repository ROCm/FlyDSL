# Lesson 18 (NEGATIVE) — 128-kv-per-softmax: optimizing the wrong bottleneck

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_18_neg_128kv_softmax.py`
Live (sq16384): `8wave 46 → v5 45 TF` (neutral/slightly worse).

## The idea (plausible)
Online-softmax bookkeeping (running max/sum, the `corr` rescale of the o-accumulator, m/l updates)
runs once per kv-**tile**. Make the tile 4× bigger — one softmax over 128 kv instead of four over 32
— and you do that bookkeeping 4× less. Less VALU.

## Why it failed
VALU bookkeeping **wasn't the binding bottleneck**. Lesson 12's PMC: the kernel is LDS-bound
(LDS-wait ≈ 54%); the softmax VALU we amortized wasn't on the critical path. Worse, the coarse 128-kv
grouping loads all 4 subtiles before one softmax, which **reduced load/compute overlap** vs the
per-32 pipeline. Net: slight regression.

## The lesson
Amortizing an op only helps if that op is the bottleneck. **Classify first** (LDS-wait/busy,
VALU:MFMA), then attack the binding cost — not the one that's easiest to cut. This is the negative
mirror of Lesson 09/13 (a real but non-binding improvement = no speedup).

A/B kernels: `kernels/fmha_prefill_fp8_8wave.py` vs `kernels/fmha_prefill_fp8_v5.py`.
