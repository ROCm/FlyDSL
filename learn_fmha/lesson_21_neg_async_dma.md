# Lesson 21 (NEGATIVE) — async DMA + KT=128: copying CK's config ≠ CK's speed

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_21_neg_async_dma.py`
Live (sq16384): `ck 61 → ck_async 32 TF` (~2× slower).

## The idea
CK-Tile (the production library, ~141 TF) uses `kN0=128` (a 4× bigger KV outer tile) **and** async
global→LDS DMA (`buffer_load_to_lds`, streaming the next K tile into LDS while MFMAs run). We even
recovered CK's exact config by decoding its JIT blob filename:
`...b128x128x32..._qr_async_vc...`. So: copy it — `KT=128` + async DMA.

## What we learned — one good, one humbling
- **Good:** `buffer_load_to_lds` is **not** broken (an earlier belief). The real requirement is a
  **wave-uniform LDS destination pointer** — the hardware forces `M0 = readfirstlane(ptr)`, so a
  per-lane pointer corrupts it. With a uniform pointer it works and is correct.
- **Humbling:** it's still **~2× slower** (32/35 vs 61/70 TF). Async frees only ~1 VGPR, so `KT=128`
  *still* can't fit the occupancy budget (a 128-kv tile inflates LDS + registers; the occupancy loss
  dominates). CK makes `KT=128` pay because its mature **C++ codegen** produces far tighter register
  usage than the FlyDSL 0.2.0 wheel.

## The ceiling (the tutorial's closing truth)
The remaining gap to CK is **register-allocation + instruction-scheduling quality**, not algorithm or
tile size. That is exactly what a DSL abstracts away, and what hand assembly (PyISA ~240 TF) and CK's
C++ codegen control. **Copying the tile config without the codegen quality doesn't transfer.**

A/B kernels: `kernels/fmha_prefill_fp8_ck.py` vs `kernels/fmha_prefill_fp8_ck_async.py`.

## Next
Lesson 22: the capstone scoreboard — where all of this lands vs CK-Tile and PyISA, and where the
wall is.
