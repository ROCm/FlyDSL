# SPDX-License-Identifier: Apache-2.0
"""Lesson 21 (NEGATIVE) — async global->LDS DMA + KT=128: matching CK's tile, ~2x slower.

A/B:  fmha_prefill_fp8_ck (best, KT=32, sync loads)
      vs  fmha_prefill_fp8_ck_async (KT=128 + buffer_load_to_lds async DMA).

THE IDEA: CK-Tile's compiled kernel (the production library, ~141 TF) uses kN0=128 (a
4x bigger KV outer tile) AND async global->LDS DMA (buffer_load_to_lds, which streams the
next K tile into LDS while MFMAs run). We found CK's exact config by decoding its JIT blob
filename: b128x128x32..._qr_async_vc. So: copy it — KT=128 + async DMA.

WHAT WE LEARNED (two things, one good, one humbling):
  GOOD: buffer_load_to_lds is NOT broken (an earlier belief). The real requirement is a
        WAVE-UNIFORM LDS destination pointer (the hw forces M0 = readfirstlane(ptr)); the
        old attempts passed a per-lane pointer. With a uniform pointer it works & is correct.
  HUMBLING: it's still ~2x SLOWER (32/35 vs 61/70 TF). Async frees only ~1 VGPR, so KT=128
        STILL can't fit the occupancy budget (a 128-kv tile inflates LDS + registers, and
        the occupancy loss dominates). CK makes KT=128 pay because its C++ codegen produces
        far tighter register usage than the FlyDSL 0.2.0 wheel does.

THE CEILING: the remaining gap to CK is register-allocation + instruction-scheduling
QUALITY, not algorithm or tile size. That's exactly what a DSL abstracts away and what hand
assembly (PyISA, ~240 TF) and CK's mature C++ codegen control. Copying the tile config
without the codegen quality doesn't transfer.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_21_neg_async_dma.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [16384, 32768]
    compare(["fmha_prefill_fp8_ck", "fmha_prefill_fp8_ck_async"], seqs=seqs)
    print("\nck(KT=32) -> ck_async(KT=128 + DMA): 61/70 -> 32/35 TF (~2x SLOWER).")
    print("buffer_load_to_lds works (needs a WAVE-UNIFORM ptr), but KT=128 still busts occupancy")
    print("(async frees ~1 VGPR). The gap to CK is codegen QUALITY (regalloc+scheduling), not tile")
    print("size — the part a DSL hides and hand/C++ codegen controls. Copying the config != the perf.")
