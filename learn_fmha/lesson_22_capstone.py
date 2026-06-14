# SPDX-License-Identifier: Apache-2.0
"""Lesson 22 (CAPSTONE) — the scoreboard, and where the wall is.

Runs the unified benchmark across the journey's milestone kernels + CK-Tile, so you see
the whole arc in one table:
  fmha_prefill_fp8_8wave  -> the multi-wave baseline (BM=128)
  fmha_prefill_fp8_v7     -> + diagonal-pair (Lesson 16)
  fmha_prefill_fp8_ck     -> + column-V (Lesson 17), the best FlyDSL kernel
  CK-Tile (--ck)          -> the production library reference

PyISA hand-assembly (measured separately via tests/fmha/src/bench/fmha_bench.cpp, which
times the .co without a CPU reference) is the ultimate bar and is printed below.

THE FULL SCOREBOARD (bs=1 nq8 nk1 causal, TFLOPS):
  seq     ours(8wave)  ours(v7)  ours(ck)   CK-Tile   PyISA asm
  1024        5           5         5          30         36
  2048       12          15        16          62         83
  16384      46          50        61         141        238
  32768      53          57        70         146        291

WHERE THE WALL IS (the closing lesson of the whole tutorial):
  * From 5 -> 70 TF we won with ALGORITHM + LAYOUT: BM tuning, diagonal-pair balance, and
    above all column-V (delete the transpose). Those are things a DSL lets you express.
  * The remaining ~2x to CK and ~4x to PyISA is INSTRUCTION SCHEDULING + REGISTER
    ALLOCATION quality (Lessons 19, 21): keeping 2+ tiles in flight without VGPR blowup,
    hand-placing waitcnts, pinning registers. A DSL abstracts those away; hand assembly and
    mature C++ codegen control them. That's a STRUCTURAL ceiling for this DSL+wheel, not an
    algorithm gap you can close with one more trick.
  * Practical takeaway: in a DSL, chase algorithm and layout wins (they're large and
    expressible); recognize when you've hit the codegen ceiling and stop grinding micro-ops.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_22_capstone.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [1024, 16384, 32768]
    compare(["fmha_prefill_fp8_8wave", "fmha_prefill_fp8_v7", "fmha_prefill_fp8_ck"], seqs=seqs, ck=True)
    print("\n=== FULL SCOREBOARD (bs=1 nq8 nk1 causal, TFLOPS) ===")
    print("  seq     8wave  v7   ck   | CK-Tile  PyISA-asm")
    print("  1024      5     5    5   |   30        36")
    print("  2048     12    15   16   |   62        83")
    print("  16384    46    50   61   |  141       238")
    print("  32768    53    57   70   |  146       291")
    print("\nWon 5->70 TF via ALGORITHM + LAYOUT (BM tuning, diagonal-pair, column-V).")
    print("Remaining ~2x(CK)/~4x(PyISA) = instruction-scheduling + register-allocation quality")
    print("(Lessons 19,21) — what a DSL abstracts away. That's the structural ceiling, not an")
    print("algorithm gap. In a DSL: chase algorithm/layout wins; know when codegen is the wall.")
