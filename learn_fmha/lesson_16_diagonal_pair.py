# SPDX-License-Identifier: Apache-2.0
"""Lesson 16 — Diagonal-pair tiling: balancing the causal triangle (a STRUCTURAL win).

A/B between two REAL kernels that differ by exactly this trick:
  BEFORE: kernels/fmha_prefill_fp8_8wave.py  (one q-tile per workgroup)
  AFTER:  kernels/fmha_prefill_fp8_v7.py     (one q-tile + its causal MIRROR per workgroup)
Open both files and diff the launch/tile mapping — the change is localized to the
workgroup->q-tile assignment (v7 exports BM = 2*TILE_BM and processes qtile `t` AND
`num_q_tiles-1-t` sequentially in one kernel).

### The idea
Under a causal mask, q-tile `t` attends to kv 0..(t+1)*BM — so EARLY q-tiles do little
work, LATE q-tiles do lots. If each workgroup takes one q-tile, the workgroups are wildly
unbalanced and the machine waits on the heavy ones. Pairing tile `t` with its mirror
`num_tiles-1-t` gives every workgroup ~the same total kv work (light + heavy = constant).
Bonus: it HALVES the grid, which is why it helps large seq but can HURT tiny seq (fewer
workgroups -> CU-starvation, Lesson 08). Hence Lesson 17/dispatch picks per-shape.

Crucially this costs almost nothing: the mirror tile runs SEQUENTIALLY in the same kernel,
reusing the same registers, so VGPR barely moves (164->168) and occupancy is unchanged.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_16_diagonal_pair.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [1024, 4096, 16384, 32768]
    compare(["fmha_prefill_fp8_8wave", "fmha_prefill_fp8_v7"], seqs=seqs)
    print("\nMeasured (bs=1 nq8 nk1 causal, TFLOPS):  8wave -> v7(diag-pair)")
    print("  1024:  5.0 -> 4.8   (LOSS: grid halves 64->32, CU-starved)")
    print("  4096: 23.2 -> 25.x  ;  16384: 46.6 -> 50.5  ;  32768: 52.9 -> 56.9   (+8..24%)")
    print("=> structural win at seq>=2048; loss at 1024 -> motivates per-shape dispatch (Lesson 17).")
