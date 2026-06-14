# SPDX-License-Identifier: Apache-2.0
"""Lesson 08 — From 1 wave to a multi-wave workgroup: occupancy & CU-starvation.

This is the first STRUCTURAL performance lesson, and the first that needs a real grid
(many workgroups) to mean anything. Lessons 00-07 used ONE wavefront on ONE of 80 CUs —
fine for learning layout/correctness, useless for throughput.

Rather than re-derive a multi-wave attention kernel from scratch (that IS the production
kernels/fmha_prefill_fp8_8wave.py, already validated), this lesson DRIVES that kernel at
three workgroup sizes via the FMHA_NWAVES env var and reads the result:
    NWAVES=8 -> BM=256 (8 waves x 32 q-rows)   <- the "obvious" big tile
    NWAVES=4 -> BM=128 (4 waves x 32 q-rows)   <- the measured sweet spot
    NWAVES=2 -> BM=64
The A/B "diff" is literally one number (waves per workgroup -> tile height BM).

WHAT TO LOOK FOR (the profiling habit):
  - grid = ceil(seqlen / BM) * nheads.  With 80 CUs, you want grid >> 80 to fill the
    machine. Smaller BM -> more workgroups -> better occupancy at small seqlen.
  - BUT a too-small wave count also shrinks per-wg work and can hurt at large seqlen.
  - VGPR limits occupancy: 512 VGPR-banks/SIMD, so waves/SIMD = floor(512 / vgpr_count).

This script just prints the bench across NWAVES and seqlens (it shells out to the unified
benchmark, one NWAVES per subprocess because BM is a compile-time constant baked per module
load). See the .md for the measured numbers and the occupancy explanation.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_08_multiwave.py
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "tests", "kernels", "bench_fmha_compare.py")


def run(nwaves, seqs):
    env = dict(os.environ, HIP_VISIBLE_DEVICES="2", FMHA_NWAVES=str(nwaves))
    cmd = [sys.executable, BENCH, "--kernels", "fmha_prefill_fp8_8wave", "--no-pyisa",
           "--seqs", *map(str, seqs)]
    print(f"\n===== FMHA_NWAVES={nwaves}  (BM = {nwaves*32}) =====")
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [1024, 16384]
    for nw in (8, 4, 2):
        run(nw, seqs)
    print("\nMeasured this session (bs=1 nq8 nk1 causal, TFLOPS):")
    print("  seq    NWAVES=8/BM256   NWAVES=4/BM128   NWAVES=2/BM64")
    print("  1024        4.3              5.0              4.8")
    print("  4096       18.8             23.2             22.3")
    print("  16384      39.7             46.6             38.3")
    print("  32768      ~45              52.8             41.9")
    print("=> BM=128 / 4-wave is the UNIFORM sweet spot (+17-23%). See lesson_08_multiwave.md.")
