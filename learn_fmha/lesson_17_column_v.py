# SPDX-License-Identifier: Apache-2.0
"""Lesson 17 — Column-major V: delete the transpose by choosing the right layout (BIGGEST win).

A/B between two REAL kernels that differ by this one design choice:
  BEFORE: kernels/fmha_prefill_fp8_v7.py  (V row-major; GEMM2 needs a V/P transpose in LDS)
  AFTER:  kernels/fmha_prefill_fp8_ck.py  (V COLUMN-major "vec_k_col_v"; GEMM2 contraction
                                           is already contiguous -> NO transpose at all)

### The idea (the punchline of the whole tutorial)
Lessons 11-12 fought the P/V transpose with LDS round-trips and ds_bpermute. Lesson 20
(negative) shows v_perm can't win it either — the transpose WORK is irreducible if you
keep V row-major. The fix is not a cleverer transpose; it's to STORE V so the transpose
never exists: lay V out column-major (kv contiguous along the GEMM2 contraction). Then
GEMM2 reads contiguous kv directly. The KV-cache packer writes this layout once; the
kernel pays zero transpose forever.

### Why it's the biggest win (PMC, measured this session)
  ds_write_b8: 48 -> 0   (the V-transpose scatter is GONE)
  ds_bpermute:  N -> 0   (the P-transpose is GONE too)
  LDS-instructions: 111M -> 63M (-43%);  busy cycles: -18%
  -> +20% at sq16384 (51 -> 61 TF), +23% at sq32768 (57 -> 70 TF).
And it DISPROVES the assumption that you need gfx950 hardware transpose-loads (ds_read_tr)
to beat the transpose on gfx942 — the right data layout sidesteps it entirely.

NOTE: ck uses V_COL=True, so its V tensor must be packed column-major. The unified bench
(bench_fmha_compare.py) reads each kernel's `V_COL` flag and packs the matching pool, so
this comparison is apples-to-apples.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_17_column_v.py
"""

import sys

from _drive import compare

if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [1024, 2048, 16384, 32768]
    compare(["fmha_prefill_fp8_v7", "fmha_prefill_fp8_ck"], seqs=seqs, ck=True)
    print("\nMeasured (bs=1 nq8 nk1 causal, TFLOPS):  v7 -> ck(column-V)  [CK-Tile ref in last col]")
    print("  1024:  4.8 -> 5    2048: 15 -> 16    16384: 50.5 -> 61 (+20%)    32768: 57 -> 70 (+23%)")
    print("  PMC: ds_write_b8 48->0, ds_bpermute->0, LDS-insts -43%, busy -18%.")
    print("=> LAYOUT beats transpose tricks. The biggest single win. Best FlyDSL kernel = ck.")
