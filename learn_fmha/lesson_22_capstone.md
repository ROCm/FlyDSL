# Lesson 22 (CAPSTONE) — the scoreboard, and where the wall is

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_22_capstone.py`
Live (sq16384): `8wave 46 → v7 51 → ck 61 TF`.

## The full scoreboard (bs=1, nq8, nk1, causal, TFLOPS)
| seq | ours 8wave | ours v7 | **ours ck** | CK-Tile | PyISA asm |
|---|---|---|---|---|---|
| 1024 | 5 | 5 | 5 | 30 | 36 |
| 2048 | 12 | 15 | 16 | 62 | 83 |
| 16384 | 46 | 50 | **61** | 141 | **238** |
| 32768 | 53 | 57 | **70** | 146 | **291** |

(PyISA hand-assembly is timed via `tests/fmha/src/bench/fmha_bench.cpp`, which runs the `.co` with
no CPU reference — the only way to time it at long seq.)

## What we won, and how
From the bf16 single-tile kernel (Lessons 00–06) to fp8 (07) to **70 TF** (17), the wins came from
**algorithm and layout**, all expressible in the DSL:
- **BM=128 / 4-wave** sweet spot (Lesson 08) — occupancy.
- **Diagonal-pair tiling** (Lesson 16) — balance the causal triangle, +8–24%.
- **Column-major V** (Lesson 17) — *delete* the transpose, +20%. The single biggest win.

And we learned just as much from what *didn't* work (Lessons 18–21): amortizing a non-bottleneck,
register-blowup from concurrent tiles, shuffling an irreducible transpose between units, and copying
a tile config without the codegen to back it.

## Where the wall is (the closing lesson)
The remaining **~2× to CK-Tile** and **~4× to PyISA** is **instruction-scheduling + register-
allocation quality** — keeping 2+ tiles in flight without VGPR blowup (Lesson 19), hand-placing
`waitcnt`s, pinning registers, making `KT=128` fit occupancy (Lesson 21). A DSL **abstracts those
away**; hand assembly (PyISA) and mature C++ codegen (CK) control them. This is a **structural
ceiling for this DSL + 0.2.0 wheel**, not an algorithm gap you can close with one more trick.

## The transferable method (what this whole tutorial taught)
1. **Build correct first**, verifying each layer against a reference (never trust a layout comment).
2. **Classify the bottleneck** with PMC before optimizing: LDS-wait/busy, VALU:MFMA, GB/s vs peak,
   grid vs CUs. (Lessons 09, 12, 13, 14, 18 all hinge on this.)
3. **Change one thing, re-measure, read the *binding* counter** — not the one you hoped to move.
4. **Prefer algorithm + layout wins** (large, expressible) over micro-ops; when a cost is irreducible
   in the current layout, **change the layout** (Lesson 17), don't shuffle the cost between units
   (Lessons 11/12/20).
5. **Know when you've hit the codegen ceiling** and stop grinding (Lessons 19, 21).

## Going further
- The optimization toolkit/runbook: `mlse-tools-internal/performance/kernel_optimization/`.
- The production kernels you've been reading: `kernels/fmha_prefill_fp8*.py`.
- The unified bench: `tests/kernels/bench_fmha_compare.py` (add `--ck` for CK-Tile).
- To chase the ceiling itself, you'd need external-LLVM codegen control or hand assembly — out of
  scope for DSL-level tuning, but now you know exactly *why*.
