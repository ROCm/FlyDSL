# Lesson 02 — A real GEMM tile (K-loop) + first throughput measurement

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_02_gemm_tiles.py` → `PASS` + a timing line.

## What we build
`C[16,16] = A[16,64] @ B[16,64]ᵀ` by chaining **4** `16x16x16` MFMAs through the accumulator
(K=64 = 4×16). Then a first `do_bench` throughput print.

## The K-loop = accumulator chaining
The MFMA's `c` operand is the running accumulator. Start at zero, then for each 16-wide K-slice feed
last step's result back in:
```
acc = 0
for ks in 0..3:        # range_constexpr -> fully unrolled (K is compile-time)
    acc = mfma(A_k, B_k, acc)
```
The C fragment stays in **registers** the whole time — no LDS, no round-trips. (CK Tile: this is the
`k_loops` inside `BlockGemm`; same accumulator-in-register idea.) The per-lane A/B indexing is
identical to Lesson 01, just shifted by the K-step base `k0 = ks*16`.

## Wide loads
Each lane needs 4 contiguous-K bf16 per step. 4×bf16 = 8 bytes → one `vec_width=4` load instead of
4 scalar loads. This is the cheapest "wide load" form. (Lesson 09 measures whether wide loads
actually help a given kernel — spoiler: for compute-bound attention they're neutral, which is itself
a useful result.)

## `range_constexpr` vs runtime `range`
- `fx.range_constexpr(N)` — N known at compile time → the loop is **unrolled** (4 MFMAs emitted
  back-to-back, accumulator chained in registers). Use for K-steps, head-dim tiles, etc.
- `range(start, stop, step, init=[...])` + `yield` — a real `scf.for` with loop-carried state, for
  **runtime** bounds like the sequence-length loop (Lesson 06). You must thread the carried values
  (running max/sum/output) through `init=` and `yield`.

## The first profiling lesson: at tiny sizes you measure the *launch*, not the math
`do_bench(fn, warmup=20, rep=100)` returns the **median** time in ms (warmup iters discarded — they
include JIT compile + cold cache). Here it reports ~120 µs for a 16×16×64 tile = **0.28 MFLOP/s**,
absurdly low. That's not a bug: a 16×16 tile is **one wavefront on one of 80 CUs**, so the ~120 µs
is essentially kernel-launch latency. **Takeaway:** throughput numbers are only meaningful once the
kernel fills the machine (real tile sizes + a grid of hundreds of workgroups). We start trusting
TFLOPS from Lesson 08 onward.

## `do_bench` cheat-sheet (used in every later lesson)
```python
from flydsl.autotune import do_bench
ms = do_bench(lambda: run_kernel(...), warmup=10, rep=50)   # returns median ms
tflops = flop / 1e9 / ms                                     # flop = 2*M*N*K (GEMM)
```

## Next
Lesson 03: GEMM1 of attention — `S = K @ Qᵀ`, the scores matrix, laid out `[kv, q]`.
