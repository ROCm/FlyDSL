# Lesson 22c — Multi-head: fixing the grid

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_22c_multihead.py` → PASS across shapes +
a benched perf shape. Single shape + bench:
`python3 learn_fmha/lesson_22c_multihead.py 8 16384 16384 1 bench`.

## What changed vs 22b (and why)
22b was **single-head**, so its grid was only `ceil(sq/BM)` workgroups — at sq1024 that's **8
workgroups on 80 CUs**, badly CU-starved (Lesson 08), giving ~1 TF. Real attention has `nq` heads,
and **each (head, q-tile) is an independent workgroup**. The fix is one idea:
```
grid = nq * ceil(sq / BM)
```
In the kernel: decode `(head, qtile)` from `block_idx`, then offset every tensor base by the head.
Tensors are laid out `[head, seq, hd]` (V column-major `[head, hd, kv]`). Everything else —
multiwave, column-V, register-P, causal-bound, fast-exp2 — is **identical to 22b**.

## Measured (nq=8, causal, fp8; honest wall-clock)
| seq | 22b (single-head) | **22c (multi-head)** | production ck |
|---|---|---|---|
| 1024 | ~1 | ~7 | 5 |
| 16384 | ~31 | **~38** | 61 |
| 32768 | ~50 | **~38** | 70 |

The grid fix lifts the small/mid shapes a lot (grid no longer starved). 22c lands at ~38 TF —
below production ck (61/70) by the deliberately-omitted tricks (diagonal-pair, ping-pong, production
tiling), which is the expected price of a readable teaching kernel.

## A profiling gotcha worth keeping
`flydsl.autotune.do_bench` reported absurd numbers here (86 TF, then 340 TF at sq32768 — physically
impossible, *faster* per-FLOP at 4× the work). It mis-times this kernel at large seq. The fix: a
plain warmup + `time.perf_counter()` loop with `torch.cuda.synchronize()`. **Lesson: sanity-check
any benchmark against the physics** — if doubling the work doesn't roughly scale the time, the timer
is lying, not the kernel. (Causal sq32768 has ~4× the FLOPs of sq16384, so its time must be ~4×.)

## When to use 22b vs 22c
- **22b** — the cleanest read of the *kernel body* (all the tricks, no head bookkeeping).
- **22c** — the same kernel with the *grid* right, so the TFLOPS are meaningful and comparable to the
  production column. The only delta is head-decode + per-head pointer offsets.

## The takeaway
Occupancy/grid is not a kernel-body property — it's set by how you map work to workgroups. A perfectly
optimized inner loop still runs at ~1 TF if the grid is 8 workgroups on 80 CUs. **Always check
`grid vs CU count` before trusting (or despairing at) a throughput number** — exactly the habit from
Lesson 08, now shown end to end.
