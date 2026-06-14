# Lesson 16 — Diagonal-pair tiling: balancing the causal triangle

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_16_diagonal_pair.py`
Live (sq4096 / sq16384): `8wave 23/46 → v7 30/51 TF`.

## The A/B (two real kernels, one localized change)
- **BEFORE** `kernels/fmha_prefill_fp8_8wave.py` — one q-tile per workgroup.
- **AFTER** `kernels/fmha_prefill_fp8_v7.py` — one q-tile **plus its causal mirror** per workgroup.

Diff is in the workgroup→q-tile mapping: v7 exports `BM = 2*TILE_BM` and, inside one kernel,
processes q-tile `t` and `num_q_tiles-1-t` **sequentially**. Open both and compare the launch/tile
section — that's the whole lesson.

## Why it works
Under a causal mask, q-tile `t` attends to kv `0..(t+1)·BM`. So **early q-tiles do little work, late
q-tiles do lots** — a triangle. One-tile-per-workgroup makes the workgroups wildly unbalanced; the
machine stalls on the heavy ones while light ones finish early and idle. Pairing `t` with its mirror
`num_tiles-1-t` makes every workgroup's total kv work ≈ constant (light + heavy), so they finish
together and the CUs stay busy.

## Measured (bs=1, nq8, nk1, causal, TFLOPS)
| seq | 8wave | v7 (diag-pair) |
|---|---|---|
| 1024 | 5.0 | 4.8  *(loss)* |
| 4096 | 23.2 | ~30 |
| 16384 | 46.6 | 50.5 |
| 32768 | 52.9 | 56.9 |

**+8–24% at seq ≥ 2048**, but a **loss at sq1024**. Why the loss: pairing **halves the grid**
(64→32 workgroups), and at small seq that drops below the 80 CUs → CU-starvation (Lesson 08). This
shape-dependence is exactly why production dispatches per-shape (baseline ≤ sq1024, diagonal above).

## Why it's nearly free
The mirror tile runs **sequentially in the same kernel**, reusing the same registers — so VGPR
barely moves (164 → 168) and occupancy is unchanged. Contrast Lesson 19 (the 2-rep pipeline), where
trying to run two tiles *concurrently* blew VGPR 164→215 and lost occupancy. **Sequential reuse =
balance without register cost; concurrent = register blowup.** That distinction is the deep lesson.

## The habit
When a kernel has data-dependent per-workgroup work (causal, ragged batches), check **load balance**,
not just per-op efficiency: are some workgroups doing 10× the others? Balancing the schedule can beat
any micro-opt — and watch the grid size so you don't trade balance for starvation.

## Next
Lesson 17: column-major V — the biggest win, which *deletes* the transpose entirely.
