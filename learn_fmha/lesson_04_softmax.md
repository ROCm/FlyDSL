# Lesson 04 — Softmax over kv: the cross-lane reduction

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_04_softmax.py` → `PASS`.

## What we build
From the scores `S=[kv,q]` (recomputed inline via Lesson 03's GEMM1), compute per-query softmax
weights `P[kv,q] = exp(S - max_kv S) / Σ_kv exp(...)`. The reduction is **over kv, for each query** —
and the tricky part is that a query's kv values live in *different lanes*.

This lesson does softmax over one 16-kv tile (fits in registers). The streaming "online" version
that folds many kv-tiles with a running max/sum is Lesson 06 — the math here is its per-tile core.

## The key mechanic: reducing across lanes
With `S=[kv=row, q=col]`, lane `(k_outer=lane//16, mn=lane%16)` holds `S[kv=k_outer*4+e, q=mn]`,
e=0..3. So for a fixed query `q=mn`, its 16 kv values are spread across **4 lanes**: this lane plus
`mn+16, mn+32, mn+48` (the other `k_outer` groups). Reduce in two stages:

1. **Intra-lane** — max/sum over this lane's own 4 registers (plain scalar ops over e=0..3).
2. **Cross-lane** — merge the 4 `k_outer` groups with `shuffle_xor(mask, width)`: a lane reads the
   value from `other_lane = my_lane XOR mask`. Since `k_outer = lane//16`:
   - `XOR 16` flips `k_outer` bit0 (groups 0↔1, 2↔3),
   - `XOR 32` flips `k_outer` bit1 (groups 0↔2, 1↔3).
   Doing both folds all four groups; afterwards every lane in column `mn` holds the fully-reduced
   max (then sum). `width=64` = full wavefront.

> CK Tile analogy: this is the warp-level reduction `block_tile_reduce` / the cross-lane part of a
> softmax, but here you write the shuffle pattern yourself, matched to the MFMA's fragment layout.

The production kernel uses `mfma_32x32x16` (16 regs/lane, kv split across only **2** lane-groups → a
**single** `shuffle_xor(32)`). We use the 16-wide MFMA here (4 regs/lane, 4 groups → **two**
shuffles). Same idea, different fan-out — a reminder that *the reduction pattern is dictated by the
MFMA fragment layout you chose in Lesson 03*.

## Numerical stability + the all-masked guard
- Subtract the row max before `exp` (standard softmax stability).
- `exp(x) = 2^(x·log2e)` via `rocdl.exp2` (one hardware `v_exp_f32`; Lesson 13 shows why this beats
  the generic `Float32.exp2()`).
- **All-masked guard:** if the max is `-inf` (a query with no valid kv — happens under causal
  masking in Lesson 06), force it to 0 so `exp(s - 0)` with `s=-inf` gives 0, not `NaN`. Harmless on
  this dense tile; introduced now so the pattern is familiar before masking arrives.

## Verify
Against `torch.softmax(S, dim=0)` (dim 0 = kv) → err 0.0.

## Next
Lesson 05: GEMM2 `O = Vᵀ@P` — and the **P-transpose problem** that the `[kv,q]` choice created.
