# Lesson 03 — Attention GEMM1: scores `S = K @ Qᵀ`

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_03_qk_scores.py` → `PASS`.

## What we build
The first of attention's two GEMMs: from `Q[q,hd]` and `K[kv,hd]` compute the scores tile, scaled by
`1/√hd`. Mechanically it's Lesson 02's K-loop GEMM (loop over head_dim), just with attention names.

## The layout decision that shapes the whole kernel
The MFMA computes `A @ Bᵀ`. We pick **A = K, B = Q**, so the result is `K @ Qᵀ` with shape
**`[kv=row, q=col]`**. This is a *deliberate* choice, not an accident:

- **Softmax reduces over kv** (Lesson 04). With `S=[kv,q]`, a lane's accumulator registers hold
  several kv values for one query column → the reduction is "reduce my own registers, then a couple
  of cross-lane shuffles." Cheap.
- **The cost:** GEMM2 (`O = Pᵀ·V` style, Lesson 05) needs kv as the *contraction* index, and the
  C-fragment from GEMM1 doesn't hand kv to GEMM2 in the layout it wants → the **P-transpose
  problem**. Every attention kernel pays a transpose *somewhere*; choosing `S=[kv,q]` pays it on the
  small P matrix, which is cheaper than transposing Q or the output. CK Tile's QRKSVS pipeline makes
  the same call.

The lesson: **operand/output layout is a global design decision.** You choose where the unavoidable
transpose lands; you don't avoid it.

## Concretely, who holds what
Reusing Lesson 01's C-fragment rule (`lane (k_outer, mn) → C[k_outer*4+e, mn]`), with C row=kv,
col=q: lane holds `S[kv = (lane//16)*4 + e, q = lane%16]`, e=0..3. We fold `sm_scale` in at store
time (in the fused kernel it multiplies S right before softmax).

## Verify
Checked against torch `(K @ Q.T) * sm` with the *same* `[kv,q]` orientation → err 0.0. Note the
reference is `K@Q.T`, not `Q@K.T` — matching our A=K/B=Q choice. Getting that backwards is the most
common first bug; the assert catches it.

## Next
Lesson 04: softmax over kv — the cross-lane reduction.
