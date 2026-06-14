# Lesson 05 — GEMM2 (`O = P·V`) and the first fully fused attention tile

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_05_pv_fused.py` → `PASS` (err ~0.002).

## What we build
End-to-end attention for one tile, one wavefront, **no LDS**: GEMM1 → softmax → GEMM2 → store.
`O[q,d] = Σ_kv softmax(QKᵀ)[q,kv] · V[kv,d]`.

## GEMM2 orientation
MFMA gives `result[m,n] = Σ_k A[m,k]·B[n,k]`. We want `O[q,d] = Σ_kv P[q,kv]·V[kv,d]`, so set
`m=q, n=d, k=kv`: **A = P[q,kv]**, **B[d,kv] = V[kv,d]** (i.e. B is `Vᵀ` — V loaded with d on the
row). P doesn't depend on d, so the A-fragment is reused across all `DT` output d-tiles.

## The headline lesson: "P-transpose problem" — and why it DOESN'T bite here
GEMM2's A-operand wants `P[q=mn, kv=k_outer*4+e]`. Softmax (Lesson 04) left this lane holding
`P[kv=k_outer*4+e, q=mn]` — **the same four scalars in the same registers.** For the `16x16x16` MFMA
the GEMM1 output C-fragment grouping matches the GEMM2 A-fragment grouping, so **P is already where
GEMM2 needs it — no transpose.**

That is a *property of this MFMA shape*, not of attention. The production fp8 kernel uses
`mfma_32x32x16`, whose C-fragment scatters kv differently than its A-fragment wants → there the
transpose is **real** and unavoidable. We hit it in Lesson 07 (fp8) and pay for it two ways:
through LDS (Lesson 11) and via `ds_bpermute` registers (Lesson 12), then finally *sidestep* it with
column-major V (Lesson 17). **Whether you pay a transpose depends on the MFMA shape you picked.**

## Verify
Against `torch.softmax(Q@Kᵀ·sm, dim=1) @ V` → err ~0.002 (bf16 rounding, not a bug).

## Next
Lesson 06: make it *streaming* — a runtime kv-loop with online softmax + causal mask.
