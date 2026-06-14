# Lesson 06 — Streaming attention: runtime kv-loop, online softmax, causal mask

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_06_causal_seqloop.py` → 3× `PASS`
(causal, non-causal, sk≠sq).

## What we build
The real flash-attention structure: one query tile vs **all** kv tiles, folded with online softmax,
over a **runtime** sequence length, with the causal mask. This is the last bf16 lesson — Lesson 07
swaps in fp8 and nothing structural changes.

## Online softmax (the flash trick)
We never materialize the full score row. Per query we keep a running max `m`, running denominator
`l`, and unnormalized output `o`. Each kv-tile:
```
m_new = max(m_old, tile_max)
corr  = exp(m_old - m_new)      # how much to shrink everything already accumulated
p     = exp(s - m_new)          # this tile's (unnormalized) weights
l     = l*corr + sum(p)
o     = o*corr + (Vᵀ @ p)       # rescale old o, add this tile's contribution
```
At the end `O = o / l`. Mathematically exact, O(seqlen) memory.

## FlyDSL: the runtime loop
Compile-time loops use `range_constexpr` (unrolled). A loop whose bound depends on `seqlen` uses an
`scf.for` with **loop-carried state**:
```python
for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=[m, l, o0, o1, o2, o3]):
    m_run, l_run = st[0], st[1];  o_acc = [st[2+d] for d in range(DT)]
    ...
    st = yield [m_new, l_new] + new_o
# after the loop, st holds the final carried values
```
There are **no mutable captured variables across iterations** — every value that must survive an
iteration is threaded through `init=` and `yield`. (CK Tile hides this in its pipeline driver; here
you write the carry explicitly. It's verbose but there's zero hidden state.)

## Causal mask — don't forget the offset
Query at absolute row `qrow` attends to key `kv` only if `kv <= qrow + (sk - sq)`. The `(sk - sq)`
term aligns the diagonal when the KV sequence is longer than the query sequence (prefix/KV-cache).
Drop it and every `sk != sq` case silently breaks — which is why a test case uses `sk=48, sq=16`.
Masked scores become `-inf` → `p=0`; the **all-masked guard** (Lesson 04) keeps fully-masked rows
from becoming `NaN`.

## War story baked into this lesson: the orientation bug
This kernel failed on the first two tries with `err ~0.9`, and the fix is the most instructive part.
- **Bug:** GEMM2 produced `O[q=row, d=col]`, but the running normalizer `l` is indexed by the
  *lane's query* (`q = mn`, the column). After the MFMA the query axis had moved to the **row**, so
  each output was divided by the **wrong query's** `l`. (One column happened to match, masking it as
  "almost right" — a classic partial-correctness red herring.)
- **Fix:** orient GEMM2 as `O[d=row, q=col=mn]` (A = `Vᵀ`, B = `P`). Now every element a lane writes
  has `q = mn`, so its single `inv_l` applies to all of them. This is exactly why the production
  kernel uses the `O[d,q]` orientation.
- **Second bug found by the same assert:** the output `d` index must be `dt*16 + (k_outer*4+e)`
  (the C-fragment row), not just `k_outer*4+e`.

**Lesson:** the reduction axis (`l` per query) and the MFMA output axes must be kept consistent;
when in doubt, choose the GEMM2 orientation that keeps the softmax-reduced dimension on the output
**column** so the per-lane normalizer lines up. And: an `err` that's small for *some* outputs is not
"almost correct" — it's a layout bug. Trust the assert, find the index.

## Next
Lesson 07: convert to fp8 (e4m3 FNUZ) — the descale/p_scale/packing step, and where the *real*
P-transpose problem finally appears.
