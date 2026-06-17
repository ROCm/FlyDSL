# Lesson 20 (NEGATIVE) — v_perm wide V-transpose: crush one counter, lose overall

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_20_neg_vperm_transpose.py`
Live (sq16384): `8wave 46 → v14 44 TF` (regressed).

## The idea
Lesson 11/12 showed the V transpose is built from 48 narrow `ds_write_b8` — the **dominant LDS op**
(Lesson 12 PMC). Replace them with an in-register 4×4 byte transpose using `v_perm_b32`, then a few
**wide** `ds_write_b32`. Fewer, wider LDS writes → less LDS pressure. Should be a clear win.

## What the profile said — it hit the target and *still* lost
```
ds_write_b8:    48 -> 0       (goal achieved)
LDS-wait/busy:  ~102% -> ~27% (LDS pressure crushed!)
busy cycles:    UP
TFLOPS:         ~46 -> ~44    (regressed)
```
The `v_perm` transpose added **~48 VALU ops/tile** and pushed **VGPR 164 → 179** (occupancy down).
The transpose **work is irreducible** — you pay it on the **DS unit** (`ds_write_b8`) *or* on the
**VALU** (`v_perm`). Trading DS for VALU just relocated the bottleneck onto a unit that, here, costs
more.

## The `perm_b32` gotcha (if you try it)
`rocdl.perm_b32(hi, lo, sel)`: `sel` is **one byte per output byte** (value 0–7 = source byte index
into the 8-byte `{hi:lo}`), **not** nibbles. Getting this wrong gives `0xff` garbage bytes — verify
the transpose on a tiny standalone kernel before wiring it in.

## The punchline (ties Part D together)
Lessons 11 (LDS scatter), 12 (`ds_bpermute`), 20 (`v_perm`) are **three different ways to move the
same transpose cost** between hardware units — and none wins, because the cost is irreducible *in the
row-major V layout*. **Lesson 17 deletes it** by changing the layout (column-major V). When a cost
won't reduce no matter which unit you push it to, stop shuffling and change the data layout.

A/B kernels: `kernels/fmha_prefill_fp8_8wave.py` vs `kernels/fmha_prefill_fp8_v14.py`.
