# Lesson 17 — Column-major V: delete the transpose with the right layout (the biggest win)

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_17_column_v.py`
Live (sq16384): `v7 51 → ck 61 TF (+20%)`.

## The A/B (two real kernels, one design choice)
- **BEFORE** `kernels/fmha_prefill_fp8_v7.py` — V row-major; GEMM2 needs a V/P transpose (LDS scatter
  + `ds_bpermute`, Lessons 11–12).
- **AFTER** `kernels/fmha_prefill_fp8_ck.py` — V **column-major** (`vec_k_col_v`); GEMM2's
  contraction over kv is already contiguous → **no transpose at all**.

## The punchline of the whole tutorial
Lessons 11–12 fought the transpose with LDS round-trips and register `ds_bpermute`. Lesson 20
(negative) shows even a `v_perm` register transpose can't win it — the transpose **work is
irreducible** *if you keep V row-major*. The fix isn't a cleverer transpose; it's to **store V so the
transpose never exists**. Lay V out column-major (kv contiguous along the GEMM2 contraction); the
KV-cache packer writes that layout once, and the kernel pays **zero** transpose forever.

## Measured (bs=1, nq8, nk1, causal, TFLOPS)
| seq | v7 | **ck (column-V)** | CK-Tile ref |
|---|---|---|---|
| 1024 | 4.8 | 5 | 30 |
| 2048 | 15 | 16 | 62 |
| 16384 | 50.5 | **61** | 141 |
| 32768 | 57 | **70** | 146 |

PMC evidence (the *why*, measured this session):
```
ds_write_b8:  48 -> 0     (V-transpose scatter GONE)
ds_bpermute:   N -> 0     (P-transpose GONE too)
LDS-instructions: 111M -> 63M  (-43%)
busy cycles:               -18%
```
That −43% LDS traffic / −18% busy directly explains the +20%. **ck is the best FlyDSL kernel** we
have — and it **disproves** the assumption that you need gfx950 hardware transpose-loads
(`ds_read_tr`) to beat the transpose on gfx942: the right data layout sidesteps it.

## The habit (the meta-lesson of Part D)
When a cost is *irreducible in the current layout*, stop optimizing the operation and **change the
layout** so the operation disappears. Transpose, bank conflicts, uncoalesced loads — many "hard"
kernel problems are really *layout* problems wearing a costume. Profile first (here: the dominant op
was `ds_write_b8` from the transpose), then ask "what layout makes this op unnecessary?"

## Apples-to-apples note
ck sets `V_COL=True`, so its V pool must be packed column-major. `bench_fmha_compare.py` reads each
kernel's `V_COL` flag and packs the matching pool, so the comparison is fair (use `ck_check.py` for
correctness — a row-major checker would falsely report ~4.0 error on ck).

## Next
Lesson 18 (Part E): the first instructive *failure* — 128-kv-per-softmax, which optimized the wrong
bottleneck.
