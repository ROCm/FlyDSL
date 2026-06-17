# Lesson 07 — fp8 (e4m3 FNUZ): packing, descale, p_scale, and the REAL P-transpose

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_07_fp8_quant.py` → 3× `PASS` (err ~0.014).
Single shape: `python3 learn_fmha/lesson_07_fp8_quant.py 16 64 1`.

## What we build
Lesson 06's streaming attention with **one thing changed: the dtype → fp8**. This is the bridge to
the production kernel. The dtype switch is never "just a cast" — it forces four new mechanics, each a
lesson, plus it finally makes the P-transpose *real*.

## The four fp8 mechanics
1. **fp8 MFMA** `mfma_f32_16x16x32_fp8_fp8` — K=**32** (vs 16 for bf16). Each lane feeds 8 fp8 packed
   into an i64.
2. **Packing — there is no usable `.to(fp8)`.** `.to(fp8)` emits a bare `arith.truncf` the backend
   rejects (you'll see a lowering error). Use `rocdl.cvt_pk_fp8_f32(res, a, b, old, word_sel)`: it
   packs **two f32 → two fp8 bytes** into an i32 (`word_sel` picks low/high half; chain 2 calls for
   4 fp8, 4 calls for an i64 of 8). Loading fp8: read as i32/i64 dwords (`vec_width=2` Int32 →
   bitcast i64), never as `v8i8` (backend crash).
3. **Descale.** fp8 has ~2 decimal digits, so Q/K/V are quantized with scales. The MFMA runs on
   raw fp8; you multiply the f32 result by the descales: Q/K descale applied to **S after the QK
   MFMA**, V descale folded into the **output epilogue** (`O *= v_descale / l`). (Here scales are
   per-tensor for simplicity; production is per-token-per-head for Q/K, per-head for V — same
   placement, finer granularity.)
4. **p_scale** (wired, set to 1.0). The probabilities P are small; multiplying by a per-head
   `p_scale` before the fp8 cast keeps them in e4m3 range, then it **cancels** in `O/l`. Implemented
   as `+log2(p_scale)` in the exp bias. Pure precision knob — left at 1 here.

## The REAL P-transpose (why fp8 is harder than bf16)
In Lesson 05 (16×16×16 bf16) the GEMM1 output fragment happened to match GEMM2's P-input fragment,
so **no transpose**. With the **16×16×32 fp8** MFMA that luck is gone:
- GEMM2's P operand needs **8 contiguous kv per lane** (`P[q=mn, kv=k_outer*8+e]`),
- but GEMM1/softmax produced only **4 kv per lane**, grouped differently.

4 ≠ 8 and the grouping differs → the registers do **not** line up. So we **must** transpose P. The
simplest fix (this lesson): round-trip P through **LDS** — store it q-major as fp8
(`p_lds[q*BKV + kv]`), `barrier`, reload 8 contiguous kv per lane as the i64 B-operand. Lessons 11–12
make this cheaper (V-transpose in LDS, then register `ds_bpermute`); Lesson 17 removes the V side
entirely with column-major V. **The transpose is a consequence of the MFMA shape, not of attention.**

## Two structural facts the fp8 MFMA forces
- **Two GEMM1 sub-tiles per kv-tile.** The MFMA outputs M=16 kv, but GEMM2 contracts K=32 kv. So one
  32-kv tile = **two** GEMM1 MFMAs (kv 0–15, then 16–31), giving 8 scores/lane that fill GEMM2's
  K=32 exactly. This "sub-tile" structure is exactly why the production kernel has an inner loop.
  *(This was a real bug while writing the lesson: with `BKV=32` but only one GEMM1, half the keys
  were silently skipped → err 0.9. The fix is the two sub-tiles.)*

## Two gotchas baked in
- **`scf.for` + shared memory:** recreate the `SmemPtr` view **inside** the loop body. A view made
  before the loop is invalidated by it → `'memref.get_global' ... not a valid global memref`.
- **One shape per process:** the module-global `SmemAllocator` is `finalize()`d once (it has a
  `finalized` guard). JIT a second shape in the same process and the kernel recompiles but finalize
  is skipped → the same "not a valid global" error. So `__main__` forks a subprocess per shape (just
  like `tests/ck_check.py`). *This is why the production test harness runs one shape per subprocess.*

## Verify
Reference is built from the **dequantized fp8 inputs** (`Qq.float()*qd @ ...`) so we measure the
*kernel*, not quantization noise. err ~0.014 — that residual is fp8 rounding of P, expected.

## Next
Lesson 08: widen from 1 wave to a multi-wave workgroup — the first real *performance* lesson, where
we start trusting TFLOPS and profiling occupancy.
