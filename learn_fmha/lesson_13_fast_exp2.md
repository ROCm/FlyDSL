# Lesson 13 — Fast exp: `rocdl.exp2` vs `Float32.exp2()`

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_13_fast_exp2.py`

```
Float32.exp2()      err=5.96e-08    95.9 us
fast rocdl.exp2     err=5.96e-08    97.9 us
```
ISA instruction diff (the real evidence):
```
slow: v_exp_f32=1  v_ldexp=1
fast: v_exp_f32=1  v_ldexp=0     <- the range-reduction op is gone
```

## The trick (the whole diff)
Softmax calls `exp` once per score. Two ways to get it in FlyDSL:
- `Float32.exp2(x)` — generic/safe: emits `v_exp_f32` **plus** a `v_ldexp` range-reduction (~2–3
  VALU/element).
- `rocdl.exp2(f32, x)` — the hardware `v_exp_f32` **alone** (1 VALU). Valid only when the input is
  already range-safe. In softmax it is: we always compute `exp(s - m)` with `s - m ≤ 0` (we even
  clamp here, mirroring the `safe_m` subtraction). `exp(x) = 2^(x·log2e)`, so scale by `log2e` first.

The diff is one line; the ISA dump confirms one fewer VALU op per element (`v_ldexp` removed).

## Why the wall time didn't move *here* (and why it matters in attention)
On this microkernel, fast and slow are the same ~96 µs — because the kernel is **memory-bound**
(stream 1M f32 in/out), so removing a VALU op doesn't touch the bottleneck. *Same meta-lesson as
Lesson 09:* a VALU optimization only shows up when you're VALU-bound.

In the attention kernel you **are** VALU-heavy — Lesson 12's PMC measures **VALU:MFMA ≈ 23:1**, and
softmax calls `exp` 16–64× per kv-tile. There, shaving `v_ldexp` per call is a real (if small)
contributor. The production kernel uses `rocdl.exp2` for exactly this reason.

## The method this reinforces
- Don't measure micro-optimizations only by wall time on a toy — also **count instructions** in the
  ISA (`grep -c v_ldexp`). A change can be correct and provably leaner yet invisible on a kernel
  bound by something else.
- Apply it where the bottleneck is: VALU tricks → VALU-bound kernels (check VALU:MFMA first).

## FlyDSL gotcha caught here
The `@flyc.kernel` AST rewriter rejects a value **assigned inside an `if`/`else` statement and used
after it** — even when the condition is a Python compile-time bool. Use a **ternary expression**
(`y = A if fast else B`) or a host-side helper that returns the chosen expression, so `y` is a single
assignment. (Same rule bit us in Lesson 06's softmax restructure.)

## Next
Lesson 10: cooperative K/V → LDS (load the shared tile once for all waves; the barrier cost).
