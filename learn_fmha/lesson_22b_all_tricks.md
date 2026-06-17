# Lesson 22b — The all-tricks kernel (tutorial style), in one file

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_22b_all_tricks.py` → 4× `PASS`
(causal, non-causal, sk≠sq; err ~0.01–0.034). Single shape:
`python3 learn_fmha/lesson_22b_all_tricks.py 1 1024 1024 1`.

## What this is
The companion to the lesson-22 *scoreboard*: a single self-contained fp8 causal attention kernel
(~250 lines) with the Part-D wins **folded in**, so you can read the whole optimized kernel at once
instead of diffing production files. It starts from lesson 07 (fp8) and adds:

| trick | lesson | what's in this file |
|---|---|---|
| **Multiwave** (BM=128/4-wave) | 08 | `NWAVES` waves/workgroup, each owns 32 q-rows; a real grid |
| **Column-V** | 17 | V stored `[d, kv]`; GEMM2's V read is contiguous → **V transpose deleted** |
| **Register-P transpose** | 12 | `ds_bpermute` (4×), no P-LDS round-trip |
| **Causal kv-bound** | 14 | kv loop capped at the causal limit; per-lane `eff_bound` |
| **Fast exp2** | 13 | `rocdl.exp2` in softmax |

It uses the **32×32×16** fp8 MFMA (like production) so the register-P `ds_bpermute` path is clean
(16 C-regs/lane, kv in 2 half-groups → one `shuffle_xor`). The lane↔element layout was derived and
verified the Lesson-01 way.

## What's deliberately omitted (and why)
Paged KV, GQA, p_scale, diagonal-pair, ping-pong double-buffer. Each adds real code for incremental
perf; leaving them out keeps this a kernel you can hold in your head. They all live in the production
`kernels/fmha_prefill_fp8_ck.py` (61/70 TF) — this file trades that last ~15% for readability.

## The key thing to study: column-V deletes a whole transpose
Compare the GEMM2 here to lesson 07's. In 07, V was row-major `[kv, d]`, so GEMM2 needed P/V
transposed through LDS. Here V is stored **column-major `[d, kv]`** (the host does
`Vq.t().contiguous()` and the coop-load copies it straight into LDS as `[d, kv]`). Now GEMM2's V
operand — 8 contiguous kv for a fixed d — is a **single wide `ds_read`**, no scatter, no transpose.
That one layout choice is the biggest single win in the whole tutorial (Lesson 17).

Note P **still** needs its transpose (`ds_bpermute`) — column-V only removes the *V* side. That
asymmetry (V layout fixable, P inherently produced transposed by GEMM1) is worth internalizing.

## Gotchas this kernel demonstrates
- **One shape per process** (the `SmemAllocator.finalize()` guard): `__main__` forks a subprocess
  per shape, like `tests/ck_check.py`. Benchmarking multiple shapes in one process throws "not a
  valid global memref."
- **`scf.for` + smem:** the `SmemPtr` views are recreated inside the loop body.
- **`ArithValue` vs IR value:** wrap a computed index in `fx.Int32(...)` before `.ir_value()` for
  `buffer_store` (an `expr * + ` product is an `ArithValue` without `.ir_value()`).

## How it relates to the scoreboard (lesson 22)
This is the readable embodiment of the "ours (ck)" column. For the *measured* 61/70 TF and the
comparison to CK-Tile (141/146) and PyISA (238/291) — and the discussion of the codegen ceiling —
see `lesson_22_capstone.md`.

## You've finished the series
You built attention from a single MFMA to a fully-optimized fp8 kernel, learned to classify
bottlenecks with PMC, saw which tricks win and which regress and *why*, and now have the whole
optimized kernel in one file. The transferable method is in `lesson_22_capstone.md`'s closing list.
