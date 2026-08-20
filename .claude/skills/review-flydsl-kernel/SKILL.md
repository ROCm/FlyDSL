---
name: review-flydsl-kernel
description: FlyDSL-specific review rules distilled from this repo's own maintainer review history. Loaded by review-pr when a PR touches FlyDSL kernels; encodes what coderfeli, sjfeng1999, xudoyuan and yanguahe reliably ask for, so an agent reviewer raises it before a human has to. Every rule cites the PRs it came from.
argument-hint: <PR number>
---

# FlyDSL kernel review

These rules are not invented. They are distilled from **479 human review comments across the
163 of this repo's 865 PRs that carry one**, and each rule cites the PRs where a maintainer
actually made that objection. The premise is that the dominant reviewers' taste is consistent:

| reviewer | comments | PRs | active |
|---|---|---|---|
| coderfeli | 273 | 101 | 2026-01-05 → 2026-08-20 |
| sjfeng1999 | 57 | 32 | 2026-03-20 → 2026-08-20 |
| xudoyuan | 20 | 13 | 2026-03-12 → 2026-08-14 |
| yanguahe | 11 | 5 | 2026-06-18 → 2026-08-17 |

Rules are ordered by how often the objection actually occurred, not by how interesting it is.

**Severity convention matches `review-pr`:** 🔴 block / ⚠️ should fix / 📝 note. Findings still
obey that skill's output contract — at most 5, most severe first, each ending in an action.

---

## F1 — Legacy spelling where the current one exists ⚠️

**46 of the 479 comments are this one objection.** It is also the only family here that a grep
can find, so do not spend reviewer attention recalling it — Step 1 of `review-pr` prints the
candidates via `scan_legacy_spelling.py`. Work that list.

| legacy | current | maintainers |
|---|---|---|
| `ir.*` / `_mlir.*` / `ArithValue` / manual wrap-unwrap | internal fx types (`fx.Float32`, `fx.Int32`, `expr/numeric.py`) | coderfeli #202 #250 #300 #326 #426 #850 |
| `scf.IfOp` / `scf.ForOp` | ordinary Python `if` / `for` | coderfeli #33 #433 #540 #582 |
| `buffer_ops.*` | `fx.copy` / a copy atom | coderfeli #404 #416 #894 #1032 |
| `arith` wrapper | raw op, with the `fastmath` hint | coderfeli #433 #848 #894 |
| `SmemAllocator` | `SharedAllocator` (old interface is going away) | sjfeng1999 #549 #567 |
| `make_ptr` to retype a pointer | `recast_iter` | sjfeng1999 #288 #745 |
| hand-rolled partition | `tiledCopy.partition_src/dst` | sjfeng1999 #564 |

FP self-check: fire only when the legacy spelling is on an **added** line. Pre-existing usage in
an untouched part of the file is not this PR's debt — say so and move on.
→ `⚠️ F1: [file:line] uses [legacy] where [current] exists — [maintainer] has asked for this on [PR]`

## F2 — A default argument that silently aliases another buffer 🔴

The most damaging pattern in this repo's review history, because it type-checks, runs, and
corrupts an unrelated tensor. Trigger: a kernel/builder parameter defaulting to another live
tensor (`if x is None: x = O`), where the kernel then writes a *different* dtype or semantic
into it.
Real examples: `return_lse=True` with no `lse` passed falls back to `Out`, so the kernel writes
fp32 LSE into the output tensor — three separate blockers on the same PR (yanguahe #844). The
same shape appeared independently in `debug_counts = O` on the gfx950 FMHA path (#629).
FP self-check: if the fallback target has identical dtype *and* the kernel writes the same
quantity, it is a convenience default, not this bug.
→ `🔴 F2: [param] defaults to [other buffer]; when [condition] the kernel writes [what] into it — require the argument or allocate`

## F3 — A duplicated code path that should be shared ⚠️

Trigger: two paths in the diff (or one new path against an existing one) that differ only in an
atom, an address, or a vector width.
Real examples: RoPE rotation copy-pasted from `q_rope_kernel`, maintainer asked for an
`_apply_neox_rope` helper (coderfeli #300); "vec and no-vec are two totally different code
paths? only atom and some addr diff?" (coderfeli #855); "reuse the code with decode?"
(coderfeli #540); "there is a similar function `_unwrap_to_raw` in kernel_function.py — same
purpose?" (xudoyuan #238).
→ `⚠️ F3: [path A] and [path B] differ only in [what] — extract the shared body`

## F4 — Config, dispatch or tile selection living inside the kernel file ⚠️

This repo keeps kernels buildable-as-told and puts selection one layer out. Trigger: a new
kernel file that also picks its own tile size, chooses a backend, or carries a config table;
or config added under `expr/`.
Real examples: "could you add an extra dispatcher out of the kernel for compile and tile
selection?" (coderfeli #780); "should not add too many configs in expr folder — leave the raw
interface there and the config in the kernel" (coderfeli #608); "move the act out of this moe
gemm kernel file" (coderfeli #823); "move to compiler.py? other kernels may also need it"
(coderfeli #108).
→ `⚠️ F4: [what] belongs in [dispatcher/config layer], not in the kernel file`

## F5 — A new test that the entry point CI runs cannot reach ⚠️

A test that exists but never executes is worse than none: it reports coverage that is not there.
Trigger: new tests added as pytest functions in a file that CI also executes as a script, where
`__main__` calls only a subset; or a shape list too narrow to exercise the path the PR adds.
Real examples: "these new variant tests are pytest-only today. `run_benchmark.sh` executes this
file as a script, but `__main__` only calls `test_all()`, so the fused/quant variants are not
exercised" (coderfeli #481); "more shape cover — use 4, 8, 32, 64, 128, 8192, 16384"
(coderfeli #318).
Pairs with `validate-kernel-pr`: that skill runs its own shape grid precisely because a repo's
declared coverage and its executed coverage differ. If a validation report is present, read its
`test_policy.commented_out_shape_rows` before writing this finding.
→ `⚠️ F5: [test] is not reached by [entry point] — wire it in, or state which job runs it`

## F6 — Unvalidated cast at the C ABI boundary ⚠️

Trigger: a value crossing the Python/C boundary that is cast to a narrower or unsigned type
without a range check.
Real example: "since this is a C ABI boundary, it would be safer to validate cluster dims before
casting from `intptr_t` to `unsigned`; a zero/negative or overflowing value would become a
confusing launch failure" (coderfeli #532).
Related but distinct from `review-pr`'s D9: D9 is about device-side index arithmetic overflowing
2^31; F6 is about host-side casts losing the sign or the range before the launch.
→ `⚠️ F6: [value] cast to [type] at the ABI boundary with no range check — validate before casting`

---

## What this skill deliberately does not do

- **It does not restate `flydsl-kernel-authoring` or `kernel-code-cleanup`.** Those tell an author
  how to write a kernel; F1 exists because reviewers still have to ask for it 46 times. If a
  finding is really "read the authoring guide", cite the guide instead of writing a finding.
- **It does not carry correctness rules that `review-pr` already covers** (masks, dispatch gates,
  fake impls, atomics on uninitialised memory). Those apply to FlyDSL kernels unchanged.
- **It does not judge performance.** No rule here fires on a number; `validate-kernel-pr` measures.

## Keeping it honest

Every rule above cites PRs. When adding one:

1. It must come from a real review comment in this repo, cited by PR number — not from a
   hypothetical failure mode.
2. Prefer a mechanical check over prose. F1 became `scan_legacy_spelling.py` because a rule that
   depends on the reviewer remembering to look does not fire; that was measured, not assumed —
   a rule rewritten in prose alone caught 0 of 3 target defects across four controlled arms.
3. Record how often the objection actually occurred. A family seen once is a note; a family seen
   46 times earns a scanner.
