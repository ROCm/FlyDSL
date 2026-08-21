---
name: review-flydsl-kernel
description: FlyDSL review rules distilled from this repo's own maintainer history. Currently one rule — legacy DSL spellings that maintainers repeatedly ask to be replaced — shipped as a diff scanner. Loaded by review-pr when a PR touches FlyDSL kernels.
argument-hint: <PR number>
---

# FlyDSL kernel review

## What is in here, and what is not

One rule. It is here because it is the single most repeated objection in this repository's
review history **and** because it survived a test: the scanner it ships as flags 10 of the 12
files where a maintainer actually raised it.

Five further families were distilled from the same corpus — default arguments aliasing another
buffer, duplicated code paths, config living inside the kernel file, tests unreachable from the
CI entry point, unvalidated casts at the ABI boundary. They were **removed before merge** after a
controlled test: seeded into real kernels and reviewed with and without this skill, `review-pr`
alone already caught four of the five. Adding rules that change nothing costs reviewer attention
and dilutes the ones that matter. They can come back if evidence does.

That is the standard for anything added below.

---

## Provenance

Mined from all 865 PRs in this repo; 163 carry human review, 479 comments in total.

| reviewer | comments | PRs | active |
|---|---|---|---|
| coderfeli | 273 | 101 | 2026-01-05 → 2026-08-20 |
| sjfeng1999 | 57 | 32 | 2026-03-20 → 2026-08-20 |
| xudoyuan | 20 | 13 | 2026-03-12 → 2026-08-14 |
| yanguahe | 11 | 5 | 2026-06-18 → 2026-08-17 |

Eight months of consistent taste from the top two is what makes any of this encodable. The
corpus stops at 2026-08-20; re-mine before trusting the counts.

---

## F1 — Legacy spelling where the current one exists ⚠️

**46 of the 479 comments are this one objection** — by a wide margin the largest family, and the
one thing a maintainer should never have to type again.

Do not review this from memory. Step 1 of `review-pr` runs the scanner and prints the candidates;
work that list.

```bash
.claude/skills/review-flydsl-kernel/scan_legacy_spelling.py <owner/repo> <PR>
.claude/skills/review-flydsl-kernel/scan_legacy_spelling.py --diff /path/to.diff
```

| legacy | current | maintainers |
|---|---|---|
| `ir.*` / `_mlir.*` / `ArithValue` / manual wrap-unwrap | internal fx types (`fx.Float32`, `fx.Int32`, `expr/numeric.py`) | coderfeli #202 #250 #300 #326 #426 #850 |
| `scf.IfOp` / `scf.ForOp` | ordinary Python `if` / `for` | coderfeli #33 #433 #540 #582 |
| `buffer_ops.*` | `fx.copy` / a copy atom | coderfeli #404 #416 #894 #1032 |
| `arith` wrapper | raw op, with the `fastmath` hint | coderfeli #433 #848 #894 |
| `SmemAllocator` | `SharedAllocator` (old interface is going away) | sjfeng1999 #549 #567 |
| `make_ptr` to retype a pointer | `recast_iter` | sjfeng1999 #288 #745 |
| hand-rolled partition | `tiledCopy.partition_src/dst` | sjfeng1999 #564 |

**FP self-check.** Fire only when the legacy spelling is on an **added** line. Pre-existing usage
elsewhere in a touched file is not this PR's debt — say so and move on. The scanner already
restricts itself to added lines; do not widen it by hand.

**Severity.** ⚠️ by default. It is not a correctness defect, and it does not get one of the five
finding slots ahead of something that produces a wrong number. Group all instances into a single
finding rather than one per site.

→ `⚠️ [file:line] uses [legacy] where [current] exists — [maintainer] has asked for this on [PR]`

---

## Why this is a scanner and not a paragraph

A rule that depends on the reviewer remembering to look does not fire. This was measured, not
assumed: on aiter, a rule covering 32-bit overflow was rewritten from a variable-name list to a
structural criterion, and across four controlled arms over 14 PRs it caught **0 of 3** of the
defects it targeted — including after the instruction was relocated into the step the agent
reliably executes, with the session transcript confirming the scan ran and its output was in
context. The scanner found all three offline.

Prefer a check that runs over a rule that must be recalled.

## Adding a rule here

1. It must come from a real review comment in this repo, cited by PR number.
2. Record how often the objection actually occurred. Once is a note; 46 times earns a scanner.
3. **Show it changes an outcome.** Seed the pattern into a real kernel, review it with and
   without the rule, and keep it only if the rule catches something `review-pr` alone does not.
   Four of the first six families failed that test and are not in this file.
