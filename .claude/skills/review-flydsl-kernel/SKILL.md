---
name: review-flydsl-kernel
description: FlyDSL review rules distilled from this repo's own maintainer history. Two rules, both shipped as diff scanners: legacy DSL spellings maintainers repeatedly ask to be replaced, and tests a PR adds that the script entry point never runs. Loaded by review-pr when a PR touches FlyDSL kernels.
argument-hint: <PR number>
---

# FlyDSL kernel review

## What is in here, and what is not

Two rules. Both are the most repeated objections in this repository's review history, both ship
as scanners rather than prose, and both were kept because they passed the same test:

| rule | corpus support | evidence it earns its place |
|---|---|---|
| F1 legacy spelling | 46 of 479 comments | scanner flags 10 of the 12 files a maintainer raised it on |
| F5 added test unreachable from the script entry point | coderfeli #481, #318 | the one family `review-pr` alone missed in a controlled seeded test (4/5 → 5/5); its scanner reproduces #481's objection exactly |

Four further families were distilled from the same corpus — default arguments aliasing another
buffer, duplicated code paths, config living inside the kernel file, unvalidated casts at the ABI
boundary. They are **not here**: seeded into real kernels and reviewed with and without this
skill, `review-pr` alone caught all four. Rules that change nothing cost reviewer attention and
dilute the five finding slots. They can return with evidence.

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

## F5 — A test this PR adds that the script entry point cannot reach ⚠️

Several test files here run two ways: pytest collects `test_*`, and `scripts/run_benchmark.sh`
runs the same file as a script, where only what `__main__` calls happens. A pytest test added to
such a file does not run in the script path — the file reports coverage it does not have.

> coderfeli on #481: *"These new variant tests are pytest-only today. run_benchmark.sh executes
> this file as a script, but `__main__` only calls `test_all()`, so the fused/quant variants are
> not exercised in that path."*

```bash
.claude/skills/review-flydsl-kernel/scan_unreachable_tests.py --diff <diff> <worktree-root>
```

It parses the file, resolves what `__main__` reaches transitively, and reports **only tests the
diff adds**. Replaying #481 it names exactly the three the maintainer named.

**FP self-check.** Pre-existing unreachable tests are the norm here — most files are pytest-first
with a one-line `__main__`, and a first version of this scanner fired on 6 of 7 untouched tests
in one file. Only added tests count. A file with no `__main__` at all is pytest-only by design
and is not a finding.

This is the static half of a problem `validate-kernel-pr` covers from the other side: F5 catches
a test that will not run, its shape grid catches shapes a test that does run never reaches.

→ `⚠️ [file] adds [tests] that `__main__` does not reach — wire them in, or say which job runs them`

---

## Why these are scanners and not paragraphs

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
