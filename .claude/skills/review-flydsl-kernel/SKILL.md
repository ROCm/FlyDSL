---
name: review-flydsl-kernel
description: >-
  Review FlyDSL kernel PRs using maintainer review evidence, added-line legacy
  spelling candidates, and test entry-point checks. Use for kernel reviews or
  review-comment follow-up; scanner output needs source-level adjudication.
argument-hint: <PR number or URL>
---

# FlyDSL kernel review

Run this skill directly with a PR number or URL. It supplies two static candidate
scanners and focused review questions; it does not require a separate review skill
to invoke them. Review-only requests authorize inspection, not edits or publication.

## Run the checks

1. Read the PR description, review threads, and repository guidance. Record the
   base and head OIDs with `gh pr view <PR> --repo ROCm/FlyDSL --json baseRefOid,headRefOid`.
   Fetch the objects and use their merge base for the comparison. Preserve the
   user's checkout; prepare a detached worktree at that exact head.
2. Use a unique directory for each review's artifacts. From the repository that
   contains this skill, with `base` set to the merge base and `head` to the recorded
   head OID:

   ```bash
   skill_dir="$PWD/.claude/skills/review-flydsl-kernel"
   review_dir=$(mktemp -d /tmp/flydsl-review.XXXXXX)
   git worktree add --detach "$review_dir/head" "$head"
   git diff --no-ext-diff --unified=3 "$base" "$head" > "$review_dir/pr.diff"
   python3 "$skill_dir/scan_legacy_spelling.py" --diff "$review_dir/pr.diff"
   python3 "$skill_dir/scan_unreachable_tests.py" --diff "$review_dir/pr.diff" "$review_dir/head"
   ```

   Run both scanners even if the first returns candidates. Both use exit **0** for
   no candidates in the supported scope, **1** for candidates needing review, and
   **2** for invalid arguments, input, or tool failure. An error is an incomplete
   check, never a clean result. The spelling scanner also accepts
   `python3 <skill-dir>/scan_legacy_spelling.py ROCm/FlyDSL <PR>`; use the local
   diff form when the evidence must stay pinned to an OID.
3. Read each candidate in the head source and compare with the base. Check the
   current API and actual test command before promoting it to a finding. Apply
   the relevant questions in [review-evidence.md](references/review-evidence.md)
   when the diff changes dispatch, numerical flags, or kernel interfaces.
4. Report concrete triggers, consequences, locations, and actions. Prioritize
   wrong results over style; group spelling instances into one finding. State
   which checks ran and which behavior remains unverified. Before an authorized
   GitHub reply or review, re-check the head OID and any intervening discussion.

These scanners only parse text; they do not execute the reviewed Python. Runtime
validation needs a checkout-compatible FlyDSL build, explicit device selection,
and tests proportionate to the changed behavior. Static output establishes neither
kernel correctness nor performance.

## F1 — Legacy spelling candidates

`scan_legacy_spelling.py` checks **added lines in kernel Python files**. It excludes
the low-level implementation in `kernels/common/buffer_ops.py`, imports, and
comments/string literals it can identify in the diff. It does not scan compiler
internals, tests, documentation, or its own implementation.

| Scanned spelling | Inspect before recommending a replacement | Review evidence |
|---|---|---|
| `ir.*`, `_mlir.*`, `ArithValue`, explicit unwraps | Prefer typed `fx.Float32` / `fx.Int32` values where the consumer accepts them; raw MLIR boundaries may need unwrapping. | coderfeli #202 #250 #300 #326 #426 #850 |
| `scf.IfOp` / `scf.ForOp` and related builders | Prefer traced Python control flow when it preserves the SSA results, carried state, and side effects. | coderfeli #33 #433 #540 #582 |
| `buffer_ops.*` | Prefer layout-aware copies/copy atoms when they express the same masking and addressing. | coderfeli #404 #416 #894 #1032 |
| `SmemAllocator` | Prefer `fx.SharedAllocator` for new kernels; verify static/dynamic LDS launch semantics. | sjfeng1999 #549 #567 |
| `make_ptr` | Use `fx.recast_iter` **only for retyping an existing pointer**; constructing a pointer is legitimate. | sjfeng1999 #288 #745 |

The scanner recognizes spellings, not import identity or semantic equivalence.
Aliases and multiline expressions can escape it. A diff hunk starting inside an
existing multiline string lacks its opening delimiter; inspect the complete source
before accepting that match. A candidate is not proof of deprecation or a defect.
Use `docs/api_stability.md` and the **api-stability** skill for compatibility claims.

Two historical objections require separate checks:

- **Arithmetic:** run the existing `scripts/check_typed_arithmetic_usage.py`
  with `--base <base> --head <head>` **from the head worktree**. It checks the
  current typed-arithmetic policy. Inspect NaN and fastmath semantics before
  replacing operations; the historical suggestion to prefer raw `arith` builders
  is not a current blanket rule.
- **Hand-written partitioning:** compare the mapping with the existing tiled-copy
  partition methods (sjfeng1999 #564). This is a manual layout-equivalence check;
  no spelling regex covers it.

## F5 — Added tests and script entry points

`scan_unreachable_tests.py` compares test definitions whose `def` line is added
by the diff with the head file's statically visible `__main__` call paths. It
identifies module tests and `Test*` methods by qualified name and source location.
It follows direct local calls and recognizes an unfiltered
`pytest.main([__file__])` entry point, including common import aliases. Selected
or dynamic pytest invocations require manual review.

The scanner reports potential gaps and uncertain paths, not proof that a test ran.
Conditions, callbacks, fixtures, plugins, and external dispatch need inspection.
Files without a supported script guard are outside this check. Editing an existing
test's signature can also add a `def` line; compare with the base before calling it
new coverage debt. Changes only inside an existing test body are outside scope.

Before filing a coverage finding, identify the command in `scripts/run_tests.sh`,
`scripts/run_benchmark.sh`, or the relevant workflow that is supposed to execute
the test. Pytest correctness coverage and script benchmark coverage are distinct.
An added pytest test does not automatically belong in a benchmark entry point.
Historical example: coderfeli #481 identified fused/quant tests missing from the
specific script path under discussion.

## Maintain the evidence

The original August 20 corpus and the September 10 refresh, with source links,
scope, and limitations, are in [review-evidence.md](references/review-evidence.md).
Historical frequencies are not current API policy or measured review accuracy.

Add guidance only when a cited review changes a concrete decision. A new scanner
needs a reproducible positive case, a clean control, and evidence it adds coverage
beyond existing checks. Keep one-off concerns as scoped review questions until
that evidence exists. Run the scanner regression suites without a GPU:

```bash
python3 tests/unit/test_review_legacy_spelling.py
python3 tests/unit/test_review_unreachable_tests.py
```

The same tests are collected under `tests/unit/` by `scripts/run_tests.sh`.
