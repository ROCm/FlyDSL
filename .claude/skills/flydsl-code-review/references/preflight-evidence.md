# Why these two checks

The scanners and their CPU regressions are adapted from
[PR #1047 at `2480f877`](https://github.com/ROCm/FlyDSL/tree/2480f877aa7f74a24d1ce857b48ba92ff0699cf5/.claude/skills/flydsl-review-checks).
They supplement the existing review runner; their output is evidence to inspect,
not an independent review verdict or another skill entry point.

That PR's [selection record](https://github.com/ROCm/FlyDSL/blob/2480f877aa7f74a24d1ce857b48ba92ff0699cf5/.claude/skills/flydsl-review-checks/references/review-evidence.md)
reports two useful signals:

- Legacy API spellings were recurring maintainer feedback, and prompt-only
  reviewers did not reliably surface the same candidates.
- In a small seeded comparison the baseline caught four of five candidate
  families. Checking script entry paths supplied the missing family. A replay
  of [#481](https://github.com/ROCm/FlyDSL/pull/481) at
  `7e117e29c9c4582158aed33da4ee6dfe14d2c76e` identified three added fused/quant
  tests omitted from the file's script path.

Those are historical and seeded selection results, not held-out precision,
recall, or proof of incremental benefit over this runner. Other proposed rule
families already had baseline coverage in that experiment, so they are not
duplicated here. Existing technical skills continue to own their semantic rules.

A new deterministic check should have a reproducible positive case, a clean
control, and evidence of coverage beyond the existing reviewer. Preserve the
inputs and outputs before claiming a review-quality improvement.
