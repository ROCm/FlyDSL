# Selection evidence and limits

## Historical snapshot

The [original August 20 version](https://github.com/ROCm/FlyDSL/blob/eb61f68c6be23fe1624abe7fe44dfaa47e1d42ff/.claude/skills/review-flydsl-kernel/SKILL.md)
reported 479 human review comments across 163 of 865 FlyDSL PRs. It grouped
repeated feedback into six candidate families. These counts are the original
author's corpus report; the September refresh did not reproduce the full census.

Two checks survived the selection process:

- **Legacy API spellings (F1):** the historical corpus attributed 46 comments to
  this family. Its replay reported candidates in 10 of 12 files where a
  maintainer had raised the issue. Prompt-only review variants did not reliably
  surface the same candidates, which supports running the mechanical check before
  semantic review.
- **Tests unreachable from script entry points (F5):** in a seeded comparison,
  the baseline reviewer caught four of five candidate families. Adding the F5
  check changed the result from 4/5 to 5/5. Replaying PR
  [#481](https://github.com/ROCm/FlyDSL/pull/481) at
  `7e117e29c9c4582158aed33da4ee6dfe14d2c76e` reports the same three
  fused/quant test definitions omitted from that file's script path that the
  maintainer identified.

F2 default-buffer aliasing, F3 duplicated paths, F4 in-kernel configuration, and
F6 unchecked C-ABI casts were already caught by the baseline reviewer in the
seeded comparison. They are not active rules here because the experiment did not
show incremental coverage.

Small seeded diffs expose one defect more directly than large real PRs. The
comparison had no held-out real-PR set, so it does not establish current
precision, recall, or general incremental benefit.

## September 10 refresh

The refresh fetched all pages of `repos/ROCm/FlyDSL/pulls/comments` with
`since=2026-08-20T00:00:00Z&per_page=100`, then filtered creation timestamps to
the configured window. It found 233 inline comment records on 38 PRs, including
author replies and bot output; these are not 233 independent objections and
must not be added to the historical count.

Recent maintainer comments continued to support checking typed/control-flow
spellings: coderfeli requested `arith` cleanup in
[#1067](https://github.com/ROCm/FlyDSL/pull/1067#discussion_r3860524512),
removal of `ArithValue` in
[#1066](https://github.com/ROCm/FlyDSL/pull/1066#discussion_r3919994660),
and ordinary `if/else` in the
[same review](https://github.com/ROCm/FlyDSL/pull/1066#discussion_r3920001758).
These comments support candidate generation; the current API and semantics still
decide whether a particular replacement is valid.

## Promotion standard

A scanner match is not a defect. Before reporting one, inspect the complete head
source, compare the base, and identify the current API or test-entry contract.
Aliases, partial diff context, dynamic dispatch, fixtures, plugins, and runtime
conditions remain outside these static checks.

Add another check only with a reproducible positive case, a clean control, and
evidence that it catches something the existing reviewer misses. Preserve the
inputs and observed difference before claiming improved review coverage.
