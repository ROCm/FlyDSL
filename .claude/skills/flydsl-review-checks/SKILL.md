---
name: flydsl-review-checks
description: >-
  Add two deterministic candidate checks to an existing FlyDSL kernel review:
  legacy API spellings on added kernel lines and added tests without a visible
  script entry path. Use after the caller has pinned a diff and matching head
  checkout; this skill does not retrieve PRs or replace the full review.
argument-hint: --diff <file> --head <dir>
---

# FlyDSL review checks

Use these checks as a preflight inside an existing review workflow. The caller
owns PR retrieval, base/head selection, checkout isolation, correctness and API
review, runtime validation, and any published review. Run the checks once against
the same pinned diff and head checkout used by that workflow.

## Run

```bash
skill_dir="$PWD/.claude/skills/flydsl-review-checks"
python3 "$skill_dir/check_candidates.py" \
  --diff "$review_diff" \
  --head "$head_worktree"
```

The combined command always runs both scanners:

- exit `0`: no candidates in either supported scope;
- exit `1`: at least one candidate needs source review;
- exit `2`: an input or scanner failed, so the preflight is incomplete.

Exit `0` is not a clean PR verdict. Continue the normal semantic, API,
correctness, performance, and test review regardless of scanner output.

## Adjudicate candidates

Read each reported location in the head checkout and compare it with the base.
Promote a candidate to a finding only after establishing the concrete consequence
and the applicable current API or test command. Group repeated legacy spellings
that have the same action into one finding.

### Legacy API spellings

`scan_legacy_spelling.py` checks added lines in `kernels/**/*.py`, excluding
`kernels/common/buffer_ops.py`. It filters comments, strings, and imports visible
in each diff hunk.

It reports these spelling families:

- raw `ir.*`, `_mlir.*`, `ArithValue`, and explicit unwraps;
- `scf.*` control-flow builders;
- `buffer_ops.*`;
- `SmemAllocator`;
- `make_ptr(...)`, which is only actionable when it retypes an existing pointer.

The scanner does not resolve import identity or prove a replacement is equivalent.
Raw MLIR can be required at an implementation boundary, ordinary pointer
construction is valid, and partial diff context can hide multiline lexical state.

### Added tests and script entry points

`scan_unreachable_tests.py` uses the head AST to match added `test_*` definitions
and `Test*` methods with statically visible `__main__` paths. It follows direct
local calls and recognizes unfiltered `pytest.main([__file__])` dispatch with
common aliases.

Before reporting a gap, identify the command that is supposed to execute the
test. Pytest correctness coverage and a benchmark script's direct call path are
different contracts. Dynamic dispatch, selectors, fixtures, plugins, decorators,
and runtime branches require manual inspection.

## Evidence boundary

[review-evidence.md](references/review-evidence.md) records why these two checks
were retained and why the other candidate rule families were not. Historical
frequencies and seeded comparisons are selection evidence, not current precision
or recall. A new check requires a reproducible positive case, a clean control, and
evidence that it adds coverage beyond the existing reviewer.
