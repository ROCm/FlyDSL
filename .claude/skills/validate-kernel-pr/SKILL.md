---
name: validate-kernel-pr
description: >
  Reproducible validation executor for FlyDSL kernel PRs. Applies an explicit base-to-head
  patch in an isolated worktree, runs it on a verified-idle GPU, and compares correctness
  AND performance against the same base under a measured noise floor. Emits a head-bound
  validation_report.json for review-pr to consume. Use when asked to validate a FlyDSL PR,
  check for a performance regression, or produce deterministic evidence for a review.
allowed-tools: Read Bash Grep Glob
---

# validate-kernel-pr — deterministic evidence for FlyDSL PRs

`review-pr` is static. It never builds and never runs, so three failure classes are
invisible to it:

1. **The PR's own tests pass while the kernel is wrong.** A suite whose non-aligned shapes
   are commented out reports green on an out-of-bounds tail store.
2. **A green suite that cannot fail.** A widened tolerance leaves every test passing and
   the kernel unguarded.
3. **A performance regression.** This one is not merely invisible to `review-pr` — it is
   invisible to FlyDSL's CI as well.

## Why the performance stage exists

`.github/workflows/flydsl.yaml` runs `scripts/run_benchmark.sh` and then
`scripts/compare_benchmark.py`. That comparator prints ratios and ends with:

```python
    print("\nBenchmark comparison report completed.")
    return 0
```

`return 0`, unconditionally. **No performance regression can turn a FlyDSL PR red.** The
numbers are printed into a job log that nobody is required to read, and there is no
threshold anywhere in the pipeline. PR #1009 is what that costs: a default-off trait
placed in front of the bf16 XCD remap shipped a silent regression that lived in main for
two weeks, and restoring the mapping recovered +17.4% and +19.4% at two production
sequence lengths with bit-identical output.

This executor supplies the missing gate.

---

## Invocation

```bash
# 1. pin the PR identity and put its base in an isolated worktree
BASE_REF=$(gh pr view "$PR" --repo ROCm/FlyDSL --json baseRefName --jq .baseRefName)
BASE=$(gh api "repos/ROCm/FlyDSL/branches/$BASE_REF" --jq .commit.sha)
HEAD=$(gh pr view "$PR" --repo ROCm/FlyDSL --json headRefOid --jq .headRefOid)
git worktree add --detach /tmp/fly-base "$BASE"
gh pr diff "$PR" --repo ROCm/FlyDSL > /tmp/pr.patch

# 2. validate base and head under the same runner and the same claimed GPU
python3 .claude/skills/validate-kernel-pr/validate_pr.py \
    --repo /tmp/fly-base \
    --patch /tmp/pr.patch \
    --head-sha "$HEAD" \
    --tests tests/kernels/test_softmax.py \
    --bench-cmd "bash scripts/run_benchmark.sh" \
    --perf-rounds 5 \
    --out validation_report.json
```

For a local candidate with no remote head, omit `--head-sha`. The report then records
`repo.head: null`; it stays useful locally, but `review-pr` will refuse it as PR evidence.

| flag | meaning |
|---|---|
| `--repo` | clean checkout at the PR base commit (required) |
| `--patch` | base-to-head patch; a conflict is a blocker |
| `--head-sha` | exact remote PR head the patch represents |
| `--tests` | correctness target; the runner is detected from the file |
| `--bench-cmd` | benchmark command run inside each side's worktree |
| `--perf-rounds` | A/B/A rounds; each round runs base, head, base (default 5) |
| `--min-effect` | smallest change treated as real before noise is considered (default 3%) |
| `--lower-is-better` | the metric is latency, not throughput |
| `--metric-format` | `flydsl-table` (the repo's own 5-column output) or `regex` |

---

## Stages

Each stage writes its own status. A stage that could not run says `skip` with a reason; it
never reports `pass` for work it did not do.

### 1 — `merge_sim`

Apply the patch onto the base commit. A conflict short-circuits everything: no number
produced downstream would describe the merged code. Records the base commit, the patch
SHA-256, and the caller-supplied head OID.

### 2 — `gpu_claim`

Claims a GPU over a **sampling window**, not a single reading — a neighbouring job between
kernel launches reads 0% exactly like an unused device. Every sample must stay under the
threshold. Records BDF, arch, and the full activity trace.

If nothing stays idle, `degraded_mode` is `NO_GPU`, correctness and perf are `skip`, and
the verdict is `INCONCLUSIVE`. It does not claim `compile-only`, because no
architecture-specific compile was attempted.

### 3 — `runtime_compat`

Does the checkout's own `flydsl` import against the runtime actually installed? FlyDSL
kernels import symbols from a compiled runtime, so a prebuilt package that drifts behind
the tree raises an `ImportError` **that looks exactly like a defect in the PR**.

This is not hypothetical. On a machine where `build-fly/python_packages/flydsl` is 0.3.1
while `python/flydsl` is 0.3.2, every kernel import dies on
`cannot import name 'get_warp_size' from 'flydsl.runtime.device'` — a symbol PR #1024
added. A validator without this stage would run the tests, collect that error, and file a
red result against an innocent author. A version or symbol mismatch is an **environment
fact**: correctness and perf are skipped, the verdict is `INCONCLUSIVE`, and nothing is
attributed to the PR.

### 4 — `test_policy` — runs **before** the suite

A suite that cannot fail is worse than no suite, because it reports green.

- **Tolerance, compared head-vs-base.** A test-only widening is a deterministic blocker. A
  widening alongside a kernel change is `NEEDS_WORK` pending numerical justification,
  rather than a false block.
- **Shape rows newly disabled** by this change are `NEEDS_WORK`.

### 5 — `correctness`

**The runner is detected, not assumed.** FlyDSL's `tests/kernels/*.py` carry both pytest
functions and a `__main__` guard, and CI uses each in a different job. A target with a
`test_*` function runs under pytest; one with only `__main__` runs as a script; one with
neither is a `skip` with a stated reason, never a `fail` — the absence of a runnable test
is a fact about the PR's packaging and belongs in `review-pr`'s advisory judgement, not
disguised as a reproducible kernel defect.

Base and head run the same target under **separate JIT cache directories**. Sharing one
would let the base run populate an artifact the head run then loads, and the comparison
would silently measure the same kernel twice.

### 6 — `perf` — the stage FlyDSL's CI does not have

**Design: A/B/A interleaved, with the noise floor measured during the run.**

Each round runs base, then head, then base again:

- Sandwiching head between two base runs cancels monotonic drift. A design that ran all of
  base and then all of head would charge every clock ramp and every neighbouring tenant to
  the patch.
- The two base runs form a genuine **A/A control**. Their disagreement *is* this machine's
  noise floor for this row, measured now — not a threshold guessed in advance. A
  head-vs-base delta smaller than the control's own disagreement is not evidence.

Why a fixed threshold fails here, in the repo's own words (`scripts/run_benchmark.sh`):

> M is chosen for occupancy, not just tier coverage: the widest tier at M=64 fills 0.25
> workgroups per CU on a 256-CU gfx950 and **swings 24% run to run**, which would flap the
> dashboard rather than report it. At M=1024 the same tier holds ~1%.

A 5% gate would call that row a regression on roughly every other run. The measured floor
absorbs it, and the regression tests pin that behavior across twelve seeds.

Per row the report records `base_median`, `head_median`, `gain`, `change_pct`,
`noise_floor`, `control_deviation`, `base_spread` and both sample counts. A row present on
only one side is `incomparable` — never silently "fine".

**Two FlyDSL-specific hazards the stage handles:**

- **Warm-cache measurement.** `CLAUDE.md`: the JIT disk cache "normally invalidates on
  kernel source and closure changes. Disable it when debugging stale artifacts, changing
  C++ passes, or changing helper code that is not part of the traced closure." So when the
  patch touches `lib/`, `include/`, `python/flydsl/`, `kernels/common/` or `tools/`, the
  key does not move with the change and a warm cache would serve the **previous** kernel —
  producing a perfectly reproducible measurement of the wrong binary. The executor detects
  those paths and forces `FLYDSL_RUNTIME_ENABLE_CACHE=0`, recording the decision and its
  reason in `environment.cold_cache_reason`.
- **Autotune nondeterminism.** `FLYDSL_AUTOTUNE=0` by default, matching
  `scripts/run_tests.sh`, so a tuning search does not masquerade as a kernel change.

**Metric parsing keeps rows labelled.** PR #654 shipped a parser that looped
`for m in re.finditer(...): pass` and kept the *last* match, so layernorm was reported at
1.69 TB/s for months against a real 5.6 — and the current-vs-main gate could not catch it,
because main was mislabelled the same way. Parsing into a dict keyed by `op|shape|dtype`
makes that collapse impossible, and a regression test pins it.

### 7 — `diff_scan`

Runs `review-pr/scan_flydsl_diff.py` over the patch and records the candidate counts.
Informational: candidates, not verdicts.

### 8 — verdict

`BLOCK` if a reproducible defect fired (including a perf regression), `NEEDS_WORK` for a
deterministic policy concern, `INCONCLUSIVE` if any required stage did not complete, else
`PASS`. **`PASS` therefore means the perf stage actually ran** — a skipped perf stage can
never be reported as `PASS`, and there is a regression test for exactly that.

---

## Honesty rules the report enforces

These are fields, not prose, so a report cannot overclaim by omission.

- **`stages`** — every declared stage is always present. A stage that did not run is an
  object with `status: skip` and a reason; it never disappears and never becomes a string.
- **`environment.cold_cache_required` / `cold_cache_reason`** — whether the measurement
  could have been served by a stale artifact, and why the executor decided so.
- **`environment.isolation`** — the real level. Where no container runtime is available it
  is `git-worktree + private JIT caches`, and `container` is `false`.
- **`degraded_mode`** — `NO_GPU` when nothing was claimable; required stages then force
  `INCONCLUSIVE`.
- **`test_selection`** — the exact target, runner, benchmark command and round count. A
  verdict applies only to those named inputs.
- **`perf` rows keep their provenance** — every number carries the noise floor it was
  judged against and the control deviation that produced it.

---

## Regression assets

Every stage here has been observed failing on a seeded defect **and** passing on a matched
control. That pairing is the point: a stage that has only ever been observed passing is
decoration, not a check.

```bash
python3 -m pytest .claude/skills/validate-kernel-pr/tests/test_validator.py -q
```

Covered:

| Property | Seeded defect | Negative control |
|---|---|---|
| Perf regression detected | head 20% and 6% slower | identical code across 12 seeds stays green |
| Noise not mistaken for signal | — | 24% run-to-run noise across 12 seeds never blocks |
| Drift cancelled by A/B/A | real regression under 3%/round drift | drift alone is not a regression |
| Unresolvable case reported honestly | 5% loss under 24% noise | reported `unchanged`, not a coin flip |
| Direction correct (PR #848 shipped inverted columns) | latency vs throughput both ways | — |
| Row labels preserved (PR #654) | two rows, same metric name | last row must not overwrite the first |
| Cold cache required | `lib/`, `python/flydsl/`, `kernels/common/` | a leaf kernel does not force it |
| Cache isolation | — | base and head get different cache dirs |
| Test-only tolerance widening blocks | widening with no kernel change | widening with a kernel change is `NEEDS_WORK` |
| Incomplete run cannot claim PASS | perf skipped | all stages passed |
| End-to-end perf wiring | seeded −20% through real subprocesses | identical command passes |

---

## Not implemented yet

Deliberately absent rather than half-built. Everything shipped above has been observed
failing on a seeded defect; these have not been:

- **PR fetch orchestration.** There is no `--pr N`; the caller creates the worktree as
  shown. Choosing the right `--tests` and `--bench-cmd` from a diff is the unsolved part,
  and an irrelevant target can still produce `PASS`. The report names both so a reviewer
  can reject that evidence, but the executor cannot judge relevance itself.
- **Cross-architecture coverage.** The executor validates the architecture it has a device
  for and says so in `environment.arch`. It never claims coverage of another. Note that no
  runner in FlyDSL's CI matrix is gfx1250, so gfx1250 PRs have no automated coverage at
  all — from CI or from this executor without such a device.
- **Building the base wheel against the base LLVM pin.** PR #1071 does this in CI. Here,
  if the patch changes the LLVM pin, both sides must be rebuilt or the comparison is
  meaningless; the executor does not yet detect that and cannot yet rebuild.

## What this skill does not do

- It does not replace `review-pr`. It produces evidence; the judgement stays there.
- It does not write findings about design, style, or API shape.
- It does not merge or publish a decision. A `BLOCK` is reproducible executor evidence;
  `review-pr` keeps its separate advisory verdict.
- It does not validate an architecture it has no device for.

---

## Adding a stage

A new stage must be able to **fail on a seeded defect**. Before adding one: seed the defect
it is meant to catch, confirm the stage goes red, confirm the clean baseline stays green,
and add both to `.claude/skills/validate-kernel-pr/tests/test_validator.py`. A stage that has never been observed failing is
not a check.
