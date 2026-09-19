---
name: flydsl-code-review
description: >
  Review a FlyDSL diff, branch, commit range, or PR for correctness bugs and
  convention violations using the repository's existing skills and policy docs.
  Uses one resumable runner to pin the reviewed tree, run deterministic checks
  and all nine review angles, optionally in one grouped pass, verify every candidate,
  and preserve the evidence.
  Pass --comment to publish a completed PR review. Use when asked to review a diff,
  review a PR, or check changes before pushing.
allowed-tools: Read Bash
---

# FlyDSL Code Review

Find real defects in a change, then prove each one before reporting it.

The sole execution entry is `.claude/skills/flydsl-code-review/scripts/run_review.py`. It runs a deterministic preflight, independent or grouped
finders, one verifier per candidate, a challenger for every CONFIRMED, and a fresh sweep.
Code constructs the final ranked report directly from those records. The sections
below supply its review method; they are not an alternative manual execution path.

## Invocation

```text
/flydsl-code-review                       # current branch vs main, including local changes
/flydsl-code-review HEAD~3                # a ref or ref range
/flydsl-code-review 1100                  # a PR number
/flydsl-code-review kernels/attention/    # restrict to a path
/flydsl-code-review focus on the LDS changes   # free-form instruction
/flydsl-code-review 1100 --comment         # publish confirmed P0/P1 findings
/flydsl-code-review 1100 --comment --publish-severity P0
```

Pass only review scope and runner options to `run_review.py`; `--comment` and
`--publish-severity` belong to `post_review.py` after a COMPLETE result. Honor
user scope restrictions verbatim. Use `--instructions` for free-form focus,
`--path` for paths, and explicit `--base`/`--head` for a frozen comparison.

## Step 1 — Run the review

Invoke the runner with Bash from the repository root:

```bash
python3 .claude/skills/flydsl-code-review/scripts/run_review.py 1100
python3 .claude/skills/flydsl-code-review/scripts/run_review.py --base HEAD~3 --head HEAD
python3 .claude/skills/flydsl-code-review/scripts/run_review.py --path kernels/attention --instructions 'focus on LDS'
python3 .claude/skills/flydsl-code-review/scripts/run_review.py 1100 --group-finders
python3 .claude/skills/flydsl-code-review/scripts/run_review.py --resume /tmp/flydsl-review-<run-directory>
```

Do not invoke Workflow or improvise an inline Agent sequence. The runner requires
Python 3.10+, Git, and Claude Code CLI with `--json-schema`; direct PR targets need
authenticated `gh`, while offline bot scopes use a trusted manifest and local objects.

The runner prints its run ID and directory immediately. Its default directory is
under `/tmp`; use `--run-dir <new-empty-directory>` outside the checkout for
longer-lived artifacts.
It copies the required Git objects into an independent checkout, fixes base,
merge-base and head OIDs, and hashes the diff. Default branch/path reviews include
local tracked and untracked changes in a synthetic commit there. Explicit refs
and PRs review committed trees. Every agent reads that checkout and fixed diff;
read the enclosing functions as well as changed lines. No phase rereads a moving
PR diff or the caller's working tree.

Defaults are 3 concurrent agents, 600 seconds per agent and 1800 seconds per
phase including queue time. Override with `--concurrency`, `--agent-timeout` and
`--phase-timeout`. Ctrl-C/SIGTERM cancels child process groups. Completed stages
and all attempt logs are checkpointed in `state.json`; `--resume` retries only
incomplete stages with the saved scope, model and configuration. Changed runner,
scanner or skill content requires a new run. Model and effort use the CLI defaults unless
the user supplies `--model`/`--effort`; do not silently select a different model.

The untrusted-container profile gives model sessions only Read, Grep, and Glob.
They read the authoritative saved `diff.patch` directly and cannot invoke Bash
or code from the reviewed tree. Small arithmetic is written out by the finder
and independently recomputed during adjudication; without executable evidence,
an arithmetic-dependent verdict remains PLAUSIBLE.

Read `result.json` after the runner exits. Exit 0 means COMPLETE; exit 1 means
INCOMPLETE. A missing result, running process, task notification or partial
transcript is not a completed review. Preserve the run directory when reporting
an interruption so the user can resume it.

## Deterministic preflight

After pinning a nonempty diff, the runner runs two source scanners from its own
`scripts/` directory against that diff and checkout. Both are required stages:
exit `0` means no leads in the supported scope, `1` means leads need inspection.
The runner requests `--json` and requires a COMPLETE result whose exit code
matches the process exit code. An exception, missing/malformed result or timeout
makes the review INCOMPLETE; exit code `1` alone never proves success. Artifact
schema v6 requires this completion record, grouped-finder provenance, verified
severity, and offline bot scope identity.
The artifact retains each scanner's output, exit status and run history; resume
reuses completed checks.
Neither scanner executes or imports the reviewed code.

- **Angle G:** `scan_legacy_spelling.py` checks added `kernels/**/*.py` lines,
  excluding `kernels/common/buffer_ops.py`, for raw IR/unwraps, SCF builders,
  `buffer_ops.*`, `SmemAllocator` and `make_ptr`. It filters visible comments,
  string literals and imports, but does not resolve API identity. Ordinary
  pointer construction and raw IR at implementation boundaries can be valid.
- **Angle I:** `scan_unreachable_tests.py` matches added test definition lines
  against the head AST, follows direct local calls from `__main__`, and recognizes
  unfiltered `pytest.main([__file__])` with common aliases. Before reporting a gap,
  identify the actual test or benchmark command: pytest coverage and a script's
  direct call path are different contracts.

The corresponding finder receives the full raw output. Inspect the complete
source and the applicable policy at the reviewed revision before promoting a
lead to a candidate. Group repeated spellings with the same corrective action;
the six-candidate limit applies to the resulting defect candidates. Raw leads
are retained for inspection, not individually certified as bugs or as clean.
All promoted candidates go through the same independent verifier and challenge
rules as other candidates. An empty scan never skips the semantic review.

Aliases, partial diff lexical context, dynamic dispatch, runtime branches,
pytest selectors, fixtures, decorators and plugins can require manual review.
See [preflight-evidence.md](references/preflight-evidence.md) for the selection
evidence and its limits; these checks do not establish review precision or recall.

## Reusing existing skills

The linked skills and policy docs own the technical rules. Each angle below
selects the relevant sources; read their applicable sections at the reviewed
revision before judging a candidate. Follow their conditions, exceptions, and
semantic constraints rather than treating the angle's topic labels as rules.
If linked guidance disagrees, check the policy and implementation at that
revision before reporting.

Reuse their analysis or check-only procedures within the scope from Step 1,
leaving the reviewed files unchanged. Authoring, migration, formatting fixes, and
intrusive debugging are separate tasks. Use this skill's candidate format,
verification, and final report instead of concatenating standalone skill reports.
A scoped review does not establish a full API audit's PASS or STABLE-ONLY result.

For a finding based on a shared rule, cite the source file and section in
`failure_scenario`, alongside the code evidence and concrete consequence.
Verifiers and challengers must read that source and check its applicability.

## Severity and publication

Severity measures impact **if the candidate is true**; verdict measures confidence. Finders propose
severity, but the verifier owns it and a challenger may only keep or lower it.

- **P0** — broad critical failure: widespread silent corruption, an unsafe boundary, or a primary path unusable with no practical escape.
- **P1** — concrete merge blocker on a supported path: wrong output, OOB, crash/hang, API break, required CI failure, or measured contract regression.
- **P2** — real but non-blocking localized defect, unproven performance concern, or bounded diagnostic/test/documentation/convention gap.
- **P3** — optional cleanup or maintainability improvement with no demonstrated present correctness, compatibility, CI, or performance effect.

`result.json` retains every candidate and adjudication. GitHub publication defaults to CONFIRMED
P0/P1, filtering the full verified set before its 12-item cap. `--publish-severity
P0|P1|P2|P3` changes the threshold; PLAUSIBLE and lower severities remain artifact-only.

## Step 2 — Run the nine angles

By default the runner starts nine independent finders, one per angle, with up
to 6 candidates each. `--group-finders` instead gives all nine checklists to one
finder so source reads are shared. Its structured result must acknowledge every
angle and assign each candidate one primary owning angle; the grouped limits are
**up to 6 candidates per angle and 12 overall**. Group observations with the
same root cause instead of duplicating them across angles; keep genuinely
different mechanisms even on the same line. Each candidate needs a
repository-relative `file`, a positive integer `line` (or null), a one-line
`summary`, a specific `mechanism`/root cause, `severity` (P0–P3), and a concrete
`failure_scenario`.

Angles A–F hunt correctness bugs. Angles G–I hunt convention violations and
cleanup; for those, `failure_scenario` states the concrete cost (what breaks in
CI, what is duplicated, what becomes arch-fragile) rather than a crash.

## Angle A — trace-time vs runtime semantics

Read the **flydsl-kernel-authoring** skill
([SKILL.md](../flydsl-kernel-authoring/SKILL.md)), §3: **Control Flow**,
**Runtime vs Compile-Time Conditions**, **Frontend Semantic Restrictions**, and
**Runtime Loops with Loop-Carried Values**. The **debug-flydsl-kernel** skill
([SKILL.md](../debug-flydsl-kernel/SKILL.md)), §6 **Compilation Errors**, supplies
concrete failure patterns.

Trace values from their definitions to uses across branches, helpers, and loop
boundaries. Identify where the changed code violates those frontend contracts.

## Angle B — memory addressing and out-of-bounds

Use the **oob-detection** skill ([SKILL.md](../oob-detection/SKILL.md)), §1
**Classify the OOB** and §2 **Static Interval Analysis**, plus §4's layout and
integer-overflow guidance. For raw buffer access, read the offset contract in the
**kernel-code-cleanup** skill ([SKILL.md](../kernel-code-cleanup/SKILL.md)), §2.

Apply the analysis to changed accesses and their corresponding writer/reader
layouts. Connect the failing range to an observable result under Step 3.

## Angle C — synchronization, LDS, and value lifetime

Read the sources relevant to the changed synchronization or storage:

- **debug-flydsl-kernel** skill ([SKILL.md](../debug-flydsl-kernel/SKILL.md)),
  §7.2 **Barrier deadlock**.
- **lds-optimization** skill ([SKILL.md](../lds-optimization/SKILL.md)),
  **LDS Instruction Model**, for dependency and cross-wave synchronization.
- **flydsl-tile-programming** skill ([SKILL.md](../flydsl-tile-programming/SKILL.md)),
  **Step 5: Add Synchronization**, for target-specific wait operations.
- **kernel-code-cleanup** skill ([SKILL.md](../kernel-code-cleanup/SKILL.md)),
  §3c and §4, for wait-counter migration constraints and shared-view lifetime.
- **flydsl-kernel-authoring** skill ([SKILL.md](../flydsl-kernel-authoring/SKILL.md)),
  §5 **Shared Memory (LDS)**, and [CLAUDE.md](../../../CLAUDE.md)'s
  **GPU Architecture Support** and **Kernel Authoring Conventions**, for
  allocation, launch, and capacity contracts.

Trace producer/consumer ordering and value lifetime across branches, loops,
pipeline stages, and the launch boundary.

## Angle D — architecture and atom contracts

Read [CLAUDE.md](../../../CLAUDE.md)'s **GPU Architecture Support** and the
**flydsl-kernel-authoring** skill ([SKILL.md](../flydsl-kernel-authoring/SKILL.md)),
§6 **MFMA Integration**, for target capabilities and operand contracts. Reuse the
**kernel-code-cleanup** skill ([SKILL.md](../kernel-code-cleanup/SKILL.md)),
§3b and §§6–7, for pointer boundaries, fragments, and TV layouts.

When the diff implements backend atoms, also apply the **add-target-atom-op**
skill ([SKILL.md](../add-target-atom-op/SKILL.md)), §1 **Inherent Design** and
§6 **Review and verification**.
Check lane math, dtype support, dispatch, and operand/layout assumptions against
every target the changed code claims to support.

### Compiler target decisions

For compiler-side target selection, identify each target property's owner and every architecture
admitted by the predicate. Compare wave size, dialect/intrinsic support, address-space mapping, and
pass options with the selected backend and `CLAUDE.md`; do not infer one property from another
classification. Name an admitted target and the wrong emitted IR, option, diagnostic, ISA, or result.

## Angle E — removed-behavior auditor

For every line the diff **deletes or replaces**, name the invariant or behavior
it enforced, then search the new code for where that invariant is
re-established. If you cannot find it, that is a candidate.

In this repo the recurring instances are: a dropped bounds guard or mask; a
removed `s_waitcnt` or `gpu.barrier()`; a NaN or divide-by-zero guard removed
during a refactor; a narrowed dtype or arch validation; a `.mlir` FileCheck line
deleted rather than updated; a test case deleted because it started failing.

## Angle F — cross-layer tracer

Two directions.

**Horizontal.** For each changed function, search its callers and callees for
assumptions the diff invalidates, including interactions between changed
functions. Report concrete broken call sites; Angle G owns public-API stability
classification.

**Vertical.** Trace changed operations through definitions, lowering, bindings,
and consumers. Use the **flydsl-kernel-authoring** skill
([SKILL.md](../flydsl-kernel-authoring/SKILL.md)), §1 **Architecture and
Compilation**, as the layer map. For atom changes, use the **add-target-atom-op**
skill ([SKILL.md](../add-target-atom-op/SKILL.md)), §2 **The Files You Will
Touch** and the applicable recipe's integration and verification steps.

Check that the affected layers, supported targets, and FileCheck expectations
remain consistent with the changed behavior.

### Compiler, dialect, and conversion changes

Classify the changed boundary: target-neutral compiler protocol/Fly interface
or transform; backend-shared ROCm pipeline/conversion; or target-specific
FlyROCDL payload and `expr/rocdl` factory. Backend payloads, intrinsics, address
spaces, and chip options must not leak into neutral owners. Inspect every
in-tree implementation/consumer of a neutral contract and every target admitted
by a changed backend dispatch.

Trace the contract through Python producer, TableGen verifier/type inference,
neutral transforms, type conversion/legality, backend payload and intrinsic,
pass registration/order, final ISA, and observable output. Derive scenario
families from real predicates and overloads: applicable static/dynamic,
scalar/vector, pointer/memref, predicate, reachable memref/full-SSA/mixed,
supported/rejected type/layout, shape boundary, and target dispatch. Cover one
reachable representative per distinct path and interacting pairs, not an
irrelevant Cartesian product; name the uncovered family and prove it reaches
the changed assumption.

## Angle G — repo conventions and API stability

- **Kernel conventions.** Read the **kernel-code-cleanup** skill
  ([SKILL.md](../kernel-code-cleanup/SKILL.md)), **Cautions** and §10's review-only
  **Find / Triage** procedure. Apply the relevant replacement tables and their
  semantic constraints, including §3, to the resolved usages.
- **API stability.** Read the **api-stability** skill
  ([SKILL.md](../api-stability/SKILL.md)). Apply §1 **Producer review** to public
  surface changes and §2 **Consumer review** to FlyDSL usage in the review scope.
  Use its classifications, severity rules, and evidence requirements. Keep
  stability judgments distinct from kernel-migration preferences.
- **Formatting and lint.** For a style question or suspected style-gate failure,
  use the **format-code** skill ([SKILL.md](../format-code/SKILL.md)),
  **Check only**. Report which part of the review scope the check actually covers.
- **Other repo conventions.** Read [CLAUDE.md](../../../CLAUDE.md)'s **Kernel
  Authoring Conventions** and **Environment Variables**. For pre-check changes,
  check registration against `scripts/check_repo.py` and
  `.github/workflows/pre-checks.yaml`.

CI independently enforces a subset of the arithmetic rules on added kernel
lines with `scripts/check_typed_arithmetic_usage.py`, invoked through
`scripts/check_repo.py`. Its AST scan is optional corroboration for a CI-failure
claim; it is not a required review step.

## Angle H — reuse, simplification, and altitude

Use [CLAUDE.md](../../../CLAUDE.md)'s **Kernel Authoring Conventions**, especially
**Helper placement**, for reuse and module ownership. Read the
**kernel-code-cleanup** skill ([SKILL.md](../kernel-code-cleanup/SKILL.md)),
§8 **Trim comments and dead code** and §9 **Cut launch overhead**, for cleanup
criteria and performance suggestions.

Search for an existing implementation before proposing reuse; name the helper
and its home. Assess redundant state, copy-paste, and new abstractions against
their actual call sites. For a special case added to shared infrastructure,
identify whether the underlying mechanism should handle it generally and state
the concrete maintenance or performance cost.

### Compiler extension generality

Search sibling overloads, interfaces, type converters, mapping tables, and
callers before accepting a local special case. Put the rule at the lowest layer
that owns the invariant; use backend data/interfaces instead of arch strings in
neutral code. Check paired overloads for drift, rewrites for a decreasing
measure and multi-use values, and unsupported states for an early verifier or
diagnostic. “Could be more general” alone is not a finding: name the missed
sibling/caller, duplicated owner, non-converging rewrite, or reachable invalid IR.

## Angle I — test and documentation contract

Read `tests/README.md` and `tests/pytest.ini` for tier, backend, and marker
contracts, and [CLAUDE.md](../../../CLAUDE.md)'s **Testing Notes** and **Kernel
Entry Points** for multi-GPU requirements and new-kernel test/documentation
coverage. Check new or moved tests against their actual dependencies and device
requirements.

Audit test observability, not only execution count. Count how many distinct
operation results can affect an assertion. In loops, graph capture/replay, or
repeated launches that write the same buffer, check whether a later write
overwrites earlier results before they are observed; replaying ten operations
but asserting only the final buffer is one observation, not ten.

For changed atoms, use the **add-target-atom-op** skill
([SKILL.md](../add-target-atom-op/SKILL.md))'s applicable verification steps.
For other changed ops or lowerings, inspect the corresponding FileCheck coverage
described in `tests/README.md`.

Do not flag general "needs more tests" — only these specific contract breaks.

### Compiler regression coverage

Map changed predicates, overloads, legality rules, and target dispatch to tests.
For each distinct reachable path, identify a test or prove another case exercises
the same branch. Include the trigger, unaffected boundary, sibling-target
control for shared code, and positive/negative verifier or dispatch cases. Use
`tests/mlir/Transforms` for neutral rewrites,
`tests/mlir/Conversion` for ROCm lowering, Python unit/system tests for tracing,
protocol/cache behavior, and L2 only for a hardware-observable contract. Verify
each MLIR `RUN` line reaches the changed pass and FileCheck asserts the semantic
invariant and absence of the old failure.

---

## Step 3 — Verify every candidate

The runner deduplicates only on the same normalized file, exact line and mechanism,
preserving all source observations. Different mechanisms on nearby or identical
lines remain distinct. It verifies every remaining candidate, without a shared
admission budget, in a deterministic order that puts correctness first.

Each assigned **independent verifier** receives the owning angle text, reads the
fixed diff and relevant files, and returns one verdict, one independently assigned
severity, and evidence. A verifier judges its assigned candidate; it does not
launch other agents.

- **CONFIRMED** — can name the inputs, state, or target that trigger it and the
  resulting wrong output, crash, hang, or CI failure. Quote the line.
- **PLAUSIBLE** — the mechanism is real but the trigger is uncertain (timing,
  architecture, config, shape). State what would confirm it.
- **REFUTED** — factually wrong, or already guarded. Quote the line that proves it.

**Default to PLAUSIBLE.** Do not refute a candidate for being "speculative" or
for "depending on runtime state" when the state is realistic. On a GPU the
following are all PLAUSIBLE, not REFUTED: a race between waves, an OOB on a
boundary tile the code does not exclude, a NaN on an all-masked partition, a
divergent barrier on a path taken only by the last workgroup, a wave32 target
the kernel was not tested on, an `i32` overflow at a large shape.

**REFUTED only when constructible from the code:** factually wrong (quote the
actual line); provably impossible from a type, constant, or invariant (show it);
already handled in this diff (cite the guard); or pure style with no observable
effect.

### The bar for CONFIRMED

The ladder above is built to stop you refuting real bugs. These two rules exist
to stop the opposite failure, which is worse: a detailed, line-accurate,
arithmetically confident causal chain whose last step is simply asserted. Detail
is not evidence. Both rules cap the verdict at PLAUSIBLE when unmet — PLAUSIBLE
is not a demotion, it is the honest label for an unfinished proof.

**Run the arithmetic; do not narrate it.** If the argument depends on index
arithmetic, offsets, strides, shapes, bounds, or bitfield widths, write a short
script that enumerates the actual index ranges over every relevant loop and wave
variable, run it, and paste its output into `evidence`. Prose arithmetic caps at
PLAUSIBLE no matter how carefully it reads. Watch for unit confusion in
particular — a 16-row tile index is not a 32-row super-row index, an element
offset is not a byte offset, a dword count is not a byte count. Substituting one
for the other produces a chain that is wrong only in its final number, which is
exactly the error that survives review.

**Walk the chain to an observable.** A defect that never reaches an output is
not a defect. For any memory, numeric, or OOB candidate, name the specific
stored element or returned value that carries the corruption, then show it is
**not** discarded downstream — check masks, `col_valid`-style guards, buffer
descriptor `num_records` bounds, and grid tails. Kernels here routinely compute
garbage for rows past `c_m` and rely on the C descriptor to drop the stores;
that is the design, not a bug. If every affected element turns out to be
discarded, the verdict is REFUTED.

**Do not promote evidence across compiler layers.** A type round-trip proves
construction; Fly/ROCDL FileCheck proves only the checked intermediate lowering;
resource counts prove resources, not instruction identity; normalized final ISA
proves opcode/operand/modifier equivalence for that specialization; target
execution against an independent oracle proves observable semantics. Atom
layout, operand, state, signedness, predicate, or packing changes require both
the relevant lowering/ISA evidence and hardware numerics before CONFIRMED.

### Challenge the CONFIRMED ones

The runner gives every candidate marked CONFIRMED one more agent whose only job is
to refute it, told to assume the prior verifier narrated its arithmetic instead
of running it and to re-derive every number itself. If the challenger returns
PLAUSIBLE or REFUTED, take the lower verdict. Only CONFIRMED pays for this —
typically a handful of candidates, and a wrong CONFIRMED costs more credibility
than six hedged findings.

Keep candidates whose verdict is CONFIRMED or PLAUSIBLE. A failed or absent
challenger leaves the candidate unresolved, not CONFIRMED. Both verifier and
challenger evidence are retained, including when they agree.

## Step 4 — Sweep for gaps

The runner injects this section into one more finder holding the verified list. Re-read the
diff and the enclosing functions looking **only** for defects not already
listed — do not re-derive or re-confirm anything on it.

Focus on what a first pass misses: a bug in unchanged lines of a touched
function; code that moved between files and lost a guard or an anchor on the
way; setup/teardown asymmetry in tests; a default value flipped; a constant
changed in one place but not its mirror; an interaction between two separately
correct hunks. Up to 8 additional candidates, each verified like the rest. If
nothing new, return nothing — do not pad.
For compiler scopes, sweep for an uncovered scenario family, stale sibling
implementation, neutral/backend ownership leak, or test that stops before the
changed pass, final ISA, or observable boundary.

## Step 5 — Report

Synthesis is deterministic code. It retains candidate IDs, kinds, independently
adjudicated severities, source observations, verdicts and evidence; a model cannot add an unverified
finding or upgrade a verdict while rewriting the report. **Correctness findings
(A–F) always outrank convention findings (G–I) when the cap forces a cut.**
CONFIRMED outranks PLAUSIBLE within each group, then severity and stable location
break ties. Keep at most **12** across confirmed findings and plausible risks.

The artifact's `findings` contains CONFIRMED only; `risks` contains PLAUSIBLE,
reported separately and not as merge blockers. `reported_ids` preserves their
combined rank. The full candidate list, refutations, stage attempts, failures,
usage and OIDs remain in the artifact even when the display cap excludes them.
Cost is labelled as a lower bound if any attempt lacks a usage record.

Any failed, timed-out, denied, skipped or unresolved required stage yields
`status: INCOMPLETE`, stage failure labels and unresolved candidate IDs. Valid
partial findings are retained as `partial_findings`, never as a clean review or
a publishable result. Only a completed run may report that nothing survived.

## Posting to GitHub (`--comment`)

Only publish when the user requests it and the artifact identifies a GitHub PR.
Use `.claude/skills/flydsl-code-review/scripts/post_review.py` with the runner's complete `result.json`; do not
extract a findings array, rewrite the artifact, or hand-roll API calls.

```bash
python3 .claude/skills/flydsl-code-review/scripts/post_review.py \
    --findings /tmp/flydsl-review-<run-directory>/result.json \
    --publish-severity P1 --dry-run
```

Show the dry-run payload and routing. Independently recheck the load-bearing
step of each finding selected for publication, executing arithmetic where
needed; retain the artifact's verdict and evidence. Publish the checked payload
only with the user's authorization, using the same command without `--dry-run`.
Existing explicit authorization applies; do not ask for it again unnecessarily.

The publisher reconstructs findings from the saved verifier records and rejects
incomplete, malformed or altered results. Repository, PR, base and head come from
that artifact; optional `--repo`, `--pr` and `--expected-head` assert equality.
It checks both PR OIDs again after reading the patches and immediately before
posting. A moved or closed PR requires a new review.

All selected inline findings and deferred text go in one
`POST /pulls/{pr}/reviews` with `event: COMMENT` and the reviewed `commit_id`.
Deferred findings retain verdict, scenario and verifier/challenger evidence.
Plausible and below-threshold records never enter the GitHub payload. The body
includes published candidate IDs, threshold, run ID, reviewed OIDs, diff hash,
counts and usage metrics.

A pinned-diff marker ignores unrelated base-tip movement and stochastic reruns; the first successful
threshold owns that merge-base/head/diff. Lost responses are reconciled by marker, never by retrying or switching routes.
GitHub has no conditional review-write API: a push racing the final check may
make the review outdated, but cannot change its pinned commit. Stop and report
any publication error with the artifact path. A denied write is not permission
to use another posting route.
