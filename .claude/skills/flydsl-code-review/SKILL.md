---
name: flydsl-code-review
description: >
  Review a FlyDSL diff, branch, commit range, or PR for correctness bugs and
  convention violations using the repository's existing skills and policy docs.
  Runs nine independent review angles, verifies every candidate with an
  independent verifier, and reports a ranked, capped findings list. Pass --comment
  to post findings as inline PR comments. Use when asked to review a diff, review
  a PR, or check changes before pushing.
allowed-tools: Read Bash Grep Glob Agent Workflow
---

# FlyDSL Code Review

Find real defects in a change, then prove each one before reporting it.

The structure is: **many independent finders, one independent verifier per
candidate, then rank and cap.** Finders are cheap and biased toward surfacing;
verifiers are the filter. Do not merge those two jobs — a finder that quietly
discards its own half-believed candidates bypasses verification entirely, and
that is the single largest cause of missed bugs.

## Invocation

```text
/flydsl-code-review                       # current branch vs upstream
/flydsl-code-review HEAD~3                # a ref or ref range
/flydsl-code-review 1100                  # a PR number
/flydsl-code-review kernels/attention/    # restrict to a path
/flydsl-code-review focus on the LDS changes   # free-form instruction
/flydsl-code-review --comment             # also post to the PR
```

Any argument that is not `--comment` is the review target. Honor it verbatim:
a path or free-form instruction narrows scope and focus, and findings the
instruction asks you to skip must not be surfaced.

## Step 1 — Gather the diff

Run `git diff @{upstream}...HEAD`. If there is no upstream, fall back to
`git diff main...HEAD`, then `git diff HEAD~1`. If the range diff is empty or
there are uncommitted changes, also run `git diff HEAD` and include the working
tree — this review usually runs before the commit.

If a PR number, branch, ref range, or path was passed, review that instead. Use
`gh pr diff <n>` for a PR.

That diff is the review scope. Read the enclosing function for each hunk: bugs
on unchanged lines of a touched function are in scope, because the change either
re-exposes them or failed to fix them.

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

## Step 2 — Run the nine angles

If the Workflow tool is available, run the fan-out as a workflow instead of
inline — it pipelines verification against finding, so verifiers start before
the last finder returns:

```text
Workflow({name: 'flydsl-code-review', args: '<target>'})
```

Otherwise run the angles inline with the Agent tool. Nine independent finders,
**up to 6 candidates each**, one angle per agent. Do not let one angle's
conclusions suppress another's: if two angles flag the same line for different
reasons, record both. Each candidate needs a `file`, a `line`, a one-line
`summary`, and a concrete `failure_scenario`.

Angles A–F hunt correctness bugs. Angles G–I hunt convention violations and
cleanup; for those, `failure_scenario` states the concrete cost (what breaks in
CI, what is duplicated, what becomes arch-fragile) rather than a crash.

---

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
skill ([SKILL.md](../add-target-atom-op/SKILL.md)), §1 **Inherent Design**.
Check lane math, dtype support, dispatch, and operand/layout assumptions against
every target the changed code claims to support.

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

## Angle I — test and documentation contract

Read `tests/README.md` and `tests/pytest.ini` for tier, backend, and marker
contracts, and [CLAUDE.md](../../../CLAUDE.md)'s **Testing Notes** and **Kernel
Entry Points** for multi-GPU requirements and new-kernel test/documentation
coverage. Check new or moved tests against their actual dependencies and device
requirements.

For changed atoms, use the **add-target-atom-op** skill
([SKILL.md](../add-target-atom-op/SKILL.md))'s applicable verification steps.
For other changed ops or lowerings, inspect the corresponding FileCheck coverage
described in `tests/README.md`.

Do not flag general "needs more tests" — only these specific contract breaks.

---

## Step 3 — Verify every candidate

Dedup candidates pointing at the same line and mechanism, keeping the one with
the most concrete failure scenario. For each remaining candidate run **one
independent verifier** with the Agent tool. Give it the diff, the relevant
files, and the candidate. It returns exactly one verdict:

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

### Challenge the CONFIRMED ones

Every candidate a verifier marks CONFIRMED gets one more agent whose only job is
to refute it, told to assume the prior verifier narrated its arithmetic instead
of running it and to re-derive every number itself. If the challenger returns
PLAUSIBLE or REFUTED, take the lower verdict. Only CONFIRMED pays for this —
typically a handful of candidates, and a wrong CONFIRMED costs more credibility
than six hedged findings.

Keep candidates whose verdict is CONFIRMED or PLAUSIBLE.

## Step 4 — Sweep for gaps

Run one more finder as a fresh reviewer holding the verified list. Re-read the
diff and the enclosing functions looking **only** for defects not already
listed — do not re-derive or re-confirm anything on it.

Focus on what a first pass misses: a bug in unchanged lines of a touched
function; code that moved between files and lost a guard or an anchor on the
way; setup/teardown asymmetry in tests; a default value flipped; a constant
changed in one place but not its mirror; an interaction between two separately
correct hunks. Up to 8 additional candidates, each verified like the rest. If
nothing new, return nothing — do not pad.

## Step 5 — Report

Merge findings that describe the same root cause and combine their evidence.
Rank most-severe first. **Correctness findings (A–F) always outrank convention
findings (G–I) when the cap forces a cut.** CONFIRMED outranks PLAUSIBLE within
each group. Keep at most **12**.

For each finding: the file and line, one sentence on what is wrong, the concrete
failure scenario, and the verdict. Cite the line. Lead with a two-to-three
sentence summary of the review.

If nothing survives verification, say so plainly. An empty review is a valid
result and is better than a padded one.

## Posting to GitHub (`--comment`)

Only when `--comment` was passed **and** the review target is a GitHub PR. If
the target is not a PR, print the findings and say that `--comment` was ignored.

Do not hand-roll the `gh api` calls. GitHub rejects an inline comment on any line
that is not in the PR diff, so placement has to be decided against the actual
patch. `.claude/skills/flydsl-code-review/scripts/post_review.py` does that: it
parses each changed file's patch
into the set of commentable RIGHT-side lines, posts what fits inline, and rolls
everything else — untouched files, lines outside a hunk, findings with no line —
into one summary comment so nothing is dropped.

Write the findings to a JSON file, then:

```bash
python3 .claude/skills/flydsl-code-review/scripts/post_review.py \
    --pr <number> --findings <file.json> --dry-run
```

Each finding is `{"file", "line", "summary", "failure_scenario", "verdict"}`;
`file` may be absolute or repo-relative, and `line` may be omitted. A bare JSON
array works, as does the workflow's result object with its `findings` key.

**Always `--dry-run` first** and show the user the routing — how many land
inline, how many defer, and on which lines. Drop `--dry-run` only after they
confirm. Posting is not reversible: every comment notifies the PR's
participants, and deleting one later does not unsend the mail.

**Re-derive every CONFIRMED finding yourself before posting.** Not "read the
evidence and find it convincing" — independently reproduce the load-bearing
step, running the arithmetic where there is arithmetic. This is the only check
in the pipeline that does not depend on an agent doubting its own reasoning,
which makes it the one that actually holds. On PR #1107 it was skipped and a
false CONFIRMED reached the author's inbox: the chain quoted real lines and real
constants but conflated a 16-row tile index with a 32-row super-row index, so
the out-of-bounds reads it correctly identified all landed on rows the C
descriptor discards. A fifteen-line enumeration would have caught it. State the
re-derivation result alongside the dry-run routing so the user is approving a
checked list, not a plausible one. PLAUSIBLE findings do not need this — they
are labelled uncertain and cost the author little.

The script refuses to comment on a PR that is not open.

Expect the post to be denied even when `Bash(*)` is allowed: the permission
classifier cannot evaluate what the script does and blocks it independently of
the allow list. When that happens, ask the user to run the command themselves.
Do not reach for `gh api` instead — that evades the same evaluation the block
exists to force, and hand-computed line numbers are what the script is for.
