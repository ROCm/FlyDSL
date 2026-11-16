---
name: review-pr
description: >
  Advisory AI code review for a FlyDSL pull request. Catches silent-correctness bugs, JIT
  cache-key mistakes, architecture-capability holes, index-width overflow, unbounded buffer
  descriptors, layout-algebra errors, dead flags, and tests or benchmarks that cannot fail.
  Never a merge gate. Invoke with a PR number or owner/repo#N.
argument-hint: <PR number | owner/repo#N>
allowed-tools: Read Write Bash Grep Glob Agent
---

# FlyDSL PR Review — advisory tier

This skill gives a human reviewer hints. Its judgement is stochastic and never blocks a merge,
and it never posts to GitHub: the output is a card in `$WORK/card.md`, and publishing it is a
human decision. The `review-pr` workflow posts a card only after a maintainer adds the `ai-review`
label to the PR.

**What makes FlyDSL different from a kernel library.** It is a compiler *and* a kernel
collection, so a defect can live at four depths: the Python DSL front end, the JIT cache, the
MLIR passes, or one kernel. The blast radius differs by orders of magnitude, and Step 4 places
the diff before any rule fires.

**The dominant failure mode here is silence.** Of about thirty defects reconstructed from merged
bugfix PRs, eleven returned silently wrong numbers, five were silent performance regressions,
two were tests or benchmarks that lied, and six crashed. Weight the review toward what would fail
without saying anything.

Every step below leaves a file in `$WORK`, and Step 9 refuses the card until those files exist
and say something. A review that skipped a step must not look like one that did it.

---

## Step 1 — Fetch

```bash
# Everything Step 1 and Step 1b do is in fetch.sh; read its output and keep the printed $WORK.
"$(git rev-parse --show-toplevel)/.claude/skills/review-pr/fetch.sh" "$@"
```

Read `$WORK/pr.diff` and the PR body in `$WORK/pr_meta.json` before going further. Each artifact
answers one question:

| file | what it answers | the trap it exists for |
|---|---|---|
| `rules_expanded.txt` | the full text of exactly the rules this diff derives | reading every rule means attending to none |
| `scan.txt` | candidate sites per category, with line numbers | a category with no candidates still has to be said out loud |
| `merge_target.txt` | where the base tree is checked out — **read base files there** | grepping your own checkout answers about the wrong branch |
| `applies.txt` | whether the diff still applies to its merge target | a stale PR's CI result describes a tree that moved |
| `core_expected.txt` | the Tier 1 / 1c / 2 files the diff touches | Step 4 skipped looks identical to Step 4 done |
| `ci_checks.txt` | every check at this head, and the labels | a green tick proves only what that job ran |
| `bench_compare.txt` | CI benchmark rows at this head, and which changed kernels no benchmark reaches | the benchmark step cannot fail, so nobody reads it |
| `bounds.txt` | added buffer descriptors with no byte bound | `max_size` defaults to True, so the omission is the bug |
| `ci_coverage.txt` | where each added test actually runs | a test in a directory nothing scans is green forever |
| `comments.txt` | human review comments so far (bots filtered) | repeating a point a maintainer already made |
| `head/`, `merge-target/` | the PR's file images and the base tree | reading files from the wrong revision |

**What CI green means here — state it accurately or not at all.**

| Check | What a green tick proves | What it does not prove |
|---|---|---|
| `test (linux-flydsl-mi325-1 / mi35x-1 / navi-2)` | `scripts/run_tests.sh` passed on gfx942, gfx950 and one Navi runner | anything on gfx1250, or behind a skip none of them meets |
| benchmark steps in the same job | the benchmark **ran** | anything about speed: `scripts/compare_benchmark.py` returns 0 unconditionally |
| `multi-gpu` | — | it is skipped unless a maintainer adds the `multi-gpu` label |
| ATOM / SGLang / vLLM integration | — | these run nightly or by manual dispatch, never on a PR |
| a `.claude/`, docs or Markdown-only diff | — | `detect-changes` skips the GPU matrix and reports it green |

**Cross-file verification, before any kernel or compiler finding.** The other half of a mask, a
barrier or an address often lives in a sibling module (`kernels/common/`,
`kernels/attention/flash_attn_utils.py`); read the whole function in `head/` or `merge-target/`,
not the hunk. Layout behavior is split between Python (`python/flydsl/expr/primitive.py` emits
`fly.*` ops) and C++ (`lib/Dialect/Fly/`); grep both before claiming something is missing.
Before claiming a value is not widened, check the line above it.

**Classify every red check before blaming the PR.** Compare with main in the same window; a job
that fails the same way on main is baseline noise. CI clones aiter at `main`, so an aiter-side
change can redden PRs that touch nothing related (PR #710).

---

## Step 2 — Semantic understanding

Answer from the diff, not from the description, and **write one line per answer into
`$WORK/answers.txt`**: `Q1:` … `Q5:`. Step 9 refuses a card without them.

- **Q1 — What changed computationally?** Which algorithm, address computation, layout or
  lowering — not "improves perf".
- **Q2 — At what depth?** Kernel / DSL front end (`python/flydsl/expr/`) / JIT and cache
  (`python/flydsl/compiler/`) / MLIR pass (`lib/`).
- **Q3 — Hardware and dtype scope.** gfx942 (wave64, 64KB LDS) / gfx950 (wave64, 160KB) / RDNA
  gfx11, gfx120x (wave32) / gfx1250 (wave32, 320KB, WMMA + TDM)? bf16 / f16 / fp8 / mxfp4?
  Prefill or decode?
- **Q4 — If performance is claimed, what is the mechanism?** Fewer HBM round trips, XCD locality,
  deeper prefetch, a fastmath flag that now reaches the op?
- **Q5 — Does the description explain WHY, or only WHAT?** Surface-only descriptions correlate
  with generated code; lean harder on Steps 6 and 7.

---

## Step 3 — Rules: derived, not chosen

`$WORK/rules.txt` lists the families this diff triggers and their rule ids; `$WORK/rules_expanded.txt`
holds exactly their text. **Work only those rules.** Do not classify the PR by hand: a reviewer
asked to tick types against every rule attends to none of them. The derivation comes from paths,
added and deleted lines, and `scan.txt`; it is conservative, so a family it cannot decide is
included, never dropped. `MAPPING.md` shows which family emits which rule.

---

## Step 4 — Blast radius

**Write one line per file in `$WORK/core_expected.txt` into `$WORK/core_files.txt`**:

```
<path> TIER1|TIER1C|TIER2 COVERED|GAP|N/A -- <reason naming what THIS PR changed>
```

`COVERED` means this PR's tests exercise the blast radius below; `GAP` means they do not, and it
goes on the card; `N/A` means the change cannot reach it (comment, docstring). If no backbone file
is touched, write one `NONE -- <reason>` line naming the Tier 3 files. The gate rejects a missing
line, a tier other than the table's, and a reason under 30 characters.

| Tier | Files | Failure mode |
|---|---|---|
| **1** | `python/flydsl/compiler/jit_function.py`, `kernel_function.py`, `ast_rewriter.py`, `backends/rocm.py`; `python/flydsl/expr/__init__.py`, `typing.py`, `numeric.py`, `primitive.py`; `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`, `lib/Dialect/Fly/Transforms/LayoutLowering.cpp`, `lib/Dialect/Fly/IR/FlyOps.cpp`, `lib/Dialect/Fly/Utils/CoreBridge.cpp`, `lib/Core/EmitOp.cpp`, `lib/Core/Trait.cpp` | every kernel: wrong codegen or mass compile failure |
| **1c** | `python/flydsl/compiler/jit_argument.py`, `protocol.py`, `_aot.py`; `python/flydsl/autotune.py` | a stale or missed artifact — silent by construction |
| **2** | `python/flydsl/runtime/device.py`, `python/flydsl/expr/rocdl/`, `lib/Dialect/FlyROCDL/`, `tests/arch_compat.py`, and any `kernels/` module seven or more files import | one architecture family, or every kernel sharing the helper |
| **3** | everything else: one kernel or one op | that kernel |

A good reason answers: which public symbols changed and who calls them
(`rg -n '<symbol>' python/ kernels/ tests/`); for Tier 1c, the direction (over- or under-keying)
and which existing caches it invalidates or wrongly keeps; for `lib/`, which FileCheck test pins
the emitted IR; and, if the change is wrong, what breaks and who would notice.

---

## Step 5 — Rule pass

**Adjudicate every rule in `$WORK/rules.txt` into `$WORK/verdicts.txt`, one line each:**

```
<RULE-ID> FIRE|CLEAR|N/A -- <specific reason naming file:line, symbol or condition>
```

`CLEAR` is a claim that you looked and it does not apply to this diff; the reason makes it
checkable. "ok", "n/a", "fine" are rejected. A `FIRE` must cite a file this PR changes, and every
`FIRE` must reach the card unless the line ends `-- not reported: <reason>`. Where `scan.txt` or
`bounds.txt` lists candidates for a rule, clear or fire each one; an empty list is said, not
skipped.

Severity is advisory: 🔴 high risk / ⚠️ should fix / 📝 note.

**🔴 evidence threshold.** Before firing any 🔴, write down the concrete input that triggers it:
the shape, dtype, arch, sequence length or flag combination. "This multiply could overflow" is not
a finding; "at `head_dim=128, H=64`, `S >= 131072` puts the flattened element count past 2^31"
is. If you cannot name the triggering case, downgrade to ⚠️ or drop it.

---

## Step 6 — The sibling diff

**This is the highest-yield move in this repo.** In most reconstructed defects the correct code
already existed in a sibling path, and the bug was the asymmetry:

| PR | Sibling that was already right |
|---|---|
| #1033 | the eager path took `m_new = maxnumf(...)` first; only the lazy path did not |
| #969 | the bf16 kernel guarded rows with `_row_chk`; only the fp8 non-atomic store did not |
| #1035 | `maximumf` / `minimumf` carried `@dsl_math_wrap_result`; only `maxnumf` did not |
| #1009 | the bf16 builder had the XCD remap until a default-off trait was placed in front of it |

For every function the diff touches, list its siblings — `_opt` / `_v2`, prefill vs decode, fwd
vs bwd, bf16 vs fp8, gfx942 vs gfx950 vs gfx1250, vector vs scalar tail — and compare field by
field: masking, index width, rounding, decorator, clamp order. Report which side is right and why.

---

## Step 7 — Generated-code diagnostic

A clean description does not make the code correct. When the diff changes code, run these six
checks and **write one line per check into `$WORK/ai_diagnostic.txt`** (`1:` … `6:`), naming what
you looked at and what you found — "clean" alone is rejected:

1. **Hallucinated symbols.** Grep every symbol new to this diff — function, kwarg, trait,
   attribute, import — against its definition on head. PR #1136 found a `bounds_checked=` argument
   that `make_buffer_tensor` never had, failing 33 of 44 tests at trace time.
2. **Twin divergence.** Step 6, applied to mirrored code the PR itself adds.
3. **Claim against code, and number provenance.** Does the code enforce what the description
   says? Trace the most impressive number to a log or script output; an untraceable number is
   `[unverified]`, never repeated as fact.
4. **Safety theater.** Is each new guard reachable, will it fire, does it swallow a real error?
5. **Tests calibrated to pass.** Is the reference a structural twin of the kernel? Was a
   tolerance widened? Does the test assert, or return?
6. **Magic constants.** A new tile size, threshold or epsilon with no stated derivation.

If three or more fire, note "elevated generated-code risk" on the card.

---

## Step 8 — Free-form review

Read the diff as a domain expert:

- **LDS budget** (`SMEM_CAPACITY_MAP` in `python/flydsl/runtime/device.py`): gfx942 64KB, gfx950
  160KB, gfx1250 320KB. On gfx1250 `ds_read`/`ds_write` immediate offsets are 16-bit, so an
  allocation past 64KB can push addressing into VGPRs.
- **Wave size**: wave64 on gfx942/gfx950; wave32 on gfx11, gfx120x **and gfx1250**. Any reduction,
  ballot or lane-indexed layout written for wave64 needs an explicit check.
- **XCD mapping**: a remap is a bijection, so a correct XCD change is bit-identical; changed output
  means a wrong remap.
- Does the tiling fit the target's MFMA (CDNA) or WMMA (RDNA, gfx1250) atom shapes? Are hardware
  tile constants named rather than scattered literals? Is fp8 fn vs fnuz decided by dtype, not by
  arch name?

When a candidate needs a deeper procedure, use the skill that owns it instead of re-deriving it:
the **oob-detection** skill for interval analysis of an access, the **flydsl-kernel-authoring**
skill for trace-time versus runtime control flow, the **kernel-code-cleanup** skill for legacy APIs
and their replacements, the **lds-optimization** skill for LDS and cross-wave ordering, the
**add-target-atom-op** skill for atom and lowering changes, and the **api-stability** skill for a
public-surface change.

---

## Step 8.5 — Blind-spot check

Answer in full: **"Is there any correctness risk, resource hazard or behavioral edge case in this
diff that none of Steps 1–8 caught?"** Look where a first pass misses: unchanged lines of a touched
function; code moved between files that lost a guard; setup/teardown asymmetry in a test; a default
flipped; a constant changed in one place but not its mirror; two hunks that are each correct and
wrong together. Append the answer to `$WORK/answers.txt` as a `BLIND:` line; a bare "no" is
rejected, so say what you looked for. If yes, add it to the findings.

## Step 8.6 — Refute your own findings

Try to kill each finding before it reaches the card. Attack, in this order: **the premise** (read
the code the claim depends on, not only the code it is about); **the tree** (anchor line numbers in
`merge-target/` or `head/`, not your checkout); **reachability** (can the trigger occur, or does a
caller, guard or arch gate prevent it?); **suspicion vs defect** (no inputs that produce a wrong
result means a question, which is 📝); **severity** (🔴 needs a reachable trigger). Write one line
per attempt to `$WORK/refutations.txt`:

```
RED|WARN|NOTE SURVIVED|KILLED -- what you opened and what it said
```

A killed finding never reaches the card. Name a file, a symbol or a command; "I checked" is rejected.

## Step 8.7 — Independent refutation

Self-refutation catches a premise you never tested; it does not catch a conclusion you are
committed to. **Hand `$WORK/card.md`, `$WORK/pr.diff` and `$WORK/merge_target.txt` to a reader who
has not seen your reasoning**: spawn a subagent whose whole prompt is
`.claude/skills/review-pr/prompts/refuter.md` with `{{WORK}}` filled in, or ask a person. It writes
`$WORK/independent.txt`, one `SURVIVED|KILLED -- what they opened` line per finding, in card order;
remove the killed findings from the card and fix what any `Correction:` on a surviving line says.
With no such reader, write `NONE AVAILABLE -- <reason>` and put `not independently refuted` on the
review line. Under `run_one.sh` the runner does this step, and `triage.py apply-refutation` drops
the killed findings and lists the corrections under the card.

---

## Step 9 — Verdict

Write the card to `$WORK/card.md`, then run the gates. A red gate means a step did not happen; go
back to it rather than reporting.

```bash
"$(git rev-parse --show-toplevel)/.claude/skills/review-pr/triage.py" gates "$WORK" "$(git rev-parse --show-toplevel)"
```

The seven gates are `answers`, `corefiles`, `ledger`, `diagnostic`, `card`, `refutations` and
`independent`; each failure names its class (`UNANSWERED`, `TIER-MISMATCH`, `UNADJUDICATED`,
`UNREPORTED-FIRE`, `UNPROVEN-RED`, `UNANCHORED`, `COUNT-MISMATCH` …) and the file to fix. They
check that the work happened and that the card reports that work. Whether a finding is *correct*
is not checked; that is what a human reads the card for.

**Output rules (strictly enforced):**
- Run Steps 1–8 internally. Do **not** narrate steps, show checklists, or name which rules fired.
- Output **only** the card below — nothing before it, nothing after it.
- If there are no findings, omit the findings section.
- "What it does" is one sentence for a reviewer who has not read the diff.
- **At most 5 findings, most severe first** (severity, then blast radius); when the cap forces a
  cut, correctness findings outrank convention and housekeeping ones. This is a readability limit,
  not a recall claim.
- **The CI line** states what the checks at this head actually ran, using `ci_checks.txt`, and
  classifies any red check as PR, baseline or infra.
- **The Perf line** comes from `bench_compare.txt`. CI rows are one run on a shared runner with no
  noise floor, so they are advisory: give the ratio range and row count for the ops that cover the
  changed kernels, or `NOT MEASURED` with the reason (no benchmark reaches the kernel, job still
  running, log expired). A hand-run number is advisory too; say base vs head and the sample count.
- With CI pending or red, or perf not measured, no finding may assert runtime behavior as fact;
  such findings are `[inferred]` and phrased as questions.
- Do **not** use rule ids (A3, D1, T7 …) on the card; they are internal labels.

```
## FlyDSL PR #NNN — [title]

**[One sentence: what this PR does, in plain terms.]**

Review (advisory): [✅ NO FINDINGS | ⚠️ NEEDS WORK | 🔴 HIGH RISK]
CI (head <sha7>): [each GPU test check and its result; a red check classified as PR / baseline / infra | PENDING — <checks> | SKIPPED — docs-only change]
Perf (CI benchmark, advisory): [<op rows covering the changed kernels>: ratio <min>–<max>x vs <baseline>, <n> rows, one run | NOT MEASURED — <reason>]

🔴 [finding]
⚠️ [finding]
📝 [note]
```

Each finding has **three parts** and a tag:

1. **Problem** — what is wrong, with `file:line`.
2. **Impact** — what happens at runtime if it is not fixed (silent wrong values / NaN / crash /
   stale artifact / perf regression).
3. **Action** — ends with "**Author must** …" or "**Reviewer should ask** …". No action, no finding.

Tag every finding `[verified]` (traced through the actual code) or `[inferred]` (plausible but
unconfirmed: frame it as a question, never as a root cause). Two things keep a finding `[inferred]`
however carefully it reads. **Arithmetic that was narrated, not run:** if the claim rests on index,
offset, stride, shape or bitfield arithmetic, run a short script over the real loop and lane ranges
and quote its output; watch element vs byte and tile vs super-row units. **A chain that stops before
an observable:** name the stored element or returned value that carries the error and show nothing
downstream drops it — masks, the buffer descriptor's `num_records`, grid tails. Kernels here often
compute garbage past the edge and let the descriptor drop the stores; that is the design, not a bug.

Good findings:

- `🔴 [verified] kernels/attention/pa_decode_tile.py:549 multiplies phys_row[sub] by n_kv, STEPS_PER_PAGE and head_dim in int32. At head_dim=128 with a KV cache above 2 GiB the offset wraps to a lower page, so decode attends to the wrong tokens with no error. **Author must** widen the page id with fx.Int64 before the multiply chain.`
- `🔴 [verified] The new causal_lpt flag is accepted by the public entry point, but neither _build_splitk nor the long _build_varlen route forwards it, so on those paths the builder default silently re-enables LPT; every test passes the flag explicitly. **Author must** thread it through both builders and add a case that relies on the default.`
- `⚠️ [inferred] The 1.31x number is not traceable to any output in the PR, and the diff edits a helper outside the traced closure, so a warm JIT cache may have served the previous kernel. **Author must** re-run with FLYDSL_RUNTIME_ENABLE_CACHE=0 and attach the log.`

Bad findings — do not produce these:

- `⚠️ Missing perf numbers` — no impact, no action.
- `🔴 A3 violation` — a rule id means nothing to a reviewer.
- `⚠️ This multiply might overflow` — no triggering input, so it fails the 🔴 threshold.

---

## Running headless

`run_one.sh <PR | owner/repo#N>` runs the whole review without a person: `fetch.sh` on the host
(it needs `gh` auth), then a worker agent for Steps 1b–9 and a separate refuter agent for Step 8.7,
then the gates. With `FLYDSL_REVIEW_SANDBOX_IMAGE` set, both agents run in a Docker container that
mounts the checkout read-only and `$WORK` read-write and holds no GitHub credentials, so a review of
untrusted PR text cannot write to GitHub or read the host's tokens. The card, the gate report and
`usage.txt` (each agent's wall time and token use) are left in `$WORK`; nothing is posted.

`.github/workflows/review-pr.yml` runs this on the self-hosted `flydsl-review` runner when a PR
gets the `ai-review` label. Its review job alone uses the `flydsl-review` environment, which holds
the API key and serves `main` only. `gateway.sh start` hands the key to a gateway container and
puts the sandbox on a network whose only reachable peer is that gateway, so the agents never see
the key and reach neither the host nor the internet. A separate job posts the card.

---

## Adding new rules

When a human reviewer catches something this skill missed:

1. Add the rule body to `rules.md` under the right category, **with a real PR number as evidence**.
2. Add its id to the family in `triage.py` whose diff would trigger it, and regenerate
   `MAPPING.md` with `python3 triage.py mapping > MAPPING.md`. A rule no family emits is never read;
   `triage.py mapping` names any such rule.
3. Commit as `[Skills] review-pr: add <rule> from PR #<NNN>`.

The skill grows from real review history, not hypothetical patterns. **Rule text goes in
`rules.md`, executable steps go in a script; this file holds only what every review needs** and
stays under 400 lines, because a longer entry file is less likely to be read in full.
