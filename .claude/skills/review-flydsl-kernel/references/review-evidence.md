# Review evidence and refresh boundaries

## Historical snapshot: August 20, 2026

The [original PR #1047 skill](https://github.com/ROCm/FlyDSL/blob/eb61f68c6be23fe1624abe7fe44dfaa47e1d42ff/.claude/skills/review-flydsl-kernel/SKILL.md)
reported 479 human review comments across 163 of 865 PRs, including 46 legacy
spelling objections. Its reviewer totals were coderfeli 273, sjfeng1999 57,
xudoyuan 20, and yanguahe 11. These are the original author's corpus report,
not a census reproduced by this refresh.

The same skill reported a 10/12-file scanner replay and seeded review
comparisons: four of five additional families were already caught by the control;
test entry-point coverage changed an outcome. Those historical experiments were
not rerun here. Small seeded diffs expose their single defect more directly than
large real PRs; there was no held-out real set for the dropped families. These
experiments do not establish current precision, recall, or incremental benefit.

## September 10, 2026 refresh

Fetched all pages of `repos/ROCm/FlyDSL/pulls/comments` with
`since=2026-08-20T00:00:00Z&per_page=100`. GitHub's `since` filters update time;
the counts below additionally filter **creation time** to
`2026-08-20T00:00:00Z <= created_at < 2026-09-11T00:00:00Z` in the retrieved
snapshot. Its newest record was created September 10 at 03:10:33 UTC.
There were **233 inline comment records on 38 PRs**, including author replies.
They are not 233 independent objections. This window overlaps the historical
snapshot day, so the two corpus totals must not be added.

| Account | Records | PRs | Interpretation |
|---|---:|---:|---|
| coderfeli | 18 | 8 | Named maintainer, including replies |
| sjfeng1999 | 5 | 4 | Named maintainer, including replies |
| yanguahe | 1 | 1 | Named maintainer |
| xudoyuan | 0 | 0 | No new inline record in this window |
| jhinpan | 55 | 8 | Separate case evidence; not maintainer-consensus frequency |
| Copilot | 38 | 20 | Bot output; excluded from maintainer evidence |
| zhiding512 | 30 | 4 | Separate review account; not pooled with maintainer habits |

Account names do not establish whether a comment was tool-assisted. Keep the
named maintainer evidence separate from jhinpan's review experiments, bot output,
and other review accounts. Author acknowledgements support a case's resolution;
they are not additional independent objections or new benchmark measurements.

The refresh also read #1047's reviews and issue comments: no issue comments,
nine empty review bodies accompanying zhiding512's nine inline comments, and one
Copilot quota message. There were no additional substantive body-only requests.

## Three review questions retained

### 1. Does the current typed API preserve the required semantics?

Recent maintainer evidence confirms the typed/control-flow direction:
[coderfeli, August 26, #1067](https://github.com/ROCm/FlyDSL/pull/1067#discussion_r3860524512)
asked to clean `arith`; [September 3, #1066](https://github.com/ROCm/FlyDSL/pull/1066#discussion_r3919994660)
asked to remove `ArithValue`, and [the same review](https://github.com/ROCm/FlyDSL/pull/1066#discussion_r3920001758)
asked for ordinary `if/else` instead of manual dispatch state.
[sjfeng1999, August 20, #1035](https://github.com/ROCm/FlyDSL/pull/1035#discussion_r3818409063)
requested `dsl_math_wrap_result` so `maxnumf` honors ambient fastmath.
[Phil-amd, September 1, #1082](https://github.com/ROCm/FlyDSL/pull/1082#discussion_r3900600512)
also identified documentation recommending an `arith.minnumf` spelling rejected
by the typed-arithmetic guard; that comment explicitly treated it as non-blocking.
The policy and existing guard were checked at `main` commit `ed701427` during
this refresh; the reviewed PR's own base/head still determine its obligations.

**Trigger:** new kernel raw builders, manual numeric wrappers/control flow, or
changes to typed arithmetic wrappers. Run the existing typed-arithmetic guard;
inspect scalar/vector result types, NaN behavior, and ambient/explicit fastmath.
The historical “arith wrapper -> raw op” row is not current blanket guidance.

**Boundary:** raw builders remain appropriate at implementation boundaries when
the typed path cannot express the operation. `maxnumf` and `maximumf` differ on
NaNs; spelling similarity does not make substitution safe. Current
`docs/api_stability.md` allows upstream MLIR calls but classifies them unstable.
Use the policy at the reviewed revision for compatibility claims; an unstable
dependency alone is not a demonstrated runtime defect.

**Incremental-value check:** the guard already detects several arithmetic
spellings. A useful additional review must identify a missed semantic difference,
for example with emitted-IR/result-wrapper probes for ambient versus explicit
fastmath. Compare a broken wrapper with a correct sibling; do not add a second
spelling scanner or claim benefit from rediscovering the existing guard's output.

### 2. Does an omitted argument exercise the promised default?

[jhinpan, August 21, #1020](https://github.com/ROCm/FlyDSL/pull/1020#discussion_r3828245604)
found every test explicitly selected lazy rescaling while the CLI still overrode
the new FP8 default. [JohnQinAMD's same-day reply](https://github.com/ROCm/FlyDSL/pull/1020#discussion_r3831731969)
reports forwarding `None` through the CLI/config and a test that failed after
reverting the library default. This is a supported case, not a frequency-based
maintainer rule.

**Trigger:** changed defaults, automatic dispatch, or CLI/config forwarding.
Trace the omitted-argument call through every forwarding layer. Use an observable
positive control that distinguishes the intended selection; two explicit modes
comparing equal cannot establish which default was used.

**Boundary:** explicit-only contracts need no omitted-argument test. Output
equality is insufficient when both branches can silently select the same path;
choose discriminating inputs or observe dispatch directly.

**Incremental-value check:** revert only the default or insert the old forwarding
override in a scratch checkout. The new test should fail while explicit-mode
tests still pass. This adds behavioral coverage beyond the entry-point scanner,
which only checks whether a test can be reached.

### 3. Does a per-step numerical bound imply the accumulated invariant?

[jhinpan, August 21, #1033](https://github.com/ROCm/FlyDSL/pull/1033#discussion_r3826706430)
showed repeated permitted downward rebases overflowing FP8 attention accumulators:
`B=1, S=2048, H=8, D=128` produced 1,048,576 NaNs despite the per-step `-16` cap.
[JohnQinAMD's reply](https://github.com/ROCm/FlyDSL/pull/1033#discussion_r3827445645)
acknowledged that a step cap cannot bound the product and used a monotonic running
maximum. [The follow-up review](https://github.com/ROCm/FlyDSL/pull/1033#discussion_r3828248258)
showed the submitted random regression still passed the defective capped head.

**Trigger:** lazy/online rescaling, repeated multiplicative corrections, or
wave-uniform branches updating rows with different histories. For this softmax
form, prove `m_new >= m_row` and `corr <= 1`, including every helper path.

**Boundary:** this is a case-derived question, not a requirement that every
running statistic be monotonic. A different algorithm may supply another valid
cumulative bound; a single large random input does not prove or refute it.

**Incremental-value check:** use the cited descending-partition construction with
`V=ones` and explicitly enabled FP8 lazy rescaling; assert finite output and
`max(abs(out - 1)) < 0.01`. It must distinguish the capped defective version from
the fix. GPU results above are cited historical evidence, not rerun measurements.

## What was checked in this update

The F5 scanner was replayed on #481 at head
`7e117e29c9c4582158aed33da4ee6dfe14d2c76e`: it reports the three added variant
tests omitted by that file's script entry point. That head's
`scripts/run_benchmark.sh:524` invokes the script. This validates that static
candidate on the historical source; it does not measure review effectiveness or
establish that pytest correctness coverage was absent.

No new candidate family earned a new scanner in this refresh. To promote one,
use a held-out positive case and a clean control, keep the reviewed diff fixed,
and compare its findings with the existing checks. Retain the artifacts and the
observed difference before claiming improved review accuracy; share them when
publication is authorized.
