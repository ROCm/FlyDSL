---
name: flydsl-code-review
description: >
  Review a FlyDSL diff, branch, commit range, or PR for correctness bugs and
  convention violations. Fans out nine independent review angles tuned to this
  repo's real failure modes (trace-time vs runtime semantics, buffer addressing
  and OOB, LDS/barrier lifetime, arch and MMA atom contracts, removed behavior,
  cross-layer Python/C++/FileCheck drift, repo conventions and API stability,
  reuse and altitude, test and doc contracts), verifies every candidate with an
  independent verifier, and reports a ranked, capped findings list. Pass
  --comment to post findings as inline PR comments. Use when asked to review a
  diff, review a PR, or check changes before pushing.
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

`@flyc.kernel` and `@flyc.jit` bodies are traced, not executed. The recurring
defect is code that assumes Python semantics where the value is an SSA value, or
vice versa.

- `for i in range(N)` where `i` then indexes a Python list, tuple, or dict.
  `i` is a runtime SSA value; this needs `range_constexpr(N)`. Conversely,
  `range(start, stop, step, init=[...])` is the `scf.for` form — runtime bounds
  must be typed (`fx.Int64` / `fx.Int32`) or the rewriter unrolls the loop and
  silently drops `init=`.
- `const_expr(...)` wrapping a runtime value. `gpu.thread_id`, lane id, and warp
  id are always runtime SSA even when the block size is a known constant.
  `const_expr` is only for compile-time decisions.
- A value defined only inside one branch of an `if`/`else` and used after the
  branch. MLIR result types become ill-defined. It must be hoisted or returned
  as a single merged value.
- Early `return`, or a branch-local `return` / `yield`, inside a traced
  function. There must be one explicit exit path.
- A nested helper inside `@flyc.kernel` / `@flyc.jit` that mutates a captured
  outer variable. Reading captures is fine; mutation is not. State must be
  passed in and returned.
- Loop-carried state wrapped in a raw value instead of a concrete type
  (`fx.Int32`, `fx.Float32`, `fx.Vector`). Unwrap only at hard low-level
  boundaries.

## Angle B — memory addressing and out-of-bounds

Do interval arithmetic, not eyeballing. For each load and store in the diff,
derive `start = base + offset_expr`, `end = start + vec_width - 1`, and the
legal range of the object being addressed. Substitute the known ranges: lane and
thread ids, `range_constexpr(N)` giving `i in [0, N-1]`, vector widths, tensor
shapes and strides. If `max(end)` exceeds the object's last element, or
`min(start)` precedes its base, the OOB is proven, not suspected.

- `buffer_load` / `buffer_store` offsets are in **elements**, not bytes. A `//4`
  or `* 4` near an offset is the fingerprint of a manual byte-to-element
  conversion that is either wrong or should not exist.
- Logical OOB: the address stays inside the allocation but crosses a row, head,
  tile, or lane-owned slot. A physical bounds check will not catch this.
- LDS index exceeding the allocated shared-memory extent.
- Stride or layout mismatch between writer and reader — output written to an
  absolute index where a relative slot was meant, or a 4D layout addressed as 5D.
- An offset that overflows `i32` before being widened.

A mask or clamp only narrows the interval if it **dominates** the access. A
`select` computed after the load does not.

The **oob-detection** skill has the full method and worked examples.

## Angle C — synchronization, LDS, and value lifetime

- `gpu.barrier()` reached by only some threads of a workgroup — inside a runtime
  conditional, or after an early exit. FlyDSL does not support divergent
  barriers; this hangs the GPU.
- An LDS read following an LDS write with no barrier or `s_waitcnt` between
  them. This is required every time, not only across loop iterations.
- Manual `s_waitcnt` bitfield encoding (a hex literal, or a hand-rolled
  `_encode_waitcnt`). Counter field widths differ by architecture — `lgkmcnt`
  tops out at 15 on CDNA3 and 63 on RDNA — so a hardcoded mask is silently wrong
  on the other target. Use `fx.rocdl.s_waitcnt(vmcnt=..., lgkmcnt=...)`.
- A legacy `SmemPtr` view reused after an `scf.for` without clearing
  `_view_cache`. The cached view no longer dominates its use, and MLIR rejects it.
- `SharedAllocator` mode against the launch. In the default `static=True` mode
  the compiler sizes LDS and `launch(smem=...)` must be left unset; only
  `static=False` infers from `allocated_bytes`, and an explicit `smem` must be
  at least that large.
- LDS budget: 160KB on gfx950, 320KB on gfx1250, 64KB elsewhere. A tile-size or
  double-buffer change that grows the allocation past the target's limit fails
  at launch, not at compile time.

## Angle D — architecture and atom contracts

- Wave size assumed. 64 on CDNA (`gfx942`, `gfx950`), 32 on RDNA (`gfx11*`,
  `gfx120*`, `gfx1250`). A literal `64`, a `>> 6`, or a `& 63` in lane math is a
  candidate. `get_warp_size` in `kernels/common/kernels_common.py` is the helper.
- MFMA path taken on an RDNA target, or WMMA shapes assumed on CDNA. Note the
  WMMA ABI split: `gfx11*` uses the legacy v16-operand form, `gfx120*` and
  `gfx1250` the v8-operand form.
- FP8 used on `gfx11*`, which has no native FP8 — this must fail fast, not
  silently degrade.
- Hardcoded LLVM address-space numbers (`address_space=1`, `address_space=3`, an
  `IntToPtrOp` with `<1>` or `<3>`). Use `fx.to_llvm_ptr(...)`, which maps the
  pointer's semantic address space per backend.
- Inline `gfx*` string comparisons scattered through kernel logic instead of an
  arch-specific helper module or constant.
- MMA operand contract: the accumulator comes **first** in
  `fx.gemm(mma, d, a, b, c)`, and LHS maps to M while RHS maps to N. Swapped
  operands compile cleanly and produce a transposed or wrong result. Fragments
  must be built with `fx.make_fragment_like` or the `make_fragment_*` helpers.
- TV layouts passed to `fx.make_layout_tv` must be static plain Python ints.

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

**Horizontal.** For each function the diff changes, Grep for its callers and
check whether the change breaks any call site: a new precondition, a changed
return shape or tuple arity, a new exception, a changed default, a new ordering
dependency. Then check callees — does another change in the same diff make an
existing call unsafe?

**Vertical.** FlyDSL changes span layers that must move together:

- a Python DSL surface in `python/flydsl/expr/`,
- the op definition in `include/flydsl/` and its implementation in `lib/`,
- the lowering in `lib/Dialect/FlyROCDL/`,
- FileCheck expectations in `tests/mlir/`,
- kernels in `kernels/` that call it.

A changed op signature, attribute, or lowering with an unchanged `.mlir`
expectation is a break. A new atom added for one subtarget but not registered
for the others it claims to support is a break. Check that every layer the
change touches moved with it.

## Angle G — repo conventions and API stability

These are the rules CI or a maintainer will enforce; each is checkable from the
diff.

- **Banned arithmetic spellings in `kernels/`.** `arith.maximumf`, `minimumf`,
  `maxsi`, `maxui`, `minsi`, `minui`, `ceildivsi`, `ceildivui` must be
  `fx.max` / `fx.min` / `fx.ceildiv`; `arith.maxnumf` and `minnumf` must be
  `fx.maxnumf` / `fx.minnumf`. These last two are **not** interchangeable with
  `fx.max` / `fx.min` — the NaN semantics differ, so a swap is a correctness
  bug, not a style fix. `scripts/check_typed_arithmetic_usage.py` gates added
  lines in `kernels/`.
- **`expr/` target neutrality.** Direct children of `python/flydsl/expr/`
  (`typing`, `primitive`, `gpu`, `derived`, `struct`, `arith`, `math`, `enum`,
  `numeric`, `meta`, `extern`, `utils/`) may not import `rocdl`,
  `_mlirDialectsFlyROCDL`, or `fly_rocdl`. Target-specific code belongs in
  `python/flydsl/expr/rocdl/`, and a new backend module must be added to the
  lazy `_BACKEND_MODULES` map in `python/flydsl/expr/__init__.py` rather than
  eager-imported. `tests/unit/test_expr_optional_rocdl.py` enforces this.
- **Legacy constructs in new code.** `copy_atom_call` / `mma_atom_call` (both
  the loop and single-atom forms; the `*_ssa` variants are a different
  primitive and stay), `kernels/common/buffer_ops.py` raw buffer intrinsics,
  `fx.Index` (maps to MLIR `index`, platform-defined width — widening a counter
  that must stay `i32` is a correctness bug), raw `rocdl.mfma_*`, `SmemAllocator`
  / `SmemPtr`, and redundant double-wrapping like `fx.Int32(fx.Int32(x))`.
  The **kernel-code-cleanup** skill has the full replacement table.
- **Direct upstream MLIR dialect operations** — `arith`, `scf`, `vector`,
  `llvm`, `memref`, `math`, `gpu`, `func`, `builtin` imported from
  `flydsl._mlir.dialects` or `mlir.dialects`. Unstable surface.
- **Writes to underscore-prefixed attributes** of a FlyDSL object, including
  `setattr` and `__dict__` forms. This bypasses validation or mutates shared
  state across calls. Rank it above other convention findings.
- **Environment variable spellings** not present in `python/flydsl/utils/env.py`.
- **New repo pre-checks** must be registered in `scripts/check_repo.py`'s
  `CHECKS` list, not added as a separate CI workflow step.

For a change to `python/flydsl/`'s public surface, judge breaking-change status
against `docs/api_stability.md`; the **api-stability** skill covers that review
in depth.

## Angle H — reuse, simplification, and altitude

- **Reuse.** New code re-implementing something the repo already has. Grep the
  shared modules before believing a helper is new: `kernels/common/kernels_common.py`
  (wave size, dtype mapping, MoE dtype validation, the `_if_then` SCF context
  manager, LLVM-ptr and stream helpers), `kernels/moe/moe_common.py`,
  `kernels/common/layout_utils.py`, `kernels/gemm/fp8_gemm_utils.py`,
  `kernels/common/dpp_utils.py`, `kernels/common/mma/`. Name the existing helper
  to call instead.
- **Placement.** A shared helper added to an unrelated kernel module instead of
  its topical home; a DSL-level numeric or type helper added under `kernels/`
  instead of `python/flydsl/expr/utils/arith.py` or
  `python/flydsl/expr/numeric.py`; a compiler or runtime utility outside
  `python/flydsl/utils/`.
- **Simplification.** Redundant or derivable state, copy-paste with a small
  variation, dead code the diff leaves behind, an abstraction introduced for a
  single call site.
- **Altitude.** A special case layered onto shared infrastructure where
  generalizing the underlying mechanism is the real fix. A new `gfx*` branch
  added to a function that already has three is the common shape here — the
  bandaid works and the next architecture will need a fourth.
- **Launch overhead.** A hot path calling a `@flyc.jit` wrapper per iteration
  re-runs argument marshalling and cache lookup every call; `_run_compiled` in
  `kernels/common/tensor_shim.py` is the fast path.

## Angle I — test and documentation contract

Marker rules are in `tests/README.md` and `tests/pytest.ini`.

- A new or moved test under `tests/kernels/` that lacks both `l2_device` and
  `rocm_lower`.
- `rocm_lower` applied without a tier marker alongside it.
- A test marked `l0_backend_agnostic` that imports or exercises `rocdl`,
  `fly_rocdl`, or `_mlirDialectsFlyROCDL` — it is misclassified and will fail on
  a non-ROCm host.
- A multi-GPU test without the `multi_gpu` marker, or one that does not skip
  below its required GPU count.
- A new kernel family or public kernel API with no `tests/kernels/test_*.py`
  coverage, or absent from `docs/prebuilt_kernels_guide.md`.
- A changed op or lowering with no corresponding `tests/mlir/` FileCheck update.

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

The script refuses to comment on a PR that is not open.

Its dry-run path and line routing are tested, but **the live POST has never been
exercised** — no comment has been posted with it. The first real use is also its
first test, so run it against a PR you own before pointing it at someone else's.
Delete this paragraph once it has posted successfully.

This skill does not edit code. Report findings; fixing them is a separate request.

## Related skills

Reviewing is not debugging. When a finding needs a reproduction rather than an
argument, hand off: the **debug-flydsl-kernel** skill for isolating a wrong
result, the **oob-detection** skill for proving a bounds violation, the
**kernel-code-cleanup** skill for migrating legacy constructs, and the
**api-stability** skill for a full breaking-change audit.
