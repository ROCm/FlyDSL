---
name: api-stability
description: >
  Review a FlyDSL PR, commit, branch, kernel, or consuming module for API-stability
  compliance. Detect breaking changes to stable APIs, usage of unstable FlyDSL APIs,
  and direct upstream MLIR dialect operations. Use when asked to review API
  compatibility, a stable-API change, or whether a kernel/module uses only stable APIs.
allowed-tools: Read Edit Write Bash Grep Glob
---

# API Stability Review

Use this skill for two distinct review modes:

1. **Producer review** — determine whether a PR, commit, or branch breaks an
   existing stable FlyDSL API.
2. **Consumer review** — determine whether a kernel or other module uses only
   stable FlyDSL APIs. Direct upstream MLIR operation use must receive a
   separate, prominent reminder.

## Invocation

Accept any of these request forms:

```text
/api-stability pr <number-or-URL>
/api-stability commit <revision>
/api-stability branch [base-ref]
/api-stability usage <file-or-directory>
```

Infer the mode when the user's target is unambiguous. If they only request an
“API stability review” without identifying whether they want producer or
consumer review, ask which target to audit before proceeding.

## Scope and review rules

- The public surface classified by the policy is `python/flydsl/`. The
  `kernels/` tree is not itself a classified public API surface, but its imports
  and calls can be audited as a consumer of FlyDSL APIs.
- Review the source at the revision being examined. Do not classify a historic
  commit using only the current checkout's manifests.
- Resolve the path that the caller actually spells, including aliases and
  direct-import forms; do not classify by Python object identity.
- Do not edit, commit, push, or change a reviewed PR unless the user separately
  requests a fix. A review reports evidence and an outcome.
- State uncertainty explicitly. Dynamic `getattr`, `importlib`, wildcard
  imports, generated source, or an alias that cannot be resolved is
  **UNRESOLVED**, not proof of stable usage.

## Policy source

`docs/api_stability.md` is the single source of truth for API stability. Before
classifying an API, read the policy at the reviewed revision, apply it exactly,
and cite the relevant section for every non-obvious finding. Do not infer
stability from whether an attribute happens to be importable or callable.

## Reference material and static catalog

Read these implementation manifests at the reviewed revision as applicable:

- `python/flydsl/expr/__init__.py` — direct-child aggregation and
  `_BACKEND_MODULES`.
- `python/flydsl/compiler/__init__.py` and
  `python/flydsl/compiler/protocol.py` — compiler export manifests.
- The relevant module or package `__all__` declarations.

Use the static catalog as an aid:

```bash
python3 scripts/list_stable_apis.py --format json
```

The catalog intentionally does **not** import FlyDSL, omits equivalent
top-level `expr` aliases, excludes §3 deprecated APIs, and does not enumerate
public type or result-object members. It can detect declared export-chain
changes, but it cannot by itself prove signature or semantic compatibility.

## 1. Producer review: PR, commit, or branch

### Resolve the comparison

Use an explicit base and head before examining the diff:

- **PR**: obtain its base and head commits with `gh pr view <PR-or-URL>`; compare
  the PR head with the merge base of its base branch and head.
- **Single commit**: compare `<commit>^` to `<commit>`; for a merge commit, ask
  which parent represents the intended baseline if it is not clear.
- **Current branch**: compare `HEAD` with its merge base against the nominated
  base branch (normally `origin/main`).

Inspect the complete relevant diff, including renames:

```bash
git diff --find-renames --find-copies --unified=80 "$base" "$head" -- \
  python/flydsl docs/api_stability.md
```

Do not limit the review to edited function bodies. Inspect all touched
`__init__.py`, `__all__`, `_BACKEND_MODULES`, `compiler.protocol`, and §2.3 / §3
documentation-table changes.

### Compare declared stable surfaces

Run `scripts/list_stable_apis.py` against **both** revisions. Prefer disposable
detached worktrees so the primary worktree is not changed. Compare the two JSON
outputs:

- a path present at the base and absent at the head is a candidate
  **BLOCKER**;
- a new path is not breaking, but creates a new compatibility commitment and
  should be called out;
- inspect §3 separately, because the catalog deliberately excludes deprecated
  stable APIs.

The catalog comparison is a starting point only. Review every changed API that
was stable at the base revision, including a stable class's public methods and
dunder methods.

### Check for breaking changes

For each base-stable API affected by the diff, compare base and head for all
§4 break conditions:

- removal, loss of an export-chain link, or removal from the explicit §2.3
  table;
- removed parameters, renamed keyword parameters, positional reordering,
  removed defaults, or changed defaults;
- narrower accepted types, architectures, or value ranges;
- changed return type, tuple arity, public type members, numerical semantics,
  layout semantics, or emitted-operation semantics for previously valid input.

Treat an implementation refactor that changes observable behavior as a breaking
change even when the function signature is unchanged. Conversely, do not
mislabel a new API, widened input, a behavior-preserving optional keyword,
improved diagnostics, or an unstable-only change as breaking.

For a retirement, verify all §5 requirements: a stable replacement, a
deprecation marker and §3 entry, retention in releases `N` and `N+1`, and
removal no earlier than `N+2`.

### Validate and report

At minimum, run the static catalog in each applicable reviewed worktree. When
the PR changes Python behavior, run focused tests that exercise the affected
public API when the environment permits; report commands that could not be run
rather than assuming success.

Use this result shape:

```text
## API-stability producer review — PASS | NEEDS CHANGES | MANUAL FOLLOW-UP
Scope: <PR/commit/base...head>

### Breaking changes
- [BLOCKER] <stable API path> — <base behavior> → <head behavior>; §<policy section>.

### Public-contract additions and deprecations
- [INFO/WARN] <path> — <new commitment, migration state, or retirement issue>.

### Validation
- <catalog/test command> — <result or why it was not run>.
```

Only return **PASS** after reviewing the complete public-surface diff and all
affected base-stable APIs. A clean catalog diff alone is insufficient.

## 2. Consumer review: kernel or FlyDSL-using module

### Inventory and resolve FlyDSL usage

Audit the requested file or directory only; do not expand to unrelated callers.
For Python source, first locate imports and then trace their aliases to actual
attribute accesses and calls. A useful initial search is:

```bash
rg -n --glob '*.py' '^\s*(from|import)\s+(flydsl|mlir)(\.|$)|flydsl\._mlir' <scope>
```

Resolve examples such as:

- `import flydsl.expr as fx` + `fx.foo` → `flydsl.expr.foo`;
- `from flydsl.expr import arith as ea` + `ea.addi` →
  `flydsl.expr.arith.addi`;
- `from flydsl.expr.typing import Vector as Vec` + `Vec.method` →
  `flydsl.expr.typing.Vector.method`;
- `from flydsl.compiler import kernel` → `flydsl.compiler.kernel`.

Classify every direct FlyDSL import and every accessed/called FlyDSL member
under `docs/api_stability.md`. For a fluent chain such as
`factory(...).member`, resolve the factory first, then apply the policy's
returned-object rule. Do not infer stability from a dynamic binding. Report
direct imports that are unused separately from actual calls.

Use these statuses:

- **STABLE** — the exact path passes the policy.
- **DEPRECATED** — still stable for compatibility, but disallowed in new code;
  it prevents a “stable-only” result.
- **UNSTABLE** — the exact path does not pass the policy, including raw
  `flydsl._mlir.*` access.
- **UNRESOLVED** — static inspection cannot establish the path.
- **UPSTREAM-MLIR** — an additional classification and reminder for direct
  upstream MLIR operation use; it is also unstable for this audit.
- **PRIVATE-WRITE** — an assignment to an underscore-prefixed attribute of a
  FlyDSL object; a strictly higher-severity form of UNSTABLE.

A module is **stable-only** only if it has no DEPRECATED, UNSTABLE,
PRIVATE-WRITE, UPSTREAM-MLIR, or UNRESOLVED FlyDSL uses. Existing internal code may
intentionally rely on unstable APIs; report that fact rather than treating it
as a producer-API compatibility break.

### Mandatory private-field-write warning

Reading an unstable member is a compatibility bet; *writing* one mutates FlyDSL
internal state, so it can break FlyDSL's invariants at the reviewed revision, not
only after an upgrade. Rank these above every other unstable finding.

Give a `[PRIVATE-WRITE]` finding for each assignment to an underscore-prefixed
attribute of a FlyDSL object — overwriting a field FlyDSL sets, attaching one it
does not define, and the `setattr` / `__dict__` forms. Locate where FlyDSL
assigns, validates, and reads the field, then state the concrete consequence and
any aggravating factor: **bypassed validation** (the public path runs a check the
write skips) or **shared mutable object** (the write leaks across calls, configs,
or threads). Note the missing public API, since that is why the workaround
exists.

```text
[PRIVATE-WRITE] kernels/example.py:1003: kernel_impl._known_block_size = [...]
Writes a FlyDSL-internal field of a stable object; unstable under §1
(underscore rule). Bypasses _validate_known_block_size() in
compiler/kernel_function.py and mutates a module-level kernel shared
across configs.
```

### Mandatory upstream-MLIR reminder

Give a separate `[UPSTREAM-MLIR]` finding for every direct use of an upstream
MLIR dialect operation, with its import path, alias, call site, and line number.
This applies to direct imports from `mlir.dialects.*` or from
`flydsl._mlir.dialects` for external dialects such as:

```text
arith, builtin, func, gpu, llvm, math, memref, rocdl, scf, vector
```

It also applies to direct generated ODS-builder modules such as
`flydsl._mlir.dialects._arith_ops_gen`. Typical findings include
`arith.addi`, `scf.ForOp`, `vector.LoadOp`, and `llvm.*` calls.

Use wording equivalent to:

```text
[UPSTREAM-MLIR] kernels/example.py:42: vector.LoadOp
Direct upstream MLIR builder; allowed, but unstable under
docs/api_stability.md §2.4. FlyDSL does not guarantee its name, signature,
or semantics across releases. Prefer a stable FlyDSL wrapper when one exists.
```

`flydsl._mlir.ir` alone is raw, unstable MLIR infrastructure but is not by
itself an operation-use reminder. Raw `fly` and `fly_rocdl` dialect bindings
are FlyDSL-specific rather than upstream; classify them as **UNSTABLE** raw
FlyDSL bindings, not as `UPSTREAM-MLIR`. Do not suppress the reminder merely
because an external dialect is accessed through an alias.

### Report

Use this result shape:

```text
## Stable API usage audit — STABLE-ONLY | NOT STABLE-ONLY | MANUAL FOLLOW-UP
Scope: <file or directory>

### Stable uses
- <path> — <locations>

### Private-field writes
- [PRIVATE-WRITE] <target expression> — <locations>; <what FlyDSL uses the field
  for, plus bypassed validation / shared-object impact>.

### Deprecated or unstable uses
- [DEPRECATED/UNSTABLE] <resolved path> — <locations>; <policy reason>.

### Upstream MLIR operations
- [UPSTREAM-MLIR] <operation> — <locations>; §2.4 reminder.

### Unresolved paths
- <source expression> — <why static resolution was insufficient>.
```

Do not collapse private-field-write or upstream-MLIR findings into a generic
unstable list: both separate sections are required even when the audit result is
already `NOT STABLE-ONLY`. Order the report so private-field writes come first.

## Completion checklist

- Use `docs/api_stability.md` at the correct revision as the source of truth.
- For producer reviews, compare base and head exports and manually inspect
  changed base-stable signatures and semantics.
- Account for §3 deprecated APIs separately from the generated catalog.
- For consumer reviews, resolve aliases and fluent result chains before
  classifying usage.
- Emit a distinct finding for each direct upstream MLIR operation.
- Emit a distinct `[PRIVATE-WRITE]` finding for each assignment to an
  underscore-prefixed attribute of a FlyDSL object, and rank those first.
- Distinguish a consumer's unstable dependency from a breaking change to the
  FlyDSL public API.
