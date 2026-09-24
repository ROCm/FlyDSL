---
name: release-api-check
description:
  Review a FlyDSL release candidate against published releases for API stability
  violations, including export paths, signatures, result interfaces, semantics,
  and deprecation windows. Use for release compatibility reviews and
  breaking-change checks.
---

# FlyDSL Release API Compatibility Review

Work in the FlyDSL repository and use `docs/api_stability.md` as the policy
source. Produce a compatibility report with source locations and validation
evidence. A review does not itself authorize tagging, publishing packages, or
changing release status.

## Establish the comparison

- Identify two endpoints and explicitly assign them to `--old` and `--new`.
  Either endpoint can be a commit SHA or tag; commit-to-tag and tag-to-commit
  comparisons are supported. Honor the user's chosen direction without sorting
  by tag name, timestamp, or ancestry. Record both references and resolved SHAs.
  Ask for the missing endpoint or direction if the intended comparison is unclear.
- The script compares committed snapshots. Staged, unstaged, and untracked
  files are excluded. For a requested working-tree review, inspect local
  changes separately and do not present a `HEAD` comparison as verification of
  those changes.
- For a release compatibility conclusion, identify the intended release version
  and verify the baseline's published commitments. Commit comparisons can
  establish path inclusion, while the policy guarantees compatibility between
  releases. An intermediate main-branch state is not itself a release baseline.
- Read the policy and export manifests at both endpoints. A target cannot
  withdraw an earlier release's commitments by changing the policy, removing
  `__all__` entries, or moving APIs into `experimental`. Newly stable surfaces
  become protected when first released under that commitment; do not apply
  extension stability retroactively to earlier releases.
- Compatibility guarantees apply between releases. Intermediate main-branch
  semantic adjustments are permitted. Patch releases must preserve stable APIs;
  assess minor-release retirements under the applicable deprecation rules.
- Prefer `git show`, `git diff`, and static source inspection for history. If a
  worktree is needed, use the main project's `.codex/worktrees/` directory;
  identify the main checkout with `git worktree list --porcelain` and preserve
  the user's current checkout.

## Build the protected interface inventory

Use the repository's `scripts/list_stable_apis.py` to read source without
importing FlyDSL. From the repository root:

```bash
python3 scripts/list_stable_apis.py --old v0.3.3 --new HEAD --format json
```

Replace the example endpoints with the requested old and new revisions.
Use `--repo-root` to select another Git repository. The script reads temporary
source snapshots without changing the checkout or index. Personal installations
of this skill also use the repository's script.

Comparison requires `old_paths <= new_paths`, including deprecated exports and
equivalent public aliases. A larger count alone is insufficient. JSON output
records both references and SHAs, `is_superset`, and `added`, `removed`, and
`retained` path lists. Exit codes are `0` for inclusion, `1` for missing old
paths, and `2` for invalid arguments or an unreadable snapshot. A permitted
retirement still fails this strict inclusion check; assess its release window
separately rather than treating every missing path as a policy violation.

Single-tree listing remains available with `--format json --include-deprecated`.
The default listing omits deprecated APIs and equivalent aliases, so its counts
can differ from comparison output. Read both versions' policies and inspect
the deprecated tables. The collector handles the earlier policy without stable
extensions and the old extension manifest name. If a policy or manifest cannot
be interpreted, use that revision's tooling or reconstruct commitments from
its documentation, source, and release notes; report remaining evidence gaps.

Trace the actual export chains:

- Direct-child `expr` modules: root star imports and each child's `__all__`,
  including equivalent top-level aliases and direct imports.
- Backends: `_BACKEND_MODULES` entries, each parent package's `__all__`, and
  final exports.
- Extensions: `_EXTENSION_MODULES` entries and recursive `__all__` chains, with
  equivalent `fx.<extension>` and `flydsl.extension` paths. The catalog uses
  canonical paths; separately check removed, renamed, or retargeted aliases.
- Compiler: root `__all__`, `protocol.__all__`, and documented exceptions.
- Private paths and module segments named `experimental` at any depth: apply the
  policy's exclusions. Classify re-exports by their access paths rather than
  implementation filenames. Moving an old stable path into an experimental
  namespace still requires preserving its prior commitment.

Use catalog differences to locate changes. A path list does not establish
runtime importability, enumerate every member, or validate semantics. Neither an
empty diff nor a successful command proves compatibility. Confirm that
explicitly documented paths still exist in source.

## Review behavior and retirement windows

Inspect the baseline-to-target diff for old stable APIs and their dependencies.
For a working-tree target, combine
`git diff BASE -- python/flydsl docs/api_stability.md` with relevant untracked
files. Also inspect changes in `lib/`, `include/`, generated code, and upstream
dependencies that affect the stable Python surface.

For each affected interface, check:

- Existing import paths, export chains, aliases, and call forms; removed or
  renamed parameters, positional order, defaults, parameter kinds, and newly
  required arguments.
- Previously accepted types, architectures, shapes, layouts, and value ranges;
  numerical results, synchronization, memory behavior, layout semantics, and
  emitted operations. Decorators, dispatch, inheritance, and shared helpers can
  change behavior without changing a public signature.
- Return types, tuple arity, and stable types' public interfaces. A stable API's
  returned objects protect public members and Python special methods
  recursively; their concrete implementation classes and constructors do not
  automatically become stable.
- Policy-permitted extensions, diagnostic changes, and undefined behavior
  becoming an error, distinguished from breaks for formerly valid inputs. Ground
  findings in the actual preconditions and old behavior.

For removals or new deprecations, verify the replacement, declaration, table
entry, and earliest permitted removal release. Find the published release `N`
that first declared deprecation; require compatibility through `N` and the next
minor release `N+1`, with removal no earlier than `N+2`. Main-branch commit
dates and dev tags do not advance the window, and patch releases do not count as
minor releases. A declared removal version is insufficient without release
history confirming the retention period.

## Validate and report

Provide a minimal old call or behavioral assertion for suspected breaks. Run
focused tests at both endpoints when needed. Use available AMDGPU hardware for
minimal compilation or execution checks; do not run the full local suite. For
mixed CPU/GPU tensor tests, specify devices explicitly, run relevant cases under
both CPU and CUDA default devices, and restore the original setting.

Keep unavailable old environments, unsupported target architectures, and
uncertain historical commitments as unresolved findings. Distinguish local
verification from remote CI; read failed job logs before reproducing shared test
state.

Lead the report with one of: "compatibility violations found", "no violations
found within the verified scope", or "insufficient evidence to determine
compatibility". Include:

- Baseline release/SHA, target SHA or working-tree state, intended version, and
  policy versions used.
- Each affected API, prior contract, change, source location, reproduction or
  source evidence, and applicable retirement deadline.
- Concrete remedies such as preserving old paths, signatures, and behavior,
  adding compatibility wrappers, or completing the deprecation window. Adding a
  separate API alone usually does not preserve an existing commitment.
- Checks actually run, results, and remaining gaps. Identify newly stable APIs
  and permitted removals separately.

For a review-only request, deliver the report. If the user also authorized
fixes, complete the relevant fixes and focused validation, then update it.
