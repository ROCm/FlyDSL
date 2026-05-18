---
name: check-python-style
description: Reproduce FlyDSL's CI Python style check locally with scripts/check_python_style.sh. Use when Check Python Code Style fails, before pushing Python changes, or when working on Black/Ruff formatting issues.
---

# Check Python Style

Use this skill when the user asks about FlyDSL CI format failures, local style
checks, Black, Ruff, or the `Check Python Code Style` job.

## Default Command

Run the local wrapper from the repository root:

```bash
bash scripts/check_python_style.sh
```

If `black` or `ruff` is missing, install the same tools used by CI:

```bash
bash scripts/check_python_style.sh --install
```

## What It Checks

The script first calls `.github/scripts/check_python_style.sh` on the current
branch's committed Python diff against `origin/main`.

By default it only checks the committed branch range, matching pushed CI
contents. To also check uncommitted, staged, and untracked Python files, use:

```bash
bash scripts/check_python_style.sh --include-local
```

## Fixing Failures

To format the same committed Python diff before checking, use:

```bash
bash scripts/check_python_style.sh --fix
```

To also format local uncommitted, staged, and untracked Python files, use:

```bash
bash scripts/check_python_style.sh --fix --include-local
```

If local checks pass but PR CI still checks unrelated files, the PR branch is
likely behind `main`. Fetch `origin/main`, merge it into the PR branch when
appropriate, rerun this script, then push the merge commit.
