---
name: format-code
description: >
  Format, clean up, and style-check changed files, matching the project's CI style gate.
  Formats Python with black + ruff and C/C++ with clang-format using the repository's
  .clang-format, and can also run check-only to reproduce the CI gate locally without editing
  files. Use when the user says "format code", "clean up code", "lint", "format before commit",
  "/format-code", wants to reproduce the "Check Python Code Style" CI job locally, is fixing a
  CI style failure, is about to push Python changes, or mentions black, ruff, or clang-format.
---

# Format Code

Format and clean up changed files before committing. Prefer repository-provided style
wrappers when they exist, because they encode the same file selection and tool flags as CI.
For FlyDSL, use `bash scripts/check_python_style.sh --fix` for Python style fixes.

If there is no project wrapper, operate only on files that are staged (`git diff --cached`)
or modified in the working tree (`git diff`), so unchanged files are never touched.

## Pipeline

For each changed file, the pipeline runs in order:

1. **Project wrapper**: run the repo's style script when present, e.g. FlyDSL's `scripts/check_python_style.sh --fix`
2. **Python (.py)**: `ruff check --fix` (remove unused imports/variables, sort imports) -> `black` (format)
3. **C/C++ (.c, .cc, .cpp, .cxx, .h, .hh, .hpp, .hxx, .cu, .cuh)**: clang-format using the repository's `.clang-format`

## Steps

### 1. Prefer the project wrapper

First check whether the repo has a formatter wrapper. In FlyDSL, run:

```bash
bash scripts/check_python_style.sh --fix
```

This wrapper runs `black -> ruff check --fix -> black` on the committed diff, exactly matching
CI. If it succeeds, inspect the resulting diff and skip the generic Python steps below unless
additional non-Python formatting is still needed. If required tooling is missing, use the
repo's install option, e.g. `bash scripts/check_python_style.sh --install`.

### 2. Ensure generic tools are installed

Check each tool and install any that are missing. Match the versions CI uses so local output
matches the CI gate: FlyDSL's CI runs `clang-format-18`.

```bash
# Check availability
command -v black &>/dev/null || NEED_PY=1
command -v ruff &>/dev/null || NEED_PY=1
command -v clang-format-18 &>/dev/null || command -v clang-format &>/dev/null || NEED_CF=1

# Install if needed
if [ -n "$NEED_PY" ]; then
  pip install black ruff
fi
if [ -n "$NEED_CF" ]; then
  sudo apt-get install -y clang-format-18 2>/dev/null || pip install clang-format
fi
```

### 3. Collect changed files

Gather the union of staged and unstaged changed files (no duplicates):

```bash
(git diff --name-only --cached; git diff --name-only) | sort -u
```

If no files are changed, tell the user there is nothing to format and stop.

### 4. Format Python files

For every `.py` file in the changed set (run from the repo root so `pyproject.toml` config is
picked up). This mirrors CI, which checks `black` formatting and `ruff check` (rules E/W/F/I):

```bash
# Remove unused imports/variables and sort imports, then format
ruff check --fix "$file"
black "$file"
```

### 5. Format C/C++ files

For every `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hh`, `.hpp`, `.hxx`, `.cu`, `.cuh` file in the
changed set. Do **not** pass `--style`: clang-format reads the repository's `.clang-format`
(FlyDSL uses LLVM style with ColumnLimit 100), which is what CI checks. Prefer the CI version:

```bash
${CLANG_FORMAT:-clang-format-18} -i "$file"
```

### 6. Inspect diff and report summary

Always inspect the diff after automatic fixes. Ruff can remove variables that appear unused
after simplifying a branch, but those variables may have been intentionally preserving a
compile-time decision or local readability. If an auto-fix changes behavior instead of only
format/import hygiene, restore the behavioral logic and rerun the formatter.

After formatting, print a summary listing:
- How many Python files were cleaned and formatted
- How many C/C++ files were formatted
- The names of all formatted files

If any files were staged before formatting, remind the user to re-stage them
(`git add <files>`) since the in-place edits made them show as modified again.

## Check only (reproduce the CI gate without editing files)

When the user wants to know whether the `Check Python Code Style` CI job will pass -- before
pushing, or when that job has already failed -- run the wrapper without `--fix` so nothing is
modified:

```bash
bash scripts/check_python_style.sh            # check committed Python diff vs origin/main (matches CI)
bash scripts/check_python_style.sh --install  # install the black/ruff versions CI uses, if missing
```

What it checks:

- By default it only checks the committed branch range (`origin/main`..HEAD), matching what CI
  sees on a pushed branch.
- Add `--include-local` to also check uncommitted, staged, and untracked Python files:
  `bash scripts/check_python_style.sh --include-local`.

To fix what the check reports, re-run with `--fix` (the formatting path above); add
`--include-local` to also format local uncommitted/untracked files:

```bash
bash scripts/check_python_style.sh --fix
bash scripts/check_python_style.sh --fix --include-local
```

If local checks pass but PR CI still flags unrelated files, the PR branch is likely behind
`main`. Fetch `origin/main`, merge it into the PR branch when appropriate, rerun the check,
then push the merge commit.

## Notes

- This skill never adds or removes files from git staging -- it only modifies file contents in place.
- Files that are not Python or C/C++ are silently skipped.
- In FlyDSL, `scripts/check_python_style.sh --fix` is the source of truth for CI Python style.
  It checks the committed Python diff against `origin/main`; use `--include-local` when the
  user explicitly wants uncommitted or untracked Python files included too.
- `ruff check --fix` removes unused imports (F401) and simple unused variables (F841) and sorts
  imports (I). It does not remove unused functions or classes -- that requires manual review.
- black and ruff read `pyproject.toml` ([tool.black] / [tool.ruff], line-length 120) automatically
  when run from the repo root. Pin clang-format to the CI version (clang-format-18) so local
  formatting does not drift from the CI check.
