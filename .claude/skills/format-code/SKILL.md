---
name: format-code
description: >
  Format and clean up code before committing. Removes unused imports/variables from Python files
  (autoflake/ruff), formats Python with black, and formats C/C++ with clang-format (Google style).
  Use this skill whenever the user says "format code", "clean up code", "lint", "format before commit",
  "code formatting", "/format-code", or wants to tidy up changed files before a git commit.
  Also trigger when the user mentions autoflake, black, ruff, Python style checks, CI style failures,
  or clang-format in the context of cleaning up their working tree.
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
2. **Python (.py)**: autoflake/ruff (remove unused imports & variables) -> black (format)
3. **C/C++ (.c, .cc, .cpp, .cxx, .h, .hpp, .hxx)**: clang-format with Google style

## Steps

### 1. Ensure tools are installed

First check whether the repo has a formatter wrapper. In FlyDSL, run:

```bash
bash scripts/check_python_style.sh --fix
```

If that succeeds, inspect the resulting diff and skip the generic Python steps below unless
additional non-Python formatting is still needed. If required tooling is missing, use the
repo's install option if available, e.g. `bash scripts/check_python_style.sh --install`.

### 2. Ensure generic tools are installed

Check each tool and install any that are missing. Do all checks first, then install in one batch.

```bash
# Check availability
command -v autoflake &>/dev/null || NEED_PY=1
command -v black &>/dev/null || NEED_PY=1
command -v clang-format &>/dev/null || NEED_CF=1

# Install if needed
if [ -n "$NEED_PY" ]; then
  pip install autoflake black
fi
if [ -n "$NEED_CF" ]; then
  sudo apt-get install -y clang-format 2>/dev/null || pip install clang-format
fi
```

### 3. Collect changed files

Gather the union of staged and unstaged changed files (no duplicates):

```bash
(git diff --name-only --cached; git diff --name-only) | sort -u
```

If no files are changed, tell the user there is nothing to format and stop.

### 4. Format Python files

For every `.py` file in the changed set:

```bash
# Remove unused imports and variables (in-place)
autoflake --in-place --remove-all-unused-imports --remove-unused-variables "$file"

# Format with black (default settings)
black "$file"
```

### 5. Format C/C++ files

For every `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.hxx` file in the changed set:

```bash
clang-format -i --style=Google "$file"
```

### 6. Inspect diff and report summary

Always inspect the diff after automatic fixes. Ruff/autoflake can remove variables that appear
unused after simplifying a branch, but those variables may have been intentionally preserving a
compile-time decision or local readability. If an auto-fix changes behavior instead of only
format/import hygiene, restore the behavioral logic and rerun the formatter.

After formatting, print a summary listing:
- How many Python files were cleaned and formatted
- How many C/C++ files were formatted
- The names of all formatted files

If any files were staged before formatting, remind the user to re-stage them
(`git add <files>`) since the in-place edits made them show as modified again.

## Notes

- This skill never adds or removes files from git staging -- it only modifies file contents in place.
- Files that are not Python or C/C++ are silently skipped.
- In FlyDSL, `scripts/check_python_style.sh --fix` is the source of truth for CI Python style.
  It checks the committed Python diff against `origin/main`; use `--include-local` when the
  user explicitly wants uncommitted or untracked Python files included too.
- autoflake's `--remove-unused-variables` only removes simple unused assignments (e.g. `x = 1`
  where `x` is never read). It does not remove unused functions or classes -- that requires
  manual review.
- black uses its default configuration. If the project has a `pyproject.toml` with `[tool.black]`
  settings, black will pick those up automatically.
