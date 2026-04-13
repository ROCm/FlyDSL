---
name: format-code
description: >
  Format and clean up code before committing. Removes unused imports/variables from Python files
  (autoflake), formats Python with black, and formats C/C++ with clang-format (Google style).
  Use this skill whenever the user says "format code", "clean up code", "lint", "format before commit",
  "code formatting", "/format-code", or wants to tidy up changed files before a git commit.
  Also trigger when the user mentions autoflake, black formatting, or clang-format in the context
  of cleaning up their working tree.
---

# Format Code

Format and clean up changed files before committing. Operates only on files that are staged
(`git diff --cached`) or modified in the working tree (`git diff`), so unchanged files are
never touched.

## Pipeline

For each changed file, the pipeline runs in order:

1. **Python (.py)**: autoflake (remove unused imports & variables) -> black (format)
2. **C/C++ (.c, .cc, .cpp, .cxx, .h, .hpp, .hxx)**: clang-format with Google style

## Steps

### 1. Ensure tools are installed

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

### 2. Collect changed files

Gather the union of staged and unstaged changed files (no duplicates):

```bash
(git diff --name-only --cached; git diff --name-only) | sort -u
```

If no files are changed, tell the user there is nothing to format and stop.

### 3. Format Python files

For every `.py` file in the changed set:

```bash
# Remove unused imports and variables (in-place)
autoflake --in-place --remove-all-unused-imports --remove-unused-variables "$file"

# Format with black (default settings)
black "$file"
```

### 4. Format C/C++ files

For every `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.hxx` file in the changed set:

```bash
clang-format -i --style=Google "$file"
```

### 5. Report summary

After formatting, print a summary listing:
- How many Python files were cleaned and formatted
- How many C/C++ files were formatted
- The names of all formatted files

If any files were staged before formatting, remind the user to re-stage them
(`git add <files>`) since the in-place edits made them show as modified again.

## Notes

- This skill never adds or removes files from git staging -- it only modifies file contents in place.
- Files that are not Python or C/C++ are silently skipped.
- autoflake's `--remove-unused-variables` only removes simple unused assignments (e.g. `x = 1`
  where `x` is never read). It does not remove unused functions or classes -- that requires
  manual review.
- black uses its default configuration. If the project has a `pyproject.toml` with `[tool.black]`
  settings, black will pick those up automatically.
