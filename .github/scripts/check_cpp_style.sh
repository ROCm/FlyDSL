#!/bin/bash

# Check C/C++ style for files touched by the current push or pull request.
# The repository is expected to stay clang-format-clean; checking the diff keeps
# this fast pre-check focused on newly submitted code.

set -euo pipefail

BASE_SHA="${BASE_SHA:-}"
HEAD_SHA="${HEAD_SHA:-${GITHUB_SHA:-HEAD}}"

resolve_base_sha() {
  if [ -n "${BASE_SHA}" ] && git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
    printf '%s\n' "${BASE_SHA}"
    return
  fi

  if [ -n "${GITHUB_BASE_REF:-}" ]; then
    git fetch --no-tags --depth=1 origin "${GITHUB_BASE_REF}" || true
    if git rev-parse "origin/${GITHUB_BASE_REF}" >/dev/null 2>&1; then
      git merge-base "${HEAD_SHA}" "origin/${GITHUB_BASE_REF}"
      return
    fi
  fi

  if git rev-parse "${HEAD_SHA}^" >/dev/null 2>&1; then
    git rev-parse "${HEAD_SHA}^"
    return
  fi

  printf '\n'
}

BASE="$(resolve_base_sha)"

if [ -z "${BASE}" ]; then
  echo "Could not determine a base commit for style checks." >&2
  exit 1
fi

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format is required for C++ style checks." >&2
  exit 1
fi

echo "Checking C++ style for changes between ${BASE} and ${HEAD_SHA}."

mapfile -t CPP_FILES < <(
  git diff --name-only --diff-filter=ACMR "${BASE}" "${HEAD_SHA}" -- \
    '*.c' '*.cc' '*.cpp' '*.cxx' \
    '*.h' '*.hh' '*.hpp' '*.hxx' \
    '*.cu' '*.cuh' |
    python3 -c '
import sys

excluded_prefixes = (".claude/", "build/", "build-fly/", "thirdparty/")
cpp_suffixes = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh")
for path in sys.stdin:
    path = path.strip()
    if not path:
        continue
    if path.startswith(excluded_prefixes):
        continue
    if path.startswith("build_"):
        continue
    if path.endswith(cpp_suffixes):
        print(path)
'
)

if [ "${#CPP_FILES[@]}" -eq 0 ]; then
  echo "No changed C++ files to check."
  exit 0
fi

printf 'Changed C++ files:\n'
printf '  %s\n' "${CPP_FILES[@]}"

clang-format --dry-run --Werror "${CPP_FILES[@]}"
