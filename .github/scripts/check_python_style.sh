#!/bin/bash

# Check Python style for files touched by the current push or pull request.
# This keeps the fast pre-check focused on submitted code instead of requiring
# the whole historical tree to become black-clean in one CI change.

set -euo pipefail

BASE_SHA="${BASE_SHA:-}"
HEAD_SHA="${HEAD_SHA:-${GITHUB_SHA:-HEAD}}"
USE_REVIEWDOG="${USE_REVIEWDOG:-false}"
REVIEWDOG_REPORTER="${REVIEWDOG_REPORTER:-github-check}"
REVIEWDOG_FILTER_MODE="${REVIEWDOG_FILTER_MODE:-nofilter}"

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

echo "Checking Python style for changes between ${BASE} and ${HEAD_SHA}."

mapfile -t PY_FILES < <(
  git diff --name-only --diff-filter=ACMR "${BASE}" "${HEAD_SHA}" -- '*.py' |
    python3 -c '
import sys

excluded_prefixes = (".claude/", "build/", "build-fly/", "thirdparty/")
for path in sys.stdin:
    path = path.strip()
    if not path:
        continue
    if path.startswith(excluded_prefixes):
        continue
    if path.startswith("build_"):
        continue
    print(path)
'
)

if [ "${#PY_FILES[@]}" -eq 0 ]; then
  echo "No changed Python files to check."
  exit 0
fi

printf 'Changed Python files:\n'
printf '  %s\n' "${PY_FILES[@]}"

python3 -m black --check --diff "${PY_FILES[@]}"

if [ "${USE_REVIEWDOG}" = "true" ]; then
  if ! command -v reviewdog >/dev/null 2>&1; then
    echo "USE_REVIEWDOG=true but reviewdog is not installed." >&2
    exit 1
  fi

  python3 -m ruff check \
    --output-format=rdjson \
    --exit-zero \
    --no-fix \
    "${PY_FILES[@]}" \
  | reviewdog \
      -f=rdjson \
      -name="ruff" \
      -reporter="${REVIEWDOG_REPORTER}" \
      -filter-mode="${REVIEWDOG_FILTER_MODE}" \
      -fail-on-error=true
else
  python3 -m ruff check "${PY_FILES[@]}"
fi
