#!/usr/bin/env bash

# Reproduce the Python style gate used by CI for the committed branch range.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/check_python_style.sh [options]

Options:
  --base <ref>        Base ref for the branch check. Defaults to origin/main.
  --head <ref>        Head ref for the branch check. Defaults to HEAD.
  --include-local     Also check uncommitted, staged, and untracked Python files.
  --install           Install/upgrade black and ruff before checking.
  -h, --help          Show this help text.

Environment overrides:
  BASE_REF            Default base ref when --base is not provided.
  BASE_SHA            Exact base commit; skips merge-base resolution.
  HEAD_SHA            Default head ref when --head is not provided.

Examples:
  bash scripts/check_python_style.sh --install
  bash scripts/check_python_style.sh --base origin/main --head HEAD
  bash scripts/check_python_style.sh --include-local
EOF
}

BASE_REF="${BASE_REF:-origin/main}"
HEAD_REF="${HEAD_SHA:-HEAD}"
EXPLICIT_BASE_SHA="${BASE_SHA:-}"
INSTALL_TOOLS=false
INCLUDE_LOCAL=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base)
      if [ "$#" -lt 2 ]; then
        echo "--base requires a ref argument." >&2
        exit 2
      fi
      BASE_REF="$2"
      EXPLICIT_BASE_SHA=""
      shift 2
      ;;
    --head)
      if [ "$#" -lt 2 ]; then
        echo "--head requires a ref argument." >&2
        exit 2
      fi
      HEAD_REF="$2"
      shift 2
      ;;
    --include-local)
      INCLUDE_LOCAL=true
      shift
      ;;
    --install)
      INSTALL_TOOLS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ensure_style_tools() {
  local missing=()
  python3 -m black --version >/dev/null 2>&1 || missing+=("black")
  python3 -m ruff --version >/dev/null 2>&1 || missing+=("ruff")

  if [ "${#missing[@]}" -eq 0 ]; then
    return
  fi

  if [ "${INSTALL_TOOLS}" = true ]; then
    python3 -m pip install --upgrade black ruff
    return
  fi

  echo "Missing Python style tool(s): ${missing[*]}" >&2
  echo "Install them with: python3 -m pip install black ruff" >&2
  echo "Or rerun this script with --install." >&2
  exit 1
}

resolve_base_sha() {
  if [ -n "${EXPLICIT_BASE_SHA}" ]; then
    git cat-file -e "${EXPLICIT_BASE_SHA}^{commit}"
    printf '%s\n' "${EXPLICIT_BASE_SHA}"
    return
  fi

  if ! git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
    if [[ "${BASE_REF}" == */* ]]; then
      local remote="${BASE_REF%%/*}"
      local branch="${BASE_REF#*/}"
      git fetch --no-tags "${remote}" "${branch}"
    fi
  fi

  git merge-base "${HEAD_REF}" "${BASE_REF}"
}

filter_python_files() {
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
    if path.endswith(".py"):
        print(path)
'
}

ensure_style_tools

RESOLVED_BASE_SHA="$(resolve_base_sha)"
echo "Checking committed Python changes between ${RESOLVED_BASE_SHA} and ${HEAD_REF}."
BASE_SHA="${RESOLVED_BASE_SHA}" HEAD_SHA="${HEAD_REF}" USE_REVIEWDOG=false \
  bash .github/scripts/check_python_style.sh

if [ "${INCLUDE_LOCAL}" != true ]; then
  exit 0
fi

mapfile -t LOCAL_PY_FILES < <(
  {
    git diff --name-only --diff-filter=ACMR
    git diff --name-only --cached --diff-filter=ACMR
    git ls-files --others --exclude-standard -- '*.py'
  } | filter_python_files | sort -u
)

if [ "${#LOCAL_PY_FILES[@]}" -eq 0 ]; then
  echo "No uncommitted Python files to check."
  exit 0
fi

printf 'Checking uncommitted Python files:\n'
printf '  %s\n' "${LOCAL_PY_FILES[@]}"
python3 -m black --check --diff "${LOCAL_PY_FILES[@]}"
python3 -m ruff check "${LOCAL_PY_FILES[@]}"
