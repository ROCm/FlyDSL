#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ORIGINAL="${FLYDSL_ORIGINAL:-}"
FIXED="${FLYDSL_FIXED:-${ROOT_DIR}}"
OUT_DIR="${SCRIPT_DIR}/results"
EXTRA_ARGS=()

pythonpath_for_repo() {
  local repo="$1"
  local path=""

  if [[ -d "${repo}/build-fly/python_packages" ]]; then
    path="${repo}/build-fly/python_packages"
  elif [[ -d "${repo}/build/python_packages" ]]; then
    path="${repo}/build/python_packages"
  fi

  if [[ -d "${repo}/python" ]]; then
    path="${path:+${path}:}${repo}/python"
  fi

  path="${path:+${path}:}${repo}"
  printf '%s' "${path}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --original)
      ORIGINAL="$2"
      shift 2
      ;;
    --fixed)
      FIXED="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

run_one() {
  local label="$1"
  local repo="$2"
  local out="${OUT_DIR}/${label}.json"

  if [[ ! -d "${repo}" ]]; then
    echo "missing FlyDSL repo for ${label}: ${repo}" >&2
    exit 2
  fi

  mkdir -p "${OUT_DIR}/cache-${label}"
  echo
  echo "=== ${label} ==="
  echo "repo: ${repo}"
  echo "out:  ${out}"

  PYTHONPATH="$(pythonpath_for_repo "${repo}")${PYTHONPATH:+:${PYTHONPATH}}" \
  FLYDSL_RUNTIME_CACHE_DIR="${OUT_DIR}/cache-${label}" \
  python "${SCRIPT_DIR}/bench_flydsl_hotpath.py" \
    --label "${label}" \
    --output "${out}" \
    "${EXTRA_ARGS[@]}"
}

if [[ -z "${ORIGINAL}" ]]; then
  echo "missing original FlyDSL path; pass --original /path/to/original or set FLYDSL_ORIGINAL" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
run_one original "${ORIGINAL}"
run_one fixed "${FIXED}"

python - "${OUT_DIR}/original.json" "${OUT_DIR}/fixed.json" <<'PY'
import json
import sys
from pathlib import Path

orig = json.loads(Path(sys.argv[1]).read_text())
fixed = json.loads(Path(sys.argv[2]).read_text())

def get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

print("\n=== summary ===")
for kernel in sorted(orig.get("per_kernel", {})):
    o = get(orig, ["per_kernel", kernel, "paths", "jit_keyword_stream", "host_wall_us_per_call"])
    f = get(fixed, ["per_kernel", kernel, "paths", "jit_keyword_stream", "host_wall_us_per_call"])
    oc = get(orig, ["per_kernel", kernel, "paths", "compiled_positional", "host_wall_us_per_call"])
    fc = get(fixed, ["per_kernel", kernel, "paths", "compiled_positional", "host_wall_us_per_call"])
    if o is not None and f is not None:
        print(f"{kernel:<16s} jit original={o:8.2f} us  fixed={f:8.2f} us  speedup={o / f:5.2f}x")
    if oc is not None and fc is not None:
        print(f"{'':<16s} cmp original={oc:8.2f} us  fixed={fc:8.2f} us  speedup={oc / fc:5.2f}x")

om = get(orig, ["mixed_replay", "jit", "host_wall_us_per_call"])
fm = get(fixed, ["mixed_replay", "jit", "host_wall_us_per_call"])
if om is not None and fm is not None:
    print(f"mixed jit        original={om:8.2f} us  fixed={fm:8.2f} us  speedup={om / fm:5.2f}x")
PY
