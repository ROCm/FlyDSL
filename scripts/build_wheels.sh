#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Build FlyDSL wheels for Python 3.10 and 3.12.

This script:
  - Builds cp310 + cp312 wheels via `python setup.py bdist_wheel`
  - Uses separate FLIR_BUILD_DIR per Python to avoid ABI mixing
  - Relies on setup.py's built-in strip + auditwheel repair (manylinux tagging)

Usage:
  bash scripts/build_wheels.sh [--skip-build] [--install-deps]

Required env:
  MLIR_PATH=/path/to/llvm-project/build   (defaults to ./llvm-project/buildmlir if present)

Other knobs:
  FLIR_REBUILD=1|auto|0   # default: 1 (force rebuild per wheel)
  EXPECTED_GLIBC=2.35     # default: 2.35 (abort if glibc differs)
  ALLOW_ANY_GLIBC=1       # default: 0 (set to 1 to skip glibc check)
EOF
}

SKIP_BUILD=0
INSTALL_DEPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

FLIR_REBUILD="${FLIR_REBUILD:-1}"
EXPECTED_GLIBC="${EXPECTED_GLIBC:-2.35}"
ALLOW_ANY_GLIBC="${ALLOW_ANY_GLIBC:-0}"

PY310_BIN="${PY310_BIN:-python3.10}"
PY312_BIN="${PY312_BIN:-python3.12}"

VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venvs/release}"
VENV_310="${VENV_ROOT}/cp310"
VENV_312="${VENV_ROOT}/cp312"

FLIR_BUILD_DIR_310="${FLIR_BUILD_DIR_310:-.flir/build_py310}"
FLIR_BUILD_DIR_312="${FLIR_BUILD_DIR_312:-.flir/build_py312}"

MLIR_PATH="${MLIR_PATH:-${REPO_ROOT}/llvm-project/buildmlir}"
if [[ ! -d "${MLIR_PATH}" ]]; then
  echo "Error: MLIR_PATH not found: ${MLIR_PATH}" >&2
  echo "Set MLIR_PATH to your llvm-project build dir (must contain lib/cmake/mlir)." >&2
  exit 1
fi

need_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    return 1
  fi
}

apt_install_if_possible() {
  local pkgs=("$@")
  if [[ "${INSTALL_DEPS}" != "1" ]]; then
    return 1
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi
  if [[ "$(id -u)" != "0" ]]; then
    echo "Error: --install-deps requested, but not running as root." >&2
    return 1
  fi
  apt-get update
  apt-get install -y "${pkgs[@]}"
}

ensure_host_deps() {
  local missing=0
  if ! need_cmd cmake; then
    echo "Missing: cmake" >&2
    missing=1
  fi
  if ! need_cmd gcc; then
    echo "Missing: gcc" >&2
    missing=1
  fi
  if ! need_cmd g++; then
    echo "Missing: g++" >&2
    missing=1
  fi
  if ! need_cmd patchelf; then
    echo "Missing: patchelf (needed by auditwheel repair)" >&2
    missing=1
  fi

  if [[ "${missing}" == "1" ]]; then
    echo "" >&2
    echo "Attempting to install deps (because --install-deps was set)..." >&2
    apt_install_if_possible cmake gcc g++ patchelf || true
  fi

  # Re-check after optional install.
  for c in cmake gcc g++ patchelf; do
    if ! need_cmd "${c}"; then
      echo "Error: required command not found: ${c}" >&2
      echo "Install it and re-run (or use --install-deps as root on Ubuntu/Debian)." >&2
      exit 1
    fi
  done
}

glibc_version() {
  # Example: "ldd (Ubuntu GLIBC 2.35-0ubuntu3.11) 2.35"
  # We extract the last token of the first line.
  local line
  line="$(ldd --version 2>/dev/null | head -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  echo "${line}" | awk '{print $NF}'
}

ensure_glibc() {
  if [[ "${ALLOW_ANY_GLIBC}" == "1" ]]; then
    return 0
  fi
  if ! need_cmd ldd; then
    echo "Warning: ldd not found; skipping glibc check." >&2
    return 0
  fi
  local got
  got="$(glibc_version)"
  if [[ -z "${got}" ]]; then
    echo "Warning: failed to detect glibc version via ldd; skipping glibc check." >&2
    return 0
  fi
  if [[ "${got}" != "${EXPECTED_GLIBC}" ]]; then
    echo "Error: glibc version mismatch. Expected ${EXPECTED_GLIBC}, got ${got}." >&2
    echo "This affects the best possible manylinux tag auditwheel can assign." >&2
    echo "Options:" >&2
    echo "  - Build on glibc ${EXPECTED_GLIBC}, or" >&2
    echo "  - Use scripts/build_manylinux_2_28.sh to build in a manylinux container, or" >&2
    echo "  - Override: EXPECTED_GLIBC=${got} or set ALLOW_ANY_GLIBC=1" >&2
    exit 1
  fi
}

ensure_python_bins() {
  local missing=0
  if ! command -v "${PY310_BIN}" >/dev/null 2>&1; then
    echo "Missing: ${PY310_BIN}" >&2
    missing=1
  fi
  if ! command -v "${PY312_BIN}" >/dev/null 2>&1; then
    echo "Missing: ${PY312_BIN}" >&2
    missing=1
  fi

  if [[ "${missing}" == "1" ]]; then
    echo "" >&2
    echo "Attempting to install Python deps (because --install-deps was set)..." >&2

    local pkgs=()
    if ! command -v "${PY310_BIN}" >/dev/null 2>&1; then
      pkgs+=( python3.10 python3.10-dev python3.10-venv )
    fi
    if ! command -v "${PY312_BIN}" >/dev/null 2>&1; then
      pkgs+=( python3.12 python3.12-dev python3.12-venv )
    fi
    if [[ "${#pkgs[@]}" -gt 0 ]]; then
      apt_install_if_possible "${pkgs[@]}" || true
    fi
  fi

  # Re-check after optional install.
  for py in "${PY310_BIN}" "${PY312_BIN}"; do
    if ! command -v "${py}" >/dev/null 2>&1; then
      echo "Error: Python binary not found on PATH: ${py}" >&2
      echo "Install it (and its -dev/-venv packages) or override PY310_BIN/PY312_BIN." >&2
      exit 1
    fi
  done
}

create_venv_and_deps() {
  local pybin="$1"
  local venv="$2"

  rm -rf "${venv}"
  mkdir -p "$(dirname "${venv}")"
  "${pybin}" -m venv "${venv}"

  "${venv}/bin/python" -m pip install -U pip setuptools wheel
  # build deps + packaging tooling
  "${venv}/bin/python" -m pip install -U numpy nanobind pybind11 auditwheel twine
}

build_one() {
  local pybin="$1"
  local venv="$2"
  local build_dir_rel="$3"
  local py_tag="$4"

  echo "[build] ${py_tag} using ${pybin}"
  create_venv_and_deps "${pybin}" "${venv}"

  # Ensure venv tools are on PATH so setup.py can find `auditwheel`.
  PATH="${venv}/bin:${PATH}" \
  MLIR_PATH="${MLIR_PATH}" \
  FLIR_BUILD_DIR="${build_dir_rel}" \
  FLIR_REBUILD="${FLIR_REBUILD}" \
  "${venv}/bin/python" setup.py bdist_wheel

  # Validate artifact
  if ! ls -1 "dist/"*"-${py_tag}-${py_tag}-manylinux_"*.whl >/dev/null 2>&1; then
    echo "Error: expected a manylinux wheel for ${py_tag} under dist/ but didn't find one." >&2
    echo "Hint: ensure auditwheel+patchelf are available and setup.py auditwheel repair succeeded." >&2
    echo "dist contents:" >&2
    ls -1 dist || true
    exit 1
  fi
}


main() {
  ensure_host_deps
  ensure_glibc
  ensure_python_bins

  mkdir -p dist

  if [[ "${SKIP_BUILD}" != "1" ]]; then
    rm -rf dist
    mkdir -p dist

    build_one "${PY310_BIN}" "${VENV_310}" "${FLIR_BUILD_DIR_310}" "cp310"
    build_one "${PY312_BIN}" "${VENV_312}" "${FLIR_BUILD_DIR_312}" "cp312"
  fi

  echo ""
  echo "[done] dist artifacts:"
  ls -lh dist/*.whl
}

main

