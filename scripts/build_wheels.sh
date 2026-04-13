#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Python versions to build wheels for. Aligned with TheRock's supported set
# (https://github.com/ROCm/TheRock RELEASES.md): 3.10, 3.11, 3.12, 3.13.
# Override via env: PYTHON_VERSIONS="3.10 3.13" bash scripts/build_wheels.sh
DEFAULT_PYTHON_VERSIONS="3.10 3.11 3.12 3.13"
IFS=' ' read -r -a PYTHON_VERSIONS <<< "${PYTHON_VERSIONS:-${DEFAULT_PYTHON_VERSIONS}}"

usage() {
  cat <<EOF
Build FlyDSL wheels for one or more CPython versions.

Default versions: ${DEFAULT_PYTHON_VERSIONS}

Usage:
  bash scripts/build_wheels.sh [--skip-build] [--install-deps]

Override versions:
  PYTHON_VERSIONS="3.10 3.13" bash scripts/build_wheels.sh

Required env:
  MLIR_PATH    path to llvm-project build (defaults to ./llvm-project/mlir_install)

Other knobs:
  FLY_REBUILD=1|auto|0   (default: 1)
  EXPECTED_GLIBC=2.35    (default: 2.35, set ALLOW_ANY_GLIBC=1 to skip check)
  PYTHON_VERSIONS="..."  whitespace-separated list, default "${DEFAULT_PYTHON_VERSIONS}"
  PY<MAJ><MIN>_BIN       per-version python binary override (e.g. PY313_BIN=/opt/py313/bin/python3.13)
  FLYDSL_PACKAGE_VERSION_OVERRIDE=...  exact package version for release automation
  FLY_REBUILD=1|auto|0                 (default: 1)
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

FLY_REBUILD="${FLY_REBUILD:-1}"
EXPECTED_GLIBC="${EXPECTED_GLIBC:-2.35}"
ALLOW_ANY_GLIBC="${ALLOW_ANY_GLIBC:-0}"

VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venvs/release}"

# Resolve the python binary for a version like "3.13".
# Honors PY<MAJ><MIN>_BIN env override (e.g. PY313_BIN), else falls back to
# `python<ver>` on PATH.
py_bin_for_version() {
  local ver="$1"
  local env_var_name="PY${ver//./}_BIN"
  local val="${!env_var_name:-python${ver}}"
  echo "${val}"
}

# Convert "3.13" -> "cp313".
py_tag_for_version() {
  local ver="$1"
  echo "cp${ver//./}"
}

if [[ -z "${MLIR_PATH:-}" ]]; then
  MLIR_PATH="${REPO_ROOT}/llvm-project/mlir_install"
fi
if [[ ! -d "${MLIR_PATH}" ]]; then
  echo "Error: MLIR_PATH not found: ${MLIR_PATH}" >&2
  echo "Set MLIR_PATH to your llvm-project build dir (must contain lib/cmake/mlir)." >&2
  exit 1
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

apt_install_if_possible() {
  local pkgs=("$@")
  [[ "${INSTALL_DEPS}" == "1" ]] || return 1
  command -v apt-get >/dev/null 2>&1 || return 1
  if [[ "$(id -u)" != "0" ]]; then
    echo "Error: --install-deps requested, but not running as root." >&2
    return 1
  fi
  apt-get update
  apt-get install -y "${pkgs[@]}"
}

ensure_host_deps() {
  local missing=0
  for c in cmake gcc g++ patchelf; do
    if ! need_cmd "${c}"; then
      echo "Missing: ${c}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" == "1" ]]; then
    apt_install_if_possible cmake gcc g++ patchelf || true
  fi

  for c in cmake gcc g++ patchelf; do
    if ! need_cmd "${c}"; then
      echo "Error: required command not found: ${c}" >&2
      echo "Install it and re-run (or use --install-deps as root on Ubuntu/Debian)." >&2
      exit 1
    fi
  done
}

glibc_version() {
  local line
  line="$(ldd --version 2>/dev/null | head -n 1 || true)"
  [[ -n "${line}" ]] && echo "${line}" | awk '{print $NF}' || echo ""
}

ensure_glibc() {
  [[ "${ALLOW_ANY_GLIBC}" != "1" ]] || return 0
  need_cmd ldd || { echo "Warning: ldd not found; skipping glibc check." >&2; return 0; }
  local got
  got="$(glibc_version)"
  [[ -n "${got}" ]] || { echo "Warning: cannot detect glibc version; skipping." >&2; return 0; }
  if [[ "${got}" != "${EXPECTED_GLIBC}" ]]; then
    echo "Error: glibc ${EXPECTED_GLIBC} expected, got ${got}." >&2
    echo "Override: EXPECTED_GLIBC=${got} or ALLOW_ANY_GLIBC=1" >&2
    exit 1
  fi
}

ensure_python_bins() {
  local ver py_bin missing=0
  for ver in "${PYTHON_VERSIONS[@]}"; do
    py_bin="$(py_bin_for_version "${ver}")"
    if ! command -v "${py_bin}" >/dev/null 2>&1; then
      echo "Missing: ${py_bin}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" == "1" ]]; then
    local pkgs=()
    for ver in "${PYTHON_VERSIONS[@]}"; do
      py_bin="$(py_bin_for_version "${ver}")"
      if ! command -v "${py_bin}" >/dev/null 2>&1; then
        pkgs+=( "python${ver}" "python${ver}-dev" "python${ver}-venv" )
      fi
    done
    [[ "${#pkgs[@]}" -eq 0 ]] || apt_install_if_possible "${pkgs[@]}" || true
  fi

  for ver in "${PYTHON_VERSIONS[@]}"; do
    py_bin="$(py_bin_for_version "${ver}")"
    if ! command -v "${py_bin}" >/dev/null 2>&1; then
      echo "Error: Python not found: ${py_bin}" >&2
      exit 1
    fi
  done
}

create_venv_and_deps() {
  local pybin="$1"
  local venv="$2"

  if [[ -x "${venv}/bin/python" ]] && "${venv}/bin/python" -c "import nanobind, auditwheel" 2>/dev/null; then
    echo "[venv] Reusing existing venv at ${venv}"
    return 0
  fi

  echo "[venv] Creating venv at ${venv} ..."
  rm -rf "${venv}"
  mkdir -p "$(dirname "${venv}")"
  "${pybin}" -m venv "${venv}"

  "${venv}/bin/python" -m pip install -U pip setuptools wheel
  local nanobind_ver="${NANOBIND_VERSION:-2.12.0}"
  "${venv}/bin/python" -m pip install -U numpy "nanobind==${nanobind_ver}" pybind11 auditwheel twine
}

build_one() {
  local pybin="$1"
  local venv="$2"
  local build_dir_rel="$3"
  local py_tag="$4"
  local log_file="${5:-/dev/stderr}"

  echo "[build] ${py_tag} using ${pybin}"
  create_venv_and_deps "${pybin}" "${venv}"

  # Use a per-version setuptools build base to avoid race conditions
  # when building multiple versions in parallel (shared build/ dir).
  local setup_build_base="${build_dir_rel}/setup_build"
  mkdir -p "${setup_build_base}"

  # Build C++ and Python packages
  # Use --egg-base to isolate egg-info per version (avoid race on shared python/flydsl.egg-info/).
  # FLY_WHEEL_BUILD=1 prevents setup.py and build.sh from mutating the shared source tree
  # (python/flydsl/_mlir symlink), which would race across parallel builds.
  PATH="${venv}/bin:${PATH}" \
  MLIR_PATH="${MLIR_PATH}" \
  FLY_BUILD_DIR="${build_dir_rel}" \
  FLY_REBUILD="${FLY_REBUILD}" \
  FLY_WHEEL_BUILD=1 \
  FLYDSL_RELEASE_TYPE="${FLYDSL_RELEASE_TYPE:-}" \
  FLYDSL_PACKAGE_VERSION_OVERRIDE="${FLYDSL_PACKAGE_VERSION_OVERRIDE:-}" \
  "${venv}/bin/python" setup.py egg_info --egg-base "${setup_build_base}" build -b "${setup_build_base}"

  # Strip shared libs in the build tree BEFORE bdist_wheel copies them.
  # Remove unversioned libFlyPythonCAPI.so (symlink that bdist_wheel
  # would copy as a second full-size file, doubling the wheel size).
  local _mlir_libs="${build_dir_rel}/python_packages/flydsl/_mlir/_mlir_libs"
  if [[ -d "${_mlir_libs}" ]]; then
    rm -f "${_mlir_libs}/libFlyPythonCAPI.so"
    find "${_mlir_libs}" \( -name '*.so' -o -name '*.so.*' \) ! -name '*nanobind*' \
      -exec strip --strip-unneeded {} + 2>/dev/null || true
    # Also strip the setuptools copy if it already exists
    for d in "${setup_build_base}"/lib.*/flydsl/_mlir/_mlir_libs; do
      [[ -d "${d}" ]] || continue
      rm -f "${d}/libFlyPythonCAPI.so"
      find "${d}" \( -name '*.so' -o -name '*.so.*' \) ! -name '*nanobind*' \
        -exec strip --strip-unneeded {} + 2>/dev/null || true
    done
  fi

  # Package wheel (C++ already built, skip rebuild)
  PATH="${venv}/bin:${PATH}" \
  MLIR_PATH="${MLIR_PATH}" \
  FLY_BUILD_DIR="${build_dir_rel}" \
  FLY_REBUILD=0 \
  FLY_WHEEL_BUILD=1 \
  FLYDSL_RELEASE_TYPE="${FLYDSL_RELEASE_TYPE:-}" \
  FLYDSL_PACKAGE_VERSION_OVERRIDE="${FLYDSL_PACKAGE_VERSION_OVERRIDE:-}" \
  "${venv}/bin/python" setup.py egg_info --egg-base "${setup_build_base}" build -b "${setup_build_base}" bdist_wheel

  if ! ls -1 "dist/"*"-${py_tag}-${py_tag}-manylinux_"*.whl >/dev/null 2>&1; then
    echo "Error: expected a manylinux wheel for ${py_tag} under dist/ but didn't find one." >&2
    ls -1 dist || true
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Parallel build: max concurrency = PARALLEL_WHEELS (default: 3)
# Each version gets its own build dir, venv, and log file — no conflicts.
# ---------------------------------------------------------------------------
PARALLEL_WHEELS="${PARALLEL_WHEELS:-3}"

main() {
  echo "Building wheels for Python versions: ${PYTHON_VERSIONS[*]}"
  echo "Parallel jobs: ${PARALLEL_WHEELS}"

  ensure_host_deps
  ensure_glibc
  ensure_python_bins

  mkdir -p dist

  if [[ "${SKIP_BUILD}" != "1" ]]; then
    # Initialize submodules once before parallel builds to avoid git race conditions.
    git -C "${REPO_ROOT}" submodule update --init --recursive

    rm -rf dist
    mkdir -p dist

    local log_dir
    log_dir="$(mktemp -d "${REPO_ROOT}/.wheel_build_logs.XXXXXX")"
    local pids=()
    local tags=()
    local logs=()
    local running=0

    for ver in "${PYTHON_VERSIONS[@]}"; do
      local py_bin py_tag venv build_dir log_file
      py_bin="$(py_bin_for_version "${ver}")"
      py_tag="$(py_tag_for_version "${ver}")"
      venv="${VENV_ROOT}/${py_tag}"
      build_dir="build-fly/build_py${ver//./}"
      log_file="${log_dir}/${py_tag}.log"

      echo "[parallel] Starting ${py_tag} (log: ${log_file})"
      build_one "${py_bin}" "${venv}" "${build_dir}" "${py_tag}" "${log_file}" \
        > "${log_file}" 2>&1 &
      pids+=($!)
      tags+=("${py_tag}")
      logs+=("${log_file}")
      running=$((running + 1))

      # Throttle: wait for a slot if we hit the concurrency limit
      if [[ "${running}" -ge "${PARALLEL_WHEELS}" ]]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
      fi
    done

    # Wait for all remaining builds to finish
    local failed=0
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "FAILED: ${tags[$i]} (see ${logs[$i]})" >&2
        tail -20 "${logs[$i]}" >&2
        failed=1
      else
        echo "OK: ${tags[$i]}"
      fi
    done

    # Show logs for any failures, then clean up
    if [[ "${failed}" == "1" ]]; then
      echo "Some wheel builds failed. Logs preserved at: ${log_dir}" >&2
      exit 1
    fi
    rm -rf "${log_dir}"
  fi

  echo ""
  echo "[done] dist artifacts:"
  ls -lh dist/*.whl
}

main
