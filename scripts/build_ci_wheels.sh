#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MLIR_PATH:?MLIR_PATH must point to the prepared MLIR install}"
: "${BASE_REPO_NAME:?BASE_REPO_NAME must be set}"
: "${BASE_COMMIT_SHA:?BASE_COMMIT_SHA must be set}"

if [[ ! "${BASE_REPO_NAME}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid BASE_REPO_NAME: ${BASE_REPO_NAME}" >&2
  exit 2
fi
if [[ ! "${BASE_COMMIT_SHA}" =~ ^[0-9a-fA-F]{40}$ && "${BASE_COMMIT_SHA}" != "main" ]]; then
  echo "BASE_COMMIT_SHA must be a 40-character commit or main: ${BASE_COMMIT_SHA}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${CI_WHEEL_OUTPUT_DIR:-/tmp/flydsl-ci-wheels}"
VENV_ROOT="${CI_WHEEL_VENV_ROOT:-/tmp/flydsl-ci-wheel-venvs}"
BASE_WORKTREE="${CI_BASE_WORKTREE:-/tmp/flydsl-ci-base}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

build_wheel() {
  local source_dir="$1"
  local label="$2"
  local destination="${OUTPUT_DIR}/${label}"
  local wheels=()
  local fly_opt="${source_dir}/build-fly/build_py312/bin/fly-opt"

  echo "Building ${label} wheel from ${source_dir}"
  (
    cd "${source_dir}"
    PYTHON_VERSIONS=3.12 \
      PY312_BIN="${PYTHON_BIN}" \
      VENV_ROOT="${VENV_ROOT}" \
      ALLOW_ANY_GLIBC=1 \
      MLIR_PATH="${MLIR_PATH}" \
      bash scripts/build_wheels.sh
  )

  shopt -s nullglob
  wheels=("${source_dir}"/dist/*.whl)
  shopt -u nullglob
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one ${label} wheel, found ${#wheels[@]}" >&2
    exit 1
  fi
  if [[ ! -x "${fly_opt}" ]]; then
    echo "fly-opt not found: ${fly_opt}" >&2
    exit 1
  fi

  mkdir -p "${destination}"
  cp "${wheels[0]}" "${destination}/"
  cp "${fly_opt}" "${destination}/fly-opt"
  (
    cd "${destination}"
    sha256sum ./*.whl fly-opt >SHA256SUMS
  )
}

build_wheel "${REPO_ROOT}" pr

if [[ -e "${BASE_WORKTREE}" ]]; then
  echo "Base worktree path already exists: ${BASE_WORKTREE}" >&2
  exit 1
fi

base_repo_url="https://github.com/${BASE_REPO_NAME}.git"
git -C "${REPO_ROOT}" fetch "${base_repo_url}" "${BASE_COMMIT_SHA}" --no-tags --depth=1
git -C "${REPO_ROOT}" worktree add --detach "${BASE_WORKTREE}" FETCH_HEAD
cleanup_base_worktree() {
  git -C "${REPO_ROOT}" worktree remove --force "${BASE_WORKTREE}" || true
}
trap cleanup_base_worktree EXIT
build_wheel "${BASE_WORKTREE}" base
cleanup_base_worktree
trap - EXIT

if [[ ! -x "${MLIR_PATH}/bin/FileCheck" ]]; then
  echo "FileCheck not found: ${MLIR_PATH}/bin/FileCheck" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}/tools"
cp "${MLIR_PATH}/bin/FileCheck" "${OUTPUT_DIR}/tools/FileCheck"
(
  cd "${OUTPUT_DIR}/tools"
  sha256sum FileCheck >SHA256SUMS
)

pr_wheels=("${OUTPUT_DIR}"/pr/*.whl)
"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${pr_wheels[0]}"
"${PYTHON_BIN}" -c "import flydsl; from flydsl._mlir.ir import Context; print('Validated FlyDSL CI wheel')"

echo "FlyDSL CI wheel artifacts:"
du -sh "${OUTPUT_DIR}" "${OUTPUT_DIR}/pr" "${OUTPUT_DIR}/base"
