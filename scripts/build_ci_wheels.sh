#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MLIR_PATH:?MLIR_PATH must point to the prepared MLIR install}"
: "${ROCM_PATH:?ROCM_PATH must point to the ROCm SDK root}"
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

PYTHON_REQUESTED="${PYTHON_BIN:-python3}"
if ! PYTHON_BIN="$(command -v "${PYTHON_REQUESTED}")"; then
  echo "Python interpreter not found: ${PYTHON_REQUESTED}" >&2
  exit 1
fi
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ ! "${PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "Unable to determine Python major.minor version: ${PYTHON_VERSION}" >&2
  exit 1
fi
PYTHON_SUFFIX="${PYTHON_VERSION//./}"
PYTHON_BIN_ENV="PY${PYTHON_SUFFIX}_BIN"
OUTPUT_DIR="${CI_WHEEL_OUTPUT_DIR:-/tmp/flydsl-ci-wheels}"
VENV_ROOT="${CI_WHEEL_VENV_ROOT:-/tmp/flydsl-ci-wheel-venvs}"
BASE_WORKTREE="${CI_BASE_WORKTREE:-/tmp/flydsl-ci-base}"

echo "Building CI wheels with Python ${PYTHON_VERSION} (${PYTHON_BIN})"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

smoke_test_rocdl() {
  local chip input output
  local smoke_dir="${CI_ROCDL_SMOKE_DIR:-/tmp/flydsl-rocdl-smoke}"

  if [[ ! -x "${MLIR_PATH}/bin/mlir-opt" ]]; then
    echo "mlir-opt not found: ${MLIR_PATH}/bin/mlir-opt" >&2
    return 1
  fi

  rm -rf "${smoke_dir}"
  mkdir -p "${smoke_dir}"
  for chip in gfx942 gfx950; do
    input="${smoke_dir}/${chip}-input.mlir"
    output="${smoke_dir}/${chip}-output.mlir"
    cat >"${input}" <<MLIR
module attributes {gpu.container_module} {
  gpu.module @test [#rocdl.target<chip = "${chip}">] {
    llvm.func @kernel() attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
  }
}
MLIR

    "${MLIR_PATH}/bin/mlir-opt" "${input}" \
      --pass-pipeline='builtin.module(gpu-module-to-binary{format=fatbin})' \
      -o "${output}"
    grep -q "gpu.binary" "${output}"
    grep -q "bin = " "${output}"
    echo "ROCDL to HSACO smoke test passed for ${chip}"
  done
  rm -rf "${smoke_dir}"
}

build_wheel() {
  local source_dir="$1"
  local label="$2"
  # The baseline wheel needs the base commit's own MLIR: on a pin bump the base
  # source predates the API change the new pin requires. The default keeps every
  # other wheel on the shared install.
  local mlir_path="${3:-${MLIR_PATH}}"
  local destination="${OUTPUT_DIR}/${label}"
  local wheels=()
  local fly_opt="${source_dir}/build-fly/build_py${PYTHON_SUFFIX}/bin/fly-opt"

  echo "Building ${label} wheel from ${source_dir} (MLIR: ${mlir_path})"
  if ! (
    cd "${source_dir}"
    env \
      PYTHON_VERSIONS="${PYTHON_VERSION}" \
      "${PYTHON_BIN_ENV}=${PYTHON_BIN}" \
      VENV_ROOT="${VENV_ROOT}" \
      ALLOW_ANY_GLIBC=1 \
      MLIR_PATH="${mlir_path}" \
      bash scripts/build_wheels.sh
  ); then
    return 1
  fi

  shopt -s nullglob
  wheels=("${source_dir}"/dist/*.whl)
  shopt -u nullglob
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one ${label} wheel, found ${#wheels[@]}" >&2
    return 1
  fi
  if [[ ! -x "${fly_opt}" ]]; then
    echo "fly-opt not found: ${fly_opt}" >&2
    return 1
  fi

  mkdir -p "${destination}" || return 1
  cp "${wheels[0]}" "${destination}/" || return 1
  cp "${fly_opt}" "${destination}/fly-opt" || return 1
  (
    cd "${destination}"
    sha256sum ./*.whl fly-opt >SHA256SUMS
  ) || return 1
}

smoke_test_rocdl
build_wheel "${REPO_ROOT}" pr

base_repo_url="https://github.com/${BASE_REPO_NAME}.git"
cleanup_base_worktree() {
  git -C "${REPO_ROOT}" worktree remove --force "${BASE_WORKTREE}" || true
  git -C "${REPO_ROOT}" worktree prune || true
}

build_base_wheel() {
  if [[ "${BASE_COMMIT_SHA}" =~ ^0{40}$ ]]; then
    echo "Base commit is unavailable for an initial push" >&2
    return 1
  fi

  git -C "${REPO_ROOT}" worktree prune || true
  if [[ -e "${BASE_WORKTREE}" ]]; then
    echo "Base worktree path already exists: ${BASE_WORKTREE}" >&2
    return 1
  fi
  if ! git -C "${REPO_ROOT}" fetch "${base_repo_url}" "${BASE_COMMIT_SHA}" --no-tags --depth=1; then
    echo "Failed to fetch baseline ${BASE_REPO_NAME}@${BASE_COMMIT_SHA}" >&2
    return 1
  fi
  if ! git -C "${REPO_ROOT}" worktree add --detach "${BASE_WORKTREE}" FETCH_HEAD; then
    echo "Failed to create baseline worktree at ${BASE_WORKTREE}" >&2
    return 1
  fi

  trap cleanup_base_worktree EXIT
  if ! build_wheel "${BASE_WORKTREE}" base "${BASE_MLIR_PATH:-${MLIR_PATH}}"; then
    cleanup_base_worktree
    trap - EXIT
    return 1
  fi
  cleanup_base_worktree
  trap - EXIT
}

if ! build_base_wheel; then
  rm -rf "${OUTPUT_DIR}/base"
  echo "::warning title=Baseline wheel unavailable::PR tests will continue without the exact base benchmark."
fi

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
du -sh "${OUTPUT_DIR}" "${OUTPUT_DIR}/pr"
if [[ -d "${OUTPUT_DIR}/base" ]]; then
  du -sh "${OUTPUT_DIR}/base"
fi
