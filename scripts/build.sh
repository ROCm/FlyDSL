#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Build directory (default: build-fly/, overridable via FLY_BUILD_DIR)
# ---------------------------------------------------------------------------
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

# ---------------------------------------------------------------------------
# Parallelism: default $(nproc), overridable via -jN argument
# ---------------------------------------------------------------------------
PARALLEL_JOBS=$(nproc)
for arg in "$@"; do
  if [[ "$arg" =~ ^-j([0-9]+)$ ]]; then
    PARALLEL_JOBS="${BASH_REMATCH[1]}"
  fi
done

# ---------------------------------------------------------------------------
# Discover MLIR_PATH
# ---------------------------------------------------------------------------
if [ -z "${MLIR_PATH:-}" ]; then
  candidates=(
    "${BASE_DIR}/llvm-project-flydsl/build-flydsl/mlir_install"
    "${BASE_DIR}/llvm-project/build-flydsl/mlir_install"
    "${BASE_DIR}/llvm-project/mlir_install"
  )
  for p in "${candidates[@]}"; do
    if [ -d "${p}/lib/cmake/mlir" ]; then
      echo "Auto-detected MLIR_PATH: ${p}"
      export MLIR_PATH="${p}"
      break
    fi
  done
fi

if [ -z "${MLIR_PATH:-}" ]; then
  echo "Error: MLIR_PATH not set and could not be auto-detected." >&2
  echo "Build LLVM/MLIR first:  bash scripts/build_llvm.sh" >&2
  echo "Or set:  export MLIR_PATH=/path/to/mlir_install" >&2
  exit 1
fi

echo "=============================================="
echo "FlyDSL Build"
echo "  REPO_ROOT:  ${REPO_ROOT}"
echo "  BUILD_DIR:  ${BUILD_DIR}"
echo "  MLIR_PATH:  ${MLIR_PATH}"
echo "  PARALLEL:   -j${PARALLEL_JOBS}"
echo "=============================================="

# ---------------------------------------------------------------------------
# Initialize git submodules if needed
# ---------------------------------------------------------------------------
if [ ! -f "${REPO_ROOT}/thirdparty/dlpack/include/dlpack/dlpack.h" ]; then
  echo "Initializing git submodules..."
  git -C "${REPO_ROOT}" submodule update --init --recursive
fi

# ---------------------------------------------------------------------------
# CMake configure
# ---------------------------------------------------------------------------
NANOBIND_DIR=$(python3 -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__) + '/cmake')" 2>/dev/null || true)

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake_args=(
  "${REPO_ROOT}"
  -DMLIR_DIR="${MLIR_PATH}/lib/cmake/mlir"
  -DPython3_EXECUTABLE="$(which python3)"
)
if [ -n "${NANOBIND_DIR}" ]; then
  cmake_args+=(-Dnanobind_DIR="${NANOBIND_DIR}")
fi

echo "Configuring CMake..."
cmake "${cmake_args[@]}"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "Building with -j${PARALLEL_JOBS}..."
cmake --build . -j"${PARALLEL_JOBS}"

# ---------------------------------------------------------------------------
# Ensure python/flydsl/_mlir symlink points to build output so that
# `import flydsl` works without PYTHONPATH or `pip install -e .`.
# The CMake copy step (CopyFlyPythonSources) now skips _mlir via
# cmake/copy_flydsl_sources.cmake, so this symlink is safe to keep.
# ---------------------------------------------------------------------------
_EDITABLE_MLIR_LINK="${REPO_ROOT}/python/flydsl/_mlir"
_MLIR_BUILD_TARGET="${BUILD_DIR}/python_packages/flydsl/_mlir"
_RELATIVE_TARGET="../../${BUILD_DIR##${REPO_ROOT}/}/python_packages/flydsl/_mlir"

if [ -d "${_MLIR_BUILD_TARGET}" ]; then
  if [ -L "${_EDITABLE_MLIR_LINK}" ]; then
    rm -f "${_EDITABLE_MLIR_LINK}"
  fi
  if [ ! -e "${_EDITABLE_MLIR_LINK}" ]; then
    ln -s "${_RELATIVE_TARGET}" "${_EDITABLE_MLIR_LINK}"
    echo "Created symlink: ${_EDITABLE_MLIR_LINK} -> ${_RELATIVE_TARGET}"
  fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
