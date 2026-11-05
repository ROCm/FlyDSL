#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Build and run the Core C++ unit tests (tests/core) as the single flydsl-core-tests
# binary, in a build tree scripts/build.sh has configured. Its MLIR must provide
# llvm_gtest: an LLVM build tree, or an install made by scripts/build_llvm.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Build directory (default: build-fly/, overridable via FLY_BUILD_DIR)
# ---------------------------------------------------------------------------
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi
if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "No configured build tree at ${BUILD_DIR}; run scripts/build.sh first." >&2
  exit 1
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

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DFLY_UNITTEST=ON
cmake --build "${BUILD_DIR}" --target flydsl-core-tests -j"${PARALLEL_JOBS}"
"${BUILD_DIR}/tests/core/flydsl-core-tests"
