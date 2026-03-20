#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# FlyDSL Test Suite
# Fail-fast: exits immediately on first test failure.
#
# Local (default): skips large_shape tests for fast iteration.
# CI:              RUN_TESTS_FULL=1 bash scripts/run_tests.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
MLIR_LIBS_DIR="${BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs"

export PYTHONPATH="${BUILD_DIR}/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
export FLYDSL_RUN_QUANT=1
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MLIR_LIBS_DIR}:"* ]]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

pytest_args=(-v --no-header --tb=short)
if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
    pytest_args+=(-m "not large_shape")
fi

# ---------------------------------------------------------------------------
# 1. All pytest-based tests (kernels + pyir + examples)
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "Pytest: kernels + pyir + examples"
echo "========================================================================"

python3 -m pytest \
    tests/kernels/ \
    tests/pyir/ \
    tests/unit/ \
    tests/python/examples/ \
    "${pytest_args[@]}"

# ---------------------------------------------------------------------------
# 2. Standalone example scripts (not pytest)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Examples (examples/)"
echo "========================================================================"

for example in "${REPO_ROOT}"/examples/*.py; do
    [ -f "${example}" ] || continue
    name="$(basename "${example}")"
    output=$(python3 "${example}" 2>&1) || {
        echo "  FAIL  ${name}"; echo "$output" | tail -10 | sed 's/^/        /'; exit 1
    }
    if echo "$output" | grep -qE "Result correct: False|All passed: False"; then
        echo "  FAIL  ${name}"; echo "$output" | tail -10 | sed 's/^/        /'; exit 1
    fi
    echo "  PASS  ${name}"
done

# ---------------------------------------------------------------------------
# 3. MLIR FileCheck tests
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "MLIR FileCheck Tests"
echo "========================================================================"

FLY_OPT="${BUILD_DIR}/bin/fly-opt"
FILECHECK=""
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    _mlir_dir=$(grep '^MLIR_DIR:' "${BUILD_DIR}/CMakeCache.txt" | sed 's|^MLIR_DIR:[A-Z]*=||')
    [ -n "${_mlir_dir}" ] && FILECHECK="${_mlir_dir}/../../../bin/FileCheck"
fi
[ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ] && FILECHECK="$(which FileCheck 2>/dev/null || true)"

if [ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ]; then
    echo "  SKIP  FileCheck not found; skipping MLIR lit tests."
else

for f in $(find "${REPO_ROOT}/tests/mlir" -name "*.mlir" -type f 2>/dev/null | sort); do
    run_line=$(grep '^// RUN:' "$f" | head -1 | sed 's|^// RUN: *||')
    [ -z "$run_line" ] && continue
    cmd=$(echo "$run_line" | sed "s|%fly-opt|${FLY_OPT}|g; s|%FileCheck|${FILECHECK}|g; s|%s|${f}|g; s|FileCheck|${FILECHECK}|g")
    if eval "$cmd" > /tmp/filecheck_out.log 2>&1; then
        echo "  PASS  ${f#${REPO_ROOT}/tests/mlir/}"
    else
        echo "  FAIL  ${f#${REPO_ROOT}/tests/mlir/}"
        tail -5 /tmp/filecheck_out.log | sed 's/^/        /'
        exit 1
    fi
done

fi

echo ""
echo "========================================================================"
echo "All tests passed."
echo "========================================================================"
