#!/bin/bash
# Kernel Test Suite - GEMM, LayerNorm, RMSNorm, Softmax, Examples, FileCheck
# Fail-fast: exits immediately on first test failure.
#
# Prerequisites: bash scripts/build.sh && pip install -e .
#   (or: export PYTHONPATH=build-fly/python_packages:$REPO_ROOT)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
MLIR_LIBS_DIR="${BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs"

# Ensure REPO_ROOT and build packages are always on PYTHONPATH.
export PYTHONPATH="${BUILD_DIR}/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"

# Ensure MLIR runtime shared libraries are discoverable.
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MLIR_LIBS_DIR}:"* ]]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "========================================================================"
echo "GEMM Test Suite"
echo "========================================================================"
echo ""

# By default, skip large_shape-marked tests (slow).
# Set RUN_TESTS_FULL=1 to run all parametrized cases (CI).
pytest_extra_args=()
if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
    pytest_extra_args+=(-m "not large_shape")
fi

python3 -m pytest tests/kernels/test_preshuffle_gemm.py "${pytest_extra_args[@]}" -v --no-header --tb=short

# ---------------------------------------------------------------------------
# Norm & Softmax Kernels (LayerNorm, RMSNorm, Softmax)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Norm & Softmax Kernels"
echo "========================================================================"
echo ""

python3 -m pytest tests/kernels/test_layernorm.py tests/kernels/test_rmsnorm.py tests/kernels/test_softmax.py -v --no-header --tb=short

# ---------------------------------------------------------------------------
# Python Examples (tests/python/examples/) via pytest
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Python Examples"
echo "========================================================================"
echo ""

python3 -m pytest tests/python/examples/*.py -v --no-header --tb=short

# ---------------------------------------------------------------------------
# MLIR FileCheck Tests (test/**/*.mlir)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "MLIR FileCheck Tests"
echo "========================================================================"
echo ""

FLY_OPT="${BUILD_DIR}/bin/fly-opt"
FILECHECK="${MLIR_PATH:+${MLIR_PATH}/bin/FileCheck}"
if [ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ]; then
    FILECHECK="$(which FileCheck 2>/dev/null || true)"
fi

if [ ! -x "${FLY_OPT}" ]; then
    echo "[SKIP] fly-opt not found at ${FLY_OPT}"
elif [ ! -x "${FILECHECK}" ]; then
    echo "[SKIP] FileCheck not found at ${FILECHECK}"
else
    for test_file in $(find "${REPO_ROOT}/test" -name "*.mlir" -type f 2>/dev/null | sort); do
        test_name="${test_file#${REPO_ROOT}/test/}"
        run_line=$(grep '^// RUN:' "$test_file" | head -1 | sed 's|^// RUN: *||')
        if [ -z "$run_line" ]; then
            continue
        fi
        cmd=$(echo "$run_line" | sed "s|%fly-opt|${FLY_OPT}|g; s|%FileCheck|${FILECHECK}|g; s|%s|${test_file}|g; s|FileCheck|${FILECHECK}|g")
        if eval "$cmd" > /tmp/filecheck_out.log 2>&1; then
            echo "  PASS  ${test_name}"
        else
            echo "  FAIL  ${test_name}"
            tail -5 /tmp/filecheck_out.log | sed 's/^/        /'
            exit 1
        fi
    done
fi

echo ""
echo "========================================================================"
echo "All tests passed."
echo "========================================================================"
