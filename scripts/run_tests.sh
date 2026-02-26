#!/bin/bash
# GEMM Test Suite - runs preshuffle GEMM tests via pytest

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${FLIR_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi

PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages/flydsl"
export PYTHONPATH="${REPO_ROOT}/flydsl/src:${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH}"

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

python3 -m pytest tests/kernels/test_preshuffle_gemm.py "${pytest_extra_args[@]}" -v --no-header --tb=short 2>&1 | tee /tmp/test_preshuffle_gemm.log
exit_code=${PIPESTATUS[0]}

summary=$(grep -P '^\s*=+\s+.*(passed|failed|error|skipped|no tests ran).*=+\s*$' /tmp/test_preshuffle_gemm.log | tail -1)
passed=$(echo "$summary" | grep -oP '\d+(?= passed)' || echo "0")
failed=$(echo "$summary" | grep -oP '\d+(?= failed)' || echo "0")
skipped=$(echo "$summary" | grep -oP '\d+(?= skipped)' || echo "0")

echo ""
echo "========================================================================"
echo "Summary: ${passed} passed, ${failed} failed, ${skipped} skipped"
echo "========================================================================"

exit $exit_code
