#!/bin/bash
# GEMM Test Suite - runs preshuffle GEMM tests via pytest
#
# Prerequisites: bash scripts/build.sh && pip install -e .
#   (or: export PYTHONPATH=build-fly/python_packages:$REPO_ROOT)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
MLIR_LIBS_DIR="${BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs"

# If flydsl is not importable (no pip install -e .), fall back to PYTHONPATH.
if ! python3 -c "import flydsl" 2>/dev/null; then
  export PYTHONPATH="${BUILD_DIR}/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
fi

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
