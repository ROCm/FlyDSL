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
echo "GEMM Tests: ${passed} passed, ${failed} failed, ${skipped} skipped"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Python Examples (tests/python/examples/) via pytest
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Python Examples"
echo "========================================================================"
echo ""

python3 -m pytest tests/python/examples/*.py -v --no-header --tb=short 2>&1 | tee /tmp/test_examples.log
example_exit=${PIPESTATUS[0]}
if [ $example_exit -ne 0 ]; then
    exit_code=1
fi

example_summary=$(grep -P '^\s*=+\s+.*(passed|failed|error|skipped|no tests ran).*=+\s*$' /tmp/test_examples.log | tail -1)
example_passed=$(echo "$example_summary" | grep -oP '\d+(?= passed)' || echo "0")
example_failed=$(echo "$example_summary" | grep -oP '\d+(?= failed)' || echo "0")

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
mlir_pass=0
mlir_fail=0

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
            mlir_pass=$((mlir_pass + 1))
        else
            echo "  FAIL  ${test_name}"
            tail -5 /tmp/filecheck_out.log | sed 's/^/        /'
            mlir_fail=$((mlir_fail + 1))
            exit_code=1
        fi
    done
fi

echo ""
echo "========================================================================"
echo "Summary"
echo "  GEMM:      ${passed} passed, ${failed} failed, ${skipped} skipped"
echo "  Examples:  ${example_passed} passed, ${example_failed} failed"
echo "  FileCheck: ${mlir_pass} passed, ${mlir_fail} failed"
echo "========================================================================"

exit $exit_code
