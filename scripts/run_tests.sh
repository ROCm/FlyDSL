#!/bin/bash
# Kernel Test Suite - GEMM, MoE GEMM, LayerNorm, RMSNorm, Softmax, VecAdd, Quant, Examples, FileCheck
# Fail-fast: exits immediately on first test failure.
#
# Prerequisites: bash scripts/build.sh && pip install -e .
#   (or: export PYTHONPATH=build-fly/python_packages:$REPO_ROOT)
#
# By default, large_shape tests are skipped for fast local iteration.
# Set RUN_TESTS_FULL=1 for CI (runs all parametrized cases).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
MLIR_LIBS_DIR="${BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs"

export PYTHONPATH="${BUILD_DIR}/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MLIR_LIBS_DIR}:"* ]]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

pytest_extra_args=()
if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
    pytest_extra_args+=(-m "not large_shape")
fi

run_pytest() {
    local title="$1"; shift
    echo ""
    echo "========================================================================"
    echo "${title}"
    echo "========================================================================"
    echo ""
    python3 -m pytest "$@" -v --no-header --tb=short
}

# ---------------------------------------------------------------------------
# Pytest-based kernel tests (all use ${pytest_extra_args[@]} for large_shape)
# ---------------------------------------------------------------------------
run_pytest "GEMM Test Suite" \
    tests/kernels/test_preshuffle_gemm.py "${pytest_extra_args[@]}"

run_pytest "Blockscale GEMM" \
    tests/kernels/test_blockscale_preshuffle_gemm.py "${pytest_extra_args[@]}"

run_pytest "MoE GEMM Kernels" \
    tests/kernels/test_moe_gemm.py "${pytest_extra_args[@]}"

# test_moe_blockscale.py is a standalone benchmark (not pytest-compatible).
# Run via: python tests/kernels/test_moe_blockscale.py -m 32 -dim 7168 -idim 256 -e 256 -k 8

run_pytest "Norm & Softmax Kernels" \
    tests/kernels/test_layernorm.py tests/kernels/test_rmsnorm.py tests/kernels/test_softmax.py \
    "${pytest_extra_args[@]}"

run_pytest "Vector Addition" \
    tests/kernels/test_vec_add.py "${pytest_extra_args[@]}"

run_pytest "MoE Reduce Kernel" \
    tests/kernels/test_moe_reduce.py "${pytest_extra_args[@]}"

FLYDSL_RUN_QUANT=1 \
run_pytest "Per-Token Quantization" \
    tests/kernels/test_quant.py "${pytest_extra_args[@]}"

run_pytest "Python Examples" \
    tests/python/examples/*.py "${pytest_extra_args[@]}"

run_pytest "Layout Algebra & PyIR Tests" \
    tests/pyir/test_layout_algebra.py tests/pyir/test_static_vs_dynamic.py tests/pyir/test_rocir_print.py

# ---------------------------------------------------------------------------
# Examples (examples/*.py) — standalone scripts, not pytest
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Examples (examples/)"
echo "========================================================================"
echo ""

for example in "${REPO_ROOT}"/examples/*.py; do
    [ -f "${example}" ] || continue
    example_name="$(basename "${example}")"
    output=$(python3 "${example}" 2>&1) || {
        echo "  FAIL  ${example_name}"
        echo "$output" | tail -10 | sed 's/^/        /'
        exit 1
    }
    if echo "$output" | grep -qE "Result correct: False|All passed: False"; then
        echo "  FAIL  ${example_name}"
        echo "$output" | tail -10 | sed 's/^/        /'
        exit 1
    fi
    echo "  PASS  ${example_name}"
done

# ---------------------------------------------------------------------------
# MLIR FileCheck Tests (tests/mlir/**/*.mlir)
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
    for test_file in $(find "${REPO_ROOT}/tests/mlir" -name "*.mlir" -type f 2>/dev/null | sort); do
        test_name="${test_file#${REPO_ROOT}/tests/mlir/}"
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
