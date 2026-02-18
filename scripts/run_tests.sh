#!/bin/bash
# Flir Test Suite - Organized by test type

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
COMPARE_AITER_CK=0
# Locate the build directory (default: .flir/build; fallback: build/).
BUILD_DIR="${FLIR_BUILD_DIR:-${FLIR_BUILD_DIR:-${REPO_ROOT}/.flir/build}}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi

# Prefer the new tool location (LLVM_RUNTIME_OUTPUT_INTDIR = build/bin),
# but keep a fallback for older build layouts.
FLIR_OPT="${BUILD_DIR}/bin/flir-opt"
if [ ! -x "${FLIR_OPT}" ]; then
  FLIR_OPT="${BUILD_DIR}/tools/flir-opt/flir-opt"
fi
if [ ! -x "${FLIR_OPT}" ]; then
  if [ -d "${BUILD_DIR}" ]; then
    echo "flir-opt not found. Building it..."
    cmake --build "${BUILD_DIR}" --target flir-opt -j"$(nproc)" || {
      echo "Error: failed to build flir-opt"
      exit 1
    }
  fi
  # Re-detect after build: modern builds place it under ${BUILD_DIR}/bin.
  FLIR_OPT="${BUILD_DIR}/bin/flir-opt"
  if [ ! -x "${FLIR_OPT}" ]; then
    FLIR_OPT="${BUILD_DIR}/tools/flir-opt/flir-opt"
  fi
  if [ ! -x "${FLIR_OPT}" ]; then
    echo "Error: flir-opt not found."
    echo "  Try: ./flir/build.sh"
    echo "  Or:  cmake --build build --target flir-opt -j\$(nproc)"
    exit 1
  fi
fi
PASS="--flir-to-standard"

echo "========================================================================"
echo "Flir Test Suite"
echo "========================================================================"
echo ""

PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages/flydsl"
export PYTHONPATH="${REPO_ROOT}/flydsl/src:${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH}"
echo "Using in-tree Python sources + embedded build packages via PYTHONPATH."


#=============================================================================
MLIR_TEST_COUNT=0
MLIR_PASS_COUNT=0

for test_file in tests/mlir/*.mlir; do
    if [ -f "$test_file" ]; then
        MLIR_TEST_COUNT=$((MLIR_TEST_COUNT + 1))
        test_name=$(basename "$test_file" .mlir)
        echo "Running: $test_name"
        $FLIR_OPT $PASS "$test_file" > /tmp/${test_name}.log 2>&1
        if [ $? -eq 0 ]; then
            echo "   PASS"
            MLIR_PASS_COUNT=$((MLIR_PASS_COUNT + 1))
        else
            echo "   FAIL"
            echo "      Log: /tmp/${test_name}.log"
        fi
    fi
done

echo ""
echo "MLIR Tests: $MLIR_PASS_COUNT/$MLIR_TEST_COUNT passed"
echo ""
#=============================================================================
# Part 2: Python IR Tests (MLIR IR generation via Python)
#=============================================================================
echo "========================================================================"
echo "Part 2: Python IR Tests (MLIR generation, no GPU execution)"
echo "========================================================================"
echo ""

# Use pytest to run all parametrized test cases (not just __main__ defaults).
python3 -m pytest tests/pyir/ -v --tb=short 2>&1 | tee /tmp/pyir_tests.log
IR_EXIT=${PIPESTATUS[0]}
# Extract pass/fail counts from pytest summary line
IR_SUMMARY=$(grep -P '^\s*=+\s+.*\d+ (passed|failed|error)' /tmp/pyir_tests.log | tail -1)
if [ $IR_EXIT -eq 0 ]; then
    IR_PASS_COUNT=$(echo "$IR_SUMMARY" | grep -oP '\d+(?= passed)' || echo "0")
    IR_TEST_COUNT=$IR_PASS_COUNT
    echo "   All tests passed: $IR_SUMMARY"
else
    IR_PASS_COUNT=$(echo "$IR_SUMMARY" | grep -oP '\d+(?= passed)' || echo "0")
    IR_FAIL_COUNT=$(echo "$IR_SUMMARY" | grep -oP '\d+(?= failed)' || echo "0")
    IR_TEST_COUNT=$((IR_PASS_COUNT + IR_FAIL_COUNT))
    echo "   Some tests failed: $IR_SUMMARY"
    echo "   Log: /tmp/pyir_tests.log"
fi

echo ""
echo "IR Tests: $IR_PASS_COUNT/$IR_TEST_COUNT passed"
echo ""

#=============================================================================
# Part 3: GPU Execution Tests (Real GPU kernels)
#=============================================================================
echo "========================================================================"
echo "Part 3: GPU Execution Tests (Compile + Run on GPU)"
echo "========================================================================"
echo ""

if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'GPU\[\d+\].*' | grep 'gfx' | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo "GPU detected: $GPU_NAME"
    else
        echo "GPU detected (ROCm available)"
    fi
    echo ""
    
    # Use pytest to run all parametrized test cases per file.
    # We run each test file in a separate pytest process so that a GPU abort
    # (e.g. Fatal Python error in preshuffle_gemm) doesn't kill remaining tests.
    #
    # Speed optimization: large-shape tests are marked with @pytest.mark.large_shape
    # and skipped by default. Set RUN_TESTS_FULL=1 to run all parametrized cases
    # (including large shapes) — this is intended for CI.
    GPU_PASS_COUNT=0
    GPU_FAIL_COUNT=0
    GPU_SKIP_COUNT=0

    for test_file in tests/kernels/test_*.py; do
        [ -f "$test_file" ] || continue
        test_name=$(basename "$test_file" .py)

        # Build per-file pytest filters.
        # - By default, skip large_shape-marked tests (slow); set RUN_TESTS_FULL=1 to include them.
        # - Additional per-file -k filters can be added below for correctness issues (e.g. fp16).
        pytest_extra_args=()
        pytest_k_filter=""

        # Speed filters (local only; CI runs everything via RUN_TESTS_FULL=1).
        if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
            pytest_extra_args+=(-m "not large_shape")
        fi

        if [ -n "$pytest_k_filter" ]; then
            pytest_extra_args+=(-k "$pytest_k_filter")
        fi

        echo "Running: $test_name"
        if [ ${#pytest_extra_args[@]} -gt 0 ]; then
            python3 -m pytest "$test_file" "${pytest_extra_args[@]}" -v --no-header --tb=short 2>&1 | tee "/tmp/${test_name}.log"
        else
            python3 -m pytest "$test_file" -v --no-header --tb=short 2>&1 | tee "/tmp/${test_name}.log"
        fi
        file_exit=${PIPESTATUS[0]}

        # Parse the pytest summary line: "N passed, M failed in Xs" etc.
        file_summary=$(grep -P '^\s*=+\s+.*(passed|failed|error|skipped|no tests ran).*=+\s*$' "/tmp/${test_name}.log" | tail -1)
        file_passed=$(echo "$file_summary" | grep -oP '\d+(?= passed)' || echo "0")
        file_failed=$(echo "$file_summary" | grep -oP '\d+(?= failed)' || echo "0")
        file_skipped=$(echo "$file_summary" | grep -oP '\d+(?= skipped)' || echo "0")

        # One-liner per test file
        if [ -z "$file_summary" ]; then
            file_failed=1
            file_passed=0
            echo "  CRASH  $test_name (see /tmp/${test_name}.log)"
        elif [ "$file_exit" -eq 0 ]; then
            if [ "$file_passed" -eq 0 ] && [ "$file_skipped" -gt 0 ]; then
                echo "  SKIP   $test_name ($file_skipped skipped)"
            elif [ "$file_passed" -eq 0 ]; then
                echo "  SKIP   $test_name"
            else
                echo "  PASS   $test_name ($file_passed passed)"
            fi
        else
            echo "  FAIL   $test_name ($file_passed passed, $file_failed failed)"
            grep "^FAILED" "/tmp/${test_name}.log" | sed 's/^/           /'
        fi

        GPU_PASS_COUNT=$((GPU_PASS_COUNT + file_passed))
        GPU_FAIL_COUNT=$((GPU_FAIL_COUNT + file_failed))
        GPU_SKIP_COUNT=$((GPU_SKIP_COUNT + file_skipped))
    done

    GPU_TEST_COUNT=$((GPU_PASS_COUNT + GPU_FAIL_COUNT))
    echo ""
    echo "GPU Tests: $GPU_PASS_COUNT/$GPU_TEST_COUNT passed ($GPU_SKIP_COUNT skipped, $GPU_FAIL_COUNT failed)"
    
    ALL_GPU_PASSED=$((GPU_FAIL_COUNT == 0 ? 1 : 0))
else
    echo "No GPU detected (ROCm not found)"
    echo "   Install ROCm to run GPU execution tests"
    echo ""
    ALL_GPU_PASSED=0
    GPU_TEST_COUNT=0
    GPU_PASS_COUNT=0
fi


#=============================================================================
# Final Summary
#=============================================================================
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo ""
echo "MLIR IR Tests (Lowering):        $MLIR_PASS_COUNT/$MLIR_TEST_COUNT passed"
echo "Python IR Tests (Generation):    $IR_PASS_COUNT/$IR_TEST_COUNT passed"

if command -v rocm-smi >/dev/null 2>&1; then
    echo "GPU Execution Tests:             $GPU_PASS_COUNT/$GPU_TEST_COUNT passed ($GPU_SKIP_COUNT skipped, $GPU_FAIL_COUNT failed)"
else
    echo "GPU Execution Tests:             Skipped (no GPU)"
fi

if [ $GPU_PASS_COUNT -eq $GPU_TEST_COUNT ] && [ $IR_PASS_COUNT -eq $IR_TEST_COUNT ]; then
    echo ""
    echo ""
    echo "Verified Capabilities:"
    echo "  * Flir IR generation and lowering"
    echo "  * GPU kernel compilation and execution (MLIR → HSACO)"
    echo ""
    exit 0
else
    if command -v rocm-smi >/dev/null 2>&1; then
        echo ""
        if [ $GPU_PASS_COUNT -ne $GPU_TEST_COUNT ]; then
            echo "Some GPU tests failed"
        fi
        exit 1
    else
        echo ""
        echo "All available tests passed"
        echo "   (GPU tests skipped - no ROCm GPU detected)"
        exit 0
    fi
fi
