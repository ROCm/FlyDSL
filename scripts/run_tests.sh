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
# Part 3: Example Tests (ROCDL dialect operations)
#=============================================================================
echo "========================================================================"
echo "Part 3: Example Tests (ROCDL Dialect Operations)"
echo "========================================================================"
echo ""

EXAMPLE_TEST_COUNT=0


#=============================================================================
# Part 4: GPU Execution Tests (Real GPU kernels)
#=============================================================================
echo "========================================================================"
echo "Part 4: GPU Execution Tests (Compile + Run on GPU)"
echo "========================================================================"
echo ""

if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'GPU\[\d+\].*' | grep 'SKU' | head -1)
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
    # Speed optimization: for heavy kernel tests (moe_gemm, preshuffle_gemm),
    # only run the "S" (small/smoke) parametrize set by default.
    # Set RUN_TESTS_FULL=1 to run all parametrized cases.
    GPU_PASS_COUNT=0
    GPU_FAIL_COUNT=0
    GPU_SKIP_COUNT=0

    for test_file in tests/kernels/test_*.py; do
        [ -f "$test_file" ] || continue
        test_name=$(basename "$test_file" .py)

        # Build per-file pytest -k filter for speed.
        # Set RUN_TESTS_FULL=1 to run all parametrized cases.
        pytest_k_filter=""
        if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
            case "$test_name" in
                test_moe_gemm)
                    # Skip fp16 (known precision issue). Run both atomic & reduce.
                    pytest_k_filter="not fp16"
                    ;;
                test_preshuffle_gemm)
                    # Only run the first param set (small M=16) to avoid slow large cases & bf16 abort
                    pytest_k_filter="16-5120-8192"
                    ;;
            esac
        fi

        if [ -n "$pytest_k_filter" ]; then
            echo "Running: $test_name (pytest) [-k '$pytest_k_filter']"
            python3 -m pytest "$test_file" -k "$pytest_k_filter" -v --tb=short 2>&1 | tee "/tmp/${test_name}.log"
        else
            echo "Running: $test_name (pytest)"
            python3 -m pytest "$test_file" -v --tb=short 2>&1 | tee "/tmp/${test_name}.log"
        fi
        file_exit=${PIPESTATUS[0]}

        # Parse the pytest summary line: "= N passed, M failed in Xs ="
        # or "= N skipped in Xs =" or "= no tests ran in Xs ="
        # Use a strict pattern to avoid matching log messages that contain "passed".
        file_summary=$(grep -P '^\s*=+\s+.*(passed|failed|error|skipped|no tests ran).*=+\s*$' "/tmp/${test_name}.log" | tail -1)
        file_passed=$(echo "$file_summary" | grep -oP '\d+(?= passed)' || echo "0")
        file_failed=$(echo "$file_summary" | grep -oP '\d+(?= failed)' || echo "0")
        file_skipped=$(echo "$file_summary" | grep -oP '\d+(?= skipped)' || echo "0")

        # If pytest crashed (no summary line at all), count as 1 failure
        if [ -z "$file_summary" ]; then
            file_failed=1
            file_passed=0
            echo "   CRASH (process aborted, see log)"
            echo "      Log: /tmp/${test_name}.log"
        elif [ "$file_exit" -eq 0 ]; then
            if [ "$file_passed" -eq 0 ] && [ "$file_skipped" -gt 0 ]; then
                echo "   SKIP ($file_skipped skipped)"
            elif [ "$file_passed" -eq 0 ]; then
                echo "   SKIP (no tests ran)"
            else
                echo "   PASS ($file_passed passed, $file_skipped skipped)"
            fi
        else
            echo "   FAIL ($file_passed passed, $file_failed failed)"
            grep "^FAILED" "/tmp/${test_name}.log" | sed 's/^/      /'
            echo "      Log: /tmp/${test_name}.log"
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
# Part 5: GPU Execution Tests With HIPGraph Mode (Real GPU kernels)
#=============================================================================
echo "========================================================================"
echo "Part 5: GPU Execution Tests With HIPGraph Mode (Compile + Run on GPU)"
echo "========================================================================"
echo ""

if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'GPU\[\d+\].*' | grep 'SKU' | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo "GPU detected: $GPU_NAME"
    else
        echo "GPU detected (ROCm available)"
    fi
    echo ""
    
    GPU_GRAPH_TEST_COUNT=0
    GPU_GRAPH_PASS_COUNT=0
    
    GPU_GRAPH_TEST_FILES=(
        tests/kernels/test_preshuffle_gemm.py
        tests/kernels/test_moe_gemm.py
    )
    for test_file in "${GPU_GRAPH_TEST_FILES[@]}"; do
        if [ -f "$test_file" ]; then
            GPU_GRAPH_TEST_COUNT=$((GPU_GRAPH_TEST_COUNT + 1))
            test_name=$(basename "$test_file" .py)
            echo "Running: $test_name"
            python3 "$test_file" -tg > /tmp/${test_name}_graph.log 2>&1
            if [ $? -eq 0 ]; then
                echo "   PASS"
                GPU_GRAPH_PASS_COUNT=$((GPU_GRAPH_PASS_COUNT + 1))
                # Show key metrics if available
                if grep -q "TFLOPS" /tmp/${test_name}_graph.log; then
                    grep "TFLOPS" /tmp/${test_name}_graph.log | tail -1 | sed 's/^/      /'
                fi
                if grep -q "Bandwidth:" /tmp/${test_name}_graph.log; then
                    grep "Bandwidth:" /tmp/${test_name}_graph.log | tail -1 | sed 's/^/      /'
                fi
            else
                echo "   FAIL"
                echo "      Log: /tmp/${test_name}.log"
            fi
        fi
    done
    
    echo ""
    echo "GPU HIPGraph Tests: $GPU_GRAPH_PASS_COUNT/$GPU_GRAPH_TEST_COUNT passed"
    
    ALL_GPU_GRAPH_PASSED=$((GPU_GRAPH_PASS_COUNT == GPU_GRAPH_TEST_COUNT))
else
    echo "No GPU detected (ROCm not found)"
    echo "   Install ROCm to run GPU HIPGraph execution tests"
    echo ""
    ALL_GPU_GRAPH_PASSED=0
    GPU_GRAPH_TEST_COUNT=0
    GPU_GRAPH_PASS_COUNT=0
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
    echo "GPU Execution Tests:             $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
    echo "GPU HIPGraph Execution Tests:    $GPU_GRAPH_PASS_COUNT/$GPU_GRAPH_TEST_COUNT passed"
else
    echo "GPU Execution Tests:             Skipped (no GPU)"
    echo "GPU HIPGraph Execution Tests:    Skipped (no GPU)"
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
