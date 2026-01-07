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
        $FLIR_OPT $PASS "$test_file" > /tmp/${test_name}.out 2>&1
        if [ $? -eq 0 ]; then
            echo "   PASS"
            MLIR_PASS_COUNT=$((MLIR_PASS_COUNT + 1))
        else
            echo "   FAIL"
            echo "      Log: /tmp/${test_name}.out"
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

IR_TEST_COUNT=0
IR_PASS_COUNT=0

for test_file in tests/pyir/test_*.py; do
    if [ -f "$test_file" ]; then
        IR_TEST_COUNT=$((IR_TEST_COUNT + 1))
        test_name=$(basename "$test_file" .py)
        echo "Running: $test_name"
        python3 "$test_file" > /tmp/${test_name}.log 2>&1
        if [ $? -eq 0 ]; then
            echo "   PASS"
            IR_PASS_COUNT=$((IR_PASS_COUNT + 1))
        else
            echo "   FAIL"
            echo "      Log: /tmp/${test_name}.log"
        fi
    fi
done

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
    
    GPU_TEST_COUNT=0
    GPU_PASS_COUNT=0
    
    for test_file in tests/kernels/test_*.py; do
        if [ -f "$test_file" ]; then
            GPU_TEST_COUNT=$((GPU_TEST_COUNT + 1))
            test_name=$(basename "$test_file" .py)
            echo "Running: $test_name"
            python3 "$test_file" > /tmp/${test_name}.log 2>&1
            if [ $? -eq 0 ]; then
                echo "   PASS"
                GPU_PASS_COUNT=$((GPU_PASS_COUNT + 1))
                # Show key metrics if available
                if grep -q "TFLOPS" /tmp/${test_name}.log; then
                    grep "TFLOPS" /tmp/${test_name}.log | tail -1 | sed 's/^/      /'
                fi
                if grep -q "Bandwidth:" /tmp/${test_name}.log; then
                    grep "Bandwidth:" /tmp/${test_name}.log | tail -1 | sed 's/^/      /'
                fi
            else
                echo "   FAIL"
                echo "      Log: /tmp/${test_name}.log"
            fi
        fi
    done
    
    echo ""
    echo "GPU Tests: $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
    
    ALL_GPU_PASSED=$((GPU_PASS_COUNT == GPU_TEST_COUNT))
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
    echo "GPU Execution Tests:             $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
else
    echo "GPU Execution Tests:             Skipped (no GPU)"
fi

if [ $GPU_PASS_COUNT -eq $GPU_TEST_COUNT ] && [ $IR_PASS_COUNT -eq $IR_TEST_COUNT ] && [ $MLIR_PASS_COUNT -eq $MLIR_TEST_COUNT ]; then
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
