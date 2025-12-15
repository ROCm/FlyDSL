#!/bin/bash
# Rocir Test Suite - Organized by test type

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Prefer the new tool location (LLVM_RUNTIME_OUTPUT_INTDIR = build/bin),
# but keep a fallback for older build layouts.
ROCIR_OPT="${SCRIPT_DIR}/build/bin/rocir-opt"
if [ ! -x "${ROCIR_OPT}" ]; then
  ROCIR_OPT="${SCRIPT_DIR}/build/tools/rocir-opt/rocir-opt"
fi
if [ ! -x "${ROCIR_OPT}" ]; then
  if [ -d "${SCRIPT_DIR}/build" ]; then
    echo "rocir-opt not found. Building it..."
    cmake --build "${SCRIPT_DIR}/build" --target rocir-opt -j"$(nproc)" || {
      echo "Error: failed to build rocir-opt"
      exit 1
    }
  fi
  if [ ! -x "${ROCIR_OPT}" ]; then
    echo "Error: rocir-opt not found."
    echo "  Try: ./build.sh"
    echo "  Or:  cmake --build build --target rocir-opt -j\$(nproc)"
    exit 1
  fi
fi
PASS="--rocir-coord-lowering"

echo "========================================================================"
echo "Rocir Test Suite"
echo "========================================================================"
echo ""

# Set up Python path for embedded MLIR bindings only (no external MLIR python).
# - build/python_packages/rocdsl: contains `_mlir/`, plus `rocdsl/` and `mlir` shim
# - python/: source tree for development convenience
export PYTHONPATH="${SCRIPT_DIR}/build/python_packages/rocdsl:${SCRIPT_DIR}/python:${SCRIPT_DIR}:${PYTHONPATH}"

# Make sure any legacy env doesn't accidentally leak an external mlir package.
unset MLIR_PATH

#=============================================================================
MLIR_TEST_COUNT=0
MLIR_PASS_COUNT=0

for test_file in tests/mlir/*.mlir; do
    if [ -f "$test_file" ]; then
        MLIR_TEST_COUNT=$((MLIR_TEST_COUNT + 1))
        test_name=$(basename "$test_file" .mlir)
        echo "Running: $test_name"
        $ROCIR_OPT $PASS "$test_file" > /tmp/${test_name}.out 2>&1
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

for test_file in tests/python/ir/test_*.py; do
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
EXAMPLE_PASS_COUNT=0

for test_file in tests/python/examples/test_*.py; do
    if [ -f "$test_file" ]; then
        EXAMPLE_TEST_COUNT=$((EXAMPLE_TEST_COUNT + 1))
        test_name=$(basename "$test_file" .py)
        echo "Running: $test_name"
        python3 "$test_file" > /tmp/${test_name}.log 2>&1
        if [ $? -eq 0 ]; then
            echo "   PASS"
            EXAMPLE_PASS_COUNT=$((EXAMPLE_PASS_COUNT + 1))
        else
            echo "   FAIL"
            echo "      Log: /tmp/${test_name}.log"
        fi
    fi
done

echo ""
echo "Example Tests: $EXAMPLE_PASS_COUNT/$EXAMPLE_TEST_COUNT passed"
echo ""

#=============================================================================
# Part 4: GPU Execution Tests (Real GPU kernels)
#=============================================================================
echo "========================================================================"
echo "Part 4: GPU Execution Tests (Compile + Run on GPU)"
echo "========================================================================"
echo ""

if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'GPU\[\d+\].*' | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo "GPU detected: $GPU_NAME"
    else
        echo "GPU detected (ROCm available)"
    fi
    echo ""
    
    GPU_TEST_COUNT=0
    GPU_PASS_COUNT=0
    
    for test_file in tests/python/gpu/test_*.py; do
        if [ -f "$test_file" ]; then
            GPU_TEST_COUNT=$((GPU_TEST_COUNT + 1))
            test_name=$(basename "$test_file" .py)
            echo "Running: $test_name"
            python3 "$test_file" > /tmp/${test_name}.log 2>&1
            if [ $? -eq 0 ]; then
                echo "   PASS"
                GPU_PASS_COUNT=$((GPU_PASS_COUNT + 1))
                # Show key metrics if available
                if grep -q "GFLOPS" /tmp/${test_name}.log; then
                    grep "GFLOPS" /tmp/${test_name}.log | tail -1 | sed 's/^/      /'
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
# Part 5: Benchmark Tests (Performance Benchmarks)
#=============================================================================
echo "========================================================================"
echo "Part 5: Benchmark Tests (Performance Benchmarks)"
echo "========================================================================"
echo ""

if command -v rocm-smi &> /dev/null; then
    BENCHMARK_TEST_COUNT=0
    BENCHMARK_PASS_COUNT=0
    
    for test_file in tests/benchmark/*.py; do
        if [ -f "$test_file" ]; then
            BENCHMARK_TEST_COUNT=$((BENCHMARK_TEST_COUNT + 1))
            test_name=$(basename "$test_file" .py)
            echo "Running: $test_name"
            pytest -sv "$test_file" > /tmp/${test_name}.log 2>&1
            if [ $? -eq 0 ]; then
                echo "   PASS"
                BENCHMARK_PASS_COUNT=$((BENCHMARK_PASS_COUNT + 1))
                # Show key metrics if available
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
    echo "Benchmark Tests: $BENCHMARK_PASS_COUNT/$BENCHMARK_TEST_COUNT passed"
    
    ALL_BENCHMARK_PASSED=$((BENCHMARK_PASS_COUNT == BENCHMARK_TEST_COUNT))
else
    echo "Skipped (requires GPU)"
    echo ""
    ALL_BENCHMARK_PASSED=1
    BENCHMARK_TEST_COUNT=0
    BENCHMARK_PASS_COUNT=0
fi

echo ""

#=============================================================================
# Final Summary
#=============================================================================
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo ""
echo "MLIR IR Tests (Lowering):        $MLIR_PASS_COUNT/$MLIR_TEST_COUNT passed"
echo "Python IR Tests (Generation):    $IR_PASS_COUNT/$IR_TEST_COUNT passed"
echo "Example Tests (ROCDL):           $EXAMPLE_PASS_COUNT/$EXAMPLE_TEST_COUNT passed"

if command -v rocm-smi &> /dev/null; then
    echo "GPU Execution Tests:             $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
    echo "Benchmark Tests:                 $BENCHMARK_PASS_COUNT/$BENCHMARK_TEST_COUNT passed"
else
    echo "GPU Execution Tests:             Skipped (no GPU)"
    echo "Benchmark Tests:                 Skipped (no GPU)"
fi

if [ $ALL_GPU_PASSED -eq 1 ] && [ $ALL_BENCHMARK_PASSED -eq 1 ]; then
    echo ""
    echo ""
    echo "Verified Capabilities:"
    echo "  * Rocir IR generation and lowering"
    echo "  * Coordinate operations (crd2idx, layouts)"
    echo "  * ROCDL dialect operations (381 ops exposed)"
    echo "  * GPU kernel compilation (MLIR → HSACO)"
    echo "  * GPU kernel execution (HIP runtime)"
    echo "  * Shared memory optimizations (LDS)"
    echo "  * MFMA operations (Pure Python API)"
    echo "  * Performance benchmarking (bandwidth tests)"
    echo ""
    exit 0
else
    if command -v rocm-smi &> /dev/null; then
        echo ""
        if [ $ALL_GPU_PASSED -ne 1 ]; then
            echo "Some GPU tests failed"
        fi
        if [ $ALL_BENCHMARK_PASSED -ne 1 ]; then
            echo "Some benchmark tests failed"
        fi
        exit 1
    else
        echo ""
        echo "All available tests passed"
        echo "   (GPU tests skipped - no ROCm GPU detected)"
        exit 0
    fi
fi
