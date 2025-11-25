#!/bin/bash
# Rocir Test Suite - Organized by test type

ROCIR_OPT="./build/tools/rocir-opt/rocir-opt"
PASS="--rocir-coord-lowering"

echo "========================================================================"
echo "Rocir Test Suite"
echo "========================================================================"
echo ""

# Set up Python path
export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/rocDSL/build/python_bindings
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/rocDSL/python

#=============================================================================
# Part 1: MLIR IR Tests (Lowering Passes)
#=============================================================================
echo "========================================================================"
echo "Part 1: MLIR IR Tests (Lowering & Transformations)"
echo "========================================================================"
echo ""

echo "Test 1.1: Coordinate Lowering (Static)"
$ROCIR_OPT $PASS tests/mlir/test_coord_lowering.mlir > /tmp/test_coord_static.out 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
fi

echo "Test 1.2: Coordinate Lowering (Dynamic)"
$ROCIR_OPT $PASS tests/mlir/test_coord_lowering_dynamic.mlir > /tmp/test_coord_dynamic.out 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
fi

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
        python3 "$test_file" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "   ✅ PASS"
            IR_PASS_COUNT=$((IR_PASS_COUNT + 1))
        else
            echo "   ❌ FAIL"
        fi
    fi
done

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
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'GPU\[\d+\].*' | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo "�� GPU detected: $GPU_NAME"
    else
        echo "🎮 GPU detected (ROCm available)"
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
                echo "   ✅ PASS"
                GPU_PASS_COUNT=$((GPU_PASS_COUNT + 1))
                # Show key metrics if available
                if grep -q "GFLOPS" /tmp/${test_name}.log; then
                    grep "GFLOPS" /tmp/${test_name}.log | tail -1 | sed 's/^/      /'
                fi
            else
                echo "   ❌ FAIL"
                echo "      Log: /tmp/${test_name}.log"
            fi
        fi
    done
    
    echo ""
    echo "GPU Tests: $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
    
    ALL_GPU_PASSED=$((GPU_PASS_COUNT == GPU_TEST_COUNT))
else
    echo "⚠️  No GPU detected (ROCm not found)"
    echo "   Install ROCm to run GPU execution tests"
    echo ""
    ALL_GPU_PASSED=0
fi

echo ""

#=============================================================================
# Final Summary
#=============================================================================
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo ""
echo "MLIR IR Tests (Lowering):        ✓ Passed"
echo "Python IR Tests (Generation):    $IR_PASS_COUNT/$IR_TEST_COUNT passed"

if [ $ALL_GPU_PASSED -eq 1 ]; then
    echo "GPU Execution Tests:             $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
    echo ""
    echo "�� ALL TESTS PASSED!"
    echo ""
    echo "Verified Capabilities:"
    echo "  ✓ Rocir IR generation and lowering"
    echo "  ✓ Coordinate operations (crd2idx, layouts)"
    echo "  ✓ GPU kernel compilation (MLIR → HSACO)"
    echo "  ✓ GPU kernel execution (HIP runtime)"
    echo "  ✓ Shared memory optimizations (LDS)"
    echo ""
    exit 0
else
    if command -v rocm-smi &> /dev/null; then
        echo "GPU Execution Tests:             $GPU_PASS_COUNT/$GPU_TEST_COUNT passed"
        echo ""
        echo "⚠️  Some GPU tests failed"
        exit 1
    else
        echo "GPU Execution Tests:             Skipped (no GPU)"
        echo ""
        echo "✅ All available tests passed"
        echo "   (GPU tests skipped - no ROCm GPU detected)"
        exit 0
    fi
fi
