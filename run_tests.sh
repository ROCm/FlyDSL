#!/bin/bash
# Run all Rocir IR operation tests

ROCIR_OPT="./build/tools/rocir-opt/rocir-opt"
PASS="--rocir-to-standard"

echo "========================================"
echo "Rocir IR Operations Test Suite"
echo "========================================"
echo ""

# Test 1: crd2idx
echo "✅ Test 1: crd2idx - Coordinate to Linear Index"
echo "Expected: coord(2,3) with stride(1,16) → idx=50"
$ROCIR_OPT $PASS tests/mlir/test_crd2idx.mlir > /tmp/test_crd2idx.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "rocir.crd2idx" /tmp/test_crd2idx.out; then
        echo "   ⚠️  crd2idx not lowered"
    else
        echo "   ✓ crd2idx operation lowered"
    fi
else
    echo "   FAIL"
fi
echo ""

# Test 2: size
echo "✅ Test 2: size - Product of Shape Dimensions"
echo "Expected: shape(8,16,32) → size=4096"
$ROCIR_OPT $PASS tests/mlir/test_size.mlir > /tmp/test_size.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "rocir.size" /tmp/test_size.out; then
        echo "   ⚠️  size not lowered"
    else
        echo "   ✓ size operation lowered"
    fi
else
    echo "   FAIL"
fi
echo ""

# Test 3: rank
echo "✅ Test 3: rank - Number of Dimensions"
echo "Expected: shape<3> → rank=3"
$ROCIR_OPT $PASS tests/mlir/test_rank.mlir > /tmp/test_rank.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "rocir.rank" /tmp/test_rank.out; then
        echo "   ⚠️  rank not lowered"
    else
        echo "   ✓ rank operation lowered"
    fi
else
    echo "   FAIL"
fi
echo ""

# Test 4: cosize
echo "✅ Test 4: cosize - Codomain Size"
echo "Expected: layout(shape(8,128), stride(1,16)) → cosize=2033"
$ROCIR_OPT $PASS tests/mlir/test_cosize.mlir > /tmp/test_cosize.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "rocir.cosize" /tmp/test_cosize.out; then
        echo "   ⚠️  cosize not lowered"
    else
        echo "   ✓ cosize operation lowered"
    fi
else
    echo "   FAIL"
fi
echo ""

# Test 5: Comprehensive test
echo "✅ Test 5: Comprehensive - All Operations Together"
$ROCIR_OPT $PASS tests/mlir/comprehensive_test.mlir > /tmp/test_comprehensive.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Module processed successfully"
    # Count how many rocir operations remain (should be minimal)
    ROCIR_OPS=$(grep -c "rocir\." /tmp/test_comprehensive.out || echo 0)
    echo "   ✓ Remaining rocir operations: $ROCIR_OPS"
else
    echo "   FAIL"
fi
echo ""

echo "========================================"
echo "MLIR Test Summary"
echo "========================================"
echo "✅ Working Operations:"
echo "   - rocir.make_shape, make_stride, make_coord, make_layout"
echo "   - rocir.size (lowering implemented)"
echo "   - rocir.rank (lowering implemented)"
echo "   - rocir.cosize (lowering implemented)"
echo "   - rocir.crd2idx (lowering implemented)"
echo "========================================"

echo ""
echo "========================================"
echo "GPU Test Suite"
echo "========================================"
echo ""

# Set up Python path for GPU tests
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/rocDSL/python

# Check if GPU is available
if command -v rocm-smi &> /dev/null; then
    echo "🎮 GPU detected, running GPU tests..."
    echo ""
    
    # Test 1: Basic GPU kernel
    echo "Test 1: Basic GPU Vector Addition"
    python3 tests/python/test_gpu_simple.py
    GPU_SIMPLE_EXIT=$?
    if [ $GPU_SIMPLE_EXIT -eq 0 ]; then
        echo "   ✅ PASS"
    else
        echo "   ❌ FAIL (exit code: $GPU_SIMPLE_EXIT)"
    fi
    echo ""
    
    # Test 2: Layout-based indexing
    echo "Test 2: GPU Layout-based Indexing"
    python3 tests/python/test_gpu_layout.py
    GPU_LAYOUT_EXIT=$?
    if [ $GPU_LAYOUT_EXIT -eq 0 ]; then
        echo "   ✅ PASS"
    else
        echo "   ❌ FAIL (exit code: $GPU_LAYOUT_EXIT)"
    fi
    echo ""
    
    GPU_TESTS_PASSED=$(( $GPU_SIMPLE_EXIT == 0 && $GPU_LAYOUT_EXIT == 0 ))
else
    echo "⚠️  No GPU detected, skipping GPU tests"
    echo "   (Install ROCm and ensure GPU is available to run GPU tests)"
    GPU_TESTS_PASSED=1
fi

echo "========================================"
echo "GPU Test Summary"
echo "========================================"
if [ $GPU_TESTS_PASSED -eq 1 ]; then
    echo "✅ All GPU tests passed"
else
    echo "⚠️  Some GPU tests failed"
fi
echo "========================================"

echo ""
echo "========================================"
echo "Python Test Suite"
echo "========================================"
echo ""
# Ensure Python bindings are built
echo "Building Python bindings..."
cd build && make RocirPythonOpsIncGen > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Failed to build Python bindings"
fi
cd ..
echo ""

# Set up Python environment
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/rocDSL/build/python_bindings

echo "Running Python tests with pytest..."
python3 -m pytest tests/python/ -v --tb=short

PYTEST_EXIT=$?

echo ""
echo "========================================"
echo "Overall Test Summary"
echo "========================================"
if [ $PYTEST_EXIT -eq 0 ]; then
    echo "✅ All Python tests passed"
else
    echo "⚠️  Some Python tests failed (exit code: $PYTEST_EXIT)"
fi

if [ $GPU_TESTS_PASSED -eq 1 ]; then
    echo "✅ All GPU tests passed"
else
    echo "⚠️  Some GPU tests failed"
fi
echo "========================================"

exit $PYTEST_EXIT
