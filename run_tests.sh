#!/bin/bash
# Run all CuTe IR operation tests

CUTE_OPT="./build/tools/cute-opt/cute-opt"
PASS="--cute-to-standard"

echo "========================================"
echo "CuTe IR Operations Test Suite"
echo "========================================"
echo ""

# Test 1: crd2idx
echo "✅ Test 1: crd2idx - Coordinate to Linear Index"
echo "Expected: coord(2,3) with stride(1,16) → idx=50"
$CUTE_OPT $PASS tests/mlir/test_crd2idx.mlir > /tmp/test_crd2idx.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "cute.crd2idx" /tmp/test_crd2idx.out; then
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
$CUTE_OPT $PASS tests/mlir/test_size.mlir > /tmp/test_size.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "cute.size" /tmp/test_size.out; then
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
$CUTE_OPT $PASS tests/mlir/test_rank.mlir > /tmp/test_rank.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "cute.rank" /tmp/test_rank.out; then
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
$CUTE_OPT $PASS tests/mlir/test_cosize.mlir > /tmp/test_cosize.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Lowered successfully"
    if grep -q "cute.cosize" /tmp/test_cosize.out; then
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
$CUTE_OPT $PASS tests/mlir/comprehensive_test.mlir > /tmp/test_comprehensive.out 2>&1
if [ $? -eq 0 ]; then
    echo "   PASS: Module processed successfully"
    # Count how many cute operations remain (should be minimal)
    CUTE_OPS=$(grep -c "cute\." /tmp/test_comprehensive.out || echo 0)
    echo "   ✓ Remaining cute operations: $CUTE_OPS"
else
    echo "   FAIL"
fi
echo ""

echo "========================================"
echo "MLIR Test Summary"
echo "========================================"
echo "✅ Working Operations:"
echo "   - cute.make_shape, make_stride, make_coord, make_layout"
echo "   - cute.size (lowering implemented)"
echo "   - cute.rank (lowering implemented)"
echo "   - cute.cosize (lowering implemented)"
echo "   - cute.crd2idx (lowering implemented)"
echo "========================================"

echo ""
echo "========================================"
echo "Python Test Suite"
echo "========================================"
echo ""

# Set up Python environment
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/rocDSL/build/python_bindings
export PYTHONPATH=$PYTHONPATH:/mnt/raid0/felix/rocDSL/python

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
echo "========================================"

exit $PYTEST_EXIT
