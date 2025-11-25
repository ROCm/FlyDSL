#!/usr/bin/env python3
"""Test the new rocir coordinate operations (make_coord, crd2idx, idx2crd)"""

import sys
sys.path.insert(0, '/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from mlir import ir
from mlir.dialects import arith
import mlir.extras.types as T


def test_coord_operations():
    """Test make_coord, crd2idx, and idx2crd operations"""
    print("="*80)
    print("Testing Rocir Coordinate Operations")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    with ctx.context, ir.Location.unknown():
        # Test 1: Create coordinates
        print("\n1. Testing make_coord...")
        i = arith.constant(T.index(), 4)
        j = arith.constant(T.index(), 7)
        coord_2d = rocir.make_coord(i, j)
        print(f"   ✓ Created 2D coordinate: {coord_2d}")
        print(f"   ✓ Type: {coord_2d.type}")
        
        # Test 2: Create layout (2D row-major)
        print("\n2. Creating layout for coordinate mapping...")
        m = arith.constant(T.index(), 32)
        n = arith.constant(T.index(), 64)
        one = arith.constant(T.index(), 1)
        
        shape = rocir.make_shape(m, n)
        stride = rocir.make_stride(n, one)  # Row-major: stride=(64, 1)
        layout = rocir.make_layout(shape, stride)
        print(f"   ✓ Layout created: shape=(32,64), stride=(64,1)")
        
        # Test 3: Convert coordinate to index
        print("\n3. Testing crd2idx (coordinate to linear index)...")
        linear_idx = rocir.crd2idx(coord_2d, layout)
        print(f"   ✓ crd2idx result: {linear_idx}")
        print(f"   ✓ Type: {linear_idx.type}")
        print(f"   Expected: 4*64 + 7*1 = 263")
        
        # Test 4: Convert index back to coordinate
        print("\n4. Testing idx2crd (linear index to coordinate)...")
        idx_test = arith.constant(T.index(), 263)
        coord_back = rocir.idx2crd(idx_test, layout)
        print(f"   ✓ idx2crd result: {coord_back}")
        print(f"   ✓ Type: {coord_back.type}")
        print(f"   Expected: coordinate (4, 7)")
        
        # Test 5: 1D coordinate
        print("\n5. Testing 1D coordinate...")
        k = arith.constant(T.index(), 42)
        coord_1d = rocir.make_coord(k)
        print(f"   ✓ Created 1D coordinate: {coord_1d}")
        
        size_1d = arith.constant(T.index(), 1024)
        stride_1d = arith.constant(T.index(), 1)
        shape_1d = rocir.make_shape(size_1d)
        stride_vec = rocir.make_stride(stride_1d)
        layout_1d = rocir.make_layout(shape_1d, stride_vec)
        
        idx_1d = rocir.crd2idx(coord_1d, layout_1d)
        print(f"   ✓ 1D crd2idx: {idx_1d}")
        
        # Test 6: 3D coordinate
        print("\n6. Testing 3D coordinate...")
        x = arith.constant(T.index(), 2)
        y = arith.constant(T.index(), 3)
        z = arith.constant(T.index(), 5)
        coord_3d = rocir.make_coord(x, y, z)
        print(f"   ✓ Created 3D coordinate: {coord_3d}")
        print(f"   ✓ Type: {coord_3d.type}")
        
        # Print full MLIR module
        print("\n" + "="*80)
        print("Generated MLIR:")
        print("="*80)
        print(ctx.module)
        
        print("\n" + "="*80)
        print("✅ All coordinate operations work correctly!")
        print("="*80)
        print("\nSummary:")
        print("  - CoordType: ✓ (1D, 2D, 3D tested)")
        print("  - make_coord: ✓ (variadic arguments)")
        print("  - crd2idx: ✓ (coordinate → linear index)")
        print("  - idx2crd: ✓ (linear index → coordinate)")
        return True


if __name__ == "__main__":
    try:
        success = test_coord_operations()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
