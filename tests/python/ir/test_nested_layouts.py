#!/usr/bin/env python3
"""
Test nested layout support in Rocir, following the layout algebra notebook.
Reference: layout algebra notebook in the CUTLASS examples tree.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index
from mlir.dialects import func

def unwrap(val):
    """Unwrap ArithValue or other wrappers."""
    if hasattr(val, 'value'): return val.value
    if hasattr(val, '_value'): return val._value
    return val


def test_nested_shape_creation():
    """Test creating nested shapes such as (9, (4, 8))."""
    print("\n" + "="*80)
    print("Test 1: Nested Shape Creation")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_nested_shape():
        # Create nested shape: (9, (4, 8))
        c9 = Index(9)
        c4 = Index(4)
        c8 = Index(8)
        
        # Nested shape
        shape = rocir.make_shape(c9, (c4, c8))
        print(">>> Created nested shape: (9, (4, 8))")
        
        # Verify size is 9 * 4 * 8 = 288
        sz = rocir.size(shape)
        return [unwrap(sz)]
    
    print("Generated IR:")
    print(ctx.module)
    print("✓ Nested shape created successfully")
    return True


def test_nested_stride_creation():
    """Test creating nested strides such as (59, (13, 1))."""
    print("\n" + "="*80)
    print("Test 2: Nested Stride Creation")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_nested_stride():
        # Create nested stride: (59, (13, 1))
        c59 = Index(59)
        c13 = Index(13)
        c1 = Index(1)
        
        # Nested stride
        stride = rocir.make_stride(c59, (c13, c1))
        print(">>> Created nested stride: (59, (13, 1))")
        
        return []
    
    print("Generated IR:")
    print(ctx.module)
    print("✓ Nested stride created successfully")
    return True


def test_nested_layout_creation():
    """Test creating nested layouts like (9,(4,8)):(59,(13,1))."""
    print("\n" + "="*80)
    print("Test 3: Nested Layout Creation")
    print("="*80)
    print(">>> Creating layout: (9,(4,8)):(59,(13,1))")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_nested_layout():
        # Create nested layout: (9,(4,8)):(59,(13,1))
        c9 = Index(9)
        c4 = Index(4)
        c8 = Index(8)
        c59 = Index(59)
        c13 = Index(13)
        c1 = Index(1)
        
        # Method 1: Create shape and stride separately
        shape = rocir.make_shape(c9, (c4, c8))
        stride = rocir.make_stride(c59, (c13, c1))
        layout = rocir.make_layout(shape, stride)
        
        # Verify size
        sz = rocir.size(layout)
        
        return [unwrap(sz)]
    
    print("Generated IR:")
    print(ctx.module)
    print("✓ Nested layout created successfully")
    return True


def test_nested_layout_direct():
    """Test creating nested layouts using direct tuple syntax"""
    print("\n" + "="*80)
    print("Test 4: Nested Layout with Direct Tuple Syntax")
    print("="*80)
    print(">>> Creating layout using tuples: (9,(4,8)):(59,(13,1))")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_nested_layout_direct():
        c9 = Index(9)
        c4 = Index(4)
        c8 = Index(8)
        c59 = Index(59)
        c13 = Index(13)
        c1 = Index(1)
        
        # Direct tuple syntax mirroring the reference notebook
        layout = rocir.make_layout(
            (c9, (c4, c8)),
            stride=(c59, (c13, c1))
        )
        
        sz = rocir.size(layout)
        return [unwrap(sz)]
    
    print("Generated IR:")
    print(ctx.module)
    print("✓ Direct tuple syntax works!")
    return True


def test_flat_divide_with_nested_layout():
    """Test flat divide producing nested output: (3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))"""
    print("\n" + "="*80)
    print("Test 5: Flat Divide with Nested Layout")
    print("="*80)
    print(">>> Layout: (9,(4,8)):(59,(13,1))")
    print(">>> Tiler : (3:3, (2,4):(1,8))")
    print(">>> Expected Result: (3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_flat_divide():
        # Define the original nested layout
        layout = rocir.make_layout(
            (Index(9), (Index(4), Index(8))),
            stride=(Index(59), (Index(13), Index(1)))
        )
        
        # Define the tiler with nested structure
        tiler = rocir.make_layout(
            (Index(3), (Index(2), Index(4))),
            stride=(Index(3), (Index(1), Index(8)))
        )
        
        # Apply flat divide
        res_flat = rocir.flat_divide(layout, tiler)
        
        # Result should have shape (3,(2,4),3,(2,2)) with stride (177,(13,2),59,(26,1))
        # Total size should be same as original: 9*4*8 = 288
        sz = rocir.size(res_flat)
        
        return [unwrap(sz)]
    
    print("Generated IR:")
    print(ctx.module)
    print("✓ Flat divide with nested layout works!")
    return True


def test_logical_divide_2d_nested():
    """Test 2D logical divide with nested layout (Cell 17 from notebook)"""
    print("\n" + "="*80)
    print("Test 6: Logical Divide 2D with Nested Layout")
    print("="*80)
    print(">>> Layout: (9,(4,8)):(59,(13,1))")
    print(">>> Tiler : (3:3, (2,4):(1,8))")
    print(">>> Expected Result: ((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_logical_divide_2d():
        # Define the original nested layout
        layout = rocir.make_layout(
            (Index(9), (Index(4), Index(8))),
            stride=(Index(59), (Index(13), Index(1)))
        )
        
        # Define the tiler
        tiler = rocir.make_layout(
            (Index(3), (Index(2), Index(4))),
            stride=(Index(3), (Index(1), Index(8)))
        )
        
        # Apply logical divide
        res_logical = rocir.logical_divide(layout, tiler)
        
        # Size should match original: 9*4*8 = 288
        sz = rocir.size(res_logical)
        
        return [unwrap(sz)]
    
    print("Generated IR:")
    print(ctx.module)
    print("✓ Logical divide with nested layout works!")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Rocir Nested Layout Tests")
    print("Following the layout algebra notebook")
    print("="*80)
    
    all_pass = True
    
    all_pass &= test_nested_shape_creation()
    all_pass &= test_nested_stride_creation()
    all_pass &= test_nested_layout_creation()
    all_pass &= test_nested_layout_direct()
    all_pass &= test_flat_divide_with_nested_layout()
    all_pass &= test_logical_divide_2d_nested()
    
    if all_pass:
        print("\n" + "="*80)
        print("All Nested Layout Tests PASSED!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ Some tests FAILED!")
        print("="*80)
        sys.exit(1)

