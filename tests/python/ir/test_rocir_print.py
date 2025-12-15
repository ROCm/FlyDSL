#!/usr/bin/env python3
"""
Test rocir.print and rocir.printf functionality.
Following the layout print notebook example.

This test demonstrates the difference between static (compile-time) and
dynamic (runtime) printing.
"""

import sys
import os


from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index
from mlir.dialects import func
import rocdsl


def test_print_basic():
    """Test that rocir.print is available and works as Python's print."""
    print("\n" + "="*80)
    print("Test 1: Basic Print")
    print("="*80)
    
    # rocir.print should just be Python's built-in print
    assert rocdsl.rocir.print is print
    
    # Test basic printing (just make sure it doesn't crash)
    rocdsl.rocir.print(">>> Testing static print")
    rocdsl.rocir.print(">>> Value:", 42)
    print("✓ rocir.print works as Python's built-in print")
    return True


def test_printf_ir_generation():
    """Test that rocir.printf generates the correct MLIR operations."""
    print("\n" + "="*80)
    print("Test 2: Printf IR Generation")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_printf_func():
        x = Index(42)
        # Print a dynamic index value
        rocdsl.rocir.printf(">?? {}", x)
        return []
    
    # Verify the IR contains gpu.printf operation
    ir_str = str(ctx.module)
    assert "gpu.printf" in ir_str
    print("Generated IR:")
    print(ir_str)
    print("✓ rocir.printf generates gpu.printf operation")
    return True


def test_printf_with_multiple_args():
    """Test printf with multiple arguments."""
    print("\n" + "="*80)
    print("Test 3: Printf with Multiple Arguments")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_multi_printf():
        a = Index(10)
        b = Index(20)
        # Print multiple values
        rocdsl.rocir.printf("a: {}, b: {}", a, b)
        return []
    
    ir_str = str(ctx.module)
    assert "gpu.printf" in ir_str
    # Check that format string is correct
    assert "a: {}, b: {}" in ir_str or "a: %lld, b: %lld" in ir_str
    print("Generated IR:")
    print(ir_str)
    print("✓ Multiple argument printf works")
    return True


def test_print_vs_printf_concept():
    """Conceptual test showing the difference between print and printf.
    
    This demonstrates the key difference highlighted in the reference notebook:
    - rocir.print: Shows static/compile-time values
    - rocir.printf: Shows dynamic/runtime values
    """
    print("\n" + "="*80)
    print("Test 4: Print vs Printf Concept")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def print_example():
        # Dynamic value
        a = Index(8)
        
        # Static value
        b = 2
        
        # Python's print: shows compile-time info
        rocdsl.rocir.print(">>>", b)  # Will print: >>> 2
        rocdsl.rocir.print(">>> a is dynamic:", a)  # Will print object representation
            
        # rocir.printf: shows runtime values
        rocdsl.rocir.printf(">?? {}", a)  # Will print actual value at runtime
        
        c2 = Index(2)
        rocdsl.rocir.printf(">?? {}", c2)  # Will print 2 at runtime
            
        return []
    
    # Verify both types of output exist
    ir_str = str(ctx.module)
    assert "gpu.printf" in ir_str
    print("Generated IR:")
    print(ir_str)
    print("✓ Print vs printf distinction demonstrated")
    return True


def test_printf_with_layout_types():
    """Test printf with rocir layout types."""
    print("\n" + "="*80)
    print("Test 5: Printf with Layout Types")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def layout_printf_example():
        dim0 = Index(9)
        dim1 = Index(4)
        dim2 = Index(8)
        
        # Create a nested shape
        shape = rocdsl.rocir.make_shape(dim0, (dim1, dim2))
            
            # Print the shape dimensions
        rocdsl.rocir.printf("Shape dims: {} x ({} x {})", dim0, dim1, dim2)
            
        return []
    
    ir_str = str(ctx.module)
    assert "rocir.make_shape" in ir_str
    assert "gpu.printf" in ir_str
    print("Generated IR:")
    print(ir_str)
    print("✓ Printf with layout types works")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Rocir Print/Printf Tests")
    print("Following the layout print notebook")
    print("="*80)
    
    all_pass = True
    all_pass &= test_print_basic()
    all_pass &= test_printf_ir_generation()
    all_pass &= test_printf_with_multiple_args()
    all_pass &= test_print_vs_printf_concept()
    all_pass &= test_printf_with_layout_types()
    
    if all_pass:
        print("\n" + "="*80)
        print("All Print/Printf Tests PASSED!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ Some tests FAILED!")
        print("="*80)
        sys.exit(1)
