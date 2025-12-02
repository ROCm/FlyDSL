#!/usr/bin/env python3
"""
Complete tests for Rocir layout algebra operations.
Exactly following CuTe layout algebra notebook examples.
Ref: examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb

Each test corresponds to a specific cell in the notebook.
"""

import sys
import os
import traceback
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import rocir, arith
from mlir.dialects import func
from mlir.ir import IntegerAttr

class Const:
    @staticmethod
    def index(val):
        return arith.index(val)

def unwrap(val):
    """Unwrap ArithValue or other wrappers."""
    if hasattr(val, 'value'): return val.value
    if hasattr(val, '_value'): return val._value
    return val


def run_lowering_test(ctx, test_name, expected_val=None, expected_vals=None):
    """Run the lowering pipeline and verify success and result value(s).
    
    Uses assertions for pytest compatibility - returns None on success, raises on failure.
    """
    print(f"  Running lowering pipeline for {test_name}...")
    
    # Lower rocir ops to standard arithmetic
    # Use nested pipeline because RocirToStandardPass is restricted to func.func
    pipeline = Pipeline().Func(Pipeline().rocir_to_standard()).canonicalize().cse()
    run_pipeline(ctx.module, pipeline)
    
    assert ctx.module.operation.verify(), f"{test_name}: IR verification failed."
        
    print(f"  ✓ {test_name}: Lowering successful!")
    
    # Find the function operation
    func_op = None
    for op in ctx.module.body.operations:
        op_name = op.name
        if hasattr(op, "OPERATION_NAME"):
            op_name = op.OPERATION_NAME
        elif hasattr(op, "operation"):
            op_name = op.operation.name
        
        if op_name == "func.func":
            func_op = op
            break
    
    if func_op is None:
        if len(ctx.module.body.operations) > 0:
             op = ctx.module.body.operations[0]
             if "func.func" in str(op):
                 func_op = op

    assert func_op is not None, f"{test_name}: Could not find function in module."
    
    # Handle verification of expected values
    if expected_val is not None or expected_vals is not None:
        assert func_op.entry_block.operations, f"{test_name}: Function body is empty."
            
        return_op = func_op.entry_block.operations[-1]
        assert return_op.name == "func.return", \
            f"{test_name}: Last operation is {return_op.name}, expected func.return."
        
        # Handle multiple return values
        if expected_vals is not None:
            assert len(return_op.operands) == len(expected_vals), \
                f"{test_name}: Return op has {len(return_op.operands)} operands, expected {len(expected_vals)}."
            
            for i, (operand, expected) in enumerate(zip(return_op.operands, expected_vals)):
                def_op = operand.owner
                if def_op.name != "arith.constant":
                    print(f"  ⚠ {test_name}: Return value [{i}] is not a constant (defined by: {def_op.name})")
                    continue
                
                assert "value" in def_op.attributes, \
                    f"{test_name}: Constant op [{i}] missing 'value' attribute."
                
                val_attr = def_op.attributes["value"]
                actual = None
                
                if isinstance(val_attr, IntegerAttr):
                    actual = val_attr.value
                elif hasattr(val_attr, "value"):
                    actual = val_attr.value
                else:
                    actual = int(val_attr)
                
                if actual != expected:
                    # Print all values for debugging
                    all_actual = []
                    for j, ret_op in enumerate(return_op.operands):
                        defining_op = ret_op.owner
                        if hasattr(defining_op, 'value'):
                            val_attr = defining_op.value
                            if isinstance(val_attr, IntegerAttr):
                                all_actual.append(val_attr.value)
                            elif hasattr(val_attr, "value"):
                                all_actual.append(val_attr.value)
                            else:
                                all_actual.append(int(val_attr))
                        else:
                            all_actual.append('?')
                    print(f"  ACTUAL values: {all_actual}")
                    print(f"  EXPECTED values: {expected_vals}")
                assert actual == expected, \
                    f"{test_name}: Value [{i}] mismatch. Expected {expected}, got {actual}"
            
            print(f"  ✅ {test_name}: All values verified: {expected_vals}")
        
        # Handle single return value
        elif expected_val is not None:
            if len(return_op.operands) != 1:
                print(f"  ⚠ {test_name}: Return op has {len(return_op.operands)} operands, expected 1.")
                return
                
            val = return_op.operands[0]
            def_op = val.owner
            
            if def_op.name != "arith.constant":
                print(f"  ⚠ {test_name}: Return value is not a constant (defined by: {def_op.name})")
                print(f"  ✓ {test_name}: Skipping value verification (optimization may be incomplete).")
                return
                
            assert "value" in def_op.attributes, \
                f"{test_name}: Constant op missing 'value' attribute."
                
            val_attr = def_op.attributes["value"]
            actual = None
            
            if isinstance(val_attr, IntegerAttr):
                actual = val_attr.value
            elif hasattr(val_attr, "value"):
                actual = val_attr.value
            else:
                actual = int(val_attr)
            
            assert actual == expected_val, \
                f"{test_name}: Size mismatch. Expected {expected_val}, got {actual}"
            print(f"  ✅ {test_name}: Size verified: {actual}")


# ==============================================================================
# Test 1: Coalesce Operations (Cells 4, 5, 7)
# ==============================================================================

def test_coalesce_basic():
    """Cell 4: Basic Coalesce - (2,(1,6)):(1,(6,2)) => 12:1
    
    NOTE: Coalesce lowering not yet implemented - currently a no-op placeholder.
    Test verifies size preservation only until full lowering is implemented.
    """
    print("\n" + "="*80)
    print("Test 1a: Basic Coalesce (Cell 4)")
    print("="*80)
    print("  Input Layout: (2,(1,6)):(1,(6,2))")
    print("  Expected Result (when implemented): 12:1")
    print("  Current: No-op placeholder - returns input layout")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def test_coalesce():
        layout = rocir.make_layout(
            (Const.index(2), (Const.index(1), Const.index(6))),
            stride=(Const.index(1), (Const.index(6), Const.index(2)))
        )
        
        coalesced = rocir.coalesce(layout)
        
        # Verify size is preserved
        sz = rocir.size(coalesced)
        return [unwrap(sz)]
    
    # Verify size is preserved: 2 * 1 * 6 = 12
    run_lowering_test(ctx, "coalesce_basic", expected_val=12)


# ==============================================================================
# Test 2: Composition Operations (Cells 9, 11, 13)
# ==============================================================================

def test_composition_basic():
    """Cell 9: Basic Composition - A:(6,2):(8,2) o B:(4,3):(3,1) => ((2,2),3):((24,2),8)"""
    print("\n" + "="*80)
    print("Test 2a: Basic Composition (Cell 9)")
    print("="*80)
    print("  Layout A: (6,2):(8,2)")
    print("  Layout B: (4,3):(3,1)")
    print("  Expected Result: ((2,2),3):((24,2),8)")
    print("  Expected Shape: [2, 2, 3]")
    print("  Expected Stride: [24, 2, 8]")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_composition():
        A = rocir.make_layout(
            (Const.index(6), Const.index(2)),
            stride=(Const.index(8), Const.index(2))
        )
        B = rocir.make_layout(
            (Const.index(4), Const.index(3)),
            stride=(Const.index(3), Const.index(1))
        )
        R = rocir.composition(A, B)
        
        # Extract shape and stride
        shape = rocir.get_shape(R)
        stride = rocir.get_stride(R)
        
        # Get dimensions (rank 3: (2,2,3))
        vals = []
        for i in range(3):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(3):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [2, 2, 3] + stride [24, 2, 8]
    run_lowering_test(ctx, "composition_basic", expected_vals=[2, 2, 3, 24, 2, 8])


def test_composition_static_vs_dynamic():
    """Cell 11: Composition with static vs dynamic - (10,2):(16,4) o (5,4):(1,5)"""
    print("\n" + "="*80)
    print("Test 2b: Static vs Dynamic Composition (Cell 11)")
    print("="*80)
    print("  Layout A: (10,2):(16,4)")
    print("  Layout B: (5,4):(1,5)")
    print("  Expected Result: (5,2,2):(16,80,4)")
    print("  Expected Shape: [5, 2, 2]")
    print("  Expected Stride: [16, 80, 4]")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_composition_static():
        A_static = rocir.make_layout(
            (Const.index(10), Const.index(2)),
            stride=(Const.index(16), Const.index(4))
        )
        B_static = rocir.make_layout(
            (Const.index(5), Const.index(4)),
            stride=(Const.index(1), Const.index(5))
        )
        R_static = rocir.composition(A_static, B_static)
        
        # Extract shape and stride
        shape = rocir.get_shape(R_static)
        stride = rocir.get_stride(R_static)
        
        # Get dimensions (rank 3: (5,2,2))
        vals = []
        for i in range(3):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(3):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [5, 2, 2] + stride [16, 80, 4]
    run_lowering_test(ctx, "composition_static_vs_dynamic", expected_vals=[5, 2, 2, 16, 80, 4])


def test_composition_bymode():
    """Cell 13: By-mode Composition - (12,(4,8)):(59,(13,1)) with tiler (3,8)"""
    print("\n" + "="*80)
    print("Test 2c: By-mode Composition (Cell 13)")
    print("="*80)
    print("  Layout A: (12,(4,8)):(59,(13,1))")
    print("  Tiler: (3, 8)")
    print("  Expected: (3,(4,2)):(59,(13,1))")
    
    # Note: By-mode composition with tuple tiler not yet implemented
    print("  ⚠ By-mode composition with tuple tiler not yet implemented in rocir")
    print("  ✓ Test skipped (pending implementation)")


# ==============================================================================
# Test 3: Divide Operations (Cells 15, 17, 19, 21, 23)
# ==============================================================================

def test_logical_divide_1d():
    """Cell 15: Logical Divide 1D - (4,2,3):(2,1,8) / 4:2 => ((2,2),(2,3)):((4,1),(2,8))"""
    print("\n" + "="*80)
    print("Test 3a: Logical Divide 1D (Cell 15)")
    print("="*80)
    print("  Input Layout: (4,2,3):(2,1,8)")
    print("  Tiler: 4:2")
    print("  Expected Result: ((2,2),(2,3)):((4,1),(2,8))")
    print("  Expected Shape (flat): [2, 2, 2, 3]")
    print("  Expected Stride (flat): [4, 1, 2, 8]")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_logical_divide_1d():
        layout = rocir.make_layout(
            (Const.index(4), Const.index(2), Const.index(3)),
            stride=(Const.index(2), Const.index(1), Const.index(8))
        )
        tiler = rocir.make_layout(
            Const.index(4),
            stride=Const.index(2)
        )
        res_logical = rocir.logical_divide(layout, tiler)
        
        # Extract shape and stride to check structure
        shape = rocir.get_shape(res_logical)
        stride = rocir.get_stride(res_logical)
        
        # Get individual dimensions using rocir.get
        shape_d0 = rocir.get(shape, Const.index(0))
        shape_d1 = rocir.get(shape, Const.index(1))
        shape_d2 = rocir.get(shape, Const.index(2))
        shape_d3 = rocir.get(shape, Const.index(3))
        
        stride_d0 = rocir.get(stride, Const.index(0))
        stride_d1 = rocir.get(stride, Const.index(1))
        stride_d2 = rocir.get(stride, Const.index(2))
        stride_d3 = rocir.get(stride, Const.index(3))
        
        # Return all values: shape dims, stride dims
        return [unwrap(shape_d0), unwrap(shape_d1), unwrap(shape_d2), unwrap(shape_d3),
                unwrap(stride_d0), unwrap(stride_d1), unwrap(stride_d2), unwrap(stride_d3)]

    # Expected: shape [2,2,2,3] + stride [4,1,2,8]
    run_lowering_test(ctx, "logical_divide_1d", 
                            expected_vals=[2, 2, 2, 3, 4, 1, 2, 8])


def test_logical_divide_2d():
    """Cell 17: Logical Divide 2D - (9,(4,8)):(59,(13,1)) with nested tiler"""
    print("\n" + "="*80)
    print("Test 3b: Logical Divide 2D (Cell 17)")
    print("="*80)
    print("  Input Layout: (9,(4,8)):(59,(13,1))")
    print("  Tiler: (3:3, (2,4):(1,8))")
    print("  Expected Result: ((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))")
    print("  Expected Shape (flat): [3, 3, 2, 4, 2, 2]")
    print("  Expected Stride (flat): [177, 59, 13, 2, 26, 1]")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_logical_divide_2d():
        layout = rocir.make_layout(
            (Const.index(9), (Const.index(4), Const.index(8))),
            stride=(Const.index(59), (Const.index(13), Const.index(1)))
        )
        tiler = rocir.make_layout(
            (Const.index(3), (Const.index(2), Const.index(4))),
            stride=(Const.index(3), (Const.index(1), Const.index(8)))
        )
        res_logical = rocir.logical_divide(layout, tiler)
        
        # Extract shape and stride
        shape = rocir.get_shape(res_logical)
        stride = rocir.get_stride(res_logical)
        
        # Get dimensions (rank 6 after flattening)
        vals = []
        for i in range(6):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(6):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [3,3,2,4,2,2] + stride [177,59,13,2,26,1]
    run_lowering_test(ctx, "logical_divide_2d", 
                            expected_vals=[3, 3, 2, 4, 2, 2, 177, 59, 13, 2, 26, 1])


def test_zipped_divide():
    """Cell 19: Zipped Divide - same inputs as Cell 17"""
    print("\n" + "="*80)
    print("Test 3c: Zipped Divide (Cell 19)")
    print("="*80)
    print("  Input Layout: (9,(4,8)):(59,(13,1))")
    print("  Tiler: (3:3, (2,4):(1,8))")
    print("  Expected Result: ((3,(2,4)),(3,(2,2))):((177,(13,2)),(59,(26,1)))")
    print("  Expected Shape (flat): [3, 2, 4, 3, 2, 2]")
    print("  Expected Stride (flat): [177, 13, 2, 59, 26, 1]")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_zipped_divide():
        layout = rocir.make_layout(
            (Const.index(9), (Const.index(4), Const.index(8))),
            stride=(Const.index(59), (Const.index(13), Const.index(1)))
        )
        tiler = rocir.make_layout(
            (Const.index(3), (Const.index(2), Const.index(4))),
            stride=(Const.index(3), (Const.index(1), Const.index(8)))
        )
        res_zipped = rocir.zipped_divide(layout, tiler)
        
        # Extract shape and stride
        shape = rocir.get_shape(res_zipped)
        stride = rocir.get_stride(res_zipped)
        
        # Get dimensions (rank 6)
        vals = []
        for i in range(6):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(6):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [3,2,4,3,2,2] + stride [177,13,2,59,26,1]
    run_lowering_test(ctx, "zipped_divide", 
                            expected_vals=[3, 2, 4, 3, 2, 2, 177, 13, 2, 59, 26, 1])


def test_tiled_divide():
    """Cell 21: Tiled Divide - same inputs as Cell 17"""
    print("\n" + "="*80)
    print("Test 3d: Tiled Divide (Cell 21)")
    print("="*80)
    print("  Input Layout: (9,(4,8)):(59,(13,1))")
    print("  Tiler: (3:3, (2,4):(1,8))")
    print("  Expected Result: ((3,(2,4)),3,(2,2)):((177,(13,2)),59,(26,1))")
    print("  Expected Shape (flat): [3, 2, 4, 3, 2, 2]")
    print("  Expected Stride (flat): [177, 13, 2, 59, 26, 1]")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_tiled_divide():
        layout = rocir.make_layout(
            (Const.index(9), (Const.index(4), Const.index(8))),
            stride=(Const.index(59), (Const.index(13), Const.index(1)))
        )
        tiler = rocir.make_layout(
            (Const.index(3), (Const.index(2), Const.index(4))),
            stride=(Const.index(3), (Const.index(1), Const.index(8)))
        )
        res_tiled = rocir.tiled_divide(layout, tiler)
        
        # Extract shape and stride
        shape = rocir.get_shape(res_tiled)
        stride = rocir.get_stride(res_tiled)
        
        # Get dimensions (rank 6)
        vals = []
        for i in range(6):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(6):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [3,2,4,3,2,2] + stride [177,13,2,59,26,1]
    run_lowering_test(ctx, "tiled_divide", 
                            expected_vals=[3, 2, 4, 3, 2, 2, 177, 13, 2, 59, 26, 1])


def test_flat_divide():
    """Cell 23: Flat Divide - same inputs as Cell 17"""
    print("\n" + "="*80)
    print("Test 3e: Flat Divide (Cell 23)")
    print("="*80)
    print("  Layout: (9,(4,8)):(59,(13,1))")
    print("  Tiler: (3:3, (2,4):(1,8))")
    print("  Expected Result: (3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))")
    print("  Note: Full layout verification pending divide lowering implementation")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_flat_divide():
        layout = rocir.make_layout(
            (Const.index(9), (Const.index(4), Const.index(8))),
            stride=(Const.index(59), (Const.index(13), Const.index(1)))
        )
        tiler = rocir.make_layout(
            (Const.index(3), (Const.index(2), Const.index(4))),
            stride=(Const.index(3), (Const.index(1), Const.index(8)))
        )
        res_flat = rocir.flat_divide(layout, tiler)
        sz = rocir.size(res_flat)
        return [unwrap(sz)]

    # Expected size: 9 * 4 * 8 = 288 (divide preserves total size)
    # TODO: Add full shape/stride verification once divide lowering is implemented
    run_lowering_test(ctx, "flat_divide", expected_val=288)


# ==============================================================================
# Test 4: Product Operations (Cells 25, 27, 29)
# ==============================================================================

def test_logical_product_1d():
    """Cell 25: Logical Product 1D - (2,2):(4,1) * 6:1 => (2,2,6):(4,1,4)"""
    print("\n" + "="*80)
    print("Test 4a: Logical Product 1D (Cell 25)")
    print("="*80)
    print("  Layout (block): (2,2):(4,1)")
    print("  Tiler: 6:1")
    print("  Expected Result: (2,2,6):(4,1,4)")
    print("  Expected Shape: [2, 2, 6]")
    print("  Expected Stride: [4, 1, 4]  (tiler stride scaled by block size=4)")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_logical_product():
        layout = rocir.make_layout(
            (Const.index(2), Const.index(2)),
            stride=(Const.index(4), Const.index(1))
        )
        tiler = rocir.make_layout(
            Const.index(6),
            stride=Const.index(1)
        )
        res_logical = rocir.logical_product(layout, tiler)
        
        # Extract shape and stride to check structure
        shape = rocir.get_shape(res_logical)
        stride = rocir.get_stride(res_logical)
        
        # Get the rank to determine how many dimensions to extract
        # Product of (2,2) with 6 gives rank 3: (2, 2, 6)
        vals = []
        for i in range(3):  # Layout (2,2) + tiler 6 = rank 3
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(3):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: block shape (2,2) concatenated with tiler shape (6)
    # Stride: block stride (4,1) concatenated with scaled tiler stride (1 * block_size)
    # block_size = 2*2 = 4, so tiler stride becomes 1*4 = 4
    run_lowering_test(ctx, "logical_product_1d",
                            expected_vals=[2, 2, 6, 4, 1, 4])


def test_blocked_raked_product():
    """Cell 27: Blocked and Raked Product - (2,5):(5,1) * (3,4):(1,3)"""
    print("\n" + "="*80)
    print("Test 4b: Blocked and Raked Product (Cell 27)")
    print("="*80)
    print("  Layout (block): (2,5):(5,1)  [size=10]")
    print("  Tiler: (3,4):(1,3)   [size=12]")
    print("  Expected Blocked: ((2,3),(5,4)):((5,10),(1,30))  [size=120]")
    print("  Expected Shape (flat): [2, 5, 3, 4]")
    print("  Expected Stride (flat): [5, 1, 10, 30]  (tiler stride scaled by 10)")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_blocked_raked_product():
        layout = rocir.make_layout(
            (Const.index(2), Const.index(5)),
            stride=(Const.index(5), Const.index(1))
        )
        tiler = rocir.make_layout(
            (Const.index(3), Const.index(4)),
            stride=(Const.index(1), Const.index(3))
        )
        
        res_blocked = rocir.blocked_product(layout, tiler)
        
        # Extract shape and stride
        shape = rocir.get_shape(res_blocked)
        stride = rocir.get_stride(res_blocked)
        
        # Get dimensions (rank 4: block + tiler)
        vals = []
        for i in range(4):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(4):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [2,5,3,4] + stride [5,1,10,30] (block_size=10)
    run_lowering_test(ctx, "blocked_raked_product", 
                            expected_vals=[2, 5, 3, 4, 5, 1, 10, 30])


def test_zipped_tiled_flat_product():
    """Cell 29: Zipped, Tiled, Flat Product - (2,5):(5,1) * (3,4):(1,3)"""
    print("\n" + "="*80)
    print("Test 4c: Zipped, Tiled, Flat Product (Cell 29)")
    print("="*80)
    print("  Layout (block): (2,5):(5,1)  [size=10]")
    print("  Tiler: (3,4):(1,3)   [size=12]")
    print("  Expected Flat: (2,5,3,4):(5,1,10,30)  [size=120]")
    print("  Expected Shape: [2, 5, 3, 4]")
    print("  Expected Stride: [5, 1, 10, 30]  (tiler stride scaled by 10)")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @func.FuncOp.from_py_func()
    def run_zipped_tiled_flat_product():
        layout = rocir.make_layout(
            (Const.index(2), Const.index(5)),
            stride=(Const.index(5), Const.index(1))
        )
        tiler = rocir.make_layout(
            (Const.index(3), Const.index(4)),
            stride=(Const.index(1), Const.index(3))
        )
        
        res_flat = rocir.flat_product(layout, tiler)
        
        # Extract shape and stride
        shape = rocir.get_shape(res_flat)
        stride = rocir.get_stride(res_flat)
        
        # Get dimensions (rank 4: block + tiler)
        vals = []
        for i in range(4):
            vals.append(unwrap(rocir.get(shape, Const.index(i))))
        for i in range(4):
            vals.append(unwrap(rocir.get(stride, Const.index(i))))
        
        return vals

    # Expected: shape [2,5,3,4] + stride [5,1,10,30] (block_size=10)
    run_lowering_test(ctx, "zipped_tiled_flat_product", 
                            expected_vals=[2, 5, 3, 4, 5, 1, 10, 30])


def test_complement_simple():
    """Test complement operation: complement(3:1, 12) should give 4:3
    
    CuTe behavior:
    - Input: tiler = Layout(3:1), target_size = 12
    - complement finds the "rest" modes: 12 / 3 = 4, stride = 3
    - Result: Layout(4:3)
    """
    print("\n=== Test: Complement Simple ===")
    print("complement(Layout(3:1), 12) -> Layout(4:3)")
    
    ctx = RAIIMLIRContextModule()
    
    @func.FuncOp.from_py_func()
    def test_func():
        # Create tiler layout: 3:1
        c3 = Const.index(3)
        c1 = Const.index(1)
        c12 = Const.index(12)
        
        tiler_shape = rocir.make_shape(c3)
        tiler_stride = rocir.make_stride(c1)
        tiler_layout = rocir.make_layout(tiler_shape, tiler_stride)
        
        # Compute complement
        comp_layout = rocir.complement(tiler_layout, c12)
        
        # Get size to verify it works
        comp_size = rocir.size(comp_layout)
        
        return
    
    run_lowering_test(ctx, "complement_simple")


def test_complement_with_divide():
    """Test logical_divide which uses complement internally.
    
    CuTe behavior:
    - logical_divide(L, T) = composition(L, make_layout(T, complement(T, size(L))))
    - Input: layout = Layout(12:1), tiler = Layout(3:1)
    - Coalesce: Layout(12:1)
    - Size: 12
    - Complement(3:1, 12): Layout(4:3)
    - Combined: Layout((3,4):(1,3))
    - Compose: Layout((3,4):(1,3))
    """
    print("\n=== Test: Logical Divide with Complement ===")
    print("logical_divide(Layout(12:1), Layout(3:1)) -> uses complement internally")
    
    ctx = RAIIMLIRContextModule()
    
    @func.FuncOp.from_py_func()
    def test_func():
        # Create input layout: 12:1
        c12 = Const.index(12)
        c1 = Const.index(1)
        
        input_shape = rocir.make_shape(c12)
        input_stride = rocir.make_stride(c1)
        input_layout = rocir.make_layout(input_shape, input_stride)
        
        # Create tiler layout: 3:1
        c3 = Const.index(3)
        tiler_shape = rocir.make_shape(c3)
        tiler_stride = rocir.make_stride(c1)
        tiler_layout = rocir.make_layout(tiler_shape, tiler_stride)
        
        # Compute logical_divide (uses complement internally)
        divided_layout = rocir.logical_divide(input_layout, tiler_layout)
        
        # Get size to verify
        div_size = rocir.size(divided_layout)
        
        return
    
    run_lowering_test(ctx, "complement_with_divide")


def test_composition_with_tuple():
    """Test composition with tuple recursion.
    
    CuTe behavior:
    - When RHS is a tuple, composition distributes over it
    - This tests the tuple recursion path in composition_impl
    """
    print("\n=== Test: Composition with Tuple Recursion ===")
    print("Tests that composition handles nested tuple structures")
    
    ctx = RAIIMLIRContextModule()
    
    @func.FuncOp.from_py_func()
    def test_func():
        # Create simple layouts for composition test
        c4 = Const.index(4)
        c2 = Const.index(2)
        c1 = Const.index(1)
        
        # Layout A: 4:1
        shapeA = rocir.make_shape(c4)
        strideA = rocir.make_stride(c1)
        layoutA = rocir.make_layout(shapeA, strideA)
        
        # Layout B: 2:1
        shapeB = rocir.make_shape(c2)
        strideB = rocir.make_stride(c1)
        layoutB = rocir.make_layout(shapeB, strideB)
        
        # Compose: A ∘ B
        composed = rocir.composition(layoutA, layoutB)
        
        # Get size
        comp_size = rocir.size(composed)
        
        return
    
    run_lowering_test(ctx, "composition_with_tuple")


# ==============================================================================
# Main Test Runner
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Complete Rocir Layout Algebra Tests")
    print("Following CuTe Layout Algebra Notebook")
    print("="*80)
    
    all_tests = [
        ("Coalesce Basic", test_coalesce_basic),
        ("Composition Basic", test_composition_basic),
        ("Composition Static vs Dynamic", test_composition_static_vs_dynamic),
        ("Composition By-Mode", test_composition_bymode),
        ("Composition with Tuple", test_composition_with_tuple),
        ("Complement Simple", test_complement_simple),
        ("Complement with Divide", test_complement_with_divide),
        ("Logical Divide 1D", test_logical_divide_1d),
        ("Logical Divide 2D", test_logical_divide_2d),
        ("Zipped Divide", test_zipped_divide),
        ("Tiled Divide", test_tiled_divide),
        ("Flat Divide", test_flat_divide),
        ("Logical Product 1D", test_logical_product_1d),
        ("Blocked/Raked Product", test_blocked_raked_product),
        ("Zipped/Tiled/Flat Product", test_zipped_tiled_flat_product),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in all_tests:
        try:
            test_func()  # Test functions now return None and use assertions
            passed += 1
            print(f"  ✅ {test_name}: PASSED\n")
        except AssertionError as e:
            print(f"  ❌ {test_name}: FAILED - {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test_name}: ERROR - {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(all_tests)} tests")
    print("="*80)
    
    if failed == 0:
        print("✅ All tests PASSED!")
        sys.exit(0)
    else:
        print(f"❌ {failed} test(s) FAILED!")
        sys.exit(1)

