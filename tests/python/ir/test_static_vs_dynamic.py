#!/usr/bin/env python3
"""Static vs dynamic layout types test (mirrors a reference notebook Cell 11)"""

import sys
import os


from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index
from mlir.dialects import func as mlir_func
from mlir.ir import IndexType

def unwrap(val):
    if hasattr(val, 'value'): return val.value
    if hasattr(val, '_value'): return val._value
    return val


def test_layout_static_types():
    """Test static layout with Index() - all values become arith.constant"""
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @mlir_func.FuncOp.from_py_func()
    def static_layout():
        layout = rocir.make_layout(
            (Index(10), Index(2)),
            stride=(Index(16), Index(4))
        )
        shape = rocir.get_shape(layout)
        stride = rocir.get_stride(layout)
        return [
            unwrap(rocir.get(shape, Index(0))),
            unwrap(rocir.get(shape, Index(1))),
            unwrap(rocir.get(stride, Index(0))),
            unwrap(rocir.get(stride, Index(1))),
            unwrap(rocir.size(layout)),
        ]
    
    pipeline = Pipeline().Func(Pipeline().rocir_to_standard()).canonicalize().cse()
    run_pipeline(ctx.module, pipeline)
    assert ctx.module.operation.verify()


def test_layout_dynamic_types():
    """Test dynamic layout with function args - values remain as block arguments"""
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    index_type = IndexType.get()
    
    @mlir_func.FuncOp.from_py_func(index_type, index_type, index_type, index_type)
    def dynamic_layout(dim0, dim1, stride0, stride1):
        layout = rocir.make_layout((dim0, dim1), stride=(stride0, stride1))
        shape = rocir.get_shape(layout)
        stride = rocir.get_stride(layout)
        
        rocir.printf("Dynamic layout: ({},{}):({}{})\n", dim0, dim1, stride0, stride1)
        
        return [
            unwrap(rocir.get(shape, Index(0))),
            unwrap(rocir.get(shape, Index(1))),
            unwrap(rocir.get(stride, Index(0))),
            unwrap(rocir.get(stride, Index(1))),
            unwrap(rocir.size(layout)),
        ]
    
    pipeline = Pipeline().Func(Pipeline().rocir_to_standard()).canonicalize().cse()
    run_pipeline(ctx.module, pipeline)
    assert ctx.module.operation.verify()


def test_composition_static_vs_dynamic():
    """Test composition: static (Index) vs dynamic (function args)"""
    # Static composition
    ctx_static = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    
    @mlir_func.FuncOp.from_py_func()
    def static_composition():
        A = rocir.make_layout((Index(10), Index(2)), stride=(Index(16), Index(4)))
        B = rocir.make_layout((Index(5), Index(4)), stride=(Index(1), Index(5)))
        R = rocir.composition(A, B)
        
        shape = rocir.get_shape(R)
        stride = rocir.get_stride(R)
        vals = []
        for i in range(3):
            vals.append(unwrap(rocir.get(shape, Index(i))))
        for i in range(3):
            vals.append(unwrap(rocir.get(stride, Index(i))))
        return vals
    
    pipeline = Pipeline().Func(Pipeline().rocir_to_standard()).canonicalize().cse()
    run_pipeline(ctx_static.module, pipeline)
    assert ctx_static.module.operation.verify()
    
    # Dynamic composition
    ctx_dynamic = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    index_type = IndexType.get()
    
    @mlir_func.FuncOp.from_py_func(index_type, index_type, index_type, index_type,
                                    index_type, index_type, index_type, index_type)
    def dynamic_composition(a_d0, a_d1, a_s0, a_s1, b_d0, b_d1, b_s0, b_s1):
        A = rocir.make_layout((a_d0, a_d1), stride=(a_s0, a_s1))
        B = rocir.make_layout((b_d0, b_d1), stride=(b_s0, b_s1))
        R = rocir.composition(A, B)
        
        rocir.printf("Composition: A({},{}) o B({},{})\n", a_d0, a_d1, b_d0, b_d1)
        
        shape = rocir.get_shape(R)
        stride = rocir.get_stride(R)
        vals = []
        for i in range(3):
            vals.append(rocir.get(shape, Index(i)))
        for i in range(3):
            vals.append(rocir.get(stride, Index(i)))
        return [unwrap(v) for v in vals]
    
    run_pipeline(ctx_dynamic.module, pipeline)
    assert ctx_dynamic.module.operation.verify()


def test_mixed_static_dynamic():
    """Test mixed layout: (arg, 8):(16, arg) - some static, some dynamic"""
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    index_type = IndexType.get()
    
    @mlir_func.FuncOp.from_py_func(index_type, index_type)
    def mixed_layout(runtime_extent, runtime_stride):
        layout = rocir.make_layout(
            (runtime_extent, Index(8)),
            stride=(Index(16), runtime_stride)
        )
        shape = rocir.get_shape(layout)
        stride = rocir.get_stride(layout)
        
        rocir.printf("Mixed: ({},8):(16,{})\n", runtime_extent, runtime_stride)
        
        return [
            unwrap(rocir.get(shape, Index(0))),
            unwrap(rocir.get(shape, Index(1))),
            unwrap(rocir.get(stride, Index(0))),
            unwrap(rocir.get(stride, Index(1))),
        ]
    
    pipeline = Pipeline().Func(Pipeline().rocir_to_standard()).canonicalize().cse()
    run_pipeline(ctx.module, pipeline)
    assert ctx.module.operation.verify()


if __name__ == "__main__":
    tests = [
        ("Static Layout", test_layout_static_types),
        ("Dynamic Layout", test_layout_dynamic_types),
        ("Composition", test_composition_static_vs_dynamic),
        ("Mixed", test_mixed_static_dynamic),
    ]
    
    for name, func in tests:
        try:
            func()
            print(f"{name}: PASSED")
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

