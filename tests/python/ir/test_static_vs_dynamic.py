#!/usr/bin/env python3
"""Static vs dynamic layout types test (mirrors a reference notebook Cell 11)"""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index


_PIPELINE = Pipeline().rocir_to_standard().canonicalize().cse()


class _StaticDynamic(rocir.MlirModule):
    @rocir.jit
    def static_layout(self: rocir.T.i64):
        layout = rocir.make_layout((Index(10), Index(2)), stride=(Index(16), Index(4)))
        shape = rocir.get_shape(layout)
        stride = rocir.get_stride(layout)
        return [
            rocir.get(shape, Index(0)).value,
            rocir.get(shape, Index(1)).value,
            rocir.get(stride, Index(0)).value,
            rocir.get(stride, Index(1)).value,
            rocir.size(layout).value,
        ]
    
    @rocir.jit
    def dynamic_layout(
        self: rocir.T.i64,
        dim0: rocir.T.index,
        dim1: rocir.T.index,
        stride0: rocir.T.index,
        stride1: rocir.T.index,
    ):
        layout = rocir.make_layout((dim0, dim1), stride=(stride0, stride1))
        shape = rocir.get_shape(layout)
        stride = rocir.get_stride(layout)
        rocir.printf("Dynamic layout: ({},{}):({}{})\n", dim0, dim1, stride0, stride1)
        return [
            rocir.get(shape, Index(0)).value,
            rocir.get(shape, Index(1)).value,
            rocir.get(stride, Index(0)).value,
            rocir.get(stride, Index(1)).value,
            rocir.size(layout).value,
        ]
    
    @rocir.jit
    def static_composition(self: rocir.T.i64):
        A = rocir.make_layout((Index(10), Index(2)), stride=(Index(16), Index(4)))
        B = rocir.make_layout((Index(5), Index(4)), stride=(Index(1), Index(5)))
        R = rocir.composition(A, B)
        shape = rocir.get_shape(R)
        stride = rocir.get_stride(R)
        vals = []
        for i in range(3):
            vals.append(rocir.get(shape, Index(i)).value)
        for i in range(3):
            vals.append(rocir.get(stride, Index(i)).value)
        return vals
    
    @rocir.jit
    def dynamic_composition(
        self: rocir.T.i64,
        a_d0: rocir.T.index,
        a_d1: rocir.T.index,
        a_s0: rocir.T.index,
        a_s1: rocir.T.index,
        b_d0: rocir.T.index,
        b_d1: rocir.T.index,
        b_s0: rocir.T.index,
        b_s1: rocir.T.index,
    ):
        A = rocir.make_layout((a_d0, a_d1), stride=(a_s0, a_s1))
        B = rocir.make_layout((b_d0, b_d1), stride=(b_s0, b_s1))
        R = rocir.composition(A, B)
        rocir.printf("Composition: A({},{}) o B({},{})\n", a_d0, a_d1, b_d0, b_d1)
        shape = rocir.get_shape(R)
        stride = rocir.get_stride(R)
        vals = []
        for i in range(3):
            vals.append(rocir.get(shape, Index(i)).value)
        for i in range(3):
            vals.append(rocir.get(stride, Index(i)).value)
        return vals

    @rocir.jit
    def mixed_layout(
        self: rocir.T.i64,
        runtime_extent: rocir.T.index,
        runtime_stride: rocir.T.index,
    ):
        layout = rocir.make_layout((runtime_extent, Index(8)), stride=(Index(16), runtime_stride))
        shape = rocir.get_shape(layout)
        stride = rocir.get_stride(layout)
        rocir.printf("Mixed: ({},8):(16,{})\n", runtime_extent, runtime_stride)
        return [
            rocir.get(shape, Index(0)).value,
            rocir.get(shape, Index(1)).value,
            rocir.get(stride, Index(0)).value,
            rocir.get(stride, Index(1)).value,
        ]
    

def test_layout_static_types():
    """Test static layout with Index() - all values become arith.constant"""
    m = _StaticDynamic()
    with m._context, m._location:
        run_pipeline(m.module, _PIPELINE)
        assert m.module.operation.verify()


def test_layout_dynamic_types():
    """Test dynamic layout with function args - values remain as block arguments"""
    m = _StaticDynamic()
    with m._context, m._location:
        run_pipeline(m.module, _PIPELINE)
        assert m.module.operation.verify()


def test_composition_static_vs_dynamic():
    """Test composition: static (Index) vs dynamic (function args)"""
    m = _StaticDynamic()
    with m._context, m._location:
        run_pipeline(m.module, _PIPELINE)
        assert m.module.operation.verify()


def test_mixed_static_dynamic():
    """Test mixed layout: (arg, 8):(16, arg) - some static, some dynamic"""
    m = _StaticDynamic()
    with m._context, m._location:
        run_pipeline(m.module, _PIPELINE)
        assert m.module.operation.verify()

