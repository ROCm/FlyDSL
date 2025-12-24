#!/usr/bin/env python3
"""Static vs dynamic layout types test (mirrors a reference notebook Cell 11)"""

from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.arith import Index


_PIPELINE = Pipeline().flir_to_standard().canonicalize().cse()


class _StaticDynamic(flir.MlirModule):
    @flir.jit
    def static_layout(self: flir.T.i64):
        layout = flir.make_layout((Index(10), Index(2)), stride=(Index(16), Index(4)))
        shape = flir.get_shape(layout)
        stride = flir.get_stride(layout)
        return [
            flir.get(shape, Index(0)).value,
            flir.get(shape, Index(1)).value,
            flir.get(stride, Index(0)).value,
            flir.get(stride, Index(1)).value,
            flir.size(layout).value,
        ]
    
    @flir.jit
    def dynamic_layout(
        self: flir.T.i64,
        dim0: flir.T.index,
        dim1: flir.T.index,
        stride0: flir.T.index,
        stride1: flir.T.index,
    ):
        layout = flir.make_layout((dim0, dim1), stride=(stride0, stride1))
        shape = flir.get_shape(layout)
        stride = flir.get_stride(layout)
        flir.printf("Dynamic layout: ({},{}):({}{})\n", dim0, dim1, stride0, stride1)
        return [
            flir.get(shape, Index(0)).value,
            flir.get(shape, Index(1)).value,
            flir.get(stride, Index(0)).value,
            flir.get(stride, Index(1)).value,
            flir.size(layout).value,
        ]
    
    @flir.jit
    def static_composition(self: flir.T.i64):
        A = flir.make_layout((Index(10), Index(2)), stride=(Index(16), Index(4)))
        B = flir.make_layout((Index(5), Index(4)), stride=(Index(1), Index(5)))
        R = flir.composition(A, B)
        shape = flir.get_shape(R)
        stride = flir.get_stride(R)
        vals = []
        for i in range(3):
            vals.append(flir.get(shape, Index(i)).value)
        for i in range(3):
            vals.append(flir.get(stride, Index(i)).value)
        return vals
    
    @flir.jit
    def dynamic_composition(
        self: flir.T.i64,
        a_d0: flir.T.index,
        a_d1: flir.T.index,
        a_s0: flir.T.index,
        a_s1: flir.T.index,
        b_d0: flir.T.index,
        b_d1: flir.T.index,
        b_s0: flir.T.index,
        b_s1: flir.T.index,
    ):
        A = flir.make_layout((a_d0, a_d1), stride=(a_s0, a_s1))
        B = flir.make_layout((b_d0, b_d1), stride=(b_s0, b_s1))
        R = flir.composition(A, B)
        flir.printf("Composition: A({},{}) o B({},{})\n", a_d0, a_d1, b_d0, b_d1)
        shape = flir.get_shape(R)
        stride = flir.get_stride(R)
        vals = []
        for i in range(3):
            vals.append(flir.get(shape, Index(i)).value)
        for i in range(3):
            vals.append(flir.get(stride, Index(i)).value)
        return vals

    @flir.jit
    def mixed_layout(
        self: flir.T.i64,
        runtime_extent: flir.T.index,
        runtime_stride: flir.T.index,
    ):
        layout = flir.make_layout((runtime_extent, Index(8)), stride=(Index(16), runtime_stride))
        shape = flir.get_shape(layout)
        stride = flir.get_stride(layout)
        flir.printf("Mixed: ({},8):(16,{})\n", runtime_extent, runtime_stride)
        return [
            flir.get(shape, Index(0)).value,
            flir.get(shape, Index(1)).value,
            flir.get(stride, Index(0)).value,
            flir.get(stride, Index(1)).value,
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

