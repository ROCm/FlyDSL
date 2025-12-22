#!/usr/bin/env python3

import pytest

from _mlir.ir import IntegerAttr

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index


def _lower_and_get_single_return_int(module) -> int:
    pipeline = Pipeline().rocir_to_standard().canonicalize().cse()
    run_pipeline(module, pipeline)
    assert module.operation.verify()

    func_op = None
    for op in module.body.operations:
        if getattr(op.operation, "name", None) == "func.func":
            func_op = op
            break
    assert func_op is not None

    ret = func_op.entry_block.operations[-1]
    assert ret.name == "func.return"
    assert len(ret.operands) == 1

    v = ret.operands[0]
    if hasattr(v, "value"):
        v = v.value
    def_op = v.owner
    assert def_op.name == "arith.constant"
    val_attr = def_op.attributes["value"]
    assert isinstance(val_attr, IntegerAttr)
    return val_attr.value


def test_nested_layout_direct():
    class _M(rocir.MlirModule):
        @rocir.jit
        def nested_layout_direct(self: rocir.T.i64):
            c9 = Index(9)
            c4 = Index(4)
            c8 = Index(8)
            c59 = Index(59)
            c13 = Index(13)
            c1 = Index(1)
            layout = rocir.make_layout((c9, (c4, c8)), stride=(c59, (c13, c1)))
            return [rocir.size(layout).value]

    m = _M()
    assert "rocir.make_layout" in str(m.module)
    assert _lower_and_get_single_return_int(m.module) == 9 * 4 * 8


def test_multi_level_nested_shape_and_stride_spec():
    class _M(rocir.MlirModule):
        @rocir.jit
        def multi_level(self: rocir.T.i64):
            c9 = Index(9)
            c4 = Index(4)
            c8 = Index(8)
            c2 = Index(1) + Index(1)
            c59 = Index(59)
            c13 = Index(13)
            c1 = Index(1)
            shape = rocir.make_shape(c9, (c4, (c8, c2)))
            stride = rocir.make_stride(c59, (c13, (c2, c1)))
            layout = rocir.make_layout(shape, stride)
            return [rocir.size(layout).value]

    ir_text = str(_M().module)
    assert "!rocir.shape<(9,(4,(8,2)))>" in ir_text
    assert "!rocir.stride<(59,(13,(2,1)))>" in ir_text


def test_flat_divide_with_nested_layout():
    class _M(rocir.MlirModule):
        @rocir.jit
        def flat_divide_nested(self: rocir.T.i64):
            layout = rocir.make_layout(
                (Index(9), (Index(4), Index(8))),
                stride=(Index(59), (Index(13), Index(1))),
            )
            tiler = rocir.make_layout(
                (Index(3), (Index(2), Index(4))),
                stride=(Index(3), (Index(1), Index(8))),
            )
            res = rocir.flat_divide(layout, tiler)
            return [rocir.size(res).value]

    assert _lower_and_get_single_return_int(_M().module) == 9 * 4 * 8


def test_logical_divide_2d_nested():
    class _M(rocir.MlirModule):
        @rocir.jit
        def logical_divide_nested(self: rocir.T.i64):
            layout = rocir.make_layout(
                (Index(9), (Index(4), Index(8))),
                stride=(Index(59), (Index(13), Index(1))),
            )
            tiler = rocir.make_layout(
                (Index(3), (Index(2), Index(4))),
                stride=(Index(3), (Index(1), Index(8))),
            )
            res = rocir.logical_divide(layout, tiler)
            return [rocir.size(res).value]

    assert _lower_and_get_single_return_int(_M().module) == 9 * 4 * 8


def test_idx2crd_crd2idx_with_nested_layout_rank():
    class _M(rocir.MlirModule):
        @rocir.jit
        def roundtrip(self: rocir.T.i64):
            layout = rocir.make_layout(
                (Index(9), (Index(4), Index(8))),
                stride=(Index(59), (Index(13), Index(1))),
            )
            idx = Index(50)
            c = rocir.idx2crd(idx.value, layout)
            back = rocir.crd2idx(c, layout)
            return [back.value]

    # Make sure idx2crd uses the shape-rank (3 leaves), not shape+stride (6 leaves).
    ir_text = str(_M().module)
    assert "rocir.idx2crd" in ir_text
    # Coord type should preserve nested structure (domain shape), matching the layout's shape spec.
    assert "!rocir.coord<(9,(4,8))>" in ir_text


def test_composition_infers_structured_layout_type():
    class _M(rocir.MlirModule):
        @rocir.jit
        def composed(self: rocir.T.i64):
            A = rocir.make_layout((Index(6), Index(2)), stride=(Index(8), Index(2)))
            B = rocir.make_layout((Index(4), Index(3)), stride=(Index(3), Index(1)))
            R = rocir.composition(A, B)
            return [rocir.size(R).value]

    ir_text = str(_M().module)
    # Wrapper should infer a structured result type, not layout<-1>.
    assert "rocir.composition" in ir_text
    assert "!rocir.layout<-1>" not in ir_text
    # This is the canonical expected structure from the reference notebook/tests.
    assert "!rocir.layout<((2,2),3):((24,2),8)>" in ir_text


def test_crd2idx_on_composed_layout_pipeline_does_not_crash():
    # Regression test for the previous crash when crd2idx saw nested strides produced by composition lowering.
    class _M(rocir.MlirModule):
        @rocir.jit
        def f(self: rocir.T.i64):
            A = rocir.make_layout((Index(6), Index(2)), stride=(Index(8), Index(2)))
            B = rocir.make_layout((Index(4), Index(3)), stride=(Index(3), Index(1)))
            R = rocir.composition(A, B)
            # Composition above yields rank-3 (flattened leaf count), so use a rank-3 coord.
            c = rocir.make_coord(Index(1).value, Index(2).value, Index(0).value)
            out = rocir.crd2idx(c, R)
            return [out.value]

    m = _M()
    pipeline = Pipeline().rocir_to_standard().canonicalize().cse()
    run_pipeline(m.module, pipeline)
    assert m.module.operation.verify()


