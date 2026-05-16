# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir.dialects import func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, Location, Module

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _build_ir(build_fn):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("test_make_rmem_tensor", FunctionType.get([], []))
                with InsertionPoint(f.add_entry_block()):
                    build_fn()
                    func.ReturnOp([])

            assert module.operation.verify()
            return str(module)


@pytest.mark.parametrize(
    ("shape", "expected_layout"),
    [
        (8, "8:1"),
        ((2, 3), "(2,3):(1,2)"),
        ((2, (3, 4)), "(2,(3,4)):(1,(2,6))"),
    ],
)
def test_make_rmem_tensor_builds_ordered_layout_from_shape(shape, expected_layout):
    ir = _build_ir(lambda: fx.make_rmem_tensor(shape, fx.Float32))

    assert "fly.make_ordered_layout" in ir
    assert f"!fly.memref<f32, register, {expected_layout}>" in ir


def test_make_rmem_tensor_builds_ordered_layout_from_shape_value():
    def build():
        shape = fx.make_shape(2, 3)
        fx.make_rmem_tensor(shape, fx.Float32)

    ir = _build_ir(build)

    assert "fly.make_ordered_layout" in ir
    assert "!fly.memref<f32, register, (2,3):(1,2)>" in ir


def test_make_rmem_tensor_preserves_layout_argument():
    def build():
        layout = fx.make_layout((2, 3), (8, 1))
        fx.make_rmem_tensor(layout, fx.Float16)

    ir = _build_ir(build)

    assert "fly.make_layout" in ir
    assert "fly.make_ordered_layout" not in ir
    assert "!fly.memref<f16, register, (2,3):(8,1)>" in ir
