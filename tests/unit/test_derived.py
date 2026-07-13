# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import contextlib

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


@contextlib.contextmanager
def _trace_context():
    """Open a context with an insertion point so ``fx.static`` ops can be built."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                yield


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


# ``Tile.unpack()`` recovers the per-mode value, mirroring ``IntTuple.unpack``.


def test_tile_unpack_int_modes():
    # Mirrors the host-side ``tiled_copy.tile_mn.unpack()`` use in 01-vectorAdd.py.
    with _trace_context():
        copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        tiled_copy = fx.make_tiled_copy_tv(
            copy_atom,
            fx.make_ordered_layout((8, 16), order=(1, 0)),
            fx.make_ordered_layout((1, 4), order=(0, 1)),
        )
        assert tiled_copy.tile_mn.unpack() == (8, 64)


def test_tile_unpack_single_mode_is_leaf():
    # A single (non-list) mode builds a leaf tile.
    with _trace_context():
        assert fx.make_tile(256).unpack() == 256


def test_tile_unpack_nested_modes():
    with _trace_context():
        assert fx.make_tile(128, (8, 4)).unpack() == (128, (8, 4))


def test_tile_unpack_layout_modes():
    with _trace_context():
        lt = fx.LayoutType.get(16, 1)
        modes = fx.static(fx.TileType.get([lt, lt])).unpack()
        assert isinstance(modes, tuple) and len(modes) == 2
        assert all(isinstance(m, fx.Layout) for m in modes)
        for m in modes:
            assert m.shape.unpack() == 16
            assert m.stride.unpack() == 1


def test_tile_unpack_mixed_int_and_layout_modes():
    with _trace_context():
        lt = fx.LayoutType.get(8, 1)
        modes = fx.static(fx.TileType.get([32, lt])).unpack()
        assert modes[0] == 32
        assert isinstance(modes[1], fx.Layout)
        assert modes[1].shape.unpack() == 8
