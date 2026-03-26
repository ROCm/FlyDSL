#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Layout algebra tests using the Fly dialect API with static types.

Each test corresponds to a specific cell in the reference layout-algebra notebook.
Uses fly.static + IntTupleType.get() for fully static layouts.
"""

import sys
import pytest

from flydsl._mlir.passmanager import PassManager
from flydsl._mlir.ir import (
    Context, Location, Module, InsertionPoint,
    FunctionType, IntegerType, IndexType, BlockArgument,
)
from flydsl._mlir.dialects.fly import IntTupleType
from flydsl._mlir.dialects import fly, arith, func
import flydsl.expr as fx

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


FLY_PIPELINE = (
    "builtin.module("
    "fly-canonicalize,"
    "fly-layout-lowering,"
    "fly-canonicalize,"
    "convert-fly-to-rocdl,"
    "canonicalize,"
    "cse)"
)


def _S(spec):
    """Create a static int_tuple via fly.static."""
    return fly.static(IntTupleType.get(spec))


def _L(shape_spec, stride_spec):
    """Create a static layout from Python tuple specs."""
    return fly.make_layout(_S(shape_spec), stride=_S(stride_spec))


def _build_and_verify(name, build_fn, expected_vals):
    """Build IR, lower, and verify constant-folded return values."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            idx = IndexType.get()
            n = len(expected_vals)
            with InsertionPoint(module.body):
                f = func.FuncOp(name, FunctionType.get([], [idx] * n))
                with InsertionPoint(f.add_entry_block()):
                    results = build_fn()
                    ret_vals = []
                    for r in results:
                        ty = str(r.type) if hasattr(r, "type") else ""
                        if "int_tuple" in ty:
                            r = fly.get_scalar(r)
                        if hasattr(r, "type") and str(r.type) != "index":
                            r = arith.IndexCastOp(idx, r).result
                        ret_vals.append(r)
                    func.ReturnOp(ret_vals)

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify(), f"{name}: IR verification failed"

            func_op = list(module.body.operations)[0]
            ret_op = list(func_op.entry_block.operations)[-1]
            for i, (operand, expected) in enumerate(zip(ret_op.operands, expected_vals)):
                if expected is None:
                    continue
                actual = int(operand.owner.attributes["value"])
                assert actual == expected, f"{name}[{i}]: expected {expected}, got {actual}"


def _build_and_verify_ir(name, build_fn, check_fn):
    """Build IR (no lowering) and check the IR text."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp(name, FunctionType.get([], []))
                with InsertionPoint(f.add_entry_block()):
                    build_fn()
                    func.ReturnOp([])
            check_fn(str(module))


# ==============================================================================
# 1. Basic layout construction & size (Cells 1-3)
# ==============================================================================

def test_basic_size():
    """make_layout((3,9):(1,3)) => size = 27"""
    def build():
        return [fly.size(_L((3, 9), (1, 3)))]
    _build_and_verify("basic_size", build, [27])


def test_nested_layout_size():
    """(9,(4,8)):(59,(13,1)) => size = 288"""
    def build():
        return [fly.size(_L((9, (4, 8)), (59, (13, 1))))]
    _build_and_verify("nested_layout_size", build, [288])


def test_shape_stride_type_nested_spec_printing():
    """Nested shape/stride types print in tuple form."""
    def build():
        s = _S((9, (4, 8)))
        d = _S((59, (13, 1)))
        _L((9, (4, 8)), (59, (13, 1)))
    def check(ir):
        assert "!fly.int_tuple<(9,(4,8))>" in ir
        assert "!fly.int_tuple<(59,(13,1))>" in ir
    _build_and_verify_ir("nested_spec_printing", build, check)


# ==============================================================================
# 2. Coalesce (Cells 4, 5, 7)
# ==============================================================================

def test_coalesce_basic():
    """Cell 4: coalesce((3,(1,9)):(1,(9,3))) => size preserved = 27"""
    def build():
        return [fly.size(fly.coalesce(_L((3, (1, 9)), (1, (9, 3)))))]
    _build_and_verify("coalesce_basic", build, [27])


def test_coalesce_dynamic_stride():
    """Cell 4 dynamic: verify mixed static/dynamic survives lowering."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("coalesce_dyn", FunctionType.get([i32], [idx]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    runtime_stride = entry.arguments[0]
                    shape = fx.make_shape(
                        arith.ConstantOp(i32, 2).result,
                        (arith.ConstantOp(i32, 1).result, arith.ConstantOp(i32, 6).result),
                    )
                    stride = fx.make_stride(
                        arith.ConstantOp(i32, 1).result,
                        (runtime_stride, arith.ConstantOp(i32, 2).result),
                    )
                    layout = fx.make_layout(shape, stride)
                    coalesced = fly.coalesce(layout)
                    sz = fly.size(coalesced)
                    sc = fly.get_scalar(sz)
                    func.ReturnOp([arith.IndexCastOp(idx, sc).result])

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()


# ==============================================================================
# 3. Composition (Cells 9, 11, 13)
# ==============================================================================

def test_composition_basic():
    """Cell 9: (6,9):(19,69) o (6,3):(3,1) => size = 18"""
    def build():
        return [fly.size(fly.composition(_L((6, 9), (19, 69)), _L((6, 3), (3, 1))))]
    _build_and_verify("composition_basic", build, [18])


def test_composition_static_vs_dynamic():
    """Cell 11: Static composition + dynamic composition both lower successfully.

    Static: (5,15):(19,51) o (3,5):(1,5) => size = 15.
    Dynamic: function-arg layouts lower without error.
    """
    # Part 1: Static
    def build_static():
        return [fly.size(fly.composition(_L((5, 15), (19, 51)), _L((3, 5), (1, 5))))]
    _build_and_verify("composition_static", build_static, [15])

    # Part 2: Dynamic
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("comp_dyn", FunctionType.get([i32] * 8, [idx]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    args = list(entry.arguments)
                    A = fx.make_layout(fx.make_shape(args[0], args[1]), fx.make_stride(args[2], args[3]))
                    B = fx.make_layout(fx.make_shape(args[4], args[5]), fx.make_stride(args[6], args[7]))
                    R = fx.composition(A, B)
                    sz = fx.size(R)
                    sc = fx.get_scalar(sz)
                    func.ReturnOp([arith.IndexCastOp(idx, sc).result])
            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()


def test_composition_bymode():
    """Cell 13: By-mode composition using fly.make_tile."""
    def build():
        layout = _L((9, (4, 8)), (59, (13, 1)))
        tile_m0 = _L((3,), (3,))
        tile_m1 = _L((2, 4), (1, 8))
        tiler = fly.make_tile([tile_m0, tile_m1])
        R = fly.logical_divide(layout, tiler)
        return [fly.size(R)]
    _build_and_verify("composition_bymode", build, [288])


def test_composition_with_tuple():
    """4:1 o 2:1 => size = 2"""
    def build():
        return [fly.size(fly.composition(_L((4,), (1,)), _L((2,), (1,))))]
    _build_and_verify("composition_with_tuple", build, [2])


# ==============================================================================
# 4. Complement
# ==============================================================================

def test_complement_simple_rank_1():
    """complement(3:1, 12) => size = 4"""
    def build():
        return [fly.size(fly.complement(_L((3,), (1,)), codomain_size=_S(12)))]
    _build_and_verify("complement_simple_rank_1", build, [4])


def test_complement_simple_rank_2():
    """complement((3,2):(2,1), 12) => size = 2"""
    def build():
        return [fly.size(fly.complement(_L((3, 2), (2, 1)), codomain_size=_S(12)))]
    _build_and_verify("complement_simple_rank_2", build, [2])


def test_complement_rank_2_error():
    """Rank-2 non-injective complement: (3,2):(1,2).

    Fly dialect does NOT raise on non-injective layouts (unlike the legacy dialect).
    Verify it runs without crash and returns a result.
    """
    def build():
        comp = fly.complement(_L((3, 2), (1, 2)), codomain_size=_S(12))
        return [fly.size(comp)]
    # Fly returns a result (doesn't error); just verify no crash.
    _build_and_verify("complement_rank_2_error", build, [0])


def test_complement_rank_1_error():
    """Rank-1 non-injective complement: 3:0.

    Fly dialect does NOT raise on non-injective layouts.
    """
    def build():
        comp = fly.complement(_L((3,), (0,)), codomain_size=_S(12))
        return [fly.size(comp)]
    _build_and_verify("complement_rank_1_error", build, [12])


@pytest.mark.skip(reason="fly.complement with dynamic rank-2 stride crashes in C++ op construction")
def test_complement_rank_2_dynamic_stride_error():
    """Rank-2 complement with dynamic stride: lowering should still succeed."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("compl_dyn", FunctionType.get([i32], [idx]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    runtime_stride = entry.arguments[0]
                    shape = fx.make_shape(arith.ConstantOp(i32, 3).result, arith.ConstantOp(i32, 2).result)
                    stride = fx.make_stride(runtime_stride, arith.ConstantOp(i32, 1).result)
                    tiler = fx.make_layout(shape, stride)
                    comp = fx.complement(tiler, 12)
                    sz = fx.size(comp)
                    sc = fx.get_scalar(sz)
                    func.ReturnOp([arith.IndexCastOp(idx, sc).result])

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()


def test_complement_with_divide():
    """logical_divide(12:1, 3:1) uses complement internally => size = 12"""
    def build():
        return [fly.size(fly.logical_divide(_L((12,), (1,)), _L((3,), (1,))))]
    _build_and_verify("complement_with_divide", build, [12])


# ==============================================================================
# 5. Divide Operations (Cells 15, 17, 19, 21, 23)
# ==============================================================================

def test_logical_divide_1d():
    """Cell 15: 16:1 / 4:1 => size = 16"""
    def build():
        return [fly.size(fly.logical_divide(_L((16,), (1,)), _L((4,), (1,))))]
    _build_and_verify("logical_divide_1d", build, [16])


def test_logical_divide_2d():
    """Cell 17: (4,8):(1,4) / (2,4):(1,2) => size = 32"""
    def build():
        return [fly.size(fly.logical_divide(_L((4, 8), (1, 4)), _L((2, 4), (1, 2))))]
    _build_and_verify("logical_divide_2d", build, [32])


def test_zipped_divide():
    """Cell 19: zipped_divide preserves size = 32"""
    def build():
        return [fly.size(fly.zipped_divide(_L((4, 8), (1, 4)), _L((2, 4), (1, 2))))]
    _build_and_verify("zipped_divide", build, [32])


def test_tiled_divide():
    """Cell 21: tiled_divide preserves size = 32"""
    def build():
        return [fly.size(fly.tiled_divide(_L((4, 8), (1, 4)), _L((2, 4), (1, 2))))]
    _build_and_verify("tiled_divide", build, [32])


def test_flat_divide():
    """Cell 23: flat_divide preserves size = 32"""
    def build():
        return [fly.size(fly.flat_divide(_L((4, 8), (1, 4)), _L((2, 4), (1, 2))))]
    _build_and_verify("flat_divide", build, [32])


# ==============================================================================
# 6. Product Operations (Cells 25, 27, 29)
# ==============================================================================

def test_logical_product_1d():
    """Cell 25: (8):(1) * (4):(1) => size = 32"""
    def build():
        return [fly.size(fly.logical_product(_L((8,), (1,)), _L((4,), (1,))))]
    _build_and_verify("logical_product_1d", build, [32])


def test_blocked_raked_product():
    """Cell 27: (3,6):(6,1) * (4,5):(1,4) => size = 360"""
    def build():
        return [fly.size(fly.blocked_product(_L((3, 6), (6, 1)), _L((4, 5), (1, 4))))]
    _build_and_verify("blocked_raked_product", build, [360])


def test_zipped_tiled_flat_product():
    """Cell 29: flat_product (3,6):(6,1) * (4,5):(1,4) => size = 360"""
    def build():
        return [fly.size(fly.flat_product(_L((3, 6), (6, 1)), _L((4, 5), (1, 4))))]
    _build_and_verify("zipped_tiled_flat_product", build, [360])


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
