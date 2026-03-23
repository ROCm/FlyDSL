#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Static vs dynamic layout types test (mirrors a reference notebook Cell 11).

Ported from legacy dialect to Fly dialect API.
"""

import pytest

from flydsl._mlir.passmanager import PassManager
from flydsl._mlir.ir import (
    Context, Location, Module, InsertionPoint,
    FunctionType, IntegerType, IndexType, BlockArgument,
)
from flydsl._mlir.dialects.fly import IntTupleType
from flydsl._mlir.dialects import fly, arith, func


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
    return fly.static(IntTupleType.get(spec))


def _L(shape_spec, stride_spec):
    return fly.make_layout(_S(shape_spec), stride=_S(stride_spec))


def test_layout_static_types():
    """All-static layout: values become arith.constant after lowering."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("static_layout", FunctionType.get([], [idx] * 5))
                with InsertionPoint(f.add_entry_block()):
                    layout = _L((10, 2), (16, 4))
                    shape = fly.get_shape(layout)
                    stride = fly.get_stride(layout)
                    sz = fly.size(layout)
                    vals = [
                        fly.get_scalar(fly.select(shape, indices=[0])),
                        fly.get_scalar(fly.select(shape, indices=[1])),
                        fly.get_scalar(fly.select(stride, indices=[0])),
                        fly.get_scalar(fly.select(stride, indices=[1])),
                        fly.get_scalar(sz),
                    ]
                    func.ReturnOp([arith.IndexCastOp(idx, v).result for v in vals])

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()

            func_op = list(module.body.operations)[0]
            ret_op = list(func_op.entry_block.operations)[-1]
            actuals = [int(op.owner.attributes["value"]) for op in ret_op.operands]
            assert actuals == [10, 2, 16, 4, 20]


def test_layout_dynamic_types():
    """Dynamic layout: function args remain as block arguments after lowering."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("dynamic_layout", FunctionType.get([i32] * 4, [idx]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    dim0, dim1, stride0, stride1 = entry.arguments
                    import flydsl.expr as fx
                    shape = fx.make_shape(dim0, dim1)
                    stride = fx.make_stride(stride0, stride1)
                    layout = fx.make_layout(shape, stride)
                    sz = fx.size(layout)
                    sc = fx.get_scalar(sz)
                    func.ReturnOp([arith.IndexCastOp(idx, sc).result])

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()


def test_composition_static():
    """Static composition: (10,2):(16,4) o (5,4):(1,5) => size = 20"""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("comp_static", FunctionType.get([], [idx]))
                with InsertionPoint(f.add_entry_block()):
                    A = _L((10, 2), (16, 4))
                    B = _L((5, 4), (1, 5))
                    R = fly.composition(A, B)
                    sz = fly.size(R)
                    sc = fly.get_scalar(sz)
                    func.ReturnOp([arith.IndexCastOp(idx, sc).result])

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()

            func_op = list(module.body.operations)[0]
            ret_op = list(func_op.entry_block.operations)[-1]
            assert int(ret_op.operands[0].owner.attributes["value"]) == 20


def test_mixed_static_dynamic():
    """Mixed layout: some static (fly.static), some dynamic (function args)."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            idx = IndexType.get()
            with InsertionPoint(module.body):
                f = func.FuncOp("mixed_layout", FunctionType.get([i32, i32], [idx]))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    runtime_extent, runtime_stride = entry.arguments
                    c8 = arith.ConstantOp(i32, 8).result
                    c16 = arith.ConstantOp(i32, 16).result
                    import flydsl.expr as fx
                    shape = fx.make_shape(runtime_extent, c8)
                    stride = fx.make_stride(c16, runtime_stride)
                    layout = fx.make_layout(shape, stride)
                    sz = fx.size(layout)
                    sc = fx.get_scalar(sz)
                    func.ReturnOp([arith.IndexCastOp(idx, sc).result])

            pm = PassManager.parse(FLY_PIPELINE, ctx)
            pm.run(module.operation)
            assert module.operation.verify()
