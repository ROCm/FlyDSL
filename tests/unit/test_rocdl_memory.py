# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func

pytestmark = pytest.mark.l0_backend_agnostic


def _build_module(build_fn):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                i64 = ir.IntegerType.get_signless(64)
                function = func.FuncOp("test", ir.FunctionType.get([i64], []))
                with ir.InsertionPoint(function.add_entry_block()):
                    build_fn(fx.Int64(function.entry_block.arguments[0]))
                    func.ReturnOp([])
            module.operation.verify()
            return str(module)


def test_ordered_memory_and_sleep_ir():
    def build(address):
        value = fx.rocdl.global_load(
            address,
            fx.Int64,
            alignment=8,
            memory_order=fx.rocdl.MemoryOrder.Acquire,
            syncscope=fx.rocdl.SyncScope.OneAs,
        )
        assert isinstance(value, fx.Int64)
        assert (
            fx.rocdl.global_store(
                address,
                value,
                alignment=8,
                memory_order=fx.rocdl.MemoryOrder.Release,
                syncscope=fx.rocdl.SyncScope.OneAs,
            )
            is None
        )
        assert fx.rocdl.sleep(1) is None

    text = _build_module(build)
    assert "llvm.load" in text and "acquire" in text
    assert "llvm.store" in text and "release" in text
    assert 'syncscope("one-as")' in text
    assert "s_sleep 1" in text


def test_vector_nontemporal_memory_ir():
    def build(address):
        value = fx.rocdl.global_load(
            address,
            fx.Int32,
            vector_width=4,
            alignment=16,
            nontemporal=True,
        )
        assert isinstance(value, fx.Vector)
        assert value.shape == (4,)
        assert value.dtype is fx.Int32
        fx.rocdl.global_store(address, value, alignment=16, nontemporal=True)

    text = _build_module(build)
    assert text.count("nontemporal") == 2
    assert "vector<4xi32>" in text


@pytest.mark.parametrize("vector_width", [0, -1, 1.5])
def test_global_load_rejects_invalid_vector_width(vector_width):
    with pytest.raises(ValueError, match="vector_width"):
        _build_module(lambda address: fx.rocdl.global_load(address, fx.Int32, vector_width=vector_width))


def test_memory_order_validation():
    with pytest.raises(ValueError, match="invalid load memory order"):
        _build_module(
            lambda address: fx.rocdl.global_load(
                address,
                fx.Int32,
                memory_order=fx.rocdl.MemoryOrder.Release,
            )
        )
    with pytest.raises(ValueError, match="syncscope requires"):
        _build_module(
            lambda address: fx.rocdl.global_load(
                address,
                fx.Int32,
                syncscope=fx.rocdl.SyncScope.OneAs,
            )
        )
    with pytest.raises(ValueError, match="invalid store memory order"):
        _build_module(
            lambda address: fx.rocdl.global_store(
                address,
                fx.Int32(0),
                memory_order=fx.rocdl.MemoryOrder.Acquire,
            )
        )


def test_global_load_rejects_raw_mlir_dtype():
    with pytest.raises(TypeError, match="FlyDSL scalar type"):
        _build_module(
            lambda address: fx.rocdl.global_load(
                address,
                ir.IntegerType.get_signless(32),
            )
        )


def test_memory_primitives_are_public_rocdl_exports():
    assert fx.rocdl.global_load.__module__ == "flydsl.expr.rocdl.memory"
    assert fx.rocdl.global_store.__module__ == "flydsl.expr.rocdl.memory"
    assert fx.rocdl.sleep.__module__ == "flydsl.expr.rocdl.memory"
