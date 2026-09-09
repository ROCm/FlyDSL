# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func

pytestmark = pytest.mark.l0_backend_agnostic


def _build_pointer_module(build_fn, *, dtype=fx.Int64, alignment=8, address_space=fx.AddressSpace.Global):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ptr_type = fx.PointerType.get(dtype.ir_type, address_space, alignment)
                function = func.FuncOp("test", ir.FunctionType.get([ptr_type], []))
                with ir.InsertionPoint(function.add_entry_block()):
                    build_fn(function.entry_block.arguments[0])
                    func.ReturnOp([])
            module.operation.verify()
            return str(module)


def _build_i64_module(build_fn):
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
            return str(module)


def test_ordered_memory_ir():
    def build(ptr):
        value = fx.global_load(
            ptr,
            fx.Int64,
            memory_order=fx.AtomicOrdering.Acquire,
            syncscope=fx.rocdl.SyncScope.OneAs,
        )
        assert isinstance(value, fx.Int64)
        assert (
            fx.global_store(
                ptr,
                value,
                memory_order=fx.AtomicOrdering.Release,
                syncscope=fx.rocdl.SyncScope.OneAs,
            )
            is None
        )

    text = _build_pointer_module(build, alignment=16)
    assert "llvm.load" in text and "acquire" in text
    assert "llvm.store" in text and "release" in text
    assert text.count("alignment = 16") == 2
    assert 'syncscope("one-as")' in text


def test_nontemporal_memory_ir():
    def build(ptr):
        value = fx.global_load(ptr, fx.Int32, nontemporal=True)
        assert isinstance(value, fx.Int32)
        fx.global_store(ptr, value, nontemporal=True)

    text = _build_pointer_module(build, dtype=fx.Int32, alignment=32)
    assert text.count("nontemporal") == 2
    assert text.count("alignment = 32") == 2


def test_memory_order_validation():
    with pytest.raises(ValueError, match="invalid load memory order"):
        _build_pointer_module(
            lambda ptr: fx.global_load(
                ptr,
                fx.Int64,
                memory_order=fx.AtomicOrdering.Release,
            )
        )
    with pytest.raises(ValueError, match="syncscope requires"):
        _build_pointer_module(
            lambda ptr: fx.global_load(
                ptr,
                fx.Int64,
                syncscope=fx.rocdl.SyncScope.OneAs,
            )
        )
    with pytest.raises(ValueError, match="invalid store memory order"):
        _build_pointer_module(
            lambda ptr: fx.global_store(
                ptr,
                fx.Int64(0),
                memory_order=fx.AtomicOrdering.Acquire,
            )
        )


def test_global_memory_rejects_non_fly_pointer():
    with pytest.raises(TypeError, match="fx.Pointer"):
        _build_i64_module(lambda address: fx.global_load(address, fx.Int64))
    with pytest.raises(TypeError, match="fx.Pointer"):
        _build_i64_module(lambda address: fx.global_store(address, fx.Int64(0)))


def test_global_memory_rejects_non_global_pointer():
    with pytest.raises(ValueError, match="global-address-space"):
        _build_pointer_module(
            lambda ptr: fx.global_load(ptr, fx.Int64),
            address_space=fx.AddressSpace.Shared,
        )


def test_global_load_rejects_raw_mlir_dtype():
    with pytest.raises(TypeError, match="FlyDSL scalar type"):
        _build_pointer_module(lambda ptr: fx.global_load(ptr, ir.IntegerType.get_signless(32)))


def test_memory_primitives_are_public_llvm_exports():
    assert fx.global_load.__module__ == "flydsl.expr.llvm"
    assert fx.global_store.__module__ == "flydsl.expr.llvm"
    assert not hasattr(fx.rocdl, "global_load")
    assert not hasattr(fx.rocdl, "global_store")
    assert not hasattr(fx.rocdl, "sleep")
    assert not hasattr(fx.rocdl, "atomic_fetch_add")
    assert not hasattr(fx.rocdl, "memory_fence")
    assert not hasattr(fx.rocdl, "MemoryOrder")
    assert fx.atomic_add.__module__ == "flydsl.expr.llvm"
    assert fx.memory_fence.__module__ == "flydsl.expr.llvm"
