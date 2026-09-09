# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for the LLVM-dialect DSL wrappers."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func

try:
    import torch
except ImportError:
    torch = None

_HAS_DEVICE = torch is not None and torch.cuda.is_available()


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


class TestLlvmWrapperIR:
    pytestmark = pytest.mark.l0_backend_agnostic

    def test_generic_memory_attributes(self):
        def build(ptr):
            ordered = fx.generic_load(
                ptr,
                dtype=fx.Int64,
                memory_order=fx.AtomicOrdering.Acquire,
                syncscope=fx.rocdl.SyncScope.OneAs,
            )
            fx.generic_store(
                ptr,
                ordered,
                memory_order=fx.AtomicOrdering.Release,
                syncscope=fx.rocdl.SyncScope.OneAs,
            )
            nontemporal = fx.generic_load(ptr, dtype=fx.Int64, nontemporal=True)
            fx.generic_store(ptr, nontemporal, nontemporal=True)
            volatile = fx.generic_load(ptr, dtype=fx.Int64, volatile=True)
            fx.generic_store(ptr, volatile, volatile=True)

        text = _build_pointer_module(build, alignment=16)
        assert text.count("alignment = 16") == 6
        assert "llvm.load" in text and "acquire" in text
        assert "llvm.store" in text and "release" in text
        assert text.count('syncscope("one-as")') == 2
        assert text.count("nontemporal") == 2
        assert text.count(" volatile ") == 2

    def test_generic_load_result_forms(self):
        def build(ptr):
            inferred_scalar = fx.generic_load(ptr)
            inferred_vector = fx.generic_load(ptr, count=4)
            scalar = fx.generic_load(ptr, dtype=fx.Int32)
            scalar_count = fx.generic_load(ptr, dtype=fx.Int32, count=1)
            vector_count = fx.generic_load(ptr, dtype=fx.Int32, count=4)
            vector_dtype = fx.generic_load(ptr, dtype=fx.Int32x4)

            assert isinstance(inferred_scalar, fx.Int32)
            assert isinstance(inferred_vector, fx.Vector)
            assert inferred_vector.dtype is fx.Int32 and inferred_vector.shape == (4,)
            assert isinstance(scalar, fx.Int32)
            assert isinstance(scalar_count, fx.Int32)
            assert isinstance(vector_count, fx.Vector)
            assert vector_count.dtype is fx.Int32 and vector_count.shape == (4,)
            assert isinstance(vector_dtype, fx.Int32x4)

        text = _build_pointer_module(build, dtype=fx.Int32)
        assert text.count("llvm.load") == 6
        assert "-> i32" in text
        assert "-> vector<4xi32>" in text

    @pytest.mark.parametrize(
        "address_space",
        [fx.AddressSpace.Generic, fx.AddressSpace.Global, fx.AddressSpace.Shared],
    )
    def test_generic_memory_address_spaces(self, address_space):
        def build(ptr):
            value = fx.generic_load(ptr)
            fx.generic_store(ptr, value)

        text = _build_pointer_module(build, address_space=address_space)
        assert f"!fly.ptr<i64, {address_space}>" in text
        assert "llvm.load" in text
        assert "llvm.store" in text

    @pytest.mark.parametrize(
        "atomic_op",
        [
            fx.atomic_add,
            fx.atomic_sub,
            fx.atomic_min,
            fx.atomic_max,
            fx.atomic_and,
            fx.atomic_or,
            fx.atomic_xor,
            fx.atomic_xchg,
        ],
    )
    def test_atomic_rmw_uses_fly_pointer_alignment(self, atomic_op):
        text = _build_pointer_module(lambda ptr: atomic_op(ptr, fx.Int64(1)), alignment=32)
        assert "alignment = 32" in text

    @pytest.mark.parametrize("atomic_op", [fx.atomic_fmin, fx.atomic_fmax])
    def test_float_minmax_uses_fly_pointer_alignment(self, atomic_op):
        text = _build_pointer_module(
            lambda ptr: atomic_op(ptr, fx.Float32(1.0), is_positive=True),
            dtype=fx.Float32,
            alignment=16,
        )
        assert "alignment = 16" in text

    def test_atomic_cas_uses_fly_pointer_alignment(self):
        text = _build_pointer_module(
            lambda ptr: fx.atomic_cas(ptr, fx.Int64(0), fx.Int64(1)),
            alignment=64,
        )
        assert "alignment = 64" in text

    def test_public_exports(self):
        assert fx.generic_load.__module__ == "flydsl.expr.llvm"
        assert fx.generic_store.__module__ == "flydsl.expr.llvm"
        assert not hasattr(fx, "global_load")
        assert not hasattr(fx, "global_store")
        assert fx.atomic_add.__module__ == "flydsl.expr.llvm"
        assert fx.memory_fence.__module__ == "flydsl.expr.llvm"
        for removed_name in ("global_load", "global_store", "sleep", "atomic_fetch_add", "memory_fence", "MemoryOrder"):
            assert not hasattr(fx.rocdl, removed_name)


def bare_atomic_add_kernel(
    A: fx.Pointer,
    Out: fx.Pointer,
    block_dim: fx.Constexpr[int],
    syncscope: fx.Constexpr[str],
):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    fx.atomic_add(Out, (A + idx).load(), syncscope=syncscope)


def bare_atomic_add(
    A: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    syncscope: fx.Constexpr[str],
    stream: fx.Stream = fx.Stream(None),
):
    bare_atomic_add_kernel(A, Out, block_dim, syncscope).launch(
        grid=((n + block_dim - 1) // block_dim, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def bare_atomic_ticket_kernel(Counter: fx.Pointer, Out: fx.Pointer, block_dim: fx.Constexpr[int]):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    (Out + idx).store(fx.atomic_add(Counter, fx.Int32(1)))


def bare_atomic_ticket(
    Counter: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    bare_atomic_ticket_kernel(Counter, Out, block_dim).launch(
        grid=((n + block_dim - 1) // block_dim, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def bare_atomic_unsigned_min_kernel(A: fx.Pointer, Out: fx.Pointer, block_dim: fx.Constexpr[int]):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    fx.atomic_min(Out, (A + idx).load(fx.Uint32))


def bare_atomic_unsigned_min(
    A: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    bare_atomic_unsigned_min_kernel(A, Out, block_dim).launch(
        grid=((n + block_dim - 1) // block_dim, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def bare_atomic_fminmax_kernel(
    A: fx.Pointer,
    Out: fx.Pointer,
    block_dim: fx.Constexpr[int],
    take_max: fx.Constexpr[bool],
):
    value = (A + fx.block_idx.x * block_dim + fx.thread_idx.x).load()
    if take_max:
        fx.atomic_fmax(Out, value)
    else:
        fx.atomic_fmin(Out, value)


def bare_atomic_fminmax(
    A: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    take_max: fx.Constexpr[bool],
    stream: fx.Stream = fx.Stream(None),
):
    bare_atomic_fminmax_kernel(A, Out, block_dim, take_max).launch(
        grid=((n + block_dim - 1) // block_dim, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


if _HAS_DEVICE:
    bare_atomic_add_kernel = flyc.kernel(bare_atomic_add_kernel)
    bare_atomic_add = flyc.jit(bare_atomic_add)
    bare_atomic_ticket_kernel = flyc.kernel(bare_atomic_ticket_kernel)
    bare_atomic_ticket = flyc.jit(bare_atomic_ticket)
    bare_atomic_unsigned_min_kernel = flyc.kernel(bare_atomic_unsigned_min_kernel)
    bare_atomic_unsigned_min = flyc.jit(bare_atomic_unsigned_min)
    bare_atomic_fminmax_kernel = flyc.kernel(bare_atomic_fminmax_kernel)
    bare_atomic_fminmax = flyc.jit(bare_atomic_fminmax)


class TestAtomicDevice:
    pytestmark = [
        pytest.mark.l2_device,
        pytest.mark.rocm_lower,
        pytest.mark.skipif(not _HAS_DEVICE, reason="CUDA/ROCm not available"),
    ]

    @pytest.mark.parametrize("syncscope", [fx.SyncScope.System, fx.rocdl.SyncScope.Agent])
    def test_atomic_add(self, syncscope):
        block_dim = 64
        n = block_dim * 4
        values = torch.ones(n, device="cuda", dtype=torch.float32)
        result = torch.zeros(1, device="cuda", dtype=torch.float32)

        stream = torch.cuda.Stream()
        bare_atomic_add(
            flyc.from_c_void_p(fx.Float32, values.data_ptr()),
            flyc.from_c_void_p(fx.Float32, result.data_ptr()),
            n,
            block_dim,
            syncscope,
            stream=stream,
        )
        torch.cuda.synchronize()
        assert result.item() == pytest.approx(float(n), abs=1e-3)

    def test_atomic_add_return(self):
        block_dim = 64
        n = block_dim * 4
        counter = torch.zeros(1, device="cuda", dtype=torch.int32)
        tickets = torch.zeros(n, device="cuda", dtype=torch.int32)

        stream = torch.cuda.Stream()
        bare_atomic_ticket(
            flyc.from_c_void_p(fx.Int32, counter.data_ptr()),
            flyc.from_c_void_p(fx.Int32, tickets.data_ptr()),
            n,
            block_dim,
            stream=stream,
        )
        torch.cuda.synchronize()
        assert counter.item() == n
        assert torch.equal(tickets.cpu().sort().values, torch.arange(n, dtype=torch.int32))

    def test_atomic_unsigned_min(self):
        values = torch.tensor([-1, -(2**31), 7, 23], device="cuda", dtype=torch.int32)
        result = torch.full((1,), -1, device="cuda", dtype=torch.int32)

        stream = torch.cuda.Stream()
        bare_atomic_unsigned_min(
            flyc.from_c_void_p(fx.Uint32, values.data_ptr()),
            flyc.from_c_void_p(fx.Uint32, result.data_ptr()),
            values.numel(),
            values.numel(),
            stream=stream,
        )
        torch.cuda.synchronize()
        assert result.item() == 7

    @pytest.mark.parametrize("take_max", [True, False])
    @pytest.mark.parametrize("sign", ["mixed", "positive", "negative"])
    def test_atomic_float_minmax(self, take_max, sign):
        block_dim = 64
        n = block_dim * 4
        values = (torch.arange(n, dtype=torch.float32) - n // 2) / 8.0
        if sign == "positive":
            values = values.abs() + 1.0
        elif sign == "negative":
            values = -(values.abs() + 1.0)

        device_values = values.cuda()
        initial = float("-inf") if take_max else float("inf")
        result = torch.full((1,), initial, device="cuda", dtype=torch.float32)
        stream = torch.cuda.Stream()
        bare_atomic_fminmax(
            flyc.from_c_void_p(fx.Float32, device_values.data_ptr()),
            flyc.from_c_void_p(fx.Float32, result.data_ptr()),
            n,
            block_dim,
            take_max,
            stream=stream,
        )
        torch.cuda.synchronize()

        expected = values.max().item() if take_max else values.min().item()
        assert result.item() == expected
