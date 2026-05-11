#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for Arena / SharedAllocator."""

from contextlib import contextmanager

import pytest

import flydsl.expr as fx
from flydsl._mlir.dialects import func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, Location, Module
from flydsl.compiler.kernel_function import KernelFunction
from flydsl.expr import Arena, SharedAllocator
from flydsl.expr.numeric import Float16, Float32, Int32, Uint8
from flydsl.expr.struct import Align, Array, Storage, _align_up

pytestmark = pytest.mark.l1a_compile_no_target_dialect


def _in_module(build_fn):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("test", FunctionType.get([], []))
                with InsertionPoint(f.add_entry_block()):
                    result = build_fn()
                    func.ReturnOp([])
            return result


@contextmanager
def _kernel_context():
    dummy_kf = object.__new__(KernelFunction)
    dummy_kf._shared_allocator = None
    dummy_kf._kernel_name = "test_kernel"
    dummy_kf._func = lambda: None
    prev = KernelFunction._current
    KernelFunction._current = dummy_kf
    try:
        yield dummy_kf
    finally:
        KernelFunction._current = prev


# =====================================================================
# Arena base class
# =====================================================================


class TestArenaAlignUpAndBump:
    def test_align_up_basic(self):
        assert _align_up(0, 4) == 0
        assert _align_up(1, 4) == 4
        assert _align_up(4, 4) == 4
        assert _align_up(5, 4) == 8
        assert _align_up(1, 16) == 16

    def test_align_up_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            _align_up(0, 0)
        with pytest.raises(ValueError, match="positive"):
            _align_up(0, -1)

    def test_bump_sequence(self):
        arena = Arena(base_alignment=16)
        assert arena._bump(100, 16) == 0
        assert arena._bump(8, 16) == 112
        assert arena.allocated_bytes == 120

    def test_bump_with_alignment_gaps(self):
        arena = Arena(base_alignment=4)
        arena._bump(3, 1)  # offset 0..2, _offset=3
        assert arena.allocated_bytes == 3
        off = arena._bump(4, 4)  # align 3→4, then 4..7
        assert off == 4
        assert arena.allocated_bytes == 8


class TestArenaBasePtr:
    def test_abstract_base_ptr_raises(self):
        arena = Arena()
        with pytest.raises(NotImplementedError):
            _ = arena.base_ptr


class TestArenaAllocatedBytes:
    def test_initial_zero(self):
        arena = Arena()
        assert arena.allocated_bytes == 0

    def test_tracks_cumulative_bumps(self):
        arena = Arena(base_alignment=4)
        arena._bump(10, 4)
        assert arena.allocated_bytes == 10
        arena._bump(20, 4)
        assert arena.allocated_bytes == 32  # align_up(10,4)=12, 12+20=32


# =====================================================================
# SharedAllocator — requires kernel context
# =====================================================================


class TestSharedAllocatorRequiresKernel:
    def test_raises_outside_kernel(self):
        def build():
            with pytest.raises(RuntimeError, match="@kernel"):
                SharedAllocator()

        _in_module(build)

    def test_base_ptr_is_shared(self):
        def build():
            with _kernel_context():
                alloc = SharedAllocator()
                assert alloc.base_ptr.address_space == fx.AddressSpace.Shared

        _in_module(build)


# =====================================================================
# allocate — struct / union types
# =====================================================================


class TestAllocateStruct:
    def test_returns_storage_with_field_access(self):
        def build():
            @fx.struct
            class S:
                a: Array[Float16, 32]
                counter: Int32

            with _kernel_context():
                alloc = SharedAllocator()
                storage = alloc.allocate(S)

            assert isinstance(storage, Storage)
            assert storage._target_type is S
            assert isinstance(storage.a, Storage)
            assert isinstance(storage.counter, Storage)
            assert storage.a._target_type is Array[Float16, 32]
            assert storage.counter._target_type is Int32

        _in_module(build)

    def test_allocated_bytes_matches_struct_size(self):
        def build():
            @fx.struct
            class S:
                x: Int32
                y: Int32

            with _kernel_context():
                alloc = SharedAllocator()
                alloc.allocate(S)
                assert alloc.allocated_bytes == S.__dsl_size_of__()

        _in_module(build)

    def test_constexpr_field_excluded_from_storage(self):
        def build():
            @fx.struct
            class S:
                mode: fx.Constexpr[int]
                value: Float32

            with _kernel_context():
                alloc = SharedAllocator()
                storage = alloc.allocate(S)
                assert alloc.allocated_bytes == Float32.__dsl_size_of__()
            with pytest.raises(AttributeError, match="compile-time only"):
                _ = storage.mode

        _in_module(build)


class TestAllocateUnion:
    def test_overlapping_variants(self):
        def build():
            @fx.union
            class U:
                small: Array[Float16, 8]
                large: Array[Float32, 32]

            with _kernel_context():
                alloc = SharedAllocator()
                storage = alloc.allocate(U)
                assert isinstance(storage, Storage)
                assert storage._target_type is U
                assert alloc.allocated_bytes == 128
            assert storage.small._target_type is Array[Float16, 8]
            assert storage.large._target_type is Array[Float32, 32]

        _in_module(build)

    def test_union_size_is_max_of_fields(self):
        def build():
            @fx.union
            class U:
                a: Int32  # 4 bytes
                b: Array[Float32, 4]  # 16 bytes

            with _kernel_context():
                alloc = SharedAllocator()
                alloc.allocate(U)
                assert alloc.allocated_bytes == 16

        _in_module(build)


class TestAllocateNestedComposite:
    def test_nested_struct_and_inline_union_field(self):
        def build():
            @fx.struct
            class Inner:
                a: Array[Float16, 32]

            @fx.struct
            class Outer:
                inner: Inner
                scratch: fx.Union["f16" : Array[Float16, 32], "f32" : Array[Float32, 32]]  # noqa: F821

            with _kernel_context():
                alloc = SharedAllocator()
                storage = alloc.allocate(Outer)
            assert isinstance(storage.inner, Storage)
            assert storage.inner._target_type is Inner
            assert isinstance(storage.scratch, Storage)
            assert storage.inner.a._target_type is Array[Float16, 32]
            assert storage.scratch.f32._target_type is Array[Float32, 32]

        _in_module(build)


class TestAllocateAlignOverride:
    def test_align_field_in_struct(self):
        def build():
            @fx.struct
            class S:
                head: Int32
                payload: Align[Int32, 16]

            with _kernel_context():
                alloc = SharedAllocator()
                alloc.allocate(S)
            assert S.__dsl_align_of__() == 16
            # head(4) + padding(12) + payload(4) = 20, round up to align 16 → 32
            assert S.__dsl_size_of__() == 32
            assert alloc.allocated_bytes == 32

        _in_module(build)

    def test_explicit_alignment_parameter(self):
        def build():
            with _kernel_context():
                alloc = SharedAllocator()
                alloc.allocate(Int32, alignment=32)
                alloc.allocate(Int32)
                # offset after first: 4, align_up(4, 4)=4, so second at 4, total=8
                assert alloc.allocated_bytes == 8

        _in_module(build)


# =====================================================================
# allocate — raw int bytes
# =====================================================================


class TestAllocateRawBytes:
    def test_returns_storage_of_uint8_array(self):
        def build():
            with _kernel_context():
                alloc = SharedAllocator()
                storage = alloc.allocate(64)
            assert isinstance(storage, Storage)
            assert storage._target_type is Array[Uint8, 64]
            assert alloc.allocated_bytes == 64

        _in_module(build)

    def test_rejects_zero_or_negative(self):
        def build():
            with _kernel_context():
                alloc = SharedAllocator()
                with pytest.raises(ValueError, match="must be > 0"):
                    alloc.allocate(0)
                with pytest.raises(ValueError, match="must be > 0"):
                    alloc.allocate(-1)

        _in_module(build)


# =====================================================================
# Multiple sequential allocations
# =====================================================================


class TestMultipleAllocations:
    def test_sequential_allocations_accumulate(self):
        def build():
            with _kernel_context():
                alloc = SharedAllocator()
                alloc.allocate(Int32)
                alloc.allocate(Int32)
                alloc.allocate(Int32)
                assert alloc.allocated_bytes == 12

        _in_module(build)

    def test_alignment_padding_between_allocations(self):
        def build():
            @fx.struct
            class Small:
                x: Int32  # 4 bytes, align 4

            @fx.struct
            class Big:
                a: Array[Float32, 4]  # 16 bytes, align 4

            with _kernel_context():
                alloc = SharedAllocator()
                alloc.allocate(Small)  # 0..3, _offset=4
                alloc.allocate(Big)  # 4..19, _offset=20
                assert alloc.allocated_bytes == 20

        _in_module(build)

    def test_same_type_multiple_times(self):
        def build():
            @fx.struct
            class S:
                data: Array[Float32, 8]  # 32 bytes

            with _kernel_context():
                alloc = SharedAllocator()
                s1 = alloc.allocate(S)
                s2 = alloc.allocate(S)
                assert alloc.allocated_bytes == 64
            assert s1._target_type is S
            assert s2._target_type is S

        _in_module(build)


# =====================================================================
# Error handling
# =====================================================================


class TestAllocateErrors:
    def test_rejects_non_storable_field(self):
        def build():
            @fx.struct
            class Bad:
                tensor: fx.Tensor

            with _kernel_context():
                alloc = SharedAllocator()
                with pytest.raises(TypeError, match="Storable"):
                    alloc.allocate(Bad)

        _in_module(build)


# =====================================================================
# KernelFunction registration
# =====================================================================


class TestKernelFunctionRegistration:
    def test_registers_and_rejects_duplicate(self):
        def build():
            with _kernel_context() as kf:
                alloc = SharedAllocator()
                assert kf._shared_allocator is alloc
                with pytest.raises(RuntimeError, match="Only one SharedAllocator"):
                    SharedAllocator()

        _in_module(build)
