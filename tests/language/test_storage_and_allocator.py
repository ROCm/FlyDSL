#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Conformance tests for ``docs/language/storage_and_allocator.md``.

Same scope as that spec: ``Storage[T]`` as a typed address (the DSL's ``T*``),
what a `Storage` may point at, the byte layout it navigates, and the allocators
that produce one. Keep the two in sync when either changes.

    Part 1  →  ## fx.Storage[T]: a typed address
    Part 2  →  ## What a Storage can point at
    Part 3  →  ## Byte layout
    Part 4  →  ## Allocators

Declaring the composites themselves is ``test_composite_types.py``.

Frontend cases trace without a GPU; target and device cases exercise the full
compiler and runtime through the matching verification-tier markers.
"""

import importlib
import re

import pytest
from lang_utils import launch_ir, source_ir

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.protocol import (
    Storable,
    construct_from_ir_values,
    dsl_align_of,
    dsl_size_of,
    extract_to_ir_values,
)
from flydsl.expr.struct import _storage_layout

# ###########################################################################
# Shared fixtures & helpers
#   (docs/language/storage_and_allocator.md → types reused across the parts)
# ###########################################################################


@fx.struct
class Pair:
    left: fx.Int32
    right: fx.Float32


@fx.struct
class Inner:
    x: fx.Int32
    y: fx.Int32


@fx.struct
class Outer:
    head: fx.Int32
    inner: Inner
    tail: fx.Float32


@fx.struct
class Padded:
    head: fx.Int32
    payload: fx.Align[fx.Int32, 16]


@fx.struct
class Shared:
    a: fx.Array[fx.Float32, 128, 16]
    b: fx.Array[fx.Float32, 128, 16]


@fx.struct
class Params:
    tile: fx.Constexpr[int]
    scale: fx.Float32


@fx.union
class Scratch:
    fp16: fx.Array[fx.Float16, 128]
    fp32: fx.Array[fx.Float32, 64]


# ── A Storable leaf that records the pointer it is read from / written to ──


class Word:
    width = 4
    poked: list = []

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    @classmethod
    def __dsl_size_of__(cls):
        return cls.width

    @classmethod
    def __dsl_align_of__(cls):
        return cls.width

    @classmethod
    def __peek_from_ptr__(cls, ptr):
        return cls(("peek", ptr))

    @classmethod
    def __poke_into_ptr__(cls, ptr, value):
        cls.poked.append((ptr, value.value))


class Wide(Word):
    width = 8


@pytest.fixture
def symbolic_offsets(monkeypatch):
    """Make ``add_offset`` symbolic, so a peeked/poked address reads as ``(base, offset)``."""
    struct_module = importlib.import_module("flydsl.expr.struct")
    monkeypatch.setattr(struct_module, "add_offset", lambda ptr, offset: (ptr, offset))
    Word.poked.clear()
    yield
    Word.poked.clear()


def _offsets(dsl_type):
    """Field byte offsets of a storable type."""
    return _storage_layout(dsl_type)[2]


def _make_ptr_lines(ir_text):
    return [line.strip() for line in ir_text.splitlines() if "fly.make_ptr" in line]


@pytest.fixture(
    params=[
        pytest.param("gfx942", marks=pytest.mark.l1b_target_dialect, id="compile-gfx942"),
        pytest.param("gfx1100", marks=pytest.mark.l1b_target_dialect, id="compile-gfx1100"),
        pytest.param("device", marks=pytest.mark.l2_device, id="gpu"),
    ]
)
def storage_target(request, monkeypatch):
    """Run each storage scenario through both target compilers and on the GPU."""
    torch = pytest.importorskip("torch")
    on_device = request.param == "device"
    if on_device and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if not on_device:
        monkeypatch.setenv("ARCH", request.param)
    monkeypatch.setenv("COMPILE_ONLY", "0" if on_device else "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    return torch, "cuda" if on_device else "cpu"


# ###########################################################################
# Part 1 — fx.Storage[T]: a typed address
#   (docs/language/storage_and_allocator.md → ## fx.Storage[T]: a typed address)
# ###########################################################################


@pytest.mark.l1a_compile_no_target_dialect
class TestTypedAddress:
    """`Storage[T]` is the DSL's `T*`: one MLIR pointer plus a trace-time `T`."""

    @pytest.mark.parametrize("target", [Pair, Scratch, fx.Array[fx.Int32, 4]])
    def test_an_mlir_pointer_cannot_name_a_composite(self, target):
        """Why the wrapper exists: a pointer's element type must be an MLIR type."""
        with pytest.raises(TypeError):
            fx.PointerType.get(elem_ty=target)

    def test_the_target_type_lives_in_python_not_in_the_pointer(self):
        def body():
            ptr = fx.get_iter(fx.make_rmem_tensor(2, fx.Int32))
            storage = fx.Storage[Pair](ptr)
            field_view = storage.right

            assert type(storage)._target_type is Pair
            assert type(field_view)._target_type is fx.Float32
            # Only the Python-side type differs; the MLIR pointer type is unchanged.
            assert str(object.__getattribute__(field_view, "_ptr").type) == str(ptr.type)

        source_ir(body)


@pytest.mark.l1a_compile_no_target_dialect
class TestStorageView:

    def test_storage_targets_its_type_and_is_cached(self):
        assert fx.Storage[Pair]._target_type is Pair
        assert fx.Storage[Pair] is fx.Storage[Pair]
        assert fx.Storage[fx.Int32].__name__ == "Storage[Int32]"

    def test_unknown_field_is_rejected(self):
        with pytest.raises(AttributeError, match="has no field"):
            fx.Storage[Pair](None).missing

    def test_constexpr_field_has_no_view(self):
        with pytest.raises(AttributeError, match="compile-time only"):
            fx.Storage[Params](None).tile

    def test_peek_reads_each_field_at_its_offset(self, symbolic_offsets):
        @fx.struct
        class Mixed:
            head: Word
            tail: Wide

        value = fx.Storage[Mixed]("base").peek()
        assert value.head == Word(("peek", ("base", 0)))
        assert value.tail == Wide(("peek", ("base", 8)))

    def test_poke_descends_into_a_nested_struct(self, symbolic_offsets):
        @fx.struct
        class Leaves:
            x: Word
            y: Word

        @fx.struct
        class Tree:
            head: Word
            leaves: Leaves
            tail: Word

        fx.Storage[Tree]("base").poke(Tree(head=Word(1), leaves=Leaves(x=Word(2), y=Word(3)), tail=Word(4)))
        assert Word.poked == [
            (("base", 0), 1),
            ((("base", 4), 0), 2),
            ((("base", 4), 4), 3),
            (("base", 12), 4),
        ]

    def test_constexpr_fields_are_skipped_by_both(self, symbolic_offsets):
        @fx.struct
        class Config:
            n: fx.Constexpr[int]
            value: Word

        value = Config(n=32, value=Word(7))
        peeked = fx.Storage[type(value)]("base").peek()
        fx.Storage[Config]("base").poke(value)

        assert peeked.n == 32
        assert peeked.value == Word(("peek", ("base", 0)))
        assert Word.poked == [(("base", 0), 7)]

    def test_peek_and_poke_reach_the_backing_memory(self):
        def body():
            storage = fx.Storage[Pair](fx.get_iter(fx.make_rmem_tensor(2, fx.Int32)))
            storage.poke(Pair(fx.Int32(1), fx.Float32(2.0)))
            value = storage.peek()
            _ = value.left + fx.Int32(1)

        ir_text = source_ir(body)
        assert ir_text.count("fly.ptr.store") == 2
        assert ir_text.count("fly.ptr.load") == 2

    def test_union_has_no_value_form_to_peek(self):
        with pytest.raises(NotImplementedError):
            Scratch.__peek_from_ptr__(None)
        with pytest.raises(NotImplementedError):
            Scratch.__poke_into_ptr__(None, None)

    def test_union_variants_view_the_same_address(self, symbolic_offsets):
        storage = fx.Storage[Scratch]("base")
        assert object.__getattribute__(storage.fp16, "_ptr") == ("base", 0)
        assert object.__getattribute__(storage.fp32, "_ptr") == ("base", 0)


# ###########################################################################
# Part 2 — What a Storage can point at
#   (docs/language/storage_and_allocator.md → ## What a Storage can point at)
# ###########################################################################


@pytest.mark.l1a_compile_no_target_dialect
class TestStorableLeaves:

    @pytest.mark.parametrize(
        "dtype, size",
        [(fx.Int8, 1), (fx.Int32, 4), (fx.Float32, 4), (fx.Int64, 8), (fx.Float64, 8)],
    )
    def test_numeric_leaves(self, dtype, size):
        assert (dsl_size_of(dtype), dsl_align_of(dtype)) == (size, size)

    @pytest.mark.parametrize("dtype", [fx.Boolean, fx.Int4])
    def test_sub_byte_numerics_are_not_storable(self, dtype):
        with pytest.raises(TypeError, match="sub-byte|Storable"):
            dsl_size_of(dtype)

    @pytest.mark.parametrize("dtype", [fx.Vector, fx.Pointer, fx.Tensor])
    def test_device_values_are_not_storable(self, dtype):
        with pytest.raises(TypeError, match="Storable"):
            dsl_size_of(dtype)


@pytest.mark.l1a_compile_no_target_dialect
class TestVectorLeaf:
    @pytest.mark.parametrize(
        "dtype, lanes, size, alignment",
        [
            (fx.Float32, 1, 4, 4),
            (fx.Float32, 3, 12, 4),
            (fx.Float32, 4, 16, 4),
            (fx.Float32, 8, 32, 4),
            (fx.Float16, 8, 16, 2),
            (fx.Float64, 3, 24, 8),
            (fx.Uint8, 5, 5, 1),
            (fx.Int4, 2, 1, 1),
            (fx.Boolean, 8, 1, 1),
            (fx.Float4E2M1FN, 2, 1, 1),
            (fx.Float6E2M3FN, 4, 3, 1),
            (fx.Float32, (2, 3), 24, 4),
            (fx.Uint32, ((2, 2),), 16, 4),
            (fx.Float6E2M3FN, ((2,), (2,)), 3, 1),
        ],
    )
    def test_layout_without_mlir_context(self, dtype, lanes, size, alignment):
        target = fx.Vector[dtype, lanes]
        assert issubclass(target, Storable)
        assert (dsl_size_of(target), dsl_align_of(target)) == (size, alignment)
        assert fx.Storage[target] is fx.Storage[fx.VectorAlias(dtype, lanes)]
        assert not issubclass(fx.Vector, Storable)

    def test_vector_fields_use_element_alignment_and_no_payload_padding(self):
        vector_type = fx.Vector[fx.Float32, 3]
        record = fx.Struct["tag" : fx.Uint8, "value":vector_type, "tail" : fx.Uint8]
        assert _storage_layout(record) == (20, 4, {"tag": 0, "value": 4, "tail": 16})
        assert dsl_size_of(fx.Array[record, 2]) == 40

        aligned = fx.Align[vector_type, 16]
        assert (dsl_size_of(aligned), dsl_align_of(aligned)) == (12, 16)
        record = fx.Struct["tag" : fx.Uint8, "value":aligned, "tail" : fx.Uint8]
        assert _storage_layout(record) == (32, 16, {"tag": 0, "value": 16, "tail": 28})

    @pytest.mark.parametrize("static", [True, False])
    def test_explicit_alignment_is_preserved_on_vector_access(self, static):
        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(1)
            storage = allocator.allocate(fx.Align[fx.Float32x4, 16])
            assert storage._ptr.alignment == 16
            assert allocator.allocated_bytes == 32
            storage.poke(fx.Float32x4(1.0))
            assert type(storage.peek()) is fx.Float32x4

        text = launch_ir(kernel)
        if static:
            assert "allocAlign = 16" in text

    @pytest.mark.parametrize(
        "target",
        [fx.Vector[fx.Index, 4], fx.Vector[fx.Int4, 3], fx.Vector[fx.Boolean, 1], fx.Vector[fx.Int4, ((1, 3),)]],
    )
    def test_unsupported_layout_is_rejected_before_pointer_access(self, target):
        for query in (dsl_size_of, dsl_align_of):
            with pytest.raises(TypeError, match="not Storable"):
                query(target)
        with pytest.raises(TypeError, match="not Storable"):
            target.__peek_from_ptr__(None)
        with pytest.raises(TypeError, match="not Storable"):
            target.__poke_into_ptr__(None, None)

    @pytest.mark.parametrize("static", [True, False])
    def test_allocation_coercion_and_jit_annotation(self, static):
        @flyc.jit
        def carry(value: fx.Vector[fx.Float32, 4]):
            return value

        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(3)
            storage = allocator.allocate(fx.Vector[fx.Float32, 4])
            storage.poke(carry(fx.Float32x4(1.0)) + 2.0)
            loaded = storage.peek()
            assert type(loaded) is fx.Float32x4
            assert loaded.shape == (4,) and loaded.dtype is fx.Float32
            assert allocator.allocated_bytes == 20
            storage.poke([1.0, 2.0, 3.0, 4.0])
            with pytest.raises(TypeError, match="expects Float32x4"):
                storage.poke(fx.Vector[fx.Float32, 2](0.0))
            with pytest.raises(TypeError, match="expects Float32x4"):
                storage.poke(fx.Vector[fx.Int32, 4](0))

        text = launch_ir(kernel)
        assert "fly.ptr.load" in text and "fly.ptr.store" in text
        assert "vector<4xf32>" in text

    @pytest.mark.parametrize("shape", [4, (2, 2), ((2, 2),)])
    @pytest.mark.parametrize("static", [True, False])
    def test_storage_and_explicit_load_type_restore_specialization(self, shape, static):
        target = fx.Vector[fx.Float32, shape]
        expected_shape = (shape,) if isinstance(shape, int) else shape

        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=static)
            storage = allocator.allocate(target)
            value = target(1.0) + 2.0
            assert type(value) is fx.Vector
            with pytest.raises(TypeError, match="Storable"):
                allocator.allocate(type(value))
            storage.poke(value)
            assert type(value) is fx.Vector
            loaded = storage.peek()
            assert type(loaded) is target and loaded.shape == expected_shape
            loaded = fx.generic_load(storage._ptr, dtype=target)
            assert type(loaded) is target and loaded.shape == expected_shape
            assert type(fx.generic_load(storage._ptr, dtype=fx.Float32, count=4)) is fx.Vector

        launch_ir(kernel)


@pytest.mark.rocm_lower
class TestVectorStorage:
    @pytest.mark.l1b_target_dialect
    @pytest.mark.parametrize("static", [True, False], ids=["static", "dynamic"])
    def test_explicit_alignment_enables_ds_read_b128(self, monkeypatch, tmp_path, static):
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
        monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
        monkeypatch.setenv("ARCH", "gfx942")
        monkeypatch.setenv("COMPILE_ONLY", "1")
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
        monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))

        @flyc.kernel
        def kernel(src: fx.Tensor, dst: fx.Tensor):
            tid = fx.thread_idx.x
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(3)
            value = allocator.allocate(fx.Align[fx.Float32x4, 16])
            if tid == 0:
                fx.Storage[fx.Float32x4](value._ptr).poke(fx.generic_load(src.iter, dtype=fx.Float32x4))
            fx.barrier()
            # Read another thread's data and write it globally to retain the LDS read.
            fx.generic_store(dst.iter + tid * 4, value.peek())

        @flyc.jit
        def launch(src: fx.Tensor, dst: fx.Tensor):
            kernel(src, dst).launch(grid=1, block=64)

        # CPU tensors suffice for target compilation; this does not launch on a GPU.
        launch(torch.empty(256, dtype=torch.float32), torch.empty(256, dtype=torch.float32))
        files = list(tmp_path.glob("*/*_final_isa.s"))
        assert len(files) == 1
        isa = files[0].read_text()
        assert re.search(r"^\s*ds_read_b128\s", isa, re.MULTILINE), isa

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize(
        "dtype, shape, numel",
        [
            (fx.Float32, 3, 3),
            (fx.Uint32, 4, 4),
            (fx.Float8E4M3FN, 4, 4),
            (fx.Float4E2M1FN, 2, 2),
            (fx.Float6E2M3FN, 4, 4),
            (fx.Int4, 2, 2),
            (fx.Float32, (2, 3), 6),
            (fx.Uint32, ((2, 2),), 4),
            (fx.Float4E2M1FN, (1, 2), 2),
            (fx.Float6E2M3FN, ((2,), (2,)), 4),
        ],
    )
    def test_shared_and_struct_array_preserve_payload_bytes(self, storage_target, static, dtype, shape, numel):
        torch, device = storage_target
        vector_type = fx.Vector[dtype, shape]
        expected_shape = (shape,) if isinstance(shape, int) else shape
        payload = dtype.width * numel // 8
        # Keep a guard after each payload, even for types with no trailing padding.
        stride = dsl_size_of(vector_type) + dsl_align_of(vector_type)
        block_size = 64

        @fx.struct
        class Item:
            tag: fx.Int32
            value: vector_type

        @flyc.kernel
        def kernel(raw: fx.Tensor, out: fx.Tensor, tags: fx.Tensor):
            tid = fx.thread_idx.x
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(3)
            direct = allocator.allocate(vector_type)
            items = allocator.allocate(fx.Array[Item, block_size]).peek()
            value = fx.Storage[vector_type](raw.iter + tid * stride).peek()
            assert type(value) is vector_type and value.shape == expected_shape
            items[tid] = Item(tid, value)
            if tid == 0:
                fx.Storage[vector_type](direct._ptr).poke(value)
            fx.barrier()
            peer = items[block_size - 1 - tid]
            assert type(peer.value) is vector_type and peer.value.shape == expected_shape
            fx.Storage[vector_type](out.iter + tid * stride).poke(peer.value)
            tags[tid] = peer.tag
            if tid == block_size - 1:
                loaded = fx.Storage[vector_type](direct._ptr).peek()
                fx.Storage[vector_type](out.iter + block_size * stride).poke(loaded)

        @flyc.jit
        def launch(raw: fx.Tensor, out: fx.Tensor, tags: fx.Tensor):
            kernel(raw, out, tags).launch(grid=1, block=block_size)

        raw = torch.arange(block_size * stride, dtype=torch.int32, device=device).to(torch.uint8)
        out = torch.full(((block_size + 1) * stride,), 0x5A, dtype=torch.uint8, device=device)
        tags = torch.empty(block_size, dtype=torch.int32, device=device)
        launch(raw, out, tags)
        assert launch._last_compiled is not None
        if device == "cpu":
            return

        # A vector store must leave the guard bytes and adjacent payloads intact.
        expected = torch.full_like(out.cpu(), 0x5A).reshape(block_size + 1, stride)
        source = raw.cpu().reshape(block_size, stride)
        expected[:block_size, :payload] = source.flip(0)[:, :payload]
        expected[block_size, :payload] = source[0, :payload]
        torch.testing.assert_close(out.cpu().reshape_as(expected), expected, rtol=0, atol=0)
        torch.testing.assert_close(tags.cpu(), torch.arange(block_size - 1, -1, -1, dtype=torch.int32))


@pytest.mark.l1a_compile_no_target_dialect
class TestPointerLeaf:
    @pytest.mark.parametrize("space", [fx.AddressSpace.Global, fx.AddressSpace.Shared])
    @pytest.mark.parametrize(
        "dtype, elem_bytes",
        [
            (fx.Boolean, 1),
            (fx.Int4, 1),
            (fx.Float6E2M3FN, 1),
            (fx.Uint8, 1),
            (fx.Float16, 2),
            (fx.Float32, 4),
            (fx.Float64, 8),
        ],
    )
    def test_default_alignment_and_type_names(self, space, dtype, elem_bytes):
        target = fx.Pointer[dtype, space]
        assert target is fx.Pointer[dtype, space, elem_bytes] is fx.Pointer[dtype, space, None]
        name = f"Pointer[{dtype.__name__}, {space}]"
        assert target.__name__ == name
        assert str(target) == f"<class 'flydsl.expr.typing.{name}'>"
        stronger = fx.Pointer[dtype, space, elem_bytes * 2]
        assert stronger is not target
        assert stronger.__name__ == f"Pointer[{dtype.__name__}, {space}, {elem_bytes * 2}]"
        with ir.Context(), ir.Location.unknown():
            assert target.ir_type.alignment == elem_bytes
            assert target.ir_type.swizzle.mask == 0

    @pytest.mark.parametrize("space, size", [(fx.AddressSpace.Global, 8), (fx.AddressSpace.Shared, 4)])
    @pytest.mark.parametrize("alignment", [4, 16, 128])
    def test_layout_and_identity_without_mlir_context(self, space, size, alignment):
        target = fx.Pointer[fx.Float32, space, alignment]
        assert issubclass(target, fx.Pointer) and issubclass(target, Storable)
        assert (dsl_size_of(target), dsl_align_of(target)) == (size, size)
        assert target is fx.Pointer[fx.Float32, int(space), alignment]
        assert target is not fx.Pointer[fx.Int32, space, alignment]
        assert not issubclass(fx.Pointer, Storable)
        record = fx.Struct["tag" : fx.Uint8, "ptr":target]
        assert _storage_layout(record) == (2 * size, size, {"tag": 0, "ptr": size})

    @pytest.mark.parametrize("params", [fx.Float32, (), (fx.Float32,), (fx.Float32, fx.AddressSpace.Global, 4, None)])
    def test_subscription_requires_two_or_three_parameters(self, params):
        with pytest.raises(TypeError, match="Pointer expects"):
            fx.Pointer[params]

    @pytest.mark.parametrize("dtype", [None, float, fx.Numeric, fx.Index, fx.Vector, [fx.Float32]])
    def test_invalid_element_type(self, dtype):
        with pytest.raises(TypeError, match="fixed-width Numeric"):
            fx.Pointer[dtype, fx.AddressSpace.Global, 4]

    @pytest.mark.parametrize("space", [fx.AddressSpace.Register, fx.AddressSpace.Generic, None, True])
    def test_unsupported_address_space(self, space):
        with pytest.raises(TypeError, match="Global.*Shared"):
            fx.Pointer[fx.Float32, space, 4]

    @pytest.mark.parametrize("alignment", [True, 4.0, 0, -4, 2, 6, 2**31])
    def test_invalid_alignment(self, alignment):
        with pytest.raises((TypeError, ValueError), match="alignment"):
            fx.Pointer[fx.Float32, fx.AddressSpace.Global, alignment]

    def test_cached_class_rebuilds_ir_type_in_current_context(self):
        target = fx.Pointer[fx.Uint32, fx.AddressSpace.Shared]
        with ir.Context() as first, ir.Location.unknown():
            first_type = target.ir_type
            assert isinstance(first_type, fx.PointerType) and first_type.context is first
            assert first_type.element_type == fx.Uint32.ir_type
            assert first_type.alignment == 4
            assert target is fx.Pointer[fx.Uint32, first_type.address_space, 4]
        with ir.Context() as second, ir.Location.unknown():
            assert target.ir_type.context is second and target.ir_type != first_type
            assert target is fx.Pointer[fx.Uint32, fx.AddressSpace.Shared]
        with pytest.raises(TypeError, match="already specialized"):
            target[fx.Int32, fx.AddressSpace.Global]

    def test_typed_construction_coercion_and_pointer_arithmetic(self):
        strong = fx.Pointer[fx.Uint32, fx.AddressSpace.Global, 16]
        weak = fx.Pointer[fx.Uint32, fx.AddressSpace.Global]

        def body(offset: fx.Int32):
            raw = fx.inttoptr(strong.ir_type, fx.Uint64(0x8000000000001000))
            value = strong(raw)
            assert type(value) is strong and value.dtype is fx.Uint32
            assert strong.__coerce__(value) is value
            reduced = weak(value)
            assert type(reduced) is weak and reduced.type == weak.ir_type
            for result, alignment in (
                (value + 1, 4),
                (1 + value, 4),
                (value - 1, 4),
                (value + offset, 4),
                (value - offset, 4),
                (value + 0, 16),
                (value + 4, 16),
            ):
                assert type(result) is fx.Pointer
                assert result.alignment == alignment
                assert not isinstance(result, Storable)
                assert type(weak(result)) is weak
            assert type(strong(value + 4)) is strong
            assert type(reduced.load()) is fx.Uint32
            rebuilt = construct_from_ir_values(strong, value, extract_to_ir_values(value))
            assert type(rebuilt) is strong and rebuilt.type == strong.ir_type
            with pytest.raises(TypeError, match="requires alignment"):
                strong(reduced)
            with pytest.raises(TypeError, match="requires alignment"):
                strong(value + 1)
            with pytest.raises(TypeError, match="expects a Pointer"):
                strong(0)

        source_ir(body, 1)

    def test_storage_rejects_incompatible_pointers(self):
        target = fx.Pointer[fx.Float32, fx.AddressSpace.Global, 16]

        @flyc.kernel
        def kernel():
            storage = fx.SharedAllocator().allocate(target)
            for wrong in (
                fx.Pointer[fx.Int32, fx.AddressSpace.Global, 16],
                fx.Pointer[fx.Float32, fx.AddressSpace.Shared, 16],
                fx.Pointer[fx.Float32, fx.AddressSpace.Global, 4],
                fx.Pointer[fx.Float32, fx.AddressSpace.Global, 24],
            ):
                address = fx.Uint32(0) if wrong.ir_type.address_space == fx.AddressSpace.Shared else fx.Uint64(0)
                value = fx.inttoptr(wrong.ir_type, address)
                with pytest.raises(TypeError, match="expects matching|requires alignment"):
                    storage.poke(value)

        launch_ir(kernel)

    def test_swizzled_pointer_is_rejected_before_storage(self):
        target = fx.Pointer[fx.Float32, fx.AddressSpace.Shared]

        @flyc.kernel
        def kernel():
            storage = fx.SharedAllocator().allocate(target)
            raw = fx.inttoptr(target.ir_type, fx.Uint32(128))
            swizzled = fx.apply_swizzle(raw, fx.static(fx.SwizzleType.get(2, 2, 2)))
            with pytest.raises(TypeError, match="does not support swizzled pointers"):
                target(swizzled)
            with pytest.raises(TypeError, match="does not support swizzled pointers"):
                storage.poke(swizzled)

        launch_ir(kernel)

    @pytest.mark.parametrize("static", [True, False])
    def test_storage_preserves_type_through_struct_and_kernel(self, static):
        target = fx.Pointer[fx.Uint32, fx.AddressSpace.Shared]
        record = fx.Struct["ptr":target]

        @flyc.kernel
        def consume(value: fx.Pointer, item: record):
            assert type(value) is target and value.type == target.ir_type
            assert type(item.ptr) is target and item.ptr.dtype is fx.Uint32
            for _ in range(fx.Int32(2)):
                value = value + 1
                assert type(value) is fx.Pointer
            assert type(value) is target and value.type == target.ir_type

        def body():
            value = target(fx.inttoptr(target.ir_type, fx.Uint32(128)))
            consume(value, record(value)).launch(grid=1, block=64)

        source_ir(body)

        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(1)
            storage = allocator.allocate(fx.Align[target, 16])
            original = target(fx.inttoptr(target.ir_type, fx.Uint32(128)))
            storage.poke(original)
            loaded = storage.peek()
            assert type(loaded) is target and loaded.type == target.ir_type
            assert storage._ptr.alignment == 16 and loaded.alignment == 4
            assert loaded.dtype is fx.Uint32
            assert type(fx.ptrtoint(loaded).to(fx.Uint32).to(fx.Int64)) is fx.Int64
            assert allocator.allocated_bytes == 20

        text = launch_ir(kernel)
        assert "fly.ptrtoint" in text and "fly.inttoptr" in text
        assert "fly.ptr.store" in text and "fly.ptr.load" in text
        assert "arith.extui" in text and "arith.extsi" not in text


@pytest.mark.rocm_lower
class TestPointerStorage:
    @pytest.mark.parametrize("static", [True, False])
    def test_address_high_bits_are_preserved(self, storage_target, static):
        torch, device = storage_target
        global_type = fx.Pointer[fx.Uint8, fx.AddressSpace.Global]
        shared_type = fx.Pointer[fx.Uint8, fx.AddressSpace.Shared]
        record = fx.Struct["global_ptr":global_type, "shared_ptr":shared_type]

        @flyc.kernel
        def kernel(raw: fx.Tensor, out: fx.Tensor):
            allocator = fx.SharedAllocator(static=static)
            slots = allocator.allocate(fx.Array[record, 64]).peek()
            tid = fx.thread_idx.x
            global_bits = raw[tid * 2].to(fx.Uint64)
            shared_bits = raw[tid * 2 + 1].to(fx.Uint32)
            slots[tid] = record(
                fx.inttoptr(global_type.ir_type, global_bits), fx.inttoptr(shared_type.ir_type, shared_bits)
            )
            fx.barrier()
            peer = slots[63 - tid]
            # Synthetic addresses are compared as bits and never dereferenced.
            out[tid * 2] = fx.ptrtoint(peer.global_ptr).to(fx.Uint64).to(fx.Int64)
            out[tid * 2 + 1] = fx.ptrtoint(peer.shared_ptr).to(fx.Uint32).to(fx.Int64)

        @flyc.jit
        def launch(raw: fx.Tensor, out: fx.Tensor):
            kernel(raw, out).launch(grid=1, block=64)

        raw = torch.tensor([-(2**63) + 0x1000, 0x80001000], dtype=torch.int64, device=device).repeat(64)
        raw += torch.arange(64, device=device).repeat_interleave(2)
        out = torch.empty_like(raw)
        launch(raw, out)
        if device != "cpu":
            torch.testing.assert_close(out.cpu().reshape(64, 2), raw.cpu().reshape(64, 2).flip(0), rtol=0, atol=0)

    @pytest.mark.parametrize("static", [True, False])
    def test_global_and_shared_pointer_round_trip(self, storage_target, static):
        torch, device = storage_target
        global_type = fx.Pointer[fx.Int32, fx.AddressSpace.Global]
        shared_type = fx.Pointer[fx.Int32, fx.AddressSpace.Shared]
        record = fx.Struct["global_ptr":global_type, "shared_ptr":shared_type]

        @flyc.kernel
        def kernel(src: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            tid = fx.thread_idx.x
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(1)
            values = allocator.allocate(fx.Array[fx.Int32, 64, 64]).peek()
            global_ptr = src.iter + tid
            shared_ptr = values.ptr + tid
            global_ptr = global_type(global_ptr)
            shared_ptr = shared_type(shared_ptr)
            shared_ptr[0] = tid * 7
            direct = allocator.allocate(global_type)
            if tid == 0:
                fx.Storage[global_type](direct._ptr).poke(global_ptr)
            items = allocator.allocate(fx.Array[record, 64]).peek()
            items[tid] = record(global_ptr, shared_ptr)
            fx.barrier()
            peer = items[63 - tid]
            assert type(peer.global_ptr) is global_type and peer.global_ptr.type == global_type.ir_type
            assert type(peer.shared_ptr) is shared_type and peer.shared_ptr.type == shared_type.ir_type
            out[tid] = peer.global_ptr[0]
            out[64 + tid] = peer.shared_ptr[0]
            out[128 + tid] = direct.peek()[0]
            addresses[tid] = fx.ptrtoint(peer.global_ptr).to(fx.Int64)
            addresses[64 + tid] = (fx.ptrtoint(peer.shared_ptr) == fx.ptrtoint(values.ptr + 63 - tid)).to(fx.Int64)

        @flyc.jit
        def launch(src: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            kernel(src, out, addresses).launch(grid=1, block=64)

        src = torch.arange(64, dtype=torch.int32, device=device) * 11
        out = torch.empty(192, dtype=torch.int32, device=device)
        addresses = torch.empty(128, dtype=torch.int64, device=device)
        launch(src, out, addresses)
        if device == "cpu":
            return
        peer = torch.arange(63, -1, -1)
        expected = torch.cat([src.cpu()[peer], (peer * 7).to(torch.int32), src.cpu()[0].expand(64)])
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(addresses[:64].cpu(), src.data_ptr() + peer * 4)
        assert torch.all(addresses[64:].cpu() == 1)


@pytest.mark.l1a_compile_no_target_dialect
class TestArrayLeaf:

    def test_size_and_alignment(self):
        Tile = fx.Array[fx.Float32, 32, 16]
        assert (Tile.size, Tile.align) == (32, 16)
        assert (dsl_size_of(Tile), dsl_align_of(Tile)) == (128, 16)

    @pytest.mark.parametrize(
        "dtype, align",
        [(fx.Float32, 4), (fx.Float16, 2), (fx.Uint8, 1), (fx.Int64, 8)],
    )
    def test_alignment_defaults_to_the_element_byte_size(self, dtype, align):
        assert fx.Array[dtype, 32].align == align
        assert dsl_align_of(fx.Array[dtype, 32]) == align

    def test_array_types_are_cached(self):
        assert fx.Array[fx.Int32, 64] is fx.Array[fx.Int32, 64]
        assert fx.Array[fx.Int32, 64] is not fx.Array[fx.Int32, 64, 8]

    def test_indexing_and_view(self):
        def body():
            arr = fx.Array[fx.Float32, 8].__peek_from_ptr__(fx.get_iter(fx.make_rmem_tensor(8, fx.Float32)))
            arr[0] = fx.Float32(1.0)
            _ = arr[1]
            _ = arr.view(fx.make_layout(8, 1))

        ir_text = source_ir(body)
        assert "fly.ptr.store" in ir_text
        assert "fly.ptr.load" in ir_text
        assert "fly.make_view" in ir_text

    def test_whole_array_poke_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            fx.Array[fx.Int32, 64].__poke_into_ptr__(None, None)

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: fx.Array[object, 4], "Numeric subclass"),
            (lambda: fx.Array[fx.Int32, 0], "positive integer"),
            (lambda: fx.Array[fx.Int32, -1], "positive integer"),
            (lambda: fx.Array[fx.Int32, 4, 0], "positive integer"),
            (lambda: fx.Array[fx.Int32], r"Array\[dtype, size\]"),
        ],
    )
    def test_parameter_errors(self, make, match):
        with pytest.raises(TypeError, match=match):
            make()


@pytest.mark.l1a_compile_no_target_dialect
class TestArrayStruct:

    @pytest.mark.parametrize("dtype, stride, align", [(Pair, 8, 4), (Outer, 16, 4), (Padded, 32, 16)])
    def test_layout_includes_element_padding(self, dtype, stride, align):
        Array = fx.Array[dtype, 3]
        assert (dsl_size_of(Array), dsl_align_of(Array)) == (3 * stride, align)
        assert Array is fx.Array[dtype, 3, align]
        assert (Array.dtype, Array.size) == (dtype, 3)

    def test_array_alignment_does_not_change_element_stride(self):
        Array = fx.Array[Pair, 3, 32]
        assert (dsl_size_of(Array), dsl_align_of(Array)) == (24, 32)

    @pytest.mark.parametrize("align", [4, 8, 24])
    def test_alignment_must_preserve_element_alignment(self, align):
        with pytest.raises(ValueError, match="multiple of the element alignment 16"):
            fx.Array[Padded, 3, align]

    @pytest.mark.parametrize("align", [12, 24])
    def test_alignment_must_be_a_power_of_two(self, align):
        with pytest.raises(ValueError, match="power of two"):
            fx.Array[Pair, 3, align]

    @pytest.mark.parametrize("field_type", [fx.Vector, fx.Pointer, fx.Boolean])
    def test_non_storable_fields_are_rejected(self, field_type):
        Item = fx.Struct["value":field_type]
        with pytest.raises(TypeError, match="field 'value'.*Storable"):
            fx.Array[Item, 4]

    def test_union_elements_are_rejected(self):
        with pytest.raises(TypeError, match="storable Struct"):
            fx.Array[Scratch, 4]

    def test_nested_field_addresses_use_aos_stride(self, ctx, symbolic_offsets, monkeypatch):
        typing_module = importlib.import_module("flydsl.expr.typing")
        monkeypatch.setattr(typing_module, "recast_iter", lambda dtype, ptr: ptr)
        monkeypatch.setattr(typing_module, "add_offset", lambda ptr, offset: (ptr, offset))
        Fields = fx.Struct["small":Word, "wide":Wide]
        Item = fx.Struct["tag":Word, "fields":Fields]
        items = fx.Array[Item, 4].__peek_from_ptr__("base")

        # Item has offsets 0, 8, 16, size 24 and alignment 8.
        value = items[2]
        assert value.tag == Word(("peek", (("base", 48), 0)))
        assert value.fields.small == Word(("peek", ((("base", 48), 8), 0)))
        assert value.fields.wide == Wide(("peek", ((("base", 48), 8), 8)))
        items[2] = Item(Word(7), Fields(Word(11), Wide(13)))
        assert Word.poked == [
            ((("base", 48), 0), 7),
            (((("base", 48), 8), 0), 11),
            (((("base", 48), 8), 8), 13),
        ]

    def test_dynamic_indexing_and_ir_round_trip(self):
        def body(index: fx.Int32):
            # Start with an i32 pointer: struct indexing must first make it a
            # byte pointer rather than multiplying the byte stride by four.
            ptr = fx.get_iter(fx.make_rmem_tensor(16, fx.Int32))
            Array = fx.Array[Outer, 4]
            items = Array.__peek_from_ptr__(ptr)
            assert items.ptr.type.element_type == fx.Uint8.ir_type
            assert items._element_ptr(index).alignment == 4
            items[index] = Outer(1, Inner(2, 3), 4.0)
            flat = extract_to_ir_values(items)
            assert len(flat) == 1
            rebuilt = construct_from_ir_values(Array, items, flat)
            assert rebuilt.dtype is Outer and rebuilt.size == 4
            assert "Array[Outer, 4]" in repr(rebuilt)
            value = rebuilt[index]
            assert isinstance(value, Outer) and isinstance(value.inner, Inner)
            assert isinstance(value.tail, fx.Float32)

        ir_text = source_ir(body, 2)
        assert ir_text.count("fly.ptr.store") == 4
        assert ir_text.count("fly.ptr.load") == 4
        assert "fly.int_tuple_mul" in ir_text

    def test_mismatched_assignment_and_tensor_view_are_rejected(self):
        def body():
            ptr = fx.get_iter(fx.make_rmem_tensor(4, fx.Int32))
            items = fx.Array[Pair, 2].__peek_from_ptr__(ptr)
            with pytest.raises(TypeError, match="expects Pair value"):
                items[0] = Inner(1, 2)
            with pytest.raises(TypeError, match="requires Numeric elements"):
                items.view(fx.make_layout(2, 1))

        source_ir(body)

    @pytest.mark.parametrize("static", [True, False])
    def test_shared_allocation_is_contiguous_even_inside_a_struct(self, static):
        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=static)
            ScratchArray = fx.Struct["items" : fx.Array[Padded, 3]]
            items = allocator.allocate(ScratchArray).peek().items
            assert allocator.allocated_bytes == 96
            assert items.ptr.address_space == fx.AddressSpace.Shared

        ir_text = launch_ir(kernel)
        if static:
            make_ptrs = _make_ptr_lines(ir_text)
            assert len(make_ptrs) == 1
            assert "allocBytes = 96" in make_ptrs[0]
            assert "allocAlign = 16" in make_ptrs[0]
        else:
            assert "dynamic_shared_memory_size %c96_i32" in ir_text


# ── fx.Align[T, A] ──────────────────────────────────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
class TestAlignModifier:
    """`Align` only overrides alignment; it delegates size and access to `T`."""

    def test_size_is_unchanged_and_alignment_is_raised(self):
        Aligned = fx.Align[fx.Int32, 16]
        assert Aligned.dtype is fx.Int32
        assert (dsl_size_of(Aligned), dsl_align_of(Aligned)) == (4, 16)

    def test_access_is_delegated_to_the_inner_type(self, symbolic_offsets):
        Aligned = fx.Align[Word, 16]
        assert Aligned.__peek_from_ptr__("base") == Word(("peek", "base"))
        Aligned.__poke_into_ptr__("base", Word(7))
        assert Word.poked == [("base", 7)]

    @pytest.mark.parametrize(
        "align, match",
        [
            (0, "positive"),
            (-1, "positive"),
            (3, "power of two"),
            (24, "power of two"),
            (2, "smaller than natural alignment"),
        ],
    )
    def test_alignment_value_errors(self, align, match):
        with pytest.raises(ValueError, match=match):
            fx.Align[fx.Int32, align]

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: fx.Align[fx.Int32, 1.0], "must be an int"),
            (lambda: fx.Align[fx.Int32, True], "must be an int"),
            (lambda: fx.Align[fx.Int32], r"Align\[Type, N\]"),
        ],
    )
    def test_alignment_type_errors(self, make, match):
        with pytest.raises(TypeError, match=match):
            make()


# ###########################################################################
# Part 3 — Byte layout
#   (docs/language/storage_and_allocator.md → ## Byte layout)
# ###########################################################################


@pytest.mark.l1a_compile_no_target_dialect
class TestProductLayout:
    """Sequential placement, per-field alignment, trailing padding."""

    def test_sequential_fields(self):
        assert _offsets(Pair) == {"left": 0, "right": 4}
        assert (dsl_size_of(Pair), dsl_align_of(Pair)) == (8, 4)

    def test_alignment_gap_and_trailing_padding(self):
        assert _offsets(Padded) == {"head": 0, "payload": 16}
        assert (dsl_size_of(Padded), dsl_align_of(Padded)) == (32, 16)

    def test_nested_struct_layout_is_recursive(self):
        assert _offsets(Outer) == {"head": 0, "inner": 4, "tail": 12}
        assert (dsl_size_of(Outer), dsl_align_of(Outer)) == (16, 4)

    def test_array_fields_use_their_element_alignment_by_default(self):
        @fx.struct
        class Tiles:
            a: fx.Array[fx.Float32, 32]
            b: fx.Array[fx.Float32, 32]

        assert _offsets(Tiles) == {"a": 0, "b": 128}
        assert (dsl_size_of(Tiles), dsl_align_of(Tiles)) == (256, 4)

    def test_array_fields_carry_an_explicit_alignment(self):
        assert _offsets(Shared) == {"a": 0, "b": 512}
        assert (dsl_size_of(Shared), dsl_align_of(Shared)) == (1024, 16)

    def test_constexpr_fields_have_no_offset(self):
        assert dsl_size_of(Params) == 4
        assert "tile" not in _offsets(Params)


@pytest.mark.l1a_compile_no_target_dialect
class TestUnionLayout:
    """Every field at offset zero; max size and max alignment."""

    def test_offsets_are_all_zero(self):
        assert _offsets(Scratch) == {"fp16": 0, "fp32": 0}

    def test_size_is_the_maximum_field_size(self):
        assert (dsl_size_of(Scratch), dsl_align_of(Scratch)) == (256, 4)

    def test_size_is_rounded_up_to_the_maximum_alignment(self):
        @fx.union
        class Mixed:
            small: fx.Array[fx.Uint8, 6]
            wide: fx.Align[fx.Int32, 16]

        assert (dsl_size_of(Mixed), dsl_align_of(Mixed)) == (16, 16)

    def test_union_inside_a_struct(self):
        @fx.struct
        class WithUnion:
            head: fx.Int32
            scratch: Scratch

        assert _offsets(WithUnion) == {"head": 0, "scratch": 4}
        assert dsl_size_of(WithUnion) == 260

    def test_struct_inside_a_union(self):
        @fx.union
        class WithStruct:
            pair: Inner
            single: fx.Float32

        assert _offsets(WithStruct) == {"pair": 0, "single": 0}
        assert (dsl_size_of(WithStruct), dsl_align_of(WithStruct)) == (8, 4)


# ###########################################################################
# Part 4 — Allocators
#   (docs/language/storage_and_allocator.md → ## Allocators)
# ###########################################################################


# ── fx.Arena — the target-neutral bump allocator ────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
class TestArena:

    def test_starts_empty(self):
        assert fx.Arena().allocated_bytes == 0

    def test_base_pointer_is_supplied_by_a_subclass(self):
        with pytest.raises(NotImplementedError):
            fx.Arena().base_ptr


# ── fx.SharedAllocator — the LDS allocator ──────────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
class TestAllocate:

    def test_allocate_returns_a_storage_tree(self):
        @flyc.kernel
        def tree_kernel():
            @fx.struct
            class Nested:
                inner: Inner
                scratch: Scratch

            storage = fx.SharedAllocator().allocate(Nested)
            assert storage._target_type is Nested
            assert storage.inner._target_type is Inner
            assert storage.inner.x._target_type is fx.Int32
            assert storage.scratch.fp32._target_type is fx.Array[fx.Float32, 64]

        launch_ir(tree_kernel)

    def test_allocated_bytes_follows_the_logical_layout(self):
        @flyc.kernel
        def bytes_kernel():
            allocator = fx.SharedAllocator()
            allocator.allocate(Shared)
            assert allocator.allocated_bytes == 1024
            allocator.allocate(Scratch)
            assert allocator.allocated_bytes == 1280

        launch_ir(bytes_kernel)

    def test_sequential_allocations_pad_to_alignment(self):
        @flyc.kernel
        def padding_kernel():
            allocator = fx.SharedAllocator()
            allocator.allocate(fx.Int32)  # 0..3
            allocator.allocate(Padded)  # aligned to 16 → 16..47
            assert allocator.allocated_bytes == 48

        launch_ir(padding_kernel)

    def test_explicit_alignment_raises_only_that_allocation(self):
        @flyc.kernel
        def alignment_kernel():
            allocator = fx.SharedAllocator()
            allocator.allocate(fx.Int32, alignment=32)
            allocator.allocate(fx.Int32)
            assert allocator.allocated_bytes == 8

        launch_ir(alignment_kernel)

    def test_constexpr_field_is_not_allocated(self):
        @flyc.kernel
        def constexpr_kernel():
            allocator = fx.SharedAllocator()
            storage = allocator.allocate(Params)
            assert allocator.allocated_bytes == 4
            with pytest.raises(AttributeError, match="compile-time only"):
                _ = storage.tile

        launch_ir(constexpr_kernel)

    def test_raw_byte_allocation(self):
        @flyc.kernel
        def raw_kernel():
            storage = fx.SharedAllocator().allocate(256)
            assert storage._target_type is fx.Array[fx.Uint8, 256]

        make_ptrs = _make_ptr_lines(launch_ir(raw_kernel))
        assert len(make_ptrs) == 1
        assert "allocBytes = 256" in make_ptrs[0]

    def test_raw_byte_allocation_rejects_non_positive_sizes(self):
        @flyc.kernel
        def bad_size_kernel():
            allocator = fx.SharedAllocator()
            with pytest.raises(ValueError, match="must be > 0"):
                allocator.allocate(0)
            with pytest.raises(ValueError, match="must be > 0"):
                allocator.allocate(-1)

        launch_ir(bad_size_kernel)

    def test_non_storable_type_is_rejected(self):
        @flyc.kernel
        def bad_type_kernel():
            allocator = fx.SharedAllocator()
            with pytest.raises(TypeError, match="Storable"):
                allocator.allocate(fx.Struct["t" : fx.Tensor])

        launch_ir(bad_type_kernel)

    def test_allocator_requires_a_kernel(self):
        def body():
            with pytest.raises(RuntimeError, match="@kernel"):
                fx.SharedAllocator()

        source_ir(body)

    def test_one_allocator_per_kernel(self):
        @flyc.kernel
        def two_allocators_kernel():
            fx.SharedAllocator()
            fx.SharedAllocator()

        with pytest.raises(RuntimeError, match="Only one SharedAllocator"):
            launch_ir(two_allocators_kernel)


@pytest.mark.l1a_compile_no_target_dialect
class TestStaticPlacement:
    """`static=True` (default): one LDS allocation per struct leaf."""

    def test_one_allocation_per_leaf(self):
        @flyc.kernel
        def static_kernel():
            allocator = fx.SharedAllocator()
            assert allocator.is_static is True
            lds = allocator.allocate(Shared).peek()
            _ = lds.a.view(fx.make_layout(128, 1))
            _ = lds.b.view(fx.make_layout(128, 1))

        ir_text = launch_ir(static_kernel)
        make_ptrs = _make_ptr_lines(ir_text)
        assert len(make_ptrs) == 2
        assert all("allocBytes = 512" in line for line in make_ptrs)
        assert all("shared" in line for line in make_ptrs)
        assert "dynamic_shared_memory_size" not in ir_text

    def test_nested_struct_emits_one_allocation_per_leaf(self):
        @flyc.kernel
        def nested_kernel():
            @fx.struct
            class Nested:
                inner: Inner
                tail: fx.Float32

            fx.SharedAllocator().allocate(Nested)

        assert len(_make_ptr_lines(launch_ir(nested_kernel))) == 3

    def test_union_leaf_is_one_allocation_sized_to_the_widest_variant(self):
        @flyc.kernel
        def union_leaf_kernel():
            @fx.struct
            class WithUnion:
                head: fx.Int32
                scratch: Scratch

            storage = fx.SharedAllocator().allocate(WithUnion)
            assert object.__getattribute__(storage.scratch.fp16, "_ptr") is object.__getattribute__(
                storage.scratch.fp32, "_ptr"
            )

        make_ptrs = _make_ptr_lines(launch_ir(union_leaf_kernel))
        assert len(make_ptrs) == 2
        assert any("allocBytes = 256" in line for line in make_ptrs)

    def test_static_mode_has_no_base_pointer(self):
        @flyc.kernel
        def base_ptr_kernel():
            allocator = fx.SharedAllocator()
            with pytest.raises(RuntimeError, match="no shared base pointer"):
                _ = allocator.base_ptr

        launch_ir(base_ptr_kernel)


@pytest.mark.l1a_compile_no_target_dialect
class TestDynamicPlacement:
    """`static=False`: one dynamic base pointer, sized at launch."""

    def test_single_base_pointer_and_inferred_smem(self):
        @flyc.kernel
        def dynamic_kernel():
            allocator = fx.SharedAllocator(static=False)
            assert allocator.is_static is False
            lds = allocator.allocate(Shared).peek()
            _ = lds.a.view(fx.make_layout(128, 1))

        ir_text = launch_ir(dynamic_kernel)
        assert ir_text.count("fly.get_dyn_shared") == 1
        assert "fly.make_ptr" not in ir_text
        assert "dynamic_shared_memory_size %c1024_i32" in ir_text

    def test_base_pointer_is_in_the_shared_address_space(self):
        @flyc.kernel
        def base_ptr_kernel():
            allocator = fx.SharedAllocator(static=False)
            assert allocator.base_ptr.address_space == fx.AddressSpace.Shared

        launch_ir(base_ptr_kernel)
