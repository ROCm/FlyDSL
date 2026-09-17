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
from flydsl._mlir.dialects import func
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
        pytest.param("device-default-cuda", marks=pytest.mark.l2_device, id="gpu-default-cuda"),
    ]
)
def storage_target(request, monkeypatch):
    """Compile both targets and run on the GPU with CPU and CUDA tensor defaults."""
    torch = pytest.importorskip("torch")
    on_device = request.param in ("device", "device-default-cuda")
    if on_device and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    if not on_device:
        monkeypatch.setenv("ARCH", request.param)
    monkeypatch.setenv("COMPILE_ONLY", "0" if on_device else "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    # Kernel tests can leave CUDA as the process default. Exercise both defaults
    # in isolation, restoring the caller's device context after each scenario.
    default_device = "cuda" if request.param == "device-default-cuda" else "cpu"
    with torch.device(default_device):
        yield torch, "cuda" if on_device else "cpu"


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
        launch(torch.empty(256, dtype=torch.float32, device="cpu"), torch.empty(256, dtype=torch.float32, device="cpu"))
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
        torch.testing.assert_close(tags.cpu(), torch.arange(block_size - 1, -1, -1, dtype=torch.int32, device="cpu"))


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
        peer = torch.arange(63, -1, -1, device="cpu")
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
            (lambda: fx.Array[object, 4], "Storable"),
            (lambda: fx.Array[fx.Int32, 0], "positive integer"),
            (lambda: fx.Array[fx.Int32, -1], "positive integer"),
            (lambda: fx.Array[fx.Int32, 4, 0], "positive integer"),
            (lambda: fx.Array[fx.Int32], r"Array\[dtype, size\]"),
        ],
    )
    def test_parameter_errors(self, make, match):
        with pytest.raises(TypeError, match=match):
            make()


def _numeric_array_access_ir(dtype, space, alignment, array_type=None):
    """Emit unoptimized MLIR for Array access or its typed-Pointer equivalent."""
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        ptr_type = fx.PointerType.get(fx.Uint8.ir_type, space, alignment)
        function = func.FuncOp("array_access", ([ptr_type, fx.Int32.ir_type, dtype.ir_type], []))
        with ir.InsertionPoint(function.add_entry_block()):
            ptr, index_value, value = function.entry_block.arguments
            items = array_type.__peek_from_ptr__(ptr) if array_type else fx.recast_iter(dtype, ptr)
            index = fx.Int32(index_value)
            literal = 1.0 if dtype.is_float else 1
            items[0] = literal
            items[2] = dtype(value)
            items[index] = value
            items[fx.make_int_tuple(index)] = dtype(literal)
            # A differently typed Numeric must retain Pointer.store's behavior.
            items[1] = fx.Int32(index_value)
            for offset in (0, 2, index, index_value, fx.make_int_tuple(index)):
                items[offset]
            items.view(fx.make_layout(8, 1))
            if array_type:
                rebuilt = construct_from_ir_values(array_type, items, extract_to_ir_values(items))
                rebuilt[index]
            else:
                items[index]
            func.ReturnOp([])
    module.operation.verify()
    return module.operation.get_asm(enable_debug_info=False)


def _numeric_array_allocation_ir(dtype, count, alignment, static, array_type=None):
    def allocate_items(allocator):
        if array_type is not None:
            return allocator.allocate(array_type).peek()
        storage = allocator.allocate(max(1, dtype.width * count // 8), alignment=alignment)
        return fx.recast_iter(dtype, object.__getattribute__(storage, "_ptr"))

    @flyc.kernel
    def kernel():
        allocator = fx.SharedAllocator(static=static)
        allocator.allocate(3)
        items = allocate_items(allocator)
        index = fx.thread_idx.x % count
        items[index] = 1.0 if dtype.is_float else 1
        items[index]
        items.view(fx.make_layout(count, 1))
        allocator.allocate(4)

    source = launch_ir(kernel)
    with ir.Context():
        module = ir.Module.parse(source)
        return module.operation.get_asm(enable_debug_info=False)


@pytest.mark.l1a_compile_no_target_dialect
class TestNumericArrayCompatibility:
    @pytest.mark.parametrize(
        "dtype",
        [
            fx.Boolean,
            fx.Int4,
            fx.Float4E2M1FN,
            fx.Int8,
            fx.Uint8,
            fx.Int16,
            fx.Uint16,
            fx.Int32,
            fx.Uint32,
            fx.Float16,
            fx.BFloat16,
            fx.Float32,
            fx.Float64,
            fx.Int64,
            fx.Uint64,
        ],
    )
    @pytest.mark.parametrize("space", [fx.AddressSpace.Global, fx.AddressSpace.Shared])
    @pytest.mark.parametrize("alignment_factor", [1, 3, 4])
    def test_mlir_is_identical_to_typed_pointer_access(self, ctx, dtype, space, alignment_factor):
        alignment = max(1, dtype.width // 8) * alignment_factor
        actual = _numeric_array_access_ir(dtype, space, alignment, fx.Array[dtype, 8, alignment])
        expected = _numeric_array_access_ir(dtype, space, alignment)
        assert actual == expected

    @pytest.mark.parametrize(
        "dtype, count, alignment",
        [(fx.Float32, 8, 4), (fx.Float32, 8, 12), (fx.Float16, 3, 2), (fx.Int4, 9, 1), (fx.Boolean, 9, 1)],
    )
    @pytest.mark.parametrize("static", [True, False])
    def test_allocation_mlir_is_identical_to_packed_bytes(self, dtype, count, alignment, static):
        array = fx.Array[dtype, count, alignment]
        actual = _numeric_array_allocation_ir(dtype, count, alignment, static, array)
        expected = _numeric_array_allocation_ir(dtype, count, alignment, static)
        assert actual == expected

    @pytest.mark.parametrize(
        "dtype, count, nbytes",
        [(fx.Boolean, 9, 1), (fx.Int4, 3, 1), (fx.Float4E2M1FN, 4, 2), (fx.Float6E2M3FN, 3, 2)],
    )
    def test_packed_layout(self, dtype, count, nbytes):
        array = fx.Array[dtype, count]
        assert dsl_size_of(array) == nbytes
        assert dsl_align_of(array) == 1
        assert array is fx.Array[dtype, count, 1]
        assert array.__name__ == f"Array[{dtype.__name__}, {count}]"

    @pytest.mark.parametrize("alignment", [1, 2, 3, 12])
    def test_explicit_alignment_accepts_any_positive_integer(self, alignment):
        array = fx.Array[fx.Float32, 4, alignment]
        assert (dsl_size_of(array), dsl_align_of(array)) == (16, alignment)

    @pytest.mark.parametrize("alignment", [1, 2])
    def test_invalid_pointer_alignment_keeps_the_pointer_error(self, ctx, alignment):
        array = fx.Array[fx.Float32, 8, alignment]
        with pytest.raises(ValueError) as actual:
            _numeric_array_access_ir(fx.Float32, fx.AddressSpace.Shared, alignment, array)
        with pytest.raises(ValueError) as expected:
            _numeric_array_access_ir(fx.Float32, fx.AddressSpace.Shared, alignment)
        assert str(actual.value) == str(expected.value)


@pytest.mark.l1a_compile_no_target_dialect
class TestArrayStorable:
    def test_custom_element_layout_and_access_use_the_protocol(self, ctx, symbolic_offsets, monkeypatch):
        class Payload(Word):
            @classmethod
            def __dsl_size_of__(cls):
                return 12

        typing_module = importlib.import_module("flydsl.expr.typing")
        monkeypatch.setattr(typing_module, "recast_iter", lambda dtype, ptr: ptr)
        monkeypatch.setattr(typing_module, "add_offset", lambda ptr, offset: (ptr, offset))

        array = fx.Array[Payload, 3]
        assert array is fx.Array[Payload, 3, 4]
        assert (dsl_size_of(array), dsl_align_of(array)) == (36, 4)
        aligned = fx.Array[Payload, 3, 16]
        assert (dsl_size_of(aligned), dsl_align_of(aligned)) == (36, 16)
        items = aligned.__peek_from_ptr__("base")
        assert items[2] == Payload(("peek", ("base", 24)))
        items[2] = Payload(7)
        assert Word.poked == [(("base", 24), 7)]

    @pytest.mark.parametrize(
        "missing", ["__dsl_size_of__", "__dsl_align_of__", "__peek_from_ptr__", "__poke_into_ptr__"]
    )
    def test_all_storable_hooks_are_required(self, missing):
        incomplete = type(
            "Incomplete",
            (),
            {name: hook for name, hook in vars(Word).items() if isinstance(hook, classmethod) and name != missing},
        )
        with pytest.raises(TypeError, match="Storable"):
            fx.Array[incomplete, 4]

    @pytest.mark.parametrize("alignment", [2, 12])
    def test_alignment_must_preserve_element_alignment_and_be_a_power_of_two(self, alignment):
        with pytest.raises(ValueError, match="multiple of the element alignment|power of two"):
            fx.Array[Word, 4, alignment]


@pytest.mark.l1a_compile_no_target_dialect
class TestArraySpecialized:
    @pytest.mark.parametrize(
        "dtype, stride, alignment",
        [
            (fx.Vector[fx.Float32, 3], 12, 4),
            (fx.Vector[fx.Uint32, (2, (1, 2))], 16, 4),
            (fx.Vector[fx.Float6E2M3FN, 4], 3, 1),
            (fx.Vector[fx.Boolean, 8], 1, 1),
            (fx.Pointer[fx.Int32, fx.AddressSpace.Global, 16], 8, 8),
            (fx.Pointer[fx.Int32, fx.AddressSpace.Shared, 16], 4, 4),
        ],
    )
    def test_layout_and_cache(self, dtype, stride, alignment):
        array = fx.Array[dtype, 5]
        assert array is fx.Array[dtype, 5, alignment]
        assert array.dtype is dtype
        assert (dsl_size_of(array), dsl_align_of(array)) == (5 * stride, alignment)
        stronger = fx.Array[dtype, 5, 32]
        assert (dsl_size_of(stronger), dsl_align_of(stronger)) == (5 * stride, 32)

    @pytest.mark.parametrize(
        "dtype", [fx.Vector, fx.Pointer, fx.Vector[fx.Index, 4], fx.Vector[fx.Int4, 3], fx.Vector[fx.Boolean, 1]]
    )
    def test_non_storable_elements_are_rejected_at_declaration(self, dtype):
        with pytest.raises(TypeError, match="Storable"):
            fx.Array[dtype, 4]

    @pytest.mark.parametrize("dtype", [fx.Float32x4, fx.Pointer[fx.Float32, fx.AddressSpace.Global]])
    @pytest.mark.parametrize("alignment", [2, 12])
    def test_alignment_must_preserve_elements_and_be_a_power_of_two(self, dtype, alignment):
        with pytest.raises(ValueError, match="multiple of the element alignment|power of two"):
            fx.Array[dtype, 4, alignment]

    @pytest.mark.parametrize("static", [True, False])
    def test_vector_indexing_and_reconstruction_inside_struct(self, static):
        vector = fx.Vector[fx.Uint32, (2, (1, 2))]
        array = fx.Array[vector, 4, 32]
        record = fx.Struct["values":array]

        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(3)
            values = allocator.allocate(record).peek().values
            index = fx.thread_idx.x % 4
            values[0] = vector(1)
            values[index] = vector(2) + fx.Uint32(3)
            rebuilt = construct_from_ir_values(array, values, extract_to_ir_values(values))
            assert type(rebuilt) is array and rebuilt.ptr.type.element_type == fx.Uint8.ir_type
            assert values.ptr.alignment == 32 and values._element_ptr(index).alignment == 16
            for value in (rebuilt[0], rebuilt[index]):
                assert type(value) is vector and value.shape == (2, (1, 2)) and value.dtype is fx.Uint32
            with pytest.raises(TypeError, match="expects"):
                values[index] = fx.Vector[fx.Uint32, 2](0)
            with pytest.raises(TypeError, match="expects"):
                values[index] = fx.Vector[fx.Float32, (2, (1, 2))](0.0)
            with pytest.raises(TypeError, match="requires Numeric elements"):
                values.view(fx.make_layout(4, 1))

        launch_ir(kernel)

    @pytest.mark.parametrize("space", [fx.AddressSpace.Global, fx.AddressSpace.Shared])
    def test_pointer_assignment_keeps_pointee_alignment_separate(self, space):
        pointer = fx.Pointer[fx.Int32, space, 16]
        array = fx.Array[pointer, 4, 32]

        @flyc.kernel
        def kernel():
            items = fx.SharedAllocator().allocate(fx.Struct["items":array]).peek().items
            index = fx.thread_idx.x % 4
            raw = fx.inttoptr(pointer.ir_type, fx.Uint64(128))
            assert type(raw) is fx.Pointer
            items[0] = raw
            items[index] = pointer(raw)
            rebuilt = construct_from_ir_values(array, items, extract_to_ir_values(items))
            for value in (rebuilt[0], rebuilt[index]):
                assert type(value) is pointer and value.type == pointer.ir_type
                assert value.alignment == 16
            assert items._element_ptr(index).alignment == dsl_size_of(pointer)
            with pytest.raises(TypeError, match="requires alignment"):
                items[index] = raw + 1
            wrong = fx.Pointer[fx.Float32, space, 16]
            with pytest.raises(TypeError, match="matching element type"):
                items[index] = fx.inttoptr(wrong.ir_type, fx.Uint64(128))

        launch_ir(kernel)


@pytest.mark.rocm_lower
class TestArraySpecializedStorage:
    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize(
        "dtype, shape",
        [
            (fx.Float32, 3),
            (fx.Uint32, (2, (1, 2))),
            (fx.Float4E2M1FN, 2),
            (fx.Float6E2M3FN, 4),
            (fx.Boolean, 8),
        ],
    )
    def test_vector_array_preserves_dense_payloads_and_guards(self, storage_target, static, dtype, shape):
        torch, device = storage_target
        vector = fx.Vector[dtype, shape]
        stride = dsl_size_of(vector)
        array = fx.Array[vector, 64, 32]

        @flyc.kernel
        def kernel(raw: fx.Tensor, out: fx.Tensor):
            tid = fx.thread_idx.x
            allocator = fx.SharedAllocator(static=static)
            before = allocator.allocate(fx.Uint8)
            items = allocator.allocate(array).peek()
            after = allocator.allocate(fx.Uint8)
            if tid == 0:
                fx.Storage[fx.Uint8](before._ptr).poke(fx.Uint8(0xA5))
                fx.Storage[fx.Uint8](after._ptr).poke(fx.Uint8(0x5A))
            fx.barrier()
            items[tid] = fx.Storage[vector](raw.iter + tid * stride).peek()
            fx.barrier()
            value = items[63 - tid]
            assert type(value) is vector and value.dtype is dtype
            fx.Storage[vector](out.iter + 1 + tid * stride).poke(value)
            if tid == 0:
                out[0] = fx.Storage[fx.Uint8](before._ptr).peek()
                out[1 + 64 * stride] = fx.Storage[fx.Uint8](after._ptr).peek()

        @flyc.jit
        def launch(raw: fx.Tensor, out: fx.Tensor):
            kernel(raw, out).launch(grid=1, block=64)

        raw = torch.arange(64 * stride, dtype=torch.int32, device=device).to(torch.uint8)
        out = torch.empty(64 * stride + 2, dtype=torch.uint8, device=device)
        launch(raw, out)
        if device != "cpu":
            expected = torch.cat(
                [
                    torch.tensor([0xA5], dtype=torch.uint8, device="cpu"),
                    raw.cpu().reshape(64, stride).flip(0).flatten(),
                    torch.tensor([0x5A], dtype=torch.uint8, device="cpu"),
                ]
            )
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("static", [True, False])
    def test_pointer_arrays_round_trip_and_dereference(self, storage_target, static):
        torch, device = storage_target
        global_pointer = fx.Pointer[fx.Int32, fx.AddressSpace.Global, 16]
        shared_pointer = fx.Pointer[fx.Int32, fx.AddressSpace.Shared, 16]

        @flyc.kernel
        def kernel(src: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            tid = fx.thread_idx.x
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(3)
            values = allocator.allocate(fx.Array[fx.Int32, 256, 64]).peek()
            global_ptrs = allocator.allocate(fx.Array[global_pointer, 64, 32]).peek()
            shared_ptrs = allocator.allocate(fx.Array[shared_pointer, 64, 16]).peek()
            # Preserve the offset's divisibility in the pointer alignment metadata.
            offset = fx.int_tuple_mul(tid, 4)
            global_ptrs[tid] = src.iter + offset
            shared_ptrs[tid] = values.ptr + offset
            values[tid * 4] = tid * 7
            fx.barrier()
            g = global_ptrs[63 - tid]
            s = shared_ptrs[63 - tid]
            assert type(g) is global_pointer and type(s) is shared_pointer
            assert g.alignment == s.alignment == 16
            assert global_ptrs._element_ptr(tid).alignment == 8 and shared_ptrs._element_ptr(tid).alignment == 4
            out[tid] = g[0]
            out[64 + tid] = s[0]
            addresses[tid] = fx.ptrtoint(g).to(fx.Int64)
            addresses[64 + tid] = (fx.ptrtoint(s) == fx.ptrtoint(values.ptr + (63 - tid) * 4)).to(fx.Int64)

        @flyc.jit
        def launch(src: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            kernel(src, out, addresses).launch(grid=1, block=64)

        src = torch.arange(256, dtype=torch.int32, device=device) * 11
        out = torch.empty(128, dtype=torch.int32, device=device)
        addresses = torch.empty(128, dtype=torch.int64, device=device)
        launch(flyc.from_dlpack(src, assumed_align=16), out, addresses)
        if device != "cpu":
            peer = torch.arange(63, -1, -1, device="cpu")
            expected = torch.cat([src.cpu()[peer * 4], (peer * 7).to(torch.int32)])
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(addresses[:64].cpu(), src.data_ptr() + peer * 16)
            assert torch.all(addresses[64:].cpu() == 1)


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

    def test_element_access_preserves_unsupported_operation_errors(self):
        array = fx.Array[Scratch, 4]
        assert dsl_size_of(array) == 4 * dsl_size_of(Scratch)

        def body():
            items = array.__peek_from_ptr__(fx.get_iter(fx.make_rmem_tensor(dsl_size_of(array), fx.Uint8)))
            with pytest.raises(NotImplementedError, match="does not support __peek_from_ptr__"):
                items[0]
            with pytest.raises(NotImplementedError, match="does not support __poke_into_ptr__"):
                items[0] = None

        source_ir(body)

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


@pytest.mark.rocm_lower
class TestStructArrayStorage:
    """### fx.Array[E, N, A]: nested Struct elements use their padded byte stride."""

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("dtype", ["uint8", "int32"])
    def test_nested_fields_and_padding_survive_shared_memory_copy(self, storage_target, static, dtype):
        torch, device = storage_target
        block_size = 64

        @fx.struct
        class Nested:
            bits: fx.Uint32
            weight: fx.Float64

        @fx.struct
        class Item:
            tag: fx.Int8
            nested: Nested
            tail: fx.Int32

        @fx.struct
        class ScratchArray:
            items: fx.Array[Item, block_size]

        @flyc.jit
        def carry(items):
            return items

        @flyc.kernel(known_block_size=[block_size, 1, 1])
        def kernel(raw: fx.Tensor, out: fx.Tensor):
            tid = fx.thread_idx.x
            # Both byte and i32 backing pointers must index by Item's byte stride.
            items = fx.Array[Item, block_size].__peek_from_ptr__(raw.iter)
            items[tid] = Item(
                (tid - 32).to(fx.Int8),
                Nested(fx.Uint32(0x80000000) + tid.to(fx.Uint32), tid.to(fx.Float64) * 0.25 + 0.5),
                -(tid * 7 + 1),
            )
            allocator = fx.SharedAllocator(static=static)
            allocator.allocate(3)
            shared = carry(allocator.allocate(ScratchArray).peek().items)
            shared[tid] = items[tid]
            fx.barrier()
            value = shared[block_size - 1 - tid]
            out[tid] = value.tag.to(fx.Float64)
            out[block_size + tid] = value.nested.bits.to(fx.Float64)
            out[2 * block_size + tid] = value.nested.weight
            out[3 * block_size + tid] = value.tail.to(fx.Float64)

        @flyc.jit
        def launch(raw: fx.Tensor, out: fx.Tensor):
            kernel(raw, out).launch(grid=(1, 1, 1), block=(block_size, 1, 1))

        # Item: tag at 0, nested at 8, tail at 24, then padding to a 32-byte stride.
        # The sentinel makes both internal and trailing padding observable.
        raw = torch.full((block_size * 32,), 0x5A, dtype=torch.uint8, device=device).view(getattr(torch, dtype))
        out = torch.empty(block_size * 4, dtype=torch.float64, device=device)
        launch(flyc.from_dlpack(raw, assumed_align=8), out)
        assert launch._last_compiled is not None
        if device == "cpu":
            return

        tag, bits, weight, tail = out.cpu().reshape(4, block_size)
        peer = torch.arange(block_size - 1, -1, -1, dtype=torch.float64, device="cpu")
        torch.testing.assert_close(tag, peer - 32, rtol=0, atol=0)
        torch.testing.assert_close(bits, peer + 0x80000000, rtol=0, atol=0)
        torch.testing.assert_close(weight, peer * 0.25 + 0.5, rtol=0, atol=0)
        torch.testing.assert_close(tail, -(peer * 7 + 1), rtol=0, atol=0)

        expected_bytes = bytearray([0x5A] * block_size * 32)
        for i in range(block_size):
            struct.pack_into("<b", expected_bytes, i * 32, i - 32)
            struct.pack_into("<I", expected_bytes, i * 32 + 8, 0x80000000 + i)
            struct.pack_into("<d", expected_bytes, i * 32 + 16, i * 0.25 + 0.5)
            struct.pack_into("<i", expected_bytes, i * 32 + 24, -(i * 7 + 1))
        assert bytes(raw.cpu().view(torch.uint8).tolist()) == expected_bytes


# ── fx.Align[T, A] ──────────────────────────────────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
class TestAlignModifier:
    """### fx.Align[T, A]: placement, underlying values, and parameter constraints."""

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
            (12, "power of two"),
            (24, "power of two"),
            (40, "power of two"),
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

    @pytest.mark.parametrize("alignment", [8, 16, 32])
    def test_documented_field_value_and_layout(self, alignment):
        """Align example: the field stays Float64; only the Struct adds padding."""
        Field = fx.Align[fx.Float64, alignment]
        Item = fx.Struct["weight":Field]
        item = Item(1.0)
        assert type(item.weight) is fx.Float64
        assert item.weight == 1.0
        assert item.replace(weight=2.0).weight == 2.0
        assert Item.__annotations__["weight"] is Field
        assert (dsl_size_of(Field), dsl_align_of(Field)) == (8, alignment)
        assert (dsl_size_of(Item), dsl_align_of(Item)) == (alignment, alignment)
        with pytest.raises(TypeError, match="weight"):
            Item(object())

    def test_nested_field_reconstruction_returns_the_inner_type(self):
        """Align delegates to T, including reconstruction of a nested Struct value."""
        Pair = fx.Struct["key" : fx.Uint32, "weight" : fx.Float64]
        Field = fx.Align[Pair, 16]
        Item = fx.Struct["pair":Field, "tail" : fx.Int32]

        def body(key: fx.Uint32):
            value = Item(Pair(key, 2.0), 3)
            assert type(value.pair) is Pair
            assert Item.__annotations__["pair"] is Field
            flat = extract_to_ir_values(value)
            assert len(flat) == 3
            for exemplar in (value, Item):
                rebuilt = construct_from_ir_values(Item, exemplar, flat)
                assert type(rebuilt.pair) is Pair
                assert type(rebuilt.pair.key) is fx.Uint32
                assert type(rebuilt.pair.weight) is fx.Float64

        source_ir(body, 0x80000000)

    def test_nested_field_preserves_constexpr_specialization(self):
        """The underlying Struct retains its compile-time fields through Align."""
        Config = fx.Struct["n" : fx.Constexpr[int], "value" : fx.Int32]
        Item = fx.Struct["config" : fx.Align[Config, 8]]

        def body(value: fx.Int32):
            item = Item(Config(3, value))
            assert type(item) is type(Item(Config(3, value)))
            rebuilt = construct_from_ir_values(type(item), item, extract_to_ir_values(item))
            assert rebuilt.config.n == 3
            assert type(rebuilt.config.value) is fx.Int32
            assert dsl_align_of(type(rebuilt)) == 8

        source_ir(body, 7)

    def test_host_argument_uses_the_inner_value(self):
        """No intermediate Align value or extra ABI slot is introduced at a JIT boundary."""
        Item = fx.Struct["weight" : fx.Align[fx.Float64, 16]]
        item = Item(1.0)
        with ir.Context(), ir.Location.unknown():
            assert len(c_abi_spec(item)) == 1

        def body(arg: Item):
            assert type(arg.weight) is fx.Float64
            _ = arg.weight + 1.0

        source_ir(body, item)


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

    def test_aligned_struct_variant_uses_maximum_alignment(self):
        """Byte layout: union variants start at zero and share the maximum alignment."""
        Item = fx.Struct["x" : fx.Align[fx.Int32, 16], "y" : fx.Int64]
        Overlay = fx.Union["item":Item, "other" : fx.Align[fx.Int32, 32]]
        assert _storage_layout(Overlay) == (32, 32, {"item": 0, "other": 0})


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

    @pytest.mark.parametrize("static", [True, False])
    def test_aligned_nested_struct_preserves_values_in_both_modes(self, static):
        """SharedAllocator exposes the same nested field values in static and dynamic modes."""
        Config = fx.Struct["n" : fx.Constexpr[int], "value" : fx.Int32]
        Item = fx.Struct["config" : fx.Align[Config, 8]]

        @flyc.kernel
        def kernel():
            item = Item(Config(3, fx.thread_idx.x))
            storage = fx.SharedAllocator(static=static).allocate(type(item))
            storage.poke(item)
            rebuilt = storage.peek()
            assert type(rebuilt) is type(item)
            assert rebuilt.config.n == 3
            assert type(rebuilt.config.value) is fx.Int32

        launch_ir(kernel)


@pytest.mark.rocm_lower
class TestAlignedStorage:
    """## Allocators: honor field and array alignment after a preceding allocation."""

    @pytest.mark.parametrize("mode", ["static", "dynamic", "global"])
    @pytest.mark.parametrize("alignment", [16, 32])
    def test_aligned_fields_keep_underlying_values(self, storage_target, mode, alignment):
        """### fx.Align[T, A]: storage and control flow expose T values directly."""
        torch, device = storage_target
        block_size = 64
        Item = fx.Struct["weight" : fx.Align[fx.Float64, alignment], "key" : fx.Align[fx.Uint32, 16]]
        Items = fx.Array[Item, block_size]
        stride = dsl_size_of(Item)

        @flyc.jit
        def carry(item):
            return item

        @flyc.kernel(known_block_size=[block_size, 1, 1])
        def kernel(raw: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            tid = fx.thread_idx.x
            if fx.const_expr(mode == "global"):
                allocator = _GlobalArena(raw.iter)
            else:
                allocator = fx.SharedAllocator(static=mode == "static")
            allocator.allocate(3)
            items = allocator.allocate(Items).peek()
            item = carry(Item(tid.to(fx.Float64) + 0.5, fx.Uint32(0x80000000) + tid.to(fx.Uint32)))
            if tid % 2 == 0:
                item = item.replace(weight=item.weight + 2.0)
            for step in range(tid % 3):
                item = item.replace(weight=item.weight + 1.0)
            remaining = tid % 2
            while remaining > 0:
                item = item.replace(weight=item.weight + 4.0)
                remaining = remaining - 1
            assert type(item.weight) is fx.Float64
            assert type(item.key) is fx.Uint32
            items[tid] = item
            ptr = fx.add_offset(items.ptr, fx.int_tuple_mul(tid, stride))
            weight_ptr = fx.recast_iter(fx.Float64, ptr)
            fx.generic_store(weight_ptr, fx.generic_load(weight_ptr, dtype=fx.Float64))
            addresses[tid] = fx.ptrtoint(ptr).to(fx.Int64)
            fx.barrier()
            peer = items[block_size - 1 - tid]
            out[tid] = peer.weight
            out[block_size + tid] = peer.key.to(fx.Float64)

        @flyc.jit
        def launch(raw: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            kernel(raw, out, addresses).launch(grid=(1, 1, 1), block=(block_size, 1, 1))

        raw = torch.empty(alignment + block_size * stride, dtype=torch.uint8, device=device)
        out = torch.empty(block_size * 2, dtype=torch.float64, device=device)
        addresses = torch.empty(block_size, dtype=torch.int64, device=device)
        # A global Arena uses the caller's aligned base pointer.
        assert raw.data_ptr() % alignment == 0
        launch(flyc.from_dlpack(raw, assumed_align=alignment), out, addresses)
        assert launch._last_compiled is not None
        if device == "cpu":
            return

        weight, key = out.cpu().reshape(2, block_size)
        peer = torch.arange(block_size - 1, -1, -1, dtype=torch.float64, device="cpu")
        expected_weight = peer + 0.5 + (peer % 2 == 0) * 2 + peer % 3 + (peer % 2) * 4
        torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(key, peer + 0x80000000, rtol=0, atol=0)
        addresses = addresses.cpu()
        assert torch.all(addresses % alignment == 0)
        assert torch.all(addresses[1:] - addresses[:-1] == stride)
        if mode == "global":
            assert addresses[0].item() == raw.data_ptr() + alignment

    @pytest.mark.parametrize("mode", ["static", "dynamic", "global"])
    @pytest.mark.parametrize("alignment", [16, 32])
    def test_array_alignment_changes_base_but_not_element_stride(self, storage_target, mode, alignment):
        """### fx.Array[E, N, A]: A aligns the base; elements remain sizeof(E) apart."""
        torch, device = storage_target
        block_size = 64

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

    @pytest.mark.parametrize("alignment", [16, 32])
    def test_aligned_leaf_uses_declared_size_and_alignment(self, alignment):
        """Static placement allocates one leaf at its declared size and alignment."""
        Field = fx.Align[fx.Float32, alignment]

        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator()
            storage = allocator.allocate(Field)
            storage.poke(fx.Float32(1.0))
            assert type(storage.peek()) is fx.Float32
            assert allocator.allocated_bytes == 4

        text = launch_ir(kernel)
        assert "allocBytes = 4" in text
        assert f"allocAlign = {alignment}" in text

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

    def test_aligned_leaf_padding_is_included_in_allocated_bytes(self):
        """Dynamic placement includes alignment padding in allocated_bytes and launch smem."""
        Field = fx.Align[fx.Float64, 16]

        @flyc.kernel
        def kernel():
            allocator = fx.SharedAllocator(static=False)
            allocator.allocate(fx.Array[fx.Uint8, 3])
            storage = allocator.allocate(Field)
            storage.poke(fx.Float64(1.0))
            assert type(storage.peek()) is fx.Float64
            assert allocator.allocated_bytes == 24

        text = launch_ir(kernel)
        assert "dynamic_shared_memory_size %c24_i32" in text
