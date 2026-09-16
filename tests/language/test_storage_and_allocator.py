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

Cases stay beside their documented topic across verification tiers: L1a checks
frontend semantics, L1b runs full target compilation, and L2 verifies GPU results.
The markers select the matching harness in ``conftest.py``. GPU availability is
checked only by the L2 cases, so ordinary language checks still run without a GPU.
"""

import importlib
import struct

import pytest
from lang_utils import launch_ir, source_ir

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.protocol import (
    c_abi_spec,
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


class _GlobalArena(fx.Arena):
    def __init__(self, ptr):
        super().__init__()
        self._ptr = ptr

    @property
    def base_ptr(self):
        return self._ptr


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

    def test_aligned_array_fields_preserve_element_stride_and_trailing_padding(self):
        """Byte layout: Array size includes element padding; the outer Struct pads to its alignment."""
        Item = fx.Struct["x" : fx.Align[fx.Int32, 16], "y" : fx.Int64]
        assert _storage_layout(Item) == (16, 16, {"x": 0, "y": 8})
        Array = fx.Array[Item, 3]
        assert (dsl_size_of(Array), dsl_align_of(Array)) == (48, 16)
        Outer = fx.Struct["items":Array, "tail" : fx.Align[fx.Int32, 32]]
        assert _storage_layout(Outer) == (96, 32, {"items": 0, "tail": 64})

    def test_legacy_numeric_array_alignment_is_preserved(self):
        """Byte layout uses the maximum field alignment, including existing Numeric Arrays."""
        # Legacy Numeric Array declarations retain their original layout rules.
        Legacy = fx.Struct["items" : fx.Array[fx.Int32, 2, 12], "tail" : fx.Align[fx.Int32, 16]]
        assert _storage_layout(Legacy) == (32, 16, {"items": 0, "tail": 16})
        # Power-of-two Align previously checked only that A >= natural alignment.
        Wrapped = fx.Align[fx.Array[fx.Int32, 2, 12], 16]
        assert (dsl_size_of(Wrapped), dsl_align_of(Wrapped)) == (8, 16)
        Outer = fx.Struct["items":Wrapped, "tail" : fx.Int32]
        assert _storage_layout(Outer) == (16, 16, {"items": 0, "tail": 8})


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

        @fx.struct
        class Item:
            weight: fx.Float64
            key: fx.Uint32

        Items = fx.Array[Item, block_size, alignment]
        assert (dsl_size_of(Item), dsl_align_of(Item)) == (16, 8)
        assert (dsl_size_of(Items), dsl_align_of(Items)) == (block_size * 16, alignment)

        @flyc.kernel(known_block_size=[block_size, 1, 1])
        def kernel(raw: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            tid = fx.thread_idx.x
            if fx.const_expr(mode == "global"):
                allocator = _GlobalArena(raw.iter)
            else:
                allocator = fx.SharedAllocator(static=mode == "static")
            allocator.allocate(3)
            items = allocator.allocate(Items).peek()
            items[tid] = Item(tid.to(fx.Float64) + 0.5, fx.Uint32(0x80000000) + tid.to(fx.Uint32))
            ptr = fx.add_offset(items.ptr, fx.int_tuple_mul(tid, 16))
            addresses[tid] = fx.ptrtoint(ptr).to(fx.Int64)
            fx.barrier()
            peer = items[block_size - 1 - tid]
            out[tid] = peer.weight
            out[block_size + tid] = peer.key.to(fx.Float64)

        @flyc.jit
        def launch(raw: fx.Tensor, out: fx.Tensor, addresses: fx.Tensor):
            kernel(raw, out, addresses).launch(grid=(1, 1, 1), block=(block_size, 1, 1))

        raw = torch.empty(alignment + block_size * 16, dtype=torch.uint8, device=device)
        out = torch.empty(block_size * 2, dtype=torch.float64, device=device)
        addresses = torch.empty(block_size, dtype=torch.int64, device=device)
        assert raw.data_ptr() % alignment == 0
        launch(flyc.from_dlpack(raw, assumed_align=alignment), out, addresses)
        assert launch._last_compiled is not None
        if device == "cpu":
            return

        weight, key = out.cpu().reshape(2, block_size)
        peer = torch.arange(block_size - 1, -1, -1, dtype=torch.float64, device="cpu")
        torch.testing.assert_close(weight, peer + 0.5, rtol=0, atol=0)
        torch.testing.assert_close(key, peer + 0x80000000, rtol=0, atol=0)
        addresses = addresses.cpu()
        assert addresses[0].item() % alignment == 0
        assert torch.all(addresses % 8 == 0)
        assert torch.all(addresses[1:] - addresses[:-1] == 16)
        if mode == "global":
            assert addresses[0].item() == raw.data_ptr() + alignment


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
